from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import torch
import torch.distributed as dist

import sevenn._keys as KEY
from sevenn.train.loss import LossDefinition

from .train.optim import loss_dict

if TYPE_CHECKING:
    from .atom_graph_data import AtomGraphData

_ERROR_TYPES = {
    'TotalEnergy': {
        'name': 'Energy',
        'ref_key': KEY.ENERGY,
        'pred_key': KEY.PRED_TOTAL_ENERGY,
        'unit': 'eV',
        'vdim': 1,
    },
    'Energy': {  # by default per-atom for energy
        'name': 'Energy',
        'ref_key': KEY.ENERGY,
        'pred_key': KEY.PRED_TOTAL_ENERGY,
        'unit': 'eV/atom',
        'per_atom': True,
        'vdim': 1,
    },
    'Force': {
        'name': 'Force',
        'ref_key': KEY.FORCE,
        'pred_key': KEY.PRED_FORCE,
        'unit': 'eV/Å',
        'vdim': 3,
    },
    'Stress': {
        'name': 'Stress',
        'ref_key': KEY.STRESS,
        'pred_key': KEY.PRED_STRESS,
        'unit': 'kbar',
        'coeff': 1602.1766208,
        'vdim': 6,
    },
    'Stress_GPa': {
        'name': 'Stress',
        'ref_key': KEY.STRESS,
        'pred_key': KEY.PRED_STRESS,
        'unit': 'GPa',
        'coeff': 160.21766208,
        'vdim': 6,
    },
    'BornEffectiveCharges': {
        'name': 'BornEffectiveCharges',
        'ref_key': KEY.BORN_EFFECTIVE_CHARGES,
        'pred_key': KEY.PRED_BORN_EFFECTIVE_CHARGES,
        'unit': 'e',
        'vdim': 9,
    },
    'DielectricTensor': {
        'name': 'DielectricTensor',
        'ref_key': KEY.DIELECTRIC_TENSOR,
        'pred_key': KEY.PRED_DIELECTRIC_TENSOR,
        'unit': '',
        'vdim': 9,
    },
    'TotalLoss': {
        'name': 'TotalLoss',
        'unit': None,
    },
}


def get_err_type(name: str) -> Dict[str, Any]:
    return deepcopy(_ERROR_TYPES[name])


def _get_loss_function_from_name(loss_functions, name):
    for loss_def, w in loss_functions:
        if loss_def.name.lower() == name.lower():
            return loss_def, w
    return None, None


class AverageNumber:
    def __init__(self):
        self._sum = 0.0
        self._count = 0

    def update(self, values: torch.Tensor) -> None:
        self._sum += values.sum().item()
        self._count += values.numel()

    def _ddp_reduce(self, device):
        _sum = torch.tensor(self._sum, device=device)
        _count = torch.tensor(self._count, device=device)
        dist.all_reduce(_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(_count, op=dist.ReduceOp.SUM)
        self._sum = _sum.item()
        self._count = _count.item()

    def get(self) -> float:
        if self._count == 0:
            return torch.nan
        return self._sum / self._count


class ErrorMetric:
    """
    Base class for error metrics We always average error by # of structures,
    and designed to collect errors in the middle of iteration (by AverageNumber)
    """

    def __init__(
        self,
        name: str,
        ref_key: str,
        pred_key: str,
        coeff: float = 1.0,
        unit: Optional[str] = None,
        per_atom: bool = False,
        ignore_unlabeled: bool = True,
        **kwargs,
    ) -> None:
        self.name = name
        self.unit = unit
        self.coeff = coeff
        self.ref_key = ref_key
        self.pred_key = pred_key
        self.per_atom = per_atom
        self.ignore_unlabeled = ignore_unlabeled
        self.value = AverageNumber()

        self.is_bec = (
            self.ref_key == KEY.BORN_EFFECTIVE_CHARGES
            and self.pred_key == KEY.PRED_BORN_EFFECTIVE_CHARGES
        )
        self.is_dielectric = (
            self.ref_key == KEY.DIELECTRIC_TENSOR
            and self.pred_key == KEY.PRED_DIELECTRIC_TENSOR
        )

    def _get_cartesian_tensor(self) -> Any:
        if getattr(self, '_ct', None) is None:
            import e3nn.io
            from e3nn.io import CartesianTensor
            if self.is_bec:
                self._ct = CartesianTensor('ij')
            elif self.is_dielectric:
                self._ct = CartesianTensor('ij=ji')
            else:
                self._ct = None
            if self._ct is not None:
                self._rtp = self._ct.reduced_tensor_products()
        return self._ct, getattr(self, '_rtp', None)

    def update(self, output: 'AtomGraphData') -> None:
        raise NotImplementedError

    def _retrieve(
        self, output: 'AtomGraphData'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        y_ref = output[self.ref_key] * self.coeff
        y_pred = output[self.pred_key] * self.coeff

        # If BornEffectiveCharges or Dielectric, convert irreps (pred) to cartesian
        if self.is_bec or self.is_dielectric:
            ct, rtp = self._get_cartesian_tensor()
            if y_pred.shape[-1] == 9 or (
                self.is_dielectric and y_pred.shape[-1] == 6
            ):
                y_pred = ct.to_cartesian(y_pred, rtp.to(y_pred.device))
                y_pred = y_pred.view(-1, 9)
            if y_ref.shape[-1] == 3 and y_ref.dim() >= 3:
                y_ref = y_ref.view(-1, 9)

        if self.per_atom:
            if y_ref.dim() > 1:
                y_ref = y_ref.view(-1)
            if y_pred.dim() > 1:
                y_pred = y_pred.view(-1)
            assert y_ref.dim() == 1 and y_pred.dim() == 1
            natoms = output[KEY.NUM_ATOMS]
            y_ref = y_ref / natoms
            y_pred = y_pred / natoms
        if self.ignore_unlabeled:
            if y_ref.dim() > 1:
                unlabelled_idx = (
                    torch.isnan(y_ref).view(y_ref.shape[0], -1).any(dim=1)
                )
            else:
                unlabelled_idx = torch.isnan(y_ref)
            y_ref = y_ref[~unlabelled_idx]
            y_pred = y_pred[~unlabelled_idx]
        return y_ref, y_pred

    def ddp_reduce(self, device: torch.device) -> None:
        self.value._ddp_reduce(device)

    def reset(self) -> None:
        self.value = AverageNumber()

    def get(self) -> float:
        return self.value.get()

    def key_str(self, with_unit: bool = True) -> str:
        if self.unit is None or not with_unit:
            return self.name
        else:
            return f'{self.name} ({self.unit})'

    def __str__(self):
        return f'{self.key_str()}: {self.value.get():.6f}'


class BECDiagRMSError(ErrorMetric):
    """
    Computes RMSE strictly on the diagonal elements of a
    3x3 Born Effective Charge tensor.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._se = torch.nn.MSELoss(reduction='none')

    def update(self, output: 'AtomGraphData') -> None:
        y_ref, y_pred = self._retrieve(output)
        if len(y_ref) == 0:
            return
        # Assumes y_ref and y_pred are flattened N*9 arrays, reshape to N, 3, 3
        y_ref = y_ref.view(-1, 3, 3)
        y_pred = y_pred.view(-1, 3, 3)

        diag_idx = torch.arange(3)
        y_ref_diag = y_ref[:, diag_idx, diag_idx].reshape(-1)
        y_pred_diag = y_pred[:, diag_idx, diag_idx].reshape(-1)

        se = self._se(y_ref_diag, y_pred_diag)
        self.value.update(se)

    def get(self) -> float:
        return self.value.get() ** 0.5


class BECOffDiagRMSError(ErrorMetric):
    """
    Computes RMSE strictly on the off-diagonal elements of a
    3x3 Born Effective Charge tensor.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._se = torch.nn.MSELoss(reduction='none')

    def update(self, output: 'AtomGraphData') -> None:
        y_ref, y_pred = self._retrieve(output)
        if len(y_ref) == 0:
            return
        # Assumes y_ref and y_pred are flattened N*9 arrays, reshape to N, 3, 3
        y_ref = y_ref.view(-1, 3, 3)
        y_pred = y_pred.view(-1, 3, 3)

        # Create mask for off-diagonal elements
        mask = ~torch.eye(3, dtype=torch.bool, device=y_ref.device)
        y_ref_off = y_ref[:, mask].reshape(-1)
        y_pred_off = y_pred[:, mask].reshape(-1)

        se = self._se(y_ref_off, y_pred_off)
        self.value.update(se)

    def get(self) -> float:
        return self.value.get() ** 0.5


class RMSError(ErrorMetric):
    """
    Vector squared error
    """

    def __init__(self, vdim: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.vdim = vdim
        self._se = torch.nn.MSELoss(reduction='none')

    def _square_error(
        self, y_ref: torch.Tensor, y_pred: torch.Tensor, vdim: int
    ) -> torch.Tensor:
        return self._se(y_ref.view(-1, vdim), y_pred.view(-1, vdim)).sum(dim=1)

    def update(self, output: 'AtomGraphData') -> None:
        y_ref, y_pred = self._retrieve(output)
        se = self._square_error(y_ref, y_pred, self.vdim)
        self.value.update(se)

    def get(self) -> float:
        return self.value.get() ** 0.5


class ComponentRMSError(ErrorMetric):
    """
    Ignore vector dim and just average over components
    Results smaller error
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._se = torch.nn.MSELoss(reduction='none')

    def _square_error(
        self, y_ref: torch.Tensor, y_pred: torch.Tensor
    ) -> torch.Tensor:
        return self._se(y_ref, y_pred)

    def update(self, output: 'AtomGraphData') -> None:
        y_ref, y_pred = self._retrieve(output)
        y_ref = y_ref.view(-1)
        y_pred = y_pred.view(-1)
        se = self._square_error(y_ref, y_pred)
        self.value.update(se)

    def get(self) -> float:
        return self.value.get() ** 0.5


class MAError(ErrorMetric):
    """
    Average over all component
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _square_error(
        self, y_ref: torch.Tensor, y_pred: torch.Tensor
    ) -> torch.Tensor:
        return torch.abs(y_ref - y_pred)

    def update(self, output: 'AtomGraphData') -> None:
        y_ref, y_pred = self._retrieve(output)
        y_ref = y_ref.reshape((-1,))
        y_pred = y_pred.reshape((-1,))
        se = self._square_error(y_ref, y_pred)
        self.value.update(se)


class CustomError(ErrorMetric):
    """
    Custom error metric
    Args:
        func: a function that takes y_ref and y_pred
              and returns a list of errors
    """

    def __init__(self, func: Callable, **kwargs) -> None:
        super().__init__(**kwargs)
        self.func = func

    def update(self, output: 'AtomGraphData') -> None:
        y_ref, y_pred = self._retrieve(output)
        se = self.func(y_ref, y_pred) if len(y_ref) > 0 else torch.tensor([])
        self.value.update(se)


class LossError(ErrorMetric):
    """
    Error metric that record loss
    """

    def __init__(
        self,
        name: str,
        loss_def: LossDefinition,
        **kwargs,
    ) -> None:
        super().__init__(
            name,
            ignore_unlabeld=loss_def.ignore_unlabeled,
            **kwargs,
        )
        self.loss_def = loss_def

    def update(self, output: 'AtomGraphData') -> None:
        loss = self.loss_def.get_loss(output)  # type: ignore
        self.value.update(loss)  # type: ignore


class CombinedError(ErrorMetric):
    """
    Combine multiple error metrics with weights
    corresponds to a weighted sum of errors (normally used in loss)
    """

    def __init__(self, metrics: List[Tuple[ErrorMetric, float]], **kwargs) -> None:
        super().__init__(**kwargs)
        self.metrics = metrics
        assert kwargs['unit'] is None

    def update(self, output: 'AtomGraphData') -> None:
        for metric, _ in self.metrics:
            metric.update(output)

    def reset(self) -> None:
        for metric, _ in self.metrics:
            metric.reset()

    def ddp_reduce(self, device) -> None:  # override
        for metric, _ in self.metrics:
            metric.value._ddp_reduce(device)

    def get(self) -> float:
        val = 0.0
        for metric, weight in self.metrics:
            val += metric.get() * weight
        return val


class ErrorRecorder:
    """
    record errors of a model
    """

    METRIC_DICT = {
        'RMSE': RMSError,
        'ComponentRMSE': ComponentRMSError,
        'MAE': MAError,
        'Loss': LossError,
        'DiagRMSE': BECDiagRMSError,
        'OffDiagRMSE': BECOffDiagRMSError,
    }

    def __init__(self, metrics: List[ErrorMetric]) -> None:
        self.history = []
        self.metrics = metrics

    def _update(self, output: 'AtomGraphData') -> None:
        for metric in self.metrics:
            metric.update(output)

    def update(self, output: 'AtomGraphData', no_grad=True) -> None:
        if no_grad:
            with torch.no_grad():
                self._update(output)
        else:
            self._update(output)

    def get_metric_dict(self, with_unit=True) -> Dict[str, float]:
        return {metric.key_str(with_unit): metric.get() for metric in self.metrics}

    def get_current(self) -> Dict[str, Union[float, str]]:
        dct = {}
        for metric in self.metrics:
            dct[metric.name] = {
                'value': metric.get(),
                'unit': metric.unit,
                'ref_key': metric.ref_key,
                'pred_key': metric.pred_key,
            }
        return dct

    def get_dct(self, prefix: str = '') -> Dict[str, str]:
        dct = {}
        if prefix.endswith('_') is False and prefix != '':
            prefix = prefix + '_'
        for metric in self.metrics:
            dct[f'{prefix}{metric.name}'] = f'{metric.get():6f}'
        return dct

    def get_key_str(self, name: str) -> Optional[str]:
        # TODO: why optional return?
        for metric in self.metrics:
            if name == metric.name:
                return metric.key_str()
        return None

    def epoch_forward(self) -> Dict[str, float]:
        self.history.append(self.get_current())
        pretty = self.get_metric_dict(with_unit=True)
        for metric in self.metrics:
            metric.reset()
        return pretty  # for print

    @staticmethod
    def init_total_loss_metric(
        config: Dict[str, Any],
        criteria: Optional[Callable] = None,
        loss_functions: Optional[List[Tuple[LossDefinition, float]]] = None,
    ) -> ErrorMetric:
        if criteria is None and loss_functions is None:
            raise ValueError('both criteria and loss functions not given')

        is_stress = config[KEY.IS_TRAIN_STRESS]
        metrics = []
        if criteria is not None:
            energy_metric = CustomError(criteria, **get_err_type('Energy'))
            metrics.append((energy_metric, 1))
            force_metric = CustomError(criteria, **get_err_type('Force'))
            metrics.append((force_metric, config[KEY.FORCE_WEIGHT]))
            if is_stress:
                stress_metric = CustomError(criteria, **get_err_type('Stress'))
                metrics.append((stress_metric, config[KEY.STRESS_WEIGHT]))
        else:  # TODO: this is hard-coded
            for efs in [
                'Energy', 'Force', 'Stress',
                'BornEffectiveCharges', 'DielectricTensor'
            ]:
                if efs == 'Stress' and not is_stress:
                    continue
                if efs == 'BornEffectiveCharges' and not config.get(
                    KEY.IS_TRAIN_BEC, False
                ):
                    continue
                if efs == 'DielectricTensor' and not config.get(
                    KEY.IS_TRAIN_DIELECTRIC, False
                ):
                    continue
                lf, w = _get_loss_function_from_name(loss_functions, efs)
                if lf is None:
                    raise ValueError(f'{efs} not found from loss_functions')
                metric = LossError(loss_def=lf, **get_err_type(efs))
                metrics.append((metric, w))

        total_loss_metric = CombinedError(
            metrics, name='TotalLoss', unit=None, ref_key=None, pred_key=None
        )
        return total_loss_metric

    @staticmethod
    def from_config(
        config: Dict[str, Any],
        loss_functions: Optional[List[Tuple[LossDefinition, float]]] = None,
    ) -> 'ErrorRecorder':
        loss_cls = loss_dict[config.get(KEY.LOSS, 'mse').lower()]
        loss_param = config.get(KEY.LOSS_PARAM, {})
        criteria = loss_cls(**loss_param) if loss_functions is None else None

        err_config = config.get(KEY.ERROR_RECORD, False)
        if not err_config:
            raise ValueError(
                'No error_record config found. Consider util.get_error_recorder'
            )
        err_config_n = []
        if not config.get(KEY.IS_TRAIN_STRESS, True):
            for err_type, metric_name in err_config:
                if 'Stress' in err_type:
                    continue
                err_config_n.append((err_type, metric_name))
            err_config = err_config_n

        err_metrics = []
        for err_type, metric_name in err_config:
            metric_kwargs = get_err_type(err_type)
            if err_type == 'TotalLoss':  # special case
                err_metrics.append(
                    ErrorRecorder.init_total_loss_metric(
                        config, criteria, loss_functions
                    )
                )
                continue
            metric_cls = ErrorRecorder.METRIC_DICT[metric_name]
            assert isinstance(metric_kwargs['name'], str)
            if metric_name == 'Loss':
                if loss_functions is not None:
                    metric_cls = LossError
                    metric_kwargs['loss_def'], _ = _get_loss_function_from_name(
                        loss_functions, metric_kwargs['name']
                    )
                else:
                    metric_cls = CustomError
                    metric_kwargs['func'] = criteria
                metric_kwargs.pop('unit', None)
            metric_kwargs['name'] += f'_{metric_name}'
            err_metrics.append(metric_cls(**metric_kwargs))
        return ErrorRecorder(err_metrics)
