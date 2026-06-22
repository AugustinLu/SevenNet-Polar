from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

import sevenn._keys as KEY


class LossDefinition:
    """
    Base class for loss definition
    weights are defined in outside of the class
    """

    def __init__(
        self,
        name: str,
        unit: Optional[str] = None,
        criterion: Optional[Callable] = None,
        ref_key: Optional[str] = None,
        pred_key: Optional[str] = None,
        use_weight: bool = False,
        ignore_unlabeled: bool = True,
    ) -> None:
        self.name = name
        self.unit = unit
        self.criterion = criterion
        self.ref_key = ref_key
        self.pred_key = pred_key
        self.use_weight = use_weight
        self.ignore_unlabeled = ignore_unlabeled

    def __repr__(self):
        return self.name

    def assign_criteria(self, criterion: Callable) -> None:
        if self.criterion is not None:
            raise ValueError('Loss uses its own criterion.')
        self.criterion = criterion

    def _preprocess(
        self, batch_data: Dict[str, Any], model: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.pred_key is None or self.ref_key is None:
            raise NotImplementedError('LossDefinition is not implemented.')
        pred = torch.reshape(batch_data[self.pred_key], (-1,))
        ref = torch.reshape(batch_data[self.ref_key], (-1,))
        return pred, ref, None

    def _ignore_unlabeled(
        self,
        pred: torch.Tensor,
        ref: torch.Tensor,
        data_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        unlabeled = torch.isnan(ref)
        pred = pred[~unlabeled]
        ref = ref[~unlabeled]
        if data_weights is not None:
            data_weights = data_weights[~unlabeled]
        return pred, ref, data_weights

    def get_loss(self, batch_data: Dict[str, Any], model: Optional[Callable] = None):
        """
        Function that return scalar
        """
        if self.criterion is None:
            raise NotImplementedError('LossDefinition has no criterion.')
        pred, ref, w_tensor = self._preprocess(batch_data, model)

        if self.ignore_unlabeled:
            pred, ref, w_tensor = self._ignore_unlabeled(pred, ref, w_tensor)

        if len(pred) == 0:
            assert self.ref_key is not None
            return torch.zeros(1, device=batch_data[self.ref_key].device)

        loss = self.criterion(pred, ref)
        if self.use_weight:
            loss = torch.mean(loss * w_tensor)
        return loss


class PerAtomEnergyLoss(LossDefinition):
    """
    Loss for per atom energy
    """

    def __init__(
        self,
        name: str = 'Energy',
        unit: str = 'eV/atom',
        criterion: Optional[Callable] = None,
        ref_key: str = KEY.ENERGY,
        pred_key: str = KEY.PRED_TOTAL_ENERGY,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            unit=unit,
            criterion=criterion,
            ref_key=ref_key,
            pred_key=pred_key,
            **kwargs,
        )

    def _preprocess(
        self, batch_data: Dict[str, Any], model: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        num_atoms = batch_data[KEY.NUM_ATOMS]
        assert isinstance(self.pred_key, str) and isinstance(self.ref_key, str)
        pred = batch_data[self.pred_key] / num_atoms
        ref = batch_data[self.ref_key] / num_atoms
        w_tensor = None

        if self.use_weight:
            loss_type = self.name.lower()
            weight = batch_data[KEY.DATA_WEIGHT][loss_type]
            w_tensor = torch.repeat_interleave(weight, 1)

        return pred, ref, w_tensor


class ForceLoss(LossDefinition):
    """
    Loss for force
    """

    def __init__(
        self,
        name: str = 'Force',
        unit: str = 'eV/A',
        criterion: Optional[Callable] = None,
        ref_key: str = KEY.FORCE,
        pred_key: str = KEY.PRED_FORCE,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            unit=unit,
            criterion=criterion,
            ref_key=ref_key,
            pred_key=pred_key,
            **kwargs,
        )

    def _preprocess(
        self, batch_data: Dict[str, Any], model: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        assert isinstance(self.pred_key, str) and isinstance(self.ref_key, str)
        pred = torch.reshape(batch_data[self.pred_key], (-1,))
        ref = torch.reshape(batch_data[self.ref_key], (-1,))
        w_tensor = None

        if self.use_weight:
            loss_type = self.name.lower()
            weight = batch_data[KEY.DATA_WEIGHT][loss_type]
            w_tensor = weight[batch_data[KEY.BATCH]]
            w_tensor = torch.repeat_interleave(w_tensor, 3)

        return pred, ref, w_tensor


class StressLoss(LossDefinition):
    """
    Loss for stress this is kbar
    """

    def __init__(
        self,
        name: str = 'Stress',
        unit: str = 'kbar',
        criterion: Optional[Callable] = None,
        ref_key: str = KEY.STRESS,
        pred_key: str = KEY.PRED_STRESS,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            unit=unit,
            criterion=criterion,
            ref_key=ref_key,
            pred_key=pred_key,
            **kwargs,
        )
        self.TO_KB = 1602.1766208  # eV/A^3 to kbar

    def _preprocess(
        self, batch_data: Dict[str, Any], model: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        assert isinstance(self.pred_key, str) and isinstance(self.ref_key, str)

        pred = torch.reshape(batch_data[self.pred_key] * self.TO_KB, (-1,))
        ref = torch.reshape(batch_data[self.ref_key] * self.TO_KB, (-1,))
        w_tensor = None

        if self.use_weight:
            loss_type = self.name.lower()
            weight = batch_data[KEY.DATA_WEIGHT][loss_type]
            w_tensor = torch.repeat_interleave(weight, 6)

        return pred, ref, w_tensor


class BECLoss(LossDefinition):
    """
    Loss for Born Effective Charges
    """

    def __init__(
        self,
        name: str = 'BornEffectiveCharges',
        unit: str = 'e',
        criterion: Optional[Callable] = None,
        ref_key: str = KEY.BORN_EFFECTIVE_CHARGES,
        pred_key: str = KEY.PRED_BORN_EFFECTIVE_CHARGES,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            unit=unit,
            criterion=criterion,
            ref_key=ref_key,
            pred_key=pred_key,
            **kwargs,
        )

    def _get_cartesian_tensor(self) -> Any:
        if getattr(self, '_ct', None) is None:
            import e3nn.io
            from e3nn.io import CartesianTensor
            self._ct = CartesianTensor('ij')
            self._rtp = self._ct.reduced_tensor_products()
        return self._ct, self._rtp

    def _preprocess(
        self, batch_data: Dict[str, Any], model: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        assert isinstance(self.pred_key, str) and isinstance(self.ref_key, str)

        # pred is 9 components (1x0e+1x1e+1x2e irreps format)
        pred = batch_data[self.pred_key]

        # ref is Cartesian tensor 3x3 format (or 9 flat cartesian)
        ref_cartesian = batch_data[self.ref_key]
        if ref_cartesian.shape[-1] == 9 and ref_cartesian.dim() == 2:
            ref_cartesian = ref_cartesian.reshape(-1, 3, 3)

        # Convert true cartesian to irreps format (N, 9)
        ct, rtp = self._get_cartesian_tensor()
        ref_irreps = ct.from_cartesian(ref_cartesian, rtp.to(ref_cartesian.device))

        pred = torch.reshape(pred, (-1,))
        ref = torch.reshape(ref_irreps, (-1,))
        w_tensor = None

        if self.use_weight:
            loss_type = self.name.lower()
            weight = batch_data[KEY.DATA_WEIGHT][loss_type]
            w_tensor = weight[batch_data[KEY.BATCH]]
            w_tensor = torch.repeat_interleave(w_tensor, 9)

        return pred, ref, w_tensor

    def get_loss(self, batch_data: Dict[str, Any], model: Optional[Callable] = None):
        """
        Function that return scalar.
        Overridden for BECLoss to compensate for 9-component flattening.
        Flattening divides the mean loss by N*9 instead of N. We multiply by 9
        to restore per-atom loss scaling, ensuring consistent gradient magnitudes.
        """
        loss = super().get_loss(batch_data, model)
        return loss * 9.0


class DielectricLoss(LossDefinition):
    """
    Loss for Macroscopic Static Dielectric Tensor
    """

    def __init__(
        self,
        name: str = 'DielectricTensor',
        unit: str = '',
        criterion: Optional[Callable] = None,
        ref_key: str = KEY.DIELECTRIC_TENSOR,
        pred_key: str = KEY.PRED_DIELECTRIC_TENSOR,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            unit=unit,
            criterion=criterion,
            ref_key=ref_key,
            pred_key=pred_key,
            **kwargs,
        )

    def _get_cartesian_tensor(self) -> Any:
        if getattr(self, '_ct', None) is None:
            import e3nn.io
            from e3nn.io import CartesianTensor
            self._ct = CartesianTensor('ij=ji')
            self._rtp = self._ct.reduced_tensor_products()
        return self._ct, self._rtp

    def _preprocess(
        self, batch_data: Dict[str, Any], model: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        assert isinstance(self.pred_key, str) and isinstance(self.ref_key, str)

        pred = batch_data[self.pred_key]

        ref_cartesian = batch_data[self.ref_key]
        if ref_cartesian.shape[-1] == 9 and ref_cartesian.dim() == 2:
            ref_cartesian = ref_cartesian.reshape(-1, 3, 3)

        ct, rtp = self._get_cartesian_tensor()
        ref_irreps = ct.from_cartesian(ref_cartesian, rtp.to(ref_cartesian.device))

        pred = torch.reshape(pred, (-1,))
        ref = torch.reshape(ref_irreps, (-1,))
        w_tensor = None

        if self.use_weight:
            loss_type = self.name.lower()
            weight = batch_data[KEY.DATA_WEIGHT][loss_type]
            w_tensor = weight[batch_data[KEY.BATCH]]
            # Dielectric Tensor is a per-graph property,
            # batch should be unique for it,
            # but wait, since it's 6 components we need to repeat.
            w_tensor = torch.repeat_interleave(w_tensor, 6)

        return pred, ref, w_tensor

    def get_loss(self, batch_data: Dict[str, Any], model: Optional[Callable] = None):
        """
        Function that return scalar.
        Multiply by 6 to compensate for 6-component flattening.
        """
        loss = super().get_loss(batch_data, model)
        return loss * 6.0


def get_loss_functions_from_config(
    config: Dict[str, Any],
) -> List[Tuple[LossDefinition, float]]:
    from sevenn.train.optim import loss_dict

    loss_functions = []  # list of tuples (loss_definition, weight)

    loss = loss_dict[config[KEY.LOSS].lower()]
    loss_param = config.get(KEY.LOSS_PARAM, {})

    use_weight = config.get(KEY.USE_WEIGHT, False)
    if use_weight:
        loss_param['reduction'] = 'none'
    criterion = loss(**loss_param)

    commons = {'use_weight': use_weight}

    loss_functions.append(
        (PerAtomEnergyLoss(**commons), config.get(KEY.ENERGY_WEIGHT, 1.0))
    )
    loss_functions.append((ForceLoss(**commons), config[KEY.FORCE_WEIGHT]))
    if config[KEY.IS_TRAIN_STRESS]:
        loss_functions.append((StressLoss(**commons), config[KEY.STRESS_WEIGHT]))
    if config.get(KEY.IS_TRAIN_BEC, False):
        loss_functions.append((BECLoss(**commons), config[KEY.BEC_WEIGHT]))
    if config.get(KEY.IS_TRAIN_DIELECTRIC, False):
        loss_functions.append(
            (DielectricLoss(**commons), config[KEY.DIELECTRIC_WEIGHT]))

    for loss_function, _ in loss_functions:  # why do these?
        if loss_function.criterion is None:
            loss_function.assign_criteria(criterion)

    return loss_functions
