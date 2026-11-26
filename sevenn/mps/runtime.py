"""Per-calculator state on the Apple GPU: the switch, the padding bucket, the report."""

import copy
import os
import warnings
from typing import Dict, Optional

import torch

import sevenn._keys as KEY
from sevenn.nn.sequential import AtomGraphSequential

from .modules import (
    FusedConvolution,
    FusedGate,
    FusedLinear,
    FusedSelfConnection,
    check_torch,
    disable,
    enable,
)
from .padding import pad_edges, padding_is_safe


class MpsRuntime:
    """What a SevenNetCalculator keeps for the Apple GPU: the switch, the padding
    bucket and a report of what runs how."""

    def __init__(self, model, mode: str, bucket: int, cutoff: float):
        if isinstance(model, AtomGraphSequential):
            disable(
                model
            )  # an instance may come fused: 'off' has to mean the e3nn code
        self.mode = mode
        self.bucket = 0
        self.cutoff = cutoff
        self.num_edges = 0
        self.plan = None
        self.report: Dict[str, int] = {}
        if mode == 'off':
            return
        self.report = enable(model)
        self.plan = model._mps_plan
        if bucket > 1:
            # a convolution left on e3nn visits pad edges, and their derivatives are
            # noise rather than zero
            if self.report['conv_e3nn'] == 0 and padding_is_safe(
                model, cutoff, all_fused=True
            ):
                self.bucket = int(bucket)
            else:
                warnings.warn(
                    'edge padding disabled: pad edges would not be silent for this '
                    'model'
                )

    @classmethod
    def for_calculator(
        cls, calc, requested: Optional[str], bucket: int, is_instance: bool
    ):
        env = os.environ.get('SEVENNET_MPS_OPT')
        mode = (
            requested
            if requested is not None
            else (env if env is not None else 'fused')
        )
        if mode not in ('fused', 'off'):
            raise ValueError(f"mps_opt must be 'fused' or 'off', got {mode!r}")
        if calc.device.type != 'mps':
            if mode == 'fused' and requested is not None:
                raise ValueError(
                    f"mps_opt='fused' needs device='mps', got {calc.device}"
                )
            return None
        check_torch('SevenNetCalculator')
        if mode == 'fused':
            problem = None
            if not isinstance(calc.model, AtomGraphSequential):
                problem = 'a deployed TorchScript model has no modules to switch'
            elif not hasattr(torch.mps, 'compile_shader'):
                problem = (
                    f'torch {torch.__version__} has no '
                    f'torch.mps.compile_shader (needs 2.7 or later; all tests pass '
                    f'from 2.10)'
                )
            if problem is not None and requested is not None:
                raise ValueError(f"mps_opt='fused' is not possible: {problem}")
            if problem is not None:
                warnings.warn(
                    f'running the reference e3nn code on the GPU: {problem}'
                )
                mode = 'off'
        if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
            warnings.warn(
                'PYTORCH_ENABLE_MPS_FALLBACK=1 lets PyTorch run unsupported '
                'operations on the '
                'CPU without telling you. SevenNet does not need it on the Apple '
                'GPU.'
            )
        fused_already = (
            any(isinstance(m, _FUSED) for m in calc.model.modules())
            if isinstance(calc.model, AtomGraphSequential)
            else False
        )
        if is_instance and (mode == 'fused' or fused_already):
            try:
                # the instance may be shared, e.g. with a training loop: this keeps
                # it on the e3nn code, but the
                # constructor has already set eval mode and moved it to the device,
                # as it does on any device
                calc.model = copy.deepcopy(calc.model)
            except Exception as e:  # e.g. a TorchScript activation cannot be copied
                if requested is not None:
                    raise ValueError(
                        f"mps_opt='fused' on a model instance needs a copy of it: "
                        f'{e}'
                    ) from e
                warnings.warn(
                    f'running the reference e3nn code on the GPU, the model cannot '
                    f'be copied: {e}'
                )
                mode = 'off'
        return cls(calc.model, mode, bucket, calc.cutoff)

    def finish(self) -> None:
        """A call that failed before its first convolution must not leave its padding
        note for another graph."""
        if self.plan is not None:
            self.plan.expect_padded = None

    def prepare(self, graph: Dict) -> Dict:
        self.num_edges = graph[KEY.EDGE_IDX].shape[1]
        if self.plan is None:
            return graph
        if torch.is_autocast_enabled('mps'):  # the convolutions would use e3nn
            return graph
        graph = pad_edges(graph, self.cutoff, self.bucket)
        self.plan.expect_padded = (self.num_edges, graph[KEY.EDGE_IDX].shape[1])
        return graph


_FUSED = (FusedConvolution, FusedLinear, FusedSelfConnection, FusedGate)
