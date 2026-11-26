"""What happens when a SevenNet model moves to the Apple GPU (AtomGraphSequential._apply
calls apply() for a move to or from 'mps').

Before the move, float64 parameters and buffers become float32 (MPS has no float64).
After it, supported modules switch to the fused kernels (modules.enable) and two
forward hooks pad and unpad the edges of every input, single graphs and batches
alike, so that the GPU sees few tensor shapes. Moving the model off the Apple GPU
undoes both; it stays in float32. SEVENN_DISABLE_MPS=1 keeps the reference e3nn code
(still float32 on the GPU) and no padding.

The model is changed in place: moving a model shared with other code to 'mps' fuses
that object, as moving it changes its device for everyone. Nothing is registered in
the state dict.
"""

import warnings
from typing import Dict, Optional

import torch
from e3nn.o3 import TensorProduct

from sevenn.nn.convolution import IrrepsConvolution

from . import _policy
from .modules import check_torch, disable, enable
from .padding import pad, padding_is_safe, unpad

# pad the edge list to a multiple of this; MD and shuffled batches then reuse a few
# shapes instead of compiling a new MPS graph for almost every call
BUCKET = 256


class MpsState:
    """Kept on the model as `_mps` while it is on the Apple GPU."""

    def __init__(self, report: Dict[str, int], plan, bucket: int, cutoff: float):
        self.report = report  # how many modules run which way; {} with e3nn only
        self.plan = plan  # edge plan shared by the fused convolutions, or None
        self.bucket = bucket  # 0: no padding
        self.cutoff = cutoff
        self.n_real: Optional[int] = None  # edges of the call in progress


def _device_type_after(model, fn) -> Optional[str]:
    p = next(model.parameters(), None)
    probe = torch.zeros(0) if p is None else p.new_zeros(0)
    try:
        return fn(probe).device.type
    except Exception:
        return None


def apply(model, fn, parent_apply, *args, **kwargs):
    """Module._apply of AtomGraphSequential for a move to or from the Apple GPU:
    parent_apply is nn.Module._apply bound to the model."""
    if _device_type_after(model, fn) != 'mps':
        restore(model)  # leaving the Apple GPU, or a conversion off it
        return parent_apply(fn, *args, **kwargs)
    check_torch('SevenNet')
    restore(model)
    to_float32(model)
    out = parent_apply(fn, *args, **kwargs)
    prepare(model)
    return out


def to_float32(model) -> None:
    """float64 parameters and buffers (and gradients) to float32, keeping the
    Parameter objects, so optimizers keep their references."""
    for m in model.modules():
        for p in m._parameters.values():
            if p is not None and p.dtype == torch.float64:
                p.data = p.data.float()
                if p.grad is not None:
                    p.grad = p.grad.float()
        for k, b in m._buffers.items():
            if b is not None and b.dtype == torch.float64:
                m._buffers[k] = b.float()


def prepare(model, bucket: int = BUCKET) -> Dict[str, int]:
    """Fused kernels and edge padding for a model on the Apple GPU; starts from
    scratch on every call. Returns the report (empty with the reference code)."""
    restore(model)
    cuda_only = sorted(
        {
            type(m.convolution).__name__
            for m in model.modules()
            if isinstance(m, IrrepsConvolution)
            and not isinstance(m.convolution, TensorProduct)
            and not m.is_parallel
        }
    )
    if cuda_only:
        raise ValueError(
            f'the Apple GPU runs the e3nn tensor product, and this model has '
            f'{", ".join(cuda_only)}: build it without enable_cueq, enable_flash, '
            f'enable_oeq or enable_lunatp'
        )
    cutoff = float(getattr(model, 'cutoff', 0.0) or 0.0)
    if _policy.disabled():
        model.__dict__['_mps'] = MpsState({}, None, 0, cutoff)
        return {}
    problem = None
    if not hasattr(torch.mps, 'compile_shader'):
        problem = (
            f'torch {torch.__version__} has no torch.mps.compile_shader (needs 2.7 '
            f'or later)'
        )
    other = {p.dtype for p in model.parameters() if p.is_floating_point()}
    other -= {torch.float32}
    if other:
        problem = f'the fused kernels need float32 weights, not {sorted(map(str, other))}'
    if problem is not None:
        warnings.warn(f'the reference e3nn code runs on the Apple GPU: {problem}')
        model.__dict__['_mps'] = MpsState({}, None, 0, cutoff)
        return {}
    report = enable(model)
    plan = model._mps_plan
    padded = 0
    if bucket > 1:
        # a convolution left on e3nn visits pad edges, and their derivatives are
        # noise rather than zero
        if (
            report['conv_e3nn'] == 0
            and cutoff > 0.0
            and padding_is_safe(model, cutoff, all_fused=True)
        ):
            padded = int(bucket)
        else:
            warnings.warn(
                'edge padding disabled: pad edges would not be silent for this model, '
                'and memory may grow with each new edge count'
            )
    model.__dict__['_mps'] = MpsState(report, plan, padded, cutoff)
    if padded:
        model.register_forward_pre_hook(_pad_hook)
        model.register_forward_hook(_unpad_hook)
    return report


def restore(model) -> None:
    """Undo prepare(): the e3nn modules and no hooks. Parameters stay float32."""
    if '_mps' not in model.__dict__ and '_mps_plan' not in model.__dict__:
        return
    disable(model)
    # by identity, not by handle: a deepcopy of the model has its own hook dicts
    for hooks, fn in (
        (model._forward_pre_hooks, _pad_hook),
        (model._forward_hooks, _unpad_hook),
    ):
        for k in [k for k, h in hooks.items() if h is fn]:
            del hooks[k]
    model.__dict__.pop('_mps', None)


def report(model) -> Dict[str, int]:
    """What runs how on the Apple GPU ({} off it or with the reference code)."""
    st = model.__dict__.get('_mps')
    return dict(st.report) if st is not None else {}


def _pad_hook(model, args):
    st = model.__dict__['_mps']
    st.n_real = None
    st.plan.expect_padded = None
    data = args[0]
    # under autocast the convolutions fall back to e3nn, which would visit pad edges
    if not torch.is_autocast_enabled('mps'):
        st.n_real = pad(data, st.cutoff, st.bucket, st.plan)
    return None


def _unpad_hook(model, args, out):
    st = model.__dict__['_mps']
    st.plan.expect_padded = None
    n_real, st.n_real = st.n_real, None
    if n_real is not None:
        unpad(out, n_real)
    return None

