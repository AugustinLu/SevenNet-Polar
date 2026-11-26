"""Edge padding on the Apple GPU: pad edges are silent self-edges at the cutoff, so tensor
shapes repeat (MPS compiles a graph per shape) and results are those of the real edges."""

import warnings
from typing import Dict

import numpy as np
import torch

import sevenn._keys as KEY
from sevenn.nn.convolution import IrrepsConvolution


def _is_edge_key(key: str) -> bool:
    """Per-edge tensors of a batch or an output, by name: never a per-atom tensor,
    even when a small batch has as many atoms as edges."""
    return key in (KEY.EDGE_VEC, KEY.EDGE_LENGTH, KEY.EDGE_ATTR, KEY.EDGE_EMBEDDING,
                   KEY.CELL_SHIFT) or key.startswith('edge_')


class BatchPadder:
    """Pads the edges of a batch (training, validation, `sevenn inference`) to a
    multiple of `bucket`, so that the Apple GPU sees few distinct tensor shapes.

    PyTorch's MPS backend compiles a graph for every new shape of an operation
    and keeps it for the life of the process. A shuffled data set gives almost
    every batch a new number of edges, so without padding the process grows by
    one to two gigabytes a minute in training. Pad edges are those of pad_edges: atom
    0 to atom 0 with vector (cutoff, 0, 0), placed after the real edges, where the
    kernels never look. Energy, forces, stress, Born charges and every gradient
    are therefore those of the unpadded batch. Off (bucket 0) when a convolution
    is left on e3nn or a pad edge would not be silent, as in the calculator.

    The batch is changed in place and is meant for one model call. Padding it again
    adds nothing; its pad edges then count as real edges, which is slower but gives
    the same results, since padding is on only where pad edges are silent.

    """

    def __init__(self, model, bucket: int = 256):
        self.bucket = 0
        self.plan = getattr(model, '_mps_plan', None)
        self.cutoff = float(getattr(model, 'cutoff', 0.0) or 0.0)
        if self.plan is None or bucket <= 1:
            return
        on_e3nn = any(type(m) is IrrepsConvolution for m in model.modules())
        if on_e3nn or self.cutoff <= 0.0 or not padding_is_safe(
            model, self.cutoff, all_fused=True
        ):
            warnings.warn(
                'edge padding of batches disabled: pad edges would not be silent '
                'for this model, and memory may grow with each new batch shape'
            )
            return
        self.bucket = int(bucket)

    def run(self, model, batch):
        """model(self(batch)), with the padding note cleared however the call ends:
        a call that fails before its first convolution must not leave it for the
        next graph."""
        try:
            return model(self(batch))
        finally:
            if self.plan is not None:
                self.plan.expect_padded = None

    def __call__(self, batch):
        # under autocast the convolutions fall back to e3nn, which would visit pads
        if not self.bucket or torch.is_autocast_enabled('mps'):
            return batch
        idx = batch[KEY.EDGE_IDX]
        n_real = int(idx.shape[1])
        n_pad = (-n_real) % self.bucket
        if n_pad == 0 or n_real == 0:
            return batch
        for key in list(batch.keys()):
            t = batch[key]
            if (
                key == KEY.EDGE_IDX
                or not _is_edge_key(key)
                or not isinstance(t, torch.Tensor)
                or t.dim() == 0
                or t.shape[0] != n_real
            ):
                continue
            pad = t.new_zeros((n_pad,) + tuple(t.shape[1:]))
            if key == KEY.EDGE_VEC:
                pad[:, 0] = self.cutoff
            batch[key] = torch.cat([t, pad], 0)
        batch[KEY.EDGE_IDX] = torch.cat([idx, idx.new_zeros((2, n_pad))], 1)
        self.plan.expect_padded = (n_real, n_real + n_pad)
        return batch

    def unpad(self, out, n_real: int):
        """Drop the pad edges from a model output, e.g. before it is split back into
        structures; n_real is the edge count before padding."""
        n_all = int(out[KEY.EDGE_IDX].shape[1])
        if n_all == n_real:
            return out
        for key in list(out.keys()):
            t = out[key]
            if key == KEY.EDGE_IDX:
                out[key] = t[:, :n_real]
            elif (
                _is_edge_key(key)
                and isinstance(t, torch.Tensor)
                and t.dim() > 0
                and t.shape[0] == n_all
            ):
                out[key] = t[:n_real]
        return out


def padding_is_safe(model, cutoff: float, all_fused: bool = False) -> bool:
    """True if a pad edge (r = cutoff) gives exactly zero radial weights in every
    convolution, with finite derivatives.

    These derivatives are float32 noise, not 0 (seen up to 5e-5 in 7net-l3i5). With
    all convolutions fused the noise goes nowhere, because the kernels do not visit
    pad edges. A convolution left on e3nn would multiply it by dE/dw (forces cancel,
    as source = destination, but the stress does not), so without all_fused the
    derivatives must also stay below 1e-5.

    """
    device = next(model.parameters()).device
    vec = torch.tensor(
        [[cutoff, 0.0, 0.0]], dtype=torch.float32, device=device, requires_grad=True
    )
    if 'edge_embedding' not in model._modules:
        warnings.warn('edge padding disabled: the model has no edge_embedding')
        return False
    data = model._modules['edge_embedding']({KEY.EDGE_VEC: vec})
    emb = data[KEY.EDGE_EMBEDDING]
    tensors = [emb] + [
        m.weight_nn(emb)
        for m in model.modules()
        if getattr(m, 'weight_nn', None) is not None
    ]
    for t in tensors:
        if float(t.detach().abs().max()) != 0.0:
            return False
        for i in range(
            0, t.shape[1], max(t.shape[1] // 8, 1)
        ):  # a sample of components; the sum could cancel
            d = torch.autograd.grad(t[:, i].sum(), vec, retain_graph=True)[0]
            if not bool(torch.isfinite(d).all()) or (
                not all_fused and float(d.abs().max()) > 1e-5
            ):
                return False
    return True


def pad_edges(graph: Dict, cutoff: float, bucket: int) -> Dict:
    """Pad edge_index / edge_vec of a graph dict (as made by
    unlabeled_atoms_to_graph) to a multiple of `bucket` edges.

    MPS caches graphs by tensor shape and in MD the number of edges changes almost
    every step. Pad edges go from atom 0 to atom 0 with vector (cutoff, 0, 0), where
    the cutoff function vanishes, so their radial weights are exactly 0
    (padding_is_safe checks this on the actual model) and the kernels do not visit
    them at all.

    """
    n_pad = (-graph['edge_index'].shape[1]) % bucket if bucket > 1 else 0
    if n_pad == 0:
        return graph
    out = dict(graph)
    out['edge_index'] = np.concatenate(
        [graph['edge_index'], np.zeros((2, n_pad), dtype=graph['edge_index'].dtype)],
        axis=1,
    )
    pad_vec = np.zeros((n_pad, 3), dtype=graph['edge_vec'].dtype)
    pad_vec[:, 0] = cutoff
    out['edge_vec'] = np.concatenate([graph['edge_vec'], pad_vec], axis=0)
    if 'cell_shift' in graph:
        out['cell_shift'] = np.concatenate(
            [
                graph['cell_shift'],
                np.zeros((n_pad, 3), dtype=graph['cell_shift'].dtype),
            ],
            axis=0,
        )
    return out
