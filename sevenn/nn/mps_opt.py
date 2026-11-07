"""Apple GPU (MPS) execution of a SevenNet model: same weights, same results to
float32 noise, much faster.

enable(model) switches supported modules to subclasses that run fused Metal kernels
(nn/mps_fused.py). Convolutions run the kernels in eval and train mode, with
derivatives to any order, so force training works. Linear, self-connection and gate
layers run kernels in eval mode, and in train mode a few PyTorch operations built
from the same tables, which autograd differentiates to any order; a layout they
do not cover keeps the e3nn code. Nothing is registered in the state dict;
disable(model) undoes it, and enable() again starts from scratch.
SevenNetCalculator imports this only for device='mps' or an explicit mps_opt.
pad_edges() keeps tensor shapes stable across MD steps.

In eval mode the kernels treat the weights of linear and self-connection layers as
constants: derivatives with respect to positions, strain and field are complete,
parameter gradients are not. For those use train mode or disable(). Changes through
load_state_dict, an optimizer or in-place operations on a parameter are noticed
before the next call (parameter identity and version counter), and module conversions
(.to, .half, .float) drop the derived tables; writes through param.data are not
noticed, call disable() and enable() after those.

"""

import copy
import os
import warnings
from typing import Dict, List, Optional, Set

import numpy as np
import torch
from e3nn.nn import Gate
from e3nn.o3 import FullyConnectedTensorProduct, Linear, TensorProduct

import sevenn._keys as KEY

from .convolution import IrrepsConvolution
from .mps_fused import (
    CsrApply,
    CsrMap,
    EdgePlan,
    FusedTP,
    FusedUvu,
    GateApply,
    GateTables,
)
from .node_embedding import OnehotEmbedding
from .self_connection import SelfConnectionIntro
from .sequential import AtomGraphSequential

ALL_PARTS = frozenset({'conv', 'linear', 'selfconn', 'gate'})
# train mode: up to this many atom types, multiply by every type's matrix and select
SELFCONN_ALL_TYPES = 8
# above this PyTorch's mm beats the sparse kernel (M: 133 -> 100 ms at 324 atoms; S
# unaffected)
LINEAR_MAX_NNZ = 100_000


def _matrix_triplets(fn, ni: int, batch: int = 2048):
    """Matrix of the linear map fn as (in, out, value) triplets, read off with
    one-hot inputs on the CPU."""
    II, OO, VV = [], [], []
    with torch.no_grad():
        for s in range(0, ni, batch):
            k = min(batch, ni - s)
            x = torch.zeros(k, ni)
            x[torch.arange(k), s + torch.arange(k)] = 1.0
            r = fn(x).numpy()
            i, o = np.nonzero(r)
            II.append(i + s)
            OO.append(o)
            VV.append(r[i, o])
    return np.concatenate(II), np.concatenate(OO), np.concatenate(VV)


def _weight_map(probe, param, run):
    """Which entry of `param` and which constant give each nonzero output of the
    linear map `run`, read off by probing: weights all one give the constants,
    weights 1, 2, 3, ... the index, and random weights check that each nonzero output
    is one weight times a constant. Returns (flat output positions, constants, weight
    indices) on the CPU."""
    with torch.no_grad():
        param.fill_(1.0)
        a1 = run()
        param.copy_(
            torch.arange(1, param.numel() + 1, dtype=param.dtype).view(param.shape)
        )
        a2 = run()
        w = torch.randn(
            param.shape,
            dtype=param.dtype,
            generator=torch.Generator().manual_seed(0),
        )
        param.copy_(w)
        a3 = run()
    nz = (a1 != 0).reshape(-1)
    pos = torch.nonzero(nz).squeeze(1)
    coef = a1.reshape(-1)[pos]
    idx = torch.round(a2.reshape(-1)[pos] / coef) - 1
    in_range = bool((idx >= 0).all()) and bool((idx < param.numel()).all())
    if (
        not in_range
        or not torch.allclose(
            a3.reshape(-1)[pos],
            coef * w.reshape(-1)[idx.long()],
            rtol=1e-6,
            atol=1e-9,
        )
        or bool((a3.reshape(-1)[~nz] != 0).any())
    ):
        raise NotImplementedError('not one weight times a constant per entry')
    return pos, coef, idx.long()


def _by_rows(fn, ni: int, chunk: int = 128):
    """fn applied to the rows of the (ni, ni) identity in chunks, so that probing a
    large layer needs little memory."""
    eye = torch.eye(ni, dtype=torch.float64)
    return torch.cat([fn(eye[s : s + chunk]) for s in range(0, ni, chunk)])


class _TrainMatrix:
    """Train mode of a fused linear layer: its matrix rebuilt from the live weights
    at each call, with one indexed write, so that autograd differentiates the layer
    to any order with a few large operations."""

    def __init__(self, pos, coef, idx, shape, device):
        self.pos, self.idx = pos.to(device), idx.to(device)
        self.coef = coef.to(device, torch.float32)
        self.shape = shape

    def build(self, w):
        n = 1
        for d in self.shape:
            n *= d
        return (
            torch.zeros(n, dtype=w.dtype, device=w.device)
            .index_put((self.pos,), self.coef * w.reshape(-1)[self.idx])
            .view(self.shape)
        )


class _TypedTrainMatrix:
    """Train mode of a fused self-connection: every atom type's matrix has the same
    positions and constants, and reads the weight at idx0 + t * stride, so one small
    table serves all types."""

    def __init__(self, pos, coef, idx0, stride, nt, shape, device):
        self.pos, self.idx0, self.stride = (
            pos.to(device),
            idx0.to(device),
            stride.to(device),
        )
        self.coef = coef.to(device, torch.float32)
        self.nt, self.shape = nt, shape

    def build(self, w, t: int):
        n = self.shape[0] * self.shape[1]
        v = self.coef * w.reshape(-1)[self.idx0 + t * self.stride]
        return (
            torch.zeros(n, dtype=w.dtype, device=w.device)
            .index_put((self.pos,), v)
            .view(self.shape)
        )

    def build_all(self, w):
        """(nt, ni, no): all types at once, for few types."""
        n = self.shape[0] * self.shape[1]
        t = torch.arange(self.nt, device=w.device)[:, None]
        v = self.coef * w.reshape(-1)[self.idx0 + t * self.stride]
        return (
            torch.zeros(self.nt * n, dtype=w.dtype, device=w.device)
            .index_put(((self.pos + t * n).reshape(-1),), v.reshape(-1))
            .view(self.nt, *self.shape)
        )


def _param_key(*params):
    # identity as well as version: deepcopy and load_state_dict(assign=True) make new
    # tensors with fresh counters
    return tuple((id(p), p._version) for p in params)


class _DerivedTables:
    """Fused modules keep tables derived from their weights outside the state dict.
    A module conversion (.to, .half, .float, ...) can change the weights while
    keeping their identity and version, so it drops the tables and the next call
    rebuilds them."""

    _DERIVED = ('_map', '_ver', '_train')

    def _apply(self, fn, *args, **kwargs):
        out = super()._apply(fn, *args, **kwargs)
        _drop(self, *self._DERIVED)
        return out

    def _set_map(self, triplets, bias, ni, no) -> None:
        """Map from the train tables: same positions at every rebuild, so after the
        first only the values change."""
        old = self.__dict__.get('_map')
        if old is not None and old.order is not None:
            old.set_values(np.concatenate([t[2] for t in triplets]), bias)
        else:
            self._map = CsrMap(
                triplets, bias, ni, no, self.weight.device, keep_order=True
            )


def _fused_ok(x, *more) -> bool:
    """The kernels and the train paths read float32 only; under autocast PyTorch
    would make some products float16 further down."""
    return (
        x.device.type == 'mps'
        and x.dtype == torch.float32
        and all(t.dtype == torch.float32 and t.device == x.device for t in more)
        and not torch.is_autocast_enabled('mps')
    )


class FusedLinear(_DerivedTables, Linear):
    """e3nn Linear; in eval mode one kernel forward and one backward, with the matrix
    read off the module."""

    def _build(self):
        train = self.__dict__.get('_train')
        if train is not None:
            # trained since the last build: the train tables give the matrix
            # from the weights directly, without probing the module again
            mat, vec = train
            no = mat.shape[1]
            pos = mat.pos.cpu().numpy()
            with torch.no_grad():
                val = (mat.coef * self.weight.reshape(-1)[mat.idx]).cpu().numpy()
                bias = (
                    np.zeros(no)
                    if vec is None
                    else vec.build(self.bias).cpu().numpy()
                )
            self._set_map([(pos // no, pos % no, val)], bias, mat.shape[0], no)
            self._ver = _param_key(self.weight, self.bias)
            return
        ref = copy.deepcopy(self).cpu()
        ref.__class__ = Linear
        trip = _matrix_triplets(
            lambda x: ref(x, None, torch.zeros_like(ref.bias)), ref.irreps_in.dim
        )
        bias = ref(torch.zeros(1, ref.irreps_in.dim))[0].detach().numpy()
        self._map = CsrMap(
            [trip], bias, ref.irreps_in.dim, ref.irreps_out.dim, self.weight.device
        )
        self._ver = _param_key(self.weight, self.bias)

    def _train_tables(self):
        """Matrix and bias of the layer as functions of its weights, for train mode;
        None keeps e3nn there. Built at the first call in train mode, so inference
        never pays for it."""
        ni, no = self.irreps_in.dim, self.irreps_out.dim
        probe = copy.deepcopy(self).cpu().double()
        probe.__class__ = Linear
        for n in ('_map', '_ver', '_train'):
            probe.__dict__.pop(n, None)
        try:
            with torch.no_grad():
                if probe.bias.numel():
                    probe.bias.zero_()
            mat = _TrainMatrix(
                *_weight_map(probe, probe.weight, lambda: _by_rows(probe, ni)),
                (ni, no),
                self.weight.device,
            )
            vec = None
            if probe.bias.numel():
                with torch.no_grad():
                    probe.weight.zero_()
                vec = _TrainMatrix(
                    *_weight_map(
                        probe,
                        probe.bias,
                        lambda: probe(torch.zeros(1, ni, dtype=torch.float64))[0],
                    ),
                    (no,),
                    self.weight.device,
                )
            return mat, vec
        except NotImplementedError as e:
            warnings.warn(f'linear layer trains with the e3nn code: {e}')
            return None

    def forward(self, features, weight=None, bias=None):
        if (
            weight is not None
            or bias is not None
            or features.dim() != 2
            or not _fused_ok(features, self.weight)
        ):
            return Linear.forward(self, features, weight, bias)
        if self.training:
            if '_train' not in self.__dict__:
                self._train = self._train_tables()
            if self._train is None:
                return Linear.forward(self, features)
            mat, vec = self._train
            out = features @ mat.build(self.weight)
            return out if vec is None else out + vec.build(self.bias)
        if self.__dict__.get('_ver') != _param_key(self.weight, self.bias):
            self._build()  # e.g. load_state_dict after enable
        return CsrApply.apply(features, self._map, None)


def _selfconn_triplets(ref: FullyConnectedTensorProduct) -> List:
    """Per atom type the matrix of a self-connection, from its weights: a scalar
    operand maps each irrep component to itself."""
    probe = copy.deepcopy(ref)
    with torch.no_grad():
        probe.weight.fill_(1.0)
    sl1, slo = ref.irreps_in1.slices(), ref.irreps_out.slices()
    nt, ni = ref.irreps_in2.dim, ref.irreps_in1.dim
    II, OO, VV = (
        [[] for _ in range(nt)],
        [[] for _ in range(nt)],
        [[] for _ in range(nt)],
    )
    for _, ins, W in ref.weight_views(yield_instruction=True):
        if ins.connection_mode != 'uvw' or W.shape[1] != nt:
            raise NotImplementedError(f'self-connection instruction {ins}')
        d = ref.irreps_in1[ins.i_in1].ir.dim
        x = torch.zeros(1, ni)
        x[0, sl1[ins.i_in1].start] = 1.0
        z = torch.zeros(1, nt)
        z[0, 0] = 1.0
        with torch.no_grad():
            c = float(
                probe(x, z)[0, slo[ins.i_out].start]
            )  # normalisation of this path
        W = W.detach().numpy()
        for t in range(nt):
            u, v = np.nonzero(W[:, t, :])
            for m in range(d):
                II[t].append(sl1[ins.i_in1].start + u * d + m)
                OO[t].append(slo[ins.i_out].start + v * d + m)
                VV[t].append(c * W[u, t, v])
    return [
        (np.concatenate(a), np.concatenate(b), np.concatenate(c))
        for a, b, c in zip(II, OO, VV)
    ]


class FusedSelfConnection(_DerivedTables, FullyConnectedTensorProduct):
    """x (N, D) with one-hot atom types z: one matrix per type, one kernel forward
    and one backward."""

    def _build(self):
        train = self.__dict__.get('_train')
        if train is not None:
            # trained since the last build: every type's matrix from the train
            # tables, without probing the module again (seconds for many types)
            ni, no = train.shape
            pos = train.pos.cpu().numpy()
            with torch.no_grad():
                t = torch.arange(train.nt, device=train.pos.device)[:, None]
                w = self.weight.reshape(-1)
                val = (train.coef * w[train.idx0 + t * train.stride]).cpu().numpy()
            trips = [(pos // no, pos % no, v) for v in val]
            self._set_map(trips, np.zeros(no), ni, no)
            self._ver = _param_key(self.weight)
            return
        ref = copy.deepcopy(self).cpu()
        ref.__class__ = FullyConnectedTensorProduct
        ni, nt, no = ref.irreps_in1.dim, ref.irreps_in2.dim, ref.irreps_out.dim
        trips = _selfconn_triplets(ref)
        # the matrices must reproduce the module
        x = torch.randn(
            3, ni, generator=torch.Generator().manual_seed(0)
        )  # leaves the global generator alone
        for t in {0, nt // 2, nt - 1}:
            z = torch.zeros(3, nt)
            z[:, t] = 1.0
            r = np.zeros((3, no))
            np.add.at(r.T, trips[t][1], (x[:, trips[t][0]].numpy() * trips[t][2]).T)
            if not np.allclose(r, ref(x, z).detach().numpy(), atol=1e-3, rtol=1e-3):
                raise NotImplementedError('self-connection is not a per-type matrix')
        self._map = CsrMap(trips, np.zeros(no), ni, no, self.weight.device)
        self._ver = _param_key(self.weight)

    def _train_tables(self):
        """Every type's matrix as a function of the weights, for train mode; None
        keeps e3nn there. Read off the weight layout (weight_views of 0, 1, 2, ...)
        and the path constants, then checked against the module for every type at
        once with random weights. Built at the first call in train mode, so inference
        never pays for it."""
        ni, nt, no = self.irreps_in1.dim, self.irreps_in2.dim, self.irreps_out.dim
        ref = copy.deepcopy(self).cpu().double()
        ref.__class__ = FullyConnectedTensorProduct
        for n in ('_map', '_ver', '_train'):
            ref.__dict__.pop(n, None)
        sl1, slo = ref.irreps_in1.slices(), ref.irreps_out.slices()
        try:
            P, CF, I0, ST = [], [], [], []
            with torch.no_grad():
                ref.weight.fill_(1.0)
                z0 = torch.zeros(1, nt, dtype=torch.float64)
                z0[0, 0] = 1.0
                ids = torch.arange(ref.weight.numel(), dtype=torch.float64)
                for _, ins, W in ref.weight_views(
                    weight=ids, yield_instruction=True
                ):
                    if (
                        ins.connection_mode != 'uvw'
                        or W.dim() != 3
                        or W.shape[1] != nt
                    ):
                        raise NotImplementedError(
                            f'self-connection instruction {ins}'
                        )
                    W = W.long()
                    d = ref.irreps_in1[ins.i_in1].ir.dim
                    x = torch.zeros(1, ni, dtype=torch.float64)
                    x[0, sl1[ins.i_in1].start] = 1.0
                    c = float(
                        ref(x, z0)[0, slo[ins.i_out].start]
                    )  # normalisation of this path
                    stride = (
                        W[:, 1, :] - W[:, 0, :]
                        if nt > 1
                        else torch.zeros_like(W[:, 0, :])
                    )
                    if not torch.equal(
                        W,
                        W[:, :1, :]
                        + torch.arange(nt)[None, :, None] * stride[:, None, :],
                    ):
                        raise NotImplementedError(
                            'weights of a type are not at a fixed stride'
                        )
                    u, v = torch.meshgrid(
                        torch.arange(W.shape[0]),
                        torch.arange(W.shape[2]),
                        indexing='ij',
                    )
                    for m in range(d):
                        P.append(
                            (sl1[ins.i_in1].start + u * d + m) * no
                            + slo[ins.i_out].start
                            + v * d
                            + m
                        )
                        I0.append(W[:, 0, :])
                        ST.append(stride)
                        CF.append(torch.full(u.shape, c, dtype=torch.float64))
            pos, idx0, stride, coef = (
                torch.cat([a.reshape(-1) for a in L]) for L in (P, I0, ST, CF)
            )
            if pos.unique().numel() != pos.numel():
                raise NotImplementedError('two weights write one entry')
            # check: random weights and inputs, every type in one call of the module
            g = torch.Generator().manual_seed(0)
            w = torch.randn(ref.weight.shape, dtype=torch.float64, generator=g)
            x = torch.randn(nt, ni, dtype=torch.float64, generator=g)
            with torch.no_grad():
                ref.weight.copy_(w)
                want = ref(x, torch.eye(nt, dtype=torch.float64))
            got = torch.zeros(nt, no, dtype=torch.float64)
            for s0 in range(0, nt, 32):
                t = torch.arange(s0, min(nt, s0 + 32))[:, None]
                val = coef * w.reshape(-1)[idx0 + t * stride] * x[t, pos // no]
                flat = (t - s0) * no + pos % no
                acc = torch.zeros(t.shape[0] * no, dtype=torch.float64)
                got[s0 : s0 + t.shape[0]] = acc.index_add_(
                    0, flat.reshape(-1), val.reshape(-1)
                ).view(-1, no)
            if not torch.allclose(
                got, want, rtol=1e-9, atol=1e-9 * float(want.abs().max())
            ):
                raise NotImplementedError(
                    'self-connection is not one weight times a constant per entry'
                )
        except NotImplementedError as e:
            warnings.warn(f'self-connection trains with the e3nn code: {e}')
            return None
        return _TypedTrainMatrix(
            pos, coef, idx0, stride, nt, (ni, no), self.weight.device
        )

    def forward(self, x, y, weight=None):
        if (
            weight is not None
            or x.dim() != 2
            or not _fused_ok(x, y, self.weight)
            or y.requires_grad
            or y.shape != (x.shape[0], self.irreps_in2.dim)
        ):  # a type index has no derivative
            return FullyConnectedTensorProduct.forward(self, x, y, weight)
        if self.training:
            if '_train' not in self.__dict__:
                self._train = self._train_tables()
            if self._train is None:
                return FullyConnectedTensorProduct.forward(self, x, y)
            if self._train.nt <= SELFCONN_ALL_TYPES:
                # few types: every type's product, then the one-hot operand picks
                # one; no read back to the CPU
                return torch.einsum(
                    'nt,tno->no',
                    y,
                    torch.matmul(x, self._train.build_all(self.weight)),
                )
            types = y.argmax(dim=1)
            out = x.new_zeros((x.shape[0], self.irreps_out.dim))
            for t in torch.unique(
                types
            ).tolist():  # many types: one small product for each type present
                rows = torch.nonzero(types == t).squeeze(1)
                out = out.index_copy(
                    0,
                    rows,
                    x.index_select(0, rows) @ self._train.build(self.weight, t),
                )
            return out
        if self.__dict__.get('_ver') != _param_key(self.weight):
            self._build()
        return CsrApply.apply(x, self._map, y.argmax(dim=1).to(torch.int32))


class FusedConvolution(IrrepsConvolution):
    def forward(self, data):
        x = data[self.key_x]
        y = data[self.key_filter]
        # autocast would make the weights float16 further down, and the kernels read
        # float32 only
        if (
            x.device.type != 'mps'
            or torch.is_autocast_enabled('mps')
            or x.dtype != torch.float32
            or y.dtype != torch.float32
            or data[self.key_weight_input].dtype != torch.float32
        ):
            return IrrepsConvolution.forward(self, data)
        plan = self._plan.get(data[self.key_edge_idx], x.shape[0])
        weight = self.weight_nn(data[self.key_weight_input])
        x = FusedUvu.apply(x, y, weight, self._kernels, plan)
        data[self.key_x] = x.div(self.denominator)
        return data


def _gate_act_code(act) -> int:
    """Code of a normalize2mom-wrapped activation in the gate kernel (silu or tanh),
    or an error."""
    f = getattr(act, 'f', None)
    if f is torch.nn.functional.silu:
        return 0
    if f is torch.tanh:
        return 1
    raise NotImplementedError(f'gate activation {f}')


def _gate_tables(gate: Gate, device) -> GateTables:
    """Slot tables of an e3nn Gate, read off the module: which input slots feed each
    output slot."""
    n_in = gate.irreps_in.dim

    def acts(activation):
        out = []
        for mul, (l, _), a in activation.paths:
            if l != 0 or a is None:
                raise NotImplementedError(
                    'gate without activation or on non-scalars'
                )
            out += [(_gate_act_code(a), 1.0 if a._is_id else a.cst)] * mul
        return out

    act_s, act_g = acts(gate.act_scalars), acts(gate.act_gates)
    with torch.no_grad():
        labels = torch.arange(n_in, dtype=torch.float32, device=device)[None]
        scal, gates, gated = (
            t[0].round().long().cpu().numpy() for t in gate.sc(labels)
        )
        src, gsl, code, cst = [], [], [], []
        for s, (c, k) in zip(scal.tolist(), act_s):
            src.append(s)
            gsl.append(-1)
            code.append(c)
            cst.append(k)
        if len(gated):
            ng, nk = len(gated), len(gates)
            # elementwise product: which gated slot and gate feed each output slot,
            # and its numeric factor
            to_gated = (
                gate.mul(
                    torch.eye(ng, device=device), torch.ones(ng, nk, device=device)
                )
                .cpu()
                .numpy()
            )
            to_gate = (
                gate.mul(
                    torch.ones(nk, ng, device=device), torch.eye(nk, device=device)
                )
                .cpu()
                .numpy()
            )
            for o in range(to_gated.shape[1]):
                ig = np.nonzero(to_gated[:, o])[0]
                ik = np.nonzero(to_gate[:, o])[0]
                if len(ig) != 1 or len(ik) != 1:
                    raise NotImplementedError('gate mapping')
                c, k = act_g[ik[0]]
                src.append(int(gates[ik[0]]))
                gsl.append(int(gated[ig[0]]))
                code.append(c)
                cst.append(k * float(to_gated[ig[0], o]))
    return GateTables(src, gsl, code, cst, n_in, gate.irreps_out.dim, device)


def _gate_train_tables(tables: GateTables):
    """The gate's slot tables as index tensors and masks, for its train path."""
    src, gated, code, cst = tables.tables
    return (
        src.long(),
        gated.clamp(min=0).long(),
        (code == 1)[None],
        (gated >= 0)[None],
        cst[None],
    )


class FusedGate(_DerivedTables, Gate):
    _DERIVED = ('_tables', '_train')

    def forward(self, features):
        if features.dim() != 2 or not _fused_ok(features):
            return Gate.forward(self, features)
        if '_tables' not in self.__dict__:  # dropped by a module conversion
            self._tables = _gate_tables(self, features.device)
        if self.training:
            # out[o] = cst[o] * act(x[src[o]]) * (x[gated[o]] if gated[o] >= 0 else
            # 1): a few elementwise operations that
            # autograd differentiates to any order, from the same tables as the
            # kernel
            if '_train' not in self.__dict__:
                self._train = _gate_train_tables(self._tables)
            src, gated, is_tanh, has_gate, cst = self._train
            v = features.index_select(1, src)
            v = cst * torch.where(
                is_tanh, torch.tanh(v), torch.nn.functional.silu(v)
            )
            return torch.where(has_gate, v * features.index_select(1, gated), v)
        return GateApply.apply(features, self._tables)


def _fuse(m, cls):
    """Switch m to cls, or leave it as it was: a module without its tables could not
    run at all."""
    original = m.__class__
    m.__class__ = cls
    try:
        m._build()
    except (NotImplementedError, ValueError) as e:
        m.__class__ = original
        raise NotImplementedError(str(e)) from e


def enable(model, parts='all') -> Dict[str, int]:
    """Switch supported modules of `model`, which must be on the Apple GPU, to fused
    kernels. Convolutions use them in eval and train mode; linear, self-connection
    and gate use them in eval mode and a few PyTorch operations in train mode.
    Returns how many modules run which way. Unsupported modules keep the e3nn code on
    the GPU, never the CPU."""
    want: Set[str] = set(ALL_PARTS if parts == 'all' else parts.split('+'))
    device = next(model.parameters()).device
    if device.type != 'mps':
        raise ValueError(
            f'fused kernels need a model on the Apple GPU, this one is on {device}'
        )
    other = {p.dtype for p in model.parameters() if p.is_floating_point()} - {
        torch.float32
    }
    if other:
        raise ValueError(
            f'fused kernels need float32 weights, this model has '
            f'{sorted(map(str, other))}'
        )
    # a second call starts from scratch, so the report, the shared edge plan and the
    # set of parts are those of this call
    disable(model)
    plan = EdgePlan()
    model._mps_plan = plan
    report = dict(
        conv_fused=0,
        conv_e3nn=0,
        linear_fused=0,
        linear_e3nn=0,
        selfconn_fused=0,
        selfconn_e3nn=0,
        gate_fused=0,
        gate_e3nn=0,
    )
    onehots = {
        m.key_additional_output
        for m in model.modules()
        if isinstance(m, OnehotEmbedding)
    }
    for m in list(model.modules()):
        if 'conv' in want and type(m) is IrrepsConvolution:
            try:
                if m.is_parallel or type(m.convolution) is not TensorProduct:
                    raise NotImplementedError('parallel or non-e3nn convolution')
                m._kernels, m._plan = FusedTP(m.convolution), plan
                m.__class__ = FusedConvolution
                report['conv_fused'] += 1
            except (NotImplementedError, ValueError) as e:
                warnings.warn(f'convolution keeps the e3nn code on the GPU: {e}')
                report['conv_e3nn'] += 1
        elif (
            'linear' in want
            and type(m) is Linear
            and m.irreps_out.dim > 0
            and m.irreps_in.dim > 0
            and m.weight.dim() == 1
        ):
            try:
                _fuse(m, FusedLinear)
            except NotImplementedError as e:
                warnings.warn(f'linear layer keeps the e3nn code on the GPU: {e}')
                report['linear_e3nn'] += 1
                continue
            if m._map.fwd[1].shape[0] > LINEAR_MAX_NNZ:
                # large dense blocks: PyTorch's mm is faster than the sparse kernel
                m.__class__ = Linear
                _drop(m, '_map', '_ver', '_train')
                report['linear_e3nn'] += 1
            else:
                report['linear_fused'] += 1
        elif 'selfconn' in want and type(m) is SelfConnectionIntro:
            # the kernel selects a matrix per atom type, which needs a one-hot
            # operand
            try:
                if (
                    m.key_operand not in onehots
                    or type(m.fc_tensor_product) is not FullyConnectedTensorProduct
                ):
                    raise NotImplementedError('operand is not one-hot')
                _fuse(m.fc_tensor_product, FusedSelfConnection)
                report['selfconn_fused'] += 1
            except NotImplementedError as e:
                # _fuse restores the class it changed; anything else is untouched
                warnings.warn(f'self-connection keeps the e3nn code on the GPU: {e}')
                report['selfconn_e3nn'] += 1
        elif 'gate' in want and type(m) is Gate:
            try:
                m._tables = _gate_tables(m, device)
                m.__class__ = FusedGate
                report['gate_fused'] += 1
            except (NotImplementedError, ValueError) as e:
                warnings.warn(f'gate keeps the e3nn code on the GPU: {e}')
                report['gate_e3nn'] += 1
    return report


_MPS_TORCH_OK: Optional[bool] = None


def mps_torch_supported() -> bool:
    """Whether this PyTorch can run SevenNet on the Apple GPU at all. The message
    scatter of the e3nn code (scatter_reduce) has an MPS kernel from PyTorch 2.6."""
    global _MPS_TORCH_OK
    if _MPS_TORCH_OK is None:
        try:
            x = torch.zeros(2, device='mps')
            idx = torch.zeros(1, dtype=torch.long, device='mps')
            x.scatter_reduce_(0, idx, torch.ones(1, device='mps'), reduce='sum')
            _MPS_TORCH_OK = True
        except (NotImplementedError, RuntimeError):
            _MPS_TORCH_OK = False
    return _MPS_TORCH_OK


def check_torch(purpose: str) -> None:
    """Raise a clear error, before any work, where the Apple GPU cannot run
    SevenNet with this PyTorch."""
    if not mps_torch_supported():
        raise RuntimeError(
            f'{purpose} on the Apple GPU needs PyTorch 2.6 or later and an '
            f'available MPS device: torch {torch.__version__} cannot run the '
            f'message scatter (scatter_reduce) on MPS. The fused kernels need '
            f'2.7 or later.'
        )


def enable_or_warn(model, purpose: str) -> Optional[Dict[str, int]]:
    """enable() for the Trainer and `sevenn inference` on device='mps', or a warning
    and the e3nn code where this PyTorch cannot compile Metal kernels, as in the
    calculator."""
    if not hasattr(torch.mps, 'compile_shader'):
        warnings.warn(
            f'{purpose} with the reference e3nn code on the GPU: torch '
            f'{torch.__version__} has no torch.mps.compile_shader'
        )
        return None
    return enable(model)


def _drop(m, *names) -> None:
    for n in names:
        m.__dict__.pop(n, None)


def disable(model) -> None:
    for m in model.modules():
        if isinstance(m, FusedConvolution):
            m.__class__ = IrrepsConvolution
            del m._kernels, m._plan
        elif isinstance(m, FusedLinear):
            m.__class__ = Linear
            _drop(m, '_map', '_ver', '_train')
        elif isinstance(m, FusedSelfConnection):
            m.__class__ = FullyConnectedTensorProduct
            _drop(m, '_map', '_ver', '_train')
        elif isinstance(m, FusedGate):
            m.__class__ = Gate
            _drop(m, '_tables', '_train')
    model.__dict__.pop('_mps_plan', None)


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
        graph = pad_edges(graph, self.cutoff, self.bucket)
        self.plan.expect_padded = (self.num_edges, graph[KEY.EDGE_IDX].shape[1])
        return graph


_FUSED = (FusedConvolution, FusedLinear, FusedSelfConnection, FusedGate)


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
