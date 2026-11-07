"""Apple GPU execution (sevenn/nn/mps_opt.py, mps_fused.py). MPS tests are skipped
where no Apple GPU exists: a skip is not a pass."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk
from e3nn.nn import Gate
from e3nn.o3 import Irreps, Linear, TensorProduct

import sevenn._keys as KEY
import sevenn.train.dataload as dl
from sevenn.atom_graph_data import AtomGraphData
from sevenn.calculator import SevenNetCalculator
from sevenn.model_build import build_E3_equivariant_model
from sevenn.nn.mps_fused import generate_msl, probe_cg
from sevenn.nn.mps_opt import mps_torch_supported, pad_edges, padding_is_safe
from sevenn.util import chemical_species_preprocess

needs_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and mps_torch_supported()),
    reason='needs an Apple GPU and PyTorch 2.6 or later',
)
METHODS = ['direct']
KEYS = (
    KEY.PRED_TOTAL_ENERGY,
    KEY.PRED_FORCE,
    KEY.PRED_STRESS,
    KEY.ATOMIC_ENERGY,
    KEY.PRED_BORN_EFFECTIVE_CHARGES,
    KEY.PRED_DIELECTRIC_TENSOR,
)


def _config(
    method='direct',
    bec=True,
    diel=True,
    lmax=2,
    channel=4,
    T=3,
    cutoff=4.0,
    species=('Na', 'Cl'),
    **kw,
):
    """A small SevenNet-Polar Direct model with Z* and dielectric heads."""
    assert method == 'direct'
    cfg = {
        'cutoff': cutoff,
        'channel': channel,
        'radial_basis': {'radial_basis_name': 'bessel'},
        'cutoff_function': {'cutoff_function_name': 'poly_cut'},
        'interaction_type': 'nequip',
        'lmax': lmax,
        'is_parity': True,
        'num_convolution_layer': T,
        'weight_nn_hidden_neurons': [64, 64],
        'act_radial': 'silu',
        'act_scalar': {'e': 'silu', 'o': 'tanh'},
        'act_gate': {'e': 'silu', 'o': 'tanh'},
        'conv_denominator': 30.0,
        'train_denominator': False,
        'self_connection_type': 'nequip',
        'shift': -10.0,
        'scale': 10.0,
        'train_shift_scale': False,
        'irreps_manual': False,
        'lmax_edge': -1,
        'lmax_node': -1,
        'readout_as_fcn': False,
        'use_bias_in_linear': False,
        '_normalize_sph': True,
        KEY.IS_TRAIN_BEC: bec,
        KEY.IS_TRAIN_DIELECTRIC: diel,
    }
    cfg.update(chemical_species_preprocess(list(species)))
    cfg.update(kw)
    return cfg


def _build(cfg, seed=0):
    torch.manual_seed(seed)
    return build_E3_equivariant_model(cfg)


def _graph(seed=1, sheared=False):
    at = bulk('NaCl', 'rocksalt', a=5.63) * (2, 1, 1)
    if sheared:  # strongly anisotropic cell
        strain = [[1.0, 0.25, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.3]]
        at.set_cell(at.cell[:] @ torch.tensor(strain).numpy(), scale_atoms=True)
    at.rattle(0.15 if sheared else 0.1, seed=seed)
    return AtomGraphData.from_numpy_dict(dl.unlabeled_atoms_to_graph(at, 4.0))


def _run(model, graph, field=None):
    g = graph.clone().to('mps')
    if field is not None:
        g[KEY.EFIELD] = torch.tensor([field], dtype=torch.float32, device='mps')
    out = model(g)
    return {k: out[k].detach().cpu() for k in KEYS if k in out}


def _close(a, b, rel=2e-5):
    assert a.keys() == b.keys()
    for k in a:
        scale = max(float(a[k].abs().max()), 1.0)
        assert float((a[k] - b[k]).abs().max()) < rel * scale, k


def _model(method='direct', **kw):
    model = _build(_config(method=method, **kw)).to('mps')
    model.set_is_batch_data(False)
    return model.eval()


def _tp():
    ir_x = Irreps('4x0e+4x1o+4x2e')
    ir_y = Irreps('1x0e+1x1o+1x2e')
    ir_mid, ins = [], []
    for i, (m, a) in enumerate(ir_x):
        for j, (_, b) in enumerate(ir_y):
            for c in a * b:
                ins.append((i, j, len(ir_mid), 'uvu', True))
                ir_mid.append((m, c))
    return TensorProduct(
        ir_x, ir_y, Irreps(ir_mid), ins, shared_weights=False, internal_weights=False
    )


# ---- no GPU needed ----


def test_probed_coefficients_are_sparse_with_no_borderline_values():
    for P in probe_cg(_tp()):
        g = np.abs(P['g'])
        assert np.all((g < 1e-7) | (g > 1e-3))


def test_generated_source_has_all_kernels():
    src = generate_msl(_tp())
    for k in ('uvu_fwd', 'uvu_adj_x', 'uvu_adj_yw', 'uvu_sum_y'):
        assert f'kernel void {k}(' in src


def test_pad_edges_shapes_and_noop():
    g = dl.unlabeled_atoms_to_graph(bulk('NaCl', 'rocksalt', a=5.63), 4.0)
    n = g['edge_index'].shape[1]
    assert pad_edges(g, 4.0, 1) is g
    assert pad_edges(g, 4.0, 0) is g
    assert pad_edges(g, 4.0, n) is g
    p = pad_edges(g, 4.0, n + 7)
    assert p['edge_index'].shape[1] == n + 7 and p['edge_vec'].shape[0] == n + 7
    assert np.all(p['edge_index'][:, n:] == 0) and np.all(
        p['edge_vec'][n:, 0] == 4.0
    )
    assert g['edge_index'].shape[1] == n  # input untouched


@pytest.mark.parametrize('method', METHODS)
def test_padding_is_safe_for_default_model(method):
    assert padding_is_safe(_build(_config(method=method)), 4.0)


def test_padding_is_not_safe_with_sigmoid_radial_activation():
    model = _build(_config(act_radial='sigmoid'))
    assert not padding_is_safe(model, 4.0)
    assert not padding_is_safe(model, 4.0, all_fused=True)


def test_large_radial_weights_only_matter_with_a_convolution_on_e3nn():
    model = _build(_config())
    with torch.no_grad():
        for m in model.modules():
            if getattr(m, 'weight_nn', None) is not None:
                for p in m.weight_nn.parameters():
                    p *= 1e4
    assert padding_is_safe(model, 4.0, all_fused=True)


def test_cuda_and_cpu_paths_do_not_import_the_apple_code():
    code = (
        'import sys, numpy, sevenn.calculator, sevenn.model_build, '
        'sevenn.checkpoint, sevenn.scripts.deploy\n'
        "assert not [m for m in sys.modules if m.startswith('sevenn.nn.mps')]\n"
    )
    subprocess.run([sys.executable, '-c', code], check=True)


def test_fused_on_cpu_device_is_an_error():
    model = _build(_config())
    with pytest.raises(ValueError, match='needs device'):
        SevenNetCalculator(
            model, file_type='model_instance', device='cpu', mps_opt='fused'
        )
    with pytest.raises(ValueError, match="must be 'fused' or 'off'"):
        SevenNetCalculator(
            model, file_type='model_instance', device='cpu', mps_opt='yes'
        )
    calc = SevenNetCalculator(model, file_type='model_instance', device='cpu')
    assert calc.mps_runtime is None


# ---- Apple GPU ----


@needs_mps
@pytest.mark.parametrize(
    'method, field',
    [
        ('direct', None),
    ],
)
def test_fused_equals_e3nn(method, field):
    from sevenn.nn import mps_opt

    model = _model(method)
    graph = _graph()
    ref = _run(model, graph, field)
    rep = mps_opt.enable(model)
    assert rep['conv_fused'] == 3 and rep['conv_e3nn'] == 0 and rep['gate_e3nn'] == 0
    _close(ref, _run(model, graph, field))
    mps_opt.disable(model)
    _close(ref, _run(model, graph, field))  # MPS itself is not bitwise reproducible


@needs_mps
def test_state_dict_unchanged_and_strict_round_trip():
    from sevenn.nn import mps_opt

    model = _model()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    mps_opt.enable(model)
    after = model.state_dict()
    assert list(before) == list(after)
    for k in before:
        assert torch.equal(before[k], after[k])
    model.load_state_dict(before, strict=True)


@needs_mps
def test_load_state_dict_after_enable_refreshes_the_tables():
    from sevenn.nn import mps_opt

    graph = _graph()
    other = _build(_config(), seed=7).to('mps')
    other.set_is_batch_data(False)
    other.eval()
    ref = _run(other, graph)
    model = _model()
    mps_opt.enable(model)
    model.load_state_dict(other.state_dict(), strict=True)
    _close(ref, _run(model, graph))


@needs_mps
def test_tensor_product_kernels_equal_e3nn_on_random_graph_with_unsorted_edges():
    from sevenn.nn.mps_fused import EdgePlan, FusedTP, FusedUvu

    tp = _tp().to('mps')
    K = FusedTP(tp)
    N, E = 17, 400
    torch.manual_seed(0)
    x = torch.randn(N, tp.irreps_in1.dim, device='mps', requires_grad=True)
    y = torch.randn(E, tp.irreps_in2.dim, device='mps', requires_grad=True)
    w = torch.randn(E, tp.weight_numel, device='mps', requires_grad=True)
    idx = torch.randint(
        0, N - 2, (2, E), device='mps'
    )  # repeated, unsorted, two nodes without edges
    ref = torch.zeros(N, K.nout, device='mps').index_add_(
        0, idx[0], tp(x[idx[1]], y, w)
    )
    out = FusedUvu.apply(x, y, w, K, EdgePlan().get(idx, N))
    assert float((ref - out).detach().abs().max()) < 1e-5
    v = torch.randn_like(out)
    for a, b in zip(
        torch.autograd.grad((ref * v).sum(), [x, y, w]),
        torch.autograd.grad((out * v).sum(), [x, y, w]),
    ):
        assert float((a - b).abs().max()) < 1e-4


def _tp_unequal():
    ir_x = Irreps('8x0e+4x1o+2x2e')
    ir_y = Irreps('1x0e+1x1o+1x2e')
    ir_mid, ins = [], []
    for i, (m, a) in enumerate(ir_x):
        for j, (_, b) in enumerate(ir_y):
            for c in a * b:
                ins.append((i, j, len(ir_mid), 'uvu', True))
                ir_mid.append((m, c))
    ir_mid = Irreps(ir_mid)
    ir_sorted, p, _ = ir_mid.sort()
    ins = sorted(((i, j, p[k], c, t) for i, j, k, c, t in ins), key=lambda z: z[2])
    return TensorProduct(
        ir_x, ir_y, ir_sorted, ins, shared_weights=False, internal_weights=False
    )


@needs_mps
@pytest.mark.parametrize('tp_fn', [_tp, _tp_unequal])
def test_kernels_equal_e3nn_to_third_order(tp_fn):
    from sevenn.nn.mps_fused import EdgePlan, FusedTP, FusedUvu

    torch.manual_seed(0)
    tp = tp_fn().to('mps')
    N, E = 7, 23
    # unsorted edges, an atom with no edges, and repeated pairs
    idx = torch.randint(0, N - 1, (2, E), device='mps')
    x = torch.randn(N, tp.irreps_in1.dim, device='mps', requires_grad=True)
    y = torch.randn(E, tp.irreps_in2.dim, device='mps', requires_grad=True)
    w = torch.randn(E, tp.weight_numel, device='mps', requires_grad=True)

    def ref(x, y, w):
        return torch.zeros(N, tp.irreps_out.dim, device='mps').index_add(
            0, idx[0], tp(x[idx[1]], y, w)
        )

    def fused(x, y, w):
        return FusedUvu.apply(x, y, w, FusedTP(tp), EdgePlan().get(idx, N))

    probe = torch.randn(N, tp.irreps_out.dim, device='mps')
    got = {}
    for name, f in (('ref', ref), ('fused', fused)):
        out = f(x, y, w)
        g1 = torch.autograd.grad((out * probe).sum(), (x, y, w), create_graph=True)
        s2 = sum(g.pow(2).sum() for g in g1)
        g2 = torch.autograd.grad(s2, (x, y, w), create_graph=True)
        g3 = torch.autograd.grad(sum((g * g.detach()).sum() for g in g2), (x, y, w))
        got[name] = [out.detach()] + [g.detach() for g in g1 + g2 + g3]
    for k, (a, b) in enumerate(zip(got['ref'], got['fused'])):
        assert float((a - b).abs().max()) < 2e-5 * max(float(a.abs().max()), 1e-6), k


@needs_mps
def test_kernels_with_unequal_multiplicities_equal_e3nn():
    from sevenn.nn.mps_fused import EdgePlan, FusedTP, FusedUvu

    tp = _tp_unequal().to('mps')
    K = FusedTP(tp)
    N, E = 13, 300
    torch.manual_seed(1)
    x = torch.randn(N, tp.irreps_in1.dim, device='mps', requires_grad=True)
    y = torch.randn(E, tp.irreps_in2.dim, device='mps', requires_grad=True)
    w = torch.randn(E, tp.weight_numel, device='mps', requires_grad=True)
    idx = torch.randint(0, N - 1, (2, E), device='mps')
    ref = torch.zeros(N, K.nout, device='mps').index_add_(
        0, idx[0], tp(x[idx[1]], y, w)
    )
    out = FusedUvu.apply(x, y, w, K, EdgePlan().get(idx, N))
    assert float((ref - out).detach().abs().max()) < 1e-5
    v = torch.randn_like(out)
    for a, b in zip(
        torch.autograd.grad((ref * v).sum(), [x, y, w]),
        torch.autograd.grad((out * v).sum(), [x, y, w]),
    ):
        assert float((a - b).abs().max()) < 1e-4
    # pad edges at the end are skipped and get zero gradients
    plan = EdgePlan()
    idx = idx.clone()
    idx[:, E - 5 :] = 0
    plan.expect_padded = (E - 5, E)
    plan.get(idx, N)
    w0 = w.detach().clone()
    w0[E - 5 :] = 0.0
    w0.requires_grad_(True)
    out = FusedUvu.apply(x, y, w0, K, plan)
    gy, gw = torch.autograd.grad((out * v).sum(), [y, w0])
    assert (
        float(gy[E - 5 :].abs().max()) == 0.0
        and float(gw[E - 5 :].abs().max()) == 0.0
    )


@needs_mps
def test_model_with_unequal_channels_runs_fully_fused():
    from sevenn.nn import mps_opt

    model = _model(
        channel=[8, 8, 8, 8],
        lmax=2,
        irreps_manual=['8x0e', '8x0e+4x1o+2x2e', '8x0e+4x1o+2x2e', '8x0e'],
    )
    graph = _graph()
    ref = _run(model, graph)
    rep = mps_opt.enable(model)
    assert rep['conv_e3nn'] == 0 and rep['conv_fused'] == 3
    _close(ref, _run(model, graph))


@needs_mps
def test_kernels_respect_views_with_storage_offset():
    from sevenn.nn.mps_fused import CsrApply, CsrMap

    x = torch.arange(12.0, device='mps').reshape(4, 3)
    M = CsrMap(
        [(np.array([0, 2]), np.array([0, 1]), np.array([1.0, 1.0]))],
        np.zeros(2),
        3,
        2,
        'mps',
    )
    out = CsrApply.apply(x[1:], M, None)  # view with an offset
    assert out.cpu().tolist() == [[3.0, 5.0], [6.0, 8.0], [9.0, 11.0]]
    out = CsrApply.apply(
        x[:, 1:].reshape(4, 2).repeat(1, 2)[:, :3], M, None
    )  # non-contiguous source
    assert out.shape == (4, 2)


@needs_mps
def test_large_linear_layers_stay_on_e3nn(monkeypatch):
    from sevenn.nn import mps_opt

    model = _model()
    graph = _graph()
    ref = _run(model, graph)
    monkeypatch.setattr(mps_opt, 'LINEAR_MAX_NNZ', 0)
    rep = mps_opt.enable(model)
    assert rep['linear_fused'] == 0 and rep['linear_e3nn'] > 0
    _close(ref, _run(model, graph))
    mps_opt.disable(model)


@needs_mps
def test_fused_linear_with_bias_and_gradients():
    from sevenn.nn import mps_opt

    lin = Linear(Irreps('3x0e+2x1o'), Irreps('2x0e+1x1o'), biases=True).to('mps')
    x = torch.randn(5, lin.irreps_in.dim, device='mps', requires_grad=True)
    ref = lin(x)
    gref = torch.autograd.grad(ref.pow(2).sum(), x)[0]
    mps_opt._fuse(lin, mps_opt.FusedLinear)
    out = lin(x)
    assert float((ref - out).abs().max()) < 1e-6
    assert (
        float((gref - torch.autograd.grad(out.pow(2).sum(), x)[0]).abs().max())
        < 1e-5
    )


@needs_mps
def test_fused_gate_tables_and_gradients():
    from sevenn.nn import mps_opt

    gate = Gate(
        '3x0e+2x0o',
        [torch.nn.functional.silu, torch.tanh],
        '2x0e+2x0o',
        [torch.nn.functional.silu, torch.tanh],
        '2x1o+2x2e',
    ).to('mps')
    x = torch.randn(6, gate.irreps_in.dim, device='mps', requires_grad=True)
    ref = gate(x)
    gref = torch.autograd.grad(ref.pow(2).sum(), x)[0]
    gate._tables = mps_opt._gate_tables(gate, 'mps')
    gate.__class__ = mps_opt.FusedGate
    out = gate(x)
    assert float((ref - out).abs().max()) < 1e-6
    assert (
        float((gref - torch.autograd.grad(out.pow(2).sum(), x)[0]).abs().max())
        < 1e-5
    )


@needs_mps
def test_unsupported_gate_activation_keeps_e3nn_on_the_gpu():
    from sevenn.nn import mps_opt

    model = _model(act_scalar={'e': 'ssp', 'o': 'tanh'})
    graph = _graph()
    ref = _run(model, graph)
    with pytest.warns(UserWarning, match='gate keeps the e3nn code'):
        rep = mps_opt.enable(model)
    assert (
        rep['gate_fused'] == 0 and rep['gate_e3nn'] == 3 and rep['conv_fused'] == 3
    )
    _close(ref, _run(model, graph))


@needs_mps
@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('bucket', [37, 67, 1024])
def test_pad_edges_are_silent_in_fused_kernels(method, bucket):
    from sevenn.nn import mps_opt

    model = _model(method)
    mps_opt.enable(model)
    at = bulk('NaCl', 'rocksalt', a=5.63).repeat((2, 1, 1))
    at.rattle(0.05, seed=3)
    g0 = dl.unlabeled_atoms_to_graph(at, 4.0)
    padded = pad_edges(g0, 4.0, bucket)
    assert padded is not g0
    num_real = g0['edge_index'].shape[1]
    ref = _run(model, AtomGraphData.from_numpy_dict(dict(g0)))
    model._mps_plan.expect_padded = (num_real, padded['edge_index'].shape[1])
    out = _run(model, AtomGraphData.from_numpy_dict(dict(padded)))
    for k in (KEY.PRED_TOTAL_ENERGY, KEY.PRED_FORCE, KEY.PRED_STRESS):
        assert float((ref[k] - out[k]).abs().max()) < 3e-5 * max(
            float(ref[k].abs().max()), 1.0
        ), k
    # padded but treated as real edges: same result, and all derivatives finite
    out = _run(model, AtomGraphData.from_numpy_dict(dict(padded)))
    for k, v in out.items():
        assert torch.isfinite(v).all(), k
    _close(ref, out)


def _calc_pair(method='direct', **kw):
    cfg = _config(method=method)
    model = _build(cfg)
    a = SevenNetCalculator(
        model, file_type='model_instance', device='mps', mps_opt='off'
    )
    b = SevenNetCalculator(
        model, file_type='model_instance', device='mps', mps_opt='fused', **kw
    )
    return a, b, model


@needs_mps
@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('cutoff', [4.0, 5.3, 6.0])
def test_default_calculator_really_pads_on_the_gpu(method, cutoff):
    model = _build(_config(method=method, cutoff=cutoff))
    calc = SevenNetCalculator(model, file_type='model_instance', device='mps')
    assert calc.mps_runtime.mode == 'fused' and calc.mps_runtime.bucket == 256
    at = bulk('NaCl', 'rocksalt', a=5.63).repeat((2, 1, 1))
    at.rattle(0.05, seed=3)
    at.calc = calc
    at.get_forces()
    assert calc.mps_runtime.plan.num_edges % 256 == 0
    assert (
        calc.mps_runtime.plan.num_real
        == calc.results['num_edges']
        < calc.mps_runtime.plan.num_edges
    )


@needs_mps
def test_calculator_copies_a_shared_instance_and_reports_real_edges():
    a, b, model = _calc_pair()
    assert (
        b.model is not model
        and type(
            next(m for m in model.modules() if hasattr(m, 'convolution'))
        ).__name__
        == 'IrrepsConvolution'
    )
    assert b.mps_runtime.report['conv_fused'] == 3 and a.mps_runtime.mode == 'off'
    at = bulk('NaCl', 'rocksalt', a=5.63).repeat((2, 1, 1))
    at.rattle(0.05, seed=3)
    n = None
    for dx in (0.0, 0.03, 0.2):  # the edge count changes
        at.positions += np.random.default_rng(1).normal(
            0, dx + 1e-3, at.positions.shape
        )
        at.calc = a
        fa, ea = at.get_forces(), at.get_potential_energy()
        n = a.results['num_edges']
        at.calc = b
        fb, eb = at.get_forces(), at.get_potential_energy()
        assert b.results['num_edges'] == n
        assert abs(ea - eb) < 1e-3 and np.abs(fa - fb).max() < 2e-4


@needs_mps
@pytest.mark.parametrize('bucket', [0, 256])
def test_isolated_atoms_without_edges(bucket):
    a, b, _ = _calc_pair(mps_edge_bucket=bucket)
    at = Atoms(
        'NaCl',
        positions=[[0, 0, 0], [30.0, 0, 0]],
        cell=[60.0, 60.0, 60.0],
        pbc=True,
    )
    at.calc = a
    ea, fa = at.get_potential_energy(), at.get_forces()
    at.calc = b
    eb, fb = at.get_potential_energy(), at.get_forces()
    assert abs(ea - eb) < 1e-4 and np.all(fa == 0.0) and np.all(fb == 0.0)
    assert b.results['num_edges'] == 0


@needs_mps
@pytest.mark.parametrize('method', METHODS)
def test_train_mode_runs_the_kernels_and_supports_force_training(
    method, monkeypatch
):
    from sevenn.nn import mps_opt

    ref = _build(_config(method=method)).to('mps')
    ref.set_is_batch_data(False)
    model = _build(_config(method=method)).to('mps')
    model.set_is_batch_data(False)
    rep = mps_opt.enable(model)
    assert rep['conv_fused'] > 0 and rep['conv_e3nn'] == 0
    calls = []
    apply = mps_opt.FusedUvu.apply
    monkeypatch.setattr(
        mps_opt.FusedUvu, 'apply', lambda *a: calls.append(1) or apply(*a)
    )
    graph = _graph()
    ref.train()
    model.train()
    losses = []
    for m in (ref, model):
        out = m(graph.clone().to('mps'))
        loss = out[KEY.PRED_FORCE].pow(2).mean()
        loss.backward()
        losses.append(float(loss.detach().cpu()))
    assert abs(losses[0] - losses[1]) < 1e-5 * losses[0]
    for (n, p), (_, q) in zip(ref.named_parameters(), model.named_parameters()):
        if p.grad is not None:
            assert float((p.grad - q.grad).abs().max()) < 1e-4 * max(
                float(p.grad.abs().max()), 1e-3
            ), n
    assert len(calls) == rep['conv_fused']
    model.eval()  # back to the kernels; weights changed in place are picked up
    with torch.no_grad():
        for p, q in zip(ref.parameters(), model.parameters()):
            p += 1e-3
            q += 1e-3
    ref.eval()
    _close(_run(ref, graph), _run(model, graph))


@needs_mps
def test_train_mode_has_second_derivatives_also_with_unequal_channels():
    from sevenn.nn import mps_opt

    graph = _graph()
    for kw in (
        {},
        dict(
            channel=[8, 8, 8, 8],
            lmax=2,
            irreps_manual=['8x0e', '8x0e+4x1o+2x2e', '8x0e+4x1o+2x2e', '8x0e'],
        ),
    ):
        ref = _build(_config(**kw)).to('mps')
        model = _build(_config(**kw)).to('mps')
        for m in (ref, model):
            m.set_is_batch_data(False)
        rep = mps_opt.enable(model)
        assert rep['conv_fused'] > 0 and rep['conv_e3nn'] == 0
        ref.train()
        model.train()
        g = []
        for m in (ref, model):
            out = m(graph.clone().to('mps'))
            f = out[KEY.PRED_FORCE]
            # a second derivative through the forces, as a force loss needs
            v = torch.autograd.grad(
                f.pow(2).sum(), out[KEY.EDGE_VEC], create_graph=True
            )[0]
            w = next(p for p in m.parameters() if p.requires_grad and p.dim() > 1)
            g.append(torch.autograd.grad(v.sum(), w, allow_unused=True)[0])
        if g[0] is not None:
            assert float((g[0] - g[1]).abs().max()) < 1e-4 * max(
                float(g[0].abs().max()), 1e-3
            )


@needs_mps
@pytest.mark.parametrize('all_types', [8, 0])
def test_train_mode_node_layers_avoid_e3nn_and_equal_it(all_types, monkeypatch):
    from e3nn.nn import Gate as E3Gate
    from e3nn.o3 import FullyConnectedTensorProduct

    from sevenn.nn import mps_opt

    graph = _graph()
    ref = _build(_config()).to('mps')
    model = _build(_config()).to('mps')
    for m in (ref, model):
        m.set_is_batch_data(False)
        m.train()
    mps_opt.enable(model)
    monkeypatch.setattr(
        mps_opt, 'SELFCONN_ALL_TYPES', all_types
    )  # 0: the loop over atom types present
    grads, fell_back = [], []
    for m, fused in ((ref, False), (model, True)):
        # record which classes call e3nn's own forward, with the shape of their input
        if fused:
            for cls in (Linear, FullyConnectedTensorProduct, E3Gate):

                def spy(self, *a, _orig=cls.forward, **k):
                    fell_back.append((type(self).__name__, a[0].dim()))
                    return _orig(self, *a, **k)

                monkeypatch.setattr(cls, 'forward', spy)
        out = m(graph.clone().to('mps'))
        (
            out[KEY.PRED_FORCE].pow(2).mean()
            + out[KEY.PRED_TOTAL_ENERGY].pow(2).mean()
        ).backward()
        grads.append(
            {
                n: p.grad.detach().cpu().clone()
                for n, p in m.named_parameters()
                if p.grad is not None
            }
        )
    # a fused layer may fall back only for inputs its kernels do not take (the
    # dielectric head calls with 3-d input)
    assert not [
        f
        for f in fell_back
        if f[0] in ('FusedLinear', 'FusedSelfConnection', 'FusedGate') and f[1] == 2
    ], fell_back
    assert grads[0].keys() == grads[1].keys()
    for n in grads[0]:
        assert float((grads[0][n] - grads[1][n]).abs().max()) < 1e-4 * max(
            float(grads[0][n].abs().max()), 1e-3
        ), n


def test_weight_map_refuses_a_map_that_is_not_one_weight_per_entry():
    from sevenn.nn.mps_opt import _weight_map

    w = torch.nn.Parameter(torch.ones(3, dtype=torch.float64))
    with pytest.raises(NotImplementedError, match='one weight times a constant'):
        _weight_map(None, w, lambda: torch.stack([w[0] * w[1], w[2]]))


@needs_mps
def test_trainer_switches_on_only_for_the_apple_gpu(monkeypatch):
    from sevenn.nn import mps_opt
    from sevenn.train.trainer import Trainer

    for device, env, expect in (
        ('mps', None, True),
        ('mps', 'off', False),
        ('cpu', None, False),
    ):
        if env is None:
            monkeypatch.delenv('SEVENNET_MPS_OPT', raising=False)
        else:
            monkeypatch.setenv('SEVENNET_MPS_OPT', env)
        tr = Trainer(_build(_config()), [], torch.optim.Adam, device=device)
        assert (
            any(isinstance(m, mps_opt.FusedConvolution) for m in tr.model.modules())
            == expect
        ), (device, env)


@needs_mps
def test_second_derivatives_raise_in_the_linear_kernel():
    from sevenn.nn import mps_opt

    # the convolution kernels have them (see the third-order test); linear and gate
    # kernels do not
    lin = Linear(Irreps('3x0e+2x1o'), Irreps('2x0e+1x1o')).to('mps').eval()
    mps_opt._fuse(lin, mps_opt.FusedLinear)
    x = torch.randn(5, lin.irreps_in.dim, device='mps', requires_grad=True)
    with pytest.raises(
        RuntimeError, match='linear and gate layers have first derivatives only'
    ):
        torch.autograd.grad(lin(x).pow(2).sum(), x, create_graph=True)


@needs_mps
def test_batched_data_equals_single_graphs():
    from torch_geometric.loader.dataloader import Collater

    from sevenn.nn import mps_opt

    model = _model()
    mps_opt.enable(model)
    g1, g2 = _graph(seed=1), _graph(seed=2)
    r1, r2 = _run(model, g1), _run(model, g2)
    model.set_is_batch_data(True)
    out = model(Collater([g1, g2])([g1.clone(), g2.clone()]).to('mps'))
    F = out[KEY.PRED_FORCE].detach().cpu()
    assert (
        float((torch.cat([r1[KEY.PRED_FORCE], r2[KEY.PRED_FORCE]]) - F).abs().max())
        < 3e-5
    )
    E = out[KEY.PRED_TOTAL_ENERGY].detach().cpu()
    assert (
        abs(float(E[0] - r1[KEY.PRED_TOTAL_ENERGY])) < 1e-3
        and abs(float(E[1] - r2[KEY.PRED_TOTAL_ENERGY])) < 1e-3
    )


@needs_mps
def test_stale_padding_announcement_does_not_cut_edges():
    from sevenn.nn.mps_fused import EdgePlan

    idx = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], device='mps')
    p = EdgePlan()
    p.expect_padded = (2, 4)  # same count, but the tail is not pad edges
    p.get(idx, 3)
    assert p.num_real == 4
    p.expect_padded = (2, 9)  # other count
    p.get(idx.clone(), 3)
    assert p.num_real == 4 and p.expect_padded is None


@needs_mps
def test_instance_under_default_file_type_is_not_switched():
    model = _build(_config())
    calc = SevenNetCalculator(model, device='mps')
    assert calc.model is not model and calc.mps_runtime.mode == 'fused'
    assert type(model._modules['0_convolution']).__name__ == 'IrrepsConvolution'


@needs_mps
def test_fused_instance_with_off_runs_e3nn_and_leaves_the_input_fused():
    from sevenn.nn import mps_opt

    model = _model()
    mps_opt.enable(model)
    calc = SevenNetCalculator(
        model, file_type='model_instance', device='mps', mps_opt='off'
    )
    assert type(calc.model._modules['0_convolution']).__name__ == 'IrrepsConvolution'
    assert type(model._modules['0_convolution']).__name__ == 'FusedConvolution'
    calc = SevenNetCalculator(model, device='mps', mps_opt='fused')
    assert calc.mps_runtime.report['conv_fused'] == 3


@needs_mps
def test_deepcopy_after_load_state_dict_uses_the_new_weights():
    import copy

    from sevenn.nn import mps_opt

    graph = _graph()
    other = _build(_config(), seed=7).to('mps')
    other.set_is_batch_data(False)
    other.eval()
    ref = _run(other, graph)
    model = _model()
    mps_opt.enable(model)
    model.load_state_dict(other.state_dict(), strict=True)
    _close(ref, _run(copy.deepcopy(model), graph))


@needs_mps
def test_enable_leaves_the_global_random_generator_alone():
    from sevenn.nn import mps_opt

    model = _model()
    torch.manual_seed(123)
    a = torch.randn(3)
    torch.manual_seed(123)
    mps_opt.enable(model)
    assert torch.equal(a, torch.randn(3))


@needs_mps
def test_too_large_graph_is_refused():
    from sevenn.nn.mps_fused import EdgePlan, FusedTP, FusedUvu

    K = FusedTP(_tp().to('mps'))
    plan = EdgePlan()
    plan.num_edges, plan.num_nodes = 2**28, 10
    x = torch.zeros(1, K.nx, device='mps')
    with pytest.raises(RuntimeError, match='too large'):
        FusedUvu.apply(x, x, x, K, plan)


@needs_mps
def test_edge_plan_notices_new_tensors_and_in_place_changes():
    from sevenn.nn.mps_fused import EdgePlan

    idx = torch.tensor([[0, 1, 1], [1, 0, 2]], device='mps')
    p = EdgePlan()
    p.get(idx, 3)
    assert p.dst.cpu().tolist() == [0, 1, 1]
    idx[0, 0] = 2
    p.get(idx, 3)
    assert p.dst.cpu().tolist() == [2, 1, 1]
    p.get(torch.tensor([[1, 1], [0, 2]], device='mps'), 3)
    assert p.num_edges == 2


@needs_mps
@pytest.mark.parametrize('method', METHODS)
def test_lammps_bridge_equals_the_calculator(method, tmp_path):
    lammps = pytest.importorskip('lammps')
    from sevenn.lammps_external import run_input

    calc = SevenNetCalculator(
        _build(_config(method=method)), file_type='model_instance', device='mps'
    )
    a = 5.63
    at = bulk('NaCl', 'rocksalt', a=a, cubic=True)
    at.rattle(0.05, seed=2)
    data = tmp_path / 'at.data'
    from ase.io import write

    write(
        data, at, format='lammps-data', specorder=['Na', 'Cl'], atom_style='atomic'
    )
    inp = tmp_path / 'in.lmp'
    inp.write_text(f"""units metal
atom_style atomic
boundary p p p
read_data {data}
mass 1 22.99
mass 2 35.45
pair_style zero 4.0
pair_coeff * *
fix sevenn all external pf/callback 1 1
fix_modify sevenn energy yes virial yes
thermo_style custom step pe press pxx pyy pzz pxy pxz pyz
run 0
""")
    lmp = lammps.lammps(cmdargs=['-screen', 'none', '-log', 'none'])
    run_input(lmp, str(inp), calc, ['Na', 'Cl'])
    pe, press = lmp.get_thermo('pe'), lmp.get_thermo('pxx')
    fl = lmp.numpy.extract_atom('f')[: len(at)].copy()
    ids = lmp.numpy.extract_atom('id')[: len(at)].copy()
    lmp.close()
    at.calc = calc
    assert abs(pe - at.get_potential_energy()) < 1e-3
    f = at.get_forces()
    assert np.abs(fl[np.argsort(ids)] - f).max() < 2e-4
    # ASE stress in eV/A^3; LAMMPS pressure in bar, kinetic part is 0 at rest
    assert abs(press - (-at.get_stress()[0] * 1602176.634)) < 2.0


_PLUGIN = str(
    Path(__file__).parents[2] / 'sevenn' / 'pair_e3gnn_mps' / 'e3gnnmpsplugin.so'
)
needs_plugin = pytest.mark.skipif(
    not Path(_PLUGIN).exists(), reason='pair_e3gnn_mps/build_plugin.sh not run'
)


def _pair_lammps(tmp_path, at, coeff):
    lammps = pytest.importorskip('lammps')
    from ase.io import write

    data = tmp_path / 'at.data'
    write(
        data, at, format='lammps-data', specorder=['Na', 'Cl'], atom_style='atomic'
    )
    lmp = lammps.lammps(cmdargs=['-screen', 'none', '-log', 'none'])
    for line in (
        f'plugin load {_PLUGIN}',
        'units metal',
        'atom_style atomic',
        'boundary p p p',
        f'read_data {data}',
        'mass 1 22.99',
        'mass 2 35.45',
        'pair_style e3gnn',
        f'pair_coeff * * {coeff}',
    ):
        lmp.command(line)
    return lmp


@needs_mps
@needs_plugin
def test_pair_style_equals_the_calculator(tmp_path):
    at = bulk('NaCl', 'rocksalt', a=5.63, cubic=True)
    at.rattle(0.05, seed=2)
    lmp = _pair_lammps(tmp_path, at, '7net-0 Na Cl')
    lmp.command('thermo_style custom step pe press pxx pyy pzz pxy pxz pyz')
    lmp.command('compute pea all pe/atom')
    lmp.command('run 0')
    pe, press = lmp.get_thermo('pe'), lmp.get_thermo('pxx')
    ids = lmp.numpy.extract_atom('id')[: len(at)].copy()
    fl = lmp.numpy.extract_atom('f')[: len(at)].copy()
    lmp.close()
    calc = SevenNetCalculator('7net-0', device='mps')
    at.calc = calc
    assert abs(pe - at.get_potential_energy()) < 1e-3
    assert np.abs(fl[np.argsort(ids)] - at.get_forces()).max() < 2e-4
    assert abs(press - (-at.get_stress()[0] * 1602176.634)) < 2.0


@needs_mps
@needs_plugin
@pytest.mark.parametrize(
    'coeff, message',
    [
        ('7net-0 Na', 'not enough elements'),
        ('7net-0 Na Xx', 'unknown'),
        ('deployed.pt Na Cl', 'deployed'),
    ],
)
def test_pair_style_clear_errors(tmp_path, coeff, message):
    at = bulk('NaCl', 'rocksalt', a=5.63, cubic=True)
    with pytest.raises(Exception, match=message):
        _pair_lammps(tmp_path, at, coeff)


def _fused_against_off(at, **kw):
    results = []
    for mode in ('off', 'fused'):
        calc = SevenNetCalculator(
            _build(_config(**kw)),
            file_type='model_instance',
            device='mps',
            mps_opt=mode,
        )
        a = at.copy()
        a.calc = calc
        results.append(
            (a.get_potential_energy(), a.get_forces(), a.get_stress(), calc)
        )
    return results


@needs_mps
@pytest.mark.parametrize(
    'name, at',
    [
        (
            'no_edges',
            Atoms(
                'Na2', positions=[[0, 0, 0], [20, 0, 0]], cell=[40, 40, 40], pbc=True
            ),
        ),
        (
            'one_atom_no_edges',
            Atoms('Cl', positions=[[1, 1, 1]], cell=[30, 30, 30], pbc=True),
        ),
        (
            'isolated_atom_beside_a_dimer',
            Atoms(
                'NaClCl',
                positions=[[0, 0, 0], [2, 0, 0], [15, 15, 15]],
                cell=[30, 30, 30],
                pbc=True,
            ),
        ),
        (
            'one_atom_self_images',
            Atoms('Na', positions=[[0, 0, 0]], cell=[3.1, 3.1, 3.1], pbc=True),
        ),
    ],
)
def test_few_or_no_edges(name, at):
    (e0, f0, s0, _), (e1, f1, s1, _) = _fused_against_off(at)
    assert np.isfinite(e1) and np.isfinite(f1).all() and np.isfinite(s1).all()
    assert abs(e0 - e1) < 1e-4
    assert np.abs(f0 - f1).max() < 1e-4
    assert np.abs(s0 - s1).max() < 1e-4


@needs_mps
@pytest.mark.parametrize('method', METHODS)
def test_lmax_4(method):
    at = bulk('NaCl', 'rocksalt', a=5.63, cubic=True)
    at.rattle(0.05, seed=2)
    (e0, f0, s0, _), (e1, f1, s1, calc) = _fused_against_off(
        at, method=method, lmax=4, channel=4
    )
    report = calc.mps_runtime.report
    assert report['conv_e3nn'] == 0 and report['conv_fused'] > 0
    assert abs(e0 - e1) < 1e-3
    assert np.abs(f0 - f1).max() < 1e-4
    assert np.abs(s0 - s1).max() < 1e-4


@needs_mps
def test_weights_loaded_on_the_cpu_are_used_after_moving_back():
    at = bulk('NaCl', 'rocksalt', a=5.63, cubic=True)
    at.rattle(0.05, seed=2)
    calc = SevenNetCalculator(
        _build(_config()), file_type='model_instance', device='mps', mps_opt='fused'
    )
    a = at.copy()
    a.calc = calc
    e0 = a.get_potential_energy()
    calc.model.to('cpu')
    names = set(dict(calc.model.named_parameters()))
    state = {
        k: v.clone() * 1.01 if k in names else v.clone()
        for k, v in calc.model.state_dict().items()
    }
    calc.model.load_state_dict(state)
    calc.model.to('mps')
    calc.reset()  # ASE would return the cached result for the same geometry
    a = at.copy()
    a.calc = calc
    ref = SevenNetCalculator(
        _build(_config()), file_type='model_instance', device='mps', mps_opt='off'
    )
    ref.model.load_state_dict({k: v.to('mps') for k, v in state.items()})
    b = at.copy()
    b.calc = ref
    assert abs(a.get_potential_energy() - e0) > 1e-2  # the weights did change
    assert abs(a.get_potential_energy() - b.get_potential_energy()) < 1e-3
    assert np.abs(a.get_forces() - b.get_forces()).max() < 1e-4


@needs_mps
def test_enable_refuses_other_float_types():
    from sevenn.nn.mps_opt import enable

    model = _build(_config()).to('mps').to(torch.float16)
    with pytest.raises(ValueError, match='float32'):
        enable(model)


@needs_mps
def test_failed_call_leaves_no_padding_note():
    calc = SevenNetCalculator(
        _build(_config()), file_type='model_instance', device='mps', mps_opt='fused'
    )
    a = bulk('NaCl', 'rocksalt', a=5.63, cubic=True)
    a.calc = calc

    def boom(data):
        raise RuntimeError('boom')

    good = calc.model.forward
    calc.model.forward = boom
    with pytest.raises(RuntimeError, match='boom'):
        a.get_potential_energy()
    assert calc.mps_runtime.plan.expect_padded is None
    calc.model.forward = good
    assert np.isfinite(a.get_potential_energy())


@needs_mps
def test_no_padding_with_a_convolution_left_on_e3nn():
    from sevenn.nn.mps_opt import MpsRuntime, enable

    calc = SevenNetCalculator(
        _build(_config()), file_type='model_instance', device='mps', mps_opt='off'
    )
    real = enable

    def partly(model, parts='all'):
        report = real(model, 'linear+selfconn+gate')
        report['conv_e3nn'] = 1
        return report

    import sevenn.nn.mps_opt as opt

    opt.enable = partly
    try:
        with pytest.warns(UserWarning, match='padding disabled'):
            rt = MpsRuntime(calc.model, 'fused', 256, calc.cutoff)
    finally:
        opt.enable = real
    assert rt.bucket == 0


class _FakeLammps:
    def __init__(self, ranks, units):
        self.ranks, self.units = ranks, units

    def extract_setting(self, name):
        return self.ranks

    def extract_global(self, name):
        return self.units

    def set_fix_external_callback(self, *args):
        pass


def test_external_bridge_refuses_ranks_and_units():
    from sevenn.lammps_external import attach

    with pytest.raises(ValueError, match='one MPI rank'):
        attach(_FakeLammps(2, 'metal'), None, ['Na'])
    with pytest.raises(ValueError, match='units metal'):
        attach(_FakeLammps(1, 'real'), None, ['Na'])
    attach(_FakeLammps(1, 'metal'), None, ['Na'])


@needs_mps
def test_external_bridge_virial_on_a_slab(tmp_path):
    lammps = pytest.importorskip('lammps')
    from ase.io import write

    from sevenn.lammps_external import run_input

    calc = SevenNetCalculator('7net-0', device='mps')
    at = bulk('NaCl', 'rocksalt', a=5.63, cubic=True)
    at.cell[2, 2] += 10.0
    at.pbc = [True, True, False]
    at.rattle(0.05, seed=2)
    at.positions += 0.3
    write(
        tmp_path / 'at.data',
        at,
        format='lammps-data',
        specorder=['Na', 'Cl'],
        atom_style='atomic',
    )
    inp = tmp_path / 'in.lmp'
    inp.write_text(f"""units metal
atom_style atomic
boundary p p f
read_data {tmp_path / 'at.data'}
mass 1 22.99
mass 2 35.45
pair_style zero 5.0
pair_coeff * *
fix sevenn all external pf/callback 1 1
fix_modify sevenn energy yes virial yes
thermo_style custom step pe pxx
run 0
""")
    lmp = lammps.lammps(cmdargs=['-screen', 'none', '-log', 'none'])
    run_input(lmp, str(inp), calc, ['Na', 'Cl'])
    press = lmp.get_thermo('pxx')
    x = lmp.numpy.extract_atom('x')[: len(at)].copy()
    ids = lmp.numpy.extract_atom('id')[: len(at)].copy()
    lmp.close()
    assert abs(press) > 1.0
    at.positions = x[np.argsort(ids)]
    at.calc = calc
    assert abs(press - (-at.get_stress()[0] * 1602176.634)) < 2.0


@needs_mps
@needs_plugin
@pytest.mark.parametrize(
    'line, message',
    [('units real', 'units metal'), ('neigh_modify exclude type 1 2', 'exclude')],
)
def test_pair_style_refuses_what_it_cannot_honour(tmp_path, line, message):
    lammps = pytest.importorskip('lammps')
    from ase.io import write

    at = bulk('NaCl', 'rocksalt', a=5.63, cubic=True)
    write(
        tmp_path / 'at.data',
        at,
        format='lammps-data',
        specorder=['Na', 'Cl'],
        atom_style='atomic',
    )
    lmp = lammps.lammps(cmdargs=['-screen', 'none', '-log', 'none'])
    lines = [
        f'plugin load {_PLUGIN}',
        'units metal',
        'atom_style atomic',
        f'read_data {tmp_path / "at.data"}',
        'mass 1 22.99',
        'mass 2 35.45',
        'pair_style e3gnn',
        'pair_coeff * * 7net-0 Na Cl',
    ]
    if line == 'units real':
        lines[1] = line
    else:
        lines.append(line)
    for l in lines:
        lmp.command(l)
    with pytest.raises(Exception, match=message):
        lmp.command('run 0')


@needs_mps
@needs_plugin
def test_pair_style_refuses_an_empty_box():
    lammps = pytest.importorskip('lammps')
    lmp = lammps.lammps(cmdargs=['-screen', 'none', '-log', 'none'])
    for l in (
        f'plugin load {_PLUGIN}',
        'units metal',
        'atom_style atomic',
        'region b block 0 10 0 10 0 10',
        'create_box 2 b',
        'mass 1 22.99',
        'mass 2 35.45',
        'pair_style e3gnn',
        'pair_coeff * * 7net-0 Na Cl',
    ):
        lmp.command(l)
    with pytest.raises(Exception, match='at least one atom'):
        lmp.command('run 0')


@needs_mps
def test_autocast_never_reaches_the_kernels():
    at = bulk('NaCl', 'rocksalt', a=5.63, cubic=True)
    at.rattle(0.05, seed=2)
    calc = SevenNetCalculator(
        _build(_config()), file_type='model_instance', device='mps', mps_opt='fused'
    )
    at.calc = calc
    with torch.autocast(device_type='mps', dtype=torch.float16):
        try:
            forces = at.get_forces()
        except RuntimeError:
            return  # as the e3nn code does
    assert np.isfinite(forces).all()


def _fused_convolution():
    calc = SevenNetCalculator(
        _build(_config()), file_type='model_instance', device='mps', mps_opt='fused'
    )
    conv = calc.model._modules['0_convolution']
    K = conv._kernels
    idx = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], device='mps')
    x = torch.randn(4, K.nx, device='mps')
    y = torch.randn(4, K.ny, device='mps')
    w = torch.randn(4, K.nw, device='mps')
    return conv, K, idx, x, y, w


@needs_mps
def test_kernels_refuse_other_types_and_shapes():
    from sevenn.nn.mps_fused import EdgePlan, FusedUvu

    _, K, idx, x, y, w = _fused_convolution()
    plan = EdgePlan().get(idx, 4)
    FusedUvu.apply(x, y, w, K, plan)
    for a, b, c in (
        (x, y.half(), w),
        (x, y, w.half()),
        (x[:3], y, w),
        (x, y[:3], w),
        (x, y, w[:, :-1]),
    ):
        with pytest.raises(ValueError, match='expected float32'):
            FusedUvu.apply(a, b, c, K, plan)


@needs_mps
def test_float16_edge_input_takes_the_reference_path():
    conv, K, idx, x, y, _ = _fused_convolution()
    y16 = y.half()
    wi = torch.randn(4, conv.weight_nn.hs[0], device='mps')
    mixed = conv(
        {
            KEY.EDGE_IDX: idx,
            KEY.NODE_FEATURE: x,
            KEY.EDGE_ATTR: y16,
            KEY.EDGE_EMBEDDING: wi,
        }
    )[KEY.NODE_FEATURE]
    same = conv(
        {
            KEY.EDGE_IDX: idx,
            KEY.NODE_FEATURE: x,
            KEY.EDGE_ATTR: y16.float(),
            KEY.EDGE_EMBEDDING: wi,
        }
    )[KEY.NODE_FEATURE]
    assert float((mixed - same).abs().max()) < 1e-4


def test_edge_plan_refuses_indices_out_of_range():
    from sevenn.nn.mps_fused import EdgePlan

    for bad in ([[0, 4], [1, 0]], [[0, -1], [1, 0]], [[0, 1], [5, 0]]):
        with pytest.raises(ValueError, match='out of range'):
            EdgePlan().get(torch.tensor(bad), 4)


@needs_mps
def test_module_without_tables_stays_as_it_was():
    from e3nn.o3 import FullyConnectedTensorProduct

    from sevenn.nn.mps_opt import FusedSelfConnection, _fuse

    tp = FullyConnectedTensorProduct('4x0e', '2x0e', '4x1o')  # no paths
    with pytest.raises(NotImplementedError):
        _fuse(tp, FusedSelfConnection)
    assert type(tp) is FullyConnectedTensorProduct


@needs_mps
def test_output_irrep_with_more_channels_is_refused():
    from sevenn.nn.mps_fused import FusedTP

    tp = TensorProduct(
        '2x0e',
        '1x0e',
        '2x0e+4x1o',
        [(0, 0, 0, 'uvu', True)],
        internal_weights=False,
        shared_weights=False,
    )
    with pytest.raises(NotImplementedError, match='more channels'):
        FusedTP(tp)


@needs_mps
def test_self_connection_uses_e3nn_when_types_carry_gradients():
    calc = SevenNetCalculator(
        _build(_config()), file_type='model_instance', device='mps', mps_opt='fused'
    )
    tp = calc.model._modules['0_self_connection_intro'].fc_tensor_product
    x = torch.randn(3, tp.irreps_in1.dim, device='mps')
    y = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], device='mps', requires_grad=True
    )
    tp(x, y).sum().backward()
    assert y.grad is not None and float(y.grad.abs().max()) > 0


def test_pair_coeff_parser_says_what_is_wrong():
    from sevenn.lammps_pair import _parse

    with pytest.raises(ValueError, match='modal requires'):
        _parse(['modal'])
    # the NVIDIA pair style of this version has a Cartesian field only
    for words in (['efield_reduced', '1', '0', '0'], ['efield', '0', '0', '1',
                  'efield_mode', 'reduced']):
        with pytest.raises(ValueError, match='not available in this version'):
            _parse(words + ['Na', 'Cl'])
    assert _parse(['efield', '0', '0', '0.05', 'enforce_asr', 'no', 'Na'])[
        'efield'
    ] == [0.0, 0.0, 0.05]
    assert _parse(['modal', 'mpa', 'Zr', 'O'])['symbols'] == ['Zr', 'O']


@needs_mps
def test_driver_with_no_atoms_gives_zero():
    from sevenn.lammps_pair import PairDriver

    box = ((0.0, 0.0, 0.0), (10.0, 10.0, 10.0), 0.0, 0.0, 0.0, (0, 0, 0))
    driver = PairDriver(['7net-0', 'Na', 'Cl'], 2, box)
    assert (
        driver.compute(
            0,
            memoryview(b''),
            memoryview(bytearray()),
            memoryview(b''),
            None,
            memoryview(bytearray(48)),
            box,
        )
        == 0.0
    )


def test_run_input_joins_continued_lines(tmp_path):
    from sevenn.lammps_external import run_input

    class Fake(_FakeLammps):
        def __init__(self):
            super().__init__(1, 'metal')
            self.seen = []

        def command(self, line):
            self.seen.append(line)

    path = tmp_path / 'in.lmp'
    path.write_text(
        'units metal  # comment\n\nfix sevenn all external &\n   pf/callback 1 '
        '1\nprint "Step #1"\n'
    )
    lmp = Fake()
    run_input(lmp, str(path), None, ['Na'])
    assert any(
        ' '.join(c.split()) == 'fix sevenn all external pf/callback 1 1'
        for c in lmp.seen
    )
    assert 'print "Step #1"' in lmp.seen


def test_patch_lammps_reports_failure():
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[2]))
    p = subprocess.run(
        [
            sys.executable,
            '-m',
            'sevenn.main.sevenn_patch_lammps',
            '/no/such/folder',
            '--enable_mps',
        ],
        capture_output=True,
        env=env,
    )
    assert p.returncode != 0


# ---- edge cases ----


@needs_mps
def test_enable_twice_is_the_same_as_once():
    from sevenn.nn import mps_opt

    model = _model()
    graph = _graph()
    ref = _run(model, graph)
    first = mps_opt.enable(model, parts='conv')
    second = mps_opt.enable(model)
    third = mps_opt.enable(model)
    assert first['linear_fused'] == 0 and second == third
    assert second['selfconn_fused'] > 0 and second['selfconn_e3nn'] == 0
    convs = [m for m in model.modules() if isinstance(m, mps_opt.FusedConvolution)]
    assert convs and all(m._plan is model._mps_plan for m in convs)
    _close(ref, _run(model, graph))


@needs_mps
def test_declined_self_connection_is_left_as_it_was():
    from sevenn.nn import mps_opt
    from sevenn.nn.self_connection import SelfConnectionIntro

    class Custom(torch.nn.Module):
        def forward(self, x, y):
            return x

    model = _model()
    sc = [m for m in model.modules() if type(m) is SelfConnectionIntro][0]
    custom = Custom()
    sc.fc_tensor_product = custom
    with pytest.warns(UserWarning, match='self-connection keeps'):
        mps_opt.enable(model)
    assert sc.fc_tensor_product is custom and type(custom) is Custom


@needs_mps
def test_float64_shift_and_scale_go_to_float32_on_mps():
    # SevenNet 0.13.1 keeps the energy shift and scale in float64
    model = _build(_config())
    rescale = [m for m in model.modules() if hasattr(m, 'shift')]
    assert rescale
    ref = {}
    for m in rescale:
        m.shift.data = m.shift.data.double()
        m.scale.data = m.scale.data.double()
        ref[m] = (m.shift.data.clone(), m.scale.data.clone())
    model.to('mps')
    for m in rescale:
        assert m.shift.dtype == m.scale.dtype == torch.float32
        assert m.shift.device.type == 'mps'
        assert torch.allclose(m.shift.cpu().double(), ref[m][0])
    model.to('cpu')
    model.to(torch.float64)  # elsewhere float64 stays
    assert all(m.shift.dtype == torch.float64 for m in rescale)


@needs_mps
def test_conversions_drop_the_derived_tables():
    import io

    from sevenn.nn import mps_opt

    graph = _graph()
    # a float16 round trip rounds the weights in place, keeping identity and version
    model = _model()
    mps_opt.enable(model)
    _run(model, graph)
    model.half()
    model.float()
    ref = _build(_config()).to('mps').eval()
    ref.set_is_batch_data(False)
    ref.load_state_dict(model.state_dict())
    _close(_run(ref, graph), _run(model, graph))
    # a pickled fused model loaded on the CPU and moved back, eval and train
    buf = io.BytesIO()
    torch.save(model, buf)
    buf.seek(0)
    back = torch.load(buf, map_location='cpu', weights_only=False).to('mps')
    _close(_run(ref, graph), _run(back, graph))
    back.train()
    ref.train()
    _close(_run(ref, graph), _run(back, graph))


@needs_mps
def test_autocast_takes_the_reference_path_in_node_layers():
    from sevenn.nn import mps_opt

    species = ('H', 'Li', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Cl')
    model = _build(_config(species=species)).to('mps').train()
    mps_opt.enable(model)
    sc = [m for m in model.modules() if isinstance(m, mps_opt.FusedSelfConnection)]
    assert sc and sc[0].irreps_in2.dim > mps_opt.SELFCONN_ALL_TYPES
    x = torch.randn(6, sc[0].irreps_in1.dim, device='mps')
    y = torch.zeros(6, sc[0].irreps_in2.dim, device='mps')
    y[:, 3] = 1.0
    with torch.autocast('mps', dtype=torch.float16):
        out = sc[0](x, y)
        want = mps_opt.FullyConnectedTensorProduct.forward(sc[0], x, y)
    assert out.dtype == want.dtype and torch.allclose(out, want)


@needs_mps
@pytest.mark.parametrize('many_types', [False, True])
def test_eval_after_training_uses_the_new_weights(many_types):
    from sevenn.nn import mps_opt

    species = (
        ('H', 'Li', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Cl')
        if many_types
        else ('Na', 'Cl')
    )
    cfg = _config(species=species, use_bias_in_linear=True)
    model = _build(cfg).to('mps')
    model.set_is_batch_data(False)
    mps_opt.enable(model)
    graph = _graph()
    ref = _build(cfg).to('mps').eval()
    ref.set_is_batch_data(False)
    for _ in range(2):  # the second epoch only updates the values of the map
        model.train()
        out = model(graph.clone().to('mps'))
        out[KEY.PRED_TOTAL_ENERGY].sum().backward()  # builds the train tables
        with torch.no_grad():  # an optimizer step
            for p in model.parameters():
                p.add_(0.01 * torch.randn_like(p))
        model.eval()
        ref.load_state_dict(model.state_dict())
        _close(_run(ref, graph), _run(model, graph))
    trained = [
        m
        for m in model.modules()
        if isinstance(m, (mps_opt.FusedLinear, mps_opt.FusedSelfConnection))
    ]
    # the rebuild used the train tables (layers called with 3-d input have none)
    with_tables = sum(m.__dict__.get('_train') is not None for m in trained)
    assert with_tables > len(trained) // 2


@needs_mps
def test_trainer_keeps_e3nn_without_compile_shader(monkeypatch):
    from sevenn.nn import mps_opt
    from sevenn.train.trainer import Trainer

    monkeypatch.delenv('SEVENNET_MPS_OPT', raising=False)
    monkeypatch.delattr(torch.mps, 'compile_shader')
    with pytest.warns(UserWarning, match='compile_shader'):
        tr = Trainer(_build(_config()), [], torch.optim.Adam, device='mps')
    assert not any(isinstance(m, mps_opt._FUSED) for m in tr.model.modules())


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='needs an Apple GPU'
)
def test_old_pytorch_is_refused_with_a_clear_error(monkeypatch):
    # PyTorch before 2.6 has no scatter_reduce on MPS: refused up front
    from sevenn.nn import mps_opt
    from sevenn.train.trainer import Trainer

    monkeypatch.setattr(mps_opt, '_MPS_TORCH_OK', False)
    for opt in (None, 'off'):
        with pytest.raises(RuntimeError, match='PyTorch 2.6 or later'):
            SevenNetCalculator(
                _build(_config()), file_type='model_instance', device='mps',
                mps_opt=opt,
            )
    with pytest.raises(RuntimeError, match='PyTorch 2.6 or later'):
        Trainer(_build(_config()), [], torch.optim.Adam, device='mps')


@needs_mps
def test_type_index_must_be_int32():
    from sevenn.nn.mps_fused import CsrApply, CsrMap

    eye = (np.arange(2), np.arange(2), np.ones(2))
    M = CsrMap([eye, eye], np.zeros(2), 2, 2, torch.device('mps'))
    x = torch.ones(4, 2, device='mps')
    with pytest.raises(ValueError, match='int32'):
        CsrApply.apply(x, M, torch.ones(4, dtype=torch.int64, device='mps'))
    assert CsrApply.apply(x, M, torch.ones(4, dtype=torch.int32, device='mps')).shape


def test_external_bridge_stops_when_the_calculator_fails(tmp_path):
    lammps = pytest.importorskip('lammps')
    from ase.calculators.calculator import Calculator, all_changes
    from ase.io import write

    from sevenn.lammps_external import run_input

    class Failing(Calculator):
        implemented_properties = ['energy', 'forces', 'stress']
        calls = 0

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            Failing.calls += 1
            if Failing.calls == 3:
                raise RuntimeError('model failed')
            self.results = dict(
                energy=0.0, forces=np.zeros((len(atoms), 3)), stress=np.zeros(6)
            )

    at = bulk('NaCl', 'rocksalt', a=5.63, cubic=True)
    write(
        tmp_path / 'at.data',
        at,
        format='lammps-data',
        specorder=['Na', 'Cl'],
        atom_style='atomic',
    )
    inp = tmp_path / 'in.lmp'
    inp.write_text(f"""units metal
atom_style atomic
boundary p p p
read_data {tmp_path / 'at.data'}
mass 1 22.99
mass 2 35.45
pair_style zero 4.0
pair_coeff * *
velocity all create 300 1
fix nve all nve
fix sevenn all external pf/callback 1 1
fix_modify sevenn energy yes virial yes
run 20
print "not reached"
""")
    lmp = lammps.lammps(cmdargs=['-screen', 'none', '-log', 'none'])
    with pytest.raises(RuntimeError, match='model failed'):
        run_input(lmp, str(inp), Failing(), ['Na', 'Cl'])
    assert lmp.extract_global('ntimestep') < 5 and Failing.calls == 3
    lmp.close()


@needs_mps
@pytest.mark.parametrize('opt', ['fused', 'off', 'flag'])
def test_sevenn_inference_on_the_apple_gpu(opt, tmp_path, monkeypatch):
    from unittest import mock

    from sevenn.main.sevenn_inference import main as inference_main
    from sevenn.nn import mps_opt

    data = Path(__file__).parents[1] / 'data'
    if opt == 'off':
        monkeypatch.setenv('SEVENNET_MPS_OPT', 'off')
    else:
        monkeypatch.delenv('SEVENNET_MPS_OPT', raising=False)
    reports = []
    enable = mps_opt.enable
    monkeypatch.setattr(mps_opt, 'enable', lambda m: reports.append(enable(m)))
    out = tmp_path / 'inference_results'
    device = ['--enable_mps'] if opt == 'flag' else ['--device', 'mps']
    argv = ['sevenn_inference', '--output', str(out), '--batch', '2', *device,
            '7net-0', str(data / 'systems' / 'hfo2.extxyz')]
    with mock.patch('sys.argv', argv):
        inference_main()

    def errors(path):
        lines = (path / 'errors.txt').read_text().splitlines()
        return np.array([float(ln.split(':')[-1]) for ln in lines])

    want = errors(data / 'inferences' / 'snet0_on_hfo2')
    assert np.allclose(errors(out), want, rtol=1e-4, atol=1e-6)
    if opt != 'off':
        assert len(reports) == 1 and reports[0]['conv_fused'] > 0
        assert reports[0]['conv_e3nn'] == 0
    else:
        assert not reports


@pytest.mark.parametrize(
    'argv, message',
    [
        (['inference', '-mps', '-flashTP', 'x.pth', 'y.xyz'], 'cannot be combined'),
        (['inference', '-mps', '-d', 'cpu', 'x.pth', 'y.xyz'], 'not cpu'),
        (['train', '-mps', '-cueq', 'input.yaml'], 'cannot be combined'),
    ],
)
def test_enable_mps_flag_refuses_other_backends(argv, message, tmp_path):
    from unittest import mock

    from sevenn.main.sevenn import main as sevenn_main

    os.chdir(tmp_path)
    with mock.patch('sys.argv', ['sevenn'] + argv):
        with pytest.raises(ValueError, match=message):
            sevenn_main()


@needs_mps
def test_sevenn_train_with_enable_mps(tmp_path, monkeypatch):
    from unittest import mock

    import yaml

    from sevenn.logger import Logger
    from sevenn.main.sevenn import main as sevenn_main
    from sevenn.nn import mps_opt

    monkeypatch.delenv('SEVENNET_MPS_OPT', raising=False)
    reports = []
    enable = mps_opt.enable
    monkeypatch.setattr(mps_opt, 'enable', lambda m: reports.append(enable(m)))
    root = Path(__file__).parents[2]
    data = str(root / 'tests' / 'data' / 'systems' / 'hfo2.extxyz')
    cfg = yaml.safe_load(open(root / 'sevenn' / 'presets' / 'base.yaml'))
    cfg['train']['epoch'] = 1
    cfg['data']['load_trainset_path'] = data
    cfg['data'].pop('load_testset_path', None)
    cfg['data']['load_validset_path'] = data
    yml = tmp_path / 'input.yaml'
    yml.write_text(yaml.dump(cfg))
    Logger().switch_file(str(tmp_path / 'log.sevenn'))
    argv = ['sevenn', 'train', '-w', str(tmp_path), '--enable_mps', str(yml)]
    with mock.patch('sys.argv', argv):
        sevenn_main()
    assert len(reports) == 1 and reports[0]['conv_fused'] > 0
    ckpt = torch.load(tmp_path / 'checkpoint_best.pth', weights_only=False)
    assert ckpt['config']['device'] == 'mps'
