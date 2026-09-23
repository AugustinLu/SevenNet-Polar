"""Batched and unbatched reductions must agree.

Regression tests for AtomReduce(reduce='mean'), used by the direct dielectric
head. In batched mode the reduction scatters into a zero-initialised tensor;
with torch's default ``include_self=True`` the initial zero is counted as an
extra element, so the batched "mean" was sum/(N+1) while the unbatched path
(ASE calculator, LAMMPS) used sum/N. Models trained with the batched path
therefore predicted eps_inf too large by (N+1)/N at inference, and the error
depended on the training cell sizes.
"""

import pytest
import torch
from ase.build import bulk
from torch_geometric.loader.dataloader import Collater

import sevenn._keys as KEY
import sevenn.train.dataload as dl
from sevenn.atom_graph_data import AtomGraphData
from sevenn.model_build import build_E3_equivariant_model
from sevenn.nn.flash_helper import is_flash_available
from sevenn.nn.linear import AtomReduce
from sevenn.util import chemical_species_preprocess

CUTOFF = 4.0


# -------------------------------------------------------------------- AtomReduce
def _random_nodes(sizes, feat_shape, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn((sum(sizes),) + feat_shape, generator=gen, dtype=torch.float64)
    batch = torch.cat(
        [torch.full((n,), i, dtype=torch.long) for i, n in enumerate(sizes)]
    )
    return x, batch


def _reference(x, reduce):
    return x.sum(dim=0) if reduce == 'sum' else x.mean(dim=0)


@pytest.mark.parametrize('scripted', [False, True])
@pytest.mark.parametrize('reduce', ['sum', 'mean'])
@pytest.mark.parametrize('feat_shape', [(1,), (6,)])
@pytest.mark.parametrize('sizes', [[5], [20], [5, 20, 47], [1, 96, 31, 2]])
def test_atom_reduce_batched_equals_unbatched(scripted, reduce, feat_shape, sizes):
    x, batch = _random_nodes(sizes, feat_shape)

    batched = AtomReduce('x', 'y', reduce=reduce)
    unbatched = AtomReduce('x', 'y', reduce=reduce)
    unbatched._is_batch_data = False
    if scripted:
        batched = torch.jit.script(batched)
        unbatched = torch.jit.script(unbatched)

    out_batched = batched({'x': x.clone(), KEY.BATCH: batch})['y']
    assert out_batched.shape[0] == len(sizes)

    start = 0
    for i, n in enumerate(sizes):
        nodes = x[start:start + n]
        start += n
        out_unbatched = unbatched({'x': nodes.clone()})['y']
        ref = _reference(nodes, reduce)
        assert torch.allclose(out_batched[i].reshape(-1), ref.reshape(-1))
        assert torch.allclose(out_unbatched.reshape(-1), ref.reshape(-1))


# ------------------------------------------------------------ direct polar model
def _direct_polar_model(use_flash=False, is_parity=True, device='cpu', channel=4):
    config = {
        'cutoff': CUTOFF,
        'channel': channel,
        'radial_basis': {'radial_basis_name': 'bessel'},
        'cutoff_function': {'cutoff_function_name': 'poly_cut'},
        'interaction_type': 'nequip',
        'lmax': 2,
        'is_parity': is_parity,
        'num_convolution_layer': 2,
        'weight_nn_hidden_neurons': [16, 16],
        'act_radial': 'silu',
        'act_scalar': {'e': 'silu', 'o': 'tanh'},
        'act_gate': {'e': 'silu', 'o': 'tanh'},
        'conv_denominator': 30.0,
        'train_denominator': False,
        'self_connection_type': 'nequip',
        'shift': 0.0,
        'scale': 1.0,
        'train_shift_scale': False,
        'irreps_manual': False,
        'lmax_edge': -1,
        'lmax_node': -1,
        'readout_as_fcn': False,
        'use_bias_in_linear': False,
        '_normalize_sph': True,
        KEY.IS_TRAIN_BEC: True,
        KEY.IS_TRAIN_DIELECTRIC: True,
        KEY.USE_FLASH_TP: use_flash,
    }
    config.update(**chemical_species_preprocess(['Na', 'Cl']))
    torch.manual_seed(0)
    return build_E3_equivariant_model(config, parallel=False).to(device)


def _structures():
    prim = bulk('NaCl', 'rocksalt', a=5.63)
    conv = bulk('NaCl', 'rocksalt', a=5.63, cubic=True)
    rattled = prim.repeat((2, 2, 2))
    rattled.rattle(stdev=0.05, seed=1)
    return {'prim': prim, 'conv': conv, 'prim_222': prim.repeat((2, 2, 2)),
            'rattled_222': rattled}


def _graphs(atoms_list, device='cpu'):
    return [
        AtomGraphData.from_numpy_dict(
            dl.unlabeled_atoms_to_graph(at, CUTOFF)
        ).to(device)
        for at in atoms_list
    ]


def _predict(model, atoms_list, batched, device='cpu'):
    graphs = _graphs(atoms_list, device)
    model.set_is_batch_data(batched)
    if batched:
        out = model(Collater(graphs)(graphs))
        diel = out[KEY.PRED_DIELECTRIC_TENSOR].detach().reshape(len(graphs), -1)
        return [d for d in diel], out[KEY.PRED_BORN_EFFECTIVE_CHARGES].detach()
    diel, bec = [], []
    for g in graphs:
        out = model(g)
        diel.append(out[KEY.PRED_DIELECTRIC_TENSOR].detach().reshape(-1))
        bec.append(out[KEY.PRED_BORN_EFFECTIVE_CHARGES].detach())
    return diel, torch.cat(bec)


def test_direct_polar_batched_equals_unbatched():
    model = _direct_polar_model()
    atoms = list(_structures().values())   # 2, 8, 16, 16 atoms in one batch
    diel_b, bec_b = _predict(model, atoms, batched=True)
    diel_u, bec_u = _predict(model, atoms, batched=False)
    for db, du in zip(diel_b, diel_u):
        assert torch.allclose(db, du, rtol=1e-5, atol=1e-6), (db, du)
    assert torch.allclose(bec_b, bec_u, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('batched', [False, True])
def test_direct_dielectric_is_intensive(batched):
    """A perfect supercell has identical per-atom features: eps_inf is unchanged."""
    model = _direct_polar_model()
    s = _structures()
    diel, _ = _predict(model, [s['prim'], s['conv'], s['prim_222']], batched=batched)
    assert torch.allclose(diel[0], diel[1], rtol=1e-5, atol=1e-6), (diel[0], diel[1])
    assert torch.allclose(diel[0], diel[2], rtol=1e-5, atol=1e-6), (diel[0], diel[2])


# ---------------------------------------------------------------- CUDA / FlashTP
_needs_flash = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_flash_available()),
    reason='CUDA and flashTP required',
)


@_needs_flash
@pytest.mark.parametrize('use_flash', [False, True])
def test_direct_polar_batched_equals_unbatched_cuda(use_flash):
    # is_parity=False and channel=32 as in test_flash.py (flashTP requirements)
    model = _direct_polar_model(
        use_flash=use_flash, is_parity=False, device='cuda', channel=32
    )
    atoms = list(_structures().values())
    diel_b, bec_b = _predict(model, atoms, batched=True, device='cuda')
    diel_u, bec_u = _predict(model, atoms, batched=False, device='cuda')
    for db, du in zip(diel_b, diel_u):
        assert torch.allclose(db, du, rtol=1e-5, atol=1e-6), (db, du)
    assert torch.allclose(bec_b, bec_u, rtol=1e-5, atol=1e-5)
    s = _structures()
    diel_i, _ = _predict(
        model, [s['prim'], s['prim_222']], batched=True, device='cuda'
    )
    assert torch.allclose(diel_i[0], diel_i[1], rtol=1e-5, atol=1e-6)


@_needs_flash
@pytest.mark.parametrize('batched', [False, True])
def test_direct_polar_flash_matches_e3nn(batched):
    atoms = list(_structures().values())
    e3nn_model = _direct_polar_model(
        use_flash=False, is_parity=False, device='cuda', channel=32
    )
    flash_model = _direct_polar_model(
        use_flash=True, is_parity=False, device='cuda', channel=32
    )
    diel_e, bec_e = _predict(e3nn_model, atoms, batched=batched, device='cuda')
    diel_f, bec_f = _predict(flash_model, atoms, batched=batched, device='cuda')
    for de, df in zip(diel_e, diel_f):
        assert torch.allclose(de, df, rtol=1e-5, atol=1e-5), (de, df)
    assert torch.allclose(bec_e, bec_f, rtol=1e-5, atol=1e-5)
