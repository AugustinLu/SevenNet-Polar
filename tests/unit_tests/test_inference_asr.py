"""`sevenn inference --enforce_asr`: per-structure acoustic sum rule projection."""

import argparse

import torch
from e3nn.io import CartesianTensor
from test_atom_reduce import _direct_polar_model, _graphs, _structures
from torch_geometric.loader.dataloader import Collater

import sevenn._keys as KEY
from sevenn.main.sevenn_inference import add_args
from sevenn.scripts.inference import _bec_to_cartesian, project_bec_asr
from sevenn.util import to_atom_graph_list


def _to_cartesian(bec_irreps):
    ct = CartesianTensor('ij')
    return ct.to_cartesian(bec_irreps, ct.reduced_tensor_products())


def _batched_output(model, atoms_list):
    graphs = _graphs(atoms_list)
    model.set_is_batch_data(True)
    return model(Collater(graphs)(graphs))


def test_project_bec_asr_zero_sum_per_structure():
    model = _direct_polar_model()
    s = _structures()
    atoms = [s['prim'], s['rattled_222'], s['conv']]   # 2, 16, 8 atoms
    sizes = [len(a) for a in atoms]
    out = _batched_output(model, atoms)
    raw = _to_cartesian(out[KEY.PRED_BORN_EFFECTIVE_CHARGES].detach().clone())

    project_bec_asr(out)
    proj = _to_cartesian(out[KEY.PRED_BORN_EFFECTIVE_CHARGES].detach())

    start = 0
    for n in sizes:
        z_raw, z_proj = raw[start:start + n], proj[start:start + n]
        start += n
        assert torch.allclose(z_proj.sum(0), torch.zeros(3, 3), atol=1e-5)
        # same as Z* -= mean(Z*) on the Cartesian tensors of this structure
        assert torch.allclose(z_proj, z_raw - z_raw.mean(0), atol=1e-5)


def test_project_bec_asr_unbatched_output():
    model = _direct_polar_model()
    graph = _graphs([_structures()['rattled_222']])[0]
    model.set_is_batch_data(False)
    out = model(graph)
    raw = _to_cartesian(out[KEY.PRED_BORN_EFFECTIVE_CHARGES].detach().clone())
    project_bec_asr(out)
    proj = _to_cartesian(out[KEY.PRED_BORN_EFFECTIVE_CHARGES].detach())
    assert torch.allclose(proj, raw - raw.mean(0), atol=1e-5)


def test_predicted_bec_split_per_structure_and_cartesian():
    # `sevenn inference` writes per_atom.csv from to_atom_graph_list
    model = _direct_polar_model()
    s = _structures()
    atoms = [s['prim'], s['rattled_222'], s['conv']]
    out = _batched_output(model, atoms)
    bec = out[KEY.PRED_BORN_EFFECTIVE_CHARGES].detach()
    graphs = to_atom_graph_list(out.detach())
    start = 0
    for g, a in zip(graphs, atoms):
        n = len(a)
        got = g[KEY.PRED_BORN_EFFECTIVE_CHARGES]
        assert torch.equal(got, bec[start:start + n])
        cart = _bec_to_cartesian(got.numpy())
        assert cart.shape == (n, 3, 3)
        assert torch.allclose(
            torch.as_tensor(cart), _to_cartesian(bec[start:start + n]), atol=1e-6
        )
        start += n


def test_enforce_asr_off_by_default():
    parser = argparse.ArgumentParser()
    add_args(parser)
    assert parser.parse_args(['cp.pth', 'data.xyz']).enforce_asr is False
    args = parser.parse_args(['cp.pth', 'data.xyz', '--enforce_asr'])
    assert args.enforce_asr is True
