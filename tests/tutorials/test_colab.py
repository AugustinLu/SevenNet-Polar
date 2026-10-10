"""Integrity and scientific plumbing for the BEC tutorials."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ase.io import read

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    'polar_tutorial_support', ROOT / 'tutorials/colab/support.py'
)
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)


def test_notebooks_are_valid_and_compile():
    nbformat = pytest.importorskip('nbformat')
    paths = sorted((ROOT / 'tutorials/colab').glob('*.ipynb'))
    assert len(paths) == 2
    for path in paths:
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'code':
                compile(cell.source, f'{path.name}:cell{i}', 'exec')
                assert cell.execution_count is None
                assert not cell.outputs


def test_split_roundtrip_and_config(tmp_path):
    structures, paths, _, manifest, _ = support.prepare_data(ROOT, tmp_path / 'data')
    indices = manifest['indices']
    assert list(map(len, indices.values())) == [20, 5, 5]
    assert sorted(sum(indices.values(), [])) == list(range(30))
    assert len(set(manifest['geometry_fingerprints'])) == 30
    for path in paths.values():
        for atoms in read(path, ':'):
            assert support.reference_bec(atoms).shape == (len(atoms), 3, 3)
            assert 'dielectric_tensor' not in atoms.calc.results
    from sevenn.parse_input import read_config_yaml
    import yaml

    for multi in [False, True]:
        config = support.make_config(paths, 1, multitask=multi)
        assert 'load_testset_path' not in config['data']
        path = tmp_path / f'{multi}.yaml'
        path.write_text(yaml.safe_dump(config))
        model, train, data = read_config_yaml(str(path))
        assert train['is_train_bec'] and not train['is_train_dielectric']
        assert train['force_loss_weight'] == (0.3 if multi else 0.0)


def test_fingerprint_is_invariant():
    atoms = read(ROOT / 'example_inputs/training_polar/data.xyz', index=0)
    shifted = atoms[::-1]
    shifted.translate([0.33, -0.21, 0.11])
    assert support.geometry_fingerprint(atoms) == support.geometry_fingerprint(
        shifted
    )


def test_stress_conversion_and_bec_order():
    atoms = pd.DataFrame({'stct_id': [0]})
    graphs = pd.DataFrame(
        {
            f'{prefix}_{c}': [value]
            for prefix in ['stress', 'inferred_stress']
            for c, value in zip(
                ['xx', 'yy', 'zz', 'xy', 'yz', 'zx'], [10, 20, 30, 40, 50, 60]
            )
        }
    )
    ref, pred, unit = support.paired_values(atoms, graphs, 'Stress', [0])
    np.testing.assert_allclose(ref, [-1, -2, -3, -4, -5, -6])
    assert unit == 'GPa'
    np.testing.assert_array_equal(pred, ref)
    for prefix in ['born_effective_charges', 'inferred_born_effective_charges']:
        for i, c in enumerate(support.COMPONENTS):
            atoms[f'{prefix}_{c}'] = i + 1
    ref, pred, unit = support.paired_values(atoms, graphs, 'BEC off-diagonal', [0])
    np.testing.assert_array_equal(ref, [2, 3, 4, 6, 7, 8])


@pytest.mark.parametrize('multitask', [False, True])
def test_cpu_train_reload_and_evaluate(tmp_path, multitask):
    # One actual epoch through the production CLI, no synthetic labels.
    structures, paths, all_path, manifest, _ = support.prepare_data(
        ROOT, tmp_path / 'data'
    )
    config = support.make_config(paths, 1, multitask=multitask, channel=4, layers=2)
    checkpoint, profile = support.train(ROOT, tmp_path / 'run', config)
    atoms, graphs, inference_profile = support.evaluate(
        ROOT, tmp_path / 'run', checkpoint, all_path, 'cpu'
    )
    assert len(graphs) == 30 and len(atoms) == sum(map(len, structures))
    i = manifest['indices']['Test'][0]
    support.reload_ase(checkpoint, structures[i], 'cpu', atoms, i)
    metrics, fig = support.metrics_and_parity(
        atoms, graphs, manifest, tmp_path / 'run', multitask=multitask
    )
    assert np.isfinite(metrics.RMSE).all()
    # Verify DFT stress references against ASE, not merely the conversion code.
    if multitask:
        reference, _, _ = support.paired_values(atoms, graphs, 'Stress', [i])
        stress = structures[i].get_stress()
        np.testing.assert_allclose(
            reference, stress[[0, 1, 2, 5, 3, 4]] * 160.21766208, rtol=1e-6
        )
    assert profile['peak_cuda_allocated_bytes'] is None
    assert inference_profile['cli_seconds'] > 0
    import matplotlib.pyplot as plt

    plt.close(fig)
