"""Integrity and scientific plumbing for the BEC tutorials."""

import importlib.util
import hashlib
import json
import shutil
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
    paths = sorted(
        p
        for p in (ROOT / 'tutorials/colab').glob('*.ipynb')
        if not p.name.endswith('_output.ipynb')
    )
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
    import yaml

    from sevenn.parse_input import read_config_yaml

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


def test_nested_subset_keeps_holdouts_and_provenance(tmp_path):
    directory = tmp_path / 'example_inputs/tutorial_polar/zro2'
    directory.mkdir(parents=True)
    source = directory / 'pool.extxyz'
    shutil.copy(ROOT / 'example_inputs/training_polar/data.xyz', source)
    selection = {
        'pool_order': list(range(30)),
        'pool_indices': {
            'Train': list(range(20)),
            'Validation': list(range(20, 25)),
            'Test': list(range(25, 30)),
        },
        'pool_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'source': {'filename': 'original.xyz', 'sha256': 'original-source-hash'},
        'selection_seed': 43,
        'split_policy': 'fixed whole-structure holdouts',
    }
    (directory / 'manifest.json').write_text(json.dumps(selection))
    manifests = []
    for n in [5, 10]:
        structures, paths, all_path, manifest, _ = support.prepare_subset_data(
            tmp_path, tmp_path / f'split{n}', n
        )
        manifests.append(manifest)
        assert len(structures) == n + 10
        assert len(read(all_path, ':')) == n + 10
        assert len(read(paths['Train'], ':')) == n
        assert len(set(sum(manifest['indices'].values(), []))) == n + 10
        assert manifest['original_source']['sha256'] == 'original-source-hash'
    assert (
        manifests[0]['source_ids']['Train']
        == manifests[1]['source_ids']['Train'][:5]
    )
    for partition in ['Validation', 'Test']:
        assert (
            manifests[0]['source_ids'][partition]
            == manifests[1]['source_ids'][partition]
        )
    with pytest.raises(ValueError, match='Training count'):
        support.prepare_subset_data(tmp_path, tmp_path / 'invalid', 21)
    source.write_bytes(source.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='checksum'):
        support.prepare_subset_data(tmp_path, tmp_path / 'modified', 5)


@pytest.mark.parametrize('dataset', ['zro2', 'li3po4'])
def test_official_subset_pool_preserves_labels(tmp_path, dataset):
    structures, paths, _, manifest, _ = support.prepare_subset_data(
        ROOT, tmp_path / dataset, 50, dataset=dataset
    )
    assert list(map(len, manifest['indices'].values())) == [50, 20, 20]
    assert len(structures) == 90
    assert len(set(sum(manifest['source_ids'].values(), []))) == 90
    assert manifest['original_source']['doi'] == '10.5281/zenodo.21322761'
    reference_sums = []
    for path in paths.values():
        for atoms in read(path, ':'):
            assert support.reference_bec(atoms).shape == (len(atoms), 3, 3)
            reference_sums.append(abs(support.reference_bec(atoms).sum(0)).max())
    if dataset == 'li3po4':
        # The official vacancy labels must not be neutralized by tutorial setup.
        assert max(reference_sums) > 1.0
    else:
        assert max(reference_sums) < 2e-7


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
    # A reference baseline must use validation labels, even if test labels
    # have entirely different tensor sums.
    plot_structures = list(structures)
    for test_id in manifest['indices']['Test']:
        plot_structures[test_id] = structures[test_id].copy()
        plot_structures[test_id].set_array(
            'born_effective_charges', np.full((len(structures[test_id]), 9), 99.0)
        )
    curve, convergence = support.learning_curves(
        tmp_path / 'run',
        multitask=multitask,
        structures=plot_structures,
        manifest=manifest,
    )
    reference_sums = [
        abs(support.reference_bec(structures[j]).sum(0)).max()
        for j in manifest['indices']['Validation']
    ]
    for title, expected in [
        ('Raw ASR maximum', max(reference_sums)),
        ('Raw ASR mean', np.mean(reference_sums)),
    ]:
        axis = next(a for a in convergence.axes if a.get_title() == title)
        np.testing.assert_allclose(axis.lines[2].get_ydata()[0], expected)
    # Verify DFT stress references against ASE, not merely the conversion code.
    if multitask:
        reference, _, _ = support.paired_values(atoms, graphs, 'Stress', [i])
        stress = structures[i].get_stress()
        np.testing.assert_allclose(
            reference, stress[[0, 1, 2, 5, 3, 4]] * 160.21766208, rtol=1e-6
        )
        for title, divisor, column in [
            ('Force', np.sqrt(3), 'validset_Force_RMSE'),
            ('Stress', 10 * np.sqrt(6), 'validset_Stress_RMSE'),
        ]:
            expected = metrics[
                (metrics.partition == 'Validation') & (metrics.quantity == title)
            ].RMSE.iloc[0]
            np.testing.assert_allclose(
                curve[column].iloc[0] / divisor, expected, rtol=1e-5
            )
            axis = next(a for a in convergence.axes if a.get_title() == title)
            np.testing.assert_allclose(
                axis.lines[1].get_ydata()[0], expected, rtol=1e-5
            )
    assert profile['peak_cuda_allocated_bytes'] is None
    assert inference_profile['cli_seconds'] > 0
    asr = support.asr_diagnostics(atoms, manifest, tmp_path / 'run')
    curve = pd.read_csv(tmp_path / 'run/lc.csv')
    expected_asr = asr[asr.partition == 'Validation'].prediction_max_abs_sum_e
    # Validation evaluates one frozen checkpoint. Training uses successive
    # optimizer states, so its epoch metric need not match a checkpoint reload.
    np.testing.assert_allclose(
        curve.validset_BornEffectiveCharges_ASRMax.iloc[0],
        expected_asr.max(),
        atol=2e-5,
        rtol=2e-5,
    )
    np.testing.assert_allclose(
        curve.validset_BornEffectiveCharges_ASRMean.iloc[0],
        expected_asr.mean(),
        atol=2e-5,
        rtol=2e-5,
    )
    import matplotlib.pyplot as plt

    plt.close(convergence)
    plt.close(fig)
