"""Small tutorial utilities; all model training uses SevenNet's CLI and APIs."""

from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import torch
import yaml
from ase.io import read, write

COMPONENTS = ['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz']
DIAGONAL = ['xx', 'yy', 'zz']
OFF_DIAGONAL = ['xy', 'xz', 'yx', 'yz', 'zx', 'zy']
COLORS = {'Train': '#4477AA', 'Validation': '#228833', 'Test': '#EE7733'}


def environment(repo):
    import sevenn

    path = Path(sevenn.__file__).resolve()
    if repo.resolve() not in path.parents:
        raise RuntimeError(
            f'Wrong sevenn package: {path}. Restart the kernel and run setup.'
        )
    packages = [
        'sevenn',
        'torch',
        'e3nn',
        'ase',
        'numpy',
        'matscipy',
        'torch-geometric',
        'matplotlib',
    ]
    result = {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'packages': {p: importlib.metadata.version(p) for p in packages},
        'source_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True
        ).strip(),
        'source_dirty': bool(
            subprocess.check_output(
                ['git', 'status', '--porcelain', '--untracked-files=no'],
                cwd=repo,
                text=True,
            ).strip()
        ),
        'sevenn_origin': str(path),
        'torch_cuda_build': torch.version.cuda,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'precision': 'float32 (no autocast)',
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        result.update(
            gpu=p.name,
            capability=[p.major, p.minor],
            gpu_memory_bytes=p.total_memory,
        )
    for name, command in [
        ('cuda_compiler', ['nvcc', '--version']),
        ('cxx_compiler', ['c++', '--version']),
    ]:
        try:
            result[name] = subprocess.check_output(
                command, text=True, stderr=subprocess.STDOUT
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            result[name] = None
    return result


def reference_bec(atoms):
    value = atoms.arrays.get('born_effective_charges')
    if value is None and atoms.calc is not None:
        value = atoms.calc.results.get('born_effective_charges')
    if value is None:
        raise ValueError('Missing Born effective charges.')
    value = np.asarray(value, dtype=float).reshape(len(atoms), 3, 3)
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite BEC labels.')
    return value


def training_copy(source):
    # Copy E/F/S/B only, excluding any optional properties in the source file.
    atoms = source.copy()
    atoms.calc = None
    atoms.info.clear()
    for key in list(atoms.arrays):
        if key not in ('numbers', 'positions'):
            del atoms.arrays[key]
    atoms.info['energy'] = float(source.get_potential_energy())
    atoms.info['stress'] = source.get_stress(voigt=False)
    atoms.new_array('forces', source.get_forces())
    atoms.new_array(
        'born_effective_charges', reference_bec(source).reshape(len(source), 9)
    )
    return atoms


def geometry_fingerprint(atoms):
    # A permutation/translation/rotation invariant exact-duplicate screen.
    # Distances are minimum-image distances, grouped by chemical pair.
    distances = atoms.get_all_distances(mic=True)
    features = []
    for i, zi in enumerate(atoms.numbers):
        for j in range(i):
            features.append(
                (
                    min(int(zi), int(atoms.numbers[j])),
                    max(int(zi), int(atoms.numbers[j])),
                    round(float(distances[i, j]), 5),
                )
            )
    return hashlib.sha256(
        repr((sorted(features), round(atoms.get_volume(), 5))).encode()
    ).hexdigest()


def prepare_data(repo, destination, seed=42):
    from sklearn.model_selection import train_test_split

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    source = repo / 'example_inputs/training_polar/data.xyz'
    structures = read(source, ':', format='extxyz')
    if len(structures) != 30:
        raise ValueError(f'Expected 30 example structures, got {len(structures)}.')
    formulas = np.array([a.get_chemical_formula() for a in structures])
    fingerprints = [geometry_fingerprint(a) for a in structures]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError(
            'Duplicate geometries found: provide a grouped split before training.'
        )
    ids = np.arange(len(structures))
    train_valid, test = train_test_split(
        ids, test_size=5, stratify=formulas, random_state=seed
    )
    train, valid = train_test_split(
        train_valid,
        test_size=5,
        stratify=formulas[train_valid],
        random_state=seed + 1,
    )
    partitions = {
        'Train': sorted(train.tolist()),
        'Validation': sorted(valid.tolist()),
        'Test': sorted(test.tolist()),
    }
    paths = {}
    for name, indices in partitions.items():
        path = destination / f'{name.lower()}.extxyz'
        write(path, [training_copy(structures[i]) for i in indices], format='extxyz')
        restored = read(path, ':', format='extxyz')
        for original_id, atoms in zip(indices, restored):
            original = structures[original_id]
            np.testing.assert_allclose(
                reference_bec(atoms), reference_bec(original), atol=1e-8
            )
            np.testing.assert_allclose(
                atoms.get_forces(), original.get_forces(), atol=1e-8
            )
            np.testing.assert_allclose(
                atoms.get_stress(), original.get_stress(), atol=1e-8
            )
            np.testing.assert_allclose(
                atoms.get_potential_energy(),
                original.get_potential_energy(),
                atol=1e-8,
            )
        paths[name] = path.resolve()
    all_path = destination / 'all.extxyz'
    write(all_path, [training_copy(a) for a in structures], format='extxyz')
    manifest = {
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'seed': seed,
        'indices': partitions,
        'geometry_fingerprints': fingerprints,
        'split_policy': 'composition-stratified random demonstration; trajectory independence unknown',
    }
    (destination / 'split.json').write_text(json.dumps(manifest, indent=2))
    summary = pd.DataFrame(
        [
            {
                'partition': name,
                'structures': len(indices),
                'compositions': dict(Counter(formulas[indices])),
            }
            for name, indices in partitions.items()
        ]
    )
    return structures, paths, all_path.resolve(), manifest, summary


def make_config(
    paths,
    epochs,
    multitask=False,
    batch_size=1,
    channel=16,
    layers=3,
    device='cpu',
    seed=42,
):
    if min(epochs, batch_size, channel, layers) < 1:
        raise ValueError(
            'Epochs, batch size, channels, and layers must be positive.'
        )
    errors = [
        ['BornEffectiveCharges', 'DiagRMSE'],
        ['BornEffectiveCharges', 'OffDiagRMSE'],
        ['TotalLoss', 'None'],
    ]
    if multitask:
        errors = [['Energy', 'RMSE'], ['Force', 'RMSE'], ['Stress', 'RMSE']] + errors
    return {
        'model': {
            'chemical_species': 'auto',
            'cutoff': 6.0,
            'channel': channel,
            'lmax': 2,
            'num_convolution_layer': layers,
            'is_parity': False,
        },
        'train': {
            'random_seed': seed,
            'num_workers': 0,
            'device': device,
            'epoch': epochs,
            'per_epoch': max(1, epochs // 4),
            'best_metric': 'TotalLoss',
            'is_train_bec': True,
            'is_train_stress': multitask,
            'is_train_dielectric': False,
            'energy_loss_weight': 1.0 if multitask else 0.0,
            'force_loss_weight': 0.3 if multitask else 0.0,
            'stress_loss_weight': 1.0e-5 if multitask else 0.0,
            'bec_loss_weight': 1.0,
            'loss': 'mse',
            'optimizer': 'adam',
            'optim_param': {'lr': 0.005},
            'scheduler': 'exponentiallr',
            'scheduler_param': {'gamma': 0.992},
            'error_record': errors,
        },
        'data': {
            'batch_size': batch_size,
            'preprocess_num_cores': 1,
            'shift': 'per_atom_energy_mean',
            'scale': 1.0,
            'data_format': 'ase',
            'data_format_args': {'format': 'extxyz', 'index': ':'},
            'load_trainset_path': [str(paths['Train'])],
            'load_validset_path': [str(paths['Validation'])],
        },
    }


def run_cli(repo, run_dir, name, arguments):
    run_dir = Path(run_dir)
    log_path = run_dir / f'{name}.log'
    profile_path = run_dir / f'{name}_profile.json'
    env = os.environ.copy()
    env['PYTHONPATH'] = str(repo) + os.pathsep + env.get('PYTHONPATH', '')
    env['OMP_NUM_THREADS'] = env['MKL_NUM_THREADS'] = env.get(
        'SEVENNET_TUTORIAL_THREADS', '2'
    )
    env['MPLBACKEND'] = 'Agg'
    started = time.perf_counter()
    command = [
        sys.executable,
        str(repo / 'tutorials/colab/cli_runner.py'),
        str(profile_path),
        *map(str, arguments),
    ]
    with log_path.open('w') as log:
        process = subprocess.Popen(
            command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            while process.poll() is None:
                time.sleep(1)
                elapsed = time.perf_counter() - started
                if int(elapsed) % 30 == 0:
                    lines = log_path.read_text(errors='replace').splitlines()
                    epochs = [
                        line.strip()
                        for line in lines
                        if line.startswith('Epoch ') and 'elapsed' not in line
                    ]
                    print(
                        f'{name}: {elapsed:.0f} s'
                        + (f' · {epochs[-1]}' if epochs else ' · preparing'),
                        flush=True,
                    )
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if process.returncode:
        tail = '\n'.join(log_path.read_text(errors='replace').splitlines()[-30:])
        raise RuntimeError(
            f'{name} failed (exit {process.returncode}). Full log: {log_path}\n{tail}\n'
            'For CUDA out of memory, reduce batch_size to 1; for imports, restart and rerun setup.'
        )
    result = json.loads(profile_path.read_text())
    result['subprocess_wall_seconds'] = time.perf_counter() - started
    profile_path.write_text(json.dumps(result, indent=2))
    print(
        f'{name}: completed in {result["subprocess_wall_seconds"]:.1f} s; log: {log_path.name}'
    )
    return result


def train(repo, run_dir, config, flash=False):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    config_path = run_dir / 'input.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    arguments = ['train', config_path, '--working_dir', run_dir, '-s']
    if flash:
        arguments.append('--enable_flash')
    profile = run_cli(repo, run_dir, 'training', arguments)
    checkpoint = run_dir / 'checkpoint_best.pth'
    if not checkpoint.is_file():
        raise RuntimeError('No best validation checkpoint was produced.')
    return checkpoint, profile


def evaluate(repo, run_dir, checkpoint, all_path, device, batch_size=1):
    out = run_dir / 'inference'
    profile = run_cli(
        repo,
        run_dir,
        'inference',
        [
            'inference',
            checkpoint,
            all_path,
            '--device',
            device,
            '--nworkers',
            '1',
            '--batch',
            str(batch_size),
            '--output',
            out,
        ],
    )
    atoms = pd.read_csv(out / 'per_atom.csv')
    graphs = pd.read_csv(out / 'per_graph.csv')
    return atoms, graphs, profile


def paired_values(atoms, graphs, quantity, ids):
    rows = atoms[atoms.stct_id.isin(ids)]
    if quantity in ('BEC diagonal', 'BEC off-diagonal'):
        components = DIAGONAL if quantity == 'BEC diagonal' else OFF_DIAGONAL
        reference = rows[
            [f'born_effective_charges_{c}' for c in components]
        ].to_numpy(float)
        prediction = rows[
            [f'inferred_born_effective_charges_{c}' for c in components]
        ].to_numpy(float)
        unit = 'e'
    elif quantity == 'Force':
        reference = rows[[f'force_of_atoms_{c}' for c in 'xyz']].to_numpy(float)
        prediction = rows[[f'inferred_force_{c}' for c in 'xyz']].to_numpy(float)
        unit = 'eV/Å'
    else:
        rows = graphs.iloc[ids]
        if quantity == 'Energy per atom':
            reference = rows.total_energy.to_numpy(float) / rows.num_atoms.to_numpy(
                float
            )
            prediction = rows.inferred_total_energy.to_numpy(
                float
            ) / rows.num_atoms.to_numpy(float)
            unit = 'eV/atom'
        elif quantity == 'Stress':
            components = ['xx', 'yy', 'zz', 'xy', 'yz', 'zx']
            # CLI CSV stores pressure-sign stress in kbar. Convert to ASE's
            # tensile-positive convention and GPa for the tutorial plots.
            reference = (
                -rows[[f'stress_{c}' for c in components]].to_numpy(float) / 10
            )
            prediction = (
                -rows[[f'inferred_stress_{c}' for c in components]].to_numpy(float)
                / 10
            )
            unit = 'GPa'
        else:
            raise ValueError(quantity)
    reference, prediction = reference.ravel(), prediction.ravel()
    if (
        len(reference) == 0
        or not np.isfinite(reference).all()
        or not np.isfinite(prediction).all()
    ):
        raise ValueError(f'Empty or nonfinite {quantity} inference.')
    return reference, prediction, unit


def metrics_and_parity(atoms, graphs, manifest, run_dir, multitask=False):
    import matplotlib.pyplot as plt

    quantities = ['BEC diagonal', 'BEC off-diagonal']
    if multitask:
        quantities += ['Energy per atom', 'Force', 'Stress']
    fig, axes = plt.subplots(
        1 if not multitask else 2,
        2 if not multitask else 3,
        figsize=(11, 4.8) if not multitask else (15, 9),
        constrained_layout=True,
    )
    records = []
    for ax, quantity in zip(axes.flat, quantities):
        bounds = []
        for partition, ids in manifest['indices'].items():
            reference, prediction, unit = paired_values(atoms, graphs, quantity, ids)
            error = prediction - reference
            metric = {
                'partition': partition,
                'quantity': quantity,
                'unit': unit,
                'components': len(error),
                'RMSE': float(np.sqrt(np.mean(error**2))),
                'MAE': float(np.mean(np.abs(error))),
            }
            records.append(metric)
            ax.scatter(
                reference,
                prediction,
                s=12 if partition == 'Train' else 22,
                alpha=0.2 if partition == 'Train' else 0.7,
                color=COLORS[partition],
                label=partition,
                rasterized=True,
                edgecolors='none',
            )
            bounds.extend(
                [
                    reference.min(),
                    reference.max(),
                    prediction.min(),
                    prediction.max(),
                ]
            )
            if partition == 'Test':
                ax.set_title(
                    f'{quantity}\nTest RMSE {metric["RMSE"]:.4f} {unit}', fontsize=12
                )
        lo, hi = min(bounds), max(bounds)
        pad = max((hi - lo) * 0.06, 0.001)
        limits = [lo - pad, hi + pad]
        ax.plot(limits, limits, '--', color='#333333', lw=1)
        ax.set(
            xlabel=f'DFT reference ({unit})',
            ylabel=f'Prediction ({unit})',
            xlim=limits,
            ylim=limits,
        )
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    for ax in list(axes.flat)[len(quantities) :]:
        ax.axis('off')
    fig.savefig(run_dir / 'parity.png', dpi=180)
    fig.savefig(run_dir / 'parity.pdf')
    metrics = pd.DataFrame(records)
    metrics.to_csv(run_dir / 'metrics.csv', index=False)
    return metrics, fig


def asr_diagnostics(atoms, manifest, run_dir):
    records = []
    for partition, ids in manifest['indices'].items():
        for i in ids:
            rows = atoms[atoms.stct_id == i]
            ref = (
                rows[[f'born_effective_charges_{c}' for c in COMPONENTS]]
                .to_numpy(float)
                .reshape(-1, 3, 3)
            )
            pred = (
                rows[[f'inferred_born_effective_charges_{c}' for c in COMPONENTS]]
                .to_numpy(float)
                .reshape(-1, 3, 3)
            )
            records.append(
                {
                    'partition': partition,
                    'structure_id': i,
                    'reference_max_abs_sum_e': float(np.abs(ref.sum(0)).max()),
                    'prediction_max_abs_sum_e': float(np.abs(pred.sum(0)).max()),
                }
            )
    frame = pd.DataFrame(records)
    frame.to_csv(run_dir / 'asr.csv', index=False)
    return frame


def learning_curves(run_dir, multitask=False):
    import matplotlib.pyplot as plt

    curve = pd.read_csv(run_dir / 'lc.csv')
    targets = [
        ('BornEffectiveCharges_DiagRMSE', 'BEC diagonal', 'e'),
        ('BornEffectiveCharges_OffDiagRMSE', 'BEC off-diagonal', 'e'),
    ]
    if multitask:
        targets += [
            ('Energy_RMSE', 'Energy per atom', 'eV/atom'),
            ('Force_RMSE', 'Force', 'eV/Å'),
            ('Stress_RMSE', 'Stress', 'kbar'),
            ('TotalLoss', 'Weighted loss', 'weighted units'),
        ]
    fig, axes = plt.subplots(
        1 if not multitask else 2,
        2 if not multitask else 3,
        figsize=(11, 4) if not multitask else (14, 8),
        constrained_layout=True,
    )
    for ax, (suffix, title, unit) in zip(axes.flat, targets):
        for prefix, partition in [('trainset', 'Train'), ('validset', 'Validation')]:
            columns = [
                c for c in curve if c.startswith(prefix + '_') and suffix in c
            ]
            if len(columns) != 1:
                raise ValueError(
                    f'Missing/ambiguous learning curve column {prefix}_{suffix}: {columns}'
                )
            ax.plot(
                curve.epoch,
                curve[columns[0]],
                label=partition,
                color=COLORS[partition],
            )
        ax.set(title=title, xlabel='Epoch', ylabel=unit)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig.savefig(run_dir / 'convergence.png', dpi=160)
    return curve, fig


def reload_ase(checkpoint, source, device, inference_atoms, structure_id):
    from sevenn.calculator import SevenNetCalculator

    # SevenNetCalculator reports raw direct-head BECs, without field ASR projection.
    calc = SevenNetCalculator(str(checkpoint), device=device, enable_flash=False)
    atoms = source.copy()
    atoms.calc = calc
    bec = calc.get_property('born_effective_charges', atoms)
    rows = inference_atoms[inference_atoms.stct_id == structure_id]
    cli_bec = (
        rows[[f'inferred_born_effective_charges_{c}' for c in COMPONENTS]]
        .to_numpy(float)
        .reshape(-1, 3, 3)
    )
    np.testing.assert_allclose(bec, cli_bec, atol=2e-5, rtol=2e-5)
    return bec


def export_artifacts(run_dir, data_dir, report):
    (run_dir / 'report.json').write_text(json.dumps(report, indent=2))
    archive = run_dir.parent / (run_dir.name + '.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in run_dir.rglob('*'):
            if (
                p.is_file()
                and ('sevenn_data' not in p.parts)
                and (p.suffix != '.pth' or p.name == 'checkpoint_best.pth')
            ):
                z.write(p, p.relative_to(run_dir))
        for p in data_dir.glob('*'):
            if p.is_file():
                z.write(p, 'data/' + p.name)
    return archive
