"""Optional short LAMMPS NVT field demonstration with a multitask checkpoint."""

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.lammps import Prism
from ase.io import read, write

import support

LAMMPS_REF = 'stable_2Aug2023_update3'


def plot_md(destination):
    """Save the thermostat response from the finite trajectory, without refitting."""
    import matplotlib.pyplot as plt
    import pandas as pd

    destination = Path(destination)
    frames = read(
        destination / 'trajectory.dump',
        ':',
        format='lammps-dump-text',
        specorder=['Li', 'O', 'P'],
    )
    report = json.loads((destination / 'report.json').read_text())
    curve = pd.DataFrame(
        {
            'time_fs': [a.info['timestep'] * report['timestep_fs'] for a in frames],
            'temperature_K': [a.get_temperature() for a in frames],
            'max_force_eV_per_A': [
                np.linalg.norm(a.get_forces(), axis=1).max() for a in frames
            ],
        }
    )
    curve.to_csv(destination / 'trajectory_summary.csv', index=False)
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    axes[0].plot(curve.time_fs, curve.temperature_K, 'o-', label='Trajectory')
    axes[0].axhline(
        report['temperature_K'], color='grey', ls='--', label='NVT target'
    )
    axes[0].set_ylabel('Temperature (K)')
    axes[0].legend()
    axes[1].plot(curve.time_fs, curve.max_force_eV_per_A, 'o-')
    axes[1].set_ylabel('Maximum force norm (eV/Å)')
    for axis in axes:
        axis.set_xlabel('Time (fs)')
        axis.grid(alpha=0.2)
    figure.suptitle(f'{report["atoms"]} atoms: short NVT field demonstration')
    figure.tight_layout()
    for extension in ['png', 'pdf']:
        figure.savefig(destination / ('trajectory.' + extension), dpi=150)
    return curve, figure


def command(arguments, directory, log_path):
    started = time.perf_counter()
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = env['MKL_NUM_THREADS'] = '2'
    with Path(log_path).open('w') as log:
        process = subprocess.Popen(
            [str(a) for a in arguments],
            cwd=directory,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        while True:
            try:
                code = process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                print(
                    f'{Path(log_path).stem}: {time.perf_counter() - started:.0f} s '
                    f'(full output in {log_path})',
                    flush=True,
                )
    if code:
        raise RuntimeError(
            f'Command failed; inspect {log_path}:\n'
            + Path(log_path).read_text()[-2500:]
        )
    return time.perf_counter() - started


def build_lammps(repo, destination, jobs=2):
    """Build an isolated serial executable, using the runtime's PyTorch."""
    import torch

    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    source = destination / 'source'
    started = time.perf_counter()
    command(
        [
            'git',
            'clone',
            '--depth',
            '1',
            '--branch',
            LAMMPS_REF,
            'https://github.com/lammps/lammps.git',
            source,
        ],
        destination,
        destination / 'clone.log',
    )
    command(
        [sys.executable, '-m', 'sevenn.main.sevenn', 'patch_lammps', source],
        repo,
        destination / 'patch.log',
    )
    command(
        [
            'cmake',
            '-S',
            source / 'cmake',
            '-B',
            destination / 'build',
            '-DBUILD_MPI=OFF',
            '-DCMAKE_BUILD_TYPE=Release',
            # The PyTorch wheel supplies MKL binaries without its header path.
            # This interface does not include MKL headers; see the package guide.
            '-DMKL_INCLUDE_DIR=' + str(destination),
            '-DCMAKE_PREFIX_PATH=' + torch.utils.cmake_prefix_path,
        ],
        destination,
        destination / 'configure.log',
    )
    command(
        ['cmake', '--build', destination / 'build', '--parallel', str(jobs)],
        destination,
        destination / 'build.log',
    )
    report = {
        'lammps_ref': LAMMPS_REF,
        'source_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=source, text=True
        ).strip(),
        'build_seconds': time.perf_counter() - started,
        'jobs': jobs,
        'torch_version': torch.__version__,
    }
    (destination / 'build_report.json').write_text(
        json.dumps(report, indent=2) + '\n'
    )
    return destination / 'build/lmp', report


def supercell(source):
    """Rotate the 512-atom cell into the restricted-triclinic LAMMPS frame."""
    if source.get_chemical_formula() != 'Li12O16P4':
        raise ValueError('Use a stoichiometric 32-atom Li₃PO₄ training frame.')
    repeated = source.repeat((2, 2, 4))
    prism = Prism(repeated.cell.array, reduce_cell=False)
    return Atoms(
        numbers=repeated.numbers,
        positions=prism.vector_to_lammps(repeated.positions, wrap=True),
        cell=prism.cell,
        pbc=True,
    )


def run_field_md(repo, checkpoint, source, destination, executable, steps=100):
    """Run a short example and validate LAMMPS initial forces against ASE."""
    import torch

    from sevenn.calculator import FieldCalculator, SevenNetCalculator
    from sevenn.util import load_checkpoint

    if not 1 <= steps <= 1000:
        raise ValueError('This demonstration is bounded to 1–1000 timesteps.')
    config = load_checkpoint(str(checkpoint)).config
    if (
        config.get('energy_loss_weight', 0) <= 0
        or config.get('force_loss_weight', 0) <= 0
    ):
        raise ValueError(
            'MD requires a multitask checkpoint with trained energy and forces.'
        )
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    atoms = supercell(source)
    write(destination / 'initial.extxyz', atoms, format='extxyz')
    write(
        destination / 'initial.data',
        atoms,
        format='lammps-data',
        atom_style='atomic',
        specorder=['Li', 'O', 'P'],
        masses=True,
    )
    deploy_seconds = command(
        [
            sys.executable,
            '-m',
            'sevenn.main.sevenn',
            'get_model',
            Path(checkpoint).resolve(),
            '-o',
            destination / 'model.pt',
        ],
        repo,
        destination / 'deploy.log',
    )
    field = np.array([0.0, 0.0, 0.001])
    timestep_fs = 0.25
    script = f"""units metal
atom_style atomic
boundary p p p
read_data initial.data
pair_style e3gnn
pair_coeff * * model.pt efield 0.0 0.0 0.001 Li O P
neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes
thermo 10
thermo_style custom step time temp pe etotal press
run 0
write_dump all custom field.dump id type x y z fx fy fz modify sort id
velocity all create 100.0 42 mom yes rot no dist gaussian
timestep {timestep_fs / 1000}
fix thermostat all nvt temp 100.0 100.0 0.05
dump trajectory all custom 10 trajectory.dump id type x y z vx vy vz fx fy fz
dump_modify trajectory sort id
run {steps}
write_data final.data
"""
    (destination / 'in.lammps').write_text(script)
    # The pair style accepts its coefficients once per instance. Validate the
    # zero-field forces in a separate process using the identical initial cell.
    zero_script = script[: script.index('write_dump')] + (
        'write_dump all custom zero.dump id type x y z fx fy fz modify sort id\n'
    )
    zero_script = zero_script.replace(' efield 0.0 0.0 0.001', '')
    (destination / 'zero.lammps').write_text(zero_script)
    zero_seconds = command(
        [Path(executable).resolve(), '-in', 'zero.lammps', '-log', 'zero.log'],
        destination,
        destination / 'zero_execution.log',
    )
    wall = command(
        [Path(executable).resolve(), '-in', 'in.lammps'],
        destination,
        destination / 'execution.log',
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base = SevenNetCalculator(str(checkpoint), device=device, enable_flash=False)
    reference = atoms.copy()
    reference.calc = base
    ase_zero = reference.get_forces()
    reference.calc = FieldCalculator(base_calculator=base)
    reference.calc.set_efield(field)
    ase_field = reference.get_forces()
    zero = read(
        destination / 'zero.dump',
        format='lammps-dump-text',
        specorder=['Li', 'O', 'P'],
    )
    applied = read(
        destination / 'field.dump',
        format='lammps-dump-text',
        specorder=['Li', 'O', 'P'],
    )
    differences = {}
    for name, expected, dumped in [
        ('zero', ase_zero, zero),
        ('field', ase_field, applied),
    ]:
        actual = dumped.get_forces()
        np.testing.assert_allclose(actual, expected, atol=3e-4, rtol=3e-4)
        differences[name] = float(abs(actual - expected).max())
    frames = read(
        destination / 'trajectory.dump',
        ':',
        format='lammps-dump-text',
        specorder=['Li', 'O', 'P'],
    )
    if len(frames[-1]) != 512 or not all(
        np.isfinite(a.positions).all() and np.isfinite(a.get_forces()).all()
        for a in frames
    ):
        raise RuntimeError('Incomplete or nonfinite MD trajectory.')
    log = (destination / 'log.lammps').read_text()
    matches = re.findall(r'Loop time of ([\d.eE+-]+) on .*? for (\d+) steps', log)
    md_seconds = next(float(t) for t, n in reversed(matches) if int(n) == steps)
    report = {
        'atoms': len(atoms),
        'steps': steps,
        'timestep_fs': timestep_fs,
        'duration_fs': steps * timestep_fs,
        'temperature_K': 100.0,
        'field_V_per_A': field.tolist(),
        'deploy_seconds': deploy_seconds,
        'lammps_process_seconds': wall,
        'zero_field_check_seconds': zero_seconds,
        'md_loop_seconds': md_seconds,
        'seconds_per_step': md_seconds / steps,
        'initial_force_max_abs_difference_eV_per_A': differences,
        'max_force_eV_per_A': float(
            max(np.linalg.norm(a.get_forces(), axis=1).max() for a in frames)
        ),
        'finite_frames': len(frames),
        'final_temperature_K': float(frames[-1].get_temperature()),
        'binary_sha256': hashlib.sha256(Path(executable).read_bytes()).hexdigest(),
        'environment': support.environment(Path(repo)),
        'note': (
            'Short workflow demonstration; no claim of long-time stability. '
            'Energy/pressure are the zero-field model quantities. '
            'Field forces use the documented pair-style defaults.'
        ),
    }
    (destination / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
