"""Relaxation of atoms and cell on the Apple GPU."""

import time

import numpy as np
from _common import calculators, parser, peak_memory_gb, structure
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE

ap = parser(__doc__)
ap.add_argument('--fmax', type=float, default=0.02)
ap.add_argument('--max-steps', type=int, default=300)
args = ap.parse_args()
base, calc = calculators(args)
atoms = structure(args.repeat, rattle=0.1)
atoms.calc = calc
e0 = atoms.get_potential_energy()
t0 = time.perf_counter()
opt = FIRE(FrechetCellFilter(atoms), logfile=None)
converged = opt.run(fmax=args.fmax, steps=args.max_steps)
dt = time.perf_counter() - t0
rss, drv = peak_memory_gb()
print(
    f'converged {bool(converged)} in {opt.nsteps} steps, '
    f'{1e3 * dt / max(opt.nsteps, 1):.1f} ms per step'
)
print(
    f'energy {e0:.4f} -> {atoms.get_potential_energy():.4f} eV, max |F| '
    f'{np.abs(atoms.get_forces()).max():.4f} eV/A, volume {atoms.get_volume():.2f} '
    f'A^3'
)
print(f'peak RSS {rss:.2f} GB, Metal driver {drv:.2f} GB')
