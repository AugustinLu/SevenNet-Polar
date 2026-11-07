"""Serial LAMMPS on the Apple GPU: LAMMPS runs the dynamics, the SevenNet calculator
gives energy, forces and virial (fix external).

Needs the LAMMPS Python module (pip install lammps). Direct models, with or without a
field. Parallel LAMMPS is not supported on a Mac.

"""

import os
import time

from _common import HERE, calculators, parser, peak_memory_gb
from lammps import lammps

from sevenn.lammps_external import run_input

ap = parser(__doc__)
ap.add_argument('--input', default=os.path.join(HERE, 'in.md_external'))
ap.add_argument(
    '--types',
    nargs='+',
    default=['Zr', 'O'],
    help='element of each LAMMPS atom type',
)
ap.add_argument('--steps', type=int, default=100)
ap.add_argument('--log', default='log.lammps')
args = ap.parse_args()
base, calc = calculators(args)
rep = str(args.repeat)
lmp = lammps(
    cmdargs=[
        '-screen',
        'none',
        '-log',
        args.log,
        '-var',
        'nx',
        rep,
        '-var',
        'ny',
        rep,
        '-var',
        'nz',
        rep,
        '-var',
        'steps',
        str(args.steps),
        '-var',
        'data',
        os.path.join(HERE, 'tZrO2_12atom.data'),
    ]
)
t0 = time.perf_counter()
run_input(lmp, args.input, calc, args.types)
dt = time.perf_counter() - t0
natoms = lmp.get_natoms()
print(
    f'{natoms} atoms, {args.steps} + 10 MD steps in {dt:.1f} s, '
    f'{1e3 * dt / (args.steps + 10):.1f} ms per step including start-up'
)
print(
    f'final pe {lmp.get_thermo("pe"):.4f} eV, T {lmp.get_thermo("temp"):.1f} K, '
    f'press {lmp.get_thermo("press"):.1f} bar; log in {args.log}'
)
rss, drv = peak_memory_gb()
print(f'peak RSS {rss:.2f} GB, Metal driver {drv:.2f} GB')
lmp.close()
