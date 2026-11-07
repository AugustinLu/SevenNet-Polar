"""Short MD on the Apple GPU, Direct models, with or without a field: Langevin NVT by
default, --nve for velocity Verlet."""

import time

import numpy as np
import torch
from _common import calculators, parser, peak_memory_gb, structure
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

ap = parser(__doc__)
ap.add_argument('--steps', type=int, default=200)
ap.add_argument('--temperature', type=float, default=300.0)
ap.add_argument('--nve', action='store_true')
ap.add_argument('--log-every', type=int, default=50)
args = ap.parse_args()
base, calc = calculators(args)
atoms = structure(args.repeat, rattle=0.02)
atoms.calc = calc
MaxwellBoltzmannDistribution(
    atoms, temperature_K=args.temperature, rng=np.random.default_rng(2)
)
if args.nve:
    dyn = VelocityVerlet(atoms, timestep=1.0 * units.fs)
else:
    dyn = Langevin(
        atoms,
        1.0 * units.fs,
        temperature_K=args.temperature,
        friction=0.01 / units.fs,
        rng=np.random.default_rng(3),
    )

dyn.run(10)  # warm-up: the first steps build MPS caches
torch.mps.synchronize()
t0 = time.perf_counter()
done = 0
while done < args.steps:
    n = min(args.log_every, args.steps - done)
    dyn.run(n)
    done += n
    e, k = atoms.get_potential_energy(), atoms.get_kinetic_energy()
    print(
        f'step {done:6d}  Epot {e:12.4f}  Etot {e + k:12.4f} eV  T '
        f'{atoms.get_temperature():7.1f} K  edges {calc.results["num_edges"]}',
        flush=True,
    )
torch.mps.synchronize()
dt = time.perf_counter() - t0
rss, drv = peak_memory_gb()
print(
    f'{1e3 * dt / args.steps:.1f} ms per MD step, {len(atoms)} atoms, field '
    f'{args.efield}; peak RSS {rss:.2f} GB, Metal driver {drv:.2f} GB'
)
