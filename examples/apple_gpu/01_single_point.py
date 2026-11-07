"""Single point with forces, stress, Z* and dielectric tensor on the Apple GPU.

Direct models: Z* and eps come from the heads; with --efield the field calculator
adds the field terms.

"""

import time

import numpy as np
from _common import calculators, parser, peak_memory_gb, structure

args = parser(__doc__).parse_args()
base, calc = calculators(args)
atoms = structure(args.repeat)
atoms.calc = calc

t0 = time.perf_counter()
energy = atoms.get_potential_energy()
print(
    f'first call, includes kernel set-up: {1e3 * (time.perf_counter() - t0):.0f} ms'
)
forces, stress = atoms.get_forces(), atoms.get_stress()
results = calc.results
print(f'atoms {len(atoms)}, edges {results["num_edges"]}, energy {energy:.4f} eV')
print(
    f'max |F| {np.abs(forces).max():.4f} eV/A, stress (eV/A^3, xx yy zz yz xz xy) '
    f'{np.round(stress, 5)}'
)
if 'born_effective_charges' in results:
    Z = results['born_effective_charges']
    print(
        f'Z* shape {Z.shape}, sum rule residual '
        f'{np.abs(Z.sum(axis=0)).max():.2e}\nZ* of atom 0:\n{np.round(Z[0], 3)}'
    )
if 'dielectric_tensor' in results:
    print(f'eps_inf:\n{np.round(results["dielectric_tensor"].reshape(3, 3), 3)}')
rss, drv = peak_memory_gb()
print(f'peak RSS {rss:.2f} GB, Metal driver {drv:.2f} GB')
