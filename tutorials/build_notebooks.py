"""Write the tutorial notebooks from the cell lists below. usage: python build_notebooks.py"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
M, C = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

SETUP = C("""import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

import _tutorial as T

warnings.filterwarnings('ignore', message='.*torch.jit.script.*')   # deprecation inside a dependency, harmless here
plt.rcParams['figure.dpi'] = 200   # sharp inline figures
T.machine_report()""")

NB = {}

# --------------------------------------------------------------------------------------------- 01
NB['01_first_calculation'] = [
    M("""# 1. A first calculation, on the CPU and on the Apple GPU

A machine-learning interatomic potential (MLIP) predicts the energy of a set of atoms, and from it the
forces and the stress, at a small fraction of the cost of density-functional theory. Here we use
**7net-0**, a universal SevenNet model trained on the Materials Project, on a crystal of silicon.

We compute the same thing three ways:

1. on the **CPU**, with the reference e3nn code,
2. on the **Apple GPU**, still with the reference e3nn code,
3. on the **Apple GPU with the fused Metal kernels**, the default on `device='mps'`.

The physics must not change: we check that the three give the same energy, forces and stress, then
compare their speed. Without an Apple GPU the notebook runs the CPU part only."""),
    SETUP,
    M("""## The structure

A 2 × 2 × 2 supercell of diamond silicon (64 atoms), with the atoms displaced slightly so that the forces
are not zero."""),
    C("""from ase.build import bulk

atoms = bulk('Si', 'diamond', a=5.43, cubic=True).repeat((2, 2, 2))
atoms.rattle(0.03, seed=1)
print(len(atoms), 'atoms, cell', atoms.cell.lengths())"""),
    M("""## Energy, forces and stress with each backend

`T.calculator(model, device, mps_opt)` builds a `SevenNetCalculator`. In your own scripts this is simply
`SevenNetCalculator('7net-0', device='mps')`, which uses the fused kernels; `mps_opt='off'` keeps the
reference e3nn code."""),
    C("""results, calcs = {}, {}
for label, device, opt in T.available_backends():
    t = time.perf_counter()
    calc = T.calculator('7net-0', device, opt)
    a = atoms.copy()
    a.calc = calc
    results[label] = dict(E=a.get_potential_energy(), F=a.get_forces(), S=a.get_stress())
    setup = time.perf_counter() - t
    calcs[label] = (calc, device)
    print(f'{label:18s} E = {results[label]["E"]:.5f} eV   max |F| = {np.abs(results[label]["F"]).max():.4f} eV/Å'
          f'   (set-up and first call {setup:.1f} s)')"""),
    M("""## Do they agree?

The three backends sum the same numbers in a different order, in single precision, so they differ in
the last digits only."""),
    C("""ref = results['CPU, e3nn']
for label, r in results.items():
    if label == 'CPU, e3nn':
        continue
    print(f'{label:18s} vs CPU:  |dE|/atom = {abs(r["E"] - ref["E"]) / len(atoms):.1e} eV,'
          f'  max |dF| = {np.abs(r["F"] - ref["F"]).max():.1e} eV/Å,'
          f'  max |dσ| = {np.abs(r["S"] - ref["S"]).max():.1e} eV/Å³')"""),
    M("""Differences of order 10⁻⁶ eV/Å in the forces are float32 rounding. They are about a million times
smaller than the forces themselves, and far below the error of the model with respect to DFT.

## How fast is each?

The set-up time printed above includes loading the model and, for the fused kernels, generating and
compiling them; it is paid once. What matters for MD is the steady state: one call from new positions,
as in each MD step."""),
    C("""timing = {}
for label, (calc, device) in calcs.items():
    _, steady = T.time_calls(atoms, calc, device, n=10)
    timing[label] = steady
    print(f'{label:18s} {steady:7.1f} ms per step')

labels = list(timing)
fig, ax = plt.subplots(figsize=(6, 4))
ax.barh(labels, [timing[k] for k in labels], color=['#9AA3AF', '#D95F02', '#3B4CC0'][:len(labels)])
ax.set_xlabel('ms per step (64 atoms)')
longest = max(timing.values())
for i, k in enumerate(labels):
    ax.text(timing[k] + 0.02 * longest, i, f'{timing[k]:.1f} ms', va='center')
ax.set_xlim(0, 1.2 * longest)
plt.tight_layout()
plt.show()"""),
    M("""On an Apple GPU you should see:

* moving the reference code to the GPU helps little, or not at all, at this size: it runs many
  small operations, each writing a large temporary array, and is limited by memory traffic;
* the fused kernels compute each convolution in one pass without those arrays, and are an order of
  magnitude faster.

The answer is the same in all three cases. What changes is how large a system, and how long a
simulation, are practical: the next notebook runs molecular dynamics.

**Next experiment:** repeat the timing with a 3 × 3 × 3 supercell (216 atoms), or with the larger
model `'7net-l3i5'`."""),
]

# --------------------------------------------------------------------------------------------- 02
NB['02_molecular_dynamics'] = [
    M("""# 2. Molecular dynamics, from 64 to 1,000 atoms

We run molecular dynamics (MD) of crystalline silicon at 300 K with 7net-0: first on a small cell with
each backend, to see the same dynamics at different speeds; then on larger cells with the fused
kernels; and finally a longer run, from which we compute the radial distribution function.

MD needs the forces at every time step, so the speed of one force call is the speed of the simulation.
We report it as steps per second and as simulated nanoseconds per day at a 1 fs time step."""),
    SETUP,
    C("""from ase import units
from ase.build import bulk
from ase.constraints import FixCom
from ase.md.langevin import Langevin


def silicon(n):
    return bulk('Si', 'diamond', a=5.43, cubic=True).repeat((n, n, n))


def run_md(atoms, calc, device, steps, T_K=300.0, dt_fs=1.0, seed=1, log_every=0):
    \"\"\"Langevin NVT; returns steps per second over the measured steps and the temperature log.\"\"\"
    atoms = atoms.copy()
    atoms.set_constraint(FixCom())   # keep the centre of mass fixed, as ASE recommends for Langevin
    atoms.calc = calc
    T.thermalize(atoms, T_K, np.random.default_rng(seed))
    dyn = Langevin(atoms, dt_fs * units.fs, temperature_K=T_K, friction=0.01 / units.fs, fixcm=False,
                   rng=np.random.default_rng(seed + 1))
    dyn.run(5)                      # warm-up, not timed
    temps = []
    if log_every:
        dyn.attach(lambda: temps.append(atoms.get_temperature()), interval=log_every)
    T.sync(device)
    t = time.perf_counter()
    dyn.run(steps)
    T.sync(device)
    return steps / (time.perf_counter() - t), temps, atoms


def ns_per_day(sps, dt_fs=1.0):
    return sps * dt_fs * 86400 / 1e6"""),
    M("""## Same system, three backends

64 atoms, 20 steps each. The CPU run takes the longest; it is here to show what the fused kernels
replace."""),
    C("""small = silicon(2)
speed = {}
for label, device, opt in T.available_backends():
    sps, _, _ = run_md(small, T.calculator('7net-0', device, opt), device, steps=20)
    speed[label] = sps
    print(f'{label:18s} {sps:6.1f} steps/s  = {ns_per_day(sps):5.2f} ns/day')"""),
    M("""## Larger cells with the fused kernels

The cost grows with the number of atoms (more precisely, with the number of neighbor pairs). Cells
that would not fit in memory on this machine are skipped."""),
    C("""label, device, opt = T.best_backend()
calc = T.calculator('7net-0', device, opt)
sizes = T.cells_for_memory([2, 3, 4, 5], gb_per_1000_atoms=1.5, natoms=[64, 216, 512, 1000])
scan = []
for n in sizes:
    sps, _, _ = run_md(silicon(n), calc, device, steps=30)
    scan.append((8 * n ** 3, sps))
    print(f'{8 * n ** 3:5d} atoms: {sps:6.1f} steps/s = {ns_per_day(sps):5.2f} ns/day ({label})')
rss, drv = T.memory_gb()
print(f'peak memory of this notebook so far: {rss:.1f} GB (process), {drv:.1f} GB (GPU driver now)')

fig, ax = plt.subplots(figsize=(4.5, 2.8))
ax.plot([s[0] for s in scan], [ns_per_day(s[1]) for s in scan], 'o-', color='#3B4CC0')
ax.set_xlabel('atoms')
ax.set_ylabel('ns / day at 1 fs')
ax.set_xscale('log')
ax.set_yscale('log')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
ax.yaxis.set_minor_formatter(plt.NullFormatter())
ax.set_title(f'7net-0, silicon, {label}')
plt.tight_layout()
plt.show()"""),
    M("""## A longer run and the radial distribution function

512 atoms at 300 K: 1 ps to equilibrate, then 2 ps of production, saving a frame every 20 fs. On an
M5 Pro this takes about two minutes; on a MacBook Air a few more. Set `LONG = True` for a 20 ps run.

The trajectory can be opened in OVITO or with `ase gui md_si512.traj`."""),
    C("""from ase.io.trajectory import Trajectory

LONG = False
n_eq, n_prod = 1000, (20000 if LONG else 2000)
atoms = silicon(4)
sps, temps, atoms = run_md(atoms, calc, device, steps=n_eq, log_every=10)
dyn = Langevin(atoms, 1.0 * units.fs, temperature_K=300.0, friction=0.01 / units.fs, fixcm=False, rng=np.random.default_rng(3))
traj = Trajectory('md_si512.traj', 'w', atoms)
dyn.attach(traj.write, interval=20)
dyn.attach(lambda: temps.append(atoms.get_temperature()), interval=10)
t = time.perf_counter()
dyn.run(n_prod)
dt = time.perf_counter() - t
traj.close()
print(f'{n_prod} steps in {dt:.0f} s: {n_prod / dt:.1f} steps/s = {ns_per_day(n_prod / dt):.2f} ns/day')

fig, ax = plt.subplots(figsize=(5, 2.6))
ax.plot(np.arange(len(temps)) * 10 / 1000, temps, lw=0.8)
ax.axhline(300, color='k', lw=0.6, ls=':')
ax.axvline(n_eq / 1000, color='grey', lw=0.6)
ax.set_xlabel('time (ps)')
ax.set_ylabel('T (K)')
plt.tight_layout()
plt.show()"""),
    C("""from ase.io import read


def rdf(frames, rmax=6.0, nbins=240):
    \"\"\"g(r) averaged over frames, minimum-image distances (cubic cell, rmax below half the cell).\"\"\"
    edges = np.linspace(0, rmax, nbins + 1)
    hist = np.zeros(nbins)
    for a in frames:
        d = a.get_all_distances(mic=True)[np.triu_indices(len(a), 1)]
        hist += np.histogram(d, edges)[0]
    n, V = len(frames[0]), frames[0].get_volume()
    shell = 4 / 3 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    g = hist * 2 / (len(frames) * n * (n / V) * shell)
    return 0.5 * (edges[1:] + edges[:-1]), g


frames = read('md_si512.traj', index=':')
r, g = rdf(frames)
fig, ax = plt.subplots(figsize=(5, 2.6))
ax.plot(r, g, color='#3B4CC0')
for d, lab in ((2.35, '1st'), (3.84, '2nd'), (4.50, '3rd')):
    ax.axvline(d, color='grey', lw=0.5, ls=':')
    ax.text(d, g.max() * 0.9, lab, ha='center', fontsize=8)
ax.set_xlabel('r (Å)')
ax.set_ylabel('g(r)')
ax.set_title('silicon, 300 K, 7net-0')
plt.tight_layout()
plt.show()"""),
    M("""The peaks sit at the first, second and third neighbor distances of diamond silicon (2.35, 3.84 and
4.50 Å), broadened by thermal motion.

**Next experiments:** heat the crystal to 1700 K to watch it melt (and the g(r) change); use the larger
`'7net-l3i5'`; or, on a machine with more memory, a 6 × 6 × 6 cell (1,728 atoms)."""),
]

# --------------------------------------------------------------------------------------------- 03
NB['03_relaxation_and_phonons'] = [
    M("""# 3. Relaxation and phonons

Phonons, the vibrations of a crystal, follow from the force constants: how the force on each atom
changes when another atom moves. Phonopy obtains them from the forces in a few displaced supercells,
which an MLIP gives in seconds.

Steps: relax the cell and atoms with 7net-0, build displaced supercells with Phonopy, compute their
forces, and plot the phonon band structure and density of states of silicon.

This notebook needs `phonopy` (`pip install phonopy`)."""),
    SETUP,
    C("""from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS

label, device, opt = T.best_backend()
calc = T.calculator('7net-0', device, opt)

unit = bulk('Si', 'diamond', a=5.40, cubic=True)      # start a little off the equilibrium lattice
unit.calc = calc
BFGS(FrechetCellFilter(unit), logfile=None).run(fmax=1e-4, steps=200)
a0 = unit.cell.lengths()[0]
print(f'relaxed lattice parameter: {a0:.4f} Å (experiment 5.431 Å; DFT-PBE, the training level of 7net-0, gives about 5.47 Å)')"""),
    M("""## Force constants with Phonopy

A 3 × 3 × 3 supercell of the 8-atom cubic cell (216 atoms). By symmetry, silicon needs a single displaced
supercell."""),
    C("""from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

cell = PhonopyAtoms(symbols=unit.get_chemical_symbols(), cell=unit.cell[:], scaled_positions=unit.get_scaled_positions())
phonon = Phonopy(cell, supercell_matrix=np.diag([3, 3, 3]), primitive_matrix='F')   # F: face-centred cubic primitive cell
phonon.generate_displacements(distance=0.01)
scs = phonon.supercells_with_displacements
print(len(scs), 'displaced supercell(s) of', len(scs[0]), 'atoms')

from ase import Atoms


def forces_of(sc, calc):
    a = Atoms(sc.symbols, cell=sc.cell, scaled_positions=sc.scaled_positions, pbc=True)
    a.calc = calc
    return a.get_forces()


t = time.perf_counter()
phonon.forces = [forces_of(sc, calc) for sc in scs]
print(f'forces with {label}: {time.perf_counter() - t:.2f} s')
phonon.produce_force_constants()"""),
    C("""from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

path = [[[0, 0, 0], [0.5, 0, 0.5], [0.625, 0.25, 0.625]], [[0.375, 0.375, 0.75], [0, 0, 0], [0.5, 0.5, 0.5]]]
labels = ['Γ', 'X', 'U', 'K', 'Γ', 'L']
qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=60)
phonon.run_band_structure(qpoints, path_connections=connections, labels=labels)
phonon.run_mesh([16, 16, 16], is_gamma_center=True)
phonon.run_total_dos()
fig = phonon.plot_band_structure_and_dos()
fig.suptitle('silicon, 7net-0')
plt.show()

freqs = phonon.run_qpoints([[0, 0, 0]]).frequencies[0]
print('optical frequency at Γ: %.2f THz (experiment 15.5 THz)' % max(freqs))"""),
    M("""The band structure has the familiar shape of silicon, but the optical branch at Γ is too low: about
12.5 THz, against 15.5 THz measured and about 15 THz from DFT-PBE. This is the model, not the code: every
backend gives the same forces. Universal potentials trained mostly on structures near equilibrium are
known to soften phonons. Larger and more recently trained models do better, which we can check in a
minute, since each phonon calculation takes seconds."""),
    C("""def gamma_optical(model, **kw):
    c = T.calculator(model, device, opt, **kw)
    u = bulk('Si', 'diamond', a=5.40, cubic=True)
    u.calc = c
    BFGS(FrechetCellFilter(u), logfile=None).run(fmax=1e-4, steps=200)
    p = Phonopy(PhonopyAtoms(symbols=u.get_chemical_symbols(), cell=u.cell[:], scaled_positions=u.get_scaled_positions()),
                supercell_matrix=np.diag([3, 3, 3]), primitive_matrix='F')
    p.generate_displacements(distance=0.01)
    p.forces = [forces_of(sc, c) for sc in p.supercells_with_displacements]
    p.produce_force_constants()
    return u.cell.lengths()[0], max(p.run_qpoints([[0, 0, 0]]).frequencies[0])


# 7net-omni is a 100 MB download on first use; set to False to skip it
WITH_OMNI = True
cases = [('7net-0', {}), ('7net-l3i5', {})] + ([('7net-omni', {'modal': 'mpa'}), ('7net-omni', {'modal': 'omat24'})] if WITH_OMNI else [])
for model, kw in cases:
    t = time.perf_counter()
    a_, f_ = gamma_optical(model, **kw)
    print(f'{model:10s} {kw.get("modal", ""):7s} a = {a_:.4f} Å, optical at Γ = {f_:5.2f} THz ({time.perf_counter() - t:.0f} s)')
print('experiment: a = 5.431 Å, 15.5 THz')"""),
    M("""The trend is the useful lesson: the choice of model matters more than any setting of the calculation.
For production work on a given material, fine-tuning on a few DFT structures of that material removes
most of the softening.

**Next experiments:** compare the time of the force step on the CPU (`T.calculator('7net-0', 'cpu',
'off')`); a lower-symmetry crystal needs many more displaced supercells, where the speed matters more."""),
]

# --------------------------------------------------------------------------------------------- 04
NB['04_born_effective_charges'] = [
    M("""# 4. SevenNet-Polar: Born effective charges

In an ionic crystal, moving an atom changes the polarization. The **Born effective charge** tensor
Z\\*<sub>i,ab</sub> = ∂P<sub>a</sub>/∂r<sub>i,b</sub> measures that change; it controls infrared
activity, the splitting of longitudinal and transverse optical phonons, and the force on an atom in an
electric field. DFT gets it from expensive linear-response calculations. SevenNet-Polar predicts it
alongside energy, forces and stress.

We use two public SevenNet-Polar models (Lu *et al.*, data and models at
[doi:10.5281/zenodo.23734482](https://doi.org/10.5281/zenodo.23734482), CC-BY-4.0), downloaded once and
cached:

* **SevenNet-PM-S**, multitask: energy, forces, stress and Z\\* for ZrO₂ and Li₃PO₄;
* **SevenNet-PS-S**, specialized: Z\\* only, for ZrO₂, Li₃PO₄ and perovskite oxides.

On a machine with more memory, **SevenNet-PM-M** (`'SevenNet-PM-M.pth'`) is the larger multitask model.

The structures are 24 frames of the same record, with their DFT values of Z\\*. They come from the
data the models were trained and validated on, so the comparison below is a consistency check, not a
test on unseen structures."""),
    SETUP,
    C("""from ase.io import read

pm_s = T.zenodo_file('SevenNet-PM-S.pth')
ps_s = T.zenodo_file('SevenNet-PS-S.pth')
frames = read(T.HERE / 'data' / 'zro2_li3po4_sample.extxyz', index=':')
print(len(frames), 'frames:', sorted({a.get_chemical_formula(mode='reduce') for a in frames}))"""),
    M("""## Predict Z\\* and compare with DFT"""),
    C("""label, device, opt = T.best_backend()
models = {'PM-S': T.calculator(pm_s, device, opt), 'PS-S': T.calculator(ps_s, device, opt)}
pred = {k: [] for k in models}
dft = []
t = time.perf_counter()
for a in frames:
    dft.append(a.calc.results['born_effective_charges'].reshape(-1, 3, 3))
    for k, calc in models.items():
        b = a.copy()
        b.calc = calc
        b.get_potential_energy()
        pred[k].append(np.asarray(calc.results['born_effective_charges']).reshape(-1, 3, 3))
print(f'{len(frames)} frames, two models, {label}: {time.perf_counter() - t:.1f} s')

fig, axs = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
D = np.concatenate(dft)
for ax, k in zip(axs, models):
    P = np.concatenate(pred[k])
    diag = np.eye(3, dtype=bool)
    ax.plot(D[:, diag].ravel(), P[:, diag].ravel(), '.', ms=2, alpha=0.5, label='diagonal')
    ax.plot(D[:, ~diag].ravel(), P[:, ~diag].ravel(), '.', ms=2, alpha=0.5, label='off-diagonal')
    lim = [D.min() - 0.3, D.max() + 0.3]
    ax.plot(lim, lim, 'k:', lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    rmse = np.sqrt(np.mean((P - D) ** 2))
    ax.set_title(f'SevenNet-{k}: RMSE {rmse:.3f} e')
    ax.set_xlabel('Z* from DFT (e)')
axs[0].set_ylabel('Z* predicted (e)')
axs[0].legend(markerscale=6)
plt.tight_layout()
plt.show()"""),
    M("""## The acoustic sum rule

Moving the whole crystal rigidly does not change its polarization, so in a neutral cell the Born
charges of all atoms sum to zero for each Cartesian component. In a cell that carries a net charge Q,
the sum is Q on the diagonal instead. The sample has such cells: Li₃PO₄ with one Li removed, keeping the
electrons, so Q = −1 e. The models are not constrained to obey either rule exactly; how close they come
is a useful check."""),
    C("""from collections import defaultdict

groups = defaultdict(list)
for i, a in enumerate(frames):
    groups[a.get_chemical_formula(mode='hill')].append(i)
print(f'{"cell":12s} {"DFT":>7s} {"PM-S":>7s} {"PS-S":>7s}   (diagonal of the sum over atoms, mean over frames, e)')
for f, idx in sorted(groups.items()):
    row = [np.mean([np.trace(z[i].sum(axis=0)) / 3 for i in idx]) for z in (dft, pred['PM-S'], pred['PS-S'])]
    print(f'{f:12s} ' + ' '.join(f'{v:7.3f}' for v in row))"""),
    M("""The neutral cells sum to zero, and the charged cells to about −1 e, in DFT and in both models: they
have learnt the rule from the data. The finite-field calculator below can enforce the neutral rule
exactly (`enforce_asr=True`)."""),
    M("""## Forces in an electric field

To first order, a uniform field **E** adds a force on each atom given by its Born charge:
F<sub>i,b</sub>(E) − F<sub>i,b</sub>(0) = Σ<sub>a</sub> Z\\*<sub>i,ab</sub> E<sub>a</sub>.
`FieldCalculator` wraps a SevenNet-Polar calculator to add it, which is what MD in a field uses. We
check the relation on one ZrO₂ cell, with the sum rule enforced (`enforce_asr=True`)."""),
    C("""from sevenn.calculator import FieldCalculator

a = next(f for f in frames if set(f.get_chemical_symbols()) == {'Zr', 'O'}).copy()
base = T.calculator(pm_s, device, opt)
field = FieldCalculator(base, enforce_asr=True)
E = np.array([0.0, 0.0, 0.05])                          # V/Å
a.calc = field
field.set_efield([0.0, 0.0, 0.0])
F0 = a.get_forces().copy()
field.set_efield(E)
F1 = a.get_forces().copy()
Z = np.asarray(field.results['born_effective_charges']).reshape(-1, 3, 3)
expected = np.einsum('iab,a->ib', Z, E)
print('max |(F(E) - F(0)) - Z*ᵀE| =', f'{np.abs((F1 - F0) - expected).max():.1e} eV/Å',
      f'(field forces up to {np.abs(expected).max():.3f} eV/Å)')"""),
    M("""The field force is exactly the Born-charge term. With it, molecular dynamics in a field runs as in
the previous notebook, with `FieldCalculator` in place of the plain calculator.

**Next experiments:** compare the time of the Z\\* predictions on the CPU and the GPU
(`T.available_backends()`), or predict Z\\* along an MD trajectory of ZrO₂ from notebook 2 using PM-S."""),
]


for name, cells in NB.items():
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    nbf.write(nb, HERE / f'{name}.ipynb')
    print('wrote', name)
