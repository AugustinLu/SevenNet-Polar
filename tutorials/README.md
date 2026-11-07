# Tutorials

Notebooks that run real calculations with SevenNet and SevenNet-Polar on a laptop. Each one starts from a
small system, checks the result, and then does something larger. They run on any computer; on a Mac
with Apple silicon they use the Apple GPU and its fused kernels, which are much faster.

| Notebook | What you compute | Model |
|---|---|---|
| `01_first_calculation` | energy, forces and stress of silicon on the CPU, the Apple GPU, and the Apple GPU with fused kernels; check they agree; compare speed | 7net-0 |
| `02_molecular_dynamics` | MD of silicon at 300 K from 64 to 1,000 atoms; steps per second and ns/day; radial distribution function | 7net-0 |
| `03_relaxation_and_phonons` | relaxed lattice parameter, phonon band structure and density of states of silicon | 7net-0 |
| `04_born_effective_charges` | Born effective charges of ZrO₂ and Li₃PO₄ against DFT; the acoustic sum rule; forces in an electric field | SevenNet-PM-S, SevenNet-PS-S |

## Setup

From the root of the repository:

```bash
pip install -e .
pip install jupyter phonopy
jupyter lab tutorials/
```

The fused kernels need PyTorch 2.10 or later on an Apple silicon Mac. Without them, the notebooks run
with the reference code and say so. The SevenNet-Polar models of notebook 4 are downloaded once from
Zenodo ([doi:10.5281/zenodo.23734482](https://doi.org/10.5281/zenodo.23734482), CC-BY-4.0) into
`~/.cache/sevennet-polar`; set `SEVENN_TUTORIAL_CACHE` to use another folder.

## Hardware

Memory sets the largest system. As a guide, 7net-0 needs about 1.5 GB per 1,000 atoms with the fused
kernels; the notebooks skip cells that would not fit. On a MacBook Air, which has no fan, long runs slow
down as the machine warms up. SevenNet-PM-M, the larger multitask model, is suited to machines with
32 GB or more.

## Data

`data/zro2_li3po4_sample.extxyz` holds 24 frames (12 ZrO₂, 12 Li₃PO₄) of the Zenodo record above, with
their DFT energies, forces, stress and Born effective charges; each frame records its source index.

To rebuild the notebooks from `build_notebooks.py`: `python build_notebooks.py`.
