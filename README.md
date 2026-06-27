# SevenNet-Polar

SevenNet-Polar is based on the original [SevenNet](https://github.com/sevennet-dev/sevennet) package, a graph neural network (GNN)-based interatomic potential package.

This package extends SevenNet by adding support for Born Effective Charge (BEC) and macroscopic dielectric constant ($\epsilon_\infty$) fitting. Additionally, it features an [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/) calculator and a LAMMPS interface that support multi-GPU execution.

For general information on the base SevenNet package, please refer to the [SevenNet documentation](https://sevennet.readthedocs.io/en/latest/).

## Features
 - Born Effective Charge (BEC) and macroscopic dielectric constant ($\epsilon_\infty$) fitting
 - [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/) calculator (python) with multi-GPU support
 - GPU-parallelized molecular dynamics with LAMMPS, featuring multi-GPU support
 - Pretrained GNN interatomic potential and fine-tuning interface
 - CUDA-accelerated D3 (van der Waals) dispersion
 - Multi-fidelity training for combining multiple databases with different calculation settings
 - [Tensor product accelerators](https://sevennet.readthedocs.io/en/latest/user_guide/accelerator.html)

## Installation and user guides

Installation (including LAMMPS and D3) and user guides for the base package can be found in the [SevenNet documentation](https://sevennet.readthedocs.io/en/latest/).

The old README (prior to v0.12.0) can be found [here](./docs/old_readme/).

## Citation

If you use SevenNet-Polar, please cite our upcoming paper:
```bib
@article{sevennet_polar_placeholder,
	title = {SevenNet-Polar: [Placeholder Title]},
	journal = {arXiv placeholder},
	author = {[Placeholder Author List]},
	year = {202X},
}
```

If you use the base SevenNet code, please cite:
```bib
@article{park_scalable_2024,
	title = {Scalable Parallel Algorithm for Graph Neural Network Interatomic Potentials in Molecular Dynamics Simulations},
	volume = {20},
	doi = {10.1021/acs.jctc.4c00190},
	number = {11},
	journal = {J. Chem. Theory Comput.},
	author = {Park, Yutack and Kim, Jaesun and Hwang, Seungwoo and Han, Seungwu},
	year = {2024},
	pages = {4857--4868},
}
```
