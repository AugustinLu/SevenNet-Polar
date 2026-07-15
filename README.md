# SevenNet-Polar

SevenNet-Polar is based on the original [SevenNet](https://github.com/sevennet-dev/sevennet) package, a graph neural network (GNN)-based interatomic potential package.

This package extends SevenNet by adding support for Born Effective Charge (BEC) fitting. Additionally, it features an [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/) calculator and a LAMMPS interface that support multi-GPU execution.

For general information on the base SevenNet package, please refer to the [SevenNet documentation](https://sevennet.readthedocs.io/en/latest/).

## Features
 - Born Effective Charge (BEC) fitting
 - [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/) calculator (python) with multi-GPU support
 - GPU-parallelized molecular dynamics with LAMMPS, featuring multi-GPU support
 - Pretrained GNN interatomic potential and fine-tuning interface
 - CUDA-accelerated D3 (van der Waals) dispersion
 - Multi-fidelity training for combining multiple databases with different calculation settings
 - [Tensor product accelerators](https://sevennet.readthedocs.io/en/latest/user_guide/accelerator.html)

## Installation and user guides

Installation (including LAMMPS and D3) and user guides for the base package can be found in the [SevenNet documentation](https://sevennet.readthedocs.io/en/latest/).

The old README (prior to v0.12.0) can be found [here](./docs/old_readme/).


## Training and Using SevenNet-Polar

SevenNet-Polar introduces new configurations to train models that predict Born Effective Charges (BEC). These options are additions to the regular SevenNet training configuration.

### Training Configuration

To enable BEC training, use `is_train_bec: True` and set the `bec_loss_weight`. The validation metrics for BEC can be tracked using `['BornEffectiveCharges', 'DiagRMSE']` and `['BornEffectiveCharges', 'OffDiagRMSE']` in the `error_record`.

**Single-task (BEC only) training example:**

```yaml
model:
  chemical_species: 'auto'
  cutoff: 6.0
  channel: 64
  lmax: 3
  num_convolution_layer: 4

train:
  random_seed: 1
  num_workers: 4
  epoch: 500
  is_train_stress: False
  is_train_bec: True
  loss: 'mse'
  optimizer: 'adam'
  optim_param:
      lr: 0.005
  scheduler: 'exponentiallr'
  scheduler_param:
      gamma: 0.992
  energy_loss_weight: 0.0
  force_loss_weight: 0.0
  bec_loss_weight: 10.0
  stress_loss_weight: 0.0
  per_epoch: 10
  error_record:
    - ['Energy', 'RMSE']
    - ['Force', 'RMSE']
    - ['Stress', 'RMSE']
    - ['BornEffectiveCharges', 'DiagRMSE']
    - ['BornEffectiveCharges', 'OffDiagRMSE']
    - ['TotalLoss', 'None']

data:
  batch_size: 4
  shift: 'per_atom_energy_mean'
  scale: 1.0
  data_format: 'ase'
  data_format_args:
    format: 'extxyz'
    index: '::'

  load_trainset_path: ['train.xyz']
  load_validset_path: ['val.xyz']
  load_testset_path:  ['test.xyz']
```

**Multi-task (Energy, Forces, Stress, and BEC) training example:**

```yaml
model:
  chemical_species: 'auto'
  cutoff: 6.0
  channel: 64
  lmax: 3
  num_convolution_layer: 4

train:
  random_seed: 1
  num_workers: 4
  epoch: 500
  is_train_stress: True
  is_train_bec: True
  loss: 'mse'
  optimizer: 'adam'
  optim_param:
      lr: 0.005
  scheduler: 'exponentiallr'
  scheduler_param:
      gamma: 0.992
  energy_loss_weight: 1.0
  force_loss_weight: 0.1
  bec_loss_weight: 10.0
  stress_loss_weight: 1.0e-6
  per_epoch: 10
  error_record:
    - ['Energy', 'RMSE']
    - ['Force', 'RMSE']
    - ['Stress', 'RMSE']
    - ['BornEffectiveCharges', 'DiagRMSE']
    - ['BornEffectiveCharges', 'OffDiagRMSE']
    - ['TotalLoss', 'None']

data:
  batch_size: 4
  shift: 'per_atom_energy_mean'
  scale: 1.0
  data_format: 'ase'
  data_format_args:
    format: 'extxyz'
    index: '::'

  load_trainset_path: ['train.xyz']
  load_validset_path: ['val.xyz']
  load_testset_path:  ['test.xyz']
```

### LAMMPS Interface with Electric Field

The LAMMPS interface supports applying an external electric field directly through the `pair_coeff` command by adding the `efield` keyword followed by the field vector components (in eV/Å/e).

**Serial Calculation:**

```lammps
pair_style     e3gnn
# Apply an electric field of 0.01 in the z-direction
pair_coeff     * * model.pt efield 0.0 0.0 0.01 Zr O
```

**Parallel GPU Calculation:**

```lammps
# Use parallel pair_style
pair_style     e3gnn/parallel
# Apply an electric field of 0.01 in the z-direction (requires specifying the number of message-passing layers, e.g., 4)
pair_coeff     * * 4 deployed_parallel_model_dir efield 0.0 0.0 0.01 Zr O
```

## Citation

If you use SevenNet-Polar, please cite our upcoming paper:
```bib
@article{lu_sevennet_polar_202X,
	title = {SevenNet-Polar for MultiTask Prediction of Energy, Forces, Stress, and Born Effective Charges: Development and Application to ZrO2, Li3PO4, and Perovskites},
	journal = {arXiv preprint},
	author = {Lu, Anh Khoa Augustin and Arai, Shungo and Park, Yutack and Han, Seungwu and Miyazaki, Tsuyoshi and Watanabe, Satoshi},
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

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

SevenNet-Polar is a fork of the original SevenNet package, which is also licensed under the MIT License by Yutack Park. The original license is preserved in the [LICENSE.SevenNet](LICENSE.SevenNet) file.
