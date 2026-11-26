# Apple GPU (MPS)

On a Mac with Apple silicon, SevenNet runs on the GPU through PyTorch's MPS backend. Moving a
model to `mps` switches the e3nn layers to fused Metal kernels and pads edge lists so that the
GPU sees few tensor shapes. Nothing else changes in the inputs or the commands.

## Requirements

- macOS on Apple silicon, PyTorch 2.7 or later (2.6 runs the reference e3nn code on the GPU).
- float32: MPS has no float64. Shift and scale stored in float64 are converted when the model
  moves to the GPU, so results agree with the CPU to float32 precision, not bit for bit.

## Usage

```python
from sevenn.calculator import SevenNetCalculator

calc = SevenNetCalculator('7net-0', device='mps')   # or device='auto' on a Mac without CUDA
```

```bash
sevenn inference --device mps 7net-0 structures.extxyz
sevenn train input.yaml        # with device: mps under train
```

`Trainer`, `sevenn.scripts.inference.inference` and user scripts behave the same way: the
switch happens in `model.to('mps')`, and `model.to('cpu')` undoes it. A model instance given to
the calculator is changed in place.

## Selection and opt-out

| Request | Result |
|---|---|
| `device='cpu'` | CPU; the Apple GPU code is not imported |
| `device='mps'` | Apple GPU, fused kernels where supported, edge padding |
| `device='mps'` and `SEVENN_DISABLE_MPS=1` | Apple GPU, reference e3nn code, no padding |
| `device='auto'`, no CUDA | Apple GPU (CPU with `SEVENN_DISABLE_MPS=1`) |
| `device='auto'`, CUDA available | CUDA |
| model built with cuEquivariance, FlashTP, OpenEquivariance or LunaTP, moved to `mps` | error |

`SEVENN_DISABLE_MPS` accepts `1`, `true` or `yes`. Modules the kernels do not cover keep the
e3nn code on the GPU with a warning; nothing moves to the CPU.

## LAMMPS

TorchScript cannot hold the Metal kernels, so the Apple GPU pair style runs the model in
Python and loads the checkpoint itself: there is no `sevenn get_model` step. It is serial
(one MPI rank) and takes `pair_style e3gnn` with
`pair_coeff * * checkpoint.pth [modal NAME] element1 element2 ...`.

Patched build: on a Mac with an Apple GPU, `sevenn patch_lammps` installs this pair style
(`SEVENN_DISABLE_MPS=1` keeps the CUDA one). The binary uses the Python environment that ran
the patch.

```bash
sevenn patch_lammps ./lammps
cd lammps && mkdir build && cd build
cmake ../cmake -DBUILD_MPI=no -DBUILD_OMP=no && make -j 8
```

Plugin for `pip install lammps`, from an installed SevenNet:

```bash
SRC=$(python -c "import sevenn.mps.lammps.patch as p; print(p.SRC)")
bash $SRC/build_plugin.sh python
export LAMMPS_PLUGIN_PATH=$SRC
```

The pair style refuses what it cannot honour: several ranks, `neigh_modify include`,
`pair_style hybrid`, deployed `.pt` files.

## Performance

(to be filled from the candidate benchmark)

## Limits

- Eval mode: linear and self-connection layers give first derivatives only and treat their
  weights as constants (forces, stress and training are complete; for other parameter
  gradients use train mode or the CPU).
- 32-bit indices: a graph above about 2^31 elements per kernel buffer is refused.
- Autocast keeps the reference code.
