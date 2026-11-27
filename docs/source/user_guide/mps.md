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
switch happens in `model.to('mps')`, and `model.to('cpu')` undoes it; parameters converted to
float32 stay float32. A model instance given to the calculator is changed in place.

## Selection and opt-out

| Request | Result |
|---|---|
| `device='cpu'` | CPU; the Apple GPU code is not imported |
| `device='mps'` | Apple GPU, fused kernels where supported; edge padding when every convolution is fused |
| `device='mps'` and `SEVENN_DISABLE_MPS=1` | Apple GPU, reference e3nn code, no padding |
| `device='auto'`, no CUDA | Apple GPU if this PyTorch can run on it (2.6 or later), else CPU; CPU with `SEVENN_DISABLE_MPS=1`, and for a deployed TorchScript model |
| `device='auto'`, CUDA available | CUDA |
| model built with cuEquivariance, FlashTP, OpenEquivariance or LunaTP, moved to `mps` | error |

`SEVENN_DISABLE_MPS` accepts `1`, `true` or `yes`. Modules the kernels do not cover keep the
e3nn code on the GPU with a warning; nothing moves to the CPU, unless `PYTORCH_ENABLE_MPS_FALLBACK=1`
is set, which lets PyTorch run operations on the CPU silently (SevenNet warns about it).

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
SRC="$(python -c "import sevenn.mps.lammps.patch as p; print(p.SRC)")"
bash "$SRC/build_plugin.sh" python
export LAMMPS_PLUGIN_PATH="$SRC"
```

The plugin is built next to its sources, so the installed package must be writable. The pair
style refuses what it cannot honour: several MPI ranks, units other than `metal`,
`neigh_modify exclude` and `include`, `pair_style hybrid` and `hybrid/overlay`, deployed `.pt`
files; per-atom stress is not available.

## Performance

ZrO2, energy, forces and stress per ASE call (ms): the median of 10 calls after 3 warm-up calls,
averaged over two runs on the GPU (one on the CPU); MacBook Pro M5 Pro (48 GB), PyTorch 2.14.1:

| Model, atoms | fused | e3nn on GPU | CPU |
|---|---|---|---|
| 7net-0, 96 | 15.9 | 145 | 271 |
| 7net-0, 768 | 65.1 | 1,011 | |
| 7net-l3i5, 96 | 22.5 | 380 | 693 |
| 7net-l3i5, 768 | 103 | > 20 GB | |
| SevenNet-PM-S, 96 | 19.0 | 254 | 488 |
| SevenNet-PM-S, 768 | 76.8 | > 20 GB | |

"> 20 GB": stopped at the memory cap of the measurement; empty: not measured. Training numbers:
`examples/apple_gpu/README.md`.

## Limits

- Eval mode: linear, self-connection and gate layers give first derivatives only, and linear and
  self-connection layers treat their weights as constants (forces, stress and training are complete; for other parameter
  gradients use train mode or the CPU).
- 32-bit indices: a graph above about 2^31 elements per kernel buffer is refused.
- Autocast keeps the reference code.
