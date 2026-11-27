# SevenNet-Polar on the Apple GPU

```python
from sevenn.calculator import FieldCalculator, SevenNetCalculator

base = SevenNetCalculator('checkpoint.pth', device='mps')    # fused kernels: moving the model to 'mps' switches them on
calc = FieldCalculator(base, enforce_asr=True)
calc.set_efield([0.0, 0.0, 0.05])                             # V/Angstrom, Direct models
atoms.calc = calc

from sevenn.mps import report
print(report(base.model))                                     # what runs how
```

## The switch

The switch is the device: moving a model to `mps` (`model.to('mps')`, which the calculator, `Trainer` and `sevenn
inference` do) converts float64 parameters to float32, switches the e3nn layers to fused Metal kernels and pads edge
lists; moving it away (`model.to('cpu')`) undoes it. The model object is changed in place.

| setting | effect |
|---|---|
| `device='mps'` | fused Metal kernels, edge padding |
| `device='mps'` with the environment variable `SEVENN_DISABLE_MPS=1` | the reference e3nn code, unchanged, on the GPU (float32) |
| `device='auto'` | CUDA if available, else the Apple GPU if this PyTorch can run on it, else the CPU; the CPU also with `SEVENN_DISABLE_MPS=1` and for a deployed TorchScript model |
| `device='cpu'` or `'cuda'` | nothing changes; the Apple code is not imported |
| `sevenn inference --device mps ...` | evaluates a dataset with the fused kernels; batches padded to 256 edges |
| `device: mps` under `train` in the YAML, `Trainer(..., device='mps')` | training on the fused kernels |

The 0.3 options (`mps_opt`, `mps_edge_bucket`, `SEVENNET_MPS_OPT`, `--enable_mps`) are gone in 0.3.3.

A model built with cuEquivariance, FlashTP or OpenEquivariance cannot move to the Apple GPU (error). The fused
kernels need `torch.mps.compile_shader`, public from PyTorch 2.7; 0.3.3 is tested on PyTorch 2.14 and 2.15 (SevenNet-Polar
0.3 on 2.10 to 2.15; on 2.7 to 2.9 it ran checkpoints, but a model on the GPU could not be deep-copied). PyTorch 2.6, or
weights other than float32, run the reference code on the GPU with a warning. Before PyTorch 2.6 the GPU cannot run SevenNet at all (no `scatter_reduce` on MPS), so the move stops with an
error. A module the kernels do not cover (a uvu path whose input and output multiplicities differ, a gate activation
other than silu and tanh, a non-one-hot self-connection operand) keeps the e3nn code on the GPU, with a warning;
`report(model)` counts them. Nothing moves to the CPU. Leave `PYTORCH_ENABLE_MPS_FALLBACK` unset: SevenNet does not
need it, and it is what would let PyTorch move operations to the CPU silently.

The kernel tables are not buffers, so `state_dict` and checkpoints are not affected. Checkpoints load unchanged.
Derivatives with respect to positions, strain and field are complete. Parameter gradients in eval mode are not: the
linear and self-connection layers treat their weights as constants there and return none, with no error (the radial
networks and convolutions do return them); use train mode or the CPU. Weights written through `param.data` are not
noticed by the eval tables: load them with `load_state_dict`, or move the model to the CPU and back. Second
derivatives: the convolution kernels have derivatives to any order, including mixed ones through non-contiguous
inputs since 0.3.3; the linear and gate kernels have first derivatives only, so `create_graph=True` in eval mode
raises an error from those layers. The kernels index with 32-bit integers: a convolution needs edges × weights per
edge below 2^31 (SevenNet-PM-M: about 493,000 edges, some 6,000 atoms of ZrO2), whatever the memory; larger graphs
stop with an error that names `SEVENN_DISABLE_MPS=1`.

Training: in `model.train()` the convolutions keep running the fused kernels. The product is linear in each of its inputs, so
each derivative of the backward pass is one of the same three kernels with an input swapped; written as autograd functions that
call each other, they give derivatives to any order and keep only their inputs, as FlashTP does on NVIDIA. Linear and
self-connection layers rebuild their matrix from the live weights at each call and run one matrix product, and the gate is a
few elementwise operations, so autograd differentiates them to any order too. Their tables are built at the first call in
train mode (16 s and 4 GB in all for 7net-omni at 96 atoms, measured with 0.3). `Trainer` on `device='mps'` switches this on;
`SEVENN_DISABLE_MPS=1` keeps e3nn.

| model, batch | e3nn: step, memory kept for backward | fused: step, memory kept |
|---|---|---|
| SevenNet-PM-S, 4 × 96 atoms | 2,530 ms, 5.56 GB | 113 ms, 1.25 GB |
| SevenNet-PM-M, 4 × 12 atoms | 2,590 ms, 3.91 GB | 148 ms, 0.71 GB |
| 7net-0, 4 × 96 atoms | 1,370 ms, 3.70 GB | 90 ms, 0.93 GB |
| 7net-l3i5, 2 × 96 atoms | 2,050 ms, 4.12 GB | 92 ms, 0.69 GB |

SevenNet-Polar 0.3.3, PyTorch 2.14.1, M5 Pro, plugged in; one repeated batch of rattled ZrO2 cells (Adam with
learning rate 0, so the inputs stay fixed), median of six steps after three; the reference code through
`SEVENN_DISABLE_MPS=1`. Gradients equal to e3nn within 1.4e-6 relative, forces within 7.2e-6 eV/Å. Without a cap on
the MPS allocator, the e3nn run of SevenNet-PM-S passed 20 GB of process footprint, the limit of these runs; it was
measured with `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3`. Through `Trainer`, a 7net-0 fine-tune epoch of 8 structures took 0.24 s
against 2.9 s (measured with 0.3).

Batches are padded to a multiple of 256 edges in training and `sevenn inference`, so the GPU
sees few tensor shapes: PyTorch compiles and keeps a graph for every new shape, and memory
would otherwise grow with every batch of a shuffled data set (fixed in 0.3.2). Only the edge count is
padded. On the data sets measured (ZrO2-type cells of about 100 atoms, batches of 1 to 4, ten minutes)
memory stays flat; a data set with widely varying atom counts brings new shapes of other tensors
too, which PyTorch also keeps, so check the memory of a first epoch there.

Training on a small dataset: keep `num_workers: 0` (the default). On macOS each
DataLoader worker is a new process that imports torch, e3nn and SevenNet, about 3 s,
and the workers start again for every epoch. On the Polar example (30 ZrO₂ frames,
one per batch) an epoch took 7.6 s with 4 workers and 1.2 s without on the Apple
GPU, against 11 s on the CPU (measured with 0.3).

## LAMMPS with `lmp -in`

Serial only, with the same input as on an NVIDIA GPU. There is no deploy step: `pair_coeff` takes the checkpoint, or a pretrained name, in place of the deployed model. Keywords are those of the NVIDIA pair style (`efield Ex Ey Ez`, `enforce_asr yes|no`), plus `modal NAME` for multi-fidelity models.

```
pair_style  e3gnn
pair_coeff  * * checkpoint.pth efield 0 0 0.05 Zr O
```

On a Mac with an Apple GPU, `sevenn patch_lammps` installs this pair style in place of the TorchScript ones, also for a CPU build (`SEVENN_DISABLE_MPS=1` keeps the TorchScript pair styles, which take deployed `.pt` models; `--enable_flash`, `--enable_oeq` and `--d3` need CUDA). Keep the Python environment that ran `patch_lammps`: LAMMPS runs SevenNet inside it, and it needs no libtorch link. `SEVENN_PYTHON` and `SEVENN_ROOT` override the environment and the SevenNet folder recorded in the build; with the plugin, `SEVENN_PYTHON` must use the same Python version as the build.

**Option 1, build LAMMPS (same steps as FlashTP, `lammps_torch` page of the docs):**

```bash
git clone https://github.com/lammps/lammps.git lammps_sevenn --branch stable_22Jul2025_update4 --depth=1
```

```bash
sevenn patch_lammps ./lammps_sevenn
```

```bash
cd lammps_sevenn && mkdir build && cd build && cmake ../cmake -DBUILD_MPI=no -DBUILD_OMP=no && make -j 6
```

```bash
./lmp -in in.md_e3gnn -var ckpt SevenNet-PM-S.pth -var data tZrO2_12atom.data -var nx 2 -var ny 2 -var nz 2 -var steps 100
```

**Option 2, no build, with `pip install lammps mpich`:** the same pair style as a plugin.

```bash
SRC="$(python -c "import sevenn.mps.lammps.patch as p; print(p.SRC)")"
bash "$SRC/build_plugin.sh" python
```

```bash
export LAMMPS_PLUGIN_PATH="$SRC"
```

The plugin is built next to its sources, so the installed package must be writable.

Then `lmp -in` as above. `06_lammps_serial.py` (`fix external`, no compiler at all) stays for scripts that drive LAMMPS from Python.

Per-atom energies work; per-atom stress does not, as on NVIDIA. The model builds its own edges, so LAMMPS builds no pair neighbor list.

It needs `units metal` and one MPI rank, and stops with an error otherwise. So do `neigh_modify exclude` and `include`, which it cannot honour (it sees all atoms), and `pair_style hybrid` or `hybrid/overlay`, since it computes every atom. There is no neighbor list request, as the model builds its own edges (computes such as `rdf` request their own).

**With a field,** the reported energy is the zero-field U, as on NVIDIA. Direct models add Z*ᵀE to the forces, so the forces are not the gradient of the reported energy. Energy-based minimization and energy-conservation checks can mislead under a field, and a thermostatted run is field-driven dynamics, not an equilibrium finite-field ensemble. `enforce_asr yes` is for neutral cells.

## Examples

All take `--ckpt` (a SevenNet or SevenNet-Polar Direct checkpoint; `SevenNet-PM-S.pth` from the SevenNet-Polar models on Zenodo, doi:10.5281/zenodo.23734482, is used below). Examples for ZrO2 with 96 atoms use `--repeat 2` (the default). `--kernels e3nn` runs the reference
e3nn code on the GPU instead of the fused kernels (as `SEVENN_DISABLE_MPS=1`); `05_benchmark.py` runs both:

```bash
python 01_single_point.py --ckpt SevenNet-PM-S.pth
```

```bash
python 01_single_point.py --ckpt SevenNet-PM-S.pth --efield 0 0 0.05
```

```bash
python 02_md.py --ckpt SevenNet-PM-S.pth --steps 200 --efield 0 0 0.05
```

```bash
python 03_relax.py --ckpt SevenNet-PM-S.pth --fmax 0.02
```

```bash
python 04_plan_memory.py --ckpt SevenNet-PM-S.pth --atoms 2000 --limit-gb 14
```

```bash
python 05_benchmark.py --ckpt SevenNet-PM-S.pth --repeat 3
```

## Serial LAMMPS

`06_lammps_serial.py` runs LAMMPS through its Python module (`pip install lammps`): LAMMPS does the dynamics, and the calculator gives energy, forces and virial through `fix external`. No C++, no patched pair style, with or without a field. In your input, replace the pair style as in `in.md_external`:

```
pair_style   zero 6.0
pair_coeff   * *
fix          sevenn all external pf/callback 1 1
fix_modify   sevenn energy yes virial yes
```

```bash
python 06_lammps_serial.py --ckpt SevenNet-PM-S.pth --efield 0 0 0.05 --steps 100
```

Serial only. It runs as `python 06_lammps_serial.py`, not `lmp -in`. A deployed TorchScript model cannot use these kernels, and parallel LAMMPS is not supported on a Mac.

## Memory

`_common.py` caps the MPS allocator through `PYTORCH_MPS_HIGH_WATERMARK_RATIO` before torch is imported: 0.25 of the
recommended working set, about 10 GB on a 48 GB Mac but about 1.4 GB on an 8 GB one, where the reference e3nn code of
SevenNet-PM-S at 96 atoms already needs more; set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6` there. `04_plan_memory.py` predicts the
peak for a target size from two short runs. Never set the ratio to 0: this disables PyTorch's guard against exhausting the machine's memory.
Unified memory is shared with everything else on the Mac, so count process RSS and `torch.mps.driver_allocated_memory()` together.
