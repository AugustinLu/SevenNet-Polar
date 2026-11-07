# SevenNet-Polar on the Apple GPU

```python
from sevenn.calculator import FieldCalculator, SevenNetCalculator

base = SevenNetCalculator('checkpoint.pth', device='mps')    # fused kernels are the default on device='mps'
calc = FieldCalculator(base, enforce_asr=True)
calc.set_efield([0.0, 0.0, 0.05])                             # V/Angstrom, Direct models
atoms.calc = calc
print(base.mps_runtime.mode, base.mps_runtime.report)         # what runs how
```

## The switch

| setting | effect |
|---|---|
| `SevenNetCalculator(..., device='mps')` or `mps_opt='fused'` | fused Metal kernels, edge padding (default on the Apple GPU) |
| `mps_opt='off'`, or environment `SEVENNET_MPS_OPT=off` | the reference e3nn code, unchanged, on the GPU |
| `mps_edge_bucket=0` | no edge padding (slower MD: each new edge count costs about 20 ms at 96 atoms) |
| `device='auto'` or `'cpu'` or `'cuda'` | nothing changes; the Apple code is not imported |
| `sevenn inference --enable_mps ...` (or `--device mps`) | evaluates a dataset with the fused kernels (no edge padding); `SEVENNET_MPS_OPT=off` for e3nn |
| `sevenn train input.yaml --enable_mps` (or `device: mps` in the YAML), `Trainer(..., device='mps')` | training on the fused kernels; `SEVENNET_MPS_OPT=off` for e3nn |

Explicit `mps_opt='fused'` on another device, or on a deployed TorchScript model, is an error. With the default setting a TorchScript model, or
a PyTorch without `torch.mps.compile_shader` (2.6), runs the reference code on the GPU with a warning. Before PyTorch 2.6 the GPU cannot
run SevenNet at all (no `scatter_reduce` on MPS), so `device='mps'` stops with an error. A module the kernels do not cover (a uvu path whose input and output
multiplicities differ, a gate activation other than silu and tanh, a non-one-hot self-connection operand) keeps the e3nn code on the GPU, with a
warning; `base.mps_runtime.report` counts them. Nothing moves to the CPU. Leave `PYTORCH_ENABLE_MPS_FALLBACK` unset: SevenNet does not need
it, and it is what would let PyTorch move operations to the CPU silently (the calculator warns if it is set).

Call `disable(model)` before saving a whole model object or moving it off the Apple GPU: the kernel tables are not buffers (`state_dict` is not affected). All tests pass on PyTorch 2.10 to 2.15; on 2.7 to 2.9 checkpoints run but a model object cannot be deep-copied, and below 2.7 the reference code runs. Checkpoints load unchanged. Derivatives with respect to positions, strain and field are complete. Parameter gradients in eval mode are not: use train mode or `mps_opt='off'`. Second derivatives: the convolution kernels have derivatives to any order; the linear and gate kernels have first derivatives
only, so `create_graph=True` in eval mode raises an error from those layers.

Training: in `model.train()` the convolutions keep running the fused kernels. The product is linear in each of its inputs, so
each derivative of the backward pass is one of the same three kernels with an input swapped; written as autograd functions that
call each other, they give derivatives to any order and keep only their inputs, as FlashTP does on NVIDIA. Linear and
self-connection layers rebuild their matrix from the live weights at each call and run one matrix product, and the gate is a
few elementwise operations, so autograd differentiates them to any order too. Their tables are built at the first call in
train mode (16 s and 4 GB in all for 7net-omni at 96 atoms). `Trainer` on `device='mps'` switches this on;
`SEVENNET_MPS_OPT=off` keeps e3nn.

| model, batch | e3nn: step, memory kept for backward | fused: step, memory kept |
|---|---|---|
| S Direct, 4 × 96 atoms | 2,430 ms, 5.56 GB | 109 ms, 1.23 GB |
| M Direct, 4 × 12 atoms | 2,410 ms, 3.91 GB | 133 ms, 0.59 GB |
| 7net-0, 4 × 96 atoms | 1,400 ms, 3.70 GB | 99 ms, 0.93 GB |
| 7net-l3i5, 2 × 96 atoms | 2,110 ms, 4.13 GB | 100 ms, 0.69 GB |

M5 Pro, plugged in; gradients equal to e3nn within 1.4e-6 relative. Through `Trainer`, a 7net-0 fine-tune epoch of 8 structures
takes 0.24 s against 2.9 s.

Training on a small dataset: keep `num_workers: 0` (the default). On macOS each
DataLoader worker is a new process that imports torch, e3nn and SevenNet, about 3 s,
and the workers start again for every epoch. On the Polar example (30 ZrO₂ frames,
one per batch) an epoch takes 7.6 s with 4 workers and 1.2 s without on the Apple
GPU, against 11 s on the CPU.

## LAMMPS with `lmp -in`

Serial only, with the same input as on an NVIDIA GPU. There is no deploy step: `pair_coeff` takes the checkpoint, or a pretrained name, in place of the deployed model. Keywords are those of the NVIDIA pair style (`efield Ex Ey Ez`, `enforce_asr yes|no`), plus `modal NAME` for multi-fidelity models.

```
pair_style  e3gnn
pair_coeff  * * checkpoint.pth efield 0 0 0.05 Zr O
```

`--enable_flash`, `--enable_oeq` and `--d3` need CUDA and cannot be combined with `--enable_mps`. Keep the Python environment that ran `patch_lammps`: LAMMPS runs SevenNet inside it, and it needs no libtorch link. `SEVENN_PYTHON` and `SEVENN_ROOT` override the environment and the SevenNet folder recorded in the build; with the plugin, `SEVENN_PYTHON` must use the same Python version as the build.

**Option 1, build LAMMPS (same steps as FlashTP, `lammps_torch` page of the docs):**

```bash
git clone https://github.com/lammps/lammps.git lammps_sevenn --branch stable_22Jul2025_update4 --depth=1
```

```bash
sevenn patch_lammps ./lammps_sevenn --enable_mps
```

```bash
cd lammps_sevenn && mkdir build && cd build && cmake ../cmake -DBUILD_MPI=no -DBUILD_OMP=no && make -j 6
```

```bash
./lmp -in in.md_e3gnn -var ckpt D_base.pth -var data tZrO2_12atom.data -var nx 2 -var ny 2 -var nz 2 -var steps 100
```

**Option 2, no build, with `pip install lammps mpich`:** the same pair style as a plugin.

```bash
sh sevenn/pair_e3gnn_mps/build_plugin.sh python
```

```bash
export LAMMPS_PLUGIN_PATH=$PWD/sevenn/pair_e3gnn_mps
```

Then `lmp -in` as above. `06_lammps_serial.py` (`fix external`, no compiler at all) stays for scripts that drive LAMMPS from Python.

Per-atom energies work; per-atom stress does not, as on NVIDIA. The model builds its own edges, so LAMMPS builds no pair neighbor list.

It needs `units metal` and one MPI rank, and stops with an error otherwise. So does `neigh_modify exclude`, which it cannot honour: it sees all atoms. `pair_style hybrid/overlay` works, with all atom types mapped to elements: `NULL` is not supported, and there is no neighbor list request, as the model builds its own edges (computes such as `rdf` request their own).

**With a field,** the reported energy is the zero-field U, as on NVIDIA. Direct models add Z*E to the forces, so the forces are not the gradient of the reported energy. Energy-based minimization and energy-conservation checks can mislead under a field.

## Examples

All take `--ckpt` (a SevenNet or SevenNet-Polar Direct checkpoint). Examples for ZrO2 with 96 atoms use `--repeat 2` (the default):

```bash
python 01_single_point.py --ckpt D_base.pth
```

```bash
python 01_single_point.py --ckpt D_base.pth --efield 0 0 0.05
```

```bash
python 02_md.py --ckpt D_base.pth --steps 200 --efield 0 0 0.05
```

```bash
python 03_relax.py --ckpt D_base.pth --fmax 0.02
```

```bash
python 04_plan_memory.py --ckpt D_base.pth --atoms 2000 --limit-gb 14
```

```bash
python 05_benchmark.py --ckpt D_base.pth --repeat 3
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
python 06_lammps_serial.py --ckpt D_base.pth --efield 0 0 0.05 --steps 100
```

Serial only. It runs as `python 06_lammps_serial.py`, not `lmp -in`. A deployed TorchScript model cannot use these kernels, and parallel LAMMPS is not supported on a Mac.

## Memory

`_common.py` caps the MPS allocator through `PYTORCH_MPS_HIGH_WATERMARK_RATIO` before torch is imported. `04_plan_memory.py` predicts the
peak for a target size from two short runs. Never set the ratio to 0: this disables PyTorch's guard against exhausting the machine's memory.
Unified memory is shared with everything else on the Mac, so count process RSS and `torch.mps.driver_allocated_memory()` together.
