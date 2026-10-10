# Measured tutorial validation

Executed on 2026-10-10, developed on `tutorials/colab-bec`. The source notebooks
contain no outputs. Compact machine-readable records, learning curves, metrics,
ASR diagnostics and selected figures are in [results](results/README.md).
Full executed notebooks and checkpoint ZIPs were also saved during development.

## End-to-end runs

Both notebooks completed at their default epoch counts, including real training,
best-validation checkpoint selection, CLI inference, parity plots, ASE reload
agreement, ASR diagnostics and artifact export. Saved executed notebooks were
checked for error outputs. The final A100 run also completed every optional
FlashTP check. Runtime allocation is excluded from all times below.

| Notebook / device | Channels / epochs | Training wall time | Reloaded CLI inference | Setup to first parity | Test BEC diagonal / off-diagonal RMSE (e) |
|---|---:|---:|---:|---:|---:|
| Quickstart / Apple M5 Pro CPU | 16 / 20 | 150.6 s | 9.0 s | 162.0 s | 0.1103 / 0.0800 |
| Quickstart / Colab T4 | 16 / 20 | 73.0 s | 13.0 s | 114.6 s | 0.1103 / 0.0800 |
| Multitask / Apple M5 Pro CPU | 16 / 40 | 262.6 s | 8.0 s | 273.8 s | 0.0932 / 0.0685 |
| Multitask / Colab A100-SXM4-40GB | 32 / 40 | 110.0 s | 12.0 s | 144.3 s | 0.1013 / 0.0699 |

CPU setup used the installed environment (`--skip-install`); GPU runs fetched the
public branch and installed the fork. CPU/A100 multitask models have different
channel counts, so these rows are not an accelerator speedup comparison.
Training wall time includes CLI startup, preprocessing, construction, optimization
and checkpoint writing. Inference includes a separate CLI process and reload.

The T4 checkout fetch/install took **3.6 / 12.8 s**; final A100 **3.4 / 8.2 s**.
The supplied PyTorch wheel was retained. Quickstart T4 training peaked at
**0.402 GiB allocated / 0.502 GiB reserved**; A100 32-channel standard multitask
training at **0.704 / 1.473 GiB**. These are PyTorch process peaks, not total GPU
memory or notebook-session memory.

Colab tested **Python 3.13.15, PyTorch 2.11.0+cu130, CUDA toolkit 13.0.88,
e3nn 0.6.0, ASE 3.29.0, numpy 2.1.3, matscipy 1.3.0,
torch-geometric 2.8.0.post1, matplotlib 3.10.0**, T4 `sm_75` and A100 `sm_80`.
CPU tested Python 3.10.20, PyTorch 2.5.1, e3nn 0.6.0 and ASE 3.28.0.
All training used float32, batch size 1 and two CPU threads. Each report records
its own full dependency versions, configuration, source commit and data hash.

## Accuracy and physical limits

| Multitask held-out component RMSE | CPU, 16 channels | A100, 32 channels |
|---|---:|---:|
| Energy (eV/atom) | 0.0321 | 0.0283 |
| Force (eV/Å) | 0.2864 | 0.2897 |
| Stress (GPa) | 2.6630 | 2.6064 |

CLI BEC predictions and ASE checkpoint reloading agree within absolute/relative
2e-5 tolerance. Stress references were independently checked against ASE's DFT
labels and tensile-positive convention. Force/stress convergence plots convert
CLI norm RMSE to the component RMSE used by parity tables: divide by √3 for
force and by 10√6 for stress in GPa. Raw CLI CSVs remain unchanged.

The real 30-structure example uses an explicit composition-stratified **20/5/5**
whole-structure split. Test labels do not supply training statistics or checkpoint
selection. Geometry fingerprints screen duplicates, and E/F/S/BEC serialization
round trips pass. The data lack trajectory/parent IDs, so these errors demonstrate
training and evaluation, not independent generalization.

Raw maximum test **ΣᵢZ*ᵢ** component residuals are **2.087 e** for the T4 quickstart
and **3.435 e** for final A100 multitask predictions, versus approximately **3e-6 e**
for reference labels. No ASR projection is applied. BEC parity is informative,
but ASR, force and stress errors establish that these small fits are educational
models and need more data and validation before research use. Quickstart E/F
outputs are untrained. Neither notebook trains dielectric response.

## Default and configuration experiments

An initial 40-epoch CPU BEC fit took **293.1 s**, with test BEC RMSE
**0.0961 / 0.0663 e**. The following validation points came from that single
learning curve, not separately timed runs:

| Epoch | Validation diagonal RMSE (e) | Validation off-diagonal RMSE (e) |
|---|---:|---:|
| 5 | 0.1484 | 0.1104 |
| 10 | 0.1136 | 0.0818 |
| 20 | 0.0962 | 0.0796 |
| 30 | 0.0815 | 0.0666 |
| 40 | 0.0861 | 0.0621 |

Twenty epochs provide a useful quickstart parity plot at about half this CPU
training cost. Validation, not test error, informed that choice. T4 exploratory
runs with 8 channels / 2 layers / batch 1 / 10 epochs took **27.0 s**, with best
validation BEC RMSE **0.2453 / 0.1399 e**. A 16-channel / 3-layer / batch 4 /
10-epoch run took **23.0 s**, but reached only **2.0104 / 0.2267 e** and used
**1.457 GiB** peak allocated memory. These are different optimization budgets:
batch 4 has fewer updates per epoch. They do not establish batch-4 inferiority
at equal optimizer steps.

The initial 16-channel multitask mean-energy baseline gave CPU test energy RMSE
**0.1259 eV/atom**. Species reference energies estimated from training data improved
validation energy behavior; the final default uses that existing CLI option.
The corresponding final CPU held-out energy RMSE is **0.0321 eV/atom**.

## FlashTP: actual builds, numerical checks and equal-work timing

Source revision **`d52e41454afc9636cd51757b65df44e7bcdca178`** built and imported
on both GPUs, with architecture detection and `MAX_JOBS=2`. Fresh build/install
cost **480.1 s on A100** and **723.4 s on T4**. Compilation dominates a short
exercise. The final A100 rerun reused the installed build after checking the
pinned clean source checkout; its report explicitly records zero new build time.

The pinned kernel requires channel multiplicities divisible by **32**. The initial
16-channel attempt failed its `mul % WARPSIZE == 0` assertion after successful
compilation. Corrected 32-channel T4 and A100 runs passed actual fused-backend
selection, E/F/S/BEC output, weighted-loss and all **23 parameter-gradient** checks,
including derivatives needed for force/stress training. T4's corrected check was
an exploratory one-epoch model; final A100 used its trained 40-epoch checkpoint.

For final A100, maximum output differences were: energy **0**, force
**1.43e-6**, stress **1.12e-8**, raw BEC irreps **9.54e-7**. Maximum gradient
absolute difference was **9.87e-8**. Output absolute/relative tolerances were
1e-4 and gradient tolerances 5e-4; weighted losses agreed within 1e-4.

| A100 operation | Standard e3nn | FlashTP | e3nn / FlashTP |
|---|---:|---:|---:|
| E/F/S/BEC forward, median | 41.02 ms | 26.81 ms | 1.53× |
| Multitask forward + backward, median | 101.25 ms | 66.31 ms | 1.53× |
| Full 5-epoch CLI training wall time | 24.00 s | 20.00 s | 1.20× |
| Isolated CLI training peak allocated | 0.704 GiB | 0.189 GiB | — |

Microbenchmarks use the same checkpoint/batch, three warmups, ten timed samples,
CUDA synchronization and float32. Both models are resident, so their process-wide
microbenchmark memory is not an isolated backend comparison. Full training runs
use separate CLI processes, identical architecture/data/seed, and 100 optimizer
steps each. Build time is excluded. Five-epoch learning curves agree closely
(maximum validation weighted-loss difference 1.4e-5); both are exported.
These are single-session measurements, not guaranteed speedups across workloads.

## Automated verification

- Seven tutorial/BEC inference tests pass: real one-epoch CLI training, all-label
  round trips, split integrity, plots/physical stress conversion, ASE reload,
  and a nonsymmetric BEC tensor across single-atom/unequal batches.
- Local selected existing regressions: **142 passed, 20 skipped**.
- GitHub full unit suite: **192 passed, 51 skipped** on both Python 3.8 and 3.13.
  Accelerator/integration skips are not claimed as coverage.
- GitHub Python 3.13 CPU workflow executes both notebooks with one epoch through
  nbclient, which stops on cell errors; formatting/lint checks pass.

The official Colab CLI can continue notebook execution after a cell error.
An abandoned rerun with stale imports was discarded; its timings are excluded.
Final runs used clean kernels, verified the fork/commit, and checked saved notebook
error outputs and FlashTP status. Only named tutorial test VMs were released;
pre-existing user sessions were preserved.
