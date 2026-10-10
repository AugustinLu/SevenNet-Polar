# Measured tutorial validation

## Revised 50/20/20 Li₃PO₄ notebooks

Real end-to-end Colab runs completed on October 11, 2026 (Japan time), with 50 training, 20 validation and 20 test structures, lmax=2, four layers, batch size 4 and 100 epochs. The quickstart uses 16 channels and BEC-only loss; multitask uses 32 channels on CUDA and E/F/S/BEC loss. These are single-run measurements, with two CPU threads and float32 throughout.

| Notebook / Colab GPU | Training subprocess wall time | Separate CLI inference | Setup to first parity | Test BEC diagonal / off-diagonal RMSE (e) |
|---|---:|---:|---:|---:|
| BEC quickstart / T4 | 356.1 s (5.9 min) | 17.0 s | 410.9 s | 0.04228 / 0.02943 |
| Multitask / T4 | 492.2 s (8.2 min) | 17.0 s | 543.2 s | 0.05881 / 0.04077 |
| Multitask / A100-SXM4-40GB | 257.0 s (4.3 min) | 13.0 s | 295.5 s | 0.05881 / 0.04077 |

Training includes process startup, preprocessing, model construction, optimization and checkpoint writing. Runtime allocation and optional compilation are excluded. Installation took 15.9 / 13.0 / 8.3 s in table order. PyTorch training peaks were 1.38 / 2.45 / 2.45 GiB allocated, and 2.38 / 4.72 / 4.72 GiB reserved, respectively; these are process peaks, not total device memory.

The recorded runs include developer tensor-sum monitoring, which has subsequently been disabled in the introductory notebooks. Training data, objective and model settings are unchanged. Exact measured commits and configurations remain in each report; the final simplified notebook flow is separately checked by one-epoch CPU execution (both completed with zero cell errors). No timing is presented as a rerun of a later source revision.

Multitask test component RMSE on T4 was **0.04448 eV/atom energy, 0.27540 eV/Å force and 0.77930 GPa stress**. A100 and T4 held-out results agree within 2×10⁻⁶ in the reported units. CLI predictions agree with independent ASE checkpoint reloading. Stress references are checked against ASE's tensile-positive convention; force/stress learning curves convert CLI norm RMSE to component RMSE. Both notebooks train genuine labels and export partition-specific parity plots.

The bounded official dataset pools, checksums, composition-balanced whole-structure splits and serialization checks are documented in [DATA.md](../../example_inputs/tutorial_polar/DATA.md). The data lack trajectory/parent IDs: these metrics demonstrate training and evaluation, and do not establish independent scientific generalization. Configuration selection used validation only. Neither notebook trains dielectric response, and the BEC-only model's forces are untrained.

Compact records are in [t4_quickstart](results/subset_study/t4_quickstart), [t4_multitask](results/subset_study/t4_multitask) and [a100_multitask](results/subset_study/a100_multitask). Full executed notebooks, checkpoints and ZIP exports were retrieved and checked for cell errors and archive integrity. Colab used Python 3.13.15, PyTorch 2.11.0+cu130, CUDA toolkit 13.0.88, e3nn 0.6.0 and ASE 3.29.0; reports retain the complete environment.

## Optional FlashTP on A100

A fresh pinned **d52e41454afc9636cd51757b65df44e7bcdca178** build took **503.1 s**, excluding its 2.0 s fetch, with two compiler jobs. The 32-channel, four-layer, batch-four multitask checkpoint passed E/F/S/BEC output, weighted-loss and all **29 parameter-gradient** comparisons. Maximum output differences were 3.05×10⁻⁵ eV energy, 3.81×10⁻⁶ eV/Å force, 2.98×10⁻⁸ in the internal stress units, and 9.54×10⁻⁷ for raw BEC irreps. Maximum parameter-gradient difference was 9.72×10⁻⁷. Output tolerance was 10⁻⁴ and gradient tolerance 5×10⁻⁴.

| Same A100 checkpoint and batch | Standard e3nn | FlashTP |
|---|---:|---:|
| Median E/F/S/BEC forward | 55.94 ms | 34.96 ms |
| Median forward + backward | 137.11 ms | 84.95 ms |
| Separate full five-epoch CLI training | 24.00 s | 20.00 s |

Microbenchmarks use three warmups, ten synchronized samples, the same checkpoint and float32. Both models remain resident, so microbenchmark memory is process-wide. Equal-work CLI runs use separate processes with the same seed/configuration; build time is excluded. For a short tutorial, compilation can outweigh this measured runtime gain. T4 standard training needs no FlashTP. The [historical validation](VALIDATION_30_FRAME.md) retains earlier T4 build checks, not a claim that this final four-layer model was FlashTP-tested on T4.

## Bounded data/configuration experiments and Mosso

The [separate developer study](SUBSET_STUDY.md) records the completed 20/30/50/100/200 training-size sweep with fixed 20/20 holdouts for both ZrO₂ and Li₃PO₄. Only the bounded 240-frame pools were supplied to GPU runs; no full-dataset training was performed. Li₃PO₄ with 50 training structures gives useful plots and fits within the T4 time budget. Extra diagnostics remain in that study, outside the introductory notebook flow.

Mosso Slurm validation completed BEC-16, multitask-32 standard and multitask-32 installed-FlashTP jobs. CLI times were 200.2 / 203.2 / 143.0 s on the individually recorded A100-PCIe/A800/A100-PCIe GPUs. Different devices prevent treating these wall times as a pure backend speedup. The same-checkpoint, same-GPU FlashTP numerical/gradient comparison also passed. All training jobs exited 0:0 and have the required job-ledger entries.

## Optional 512-atom LAMMPS field demonstration

The public serial `e3gnn` interface successfully ran **512 atoms, 100 NVT steps, 0.25 fs timestep (25 fs total)** under a **0.001 V/Å z field**. The initial cell is a 2×2×4 repetition of a stoichiometric Li₃PO₄ training structure. The thermostat target is 100 K with a 50 fs damping time. This uses the multitask checkpoint and standard e3nn; the BEC-only checkpoint is explicitly rejected.

| Runtime | Fresh LAMMPS build | MD loop (100 steps) | Seconds per step | Max initial force difference from ASE |
|---|---:|---:|---:|---:|
| Colab T4 | 612.8 s (10.2 min) | 32.39 s | 0.324 | 4.51×10⁻⁵ eV/Å |
| Mosso A100-80GB PCIe | Existing installation | 6.23 s | 0.0623 | 4.64×10⁻⁵ eV/Å |

These are different model checkpoints/environments and do not isolate hardware speedup. Colab builds the supported `stable_2Aug2023_update3` LAMMPS revision **46265e36ce79e4b42c9e5229b72a0ce2485845cd** with two compiler jobs, `BUILD_MPI=OFF`, and the documented MKL include-path workaround against PyTorch 2.11.0+cu130. Build logs are exported. Its separate deployment, zero-field check and field-process costs were 11.1 / 1.9 / 34.6 s. The loop timing excludes deployment, ASE comparison and compilation.

Both zero-field and field initial forces pass independent ASE comparisons at absolute/relative tolerance 3×10⁻⁴. All 11 saved trajectory frames have finite positions and forces. Temperature rises from approximately 100 K to 204 K during the 25 fs run: **this is not an equilibrated trajectory or evidence of long-time stability**. Maximum force norm is about 3.06 eV/Å. The thermostat damping time exceeds the trajectory duration, and the initial DFT frame need not be relaxed on the small fitted model. Energy and pressure logged by LAMMPS are the model's zero-field quantities; force checks use the pair style's documented field defaults.

Short MD is practical on T4; fresh compilation is the dominant added cost. The optional notebook cell remains off by default. Input scripts, measurements and trajectory summaries/plots are in [t4_md512](results/subset_study/t4_md512) and [mosso_md512](results/subset_study/mosso_md512). Full trajectories, deployed models and build logs are retained in verified development archives and produced by the optional notebook export.

## Automated verification and reproduction

The final core unit suite passed **196 tests, with 51 skips**; all **16 tutorial/BEC regressions passed**, including real CLI training, split/label round trips, nonsymmetric BEC conversion across unequal batches, native monitoring and checkpoint reload. New MD checks verify that rotation/repetition preserves periodic neighbor distances and that BEC-only checkpoints are rejected before MD launch. GitHub also executes both output-free notebooks through nbclient with one epoch, and runs full Python 3.8/3.13 unit and formatting checks. Accelerator/integration skips are not claimed as coverage.

For self-testing, use a fresh execution directory per configuration and save the ZIP before stopping Colab. See [the tutorial README](README.md) for local, CLI and notebook commands. A one-epoch run checks plumbing; omit the override to reproduce training quality. Only named VMs created for this work are released; pre-existing user sessions are preserved.

The earlier 30-frame ZrO₂ measurements are preserved in [historical validation](VALIDATION_30_FRAME.md); they do not describe the revised defaults.
