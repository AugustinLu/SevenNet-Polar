# SevenNet-Polar Python and Google Colab tutorials

| Tutorial | Runtime | Start |
|---|---|---|
| [BEC quickstart](01_bec_quickstart.ipynb) | CPU or free T4; standard e3nn | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AugustinLu/SevenNet-Polar/blob/main/tutorials/colab/01_bec_quickstart.ipynb) |
| [Multitask E/F/S/BEC](02_multitask_a100_flashtp.ipynb) | A100 when available; standard path with optional FlashTP comparison | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AugustinLu/SevenNet-Polar/blob/main/tutorials/colab/02_multitask_a100_flashtp.ipynb) |

Choose a Colab runtime and **Run all**, or open either notebook in local Jupyter with PyTorch installed. Setup installs this fork, preserves the runtime's PyTorch wheel, and verifies `sevenn`'s source path. The notebooks fetch `main`; `report.json` records the exact source commit and dependency versions. For exact reproduction, check out that commit and run locally or set `SEVENNET_TUTORIAL_REPO` to its path. `SEVENNET_TUTORIAL_REF` selects a remote branch during development; use a fresh kernel to avoid reusing an older checkout.

Both tutorials use 50 training + 20 validation + 20 test structures from the official Li₃PO₄ dataset. Both models use lmax=2, four layers, batch size 4 and 100 epochs: the quickstart has 16 channels; the advanced notebook has 32 on CUDA and 16 on CPU. These compact models are intended for education. It is not a production potential. The quickstart optimizes only BEC loss; its energy/force outputs must not be used. The advanced notebook fits E/F/S/BEC together. Neither trains dielectric response.

See [data attribution and split limitations](../../example_inputs/tutorial_polar/DATA.md), [measured validation](VALIDATION.md), [subset comparison](SUBSET_STUDY.md), and the repository's [citation information](../../README.md#citation). Please cite the SevenNet-Polar paper and dataset, upstream SevenNet, and FlashTP when used.

## Optional LAMMPS field example

The multitask notebook includes a disabled-by-default `RUN_FIELD_MD` cell: 512 stoichiometric Li₃PO₄ atoms, 100 NVT steps at 0.25 fs, a 100 K thermostat target and a 0.001 V/Å field. It builds an isolated, compatible serial LAMMPS executable or uses a supplied `LAMMPS_EXECUTABLE`. Initial zero-field and applied-field forces are checked against ASE; the archive includes input scripts, deployed model, logs, trajectory, plots and timings. This uses standard e3nn and requires trained energy/force outputs. Build time is separate from training; [measured validation](VALIDATION.md) records tested cost. The short trajectory demonstrates the workflow and does not establish equilibration or long-term stability.

## Using VS Code with Colab

With [Google's Colab extension](https://github.com/googlecolab/colab-vscode), open the local `.ipynb`, sign in, choose **Select Kernel → Colab → Auto Connect**, and select/create a runtime. Run the setup cell first and check the printed GPU and `sevenn` source path. Code executes on the remote runtime, so local Python environments and local helper files are not automatically present there; these notebooks fetch their own repository. Save the executed notebook locally and download its exported ZIP before removing the runtime with **Colab: Remove Server**. Disconnecting the editor is not a substitute for deleting an idle compute runtime.

For repeatable comparisons, change one configuration at a time, restart the kernel before changing repository revision or backend installation, and retain each unique run directory/archive. Start with standard e3nn and leave both optional compilation sections disabled for the shortest test. The one-epoch override checks execution; the default 100 epochs produces the fitted example. Colab availability and actual GPU allocation vary.

## What each run saves

Each execution creates a unique `polar-tutorial-*` directory and a ZIP archive. It contains the best validation checkpoint, YAML configuration, split data and source hash, training/inference logs, epoch CSV, partition-specific RMSE/MAE, parity PNG/PDF, convergence figure, machine/dependency details and timings. The notebook offers a Colab download control. Runtime allocation is excluded from setup-to-first-parity timing; package installation, subprocess training, subprocess inference, and optional FlashTP build are recorded separately. GPU peaks are PyTorch allocation/reservation, not total device memory.

Training and inference use the package CLI; ASE checkpoint reloading must agree with CLI BECs. Stress CSV values are converted from pressure-sign kbar to tensile-positive GPa. BEC predictions are uncorrected, with all six off-diagonal components retained.

## Local validation

Install test dependencies in the same Python environment as SevenNet-Polar:

```bash
python -m pip install nbformat nbclient ipykernel pytest
PYTHONPATH=. MPLBACKEND=Agg python -m pytest tests/tutorials tests/unit_tests/test_bec_inference.py
python tutorials/colab/execute_notebook.py tutorials/colab/01_bec_quickstart.ipynb --output /tmp/polar-bec-executed.ipynb --skip-install
python tutorials/colab/execute_notebook.py tutorials/colab/02_multitask_a100_flashtp.ipynb --output /tmp/polar-multitask-executed.ipynb --skip-install --flash off
```

`--epochs 1` is a plumbing smoke test, not scientific accuracy validation. Omit `--skip-install` to exercise setup. `SEVENNET_TUTORIAL_THREADS` controls CPU threads (default 2). The source notebooks remain output-free; executed copies live at `--output`.

## Colab execution with the official CLI

After installing and signing in to [Google's Colab CLI](https://github.com/googlecolab/google-colab-cli), use a named session:

```bash
colab new -s polar-t4 --gpu T4
colab exec -s polar-t4 --timeout 1800 -f tutorials/colab/01_bec_quickstart.ipynb
colab stop -s polar-t4
```

Repeat on an A100 for the advanced notebook. Download each run's ZIP before stopping the session (`colab download -s NAME REMOTE_ZIP LOCAL_ZIP`). Restart an already-used test kernel with `colab restart-kernel -s NAME` first. CLI notebook execution can continue after a cell error: inspect the saved executed notebook for error outputs before accepting a run. The CLI uploads code cells, not local helper files; the notebook obtains its helpers by cloning this repository. Release test VMs after use.

## Troubleshooting

- Wrong `sevenn` origin: restart the kernel and run setup first. Installing PyPI `sevenn` alone gives upstream SevenNet.
- Different GPU: follow the detected hardware and record its name; paid subscriptions do not guarantee A100 allocation.
- Out of memory: set batch size to 1. If FlashTP compilation fails, lower build parallelism from two jobs to one or disable the optional build.
- FlashTP channels: use multiples of 32 with the pinned kernel; 16-channel models cannot use it. A fresh build took about 8 minutes on A100 and 12 minutes on T4 in our tests.
- No `nvcc`, failed build, failed forward/backward agreement: continue with standard e3nn. `flash_report.json` records the failure; no speedup is implied.
- Slow CPU: use fewer epochs for a smoke test or select T4. An incomplete fit should be labeled as such.

## Developer diagnostics

The introductory notebook flow focuses on RMSE, parity and checkpoint reuse. The [separate dataset study](SUBSET_STUDY.md) retains tensor-sum diagnostics and their reference-label caveats for future refinement. `support.make_config(..., monitor_asr=True)` opts into native monitoring; it is off in the notebooks. The validation-only experiment harness enables these extra diagnostics.

## Stoichiometric ZrO₂ alternative

The [restricted ZrO₂ study](STOICHIOMETRIC_ZRO2.md) tests 48/96-atom cells with oxygen-vacancy frames excluded. To try it, choose `DATASET='zro2_stoichiometric'`, 50 training structures, 32 channels and 100 epochs. Its fixed 20/20 holdouts and official labels are preserved. The existing Li₃PO₄ defaults remain available; the optional MD example uses Li₃PO₄.

## Reproduce the bounded subset comparison

The bundled pools contain only 240 structures per material: a nested reservoir of 200 training structures and fixed 20/20 validation/test sets. Original source IDs, checksums and composition-balanced selection are saved in the manifests. No full-dataset training is needed. Li₁₁O₁₆P₄ reference tensor sums are approximately 1 e; these labels are retained, and `sum_error_max_abs_e` measures agreement with their tensor sums.

Use a fresh output directory for each configuration. This command runs the production CLI and compares **training and validation only**, leaving the test set unused:

```bash
python tutorials/colab/subset_experiment.py --dataset li3po4 --sizes 20 30 50 100 200 --epochs 100 --channels 32 --layers 4 --batch-size 4 --output /tmp/polar-li-subsets
```

Use `--multitask` for E/F/S/BEC, or `--dataset zro2` for ZrO₂. On an HPC runtime with an installed CUDA extension, `--multitask --flash --channels 32` enables FlashTP and verifies outputs/loss/gradients against e3nn. It does not install the extension; record its version separately. Each size saves its exact configuration, timings, fixed split, validation errors, tensor-sum diagnostics and checkpoint archive. `summary.csv` collects the comparison. Runs are single-seed observations; larger subsets do not guarantee smaller worst-case sum errors.

For notebook customization, set `DATASET` and `TRAINING_STRUCTURES` in the data cell, or the environment variables `SEVENNET_TUTORIAL_DATASET` and `SEVENNET_TUTORIAL_TRAINING_STRUCTURES` before execution. The 20 validation and 20 test frames remain fixed. One epoch checks installation and execution; it does not reproduce the fitted model's accuracy.
