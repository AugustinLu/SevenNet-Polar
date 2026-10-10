# SevenNet-Polar Python and Google Colab tutorials

| Tutorial | Runtime | Start |
|---|---|---|
| [BEC quickstart](01_bec_quickstart.ipynb) | CPU or free T4; standard e3nn | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AugustinLu/SevenNet-Polar/blob/main/tutorials/colab/01_bec_quickstart.ipynb) |
| [Multitask E/F/S/BEC](02_multitask_a100_flashtp.ipynb) | A100 when available; standard path with optional FlashTP comparison | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AugustinLu/SevenNet-Polar/blob/main/tutorials/colab/02_multitask_a100_flashtp.ipynb) |

Choose a Colab runtime and **Run all**, or open either notebook in local Jupyter with PyTorch installed. Setup installs this fork, preserves the runtime's PyTorch wheel, and verifies `sevenn`'s source path. The notebooks fetch `main`; `report.json` records the exact source commit and dependency versions. For exact reproduction, check out that commit and run locally or set `SEVENNET_TUTORIAL_REPO` to its path.

Both tutorials use the public 30-structure ZrO₂ example. Both models use lmax=2 and three layers: the quickstart has 16 channels; the advanced notebook has 32 on CUDA and 16 on CPU. These compact models are intended for education. It is not a production potential. The quickstart optimizes only BEC loss; its energy/force outputs must not be used. The advanced notebook fits E/F/S/BEC together. Neither trains dielectric response.

See [data attribution and split limitations](../../example_inputs/training_polar/DATA.md), [measured validation](VALIDATION.md), and the repository's [citation information](../../README.md#citation). Please cite the SevenNet-Polar paper and dataset, upstream SevenNet, and FlashTP when used.

## What each run saves

Each execution creates a unique `polar-tutorial-*` directory and a ZIP archive. It contains the best validation checkpoint, YAML configuration, split data and source hash, training/inference logs, epoch CSV, partition-specific RMSE/MAE, raw acoustic-sum-rule residuals, parity PNG/PDF, convergence figure, machine/dependency details and timings. The notebook offers a Colab download control. Runtime allocation is excluded from setup-to-first-parity timing; package installation, subprocess training, subprocess inference, and optional FlashTP build are recorded separately. GPU peaks are PyTorch allocation/reservation, not total device memory.

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

The notebooks retain the CLI's established loss definitions and normalization. Changes for tutorial support are restricted to preserving BEC predictions during batched CLI inference and converting the BEC head's irreps to Cartesian CSV columns, with regression coverage.
