# Validation record

Development on 2026-10-10 (Asia/Tokyo), branch `tutorials/colab-bec`.

Validation is in progress. GPU results below will be updated after execution;
no accelerator compatibility or speedup is inferred from installation alone.

## Initial CPU experiment

Mac CPU, Python 3.10.20, PyTorch 2.5.1, two CPU threads. Real 20/5/5 structure split,
16 channels, lmax=2, three layers, batch size one, BEC-only loss.
Forty epochs took **293.1 s** including CLI startup/preprocessing/model construction.
Reloaded-checkpoint inference over all 30 structures took **8.0 s**.
The best validation checkpoint produced test diagonal/off-diagonal BEC RMSE
**0.0961 / 0.0663 e**, with MAE **0.0694 / 0.0515 e**.
CLI and ASE BEC predictions agree within 2e-5 absolute/relative tolerance.

| Epoch | Validation diagonal RMSE (e) | Validation off-diagonal RMSE (e) |
|---|---:|---:|
| 5 | 0.1484 | 0.1104 |
| 10 | 0.1136 | 0.0818 |
| 20 | 0.0962 | 0.0796 |
| 30 | 0.0815 | 0.0666 |
| 40 | 0.0861 | 0.0621 |

These are points on one learning curve, not separately timed runs. The quickstart
starts with 20 epochs as an initial time/accuracy tradeoff, subject to full notebook
and T4 validation. Test errors were not used to select this initial default.

The maximum test acoustic-sum-rule residual is **2.285 e** for raw model predictions,
versus about **3e-6 e** for raw reference labels. The tutorial reports this diagnostic
without projection. A visually good parity plot does not establish ASR compliance
or production-model suitability. The small split lacks trajectory IDs and is not
an independent-generalization benchmark.

## Remaining checks

- Full default notebooks through nbclient on CPU and through Colab's notebook executor.
- Fresh T4 installation and quickstart, batch/model-size timing comparisons.
- A100 multitask notebook; FlashTP build, same-checkpoint E/F/S/BEC and loss-gradient
  agreement, warmups, and equal-work training timings.
- Existing unit regressions plus the BEC CSV single-atom/unequal-batch regression.

Source notebooks are output-free. Each run exports its exact machine, dependency,
source, configuration, partition and timing records; raw execution artifacts are
kept outside the source tree during development.
