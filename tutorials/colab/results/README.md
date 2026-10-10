# Compact validation artifacts

See [measured validation](../VALIDATION.md) for interpretation and test coverage.
Records were collected on 2026-10-10. Full checkpoints, executed notebooks,
DFT label copies and raw logs remain in the development validation archive;
each tutorial exports these files in its own ZIP when run.

- `cpu_quickstart`, `t4_quickstart`: default 16-channel, 20-epoch BEC tutorial.
- `cpu_multitask`: final 16-channel, 40-epoch E/F/S/BEC tutorial.
- `a100_multitask`: final 32-channel, 40-epoch E/F/S/BEC tutorial and verified
  same-checkpoint/equal-work FlashTP checks. Includes both five-epoch curves.
- `cpu_multitask_mean_shift`: earlier energy-baseline experiment, not the default.
- `exploratory`: T4 model/batch trials, fresh accelerator build records and the
  corrected T4 32-channel numerical/gradient check. These exploratory T4 records
  mark the checkout dirty because the accelerator helper was updated in place.

JSON records retain original commits, versions, hashes, indices, configurations
and measurements. Local checkout/input paths have been shortened for publication;
CPU model identifiers have been added. Timings are single-run observations.
Training/validation/test metrics come from the best validation checkpoint;
epoch training metrics accumulate predictions while model weights change.
No synthetic DFT labels or predictions are used. Raw ASR diagnostics are retained.
