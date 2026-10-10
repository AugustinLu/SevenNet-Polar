# Compact validation artifacts

See [measured validation](../VALIDATION.md) for interpretation and test coverage.

`subset_study` contains the October 11 bounded 20/30/50/100/200 comparison for
ZrO₂ and Li₃PO₄, per-frame tensor sums, exact configurations and Mosso checks.
See [the study](../SUBSET_STUDY.md). Its `t4_quickstart`, `t4_multitask` and
`a100_multitask` directories contain the revised 50/20/20 full Colab runs.
The optional `*_md512` directories contain short LAMMPS field demonstrations. The directories below retain the earlier
30-frame ZrO₂ defaults for historical comparison.
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
CPU model identifiers have been added. Published LAMMPS logs have trailing
whitespace removed; original logs remain in the development archives. Timings are single-run observations.
Training/validation/test metrics come from the best validation checkpoint;
epoch training metrics accumulate predictions while model weights change.
No synthetic DFT labels or predictions are used. Raw ASR diagnostics are retained.
