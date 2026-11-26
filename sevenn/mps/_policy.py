"""When the Apple GPU code runs: the opt-out and the PyTorch requirements. Imports
nothing from torch.mps, so it is safe to import on any machine."""

import os

# SEVENN_DISABLE_MPS=1 (or true, yes) keeps the reference e3nn code on the Apple GPU,
# for reproducibility or debugging; device='auto' then selects the CPU on a Mac.
OPT_OUT = 'SEVENN_DISABLE_MPS'


def disabled() -> bool:
    return os.environ.get(OPT_OUT, '').strip().lower() in ('1', 'true', 'yes')
