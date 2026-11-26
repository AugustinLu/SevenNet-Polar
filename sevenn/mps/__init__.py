"""Apple GPU (MPS) backend of SevenNet: fused Metal kernels for the e3nn layers, edge
padding, and the LAMMPS pair style. Nothing here is imported unless a model moves to
the Apple GPU: AtomGraphSequential._apply calls runtime.apply() for that move, and
moving the model away undoes it. SEVENN_DISABLE_MPS=1 keeps the reference e3nn code.
"""

from ._policy import disabled
from .runtime import prepare, report, restore

__all__ = ['disabled', 'prepare', 'report', 'restore']
