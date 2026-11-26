"""`sevenn patch_lammps` on a Mac: copy the Apple GPU pair style into a LAMMPS source
tree and link it to this Python environment (src/patch_lammps.sh)."""

import os.path as osp
import subprocess
import sys

SRC = osp.join(osp.dirname(osp.abspath(__file__)), 'src')


def patch(lammps_dir: str) -> int:
    root = osp.abspath(osp.join(SRC, '..', '..', '..', '..'))
    script = osp.join(SRC, 'patch_lammps.sh')
    return subprocess.run([script, lammps_dir, sys.executable, root]).returncode
