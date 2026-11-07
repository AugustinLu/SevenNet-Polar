"""How large a system fits in your memory budget, and which MPS cap to set.

Measures the peak (process RSS + Metal driver memory) on two cells, each in its own
process, and extrapolates linearly in the number of edges (edges per atom are taken
from the larger cell). Prints the prediction and the environment settings for the
target size. The MPS allocator grows in steps of about 1 GB, so the extrapolation is
rough: compare with a short run before a long job.

"""

import argparse
import json
import os
import re
import subprocess
import sys

MEASURE = """
import json, os, sys, threading, time
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.25')
import psutil, torch
sys.path.insert(0, {here!r})
from _common import structure
from sevenn.calculator import SevenNetCalculator
peak = [0.0, 0.0]
stop = []
def watch():
    p = psutil.Process()
    while not stop:
        r, d = p.memory_info().rss / 1e9, torch.mps.driver_allocated_memory() / 1e9
        peak[0] = max(peak[0], r + d); peak[1] = max(peak[1], d)
        time.sleep(0.01)
threading.Thread(target=watch, daemon=True).start()
calc = SevenNetCalculator({ckpt!r}, device='mps', mps_opt={mode!r})
at = structure({k})
at.calc = calc
for i in range(5):
    at.positions += 1e-3
    at.get_forces()
torch.mps.synchronize()
time.sleep(0.05)
stop.append(1)
print('PEAK ' + json.dumps(dict(atoms=len(at), edges=calc.results['num_edges'], peak=peak[0], driver=peak[1])))
"""  # noqa: E501

ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
ap.add_argument('--ckpt', required=True)
ap.add_argument('--atoms', type=int, default=1000, help='target number of atoms')
ap.add_argument(
    '--limit-gb',
    type=float,
    default=14.0,
    help='budget for RSS + Metal driver memory',
)
ap.add_argument('--mps-opt', default='fused', choices=['fused', 'off'])
ap.add_argument(
    '--cells', type=int, nargs=2, default=[3, 4], help='supercells to measure'
)
args = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))
rows = []
for k in args.cells:
    code = MEASURE.format(here=here, ckpt=args.ckpt, mode=args.mps_opt, k=k)
    out = subprocess.run(
        [sys.executable, '-c', code], capture_output=True, text=True, timeout=1200
    )
    m = re.search(r'^PEAK (.*)$', out.stdout, re.M)
    if not m:
        sys.exit(
            f'measurement failed for {12 * k**3} '
            f'atoms:\n{out.stdout[-500:]}{out.stderr[-1500:]}'
        )
    rows.append(json.loads(m.group(1)))
    print(
        f'{rows[-1]["atoms"]:6d} atoms, {rows[-1]["edges"]:7d} edges: peak '
        f'{rows[-1]["peak"]:.2f} GB, driver {rows[-1]["driver"]:.2f} GB'
    )
a, b = rows
slope = max(
    (b['peak'] - a['peak']) / max(b['edges'] - a['edges'], 1), 0.0
)  # GB per edge; the allocator works in steps, so this is rough
per_atom = b['edges'] / b['atoms']
edges = args.atoms * per_atom
predicted = max(b['peak'], b['peak'] + slope * (edges - b['edges']))
fits = int(
    b['atoms'] + (args.limit_gb - 1.0 - b['peak']) / max(slope * per_atom, 1e-9)
)
print(
    f'target {args.atoms} atoms, about {edges:.0f} edges at this density: predicted '
    f'peak {predicted:.1f} GB (budget {args.limit_gb:.1f} GB)'
)
print(f'largest size within the budget, 1 GB margin: about {fits} atoms (rough)')
import torch  # noqa: E402

rec = (
    torch.mps.recommended_max_memory() / 1e9
)  # the watermark ratios are relative to this
ratio = min(max(args.limit_gb - 2.5, 1.0) / rec, 1.0)
print(
    f'set before starting python:\n  export '
    f'PYTORCH_MPS_HIGH_WATERMARK_RATIO={ratio:.2f}\n  export '
    f'PYTORCH_MPS_LOW_WATERMARK_RATIO={0.6 * ratio:.2f}'
)
print(
    'Above the cap PyTorch raises an out-of-memory error instead of using more. Do '
    'not set the ratio to 0.'
)
