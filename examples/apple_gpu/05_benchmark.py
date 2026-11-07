"""Benchmark: model call on a fixed graph and ASE step with a changing graph,
reference e3nn against fused kernels, one process per mode.

The model call covers forward + all backward passes with the graph on the device. The
ASE step covers the neighbour list, transfers, model and copies back, with new
positions and usually a new edge count each step. Timed regions are bracketed by
torch.mps.synchronize().

"""

import argparse
import json
import os
import re
import subprocess
import sys

CASE = """
import json, statistics, sys, time
sys.path.insert(0, {here!r})
from _common import calculators, structure, peak_memory_gb
import argparse
import numpy as np, torch
from sevenn.atom_graph_data import AtomGraphData
from sevenn.train.dataload import unlabeled_atoms_to_graph
args = argparse.Namespace(ckpt={ckpt!r}, mps_opt={mode!r}, efield={efield!r})
base, calc = calculators(args)
at = structure({k})
g = AtomGraphData.from_numpy_dict(unlabeled_atoms_to_graph(at, base.cutoff)); g.to('mps')
ts = []
for i in range(3 + {n}):
    torch.mps.synchronize(); t = time.perf_counter(); base.model(g.clone()); torch.mps.synchronize()
    if i >= 3: ts.append(1e3 * (time.perf_counter() - t))
at.calc = calc
rng = np.random.default_rng(1)
tb = []
for i in range(6 + {n}):
    at.positions += rng.normal(0, 0.02, at.positions.shape)
    torch.mps.synchronize(); t = time.perf_counter(); at.get_forces(); at.get_stress(); torch.mps.synchronize()
    if i >= 6: tb.append(1e3 * (time.perf_counter() - t))
print('OUT ' + json.dumps(dict(atoms=len(at), edges=calc.results['num_edges'], model=statistics.median(ts), model_min=min(ts), model_max=max(ts),
      ase=statistics.median(tb), ase_min=min(tb), ase_max=max(tb), rss_peak=peak_memory_gb()[0], driver=peak_memory_gb()[1])))
"""  # noqa: E501

ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
ap.add_argument('--ckpt', required=True)
ap.add_argument('--repeat', type=int, default=2)
ap.add_argument('--efield', type=float, nargs=3, default=[0.0, 0.0, 0.05])
ap.add_argument('--n', type=int, default=20)
ap.add_argument('--modes', default='off,fused')
args = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))
res = {}
for mode in args.modes.split(','):
    code = CASE.format(
        here=here,
        ckpt=args.ckpt,
        mode=mode,
        efield=args.efield,
        k=args.repeat,
        n=args.n,
    )
    out = subprocess.run(
        [sys.executable, '-c', code], capture_output=True, text=True, timeout=1800
    )
    m = re.search(r'^OUT (.*)$', out.stdout, re.M)
    if not m:
        sys.exit(f'{mode} failed:\n{out.stdout[-500:]}{out.stderr[-1500:]}')
    r = res[mode] = json.loads(m.group(1))
    print(
        f'{mode:6s} {r["atoms"]} atoms, {r["edges"]} edges  model call '
        f'{r["model"]:8.1f} ms [{r["model_min"]:.1f}..{r["model_max"]:.1f}]  '
        f'ASE step {r["ase"]:8.1f} ms [{r["ase_min"]:.1f}..{r["ase_max"]:.1f}]  '
        f'peak RSS {r["rss_peak"]:.2f} GB, driver {r["driver"]:.2f} GB'
    )
if 'off' in res and 'fused' in res:
    print(
        f'speed-up: model call {res["off"]["model"] / res["fused"]["model"]:.1f}x, '
        f'ASE step {res["off"]["ase"] / res["fused"]["ase"]:.1f}x'
    )
