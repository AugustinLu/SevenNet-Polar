"""Shared by the examples. Sets the MPS memory cap before torch is imported: 0.25 of
the recommended working set is about 10 GB on a 48 GB Mac. Keep the ratio above 0: 0
disables PyTorch's safeguard against exhausting the machine's memory."""

import argparse
import os
import resource

os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.25')
os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '0.15')

import torch  # noqa: E402
from ase.io import read  # noqa: E402

from sevenn.calculator import FieldCalculator, SevenNetCalculator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def parser(doc):
    ap = argparse.ArgumentParser(description=doc)
    ap.add_argument(
        '--ckpt',
        required=True,
        help='SevenNet or SevenNet-Polar (Direct) checkpoint',
    )
    ap.add_argument(
        '--repeat',
        type=int,
        default=2,
        help='supercell of the 12-atom cell, 2 gives 96 atoms',
    )
    ap.add_argument(
        '--mps-opt',
        default='fused',
        choices=['fused', 'off'],
        help="'off' runs the reference e3nn code",
    )
    ap.add_argument(
        '--efield',
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=('Ex', 'Ey', 'Ez'),
        help='V/Angstrom',
    )
    return ap


def structure(repeat, rattle=0.05):
    atoms = read(os.path.join(HERE, 'tZrO2_12atom.extxyz')).repeat(
        (repeat, repeat, repeat)
    )
    atoms.rattle(rattle, seed=1)
    return atoms


def calculators(args):
    base = SevenNetCalculator(args.ckpt, device='mps', mps_opt=args.mps_opt)
    calc = FieldCalculator(base, enforce_asr=True)
    calc.set_efield(args.efield)
    runtime = base.mps_runtime
    print(
        f'execution: {runtime.mode}, edge bucket {runtime.bucket}, modules '
        f'{runtime.report}'
    )
    return base, calc


def peak_memory_gb():
    """Process high-water mark and the Metal driver memory now, in GB."""
    return resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss / 1e9, torch.mps.driver_allocated_memory() / 1e9
