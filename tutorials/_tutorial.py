"""Small helpers shared by the tutorial notebooks: machine report, backend
choice, synchronized timing, memory, and cached downloads from Zenodo."""

import hashlib
import os
import platform
import resource
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

ZENODO = 'https://zenodo.org/records/23734482/files'
# SevenNet-Polar models and data of Lu et al. (CC-BY-4.0),
# doi:10.5281/zenodo.23734482
ZENODO_MD5 = {
    'SevenNet-PM-S.pth': 'f15ebbdaccb1faabc6b55892b4df57b4',
    'SevenNet-PM-M.pth': 'd170b80cee43d693e2e56416b7543ab7',
    'SevenNet-PS-S.pth': '4d18c0a75ad1c9f908bb1f369e754f31',
}
CACHE = Path(
    os.environ.get(
        'SEVENN_TUTORIAL_CACHE', Path.home() / '.cache' / 'sevennet-polar'
    )
)
HERE = Path(__file__).resolve().parent

# (label, device, mps_opt): the three ways to run the same model
BACKENDS = [
    ('CPU, e3nn', 'cpu', 'off'),
    ('Apple GPU, e3nn', 'mps', 'off'),
    ('Apple GPU, fused', 'mps', 'fused'),
]


def _sysctl(key):
    try:
        return subprocess.check_output(['sysctl', '-n', key], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ''


def has_apple_gpu():
    return torch.backends.mps.is_available()


def has_fused():
    """The fused kernels need an Apple GPU and torch.mps.compile_shader."""
    return has_apple_gpu() and hasattr(torch.mps, 'compile_shader')


def ram_gb():
    v = _sysctl('hw.memsize')
    if v:
        return int(v) / 1e9
    try:
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1e9
    except (ValueError, OSError):
        return 0.0


def machine_report():
    chip = (
        _sysctl('machdep.cpu.brand_string')
        or platform.processor()
        or platform.machine()
    )
    import sevenn

    print(
        f'machine : {chip}, {ram_gb():.0f} GB memory, '
        f'{platform.system()} {platform.release()}'
    )
    print(
        f'python  : {sys.version.split()[0]}, torch {torch.__version__}, '
        f'sevenn {getattr(sevenn, "__version__", "?")}'
    )
    print(
        f'Apple GPU (MPS): {"yes" if has_apple_gpu() else "no"};'
        f' fused kernels: {"available" if has_fused() else "not available"}'
    )
    if not has_apple_gpu():
        print(
            'No Apple GPU: the notebooks run on the CPU only, with smaller systems.'
        )


def available_backends():
    out = [b for b in BACKENDS if b[1] == 'cpu' or has_apple_gpu()]
    return [b for b in out if b[2] != 'fused' or has_fused()]


def best_backend():
    """The fastest available (label, device, mps_opt)."""
    return available_backends()[-1]


def calculator(model, device, mps_opt, **kw):
    from sevenn.calculator import SevenNetCalculator

    if device == 'cpu':
        return SevenNetCalculator(model, device='cpu', **kw)
    return SevenNetCalculator(model, device=device, mps_opt=mps_opt, **kw)


def sync(device):
    if device == 'mps':
        torch.mps.synchronize()


def time_calls(atoms, calc, device, n=10, rattle=0.01, seed=0):
    """First call (includes building the calculator's kernels and caches) and
    the median of n further calls, each from new positions as in MD, in ms."""
    rng = np.random.default_rng(seed)
    a = atoms.copy()
    a.positions += rng.normal(
        0, rattle, a.positions.shape
    )  # new positions, so nothing is cached
    a.calc = calc
    sync(device)
    t = time.perf_counter()
    a.get_forces()
    sync(device)
    first = 1e3 * (time.perf_counter() - t)
    ts = []
    for _ in range(n):
        a.positions += rng.normal(0, rattle, a.positions.shape)
        sync(device)
        t = time.perf_counter()
        a.get_forces()
        a.get_stress()
        sync(device)
        ts.append(1e3 * (time.perf_counter() - t))
    return first, float(np.median(ts))


def memory_gb():
    """Peak resident memory of this process and current Metal driver memory, GB."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = rss / 1e9 if sys.platform == 'darwin' else rss / 1e6
    drv = torch.mps.driver_allocated_memory() / 1e9 if has_apple_gpu() else 0.0
    return rss, drv


def zenodo_file(name):
    """Download a file of the SevenNet-Polar Zenodo record once, check it,
    and return its path."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if not path.exists():
        print(f'downloading {name} from Zenodo (doi:10.5281/zenodo.23734482) ...')
        tmp = path.with_suffix('.part')
        urllib.request.urlretrieve(f'{ZENODO}/{name}?download=1', tmp)
        tmp.rename(path)
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    if name in ZENODO_MD5 and md5 != ZENODO_MD5[name]:
        path.unlink()
        raise RuntimeError(
            f'{name}: checksum mismatch, the download was removed; '
            'run the cell again'
        )
    return str(path)


def cells_for_memory(cells, gb_per_1000_atoms, natoms):
    """Keep the cells whose estimated memory fits in 40% of this machine's memory."""
    budget = 0.4 * ram_gb()
    return [
        c for c, n in zip(cells, natoms) if n / 1000 * gb_per_1000_atoms < budget
    ]


def thermalize(atoms, temperature_K, rng):
    """Maxwell-Boltzmann velocities at temperature_K with zero total momentum,
    for old and new ASE."""
    try:
        from ase.md.velocitydistribution import (
            thermalize_momenta,  # ASE >= 3.29
        )

        thermalize_momenta(atoms, temperature_K, rng=rng)
    except ImportError:
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng)
    from ase.md.velocitydistribution import Stationary

    Stationary(atoms)
