"""Physical cell preparation and trained-force guard for the optional MD demo."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase.io import read
from ase.neighborlist import neighbor_list

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tutorials/colab'))
spec = importlib.util.spec_from_file_location(
    'polar_lammps_demo', ROOT / 'tutorials/colab/lammps_demo.py'
)
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def test_supercell_preserves_periodic_local_environments():
    frames = read(ROOT / 'example_inputs/tutorial_polar/li3po4/pool.extxyz', ':')
    source = next(a for a in frames if a.get_chemical_formula() == 'Li12O16P4')
    atoms = demo.supercell(source)
    assert len(atoms) == 512
    assert atoms.get_chemical_formula() == 'Li192O256P64'
    assert atoms.get_volume() == pytest.approx(16 * source.get_volume())
    original = neighbor_list('d', source, 6.0)
    expanded = neighbor_list('d', atoms, 6.0)
    np.testing.assert_allclose(
        np.sort(expanded), np.sort(np.tile(original, 16)), atol=1e-8
    )


def test_md_rejects_bec_only_checkpoint_before_launch(monkeypatch, tmp_path):
    import sevenn.util

    monkeypatch.setattr(
        sevenn.util,
        'load_checkpoint',
        lambda _: SimpleNamespace(
            config={'energy_loss_weight': 0.0, 'force_loss_weight': 0.0}
        ),
    )
    with pytest.raises(ValueError, match='trained energy and forces'):
        demo.run_field_md(ROOT, 'unused.pth', None, tmp_path / 'md', 'unused-lmp')
    assert not (tmp_path / 'md').exists()
