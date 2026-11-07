"""Python side of pair style e3gnn on the Apple GPU (pair_e3gnn_mps/): owns the
calculator and computes for LAMMPS.

pair_coeff * * checkpoint.pth [modal NAME] [efield Ex Ey Ez] [enforce_asr yes|no]
element1 element2 ...
Keywords as in the NVIDIA pair style; modal is for multi-fidelity models.

"""

from typing import List, Sequence

import numpy as np
import torch
from ase import Atoms
from ase.data import atomic_numbers

from sevenn.calculator import FieldCalculator, SevenNetCalculator

_VIRIAL = [
    0,
    1,
    2,
    5,
    4,
    3,
]  # ASE voigt xx yy zz yz xz xy to LAMMPS xx yy zz xy xz yz


def _cell(box) -> np.ndarray:
    lo, hi, xy, xz, yz, _ = box
    return np.array(
        [
            [hi[0] - lo[0], 0.0, 0.0],
            [xy, hi[1] - lo[1], 0.0],
            [xz, yz, hi[2] - lo[2]],
        ]
    )


def _parse(args: List[str]) -> dict:
    out = dict(modal=None, enforce_asr=True, efield=None)
    i = 0
    while i < len(args) and args[i] in (
        'modal',
        'efield',
        'efield_reduced',
        'enforce_asr',
        'efield_mode',
        'efield_virial',
    ):
        key = args[i]
        if key == 'efield':
            if out['efield'] is not None:
                raise ValueError('e3gnn: give one field keyword')
            out['efield'] = [float(x) for x in args[i + 1 : i + 4]]
            if len(out['efield']) != 3:
                raise ValueError('e3gnn: efield requires 3 values')
            i += 4
        elif key == 'modal':
            if i + 1 == len(args):
                raise ValueError('e3gnn: modal requires a name')
            out['modal'] = args[i + 1]
            i += 2
        elif key == 'enforce_asr':
            i += 1
            if i < len(args) and args[i] in ('yes', 'no'):
                out['enforce_asr'] = args[i] == 'yes'
                i += 1
        elif key == 'efield_virial':
            raise ValueError(
                'e3gnn: efield_virial is not supported: direct models derive from '
                'no field energy and have no field stress'
            )
        else:  # as in the NVIDIA pair style of this version
            raise ValueError(
                f'e3gnn: {key} is not available in this version; give efield Ex '
                f'Ey Ez in V/Angstrom'
            )
    out['symbols'] = args[i:]
    return out


class PairDriver:
    def __init__(self, args: List[str], ntypes: int, box: Sequence) -> None:
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                'e3gnn: this pair style needs the Apple GPU, and MPS is not '
                'available'
            )
        path, opt = args[0], _parse(args[1:])
        if path.endswith('.pt'):
            raise ValueError(
                f'e3gnn: {path} looks like a deployed model. TorchScript cannot '
                f'hold the Metal kernels: give the .pth checkpoint or a pretrained '
                f'name'
            )
        if len(opt['symbols']) < ntypes:
            raise ValueError(
                f'e3gnn: not enough elements in pair_coeff: {ntypes} atom types, '
                f'got {opt["symbols"]}'
            )
        self.base = SevenNetCalculator(path, device='mps', modal=opt['modal'])
        known = set(self.base.type_map)
        unknown = [s for s in opt['symbols'] if atomic_numbers.get(s) not in known]
        if unknown:
            raise ValueError(f'e3gnn: elements unknown to this model: {unknown}')
        self.calc = self.base
        field = 'none'
        if opt['efield'] is not None:
            self.calc = FieldCalculator(self.base, enforce_asr=opt['enforce_asr'])
            self.calc.set_efield(opt['efield'])
            field = (
                str(opt['efield'])
                + ' (reported energy excludes the field term, so forces are not its '
                'gradient)'
            )
        self.numbers = np.array([0] + [atomic_numbers[s] for s in opt['symbols']])
        self.cutoff = float(self.base.cutoff)
        run = (
            self.base.mps_runtime.report
            if self.base.mps_runtime
            else 'reference code'
        )
        self.banner = (
            f'e3gnn on the Apple GPU: {path}, cutoff {self.cutoff} A, '
            f'field {field}, elements {" ".join(opt["symbols"])}\n{run}'
        )

    def compute(self, nlocal: int, x, f, types, eatom, virial, box) -> float:
        if (
            nlocal == 0
        ):  # all atoms lost during a run, with thermo_modify lost ignore
            return 0.0
        lo, _, _, _, _, periodic = box
        forces = np.frombuffer(f, dtype=np.float64).reshape(nlocal, 3)
        out = np.frombuffer(virial, dtype=np.float64)
        atoms = Atoms(
            numbers=self.numbers[np.frombuffer(types, dtype=np.int32)],
            positions=np.frombuffer(x, dtype=np.float64).reshape(nlocal, 3)
            - np.array(lo),
            cell=_cell(box),
            pbc=[bool(p) for p in periodic],
        )
        atoms.calc = self.calc
        energy = atoms.get_potential_energy()
        forces += atoms.get_forces()
        out[:] = -(atoms.get_stress(voigt=True) * atoms.get_volume())[_VIRIAL]
        if eatom is not None:
            np.frombuffer(eatom, dtype=np.float64)[:] += atoms.calc.results[
                'energies'
            ]
        return float(energy)
