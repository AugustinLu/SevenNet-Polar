"""Create a bounded, composition-filtered teaching pool from the official XYZ.

python tutorials/colab/prepare_stoichiometric_zro2.py --source zro2_10103.xyz
Selection uses composition and a fixed seed, never label values or model errors.
"""

import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
from ase.io import read, write

import support

REPO = Path(__file__).resolve().parents[2]
SEED = 20261010
COMPOSITIONS = {'O32Zr16': {'O': 32, 'Zr': 16}, 'O64Zr32': {'O': 64, 'Zr': 32}}


def prepare(source, destination):
    previous = json.loads(
        (REPO / 'example_inputs/tutorial_polar/zro2/manifest.json').read_text()
    )
    if (
        hashlib.sha256(source.read_bytes()).hexdigest()
        != previous['source']['sha256']
    ):
        raise ValueError('Input must match the original official ZrO₂ checksum.')
    rows = []
    with source.open('rb') as stream:
        while True:
            offset = stream.tell()
            first = stream.readline()
            if not first:
                break
            count = int(first)
            header = stream.readline()
            if (
                not header.startswith(b'Lattice=')
                or b'Properties=species:S:1:' not in header
            ):
                raise ValueError('Expected the official extended-XYZ layout.')
            composition = Counter(
                stream.readline().split()[0].decode() for _ in range(count)
            )
            formula = next(
                (
                    name
                    for name, counts in COMPOSITIONS.items()
                    if composition == counts
                ),
                None,
            )
            rows.append(
                {
                    'id': len(rows),
                    'offset': offset,
                    'bytes': stream.tell() - offset,
                    'formula': formula,
                }
            )
    if len(rows) != previous['source']['structures']:
        raise ValueError('Incomplete source file.')
    rng = np.random.default_rng(SEED)
    old_pool = set(previous['pool_order'])
    selected = {'Train': [], 'Validation': [], 'Test': []}
    common = {'Validation': [], 'Test': []}
    eligible_counts = {}
    for formula in COMPOSITIONS:
        eligible = [r['id'] for r in rows if r['formula'] == formula]
        eligible_counts[formula] = len(eligible)
        # Retain the previous stoichiometric holdouts for a shared-label audit.
        # Additional holdouts never come from any previous pool training frame.
        fresh = [i for i in eligible if i not in old_pool]
        rng.shuffle(fresh)
        for partition in ['Validation', 'Test']:
            retained = [
                i
                for i in previous['source_ids'][partition]
                if rows[i]['formula'] == formula
            ]
            common[partition].extend(retained)
            needed = 10 - len(retained)
            selected[partition].extend(retained + fresh[:needed])
            fresh = fresh[needed:]
        held = set(selected['Validation'] + selected['Test'])
        training = [i for i in eligible if i not in held]
        rng.shuffle(training)
        selected['Train'].append(training[:100])
    selected['Train'] = [i for pair in zip(*selected['Train']) for i in pair]
    # Shuffle fixed holdouts without changing their membership.
    rng.shuffle(selected['Validation'])
    rng.shuffle(selected['Test'])
    order = sum(selected.values(), [])
    assert len(order) == len(set(order)) == 240
    pool = []
    with source.open('rb') as stream:
        for i in order:
            row = rows[i]
            stream.seek(row['offset'])
            pool.append(
                read(
                    io.StringIO(stream.read(row['bytes']).decode()), format='extxyz'
                )
            )
    fingerprints = [support.geometry_fingerprint(a) for a in pool]
    if len(set(fingerprints)) != len(pool):
        raise ValueError('Duplicate geometry in selected pool.')
    destination.mkdir(parents=True, exist_ok=False)
    path = destination / 'pool.extxyz'
    write(path, [support.training_copy(a) for a in pool], format='extxyz')
    restored = read(path, ':', format='extxyz')
    for original, copy in zip(pool, restored):
        for actual, expected in [
            (support.reference_bec(copy), support.reference_bec(original)),
            (copy.get_forces(), original.get_forces()),
            (copy.get_stress(), original.get_stress()),
            (copy.get_potential_energy(), original.get_potential_energy()),
        ]:
            np.testing.assert_allclose(actual, expected, atol=1e-8)
    manifest = {
        'dataset': 'zro2_stoichiometric',
        'source': previous['source'],
        'selection_seed': SEED,
        'composition_filter': list(COMPOSITIONS),
        'eligible_compositions': eligible_counts,
        'source_ids': selected,
        'common_previous_holdouts': common,
        'pool_order': order,
        'pool_indices': {
            'Train': list(range(200)),
            'Validation': list(range(200, 220)),
            'Test': list(range(220, 240)),
        },
        'geometry_fingerprints': fingerprints,
        'pool_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'reference_max_abs_sum_e': float(
            max(abs(support.reference_bec(a).sum(0)).max() for a in pool)
        ),
        'split_policy': (
            'Stoichiometric 48/96-atom cells only; balanced nested training '
            'reservoir; fixed 20/20 holdouts; prior stoichiometric holdouts '
            'retained; additional holdouts exclude the previous pool; '
            'no label-based selection; trajectory independence unknown.'
        ),
    }
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(
        json.dumps(
            {
                'eligible': eligible_counts,
                'selected': len(pool),
                'reference_max_abs_sum_e': manifest['reference_max_abs_sum_e'],
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument(
        '--output',
        type=Path,
        default=REPO / 'example_inputs/tutorial_polar/zro2_stoichiometric',
    )
    args = parser.parse_args()
    prepare(args.source.resolve(), args.output.resolve())
