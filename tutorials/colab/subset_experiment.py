"""Compare bounded tutorial subsets without evaluating the test partition.

Run from an installed repository checkout, locally or in a Colab terminal:
python tutorials/colab/subset_experiment.py --dataset li3po4 --sizes 20 30 50 100 200
    --epochs 100 --output /tmp/polar-subset-study
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from ase.io import write

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import support  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--dataset',
        choices=['zro2', 'zro2_stoichiometric', 'li3po4'],
        default='li3po4',
    )
    parser.add_argument(
        '--sizes', nargs='+', type=int, default=[20, 30, 50, 100, 200]
    )
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--multitask', action='store_true')
    parser.add_argument(
        '--flash',
        action='store_true',
        help='Use an installed FlashTP backend (CUDA, channels divisible by 32).',
    )
    parser.add_argument('--bec-weight', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if any(not 1 <= size <= 200 for size in args.sizes):
        parser.error(
            'Only bounded training subsets of 1–200 structures are allowed.'
        )
    if len(args.sizes) != len(set(args.sizes)):
        parser.error('Training subset sizes must be unique.')
    torch.set_num_threads(2)
    environment = support.environment(REPO)
    device = environment['device']
    if args.flash:
        from sevenn.nn.flash_helper import is_flash_available

        if (
            not args.multitask
            or device != 'cuda'
            or args.channels % 32
            or not is_flash_available()
        ):
            parser.error(
                'FlashTP needs --multitask, CUDA, its installed extension, '
                'and channels divisible by 32.'
            )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for size in args.sizes:
        run = output / f'n{size}'
        run.mkdir()
        structures, paths, _, manifest, _ = support.prepare_subset_data(
            REPO, run / 'data', size, dataset=args.dataset
        )
        config = support.make_config(
            paths,
            args.epochs,
            multitask=args.multitask,
            batch_size=args.batch_size,
            channel=args.channels,
            layers=args.layers,
            device=device,
            seed=args.seed,
            monitor_asr=True,
        )
        config['train']['bec_loss_weight'] = args.bec_weight
        checkpoint, profile = support.train(
            REPO, run / 'training', config, flash=args.flash
        )
        flash_report = {'requested': args.flash}
        if args.flash:
            import importlib.metadata

            import flash

            flash_report.update(
                package_version=importlib.metadata.version('flashTP_e3nn'),
                comparison=flash.compare_backends(
                    checkpoint, paths['Train'], batch_size=args.batch_size
                ),
            )
        # Predict training and validation only. The fixed test labels never
        # enter model/configuration comparisons or checkpoint selection.
        comparison = run / 'train_validation.extxyz'
        ids = manifest['indices']['Train'] + manifest['indices']['Validation']
        write(comparison, [support.training_copy(structures[i]) for i in ids])
        atoms, graphs, inference = support.evaluate(
            REPO,
            run / 'training',
            checkpoint,
            comparison,
            device,
            batch_size=args.batch_size,
        )
        comparison_manifest = dict(manifest)
        comparison_manifest['indices'] = {
            key: manifest['indices'][key] for key in ['Train', 'Validation']
        }
        sums = support.asr_diagnostics(
            atoms, comparison_manifest, run / 'training', structures=structures
        )
        curve, figure = support.learning_curves(
            run / 'training',
            multitask=args.multitask,
            structures=structures,
            manifest=manifest,
        )
        import matplotlib.pyplot as plt

        plt.close(figure)
        best_epoch = int(curve.loc[curve.validset_TotalLoss.idxmin(), 'epoch'])
        row = {
            'dataset': args.dataset,
            'training_structures': size,
            'best_epoch': best_epoch,
            'training_seconds': profile['cli_seconds'],
            'peak_cuda_allocated_bytes': profile['peak_cuda_allocated_bytes'],
        }
        valid = sums[sums.partition == 'Validation']
        for quantity in ['BEC diagonal', 'BEC off-diagonal']:
            reference, prediction, _ = support.paired_values(
                atoms, graphs, quantity, manifest['indices']['Validation']
            )
            row[quantity + ' RMSE e'] = float(
                ((prediction - reference) ** 2).mean() ** 0.5
            )
        for column in [
            'reference_max_abs_sum_e',
            'prediction_max_abs_sum_e',
            'sum_error_max_abs_e',
        ]:
            row[column + ' max'] = float(valid[column].max())
            row[column + ' median'] = float(valid[column].median())
        report = {
            'environment': environment,
            'configuration': config,
            'split': manifest,
            'training': profile,
            'inference': inference,
            'validation_summary': row,
            'test_evaluated': False,
            'flash': flash_report,
        }
        support.export_artifacts(run / 'training', run / 'data', report)
        records.append(row)
        pd.DataFrame(records).to_csv(output / 'summary.csv', index=False)
        print(json.dumps(row, indent=2), flush=True)


if __name__ == '__main__':
    main()
