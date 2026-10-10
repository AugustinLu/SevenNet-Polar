"""Execute a tutorial with the current Python in an isolated Jupyter kernel.

Usage: python execute_notebook.py NOTEBOOK --output OUTPUT [--epochs 1] [--skip-install]
Requires nbformat, nbclient, ipykernel. Outputs go outside the source notebook.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--skip-install', action='store_true')
    parser.add_argument('--flash', choices=['auto', 'off'], default='off')
    args = parser.parse_args()
    notebook_path = args.notebook.resolve()
    repo = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['SEVENNET_TUTORIAL_REPO'] = str(repo)
    env['SEVENNET_TUTORIAL_FLASH'] = args.flash
    env['OMP_NUM_THREADS'] = env['MKL_NUM_THREADS'] = env[
        'SEVENNET_TUTORIAL_THREADS'
    ] = '2'
    env['MPLBACKEND'] = 'Agg'
    if args.epochs:
        env['SEVENNET_TUTORIAL_EPOCHS'] = str(args.epochs)
    if args.skip_install:
        env['SEVENNET_TUTORIAL_SKIP_INSTALL'] = '1'
    notebook = nbformat.read(notebook_path, as_version=4)
    with tempfile.TemporaryDirectory() as directory:
        kernel = Path(directory) / 'polar-tutorial'
        kernel.mkdir()
        (kernel / 'kernel.json').write_text(
            json.dumps(
                {
                    'argv': [
                        sys.executable,
                        '-m',
                        'ipykernel_launcher',
                        '-f',
                        '{connection_file}',
                    ],
                    'display_name': 'Polar tutorial test',
                    'language': 'python',
                }
            )
        )
        manager = KernelManager(
            kernel_name='polar-tutorial',
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[directory]),
        )
        client = NotebookClient(
            notebook,
            km=manager,
            timeout=3600,
            resources={'metadata': {'path': str(output.parent)}},
        )
        try:
            client.execute(env=env)
        finally:
            nbformat.write(notebook, output)
    print(f'Executed: {output}')


if __name__ == '__main__':
    main()
