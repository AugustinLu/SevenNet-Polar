"""Optional FlashTP build and equal-work checks on the current CUDA runtime."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch


def install_flash(destination, revision):
    if not torch.cuda.is_available():
        raise RuntimeError('FlashTP requires a CUDA GPU. Standard training remains available.')
    if shutil.which('nvcc') is None:
        raise RuntimeError('CUDA compiler nvcc is missing. Use standard training in this runtime.')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    log_path = destination / 'build.log'
    source = destination / 'source'
    env = os.environ.copy()
    major, minor = torch.cuda.get_device_capability()
    env['CUDA_ARCH_LIST'] = f'{major}{minor}'
    env['MAX_JOBS'] = '2'
    commands = [
        ['git', 'clone', 'https://github.com/SNU-ARC/flashTP.git', str(source)],
        ['git', '-C', str(source), 'checkout', '--detach', revision],
        [sys.executable, '-m', 'pip', 'install', '--quiet', '--no-deps', '--no-build-isolation', str(source)],
    ]
    timings = []
    with log_path.open('w') as log:
        for command in commands:
            started = time.perf_counter()
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            try:
                while process.poll() is None:
                    time.sleep(1)
                    if int(time.perf_counter()-started) % 30 == 0:
                        print(f'FlashTP setup: {time.perf_counter()-started:.0f} s (full output in {log_path})', flush=True)
            except BaseException:
                process.terminate()
                process.wait()
                raise
            timings.append(time.perf_counter()-started)
            if process.returncode:
                tail = '\n'.join(log_path.read_text(errors='replace').splitlines()[-20:])
                raise RuntimeError(f'FlashTP build failed. Keep the standard backend. Log: {log_path}\n{tail}')
    return {'source_revision': revision, 'cuda_arch': env['CUDA_ARCH_LIST'], 'max_jobs': 2,
            'fetch_seconds': sum(timings[:2]), 'build_install_seconds': timings[2]}


def compare_backends(checkpoint, train_path, batch_size=1, repeats=10):
    """Compare raw outputs and every parameter gradient, including force/stress loss."""
    import sevenn._keys as KEY
    from sevenn.train.graph_dataset import SevenNetGraphDataset
    from sevenn.train.loss import get_loss_functions_from_config
    from sevenn.util import load_checkpoint
    from torch_geometric.loader import DataLoader

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for FlashTP comparison.')
    cp = load_checkpoint(str(checkpoint))
    graphs = SevenNetGraphDataset.file_to_graph_list(str(train_path), cutoff=cp.config[KEY.CUTOFF], num_cores=1)
    batch = next(iter(DataLoader(graphs, batch_size=batch_size, shuffle=False))).to('cuda')
    config = cp.config
    loss_functions = get_loss_functions_from_config(config)
    models = {name: cp.build_model(enable_flash=flash).to('cuda') for name, flash in [('e3nn', False), ('FlashTP', True)]}
    keys = [KEY.PRED_TOTAL_ENERGY, KEY.PRED_FORCE, KEY.PRED_STRESS, KEY.PRED_BORN_EFFECTIVE_CHARGES]
    outputs, gradients, losses = {}, {}, {}

    def step(model, backward):
        model.zero_grad(set_to_none=True)
        output = model(batch.clone())
        if backward:
            terms = [loss.get_loss(output, model)*weight for loss, weight in loss_functions if weight != 0]
            total = sum(terms)
            total.backward()
            return output, float(total.detach())
        return output, None

    for name, model in models.items():
        model.set_is_batch_data(True)
        model.train()
        output, losses[name] = step(model, True)
        outputs[name] = {k: output[k].detach().cpu().numpy() for k in keys}
        gradients[name] = {k: p.grad.detach().cpu().numpy() for k, p in model.named_parameters() if p.grad is not None}
        if not gradients[name] or not all(np.isfinite(x).all() for x in gradients[name].values()):
            raise AssertionError(f'{name} has missing/nonfinite gradients.')
    differences = {}
    for key in keys:
        reference, accelerated = outputs['e3nn'][key], outputs['FlashTP'][key]
        np.testing.assert_allclose(accelerated, reference, atol=1e-4, rtol=1e-4, err_msg=key)
        differences[key] = float(np.abs(accelerated-reference).max())
    if gradients['e3nn'].keys() != gradients['FlashTP'].keys():
        raise AssertionError('Gradient parameter names differ between backends.')
    gradient_max = 0.
    for key, reference in gradients['e3nn'].items():
        accelerated = gradients['FlashTP'][key]
        np.testing.assert_allclose(accelerated, reference, atol=5e-4, rtol=5e-4, err_msg=f'gradient {key}')
        gradient_max = max(gradient_max, float(np.abs(accelerated-reference).max()))

    measurements = []
    for backward in [False, True]:
        for name, model in models.items():
            model.train(backward)
            for _ in range(3):
                step(model, backward)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            samples = []
            for _ in range(repeats):
                started = time.perf_counter()
                step(model, backward)
                torch.cuda.synchronize()
                samples.append(time.perf_counter()-started)
            measurements.append({'backend': name, 'operation': 'forward+backward' if backward else 'forward (E/F/S/B)',
                                 'median_seconds': float(np.median(samples)), 'samples_seconds': samples,
                                 'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
    # Both models resident during microbenchmarks; compare timing, not per-model memory.
    return {'status': 'passed', 'output_max_abs_differences': differences,
            'gradient_max_abs_difference': gradient_max, 'gradient_parameters_checked': len(gradients['e3nn']),
            'losses': losses, 'atol_outputs': 1e-4, 'atol_gradients': 5e-4,
            'warmups': 3, 'repeats': repeats, 'batch_size': batch_size, 'measurements': measurements,
            'memory_note': 'Both models resident; memory is process-wide, not isolated per backend.'}
