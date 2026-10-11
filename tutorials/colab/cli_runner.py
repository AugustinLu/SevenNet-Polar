"""Profile the unmodified SevenNet CLI in its own process."""

import json
import os
import runpy
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(int(os.environ.get('SEVENNET_TUTORIAL_THREADS', '2')))
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
started = time.perf_counter()
profile_path = Path(sys.argv[1])
sys.argv = ['sevenn', *sys.argv[2:]]
runpy.run_module('sevenn.main.sevenn', run_name='__main__')
if torch.cuda.is_available():
    torch.cuda.synchronize()
profile_path.write_text(
    json.dumps(
        {
            'cli_seconds': time.perf_counter() - started,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()
            if torch.cuda.is_available()
            else None,
            'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved()
            if torch.cuda.is_available()
            else None,
            'cpu_threads': torch.get_num_threads(),
        },
        indent=2,
    )
)
