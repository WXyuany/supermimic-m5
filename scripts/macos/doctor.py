"""Fail early on unsupported devices; print the actual runtime."""
import json
import platform
import shutil
import subprocess
import sys
from importlib.metadata import version

import torch

assert sys.platform == 'darwin' and platform.machine() == 'arm64', 'Native Apple Silicon macOS required.'
assert torch.backends.mps.is_available(), 'MPS unavailable: use native arm64 Python and a supported macOS.'
assert sys.version_info[:2] == (3, 12), 'Python 3.12 required.'
memory = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize']))
chip = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
free = shutil.disk_usage('.').free
print(json.dumps({'chip': chip, 'memory_gib': round(memory / 2**30, 1),
                  'disk_free_gib': round(free / 2**30, 1), 'macOS': platform.mac_ver()[0],
                  'python': platform.python_version(), 'torch': torch.__version__,
                  'mujoco': version('mujoco'), 'warp': version('warp-lang'),
                  'learner': 'mps', 'physics': 'cpu'}, indent=2))
if memory < 24 * 2**30:
    print('NOTE: published 1024-environment training was tested with 24 GB RAM; smaller machines are unvalidated.')
if free < 15 * 2**30:
    print('NOTE: allow at least 15 GB free for dependencies, data, checkpoints and video.')
assert torch.isfinite((torch.ones(8, device='mps') * 2).sum()).item()
