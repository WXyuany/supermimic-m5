"""Fetch only this release's checksum-verified checkpoint."""
import hashlib
import json
import os
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[2]
manifest = json.loads((root / 'configs/release.json').read_text())
target = root / 'checkpoints' / manifest['filename']
target.parent.mkdir(parents=True, exist_ok=True)
def valid(path):
    return path.exists() and path.stat().st_size == manifest['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == manifest['sha256']
if not valid(target):
    temp = target.with_suffix('.download')
    print('Downloading the published M5 checkpoint (~216 MB)…', flush=True)
    with urllib.request.urlopen(manifest['url'], timeout=120) as src, temp.open('wb') as dst:
        while chunk := src.read(1024 * 1024): dst.write(chunk)
    if not valid(temp): raise RuntimeError('Checkpoint SHA256 mismatch; rerun to retry.')
    os.replace(temp, target)
print(target)
