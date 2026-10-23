"""Download pinned public LAFAN1 files, verify all SHA256s, and convert at 30->50 Hz."""
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
from huggingface_hub import hf_hub_download
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    os.chdir(ROOT)
    manifest = json.loads((ROOT / 'configs/lafan1_manifest.json').read_text())
    assert len(manifest['files']) == 40
    dest = ROOT / 'data/raw/lafan1'
    def fetch(entry):
        path = dest / entry['name']
        def valid():
            return (path.exists() and path.stat().st_size == entry['bytes'] and
                    hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'])
        if not valid():
            hf_hub_download(manifest['repo'], entry['name'], repo_type='dataset',
                            revision=manifest['revision'], local_dir=dest, force_download=path.exists())
        if not valid():
            raise RuntimeError(f"Checksum mismatch: {entry['name']}; rerun setup to retry.")
        print(f"Verified {entry['name']}", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, manifest['files']))
    # Failed/incomplete output is removed so a retry can recover.
    out = ROOT / 'data/npz/lafan1'
    out.mkdir(parents=True, exist_ok=True)
    names = {Path(f['name']).stem for f in manifest['files']}
    unexpected = {p.stem for p in out.glob('*.npz')} - names
    if unexpected:
        raise RuntimeError(f'Unexpected clips in LAFAN1 output: {unexpected}. Use a clean data directory.')
    def validate(path):
        with np.load(path, allow_pickle=False) as data:
            assert float(data['fps'][0]) == 50
            assert data['joint_pos'].shape[1] == 29
            assert len(data['joint_pos']) > 1
            for key in ('joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w'):
                assert np.isfinite(data[key]).all(), key
    for path in out.glob('*.npz'):
        try: validate(path)
        except (OSError, ValueError, KeyError, AssertionError): path.unlink()
    subprocess.run([sys.executable, 'scripts/data/csv_to_npz_batch.py',
                    '--input-dir', str(dest / 'g1'), '--output-dir', str(out),
                    '--input-fps', '30', '--output-fps', '50', '--workers', '4'], check=True)
    for name in sorted(names): validate(out / f'{name}.npz')
    print('All 40 clips verified and converted. No account or token required.')


if __name__ == '__main__': main()
