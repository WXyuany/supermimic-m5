# Apple M5 quick start

An adaptation of [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced), with measured M5 results. PyTorch MPS trains the policy/world model; CPU MuJoCo performs physics. No CUDA required. Tested on Apple M5, 24 GB unified memory, macOS 27.0. Other chips/versions are unvalidated.

Prerequisites: native arm64 terminal, Apple Command Line Tools (`xcode-select --install` if missing), at least 15 GB disk space, network access to GitHub/PyPI/Hugging Face, a logged-in macOS desktop for GLFW rendering.

```bash
git clone https://github.com/WXyuany/supermimic-m5.git && cd supermimic-m5
bash scripts/macos/setup.sh
bash scripts/macos/demo.sh
```

Setup installs a local environment with pinned dependencies and Python, downloads the public 40-clip LAFAN1 G1 dataset at an immutable revision, verifies SHA256 and converts 30 Hz CSVs to 50 Hz NPZs. No Hugging Face login required. Demo downloads a checksum-verified checkpoint, renders 20 seconds of walking and opens the video. First compilation/downloads take time.

```bash
bash scripts/macos/demo.sh dance1_subject1 20
bash scripts/macos/evaluate.sh
bash scripts/macos/train.sh
```

Training starts from scratch: 128 environments/batch 128 until 12500, then 1024 environments/batch 512 until 21000, followed by all-40 evaluation and videos. It takes hours. The 320000-iteration LR schedule is retained. Ctrl+C stops; rerunning resumes saved checkpoints with 40-step replay warmup. Do not run concurrent training instances. Keep the laptop plugged in and awake. Outputs are under `logs/`.

Run `bash scripts/macos/tensorboard.sh` in another terminal and open http://127.0.0.1:6006.

Published checkpoint 21000: mean normalized return **0.708095**, step-weighted **0.721223**, **25/40** clips complete, **15/40** terminate early. Evaluated on training motions, one seed only. Return is normalized over executed steps, not padded with zeros after failure. Uses 9 extra anchor-relative observation dimensions; not directly comparable to the paper's proprioception-only result. Videos are 20-second previews with independently following cameras. Original training stopped at 21200; public evaluation/video weights remain 21000.

A fresh-environment rerun with the default 4 CPU threads yielded mean 0.708108, weighted 0.722371 and 26/40 complete (original evaluation used 2 threads). One borderline sprint changed termination outcome; both raw results are retained. Bitwise simulation/termination reproducibility is not claimed.

Code is MIT; original LAFAN1 data has CC BY-NC-ND 4.0 terms. No motion data redistributed. See [NOTICE](../NOTICE.md), [per-clip results](../results/benchmark_retuned_21000.json), and [configuration/protocol](REPRODUCIBILITY.md).
