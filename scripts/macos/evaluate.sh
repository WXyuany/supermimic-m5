#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
if [[ $# == 0 ]]; then
  "$PY" scripts/macos/download_checkpoint.py
  ckpt=checkpoints/m5_lafan1_21000.pt
else ckpt="$1"; fi
mkdir -p logs/evaluation
"$PY" scripts/eval_benchmark.py --ckpt "$ckpt" --npz-dir data/npz/lafan1 --device cpu --num-envs 40 --out logs/evaluation/benchmark.json
