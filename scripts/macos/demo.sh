#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
"$PY" scripts/macos/download_checkpoint.py
clip="${1:-walk1_subject1}"
seconds="${2:-20}"
mkdir -p logs/demo
"$PY" scripts/render_video.py --ckpt checkpoints/m5_lafan1_21000.pt --clip "$clip" --seconds "$seconds" --device cpu --out "logs/demo/$clip.mp4"
if [[ "${SM_NO_OPEN:-0}" != 1 ]]; then open "logs/demo/$clip.mp4"; fi
