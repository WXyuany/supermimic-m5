#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
"$PY" scripts/macos/doctor.py
"$PY" scripts/macos/prepare_data.py
mkdir -p logs
common=(--device mps --env-device cpu --seed 0 --data.npz-dir data/npz/lafan1 --env.anchor-obs --schedule.total-iterations 320000 --schedule.checkpoint-interval 500)
name1=m5_stage1; name2=m5_stage2
stop1=12500; stop2=21000
env1=128; batch1=128; env2=1024; batch2=512
if [[ "${1:-}" == --smoke ]]; then
  name1=smoke_stage1; name2=smoke_stage2
  stop1=50; stop2=100
  env1=32; batch1=32; env2=32; batch2=32
elif [[ $# != 0 ]]; then
  echo 'Usage: bash scripts/macos/train.sh [--smoke]' >&2; exit 1
fi
stage1="logs/$name1/ckpt_latest.pt"
stage2="logs/$name2/ckpt_latest.pt"
iteration() { "$PY" -c 'import sys,torch; print(torch.load(sys.argv[1],map_location="cpu",weights_only=False)["iteration"])' "$1"; }
# Bash 3.2 on macOS rejects expanding an empty array under set -u.
resume=(--run-name "$name1")
if [[ -f "$stage1" ]]; then resume+=(--resume "$stage1"); fi
if [[ ! -f "$stage1" ]] || (( $(iteration "$stage1") < stop1 )); then
  caffeinate -i .venv/bin/sm-train "${common[@]}" --env.num-envs "$env1" --wm-trainer.batch-size "$batch1" --policy-trainer.batch-size "$batch1" --schedule.stop-at-iteration "$stop1" "${resume[@]}" 2>&1 | tee -a "logs/$name1.console.log"
fi
resume=(--resume "$stage1" --reset-sampler-memory-on-resume)
if [[ -f "$stage2" ]]; then resume=(--resume "$stage2"); fi
if [[ ! -f "$stage2" ]] || (( $(iteration "$stage2") < stop2 )); then
  caffeinate -i .venv/bin/sm-train "${common[@]}" --run-name "$name2" --env.num-envs "$env2" --wm-trainer.batch-size "$batch2" --policy-trainer.batch-size "$batch2" --env.sampler-uniform-ratio 0.8 --env.sampler-hard-ratio 0.15 --env.sampler-memory-ratio 0.05 --env.sampler-temperature 5 --schedule.stop-at-iteration "$stop2" "${resume[@]}" 2>&1 | tee -a "logs/$name2.console.log"
fi
if [[ "${1:-}" == --smoke ]]; then
  "$PY" - "$stage1" "$stage2" <<'PY'
import sys, torch
for path, step, updates in zip(sys.argv[1:], (50,100), (4,8)):
    c = torch.load(path, map_location='cpu', weights_only=False)
    assert (c['iteration'],c['num_updates']) == (step,updates)
    assert c['config']['schedule']['total_iterations'] == 320000
    assert all(torch.isfinite(v).all() for k in ('wm','policy','critic') for v in c[k].values())
print('PASS: real MPS updates, stage transition, 40-step resume warmup, finite parameters.')
PY
  echo 'Smoke complete. These random-start weights are not the published demonstration model.'
  exit 0
fi
bash scripts/macos/evaluate.sh "$stage2"
for clip in walk1_subject1 dance1_subject1 jumps1_subject1; do
  "$PY" scripts/render_video.py --ckpt "$stage2" --clip "$clip" --seconds 20 --device cpu --out "logs/m5_stage2/$clip.mp4"
done
echo 'Training, all-40 evaluation and three preview videos are complete under logs/.'
