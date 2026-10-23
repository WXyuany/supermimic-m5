#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
"$PY" scripts/macos/doctor.py
"$PY" scripts/macos/prepare_data.py
mkdir -p logs
common=(--device mps --env-device cpu --seed 0 --data.npz-dir data/npz/lafan1 --env.anchor-obs --schedule.total-iterations 320000 --schedule.checkpoint-interval 500)
stage1=logs/m5_stage1/ckpt_latest.pt
stage2=logs/m5_stage2/ckpt_latest.pt
iteration() { "$PY" -c 'import sys,torch; print(torch.load(sys.argv[1],map_location="cpu",weights_only=False)["iteration"])' "$1"; }
resume=()
if [[ -f "$stage1" ]]; then resume=(--resume "$stage1"); fi
if [[ ! -f "$stage1" ]] || (( $(iteration "$stage1") < 12500 )); then
  caffeinate -i .venv/bin/sm-train "${common[@]}" --run-name m5_stage1 --env.num-envs 128 --wm-trainer.batch-size 128 --policy-trainer.batch-size 128 --schedule.stop-at-iteration 12500 "${resume[@]}" 2>&1 | tee -a logs/m5_stage1.console.log
fi
resume=(--resume "$stage1" --reset-sampler-memory-on-resume)
if [[ -f "$stage2" ]]; then resume=(--resume "$stage2"); fi
if [[ ! -f "$stage2" ]] || (( $(iteration "$stage2") < 21000 )); then
  caffeinate -i .venv/bin/sm-train "${common[@]}" --run-name m5_stage2 --env.num-envs 1024 --wm-trainer.batch-size 512 --policy-trainer.batch-size 512 --env.sampler-uniform-ratio 0.8 --env.sampler-hard-ratio 0.15 --env.sampler-memory-ratio 0.05 --env.sampler-temperature 5 --schedule.stop-at-iteration 21000 "${resume[@]}" 2>&1 | tee -a logs/m5_stage2.console.log
fi
bash scripts/macos/evaluate.sh "$stage2"
for clip in walk1_subject1 dance1_subject1 jumps1_subject1; do
  "$PY" scripts/render_video.py --ckpt "$stage2" --clip "$clip" --seconds 20 --device cpu --out "logs/m5_stage2/$clip.mp4"
done
echo 'Training, all-40 evaluation and three preview videos are complete under logs/.'
