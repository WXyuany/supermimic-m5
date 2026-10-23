#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
unset PYTHONPATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
export PYTHONUNBUFFERED=1 MUJOCO_GL=glfw HF_HUB_DISABLE_XET=1
PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then echo 'Run bash scripts/macos/setup.sh first.' >&2; exit 1; fi
