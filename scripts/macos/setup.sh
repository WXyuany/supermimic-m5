#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
unset PYTHONPATH
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo 'This recipe requires native Apple Silicon macOS (do not use a Rosetta terminal).' >&2; exit 1
fi
if ! xcode-select -p >/dev/null 2>&1; then
  echo 'Install Apple Command Line Tools with xcode-select --install, then rerun setup.' >&2; exit 1
fi
if command -v uv >/dev/null 2>&1; then UV="$(command -v uv)"; else
  # Install the package manager locally; no sudo or global Python changes.
  if [[ ! -x .tools/bin/uv ]]; then
    python3 -m venv .tools
    .tools/bin/python -m pip install --disable-pip-version-check 'uv==0.13.5'
  fi
  UV="$PWD/.tools/bin/uv"
fi
export UV_HTTP_TIMEOUT=120 HF_HUB_DISABLE_XET=1
"$UV" sync --frozen --python 3.12.15
.venv/bin/python scripts/macos/doctor.py
.venv/bin/python scripts/macos/prepare_data.py
echo 'Ready. Demo: bash scripts/macos/demo.sh | Train: bash scripts/macos/train.sh'
