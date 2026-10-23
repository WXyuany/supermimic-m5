#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
exec .venv/bin/tensorboard --logdir logs --host 127.0.0.1 --port "${1:-6006}"
