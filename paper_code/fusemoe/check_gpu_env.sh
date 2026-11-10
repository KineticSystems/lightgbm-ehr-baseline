#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
GPU_PYTHON="$PAPER_ROOT/.export_assets/reproduction/gpu_venv/bin/python"

if [[ ! -x "$GPU_PYTHON" ]]; then
  echo "Missing GPU environment; run setup_gpu_env.sh first" >&2
  exit 1
fi

env TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  "$GPU_PYTHON" "$PAPER_ROOT/check_gpu_env.py"
