#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
GPU_ENV="$PAPER_ROOT/.export_assets/gpu_venv"
UV_CACHE_ROOT="$PAPER_ROOT/.export_assets/gpu_uv_cache"
PIP_TMP_ROOT="$PAPER_ROOT/.export_assets/gpu_pip_tmp"
GPU_PYTHON=${SMART_GPU_PYTHON:-"$PAPER_ROOT/.venv/bin/python"}

mkdir -p "$PAPER_ROOT/.export_assets" "$UV_CACHE_ROOT" "$PIP_TMP_ROOT"
if [[ ! -x "$GPU_ENV/bin/python" ]]; then
  env UV_CACHE_DIR="$UV_CACHE_ROOT" \
    uv venv --python "$GPU_PYTHON" "$GPU_ENV"
fi
"$GPU_ENV/bin/python" -m ensurepip --upgrade
env PIP_CACHE_DIR="$UV_CACHE_ROOT" TMPDIR="$PIP_TMP_ROOT" \
  "$GPU_ENV/bin/python" -m pip install \
  -r "$PAPER_ROOT/requirements-gpu.txt"
echo "Prepared $GPU_ENV"
