#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
GPU_ENV="$PAPER_ROOT/.export_assets/gpu_venv"
UV_CACHE_ROOT="$PAPER_ROOT/.export_assets/uv_cache"
PIP_TMP_ROOT="$PAPER_ROOT/.export_assets/pip_tmp"
GPU_PYTHON=${MUSE_GPU_PYTHON:-python3.10}

mkdir -p "$PAPER_ROOT/.export_assets" "$UV_CACHE_ROOT" "$PIP_TMP_ROOT"

if [[ ! -x "$GPU_ENV/bin/python" ]]; then
  env UV_CACHE_DIR="$UV_CACHE_ROOT" \
    uv venv --python "$GPU_PYTHON" "$GPU_ENV"
fi

"$GPU_ENV/bin/python" -m ensurepip --upgrade

env PIP_CACHE_DIR="$UV_CACHE_ROOT" TMPDIR="$PIP_TMP_ROOT" \
  "$GPU_ENV/bin/python" -m pip install \
  -r "$PAPER_ROOT/requirements-gpu.txt"

# CUDA 12.8 is the first official PyTorch wheel family with Blackwell support.
# Torch 2.7.0 is the oldest release in that family, minimizing the version jump.
# Install it second so the preserved Jinja/SymPy/typing pins satisfy its generic
# dependencies without asking the PyTorch wheel index to resolve them.
env PIP_CACHE_DIR="$UV_CACHE_ROOT" TMPDIR="$PIP_TMP_ROOT" \
  "$GPU_ENV/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.0

echo "Prepared $GPU_ENV"
