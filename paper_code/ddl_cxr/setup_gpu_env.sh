#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
REPRO_ROOT="$PAPER_ROOT/.export_assets/reproduction"
GPU_ENV="$REPRO_ROOT/gpu_venv"
GPU_PYTHON_ROOT="$REPRO_ROOT/gpu_python"
UV_CACHE_ROOT="$REPRO_ROOT/uv_cache"
PIP_TMP_ROOT="$REPRO_ROOT/pip_tmp"

mkdir -p "$REPRO_ROOT" "$GPU_PYTHON_ROOT" "$UV_CACHE_ROOT" "$PIP_TMP_ROOT"

if [[ ! -x "$GPU_ENV/bin/python" ]]; then
  env UV_CACHE_DIR="$UV_CACHE_ROOT" UV_PYTHON_INSTALL_DIR="$GPU_PYTHON_ROOT" \
    uv venv --python 3.10.20 --python-preference only-managed "$GPU_ENV"
fi

"$GPU_ENV/bin/python" -c \
  'import platform; assert platform.python_version() == "3.10.20", platform.python_version()'

"$GPU_ENV/bin/python" -m ensurepip --upgrade

# CUDA 12.8 is the first official PyTorch wheel family with Blackwell support;
# Torch 2.7.0 is the oldest release in that family.
env PIP_CACHE_DIR="$UV_CACHE_ROOT" TMPDIR="$PIP_TMP_ROOT" \
  "$GPU_ENV/bin/python" -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128

env PIP_CACHE_DIR="$UV_CACHE_ROOT" TMPDIR="$PIP_TMP_ROOT" \
  "$GPU_ENV/bin/python" -m pip install -r "$PAPER_ROOT/requirements-gpu.txt"

echo "Prepared $GPU_ENV"
