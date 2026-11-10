#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
GPU_ENV="$PAPER_ROOT/.export_assets/gpu_venv"
GPU_CACHE_ROOT=${GRAPHCARE_GPU_CACHE_ROOT:-/var/tmp/missing-xgboost-pip/graphcare-gpu}
GPU_TMP_ROOT=${GRAPHCARE_GPU_TMP_ROOT:-/var/tmp/missing-xgboost-pip-tmp/graphcare-gpu}
GPU_PYTHON=${GRAPHCARE_GPU_PYTHON:-python3.12}

mkdir -p "$GPU_CACHE_ROOT" "$GPU_TMP_ROOT" "$PAPER_ROOT/.export_assets"

if [[ ! -x "$GPU_ENV/bin/python" ]]; then
  uv venv --python "$GPU_PYTHON" "$GPU_ENV"
  "$GPU_ENV/bin/python" -m ensurepip --upgrade
fi

# CUDA 12.8 is the oldest official PyTorch build that supports the RTX 5090's
# Blackwell (sm_120) kernels.  Torch stays separate because it comes from the
# official PyTorch CUDA index rather than PyPI.
env TMPDIR="$GPU_TMP_ROOT" PIP_CACHE_DIR="$GPU_CACHE_ROOT" \
  "$GPU_ENV/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0
env TMPDIR="$GPU_TMP_ROOT" PIP_CACHE_DIR="$GPU_CACHE_ROOT" \
  "$GPU_ENV/bin/python" -m pip install \
  -r "$PAPER_ROOT/requirements-gpu.txt"

echo "Prepared $GPU_ENV"
