#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
GPU_ENV="$PAPER_ROOT/.export_assets/gpu_venv"
GPU_PYTHON_ROOT="$PAPER_ROOT/.export_assets/gpu_python"
UV_CACHE_ROOT="$PAPER_ROOT/.export_assets/gpu_uv_cache"
PIP_TMP_ROOT="$PAPER_ROOT/.export_assets/gpu_pip_tmp"

mkdir -p "$PAPER_ROOT/.export_assets" "$GPU_PYTHON_ROOT" "$UV_CACHE_ROOT" "$PIP_TMP_ROOT"
if [[ ! -x "$GPU_ENV/bin/python" ]]; then
  env UV_CACHE_DIR="$UV_CACHE_ROOT" UV_PYTHON_INSTALL_DIR="$GPU_PYTHON_ROOT" \
    uv venv --python 3.11.9 --python-preference only-managed "$GPU_ENV"
fi
"$GPU_ENV/bin/python" -c \
  'import platform; assert platform.python_version() == "3.11.9", platform.python_version()'
"$GPU_ENV/bin/python" -m ensurepip --upgrade
env PIP_CACHE_DIR="$UV_CACHE_ROOT" TMPDIR="$PIP_TMP_ROOT" \
  "$GPU_ENV/bin/python" -m pip install \
  -r "$PAPER_ROOT/requirements-gpu.txt"
echo "Prepared $GPU_ENV"
