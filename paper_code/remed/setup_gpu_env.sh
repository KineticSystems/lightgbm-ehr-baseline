#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
REPRO_ROOT="$PAPER_ROOT/.export_assets/reproduction"
GPU_ENV="$REPRO_ROOT/gpu_venv"
UV_CACHE_ROOT="$REPRO_ROOT/uv_cache"
PIP_TMP_ROOT="$REPRO_ROOT/pip_tmp"
AUDIT_ROOT="$REPRO_ROOT/audit"

mkdir -p "$REPRO_ROOT" "$UV_CACHE_ROOT" "$PIP_TMP_ROOT" "$AUDIT_ROOT"

if [[ ! -x "$PAPER_ROOT/.venv/bin/python" ]]; then
  echo "Missing verified REMed CPU environment: $PAPER_ROOT/.venv" >&2
  exit 1
fi

if [[ ! -x "$GPU_ENV/bin/python" ]]; then
  env UV_CACHE_DIR="$UV_CACHE_ROOT" \
    uv venv --python "$PAPER_ROOT/.venv/bin/python" "$GPU_ENV"
fi

"$GPU_ENV/bin/python" -c \
  'import platform; assert platform.python_version() == "3.10.20", platform.python_version()'

"$GPU_ENV/bin/python" -m ensurepip --upgrade

env PIP_CACHE_DIR="$UV_CACHE_ROOT" TMPDIR="$PIP_TMP_ROOT" \
  "$GPU_ENV/bin/python" -m pip install \
  -r "$PAPER_ROOT/requirements-gpu.txt"

env PIP_CACHE_DIR="$UV_CACHE_ROOT" TMPDIR="$PIP_TMP_ROOT" \
  "$GPU_ENV/bin/python" -m pip check

cd "$PAPER_ROOT"
"$GPU_ENV/bin/python" record_gpu_environment.py \
  --output "$AUDIT_ROOT/gpu_environment.json" \
  --freeze "$AUDIT_ROOT/gpu_environment_freeze.txt" \
  --requirements "$PAPER_ROOT/requirements-gpu.txt"

echo "Prepared $GPU_ENV"
