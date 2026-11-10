#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
GPU_ENV="$PAPER_ROOT/.export_assets/gpu_venv"
GPU_CACHE_ROOT="$PAPER_ROOT/.export_assets/gpu_uv_cache"
GPU_TMP_ROOT="$PAPER_ROOT/.export_assets/gpu_tmp"
GPU_PYTHON=${MMUGL_GPU_PYTHON:-3.9.25}

mkdir -p "$GPU_CACHE_ROOT" "$GPU_TMP_ROOT" "$PAPER_ROOT/.export_assets"

if [[ ! -x "$GPU_ENV/bin/python" ]]; then
  env UV_CACHE_DIR="$GPU_CACHE_ROOT" \
    uv venv --python "$GPU_PYTHON" "$GPU_ENV"
fi

if ! "$GPU_ENV/bin/python" -m pip --version >/dev/null 2>&1; then
  "$GPU_ENV/bin/python" -m ensurepip --upgrade
fi

env PIP_CACHE_DIR="$GPU_CACHE_ROOT" TMPDIR="$GPU_TMP_ROOT" \
  "$GPU_ENV/bin/python" -m pip install pip==22.2.2 setuptools==58.0.4

env PIP_CACHE_DIR="$GPU_CACHE_ROOT" TMPDIR="$GPU_TMP_ROOT" \
  "$GPU_ENV/bin/python" -m pip install -r "$PAPER_ROOT/requirements-gpu.txt"

# spaCy 3.4.1 constrains pydantic below 1.10, but that release fails with the
# typing-extensions version required by Torch 2.7. The final pydantic 1.x line
# preserves the same API and is used only by the unused QuickUMLS import path
# during MIMIC training.
env PIP_CACHE_DIR="$GPU_CACHE_ROOT" TMPDIR="$GPU_TMP_ROOT" \
  "$GPU_ENV/bin/python" -m pip install --no-deps pydantic==1.10.22

# Lightning 1.6.4's stale package metadata constrains PyYAML to a source-only
# legacy release. Its actual runtime works with the released PyYAML 6.0 pin, so
# retain Lightning itself exactly and bypass only dependency re-resolution.
env PIP_CACHE_DIR="$GPU_CACHE_ROOT" TMPDIR="$GPU_TMP_ROOT" \
  "$GPU_ENV/bin/python" -m pip install --no-deps pytorch-lightning==1.6.4

echo "Prepared $GPU_ENV"
