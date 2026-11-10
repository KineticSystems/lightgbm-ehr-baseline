#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
REPRO_ROOT="$PAPER_ROOT/.export_assets/reproduction"
GPU_ENV="$REPRO_ROOT/gpu_venv"
UV_CACHE_ROOT="$REPRO_ROOT/uv_cache"
PIP_TMP_ROOT="$REPRO_ROOT/pip_tmp"

mkdir -p "$REPRO_ROOT" "$UV_CACHE_ROOT" "$PIP_TMP_ROOT"

if [[ ! -x "$GPU_ENV/bin/python" ]]; then
  env UV_CACHE_DIR="$UV_CACHE_ROOT" \
    uv venv --python "$PAPER_ROOT/.venv/bin/python" "$GPU_ENV"
fi

"$GPU_ENV/bin/python" -c \
  'import platform; assert platform.python_version() == "3.10.20", platform.python_version()'

"$GPU_ENV/bin/python" -m ensurepip --upgrade

# CUDA 12.8 is the first official wheel family supporting Blackwell. Torch
# 2.7.0 is the oldest release in that family and is the sole intentional
# change from the released Torch 2.6.0 / torchvision 0.21.0 requirements.
env PIP_CACHE_DIR="$UV_CACHE_ROOT" TMPDIR="$PIP_TMP_ROOT" \
  "$GPU_ENV/bin/python" -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128

env PIP_CACHE_DIR="$UV_CACHE_ROOT" TMPDIR="$PIP_TMP_ROOT" \
  "$GPU_ENV/bin/python" -m pip install \
  -r "$PAPER_ROOT/requirements-gpu.txt"

"$GPU_ENV/bin/python" -m pip check
"$GPU_ENV/bin/python" - <<'PY'
import platform
import torch
import torchvision
import torchxrayvision
import transformers

print("python", platform.python_version())
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("torchvision", torchvision.__version__)
print("torchxrayvision", torchxrayvision.__version__)
print("transformers", transformers.__version__)
assert torch.__version__ == "2.7.0+cu128"
assert torchvision.__version__ == "0.22.0+cu128"
assert torchxrayvision.__version__ == "1.5.2"
assert transformers.__version__ == "5.15.1"
PY

echo "Prepared $GPU_ENV"
