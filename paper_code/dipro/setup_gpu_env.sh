#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
GPU_ENV="$PAPER_ROOT/.export_assets/gpu_venv"
GPU_PYTHON_ROOT="$PAPER_ROOT/.export_assets/gpu_python"
UV_CACHE_ROOT="$PAPER_ROOT/.export_assets/gpu_uv_cache"
PIP_TMP_ROOT="$PAPER_ROOT/.export_assets/gpu_pip_tmp"
TORCH_HOME_ROOT="$PAPER_ROOT/.export_assets/torch_home"
RESNET_SOURCE="$PAPER_ROOT/.export_assets/external_artifacts/torchvision/resnet50-0676ba61.pth"
RESNET_TARGET="$TORCH_HOME_ROOT/hub/checkpoints/resnet50-0676ba61.pth"

mkdir -p "$PAPER_ROOT/.export_assets" "$GPU_PYTHON_ROOT" "$UV_CACHE_ROOT" \
  "$PIP_TMP_ROOT" "$TORCH_HOME_ROOT/hub/checkpoints"

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

# The released Lightning pin has stale metadata requiring PyYAML <=5.4.1,
# while the released DiPro requirements explicitly pin PyYAML 6.0.1.  Install
# Lightning itself without dependency resolution after installing its actual
# runtime dependencies above; this preserves both author pins.
env PIP_CACHE_DIR="$UV_CACHE_ROOT" TMPDIR="$PIP_TMP_ROOT" \
  "$GPU_ENV/bin/python" -m pip install --no-deps pytorch-lightning==1.4.2

if [[ ! -f "$RESNET_SOURCE" ]]; then
  echo "Missing verified released ResNet-50 artifact: $RESNET_SOURCE" >&2
  exit 1
fi
if [[ -e "$RESNET_TARGET" && ! -L "$RESNET_TARGET" ]]; then
  echo "Refusing to replace non-symlink ResNet cache target: $RESNET_TARGET" >&2
  exit 1
fi
if [[ ! -e "$RESNET_TARGET" ]]; then
  ln -s "$RESNET_SOURCE" "$RESNET_TARGET"
fi

echo "Prepared $GPU_ENV"
