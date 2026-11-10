#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
REPRO_ROOT="$PAPER_ROOT/.export_assets/reproduction"
GPU_ENV="$REPRO_ROOT/gpu_venv"
GPU_PYTHON_ROOT="$REPRO_ROOT/gpu_python"
UV_CACHE_ROOT="$REPRO_ROOT/uv_cache"
PIP_TMP_ROOT="$REPRO_ROOT/pip_tmp"
MODEL_ROOT="$REPRO_ROOT/clinical_longformer_model"
TOKENIZER_ROOT="$PAPER_ROOT/.export_assets/clinical_longformer_tokenizer"
WEIGHT_PATH="$PAPER_ROOT/.export_assets/external_artifacts/clinical_longformer/pytorch_model.bin"

mkdir -p "$REPRO_ROOT" "$GPU_PYTHON_ROOT" "$UV_CACHE_ROOT" \
  "$PIP_TMP_ROOT" "$MODEL_ROOT"

if [[ ! -x "$GPU_ENV/bin/python" ]]; then
  env UV_CACHE_DIR="$UV_CACHE_ROOT" UV_PYTHON_INSTALL_DIR="$GPU_PYTHON_ROOT" \
    uv venv --python 3.10.20 --python-preference only-managed "$GPU_ENV"
fi

"$GPU_ENV/bin/python" -c \
  'import platform; assert platform.python_version() == "3.10.20", platform.python_version()'

"$GPU_ENV/bin/python" -m ensurepip --upgrade

# CUDA 12.8 is the first official PyTorch wheel family with Blackwell support.
# Torch 2.7.0 is the oldest release in that family, minimizing the version jump.
env PIP_CACHE_DIR="$UV_CACHE_ROOT" TMPDIR="$PIP_TMP_ROOT" \
  "$GPU_ENV/bin/python" -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128

# Install Torch first so the released Accelerate pin's broad torch>=1.4
# requirement cannot resolve to a newer, unintended PyTorch release.
env PIP_CACHE_DIR="$UV_CACHE_ROOT" TMPDIR="$PIP_TMP_ROOT" \
  "$GPU_ENV/bin/python" -m pip install \
  -r "$PAPER_ROOT/requirements-gpu.txt"

if [[ ! -f "$TOKENIZER_ROOT/config.json" ]]; then
  echo "Missing verified Clinical-Longformer config/tokenizer: $TOKENIZER_ROOT" >&2
  exit 1
fi
if [[ ! -f "$WEIGHT_PATH" ]]; then
  echo "Missing verified Clinical-Longformer weights: $WEIGHT_PATH" >&2
  exit 1
fi

for name in config.json merges.txt special_tokens_map.json tokenizer.json tokenizer_config.json vocab.json; do
  source_path="$TOKENIZER_ROOT/$name"
  target_path="$MODEL_ROOT/$name"
  if [[ -e "$target_path" && ! -L "$target_path" ]]; then
    echo "Refusing to replace non-symlink model artifact: $target_path" >&2
    exit 1
  fi
  if [[ ! -e "$target_path" ]]; then
    ln -s "$source_path" "$target_path"
  fi
done

weight_target="$MODEL_ROOT/pytorch_model.bin"
if [[ -e "$weight_target" && ! -L "$weight_target" ]]; then
  echo "Refusing to replace non-symlink model artifact: $weight_target" >&2
  exit 1
fi
if [[ ! -e "$weight_target" ]]; then
  ln -s "$WEIGHT_PATH" "$weight_target"
fi

echo "Prepared $GPU_ENV"
echo "Prepared $MODEL_ROOT"
