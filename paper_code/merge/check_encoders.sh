#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
REPRO_ROOT="$PAPER_ROOT/.export_assets/reproduction"
GPU_PYTHON="$REPRO_ROOT/gpu_venv/bin/python"
ENCODER_ROOT="$PAPER_ROOT/.export_assets/external_artifacts/merge_encoders"
GPU=${1:-0}

if [[ ! -x "$GPU_PYTHON" ]]; then
  echo "Missing MERGE GPU environment; run setup_gpu_env.sh first" >&2
  exit 1
fi

mkdir -p "$REPRO_ROOT/audit" "$REPRO_ROOT/matplotlib"
env CUDA_VISIBLE_DEVICES="$GPU" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  MPLCONFIGDIR="$REPRO_ROOT/matplotlib" \
  "$GPU_PYTHON" "$PAPER_ROOT/check_encoders.py" \
  --encoder-root "$ENCODER_ROOT" \
  --output "$REPRO_ROOT/audit/encoder_check.json" \
  --device cuda:0
