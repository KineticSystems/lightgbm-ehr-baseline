#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
REPRO_ROOT="$PAPER_ROOT/.export_assets/reproduction"
GPU_PYTHON="$REPRO_ROOT/gpu_venv/bin/python"
ENCODER_ROOT="$PAPER_ROOT/.export_assets/external_artifacts/merge_encoders"
CXR_SOURCE="$PAPER_ROOT/../fusemoe/.export_assets/external_artifacts/torchxrayvision_chex/chex-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"
BIOBERT_REVISION=551ca18efd7f052c8dfa0b01c94c2a8e68bc5488

if [[ ! -x "$GPU_PYTHON" ]]; then
  echo "Missing MERGE GPU environment; run setup_gpu_env.sh first" >&2
  exit 1
fi

env HF_HOME="$REPRO_ROOT/huggingface-home" \
  "$GPU_PYTHON" "$PAPER_ROOT/acquire_encoders.py" \
  --output-root "$ENCODER_ROOT" \
  --biobert-revision "$BIOBERT_REVISION" \
  --cxr-source "$CXR_SOURCE" \
  "$@"
