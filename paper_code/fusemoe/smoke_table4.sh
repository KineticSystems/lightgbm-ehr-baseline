#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
REPRO_ROOT="$PAPER_ROOT/.export_assets/reproduction"
GPU_PYTHON="$REPRO_ROOT/gpu_venv/bin/python"
INPUT_DIR="$REPRO_ROOT/input"
MODEL_ROOT="$REPRO_ROOT/clinical_longformer_model"
OUTPUT_DIR="$REPRO_ROOT/smoke"

if [[ ! -x "$GPU_PYTHON" ]]; then
  echo "Missing GPU environment; run setup_gpu_env.sh first" >&2
  exit 1
fi
if [[ ! -f "$INPUT_DIR/verification.json" ]]; then
  echo "Missing verified reproduction inputs" >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR"

env TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  "$GPU_PYTHON" "$PAPER_ROOT/smoke_table4.py" \
  --input-dir "$INPUT_DIR" \
  --model-root "$MODEL_ROOT" \
  --output-dir "$OUTPUT_DIR"

