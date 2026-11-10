#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 SEED" >&2
  exit 2
fi
SEED=$1
case "$SEED" in
  32|42|52|62|72) ;;
  *) echo "Seed must be one of: 32 42 52 62 72" >&2; exit 2 ;;
esac

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
REPRO_ROOT="$PAPER_ROOT/.export_assets/reproduction"
GPU_PYTHON="$REPRO_ROOT/gpu_venv/bin/python"
INPUT_DIR="$REPRO_ROOT/input"
MODEL_ROOT="$REPRO_ROOT/clinical_longformer_model"
RUN_DIR="$REPRO_ROOT/primary/seed_$SEED"

if [[ ! -x "$GPU_PYTHON" ]]; then
  echo "Missing GPU environment; run setup_gpu_env.sh first" >&2
  exit 1
fi
if [[ ! -f "$INPUT_DIR/verification.json" ]]; then
  echo "Missing verified reproduction inputs" >&2
  exit 1
fi
if [[ -d "$RUN_DIR" ]] && [[ -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to mix with existing run directory: $RUN_DIR" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"

env TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  "$GPU_PYTHON" "$PAPER_ROOT/reproduce_table4.py" \
  --seed "$SEED" \
  --input-dir "$INPUT_DIR" \
  --model-root "$MODEL_ROOT" \
  --run-dir "$RUN_DIR" \
  2>&1 | tee "$RUN_DIR/console.log"

