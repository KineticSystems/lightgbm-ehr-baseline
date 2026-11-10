#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
RUNTIME_ROOT="$WORKSPACE_ROOT/paper_results/smart/runtime"
GPU_PYTHON="$PAPER_ROOT/.export_assets/gpu_venv/bin/python"

if [[ $# -ne 2 ]]; then
  echo "usage: $0 smoke|full SEED" >&2
  exit 2
fi
MODE=$1
SEED=$2
case "$SEED" in
  1|42|3407) ;;
  *) echo "SMART seed must be 1, 42, or 3407" >&2; exit 2 ;;
esac
if [[ "$MODE" == "smoke" ]]; then
  if [[ "$SEED" != "1" ]]; then
    echo "the prespecified smoke seed is 1" >&2
    exit 2
  fi
  EPOCHS=1
  RUN_ROOT="$WORKSPACE_ROOT/paper_results/smart/smoke/seed_1/author_run"
  OUTPUT_ROOT="$WORKSPACE_ROOT/paper_results/smart/smoke"
elif [[ "$MODE" == "full" ]]; then
  EPOCHS=25
  RUN_ROOT="$WORKSPACE_ROOT/paper_results/smart/reproduction/seed_${SEED}/author_run"
  OUTPUT_ROOT="$WORKSPACE_ROOT/paper_results/smart/reproduction"
else
  echo "mode must be smoke or full" >&2
  exit 2
fi

PICKLE="$RUNTIME_ROOT/data/MIMIC-III/mortality_normalized.pkl"
if [[ ! -f "$PICKLE" ]]; then
  echo "missing verified runtime pickle: $PICKLE" >&2
  exit 1
fi
MODEL_DIR="$RUN_ROOT/mimic_mortality/smart"
RESULT="$OUTPUT_ROOT/seed_${SEED}/result.json"
if [[ -f "$RESULT" ]]; then
  echo "completed result already exists: $RESULT"
  exit 0
fi
if [[ -e "$MODEL_DIR" ]]; then
  echo "refusing to overwrite incomplete SMART run: $MODEL_DIR" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT"
cd "$RUNTIME_ROOT"
env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  "$GPU_PYTHON" "$PAPER_ROOT/main_pretrain.py" \
  --dataset mimic_mortality \
  --seed "$SEED" \
  --batch_size 256 \
  --local-rank 0 \
  --epochs "$EPOCHS" \
  --save_dir "$RUN_ROOT"
mv "$MODEL_DIR/training.log" "$MODEL_DIR/pretraining.log"

env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  "$GPU_PYTHON" "$PAPER_ROOT/main_finetune.py" \
  --dataset mimic_mortality \
  --seed "$SEED" \
  --batch_size 256 \
  --local-rank 0 \
  --epochs "$EPOCHS" \
  --save_dir "$RUN_ROOT"
mv "$MODEL_DIR/training.log" "$MODEL_DIR/finetuning.log"

env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  "$GPU_PYTHON" "$PAPER_ROOT/evaluate_reproduction.py" \
  --seed "$SEED" \
  --run-root "$RUN_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --epochs "$EPOCHS" \
  --batch-size 256
