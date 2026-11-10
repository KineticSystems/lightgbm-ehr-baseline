#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
GPU_PYTHON="$PAPER_ROOT/.export_assets/gpu_venv/bin/python"
ADAPTER_CHECK="$WORKSPACE_ROOT/paper_results/dpmm/runtime/reproduction_data_check.json"

if [[ $# -ne 1 ]]; then
  echo "usage: $0 smoke|clean|released-loop" >&2
  exit 2
fi
MODE=$1
case "$MODE" in
  smoke)
    OUTPUT_ROOT="$WORKSPACE_ROOT/paper_results/dpmm/smoke"
    ;;
  clean)
    OUTPUT_ROOT="$WORKSPACE_ROOT/paper_results/dpmm/reproduction/clean_primary"
    ;;
  released-loop)
    OUTPUT_ROOT="$WORKSPACE_ROOT/paper_results/dpmm/reproduction/released_loop"
    ;;
  *)
    echo "mode must be smoke, clean, or released-loop" >&2
    exit 2
    ;;
esac

if [[ ! -x "$GPU_PYTHON" ]]; then
  echo "missing DPMM GPU environment; run $PAPER_ROOT/setup_gpu_env.sh" >&2
  exit 1
fi
if [[ ! -f "$ADAPTER_CHECK" ]]; then
  echo "missing exhaustive DPMM adapter check: $ADAPTER_CHECK" >&2
  exit 1
fi

cd "$PAPER_ROOT"
env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false \
  WANDB_MODE=disabled \
  WANDB_SILENT=true \
  MPLBACKEND=Agg \
  PYTHONHASHSEED=1002 \
  "$GPU_PYTHON" -u "$PAPER_ROOT/reproduce_dpmm.py" \
  --mode "$MODE" \
  --output-root "$OUTPUT_ROOT"
