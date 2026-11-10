#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 train|validate|test" >&2
  exit 2
fi
PARTITION="$1"
case "$PARTITION" in
  train|validate|test) ;;
  *) echo "Invalid partition: $PARTITION" >&2; exit 2 ;;
esac

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"
: "${MIMIC_CXR_JPG_2_0_0_ROOT:?MIMIC_CXR_JPG_2_0_0_ROOT must identify the exact 2.0.0 payload}"
CXR_FILES_ROOT="$MIMIC_CXR_JPG_2_0_0_ROOT/files"

REPRO_ROOT="$PAPER_ROOT/.export_assets/reproduction"
GPU_ENV="$REPRO_ROOT/gpu_venv"
CHECKPOINT_ROOT="$PAPER_ROOT/.export_assets/external_artifacts/huggingface_jasmineChen_ddl-cxr_1a7b6d3776745194f1426f8b2c9f7e8b1ebd5033"
RUN_ROOT="$REPRO_ROOT/generation/$PARTITION"

if [[ -e "$RUN_ROOT" ]]; then
  echo "Refusing prior generation run directory: $RUN_ROOT" >&2
  exit 1
fi
mkdir -p "$RUN_ROOT"

cd "$PAPER_ROOT"
env CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=23 \
  "$GPU_ENV/bin/python" "$PAPER_ROOT/generate_released_latents.py" \
  --partition "$PARTITION" \
  --checkpoint "$CHECKPOINT_ROOT/ddl-cxr-ldm.ckpt" \
  --cxr-root "$CXR_FILES_ROOT" \
  --input-root "$REPRO_ROOT/released_policy" \
  --output-root "$RUN_ROOT" \
  --batch-size 48 \
  --num-workers 4 2>&1 | tee "$RUN_ROOT/console.log"
