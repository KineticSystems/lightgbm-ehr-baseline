#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"

: "${MIMIC_CXR_JPG_2_0_0_ROOT:?MIMIC_CXR_JPG_2_0_0_ROOT must identify the exact 2.0.0 payload}"
CXR_FILES_ROOT="$MIMIC_CXR_JPG_2_0_0_ROOT/files"
if [[ ! -d "$CXR_FILES_ROOT" ]]; then
  echo "Missing verified CXR image payload directory: $CXR_FILES_ROOT" >&2
  exit 1
fi

REPRO_ROOT="$PAPER_ROOT/.export_assets/reproduction"
GPU_ENV="$REPRO_ROOT/gpu_venv"
CHECKPOINT_ROOT="$PAPER_ROOT/.export_assets/external_artifacts/huggingface_jasmineChen_ddl-cxr_1a7b6d3776745194f1426f8b2c9f7e8b1ebd5033"

cd "$PAPER_ROOT"
exec env CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=23 \
  "$GPU_ENV/bin/python" "$PAPER_ROOT/smoke_ldm.py" \
  --checkpoint "$CHECKPOINT_ROOT/ddl-cxr-ldm.ckpt" \
  --cxr-root "$CXR_FILES_ROOT" \
  --metadata-root "$REPRO_ROOT/released_policy" \
  --output "$REPRO_ROOT/smoke/ldm_batch48_200steps.json" \
  --batch-size 48 \
  --ddim-steps 200
