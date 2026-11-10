#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
REPRO_ROOT="$PAPER_ROOT/.export_assets/reproduction"
GPU_PYTHON="$REPRO_ROOT/gpu_venv/bin/python"
DATA_ROOT="$REPRO_ROOT/data/ihm"
INPUT_REPORT="$REPRO_ROOT/audit/ihm_inputs.json"
OUTPUT="$REPRO_ROOT/audit/training_dtype_boundary.json"
LOG="$REPRO_ROOT/audit/verify_training_dtype_boundary.log"

for input_path in \
  "$GPU_PYTHON" \
  "$INPUT_REPORT" \
  "$DATA_ROOT/train_ihm-48-cxr-notes-missingInd-standardized_stays.pkl" \
  "$DATA_ROOT/val_ihm-48-cxr-notes-missingInd-standardized_stays.pkl" \
  "$DATA_ROOT/test_ihm-48-cxr-notes-missingInd-standardized_stays.pkl"; do
  if [[ ! -e "$input_path" ]]; then
    echo "Missing required verified input: $input_path" >&2
    exit 1
  fi
done

mkdir -p "$REPRO_ROOT/audit"
"$GPU_PYTHON" "$PAPER_ROOT/verify_training_dtype_boundary.py" \
  --input-report "$INPUT_REPORT" \
  --train "$DATA_ROOT/train_ihm-48-cxr-notes-missingInd-standardized_stays.pkl" \
  --validation "$DATA_ROOT/val_ihm-48-cxr-notes-missingInd-standardized_stays.pkl" \
  --test "$DATA_ROOT/test_ihm-48-cxr-notes-missingInd-standardized_stays.pkl" \
  --output "$OUTPUT" 2>&1 | tee "$LOG"
