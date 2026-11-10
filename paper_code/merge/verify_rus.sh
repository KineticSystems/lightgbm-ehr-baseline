#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
REPRO_ROOT="$PAPER_ROOT/.export_assets/reproduction"
GPU_PYTHON="$REPRO_ROOT/gpu_venv/bin/python"

if [[ ! -x "$GPU_PYTHON" ]]; then
  echo "Missing MERGE reproduction environment" >&2
  exit 1
fi

mkdir -p "$REPRO_ROOT/audit" "$REPRO_ROOT/matplotlib"
env MPLCONFIGDIR="$REPRO_ROOT/matplotlib" \
  "$GPU_PYTHON" "$PAPER_ROOT/verify_rus.py" \
  --rus "$REPRO_ROOT/results/ihm/rus_multimodal_all_seq48_lags8_meanpool.npy" \
  --input-report "$REPRO_ROOT/audit/ihm_inputs.json" \
  --train-pickle "$REPRO_ROOT/data/ihm/train_ihm-48-cxr-notes-missingInd-standardized_stays.pkl" \
  --run-log "$REPRO_ROOT/audit/rus_multimodal.log" \
  --output "$REPRO_ROOT/audit/rus_multimodal.json" \
  2>&1 | tee "$REPRO_ROOT/audit/verify_rus.log"

