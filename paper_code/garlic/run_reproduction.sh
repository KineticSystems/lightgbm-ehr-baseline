#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
REPRO_ROOT="$PAPER_ROOT/.export_assets/reproduction"
GPU_PYTHON="$PAPER_ROOT/.venv/bin/python"
CACHE="$PAPER_ROOT/.export_assets/author_boundary.sqlite"
ORDERS="$REPRO_ROOT/ordered_splits.json"
INPUT_REPORT="$REPRO_ROOT/audit/reproduction_inputs.json"
LIGHTGBM_RESULT="$WORKSPACE_ROOT/paper_results/garlic/in_hospital_mortality.v10/result.json"
RUN_ROOT="$REPRO_ROOT/primary"
RUN_LOG="$REPRO_ROOT/audit/primary.log"
VERIFY_LOG="$REPRO_ROOT/audit/verify_primary.log"
VERIFY_REPORT="$REPRO_ROOT/audit/aggregate.json"
GPU=${1:-0}

for input_path in \
  "$GPU_PYTHON" "$CACHE" "$ORDERS" "$INPUT_REPORT" "$LIGHTGBM_RESULT"; do
  if [[ ! -e "$input_path" ]]; then
    echo "Missing required GARLIC reproduction input: $input_path" >&2
    exit 1
  fi
done
"$GPU_PYTHON" -c \
  'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "exact"' \
  "$INPUT_REPORT"
if [[ -e "$RUN_ROOT" || -e "$RUN_LOG" || -e "$VERIFY_REPORT" ]]; then
  echo "Refusing to overwrite existing GARLIC primary artifacts" >&2
  exit 1
fi

mkdir -p "$REPRO_ROOT/audit"
env CUDA_VISIBLE_DEVICES="$GPU" \
  TOKENIZERS_PARALLELISM=false \
  "$GPU_PYTHON" "$PAPER_ROOT/run_reproduction.py" \
  --cache "$CACHE" \
  --orders "$ORDERS" \
  --input-report "$INPUT_REPORT" \
  --output-root "$RUN_ROOT" \
  --device cuda:0 2>&1 | tee "$RUN_LOG"

"$GPU_PYTHON" "$PAPER_ROOT/verify_reproduction.py" \
  --cache "$CACHE" \
  --orders "$ORDERS" \
  --input-report "$INPUT_REPORT" \
  --run-root "$RUN_ROOT" \
  --run-log "$RUN_LOG" \
  --lightgbm-result "$LIGHTGBM_RESULT" \
  --output "$VERIFY_REPORT" 2>&1 | tee "$VERIFY_LOG"
