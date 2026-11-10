#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 SEED" >&2
  exit 2
fi
SEED=$1
case "$SEED" in
  2020|2021|2022) ;;
  *) echo "seed must be one of 2020, 2021, or 2022" >&2; exit 2 ;;
esac

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
REPRO_ROOT="$PAPER_ROOT/.export_assets/reproduction"
GPU_ENV="$REPRO_ROOT/gpu_venv"
AUDIT_ROOT="$REPRO_ROOT/audit"
SAVE_ROOT="$REPRO_ROOT/runs"
RUN_DIR="$SAVE_ROOT/remed_${SEED}"
ENCODER_LOG="$SAVE_ROOT/encoder_${SEED}.console.log"
SOURCE_LOG="$SAVE_ROOT/remed_${SEED}.console.log"
OUTPUT_DIR="$RUN_DIR/verification"
CONSOLE_LOG="$RUN_DIR/verification.console.log"

if [[ ! -x "$GPU_ENV/bin/python" ]]; then
  echo "Blackwell REMed environment has not completed" >&2
  exit 1
fi
for path in \
  "$AUDIT_ROOT/gpu_environment.json" \
  "$AUDIT_ROOT/input_shim.json" \
  "$SAVE_ROOT/encoder_${SEED}/mimiciv_encoded.h5" \
  "$RUN_DIR/checkpoint_best.pt" \
  "$ENCODER_LOG" \
  "$SOURCE_LOG"; do
  if [[ ! -s "$path" ]]; then
    echo "Missing terminal seed artifact: $path" >&2
    exit 1
  fi
done
if [[ -e "$OUTPUT_DIR" || -e "$CONSOLE_LOG" ]]; then
  echo "Refusing to overwrite seed-$SEED verification artifacts" >&2
  exit 1
fi

cd "$PAPER_ROOT"
env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
  "$GPU_ENV/bin/python" evaluate_reproduction_seed.py \
  --seed "$SEED" \
  --input-root "$REPRO_ROOT/input" \
  --save-root "$SAVE_ROOT" \
  --encoder-console-log "$ENCODER_LOG" \
  --source-console-log "$SOURCE_LOG" \
  --environment-report "$AUDIT_ROOT/gpu_environment.json" \
  --input-report "$AUDIT_ROOT/input_shim.json" \
  --output-dir "$OUTPUT_DIR" 2>&1 | tee "$CONSOLE_LOG"

env REPORT="$OUTPUT_DIR/report.json" "$GPU_ENV/bin/python" -c \
  'import json,os; r=json.load(open(os.environ["REPORT"])); assert r["status"] == "passed" and r["test_accesses_total"] >= 3'
