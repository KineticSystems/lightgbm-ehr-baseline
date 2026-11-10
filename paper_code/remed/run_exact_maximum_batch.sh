#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 SEED [GPU]" >&2
  exit 2
fi
SEED=$1
GPU=${2:-0}
case "$SEED" in
  2020|2021|2022) ;;
  *) echo "seed must be one of 2020, 2021, or 2022" >&2; exit 2 ;;
esac

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
REPRO_ROOT="$PAPER_ROOT/.export_assets/reproduction"
GPU_ENV="$REPRO_ROOT/gpu_venv"
AUDIT_ROOT="$REPRO_ROOT/audit"
ENCODER_DIR="$REPRO_ROOT/runs/encoder_${SEED}"
OUTPUT="$ENCODER_DIR/exact_maximum_batch_report.json"
CONSOLE="$ENCODER_DIR/exact_maximum_batch.console.log"

if [[ ! -x "$GPU_ENV/bin/python" ]]; then
  echo "Blackwell REMed environment has not completed" >&2
  exit 1
fi
for report in \
  "$AUDIT_ROOT/gpu_environment.json" \
  "$REPRO_ROOT/smoke/small/report.json" \
  "$ENCODER_DIR/encoded_cache_report.json"; do
  if [[ ! -s "$report" ]]; then
    echo "Missing required exact-batch gate: $report" >&2
    exit 1
  fi
done
if [[ -e "$OUTPUT" || -e "$CONSOLE" ]]; then
  echo "Refusing to overwrite seed-$SEED exact maximum-batch artifacts" >&2
  exit 1
fi

cd "$PAPER_ROOT"
env CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
  "$GPU_ENV/bin/python" verify_exact_maximum_batch.py \
  --seed "$SEED" \
  --input-root "$REPRO_ROOT/input" \
  --save-root "$REPRO_ROOT/runs" \
  --cache-report "$ENCODER_DIR/encoded_cache_report.json" \
  --environment-report "$AUDIT_ROOT/gpu_environment.json" \
  --small-smoke-report "$REPRO_ROOT/smoke/small/report.json" \
  --output "$OUTPUT" 2>&1 | tee "$CONSOLE"

env REPORT="$OUTPUT" "$GPU_ENV/bin/python" -c \
  'import json,os; r=json.load(open(os.environ["REPORT"])); assert r["status"] == "passed" and r["test_metric_accesses"] == 0'
