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
ENCODER_DIR="$REPRO_ROOT/runs/encoder_${SEED}"
OUTPUT="$ENCODER_DIR/encoded_cache_report.json"
CONSOLE="$ENCODER_DIR/encoded_cache_verification.console.log"

if [[ ! -x "$GPU_ENV/bin/python" ]]; then
  echo "Blackwell REMed environment has not completed" >&2
  exit 1
fi
if [[ -e "$OUTPUT" || -e "$CONSOLE" ]]; then
  echo "Refusing to overwrite seed-$SEED encoded-cache verification" >&2
  exit 1
fi

cd "$PAPER_ROOT"
"$GPU_ENV/bin/python" verify_encoded_cache.py \
  --source-hdf "$REPRO_ROOT/input/48h/mimiciv.h5" \
  --cohort-csv "$REPRO_ROOT/input/48h/mimiciv_cohort.csv" \
  --encoded-hdf "$ENCODER_DIR/mimiciv_encoded.h5" \
  --stage-report "$ENCODER_DIR/stage_report.json" \
  --output "$OUTPUT" 2>&1 | tee "$CONSOLE"

env REPORT="$OUTPUT" "$GPU_ENV/bin/python" -c \
  'import json,os; r=json.load(open(os.environ["REPORT"])); assert r["status"] == "passed" and r["stays"] == 32449 and r["events"] == 129548126'
