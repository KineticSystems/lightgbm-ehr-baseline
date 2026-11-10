#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 small|maximum [GPU]" >&2
  exit 2
fi

PROFILE=$1
GPU=${2:-0}
case "$PROFILE" in
  small|maximum) ;;
  *) echo "profile must be small or maximum" >&2; exit 2 ;;
esac

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
REPRO_ROOT="$PAPER_ROOT/.export_assets/reproduction"
GPU_ENV="$REPRO_ROOT/gpu_venv"
AUDIT_ROOT="$REPRO_ROOT/audit"
SMOKE_ROOT="$REPRO_ROOT/smoke"
OUTPUT_DIR="$SMOKE_ROOT/$PROFILE"
CONSOLE_LOG="$SMOKE_ROOT/${PROFILE}.console.log"
COHORT_REPORT="$WORKSPACE_ROOT/paper_tasks/remed/los_7d_at_48h.hashed_vocab_20260901/audit/cohort_reconstruction.json"

if [[ ! -x "$GPU_ENV/bin/python" ]]; then
  echo "Blackwell REMed environment has not completed" >&2
  exit 1
fi
for report in "$AUDIT_ROOT/gpu_environment.json" "$AUDIT_ROOT/input_shim.json"; do
  if [[ ! -s "$report" ]]; then
    echo "Missing required REMed gate: $report" >&2
    exit 1
  fi
done
if [[ -e "$OUTPUT_DIR" || -e "$CONSOLE_LOG" ]]; then
  echo "Refusing to overwrite existing REMed $PROFILE smoke artifacts" >&2
  exit 1
fi

extra_args=()
if [[ "$PROFILE" == "maximum" ]]; then
  SMALL_REPORT="$SMOKE_ROOT/small/report.json"
  if [[ ! -s "$SMALL_REPORT" ]]; then
    echo "Small two-stage smoke has not completed" >&2
    exit 1
  fi
  extra_args=(--small-report "$SMALL_REPORT")
fi

mkdir -p "$SMOKE_ROOT"
cd "$PAPER_ROOT"
env CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
  "$GPU_ENV/bin/python" smoke_reproduction.py \
  --profile "$PROFILE" \
  --output-dir "$OUTPUT_DIR" \
  --environment-report "$AUDIT_ROOT/gpu_environment.json" \
  --input-report "$AUDIT_ROOT/input_shim.json" \
  --cohort-report "$COHORT_REPORT" \
  "${extra_args[@]}" 2>&1 | tee "$CONSOLE_LOG"

env REPORT="$OUTPUT_DIR/report.json" "$GPU_ENV/bin/python" -c \
  'import json,os; r=json.load(open(os.environ["REPORT"])); assert r["status"] == "passed" and r["test_accesses"] == 0'
