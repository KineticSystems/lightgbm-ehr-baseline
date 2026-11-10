#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
REPRO_ROOT="$PAPER_ROOT/.export_assets/reproduction"

env PYTHONHASHSEED=0 \
  "$REPRO_ROOT/gpu_venv/bin/python" "$PAPER_ROOT/verify_predictor_runs.py" \
  --runs-root "$REPRO_ROOT/predictor_runs" \
  --test-csv "$REPRO_ROOT/predictor_input/test_pred.csv" \
  --output "$REPRO_ROOT/predictor_runs/aggregate_verification.json"
