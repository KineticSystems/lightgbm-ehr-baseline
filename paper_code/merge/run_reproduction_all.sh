#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
REPRO_ROOT="$PAPER_ROOT/.export_assets/reproduction"
PRIMARY_ROOT="$REPRO_ROOT/primary"
GPU_PYTHON="$REPRO_ROOT/gpu_venv/bin/python"
GPU=${1:-0}

for seed in 0 1 2 3 4; do
  report="$PRIMARY_ROOT/reports/seed_${seed}.json"
  if [[ -f "$report" ]]; then
    "$GPU_PYTHON" -c \
      'import json,sys; report=json.load(open(sys.argv[1])); assert report["status"] == "exact"' \
      "$report"
    echo "MERGE seed $seed is already independently verified; skipping"
  else
    "$PAPER_ROOT/run_reproduction_seed.sh" "$seed" "$GPU"
  fi
done

"$GPU_PYTHON" "$PAPER_ROOT/summarize_reproduction.py" \
  --reports-dir "$PRIMARY_ROOT/reports" \
  --output "$PRIMARY_ROOT/result.json"
