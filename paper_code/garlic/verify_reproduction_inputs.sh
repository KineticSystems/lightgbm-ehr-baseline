#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
REPRO_ROOT="$PAPER_ROOT/.export_assets/reproduction"

mkdir -p "$REPRO_ROOT/audit"
"$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/verify_reproduction_inputs.py" \
  --cache "$PAPER_ROOT/.export_assets/author_boundary.sqlite" \
  --orders "$REPRO_ROOT/ordered_splits.json" \
  --order-report "$REPRO_ROOT/audit/ordered_splits.json" \
  --export-report "$WORKSPACE_ROOT/paper_tasks/garlic/in_hospital_mortality/audit/full_batch_recreation.json" \
  --output "$REPRO_ROOT/audit/reproduction_inputs.json" \
  2>&1 | tee "$REPRO_ROOT/audit/verify_reproduction_inputs.log"

