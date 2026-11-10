#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"

REPRO_ROOT="$PAPER_ROOT/.export_assets/reproduction"
mkdir -p "$REPRO_ROOT/audit"

"$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/prepare_reproduction_inputs.py" \
  --cache "$PAPER_ROOT/.export_assets/author_boundary.sqlite" \
  --mimic-root "$MIMIC_III_1_4_ROOT" \
  --orders "$REPRO_ROOT/ordered_splits.json" \
  --report "$REPRO_ROOT/audit/ordered_splits.json" \
  2>&1 | tee "$REPRO_ROOT/audit/prepare_reproduction_inputs.log"

