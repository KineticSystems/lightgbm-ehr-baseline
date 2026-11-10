#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"

: "${MIMIC_III_1_4_ROOT:?Exact MIMIC-III 1.4 is required}"
: "${KEDGN_EXPORT_CACHE_ROOT:=$PAPER_ROOT/.export_assets/source}"
TASK_ROOT="$WORKSPACE_ROOT/paper_tasks/kedgn/mortality_48h"
if [[ ! -f "$TASK_ROOT/manifest.json" ]]; then
  echo "Missing exported task: $TASK_ROOT (run ./export.sh first)" >&2
  exit 1
fi

"$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/check_cohort.py" \
  --mimic-root "$MIMIC_III_1_4_ROOT" \
  --cache-root "$KEDGN_EXPORT_CACHE_ROOT" \
  --task-root "$TASK_ROOT"

echo "Cohort-verified KEDGN mortality task at $TASK_ROOT"
