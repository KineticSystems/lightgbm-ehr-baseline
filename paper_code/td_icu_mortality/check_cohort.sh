#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"

: "${MIMIC_IV_3_0_ROOT:?Exact MIMIC-IV 3.0 is required}"
TASK_DIR="${TD_ICU_TASK_DIR:-$WORKSPACE_ROOT/paper_tasks/td_icu_mortality/mortality_28d}"
if [[ ! -f "$TASK_DIR/manifest.json" ]]; then
  echo "Missing exported task: $TASK_DIR (run ./export.sh first)" >&2
  exit 1
fi

cd "$PAPER_ROOT"
"$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/check_cohort.py" \
  --mimic-root "$MIMIC_IV_3_0_ROOT" \
  --task-dir "$TASK_DIR"
