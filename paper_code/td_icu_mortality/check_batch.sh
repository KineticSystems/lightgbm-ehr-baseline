#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
TASK_DIR="${TD_ICU_TASK_DIR:-$WORKSPACE_ROOT/paper_tasks/td_icu_mortality/mortality_28d}"
if [[ ! -f "$TASK_DIR/manifest.json" ]]; then
  echo "Missing exported task: $TASK_DIR (run ./export.sh first)" >&2
  exit 1
fi

cd "$PAPER_ROOT"
"$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/check_batch.py" \
  --task-dir "$TASK_DIR"
