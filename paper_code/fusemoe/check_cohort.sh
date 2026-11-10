#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"

: "${FUSEMOE_SOURCE_DATA_ROOT:?Set FUSEMOE_SOURCE_DATA_ROOT to the released split-pickle directory}"
TASK_DIR="$WORKSPACE_ROOT/paper_tasks/fusemoe/mortality_48h"
if [[ ! -f "$TASK_DIR/manifest.json" ]]; then
  echo "Missing exported task: $TASK_DIR" >&2
  exit 1
fi

"$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/check_cohort.py" \
  --source-root "$FUSEMOE_SOURCE_DATA_ROOT" \
  --task-dir "$TASK_DIR"

echo "Cohort-verified $TASK_DIR"
