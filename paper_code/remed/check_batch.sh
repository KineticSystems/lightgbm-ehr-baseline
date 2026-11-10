#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"

TASKS="los_7d_at_48h"
if [[ $# -ne 1 ]]; then
  echo "usage: $0 TASK" >&2
  echo "tasks: $TASKS" >&2
  exit 2
fi

case "$1" in
los_7d_at_48h)
  : "${MIMIC_IV_2_0_ROOT:?Set MIMIC_IV_2_0_ROOT to exact MIMIC-IV v2.0}"
  : "${MIMIC_IV_ED_2_0_ROOT:?Set MIMIC_IV_ED_2_0_ROOT to exact MIMIC-IV-ED v2.0}"

  SOURCE_ROOT="$PAPER_ROOT/.export_work/source_48h"
  TASK_DIR="$WORKSPACE_ROOT/paper_tasks/remed/los_7d_at_48h"
  if [[ ! -f "$TASK_DIR/manifest.json" ]]; then
    echo "Missing exported task: $TASK_DIR (run ./export.sh first)" >&2
    exit 1
  fi

  "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/check_batch.py" \
    --source-root "$SOURCE_ROOT" \
    --task-dir "$TASK_DIR"
  ;;
mortality_14d_at_48h|readmission_at_48h)
  # Sibling outcomes reuse the base task's verified event stream and
  # inherit its audit records, so they have nothing separate to check.
  echo "$1 derives from los_7d_at_48h; check that task instead" >&2
  exit 2
  ;;
*)
  echo "unknown task: $1" >&2
  echo "tasks: $TASKS" >&2
  exit 2
  ;;
esac
