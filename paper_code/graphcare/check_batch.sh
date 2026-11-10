#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"

TASKS="subsequent_visit_mortality readmission_15d mimic4_subsequent_visit_mortality mimic4_readmission_15d"
if [[ $# -ne 1 ]]; then
  echo "usage: $0 TASK" >&2
  echo "tasks: $TASKS" >&2
  exit 2
fi

case "$1" in
subsequent_visit_mortality)
  : "${MIMIC_III_1_4_ROOT:?Set the exact MIMIC-III 1.4 source root}"

  TASK_DIR="$WORKSPACE_ROOT/paper_tasks/graphcare/subsequent_visit_mortality"
  if [[ ! -f "$TASK_DIR/manifest.json" ]]; then
    echo "Missing exported task: $TASK_DIR" >&2
    exit 1
  fi

  PYTHONHASHSEED=0 \
    "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/check_batch.py" \
    --mimic-root "$MIMIC_III_1_4_ROOT" \
    --task-dir "$TASK_DIR"
  echo "Batch-verified $TASK_DIR"
  ;;
readmission_15d|mimic4_subsequent_visit_mortality|mimic4_readmission_15d)
  # The MIMIC-IV tasks and the MIMIC-III readmission task run the
  # released PyHealth path against a different source release.
  exec "$PAPER_ROOT/check_batch_phase2.sh" "$1"
  ;;
*)
  echo "unknown task: $1" >&2
  echo "tasks: $TASKS" >&2
  exit 2
  ;;
esac
