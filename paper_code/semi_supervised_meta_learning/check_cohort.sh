#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)

TASKS="decompensation mortality_48h"
if [[ $# -ne 1 ]]; then
  echo "usage: $0 TASK" >&2
  echo "tasks: $TASKS" >&2
  exit 2
fi
TASK="$1"

: "${SSML_SOURCE_ROOT:=/var/tmp/missing-xgboost-decomp-family-${UID}}"
: "${DECOMP_FAMILY_M3B_SOURCE_ROOT:=/var/tmp/missing-xgboost-stepwise-${UID}}"
TASK_DIR="$WORKSPACE_ROOT/paper_tasks/semi_supervised_meta_learning/$TASK"
if [[ ! -f "$TASK_DIR/manifest.json" ]]; then
  echo "Missing exported task: $TASK_DIR (run ./export.sh first)" >&2
  exit 1
fi
PYTHONPATH="$PAPER_ROOT:$WORKSPACE_ROOT/lightgbm_code/src"
export PYTHONPATH

case "$TASK" in
  decompensation)
    "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/check_cohort.py" \
      --source-root "$SSML_SOURCE_ROOT" \
      --task-dir "$TASK_DIR"
    ;;
  mortality_48h)
    "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/check_cohort_ihm.py" \
      --source-root "$DECOMP_FAMILY_M3B_SOURCE_ROOT" \
      --task-dir "$TASK_DIR"
    ;;
  decompensation_sampled)
    # The sampled subset reuses the base task's verified events and inherits
    # its audit records, so there is nothing separate to check here.
    echo "$TASK is drawn from decompensation; check that task instead" >&2
    exit 2
    ;;
  *)
    echo "unknown task: $TASK" >&2
    echo "tasks: $TASKS" >&2
    exit 2
    ;;
esac
