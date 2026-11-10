#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"

TASKS="decompensation ihm"
if [[ $# -ne 1 ]]; then
  echo "usage: $0 TASK" >&2
  echo "tasks: $TASKS" >&2
  exit 2
fi

case "$1" in
decompensation)
  : "${MIMIC_III_1_4_ROOT:?Set MIMIC_III_1_4_ROOT to exact MIMIC-III v1.4}"
  : "${STEPWISE_SOURCE_ROOT:=/var/tmp/missing-xgboost-stepwise-${UID}}"
  TASK_DIR="$WORKSPACE_ROOT/paper_tasks/stepwise_embeddings/decompensation"
  if [[ ! -f "$TASK_DIR/manifest.json" ]]; then
    echo "Missing exported task: $TASK_DIR (run ./export.sh first)" >&2
    exit 1
  fi

  "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/check_batch.py" \
    --source-root "$STEPWISE_SOURCE_ROOT" \
    --task-dir "$TASK_DIR"
  ;;
ihm)
  : "${MIMIC_III_1_4_ROOT:?Set MIMIC_III_1_4_ROOT to exact MIMIC-III v1.4}"
  : "${STEPWISE_SOURCE_ROOT:=/var/tmp/missing-xgboost-stepwise-${UID}}"
  TASK_DIR="$WORKSPACE_ROOT/paper_tasks/stepwise_embeddings/ihm"
  if [[ ! -f "$TASK_DIR/manifest.json" ]]; then
    echo "Missing exported task: $TASK_DIR (run ./export.sh first)" >&2
    exit 1
  fi

  "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/check_batch_ihm.py" \
    --source-root "$STEPWISE_SOURCE_ROOT" \
    --task-dir "$TASK_DIR"
  ;;
*)
  echo "unknown task: $1" >&2
  echo "tasks: $TASKS" >&2
  exit 2
  ;;
esac
