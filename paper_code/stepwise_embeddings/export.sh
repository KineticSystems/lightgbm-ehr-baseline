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

  if [[ -e "$TASK_DIR" ]]; then
    echo "Refusing to overwrite existing task: $TASK_DIR" >&2
    exit 1
  fi

  if [[ ! -f "$STEPWISE_SOURCE_ROOT/preparation.json" ]]; then
    "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/prepare_cohort_source.py" \
      --mimic-root "$MIMIC_III_1_4_ROOT" \
      --source-root "$STEPWISE_SOURCE_ROOT"
  fi

  "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/export_source.py" \
    --source-root "$STEPWISE_SOURCE_ROOT" \
    --output-dir "$TASK_DIR"

  "$WORKSPACE_ROOT/lightgbm_code/.venv/bin/python" -m \
    missing_xgboost_lightgbm.validate_task "$TASK_DIR"
  ;;
ihm)
  : "${MIMIC_III_1_4_ROOT:?Set MIMIC_III_1_4_ROOT to exact MIMIC-III v1.4}"
  : "${STEPWISE_SOURCE_ROOT:=/var/tmp/missing-xgboost-stepwise-${UID}}"
  TASK_DIR="$WORKSPACE_ROOT/paper_tasks/stepwise_embeddings/ihm"

  if [[ -e "$TASK_DIR" ]]; then
    echo "Refusing to overwrite existing task: $TASK_DIR" >&2
    exit 1
  fi

  if [[ ! -f "$STEPWISE_SOURCE_ROOT/preparation.json" ]]; then
    "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/prepare_cohort_source.py" \
      --mimic-root "$MIMIC_III_1_4_ROOT" \
      --source-root "$STEPWISE_SOURCE_ROOT"
  fi

  "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/export_ihm.py" \
    --source-root "$STEPWISE_SOURCE_ROOT" \
    --output-dir "$TASK_DIR"

  "$WORKSPACE_ROOT/lightgbm_code/.venv/bin/python" -m \
    missing_xgboost_lightgbm.validate_task "$TASK_DIR"
  ;;
*)
  echo "unknown task: $1" >&2
  echo "tasks: $TASKS" >&2
  exit 2
  ;;
esac

echo "Wrote $TASK_DIR"
