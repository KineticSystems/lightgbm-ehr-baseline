#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"

TASKS="mortality_48h los_96h_survival_at_48h"
if [[ $# -ne 1 ]]; then
  echo "usage: $0 TASK" >&2
  echo "tasks: $TASKS" >&2
  exit 2
fi

case "$1" in
mortality_48h)
  : "${MIMIC_IV_3_1_ROOT:?Set the exact MIMIC-IV 3.1 source root}"
  : "${MIMIC_IV_NOTE_2_2_ROOT:?Set the exact MIMIC-IV-Note 2.2 source root}"
  : "${MERGE_EXPORT_CACHE_ROOT:=$PAPER_ROOT/.export_assets/source_cache}"

  TASK_DIR="$WORKSPACE_ROOT/paper_tasks/merge/mortality_48h"
  if [[ -e "$TASK_DIR" ]]; then
    echo "Refusing to overwrite existing task: $TASK_DIR" >&2
    exit 1
  fi

  "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/export_source.py" \
    --mimic-root "$MIMIC_IV_3_1_ROOT" \
    --note-root "$MIMIC_IV_NOTE_2_2_ROOT" \
    --cache-root "$MERGE_EXPORT_CACHE_ROOT" \
    --output-dir "$TASK_DIR"

  "$WORKSPACE_ROOT/lightgbm_code/.venv/bin/python" -m \
    missing_xgboost_lightgbm.validate_task "$TASK_DIR"
  ;;
los_96h_survival_at_48h)
  # Never write .pyc into the frozen shared lightgbm_code tree.
  export PYTHONDONTWRITEBYTECODE=1

  : "${MIMIC_IV_3_1_ROOT:?Set the exact MIMIC-IV 3.1 source root}"
  : "${MIMIC_IV_NOTE_2_2_ROOT:?Set the exact MIMIC-IV-Note 2.2 source root}"
  : "${MERGE_EXPORT_CACHE_ROOT:=$PAPER_ROOT/.export_assets/source_cache}"

  TASK_DIR="$WORKSPACE_ROOT/paper_tasks/merge/los_96h_survival_at_48h"
  if [[ -e "$TASK_DIR" ]]; then
    echo "Refusing to overwrite existing task: $TASK_DIR" >&2
    exit 1
  fi

  # The guard renices its child by +10; starting at 5 puts the heavy phase at 15
  # so it yields to any concurrent shared fit.
  MXGB_THREADS=3 \
    nice -n 5 "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/export_source.py" \
    --task los \
    --mimic-root "$MIMIC_IV_3_1_ROOT" \
    --note-root "$MIMIC_IV_NOTE_2_2_ROOT" \
    --cache-root "$MERGE_EXPORT_CACHE_ROOT" \
    --output-dir "$TASK_DIR"

  MXGB_THREADS=3 nice -n 5 \
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
