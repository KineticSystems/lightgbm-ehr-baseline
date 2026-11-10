#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"

TASKS="mortality_90d_post_discharge readmission_15d_post_discharge"
if [[ $# -ne 1 ]]; then
  echo "usage: $0 TASK" >&2
  echo "tasks: $TASKS" >&2
  exit 2
fi

case "$1" in
mortality_90d_post_discharge)
  : "${MIMIC_IV_2_0_ROOT:?Exact MIMIC-IV 2.0 is required}"
  : "${MIMIC_IV_NOTE_2_2_ROOT:?Exact MIMIC-IV-Note 2.2 is required}"

  TASK_DIR="$WORKSPACE_ROOT/paper_tasks/muse/mortality_90d_post_discharge"
  CACHE_ROOT="$PAPER_ROOT/.export_assets/source_cache"
  if [[ -e "$TASK_DIR" ]]; then
    echo "Refusing to overwrite existing task: $TASK_DIR" >&2
    exit 1
  fi

  PYTHONHASHSEED=42 \
    "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/export_source.py" \
    --mimic-root "$MIMIC_IV_2_0_ROOT" \
    --note-root "$MIMIC_IV_NOTE_2_2_ROOT" \
    --cache-root "$CACHE_ROOT" \
    --output-dir "$TASK_DIR"

  "$WORKSPACE_ROOT/lightgbm_code/.venv/bin/python" -m \
    missing_xgboost_lightgbm.validate_task "$TASK_DIR"
  ;;
readmission_15d_post_discharge)
  source "$PAPER_ROOT/_readmission_limits.sh"

  : "${MIMIC_IV_2_0_ROOT:?Exact MIMIC-IV 2.0 is required}"
  : "${MIMIC_IV_NOTE_2_2_ROOT:?Exact MIMIC-IV-Note 2.2 is required}"

  TASK_DIR="$WORKSPACE_ROOT/paper_tasks/muse/readmission_15d_post_discharge"
  CACHE_ROOT="$PAPER_ROOT/.export_assets/source_cache"
  # paper_code/muse/.venv is unusable: its CPython 3.8.20 interpreter under
  # /tmp was deleted. The paper's own gpu_venv carries the same NumPy/pandas/
  # Polars/PyArrow pins recorded for the sealed mortality export.
  PYTHON="${MUSE_PYTHON:-$PAPER_ROOT/.export_assets/gpu_venv/bin/python}"
  if [[ -e "$TASK_DIR" ]]; then
    echo "Refusing to overwrite existing task: $TASK_DIR" >&2
    exit 1
  fi

  PYTHONHASHSEED=42 muse_run "$WORKSPACE_ROOT" \
    "$PYTHON" "$PAPER_ROOT/export_readmission.py" \
    --mimic-root "$MIMIC_IV_2_0_ROOT" \
    --note-root "$MIMIC_IV_NOTE_2_2_ROOT" \
    --cache-root "$CACHE_ROOT" \
    --output-dir "$TASK_DIR"

  muse_run "$WORKSPACE_ROOT" \
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
