#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"

TASKS="mortality_48h_partial_ehr_note mimic4_mortality_48h_partial_ehr mimic4_readmission_30d_partial_ehr readmission_30d_partial_ehr_note"
if [[ $# -ne 1 ]]; then
  echo "usage: $0 TASK" >&2
  echo "tasks: $TASKS" >&2
  exit 2
fi

case "$1" in
mortality_48h_partial_ehr_note)
  : "${MIMIC_III_1_4_ROOT:?Exact MIMIC-III 1.4 is required}"

  TASK_DIR="$WORKSPACE_ROOT/paper_tasks/dpmm/mortality_48h_partial_ehr_note"
  CACHE_ROOT="$PAPER_ROOT/.export_assets/source_cache"
  export NLTK_DATA="$PAPER_ROOT/.export_assets/nltk_data"
  if [[ -e "$TASK_DIR" ]]; then
    echo "Refusing to overwrite existing task: $TASK_DIR" >&2
    exit 1
  fi

  "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/export_source.py" \
    --mimic-root "$MIMIC_III_1_4_ROOT" \
    --cache-root "$CACHE_ROOT" \
    --output-dir "$TASK_DIR"

  "$WORKSPACE_ROOT/lightgbm_code/.venv/bin/python" -m \
    missing_xgboost_lightgbm.validate_task "$TASK_DIR"
  ;;
mimic4_mortality_48h_partial_ehr)
  : "${MIMIC_IV_2_2_ROOT:?Exact MIMIC-IV 2.2 is required}"

  TASK_DIR="$WORKSPACE_ROOT/paper_tasks/dpmm/mimic4_mortality_48h_partial_ehr"
  CACHE_ROOT="$PAPER_ROOT/.export_assets/mimic4_cache"
  if [[ -e "$TASK_DIR" ]]; then
    echo "Refusing to overwrite existing task: $TASK_DIR" >&2
    exit 1
  fi

  "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/export_mimic4.py" \
    --mimic-root "$MIMIC_IV_2_2_ROOT" \
    --cache-root "$CACHE_ROOT" \
    --task in-hospital-mortality \
    --output-dir "$TASK_DIR"

  "$WORKSPACE_ROOT/lightgbm_code/.venv/bin/python" -m \
    missing_xgboost_lightgbm.validate_task "$TASK_DIR"
  ;;
mimic4_readmission_30d_partial_ehr)
  : "${MIMIC_IV_2_2_ROOT:?Exact MIMIC-IV 2.2 is required}"

  TASK_DIR="$WORKSPACE_ROOT/paper_tasks/dpmm/mimic4_readmission_30d_partial_ehr"
  CACHE_ROOT="$PAPER_ROOT/.export_assets/mimic4_cache"
  if [[ -e "$TASK_DIR" ]]; then
    echo "Refusing to overwrite existing task: $TASK_DIR" >&2
    exit 1
  fi

  "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/export_mimic4.py" \
    --mimic-root "$MIMIC_IV_2_2_ROOT" \
    --cache-root "$CACHE_ROOT" \
    --task readmission \
    --output-dir "$TASK_DIR"

  "$WORKSPACE_ROOT/lightgbm_code/.venv/bin/python" -m \
    missing_xgboost_lightgbm.validate_task "$TASK_DIR"
  ;;
readmission_30d_partial_ehr_note)
  : "${MIMIC_III_1_4_ROOT:?Exact MIMIC-III 1.4 is required}"

  TASK_DIR="$WORKSPACE_ROOT/paper_tasks/dpmm/readmission_30d_partial_ehr_note"
  CACHE_ROOT="$PAPER_ROOT/.export_assets/source_cache"
  READM_CACHE="$PAPER_ROOT/.export_assets/readmission_cache"
  export NLTK_DATA="$PAPER_ROOT/.export_assets/nltk_data"
  if [[ -e "$TASK_DIR" ]]; then
    echo "Refusing to overwrite existing task: $TASK_DIR" >&2
    exit 1
  fi

  "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/export_readmission.py" \
    --mimic-root "$MIMIC_III_1_4_ROOT" \
    --cache-root "$CACHE_ROOT" \
    --readmission-cache "$READM_CACHE" \
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
