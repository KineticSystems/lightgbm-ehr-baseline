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
  export NLTK_DATA="$PAPER_ROOT/.export_assets/nltk_data"
  [[ -f "$TASK_DIR/manifest.json" ]] || { echo "Run export.sh first" >&2; exit 1; }

  "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/check_batch.py" \
    --mimic-root "$MIMIC_III_1_4_ROOT" \
    --cache-root "$PAPER_ROOT/.export_assets/source_cache" \
    --task-dir "$TASK_DIR"
  ;;
mimic4_mortality_48h_partial_ehr)
  : "${MIMIC_IV_2_2_ROOT:?Exact MIMIC-IV 2.2 is required}"
  TASK_DIR="$WORKSPACE_ROOT/paper_tasks/dpmm/mimic4_mortality_48h_partial_ehr"
  [[ -f "$TASK_DIR/manifest.json" ]] || { echo "Run export_mimic4_mortality.sh first" >&2; exit 1; }

  "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/check_batch_mimic4.py" \
    --mimic-root "$MIMIC_IV_2_2_ROOT" \
    --cache-root "$PAPER_ROOT/.export_assets/mimic4_cache" \
    --task-dir "$TASK_DIR"
  ;;
mimic4_readmission_30d_partial_ehr)
  : "${MIMIC_IV_2_2_ROOT:?Exact MIMIC-IV 2.2 is required}"
  TASK_DIR="$WORKSPACE_ROOT/paper_tasks/dpmm/mimic4_readmission_30d_partial_ehr"
  [[ -f "$TASK_DIR/manifest.json" ]] || { echo "Run export_mimic4_readmission.sh first" >&2; exit 1; }

  "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/check_batch_mimic4.py" \
    --mimic-root "$MIMIC_IV_2_2_ROOT" \
    --cache-root "$PAPER_ROOT/.export_assets/mimic4_cache" \
    --task-dir "$TASK_DIR"
  ;;
readmission_30d_partial_ehr_note)
  : "${MIMIC_III_1_4_ROOT:?Exact MIMIC-III 1.4 is required}"
  TASK_DIR="$WORKSPACE_ROOT/paper_tasks/dpmm/readmission_30d_partial_ehr_note"
  export NLTK_DATA="$PAPER_ROOT/.export_assets/nltk_data"
  [[ -f "$TASK_DIR/manifest.json" ]] || { echo "Run export_readmission.sh first" >&2; exit 1; }

  "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/check_batch_readmission.py" \
    --mimic-root "$MIMIC_III_1_4_ROOT" \
    --cache-root "$PAPER_ROOT/.export_assets/source_cache" \
    --readmission-cache "$PAPER_ROOT/.export_assets/readmission_cache" \
    --task-dir "$TASK_DIR"
  ;;
*)
  echo "unknown task: $1" >&2
  echo "tasks: $TASKS" >&2
  exit 2
  ;;
esac
