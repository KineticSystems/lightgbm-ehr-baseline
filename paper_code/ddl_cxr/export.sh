#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"

: "${MIMIC_IV_2_0_ROOT:?MIMIC_IV_2_0_ROOT must identify exact MIMIC-IV 2.0}"
: "${MIMIC_CXR_JPG_2_0_0_ROOT:?MIMIC_CXR_JPG_2_0_0_ROOT must identify the verified exact 2.0.0 payload}"
: "${DDL_CXR_EXPORT_WORK_ROOT:=$PAPER_ROOT/.export_assets/author_source}"

TASK_DIR="$WORKSPACE_ROOT/paper_tasks/ddl_cxr/mortality_48h"
if [[ -e "$TASK_DIR" ]]; then
  echo "Refusing to overwrite existing task: $TASK_DIR" >&2
  exit 1
fi

"$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/prepare_cohort_source.py" \
  --mimic-root "$MIMIC_IV_2_0_ROOT" \
  --cxr-root "$MIMIC_CXR_JPG_2_0_0_ROOT" \
  --work-root "$DDL_CXR_EXPORT_WORK_ROOT"

"$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/export_source.py" \
  --work-root "$DDL_CXR_EXPORT_WORK_ROOT" \
  --output-dir "$TASK_DIR"

"$WORKSPACE_ROOT/lightgbm_code/.venv/bin/python" -m \
  missing_xgboost_lightgbm.validate_task "$TASK_DIR"

echo "Wrote $TASK_DIR"
