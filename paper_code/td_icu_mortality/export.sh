#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"

: "${MIMIC_IV_3_0_ROOT:?Exact MIMIC-IV 3.0 is required}"
OUTPUT_ROOT="${TD_ICU_OUTPUT_ROOT:-$WORKSPACE_ROOT/paper_tasks/td_icu_mortality}"
TASK_DIR="$OUTPUT_ROOT/mortality_28d"
if [[ -e "$TASK_DIR" ]]; then
  echo "Refusing to overwrite existing task: $TASK_DIR" >&2
  exit 1
fi

cd "$PAPER_ROOT"
"$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/export_source.py" \
  --mimic-root "$MIMIC_IV_3_0_ROOT" \
  --output-root "$OUTPUT_ROOT"

"$WORKSPACE_ROOT/lightgbm_code/.venv/bin/python" -m \
  missing_xgboost_lightgbm.validate_task "$TASK_DIR"

echo "Wrote $TASK_DIR"
