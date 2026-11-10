#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"

R_RUNTIME_BIN="$PAPER_ROOT/.export_assets/r-env/bin"
if [[ -x "$R_RUNTIME_BIN/Rscript" ]]; then
  export PATH="$R_RUNTIME_BIN:$PATH"
fi

: "${MIMIC_IV_1_0_ROOT:?Exact MIMIC-IV 1.0 is required}"
SOURCE_ROOT="$PAPER_ROOT/.export_assets/source_cache_v1"
TASK_DIR="$WORKSPACE_ROOT/paper_tasks/cluda/age_mortality_g1_to_g2"
if [[ -e "$TASK_DIR" ]]; then
  echo "Refusing to overwrite existing task: $TASK_DIR" >&2
  exit 1
fi

"$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/prepare_cohort_source.py" \
  --mimic-root "$MIMIC_IV_1_0_ROOT" \
  --source-root "$SOURCE_ROOT"

"$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/export_source.py" \
  --source-root "$SOURCE_ROOT" \
  --output-dir "$TASK_DIR"

"$WORKSPACE_ROOT/lightgbm_code/.venv/bin/python" -m \
  missing_xgboost_lightgbm.validate_task "$TASK_DIR"

echo "Wrote $TASK_DIR"
