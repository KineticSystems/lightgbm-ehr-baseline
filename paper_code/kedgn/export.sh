#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"

: "${MIMIC_III_1_4_ROOT:?Exact MIMIC-III 1.4 is required}"
: "${KEDGN_EXPORT_CACHE_ROOT:=$PAPER_ROOT/.export_assets/source}"
: "${KEDGN_COMPATIBLE_BENCHMARK_ROOT:=$WORKSPACE_ROOT/paper_code/smart/.export_assets/source}"

TASK_ROOT="$WORKSPACE_ROOT/paper_tasks/kedgn/mortality_48h"
if [[ -e "$TASK_ROOT" ]]; then
  echo "Refusing to overwrite existing task: $TASK_ROOT" >&2
  exit 1
fi

"$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/prepare_cohort_source.py" \
  --mimic-root "$MIMIC_III_1_4_ROOT" \
  --cache-root "$KEDGN_EXPORT_CACHE_ROOT" \
  --compatible-cache "$KEDGN_COMPATIBLE_BENCHMARK_ROOT"

"$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/export_source.py" \
  --cache-root "$KEDGN_EXPORT_CACHE_ROOT" \
  --output-root "$TASK_ROOT"

"$WORKSPACE_ROOT/lightgbm_code/.venv/bin/python" -m \
  missing_xgboost_lightgbm.validate_task "$TASK_ROOT"

echo "Wrote KEDGN mortality task to $TASK_ROOT"
