#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"

: "${MIMIC_III_1_4_ROOT:?Set the exact MIMIC-III v1.4 root}"
: "${MCP_LIST_ROOT:=$PAPER_ROOT/.export_assets/google_drive_1wB-4kUrNB9cHqD1qvR5fFEOaIUXmXTxI/in-hospital-mortality}"
: "${MCP_PRODUCER_ROOT:=$PAPER_ROOT/.export_assets/mimic3-benchmarks-ea0314c7}"
: "${MCP_SOURCE_ROOT:=$PAPER_ROOT/.export_assets/author_source}"

if [[ ! -f "$MCP_PRODUCER_ROOT/mimic3benchmark/mimic3csv.py" ]]; then
  echo "Missing hash-pinned YerevaNN ea0314c7 source tree: $MCP_PRODUCER_ROOT" >&2
  exit 1
fi
if [[ ! -f "$MCP_SOURCE_ROOT/source_build.json" ]]; then
  "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/prepare_cohort_source.py" \
    --mimic-root "$MIMIC_III_1_4_ROOT" \
    --list-root "$MCP_LIST_ROOT" \
    --producer-root "$MCP_PRODUCER_ROOT" \
    --output-root "$MCP_SOURCE_ROOT"
fi

TASK_DIR="$WORKSPACE_ROOT/paper_tasks/multimodal_clinical_pretraining/mortality_48h"
if [[ -e "$TASK_DIR" ]]; then
  echo "Refusing to overwrite existing task: $TASK_DIR" >&2
  exit 1
fi

"$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/export_source.py" \
  --source-root "$MCP_SOURCE_ROOT" \
  --output-dir "$TASK_DIR"

"$WORKSPACE_ROOT/lightgbm_code/.venv/bin/python" -m \
  missing_xgboost_lightgbm.validate_task "$TASK_DIR"

echo "Wrote $TASK_DIR"
