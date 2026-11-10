#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"

: "${MIMIC_III_1_4_ROOT:?Set the exact MIMIC-III v1.4 root}"
: "${MCP_LIST_ROOT:=$PAPER_ROOT/.export_assets/google_drive_1wB-4kUrNB9cHqD1qvR5fFEOaIUXmXTxI/in-hospital-mortality}"
: "${MCP_PRODUCER_ROOT:=$PAPER_ROOT/.export_assets/mimic3-benchmarks-ea0314c7}"
: "${MCP_SOURCE_ROOT:=$PAPER_ROOT/.export_assets/author_source}"

TASK_DIR="$WORKSPACE_ROOT/paper_tasks/multimodal_clinical_pretraining/mortality_48h"
if [[ ! -f "$TASK_DIR/manifest.json" ]]; then
  echo "Missing exported task: $TASK_DIR" >&2
  exit 1
fi

"$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/check_cohort.py" \
  --mimic-root "$MIMIC_III_1_4_ROOT" \
  --list-root "$MCP_LIST_ROOT" \
  --producer-root "$MCP_PRODUCER_ROOT" \
  --source-root "$MCP_SOURCE_ROOT" \
  --task-dir "$TASK_DIR"

echo "Cohort-verified $TASK_DIR"
