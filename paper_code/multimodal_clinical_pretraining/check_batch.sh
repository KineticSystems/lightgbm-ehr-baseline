#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)

: "${MCP_SOURCE_ROOT:=$PAPER_ROOT/.export_assets/author_source}"
TASK_DIR="$WORKSPACE_ROOT/paper_tasks/multimodal_clinical_pretraining/mortality_48h"
if [[ ! -f "$TASK_DIR/manifest.json" ]]; then
  echo "Missing exported task: $TASK_DIR" >&2
  exit 1
fi

"$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/check_batch.py" \
  --source-root "$MCP_SOURCE_ROOT" \
  --task-dir "$TASK_DIR"

echo "Batch-verified $TASK_DIR"
