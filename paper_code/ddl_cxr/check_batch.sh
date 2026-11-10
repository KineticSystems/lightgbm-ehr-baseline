#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"

: "${DDL_CXR_EXPORT_WORK_ROOT:=$PAPER_ROOT/.export_assets/author_source}"

TASK_DIR="$WORKSPACE_ROOT/paper_tasks/ddl_cxr/mortality_48h"
MANIFEST="$TASK_DIR/manifest.json"
if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing exported task: $TASK_DIR" >&2
  exit 1
fi
BEFORE=$(sha256sum "$MANIFEST" | cut -d' ' -f1)

CHECK_STATUS=0
"$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/check_batch.py" \
  --work-root "$DDL_CXR_EXPORT_WORK_ROOT" \
  --task-dir "$TASK_DIR" || CHECK_STATUS=$?

AFTER=$(sha256sum "$MANIFEST" | cut -d' ' -f1)
if [[ "$BEFORE" != "$AFTER" ]]; then
  echo "Batch checker mutated manifest.json" >&2
  exit 1
fi
if [[ "$CHECK_STATUS" -ne 0 ]]; then
  exit "$CHECK_STATUS"
fi
echo "Batch-verified $TASK_DIR"
