#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"

: "${MIMIC_CXR_JPG_2_0_0_ROOT:?MIMIC_CXR_JPG_2_0_0_ROOT must identify the verified exact 2.0.0 payload}"
: "${DDL_CXR_EXPORT_WORK_ROOT:=$PAPER_ROOT/.export_assets/author_source}"

TASK_DIR="$WORKSPACE_ROOT/paper_tasks/ddl_cxr/mortality_48h"
MANIFEST="$TASK_DIR/manifest.json"
if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing exported task: $TASK_DIR" >&2
  exit 1
fi
BEFORE=$(sha256sum "$MANIFEST" | cut -d' ' -f1)

CHECK_STATUS=0
"$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/check_cohort.py" \
  --work-root "$DDL_CXR_EXPORT_WORK_ROOT" \
  --task-dir "$TASK_DIR" \
  --cxr-root "$MIMIC_CXR_JPG_2_0_0_ROOT" || CHECK_STATUS=$?

AFTER=$(sha256sum "$MANIFEST" | cut -d' ' -f1)
if [[ "$BEFORE" != "$AFTER" ]]; then
  echo "Cohort checker mutated manifest.json" >&2
  exit 1
fi
if [[ "$CHECK_STATUS" -ne 0 ]]; then
  exit "$CHECK_STATUS"
fi
echo "Cohort-verified $TASK_DIR"
