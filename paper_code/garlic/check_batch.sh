#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"

: "${MIMIC_III_1_4_ROOT:?Set MIMIC_III_1_4_ROOT to the exact MIMIC-III v1.4 release}"
: "${GARLIC_EXPORT_CACHE:=$PAPER_ROOT/.export_assets/author_boundary.sqlite}"

TASK_DIR="$WORKSPACE_ROOT/paper_tasks/garlic/in_hospital_mortality"
if [[ ! -f "$TASK_DIR/manifest.json" ]]; then
  echo "Missing exported task: $TASK_DIR" >&2
  exit 1
fi

"$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/verify_full_batch.py" \
  --cache-path "$GARLIC_EXPORT_CACHE" \
  --task-dir "$TASK_DIR"

echo "Batch-verified $TASK_DIR"
