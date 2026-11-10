#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
: "${DIPRO_EXPORT_CACHE_ROOT:=$PAPER_ROOT/.export_assets/source_cache}"
TASK_DIR="$WORKSPACE_ROOT/paper_tasks/dipro/mortality_48h"
MANIFEST="$TASK_DIR/manifest.json"
if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing exported task: $TASK_DIR" >&2
  exit 1
fi

BEFORE=$(sha256sum "$MANIFEST" | cut -d' ' -f1)
"$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/check_cohort.py" \
  --cache-root "$DIPRO_EXPORT_CACHE_ROOT" \
  --task-dir "$TASK_DIR"
AFTER=$(sha256sum "$MANIFEST" | cut -d' ' -f1)
if [[ "$BEFORE" != "$AFTER" ]]; then
  echo "Cohort checker mutated manifest.json" >&2
  exit 1
fi

echo "Cohort-verified $TASK_DIR"
