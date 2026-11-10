#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"

: "${FUSEMOE_SOURCE_DATA_ROOT:?Set FUSEMOE_SOURCE_DATA_ROOT to the directory containing the released split pickles}"
if [[ -z "${FUSEMOE_TOKENIZER:-}" ]]; then
  if [[ -d "$PAPER_ROOT/.export_assets/clinical_longformer_tokenizer" ]]; then
    FUSEMOE_TOKENIZER="$PAPER_ROOT/.export_assets/clinical_longformer_tokenizer"
  else
    FUSEMOE_TOKENIZER="yikuan8/Clinical-Longformer"
  fi
fi

TASK_DIR="$WORKSPACE_ROOT/paper_tasks/fusemoe/mortality_48h"
if [[ ! -f "$TASK_DIR/manifest.json" ]]; then
  echo "Missing exported task: $TASK_DIR" >&2
  exit 1
fi

"$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/verify_full_batch.py" \
  --source-root "$FUSEMOE_SOURCE_DATA_ROOT" \
  --task-dir "$TASK_DIR" \
  --tokenizer "$FUSEMOE_TOKENIZER"

echo "Batch-verified $TASK_DIR"
