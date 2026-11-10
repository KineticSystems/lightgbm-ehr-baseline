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
if [[ -e "$TASK_DIR" ]]; then
  echo "Refusing to overwrite existing task: $TASK_DIR" >&2
  exit 1
fi
"$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/export_source.py" \
  --source-root "$FUSEMOE_SOURCE_DATA_ROOT" \
  --mimic-root "$MIMIC_IV_2_2_ROOT" \
  --output-dir "$TASK_DIR" \
  --tokenizer "$FUSEMOE_TOKENIZER"

"$WORKSPACE_ROOT/lightgbm_code/.venv/bin/python" -m \
  missing_xgboost_lightgbm.validate_task "$TASK_DIR"

echo "Wrote $TASK_DIR"
