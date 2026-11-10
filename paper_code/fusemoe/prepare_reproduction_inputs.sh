#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"

: "${FUSEMOE_SOURCE_DATA_ROOT:?Set FUSEMOE_SOURCE_DATA_ROOT to the released split-pickle directory}"
OUTPUT_DIR="$PAPER_ROOT/.export_assets/reproduction/input"

"$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/prepare_reproduction_inputs.py" \
  --source-root "$FUSEMOE_SOURCE_DATA_ROOT" \
  --output-dir "$OUTPUT_DIR"

echo "Prepared $OUTPUT_DIR"

