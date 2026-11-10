#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"

: "${FUSEMOE_SOURCE_DATA_ROOT:?Set FUSEMOE_SOURCE_DATA_ROOT to the released split-pickle directory}"
INPUT_DIR="$PAPER_ROOT/.export_assets/reproduction/input"

if [[ ! -f "$INPUT_DIR/manifest.json" ]]; then
  echo "Missing prepared input manifest: $INPUT_DIR/manifest.json" >&2
  exit 1
fi

"$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/verify_reproduction_inputs.py" \
  --source-root "$FUSEMOE_SOURCE_DATA_ROOT" \
  --input-dir "$INPUT_DIR"

echo "Verified $INPUT_DIR"
