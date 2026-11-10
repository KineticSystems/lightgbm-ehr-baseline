#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
OUTPUT_ROOT="$PAPER_ROOT/.export_assets/reproduction/released_policy"
SUBJECTS_ROOT="$PAPER_ROOT/.export_assets/author_source/subjects"

cd "$PAPER_ROOT"
exec "$PAPER_ROOT/.venv/bin/python" \
  "$PAPER_ROOT/prepare_released_generation_ehr.py" \
  --output-root "$OUTPUT_ROOT" \
  --subjects-root "$SUBJECTS_ROOT"
