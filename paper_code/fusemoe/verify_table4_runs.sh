#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
REPRO_ROOT="$PAPER_ROOT/.export_assets/reproduction"
GPU_PYTHON="$REPRO_ROOT/gpu_venv/bin/python"

if [[ ! -x "$GPU_PYTHON" ]]; then
  echo "Missing reproduction environment" >&2
  exit 1
fi

"$GPU_PYTHON" "$PAPER_ROOT/verify_table4_runs.py" \
  --primary-dir "$REPRO_ROOT/primary" \
  --input-dir "$REPRO_ROOT/input" \
  --output "$WORKSPACE_ROOT/paper_reproductions/fusemoe/result.json"
