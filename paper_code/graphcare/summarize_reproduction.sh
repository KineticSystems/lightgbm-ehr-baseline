#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
GPU_PYTHON="$PAPER_ROOT/.export_assets/gpu_venv/bin/python"

if [[ ! -x "$GPU_PYTHON" ]]; then
  echo "Missing GraphCare GPU environment; run $PAPER_ROOT/setup_gpu_env.sh" >&2
  exit 1
fi

"$GPU_PYTHON" \
  "$PAPER_ROOT/summarize_reproduction.py" "$@"
