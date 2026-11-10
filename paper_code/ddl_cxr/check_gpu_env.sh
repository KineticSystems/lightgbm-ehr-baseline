#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
GPU_ENV="$PAPER_ROOT/.export_assets/reproduction/gpu_venv"
OUTPUT="$PAPER_ROOT/.export_assets/reproduction/environment.json"

if [[ ! -x "$GPU_ENV/bin/python" ]]; then
  echo "Missing GPU environment; run $PAPER_ROOT/setup_gpu_env.sh first" >&2
  exit 1
fi

cd "$PAPER_ROOT"
exec "$GPU_ENV/bin/python" "$PAPER_ROOT/check_gpu_env.py" --output "$OUTPUT"
