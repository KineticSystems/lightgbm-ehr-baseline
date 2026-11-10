#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
REPRO_ROOT="$PAPER_ROOT/.export_assets/reproduction"
GPU_ENV="$REPRO_ROOT/gpu_venv"
OUTPUT_DIR="$REPRO_ROOT/smoke/seed_1111"
CONSOLE_LOG="$REPRO_ROOT/smoke/seed_1111.console.log"

if [[ ! -x "$GPU_ENV/bin/python" ]]; then
  echo "GPU environment has not passed setup_gpu_env.sh" >&2
  exit 1
fi
if [[ -e "$OUTPUT_DIR" || -e "$CONSOLE_LOG" ]]; then
  echo "Refusing to overwrite an existing Step-wise smoke artifact" >&2
  exit 1
fi
mkdir -p "$REPRO_ROOT/smoke" "$REPRO_ROOT/matplotlib"

cd "$PAPER_ROOT"
env MPLCONFIGDIR="$REPRO_ROOT/matplotlib" PYTHONUNBUFFERED=1 \
  "$GPU_ENV/bin/python" smoke_reproduction.py \
  --seed 1111 \
  --output-dir "$OUTPUT_DIR" 2>&1 | tee "$CONSOLE_LOG"

test -s "$OUTPUT_DIR/smoke.json"
