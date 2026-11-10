#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
REPRO_ROOT="$PAPER_ROOT/.export_assets/reproduction"
GPU_ENV="$REPRO_ROOT/gpu_venv"
CONSOLE_LOG="$REPRO_ROOT/audit/gpu_environment.console.log"

if [[ ! -x "$GPU_ENV/bin/python" ]]; then
  echo "GPU environment has not been installed" >&2
  exit 1
fi
if [[ -e "$REPRO_ROOT/audit/gpu_environment.json" || -e "$CONSOLE_LOG" ]]; then
  echo "Refusing to overwrite an existing GPU environment record" >&2
  exit 1
fi
mkdir -p "$REPRO_ROOT/audit" "$REPRO_ROOT/matplotlib"

cd "$PAPER_ROOT"
env MPLCONFIGDIR="$REPRO_ROOT/matplotlib" PYTHONUNBUFFERED=1 \
  "$GPU_ENV/bin/python" record_gpu_environment.py 2>&1 | tee "$CONSOLE_LOG"

test -s "$REPRO_ROOT/audit/gpu_environment.json"
test -s "$REPRO_ROOT/audit/gpu_environment_freeze.txt"
