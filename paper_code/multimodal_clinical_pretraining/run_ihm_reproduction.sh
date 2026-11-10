#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)

cd "$PAPER_ROOT"
exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  "$PAPER_ROOT/.export_assets/gpu_venv/bin/python" \
  "$PAPER_ROOT/reproduce_ihm.py" "$@"
