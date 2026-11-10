#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)

PYTHONHASHSEED=42 \
  "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/check_muse_plus_inputs.py"
