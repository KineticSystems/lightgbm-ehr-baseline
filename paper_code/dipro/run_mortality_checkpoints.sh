#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "smoke" && "$1" != "evaluate" ) ]]; then
  echo "usage: $0 smoke|evaluate" >&2
  exit 2
fi

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"
: "${MIMIC_CXR_JPG_2_0_0_ROOT:?MIMIC_CXR_JPG_2_0_0_ROOT is required}"

env TORCH_HOME="$PAPER_ROOT/.export_assets/torch_home" \
  "$PAPER_ROOT/.export_assets/gpu_venv/bin/python" \
  "$PAPER_ROOT/reproduce_mortality_checkpoints.py" "$1" \
  --mimic-cxr-root "$MIMIC_CXR_JPG_2_0_0_ROOT"
