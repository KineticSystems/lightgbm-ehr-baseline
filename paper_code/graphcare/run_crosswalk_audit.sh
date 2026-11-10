#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"
: "${MIMIC_III_1_4_ROOT:?Set the exact MIMIC-III 1.4 source root}"

"$WORKSPACE_ROOT/lightgbm_code/.venv/bin/python" \
  "$PAPER_ROOT/investigate_crosswalks.py" \
  --mimic-root "$MIMIC_III_1_4_ROOT" \
  --output "$WORKSPACE_ROOT/paper_results/graphcare/crosswalk_audit.json" \
  "$@"
