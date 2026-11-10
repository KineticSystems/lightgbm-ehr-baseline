#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"

: "${MIMIC_III_1_4_ROOT:?Set the exact MIMIC-III 1.4 source root}"

PYTHONHASHSEED=0 \
  "$WORKSPACE_ROOT/lightgbm_code/.venv/bin/python" \
  "$PAPER_ROOT/check_raw_roundtrip.py" \
  --mimic-root "$MIMIC_III_1_4_ROOT" \
  --task-dir "$WORKSPACE_ROOT/paper_tasks/graphcare/subsequent_visit_mortality" \
  --output "$WORKSPACE_ROOT/paper_results/graphcare/raw_roundtrip_audit.json" \
  "$@"
