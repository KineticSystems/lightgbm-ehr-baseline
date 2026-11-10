#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"

R_RUNTIME_BIN="$PAPER_ROOT/.export_assets/r-env/bin"
if [[ -x "$R_RUNTIME_BIN/Rscript" ]]; then
  export PATH="$R_RUNTIME_BIN:$PATH"
fi

: "${MIMIC_IV_1_0_ROOT:?Exact MIMIC-IV 1.0 is required}"
SOURCE_ROOT="$PAPER_ROOT/.export_assets/source_cache_v1"
TASK_DIR="$WORKSPACE_ROOT/paper_tasks/cluda/age_mortality_g1_to_g2"
[[ -f "$TASK_DIR/manifest.json" ]] || { echo "Run export.sh first" >&2; exit 1; }

"$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/prepare_cohort_source.py" \
  --mimic-root "$MIMIC_IV_1_0_ROOT" \
  --source-root "$SOURCE_ROOT"

"$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/check_batch.py" \
  --source-root "$SOURCE_ROOT" \
  --task-dir "$TASK_DIR"
