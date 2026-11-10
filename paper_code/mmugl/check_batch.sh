#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"

# The vendored loader imports seaborn, which pulls in matplotlib.pyplot and
# tries to select a GUI backend. This host is headless with a broken
# _tkinter, so pin the non-interactive backend before any import.
export MPLBACKEND=Agg

: "${MIMIC_III_1_4_ROOT:?Set MIMIC_III_1_4_ROOT to exact MIMIC-III v1.4}"
MAPPING_ROOT="$PAPER_ROOT/.export_assets/external_artifacts/gamenet-da695b4fc9390882f3a681c82115e81291ae6380"
TASK_DIR="$WORKSPACE_ROOT/paper_tasks/mmugl/next_visit_heart_failure"
if [[ ! -f "$TASK_DIR/manifest.json" ]]; then
  echo "Missing exported task: $TASK_DIR" >&2
  exit 1
fi

"$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/check_batch.py" \
  --mimic-root "$MIMIC_III_1_4_ROOT" \
  --mapping-root "$MAPPING_ROOT" \
  --task-dir "$TASK_DIR"

echo "Batch-verified $TASK_DIR"
