#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_phase2_common.sh"
resolve_task "${1:-}"

if [[ -e "$TASK_DIR" ]]; then
  echo "Refusing to overwrite existing task: $TASK_DIR" >&2
  exit 1
fi

run_capped "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/export_phase2.py" \
  --task "$TASK" --root "$ROOT" --output-dir "$TASK_DIR"

run_capped "$WORKSPACE_ROOT/lightgbm_code/.venv/bin/python" -m \
  missing_xgboost_lightgbm.validate_task "$TASK_DIR"

echo "Wrote $TASK_DIR"
