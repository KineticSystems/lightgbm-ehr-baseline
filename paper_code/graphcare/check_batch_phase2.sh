#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_phase2_common.sh"
resolve_task "${1:-}"

if [[ ! -f "$TASK_DIR/manifest.json" ]]; then
  echo "Missing exported task: $TASK_DIR" >&2
  exit 1
fi

run_capped "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/check_batch_phase2.py" \
  --task "$TASK" --task-dir "$TASK_DIR"

echo "Batch-verified $TASK_DIR"
