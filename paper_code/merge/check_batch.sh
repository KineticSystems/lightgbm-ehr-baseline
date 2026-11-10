#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"

TASKS="mortality_48h los_96h_survival_at_48h"
if [[ $# -ne 1 ]]; then
  echo "usage: $0 TASK" >&2
  echo "tasks: $TASKS" >&2
  exit 2
fi

case "$1" in
mortality_48h)
  : "${MIMIC_IV_3_1_ROOT:?Set the exact MIMIC-IV 3.1 source root}"
  : "${MIMIC_IV_NOTE_2_2_ROOT:?Set the exact MIMIC-IV-Note 2.2 source root}"
  : "${MERGE_EXPORT_CACHE_ROOT:=$PAPER_ROOT/.export_assets/source_cache}"
  TASK_DIR="$WORKSPACE_ROOT/paper_tasks/merge/mortality_48h"
  MANIFEST="$TASK_DIR/manifest.json"
  if [[ ! -f "$MANIFEST" ]]; then
    echo "Missing exported task: $TASK_DIR" >&2
    exit 1
  fi

  BEFORE=$(sha256sum "$MANIFEST" | cut -d' ' -f1)
  mkdir -p "$PAPER_ROOT/.export_assets/matplotlib"
  export MPLCONFIGDIR="$PAPER_ROOT/.export_assets/matplotlib"
  "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/verify_full_batch.py" \
    --mimic-root "$MIMIC_IV_3_1_ROOT" \
    --note-root "$MIMIC_IV_NOTE_2_2_ROOT" \
    --cache-root "$MERGE_EXPORT_CACHE_ROOT" \
    --task-dir "$TASK_DIR"
  AFTER=$(sha256sum "$MANIFEST" | cut -d' ' -f1)
  if [[ "$BEFORE" != "$AFTER" ]]; then
    echo "Batch checker mutated manifest.json" >&2
    exit 1
  fi
  echo "Batch-verified $TASK_DIR"
  ;;
los_96h_survival_at_48h)
  # Never write .pyc into the frozen shared lightgbm_code tree.
  export PYTHONDONTWRITEBYTECODE=1

  : "${MIMIC_IV_3_1_ROOT:?Set the exact MIMIC-IV 3.1 source root}"
  : "${MIMIC_IV_NOTE_2_2_ROOT:?Set the exact MIMIC-IV-Note 2.2 source root}"
  : "${MERGE_EXPORT_CACHE_ROOT:=$PAPER_ROOT/.export_assets/source_cache}"
  TASK_DIR="$WORKSPACE_ROOT/paper_tasks/merge/los_96h_survival_at_48h"
  MANIFEST="$TASK_DIR/manifest.json"
  if [[ ! -f "$MANIFEST" ]]; then
    echo "Missing exported task: $TASK_DIR" >&2
    exit 1
  fi

  BEFORE=$(sha256sum "$MANIFEST" | cut -d' ' -f1)
  mkdir -p "$PAPER_ROOT/.export_assets/matplotlib"
  export MPLCONFIGDIR="$PAPER_ROOT/.export_assets/matplotlib"
  MXGB_THREADS=3 \
    nice -n 5 "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/verify_full_batch.py" \
    --task los \
    --mimic-root "$MIMIC_IV_3_1_ROOT" \
    --note-root "$MIMIC_IV_NOTE_2_2_ROOT" \
    --cache-root "$MERGE_EXPORT_CACHE_ROOT" \
    --task-dir "$TASK_DIR"
  AFTER=$(sha256sum "$MANIFEST" | cut -d' ' -f1)
  if [[ "$BEFORE" != "$AFTER" ]]; then
    echo "Batch checker mutated manifest.json" >&2
    exit 1
  fi
  echo "Batch-verified $TASK_DIR"
  ;;
*)
  echo "unknown task: $1" >&2
  echo "tasks: $TASKS" >&2
  exit 2
  ;;
esac
