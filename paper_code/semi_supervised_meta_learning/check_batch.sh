#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)

TASKS="decompensation mortality_48h"
if [[ $# -ne 1 ]]; then
  echo "usage: $0 TASK" >&2
  echo "tasks: $TASKS" >&2
  exit 2
fi
TASK="$1"

: "${SSML_SOURCE_ROOT:=/var/tmp/missing-xgboost-decomp-family-${UID}}"
: "${DECOMP_FAMILY_M3B_SOURCE_ROOT:=/var/tmp/missing-xgboost-stepwise-${UID}}"
TASK_DIR="$WORKSPACE_ROOT/paper_tasks/semi_supervised_meta_learning/$TASK"
if [[ ! -f "$TASK_DIR/manifest.json" ]]; then
  echo "Missing exported task: $TASK_DIR (run ./export.sh first)" >&2
  exit 1
fi
MEASUREMENT="$TASK_DIR/audit/source_precision_measurement.json"
PYTHONPATH="$PAPER_ROOT:$WORKSPACE_ROOT/lightgbm_code/src"
export PYTHONPATH

case "$TASK" in
  decompensation)
    if [[ ! -f "$MEASUREMENT" ]]; then
      "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/measure_precision.py" \
        --source-root "$SSML_SOURCE_ROOT" \
        --output "$MEASUREMENT"
    fi

    "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/check_batch.py" \
      --source-root "$SSML_SOURCE_ROOT" \
      --task-dir "$TASK_DIR"

    # PyTorch reserves more virtual address space than the 7-GiB resident-memory
    # envelope.  Require the guard's zero-swap cgroup backend for this phase rather
    # than allowing the automatic RLIMIT_AS fallback to fail during import.
    "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/check_batchgen.py" \
      --source-root "$SSML_SOURCE_ROOT" \
      --task-dir "$TASK_DIR"
    ;;
  mortality_48h)
    # The released study wires no in-hospital-mortality trainer, so there is no
    # released BatchGen phase for this task: the released MultitaskReader +
    # Discretizer + normalizer model input is the last deterministic boundary.
    if [[ ! -f "$MEASUREMENT" ]]; then
      "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/measure_precision_ihm.py" \
        --source-root "$DECOMP_FAMILY_M3B_SOURCE_ROOT" \
        --output "$MEASUREMENT"
    fi

    "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/check_batch_ihm.py" \
      --source-root "$DECOMP_FAMILY_M3B_SOURCE_ROOT" \
      --task-dir "$TASK_DIR"
    ;;
  decompensation_sampled)
    # The sampled subset reuses the base task's verified events and inherits
    # its audit records, so there is nothing separate to check here.
    echo "$TASK is drawn from decompensation; check that task instead" >&2
    exit 2
    ;;
  *)
    echo "unknown task: $TASK" >&2
    echo "tasks: $TASKS" >&2
    exit 2
    ;;
esac
