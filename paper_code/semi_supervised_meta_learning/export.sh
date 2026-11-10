#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"

TASKS="decompensation mortality_48h decompensation_sampled"
if [[ $# -ne 1 ]]; then
  echo "usage: $0 TASK" >&2
  echo "tasks: $TASKS" >&2
  exit 2
fi
TASK="$1"

: "${MIMIC_III_1_4_ROOT:?Set MIMIC_III_1_4_ROOT to exact MIMIC-III v1.4}"
: "${DECOMP_FAMILY_M3B_SOURCE_ROOT:=/var/tmp/missing-xgboost-stepwise-${UID}}"
: "${SSML_SOURCE_ROOT:=/var/tmp/missing-xgboost-decomp-family-${UID}}"
TASK_DIR="$WORKSPACE_ROOT/paper_tasks/semi_supervised_meta_learning/$TASK"
PYTHONPATH="$PAPER_ROOT:$WORKSPACE_ROOT/lightgbm_code/src"
export PYTHONPATH

if [[ -e "$TASK_DIR" ]]; then
  echo "Refusing to overwrite existing task: $TASK_DIR" >&2
  exit 1
fi

case "$TASK" in
  decompensation)
    if [[ ! -f "$SSML_SOURCE_ROOT/preparation.json" ]]; then
      "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/prepare_cohort_source.py" \
        --shared-source-root "$DECOMP_FAMILY_M3B_SOURCE_ROOT" \
        --source-root "$SSML_SOURCE_ROOT"
    fi

    "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/export_source.py" \
      --source-root "$SSML_SOURCE_ROOT" \
      --output-dir "$TASK_DIR"
    ;;
  mortality_48h)
    # The released multitask listfiles and stay files are the already sealed
    # shared mimic3-benchmarks build; nothing is rebuilt here.
    "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/export_source_ihm.py" \
      --source-root "$DECOMP_FAMILY_M3B_SOURCE_ROOT" \
      --output-dir "$TASK_DIR" \
      --batch-rows 500000
    ;;
  decompensation_sampled)
    # The 600k-landmark subset the campaign fits, drawn from the full
    # decompensation task under the shared sampling contract. It reuses that
    # task's validated events, so export decompensation first.
    BASE="$WORKSPACE_ROOT/paper_tasks/semi_supervised_meta_learning/decompensation"
    if [[ ! -f "$BASE/manifest.json" ]]; then
      echo "export decompensation first: $TASK is drawn from it" >&2
      exit 1
    fi
    "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/derive_sampled.py" \
      --base "$BASE" --output "$TASK_DIR"
    ;;
  *)
    echo "unknown task: $TASK" >&2
    echo "tasks: $TASKS" >&2
    exit 2
    ;;
esac

"$WORKSPACE_ROOT/lightgbm_code/.venv/bin/python" -m \
  missing_xgboost_lightgbm.validate_task "$TASK_DIR"

echo "Wrote $TASK_DIR"
