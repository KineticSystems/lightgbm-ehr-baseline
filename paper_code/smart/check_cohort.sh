#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"

# Each outcome is three replicas, checked together, so the parameter
# names the outcome exactly as export.sh does.
TASKS="mortality decompensation"
if [[ $# -ne 1 ]]; then
  echo "usage: $0 TASK" >&2
  echo "tasks: $TASKS" >&2
  exit 2
fi

case "$1" in
mortality)
  : "${MIMIC_III_1_4_ROOT:?Exact MIMIC-III 1.4 is required}"
  : "${SMART_EXPORT_CACHE_ROOT:=$PAPER_ROOT/.export_assets/source}"

  TASK_ROOT="$WORKSPACE_ROOT/paper_tasks/smart"
  for replica in 01 02 03; do
    if [[ ! -f "$TASK_ROOT/mortality_48h_replica_$replica/manifest.json" ]]; then
      echo "Missing exported replica: $TASK_ROOT/mortality_48h_replica_$replica/manifest.json (run ./export.sh first)" >&2
      exit 1
    fi
  done

  "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/check_cohort.py" \
    --mimic-root "$MIMIC_III_1_4_ROOT" \
    --cache-root "$SMART_EXPORT_CACHE_ROOT" \
    --task-root "$TASK_ROOT"
  echo "Cohort-verified three SMART mortality replicas below $TASK_ROOT"
  ;;
decompensation)
  : "${MIMIC_III_1_4_ROOT:?Exact MIMIC-III 1.4 is required}"
  : "${SMART_DECOMP_CACHE_ROOT:=/var/tmp/missing-xgboost-smart-decomp-${UID}}"

  "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/check_cohort_decompensation.py" \
    --mimic-root "$MIMIC_III_1_4_ROOT" \
    --cache-root "$SMART_DECOMP_CACHE_ROOT" \
    --task-root "$WORKSPACE_ROOT/paper_tasks/smart"
  ;;
*)
  echo "unknown task: $1" >&2
  echo "tasks: $TASKS" >&2
  exit 2
  ;;
esac
