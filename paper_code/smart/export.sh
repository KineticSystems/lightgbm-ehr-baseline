#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"

# Each outcome is exported as three replicas, which is how the paper
# reports it; the parameter names the outcome, not one replica.
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
    target="$TASK_ROOT/mortality_48h_replica_$replica"
    if [[ -e "$target" ]]; then
      echo "Refusing to overwrite existing replica: $target" >&2
      exit 1
    fi
  done

  prepare_args=(
    --mimic-root "$MIMIC_III_1_4_ROOT"
    --cache-root "$SMART_EXPORT_CACHE_ROOT"
  )
  if [[ -n "${SMART_EXPORT_COMPATIBLE_CACHE_ROOT:-}" ]]; then
    prepare_args+=(--compatible-cache "$SMART_EXPORT_COMPATIBLE_CACHE_ROOT")
  fi

  "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/prepare_cohort_source.py" \
    "${prepare_args[@]}"

  "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/export_source.py" \
    --cache-root "$SMART_EXPORT_CACHE_ROOT" \
    --output-root "$TASK_ROOT"

  for replica in 01 02 03; do
    "$WORKSPACE_ROOT/lightgbm_code/.venv/bin/python" -m \
      missing_xgboost_lightgbm.validate_task \
      "$TASK_ROOT/mortality_48h_replica_$replica"
  done
  echo "Wrote three SMART mortality replicas below $TASK_ROOT"
  ;;
decompensation)
  : "${MIMIC_III_1_4_ROOT:?Exact MIMIC-III 1.4 is required}"
  : "${SMART_DECOMP_CACHE_ROOT:=/var/tmp/missing-xgboost-smart-decomp-${UID}}"

  TASK_ROOT="$WORKSPACE_ROOT/paper_tasks/smart"
  for replica in 01 02 03; do
    target="$TASK_ROOT/decompensation_replica_$replica"
    if [[ -e "$target" ]]; then
      echo "Refusing to overwrite existing replica: $target" >&2
      exit 1
    fi
  done

  "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/prepare_decompensation_source.py" \
    --mimic-root "$MIMIC_III_1_4_ROOT" \
    --cache-root "$SMART_DECOMP_CACHE_ROOT"

  "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/export_decompensation.py" \
    --cache-root "$SMART_DECOMP_CACHE_ROOT" \
    --output-root "$TASK_ROOT"

  for replica in 01 02 03; do
    "$WORKSPACE_ROOT/lightgbm_code/.venv/bin/python" -m \
      missing_xgboost_lightgbm.validate_task \
      "$TASK_ROOT/decompensation_replica_$replica"
  done
  echo "Wrote three SMART decompensation replicas below $TASK_ROOT"
  ;;
*)
  echo "unknown task: $1" >&2
  echo "tasks: $TASKS" >&2
  exit 2
  ;;
esac
