#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"

: "${OC_DISTILL_REFERENCE_COHORT_ROOT:?Set OC_DISTILL_REFERENCE_COHORT_ROOT to the validated exact-count cohort directory}"
: "${OC_DISTILL_AUXILIARY_ROOT:=$PAPER_ROOT/.export_assets/student_auxiliary}"
: "${OC_DISTILL_CACHE_ROOT:=$PAPER_ROOT/.export_assets/author_cache}"

TASK_DIR="$WORKSPACE_ROOT/paper_tasks/oc_distill/mortality_48h"
if [[ ! -f "$TASK_DIR/manifest.json" ]]; then
  echo "Missing exported task: $TASK_DIR" >&2
  exit 1
fi
if [[ ! -f "$OC_DISTILL_AUXILIARY_ROOT/preparation.json" ]]; then
  "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/prepare_student_auxiliary.py" \
    --mimic-root "$MIMIC_III_1_4_ROOT" \
    --output-dir "$OC_DISTILL_AUXILIARY_ROOT"
fi

"$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/verify_full_batch.py" \
  --cohort-root "$OC_DISTILL_REFERENCE_COHORT_ROOT" \
  --mimic-root "$MIMIC_III_1_4_ROOT" \
  --auxiliary-root "$OC_DISTILL_AUXILIARY_ROOT" \
  --cache-root "$OC_DISTILL_CACHE_ROOT" \
  --task-dir "$TASK_DIR"

echo "Batch-verified $TASK_DIR"
