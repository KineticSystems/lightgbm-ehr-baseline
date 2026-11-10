#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"

: "${MIMIC_IV_2_0_ROOT:?Set the exact MIMIC-IV 2.0 source root}"
: "${MIMIC_CXR_JPG_2_0_0_ROOT:?Set the verified exact MIMIC-CXR-JPG 2.0.0 payload root}"
: "${DIPRO_EXPORT_CACHE_ROOT:=$PAPER_ROOT/.export_assets/source_cache}"
IMAGENOME_ROOT="$PAPER_ROOT/.export_assets/external_artifacts/physionet_chest_imagenome_1.0.0"
TASK_DIR="$WORKSPACE_ROOT/paper_tasks/dipro/mortality_48h"

if [[ -e "$TASK_DIR" ]]; then
  echo "Refusing to overwrite existing task: $TASK_DIR" >&2
  exit 1
fi

"$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/cohort_source.py" build-cohort \
  --mimic-root "$MIMIC_IV_2_0_ROOT" \
  --cxr-root "$MIMIC_CXR_JPG_2_0_0_ROOT" \
  --imagenome-root "$IMAGENOME_ROOT" \
  --cache-root "$DIPRO_EXPORT_CACHE_ROOT"

"$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/cohort_source.py" build-structured \
  --mimic-root "$MIMIC_IV_2_0_ROOT" \
  --cxr-root "$MIMIC_CXR_JPG_2_0_0_ROOT" \
  --imagenome-root "$IMAGENOME_ROOT" \
  --cache-root "$DIPRO_EXPORT_CACHE_ROOT"

"$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/export_source.py" \
  --cache-root "$DIPRO_EXPORT_CACHE_ROOT" \
  --output-dir "$TASK_DIR"

"$WORKSPACE_ROOT/lightgbm_code/.venv/bin/python" -m \
  missing_xgboost_lightgbm.validate_task "$TASK_DIR"

echo "Wrote $TASK_DIR"
