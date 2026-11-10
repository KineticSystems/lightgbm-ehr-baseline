#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"

REPRO_ROOT="$PAPER_ROOT/.export_assets/reproduction"
DATA_ROOT="$REPRO_ROOT/data"
AUDIT_ROOT="$REPRO_ROOT/audit"
GPU_PYTHON="$REPRO_ROOT/gpu_venv/bin/python"
LOG_PATH="$AUDIT_ROOT/create_ihm_task.log"
TASK_ROOT="$DATA_ROOT/ihm"
BASE_NAME="ihm-48-cxr-notes-missingInd-standardized"

if [[ ! -x "$GPU_PYTHON" ]]; then
  echo "Missing MERGE reproduction environment; run setup_gpu_env.sh first" >&2
  exit 1
fi

for input_path in \
  "$DATA_ROOT/ts_labs_vitals.parquet" \
  "$DATA_ROOT/rad_notes_text_embeddings.parquet" \
  "$DATA_ROOT/mimic_cxr_embeddings.parquet"; do
  if [[ ! -e "$input_path" ]]; then
    echo "Missing required author input: $input_path" >&2
    exit 1
  fi
done

for output_path in \
  "$TASK_ROOT/train_${BASE_NAME}_stays.pkl" \
  "$TASK_ROOT/val_${BASE_NAME}_stays.pkl" \
  "$TASK_ROOT/test_${BASE_NAME}_stays.pkl" \
  "$TASK_ROOT/${BASE_NAME}_irg_scaler.pkl" \
  "$TASK_ROOT/${BASE_NAME}_imputed_scaler.pkl"; do
  if [[ -e "$output_path" ]]; then
    echo "Refusing to overwrite an existing author task artifact: $output_path" >&2
    exit 1
  fi
done

mkdir -p "$AUDIT_ROOT" "$TASK_ROOT"

"$GPU_PYTHON" \
  "$PAPER_ROOT/mimiciv/data_preprocess/create_ihm_task.py" \
  --mimic_iv_dir "$MIMIC_IV_3_1_ROOT" \
  --output_dir "$DATA_ROOT" \
  --restrict_hours 48 \
  --include_notes \
  --include_cxr \
  --include_missing \
  --standardize_features \
  --seed 42 2>&1 | tee "$LOG_PATH"
