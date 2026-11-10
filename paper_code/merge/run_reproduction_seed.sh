#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
REPRO_ROOT="$PAPER_ROOT/.export_assets/reproduction"
GPU_PYTHON="$REPRO_ROOT/gpu_venv/bin/python"
DATA_ROOT="$REPRO_ROOT/data/ihm"
RUS_PATH="$REPRO_ROOT/results/ihm/rus_multimodal_all_seq48_lags8_meanpool.npy"
PRIMARY_ROOT="$REPRO_ROOT/primary"
SEED=${1:?Usage: run_reproduction_seed.sh SEED [GPU]}
GPU=${2:-0}

case "$SEED" in
  0|1|2|3|4) ;;
  *) echo "MERGE primary seed must be one of 0,1,2,3,4" >&2; exit 1 ;;
esac

RUN_NAME="seed_${SEED}"
CHECKPOINT_DIR="$PRIMARY_ROOT/ihm/checkpoints/$RUN_NAME"
CHECKPOINT="$CHECKPOINT_DIR/best_multimodal_model_mimiciv.pth"
TRAIN_LOG="$PRIMARY_ROOT/logs/seed_${SEED}_train.log"
TEST_LOG="$PRIMARY_ROOT/logs/seed_${SEED}_test.log"
REPORT="$PRIMARY_ROOT/reports/seed_${SEED}.json"

for input_path in \
  "$GPU_PYTHON" \
  "$DATA_ROOT/train_ihm-48-cxr-notes-missingInd-standardized_stays.pkl" \
  "$DATA_ROOT/val_ihm-48-cxr-notes-missingInd-standardized_stays.pkl" \
  "$DATA_ROOT/test_ihm-48-cxr-notes-missingInd-standardized_stays.pkl" \
  "$RUS_PATH" \
  "$REPRO_ROOT/audit/ihm_inputs.json" \
  "$REPRO_ROOT/audit/rus_multimodal.json" \
  "$REPRO_ROOT/audit/training_dtype_boundary.json" \
  "$REPRO_ROOT/audit/training_smoke.json"; do
  if [[ ! -e "$input_path" ]]; then
    echo "Missing required verified MERGE primary input: $input_path" >&2
    exit 1
  fi
done

"$GPU_PYTHON" -c \
  'import json,sys; report=json.load(open(sys.argv[1])); assert report["status"] == sys.argv[2]' \
  "$REPRO_ROOT/audit/ihm_inputs.json" exact
"$GPU_PYTHON" -c \
  'import json,sys; report=json.load(open(sys.argv[1])); assert report["status"] == sys.argv[2]' \
  "$REPRO_ROOT/audit/rus_multimodal.json" exact
"$GPU_PYTHON" -c \
  'import json,sys; report=json.load(open(sys.argv[1])); assert report["status"] == sys.argv[2]' \
  "$REPRO_ROOT/audit/training_dtype_boundary.json" exact
"$GPU_PYTHON" -c \
  'import json,sys; report=json.load(open(sys.argv[1])); assert report["status"] == sys.argv[2]' \
  "$REPRO_ROOT/audit/training_smoke.json" passed

if [[ -e "$CHECKPOINT_DIR" || -e "$TRAIN_LOG" || -e "$TEST_LOG" || -e "$REPORT" ]]; then
  echo "Refusing to overwrite existing MERGE seed-$SEED primary artifacts" >&2
  exit 1
fi

mkdir -p "$PRIMARY_ROOT/logs" "$PRIMARY_ROOT/reports" "$REPRO_ROOT/matplotlib"

env CUDA_VISIBLE_DEVICES="$GPU" \
  MPLCONFIGDIR="$REPRO_ROOT/matplotlib" \
  TOKENIZERS_PARALLELISM=false \
  "$GPU_PYTHON" "$PAPER_ROOT/mimiciv/train_mimiciv_multimodal.py" \
  --train_data_path "$DATA_ROOT/train_ihm-48-cxr-notes-missingInd-standardized_stays.pkl" \
  --val_data_path "$DATA_ROOT/val_ihm-48-cxr-notes-missingInd-standardized_stays.pkl" \
  --rus_data_path "$RUS_PATH" \
  --task ihm \
  --truncate_from_end \
  --seq_len 48 \
  --gpu 0 \
  --lambda_u 1 \
  --lambda_r 1 \
  --lambda_s 1 \
  --lambda_load 0.02 \
  --run_name "$RUN_NAME" \
  --seed "$SEED" \
  --lr 1e-3 \
  --epochs 20 \
  --output_dir "$PRIMARY_ROOT" 2>&1 | tee "$TRAIN_LOG"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "MERGE seed-$SEED did not create a validation-selected checkpoint" >&2
  exit 1
fi

env CUDA_VISIBLE_DEVICES="$GPU" \
  MPLCONFIGDIR="$REPRO_ROOT/matplotlib" \
  TOKENIZERS_PARALLELISM=false \
  "$GPU_PYTHON" "$PAPER_ROOT/mimiciv/test_mimiciv_multimodal.py" \
  --checkpoint_path "$CHECKPOINT" \
  --test_data_path "$DATA_ROOT/test_ihm-48-cxr-notes-missingInd-standardized_stays.pkl" \
  --rus_data_path "$RUS_PATH" \
  --batch_size 512 \
  --gpu 0 \
  --save_predictions \
  --save_metrics 2>&1 | tee "$TEST_LOG"

"$GPU_PYTHON" "$PAPER_ROOT/verify_reproduction_seed.py" \
  --seed "$SEED" \
  --checkpoint "$CHECKPOINT" \
  --predictions "$CHECKPOINT_DIR/test_predictions.npz" \
  --metrics "$CHECKPOINT_DIR/test_metrics.json" \
  --test-pickle "$DATA_ROOT/test_ihm-48-cxr-notes-missingInd-standardized_stays.pkl" \
  --input-report "$REPRO_ROOT/audit/ihm_inputs.json" \
  --rus-report "$REPRO_ROOT/audit/rus_multimodal.json" \
  --train-log "$TRAIN_LOG" \
  --test-log "$TEST_LOG" \
  --output "$REPORT"
