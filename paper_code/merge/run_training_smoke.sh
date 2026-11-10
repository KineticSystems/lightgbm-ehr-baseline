#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
REPRO_ROOT="$PAPER_ROOT/.export_assets/reproduction"
GPU_PYTHON="$REPRO_ROOT/gpu_venv/bin/python"
DATA_ROOT="$REPRO_ROOT/data/ihm"
RUS_PATH="$REPRO_ROOT/results/ihm/rus_multimodal_all_seq48_lags8_meanpool.npy"
SMOKE_ROOT="$REPRO_ROOT/training_smoke"
RUN_NAME=seed0_epoch1_smoke
CHECKPOINT_DIR="$SMOKE_ROOT/ihm/checkpoints/$RUN_NAME"
CHECKPOINT="$CHECKPOINT_DIR/best_multimodal_model_mimiciv.pth"
TRAIN_LOG="$REPRO_ROOT/audit/training_smoke.log"
EVAL_LOG="$REPRO_ROOT/audit/training_smoke_validation.log"
REPORT="$REPRO_ROOT/audit/training_smoke.json"
GPU=${1:-0}

for input_path in \
  "$GPU_PYTHON" \
  "$DATA_ROOT/train_ihm-48-cxr-notes-missingInd-standardized_stays.pkl" \
  "$DATA_ROOT/val_ihm-48-cxr-notes-missingInd-standardized_stays.pkl" \
  "$RUS_PATH" \
  "$REPRO_ROOT/audit/ihm_inputs.json" \
  "$REPRO_ROOT/audit/rus_multimodal.json" \
  "$REPRO_ROOT/audit/training_dtype_boundary.json"; do
  if [[ ! -e "$input_path" ]]; then
    echo "Missing required verified smoke input: $input_path" >&2
    exit 1
  fi
done
for report_path in \
  "$REPRO_ROOT/audit/ihm_inputs.json" \
  "$REPRO_ROOT/audit/rus_multimodal.json" \
  "$REPRO_ROOT/audit/training_dtype_boundary.json"; do
  "$GPU_PYTHON" -c \
    'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "exact"' \
    "$report_path"
done
if [[ -e "$SMOKE_ROOT" || -e "$REPORT" ]]; then
  echo "Refusing to overwrite an existing MERGE training smoke" >&2
  exit 1
fi

mkdir -p "$REPRO_ROOT/audit" "$REPRO_ROOT/matplotlib"
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
  --seed 0 \
  --lr 1e-3 \
  --epochs 1 \
  --output_dir "$SMOKE_ROOT" 2>&1 | tee "$TRAIN_LOG"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Training smoke did not create its validation-selected checkpoint" >&2
  exit 1
fi

env CUDA_VISIBLE_DEVICES="$GPU" \
  MPLCONFIGDIR="$REPRO_ROOT/matplotlib" \
  TOKENIZERS_PARALLELISM=false \
  "$GPU_PYTHON" "$PAPER_ROOT/mimiciv/test_mimiciv_multimodal.py" \
  --checkpoint_path "$CHECKPOINT" \
  --test_data_path "$DATA_ROOT/val_ihm-48-cxr-notes-missingInd-standardized_stays.pkl" \
  --rus_data_path "$RUS_PATH" \
  --batch_size 512 \
  --gpu 0 \
  --save_predictions \
  --save_metrics 2>&1 | tee "$EVAL_LOG"

mv "$CHECKPOINT_DIR/test_predictions.npz" "$CHECKPOINT_DIR/validation_predictions.npz"
mv "$CHECKPOINT_DIR/test_metrics.json" "$CHECKPOINT_DIR/validation_metrics.json"

"$GPU_PYTHON" "$PAPER_ROOT/verify_training_smoke.py" \
  --checkpoint "$CHECKPOINT" \
  --predictions "$CHECKPOINT_DIR/validation_predictions.npz" \
  --metrics "$CHECKPOINT_DIR/validation_metrics.json" \
  --validation-pickle "$DATA_ROOT/val_ihm-48-cxr-notes-missingInd-standardized_stays.pkl" \
  --input-report "$REPRO_ROOT/audit/ihm_inputs.json" \
  --rus-report "$REPRO_ROOT/audit/rus_multimodal.json" \
  --train-log "$TRAIN_LOG" \
  --evaluation-log "$EVAL_LOG" \
  --output "$REPORT"
