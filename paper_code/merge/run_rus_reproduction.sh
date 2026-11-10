#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
REPRO_ROOT="$PAPER_ROOT/.export_assets/reproduction"
GPU_PYTHON="$REPRO_ROOT/gpu_venv/bin/python"
INPUT_REPORT="$REPRO_ROOT/audit/ihm_inputs.json"
TRAIN_PICKLE="$REPRO_ROOT/data/ihm/train_ihm-48-cxr-notes-missingInd-standardized_stays.pkl"
RESULT_ROOT="$REPRO_ROOT/results"
RUS_PATH="$RESULT_ROOT/ihm/rus_multimodal_all_seq48_lags8_meanpool.npy"
LOG_PATH="$REPRO_ROOT/audit/rus_multimodal.log"
GPU=${1:-0}

if [[ ! -x "$GPU_PYTHON" ]]; then
  echo "Missing MERGE reproduction environment; run setup_gpu_env.sh first" >&2
  exit 1
fi
for input_path in "$INPUT_REPORT" "$TRAIN_PICKLE"; do
  if [[ ! -e "$input_path" ]]; then
    echo "Missing required verified RUS input: $input_path" >&2
    exit 1
  fi
done
"$GPU_PYTHON" -c \
  'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "exact"' \
  "$INPUT_REPORT"
if [[ -e "$RUS_PATH" ]]; then
  echo "Refusing to overwrite the existing primary RUS artifact: $RUS_PATH" >&2
  exit 1
fi

mkdir -p "$RESULT_ROOT/ihm" "$REPRO_ROOT/audit" "$REPRO_ROOT/matplotlib"

# The released preprocessor eagerly retains dense float64 [48,D] arrays for
# every eligible training stay. Static payload accounting is roughly 21 GiB
# before the author pickle and Python containers, so reserve one near-exclusive
# guarded job while preserving the exact released implementation.
env CUDA_VISIBLE_DEVICES="$GPU" \
  MPLCONFIGDIR="$REPRO_ROOT/matplotlib" \
  TOKENIZERS_PARALLELISM=false \
  "$GPU_PYTHON" "$PAPER_ROOT/mimiciv/mimiciv_rus_multimodal.py" \
  --train_dataset_path "$TRAIN_PICKLE" \
  --task ihm \
  --seq_len 48 \
  --num_lags 8 \
  --sequence_pooling mean \
  --gpu 0 \
  --output_dir "$RESULT_ROOT" 2>&1 | tee "$LOG_PATH"

