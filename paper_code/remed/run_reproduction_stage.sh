#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 encoder|remed SEED [GPU]" >&2
  exit 2
fi

STAGE=$1
SEED=$2
GPU=${3:-0}
case "$STAGE" in
  encoder|remed) ;;
  *) echo "stage must be encoder or remed" >&2; exit 2 ;;
esac
case "$SEED" in
  2020|2021|2022) ;;
  *) echo "seed must be one of 2020, 2021, or 2022" >&2; exit 2 ;;
esac

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
REPRO_ROOT="$PAPER_ROOT/.export_assets/reproduction"
GPU_ENV="$REPRO_ROOT/gpu_venv"
AUDIT_ROOT="$REPRO_ROOT/audit"
SAVE_ROOT="$REPRO_ROOT/runs"
RUN_DIR="$SAVE_ROOT/${STAGE}_${SEED}"
CONSOLE_LOG="$SAVE_ROOT/${STAGE}_${SEED}.console.log"
STAGE_REPORT="$RUN_DIR/stage_report.json"

if [[ ! -x "$GPU_ENV/bin/accelerate" ]]; then
  echo "Blackwell REMed environment has not completed" >&2
  exit 1
fi
for report in "$AUDIT_ROOT/gpu_environment.json" "$AUDIT_ROOT/input_shim.json"; do
  if [[ ! -s "$report" ]]; then
    echo "Missing required REMed gate: $report" >&2
    exit 1
  fi
done

if [[ -s "$STAGE_REPORT" ]]; then
  env REPORT="$STAGE_REPORT" "$GPU_ENV/bin/python" -c \
    'import json,os; r=json.load(open(os.environ["REPORT"])); assert r["status"] == "terminal_source_stage"'
  echo "REMed $STAGE seed $SEED is already terminal"
  exit 0
fi

if [[ "$STAGE" == "encoder" ]]; then
  GATE_REPORT="$REPRO_ROOT/smoke/small/report.json"
else
  GATE_REPORT="$SAVE_ROOT/encoder_${SEED}/exact_maximum_batch_report.json"
fi
if [[ ! -s "$GATE_REPORT" ]]; then
  echo "Missing required stage gate: $GATE_REPORT" >&2
  exit 1
fi
env REPORT="$GATE_REPORT" "$GPU_ENV/bin/python" -c \
  'import json,os; r=json.load(open(os.environ["REPORT"])); assert r["status"] == "passed"'

RUN_SOURCE=1
if [[ -d "$RUN_DIR" || -e "$CONSOLE_LOG" ]]; then
  if [[ -d "$RUN_DIR/checkpoint" ]]; then
    echo "Resuming interrupted REMed $STAGE seed $SEED from released Accelerator state"
  elif [[ -s "$RUN_DIR/checkpoint_best.pt" && -s "$CONSOLE_LOG" ]]; then
    if [[ "$STAGE" == "encoder" && ! -s "$RUN_DIR/mimiciv_encoded.h5" ]]; then
      echo "Inconsistent terminal encoder artifacts" >&2
      exit 1
    fi
    echo "Recovering terminal REMed $STAGE seed $SEED stage report"
    RUN_SOURCE=0
  else
    echo "Inconsistent existing REMed $STAGE seed $SEED artifacts" >&2
    exit 1
  fi
fi

if [[ "$RUN_SOURCE" -eq 1 ]]; then
  mkdir -p "$SAVE_ROOT"
  ATTEMPT_COUNT=$(rg -c '^=== source attempt ' "$CONSOLE_LOG" 2>/dev/null || true)
  ATTEMPT_COUNT=${ATTEMPT_COUNT:-0}
  ATTEMPT=$((ATTEMPT_COUNT + 1))
  echo "=== source attempt $ATTEMPT: $STAGE seed $SEED ===" | tee -a "$CONSOLE_LOG"
  cd "$PAPER_ROOT"
  shared_args=(
    --config_file config/single.json
    --num_processes 1
    --gpu_ids 0
    main.py
    --src_data mimiciv
    --input_path "$REPRO_ROOT/input"
    --save_dir "$SAVE_ROOT"
  )
  if [[ "$STAGE" == "encoder" ]]; then
    stage_args=(
      --train_type short
      --time -99999
      --pred_time 48
      --lr 5e-5
      --random_sample
      --encode_events
      --resume_name "encoder_${SEED}"
    )
  else
    stage_args=(
      --train_type remed
      --time -99999
      --pred_time 48
      --lr 1e-5
      --scorer
      --scorer_use_time
      --pretrained "encoder_${SEED}"
      --no_pretrained_checkpoint
      --resume_name "remed_${SEED}"
    )
  fi
  env CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
    "$GPU_ENV/bin/accelerate" launch \
    "${shared_args[@]}" "${stage_args[@]}" 2>&1 | tee -a "$CONSOLE_LOG"
fi

ATTEMPTS=$(rg -c '^=== source attempt ' "$CONSOLE_LOG" || true)
ATTEMPTS=${ATTEMPTS:-0}
cd "$PAPER_ROOT"
"$GPU_ENV/bin/python" record_reproduction_stage.py \
  --stage "$STAGE" \
  --seed "$SEED" \
  --save-root "$SAVE_ROOT" \
  --console-log "$CONSOLE_LOG" \
  --environment-report "$AUDIT_ROOT/gpu_environment.json" \
  --input-report "$AUDIT_ROOT/input_shim.json" \
  --output "$STAGE_REPORT" \
  --attempts "$ATTEMPTS"
