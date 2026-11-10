#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"

TASK_ROOT="$WORKSPACE_ROOT/paper_tasks/remed"
TASKS="los_7d_at_48h mortality_14d_at_48h readmission_at_48h"
if [[ $# -ne 1 ]]; then
  echo "usage: $0 TASK" >&2
  echo "tasks: $TASKS" >&2
  exit 2
fi

case "$1" in
los_7d_at_48h)
  : "${MIMIC_IV_2_0_ROOT:?Set MIMIC_IV_2_0_ROOT to exact MIMIC-IV v2.0}"
  : "${MIMIC_IV_ED_2_0_ROOT:?Set MIMIC_IV_ED_2_0_ROOT to exact MIMIC-IV-ED v2.0}"
  if [[ ! -f "$MIMIC_IV_2_0_ROOT/SHA256SUMS.txt" ]]; then
    echo "Exact MIMIC-IV v2.0 release/manifest is unavailable: $MIMIC_IV_2_0_ROOT" >&2
    exit 1
  fi
  if [[ ! -f "$MIMIC_IV_ED_2_0_ROOT/SHA256SUMS.txt" || ! -d "$MIMIC_IV_ED_2_0_ROOT/ed" ]]; then
    echo "Exact MIMIC-IV-ED v2.0 release/manifest is unavailable: $MIMIC_IV_ED_2_0_ROOT" >&2
    exit 1
  fi

  PYTHON="$PAPER_ROOT/.venv/bin/python"
  JDK_ROOT="${REMED_JDK_ROOT:-$PAPER_ROOT/.venv/temurin-21.0.12.1}"
  ARTIFACT_ROOT="${REMED_ARTIFACT_ROOT:-$PAPER_ROOT/.export_assets/external_artifacts}"
  SNUB_ROOT="${REMED_SNUB_ROOT:-$ARTIFACT_ROOT/snub_bab6c07/integrated-ehr-pipeline-bab6c07acce4a617153fba7da4f68bc32d20652f}"
  MAPPING_ROOT="${REMED_MAPPING_ROOT:-$ARTIFACT_ROOT/mappings}"
  TOKENIZER_ROOT="${REMED_TOKENIZER_ROOT:-$ARTIFACT_ROOT/Bio_ClinicalBERT/paper_era_9b5e0380b37eac696b3ff68b5f319c554523971f}"
  WORK_ROOT="$PAPER_ROOT/.export_work"
  SOURCE_ROOT="$WORK_ROOT/source_48h"
  TASK_DIR="$WORKSPACE_ROOT/paper_tasks/remed/los_7d_at_48h"
  SPARK_PYTHONPATH="$SNUB_ROOT"
  if [[ -n "${PYTHONPATH:-}" ]]; then
    SPARK_PYTHONPATH="$SPARK_PYTHONPATH:$PYTHONPATH"
  fi

  if [[ -e "$TASK_DIR" ]]; then
    echo "Refusing to overwrite existing task: $TASK_DIR" >&2
    exit 1
  fi
  if [[ ! -x "$JDK_ROOT/bin/java" ]]; then
    echo "Paper-local full Temurin 21 runtime is unavailable: $JDK_ROOT" >&2
    exit 1
  fi

  MXGB_TASKS_MAX=512 \
    env JAVA_HOME="$JDK_ROOT" \
    PYSPARK_PYTHON="$PYTHON" \
    PYSPARK_DRIVER_PYTHON="$PYTHON" \
    PYTHONPATH="$SPARK_PYTHONPATH" \
    JAVA_TOOL_OPTIONS="-XX:ActiveProcessorCount=4 -XX:ParallelGCThreads=2 -XX:ConcGCThreads=1 -XX:CICompilerCount=2" \
    "$PYTHON" "$PAPER_ROOT/prepare_cohort_source.py" \
    --mimic-root "$MIMIC_IV_2_0_ROOT" \
    --ed-root "$MIMIC_IV_ED_2_0_ROOT" \
    --snub-root "$SNUB_ROOT" \
    --mapping-root "$MAPPING_ROOT" \
    --tokenizer-root "$TOKENIZER_ROOT" \
    --java-root "$JDK_ROOT" \
    --source-root "$SOURCE_ROOT" \
    --work-root "$WORK_ROOT" \
    --threads 4

  "$PYTHON" "$PAPER_ROOT/export_source.py" \
    --source-root "$SOURCE_ROOT" \
    --output-dir "$TASK_DIR"

  "$WORKSPACE_ROOT/lightgbm_code/.venv/bin/python" -m \
    missing_xgboost_lightgbm.validate_task "$TASK_DIR"
  echo "Wrote $TASK_DIR"
  ;;
mortality_14d_at_48h|readmission_at_48h)
  # Sibling outcomes over the same cohort: same events, different label.
  # They reuse the base task's already validated event stream rather
  # than rebuilding it, so export los_7d_at_48h first.
  BASE="$TASK_ROOT/los_7d_at_48h"
  if [[ ! -f "$BASE/manifest.json" ]]; then
    echo "export los_7d_at_48h first: $1 derives from it" >&2
    exit 1
  fi
  "$PAPER_ROOT/.venv/bin/python" "$PAPER_ROOT/derive_outcomes.py" \
    --base "$BASE" --output-root "$TASK_ROOT" --only "$1"
  "$WORKSPACE_ROOT/lightgbm_code/.venv/bin/python" -m \
    missing_xgboost_lightgbm.validate_task "$TASK_ROOT/$1"
  echo "Wrote $TASK_ROOT/$1"
  ;;
*)
  echo "unknown task: $1" >&2
  echo "tasks: $TASKS" >&2
  exit 2
  ;;
esac
