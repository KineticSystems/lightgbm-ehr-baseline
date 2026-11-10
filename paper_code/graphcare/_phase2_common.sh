# Shared setup for the GraphCare phase-2 task chain.
# Defines $TASK, $ROOT, $TASK_DIR, $PAPER_ROOT, $WORKSPACE_ROOT and run_capped.
set -euo pipefail

PAPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)
WORKSPACE_ROOT=$(cd "$PAPER_ROOT/../.." && pwd)
source "$WORKSPACE_ROOT/workflows/data_paths.local.sh"

# One heavy process at a time under a hard 7 GiB address-space cap, deprioritised,
# with native thread pools capped at 3 on this shared 12-core workstation.
run_capped() {
  (
    ulimit -v 7340032
    exec env \
      PYTHONHASHSEED=0 \
      POLARS_MAX_THREADS=3 OMP_NUM_THREADS=3 RAYON_NUM_THREADS=3 \
      MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3 \
      TOKENIZERS_PARALLELISM=false \
      nice -n 15 "$@"
  )
}

resolve_task() {
  TASK="${1:-}"
  case "$TASK" in
    readmission_15d)
      # data_prepare.py:55-64 hard-codes mimiciii/1.4.
      ROOT="${MIMIC_III_1_4_ROOT:?Set the exact MIMIC-III 1.4 source root}"
      ;;
    mimic4_subsequent_visit_mortality|mimic4_readmission_15d)
      # data_prepare.py:65-77 hard-codes mimiciv/2.0/hosp.
      ROOT="${MIMIC_IV_2_0_ROOT:?Set the exact MIMIC-IV 2.0 source root}/hosp"
      ;;
    *)
      echo "usage: <readmission_15d|mimic4_subsequent_visit_mortality|mimic4_readmission_15d>" >&2
      exit 2
      ;;
  esac
  TASK_DIR="$WORKSPACE_ROOT/paper_tasks/graphcare/$TASK"
}
