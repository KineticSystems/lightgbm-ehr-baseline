# Shared resource envelope for the MUSE readmission exporter and checkers.
#
# Every heavy step runs through the workspace resource guard, exactly as the
# sealed mortality chain does: a 7 GiB cgroup memory cap, one of the shared job
# slots, nice level 15, and 3-thread caps for the native libraries. Set
muse_run() {
  local workspace_root="$1"; shift
  export POLARS_MAX_THREADS=3 OMP_NUM_THREADS=3 RAYON_NUM_THREADS=3 \
    MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3
  MXGB_THREADS=3 \
    nice -n 15 "$@"
}
