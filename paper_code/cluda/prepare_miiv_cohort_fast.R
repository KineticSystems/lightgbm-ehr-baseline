# Opt-in candidate wrapper around the unchanged canonical R producer.

.cluda_fast_args <- commandArgs(trailingOnly = TRUE)
if (length(.cluda_fast_args) != 2L) {
  stop("usage: prepare_miiv_cohort_fast.R RICU_DATA_PATH RELEASED_CSV_ROOT")
}
.cluda_fast_paper_root <- normalizePath(
  dirname(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[[1]]))
)
.cluda_canonical_wrapper <- file.path(
  .cluda_fast_paper_root, "prepare_miiv_cohort.R"
)
.cluda_upstream_process <- file.path(
  .cluda_fast_paper_root, "load_ICU_datasets", "process_miiv.R"
)
source(
  file.path(.cluda_fast_paper_root, "cluda_los_filter_fast.R"),
  local = .GlobalEnv
)
source(
  file.path(.cluda_fast_paper_root, "cluda_split_rng_compat.R"),
  local = .GlobalEnv
)

.cluda_expected_wrapper_sha256 <-
  "adc344fd4ae75227f55ca1390dc0aa4af7f0088a340bfe43dd86c9d59f7adfbb"
if (!identical(
  cluda_sha256_file(.cluda_canonical_wrapper),
  .cluda_expected_wrapper_sha256
)) {
  stop("canonical prepare_miiv_cohort.R changed; fast interception is unsafe")
}

.cluda_real_sys_source <- base::sys.source
.cluda_intercept_count <- 0L
sys.source <- function(file, envir = baseenv(), ...) {
  requested <- normalizePath(file, mustWork = TRUE)
  if (identical(requested, normalizePath(.cluda_upstream_process))) {
    .cluda_intercept_count <<- .cluda_intercept_count + 1L
    if (.cluda_intercept_count != 1L) {
      stop("process_miiv.R was intercepted more than once")
    }
    fast_derived <- tempfile(fileext = ".R")
    paper_derived <- tempfile(fileext = ".R")
    on.exit(unlink(c(fast_derived, paper_derived)), add = TRUE)
    .cluda_los_script_seal <- cluda_write_fast_process_script(
      .cluda_upstream_process, fast_derived
    )
    .cluda_split_script_seal <- cluda_write_paper_split_process_script(
      fast_derived, paper_derived, .cluda_upstream_process
    )
    .cluda_fast_script_seal <<- c(
      .cluda_los_script_seal,
      list(split_rng_transform = .cluda_split_script_seal)
    )
    return(.cluda_real_sys_source(paper_derived, envir = envir, ...))
  }
  .cluda_real_sys_source(file, envir = envir, ...)
}

# commandArgs() still identifies this file in the same paper directory, so the
# canonical wrapper resolves all paths and arguments exactly as in its normal
# execution. Only its single sys.source(process_miiv.R) call is intercepted.
source(.cluda_canonical_wrapper, local = .GlobalEnv)
if (.cluda_intercept_count != 1L ||
    !exists(".cluda_los_fast_audit") ||
    !exists(".cluda_split_rng_audit")) {
  stop("canonical wrapper did not execute both compatibility boundaries")
}
rm(sys.source, envir = .GlobalEnv)

.cluda_fast_record <- c(
  .cluda_fast_script_seal,
  list(
    filter_audit = .cluda_los_fast_audit,
    split_rng_audit = .cluda_split_rng_audit,
    invariants = c(
      "complete finite ts_data stay_id and charttime",
      "complete finite icustays_df stay_id and hadm_id comparison vectors",
      "exactly one icustays_df row per selected stay_id",
      "exactly one finite LOS row per selected hadm_id",
      "stable match-indexed one-pass logical subset"
    )
  )
)
jsonlite::write_json(
  .cluda_fast_record,
  file.path(output_root, "los_fast_path.json"),
  auto_unbox = TRUE,
  pretty = TRUE
)
