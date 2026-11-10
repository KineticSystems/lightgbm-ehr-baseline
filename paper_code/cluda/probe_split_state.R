args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("usage: probe_split_state.R RICU_DATA_PATH OUTPUT_ROOT")
}

ricu_data_path <- normalizePath(args[[1]], mustWork = TRUE)
output_root <- normalizePath(args[[2]], mustWork = FALSE)
probe_rds_arg <- Sys.getenv("CLUDA_SPLIT_PROBE_RDS", unset = NA_character_)
if (is.na(probe_rds_arg) || !nzchar(probe_rds_arg)) {
  stop("CLUDA_SPLIT_PROBE_RDS must name the probe output")
}
probe_rds <- normalizePath(probe_rds_arg, mustWork = FALSE)
expected_ricu_version <- Sys.getenv(
  "CLUDA_SPLIT_PROBE_RICU_VERSION", unset = "0.5.2"
)
paper_root <- normalizePath(
  dirname(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[[1]]))
)
canonical_wrapper <- file.path(paper_root, "prepare_miiv_cohort.R")
upstream_process <- file.path(
  paper_root, "load_ICU_datasets", "process_miiv.R"
)
source(file.path(paper_root, "cluda_los_filter_fast.R"), local = .GlobalEnv)

expected_wrapper_sha256 <-
  "adc344fd4ae75227f55ca1390dc0aa4af7f0088a340bfe43dd86c9d59f7adfbb"
if (!identical(cluda_sha256_file(canonical_wrapper), expected_wrapper_sha256)) {
  stop("canonical prepare_miiv_cohort.R changed; split probe is unsafe")
}

real_sys_source <- base::sys.source
intercept_count <- 0L
sys.source <- function(file, envir = baseenv(), ...) {
  requested <- normalizePath(file, mustWork = TRUE)
  if (!identical(requested, normalizePath(upstream_process))) {
    return(real_sys_source(file, envir = envir, ...))
  }
  intercept_count <<- intercept_count + 1L
  if (intercept_count != 1L) {
    stop("process_miiv.R was intercepted more than once")
  }

  fast_process <- tempfile(fileext = ".R")
  probed_process <- tempfile(fileext = ".R")
  on.exit(unlink(c(fast_process, probed_process)), add = TRUE)
  cluda_write_fast_process_script(upstream_process, fast_process)
  lines <- readLines(fast_process, warn = FALSE)
  insert_after <- function(current, anchor, addition) {
    location <- which(current == anchor)
    if (length(location) != 1L) {
      stop(paste("probe anchor did not occur exactly once:", anchor))
    }
    append(current, addition, after = location)
  }
  lines <- insert_after(
    lines,
    "set.seed(1234)",
    ".cluda_rng_trace <- list(after_set_seed = .Random.seed)"
  )
  lines <- insert_after(
    lines,
    "cohort <- load_concepts(static_features, \"miiv\", patient_ids=icustays_df$stay_id, verbose = FALSE)",
    ".cluda_rng_trace$after_static_concepts <- .Random.seed"
  )
  lines <- insert_after(
    lines,
    "                              patient_ids = cohort, verbose = TRUE)",
    ".cluda_rng_trace$after_timeseries_concepts <- .Random.seed"
  )
  lines <- insert_after(
    lines,
    "count <- length(list_stayids)",
    ".cluda_rng_trace$after_los_filter <- .Random.seed"
  )
  lines <- insert_after(
    lines,
    "ts_data <- fill_gaps(ts_data)",
    ".cluda_rng_trace$after_fill_gaps <- .Random.seed"
  )
  lines <- insert_after(
    lines,
    "hist(trim_q(as.numeric(adm_los$los), 0.05, 0.9), breaks=100, freq=F, main=\"Distribution of LOS after trimming\")",
    ".cluda_rng_trace$after_histograms <- .Random.seed"
  )
  lines <- insert_after(
    lines,
    "icustays_df <- icustays_df[icustays_df$stay_id %in% unique(ts_data_mask$stay_id),]",
    ".cluda_rng_trace$after_eligible_filter <- .Random.seed"
  )
  split_line <-
    "train_index <- createDataPartition(icustays_df$hospital_expire_flag, p = .7, list = FALSE)"
  location <- which(lines == split_line)
  if (length(location) != 1L) {
    stop("released split boundary did not occur exactly once")
  }
  probe <- c(
    "saveRDS(list(",
    "  status = 'exact_released_pre_split_state',",
    "  rng_kind = RNGkind(),",
    "  random_seed = .Random.seed,",
    "  rng_trace = .cluda_rng_trace,",
    "  icustays_df = as.data.frame(icustays_df[, c('subject_id', 'hadm_id', 'stay_id', 'hospital_expire_flag')]),",
    "  lengths = as.data.frame(ts_data_mask[, .(length = .N), by = stay_id]),",
    "  ages = as.data.frame(cohort[, c('stay_id', 'age')])",
    "), Sys.getenv('CLUDA_SPLIT_PROBE_RDS'))",
    "quit(save = 'no', status = 0L, runLast = FALSE)"
  )
  derived <- append(lines, probe, after = location - 1L)
  writeLines(derived, probed_process, useBytes = TRUE)
  real_sys_source(probed_process, envir = envir, ...)
}

dir.create(output_root, recursive = TRUE, showWarnings = FALSE)
Sys.setenv(CLUDA_SPLIT_PROBE_RDS = probe_rds)
wrapper_to_source <- canonical_wrapper
if (!identical(expected_ricu_version, "0.5.2")) {
  wrapper_lines <- readLines(canonical_wrapper, warn = FALSE)
  version_line <-
    'if (as.character(packageVersion("ricu")) != "0.5.2") {'
  location <- which(wrapper_lines == version_line)
  if (length(location) != 1L) {
    stop("canonical RICU version boundary did not occur exactly once")
  }
  wrapper_lines[[location]] <- sub(
    '"0.5.2"', paste0('"', expected_ricu_version, '"'),
    wrapper_lines[[location]], fixed = TRUE
  )
  wrapper_to_source <- tempfile(fileext = ".R")
  writeLines(wrapper_lines, wrapper_to_source, useBytes = TRUE)
  on.exit(unlink(wrapper_to_source), add = TRUE)
}
source(wrapper_to_source, local = .GlobalEnv)
stop("released producer returned without terminating at the split probe")
