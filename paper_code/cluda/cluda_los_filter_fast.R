# Exact one-pass replacement for the released CLUDA LOS-filter loop.

CLUDA_PROCESS_MIIV_SHA256 <-
  "1b19d6c61f012dd538003128fc81b9d8e519189cd8da4480af6cf89a1a262d46"
CLUDA_RELEASED_LOS_BLOCK_SHA256 <-
  "f151e16e73bc8f6e4d49fb0fccdc95e55a71aa2ab1d1f1ebe3e80b5e17c52466"

CLUDA_RELEASED_LOS_BLOCK <- c(
  "for(sid in list_stayids){",
  "  hadm_id <- icustays_df$hadm_id[icustays_df$stay_id == sid]",
  "  los <- icustays_df$los[icustays_df$hadm_id == hadm_id]",
  "  los <- ceiling(los*24)",
  "  cond <- ts_data$stay_id == sid & ts_data$charttime > los",
  "  ts_data <- ts_data[!cond,]",
  "  if(count %% 100 == 0){print(count)}",
  "  count <- count + 1",
  "}"
)

cluda_sha256_raw <- function(value) {
  unclass(as.character(openssl::sha256(value)))
}

cluda_sha256_file <- function(path) {
  connection <- file(path, open = "rb")
  on.exit(close(connection), add = TRUE)
  cluda_sha256_raw(connection)
}

cluda_sha256_lines <- function(lines) {
  cluda_sha256_raw(charToRaw(paste0(paste(lines, collapse = "\n"), "\n")))
}

cluda_require_finite <- function(value, name) {
  finite <- tryCatch(
    is.finite(value),
    error = function(error) {
      stop(paste(name, "does not support finite-value validation"), call. = FALSE)
    }
  )
  if (length(finite) != length(value) || anyNA(value) || any(!finite)) {
    stop(paste(name, "contains a missing or nonfinite value"), call. = FALSE)
  }
  invisible(TRUE)
}

cluda_filter_los_exact_fast <- function(ts_data, icustays_df, list_stayids) {
  required_ts <- c("stay_id", "charttime")
  required_icu <- c("stay_id", "hadm_id", "los")
  if (!all(required_ts %in% names(ts_data))) {
    stop("ts_data lacks the released stay_id/charttime fields", call. = FALSE)
  }
  if (!all(required_icu %in% names(icustays_df))) {
    stop("icustays_df lacks the released stay_id/hadm_id/los fields", call. = FALSE)
  }
  if (!identical(list_stayids, unique(ts_data$stay_id))) {
    stop("list_stayids is not the released first-appearance stay order", call. = FALSE)
  }
  if (length(list_stayids) == 0L) {
    return(list(
      data = ts_data,
      audit = list(
        status = "exact_scalar_los_fast_path",
        input_rows = nrow(ts_data),
        output_rows = nrow(ts_data),
        stays = 0L,
        removed_rows = 0L
      )
    ))
  }

  # The released expressions compare these complete vectors. An NA in an
  # otherwise unrelated icustays ID would itself be selected as NA during R
  # logical indexing and make hadm_id/los nonscalar, so validate the full ID
  # columns rather than only the ultimately matched rows.
  cluda_require_finite(ts_data$stay_id, "ts_data$stay_id")
  cluda_require_finite(ts_data$charttime, "ts_data$charttime")
  cluda_require_finite(icustays_df$stay_id, "icustays_df$stay_id")
  cluda_require_finite(icustays_df$hadm_id, "icustays_df$hadm_id")

  stay_match <- match(list_stayids, icustays_df$stay_id)
  if (anyNA(stay_match)) {
    stop("a ts_data stay_id has no icustays_df row", call. = FALSE)
  }
  relevant_stay_rows <- icustays_df$stay_id %in% list_stayids
  if (anyDuplicated(icustays_df$stay_id[relevant_stay_rows])) {
    stop("a ts_data stay_id maps to multiple icustays_df rows", call. = FALSE)
  }

  stay_hadm <- icustays_df$hadm_id[stay_match]
  relevant_hadm_rows <- icustays_df$hadm_id %in% stay_hadm
  if (anyDuplicated(icustays_df$hadm_id[relevant_hadm_rows])) {
    stop("a selected hadm_id maps to multiple LOS rows", call. = FALSE)
  }
  hadm_match <- match(stay_hadm, icustays_df$hadm_id)
  if (anyNA(hadm_match)) {
    stop("a selected hadm_id has no LOS row", call. = FALSE)
  }
  stay_los <- icustays_df$los[hadm_match]
  cluda_require_finite(stay_los, "selected icustays_df$los")

  # This is deliberately the released arithmetic, in R, before matching the
  # resulting scalar threshold back to every row of its stay.
  los_hours <- ceiling(stay_los * 24)
  stay_position <- match(ts_data$stay_id, list_stayids)
  if (anyNA(stay_position)) {
    stop("ts_data changed after list_stayids was captured", call. = FALSE)
  }
  keep <- !(ts_data$charttime > los_hours[stay_position])
  if (anyNA(keep) || length(keep) != nrow(ts_data)) {
    stop("the scalar LOS filter did not produce one Boolean per input row", call. = FALSE)
  }
  result <- ts_data[keep, ]
  if (!identical(names(result), names(ts_data)) ||
      !identical(class(result), class(ts_data))) {
    stop("one-pass subsetting changed the released table names or class", call. = FALSE)
  }
  list(
    data = result,
    audit = list(
      status = "exact_scalar_los_fast_path",
      input_rows = nrow(ts_data),
      output_rows = nrow(result),
      stays = length(list_stayids),
      removed_rows = nrow(ts_data) - nrow(result)
    )
  )
}

cluda_write_fast_process_script <- function(upstream_path, output_path) {
  actual_source_sha256 <- cluda_sha256_file(upstream_path)
  if (!identical(actual_source_sha256, CLUDA_PROCESS_MIIV_SHA256)) {
    stop(
      paste(
        "released process_miiv.R SHA-256 changed:",
        actual_source_sha256,
        "!=",
        CLUDA_PROCESS_MIIV_SHA256
      ),
      call. = FALSE
    )
  }
  actual_block_sha256 <- cluda_sha256_lines(CLUDA_RELEASED_LOS_BLOCK)
  if (!identical(actual_block_sha256, CLUDA_RELEASED_LOS_BLOCK_SHA256)) {
    stop("pinned released LOS expression changed", call. = FALSE)
  }

  lines <- readLines(upstream_path, warn = FALSE)
  width <- length(CLUDA_RELEASED_LOS_BLOCK)
  candidates <- which(lines == CLUDA_RELEASED_LOS_BLOCK[[1]])
  matches <- candidates[vapply(
    candidates,
    function(start) {
      stop <- start + width - 1L
      stop <= length(lines) && identical(lines[start:stop], CLUDA_RELEASED_LOS_BLOCK)
    },
    logical(1L)
  )]
  if (length(matches) != 1L) {
    stop("released LOS loop did not occur exactly once", call. = FALSE)
  }

  start <- matches[[1]]
  stop <- start + width - 1L
  replacement <- c(
    ".cluda_los_fast_result <- cluda_filter_los_exact_fast(ts_data, icustays_df, list_stayids)",
    "ts_data <- .cluda_los_fast_result$data",
    ".cluda_los_fast_audit <- .cluda_los_fast_result$audit",
    "rm(.cluda_los_fast_result)",
    "count <- length(list_stayids)"
  )
  derived <- c(
    if (start > 1L) lines[seq_len(start - 1L)] else character(),
    replacement,
    if (stop < length(lines)) lines[(stop + 1L):length(lines)] else character()
  )
  writeLines(derived, output_path, useBytes = TRUE)
  list(
    status = "released_los_loop_integrity_pinned_and_replaced",
    released_process_miiv_sha256 = actual_source_sha256,
    released_los_block_sha256 = actual_block_sha256,
    released_los_block_start_line = start,
    released_los_block_end_line = stop,
    derived_process_miiv_sha256 = cluda_sha256_file(output_path)
  )
}
