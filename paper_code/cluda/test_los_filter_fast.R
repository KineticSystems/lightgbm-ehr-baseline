suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(ricu))
.test_paper_root <- normalizePath(
  dirname(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[[1]]))
)
source(file.path(.test_paper_root, "cluda_los_filter_fast.R"))

released_filter <- function(ts_data, icustays_df) {
  count <- 0
  list_stayids <- unique(ts_data$stay_id)
  invisible(capture.output({
    for(sid in list_stayids){
      hadm_id <- icustays_df$hadm_id[icustays_df$stay_id == sid]
      los <- icustays_df$los[icustays_df$hadm_id == hadm_id]
      los <- ceiling(los*24)
      cond <- ts_data$stay_id == sid & ts_data$charttime > los
      ts_data <- ts_data[!cond,]
      if(count %% 100 == 0){print(count)}
      count <- count + 1
    }
  }))
  ts_data
}

expect_error <- function(expression, pattern) {
  error <- tryCatch(
    {
      force(expression)
      NULL
    },
    error = identity
  )
  if (is.null(error) || !grepl(pattern, conditionMessage(error), fixed = TRUE)) {
    stop(paste("expected failure containing:", pattern))
  }
}

ts <- data.table(
  source_row = 1:6,
  stay_id = c(20, 10, 20, 30, 10, 30),
  charttime = c(13, 5, 11, 25, 7, 20),
  value = c(NA_real_, 2, 3, 4, 5, 6),
  text = letters[1:6]
)
icu <- data.frame(
  stay_id = c(30, 10, 20),
  hadm_id = c(300, 100, 200),
  los = c(1, 0.25, 0.5)
)
reference <- released_filter(copy(ts), icu)
fast <- cluda_filter_los_exact_fast(copy(ts), icu, unique(ts$stay_id))
stopifnot(
  identical(fast$data, reference),
  identical(fast$data$source_row, c(2L, 3L, 6L)),
  identical(class(fast$data), class(ts)),
  identical(fast$audit$removed_rows, 3L)
)

# Exercise ricu's real ts_tbl/id_tbl subsetting method as well as data.table.
ts_id <- as_ts_tbl(
  data.table(
    source_row = 1:4,
    stay_id = c(10, 10, 20, 20),
    charttime = as.difftime(c(1, 7, 3, 13), units = "hours"),
    value = c(NA_real_, 2, 3, 4)
  ),
  id_vars = "stay_id",
  index_var = "charttime",
  interval = as.difftime(1, units = "hours")
)
reference_id <- released_filter(data.table::copy(ts_id), icu)
fast_id <- cluda_filter_los_exact_fast(
  data.table::copy(ts_id), icu, unique(ts_id$stay_id)
)
stopifnot(
  identical(fast_id$data, reference_id),
  identical(class(fast_id$data), class(ts_id)),
  identical(fast_id$data$source_row, c(1L, 3L))
)

missing_stay <- copy(ts)
missing_stay[1, stay_id := 40]
expect_error(
  cluda_filter_los_exact_fast(missing_stay, icu, unique(missing_stay$stay_id)),
  "has no icustays_df row"
)

duplicate_stay <- rbind(icu, data.frame(stay_id = 20, hadm_id = 201, los = 0.5))
expect_error(
  cluda_filter_los_exact_fast(copy(ts), duplicate_stay, unique(ts$stay_id)),
  "maps to multiple icustays_df rows"
)

# Different LOS values for one selected admission trigger the released global
# recycling behavior. The fast path must reject rather than reproduce it.
multiple_los <- copy(icu)
multiple_los$hadm_id[[1]] <- 200
recycled_reference <- suppressWarnings(released_filter(copy(ts), multiple_los))
stopifnot(
  !identical(recycled_reference, reference),
  identical(recycled_reference$source_row, c(1L, 2L, 3L, 6L))
)
expect_error(
  cluda_filter_los_exact_fast(copy(ts), multiple_los, unique(ts$stay_id)),
  "maps to multiple LOS rows"
)

bad_ts_id <- copy(ts)
bad_ts_id$stay_id[[1]] <- NA_real_
expect_error(
  cluda_filter_los_exact_fast(bad_ts_id, icu, unique(bad_ts_id$stay_id)),
  "contains a missing or nonfinite value"
)
bad_ts_time <- copy(ts)
bad_ts_time$charttime[[1]] <- Inf
expect_error(
  cluda_filter_los_exact_fast(bad_ts_time, icu, unique(bad_ts_time$stay_id)),
  "contains a missing or nonfinite value"
)
bad_ts_time_na <- copy(ts)
bad_ts_time_na$charttime[[1]] <- NA_real_
expect_error(
  cluda_filter_los_exact_fast(bad_ts_time_na, icu, unique(bad_ts_time_na$stay_id)),
  "contains a missing or nonfinite value"
)
bad_icu_id <- rbind(icu, data.frame(stay_id = NA_real_, hadm_id = 999, los = 1))
expect_error(
  cluda_filter_los_exact_fast(copy(ts), bad_icu_id, unique(ts$stay_id)),
  "contains a missing or nonfinite value"
)
bad_hadm <- copy(icu)
bad_hadm$hadm_id[[1]] <- Inf
expect_error(
  cluda_filter_los_exact_fast(copy(ts), bad_hadm, unique(ts$stay_id)),
  "contains a missing or nonfinite value"
)
bad_los <- copy(icu)
bad_los$los[[1]] <- NA_real_
expect_error(
  cluda_filter_los_exact_fast(copy(ts), bad_los, unique(ts$stay_id)),
  "contains a missing or nonfinite value"
)
bad_los_inf <- copy(icu)
bad_los_inf$los[[1]] <- Inf
expect_error(
  cluda_filter_los_exact_fast(copy(ts), bad_los_inf, unique(ts$stay_id)),
  "contains a missing or nonfinite value"
)
expect_error(
  cluda_filter_los_exact_fast(copy(ts), icu, rev(unique(ts$stay_id))),
  "not the released first-appearance stay order"
)

derived <- tempfile(fileext = ".R")
seal <- cluda_write_fast_process_script(
  file.path(.test_paper_root, "load_ICU_datasets", "process_miiv.R"), derived
)
derived_lines <- readLines(derived, warn = FALSE)
stopifnot(
  identical(seal$released_process_miiv_sha256, CLUDA_PROCESS_MIIV_SHA256),
  identical(seal$released_los_block_sha256, CLUDA_RELEASED_LOS_BLOCK_SHA256),
  identical(seal$released_los_block_start_line, 44L),
  identical(seal$released_los_block_end_line, 52L),
  sum(grepl("cluda_filter_los_exact_fast", derived_lines, fixed = TRUE)) == 1L,
  !any(derived_lines == CLUDA_RELEASED_LOS_BLOCK[[1]])
)
unlink(derived)

tampered <- tempfile(fileext = ".R")
tampered_lines <- readLines(
  file.path(.test_paper_root, "load_ICU_datasets", "process_miiv.R"),
  warn = FALSE
)
tampered_lines[[42]] <- "count <- 1"
writeLines(tampered_lines, tampered)
expect_error(
  cluda_write_fast_process_script(tampered, tempfile(fileext = ".R")),
  "process_miiv.R SHA-256 changed"
)
unlink(tampered)

cat("CLUDA exact LOS fast-path fixtures passed\n")
