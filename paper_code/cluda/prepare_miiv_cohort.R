args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("usage: prepare_miiv_cohort.R RICU_DATA_PATH RELEASED_CSV_ROOT")
}

ricu_data_path <- normalizePath(args[[1]], mustWork = TRUE)
output_root <- normalizePath(args[[2]], mustWork = FALSE)
paper_root <- normalizePath(dirname(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[[1]])))

Sys.setenv(RICU_DATA_PATH = ricu_data_path, RICU_SRC_LOAD = "")
suppressPackageStartupMessages(library(ricu))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(caret))

if (as.character(packageVersion("ricu")) != "0.5.2") {
  stop(paste("CLUDA requires the time-compatible ricu 0.5.2, found", packageVersion("ricu")))
}

miiv_dir <- file.path(ricu_data_path, "miiv")
if (!dir.exists(miiv_dir)) {
  stop("the prepared exact MIMIC-IV v1.0 RICU source directory is missing")
}

required_tables <- c(
  "admissions", "patients", "services", "labevents", "icustays", "chartevents"
)

# The Python adapter has already imported and validated the six required raw
# tables, one exact RICU 0.5.2 import per fresh R process.  RICU otherwise
# attaches all 27 tables in its MIMIC-IV source configuration and consequently
# marks the source unavailable when the 21 unused tables are intentionally
# absent.  Parse the same pinned source configuration after restricting only
# its table list to the sealed six-table boundary; table definitions, column
# types, ID systems, partitioning, and concept definitions remain unchanged.
raw_miiv_config <- ricu:::read_src_cfg("miiv")[[1]]
if (!all(required_tables %in% names(raw_miiv_config$tables))) {
  stop("the pinned RICU source configuration omitted a required CLUDA table")
}
raw_miiv_config$tables <- raw_miiv_config$tables[required_tables]
miiv_config <- ricu:::parse_src_cfg(raw_miiv_config)
attach_src(miiv_config, data_dir = miiv_dir, assign_env = .GlobalEnv)
if (!isTRUE(is_data_avail("miiv"))) {
  stop("the validated six-table RICU source did not attach as available")
}

# feature_selection.R computes this exact vector by inspecting both MIMIC and
# AUMC, then appends its final six fields.  The export is MIMIC-only; Table 5 of
# the accepted paper fixes the resulting order, so no unavailable AUMC rows are
# needed to repeat a deterministic availability calculation.
selected_ts_features <- c(
  "alb", "alp", "alt", "ast", "be", "bicar", "bili", "bun", "ca", "cai",
  "ck", "cl", "crea", "dbp", "etco2", "fio2", "glu", "hct", "hgb", "hr",
  "inr_pt", "k", "lact", "map", "mg", "na", "pco2", "ph", "phos", "plt",
  "po2", "ptt", "sbp", "temp", "wbc", "basos", "eos", "lymph", "neut",
  "pt", "rbc"
)
static_features <- c("adm", "age", "bmi", "height", "sex", "weight")

# The released instructions require feature_selection.R and process_miiv.R to
# run in the same R environment.  The former defines trim_q, which the latter
# calls after its expensive per-stay loop.  Executing all of feature_selection.R
# here would repeat its interactive AUMC/MIMIC exploration; instead, locate and
# evaluate its one exact top-level helper assignment from the integrity-pinned
# released file.  Fail closed if its identity or body differs.
feature_selection_path <- normalizePath(
  file.path(paper_root, "load_ICU_datasets", "feature_selection.R"),
  mustWork = TRUE
)
feature_selection_exprs <- parse(feature_selection_path, keep.source = TRUE)
is_trim_q_assignment <- function(expr) {
  is.call(expr) &&
    identical(expr[[1]], as.name("<-")) &&
    identical(expr[[2]], as.name("trim_q")) &&
    is.call(expr[[3]]) &&
    identical(expr[[3]][[1]], as.name("function"))
}
trim_q_exprs <- Filter(is_trim_q_assignment, as.list(feature_selection_exprs))
if (length(trim_q_exprs) != 1L) {
  stop("the pinned released feature_selection.R did not contain exactly one top-level trim_q function assignment")
}
eval(trim_q_exprs[[1]], envir = .GlobalEnv)
expected_trim_q_body <- quote({
  x[(x > quantile(x, lb, na.rm = T)) & (x < quantile(x, ub, na.rm = T))]
})
if (!is.function(trim_q) ||
    !identical(formals(trim_q), as.pairlist(alist(x = , lb = , ub = ))) ||
    !isTRUE(all.equal(body(trim_q), expected_trim_q_body, check.attributes = FALSE))) {
  stop("the extracted released trim_q helper failed its exact formals/body check")
}

# Preflight every function call in the released top-level process before
# entering the costly cohort construction.  Functions assigned by that same
# file are local definitions; every other callable must resolve on the exact
# package/search path or through the extracted released helper above.
process_miiv_path <- normalizePath(
  file.path(paper_root, "load_ICU_datasets", "process_miiv.R"),
  mustWork = TRUE
)
process_miiv_exprs <- parse(process_miiv_path, keep.source = TRUE)
process_miiv_body <- as.call(c(as.name("{"), as.list(process_miiv_exprs)))
process_miiv_fn <- eval(
  call("function", as.pairlist(alist()), process_miiv_body),
  envir = .GlobalEnv
)
process_function_calls <- codetools::findGlobals(
  process_miiv_fn, merge = FALSE
)$functions
unresolved_function_calls <- process_function_calls[
  !vapply(
    process_function_calls,
    exists,
    logical(1),
    mode = "function",
    inherits = TRUE
  )
]
if (length(unresolved_function_calls) != 0L) {
  stop(paste(
    "unresolved released top-level helper/function calls:",
    paste(unresolved_function_calls, collapse = ", ")
  ))
}
message(paste(
  "Loaded exact released trim_q helper from",
  feature_selection_path,
  "and resolved",
  length(process_function_calls),
  "released top-level function calls"
))

# Resolve every requested concept against the unmodified pinned dictionary
# before invoking the released producer.  This is a configuration smoke test,
# not a replacement or extension of any concept definition.
resolved_concepts <- load_dictionary(
  "miiv", c(static_features, selected_ts_features)
)
if (!identical(names(resolved_concepts), c(static_features, selected_ts_features))) {
  stop("the pinned RICU dictionary did not resolve the released concept set")
}
rm(resolved_concepts)

dir.create(output_root, recursive = TRUE, showWarnings = FALSE)
old_working_directory <- getwd()
on.exit(setwd(old_working_directory), add = TRUE)
setwd(output_root)

# Run the released cohort, label, split, hourly aggregation, forward-fill,
# scaling, mask, and CSV writer unchanged.
sys.source(process_miiv_path, envir = .GlobalEnv)

# The MIMIC file ends immediately before this tail.  These statements are the
# identifier-renamed counterpart of the released process_aumc.R tail.
static_cont_features <- setdiff(static_features, c("adm", "sex"))
static_miiv_means <- colMeans(
  cohort[cohort$stay_id %in% train_admissions, ..static_cont_features],
  na.rm = TRUE
)
static_miiv_sds <- apply(
  cohort[cohort$stay_id %in% train_admissions, ..static_cont_features],
  2,
  sd,
  na.rm = TRUE
)
cohort_miiv_scaled <- as.data.frame(cohort)
cohort_miiv_scaled <- cohort_miiv_scaled[
  , setdiff(names(cohort_miiv_scaled), c("adm", "sex")), drop = FALSE
]
for (col_name in static_cont_features) {
  cohort_miiv_scaled[[col_name]] <-
    (cohort_miiv_scaled[[col_name]] - static_miiv_means[col_name]) /
    static_miiv_sds[col_name]
}
cohort_miiv_scaled[is.na(cohort_miiv_scaled)] <- 0

miiv_output <- file.path(output_root, "miiv")
write.csv(
  cohort_miiv_scaled,
  file.path(miiv_output, "static_data_scaled.csv"),
  row.names = FALSE
)
write.csv(
  data.frame(feature = names(ts_means), mean = ts_means, sd = ts_sds),
  file.path(miiv_output, "ts_scaling.csv"),
  row.names = FALSE
)
write.csv(
  data.frame(
    feature = names(static_miiv_means),
    mean = static_miiv_means,
    sd = static_miiv_sds
  ),
  file.path(miiv_output, "static_scaling.csv"),
  row.names = FALSE
)
writeLines(selected_ts_features, file.path(miiv_output, "channel_order.txt"))
writeLines(
  c("age", "bmi", "height", "weight", "adm_0", "adm_1", "adm_2", "sex"),
  file.path(miiv_output, "static_order.txt")
)
writeLines(as.character(packageVersion("ricu")), file.path(miiv_output, "ricu_version.txt"))
capture.output(sessionInfo(), file = file.path(miiv_output, "sessionInfo.txt"))
