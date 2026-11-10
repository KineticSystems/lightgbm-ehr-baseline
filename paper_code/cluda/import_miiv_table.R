args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("usage: import_miiv_table.R RICU_DATA_PATH TABLE AUDIT_JSON")
}

ricu_data_path <- normalizePath(args[[1]], mustWork = TRUE)
table_name <- args[[2]]
audit_path <- args[[3]]
required_tables <- c(
  "admissions", "patients", "services", "labevents", "icustays", "chartevents"
)
if (!(table_name %in% required_tables)) {
  stop(paste("unsupported CLUDA table", table_name))
}

Sys.setenv(RICU_DATA_PATH = ricu_data_path, RICU_SRC_LOAD = "")
suppressPackageStartupMessages(library(ricu))

if (as.character(packageVersion("ricu")) != "0.5.2") {
  stop(paste("CLUDA requires the time-compatible ricu 0.5.2, found", packageVersion("ricu")))
}

miiv_dir <- file.path(ricu_data_path, "miiv")
if (!dir.exists(miiv_dir)) {
  stop("the prepared exact MIMIC-IV v1.0 RICU source directory is missing")
}

# Run the exact RICU 0.5.2 importer for one table in this process.  The Python
# orchestrator starts a fresh R process for every table so memory retained by
# readr/fst while partitioning one large table cannot accumulate into the next
# import.  This changes only process isolation; import_src/import_tbl and their
# parsing, renaming, partitioning, sorting, and FST writing remain unchanged.
import_src("miiv", data_dir = miiv_dir, tables = table_name)

cfg <- load_src_cfg("miiv")[[1]]
table_cfg <- ricu:::as_tbl_cfg(cfg)[table_name]
relative_files <- unlist(ricu:::fst_file_names(table_cfg), use.names = FALSE)
paths <- file.path(miiv_dir, relative_files)
if (!all(file.exists(paths))) {
  stop(paste("RICU omitted configured FST files:", paste(relative_files[!file.exists(paths)], collapse = ", ")))
}

row_counts <- vapply(paths, function(path) as.double(nrow(fst::fst(path))), numeric(1))
if (any(!is.finite(row_counts)) || any(row_counts <= 0)) {
  stop("RICU produced an unreadable or empty configured FST file")
}
hashes <- vapply(
  paths,
  function(path) {
    connection <- file(path, open = "rb")
    on.exit(close(connection))
    as.character(openssl::sha256(connection))
  },
  character(1)
)

dir.create(dirname(audit_path), recursive = TRUE, showWarnings = FALSE)
jsonlite::write_json(
  list(
    status = "complete",
    table = table_name,
    configured_rows = as.double(ricu:::n_row(table_cfg)),
    observed_rows = sum(row_counts),
    files = unname(Map(
        function(path, rows, hash) list(file = path, rows = rows, sha256 = hash),
        relative_files,
        row_counts,
        hashes
      ))
  ),
  audit_path,
  auto_unbox = TRUE,
  pretty = TRUE
)
