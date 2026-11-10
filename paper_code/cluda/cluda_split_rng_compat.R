# Fail-closed reconstruction of CLUDA's paper-evaluated split RNG boundary.

CLUDA_RELEASED_PRE_SPLIT_RNG_SHA256 <-
  "0c206a90d90b15baaaf4db06185f0716bcf8e24035561f7a4b76938ff48e3344"
CLUDA_PAPER_PRE_SPLIT_RNG_SHA256 <-
  "6dd4ba42d5e418370956f80414fafffe41a0589649706b33203454a608c140a4"
CLUDA_RELEASED_NAME_DRAWS <- 37L
CLUDA_PAPER_NAME_DRAWS <- 72L
CLUDA_ADDITIONAL_TS_NAME_DRAWS <-
  CLUDA_PAPER_NAME_DRAWS - CLUDA_RELEASED_NAME_DRAWS

cluda_rng_sha256 <- function(seed = .Random.seed) {
  unclass(as.character(openssl::sha256(serialize(seed, NULL, version = 2L))))
}

cluda_apply_paper_split_rng_compat <- function() {
  if (!exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
    stop("released CLUDA producer has no initialized RNG state", call. = FALSE)
  }
  before <- cluda_rng_sha256(.Random.seed)
  if (!identical(before, CLUDA_RELEASED_PRE_SPLIT_RNG_SHA256)) {
    stop(
      paste(
        "released CLUDA pre-split RNG state changed:", before, "!=",
        CLUDA_RELEASED_PRE_SPLIT_RNG_SHA256
      ),
      call. = FALSE
    )
  }

  # The evaluated artifacts are exactly equivalent to one additional
  # invocation of the released 41-channel RICU loader: on the pinned source it
  # generates 35 temporary names. Avoid loading the 2.2 GB table twice while
  # retaining the exact R/RICU random-state transition.
  for (index in seq_len(CLUDA_ADDITIONAL_TS_NAME_DRAWS)) {
    ricu:::new_names()
  }
  after <- cluda_rng_sha256(.Random.seed)
  if (!identical(after, CLUDA_PAPER_PRE_SPLIT_RNG_SHA256)) {
    stop(
      paste(
        "CLUDA paper-evaluated pre-split RNG state changed:", after, "!=",
        CLUDA_PAPER_PRE_SPLIT_RNG_SHA256
      ),
      call. = FALSE
    )
  }

  list(
    status = "paper_artifact_triangulated_split_rng_state",
    released_script_pre_split_rng_sha256 = before,
    paper_evaluated_pre_split_rng_sha256 = after,
    released_name_draws = CLUDA_RELEASED_NAME_DRAWS,
    additional_time_series_loader_name_draws =
      CLUDA_ADDITIONAL_TS_NAME_DRAWS,
    paper_evaluated_name_draws = CLUDA_PAPER_NAME_DRAWS,
    exact_downstream_signatures = list(
      loader_counts = c(train = 34290L, validation = 7343L, test = 7353L),
      sample_counts = c(
        train = 2398546L, validation = 513636L, test = 512454L
      ),
      training_age_mean = 63.55277,
      training_age_sd = 17.259564
    )
  )
}

cluda_write_paper_split_process_script <- function(
  input_path, output_path, upstream_path
) {
  if (!identical(
    cluda_sha256_file(upstream_path), CLUDA_PROCESS_MIIV_SHA256
  )) {
    stop("released process_miiv.R changed; split reconstruction is unsafe")
  }
  lines <- readLines(input_path, warn = FALSE)
  split_line <-
    "train_index <- createDataPartition(icustays_df$hospital_expire_flag, p = .7, list = FALSE)"
  location <- which(lines == split_line)
  if (length(location) != 1L) {
    stop("released split boundary did not occur exactly once", call. = FALSE)
  }
  derived <- append(
    lines,
    ".cluda_split_rng_audit <- cluda_apply_paper_split_rng_compat()",
    after = location - 1L
  )
  writeLines(derived, output_path, useBytes = TRUE)
  list(
    status = "released_split_boundary_integrity_pinned_and_reconstructed",
    released_process_miiv_sha256 = cluda_sha256_file(upstream_path),
    input_process_sha256 = cluda_sha256_file(input_path),
    derived_process_sha256 = cluda_sha256_file(output_path),
    split_line = split_line,
    split_line_number = location,
    released_script_pre_split_rng_sha256 =
      CLUDA_RELEASED_PRE_SPLIT_RNG_SHA256,
    paper_evaluated_pre_split_rng_sha256 =
      CLUDA_PAPER_PRE_SPLIT_RNG_SHA256
  )
}
