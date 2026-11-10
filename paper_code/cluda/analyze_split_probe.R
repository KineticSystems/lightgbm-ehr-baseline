# Summarize CLUDA split-state probes without retaining patient-level data.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L) {
  stop(
    paste(
      "usage: analyze_split_probe.R OUTPUT_JSON",
      "LABEL=PROBE_RDS [LABEL=PROBE_RDS ...]"
    )
  )
}

paper_root <- normalizePath(
  dirname(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[[1]]))
)
source(file.path(paper_root, "cluda_split_rng_compat.R"), local = .GlobalEnv)

output_path <- args[[1]]
probe_specs <- args[-1]
split_spec <- function(value) {
  location <- regexpr("=", value, fixed = TRUE)
  if (location < 2L) {
    stop(paste("invalid probe specification:", value))
  }
  list(
    label = substr(value, 1L, location - 1L),
    path = substr(value, location + 1L, nchar(value))
  )
}

draw_min <- 0L
draw_max <- 300L
expected <- list(
  loader_counts = c(train = 34290L, validation = 7343L, test = 7353L),
  sample_counts = c(
    train = 2398546L, validation = 513636L, test = 512454L
  ),
  training_age_mean = 63.55277,
  training_age_sd = 17.259564
)
age_tolerance <- 5.1e-6

seed_after_draws <- function(draws) {
  set.seed(1234)
  if (draws > 0L) {
    for (index in seq_len(draws)) {
      ricu:::new_names()
    }
  }
  .Random.seed
}

candidate_seeds <- lapply(draw_min:draw_max, seed_after_draws)
candidate_hashes <- vapply(candidate_seeds, cluda_rng_sha256, character(1))

identify_draw <- function(seed) {
  matches <- which(candidate_hashes == cluda_rng_sha256(seed))
  as.integer((draw_min:draw_max)[matches])
}

named_counts <- function(values) {
  list(
    train = as.integer(values[["train"]]),
    validation = as.integer(values[["validation"]]),
    test = as.integer(values[["test"]])
  )
}

split_signature <- function(probe, seed) {
  frame <- probe$icustays_df
  if (
    !identical(nrow(frame), 49351L) ||
    anyDuplicated(frame$stay_id) ||
    anyDuplicated(probe$lengths$stay_id) ||
    anyDuplicated(probe$ages$stay_id)
  ) {
    stop("probe does not contain the expected unique 49,351-stay boundary")
  }
  assign(".Random.seed", seed, envir = .GlobalEnv)
  train_index <- as.integer(caret::createDataPartition(
    frame$hospital_expire_flag, p = .7, list = FALSE
  ))
  remaining_index <- which(!(seq_len(nrow(frame)) %in% train_index))
  val_index <- as.integer(caret::createDataPartition(
    frame$hospital_expire_flag[remaining_index], p = .5, list = FALSE
  ))
  split_ids <- list(
    train = frame$stay_id[train_index],
    validation = frame$stay_id[remaining_index][val_index],
    test = frame$stay_id[remaining_index][-val_index]
  )
  length_by_stay <- setNames(
    as.integer(probe$lengths$length), as.character(probe$lengths$stay_id)
  )
  lengths <- lapply(
    split_ids,
    function(ids) unname(length_by_stay[as.character(ids)])
  )
  if (any(vapply(lengths, anyNA, logical(1)))) {
    stop("probe omitted a stay length")
  }
  loader_counts <- vapply(lengths, function(value) sum(value >= 4L), integer(1))
  sample_counts <- vapply(
    lengths,
    function(value) sum(pmax(value - 3L, 0L)),
    double(1)
  )
  age_by_stay <- setNames(probe$ages$age, as.character(probe$ages$stay_id))
  train_age <- unname(age_by_stay[as.character(split_ids$train)])
  if (anyNA(train_age)) {
    stop("probe omitted a training-stay age")
  }
  list(
    prefilter_counts = named_counts(vapply(split_ids, length, integer(1))),
    loader_counts = named_counts(loader_counts),
    sample_counts = named_counts(sample_counts),
    training_age_mean = unname(mean(train_age)),
    training_age_sd = unname(stats::sd(train_age))
  )
}

matches_expected <- function(signature) {
  identical(unname(unlist(signature$loader_counts)), unname(expected$loader_counts)) &&
    identical(unname(unlist(signature$sample_counts)), unname(expected$sample_counts)) &&
    abs(signature$training_age_mean - expected$training_age_mean) <= age_tolerance &&
    abs(signature$training_age_sd - expected$training_age_sd) <= age_tolerance
}

specs <- lapply(probe_specs, split_spec)
probes <- lapply(specs, function(spec) readRDS(spec$path))
if (any(vapply(probes, function(probe) probe$status, character(1)) !=
        "exact_released_pre_split_state")) {
  stop("a probe has the wrong status")
}
reference <- probes[[1]]

candidate_signatures <- lapply(candidate_seeds, function(seed) {
  split_signature(reference, seed)
})
matching_draws <- (draw_min:draw_max)[vapply(
  candidate_signatures, matches_expected, logical(1)
)]
if (!identical(as.integer(matching_draws), CLUDA_PAPER_NAME_DRAWS)) {
  stop(paste("paper signature did not have one expected draw match:", matching_draws))
}

summarize_probe <- function(spec, probe) {
  identified <- identify_draw(probe$random_seed)
  if (length(identified) != 1L) {
    stop(paste("probe state did not have one draw-count identity:", spec$label))
  }
  list(
    label = spec$label,
    input_rds_sha256 = unclass(as.character(openssl::sha256(file(spec$path)))),
    rng_kind = as.list(probe$rng_kind),
    pre_split_rng_sha256 = cluda_rng_sha256(probe$random_seed),
    identified_name_draws = identified,
    signature = split_signature(probe, probe$random_seed)
  )
}

historical <- Map(summarize_probe, specs, probes)
trace_draws <- lapply(reference$rng_trace, identify_draw)
if (any(lengths(trace_draws) != 1L)) {
  stop("a released trace state did not have one draw-count identity")
}

audit <- list(
  schema_version = "1",
  status = "complete_bounded_split_rng_search",
  upstream_commit = "60e0b10f0e967fbada08ee6c164fc7c97932c57b",
  seed = 1234L,
  rng_kind = as.list(reference$rng_kind),
  searched_name_draws = list(
    minimum = draw_min,
    maximum = draw_max,
    inclusive = TRUE,
    candidates = draw_max - draw_min + 1L
  ),
  matching_rule = list(
    loader_counts = named_counts(expected$loader_counts),
    sample_counts = named_counts(expected$sample_counts),
    training_age_mean = expected$training_age_mean,
    training_age_sd = expected$training_age_sd,
    age_absolute_tolerance = age_tolerance
  ),
  matching_name_draws = as.integer(matching_draws),
  released_script = list(
    identified_name_draws = CLUDA_RELEASED_NAME_DRAWS,
    pre_split_rng_sha256 = CLUDA_RELEASED_PRE_SPLIT_RNG_SHA256,
    trace_name_draws = trace_draws,
    signature = candidate_signatures[[CLUDA_RELEASED_NAME_DRAWS - draw_min + 1L]]
  ),
  paper_evaluated_state = list(
    identified_name_draws = CLUDA_PAPER_NAME_DRAWS,
    pre_split_rng_sha256 = CLUDA_PAPER_PRE_SPLIT_RNG_SHA256,
    signature = candidate_signatures[[CLUDA_PAPER_NAME_DRAWS - draw_min + 1L]],
    relation_to_released_state = paste(
      "exactly 35 additional RICU temporary-name draws; this is equivalent",
      "to one additional 41-channel time-series loader invocation and is not",
      "evidence that the authors literally made that invocation"
    )
  ),
  runtime_probes = historical,
  retained_data = "aggregate signatures and cryptographic hashes only; no patient rows"
)

temporary <- paste0(output_path, ".tmp")
jsonlite::write_json(audit, temporary, auto_unbox = TRUE, pretty = TRUE, digits = 16)
if (!file.rename(temporary, output_path)) {
  stop("could not atomically install split RNG search audit")
}
