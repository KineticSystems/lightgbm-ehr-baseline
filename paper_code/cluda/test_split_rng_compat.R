paper_root <- normalizePath(
  dirname(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[[1]]))
)
source(file.path(paper_root, "cluda_los_filter_fast.R"), local = TRUE)
source(file.path(paper_root, "cluda_split_rng_compat.R"), local = TRUE)

set.seed(1234)
for (index in seq_len(CLUDA_RELEASED_NAME_DRAWS)) {
  ricu:::new_names()
}
if (!identical(cluda_rng_sha256(), CLUDA_RELEASED_PRE_SPLIT_RNG_SHA256)) {
  stop("test runtime did not reproduce the released-script pre-split state")
}
audit <- cluda_apply_paper_split_rng_compat()
if (!identical(audit$status, "paper_artifact_triangulated_split_rng_state") ||
    !identical(
      audit$paper_evaluated_pre_split_rng_sha256,
      CLUDA_PAPER_PRE_SPLIT_RNG_SHA256
    ) ||
    !identical(cluda_rng_sha256(), CLUDA_PAPER_PRE_SPLIT_RNG_SHA256)) {
  stop("split RNG compatibility transition did not reach the paper boundary")
}

set.seed(1234)
error <- tryCatch(
  {
    cluda_apply_paper_split_rng_compat()
    NULL
  },
  error = identity
)
if (!inherits(error, "error")) {
  stop("split RNG compatibility did not fail closed on a changed input state")
}
cat("CLUDA split RNG compatibility tests passed\n")
