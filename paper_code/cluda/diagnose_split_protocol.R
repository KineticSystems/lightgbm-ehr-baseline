args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) {
  stop("usage: diagnose_split_protocol.R COMPACT_STAY_CSV")
}

suppressPackageStartupMessages(library(caret))
frame <- read.csv(args[[1]], stringsAsFactors = FALSE)

run_split <- function(rng_version) {
  RNGversion(rng_version)
  set.seed(1234)
  train_index <- createDataPartition(
    frame$hospital_expire_flag, p = .7, list = FALSE
  )
  remaining <- frame$hospital_expire_flag[-train_index]
  val_index <- createDataPartition(remaining, p = .5, list = FALSE)
  assignment <- rep("test", nrow(frame))
  assignment[train_index] <- "train"
  nontrain <- which(!(seq_len(nrow(frame)) %in% train_index))
  assignment[nontrain[val_index]] <- "val"

  eligible <- frame$length >= 4L
  split_order <- c("train", "val", "test")
  loader_counts <- vapply(
    split_order, function(value) sum(eligible & assignment == value), integer(1L)
  )
  sample_counts <- vapply(
    split_order,
    function(value) sum(pmax(frame$length[assignment == value] - 3L, 0L)),
    numeric(1L)
  )
  train_ages <- frame$age[assignment == "train"]
  released_match <- mean(assignment == frame$released_split)
  list(
    rng_version = rng_version,
    loader_counts = unname(loader_counts),
    sample_counts = unname(sample_counts),
    age_mean = mean(train_ages),
    age_sd = sd(train_ages),
    released_membership_fraction = released_match
  )
}

versions <- c("3.5.0", "3.6.0", "4.0.0", "4.1.3")
jsonlite::write_json(
  lapply(versions, run_split), stdout(), auto_unbox = TRUE, pretty = TRUE
)
