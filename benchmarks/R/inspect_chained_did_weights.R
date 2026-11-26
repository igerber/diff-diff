#!/usr/bin/env Rscript
# Inspect public estimator-callback inputs for the first pair; never estimates ATT.
# Usage: Rscript benchmarks/R/inspect_chained_did_weights.R input.csv output.json
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L, getRversion() == "4.5.2",
          as.character(packageVersion("cdid")) == "1.1.0.0",
          packageDescription("cdid")[["RemoteSha"]] == "1039120668cceabd8111cad82d32ab3927c2b94f")
data <- read.csv(args[[1L]], na.strings = c("", "NA"))
stopifnot(all(c("id", "t", "g", "y", "x") %in% names(data)),
          identical(sort(unique(data$t)), 0:3),
          identical(sort(unique(data$g)), c(0L, 2L)))
observed_counts <- tapply(is.finite(data$y), data$id, sum)
frame <- data[!duplicated(data$id), c("id", "g", "x")]
eligible <- frame[observed_counts[as.character(frame$id)] >= 2, ]
eligible_counts <- table(factor(eligible$g > 0, levels = c(FALSE, TRUE)))
pair <- data[data$t %in% 0:1 & is.finite(data$y), ]
pair_ids <- names(which(table(pair$id) == 2L))
pair_counts <- table(factor(frame$g[as.character(frame$id) %in% pair_ids] > 0,
                            levels = c(FALSE, TRUE)))
expected <- as.numeric(eligible_counts / pair_counts)
stopifnot(all(pair_counts > 0), all(is.finite(expected)))
inspected <- FALSE
probe <- function(y1, y0, D, covariates, i.weights, inffunc = TRUE, ...) {
  counts <- table(factor(D, levels = 0:1))
  stopifnot(identical(as.integer(counts), as.integer(pair_counts)),
            length(i.weights) == length(D),
            all(abs(i.weights - expected[D + 1L]) < 1e-12))
  jsonlite::write_json(list(
    status = "public_inputs_verified_no_estimate", pair = c(0L, 1L),
    cohorts = c(0L, 2L), eligible_counts = as.integer(eligible_counts),
    pair_counts = as.integer(counts), weights = expected,
    max_abs_weight_error = max(abs(i.weights - expected[D + 1L]))
  ), args[[2L]], auto_unbox = TRUE, pretty = TRUE, digits = 17)
  inspected <<- TRUE
  stop("PUBLIC_INPUT_INSPECTION_ONLY", call. = FALSE)
}
tryCatch(cdid::att_gt(
  yname = "y", tname = "t", idname = "id", gname = "g", data = data, frame = frame,
  xformla = ~ x, est_method = probe, control_group = "nevertreated",
  base_period = "universal", mode = "chained", selection = "none",
  gmm_weighting = "identity", pt_assumption = "post", bstrap = FALSE, cband = FALSE
), error = function(e) {
  if (!identical(conditionMessage(e), "PUBLIC_INPUT_INSPECTION_ONLY")) stop(e)
})
stopifnot(inspected)
