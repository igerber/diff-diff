#!/usr/bin/env Rscript
# Benchmark: Triple Difference (R `triplediff` package)
#
# This uses triplediff::ddd() with panel=FALSE (repeated cross-section mode),
# matching the Python TripleDifference estimator's approach.
#
# Usage:
#   Rscript benchmark_triplediff.R --data path/to/data.csv --output path/to/results.json \
#     [--method dr|reg|ipw] [--covariates true|false]

library(triplediff)
library(jsonlite)
library(data.table)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

parse_args <- function(args) {
  result <- list(
    data = NULL,
    output = NULL,
    method = "dr",
    covariates = FALSE
  )

  i <- 1
  while (i <= length(args)) {
    if (args[i] == "--data") {
      result$data <- args[i + 1]
      i <- i + 2
    } else if (args[i] == "--output") {
      result$output <- args[i + 1]
      i <- i + 2
    } else if (args[i] == "--method") {
      result$method <- args[i + 1]
      i <- i + 2
    } else if (args[i] == "--covariates") {
      result$covariates <- tolower(args[i + 1]) == "true"
      i <- i + 2
    } else {
      i <- i + 1
    }
  }

  if (is.null(result$data) || is.null(result$output)) {
    stop("Usage: Rscript benchmark_triplediff.R --data <path> --output <path> [--method dr|reg|ipw] [--covariates true|false]")
  }

  return(result)
}

config <- parse_args(args)

# Load data
message(sprintf("Loading data from: %s", config$data))
data <- fread(config$data)

# Build covariate formula
cov_cols <- grep("^cov", names(data), value = TRUE)
if (config$covariates && length(cov_cols) > 0) {
  xformla <- as.formula(paste("~", paste(cov_cols, collapse = "+")))
  message(sprintf("Using covariates: %s", paste(cov_cols, collapse = ", ")))
} else {
  xformla <- ~1
  message("No covariates")
}

# Run benchmark
message(sprintf("Running DDD estimation (method=%s, panel=FALSE)...", config$method))
timing <- system.time({
  res <- ddd(
    yname = "y",
    tname = "time",
    idname = "id",
    gname = "state",
    pname = "partition",
    data = data,
    control_group = "nevertreated",
    panel = FALSE,
    xformla = xformla,
    est_method = config$method,
    boot = FALSE
  )
})

# Collect results
output <- list(
  ATT = res$ATT,
  se = res$se,
  lci = res$lci,
  uci = res$uci,
  method = config$method,
  covariates = config$covariates,
  n_obs = nrow(data),
  elapsed_seconds = timing["elapsed"]
)

# Write results
message(sprintf("Writing results to: %s", config$output))
write(toJSON(output, pretty = TRUE, auto_unbox = TRUE, digits = 15), config$output)

message("Done.")
message(sprintf("  ATT = %.6f", res$ATT))
message(sprintf("  SE  = %.6f", res$se))
message(sprintf("  Time: %.3fs", timing["elapsed"]))
