#!/usr/bin/env Rscript
# Benchmark: Synthetic DiD (R `synthdid` package)
#
# Usage:
#   Rscript benchmark_synthdid.R --data path/to/data.csv --output path/to/results.json

library(synthdid)
library(jsonlite)
library(data.table)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

parse_args <- function(args) {
  result <- list(
    data = NULL,
    output = NULL
  )

  i <- 1
  while (i <= length(args)) {
    if (args[i] == "--data") {
      result$data <- args[i + 1]
      i <- i + 2
    } else if (args[i] == "--output") {
      result$output <- args[i + 1]
      i <- i + 2
    } else {
      i <- i + 1
    }
  }

  if (is.null(result$data) || is.null(result$output)) {
    stop("Usage: Rscript benchmark_synthdid.R --data <path> --output <path>")
  }

  return(result)
}

config <- parse_args(args)

# Load data
message(sprintf("Loading data from: %s", config$data))
data <- fread(config$data)

# synthdid requires panel.matrices format
# Data must have: unit, time, outcome, treated columns
message("Preparing data for synthdid...")

# Create treatment indicator (1 if treated in post period)
# synthdid expects 0/1 treatment indicator
data[, treatment := as.integer(treated == 1 & post == 1)]

# Convert to panel.matrices format
setup <- panel.matrices(
  as.data.frame(data),
  unit = "unit",
  time = "time",
  outcome = "outcome",
  treatment = "treatment"
)

# Run benchmark
message("Running Synthetic DiD estimation...")
start_time <- Sys.time()

tau_hat <- synthdid_estimate(setup$Y, setup$N0, setup$T0)

estimation_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

# Get weights
weights <- attr(tau_hat, "weights")
unit_weights <- weights$omega
time_weights <- weights$lambda

# Compute SE via placebo (jackknife)
message("Computing standard errors...")
se_start <- Sys.time()
se_matrix <- vcov(tau_hat, method = "placebo")
se <- as.numeric(sqrt(se_matrix[1, 1]))  # Extract scalar SE
se_time <- as.numeric(difftime(Sys.time(), se_start, units = "secs"))

total_time <- estimation_time + se_time

# Format output
results <- list(
  estimator = "synthdid::synthdid_estimate",

  # Point estimate and SE
  att = as.numeric(tau_hat),
  se = se,

  # Weights
  unit_weights = as.numeric(unit_weights),
  time_weights = as.numeric(time_weights),

  # Timing
  timing = list(
    estimation_seconds = estimation_time,
    se_seconds = se_time,
    total_seconds = total_time
  ),

  # Metadata
  metadata = list(
    r_version = R.version.string,
    synthdid_version = as.character(packageVersion("synthdid")),
    n_control = setup$N0,
    n_treated = nrow(setup$Y) - setup$N0,
    n_pre_periods = setup$T0,
    n_post_periods = ncol(setup$Y) - setup$T0
  )
)

# Write output
message(sprintf("Writing results to: %s", config$output))
write_json(results, config$output, auto_unbox = TRUE, pretty = TRUE, digits = 10)

message(sprintf("Completed in %.3f seconds", total_time))
