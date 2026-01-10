#!/usr/bin/env Rscript
# Benchmark: HonestDiD Sensitivity Analysis (R `HonestDiD` package)
#
# Usage:
#   Rscript benchmark_honest.R --beta path/to/beta.json --sigma path/to/sigma.csv --output path/to/results.json

library(HonestDiD)
library(jsonlite)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

parse_args <- function(args) {
  result <- list(
    beta = NULL,
    sigma = NULL,
    output = NULL,
    num_pre = 4,
    num_post = 1,
    method = "FLCI",
    m_grid = "0,0.5,1,1.5,2"
  )

  i <- 1
  while (i <= length(args)) {
    if (args[i] == "--beta") {
      result$beta <- args[i + 1]
      i <- i + 2
    } else if (args[i] == "--sigma") {
      result$sigma <- args[i + 1]
      i <- i + 2
    } else if (args[i] == "--output") {
      result$output <- args[i + 1]
      i <- i + 2
    } else if (args[i] == "--num-pre") {
      result$num_pre <- as.integer(args[i + 1])
      i <- i + 2
    } else if (args[i] == "--num-post") {
      result$num_post <- as.integer(args[i + 1])
      i <- i + 2
    } else if (args[i] == "--method") {
      result$method <- args[i + 1]
      i <- i + 2
    } else if (args[i] == "--m-grid") {
      result$m_grid <- args[i + 1]
      i <- i + 2
    } else {
      i <- i + 1
    }
  }

  if (is.null(result$beta) || is.null(result$sigma) || is.null(result$output)) {
    stop("Usage: Rscript benchmark_honest.R --beta <path> --sigma <path> --output <path> [--num-pre N] [--num-post N] [--method FLCI|Conditional] [--m-grid 0,0.5,1]")
  }

  return(result)
}

config <- parse_args(args)

# Load inputs
message(sprintf("Loading beta from: %s", config$beta))
beta <- fromJSON(config$beta)

message(sprintf("Loading sigma from: %s", config$sigma))
sigma <- as.matrix(read.csv(config$sigma, header = FALSE))

# Parse M grid
m_grid <- as.numeric(strsplit(config$m_grid, ",")[[1]])

message(sprintf("Running HonestDiD with %d pre-periods, %d post-periods", config$num_pre, config$num_post))
message(sprintf("M grid: %s", paste(m_grid, collapse = ", ")))

# Run Relative Magnitudes (Delta RM) sensitivity analysis
message("Running Delta RM sensitivity analysis...")
start_time_rm <- Sys.time()

rm_results <- createSensitivityResults_relativeMagnitudes(
  betahat = beta,
  sigma = sigma,
  numPrePeriods = config$num_pre,
  numPostPeriods = config$num_post,
  Mbarvec = m_grid,
  method = config$method
)

rm_time <- as.numeric(difftime(Sys.time(), start_time_rm, units = "secs"))

# Run Smoothness (Delta SD) sensitivity analysis
message("Running Delta SD sensitivity analysis...")
start_time_sd <- Sys.time()

# For Delta SD, M values are typically smaller
sd_m_grid <- m_grid / 10  # Scale down for smoothness

sd_results <- createSensitivityResults(
  betahat = beta,
  sigma = sigma,
  numPrePeriods = config$num_pre,
  numPostPeriods = config$num_post,
  Mvec = sd_m_grid,
  method = config$method
)

sd_time <- as.numeric(difftime(Sys.time(), start_time_sd, units = "secs"))

total_time <- rm_time + sd_time

# Format output
results <- list(
  estimator = "HonestDiD",
  method = config$method,

  # Delta RM results
  delta_rm = list(
    M_grid = m_grid,
    lb = rm_results$lb,
    ub = rm_results$ub
  ),

  # Delta SD results
  delta_sd = list(
    M_grid = sd_m_grid,
    lb = sd_results$lb,
    ub = sd_results$ub
  ),

  # Configuration
  config = list(
    num_pre_periods = config$num_pre,
    num_post_periods = config$num_post,
    beta = beta,
    sigma_diag = diag(sigma)
  ),

  # Timing
  timing = list(
    delta_rm_seconds = rm_time,
    delta_sd_seconds = sd_time,
    total_seconds = total_time
  ),

  # Metadata
  metadata = list(
    r_version = R.version.string,
    honestdid_version = as.character(packageVersion("HonestDiD"))
  )
)

# Write output
message(sprintf("Writing results to: %s", config$output))
write_json(results, config$output, auto_unbox = TRUE, pretty = TRUE, digits = 10)

message(sprintf("Completed in %.3f seconds", total_time))
