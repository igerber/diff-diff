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

parse_bool <- function(x, flag) {
  v <- tolower(x)
  if (v %in% c("true", "t", "1", "yes")) return(TRUE)
  if (v %in% c("false", "f", "0", "no")) return(FALSE)
  stop(sprintf("Invalid boolean for %s: '%s' (use true/false)", flag, x))
}

parse_args <- function(args) {
  result <- list(
    data = NULL,
    output = NULL,
    warmup = FALSE,
    skip_jackknife = FALSE
  )

  i <- 1
  while (i <= length(args)) {
    if (args[i] == "--data") {
      result$data <- args[i + 1]
      i <- i + 2
    } else if (args[i] == "--output") {
      result$output <- args[i + 1]
      i <- i + 2
    } else if (args[i] == "--warmup") {
      result$warmup <- parse_bool(args[i + 1], "--warmup")
      i <- i + 2
    } else if (args[i] == "--skip-jackknife") {
      result$skip_jackknife <- parse_bool(args[i + 1], "--skip-jackknife")
      i <- i + 2
    } else {
      # Unknown flags used to be silently skipped, which turned typos into
      # silent protocol changes. Fail loudly instead.
      stop(sprintf("Unknown flag: %s", args[i]))
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

# Warm-up: run the FULL timed pipeline (estimation + placebo vcov) once
# untimed so byte-compiler JIT and first-call setup stay out of the window.
if (config$warmup) {
  message("Warm-up fit (untimed)...")
  tau_warm <- synthdid_estimate(setup$Y, setup$N0, setup$T0)
  invisible(vcov(tau_warm, method = "placebo"))
  rm(tau_warm)
}

# Run benchmark
message("Running Synthetic DiD estimation...")
start_time <- Sys.time()

tau_hat <- synthdid_estimate(setup$Y, setup$N0, setup$T0)

estimation_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

# Get weights
weights <- attr(tau_hat, "weights")
unit_weights <- weights$omega
time_weights <- weights$lambda

# Compute SE via placebo
message("Computing standard errors...")
se_start <- Sys.time()
se_matrix <- vcov(tau_hat, method = "placebo")
se <- as.numeric(sqrt(se_matrix[1, 1]))  # Extract scalar SE
se_time <- as.numeric(difftime(Sys.time(), se_start, units = "secs"))

# Compute SE via jackknife (Algorithm 3). Not part of total_seconds (which
# is estimation + placebo SE, matching the Python arm); skippable because it
# wastes hours of wall-clock at large scales in timing-focused runs.
if (config$skip_jackknife) {
  message("Skipping jackknife standard errors (--skip-jackknife)")
  se_jackknife <- NA_real_
  se_jk_time <- NA_real_
} else {
  message("Computing jackknife standard errors...")
  se_jk_start <- Sys.time()
  se_jk_matrix <- vcov(tau_hat, method = "jackknife")
  se_jackknife <- as.numeric(sqrt(se_jk_matrix[1, 1]))
  se_jk_time <- as.numeric(difftime(Sys.time(), se_jk_start, units = "secs"))
}

total_time <- estimation_time + se_time  # placebo only, matches `se` field

# Compute noise level and regularization (to match Python's auto-computed values)
N0 <- setup$N0
T0 <- setup$T0
N1 <- nrow(setup$Y) - N0
T1 <- ncol(setup$Y) - T0
noise_level <- sd(apply(setup$Y[1:N0, 1:T0], 1, diff))
zeta_omega <- ((N1 * T1)^(1/4)) * noise_level
zeta_lambda <- 1e-6 * noise_level

# Format output
results <- list(
  estimator = "synthdid::synthdid_estimate",

  # Point estimate and SE
  att = as.numeric(tau_hat),
  se = se,
  se_jackknife = se_jackknife,

  # Weights (full precision; omega is in panel control-row order, lambda in
  # pre-period column order - emit the ids so comparison aligns by key)
  unit_weights = as.numeric(unit_weights),
  unit_weight_ids = rownames(setup$Y)[seq_len(N0)],
  time_weights = as.numeric(time_weights),
  time_weight_ids = colnames(setup$Y)[seq_len(T0)],

  # Regularization parameters
  noise_level = noise_level,
  zeta_omega = zeta_omega,
  zeta_lambda = zeta_lambda,

  # Timing
  timing = list(
    estimation_seconds = estimation_time,
    se_placebo_seconds = se_time,
    se_jackknife_seconds = se_jk_time,
    total_seconds = total_time
  ),

  # Metadata
  metadata = list(
    r_version = R.version.string,
    synthdid_version = as.character(packageVersion("synthdid")),
    dt_threads = data.table::getDTthreads(),
    warmup = config$warmup,
    skip_jackknife = config$skip_jackknife,
    n_control = N0,
    n_treated = N1,
    n_pre_periods = T0,
    n_post_periods = T1
  )
)

# Write output
message(sprintf("Writing results to: %s", config$output))
write_json(results, config$output, auto_unbox = TRUE, pretty = TRUE, digits = 10)

message(sprintf("Completed in %.3f seconds", total_time))
