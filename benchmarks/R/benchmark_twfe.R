#!/usr/bin/env Rscript
# Benchmark: Two-Way Fixed Effects (R `fixest` package with absorbed FE)
#
# This uses fixest::feols() with absorbed unit + post FE and unit-level clustering,
# matching the Python TwoWayFixedEffects estimator's approach.
#
# Usage:
#   Rscript benchmark_twfe.R --data path/to/data.csv --output path/to/results.json

library(fixest)
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
    warmup = FALSE
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
    } else {
      # Unknown flags used to be silently skipped, which turned typos into
      # silent protocol changes. Fail loudly instead.
      stop(sprintf("Unknown flag: %s", args[i]))
    }
  }

  if (is.null(result$data) || is.null(result$output)) {
    stop("Usage: Rscript benchmark_twfe.R --data <path> --output <path>")
  }

  return(result)
}

config <- parse_args(args)

# Load data
message(sprintf("Loading data from: %s", config$data))
data <- fread(config$data)

# Warm-up: run the full estimation once untimed so byte-compiler JIT and
# first-call setup costs stay out of the timing window.
if (config$warmup) {
  message("Warm-up fit (untimed)...")
  invisible(feols(outcome ~ treated:post | unit + post, data = data, cluster = ~unit))
}

# Run benchmark
message("Running TWFE estimation with absorbed FE...")
start_time <- Sys.time()

# TWFE with absorbed unit + post fixed effects, clustered at unit level
# This matches Python's TwoWayFixedEffects:
#   - Within-transformation removes unit and time (post) FE
#   - Cluster-robust SE at unit level (automatic)
model <- feols(
  outcome ~ treated:post | unit + post,
  data = data,
  cluster = ~unit
)

estimation_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

# Extract results
coef_name <- "treated:post"
coefs <- coef(model)
ses <- se(model)
pvals <- pvalue(model)
ci <- confint(model)

# Find the treatment effect coefficient
if (coef_name %in% names(coefs)) {
  att <- coefs[coef_name]
  att_se <- ses[coef_name]
  att_pval <- pvals[coef_name]
  att_ci <- ci[coef_name, ]
} else {
  # Try alternative name formats
  idx <- grep("treated.*post|post.*treated", names(coefs))
  if (length(idx) > 0) {
    att <- coefs[idx[1]]
    att_se <- ses[idx[1]]
    att_pval <- pvals[idx[1]]
    att_ci <- ci[idx[1], ]
    coef_name <- names(coefs)[idx[1]]
  } else {
    stop("Could not find treatment effect coefficient")
  }
}

# Format output
results <- list(
  estimator = "fixest::feols (absorbed FE)",
  cluster = "unit",

  # Treatment effect
  att = unname(att),
  se = unname(att_se),
  pvalue = unname(att_pval),
  ci_lower = unname(att_ci[1]),
  ci_upper = unname(att_ci[2]),
  coef_name = coef_name,

  # Model statistics
  model_stats = list(
    r_squared = summary(model)$r2,
    adj_r_squared = summary(model)$adj.r2,
    n_obs = model$nobs
  ),

  # Timing
  timing = list(
    estimation_seconds = estimation_time,
    total_seconds = estimation_time
  ),

  # Metadata
  metadata = list(
    r_version = R.version.string,
    fixest_version = as.character(packageVersion("fixest")),
    nthreads = fixest::getFixest_nthreads(),
    dt_threads = data.table::getDTthreads(),
    warmup = config$warmup,
    n_units = length(unique(data$unit)),
    n_periods = length(unique(data$post)),
    n_obs = nrow(data)
  )
)

# Write output
message(sprintf("Writing results to: %s", config$output))
write_json(results, config$output, auto_unbox = TRUE, pretty = TRUE, digits = 15)

message(sprintf("Completed in %.3f seconds", estimation_time))
