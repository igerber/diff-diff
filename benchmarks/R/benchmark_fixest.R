#!/usr/bin/env Rscript
# Benchmark: Basic DiD / TWFE (R `fixest` package)
#
# Usage:
#   Rscript benchmark_fixest.R --data path/to/data.csv --output path/to/results.json

library(fixest)
library(jsonlite)
library(data.table)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

parse_args <- function(args) {
  result <- list(
    data = NULL,
    output = NULL,
    cluster = "unit",
    type = "twfe"  # "basic" for simple 2x2, "twfe" for two-way FE
  )

  i <- 1
  while (i <= length(args)) {
    if (args[i] == "--data") {
      result$data <- args[i + 1]
      i <- i + 2
    } else if (args[i] == "--output") {
      result$output <- args[i + 1]
      i <- i + 2
    } else if (args[i] == "--cluster") {
      result$cluster <- args[i + 1]
      i <- i + 2
    } else if (args[i] == "--type") {
      result$type <- args[i + 1]
      i <- i + 2
    } else {
      i <- i + 1
    }
  }

  if (is.null(result$data) || is.null(result$output)) {
    stop("Usage: Rscript benchmark_fixest.R --data <path> --output <path> [--cluster unit|time] [--type basic|twfe]")
  }

  return(result)
}

config <- parse_args(args)

# Load data
message(sprintf("Loading data from: %s", config$data))
data <- fread(config$data)

# Run benchmark
message(sprintf("Running %s DiD estimation...", toupper(config$type)))
start_time <- Sys.time()

# Basic 2x2 DiD: outcome ~ treated * post
# This matches Python's DifferenceInDifferences with formula
model <- feols(
  outcome ~ treated * post,
  data = data,
  cluster = config$cluster
)

# The treatment effect is the interaction coefficient
coef_name <- "treated:post"

estimation_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

# Extract results
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
  estimator = "fixest::feols",
  type = config$type,
  cluster = config$cluster,

  # Treatment effect
  att = unname(att),
  se = unname(att_se),
  pvalue = unname(att_pval),
  ci_lower = unname(att_ci[1]),
  ci_upper = unname(att_ci[2]),
  coef_name = coef_name,

  # All coefficients (for basic model)
  coefficients = if (config$type == "basic") {
    data.frame(
      term = names(coefs),
      estimate = unname(coefs),
      se = unname(ses),
      pvalue = unname(pvals)
    )
  } else {
    NULL
  },

  # Model statistics
  model_stats = list(
    r_squared = summary(model)$r2,
    adj_r_squared = summary(model)$adj.r2,
    n_obs = model$nobs,
    n_clusters = model$nobs.origin
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
    n_units = length(unique(data$unit)),
    n_periods = length(unique(data$time)),
    n_obs = nrow(data)
  )
)

# Write output
message(sprintf("Writing results to: %s", config$output))
write_json(results, config$output, auto_unbox = TRUE, pretty = TRUE, digits = 10)

message(sprintf("Completed in %.3f seconds", estimation_time))
