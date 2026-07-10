#!/usr/bin/env Rscript
# Benchmark: MultiPeriodDiD event study (R `fixest` package)
#
# Usage:
#   Rscript benchmark_multiperiod.R --data path/to/data.csv --output path/to/results.json \
#       --n-pre 4 --n-post 4

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
    cluster = "unit",
    n_pre = NULL,
    n_post = NULL,
    reference_period = NULL,
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
    } else if (args[i] == "--cluster") {
      result$cluster <- args[i + 1]
      i <- i + 2
    } else if (args[i] == "--n-pre") {
      result$n_pre <- as.integer(args[i + 1])
      i <- i + 2
    } else if (args[i] == "--n-post") {
      result$n_post <- as.integer(args[i + 1])
      i <- i + 2
    } else if (args[i] == "--reference-period") {
      result$reference_period <- as.integer(args[i + 1])
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
    stop("Usage: Rscript benchmark_multiperiod.R --data <path> --output <path> --n-pre <int> --n-post <int>")
  }
  if (is.null(result$n_pre) || is.null(result$n_post)) {
    stop("--n-pre and --n-post are required")
  }

  # Default reference period: last pre-period
  if (is.null(result$reference_period)) {
    result$reference_period <- result$n_pre
  }

  return(result)
}

config <- parse_args(args)

# Load data
message(sprintf("Loading data from: %s", config$data))
data <- fread(config$data)

ref_period <- config$reference_period
message(sprintf("Reference period: %d", ref_period))
message(sprintf("n_pre: %d, n_post: %d", config$n_pre, config$n_post))

# Create factor for time with reference level
data[, time_f := relevel(factor(time), ref = as.character(ref_period))]

# Warm-up: run the full estimation once untimed so byte-compiler JIT and
# first-call setup costs stay out of the timing window.
cluster_formula_warm <- as.formula(paste0("~", config$cluster))
if (config$warmup) {
  message("Warm-up fit (untimed)...")
  invisible(feols(outcome ~ treated * time_f | unit, data = data,
                  cluster = cluster_formula_warm))
}

# Run benchmark
message("Running MultiPeriodDiD estimation (fixest::feols)...")
start_time <- Sys.time()

# Regression: outcome ~ treated * time_f | unit, clustered SEs
# With | unit, fixest absorbs unit fixed effects. The unit-invariant 'treated'
# main effect is collinear with unit FE and is absorbed automatically.
# Interaction coefficients treated:time_fK remain identified.
cluster_formula <- as.formula(paste0("~", config$cluster))
model <- feols(outcome ~ treated * time_f | unit, data = data, cluster = cluster_formula)

estimation_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

# Extract all coefficients and SEs
coefs <- coef(model)
ses <- se(model)
vcov_mat <- vcov(model)

# Extract interaction coefficients (treated:time_fK for each non-reference K)
interaction_mask <- grepl("^treated:time_f", names(coefs))
interaction_names <- names(coefs)[interaction_mask]
interaction_coefs <- coefs[interaction_mask]
interaction_ses <- ses[interaction_mask]

message(sprintf("Found %d interaction coefficients", length(interaction_names)))

# Build period effects list
all_periods <- sort(unique(data$time))
period_effects <- list()

for (i in seq_along(interaction_names)) {
  coef_name <- interaction_names[i]
  # Extract period value from coefficient name "treated:time_fK"
  period_val <- as.integer(sub("treated:time_f", "", coef_name))
  event_time <- period_val - ref_period

  period_effects[[i]] <- list(
    period = period_val,
    event_time = event_time,
    att = unname(interaction_coefs[i]),
    se = unname(interaction_ses[i])
  )
}

# Compute average ATT across post-periods (covariance-aware SE)
post_period_names <- c()
for (coef_name in interaction_names) {
  period_val <- as.integer(sub("treated:time_f", "", coef_name))
  if (period_val > config$n_pre) {
    post_period_names <- c(post_period_names, coef_name)
  }
}

n_post_periods <- length(post_period_names)
message(sprintf("Post-period interaction coefficients: %d", n_post_periods))

if (n_post_periods > 0) {
  avg_att <- mean(coefs[post_period_names])
  vcov_sub <- vcov_mat[post_period_names, post_period_names, drop = FALSE]
  avg_se <- sqrt(sum(vcov_sub) / n_post_periods^2)
  # NaN guard: match registry convention (REGISTRY.md lines 179-183)
  if (is.finite(avg_se) && avg_se > 0) {
    avg_t <- avg_att / avg_se
    avg_pval <- 2 * pt(abs(avg_t), df = model$nobs - length(coefs), lower.tail = FALSE)
    avg_ci_lower <- avg_att - qt(0.975, df = model$nobs - length(coefs)) * avg_se
    avg_ci_upper <- avg_att + qt(0.975, df = model$nobs - length(coefs)) * avg_se
  } else {
    avg_t <- NA
    avg_pval <- NA
    avg_ci_lower <- NA
    avg_ci_upper <- NA
  }
} else {
  avg_att <- NA
  avg_se <- NA
  avg_pval <- NA
  avg_ci_lower <- NA
  avg_ci_upper <- NA
}

message(sprintf("Average ATT: %.6f", avg_att))
message(sprintf("Average SE:  %.6f", avg_se))

# Format output
results <- list(
  estimator = "fixest::feols (multiperiod)",
  cluster = config$cluster,

  # Average treatment effect
  att = avg_att,
  se = avg_se,
  pvalue = avg_pval,
  ci_lower = avg_ci_lower,
  ci_upper = avg_ci_upper,

  # Reference period
  reference_period = ref_period,

  # Period-level effects
  period_effects = period_effects,

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
    n_periods = length(unique(data$time)),
    n_obs = nrow(data),
    n_pre = config$n_pre,
    n_post = config$n_post
  )
)

# Write output
message(sprintf("Writing results to: %s", config$output))
write_json(results, config$output, auto_unbox = TRUE, pretty = TRUE, digits = 10)

message(sprintf("Completed in %.3f seconds", estimation_time))
