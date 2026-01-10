#!/usr/bin/env Rscript
# Benchmark: Callaway-Sant'Anna Estimator (R `did` package)
#
# Usage:
#   Rscript benchmark_did.R --data path/to/data.csv --output path/to/results.json

library(did)
library(jsonlite)
library(data.table)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

parse_args <- function(args) {
  result <- list(
    data = NULL,
    output = NULL,
    method = "dr",
    control_group = "nevertreated"
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
    } else if (args[i] == "--control-group") {
      result$control_group <- args[i + 1]
      i <- i + 2
    } else {
      i <- i + 1
    }
  }

  if (is.null(result$data) || is.null(result$output)) {
    stop("Usage: Rscript benchmark_did.R --data <path> --output <path> [--method dr|ipw|reg] [--control-group nevertreated|notyettreated]")
  }

  return(result)
}

config <- parse_args(args)

# Load data
message(sprintf("Loading data from: %s", config$data))
data <- fread(config$data)

# Ensure proper column types
data[, unit := as.integer(unit)]
data[, time := as.integer(time)]

# R's did package expects first_treat=Inf for never-treated units
# Our Python implementation uses first_treat=0 for never-treated
# Convert 0 to Inf to match R's expectation
data[, first_treat := as.numeric(first_treat)]
data[first_treat == 0, first_treat := Inf]
message(sprintf("Never-treated units (first_treat=Inf): %d", sum(is.infinite(data$first_treat))))

# Run benchmark
message("Running Callaway-Sant'Anna estimation...")
start_time <- Sys.time()

out <- att_gt(
  yname = "outcome",
  tname = "time",
  idname = "unit",
  gname = "first_treat",
  xformla = NULL,
  data = data,
  est_method = config$method,
  control_group = config$control_group,
  bstrap = FALSE,  # Use analytical SEs for speed
  cband = FALSE
)

estimation_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

# Aggregate results
message("Aggregating results...")
agg_start <- Sys.time()

agg_simple <- aggte(out, type = "simple", bstrap = FALSE, cband = FALSE)
agg_dynamic <- aggte(out, type = "dynamic", bstrap = FALSE, cband = FALSE)
agg_group <- aggte(out, type = "group", bstrap = FALSE, cband = FALSE)

aggregation_time <- as.numeric(difftime(Sys.time(), agg_start, units = "secs"))
total_time <- estimation_time + aggregation_time

# Format output
results <- list(
  estimator = "did::att_gt",
  method = config$method,
  control_group = config$control_group,

  # Overall ATT
  overall_att = agg_simple$overall.att,
  overall_se = agg_simple$overall.se,

  # Group-time effects
  group_time_effects = data.frame(
    group = out$group,
    time = out$t,
    att = out$att,
    se = out$se
  ),

  # Event study (dynamic aggregation)
  event_study = data.frame(
    event_time = agg_dynamic$egt,
    att = agg_dynamic$att.egt,
    se = agg_dynamic$se.egt
  ),

  # Group aggregation
  group_effects = data.frame(
    group = agg_group$egt,
    att = agg_group$att.egt,
    se = agg_group$se.egt
  ),

  # Timing
  timing = list(
    estimation_seconds = estimation_time,
    aggregation_seconds = aggregation_time,
    total_seconds = total_time
  ),

  # Metadata
  metadata = list(
    r_version = R.version.string,
    did_version = as.character(packageVersion("did")),
    n_units = length(unique(data$unit)),
    n_periods = length(unique(data$time)),
    n_obs = nrow(data)
  )
)

# Write output
message(sprintf("Writing results to: %s", config$output))
write_json(results, config$output, auto_unbox = TRUE, pretty = TRUE, digits = 10)

message(sprintf("Completed in %.3f seconds", total_time))
