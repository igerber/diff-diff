#!/usr/bin/env Rscript
# Benchmark: Callaway-Sant'Anna Estimator (R `did` package)
#
# Usage:
#   Rscript benchmark_did.R --data path/to/data.csv --output path/to/results.json \
#     [--method dr|ipw|reg] [--control-group nevertreated|notyettreated] \
#     [--xformla "~ x1 + x2"] [--bstrap true|false] [--biters N] \
#     [--cband true|false] [--pl true|false] [--cores N] \
#     [--faster-mode true|false]
#
# Defaults reproduce the historical behavior (analytical SEs, no covariates,
# single core), so existing callers (benchmarks/run_benchmarks.py) are
# unaffected. The optional flags exist for the R-yardstick arms: bootstrap
# inference (bstrap/biters/cband applied at BOTH att_gt and aggte), covariate
# formulas, and did's parallel processing (pl/cores).

library(did)
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
    method = "dr",
    control_group = "nevertreated",
    xformla = NULL,
    bstrap = FALSE,
    biters = 1000L,
    cband = FALSE,
    pl = FALSE,
    cores = 1L,
    faster_mode = NULL  # NULL -> use did's own default for this version
  )

  i <- 1
  while (i <= length(args)) {
    flag <- args[i]
    if (i + 1 > length(args) && flag != "") {
      stop(sprintf("Missing value for flag: %s", flag))
    }
    val <- args[i + 1]
    if (flag == "--data") {
      result$data <- val
    } else if (flag == "--output") {
      result$output <- val
    } else if (flag == "--method") {
      result$method <- val
    } else if (flag == "--control-group") {
      result$control_group <- val
    } else if (flag == "--xformla") {
      result$xformla <- as.formula(val)
    } else if (flag == "--bstrap") {
      result$bstrap <- parse_bool(val, flag)
    } else if (flag == "--biters") {
      result$biters <- as.integer(val)
    } else if (flag == "--cband") {
      result$cband <- parse_bool(val, flag)
    } else if (flag == "--pl") {
      result$pl <- parse_bool(val, flag)
    } else if (flag == "--cores") {
      result$cores <- as.integer(val)
    } else if (flag == "--faster-mode") {
      result$faster_mode <- parse_bool(val, flag)
    } else {
      # Unknown flags used to be silently skipped, which turned typos into
      # silent default runs. Fail loudly instead.
      stop(sprintf("Unknown flag: %s", flag))
    }
    i <- i + 2
  }

  if (is.null(result$data) || is.null(result$output)) {
    stop("Usage: Rscript benchmark_did.R --data <path> --output <path> [--method dr|ipw|reg] [--control-group nevertreated|notyettreated] [--xformla '~ x1 + x2'] [--bstrap true|false] [--biters N] [--cband true|false] [--pl true|false] [--cores N] [--faster-mode true|false]")
  }
  if (is.na(result$biters) || result$biters <= 0) {
    stop(sprintf("--biters must be a positive integer, got '%s'", result$biters))
  }
  if (is.na(result$cores) || result$cores <= 0) {
    stop(sprintf("--cores must be a positive integer, got '%s'", result$cores))
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

att_gt_args <- list(
  yname = "outcome",
  tname = "time",
  idname = "unit",
  gname = "first_treat",
  xformla = config$xformla,
  data = data,
  est_method = config$method,
  control_group = config$control_group,
  bstrap = config$bstrap,
  biters = config$biters,
  cband = config$cband,
  pl = config$pl,
  cores = config$cores
)
# faster_mode exists in recent did releases only; pass it only when both
# requested and supported, so the script still runs on older installs.
# effective_faster_mode records what actually reached att_gt (vs the
# requested flag) so benchmark artifacts are never mislabeled.
effective_faster_mode <- "did-default"
if (!is.null(config$faster_mode)) {
  if ("faster_mode" %in% names(formals(att_gt))) {
    att_gt_args$faster_mode <- config$faster_mode
    effective_faster_mode <- config$faster_mode
  } else {
    message("faster_mode not supported by this did version; ignoring flag")
    effective_faster_mode <- "unsupported-ignored"
  }
}
out <- do.call(att_gt, att_gt_args)

estimation_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

# Aggregate results (same inference mode as att_gt so the timing arm does
# equal work end-to-end)
message("Aggregating results...")
agg_start <- Sys.time()

agg_simple <- aggte(out, type = "simple", bstrap = config$bstrap,
                    biters = config$biters, cband = FALSE)
agg_dynamic <- aggte(out, type = "dynamic", bstrap = config$bstrap,
                     biters = config$biters, cband = config$cband)
agg_group <- aggte(out, type = "group", bstrap = config$bstrap,
                   biters = config$biters, cband = FALSE)

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

  # Metadata (records the full inference/parallelism config so every
  # yardstick number is reproducible from its own artifact)
  metadata = list(
    r_version = R.version.string,
    did_version = as.character(packageVersion("did")),
    n_units = length(unique(data$unit)),
    n_periods = length(unique(data$time)),
    n_obs = nrow(data),
    xformla = if (is.null(config$xformla)) NULL else deparse(config$xformla),
    bstrap = config$bstrap,
    biters = if (config$bstrap) config$biters else NULL,
    cband = config$cband,
    pl = config$pl,
    cores = config$cores,
    faster_mode = effective_faster_mode,
    blas = tryCatch(sessionInfo()$BLAS, error = function(e) NULL)
  )
)

# Write output
message(sprintf("Writing results to: %s", config$output))
write_json(results, config$output, auto_unbox = TRUE, pretty = TRUE, digits = 10)

message(sprintf("Completed in %.3f seconds", total_time))
