#!/usr/bin/env Rscript
# Benchmark: Stacked Difference-in-Differences (Wing, Freedman & Hollingsworth 2024)
#
# Uses the reference implementation functions from:
#   https://github.com/hollina/stacked-did-weights
# embedded directly (not a CRAN package).
# Regression via fixest::feols.
#
# Compares against diff_diff.StackedDiD.
#
# Usage:
#   Rscript benchmark_stacked_did.R --data path/to/data.csv --output path/to/results.json
#   Rscript benchmark_stacked_did.R --data path/to/data.csv --output path/to/results.json --kappa-pre 2 --kappa-post 2

library(fixest)
library(jsonlite)
library(data.table)

# =============================================================================
# Reference implementation functions (from hollina/stacked-did-weights)
# =============================================================================

create_sub_exp <- function(dataset, timeID, groupID, adoptionTime,
                           focalAdoptionTime, kappa_pre, kappa_post) {
  # Copy dataset
  dt_temp <- copy(dataset)

  # Determine earliest and latest time in the data
  minTime <- dt_temp[, min(get(timeID))]
  maxTime <- dt_temp[, max(get(timeID))]

  # Include only treated groups and clean controls (not-yet-treated)
  dt_temp <- dt_temp[
    get(adoptionTime) == focalAdoptionTime |
    get(adoptionTime) > focalAdoptionTime + kappa_post |
    is.na(get(adoptionTime))
  ]

  # Limit to time periods inside the event window
  dt_temp <- dt_temp[
    get(timeID) %in% (focalAdoptionTime - kappa_pre):(focalAdoptionTime + kappa_post)
  ]

  # Make treatment group dummy
  dt_temp[, treat := 0]
  dt_temp[get(adoptionTime) == focalAdoptionTime, treat := 1]

  # Make a post variable
  dt_temp[, post := fifelse(get(timeID) >= focalAdoptionTime, 1, 0)]

  # Make event time variable
  dt_temp[, event_time := get(timeID) - focalAdoptionTime]

  # Check feasibility (IC1)
  dt_temp[, feasible := fifelse(
    focalAdoptionTime - kappa_pre >= minTime &
    focalAdoptionTime + kappa_post <= maxTime, 1, 0
  )]

  # Make a sub experiment ID
  dt_temp[, sub_exp := focalAdoptionTime]

  return(dt_temp)
}

compute_weights <- function(dataset, treatedVar, eventTimeVar, subexpVar) {
  # Create a copy
  stack_dt_temp <- copy(dataset)

  # Step 1: Compute stack-time counts for treated and control
  stack_dt_temp[, `:=`(
    stack_n = .N,
    stack_treat_n = sum(get(treatedVar)),
    stack_control_n = sum(1 - get(treatedVar))
  ), by = get(eventTimeVar)]

  # Step 2: Compute sub_exp-level counts
  stack_dt_temp[, `:=`(
    sub_n = .N,
    sub_treat_n = sum(get(treatedVar)),
    sub_control_n = sum(1 - get(treatedVar))
  ), by = list(get(subexpVar), get(eventTimeVar))]

  # Step 3: Compute sub-experiment share of totals
  stack_dt_temp[, sub_share := sub_n / stack_n]
  stack_dt_temp[, `:=`(
    sub_treat_share = sub_treat_n / stack_treat_n,
    sub_control_share = sub_control_n / stack_control_n
  )]

  # Step 4: Compute weights (aggregate weighting)
  stack_dt_temp[get(treatedVar) == 1, stack_weight := 1]
  stack_dt_temp[get(treatedVar) == 0, stack_weight := sub_treat_share / sub_control_share]

  return(stack_dt_temp)
}

# =============================================================================
# Command line argument parsing
# =============================================================================

args <- commandArgs(trailingOnly = TRUE)

parse_args <- function(args) {
  result <- list(
    data = NULL,
    output = NULL,
    kappa_pre = 2,
    kappa_post = 2
  )

  i <- 1
  while (i <= length(args)) {
    if (args[i] == "--data") {
      result$data <- args[i + 1]
      i <- i + 2
    } else if (args[i] == "--output") {
      result$output <- args[i + 1]
      i <- i + 2
    } else if (args[i] == "--kappa-pre") {
      result$kappa_pre <- as.integer(args[i + 1])
      i <- i + 2
    } else if (args[i] == "--kappa-post") {
      result$kappa_post <- as.integer(args[i + 1])
      i <- i + 2
    } else {
      i <- i + 1
    }
  }

  if (is.null(result$data) || is.null(result$output)) {
    stop("Usage: Rscript benchmark_stacked_did.R --data <path> --output <path> [--kappa-pre N] [--kappa-post N]")
  }

  return(result)
}

config <- parse_args(args)

# =============================================================================
# Load data
# =============================================================================

message(sprintf("Loading data from: %s", config$data))
data <- fread(config$data)

# Ensure proper column types
data[, unit := as.integer(unit)]
data[, time := as.integer(time)]
data[, first_treat := as.integer(first_treat)]

# Convert never-treated (first_treat=0) to NA (R convention)
data[first_treat == 0, first_treat := NA_integer_]

n_units <- length(unique(data$unit))
n_periods <- length(unique(data$time))
message(sprintf("Data: %d units, %d periods, %d obs", n_units, n_periods, nrow(data)))

# Get unique adoption events (excluding never-treated)
events <- sort(unique(data[!is.na(first_treat), first_treat]))
message(sprintf("Adoption events: %s", paste(events, collapse = ", ")))
message(sprintf("Never-treated units: %d", sum(is.na(data[, .SD[1], by = unit]$first_treat))))

kappa_pre <- config$kappa_pre
kappa_post <- config$kappa_post
message(sprintf("kappa_pre=%d, kappa_post=%d", kappa_pre, kappa_post))

# =============================================================================
# Build stacked dataset
# =============================================================================

message("Building stacked dataset...")
start_time <- Sys.time()

sub_experiments <- list()

for (j in events) {
  sub_name <- paste0("sub_", j)
  sub_experiments[[sub_name]] <- create_sub_exp(
    dataset = data,
    timeID = "time",
    groupID = "unit",
    adoptionTime = "first_treat",
    focalAdoptionTime = j,
    kappa_pre = kappa_pre,
    kappa_post = kappa_post
  )
}

# Vertically concatenate
stackfull <- rbindlist(sub_experiments)

# Remove infeasible sub-experiments (IC1)
stacked_data <- stackfull[feasible == 1]

if (nrow(stacked_data) == 0) {
  stop("All sub-experiments are infeasible. Try smaller kappa values.")
}

n_feasible <- length(unique(stacked_data$sub_exp))
message(sprintf("Feasible sub-experiments: %d / %d", n_feasible, length(events)))
message(sprintf("Stacked dataset: %d obs", nrow(stacked_data)))

# =============================================================================
# Compute Q-weights (aggregate weighting)
# =============================================================================

message("Computing Q-weights...")
stacked_data2 <- compute_weights(
  dataset = stacked_data,
  treatedVar = "treat",
  eventTimeVar = "event_time",
  subexpVar = "sub_exp"
)

stack_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
message(sprintf("Stacking + weights completed in %.3f seconds", stack_time))

# =============================================================================
# Run WLS regression (event study)
# =============================================================================

message("Running weighted event study regression...")
reg_start_time <- Sys.time()

# Weighted event study with fixed effects (matches Equation 3 in paper)
# i(event_time, treat, ref = -1) creates treat × event_time interactions, omitting -1
weight_stack <- feols(
  outcome ~ i(event_time, treat, ref = -1) | treat + event_time,
  data = stacked_data2,
  cluster = ~unit,
  weights = stacked_data2$stack_weight
)

reg_time <- as.numeric(difftime(Sys.time(), reg_start_time, units = "secs"))
total_time <- stack_time + reg_time
message(sprintf("Regression completed in %.3f seconds", reg_time))

# =============================================================================
# Extract results
# =============================================================================

# Extract event study coefficients
coef_table <- coeftable(weight_stack)
coef_names <- rownames(coef_table)

# Parse event study effects
event_study <- data.frame(
  event_time = integer(0),
  att = numeric(0),
  se = numeric(0)
)

for (i in seq_len(nrow(coef_table))) {
  name <- coef_names[i]
  # Parse "event_time::X:treat" pattern
  if (grepl("event_time::", name)) {
    et <- as.integer(gsub("event_time::|:treat", "", name))
    event_study <- rbind(event_study, data.frame(
      event_time = et,
      att = coef_table[i, "Estimate"],
      se = coef_table[i, "Std. Error"]
    ))
  }
}

event_study <- event_study[order(event_study$event_time), ]

message("Event study effects:")
for (i in seq_len(nrow(event_study))) {
  message(sprintf("  h=%d: ATT=%.6f (SE=%.6f)",
    event_study$event_time[i],
    event_study$att[i],
    event_study$se[i]))
}

# Compute overall ATT (average of post-treatment effects)
post_effects <- event_study[event_study$event_time >= 0, ]
if (nrow(post_effects) > 0) {
  # Use hypotheses() for delta-method SE
  K <- nrow(post_effects)
  hyp_terms <- paste0("`event_time::", post_effects$event_time, ":treat`")
  hyp_formula <- paste0("(", paste(hyp_terms, collapse = " + "), ") / ", K, " = 0")

  hyp_result <- tryCatch({
    marginaleffects::hypotheses(weight_stack, hyp_formula)
  }, error = function(e) {
    # Fallback: simple average without delta-method SE
    message(sprintf("  hypotheses() not available, using simple average: %s", e$message))
    NULL
  })

  if (!is.null(hyp_result)) {
    overall_att <- hyp_result$estimate
    overall_se <- hyp_result$std.error
  } else {
    overall_att <- mean(post_effects$att)
    # Approximate SE: sqrt(mean(se^2) / K)
    overall_se <- sqrt(sum(post_effects$se^2)) / K
  }
} else {
  overall_att <- NA_real_
  overall_se <- NA_real_
}

message(sprintf("Overall ATT: %.6f (SE: %.6f)", overall_att, overall_se))
message(sprintf("Total time: %.3f seconds", total_time))

# =============================================================================
# Format and write output
# =============================================================================

results <- list(
  estimator = "stacked-did-weights (Wing et al. 2024)",

  # Overall ATT
  overall_att = overall_att,
  overall_se = overall_se,

  # Event study
  event_study = event_study,

  # Timing
  timing = list(
    stacking_seconds = stack_time,
    regression_seconds = reg_time,
    total_seconds = total_time
  ),

  # Metadata
  metadata = list(
    r_version = R.version.string,
    fixest_version = as.character(packageVersion("fixest")),
    n_units = n_units,
    n_periods = n_periods,
    n_obs = nrow(data),
    n_stacked_obs = nrow(stacked_data2),
    n_sub_experiments = n_feasible,
    kappa_pre = kappa_pre,
    kappa_post = kappa_post,
    weighting = "aggregate",
    clean_control = "not_yet_treated"
  )
)

message(sprintf("Writing results to: %s", config$output))
dir.create(dirname(config$output), recursive = TRUE, showWarnings = FALSE)
write_json(results, config$output, auto_unbox = TRUE, pretty = TRUE, digits = 10)

message(sprintf("Completed in %.3f seconds", total_time))
