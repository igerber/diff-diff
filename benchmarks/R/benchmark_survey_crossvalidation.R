#!/usr/bin/env Rscript
# Benchmark: Survey Cross-Validation (R `survey` + `did` packages)
#
# Generates golden values for cross-validation of diff-diff's survey
# variance estimates against R's authoritative implementations.
#
# Tier 1: svyglm() for basic DiD under complex survey designs
# Tier 2: did::att_gt() with survey weights for CallawaySantAnna
# Tier 3: svrepdesign() + svyglm() for BRR replicate weights
#
# Usage:
#   Rscript benchmark_survey_crossvalidation.R
#
# References:
#   Lumley, T. (2004). "Analysis of Complex Survey Samples." JSS 9(8).
#   Binder, D.A. (1983). "On the Variances of Asymptotically Normal Estimators
#     from Complex Surveys." International Statistical Review 51(3).

library(survey)
library(jsonlite)

cat("=== Survey Cross-Validation Benchmark Generator ===\n\n")

# --- Output directory ---
output_dir <- file.path(dirname(dirname(getwd())), "benchmarks", "data", "synthetic")
if (!dir.exists(output_dir)) {
  output_dir <- "benchmarks/data/synthetic"
}
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

results <- list()

# ---------------------------------------------------------------------------
# Data Generation: 2x2 DiD with complex survey structure
# ---------------------------------------------------------------------------

generate_survey_did_data <- function(seed, n_units = 200, att_true = 5.0,
                                     include_covariates = FALSE) {
  set.seed(seed)

  # 4 strata with unequal sizes
  strata_sizes <- c(30, 40, 60, 70)  # total = 200 units
  n_strata <- length(strata_sizes)
  stopifnot(sum(strata_sizes) == n_units)

  # PSU assignment: 2-3 PSUs per stratum, globally unique IDs
  # Stratum 1: 2 PSUs, Stratum 2: 2 PSUs, Stratum 3: 3 PSUs, Stratum 4: 3 PSUs
  psu_per_stratum <- c(2, 2, 3, 3)  # total = 10 PSUs
  fpc_values <- c(100, 150, 200, 250)  # population sizes per stratum

  # Build unit-level data
  unit_id <- integer(0)
  stratum <- integer(0)
  psu <- integer(0)
  fpc <- numeric(0)
  weight <- numeric(0)
  treated <- integer(0)
  unit_effect <- numeric(0)

  # Covariates (generated at unit level)
  x1_vals <- numeric(0)
  x2_vals <- numeric(0)

  global_psu_id <- 0
  unit_counter <- 0

  for (h in 1:n_strata) {
    n_h <- strata_sizes[h]
    n_psu_h <- psu_per_stratum[h]
    fpc_h <- fpc_values[h]
    w_h <- fpc_h / n_h  # inverse selection probability weight

    # Assign units to PSUs within stratum (roughly equal)
    psu_assignment <- rep(1:n_psu_h, length.out = n_h)

    for (i in 1:n_h) {
      unit_counter <- unit_counter + 1
      unit_id <- c(unit_id, unit_counter)
      stratum <- c(stratum, h)
      psu <- c(psu, global_psu_id + psu_assignment[i])
      fpc <- c(fpc, fpc_h)
      weight <- c(weight, w_h)

      # Treatment: first half of each stratum is treated
      is_treated <- as.integer(i <= n_h / 2)
      treated <- c(treated, is_treated)

      # Unit-level heterogeneity
      ue <- rnorm(1, 0, 2)
      unit_effect <- c(unit_effect, ue)

      # Covariates
      if (include_covariates) {
        x1_i <- rnorm(1) + 0.5 * is_treated  # correlated with treatment
        x2_i <- as.integer(runif(1) < (0.3 + 0.1 * h))  # correlated with stratum
        x1_vals <- c(x1_vals, x1_i)
        x2_vals <- c(x2_vals, x2_i)
      }
    }
    global_psu_id <- global_psu_id + n_psu_h
  }

  # Expand to panel: 2 periods (0 = pre, 1 = post)
  n_obs <- n_units * 2
  rows <- data.frame(
    unit = rep(unit_id, each = 2),
    period = rep(c(0L, 1L), times = n_units),
    treated = rep(treated, each = 2),
    post = rep(c(0L, 1L), times = n_units),
    stratum = rep(stratum, each = 2),
    psu = rep(psu, each = 2),
    fpc = rep(fpc, each = 2),
    weight = rep(weight, each = 2)
  )

  # DGP: outcome = 10 + unit_effect + 5*post + att*treated*post [+ covariates] + noise
  ue_expanded <- rep(unit_effect, each = 2)
  noise <- rnorm(n_obs, 0, 1)

  rows$outcome <- 10 + ue_expanded + 5 * rows$post +
    att_true * rows$treated * rows$post + noise

  if (include_covariates) {
    rows$x1 <- rep(x1_vals, each = 2)
    rows$x2 <- rep(x2_vals, each = 2)
    # Add covariate effects to outcome
    rows$outcome <- rows$outcome + 2 * rows$x1 + 1.5 * rows$x2
  }

  return(rows)
}

# ---------------------------------------------------------------------------
# Helper: extract svyglm results for the interaction term
# ---------------------------------------------------------------------------

extract_svyglm_results <- function(model, design, coef_name = "treated:post") {
  coefs <- coef(model)
  ses <- SE(model)
  idx <- which(names(coefs) == coef_name)

  att <- as.numeric(coefs[idx])
  se <- as.numeric(ses[idx])
  t_stat <- att / se
  df <- degf(design)

  # Compute CI manually — confint.svyglm returns [-Inf, Inf] when
  # n_params > df, but the manual t-quantile computation is correct.
  t_crit <- qt(0.975, df)
  ci_lower <- att - t_crit * se
  ci_upper <- att + t_crit * se

  list(
    att = att,
    se = se,
    t_stat = t_stat,
    df = df,
    ci_lower = ci_lower,
    ci_upper = ci_upper
  )
}

# ---------------------------------------------------------------------------
# Tier 1: svyglm DiD under various survey designs
# ---------------------------------------------------------------------------

cat("--- Tier 1: svyglm DiD ---\n")

tier1_configs <- list(
  # Design variants at seed=42
  list(key = "strata_psu_fpc_s42",    seed = 42,  design_type = "strata_psu_fpc",
       covariates = FALSE),
  list(key = "strata_psu_nofpc_s42",  seed = 42,  design_type = "strata_psu_nofpc",
       covariates = FALSE),
  list(key = "weights_only_s42",      seed = 42,  design_type = "weights_only",
       covariates = FALSE),
  list(key = "strata_only_s42",       seed = 42,  design_type = "strata_only",
       covariates = FALSE),

  # Covariate variant at seed=42
  list(key = "strata_psu_fpc_cov_s42", seed = 42, design_type = "strata_psu_fpc",
       covariates = TRUE),

  # Multi-seed robustness (full design, no covariates)
  list(key = "strata_psu_fpc_s123",   seed = 123, design_type = "strata_psu_fpc",
       covariates = FALSE),
  list(key = "strata_psu_fpc_s789",   seed = 789, design_type = "strata_psu_fpc",
       covariates = FALSE)
)

for (cfg in tier1_configs) {
  cat(sprintf("  Running scenario: %s ...\n", cfg$key))

  data <- generate_survey_did_data(
    seed = cfg$seed,
    include_covariates = cfg$covariates
  )

  # Construct survey design based on type
  design <- switch(cfg$design_type,
    "strata_psu_fpc" = svydesign(
      ids = ~psu, strata = ~stratum, fpc = ~fpc, weights = ~weight, data = data
    ),
    "strata_psu_nofpc" = svydesign(
      ids = ~psu, strata = ~stratum, weights = ~weight, data = data
    ),
    "weights_only" = svydesign(
      ids = ~1, weights = ~weight, data = data
    ),
    "strata_only" = svydesign(
      ids = ~1, strata = ~stratum, weights = ~weight, data = data
    )
  )

  # Verify weights survived construction
  design_weights <- weights(design)
  weight_match <- all.equal(as.numeric(design_weights), data$weight, tolerance = 1e-10)
  if (!isTRUE(weight_match)) {
    cat(sprintf("    WARNING: weights changed during svydesign construction: %s\n",
                weight_match))
  }

  # Fit model
  if (cfg$covariates) {
    model <- svyglm(outcome ~ treated * post + x1 + x2, design = design)
  } else {
    model <- svyglm(outcome ~ treated * post, design = design)
  }

  est <- extract_svyglm_results(model, design)

  # Also compute naive (unweighted) estimate for reference
  if (cfg$covariates) {
    naive <- lm(outcome ~ treated * post + x1 + x2, data = data)
  } else {
    naive <- lm(outcome ~ treated * post, data = data)
  }
  naive_coefs <- coef(naive)
  naive_ses <- summary(naive)$coefficients[, "Std. Error"]
  naive_idx <- which(names(naive_coefs) == "treated:post")

  # Build data export (as lists for JSON)
  data_export <- list(
    unit = as.list(data$unit),
    period = as.list(data$period),
    treated = as.list(data$treated),
    post = as.list(data$post),
    outcome = as.list(data$outcome),
    stratum = as.list(data$stratum),
    psu = as.list(data$psu),
    fpc = as.list(data$fpc),
    weight = as.list(data$weight)
  )
  if (cfg$covariates) {
    data_export$x1 <- as.list(data$x1)
    data_export$x2 <- as.list(data$x2)
  }

  results[[cfg$key]] <- list(
    estimator = "survey::svyglm",
    design_type = cfg$design_type,
    seed = cfg$seed,
    has_covariates = cfg$covariates,
    n_units = 200,
    n_periods = 2,
    n_obs = nrow(data),
    att = est$att,
    se = est$se,
    t_stat = est$t_stat,
    df = est$df,
    ci_lower = est$ci_lower,
    ci_upper = est$ci_upper,
    n_strata = length(unique(data$stratum)),
    n_psu = length(unique(data$psu)),
    att_naive = as.numeric(naive_coefs[naive_idx]),
    se_naive = as.numeric(naive_ses[naive_idx]),
    data = data_export
  )

  cat(sprintf("    ATT = %.6f, SE = %.6f, df = %d\n", est$att, est$se, est$df))
  cat(sprintf("    CI = [%.6f, %.6f]\n", est$ci_lower, est$ci_upper))
}

# ---------------------------------------------------------------------------
# Tier 2: CallawaySantAnna with survey weights
# ---------------------------------------------------------------------------

cat("\n--- Tier 2: CallawaySantAnna with survey weights ---\n")

# Check if 'did' package is available
has_did <- requireNamespace("did", quietly = TRUE)

if (has_did) {
  library(did)

  generate_cs_survey_data <- function(seed, n_units = 300) {
    set.seed(seed)

    # 3 cohorts: first_treat=2, first_treat=3, never-treated (Inf)
    # 100 units each
    n_per_group <- n_units / 3
    stopifnot(n_per_group == floor(n_per_group))

    # 3 strata of 100 units each, weights vary
    strata_weights <- c(2.0, 3.5, 5.0)

    unit_id <- 1:n_units
    first_treat <- c(rep(2, n_per_group), rep(3, n_per_group), rep(Inf, n_per_group))
    stratum <- rep(1:3, each = n_per_group)
    w <- strata_weights[stratum]

    # Covariate: continuous, correlated with treatment timing
    x1 <- rnorm(n_units) + ifelse(is.finite(first_treat), 0.3 * first_treat, 0)

    # Unit effects
    unit_eff <- rnorm(n_units, 0, 2)

    # Expand to panel: 4 periods (1-4)
    periods <- 1:4
    rows <- expand.grid(unit = unit_id, period = periods)
    rows <- rows[order(rows$unit, rows$period), ]

    rows$first_treat <- first_treat[rows$unit]
    rows$stratum <- stratum[rows$unit]
    rows$weight <- w[rows$unit]
    rows$x1 <- x1[rows$unit]

    # Treatment indicator
    rows$treated <- as.integer(
      is.finite(rows$first_treat) & rows$period >= rows$first_treat
    )

    # Outcome DGP: ATT = 3.0 for each post-treatment period
    att_true <- 3.0
    rows$outcome <- 10 + unit_eff[rows$unit] + 2 * rows$period +
      att_true * rows$treated + rnorm(nrow(rows), 0, 1)

    # Add covariate effect
    rows$outcome <- rows$outcome + 1.5 * rows$x1

    return(rows)
  }

  cs_data <- generate_cs_survey_data(seed = 42)

  cs_scenarios <- list(
    list(key = "cs_weighted_nocov", xformla = ~1, cov_name = NULL),
    list(key = "cs_weighted_cov",   xformla = ~x1, cov_name = "x1")
  )

  for (cs_cfg in cs_scenarios) {
    cat(sprintf("  Running scenario: %s ...\n", cs_cfg$key))

    out <- tryCatch({
      att_gt(
        yname = "outcome",
        tname = "period",
        idname = "unit",
        gname = "first_treat",
        data = cs_data,
        weightsname = "weight",
        est_method = "reg",
        xformla = cs_cfg$xformla,
        control_group = "nevertreated",
        base_period = "varying",
        bstrap = FALSE,
        cband = FALSE
      )
    }, error = function(e) {
      cat(sprintf("    ERROR: %s\n", e$message))
      return(NULL)
    })

    if (is.null(out)) next

    # Simple aggregation
    agg <- aggte(out, type = "simple")

    # Export group-time effects, sorted by (group, time) for stable comparison
    gt_results <- data.frame(
      group = out$group,
      time = out$t,
      att = out$att,
      se = out$se
    )
    gt_results <- gt_results[order(gt_results$group, gt_results$time), ]

    # Data export
    cs_data_export <- list(
      unit = as.list(cs_data$unit),
      period = as.list(cs_data$period),
      first_treat = as.list(cs_data$first_treat),
      outcome = as.list(cs_data$outcome),
      weight = as.list(cs_data$weight),
      x1 = as.list(cs_data$x1),
      stratum = as.list(cs_data$stratum)
    )

    results[[cs_cfg$key]] <- list(
      estimator = "did::att_gt",
      design_type = "cs_weighted",
      seed = 42,
      estimation_method = "reg",
      control_group = "nevertreated",
      has_covariates = !is.null(cs_cfg$cov_name),
      covariate_formula = deparse(cs_cfg$xformla),
      n_units = 300,
      n_periods = 4,
      n_obs = nrow(cs_data),
      gt_groups = as.list(gt_results$group),
      gt_periods = as.list(gt_results$time),
      gt_att = as.list(gt_results$att),
      gt_se = as.list(gt_results$se),
      overall_att = agg$overall.att,
      overall_se = agg$overall.se,
      data = cs_data_export
    )

    cat(sprintf("    Overall ATT = %.6f, SE = %.6f\n",
                agg$overall.att, agg$overall.se))
    cat(sprintf("    %d group-time cells\n", nrow(gt_results)))
  }
} else {
  cat("  SKIPPED: 'did' package not available\n")
}

# ---------------------------------------------------------------------------
# Tier 3: BRR replicate weights
# ---------------------------------------------------------------------------

cat("\n--- Tier 3: BRR replicate weights ---\n")

generate_brr_data <- function(seed, n_units = 200, att_true = 5.0) {
  # BRR requires exactly 2 PSUs per stratum
  set.seed(seed)

  n_strata <- 4
  strata_sizes <- c(50, 50, 50, 50)  # equal for BRR simplicity
  psu_per_stratum <- rep(2, n_strata)  # exactly 2 PSUs per stratum
  fpc_values <- c(200, 300, 400, 500)

  unit_id <- integer(0)
  stratum <- integer(0)
  psu <- integer(0)
  fpc <- numeric(0)
  weight <- numeric(0)
  treated <- integer(0)
  unit_effect <- numeric(0)

  global_psu_id <- 0
  unit_counter <- 0

  for (h in 1:n_strata) {
    n_h <- strata_sizes[h]
    n_psu_h <- psu_per_stratum[h]
    fpc_h <- fpc_values[h]
    w_h <- fpc_h / n_h

    psu_assignment <- rep(1:n_psu_h, length.out = n_h)

    for (i in 1:n_h) {
      unit_counter <- unit_counter + 1
      unit_id <- c(unit_id, unit_counter)
      stratum <- c(stratum, h)
      psu <- c(psu, global_psu_id + psu_assignment[i])
      fpc <- c(fpc, fpc_h)
      weight <- c(weight, w_h)
      is_treated <- as.integer(i <= n_h / 2)
      treated <- c(treated, is_treated)
      unit_effect <- c(unit_effect, rnorm(1, 0, 2))
    }
    global_psu_id <- global_psu_id + n_psu_h
  }

  # Panel: 2 periods
  rows <- data.frame(
    unit = rep(unit_id, each = 2),
    period = rep(c(0L, 1L), times = n_units),
    treated = rep(treated, each = 2),
    post = rep(c(0L, 1L), times = n_units),
    stratum = rep(stratum, each = 2),
    psu = rep(psu, each = 2),
    fpc = rep(fpc, each = 2),
    weight = rep(weight, each = 2)
  )

  ue_expanded <- rep(unit_effect, each = 2)
  noise <- rnorm(nrow(rows), 0, 1)
  rows$outcome <- 10 + ue_expanded + 5 * rows$post +
    att_true * rows$treated * rows$post + noise

  return(rows)
}

cat("  Running scenario: brr_replicate ...\n")

brr_data <- generate_brr_data(seed = 42)

# Create base design
base_design <- svydesign(
  ids = ~psu, strata = ~stratum, fpc = ~fpc, weights = ~weight, data = brr_data
)

# Convert to BRR replicate design
brr_design <- as.svrepdesign(base_design, type = "BRR")

# Fit model
brr_model <- svyglm(outcome ~ treated * post, design = brr_design)

brr_est <- extract_svyglm_results(brr_model, brr_design)

# Extract replicate weights (analysis weights = combined weights)
rep_weights <- weights(brr_design, type = "analysis")
n_replicates <- ncol(rep_weights)

# Export replicate weights as list of columns
rep_weight_export <- list()
for (r in 1:n_replicates) {
  rep_weight_export[[paste0("rep_", r - 1)]] <- as.list(rep_weights[, r])
}

# Data export
brr_data_export <- list(
  unit = as.list(brr_data$unit),
  period = as.list(brr_data$period),
  treated = as.list(brr_data$treated),
  post = as.list(brr_data$post),
  outcome = as.list(brr_data$outcome),
  stratum = as.list(brr_data$stratum),
  psu = as.list(brr_data$psu),
  fpc = as.list(brr_data$fpc),
  weight = as.list(brr_data$weight)
)

results[["brr_replicate"]] <- list(
  estimator = "survey::svyglm (svrepdesign)",
  design_type = "brr_replicate",
  seed = 42,
  n_units = 200,
  n_periods = 2,
  n_obs = nrow(brr_data),
  att = brr_est$att,
  se = brr_est$se,
  t_stat = brr_est$t_stat,
  df = brr_est$df,
  ci_lower = brr_est$ci_lower,
  ci_upper = brr_est$ci_upper,
  n_strata = length(unique(brr_data$stratum)),
  n_psu = length(unique(brr_data$psu)),
  n_replicates = n_replicates,
  scale = brr_design$scale,
  rscales = as.list(brr_design$rscales),
  replicate_weights = rep_weight_export,
  data = brr_data_export
)

cat(sprintf("    ATT = %.6f, SE = %.6f, df = %d\n",
            brr_est$att, brr_est$se, brr_est$df))
cat(sprintf("    %d BRR replicates\n", n_replicates))

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

json_path <- file.path(output_dir, "survey_crossvalidation_r_results.json")
writeLines(toJSON(results, auto_unbox = TRUE, pretty = TRUE, digits = 10), json_path)
cat(sprintf("\nResults saved to: %s\n", json_path))

# Summary
cat("\n=== Summary ===\n")
cat(sprintf("Tier 1: %d scenarios (svyglm)\n",
            sum(sapply(results, function(x) x$estimator == "survey::svyglm"))))
cat(sprintf("Tier 2: %d scenarios (did::att_gt)\n",
            sum(sapply(results, function(x) x$estimator == "did::att_gt"))))
cat(sprintf("Tier 3: %d scenarios (svrepdesign)\n",
            sum(sapply(results, function(x) grepl("svrepdesign", x$estimator)))))
cat("Done.\n")
