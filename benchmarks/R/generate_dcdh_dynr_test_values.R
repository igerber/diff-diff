#!/usr/bin/env Rscript
# Generate golden values for de Chaisemartin-D'Haultfoeuille (dCDH)
# parity tests at horizon l=1.
#
# This script fits the R `DIDmultiplegtDYN` package's `did_multiplegt_dyn`
# function (the official R implementation of the dCDH dynamic-effects
# companion paper, NBER WP 29873) on a set of canonical reversible-treatment
# scenarios. At horizon l=1, did_multiplegt_dyn computes DID_1, which is
# numerically identical to DID_M of the AER 2020 paper. Phase 1 of the
# Python diff-diff dCDH implementation tests for parity against these
# golden values.
#
# Usage:
#   Rscript benchmarks/R/generate_dcdh_dynr_test_values.R
#
# Prerequisites:
#   install.packages("DIDmultiplegtDYN")  # CRAN v2.3.3+
#   install.packages("jsonlite")
#
# Output:
#   benchmarks/data/dcdh_dynr_golden_values.json
#
# Each scenario exports:
#   - data: the simulated dataset (so Python tests use identical data)
#   - params: the dCDH options used
#   - results: DID_1 point estimate, SE, CI, placebo, switcher counts

library(DIDmultiplegtDYN)
library(jsonlite)
suppressMessages(library(polars))  # required by DIDmultiplegtDYN >= 2.x

# Pin DIDmultiplegtDYN to an exact version because the `by_path` output
# slots (res$by_levels, res$by_level_i) were introduced in v2.3.3 and
# the structure is not version-stable per the R package's own docs. A
# floor constraint (`>= 2.3.3`) could silently drift the fixture schema
# when regenerated against a future release. Update this pin *and* re-
# run TestDCDHDynRParityByPath when bumping to a newer known-compatible
# release; extend to an explicit allowlist (e.g. `%in% c("2.3.3",
# "2.3.4")`) once a second version is verified.
stopifnot(packageVersion("DIDmultiplegtDYN") == "2.3.3")

cat("Generating dCDH golden values via DIDmultiplegtDYN at l=1...\n")

output_path <- file.path("benchmarks", "data", "dcdh_dynr_golden_values.json")

# ---------------------------------------------------------------------------
# Helper: Python-mirror reversible-treatment generator.
# Mirrors generate_reversible_did_data() in diff_diff/prep_dgp.py at the
# STRUCTURAL level — the two implementations apply the same pattern logic
# (single_switch / joiners_only / leavers_only / mixed_single_switch) and
# the same fixed-effect / treatment-effect / time-trend / noise model. They
# do NOT produce bit-identical draws even with the same seed: R's set.seed
# and NumPy's default_rng use different RNGs and the parity tests don't
# rely on RNG identity. Instead, the parity tests load THIS R script's
# golden-value JSON output and pass the SAME data (group/period/treatment/
# outcome columns) to the Python estimator, so both sides operate on
# byte-identical input regardless of how it was originally generated.
# ---------------------------------------------------------------------------
gen_reversible <- function(n_groups, n_periods, pattern, seed,
                           p_switch = 0.2, initial_treat_frac = 0.3,
                           cycle_length = 2, treatment_effect = 2.0,
                           heterogeneous_effects = FALSE, effect_sd = 0.5,
                           group_fe_sd = 2.0, time_trend = 0.1, noise_sd = 0.5,
                           n_never_treated = 20, n_always_treated = 20,
                           L_max = 3) {
  # n_never_treated and n_always_treated add stable control cohorts so
  # both Python (AER 2020 zero-retention) and R DIDmultiplegtDYN (dynamic
  # paper, drop-cohort) implementations have controls available at every
  # period — eliminating the methodology divergence and giving a clean
  # parity comparison. The total returned panel has
  # n_groups + n_never_treated + n_always_treated groups.
  set.seed(seed)

  # --- Build the (n_groups, n_periods) treatment matrix ---
  D <- matrix(0L, nrow = n_groups, ncol = n_periods)

  if (pattern == "single_switch") {
    initial_treated <- runif(n_groups) < initial_treat_frac
    switch_times <- sample.int(n_periods - 1, n_groups, replace = TRUE)  # 1..(n_periods-1)
    for (g in seq_len(n_groups)) {
      st <- switch_times[g] + 1L  # convert to 1-indexed switch period
      if (initial_treated[g]) {
        D[g, seq_len(st - 1L)] <- 1L
        D[g, st:n_periods] <- 0L
      } else {
        D[g, seq_len(st - 1L)] <- 0L
        D[g, st:n_periods] <- 1L
      }
    }
  } else if (pattern == "joiners_only") {
    switch_times <- sample.int(n_periods - 1, n_groups, replace = TRUE)
    for (g in seq_len(n_groups)) {
      st <- switch_times[g] + 1L
      D[g, st:n_periods] <- 1L
    }
  } else if (pattern == "leavers_only") {
    switch_times <- sample.int(n_periods - 1, n_groups, replace = TRUE)
    for (g in seq_len(n_groups)) {
      st <- switch_times[g] + 1L
      D[g, seq_len(st - 1L)] <- 1L
    }
  } else if (pattern == "mixed_single_switch") {
    switch_times <- sample.int(n_periods - 1, n_groups, replace = TRUE)
    n_joiners <- n_groups %/% 2
    for (g in seq_len(n_groups)) {
      st <- switch_times[g] + 1L
      if (g <= n_joiners) {
        D[g, st:n_periods] <- 1L
      } else {
        D[g, seq_len(st - 1L)] <- 1L
      }
    }
  } else if (pattern == "multi_path_reversible") {
    # Deterministic multi-path DGP designed for by_path R-parity:
    #   - 4 distinct joiner-style target paths with unequal frequencies
    #     (so top-k ranking produces unique ranks with no ties)
    #   - path assignment is a DETERMINISTIC FUNCTION OF F_g, so each
    #     cohort (D_{g,1}, F_g, S_g) contains switchers from a single
    #     path. This avoids cross-path cohort sharing in the
    #     cohort-recentered influence function, which otherwise blows
    #     out SE parity with R's re-run-per-path convention.
    #   - post-window treatment is stable at path[L_max+1] (clean
    #     control-pool eligibility — no post-window contamination)
    #
    # Each group:
    #   - F_g in [2, n_periods - L_max] so the length-(L_max+1) window
    #     [F_g-1, F_g-1+L_max] fits the panel
    #   - path is determined by F_g: two F_g values per path (groups in
    #     {F_g in {2,3}} share path 1, {F_g in {4,5}} share path 2,
    #     {F_g in {6}} = path 3, {F_g in {7}} = path 4); within a path,
    #     F_g distribution yields n_groups * path_prop total groups
    max_switch <- n_periods - L_max - 1L
    stopifnot(max_switch >= 1L)
    # With n_periods=10, L_max=3: max_switch=6, F_g in [2,7] (6 values).

    target_paths <- list(
      c(0L, 1L, 1L, 1L),  # sustained on (rank 1)
      c(0L, 1L, 1L, 0L),  # on then off  (rank 2)
      c(0L, 1L, 0L, 0L),  # on briefly   (rank 3)
      c(0L, 1L, 0L, 1L)   # on-off-on    (rank 4, truncated under by_path=3)
    )
    stopifnot(length(target_paths[[1]]) == L_max + 1L)

    # Per-F_g path assignment: 2 F_g values for paths 1 & 2, 1 for paths
    # 3 & 4. This keeps each (D_{g,1}, F_g, S_g) cohort single-path.
    f_g_to_path <- c(1L, 1L, 2L, 2L, 3L, 4L)
    stopifnot(length(f_g_to_path) == max_switch)
    # Group counts per F_g (rank 1 > rank 2 > rank 3 > rank 4 with unique
    # ranks — no frequency ties, robust to R's undocumented tiebreak):
    #   F_g=2: 20 groups (path 1)
    #   F_g=3: 20 groups (path 1) → rank 1 has 40 switchers total
    #   F_g=4: 15 groups (path 2)
    #   F_g=5: 10 groups (path 2) → rank 2 has 25 switchers total
    #   F_g=6: 10 groups (path 3) → rank 3 has 10 switchers
    #   F_g=7:  5 groups (path 4) → rank 4 has  5 switchers (excluded by by_path=3)
    counts_per_F_g <- c(20L, 20L, 15L, 10L, 10L, 5L)
    stopifnot(sum(counts_per_F_g) == n_groups)

    # Build the group-to-(F_g, path) assignment, deterministic with seed
    g_idx <- 1L
    for (f_idx in seq_along(counts_per_F_g)) {
      F_g <- f_idx + 1L  # F_g in [2, 7] for f_idx in [1, 6]
      path_idx <- f_g_to_path[f_idx]
      target <- target_paths[[path_idx]]
      n_here <- counts_per_F_g[f_idx]
      for (k in seq_len(n_here)) {
        g <- g_idx
        # Pre-baseline [1 .. F_g-2]: initial state
        if (F_g >= 3L) {
          D[g, 1:(F_g - 2L)] <- target[1]
        }
        # Window [F_g-1 .. F_g-1+L_max]: exactly the target path
        for (j in 0:L_max) {
          D[g, F_g - 1L + j] <- target[j + 1L]
        }
        # Post-window [F_g+L_max .. n_periods]: stable at path[L_max+1]
        if (F_g + L_max <= n_periods) {
          D[g, (F_g + L_max):n_periods] <- target[L_max + 1L]
        }
        g_idx <- g_idx + 1L
      }
    }
  } else {
    stop(sprintf("Unknown pattern: %s", pattern))
  }

  # --- Append stable control cohorts ---
  if (n_never_treated > 0) {
    D <- rbind(D, matrix(0L, nrow = n_never_treated, ncol = n_periods))
  }
  if (n_always_treated > 0) {
    D <- rbind(D, matrix(1L, nrow = n_always_treated, ncol = n_periods))
  }
  n_groups <- nrow(D)

  # --- Generate fixed effects, true effects, outcomes ---
  group_fe <- rnorm(n_groups, mean = 0, sd = group_fe_sd)
  if (heterogeneous_effects) {
    true_effects <- matrix(rnorm(n_groups * n_periods, mean = treatment_effect, sd = effect_sd),
                           nrow = n_groups, ncol = n_periods)
  } else {
    true_effects <- matrix(treatment_effect, nrow = n_groups, ncol = n_periods)
  }
  true_effects[D == 0] <- 0.0

  period_arr <- 0:(n_periods - 1)
  noise <- matrix(rnorm(n_groups * n_periods, mean = 0, sd = noise_sd),
                  nrow = n_groups, ncol = n_periods)
  Y <- 10.0 + matrix(group_fe, nrow = n_groups, ncol = n_periods) +
       matrix(time_trend * period_arr, nrow = n_groups, ncol = n_periods, byrow = TRUE) +
       true_effects + noise

  # --- Build long-format data frame ---
  group_col <- rep(seq_len(n_groups) - 1L, each = n_periods)  # 0-indexed
  period_col <- rep(period_arr, n_groups)
  treatment_col <- as.vector(t(D))
  outcome_col <- as.vector(t(Y))

  data.frame(
    group = group_col,
    period = period_col,
    treatment = treatment_col,
    outcome = outcome_col
  )
}

# ---------------------------------------------------------------------------
# Helper: extract DID_1 (l=1) results from did_multiplegt_dyn output
# ---------------------------------------------------------------------------
extract_dcdh_l1 <- function(res) {
  # did_multiplegt_dyn returns a results object with $results$Effects matrix.
  # The Effects matrix has one row per effect (Effect_1, Effect_2, ...) and
  # columns: Estimate, SE, LB CI, UB CI, N, Switchers, N.w, Switchers.w. We
  # pull the row for "Effect_1" (the l=1 effect, == DID_M of AER 2020).
  effects <- res$results$Effects
  if (is.null(effects)) {
    stop("did_multiplegt_dyn returned no Effects; check the input data")
  }

  out <- list(
    overall_att = as.numeric(effects[1, "Estimate"]),
    overall_se = as.numeric(effects[1, "SE"]),
    overall_ci_lo = as.numeric(effects[1, "LB CI"]),
    overall_ci_hi = as.numeric(effects[1, "UB CI"]),
    n_switchers = as.numeric(effects[1, "N"])
  )

  # Placebo at lag 1 if available (Placebos has the same column layout)
  placebos <- res$results$Placebos
  if (!is.null(placebos) && nrow(placebos) >= 1) {
    out$placebo_effect <- as.numeric(placebos[1, "Estimate"])
    out$placebo_se <- as.numeric(placebos[1, "SE"])
    out$placebo_ci_lo <- as.numeric(placebos[1, "LB CI"])
    out$placebo_ci_hi <- as.numeric(placebos[1, "UB CI"])
  }

  out
}

# ---------------------------------------------------------------------------
# Helper: convert data frame to exportable list
# ---------------------------------------------------------------------------
export_data <- function(df) {
  list(
    group = as.numeric(df$group),
    period = as.numeric(df$period),
    treatment = as.numeric(df$treatment),
    outcome = as.numeric(df$outcome)
  )
}

scenarios <- list()

# Golden value datasets use n_groups=80 to keep the JSON file small (~50KB)
# while still being large enough to exercise inference.
N_GOLDEN <- 80

# ---------------------------------------------------------------------------
# Scenario 1: single_switch — mix of joiners and leavers
# ---------------------------------------------------------------------------
cat("  Scenario 1: single_switch_mixed\n")
d1 <- gen_reversible(n_groups = N_GOLDEN, n_periods = 6,
                     pattern = "single_switch", seed = 101)
res1 <- did_multiplegt_dyn(
  df = d1, outcome = "outcome", group = "group", time = "period",
  treatment = "treatment", effects = 1, placebo = 1, ci_level = 95
)
scenarios$single_switch_mixed <- list(
  data = export_data(d1),
  params = list(pattern = "single_switch", n_groups = N_GOLDEN, n_periods = 6,
                seed = 101, effects = 1, placebo = 1, ci_level = 95),
  results = extract_dcdh_l1(res1)
)

# ---------------------------------------------------------------------------
# Scenario 2: joiners_only — pure staggered adoption
# ---------------------------------------------------------------------------
cat("  Scenario 2: joiners_only\n")
d2 <- gen_reversible(n_groups = N_GOLDEN, n_periods = 6,
                     pattern = "joiners_only", seed = 102)
res2 <- did_multiplegt_dyn(
  df = d2, outcome = "outcome", group = "group", time = "period",
  treatment = "treatment", effects = 1, placebo = 1, ci_level = 95
)
scenarios$joiners_only <- list(
  data = export_data(d2),
  params = list(pattern = "joiners_only", n_groups = N_GOLDEN, n_periods = 6,
                seed = 102, effects = 1, placebo = 1, ci_level = 95),
  results = extract_dcdh_l1(res2)
)

# ---------------------------------------------------------------------------
# Scenario 3: leavers_only — pure staggered removal
# ---------------------------------------------------------------------------
cat("  Scenario 3: leavers_only\n")
d3 <- gen_reversible(n_groups = N_GOLDEN, n_periods = 6,
                     pattern = "leavers_only", seed = 103)
res3 <- did_multiplegt_dyn(
  df = d3, outcome = "outcome", group = "group", time = "period",
  treatment = "treatment", effects = 1, placebo = 1, ci_level = 95
)
scenarios$leavers_only <- list(
  data = export_data(d3),
  params = list(pattern = "leavers_only", n_groups = N_GOLDEN, n_periods = 6,
                seed = 103, effects = 1, placebo = 1, ci_level = 95),
  results = extract_dcdh_l1(res3)
)

# ---------------------------------------------------------------------------
# Scenario 4: mixed_single_switch — deterministic 50/50 joiners/leavers
# ---------------------------------------------------------------------------
cat("  Scenario 4: mixed_single_switch\n")
d4 <- gen_reversible(n_groups = N_GOLDEN, n_periods = 6,
                     pattern = "mixed_single_switch", seed = 104)
res4 <- did_multiplegt_dyn(
  df = d4, outcome = "outcome", group = "group", time = "period",
  treatment = "treatment", effects = 1, placebo = 1, ci_level = 95
)
scenarios$mixed_single_switch <- list(
  data = export_data(d4),
  params = list(pattern = "mixed_single_switch", n_groups = N_GOLDEN, n_periods = 6,
                seed = 104, effects = 1, placebo = 1, ci_level = 95),
  results = extract_dcdh_l1(res4)
)

# ---------------------------------------------------------------------------
# Scenario 5: hand-calculable 4-group panel from the worked example.
# This is the panel used by tests/test_methodology_chaisemartin_dhaultfoeuille.py
# in test_hand_calculable_4group_3period_joiners_and_leavers. We capture R's
# answer here so the Python test can assert exact agreement.
# ---------------------------------------------------------------------------
cat("  Scenario 5: hand_calculable_worked_example\n")
d5 <- data.frame(
  group =     c(1, 1, 1,  2, 2, 2,  3, 3, 3,  4, 4, 4),
  period =    c(0, 1, 2,  0, 1, 2,  0, 1, 2,  0, 1, 2),
  treatment = c(0, 1, 1,  1, 1, 0,  0, 0, 0,  1, 1, 1),
  outcome   = c(10, 13, 14,  10, 11, 9,  10, 11, 12,  10, 11, 12)
)
res5 <- did_multiplegt_dyn(
  df = d5, outcome = "outcome", group = "group", time = "period",
  treatment = "treatment", effects = 1, placebo = 0, ci_level = 95
)
scenarios$hand_calculable_worked_example <- list(
  data = export_data(d5),
  params = list(description = "4-group hand-calculable panel from plan worked example",
                effects = 1, placebo = 0, ci_level = 95,
                expected_did_m = 2.5, expected_did_plus = 2.0, expected_did_minus = 3.0),
  results = extract_dcdh_l1(res5)
)

# ---------------------------------------------------------------------------
# Phase 2: Multi-horizon scenarios (effects > 1)
# ---------------------------------------------------------------------------

# Helper: extract multi-horizon results from did_multiplegt_dyn output
extract_dcdh_multi <- function(res, n_effects, n_placebos = 0) {
  effects <- res$results$Effects
  if (is.null(effects)) {
    stop("did_multiplegt_dyn returned no Effects; check the input data")
  }

  out <- list(effects = list(), placebos = list())

  for (i in seq_len(min(n_effects, nrow(effects)))) {
    out$effects[[as.character(i)]] <- list(
      overall_att = as.numeric(effects[i, "Estimate"]),
      overall_se = as.numeric(effects[i, "SE"]),
      overall_ci_lo = as.numeric(effects[i, "LB CI"]),
      overall_ci_hi = as.numeric(effects[i, "UB CI"]),
      n_switchers = as.numeric(effects[i, "N"])
    )
  }

  placebos <- res$results$Placebos
  if (!is.null(placebos) && n_placebos > 0) {
    for (i in seq_len(min(n_placebos, nrow(placebos)))) {
      out$placebos[[as.character(i)]] <- list(
        effect = as.numeric(placebos[i, "Estimate"]),
        se = as.numeric(placebos[i, "SE"]),
        ci_lo = as.numeric(placebos[i, "LB CI"]),
        ci_hi = as.numeric(placebos[i, "UB CI"])
      )
    }
  }

  out
}

# Scenario 6: joiners_only multi-horizon (L_max=3, placebo=3)
# Uses n_periods=8 to give enough room for 3 positive + 3 placebo horizons
cat("  Scenario 6: joiners_only_multi_horizon\n")
d6 <- gen_reversible(n_groups = N_GOLDEN, n_periods = 8,
                     pattern = "joiners_only", seed = 106)
res6 <- did_multiplegt_dyn(
  df = d6, outcome = "outcome", group = "group", time = "period",
  treatment = "treatment", effects = 3, placebo = 3, ci_level = 95
)
scenarios$joiners_only_multi_horizon <- list(
  data = export_data(d6),
  params = list(pattern = "joiners_only", n_groups = N_GOLDEN, n_periods = 8,
                seed = 106, effects = 3, placebo = 3, ci_level = 95),
  results = extract_dcdh_multi(res6, n_effects = 3, n_placebos = 3)
)

# Scenario 7: leavers_only multi-horizon (L_max=3, placebo=3)
cat("  Scenario 7: leavers_only_multi_horizon\n")
d7 <- gen_reversible(n_groups = N_GOLDEN, n_periods = 8,
                     pattern = "leavers_only", seed = 107)
res7 <- did_multiplegt_dyn(
  df = d7, outcome = "outcome", group = "group", time = "period",
  treatment = "treatment", effects = 3, placebo = 3, ci_level = 95
)
scenarios$leavers_only_multi_horizon <- list(
  data = export_data(d7),
  params = list(pattern = "leavers_only", n_groups = N_GOLDEN, n_periods = 8,
                seed = 107, effects = 3, placebo = 3, ci_level = 95),
  results = extract_dcdh_multi(res7, n_effects = 3, n_placebos = 3)
)

# Scenario 8: mixed_single_switch multi-horizon (L_max=5, placebo=4)
# Uses n_periods=10 for far horizons
cat("  Scenario 8: mixed_single_switch_multi_horizon\n")
d8 <- gen_reversible(n_groups = N_GOLDEN, n_periods = 10,
                     pattern = "mixed_single_switch", seed = 108)
res8 <- did_multiplegt_dyn(
  df = d8, outcome = "outcome", group = "group", time = "period",
  treatment = "treatment", effects = 5, placebo = 4, ci_level = 95
)
scenarios$mixed_single_switch_multi_horizon <- list(
  data = export_data(d8),
  params = list(pattern = "mixed_single_switch", n_groups = N_GOLDEN, n_periods = 10,
                seed = 108, effects = 5, placebo = 4, ci_level = 95),
  results = extract_dcdh_multi(res8, n_effects = 5, n_placebos = 4)
)

# Scenario 9: joiners_only long panel multi-horizon (L_max=5, placebo=5)
# Uses n_periods=12 and n_groups=80 for thorough coverage
cat("  Scenario 9: joiners_only_long_multi_horizon\n")
d9 <- gen_reversible(n_groups = N_GOLDEN, n_periods = 12,
                     pattern = "joiners_only", seed = 109)
res9 <- did_multiplegt_dyn(
  df = d9, outcome = "outcome", group = "group", time = "period",
  treatment = "treatment", effects = 5, placebo = 5, ci_level = 95
)
scenarios$joiners_only_long_multi_horizon <- list(
  data = export_data(d9),
  params = list(pattern = "joiners_only", n_groups = N_GOLDEN, n_periods = 12,
                seed = 109, effects = 5, placebo = 5, ci_level = 95),
  results = extract_dcdh_multi(res9, n_effects = 5, n_placebos = 5)
)

# ---------------------------------------------------------------------------
# Phase 3: Covariate and linear-trends scenarios
# ---------------------------------------------------------------------------

# Helper: add a covariate column to a panel. The covariate is correlated with
# switch timing (confounding) but the true effect is constant.
add_covariate <- function(df, seed = 42, x_effect = 1.5) {
  set.seed(seed)
  n <- nrow(df)
  groups <- unique(df$group)
  # Group-level base value (correlated with which groups switch)
  x_base <- setNames(rnorm(length(groups), 0, 1), groups)
  # Time-varying component
  df$X1 <- x_base[as.character(df$group)] + 0.3 * df$period + rnorm(n, 0, 0.2)
  # Add covariate effect to outcome
  df$outcome <- df$outcome + x_effect * df$X1
  df
}

# Scenario 10: joiners_only with controls (L_max=2)
cat("  Scenario 10: joiners_only_controls\n")
d10 <- gen_reversible(n_groups = N_GOLDEN, n_periods = 8,
                      pattern = "joiners_only", seed = 110)
d10 <- add_covariate(d10, seed = 210, x_effect = 1.5)
res10 <- did_multiplegt_dyn(
  df = d10, outcome = "outcome", group = "group", time = "period",
  treatment = "treatment", effects = 2, placebo = 1, ci_level = 95,
  controls = "X1"
)
scenarios$joiners_only_controls <- list(
  data = list(
    group = as.numeric(d10$group),
    period = as.numeric(d10$period),
    treatment = as.numeric(d10$treatment),
    outcome = as.numeric(d10$outcome),
    X1 = as.numeric(d10$X1)
  ),
  params = list(pattern = "joiners_only", n_groups = N_GOLDEN, n_periods = 8,
                seed = 110, effects = 2, placebo = 1, ci_level = 95,
                controls = "X1"),
  results = extract_dcdh_multi(res10, n_effects = 2, n_placebos = 1)
)

# Scenario 11: joiners_only with trends_lin (L_max=2)
cat("  Scenario 11: joiners_only_trends_lin\n")
d11 <- gen_reversible(n_groups = N_GOLDEN, n_periods = 8,
                      pattern = "joiners_only", seed = 111)
# Add group-specific linear trends to outcome
set.seed(311)
groups11 <- unique(d11$group)
g_trends <- setNames(rnorm(length(groups11), 0, 0.5), groups11)
d11$outcome <- d11$outcome + g_trends[as.character(d11$group)] * d11$period
res11 <- did_multiplegt_dyn(
  df = d11, outcome = "outcome", group = "group", time = "period",
  treatment = "treatment", effects = 2, placebo = 1, ci_level = 95,
  trends_lin = TRUE
)
scenarios$joiners_only_trends_lin <- list(
  data = export_data(d11),
  params = list(pattern = "joiners_only", n_groups = N_GOLDEN, n_periods = 8,
                seed = 111, effects = 2, placebo = 1, ci_level = 95,
                trends_lin = TRUE),
  results = extract_dcdh_multi(res11, n_effects = 2, n_placebos = 1)
)

# Scenario 12: joiners_only with both controls and trends_lin (L_max=2)
cat("  Scenario 12: joiners_only_controls_trends_lin\n")
d12 <- gen_reversible(n_groups = N_GOLDEN, n_periods = 8,
                      pattern = "joiners_only", seed = 112)
d12 <- add_covariate(d12, seed = 212, x_effect = 1.5)
# Add group-specific linear trends
set.seed(312)
groups12 <- unique(d12$group)
g_trends12 <- setNames(rnorm(length(groups12), 0, 0.5), groups12)
d12$outcome <- d12$outcome + g_trends12[as.character(d12$group)] * d12$period
res12 <- did_multiplegt_dyn(
  df = d12, outcome = "outcome", group = "group", time = "period",
  treatment = "treatment", effects = 2, placebo = 1, ci_level = 95,
  controls = "X1", trends_lin = TRUE
)
scenarios$joiners_only_controls_trends_lin <- list(
  data = list(
    group = as.numeric(d12$group),
    period = as.numeric(d12$period),
    treatment = as.numeric(d12$treatment),
    outcome = as.numeric(d12$outcome),
    X1 = as.numeric(d12$X1)
  ),
  params = list(pattern = "joiners_only", n_groups = N_GOLDEN, n_periods = 8,
                seed = 112, effects = 2, placebo = 1, ci_level = 95,
                controls = "X1", trends_lin = TRUE),
  results = extract_dcdh_multi(res12, n_effects = 2, n_placebos = 1)
)

# ---------------------------------------------------------------------------
# Phase 3 extension: by_path per-path event-study disaggregation
# ---------------------------------------------------------------------------

# Helper: extract per-path by_path results. When did_multiplegt_dyn is
# called with by_path=k, the result object has no $results slot; instead
# per-path results live at res$by_level_1, res$by_level_2, ... in rank
# order (1 = most frequent observed path). res$by_levels is a character
# vector of comma-joined path labels (e.g. "0,1,1,1") in the same order.
extract_dcdh_by_path <- function(res, n_effects, n_placebos = 0) {
  by_levels <- res$by_levels
  out <- list()
  for (i in seq_along(by_levels)) {
    slot <- res[[paste0("by_level_", i)]]
    effects <- slot$results$Effects
    horizons <- list()
    for (h in seq_len(min(n_effects, nrow(effects)))) {
      horizons[[as.character(h)]] <- list(
        effect = as.numeric(effects[h, "Estimate"]),
        se = as.numeric(effects[h, "SE"]),
        ci_lo = as.numeric(effects[h, "LB CI"]),
        ci_hi = as.numeric(effects[h, "UB CI"]),
        n_switchers = as.numeric(effects[h, "Switchers"]),
        n_obs = as.numeric(effects[h, "N"])
      )
    }
    # Per-path placebos. When did_multiplegt_dyn is called with
    # by_path=k AND placebo=N, each by_level_i has its own
    # slot$results$Placebos table with N rows. Negative-keyed
    # ("-1", "-2", ...) so the Python parity loop can iterate the
    # full forward+backward horizon set with int(k) on the keys.
    if (n_placebos > 0) {
      placebos <- slot$results$Placebos
      if (!is.null(placebos)) {
        for (h in seq_len(min(n_placebos, nrow(placebos)))) {
          horizons[[as.character(-h)]] <- list(
            effect = as.numeric(placebos[h, "Estimate"]),
            se = as.numeric(placebos[h, "SE"]),
            ci_lo = as.numeric(placebos[h, "LB CI"]),
            ci_hi = as.numeric(placebos[h, "UB CI"]),
            n_switchers = as.numeric(placebos[h, "Switchers"]),
            n_obs = as.numeric(placebos[h, "N"])
          )
        }
      }
    }
    out[[i]] <- list(
      path = by_levels[i],
      frequency_rank = i,
      horizons = horizons
    )
  }
  list(by_path = out)
}

# Helper: extract global predict_het results. R's predict_het slot is at
# res$results$predict_het, a data.frame with columns
# {effect, covariate, Estimate, SE, t, LB, UB, N, pF}. Estimate is the
# WLS coefficient on the heterogeneity covariate.
#
# `n_effects` retained for backward-compat with scenarios 20/21 callers
# but unused: we iterate ALL rows in ph$effect and partition by sign so
# placebo (negative-effect) rows are captured separately. Scenario 22
# probes whether R emits negative-effect rows when called with
# `predict_het + placebo`; resolves TODO #422.
extract_dcdh_predict_het <- function(res, n_effects) {
  ph <- res$results$predict_het
  # `structure(list(), names = character(0))` produces a named list with
  # zero entries; jsonlite serializes it as `{}` (object) rather than
  # `[]` (array). Plain `list()` would serialize as `[]`, which gives
  # the JSON contract a type-unstable shape (object when populated, array
  # when empty). Type stability matters for generic consumers — see
  # `tests/test_chaisemartin_dhaultfoeuille_parity.py::_as_dict` for the
  # defensive Python-side coercion that backstops this.
  forward_horizons <- structure(list(), names = character(0))
  placebo_horizons <- structure(list(), names = character(0))
  if (is.null(ph) || nrow(ph) == 0) {
    return(list(predict_het = forward_horizons,
                placebo_predict_het = placebo_horizons))
  }
  for (h in seq_len(nrow(ph))) {
    effect_val <- as.numeric(ph$effect[h])
    entry <- list(
      beta = as.numeric(ph$Estimate[h]),
      se = as.numeric(ph$SE[h]),
      t = as.numeric(ph$t[h]),
      ci_lo = as.numeric(ph$LB[h]),
      ci_hi = as.numeric(ph$UB[h]),
      n_obs = as.numeric(ph$N[h]),
      p_value = as.numeric(ph$pF[h])
    )
    if (effect_val > 0) {
      forward_horizons[[as.character(effect_val)]] <- entry
    } else if (effect_val < 0) {
      placebo_horizons[[as.character(effect_val)]] <- entry
    }
    # effect_val == 0: skip (not a valid event-study horizon).
  }
  list(predict_het = forward_horizons,
       placebo_predict_het = placebo_horizons)
}

# Helper: extract per-path predict_het results. Under by_path=k +
# predict_het, R's per-by_level dispatcher writes a predict_het table to
# each res$by_level_i$results$predict_het. Output mirrors
# extract_dcdh_by_path's shape with a horizons dict keyed by horizon.
extract_dcdh_by_path_predict_het <- function(res, n_effects) {
  by_levels <- res$by_levels
  out <- list()
  for (i in seq_along(by_levels)) {
    slot <- res[[paste0("by_level_", i)]]
    ph <- slot$results$predict_het
    # See extract_dcdh_predict_het comment for the named-list rationale.
    forward_horizons <- structure(list(), names = character(0))
    placebo_horizons <- structure(list(), names = character(0))
    if (!is.null(ph) && nrow(ph) > 0) {
      # Iterate ALL rows; partition by sign so placebo (negative-effect)
      # rows are captured under `placebo_horizons`. Scenario 22 probes
      # whether R emits negative-effect rows on the per-path surface.
      for (h in seq_len(nrow(ph))) {
        effect_val <- as.numeric(ph$effect[h])
        entry <- list(
          beta = as.numeric(ph$Estimate[h]),
          se = as.numeric(ph$SE[h]),
          t = as.numeric(ph$t[h]),
          ci_lo = as.numeric(ph$LB[h]),
          ci_hi = as.numeric(ph$UB[h]),
          n_obs = as.numeric(ph$N[h]),
          p_value = as.numeric(ph$pF[h])
        )
        if (effect_val > 0) {
          forward_horizons[[as.character(effect_val)]] <- entry
        } else if (effect_val < 0) {
          placebo_horizons[[as.character(effect_val)]] <- entry
        }
      }
    }
    out[[i]] <- list(
      path = by_levels[i],
      frequency_rank = i,
      horizons = forward_horizons,
      placebo_horizons = placebo_horizons
    )
  }
  list(by_path_predict_het = out)
}

# Scenario 13: mixed_single_switch + by_path=2 (basic 2-path case).
# The mixed_single_switch DGP produces joiners (path 0,1,1,1) and
# leavers (path 1,0,0,0) as its only two observed paths at L_max=3, so
# by_path=2 captures both and tests core per-path parity.
cat("  Scenario 13: mixed_single_switch_by_path\n")
d13 <- gen_reversible(n_groups = N_GOLDEN, n_periods = 8,
                      pattern = "mixed_single_switch", seed = 113)
res13 <- did_multiplegt_dyn(
  df = d13, outcome = "outcome", group = "group", time = "period",
  treatment = "treatment", effects = 3, by_path = 2, ci_level = 95
)
scenarios$mixed_single_switch_by_path <- list(
  data = export_data(d13),
  params = list(pattern = "mixed_single_switch", n_groups = N_GOLDEN,
                n_periods = 8, seed = 113, effects = 3, by_path = 2,
                ci_level = 95),
  results = extract_dcdh_by_path(res13, n_effects = 3)
)

# Scenario 14: multi_path_reversible + by_path=3 (top-k ranking case).
# The `multi_path_reversible` pattern is a DETERMINISTIC multi-path DGP:
# path assignment is a fixed function of F_g (so every (D_{g,1}, F_g,
# S_g) cohort contains switchers from a single path), path proportions
# are fixed at 20/20/15/10/10/5 across the 6 F_g values, and
# post-window treatment is stable at path[L_max+1]. by_path=3 exercises
# top-k selection when observed paths exceed k (4 observed paths, top-3
# selected). n_periods=10 gives every switch_time a complete length-
# (L_max+1) window. The old `p_switch`-driven random-toggle variant
# (pre-PR) blew out SE parity with R via cross-path cohort mixing;
# see the REGISTRY.md `Note (Phase 3 by_path ...)` Deviation bullet.
cat("  Scenario 14: multi_path_reversible_by_path\n")
d14 <- gen_reversible(n_groups = N_GOLDEN, n_periods = 10,
                      pattern = "multi_path_reversible", seed = 114,
                      L_max = 3)
res14 <- did_multiplegt_dyn(
  df = d14, outcome = "outcome", group = "group", time = "period",
  treatment = "treatment", effects = 3, by_path = 3, ci_level = 95
)
scenarios$multi_path_reversible_by_path <- list(
  data = export_data(d14),
  params = list(pattern = "multi_path_reversible", n_groups = N_GOLDEN,
                n_periods = 10, seed = 114, effects = 3, by_path = 3,
                ci_level = 95),
  results = extract_dcdh_by_path(res14, n_effects = 3)
)

# Scenario 15: multi_path_reversible + by_path=3 + placebo=2 (per-path
# backward placebo case). Same deterministic DGP and n_periods=10 as
# scenario 14 (the DGP's `f_g_to_path` is sized for max_switch=6, fixed
# at L_max=3 + n_periods=10). For placebo=2: F_g=2 cohort has backward
# index F_g-1-2=-1 out of range, so those 20 switchers contribute NaN
# at lag=2; F_g in [3..7] (60 switchers) produce a valid lag=2 estimate.
# R drops the F_g=2 cohort from Placebo_2 automatically; the parity
# test compares only over the rows that R produced.
cat("  Scenario 15: multi_path_reversible_by_path_placebo\n")
d15 <- gen_reversible(n_groups = N_GOLDEN, n_periods = 10,
                      pattern = "multi_path_reversible", seed = 115,
                      L_max = 3)
res15 <- did_multiplegt_dyn(
  df = d15, outcome = "outcome", group = "group", time = "period",
  treatment = "treatment", effects = 3, placebo = 2, by_path = 3,
  ci_level = 95
)
scenarios$multi_path_reversible_by_path_placebo <- list(
  data = export_data(d15),
  # n_switcher_groups records the switcher cohort count fed into
  # gen_reversible's `counts_per_F_g` allocator (80 = sum c(20, 20, 15,
  # 10, 10, 5)); the realized panel has 120 unique groups after the
  # default 20 never-treated + 20 always-treated control rows are
  # appended (gen_reversible defaults at line 64). Recording both fields
  # avoids the metadata-vs-data mismatch the reviewer flagged on
  # PR #371 R2: anyone reusing this scenario's metadata sees both the
  # switcher count (the load-bearing number for the DGP allocation) and
  # the realized panel size.
  params = list(pattern = "multi_path_reversible",
                n_switcher_groups = N_GOLDEN, n_realized_groups = N_GOLDEN + 40L,
                n_periods = 10, seed = 115, effects = 3, placebo = 2,
                by_path = 3, ci_level = 95),
  results = extract_dcdh_by_path(res15, n_effects = 3, n_placebos = 2)
)

# Scenario 16: multi_path_reversible + by_path=3 + controls="X1" (Phase 3
# Wave 3 #5: by_path + DID^X residualization). Same deterministic DGP
# and n_periods=10 as scenarios 14/15, with a confounding covariate X1
# added via the same `add_covariate` helper used by scenario 10's
# `joiners_only_controls`. **R re-runs `did_multiplegt_main()` per path**
# with a path-restricted subsample (path's switchers + same-baseline
# not-yet-treated controls), so its per-baseline OLS residualization
# coefficients can vary per path (verified against
# `chaisemartinPackages/did_multiplegt_dyn` source —
# `R/R/did_multiplegt_dyn.R` lines 393-411 dispatch the per-path loop;
# `did_multiplegt_by_path` is a path-classifier preprocessor only).
# Python residualizes once on the full panel before path enumeration,
# then disaggregates per path. **The two strategies coincide on
# single-baseline switcher panels** (every switcher shares D_{g,1}=0)
# because R's per-path control pool then equals the global control pool
# # — `multi_path_reversible` is built precisely for this property, so
# per-path event-study point estimates and switcher counts must match R
# bit-exactly on the one-observation-per-(g,t) DGP this generator
# produces. (On panels with multiple observations per `(g, t)` cell, the
# library's equal-cell-weighting first stage diverges from R's `N_gt`-
# weighted first stage per the existing DID^X cell-weighting deviation
# in `docs/methodology/REGISTRY.md` "Note (Phase 3 DID^X covariate
# adjustment)" — that deviation is independent of the by_path lift.)
# Per-path SE inherits the documented cross-path cohort-sharing
# deviation from R for `path_effects`. On multi-baseline switcher panels
# the residualization coefficients can diverge per path between Python
# and R; the production fit emits a `UserWarning` in that configuration.
# Single covariate keeps the scenario tight; multi-covariate is
# exercised via internal regression tests.
cat("  Scenario 16: multi_path_reversible_by_path_controls\n")
d16 <- gen_reversible(n_groups = N_GOLDEN, n_periods = 10,
                      pattern = "multi_path_reversible", seed = 116,
                      L_max = 3)
d16 <- add_covariate(d16, seed = 216, x_effect = 1.5)
res16 <- did_multiplegt_dyn(
  df = d16, outcome = "outcome", group = "group", time = "period",
  treatment = "treatment", effects = 3, by_path = 3, controls = "X1",
  ci_level = 95
)
scenarios$multi_path_reversible_by_path_controls <- list(
  data = list(
    group = as.numeric(d16$group),
    period = as.numeric(d16$period),
    treatment = as.numeric(d16$treatment),
    outcome = as.numeric(d16$outcome),
    X1 = as.numeric(d16$X1)
  ),
  params = list(pattern = "multi_path_reversible",
                n_switcher_groups = N_GOLDEN, n_realized_groups = N_GOLDEN + 40L,
                n_periods = 10, seed = 116, effects = 3, by_path = 3,
                controls = "X1", ci_level = 95),
  results = extract_dcdh_by_path(res16, n_effects = 3)
)

# Scenario 17: single-baseline multi-path + by_path=3 + trends_lin=TRUE
# (Phase 3 Wave 3 #6: by_path + DID^{fd} group-specific linear trends).
# Custom inline single-baseline DGP — `multi_path_reversible` (Scenarios
# 13-16) concentrates each path on 1-2 F_g values, which after R's
# per-path subset + trends_lin's F_g==2 filter collapses to a single F_g,
# violating the dCDH staggered design restriction inside R's per-path
# `did_multiplegt_main` call. `mixed_single_switch` (Scenario 17's first
# attempt) is MULTI-baseline (joiners + leavers), which triggers the
# documented multi-baseline divergence between Python's global-then-
# disaggregate architecture and R's per-path full-pipeline call (same
# divergence pattern as `controls`; rel diffs of 7-19% on point estimates
# observed). To get clean single-baseline parity AND avoid the trends_lin
# F_g concentration trap, this scenario uses a custom DGP: all groups
# start at D=0 (single baseline), 3 paths span F_g ∈ {3,4,5} (all >= 3
# so trends_lin's F_g==2 filter is a no-op), per-path counts unequal so
# top-k ranking is deterministic. **R returns the cumulated level effect
# delta_l per horizon, NOT the raw second-difference DID^{fd}_l** —
# verified empirically against the existing `joiners_only_trends_lin`
# parity test (`tests/test_chaisemartin_dhaultfoeuille_parity.py:403-409`
# documents this convention). The Python parity test compares Python's
# `path_cumulated_event_study[path][l]` against R's per-path Effect_l.
# Placebos under trends_lin remain RAW per-horizon (no cumulated placebo
# surface in R either), so the Python parity test compares
# `path_placebo_event_study[path][-l]` against R's per-path Placebo_l
# directly. Cumulated SE_RTOL is widened (~0.20 vs 0.12 used for
# non-cumulated by_path parity) because the conservative upper-bound SE
# (sum of per-horizon component SEs) compounds the cross-path
# cohort-sharing deviation under summation.
cat("  Scenario 17: single_baseline_multi_path_by_path_trends_lin\n")
{
  # Custom DGP: 80 switchers across 3 paths × 2 distinct F_g per path,
  # all single-baseline (D_{g,1}=0). Each F_g maps to exactly ONE path
  # (cohort-single-path, eliminating the cross-path cohort-sharing
  # deviation from R that PR #360 documented for path_effects). All
  # F_g >= 3 so trends_lin's F_g==2 filter is a no-op. Two distinct
  # F_g per path satisfies R's staggered-design requirement inside the
  # per-path `did_multiplegt_main` call. n_periods=11, L_max=3 gives
  # F_g in [2,8] = 7 values; we use F_g in {3..8} = 6 values, two per
  # path. Plus 20 never-treated + 20 always-treated controls
  # (n_realized_groups = 120).
  set.seed(117)
  n_periods17 <- 13
  L_max17 <- 3
  target_paths17 <- list(
    c(0L, 1L, 1L, 1L),  # path 1, sustained on (rank 1)
    c(0L, 1L, 1L, 0L),  # path 2, on-then-off  (rank 2)
    c(0L, 1L, 0L, 0L)   # path 3, on briefly   (rank 3)
  )
  # F_g-to-path mapping (each F_g unique to one path).
  # All F_g >= 4 to avoid the F_g=3 boundary case under trends_lin —
  # F_g=3 leaves only 1 valid pre-window Z value after the time==1
  # filter, causing R's per-path call (which re-runs the full pipeline
  # on the path subset) to handle pre-window/control eligibility
  # differently from Python's global first-differencing. F_g >= 4 gives
  # both implementations 2+ valid pre-window Z values, eliminating the
  # boundary-case divergence.
  #   F_g=4 -> path 1, F_g=5 -> path 1   (path 1 = 38)
  #   F_g=6 -> path 2, F_g=7 -> path 2   (path 2 = 24)
  #   F_g=8 -> path 3, F_g=9 -> path 3   (path 3 = 18)
  # Total switchers = 80
  fg_path_counts17 <- list(
    list(F_g = 4L, path_idx = 1L, count = 20L),
    list(F_g = 5L, path_idx = 1L, count = 18L),
    list(F_g = 6L, path_idx = 2L, count = 13L),
    list(F_g = 7L, path_idx = 2L, count = 11L),
    list(F_g = 8L, path_idx = 3L, count = 11L),
    list(F_g = 9L, path_idx = 3L, count = 7L)
  )
  n_switchers17 <- sum(sapply(fg_path_counts17, function(x) x$count))
  stopifnot(n_switchers17 == 80L)
  D17 <- matrix(0L, nrow = n_switchers17, ncol = n_periods17)
  g17 <- 1L
  for (entry in fg_path_counts17) {
    F_g <- entry$F_g
    target <- target_paths17[[entry$path_idx]]
    n_here <- entry$count
    for (k in seq_len(n_here)) {
      # Pre-baseline [1..F_g-2]: D=0 (single-baseline contract)
      if (F_g >= 3L) D17[g17, 1:(F_g - 2L)] <- 0L
      # Window [F_g-1..F_g-1+L_max]: target path
      for (j in 0:L_max17) D17[g17, F_g - 1L + j] <- target[j + 1L]
      # Post-window: stable at path[L_max+1]
      if (F_g + L_max17 <= n_periods17) {
        D17[g17, (F_g + L_max17):n_periods17] <- target[L_max17 + 1L]
      }
      g17 <- g17 + 1L
    }
  }
  # Append 20 never-treated and 20 always-treated controls
  D17 <- rbind(D17,
               matrix(0L, nrow = 20L, ncol = n_periods17),
               matrix(1L, nrow = 20L, ncol = n_periods17))
  n_total17 <- nrow(D17)
  # Generate fixed effects, treatment effects, outcomes (mirror gen_reversible
  # parameters: group_fe_sd=2.0, treatment_effect=2.0, time_trend=0.1, noise_sd=0.5)
  set.seed(117L)
  group_fe17 <- rnorm(n_total17, 0, 2.0)
  noise17 <- matrix(rnorm(n_total17 * n_periods17, 0, 0.5),
                    nrow = n_total17, ncol = n_periods17)
  period_arr17 <- 0:(n_periods17 - 1L)
  Y17 <- 10.0 +
    matrix(group_fe17, nrow = n_total17, ncol = n_periods17) +
    matrix(0.1 * period_arr17, nrow = n_total17, ncol = n_periods17, byrow = TRUE) +
    2.0 * D17 +
    noise17
  # Build long data frame
  d17 <- data.frame(
    group = rep(seq_len(n_total17) - 1L, each = n_periods17),
    period = rep(period_arr17, n_total17),
    treatment = as.vector(t(D17)),
    outcome = as.vector(t(Y17))
  )
  # Inject per-group linear trends (Scenario 11 pattern)
  set.seed(217L)
  groups17 <- sort(unique(d17$group))
  g_trends17 <- setNames(rnorm(length(groups17), 0, 0.5),
                         as.character(groups17))
  d17$outcome <- d17$outcome +
    g_trends17[as.character(d17$group)] * d17$period
  res17 <- did_multiplegt_dyn(
    df = d17, outcome = "outcome", group = "group", time = "period",
    treatment = "treatment", effects = 3, placebo = 1, by_path = 3,
    trends_lin = TRUE, ci_level = 95
  )
  scenarios$single_baseline_multi_path_by_path_trends_lin <- list(
    data = list(
      group = as.numeric(d17$group),
      period = as.numeric(d17$period),
      treatment = as.numeric(d17$treatment),
      outcome = as.numeric(d17$outcome)
    ),
    params = list(pattern = "single_baseline_multi_path",
                  n_switcher_groups = 80L, n_realized_groups = 120L,
                  n_periods = 13L, seed = 117L, effects = 3, placebo = 1,
                  by_path = 3, trends_lin = TRUE, ci_level = 95),
    results = extract_dcdh_by_path(res17, n_effects = 3, n_placebos = 1)
  )
}

# Scenario 18: multi_path_reversible + by_path=3 + trends_nonparam="state"
# (Phase 3 Wave 3 #7: by_path + state-set trends). Same deterministic DGP
# and n_periods=10 as Scenarios 16/17, with a 3-state column added
# (deterministic per-group assignment via `((group - 1) %% 3) + 1` so
# within-set controls are guaranteed to exist for each path). **R does
# NOT cumulate or first-difference under trends_nonparam** — Effect_l
# per horizon is a normal DID with set-restricted control pool. The
# Python parity test compares per-path raw DID per (path, l) directly
# against R's per-path Effect_l. Placebos likewise are raw per-horizon.
# Per-path R parity matches exactly on single-baseline panels.
cat("  Scenario 18: multi_path_reversible_by_path_trends_nonparam\n")
d18 <- gen_reversible(n_groups = N_GOLDEN, n_periods = 10,
                      pattern = "multi_path_reversible", seed = 118,
                      L_max = 3)
d18$state <- ((d18$group - 1) %% 3) + 1
res18 <- did_multiplegt_dyn(
  df = d18, outcome = "outcome", group = "group", time = "period",
  treatment = "treatment", effects = 3, placebo = 1, by_path = 3,
  trends_nonparam = "state", ci_level = 95
)
scenarios$multi_path_reversible_by_path_trends_nonparam <- list(
  data = list(
    group = as.numeric(d18$group),
    period = as.numeric(d18$period),
    treatment = as.numeric(d18$treatment),
    outcome = as.numeric(d18$outcome),
    state = as.numeric(d18$state)
  ),
  params = list(pattern = "multi_path_reversible",
                n_switcher_groups = N_GOLDEN, n_realized_groups = N_GOLDEN + 40L,
                n_periods = 10, seed = 118, effects = 3, placebo = 1,
                by_path = 3, trends_nonparam = "state", ci_level = 95),
  results = extract_dcdh_by_path(res18, n_effects = 3, n_placebos = 1)
)

# Scenario 19: by_path + non-binary integer treatment (D in {0, 1, 2}).
# Phase 3 Wave 3 #8 lift. Custom inline DGP (mirror Scenario 17 structure)
# with 3 single-baseline non-binary paths: low-dose sustained
# (0, 1, 1, 1), high-dose sustained (0, 2, 2, 2), and ramp-up
# (0, 1, 2, 2). All F_g >= 4 (defensive: avoids any pre-window boundary
# edge cases under future trends_lin combinations and matches Scenario 17).
# 78 switchers + 20 never-treated (D=0) + 20 always-treated (D=2) controls.
# n_periods=13, L_max=3.
#
# R's substr(path, 1, 1) baseline-derivation in did_multiplegt_by_path
# is correct for D in {0..9} (single-digit decimal); we stay in {0, 1, 2}
# so no R bug interferes. Python's tuple-key matching is correct
# regardless of D range.
cat("  Scenario 19: multi_path_reversible_by_path_non_binary\n")
{
  set.seed(119)
  n_periods19 <- 13
  L_max19 <- 3
  target_paths19 <- list(
    c(0L, 1L, 1L, 1L),  # path 1, low-dose sustained (rank 1)
    c(0L, 2L, 2L, 2L),  # path 2, high-dose sustained (rank 2)
    c(0L, 1L, 2L, 2L)   # path 3, ramp-up            (rank 3)
  )
  fg_path_counts19 <- list(
    list(F_g = 4L, path_idx = 1L, count = 18L),
    list(F_g = 5L, path_idx = 1L, count = 14L),
    list(F_g = 6L, path_idx = 2L, count = 14L),
    list(F_g = 7L, path_idx = 2L, count = 12L),
    list(F_g = 8L, path_idx = 3L, count = 12L),
    list(F_g = 9L, path_idx = 3L, count = 8L)
  )
  n_switchers19 <- sum(sapply(fg_path_counts19, function(x) x$count))
  stopifnot(n_switchers19 == 78L)
  D19 <- matrix(0L, nrow = n_switchers19, ncol = n_periods19)
  g19 <- 1L
  for (entry in fg_path_counts19) {
    F_g <- entry$F_g
    target <- target_paths19[[entry$path_idx]]
    n_here <- entry$count
    for (k in seq_len(n_here)) {
      if (F_g >= 3L) D19[g19, 1:(F_g - 2L)] <- 0L
      for (j in 0:L_max19) D19[g19, F_g - 1L + j] <- target[j + 1L]
      if (F_g + L_max19 <= n_periods19) {
        D19[g19, (F_g + L_max19):n_periods19] <- target[L_max19 + 1L]
      }
      g19 <- g19 + 1L
    }
  }
  # Append 20 never-treated (D=0) and 20 always-treated (D=2) controls
  D19 <- rbind(
    D19,
    matrix(0L, nrow = 20L, ncol = n_periods19),
    matrix(2L, nrow = 20L, ncol = n_periods19)
  )
  n_total19 <- nrow(D19)
  set.seed(119L)
  group_fe19 <- rnorm(n_total19, 0, 2.0)
  noise19 <- matrix(rnorm(n_total19 * n_periods19, 0, 0.5),
                    nrow = n_total19, ncol = n_periods19)
  period_arr19 <- 0:(n_periods19 - 1L)
  Y19 <- 10.0 +
    matrix(group_fe19, nrow = n_total19, ncol = n_periods19) +
    matrix(0.1 * period_arr19, nrow = n_total19, ncol = n_periods19, byrow = TRUE) +
    1.5 * D19 +
    noise19
  d19 <- data.frame(
    group = rep(seq_len(n_total19) - 1L, each = n_periods19),
    period = rep(period_arr19, n_total19),
    treatment = as.vector(t(D19)),
    outcome = as.vector(t(Y19))
  )
  res19 <- did_multiplegt_dyn(
    df = d19, outcome = "outcome", group = "group", time = "period",
    treatment = "treatment", effects = 3, placebo = 1, by_path = 3,
    ci_level = 95
  )
  scenarios$multi_path_reversible_by_path_non_binary <- list(
    data = list(
      group = as.numeric(d19$group),
      period = as.numeric(d19$period),
      treatment = as.numeric(d19$treatment),
      outcome = as.numeric(d19$outcome)
    ),
    params = list(pattern = "single_baseline_multi_path_non_binary",
                  n_switcher_groups = 78L, n_realized_groups = 118L,
                  n_periods = 13L, seed = 119L, effects = 3, placebo = 1,
                  by_path = 3, ci_level = 95),
    results = extract_dcdh_by_path(res19, n_effects = 3, n_placebos = 1)
  )
}

# Scenarios 20 + 21: predict_het R-parity (Wave 5 #11). Scenario 20 is
# the global anchor (predict_het without by_path); scenario 21 is the
# per-path version (by_path + predict_het). Both use the same DGP so the
# Python-side parity tests can calibrate atol on the global scenario and
# inherit the same atol for the per-path test. R's predict_het syntax
# requires a 2-element list: list(covariate_name, horizons_vec).
#
# DGP shape: 90 switchers + 30 never-treated controls, 10 periods, 3
# paths (0,1,1,1) / (0,1,0,0) / (0,1,1,0), F_g varies in {3,4,5}
# INDEPENDENTLY of path so each path has multiple cohorts. het_x is
# binary {0,1}, balanced across both switchers and controls. Effect =
# 5.0 + 3.0 * het_x to produce a detectable heterogeneity signal.
# Never-treated controls are required for R's predict_het to compute
# horizons l>=2 under reversal paths (otherwise R returns
# "max effects = 2" or empty cohort dummies). `dont_drop_larger_lower
# = TRUE` matches the Python by_path requirement that
# `drop_larger_lower=False`.
cat("  Scenarios 20/21: multi_path_reversible_predict_het + by_path version\n")
{
  set.seed(120L)
  n_switchers20 <- 90L
  n_controls20 <- 30L
  n_groups20 <- n_switchers20 + n_controls20
  n_periods20 <- 10L
  paths20 <- list(c(0L, 1L, 1L, 1L), c(0L, 1L, 0L, 0L), c(0L, 1L, 1L, 0L))
  D20 <- matrix(0L, nrow = n_groups20, ncol = n_periods20)
  het_x20 <- integer(n_groups20)
  group_fe20 <- rnorm(n_groups20, 0, 2.0)
  # Switchers (groups 1..n_switchers20)
  for (g in seq_len(n_switchers20)) {
    F_g_choice <- ((g - 1L) %/% 3L) %% 3L
    F_g <- 3L + F_g_choice
    path_idx <- ((g - 1L) %% 3L) + 1L
    pp <- paths20[[path_idx]]
    het_x20[g] <- if (g <= n_switchers20 %/% 2L) 1L else 0L
    for (j in seq_along(pp)) {
      t <- F_g - 1L + j
      if (t >= 1L && t <= n_periods20) D20[g, t] <- pp[j]
    }
    if (F_g - 1L + length(pp) <= n_periods20) {
      tail_t <- (F_g - 1L + length(pp)):n_periods20
      D20[g, tail_t] <- pp[length(pp)]
    }
  }
  # Never-treated controls (groups n_switchers20+1..n_groups20).
  # Balanced het_x assignment so the heterogeneity covariate has
  # variation in the control pool too.
  for (g in (n_switchers20 + 1L):n_groups20) {
    k <- g - n_switchers20
    het_x20[g] <- if (k <= n_controls20 %/% 2L) 1L else 0L
  }
  noise20 <- matrix(rnorm(n_groups20 * n_periods20, 0, 0.5),
                    nrow = n_groups20, ncol = n_periods20)
  period_arr20 <- 0:(n_periods20 - 1L)
  effect20 <- 5.0 + 3.0 * het_x20  # heterogeneity signal
  Y20 <- matrix(group_fe20, nrow = n_groups20, ncol = n_periods20) +
    matrix(0.5 * period_arr20, nrow = n_groups20, ncol = n_periods20, byrow = TRUE) +
    matrix(effect20, nrow = n_groups20, ncol = n_periods20) * D20 +
    noise20
  d20 <- data.frame(
    group = rep(seq_len(n_groups20) - 1L, each = n_periods20),
    period = rep(period_arr20, n_groups20),
    treatment = as.vector(t(D20)),
    outcome = as.vector(t(Y20)),
    het_x = rep(het_x20, each = n_periods20)
  )

  # Scenario 20: global predict_het anchor (no by_path).
  # `dont_drop_larger_lower = TRUE` preserves multi-switch cohorts so the
  # heterogeneity regression has cohort variation at every horizon.
  # Without this, R drops off-switch paths at l>=2, leaving a single
  # cohort and triggering the `prod_het_l_XX ~ het_x + ` empty-cohort
  # error. dCDH `by_path` requires `drop_larger_lower=False` on the Python
  # side anyway, so this flag is consistent with the per-path scope.
  res20 <- did_multiplegt_dyn(
    df = d20, outcome = "outcome", group = "group", time = "period",
    treatment = "treatment", effects = 3,
    dont_drop_larger_lower = TRUE,
    predict_het = list("het_x", c(1, 2, 3)),
    ci_level = 95, graph_off = TRUE
  )
  scenarios$multi_path_reversible_predict_het <- list(
    data = list(
      group = as.numeric(d20$group),
      period = as.numeric(d20$period),
      treatment = as.numeric(d20$treatment),
      outcome = as.numeric(d20$outcome),
      het_x = as.numeric(d20$het_x)
    ),
    params = list(pattern = "multi_path_reversible_predict_het",
                  n_switchers = n_switchers20, n_controls = n_controls20,
                  n_groups = n_groups20, n_periods = n_periods20,
                  seed = 120L, effects = 3,
                  predict_het_var = "het_x",
                  predict_het_horizons = c(1, 2, 3),
                  ci_level = 95,
                  dont_drop_larger_lower = TRUE),
    results = extract_dcdh_predict_het(res20, n_effects = 3)
  )

  # Scenario 21: by_path + predict_het (per-path version on same DGP).
  # `dont_drop_larger_lower = TRUE` matches scenario 20 + Python
  # by_path's `drop_larger_lower=False` requirement.
  res21 <- did_multiplegt_dyn(
    df = d20, outcome = "outcome", group = "group", time = "period",
    treatment = "treatment", effects = 3, by_path = 3,
    dont_drop_larger_lower = TRUE,
    predict_het = list("het_x", c(1, 2, 3)),
    ci_level = 95, graph_off = TRUE
  )
  scenarios$multi_path_reversible_by_path_predict_het <- list(
    data = list(
      group = as.numeric(d20$group),
      period = as.numeric(d20$period),
      treatment = as.numeric(d20$treatment),
      outcome = as.numeric(d20$outcome),
      het_x = as.numeric(d20$het_x)
    ),
    params = list(pattern = "multi_path_reversible_by_path_predict_het",
                  n_switchers = n_switchers20, n_controls = n_controls20,
                  n_groups = n_groups20, n_periods = n_periods20,
                  seed = 120L, effects = 3, by_path = 3,
                  predict_het_var = "het_x",
                  predict_het_horizons = c(1, 2, 3),
                  ci_level = 95,
                  dont_drop_larger_lower = TRUE),
    results = extract_dcdh_by_path_predict_het(res21, n_effects = 3)
  )

  # Scenario 23: GLOBAL predict_het + placebo (no by_path). Mirrors
  # scenario 22's syntax minus by_path so we have a parity anchor for
  # the GLOBAL `results.heterogeneity_effects` surface emitting both
  # forward and backward (placebo) horizons. Resolves codex R1 P1 #2:
  # the Phase 1A change extended the global heterogeneity loop to
  # cover backward horizons, so a global-surface parity test was
  # required to lock that contract independently of the per-path
  # dispatcher. Same `c(-1)` sentinel as scenario 22 (computes ALL
  # forward + ALL placebo positions); reuses `d20` for DGP parity.
  res23 <- did_multiplegt_dyn(
    df = d20, outcome = "outcome", group = "group", time = "period",
    treatment = "treatment", effects = 3, placebo = 2,
    dont_drop_larger_lower = TRUE,
    predict_het = list("het_x", c(-1)),
    ci_level = 95, graph_off = TRUE
  )
  scenarios$multi_path_reversible_predict_het_with_placebo_global <- list(
    data = list(
      group = as.numeric(d20$group),
      period = as.numeric(d20$period),
      treatment = as.numeric(d20$treatment),
      outcome = as.numeric(d20$outcome),
      het_x = as.numeric(d20$het_x)
    ),
    params = list(pattern = "multi_path_reversible_predict_het_with_placebo_global",
                  n_switchers = n_switchers20, n_controls = n_controls20,
                  n_groups = n_groups20, n_periods = n_periods20,
                  seed = 120L, effects = 3, placebo = 2,
                  predict_het_var = "het_x",
                  predict_het_horizons = c(-1),
                  ci_level = 95,
                  dont_drop_larger_lower = TRUE),
    results = extract_dcdh_predict_het(res23, n_effects = 3)
  )

  # Scenario 22: by_path + predict_het + placebo (probes TODO #422). Reuses
  # d20 from scenarios 20/21 for DGP parity. Tests whether R's
  # did_multiplegt_dyn(by_path=k, predict_het, placebo=N) per-by_level
  # dispatcher emits predict_het rows on backward (placebo) horizons.
  #
  # R's predict_het syntax with `placebo > 0` (per did_multiplegt_main
  # source `DIDmultiplegtDYN:::did_multiplegt_main` lines 1907 / 2030):
  # the SAME horizon vector is used for BOTH forward effects AND placebo
  # positions. Passing `c(1, 2, 3)` with `placebo=2` errors because
  # `max(c(1, 2, 3)) > placebo=2`. The `c(-1)` sentinel triggers "compute
  # heterogeneity for ALL forward (1..effects) AND ALL placebo
  # (1..placebo) positions" by replacing `het_effects` with `1:l_XX` in
  # the forward block and `1:l_placebo_XX` in the placebo block. Forward
  # rows are emitted with positive `effect` values (1, 2, 3); placebo
  # rows with NEGATIVE values (-1, -2) per `effect = matrix(-i, ...)` at
  # the placebo block's rbind site.
  #
  # The extended extract_dcdh_by_path_predict_het partitions the per-path
  # predict_het table by `effect` sign: forward into `horizons`, placebo
  # into `placebo_horizons`.
  res22 <- did_multiplegt_dyn(
    df = d20, outcome = "outcome", group = "group", time = "period",
    treatment = "treatment", effects = 3, placebo = 2, by_path = 3,
    dont_drop_larger_lower = TRUE,
    predict_het = list("het_x", c(-1)),
    ci_level = 95, graph_off = TRUE
  )
  scenarios$multi_path_reversible_predict_het_with_placebo <- list(
    data = list(
      group = as.numeric(d20$group),
      period = as.numeric(d20$period),
      treatment = as.numeric(d20$treatment),
      outcome = as.numeric(d20$outcome),
      het_x = as.numeric(d20$het_x)
    ),
    params = list(pattern = "multi_path_reversible_predict_het_with_placebo",
                  n_switchers = n_switchers20, n_controls = n_controls20,
                  n_groups = n_groups20, n_periods = n_periods20,
                  seed = 120L, effects = 3, placebo = 2, by_path = 3,
                  predict_het_var = "het_x",
                  predict_het_horizons = c(-1),
                  ci_level = 95,
                  dont_drop_larger_lower = TRUE),
    results = extract_dcdh_by_path_predict_het(res22, n_effects = 3)
  )
}

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------
dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)
writeLines(
  toJSON(
    list(scenarios = scenarios,
         generator = "generate_reversible_did_data v1",
         dcdh_package = paste0("DIDmultiplegtDYN ", utils::packageVersion("DIDmultiplegtDYN"))),
    auto_unbox = TRUE,
    digits = 10,
    pretty = TRUE
  ),
  output_path
)
cat(sprintf("dCDH golden values written to %s\n", output_path))
cat(sprintf("File size: %.1f KB\n", file.info(output_path)$size / 1024))
cat(sprintf("Scenarios: %d\n", length(scenarios)))
