#!/usr/bin/env Rscript
# Generate R bacondecomp parity goldens for diff-diff BaconDecomposition.
#
# Requires: install.packages("bacondecomp")  (CRAN; main function is bacon())
#           install.packages("jsonlite")
# Output:   ../data/r_bacondecomp_golden.json
#
# The diff-diff BaconDecomposition implementation (`diff_diff/bacon.py`) with
# the default ``weights="exact"`` is expected to match the values in this JSON
# at atol=1e-6 along a three-tier contract:
#   (1) aggregate TWFE coefficient + weights-sum on all 3 fixtures;
#   (2) direct per-component (treated, control, type) parity on the 2
#       non-remap fixtures AND on the 6 timing-vs-timing rows of
#       `always_treated_remapped`;
#   (3) cohort-level fold-back parity for the U bucket on
#       `always_treated_remapped` — Python's paper-footnote-11 remap folds
#       R's separate `Later vs Always Treated` + `Treated vs Untreated`
#       rows into a single `treated_vs_never` cell per cohort, so the
#       aggregate is invariant per Theorem 1 but the per-component
#       breakdown differs by convention. See REGISTRY notes:
#       `**Note (R parity convention divergence on always-treated)**` and
#       `**Deviation (first-period boundary extension on always-treated remap)**`.
# The ``weights="approximate"`` path is a library-only optimization and is
# NOT covered by this parity harness.
#
# Three fixtures:
#   1. uniform_3groups_with_never_treated — 3 timing groups + never-treated U;
#      exercises all three comparison types (treated/never, earlier/later,
#      later/earlier).
#   2. two_groups_no_never_treated — 2 timing groups only; tests the
#      timing-only decomposition where the s_{kU} terms drop.
#   3. always_treated_remapped — 3 timing groups + 1 always-treated cohort
#      (first_treat = 1). Validates the convention-divergent U-bucket
#      fold-back on Python's warn+remap of always-treated units into U.
#
# Run:
#   cd benchmarks/R && Rscript generate_bacon_golden.R

suppressPackageStartupMessages({
  library(bacondecomp)
  library(jsonlite)
})

stopifnot(packageVersion("bacondecomp") >= "0.1.0")

# ---------------------------------------------------------------------------
# DGP helpers
# ---------------------------------------------------------------------------

# Build a balanced panel with absorbing treatment.
#   n_units       : units per timing group (excluding never-treated)
#   n_periods     : panel length (1..T)
#   cohort_times  : vector of first-treatment times, one per cohort
#   always_treated_count : optional cohort treated at first_treat = 1
#                          (i.e., always-treated for the observable window)
#   never_treated_count  : units with first_treat = 0
#   true_effect          : constant ATT
#   seed                 : reproducibility
build_panel <- function(n_units_per_cohort, n_periods, cohort_times,
                        always_treated_count = 0L, never_treated_count = 0L,
                        true_effect = 2.0, seed = 42L) {
  set.seed(seed)
  units <- list()
  uid <- 1L

  # Always-treated cohort (first_treat = 1; treated in every observable period)
  if (always_treated_count > 0L) {
    for (i in seq_len(always_treated_count)) {
      units[[length(units) + 1L]] <- data.frame(
        unit = uid, time = seq_len(n_periods), first_treat = 1L
      )
      uid <- uid + 1L
    }
  }

  # Never-treated U
  if (never_treated_count > 0L) {
    for (i in seq_len(never_treated_count)) {
      units[[length(units) + 1L]] <- data.frame(
        unit = uid, time = seq_len(n_periods), first_treat = 0L
      )
      uid <- uid + 1L
    }
  }

  # Treated cohorts
  for (g in cohort_times) {
    for (i in seq_len(n_units_per_cohort)) {
      units[[length(units) + 1L]] <- data.frame(
        unit = uid, time = seq_len(n_periods), first_treat = as.integer(g)
      )
      uid <- uid + 1L
    }
  }

  df <- do.call(rbind, units)

  # Treatment indicator: D_{it} = 1 iff first_treat in {1,..,T} AND time >= first_treat.
  df$D <- as.integer(df$first_treat > 0L & df$time >= df$first_treat)

  # Outcome: unit FE + linear time + constant treatment effect + noise.
  unit_fe <- rnorm(uid - 1L, sd = 2.0)
  df$y <- unit_fe[df$unit] +
          0.1 * df$time +
          true_effect * df$D +
          rnorm(nrow(df), sd = 0.5)

  df
}

# ---------------------------------------------------------------------------
# Extract bacondecomp::bacon() output into a fixture-shaped list.
# ---------------------------------------------------------------------------

extract_bacon <- function(df, fixture_name) {
  # bacondecomp::bacon takes the OUTCOME ~ TREATMENT formula plus id_var/time_var.
  # It returns a data.frame with columns: treated, untreated, estimate, weight,
  # plus a `type` column (e.g. "Both Treated", "Treated vs Untreated"), and an
  # attribute beta_hat_w (the weighted sum, which equals the TWFE coefficient).
  res <- bacondecomp::bacon(
    formula  = y ~ D,
    data     = df,
    id_var   = "unit",
    time_var = "time"
  )

  # When the data contains a never-treated group, bacon() returns a list with
  # $two_by_twos (the per-component table) and $Omega (the variance-weighted
  # contributions). Without never-treated, it returns the data.frame directly.
  if (is.list(res) && !is.data.frame(res)) {
    components_df <- res$two_by_twos
    twfe_coef <- as.numeric(attr(res, "beta_hat_w"))
    # Fallback: re-derive TWFE from the components if attr is missing.
    if (is.null(twfe_coef) || length(twfe_coef) == 0L) {
      twfe_coef <- sum(components_df$estimate * components_df$weight)
    }
  } else {
    components_df <- res
    twfe_coef <- sum(components_df$estimate * components_df$weight)
  }

  # Components vary across bacondecomp versions; normalize the column names.
  cols <- names(components_df)
  treated_col <- if ("treated" %in% cols) "treated" else "g1"
  untreated_col <- if ("untreated" %in% cols) "untreated" else "g2"
  estimate_col <- if ("estimate" %in% cols) "estimate" else "Estimate"
  weight_col <- if ("weight" %in% cols) "weight" else "Weight"
  type_col <- if ("type" %in% cols) "type" else NA_character_

  components <- lapply(seq_len(nrow(components_df)), function(i) {
    list(
      treated_group   = as.numeric(components_df[[treated_col]][i]),
      control_group   = as.numeric(components_df[[untreated_col]][i]),
      estimate        = as.numeric(components_df[[estimate_col]][i]),
      weight          = as.numeric(components_df[[weight_col]][i]),
      type            = if (!is.na(type_col))
                          as.character(components_df[[type_col]][i])
                        else NA_character_
    )
  })

  weights_sum <- sum(sapply(components, function(c) c$weight))

  list(
    panel = list(
      unit        = as.integer(df$unit),
      time        = as.integer(df$time),
      y           = as.numeric(df$y),
      first_treat = as.integer(df$first_treat),
      treated     = as.integer(df$D)
    ),
    r_twfe_coef    = twfe_coef,
    r_components   = components,
    r_weights_sum  = weights_sum,
    n_components   = length(components)
  )
}

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

cat("Building fixture 1: uniform_3groups_with_never_treated...\n")
df1 <- build_panel(
  n_units_per_cohort   = 30L,
  n_periods            = 6L,
  cohort_times         = c(3L, 4L, 5L),
  always_treated_count = 0L,
  never_treated_count  = 30L,
  true_effect          = 2.0,
  seed                 = 101L
)
fixture_1 <- extract_bacon(df1, "uniform_3groups_with_never_treated")

cat("Building fixture 2: two_groups_no_never_treated...\n")
df2 <- build_panel(
  n_units_per_cohort   = 30L,
  n_periods            = 6L,
  cohort_times         = c(3L, 5L),
  always_treated_count = 0L,
  never_treated_count  = 0L,
  true_effect          = 2.0,
  seed                 = 202L
)
fixture_2 <- extract_bacon(df2, "two_groups_no_never_treated")

cat("Building fixture 3: always_treated_remapped...\n")
# 3 timing-cohorts (3, 4, 5) + 5 always-treated units (first_treat = 1, i.e.,
# treated in every observable period) + 25 never-treated. R's bacondecomp
# keeps the first_treat=1 cohort as a *separate* timing cohort (not in U) and
# emits a `Later vs Always Treated` comparison row for each later cohort
# alongside the standard `Treated vs Untreated` row. Python's paper-footnote-11
# convention remaps these units into the U bucket and folds R's two columns
# of components into a single `treated_vs_never` cell per treated cohort.
# The aggregate (TWFE coefficient + weights-sum) is invariant per Theorem 1,
# but the per-component breakdown differs by convention — see REGISTRY
# `**Note (R parity convention divergence on always-treated)**` and
# `**Deviation (first-period boundary extension on always-treated remap)**`.
# `tests/test_methodology_bacon.py::TestBaconParityR` carves out the U-bucket
# rows for direct per-component parity (keeping the 6 timing-vs-timing rows
# under direct parity) and asserts the U-bucket fold-back separately via
# `test_always_treated_remapped_fold_back_matches_r` at atol=1e-6.
df3 <- build_panel(
  n_units_per_cohort   = 25L,
  n_periods            = 6L,
  cohort_times         = c(3L, 4L, 5L),
  always_treated_count = 5L,
  never_treated_count  = 25L,
  true_effect          = 2.0,
  seed                 = 303L
)
fixture_3 <- extract_bacon(df3, "always_treated_remapped")

# ---------------------------------------------------------------------------
# Write JSON
# ---------------------------------------------------------------------------

out <- list(
  meta = list(
    generated_at         = format(Sys.Date()),
    bacondecomp_version  = as.character(packageVersion("bacondecomp")),
    r_version            = R.version.string,
    description          = paste(
      "Goodman-Bacon (2021) decomposition parity goldens for diff-diff",
      "BaconDecomposition. Parity target at atol=1e-6:",
      "(1) aggregate TWFE coefficient + weights-sum across all 3 fixtures;",
      "(2) direct per-component (treated, control, type) parity on the 2",
      "non-remap fixtures AND on the 6 timing-vs-timing rows of",
      "always_treated_remapped;",
      "(3) cohort-level fold-back parity for the U bucket on",
      "always_treated_remapped (Python's paper-footnote-11 remap folds",
      "R's separate Later-vs-Always-Treated + Treated-vs-Untreated rows",
      "into a single treated_vs_never cell per cohort; aggregate is",
      "invariant per Theorem 1, breakdown differs by convention).",
      "See REGISTRY Note (R parity convention divergence on always-treated)",
      "+ Deviation (first-period boundary extension)."
    )
  ),
  uniform_3groups_with_never_treated = fixture_1,
  two_groups_no_never_treated        = fixture_2,
  always_treated_remapped            = fixture_3
)

out_path <- "../data/r_bacondecomp_golden.json"
write_json(out, out_path, pretty = TRUE, digits = NA, auto_unbox = TRUE)
cat(sprintf("Wrote %s\n", out_path))
