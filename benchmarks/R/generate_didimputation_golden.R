#!/usr/bin/env Rscript
# Golden generator: ImputationDiD vs R `didimputation::did_imputation`
# (Borusyak, Jaravel & Spiess 2024).
#
# Writes a fixed staggered-adoption panel and the R reference estimates so that
# tests/test_methodology_imputation.py::TestImputationDiDParityR can pin Python
# output against R without requiring R at test time.
#
# Outputs (checked into the repo):
#   benchmarks/data/didimputation_test_panel.csv   (unit, time, first_treat, y)
#   benchmarks/data/didimputation_golden.json
#
# Usage:
#   Rscript benchmarks/R/generate_didimputation_golden.R
#
# Notes:
#   - R `didimputation` aggregates the auxiliary model by cohort x event-time and
#     uses sum(v^2 * tau)/sum(v^2); at that partition this equals the paper's
#     unit-clustered Equation 8 (<=1 obs/unit/group), i.e. diff-diff's default
#     aux_partition="cohort_horizon". See docs/methodology/REGISTRY.md
#     "## ImputationDiD" -> "Deviation from R".
#   - never-treated units are coded first_treat = 0 (matches the Python side).

suppressMessages({
  library(didimputation)
  library(jsonlite)
  library(data.table)
})

set.seed(2024)

# ---- Deterministic staggered-adoption panel (parallel trends) ----
# Cohorts 3 and 5 plus a never-treated group (first_treat = 0); 8 periods.
n_per_cohort <- 60L
n_periods <- 8L
cohorts <- c(0L, 3L, 5L)          # 0 = never-treated
tau_h <- function(k) 1.0 + 0.5 * k # heterogeneous-by-horizon effect

rows <- list()
uid <- 0L
for (g in cohorts) {
  for (j in seq_len(n_per_cohort)) {
    c_i <- rnorm(1)
    for (t in seq_len(n_periods)) {
      beta_t <- 0.5 * t
      u <- 0.2 * rnorm(1)
      treated <- (g > 0L) && (t >= g)
      eff <- if (treated) tau_h(t - g) else 0.0
      y <- c_i + beta_t + eff + u
      uid_chr <- uid
      rows[[length(rows) + 1L]] <- list(
        unit = uid_chr, time = t, first_treat = g, y = y
      )
    }
    uid <- uid + 1L
  }
}
panel <- rbindlist(rows)
panel[, unit := as.integer(unit)]
panel[, time := as.integer(time)]
panel[, first_treat := as.integer(first_treat)]

panel_path <- file.path("benchmarks", "data", "didimputation_test_panel.csv")
dir.create(dirname(panel_path), recursive = TRUE, showWarnings = FALSE)
fwrite(panel, panel_path)
message(sprintf("Wrote panel: %s (%d rows)", panel_path, nrow(panel)))

# ---- Overall ATT (static) ----
overall <- did_imputation(
  data = panel, yname = "y", gname = "first_treat",
  tname = "time", idname = "unit", cluster_var = "unit"
)
overall_att <- as.numeric(overall$estimate[1])
overall_se <- as.numeric(overall$std.error[1])
message(sprintf("Overall ATT = %.8f (SE %.8f)", overall_att, overall_se))

# ---- Event study (post-treatment horizons) ----
es <- did_imputation(
  data = panel, yname = "y", gname = "first_treat",
  tname = "time", idname = "unit", horizon = TRUE, cluster_var = "unit"
)
es_h <- as.integer(gsub("tau", "", es$term))
ord <- order(es_h)
es_h <- es_h[ord]
es_att <- as.numeric(es$estimate)[ord]
es_se <- as.numeric(es$std.error)[ord]
for (i in seq_along(es_h)) {
  message(sprintf("  h=%d: ATT=%.6f SE=%.6f", es_h[i], es_att[i], es_se[i]))
}

golden <- list(
  estimator = "didimputation::did_imputation",
  meta = list(
    r_version = R.version.string,
    didimputation_version = as.character(packageVersion("didimputation")),
    seed = 2024L,
    n_units = length(unique(panel$unit)),
    n_periods = n_periods,
    cohorts = cohorts,
    aux_partition = "cohort_horizon (cohort x event-time; R default)",
    note = paste(
      "R did_imputation auxiliary aggregator sum(v^2*tau)/sum(v^2) equals the",
      "paper's unit-clustered Eq. 8 at cohort x event-time = diff-diff",
      "aux_partition='cohort_horizon'."
    )
  ),
  overall = list(att = overall_att, se = overall_se),
  event_study = list(horizons = es_h, att = es_att, se = es_se)
)

golden_path <- file.path("benchmarks", "data", "didimputation_golden.json")
write_json(golden, golden_path, auto_unbox = TRUE, pretty = TRUE, digits = 12)
message(sprintf("Wrote golden: %s", golden_path))

# ---- Covariate scenario (time-varying X in the first stage) ----
# Appended AFTER the base scenario so the base panel's RNG draws (and the
# committed base CSV/JSON values) are unchanged on regeneration. Validates
# the covariate branch of the untreated v_it projection: the first-stage
# imputation model becomes y ~ x | unit + time on the untreated sample
# (fixest-style first_stage formula), matching diff-diff's
# ImputationDiD(covariates=["x"]).
rows_x <- list()
uid <- 0L
for (g in cohorts) {
  for (j in seq_len(n_per_cohort)) {
    c_i <- rnorm(1)
    for (t in seq_len(n_periods)) {
      beta_t <- 0.5 * t
      x_it <- 0.4 * c_i + 0.2 * t + rnorm(1)   # time-varying, unit-correlated
      u <- 0.2 * rnorm(1)
      treated <- (g > 0L) && (t >= g)
      eff <- if (treated) tau_h(t - g) else 0.0
      y <- c_i + beta_t + 0.7 * x_it + eff + u
      rows_x[[length(rows_x) + 1L]] <- list(
        unit = uid, time = t, first_treat = g, y = y, x = x_it
      )
    }
    uid <- uid + 1L
  }
}
panel_x <- rbindlist(rows_x)
panel_x[, unit := as.integer(unit)]
panel_x[, time := as.integer(time)]
panel_x[, first_treat := as.integer(first_treat)]

panel_x_path <- file.path("benchmarks", "data", "didimputation_covariate_panel.csv")
fwrite(panel_x, panel_x_path)
message(sprintf("Wrote covariate panel: %s (%d rows)", panel_x_path, nrow(panel_x)))

overall_x <- did_imputation(
  data = panel_x, yname = "y", gname = "first_treat",
  tname = "time", idname = "unit", cluster_var = "unit",
  first_stage = ~ 0 + x | unit + time
)
overall_x_att <- as.numeric(overall_x$estimate[1])
overall_x_se <- as.numeric(overall_x$std.error[1])
message(sprintf("Covariate overall ATT = %.8f (SE %.8f)", overall_x_att, overall_x_se))

es_x <- did_imputation(
  data = panel_x, yname = "y", gname = "first_treat",
  tname = "time", idname = "unit", horizon = TRUE, cluster_var = "unit",
  first_stage = ~ 0 + x | unit + time
)
es_x_h <- as.integer(gsub("tau", "", es_x$term))
ord_x <- order(es_x_h)
es_x_h <- es_x_h[ord_x]
es_x_att <- as.numeric(es_x$estimate)[ord_x]
es_x_se <- as.numeric(es_x$std.error)[ord_x]

golden$covariate <- list(
  panel = "didimputation_covariate_panel.csv",
  first_stage = "~ 0 + x | unit + time",
  overall = list(att = overall_x_att, se = overall_x_se),
  event_study = list(horizons = es_x_h, att = es_x_att, se = es_x_se)
)

write_json(golden, golden_path, auto_unbox = TRUE, pretty = TRUE, digits = 12)
message(sprintf("Re-wrote golden with covariate block: %s", golden_path))
