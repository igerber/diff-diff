# Generate cross-language end-to-end parity fixture for HAD Phase 4
# (PR #389 R-parity vs `Credible-Answers/did_had`).
#
# Purpose: validate Python `HeterogeneousAdoptionDiD.fit()` (overall,
# event-study, placebo, yatchew, trends_lin) against R `DIDHAD::did_had()`
# bit-exactly on shared input. The R package is the methodology source
# of truth (the de Chaisemartin team wrote it); matching it within
# `atol=1e-8` on point/SE/CI and `atol=1e-10` on closed-form Yatchew
# T-stats is a strictly stronger correctness signal than reproducing the
# paper's published Pierce-Schott numbers (which depend on a
# LBD-restricted analysis panel).
#
# Usage:
#   Rscript benchmarks/R/generate_did_had_golden.R
#
# Output:
#   benchmarks/data/did_had_golden.json
#
# Phase 4 of HeterogeneousAdoptionDiD (de Chaisemartin et al. 2025).
# Python test loader: tests/test_did_had_parity.py.
#
# Pin: DIDHAD == 2.0.0 (CRAN current as of 2026-04). YatchewTest >= 1.1.0.

library(jsonlite)
library(DIDHAD)
library(YatchewTest)

# PR #392 R4 P3 / R5 P3: pin exact upstream versions so future
# regeneration does not silently re-anchor the goldens to a newer
# CRAN release while CHANGELOG / REGISTRY / parity test still cite
# v2.0.0 / SHA `edc09197`. The parity contract runs through
# `nprobust` numerical paths so we pin it too. Bump these pins
# (here AND in the parity test's `test_metadata_versions_match`)
# when intentionally re-anchoring.
stopifnot(packageVersion("DIDHAD") == "2.0.0")
stopifnot(packageVersion("YatchewTest") == "1.1.1")
stopifnot(packageVersion("nprobust") == "0.5.0")

# -------------------------------------------------------------------------
# Panel builder: 5-period panel with F=4 (treatment onset at t=4).
# Pre-periods: 1, 2, 3 (D=0). Post-periods: 4, 5 (D=fixed positive dose).
# Y[g, t] = unit_fe[g] + trend[g] * (t - 1) + (dose[g] + dose[g]^2) * (t >= F) + noise
# -------------------------------------------------------------------------

build_panel <- function(G, F_treat, T_periods, dose_draws, seed,
                        unit_trend_sd = 0.05, noise_sd = 0.5) {
  set.seed(seed)
  n <- G * T_periods
  unit_fe <- rnorm(G, mean = 0, sd = 1.0)
  unit_trend <- rnorm(G, mean = 0.1, sd = unit_trend_sd)
  noise <- rnorm(n, mean = 0, sd = noise_sd)

  rows <- vector("list", n)
  k <- 1
  for (g in seq_len(G)) {
    for (t in seq_len(T_periods)) {
      treated <- as.numeric(t >= F_treat)
      y <- unit_fe[g] + unit_trend[g] * (t - 1) +
           (dose_draws[g] + dose_draws[g]^2) * treated +
           noise[k]
      d_obs <- if (treated == 1) dose_draws[g] else 0.0
      # Use short column names (g, t, d, y) matching DIDHAD's tutorial
      # convention. The package has a data-masking issue when column
      # names alias the formal parameter names (e.g., column "time" with
      # `time = "time"` resolves to the column values inside dplyr's
      # `.data[[get("time")]]` lookup), so avoid that overlap upstream.
      rows[[k]] <- data.frame(
        g = g,
        t = t,
        y = y,
        d = d_obs,
        stringsAsFactors = FALSE
      )
      k <- k + 1
    }
  }
  do.call(rbind, rows)
}

# DGP 1: D ~ Uniform(0, 1).
dgp_uniform <- function(G = 200, F_treat = 4, T_periods = 5, seed = 20260426) {
  set.seed(seed * 2L + 1L)
  d <- runif(G, min = 0.0, max = 1.0)
  list(
    name = "uniform_G200_F4_T5",
    panel = build_panel(G, F_treat, T_periods, d, seed = seed),
    G = G, F_treat = F_treat, T_periods = T_periods,
    dose_distribution = "Uniform(0, 1)",
    seed = seed
  )
}

# DGP 2: D ~ Beta(2, 2). Symmetric, bell-shaped on [0, 1].
dgp_beta22 <- function(G = 200, F_treat = 4, T_periods = 5, seed = 20260426) {
  set.seed(seed * 2L + 2L)
  d <- rbeta(G, shape1 = 2, shape2 = 2)
  list(
    name = "beta22_G200_F4_T5",
    panel = build_panel(G, F_treat, T_periods, d, seed = seed),
    G = G, F_treat = F_treat, T_periods = T_periods,
    dose_distribution = "Beta(2, 2)",
    seed = seed
  )
}

# DGP 3: D ~ Beta(0.5, 1). Heavy left tail (mass near 0); approximates
# the empirical Pierce-Schott NTR-gap distribution where many industries
# have small tariff gaps (boundary density vanishes property).
dgp_boundary <- function(G = 200, F_treat = 4, T_periods = 5, seed = 20260426) {
  set.seed(seed * 2L + 3L)
  d <- rbeta(G, shape1 = 0.5, shape2 = 1.0)
  list(
    name = "boundary_G200_F4_T5",
    panel = build_panel(G, F_treat, T_periods, d, seed = seed),
    G = G, F_treat = F_treat, T_periods = T_periods,
    dose_distribution = "Beta(0.5, 1)",
    seed = seed
  )
}

# -------------------------------------------------------------------------
# Run did_had with given options and extract the standardized result
# matrix. The R package returns a `did_had` S3 object whose `results`
# slot has `resmat` (effects + placebos) and optionally `yatchew_test`.
# -------------------------------------------------------------------------

run_did_had <- function(panel, effects = 1, placebo = 0,
                       trends_lin = FALSE, yatchew = FALSE) {
  # graph_off=TRUE suppresses the auto-print of the event-study plot.
  fit <- did_had(
    df = panel,
    outcome = "y",
    group = "g",
    time = "t",
    treatment = "d",
    effects = effects,
    placebo = placebo,
    trends_lin = trends_lin,
    yatchew = yatchew,
    graph_off = TRUE
  )
  res <- fit$results
  resmat <- res$resmat
  out <- list(
    n_effects_actual = res$res.effects,
    n_placebo_actual = res$res.placebo,
    rownames = rownames(resmat),
    estimate = unname(resmat[, "Estimate"]),
    se = unname(resmat[, "SE"]),
    ci_lo = unname(resmat[, "LB.CI"]),
    ci_hi = unname(resmat[, "UB.CI"]),
    n_per_horizon = unname(as.integer(resmat[, "N"])),
    bw_per_horizon = unname(resmat[, "BW"]),
    n_within_bw = unname(as.integer(resmat[, "N.BW"])),
    qug_t = unname(resmat[, "T"]),
    qug_p = unname(resmat[, "p.val"]),
    event_id = unname(as.integer(resmat[, "ID"]))
  )
  if (yatchew) {
    yt <- res$yatchew_test
    out$yatchew_t <- unname(yt[, "T_hr"])
    out$yatchew_p <- unname(yt[, "p-value"])
    out$yatchew_n <- unname(as.integer(yt[, "N"]))
    # Capture sigma2 components for diagnostic comparison; the column
    # names contain unicode (sigma², σ²). Use positional indexing.
    out$yatchew_sigma2_lin <- unname(yt[, 1])
    out$yatchew_sigma2_diff <- unname(yt[, 2])
  }
  out
}

# -------------------------------------------------------------------------
# Build the DGP × method-combo fixture grid.
# -------------------------------------------------------------------------

dgp_builders <- list(
  uniform = dgp_uniform,
  beta22 = dgp_beta22,
  boundary = dgp_boundary
)

# Per-DGP method matrix. Each combo runs did_had with the named flags
# and stores the resulting standardized resmat dict alongside the input
# panel arrays. Python parity test loops over combos and asserts.
#
# Why effects=2/placebo=2: F=4 with T=5 leaves 2 post-period horizons
# (t=4, 5) and 2 pre-period placebos (t=2, 1) without trends_lin. R
# auto-truncates if requested > feasible. Under trends_lin, the
# F-2 -> F-1 evolution is consumed by the slope estimator and R reduces
# max placebo by 1 (so only placebo at t=1 survives).
combos <- list(
  list(name = "overall_e1", effects = 1, placebo = 0,
       trends_lin = FALSE, yatchew = FALSE),
  list(name = "event_e2_p2", effects = 2, placebo = 2,
       trends_lin = FALSE, yatchew = FALSE),
  list(name = "event_e2_p2_yatchew", effects = 2, placebo = 2,
       trends_lin = FALSE, yatchew = TRUE),
  list(name = "event_e2_p2_trendslin", effects = 2, placebo = 2,
       trends_lin = TRUE, yatchew = FALSE),
  list(name = "event_e2_p2_yatchew_trendslin", effects = 2, placebo = 2,
       trends_lin = TRUE, yatchew = TRUE)
)

fixtures <- list()
for (dgp_name in names(dgp_builders)) {
  dgp <- dgp_builders[[dgp_name]]()
  panel <- dgp$panel
  combo_results <- list()
  for (combo in combos) {
    res <- run_did_had(
      panel = panel,
      effects = combo$effects,
      placebo = combo$placebo,
      trends_lin = combo$trends_lin,
      yatchew = combo$yatchew
    )
    combo_results[[combo$name]] <- list(
      effects = combo$effects,
      placebo = combo$placebo,
      trends_lin = combo$trends_lin,
      yatchew = combo$yatchew,
      result = res
    )
  }
  fixtures[[dgp$name]] <- list(
    name = dgp$name,
    G = dgp$G,
    F = dgp$F_treat,
    T = dgp$T_periods,
    dose_distribution = dgp$dose_distribution,
    seed = dgp$seed,
    panel = list(
      g = panel$g,
      t = panel$t,
      y = panel$y,
      d = panel$d
    ),
    combos = combo_results
  )
}

# -------------------------------------------------------------------------
# Serialize
# -------------------------------------------------------------------------

out <- list(
  metadata = list(
    description = paste(
      "DIDHAD::did_had end-to-end parity fixture for HAD Phase 4",
      "(PR #389 R-parity).",
      sep = " "
    ),
    didhad_version = as.character(packageVersion("DIDHAD")),
    yatchewtest_version = as.character(packageVersion("YatchewTest")),
    nprobust_version = as.character(packageVersion("nprobust")),
    r_version = as.character(getRversion()),
    n_dgps = length(fixtures),
    n_combos_per_dgp = length(combos),
    point_atol = 1e-8,
    se_atol = 1e-8,
    ci_atol = 1e-8,
    yatchew_atol = 1e-10,
    qug_atol = 1e-12,
    notes = paste(
      "Three synthetic DGPs (Uniform, Beta(2,2), Beta(0.5,1) approximation",
      "of the empirical Pierce-Schott NTR-gap distribution). Each DGP runs",
      "5 method combos covering overall, event-study, placebo, yatchew,",
      "and trends_lin variants. Tolerances per the Phase 4 plan.",
      sep = " "
    )
  ),
  fixtures = fixtures
)

out_dir <- "benchmarks/data"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
out_path <- file.path(out_dir, "did_had_golden.json")
write_json(out, path = out_path, digits = 17, auto_unbox = TRUE, null = "null")
message(sprintf(
  "Wrote %d DGP fixtures (each with %d combos) to %s",
  length(fixtures), length(combos), out_path
))
