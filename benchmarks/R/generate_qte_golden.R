#!/usr/bin/env Rscript
# Golden fixtures for the diff-diff ChangesInChanges / QDiD estimators vs qte 1.3.1.
#
# Regenerate with:
#   Rscript benchmarks/R/generate_qte_golden.R
#
# Output: benchmarks/data/qte_golden.json, consumed by
# tests/test_changes_in_changes_parity.py (which pytest.skips when absent).
#
# Version contract: qte is pinned to 1.3.1 here, in benchmarks/R/requirements.R
# (pinned_versions), and in the parity test's test_metadata_versions_match.
# Bump all three in lockstep when re-anchoring.
#
# Point fixtures are generated with se=FALSE so they are fully deterministic
# (qte's bootstrap is not seedable through its public API). The SE block runs
# seeded (set.seed before each call) with iters=999; Python compares those SEs
# statistically, not bit-exactly.

suppressMessages({
  library(qte)
  library(jsonlite)
})

stopifnot(packageVersion("qte") == "1.3.1")

probs <- seq(0.05, 0.95, 0.05)

# ---------------------------------------------------------------------------
# Synthetic DGPs (long format: id, period 0/1, treat 0/1 group indicator, y)
# ---------------------------------------------------------------------------

make_normal_2x2 <- function(n_treated, n_control, seed) {
  # Additive time shift + heterogeneous treatment effect monotone in the unit
  # unobservable; full support overlap so the whole grid is interior.
  set.seed(seed)
  n <- n_treated + n_control
  treat <- c(rep(1L, n_treated), rep(0L, n_control))
  u <- rnorm(n, mean = 0, sd = 1)
  y_pre <- u + rnorm(n, sd = 0.3)
  effect <- 0.8 + 0.4 * pnorm(u)
  y_post <- u + 0.5 + rnorm(n, sd = 0.3) + treat * effect
  data.frame(
    id = rep(seq_len(n), times = 2),
    period = rep(c(0L, 1L), each = n),
    treat = rep(treat, times = 2),
    y = c(y_pre, y_post)
  )
}

make_lognormal_2x2 <- function(n_treated, n_control, seed) {
  # Skewed outcome with a scale-type change over time: CiC and QDiD are
  # numerically far apart, exercising the inverse-CDF composition away from
  # the additive special case.
  set.seed(seed)
  n <- n_treated + n_control
  treat <- c(rep(1L, n_treated), rep(0L, n_control))
  u <- rnorm(n, mean = 0, sd = 0.6)
  y_pre <- exp(u + rnorm(n, sd = 0.2))
  y_post <- exp(u * 1.3 + 0.3 + rnorm(n, sd = 0.2)) + treat * (0.5 + 0.5 * exp(u / 2))
  data.frame(
    id = rep(seq_len(n), times = 2),
    period = rep(c(0L, 1L), each = n),
    treat = rep(treat, times = 2),
    y = c(y_pre, y_post)
  )
}

dgps <- list(
  normal_2x2_n500 = make_normal_2x2(250, 250, seed = 20260712),
  lognormal_2x2_n300 = make_lognormal_2x2(150, 150, seed = 20260713),
  # Small unbalanced-cell design stressing near-integer n*p type-1 boundaries.
  smalln_2x2_n60 = make_normal_2x2(35, 25, seed = 20260714)
)

# lalonde (qte-shipped): 1975/1978 two-period panel; re78 zeros give heavy ties.
data(lalonde, package = "qte")
lal <- lalonde.psid.panel[lalonde.psid.panel$year %in% c(1975, 1978), ]
dgps$lalonde_psid <- data.frame(
  id = lal$id,
  period = ifelse(lal$year == 1978, 1L, 0L),
  treat = as.integer(lal$treat),
  y = as.numeric(lal$re)
)

# ---------------------------------------------------------------------------
# Point fixtures: {CiC, QDiD} x {panel, repeated cross-section}, se = FALSE
# ---------------------------------------------------------------------------

run_point <- function(df, method, panel) {
  fn <- if (method == "cic") CiC else QDiD
  res <- fn(
    formla = y ~ treat,
    t = 1, tmin1 = 0, tname = "period",
    data = df,
    panel = panel,
    idname = if (panel) "id" else NULL,
    se = FALSE,
    probs = probs
  )
  list(ate = res$ate, qte = as.numeric(res$qte))
}

scenarios <- list()
for (dgp_name in names(dgps)) {
  df <- dgps[[dgp_name]]
  entry <- list(
    data = list(
      id = df$id,
      period = df$period,
      treat = df$treat,
      y = df$y
    ),
    results = list()
  )
  for (method in c("cic", "qdid")) {
    for (panel in c(TRUE, FALSE)) {
      key <- sprintf("%s_%s", method, if (panel) "panel" else "rcs")
      entry$results[[key]] <- run_point(df, method, panel)
      message(sprintf(
        "%s / %s: ate = %.10g", dgp_name, key, entry$results[[key]]$ate
      ))
    }
  }
  scenarios[[dgp_name]] <- entry
}

# ---------------------------------------------------------------------------
# SE block (statistical parity): normal_2x2_n500 only, seeded, iters = 999
# ---------------------------------------------------------------------------

run_se <- function(df, method, panel, seed) {
  fn <- if (method == "cic") CiC else QDiD
  set.seed(seed)
  res <- fn(
    formla = y ~ treat,
    t = 1, tmin1 = 0, tname = "period",
    data = df,
    panel = panel,
    idname = if (panel) "id" else NULL,
    se = TRUE, iters = 999,
    probs = probs
  )
  list(
    ate = res$ate, qte = as.numeric(res$qte),
    ate_se = res$ate.se, qte_se = as.numeric(res$qte.se),
    sup_t_crit = as.numeric(res$c)
  )
}

se_block <- list(
  cic_panel = run_se(dgps$normal_2x2_n500, "cic", TRUE, seed = 42),
  cic_rcs = run_se(dgps$normal_2x2_n500, "cic", FALSE, seed = 42),
  qdid_panel = run_se(dgps$normal_2x2_n500, "qdid", TRUE, seed = 42),
  qdid_rcs = run_se(dgps$normal_2x2_n500, "qdid", FALSE, seed = 42)
)
message("SE block done.")

# ---------------------------------------------------------------------------
# Micro-fixtures: raw R type-1 / type-7 quantile outputs on adversarial probs
# ---------------------------------------------------------------------------

micro_case <- function(x, p) {
  list(
    x = x,
    probs = p,
    type1 = as.numeric(quantile(x, probs = p, type = 1, names = FALSE)),
    type7 = as.numeric(quantile(x, probs = p, names = FALSE))
  )
}

set.seed(20260715)
x7 <- sort(rnorm(7))
x20 <- sort(rnorm(20))
x60 <- sort(rlnorm(60))
x101 <- sort(rnorm(101))
# ECDF-composed probabilities (the parity-critical case): ranks k/n00 from one
# sample evaluated in another - float products can land an ulp off an integer.
p_comp20 <- ecdf(x7)(x20)
p_comp60 <- ecdf(x20)(x60)
grid_edges <- c(0, 1e-12, 1 / 7, 2 / 7, 0.25, 1 / 3, 0.5, 19 / 20, 0.95, 0.999, 1)
p_near_int101 <- c((0:101) / 101, (1:100) / 101 + 1e-16, (1:100) / 101 - 1e-16)

micro <- list(
  n7_grid = micro_case(x7, grid_edges),
  n20_composed = micro_case(x20, p_comp20),
  n60_composed = micro_case(x60, p_comp60),
  n101_near_integer = micro_case(x101, pmin(pmax(p_near_int101, 0), 1))
)

# ---------------------------------------------------------------------------
# Write JSON
# ---------------------------------------------------------------------------

out <- list(
  metadata = list(
    description = paste(
      "Golden fixtures for diff-diff ChangesInChanges/QDiD parity vs qte.",
      "Point fixtures use se=FALSE (deterministic); the se_block is seeded",
      "(set.seed before each call, iters=999) and compared statistically."
    ),
    qte_version = as.character(packageVersion("qte")),
    r_version = as.character(getRversion()),
    probs = probs,
    point_atol = 1e-10,
    n_scenarios = length(scenarios)
  ),
  scenarios = scenarios,
  se_block = se_block,
  quantile_cases = micro
)

out_path <- file.path("benchmarks", "data", "qte_golden.json")
write_json(out, path = out_path, digits = 17, auto_unbox = TRUE, null = "null")
message(sprintf("Wrote %s (%.1f KB)", out_path, file.size(out_path) / 1024))
