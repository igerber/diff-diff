#!/usr/bin/env Rscript
# Generate R `pretrends` parity goldens for diff-diff PreTrendsPower (PR-C).
#
# Script committed in PR-B; JSON goldens generated and committed in PR-C
# against `jonathandroth/pretrends` commit `122731d082` (package version
# 0.1.0). Running this script writes ../data/r_pretrends_golden.json.
#
# Requires:
#   - R 4.4+ (tested on 4.5.2)
#   - install.packages("remotes")
#   - remotes::install_github("jonathandroth/pretrends", ref = "122731d082")
#   - install.packages(c("jsonlite", "MASS"))
#
# Output: ../data/r_pretrends_golden.json
#
# diff-diff PreTrendsPower with `pretest_form='nis'` (the new default per
# PR-B Step 2) matches the values in this JSON along a three-tier contract,
# both tiers at atol=1e-4:
#   (1) NIS box probability `P(beta_hat_pre in B_NIS(Sigma))` at fixed gamma
#       values on all 4 fixtures, at atol=1e-4. R hardcodes thresholdTstat
#       = 1.96 while Python uses scipy.stats.norm.ppf(0.975) =
#       1.959963984540054 (~4e-5 dz gap); on top of that, mvtnorm::pmvnorm
#       (R) and scipy.stats.multivariate_normal.cdf (Python) use Genz-Bretz
#       randomized-lattice rules with different absolute-error defaults
#       (abseps ~ 1e-3 vs 1e-5). The empirical NIS power gap is bounded by
#       ~5e-5 on the K=4 shifted-grid fixture; ~3e-5 on K=3 fixtures; ~2e-5
#       on K=1. atol=1e-4 is the realistic atol without tightening
#       thresholdTstat.Pretest in R or relaxing the Genz tolerances.
#   (2) gamma_p MDV (slope at target power 0.5 and 0.8) on regular, irregular,
#       shifted-grid, and K=1 grids, at atol=1e-4. R uniroot defaults to
#       tol = .Machine$double.eps^0.25 ~= 1.22e-4 vs Python brentq xtol=2e-12;
#       the inverse-solver tolerance gap dominates, so 1e-4 is the realistic
#       atol without tightening either solver.
#   (3) gamma-unit MDV invariance: PR-B's "skip L2 norm for linear with
#       relative_times" path produces MDV in Roth's gamma units exactly,
#       matching R's `slope_for_power()` which also reports gamma. Fixture 2
#       (irregular grid {-5, -3, -1}) and the end-to-end fit() test in
#       tests/test_methodology_pretrends.py lock this.
#
# Four fixtures (matched to test_methodology_pretrends.py expectations):
#   1. uniform_3_pre_periods_no_anticipation — K=3 regular grid (t in {-3, -2, -1}),
#      never-treated control. Default-case parity baseline.
#   2. irregular_pre_periods — K=3 with relative_times = [-5, -3, -1].
#      Exercises the PR-B gamma-unit linear-pattern fix end-to-end.
#   3. anticipation_shifted — K=4 shifted-grid case (pre-periods at
#      {-5, -4, -3, -2}, leaving a notional t=-1 anticipation gap to the
#      reference period 0). Locks R parity at a larger K than the K=3
#      fixtures and on a non-adjacent-to-reference grid. NOTE: the R-side
#      `pretrends` package has no anticipation parameter, so this fixture
#      does NOT exercise the Python-side `_extract_pre_period_params`
#      CS/SA anticipation filter (`if t < _pre_cutoff` in pretrends.py)
#      against R goldens — that filter is exercised by the existing
#      `TestPretrendsCovarianceSource` suite (PR-B Step 3) and by the
#      MC-based `TestPretrendsPropositions` tests, but a CS/SA-level
#      anticipation+R-parity test would need a synthetic
#      CallawaySantAnnaResults with `anticipation=1` and a t=-1 entry
#      that gets filtered before reaching `_compute_power_nis`. Deferred
#      to a follow-up.
#   4. single_pre_period_closed_form — K=1 with diagonal Sigma = 0.25*I
#      (Roth Proposition 2 univariate truncated-normal closed form). Locks
#      the scalar fast-path against R AND against the analytical expression
#      `1 - Phi(z - gamma/sigma) + Phi(-z - gamma/sigma)`.
#
# Run:
#   cd benchmarks/R && Rscript generate_pretrends_golden.R

suppressPackageStartupMessages({
  library(pretrends)
  library(jsonlite)
})

PRETRENDS_COMMIT <- "122731d082"

# Provenance fail-closed: refuse to regenerate goldens unless the installed
# pretrends matches the pinned (version, commit) pair. The JSON stamps the
# pinned commit string into meta.pretrends_commit, so without this check a
# future rerun could silently regenerate goldens from a drifted revision
# while still labeling the artifact with the original commit.
stopifnot(packageVersion("pretrends") == "0.1.0")
.installed_sha <- packageDescription("pretrends")$RemoteSha
if (is.null(.installed_sha) || !startsWith(.installed_sha, PRETRENDS_COMMIT)) {
  stop(sprintf(
    "pretrends provenance mismatch: expected RemoteSha to start with '%s' but got '%s'. Reinstall with: remotes::install_github('jonathandroth/pretrends', ref = '%s')",
    PRETRENDS_COMMIT,
    if (is.null(.installed_sha)) "<missing — not installed via install_github>" else .installed_sha,
    PRETRENDS_COMMIT
  ))
}
cat("pretrends provenance verified: version 0.1.0, RemoteSha =",
    .installed_sha, "\n")

# ---------------------------------------------------------------------------
# DGP helper: build a synthetic event-study coefficient vector + VCV under a
# stylized null DGP (beta = 0, Sigma_22 ~ correlated). Mirrors the simulation
# fixtures in test_methodology_pretrends.py.
# ---------------------------------------------------------------------------

build_event_study_fixture <- function(
  pre_periods,
  post_periods,
  sigma2 = 0.04,
  rho = 0.3,
  seed = 42L
) {
  # Generate a correlated equicorrelation Sigma across all (pre + post) periods.
  # Realized beta_hat drawn from N(0, Sigma) — null DGP, no real treatment
  # effect.
  set.seed(seed)
  all_periods <- c(pre_periods, post_periods)
  K_total <- length(all_periods)
  Sigma <- sigma2 * (rho * matrix(1, K_total, K_total) + (1 - rho) * diag(K_total))
  beta_hat <- MASS::mvrnorm(1, mu = rep(0, K_total), Sigma = Sigma)

  list(
    beta_hat = beta_hat,
    Sigma = Sigma,
    all_periods = all_periods,
    pre_periods = pre_periods,
    post_periods = post_periods
  )
}

# ---------------------------------------------------------------------------
# Extract R pretrends() output into a fixture-shaped list.
# ---------------------------------------------------------------------------

extract_pretrends <- function(fixture_data, fixture_name) {
  beta_hat <- fixture_data$beta_hat
  Sigma <- fixture_data$Sigma
  pre_periods <- fixture_data$pre_periods
  post_periods <- fixture_data$post_periods
  all_periods <- fixture_data$all_periods

  # R `pretrends` expects: betahat (coefficient vector), sigma (VCV matrix),
  # tVec (relative-time labels excluding the reference period 0),
  # referencePeriod = 0, and deltatrue (length-K_total hypothesized delta
  # vector — only the pre-period entries are used for the rejection
  # probability).

  # Tier 1: NIS power at fixed gamma values.
  # Build delta_pre = gamma * pre_periods per Roth's slope convention
  # delta_t = gamma * t (t < 0 for pre-periods, so delta_pre is negative;
  # the NIS box is symmetric, so the sign does not affect the rejection
  # probability).
  gamma_test_values <- c(0.0, 0.2, 0.5, 1.0)
  power_values <- sapply(gamma_test_values, function(g) {
    deltatrue_full <- rep(0, length(all_periods))
    deltatrue_full[seq_along(pre_periods)] <- g * pre_periods
    res <- pretrends(betahat = beta_hat,
                     sigma = Sigma,
                     deltatrue = deltatrue_full,
                     tVec = all_periods,
                     referencePeriod = 0)
    as.numeric(res$df_power$Power)
  })

  # Tier 2: gamma_p MDV at target powers 0.5 and 0.8.
  # R `slope_for_power()` solves uniroot on the slope -> rejection-probability
  # map; returns gamma in the same units as Roth's slope convention.
  gamma_p_values <- sapply(c(0.5, 0.8), function(p) {
    as.numeric(slope_for_power(sigma = Sigma,
                               targetPower = p,
                               tVec = all_periods,
                               referencePeriod = 0))
  })

  list(
    # Wrap vector fields in I() to prevent jsonlite's auto_unbox=TRUE from
    # collapsing length-1 vectors to scalars (matters for the K=1 fixture:
    # `pre_periods` and `post_periods` are singletons but must serialize as
    # length-1 arrays so downstream Python loaders can iterate uniformly).
    panel = list(
      pre_periods = I(as.integer(pre_periods)),
      post_periods = I(as.integer(post_periods)),
      all_periods = I(as.integer(all_periods)),
      beta_hat = I(as.numeric(beta_hat)),
      Sigma = Sigma
    ),
    r_power_at_gamma = list(
      gamma_test_values = I(as.numeric(gamma_test_values)),
      power_values = I(as.numeric(power_values))
    ),
    r_gamma_p = list(
      target_power = I(c(0.5, 0.8)),
      gamma_p_values = I(as.numeric(gamma_p_values))
    ),
    fixture_name = fixture_name
  )
}

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

cat("Building fixture 1: uniform_3_pre_periods_no_anticipation...\n")
f1 <- build_event_study_fixture(
  pre_periods = c(-3L, -2L, -1L),
  post_periods = c(1L, 2L, 3L),
  seed = 101L
)
fixture_1 <- extract_pretrends(f1, "uniform_3_pre_periods_no_anticipation")

cat("Building fixture 2: irregular_pre_periods...\n")
# K=3 with t in {-5, -3, -1}. Tests PR-B's gamma-unit linear-pattern fix:
# pre-PR-B Python with normalized count-based weights silently reported
# MDV in [0.45, 0.30, 0.15] / sqrt(0.3) units, not gamma. R
# `slope_for_power()` always reports gamma; Python's PR-B Step 4 makes the
# two match at atol=1e-4.
f2 <- build_event_study_fixture(
  pre_periods = c(-5L, -3L, -1L),
  post_periods = c(1L, 2L, 3L),
  seed = 202L
)
fixture_2 <- extract_pretrends(f2, "irregular_pre_periods")

cat("Building fixture 3: anticipation_shifted (K=4 shifted-grid)...\n")
# K=4 pre-periods at {-5, -4, -3, -2} — a shifted-grid case (gap at t=-1
# between the last pre-period and the reference period 0). Locks R parity
# at a larger K and on a non-adjacent-to-reference grid. Note: does NOT
# exercise the Python-side `_extract_pre_period_params` CS/SA
# anticipation-filter path against R goldens — see the file-header
# comment for the rationale and deferred follow-up.
f3 <- build_event_study_fixture(
  pre_periods = c(-5L, -4L, -3L, -2L),
  post_periods = c(1L, 2L, 3L),
  seed = 303L
)
fixture_3 <- extract_pretrends(f3, "anticipation_shifted")

cat("Building fixture 4: single_pre_period_closed_form...\n")
# K=1 with diagonal Sigma = 0.25*I. Locks Roth Proposition 2 univariate
# truncated-normal closed form against R AND the analytical scalar
# expression `1 - Phi(z - gamma/sigma) + Phi(-z - gamma/sigma)`.
f4 <- build_event_study_fixture(
  pre_periods = c(-1L),
  post_periods = c(1L),
  sigma2 = 0.25,
  rho = 0.0,
  seed = 404L
)
fixture_4 <- extract_pretrends(f4, "single_pre_period_closed_form")

# ---------------------------------------------------------------------------
# Write JSON
# ---------------------------------------------------------------------------

out <- list(
  meta = list(
    generated_at = format(Sys.Date()),
    pretrends_version = as.character(packageVersion("pretrends")),
    pretrends_commit = PRETRENDS_COMMIT,
    r_version = R.version.string,
    description = paste(
      "Roth (2022) PreTrendsPower parity goldens for diff-diff",
      "compute_pretrends_power / PreTrendsPower (PR-C).",
      "Three-tier parity contract, both numeric tiers at atol=1e-4:",
      "(1) NIS box probability at fixed gamma values on all 4 fixtures",
      "(atol=1e-4; R hardcodes thresholdTstat=1.96 while Python uses",
      "scipy.stats.norm.ppf(0.975) = 1.959963984540054, and mvtnorm::pmvnorm vs",
      "scipy MVN CDF Genz-Bretz randomized-lattice differences bound the",
      "K=4 NIS power gap at ~5e-5);",
      "(2) gamma_p MDV (slope at target power 0.5 and 0.8) on regular,",
      "irregular, shifted-grid (K=4), and K=1 grids (atol=1e-4; R uniroot tol",
      "vs Python brentq xtol gap dominates);",
      "(3) gamma-unit MDV invariance: PR-B's skip-L2-norm path produces MDV",
      "in Roth's gamma units exactly, matching R's slope_for_power().",
      "See diff-diff/docs/methodology/papers/roth-2022-review.md for",
      "the full derivation."
    )
  ),
  uniform_3_pre_periods_no_anticipation = fixture_1,
  irregular_pre_periods = fixture_2,
  anticipation_shifted = fixture_3,
  single_pre_period_closed_form = fixture_4
)

out_path <- "../data/r_pretrends_golden.json"
write_json(out, out_path, pretty = TRUE, digits = NA, auto_unbox = TRUE)
cat(sprintf("Wrote %s\n", out_path))
