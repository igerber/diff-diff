# Black-box parity fixture for the DMLDiD bad-control lane (Caetano, Callaway,
# Payne & Sant'Anna 2026, "Difference-in-differences with 'bad controls'",
# arXiv:2608.03881) against the authors' R package `badcontrols`.
#
# LICENSE POSTURE: `badcontrols` is GPL-3; diff-diff is MIT. This script only
# EXECUTES the package as a black-box oracle. Its source (R/*.R) is never read
# by the implementer, and nothing in diff_diff/ derives from it - the Python
# lane is derived from the paper alone (docs/methodology/papers/
# caetano-2026-review.md).
#
# Usage:
#   Rscript benchmarks/R/requirements.R    # installs the pinned packages
#   Rscript benchmarks/R/generate_badcontrols_golden.R
#
# Output:
#   benchmarks/data/badcontrols_panel.csv   (the simulated panel, shared input)
#   benchmarks/data/badcontrols_golden.json (per-(g,t) R results + metadata)
#
# Python test loader: tests/test_dml_did_bad_controls_parity.py.
#
# Parity is TOLERANCE-based, not bit-exact: `didbc()` cross-fits with its own
# fold draw (nfolds=1 errors); single-seed fold noise is ~0.25-0.35 analytical SE
# on the smallest cells, so every run is recorded as the per-cell mean over
# N_SEEDS seeds (per-seed rows kept too) and the Python side averages the same
# number of its own seeds. `att_gt$se` is bootstrap-based
# even with bstrap=FALSE, so the analytical variance `att_gt$V_analytical`
# (SE = sqrt(V/n)) is recorded instead. `overlap_threshold = 1` disables the
# package's silent fallback to its imputation estimator on high-propensity
# cells (the fallback would change both the estimand and the RNG stream).
#
# Pins: badcontrols 1.0.0 at GitHub commit 651ccc92 (no tags/releases exist),
# ptetools 1.0.0 (CRAN). Bump here AND in the parity test's
# `test_metadata_versions_match` and requirements.R when re-anchoring.

suppressPackageStartupMessages({
  library(jsonlite)
  library(badcontrols)
  library(ptetools)
})

BADCONTROLS_SHA <- "651ccc925776125bb9233d76867c862107ea0ba5"
stopifnot(packageVersion("badcontrols") == "1.0.0")
stopifnot(packageVersion("ptetools") == "1.0.0")
remote_sha <- packageDescription("badcontrols")$RemoteSha
if (is.null(remote_sha) || !identical(remote_sha, BADCONTROLS_SHA)) {
  stop(sprintf(
    "badcontrols must be installed from GitHub at %s (found RemoteSha=%s); run Rscript benchmarks/R/requirements.R",
    BADCONTROLS_SHA, if (is.null(remote_sha)) "NULL" else remote_sha
  ))
}

SEED <- 20260905
N <- 1000
T_MAX <- 4
set.seed(SEED)
sim <- simulate_bad_controls(n = N, T_max = T_MAX)
panel <- sim$data
write.csv(panel, "benchmarks/data/badcontrols_panel.csv", row.names = FALSE)

# Complete-case treated count per (g, t) cell on the panel: didbc's per-cell
# `n` is the cell size; the parity test also wants the treated mass to
# diagnose any systematic 1/pi vs 1/(1-pi) normalization gap.
n_treated_in <- function(g) sum(panel$G == g & panel$period == min(panel$period))

run_didbc <- function(run_seed, control_group, bad_control = TRUE, cov_formula = NULL) {
  set.seed(run_seed)
  args <- list(
    yname = "Y", gname = "G", tname = "period", idname = "id", data = panel,
    xformula = ~Z,
    est_method = "dr_ml", nuisance_method = "parametric",
    control_group = control_group, base_period = "varying", anticipation = 0,
    bstrap = FALSE, overlap_threshold = 1, nfolds = 5
  )
  if (bad_control) {
    args$bad_control_formula <- ~X
    if (!is.null(cov_formula)) args$bad_control_cov_formula <- cov_formula
  }
  res <- do.call(didbc, args)
  gt <- res$att_gt
  # V_analytical is the full (n_cells x n_cells) analytical covariance of the
  # cell estimates in att_gt row order; the per-cell variance is its diagonal.
  V <- gt$V_analytical
  if (is.matrix(V)) {
    stopifnot(nrow(V) == length(gt$att), ncol(V) == length(gt$att))
    V <- diag(V)
  }
  stopifnot(length(V) == length(gt$att))
  data.frame(
    group = as.numeric(gt$group),
    t = as.numeric(gt$t),
    att = as.numeric(gt$att),
    V_analytical = as.numeric(V),
    n = as.numeric(gt$n),
    n_treated = vapply(gt$group, n_treated_in, numeric(1))
  )
}

runs <- list(
  list(key = "w_nyt", seed = 101, control_group = "notyettreated", bad_control = TRUE, cov = ~W,
       description = "bad_control=X, W=[W], not-yet-treated (Proposition 2)"),
  list(key = "w_nt", seed = 102, control_group = "nevertreated", bad_control = TRUE, cov = ~W,
       description = "bad_control=X, W=[W], never-treated"),
  list(key = "ylag_nyt", seed = 103, control_group = "notyettreated", bad_control = TRUE, cov = ~Y,
       description = "bad_control=X, W=[Y] (Remark 5 lagged outcome), not-yet-treated"),
  list(key = "plain_nyt", seed = 104, control_group = "notyettreated", bad_control = FALSE, cov = NULL,
       description = "no bad control (xformula=~Z only), not-yet-treated; CHARACTERIZATION only")
)

N_SEEDS <- 10
results <- list()
for (r in runs) {
  message(sprintf("Running %s (%d seeds) ...", r$key, N_SEEDS))
  per_seed <- list()
  for (i in seq_len(N_SEEDS)) {
    seed_i <- r$seed * 10 + i
    df <- run_didbc(seed_i, r$control_group, r$bad_control, r$cov)
    per_seed[[i]] <- list(seed = seed_i, att_gt = df)
  }
  # Seed-mean of the per-cell point estimates and variances (all seeds share
  # the att_gt row order; asserted).
  base <- per_seed[[1]]$att_gt
  for (ps in per_seed) stopifnot(identical(ps$att_gt$group, base$group), identical(ps$att_gt$t, base$t))
  att_mean <- Reduce(`+`, lapply(per_seed, function(ps) ps$att_gt$att)) / N_SEEDS
  V_mean <- Reduce(`+`, lapply(per_seed, function(ps) ps$att_gt$V_analytical)) / N_SEEDS
  att_sd <- apply(do.call(cbind, lapply(per_seed, function(ps) ps$att_gt$att)), 1, sd)
  results[[r$key]] <- list(
    control_group = r$control_group,
    bad_control = r$bad_control,
    bad_control_cov_formula = if (is.null(r$cov)) NULL else deparse(r$cov),
    description = r$description,
    seeds = vapply(per_seed, function(ps) ps$seed, numeric(1)),
    att_gt_mean = data.frame(group = base$group, t = base$t, att = att_mean,
                             V_analytical = V_mean, att_seed_sd = att_sd,
                             n = base$n, n_treated = base$n_treated),
    att_gt_per_seed = lapply(per_seed, function(ps) ps$att_gt)
  )
}

true_att_gt <- sim$true_att_gt
out <- list(
  meta = list(
    generator = "benchmarks/R/generate_badcontrols_golden.R",
    generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
    r_version = R.version.string,
    badcontrols_version = as.character(packageVersion("badcontrols")),
    badcontrols_remote_sha = remote_sha,
    ptetools_version = as.character(packageVersion("ptetools")),
    seed = SEED, n = N, T_max = T_MAX,
    panel_csv = "benchmarks/data/badcontrols_panel.csv",
    columns = list(unit = "id", time = "period", first_treat = "G", outcome = "Y",
                   bad_control = "X", covariates = list("Z"), bad_control_covariate = "W"),
    python_mapping = paste(
      "DMLDiD(control_group=<cg>, base_period='varying', pscore_trim=1e-6, seed=0)",
      ".fit(panel, outcome='Y', unit='id', time='period', first_treat='G',",
      "covariates=['Z'], bad_control='X', bad_control_covariates=[<W or Y>])"
    ),
    common_args = list(est_method = "dr_ml", nuisance_method = "parametric",
                       base_period = "varying", anticipation = 0, bstrap = FALSE,
                       overlap_threshold = 1, nfolds = 5, xformula = "~Z"),
    se_convention = "SE = sqrt(V_analytical / n) with V_analytical = diag(att_gt$V_analytical) and n = att_gt$n (the panel unit count); att_gt$se is bootstrap-based even with bstrap=FALSE",
    n_seeds = N_SEEDS,
    tolerance_rationale = paste(
      "didbc cross-fits with its own fold draw (nfolds=1 errors); single-seed fold noise",
      "is ~0.25-0.35 SE per implementation on the smallest cells (att_seed_sd records it),",
      "so each run is recorded as the mean over N_SEEDS seeds and the Python side averages",
      "the same number of seeds: parity is |mean_att_py - mean_att_R| < 0.5 * SE_R per cell",
      "and |mean_se_py / SE_R - 1| < 0.3, never bit-exact"
    ),
    license_note = "badcontrols is GPL-3 and used ONLY as an executed black-box oracle; its source is never read"
  ),
  true_att_gt = true_att_gt,
  runs = results
)
write_json(out, "benchmarks/data/badcontrols_golden.json", digits = 17, auto_unbox = TRUE, pretty = TRUE, null = "null")
message("Wrote benchmarks/data/badcontrols_golden.json")
