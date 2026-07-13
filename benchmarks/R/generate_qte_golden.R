#!/usr/bin/env Rscript
# Golden fixtures for the diff-diff ChangesInChanges / QDiD estimators vs qte 1.3.1.
#
# Regenerate with:
#   Rscript benchmarks/R/generate_qte_golden.R
#
# Output: benchmarks/data/qte_golden.json, consumed by
# tests/test_changes_in_changes_parity.py (which pytest.skips when absent).
#
# Version contract: qte is pinned to 1.3.1 and quantreg to 6.1 here, in
# benchmarks/R/requirements.R (pinned_versions), and in the parity test's
# test_metadata_versions_match. Bump all in lockstep when re-anchoring.
# quantreg is part of the contract because the covariate (xformla) fixtures
# embed quantreg's rq / predict.rqs behavior, not just qte's.
#
# Point fixtures are generated with se=FALSE so they are fully deterministic.
# The SE block runs seeded (set.seed before each call) with iters=999 AND
# cores=1: qte's default cores=2 parallelizes the bootstrap via forking, whose
# per-child RNG seeding is NOT reproducible even under set.seed (verified
# empirically 2026-07-13 - back-to-back seeded runs differ). With cores=1 the
# draw sequence is fully deterministic. Python still compares these SEs
# statistically, not bit-exactly (the R draw sequence cannot be replicated
# cross-language).

suppressMessages({
  library(qte)
  library(quantreg)
  library(jsonlite)
})

stopifnot(packageVersion("qte") == "1.3.1")
stopifnot(packageVersion("quantreg") == "6.1")

probs <- seq(0.05, 0.95, 0.05)
# qte's covariate branch hardcodes this 99-tau quantile-regression grid inside
# compute.CiC / compute.QDiD; stored in metadata so Python pins its own copy.
qr_taus <- seq(0.01, 0.99, 0.01)

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

make_cov1_2x2 <- function(n_treated, n_control, seed) {
  # One covariate: composition differs by group, and both the time trend and
  # the noise scale depend on x, so covariate adjustment matters and CiC/QDiD
  # differ. x is time-invariant; continuous throughout (the conditional-rank
  # pipeline is knife-edge on outcomes that sit exactly on predicted knots, so
  # covariate fixtures avoid mass points entirely).
  set.seed(seed)
  n <- n_treated + n_control
  treat <- c(rep(1L, n_treated), rep(0L, n_control))
  x <- runif(n, 0, 2) + 0.4 * treat
  u <- rnorm(n)
  y_pre <- 0.5 + 0.8 * x + (0.4 + 0.2 * x) * u
  y_post <- 0.8 + 1.1 * x + (0.5 + 0.2 * x) * (0.7 * u + 0.5 * rnorm(n)) +
    treat * (0.6 + 0.3 * pnorm(u))
  data.frame(
    id = rep(seq_len(n), times = 2),
    period = rep(c(0L, 1L), each = n),
    treat = rep(treat, times = 2),
    x1 = rep(x, times = 2),
    y = c(y_pre, y_post)
  )
}

make_cov2_2x2 <- function(n_treated, n_control, seed) {
  # Two covariates: continuous x1 plus a binary x2 stored as 0/1 NUMERIC
  # (exercises multi-column quantile regression; the fixture never uses R
  # factors so the R and Python design matrices are identical by construction).
  set.seed(seed)
  n <- n_treated + n_control
  treat <- c(rep(1L, n_treated), rep(0L, n_control))
  x1 <- rnorm(n, mean = 1, sd = 0.5) + 0.3 * treat
  x2 <- as.numeric(runif(n) < 0.4 + 0.2 * treat)
  u <- rnorm(n)
  y_pre <- 0.2 + 0.6 * x1 - 0.4 * x2 + (0.5 + 0.15 * x1) * u
  y_post <- 0.5 + 0.9 * x1 - 0.2 * x2 + (0.5 + 0.15 * x1) * (0.6 * u + 0.6 * rnorm(n)) +
    treat * (0.5 + 0.4 * pnorm(u))
  df <- data.frame(
    id = rep(seq_len(n), times = 2),
    period = rep(c(0L, 1L), each = n),
    treat = rep(treat, times = 2),
    x1 = rep(x1, times = 2),
    x2 = rep(x2, times = 2),
    y = c(y_pre, y_post)
  )
  # Guard against a degenerate draw: x2 must vary within every (g, t) cell.
  for (gv in 0:1) for (tv in 0:1) {
    stopifnot(var(df$x2[df$treat == gv & df$period == tv]) > 0)
  }
  df
}

dgps <- list(
  normal_2x2_n500 = make_normal_2x2(250, 250, seed = 20260712),
  lognormal_2x2_n300 = make_lognormal_2x2(150, 150, seed = 20260713),
  # Small unbalanced-cell design stressing near-integer n*p type-1 boundaries.
  smalln_2x2_n60 = make_normal_2x2(35, 25, seed = 20260714)
)

# Covariate scenarios (qte xformla branch). Kept separate from `dgps`: their
# result keys and data blocks carry covariate columns.
#
# WHY END-TO-END COVARIATE PARITY IS TIE-SELECTION-BOUNDED (not ~1e-10 like
# the unconditional fixtures). Two exact-tie artifacts make bit-level
# cross-language agreement unattainable for the QR pipeline:
#   (i)  Optimal-face vertex selection: the QR check loss can have a
#        non-trivial optimal face (structural with binary covariates; data-
#        dependent otherwise). R's br simplex and Python's HiGHS then return
#        DIFFERENT exact minimizers (equal loss to ~1e-15, coefficients apart
#        by O(1e-1)).
#   (ii) Fhat knot-tie ordering: adjacent taus often share an interpolating
#        basis, so R's predicted knots are EXACTLY tied and predict.rqs's
#        stable sort orders them by tau index. Python coefficients agree only
#        to ~1e-15, which breaks those ties in arbitrary value order, so
#        ~5-10% of conditional ranks legitimately land a grid step away.
# Both are selections among equally valid solutions. The parity suite
# therefore pins the CONVENTIONS at atol=0 conditioned on R's stored
# coefficients/predictions (qr_cases below), proves the LP solver optimal
# per tau (equal coefficients OR equal loss), and gates end-to-end results
# at empirically measured tie-selection bounds (metadata cov_*_atol).
# Cell sizes are kept coprime to 100 (integer n*tau is one avoidable
# degeneracy source); data are continuous throughout.
cov_specs <- list(
  cov1_2x2_n300 = list(
    df = make_cov1_2x2(151, 149, seed = 20260716),
    xformla = ~x1, xcols = c("x1")
  ),
  cov2_2x2_n240 = list(
    df = make_cov2_2x2(123, 117, seed = 20260717),
    xformla = ~ x1 + x2, xcols = c("x1", "x2")
  )
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

run_point <- function(df, method, panel, xformla = NULL) {
  fn <- if (method == "cic") CiC else QDiD
  res <- fn(
    formla = y ~ treat,
    xformla = xformla,
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

for (cov_name in names(cov_specs)) {
  spec <- cov_specs[[cov_name]]
  df <- spec$df
  data_block <- list(
    id = df$id,
    period = df$period,
    treat = df$treat,
    y = df$y
  )
  for (cn in spec$xcols) data_block[[cn]] <- df[[cn]]
  entry <- list(
    data = data_block,
    covariates = spec$xcols,
    results = list()
  )
  for (method in c("cic", "qdid")) {
    for (panel in c(TRUE, FALSE)) {
      key <- sprintf("%s_cov_%s", method, if (panel) "panel" else "rcs")
      entry$results[[key]] <- run_point(df, method, panel, xformla = spec$xformla)
      message(sprintf(
        "%s / %s: ate = %.10g", cov_name, key, entry$results[[key]]$ate
      ))
    }
  }
  scenarios[[cov_name]] <- entry
}

# ---------------------------------------------------------------------------
# SE block (statistical parity): normal_2x2_n500 only, seeded, iters = 999
# ---------------------------------------------------------------------------

run_se <- function(df, method, panel, seed, xformla = NULL) {
  fn <- if (method == "cic") CiC else QDiD
  set.seed(seed)
  res <- fn(
    formla = y ~ treat,
    xformla = xformla,
    t = 1, tmin1 = 0, tname = "period",
    data = df,
    panel = panel,
    idname = if (panel) "id" else NULL,
    se = TRUE, iters = 999,
    # cores = 1 is REQUIRED for reproducibility: qte's default cores = 2
    # forks the bootstrap and the per-child RNG seeding ignores set.seed.
    cores = 1,
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
  qdid_rcs = run_se(dgps$normal_2x2_n500, "qdid", FALSE, seed = 42),
  # Covariate SE parity: one panel CiC + one RCS QDiD block on the 1-covariate
  # scenario (data lives under scenarios$cov1_2x2_n300).
  cic_cov_panel = run_se(cov_specs$cov1_2x2_n300$df, "cic", TRUE, seed = 42, xformla = ~x1),
  qdid_cov_rcs = run_se(cov_specs$cov1_2x2_n300$df, "qdid", FALSE, seed = 42, xformla = ~x1)
)
se_block$cic_cov_panel$covariates <- c("x1")
se_block$cic_cov_panel$scenario <- "cov1_2x2_n300"
se_block$qdid_cov_rcs$covariates <- c("x1")
se_block$qdid_cov_rcs$scenario <- "cov1_2x2_n300"
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
# QR micro-fixtures (covariate convention anchor): per-cell rq coefficients,
# prediction matrices, and the exact predict.rqs Fhat/Qhat outputs qte's
# covariate branch consumes - so Python can pin the step-function conventions
# at atol=0 given R's coefficients (isolating convention-port errors from
# LP-solver last-digit differences), and the LP solver separately.
# ---------------------------------------------------------------------------

qr_case <- function(df, xcols, xformla) {
  cells <- list(
    c00 = df[df$treat == 0 & df$period == 0, ],
    c01 = df[df$treat == 0 & df$period == 1, ],
    c10 = df[df$treat == 1 & df$period == 0, ],
    c11 = df[df$treat == 1 & df$period == 1, ]
  )
  yformla <- as.formula(paste("y ~", paste(xcols, collapse = " + ")))
  QR00 <- rq(yformla, data = cells$c00, tau = qr_taus)
  QR01 <- rq(yformla, data = cells$c01, tau = qr_taus)
  QR10 <- rq(yformla, data = cells$c10, tau = qr_taus)

  # Prediction matrices at the treated-pre covariates (n10 x 99): separates
  # dot-product-ulp failures from step-function-convention failures.
  X10d <- cbind(1, as.matrix(cells$c10[, xcols, drop = FALSE]))
  preds00 <- unname(X10d %*% coef(QR00))
  preds01 <- unname(X10d %*% coef(QR01))
  preds10 <- unname(X10d %*% coef(QR10))

  n10 <- nrow(cells$c10)
  # CiC composition (mirrors qte compute.CiC's xformla branch).
  F00 <- predict(QR00, newdata = cells$c10, type = "Fhat", stepfun = TRUE)
  cic_ranks <- sapply(seq_len(n10), function(i) F00[[i]](cells$c10$y[i]))
  Q01 <- predict(QR01, newdata = cells$c10, type = "Qhat", stepfun = TRUE)
  cic_y0t <- sapply(seq_len(n10), function(i) Q01[[i]](cic_ranks[i]))
  # QDiD composition (mirrors qte compute.QDiD's xformla branch).
  F10 <- predict(QR10, newdata = cells$c10, type = "Fhat", stepfun = TRUE)
  qdid_ranks <- sapply(seq_len(n10), function(i) F10[[i]](cells$c10$y[i]))
  Q00 <- predict(QR00, newdata = cells$c10, type = "Qhat", stepfun = TRUE)
  qdid_y0t <- cells$c10$y +
    sapply(seq_len(n10), function(i) Q01[[i]](qdid_ranks[i])) -
    sapply(seq_len(n10), function(i) Q00[[i]](qdid_ranks[i]))

  # End-to-end micro results from the actual qte calls (RCS mode): guards
  # against qte's internals diverging from raw predict.rqs usage.
  cic_res <- CiC(y ~ treat, xformla = xformla, t = 1, tmin1 = 0, tname = "period",
                 data = df, panel = FALSE, se = FALSE, probs = probs)
  qdid_res <- QDiD(y ~ treat, xformla = xformla, t = 1, tmin1 = 0, tname = "period",
                   data = df, panel = FALSE, se = FALSE, probs = probs)

  cell_block <- lapply(cells, function(cc) {
    b <- list(y = cc$y)
    for (cn in xcols) b[[cn]] <- cc[[cn]]
    b
  })
  list(
    covariates = xcols,
    cells = cell_block,
    coef00 = unname(as.matrix(coef(QR00))),  # (k+1) x 99, intercept row first
    coef01 = unname(as.matrix(coef(QR01))),
    coef10 = unname(as.matrix(coef(QR10))),
    preds00_at_x10 = preds00,
    preds01_at_x10 = preds01,
    preds10_at_x10 = preds10,
    cic_ranks = cic_ranks,
    cic_y0t = cic_y0t,
    qdid_ranks = qdid_ranks,
    qdid_y0t = qdid_y0t,
    cic_ate = cic_res$ate, cic_qte = as.numeric(cic_res$qte),
    qdid_ate = qdid_res$ate, qdid_qte = as.numeric(qdid_res$qte)
  )
}

# Micro cells sized coprime to 100 too (see the cov_specs comment). The
# qr1 case (continuous covariate) has unique QR optima, so _rq_fit matches
# R's coefficients directly; qr2's binary covariate makes optimal faces
# structural, exercising the equal-loss branch of the solver test.
qr_cases <- list(
  qr1_cov1_n41cell = qr_case(make_cov1_2x2(41, 41, seed = 20260718), c("x1"), ~x1),
  qr2_cov2_n33cell = qr_case(make_cov2_2x2(33, 29, seed = 20260719), c("x1", "x2"), ~ x1 + x2)
)
message("QR micro-fixtures done.")

# ---------------------------------------------------------------------------
# Write JSON
# ---------------------------------------------------------------------------

out <- list(
  metadata = list(
    description = paste(
      "Golden fixtures for diff-diff ChangesInChanges/QDiD parity vs qte.",
      "Point fixtures use se=FALSE (deterministic); the se_block is seeded",
      "(set.seed before each call, iters=999) and compared statistically.",
      "Covariate (xformla) fixtures also embed quantreg rq/predict.rqs",
      "behavior; qr_cases anchors those conventions at atol=0."
    ),
    qte_version = as.character(packageVersion("qte")),
    quantreg_version = as.character(packageVersion("quantreg")),
    r_version = as.character(getRversion()),
    probs = probs,
    qr_taus = qr_taus,
    point_atol = 1e-10,
    # Covariate end-to-end gates: tie-selection bounds (see the cov_specs
    # comment). The committed scenarios measure worst att 9.3e-3 / qte 8.1e-2
    # on the generation platform; a 30-dataset randomized decomposition audit
    # (2026-07-13) observed same-mechanism deviations up to 3.5e-2 / 0.40 on
    # adversarial (binary-covariate / ties-heavy / small-cell) data, so the
    # gates carry cross-platform tie-flip margin. The exact conventions are
    # pinned at atol=0 by qr_cases instead.
    cov_att_atol = 0.04,
    cov_qte_atol = 0.25,
    n_scenarios = length(scenarios)
  ),
  scenarios = scenarios,
  se_block = se_block,
  quantile_cases = micro,
  qr_cases = qr_cases
)

out_path <- file.path("benchmarks", "data", "qte_golden.json")
write_json(out, path = out_path, digits = 17, auto_unbox = TRUE, null = "null")
message(sprintf("Wrote %s (%.1f KB)", out_path, file.size(out_path) / 1024))
