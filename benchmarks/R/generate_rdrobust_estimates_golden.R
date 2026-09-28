# Golden-value generator for the diff-diff RD ESTIMATION port - sharp,
# fuzzy, AND covariate-adjusted (diff_diff/_rdrobust_port.py::rdrobust_fit
# and the public RegressionDiscontinuity estimator; 32 configs across five
# synthetic DGPs + the Senate data, incl. 7 fuzzy configs with full
# first-stage tau_T/se_T/z_T/pv_T/ci_T blocks and 9 covariate configs with
# coef_covs (gamma) pins; covariate names deliberately differ in length
# and are passed UNSORTED so every covariate config also pins rdrobust's
# order(nchar(colnames)) column sort, rdrobust.R:131).
#
# Deliberately a SEPARATE file/JSON from generate_rdrobust_golden.R so the
# bandwidth fixtures reviewed in the machinery PR are never regenerated.
# Same parity target: CRAN rdrobust 4.0.0 (tarball sha256
# 78f0d6b4bdec4091cc8f42f6f1598704747f95926446d3aaee381ea1d613a36f);
# the GitHub 4.1.0-dev tree must NOT be installed when regenerating.
#
# Outputs benchmarks/data/rdrobust_estimates_golden.json: per config the
# embedded inputs (17 significant digits), selected/echoed bandwidths, the
# full three-row (Conventional / Bias-Corrected / Robust) coef/se/z/pv/ci
# block, effective counts, and per-side order-p coefficient vectors.
#
# Run from the repo root: Rscript benchmarks/R/generate_rdrobust_estimates_golden.R

library(rdrobust)
library(jsonlite)

stopifnot(packageVersion("rdrobust") == "4.0.0")

TARBALL_SHA256 <- "78f0d6b4bdec4091cc8f42f6f1598704747f95926446d3aaee381ea1d613a36f"

run_estimate <- function(y, x, c = 0, masspoints = "adjust", kernel = "tri",
                         p = 1, q = 2, h = NULL, b = NULL, rho = NULL,
                         level = 95, bwselect = "mserd",
                         fuzzy = NULL, sharpbw = FALSE,
                         covs = NULL, covs_drop = TRUE) {
  args <- list(y = y, x = x, c = c, masspoints = masspoints, kernel = kernel,
               p = p, q = q, level = level, bwselect = bwselect,
               sharpbw = sharpbw, covs_drop = covs_drop)
  if (!is.null(h)) args$h <- h
  if (!is.null(b)) args$b <- b
  if (!is.null(rho)) args$rho <- rho
  if (!is.null(fuzzy)) args$fuzzy <- fuzzy
  if (!is.null(covs)) args$covs <- covs
  r <- suppressWarnings(do.call(rdrobust, args))
  out <- list(
    c = c, masspoints = masspoints, kernel = kernel, p = p, q = q,
    bwselect = bwselect,
    fuzzy_in = !is.null(fuzzy), sharpbw = sharpbw,
    covs_in = !is.null(covs),
    covs_names = if (is.null(covs)) NA else colnames(covs),
    covs_drop = covs_drop,
    h_in = if (is.null(h)) NA else h,
    b_in = if (is.null(b)) NA else b,
    rho_in = if (is.null(rho)) NA else rho,
    level = level,
    h_l = r$bws["h", "left"], h_r = r$bws["h", "right"],
    b_l = r$bws["b", "left"], b_r = r$bws["b", "right"],
    tau_cl = unname(r$coef[1, 1]), tau_bc = unname(r$coef[2, 1]),
    se_cl = unname(r$se[1, 1]), se_rb = unname(r$se[3, 1]),
    z = unname(r$z[, 1]), pv = unname(r$pv[, 1]),
    ci_lower = unname(r$ci[, 1]), ci_upper = unname(r$ci[, 2]),
    N = unname(r$N), N_h = unname(r$N_h), N_b = unname(r$N_b),
    bias = unname(r$bias),
    beta_p_l = unname(as.vector(r$beta_Y_p_l)),
    beta_p_r = unname(as.vector(r$beta_Y_p_r))
  )
  if (!is.null(fuzzy)) {
    # First-stage three-row block (Conventional / Bias-Corrected / Robust)
    out$tau_T <- unname(r$tau_T)
    out$se_T <- unname(r$se_T)
    out$z_T <- unname(r$z_T)
    out$pv_T <- unname(r$pv_T)
    out$ci_T_lower <- unname(r$ci_T[, 1])
    out$ci_T_upper <- unname(r$ci_T[, 2])
    out$beta_t_p_l <- unname(as.vector(r$beta_T_p_l))
    out$beta_t_p_r <- unname(as.vector(r$beta_T_p_r))
  }
  if (!is.null(covs)) {
    # Common projection coefficients gamma (dZ_kept x 1 sharp, x 2 fuzzy)
    # over the covariates KEPT after covs_drop, in R's nchar-sorted column
    # order; the row count pins WHICH columns survived the drop.
    out$coef_covs <- unname(as.matrix(r$coef_covs))
  }
  out
}

golden <- list()

golden$metadata <- list(
  rdrobust_version = as.character(packageVersion("rdrobust")),
  rdrobust_tarball_sha256 = TARBALL_SHA256,
  seeds = list(dgp_lee_smooth = 42L, dgp_ties_moderate = 123L,
               dgp_asymmetric_scaled = 777L, dgp_fuzzy = 314L,
               dgp_covs = 2718L),
  generator = "benchmarks/R/generate_rdrobust_estimates_golden.R",
  algorithm = paste(
    "rdrobust() sharp, fuzzy, AND covariate-adjusted estimation blocks",
    "(three-row coef/se/z/pv/ci, counts, per-side beta_p; fuzzy configs add",
    "the first-stage tau_T/se_T/z_T/pv_T/ci_T rows and per-side beta_T_p;",
    "covariate configs add the coef_covs gamma matrix) for the vce='nn'",
    "path, complementing the bandwidth fixtures in rdrobust_golden.json."
  ),
  r_version = R.version.string
)

# Same seeded DGPs as generate_rdrobust_golden.R (kept in sync by seed).
set.seed(42)
n <- 1000
x1 <- 2 * rbeta(n, 2, 4) - 1
y1 <- 0.48 + 1.27 * x1 + 7.18 * x1^2 + 20.21 * x1^3 + 21.54 * x1^4 +
  7.33 * x1^5 + 0.04 * (x1 >= 0) + rnorm(n, sd = 0.1295)

golden$dgp_lee_smooth <- list(
  x = x1, y = y1,
  configs = list(
    default    = run_estimate(y1, x1),
    manual_h   = run_estimate(y1, x1, h = 0.15),
    h_rho2     = run_estimate(y1, x1, h = 0.15, rho = 2),
    rho_only   = run_estimate(y1, x1, rho = 2),
    p2q3       = run_estimate(y1, x1, p = 2, q = 3),
    p0q1       = run_estimate(y1, x1, p = 0, q = 1),
    epa        = run_estimate(y1, x1, kernel = "epa"),
    uni        = run_estimate(y1, x1, kernel = "uni"),
    level90    = run_estimate(y1, x1, level = 90),
    msetwo     = run_estimate(y1, x1, bwselect = "msetwo"),
    cercomb2   = run_estimate(y1, x1, bwselect = "cercomb2")
  )
)

set.seed(123)
n2 <- 800
x2 <- round(2 * rbeta(n2, 2, 4) - 1, 2)
y2 <- 0.5 + 0.8 * x2 + (x2 >= 0) * 1.0 + rnorm(n2, sd = 0.3)

golden$dgp_ties_moderate <- list(
  x = x2, y = y2,
  configs = list(
    adjust = run_estimate(y2, x2, masspoints = "adjust"),
    off    = run_estimate(y2, x2, masspoints = "off")
  )
)

set.seed(777)
n3 <- 300
x3 <- 40 * rbeta(n3, 5, 2)
y3 <- 2 + 0.05 * x3 - 0.001 * x3^2 + 0.8 * (x3 >= 28) + rnorm(n3, sd = 0.5)

golden$dgp_asymmetric_scaled <- list(
  x = x3, y = y3,
  configs = list(
    default = run_estimate(y3, x3, c = 28)
  )
)

# Fuzzy DGP: two-sided imperfect compliance (take-up jumps 0.15 -> 0.75).
set.seed(314)
n4 <- 1500
x4 <- 2 * rbeta(n4, 2, 4) - 1
t4 <- rbinom(n4, 1, ifelse(x4 >= 0, 0.75, 0.15))
y4 <- 0.5 * x4 + 1.2 * t4 + rnorm(n4, sd = 0.3)
# One-sided perfect compliance variant (T == 0 left of the cutoff):
# exercises the perf_comp bandwidth auto-switch (rdbwselect.R:334-346).
t4_one <- ifelse(x4 >= 0, t4, 0)
# Tied running variable variant (2dp rounding; keeps all masspoints modes
# runnable in R per the sharp-golden lesson).
x4_ties <- round(x4, 2)

golden$dgp_fuzzy <- list(
  x = x4, y = y4, t = t4, t_one = t4_one, x_ties = x4_ties,
  configs = list(
    default      = run_estimate(y4, x4, fuzzy = t4),
    sharpbw_true = run_estimate(y4, x4, fuzzy = t4, sharpbw = TRUE),
    manual_h     = run_estimate(y4, x4, fuzzy = t4, h = 0.2),
    epa          = run_estimate(y4, x4, fuzzy = t4, kernel = "epa"),
    msetwo       = run_estimate(y4, x4, fuzzy = t4, bwselect = "msetwo"),
    one_sided    = run_estimate(y4, x4, fuzzy = t4_one),
    ties_adjust  = run_estimate(y4, x4_ties, fuzzy = t4)
  )
)

# Covariate DGP: two informative covariates with NAME LENGTHS that differ
# and are passed UNSORTED (c("zlong", "zb")) so R's order(nchar) column
# sort (rdrobust.R:131) is exercised by every config; zdup is an EXACT
# linear combination for the covs_drop config. covs_ties reuses the
# 2dp-rounded running variable (masspoints machinery x covariates).
set.seed(2718)
n5 <- 1200
x5 <- 2 * rbeta(n5, 2, 4) - 1
zlong <- 0.5 * x5 + rnorm(n5, sd = 0.8)
zb <- rbinom(n5, 1, 0.4)
y5 <- 0.4 * x5 + 0.9 * (x5 >= 0) + 0.7 * zlong + 0.3 * zb + rnorm(n5, sd = 0.3)
t5 <- rbinom(n5, 1, ifelse(x5 >= 0, 0.75, 0.2))
zdup <- 1.5 * zlong - 0.5 * zb
x5_ties <- round(x5, 2)
covs2 <- cbind(zlong = zlong, zb = zb)
covs3 <- cbind(zlong = zlong, zb = zb, zdup = zdup)

golden$dgp_covs <- list(
  x = x5, y = y5, t = t5, zlong = zlong, zb = zb, zdup = zdup,
  x_ties = x5_ties,
  configs = list(
    covs_default        = run_estimate(y5, x5, covs = covs2),
    covs_manual_h       = run_estimate(y5, x5, covs = covs2, h = 0.2),
    covs_msetwo         = run_estimate(y5, x5, covs = covs2,
                                       bwselect = "msetwo"),
    covs_cercomb2       = run_estimate(y5, x5, covs = covs2,
                                       bwselect = "cercomb2"),
    covs_epa            = run_estimate(y5, x5, covs = covs2, kernel = "epa"),
    covs_drop_collinear = run_estimate(y5, x5, covs = covs3),
    covs_ties           = run_estimate(y5, x5_ties, covs = covs2),
    fuzzy_covs          = run_estimate(y5, x5, covs = covs2, fuzzy = t5),
    fuzzy_covs_sharpbw  = run_estimate(y5, x5, covs = covs2, fuzzy = t5,
                                       sharpbw = TRUE)
  )
)

senate_path <- "benchmarks/data/rdrobust_senate.csv"
stopifnot(file.exists(senate_path))
senate <- read.csv(senate_path)
ok <- complete.cases(senate$vote, senate$margin)
sv <- senate$vote[ok]
sm <- senate$margin[ok]

golden$senate <- list(
  csv = "benchmarks/data/rdrobust_senate.csv",
  configs = list(
    adjust = run_estimate(sv, sm, masspoints = "adjust"),
    off    = run_estimate(sv, sm, masspoints = "off")
  )
)

out_path <- "benchmarks/data/rdrobust_estimates_golden.json"
write_json(golden, out_path, auto_unbox = TRUE, pretty = TRUE, digits = I(17))
cat("Wrote", out_path, "\n")
