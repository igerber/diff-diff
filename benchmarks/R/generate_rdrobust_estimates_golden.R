# Golden-value generator for the diff-diff sharp-RD ESTIMATION port
# (diff_diff/_rdrobust_port.py::rdrobust_fit_sharp and the public
# RegressionDiscontinuity estimator).
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
                         level = 95, bwselect = "mserd") {
  args <- list(y = y, x = x, c = c, masspoints = masspoints, kernel = kernel,
               p = p, q = q, level = level, bwselect = bwselect)
  if (!is.null(h)) args$h <- h
  if (!is.null(b)) args$b <- b
  if (!is.null(rho)) args$rho <- rho
  r <- suppressWarnings(do.call(rdrobust, args))
  list(
    c = c, masspoints = masspoints, kernel = kernel, p = p, q = q,
    bwselect = bwselect,
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
}

golden <- list()

golden$metadata <- list(
  rdrobust_version = as.character(packageVersion("rdrobust")),
  rdrobust_tarball_sha256 = TARBALL_SHA256,
  seeds = list(dgp_lee_smooth = 42L, dgp_ties_moderate = 123L,
               dgp_asymmetric_scaled = 777L),
  generator = "benchmarks/R/generate_rdrobust_estimates_golden.R",
  algorithm = paste(
    "rdrobust() sharp-RD estimation blocks (three-row coef/se/z/pv/ci,",
    "counts, per-side beta_p) for the vce='nn' no-covariate path,",
    "complementing the bandwidth fixtures in rdrobust_golden.json."
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
