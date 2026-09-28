# Generate cross-language weighted-OLS parity fixture for HAD Phase 4.5.
#
# Purpose: no public weighted-CCF reference exists for bias-corrected
# local-linear (nprobust::lprobust has no weight argument; np::npreg uses
# its own internal local-linear algorithm that does not reduce to a
# straightforward weighted OLS at the intercept). To validate the weighted
# kernel-composition machinery in diff_diff._nprobust_port cross-language,
# we record the intercept from an R implementation of the SAME formula
# (weighted-OLS with one-sided Epanechnikov kernel) that the Python port
# implements. Bit-parity at atol=1e-14 locks in numerical consistency
# across BLAS reductions.
#
# This is NOT third-party validation of the weighted-CCF methodology. It is
# a regression lock against R↔Python drift on the weighted-OLS formula
# itself. Methodology confidence under informative weights comes from:
#   1. Analytic derivation in docs/methodology/REGISTRY.md
#   2. Uniform-weights bit-parity: weights=np.ones ≡ unweighted at 1e-14
#   3. Monte Carlo oracle consistency (tests/test_had_mc.py)
#
# Usage:
#   Rscript benchmarks/R/generate_np_npreg_weighted_golden.R
#
# Output:
#   benchmarks/data/np_npreg_weighted_golden.json
#
# Phase 4.5 of HeterogeneousAdoptionDiD (de Chaisemartin et al. 2026).
# Python test loader: tests/test_np_npreg_weighted_parity.py.

library(jsonlite)

# -------------------------------------------------------------------------
# Weighted local-linear at a boundary: manual weighted OLS with Epa kernel.
# Matches diff_diff/local_linear.py::local_linear_fit exactly.
# -------------------------------------------------------------------------

weighted_local_linear <- function(d, y, weights, eval_point = 0.0, h = 0.3) {
  # One-sided epanechnikov on [0, 1]: k(u) = (3/4)(1 - u^2), zero elsewhere.
  u <- (d - eval_point) / h
  kw <- ifelse(u >= 0 & u <= 1, 0.75 * (1 - u^2), 0)
  # Combined weights: user weights * kernel weights.
  combined <- kw * weights
  # Active window (non-zero combined weight).
  active <- combined > 0
  if (sum(active) < 2) {
    stop("Active window has fewer than 2 observations.")
  }
  # Weighted OLS of y ~ 1 + (d - eval_point), intercept is mu_hat at
  # eval_point.
  fit <- lm(y[active] ~ I(d[active] - eval_point), weights = combined[active])
  mu_hat <- as.numeric(coef(fit)[1])
  slope_hat <- as.numeric(coef(fit)[2])
  list(
    mu_hat = mu_hat,
    slope = slope_hat,
    n_active = as.integer(sum(active)),
    h = h,
    eval_point = eval_point
  )
}

# -------------------------------------------------------------------------
# DGPs: deterministic seeds for reproducibility.
# -------------------------------------------------------------------------

set.seed(20260424)

dgp1 <- local({
  G <- 500
  d <- runif(G, 0, 1)
  y <- 2 * d + 0.3 * d^2 + rnorm(G, sd = 0.25)
  w <- rep(1.0, G)
  list(d = d, y = y, w = w, eval_point = 0.0, h = 0.30,
       description = "Uniform weights, G=500, boundary=0")
})

dgp2 <- local({
  G <- 400
  d <- runif(G, 0, 1)
  y <- 2 * d + 0.3 * d^2 + rnorm(G, sd = 0.25)
  w <- exp(-d * 2.0)
  list(d = d, y = y, w = w, eval_point = 0.0, h = 0.25,
       description = "Informative weights (exp decay from boundary), G=400")
})

dgp3 <- local({
  G <- 200
  d <- runif(G, 0, 1)
  y <- 3 * d - d^2 + 0.5 * d^3 + rnorm(G, sd = 0.30)
  w <- pmax(0.1, runif(G, 0.5, 1.5))
  list(d = d, y = y, w = w, eval_point = 0.0, h = 0.35,
       description = "Small G=200, nonlinear m(d), bounded heterogeneous weights")
})

dgp4 <- local({
  G <- 400
  d_lower <- 0.1
  d <- runif(G, d_lower, 1)
  y <- 2 * (d - d_lower) + 0.3 * (d - d_lower)^2 + rnorm(G, sd = 0.25)
  w <- rep(1.0, G)
  list(d = d - d_lower, y = y, w = w, eval_point = 0.0, h = 0.30,
       description = "G=400, d_lower=0.1 shifted boundary=0 (Design 1 near-d_lower)",
       d_lower = d_lower)
})

# -------------------------------------------------------------------------
# Run each DGP through the weighted local-linear reference.
# -------------------------------------------------------------------------

run_one <- function(name, dgp) {
  cat(sprintf("Running %s: %s\n", name, dgp$description))
  res <- weighted_local_linear(
    d = dgp$d, y = dgp$y, weights = dgp$w,
    eval_point = dgp$eval_point, h = dgp$h
  )
  list(
    description = dgp$description,
    n = length(dgp$d),
    d = as.numeric(dgp$d),
    y = as.numeric(dgp$y),
    weights = as.numeric(dgp$w),
    eval_point = as.numeric(res$eval_point),
    h = as.numeric(res$h),
    kernel = "epanechnikov",
    n_active = res$n_active,
    mu_hat = as.numeric(res$mu_hat),
    slope = as.numeric(res$slope)
  )
}

out <- list(
  metadata = list(
    r_version = paste(R.Version()$major, R.Version()$minor, sep = "."),
    seed = 20260424L,
    generator = "generate_np_npreg_weighted_golden.R",
    algorithm = "manual weighted OLS with one-sided Epanechnikov kernel",
    purpose = "HAD Phase 4.5 cross-language weighted-LL parity",
    note = paste(
      "Regression lock on the weighted kernel + weighted OLS formula",
      "implemented in diff_diff.local_linear.local_linear_fit. Not a",
      "third-party validation of weighted-CCF methodology; see REGISTRY",
      "'Weighted extension (Phase 4.5)' for the parity-gap acknowledgement."
    )
  ),
  dgp1 = run_one("dgp1", dgp1),
  dgp2 = run_one("dgp2", dgp2),
  dgp3 = run_one("dgp3", dgp3),
  dgp4 = run_one("dgp4", dgp4)
)

out_path <- "benchmarks/data/np_npreg_weighted_golden.json"
dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)

write_json(out, out_path, auto_unbox = TRUE, pretty = TRUE, digits = 14)
cat(sprintf("Wrote %s\n", out_path))
