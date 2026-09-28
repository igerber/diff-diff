# Generate cross-language weighted-2SLS parity fixture for HAD Phase 4.5 B
# (mass-point + weights).
#
# Purpose: validate ``_fit_mass_point_2sls(..., weights=...)`` against
# ``estimatr::iv_robust(y ~ d | Z, weights=w, se_type=...)`` bit-exactly.
# estimatr's HC1 sandwich and Stata-style CR1 under pweights match the
# Wooldridge 2010 Ch. 12 / Angrist-Pischke 4.1.3 pweight convention
# that the Python port implements (w² in the HC1 meat, w·u in the CR1
# cluster score, weighted bread Z'WX). estimatr's classical SE uses a
# different DOF / projection convention and is skipped in parity tests
# (documented deviation; diverges by O(1/n) at non-uniform weights).
#
# Usage:
#   Rscript benchmarks/R/generate_estimatr_iv_robust_golden.R
#
# Output:
#   benchmarks/data/estimatr_iv_robust_golden.json
#
# Phase 4.5 B of HeterogeneousAdoptionDiD (de Chaisemartin et al. 2026).
# Python test loader: tests/test_estimatr_iv_robust_parity.py.

library(jsonlite)
library(estimatr)

stopifnot(packageVersion("estimatr") >= "1.0")

# -------------------------------------------------------------------------
# DGP builders
# -------------------------------------------------------------------------

dgp_mass_point <- function(n, seed, weight_pattern = "uniform",
                           include_cluster = FALSE, d_lower = 0.3) {
  # Mass-point dose: a fraction at d_lower, rest uniform(d_lower, 1).
  set.seed(seed)
  n_mass <- round(0.15 * n)
  d_mass <- rep(d_lower, n_mass)
  d_cont <- runif(n - n_mass, d_lower, 1.0)
  d <- c(d_mass, d_cont)
  # Reshuffle to avoid ordered-by-dose artifacts.
  perm <- sample.int(n)
  d <- d[perm]

  # True DGP: dy = 2 * d + 0.3 * d^2 + eps
  dy <- 2.0 * d + 0.3 * d^2 + rnorm(n, sd = 0.4)

  # Weights
  w <- switch(weight_pattern,
    "uniform" = rep(1.0, n),
    "mild" = 1.0 + 0.3 * rnorm(n),
    "informative" = 1.0 + 2.0 * abs(d - 0.5) + runif(n, 0, 0.2),
    "heavy_tailed" = pmax(0.1, rlnorm(n, meanlog = 0, sdlog = 0.5))
  )
  # Clip to positive.
  w <- pmax(w, 0.01)

  cluster <- if (include_cluster) sample.int(max(4, n %/% 20), n, replace = TRUE) else NULL

  list(d = d, dy = dy, w = w, cluster = cluster, d_lower = d_lower)
}

# -------------------------------------------------------------------------
# Fit weighted 2SLS with estimatr at specified se_type.
# -------------------------------------------------------------------------

fit_iv_robust <- function(dgp, se_type, use_cluster = FALSE) {
  d <- dgp$d
  dy <- dgp$dy
  w <- dgp$w
  Z <- as.integer(d > dgp$d_lower)
  df <- data.frame(d = d, dy = dy, Z = Z, w = w)
  if (use_cluster) df$cluster <- dgp$cluster

  fit <- if (use_cluster) {
    iv_robust(dy ~ d | Z, data = df, weights = w, clusters = cluster,
              se_type = se_type)
  } else {
    iv_robust(dy ~ d | Z, data = df, weights = w, se_type = se_type)
  }

  list(
    beta = as.numeric(coef(fit)["d"]),
    se = as.numeric(fit$std.error["d"]),
    # Intercept for manual sandwich verification.
    alpha = as.numeric(coef(fit)["(Intercept)"]),
    se_intercept = as.numeric(fit$std.error["(Intercept)"]),
    n = as.integer(nobs(fit)),
    se_type = se_type
  )
}

# -------------------------------------------------------------------------
# Build the DGP × se_type fixture grid.
# -------------------------------------------------------------------------

# Each DGP × se_type combination becomes one fixture entry. DGPs vary
# sample size, weight informativeness, and cluster structure so the
# Python test exercises all three sandwich variants (HC1, classical, CR1).
fixtures <- list()

dgps <- list(
  list(n = 200, seed = 42, weight = "uniform", cluster = FALSE, name = "uniform_n200"),
  list(n = 500, seed = 123, weight = "mild", cluster = FALSE, name = "mild_n500"),
  list(n = 500, seed = 7, weight = "informative", cluster = FALSE, name = "informative_n500"),
  list(n = 1000, seed = 321, weight = "heavy_tailed", cluster = FALSE, name = "heavy_n1000"),
  list(n = 600, seed = 99, weight = "informative", cluster = TRUE, name = "informative_cluster_n600")
)

# For the non-clustered DGPs, emit HC1 + classical entries (Python
# parity tests target HC1; classical deviates by O(1/n) and is recorded
# as a reference only). For the clustered DGP, emit the Stata-style CR1
# entry (matches `diff_diff/had.py::_fit_mass_point_2sls` CR1 convention
# bit-exactly; see Gate #0 verification in the Phase 4.5 B plan).
for (dgp_spec in dgps) {
  dgp <- dgp_mass_point(
    n = dgp_spec$n,
    seed = dgp_spec$seed,
    weight_pattern = dgp_spec$weight,
    include_cluster = dgp_spec$cluster
  )

  if (dgp_spec$cluster) {
    entry <- list(
      name = dgp_spec$name,
      n = dgp_spec$n,
      d_lower = dgp$d_lower,
      weight_pattern = dgp_spec$weight,
      seed = dgp_spec$seed,
      d = dgp$d,
      dy = dgp$dy,
      w = dgp$w,
      cluster = dgp$cluster,
      cr1 = fit_iv_robust(dgp, se_type = "stata", use_cluster = TRUE)
    )
  } else {
    entry <- list(
      name = dgp_spec$name,
      n = dgp_spec$n,
      d_lower = dgp$d_lower,
      weight_pattern = dgp_spec$weight,
      seed = dgp_spec$seed,
      d = dgp$d,
      dy = dgp$dy,
      w = dgp$w,
      cluster = NULL,
      hc1 = fit_iv_robust(dgp, se_type = "HC1", use_cluster = FALSE),
      classical = fit_iv_robust(dgp, se_type = "classical", use_cluster = FALSE)
    )
  }
  fixtures[[dgp_spec$name]] <- entry
}

# -------------------------------------------------------------------------
# Serialize
# -------------------------------------------------------------------------

out <- list(
  metadata = list(
    description = "estimatr::iv_robust weighted 2SLS parity fixture for HAD Phase 4.5 B",
    estimatr_version = as.character(packageVersion("estimatr")),
    r_version = as.character(getRversion()),
    n_dgps = length(fixtures),
    hc1_atol = 1e-10,
    cr1_atol = 1e-10,
    notes = paste(
      "HC1 (se_type='HC1') and CR1 (se_type='stata') under pweights match",
      "Python's _fit_mass_point_2sls bit-exactly at atol=1e-10. Classical",
      "(se_type='classical') uses estimatr's projection-form DOF convention",
      "(n-k + X_hat'WX_hat bread) which differs from Python's sandwich form",
      "(sum(w)-k + Z'W^2Z meat); included as a reference without parity",
      "assertion.",
      sep = " "
    )
  ),
  fixtures = fixtures
)

# Ensure output directory exists.
out_dir <- "benchmarks/data"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
out_path <- file.path(out_dir, "estimatr_iv_robust_golden.json")
write_json(out, path = out_path, digits = 17, auto_unbox = TRUE, null = "null")
message(sprintf("Wrote %d DGP fixtures to %s", length(fixtures), out_path))
