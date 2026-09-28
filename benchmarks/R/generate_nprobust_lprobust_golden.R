# Generate nprobust lprobust golden values for the Phase 1c parity suite.
#
# This script calls nprobust::lprobust() at a single eval point with
# bwselect="mse-dpi" on five deterministic DGPs and records:
#   - tau.cl, tau.bc (point estimates)
#   - se.cl, se.rb   (standard errors)
#   - h, b           (bandwidths chosen by the mse-dpi selector)
#   - N              (observations in the selected kernel window)
#   - z = qnorm(1 - alpha/2)
#   - ci_low, ci_high = tau.bc +/- z * se.rb
#
# DGPs 1-3 reuse the same seed + shape as benchmarks/R/generate_nprobust_golden.R
# so the selected (h, b) are identical; Phase 1c parity is therefore isolated
# to the point-estimate + variance computation. DGP 4 adds cluster IDs for
# cluster-robust SE parity; DGP 5 shifts the support to test a non-zero
# boundary (Design 1 continuous-near-d_lower).
#
# Usage:
#   Rscript benchmarks/R/generate_nprobust_lprobust_golden.R
#
# Requirements:
#   nprobust (CRAN), jsonlite
#
# Output:
#   benchmarks/data/nprobust_lprobust_golden.json
#
# Phase 1c of the HeterogeneousAdoptionDiD implementation (de Chaisemartin,
# Ciccia, D'Haultfoeuille & Knau 2026, arXiv:2405.04465v6). Python tests at
# tests/test_bias_corrected_lprobust.py and tests/test_nprobust_port.py load
# this JSON and check agreement to tiered tolerances (1e-14 on tau_cl/se_cl,
# 1e-12 on tau_bc/se_rb, 1e-13 on CI bounds; see Phase 1c plan).

library(nprobust)
library(jsonlite)

stopifnot(packageVersion("nprobust") == "0.5.0")

extract_lprobust_single_eval <- function(d, y, eval_point = 0.0,
                                          kernel = "epa", vce = "nn",
                                          cluster = NULL, alpha = 0.05,
                                          h = NULL, b = NULL) {
  # If h (and optionally b) are passed, bypass the mse-dpi selector and
  # call lprobust() with those bandwidths directly. This is used for
  # clustered DGPs where nprobust's internal lpbwselect.mse.dpi hits a
  # singleton-cluster shape bug in lprobust.vce during the order-q+1/q+2
  # pilot fits. For unclustered DGPs, h=NULL triggers bwselect="mse-dpi".
  if (is.null(h)) {
    fit <- lprobust(y = y, x = d, eval = eval_point,
                    p = 1L, deriv = 0L, kernel = kernel,
                    bwselect = "mse-dpi", vce = vce, cluster = cluster,
                    bwcheck = 21L, bwregul = 1, nnmatch = 3L)
  } else {
    # When b is unspecified, nprobust defaults to b = h / rho with rho=1.
    fit <- lprobust(y = y, x = d, eval = eval_point,
                    p = 1L, deriv = 0L, kernel = kernel,
                    h = h, b = if (is.null(b)) h else b,
                    vce = vce, cluster = cluster,
                    bwcheck = 21L, nnmatch = 3L)
  }
  est <- fit$Estimate[1, ]
  z   <- qnorm(1 - alpha / 2)
  ci_low  <- as.numeric(est["tau.bc"] - z * est["se.rb"])
  ci_high <- as.numeric(est["tau.bc"] + z * est["se.rb"])

  list(
    eval_point = as.numeric(eval_point),
    h          = as.numeric(est["h"]),
    b          = as.numeric(est["b"]),
    n_used     = as.integer(est["N"]),
    tau_cl     = as.numeric(est["tau.us"]),
    tau_bc     = as.numeric(est["tau.bc"]),
    se_cl      = as.numeric(est["se.us"]),
    se_rb      = as.numeric(est["se.rb"]),
    ci_low     = ci_low,
    ci_high    = ci_high,
    alpha      = as.numeric(alpha),
    z          = as.numeric(z)
  )
}

set.seed(20260419)

# DGP 1: d ~ Uniform(0, 1), y = d + d^2 + N(0, 0.5)
G <- 2000L
d1 <- runif(G, 0, 1)
y1 <- d1 + d1^2 + rnorm(G, 0, 0.5)

# DGP 2: d ~ Beta(2, 2), y = d + d^2 + N(0, 0.5)   (f(0) vanishes at boundary)
d2 <- rbeta(G, 2, 2)
y2 <- d2 + d2^2 + rnorm(G, 0, 0.5)

# DGP 3: Half-normal d, y = 0.5 * d^2 + N(0, 1)
d3 <- abs(rnorm(G, 0, 1))
y3 <- 0.5 * d3^2 + rnorm(G, 0, 1)

# DGP 4: Uniform(0, 1) with 50 clusters of 40 obs (cluster-robust SE parity).
# Fewer, larger clusters avoid an nprobust-internal singleton-cluster shape
# bug in lprobust.vce that fires if a kernel window retains only one obs per
# cluster. 50 clusters x 40 obs => the mse-dpi pilot windows near the
# boundary keep enough obs per cluster to stay well-conditioned.
set.seed(20260420)
G4 <- 2000L
d4 <- runif(G4, 0, 1)
cluster4 <- rep(1:50, each = 40)[1:G4]
# Introduce within-cluster correlation in y via a cluster effect.
cluster_effect <- rnorm(50, 0, 0.3)[cluster4]
y4 <- d4 + d4^2 + cluster_effect + rnorm(G4, 0, 0.3)

# DGP 5: Uniform(0.2, 1.0) — Design 1 continuous-near-d_lower at
# boundary = d.min() > 0. Different seed to avoid aliasing DGP 1.
set.seed(20260421)
G5 <- 2000L
d5 <- runif(G5, 0.2, 1.0)
y5 <- (d5 - 0.2) + (d5 - 0.2)^2 + rnorm(G5, 0, 0.5)
eval5 <- min(d5)  # Design 1 continuous: evaluate at the realized minimum.

golden <- list(
  metadata = list(
    nprobust_version = as.character(packageVersion("nprobust")),
    nprobust_sha = "36e4e532d2f7d23d4dc6e162575cca79e0927cda",
    seeds = list(dgp1 = 20260419L, dgp2 = 20260419L, dgp3 = 20260419L,
                 dgp4 = 20260420L, dgp5 = 20260421L),
    generator = "benchmarks/R/generate_nprobust_lprobust_golden.R",
    algorithm = paste("nprobust::lprobust(..., bwselect='mse-dpi') at a single",
                       "eval point, p=1, deriv=0, kernel='epa', vce='nn'",
                       "unless noted. The Python wrapper computes its own",
                       "z_{1-alpha/2} via scipy.stats.norm.ppf inside",
                       "safe_inference(); R's z is exported here for audit",
                       "so a reviewer can verify the two critical values",
                       "agree to machine precision.")
  ),
  dgp1 = c(list(n = G, d = d1, y = y1, kernel = "epa", vce = "nn",
                description = "Uniform(0,1), polynomial m(d) = d + d^2"),
           extract_lprobust_single_eval(d1, y1, kernel = "epa", vce = "nn")),
  dgp2 = c(list(n = G, d = d2, y = y2, kernel = "epa", vce = "nn",
                description = "Beta(2,2) - boundary density vanishes at 0"),
           extract_lprobust_single_eval(d2, y2, kernel = "epa", vce = "nn")),
  dgp3 = c(list(n = G, d = d3, y = y3, kernel = "epa", vce = "nn",
                description = "Half-normal d, quadratic m(d) with unit noise"),
           extract_lprobust_single_eval(d3, y3, kernel = "epa", vce = "nn")),
  dgp4 = c(list(n = G4, d = d4, y = y4, cluster = cluster4,
                kernel = "epa", vce = "nn",
                h_manual = 0.3, b_manual = 0.3,
                description = paste("Clustered (50 groups of 40) Uniform(0,1);",
                                     "manual h=b=0.3 to sidestep nprobust's",
                                     "singleton-cluster bug in the mse-dpi",
                                     "pilot fits.")),
           extract_lprobust_single_eval(d4, y4, kernel = "epa", vce = "nn",
                                         cluster = cluster4,
                                         h = 0.3, b = 0.3)),
  dgp5 = c(list(n = G5, d = d5, y = y5, eval_point_override = eval5,
                kernel = "epa", vce = "nn",
                description = "Uniform(0.2, 1.0), Design 1 boundary = d.min()"),
           extract_lprobust_single_eval(d5, y5, eval_point = eval5,
                                         kernel = "epa", vce = "nn"))
)

out_path <- "benchmarks/data/nprobust_lprobust_golden.json"
dir.create("benchmarks/data", recursive = TRUE, showWarnings = FALSE)
write_json(golden, out_path, auto_unbox = TRUE, pretty = TRUE, digits = 14)
cat("Golden values written to", out_path, "\n")
for (name in c("dgp1", "dgp2", "dgp3", "dgp4", "dgp5")) {
  cat(sprintf("%s: tau.bc = %.6f, se.rb = %.6f, h = %.6f, b = %.6f\n",
              name, golden[[name]]$tau_bc, golden[[name]]$se_rb,
              golden[[name]]$h, golden[[name]]$b))
}
