# Golden-value generator for the diff-diff RD PLOT port
# (diff_diff/rdplot.py::RDPlot). Parity target: CRAN rdrobust 4.0.0
# (tarball sha256
# 78f0d6b4bdec4091cc8f42f6f1598704747f95926446d3aaee381ea1d613a36f);
# the GitHub 4.1.0-dev tree must NOT be installed when regenerating.
#
# Outputs benchmarks/data/rdplot_golden.json: per config the input
# arguments, the selector outputs (J / J_IMSE / J_MV / rscale / bin_avg /
# bin_med), counts, descriptive strings, per-side global-fit coefficients,
# the FULL vars_bins table, and vars_poly at a fixed downsample of the
# 1000 curve points. DGP inputs are embedded at 17 significant digits.
#
# Configs stay inside the surface where R 4.0.0's rdplot() actually runs:
# fractional scale*J products and length-2 scale CRASH R (vector-indexing
# accident / vectorized-if error on R >= 4.2) and are covered by
# Deviation-locked Python unit tests instead of goldens.
#
# Run from the repo root: Rscript benchmarks/R/generate_rdplot_golden.R

library(rdrobust)
library(jsonlite)

stopifnot(packageVersion("rdrobust") == "4.0.0")

TARBALL_SHA256 <- "78f0d6b4bdec4091cc8f42f6f1598704747f95926446d3aaee381ea1d613a36f"

POLY_IDX <- sort(unique(c(seq(1, 1000, by = 25), 500L, 501L, 1000L)))

run_rdplot <- function(name, y, x, c = 0, p = 4, nbins = NULL,
                       binselect = "esmv", scale = NULL, kernel = "uni",
                       h = NULL, support = NULL, masspoints = "adjust",
                       ci = NULL, covs = NULL, covs_drop = TRUE) {
  args <- list(y = y, x = x, c = c, p = p, binselect = binselect,
               kernel = kernel, masspoints = masspoints, hide = TRUE,
               covs_drop = covs_drop)
  if (!is.null(nbins)) args$nbins <- nbins
  if (!is.null(scale)) args$scale <- scale
  if (!is.null(h)) args$h <- h
  if (!is.null(support)) args$support <- support
  if (!is.null(ci)) args$ci <- ci
  if (!is.null(covs)) args$covs <- covs
  r <- tryCatch(
    suppressWarnings(suppressMessages(do.call(rdplot, args))),
    error = function(e) stop(sprintf("config '%s' failed in R: %s",
                                     name, conditionMessage(e)))
  )
  vb <- r$vars_bins
  vp <- r$vars_poly
  out <- list(
    # input echoes (the Python test rebuilds RDPlot(...) from these)
    args = list(
      c = c, p = p,
      nbins = if (is.null(nbins)) NA else nbins,
      binselect = binselect,
      scale = if (is.null(scale)) NA else scale,
      kernel = kernel,
      h = if (is.null(h)) NA else h,
      support = if (is.null(support)) NA else support,
      masspoints = masspoints,
      ci = if (is.null(ci)) NA else ci,
      covs_in = !is.null(covs),
      covs_names = if (is.null(covs)) NA else colnames(covs),
      covs_drop = covs_drop
    ),
    J = unname(r$J), J_IMSE = unname(r$J_IMSE), J_MV = unname(r$J_MV),
    scale_out = unname(r$scale), rscale = unname(r$rscale),
    bin_avg = unname(r$bin_avg), bin_med = unname(r$bin_med),
    h_out = unname(r$h), N = unname(r$N), N_h = unname(r$N_h),
    binselect_type = r$binselect, kernel_type = r$kernel,
    coef_left = unname(r$coef[, 1]), coef_right = unname(r$coef[, 2]),
    vars_bins = list(
      mean_bin = vb$rdplot_mean_bin, mean_x = vb$rdplot_mean_x,
      mean_y = vb$rdplot_mean_y, min_bin = vb$rdplot_min_bin,
      max_bin = vb$rdplot_max_bin, se_y = vb$rdplot_se_y,
      N = vb$rdplot_N, ci_l = vb$rdplot_ci_l, ci_r = vb$rdplot_ci_r
    ),
    vars_poly_idx = POLY_IDX,
    vars_poly_x = vp$rdplot_x[POLY_IDX],
    vars_poly_y = vp$rdplot_y[POLY_IDX]
  )
  if (!is.null(covs)) out$coef_covs <- unname(as.vector(r$coef_covs))
  out
}

golden <- list()

golden$metadata <- list(
  rdrobust_version = as.character(packageVersion("rdrobust")),
  rdrobust_tarball_sha256 = TARBALL_SHA256,
  seeds = list(dgp_ties = 123L, dgp_covs = 2718L, dgp_small = 555L,
               dgp_ladder = 999L),
  generator = "benchmarks/R/generate_rdplot_golden.R",
  algorithm = paste(
    "rdplot() bin selectors (all 8 binselect values), manual",
    "nbins/scale/h/support/ci knobs, masspoints adjust remap, covariate",
    "adjustment, and the k=4->3->2 selector-fit fallback ladder; per",
    "config the full vars_bins table and downsampled vars_poly curve."
  ),
  r_version = R.version.string
)

## --- Senate data (vendored CSV; same complete-case filter as rdplot) ----
senate_path <- "benchmarks/data/rdrobust_senate.csv"
stopifnot(file.exists(senate_path))
senate <- read.csv(senate_path)
ok <- complete.cases(senate$vote, senate$margin)
sv <- senate$vote[ok]
sm <- senate$margin[ok]

golden$senate <- list(
  csv = "benchmarks/data/rdrobust_senate.csv",
  configs = list(
    default    = run_rdplot("default", sv, sm),
    es         = run_rdplot("es", sv, sm, binselect = "es"),
    espr       = run_rdplot("espr", sv, sm, binselect = "espr"),
    esmvpr     = run_rdplot("esmvpr", sv, sm, binselect = "esmvpr"),
    qs         = run_rdplot("qs", sv, sm, binselect = "qs"),
    qspr       = run_rdplot("qspr", sv, sm, binselect = "qspr"),
    qsmv       = run_rdplot("qsmv", sv, sm, binselect = "qsmv"),
    qsmvpr     = run_rdplot("qsmvpr", sv, sm, binselect = "qsmvpr"),
    p2         = run_rdplot("p2", sv, sm, p = 2),
    p0         = run_rdplot("p0", sv, sm, p = 0),
    p1_tri_h20 = run_rdplot("p1_tri_h20", sv, sm, p = 1, kernel = "tri",
                            h = 20),
    epa_h_asym = run_rdplot("epa_h_asym", sv, sm, kernel = "epa",
                            h = c(15, 25)),
    nbins_asym = run_rdplot("nbins_asym", sv, sm, nbins = c(10, 14)),
    scale2     = run_rdplot("scale2", sv, sm, scale = 2),
    support_w  = run_rdplot("support_w", sv, sm, support = c(-110, 110)),
    ci90       = run_rdplot("ci90", sv, sm, ci = 90),
    qs_nbins   = run_rdplot("qs_nbins", sv, sm, binselect = "qs", nbins = 8)
  )
)

## --- Tied running variable (masspoints machinery) -----------------------
# Same seeded recipe as generate_rdrobust_estimates_golden.R dgp_ties.
set.seed(123)
n2 <- 800
x2 <- round(2 * rbeta(n2, 2, 4) - 1, 2)
y2 <- 0.5 + 0.8 * x2 + (x2 >= 0) * 1.0 + rnorm(n2, sd = 0.3)

golden$dgp_ties <- list(
  x = x2, y = y2,
  configs = list(
    # default esmv under adjust remaps to esmvpr (rdplot.R:130-135)
    adjust       = run_rdplot("adjust", y2, x2, masspoints = "adjust"),
    off          = run_rdplot("off", y2, x2, masspoints = "off"),
    # explicit esmvpr with detection off must EQUAL the adjust config
    # (the Python suite asserts the cross-config identity)
    esmvpr_off   = run_rdplot("esmvpr_off", y2, x2, binselect = "esmvpr",
                              masspoints = "off")
  )
)

## --- Covariates (same seeded recipe as the estimates golden) ------------
set.seed(2718)
n5 <- 1200
x5 <- 2 * rbeta(n5, 2, 4) - 1
zlong <- 0.5 * x5 + rnorm(n5, sd = 0.8)
zb <- rbinom(n5, 1, 0.4)
y5 <- 0.4 * x5 + 0.9 * (x5 >= 0) + 0.7 * zlong + 0.3 * zb + rnorm(n5, sd = 0.3)
zdup <- 1.5 * zlong - 0.5 * zb
covs2 <- cbind(zlong = zlong, zb = zb)   # UNSORTED names: pins order(nchar)
covs3 <- cbind(zlong = zlong, zb = zb, zdup = zdup)

golden$dgp_covs <- list(
  x = x5, y = y5, zlong = zlong, zb = zb, zdup = zdup,
  configs = list(
    covs_default   = run_rdplot("covs_default", y5, x5, covs = covs2),
    covs_collinear = run_rdplot("covs_collinear", y5, x5, covs = covs3)
  )
)

## --- Small sample (n = 25; just above rdplot's n >= 20 hard floor) ------
set.seed(555)
n6 <- 25
x6 <- runif(n6, -1, 1)
y6 <- 0.3 + 0.5 * x6 + 0.8 * (x6 >= 0) + rnorm(n6, sd = 0.4)

golden$dgp_small <- list(
  x = x6, y = y6,
  configs = list(default = run_rdplot("small_default", y6, x6))
)

## --- Selector-fit fallback ladder (k = 4 -> 3) --------------------------
# |x| ~ 1e40 keeps every x^4 design entry finite but overflows the k=4
# Gram (entries ~x^8 -> Inf), so R's try(qrXXinv) errors and the ladder
# drops to k=3 (rdplot.R:281-298). Extreme magnitudes make float surfaces
# BLAS-sensitive, so the Python suite asserts only the integer/string
# outputs for this DGP.
set.seed(999)
n7 <- 120
x7 <- c(-runif(50, 1e39, 2e39), runif(70, 1e39, 2e39))
y7 <- rnorm(n7)

golden$dgp_ladder <- list(
  x = x7, y = y7,
  configs = list(default = run_rdplot("ladder_default", y7, x7))
)

out_path <- "benchmarks/data/rdplot_golden.json"
write_json(golden, out_path, auto_unbox = TRUE, pretty = TRUE, digits = I(17))
cat("Wrote", out_path, "\n")
