# Golden-value generator for the diff-diff RD DENSITY TEST port
# (diff_diff/rddensity.py::RDDensityTest). Parity target: CRAN rddensity 3.0
# (tarball sha256
# a9c45ab0f6b86ead4d91084db16513d4156b7f59b0472510b63deb5dee6f305d).
#
# Outputs benchmarks/data/rddensity_golden.json: per config the input
# arguments, the full rddensity() result surface (hat / sd_asy / sd_jk /
# test / N / h / bwselectl / masspoints flag) plus the standalone
# rdbwdensity() 4-row h-table (bw / variance / biassq). Synthetic DGP
# samples are embedded at 17 significant digits - R's RNG streams are not
# reproducible from numpy, so the Python tests rebuild every fit from the
# embedded arrays. The senate and headstart sections reference the vendored
# CSVs (benchmarks/data/rdrobust_senate.csv,
# benchmarks/data/rddensity_headstart.csv) instead.
#
# Floor-gate separation contract: the four nLocalMin/nUniqueMin configs must
# demonstrably exercise BOTH regularization gates - stopifnot assertions
# below abort generation loudly if a fixture fails to separate its gate, so
# every committed golden pins a binding gate (no unverifiable hand-written
# h values).
#
# Run from the repo root: Rscript benchmarks/R/generate_rddensity_golden.R

library(rddensity)
library(jsonlite)

stopifnot(packageVersion("rddensity") == "3.0")

TARBALL_SHA256 <- "a9c45ab0f6b86ead4d91084db16513d4156b7f59b0472510b63deb5dee6f305d"

run_rddensity <- function(name, x, c = 0, p = 2, q = 0,
                          fitselect = "unrestricted", kernel = "triangular",
                          vce = "jackknife", massPoints = TRUE, h = NULL,
                          bwselect = "comb", all = FALSE, regularize = TRUE,
                          nLocalMin = NULL, nUniqueMin = NULL) {
  args <- list(X = x, c = c, p = p, q = q, fitselect = fitselect,
               kernel = kernel, vce = vce, massPoints = massPoints,
               bwselect = bwselect, all = all, regularize = regularize,
               bino = FALSE)
  if (!is.null(h)) args$h <- h
  if (!is.null(nLocalMin)) args$nLocalMin <- nLocalMin
  if (!is.null(nUniqueMin)) args$nUniqueMin <- nUniqueMin
  r <- tryCatch(
    suppressWarnings(suppressMessages(do.call(rddensity, args))),
    error = function(e) stop(sprintf("config '%s' failed in R: %s",
                                     name, conditionMessage(e)))
  )
  out <- list(
    # input echoes (the Python test rebuilds RDDensityTest(...) from these)
    args = list(c = c, p = p, q = q, fitselect = fitselect, kernel = kernel,
                vce = vce, massPoints = massPoints, h = h,
                bwselect = bwselect, all = all, regularize = regularize,
                nLocalMin = nLocalMin, nUniqueMin = nUniqueMin),
    hat = r$hat, sd_asy = r$sd_asy, sd_jk = r$sd_jk, test = r$test,
    hat_p = r$hat_p, sd_asy_p = r$sd_asy_p, sd_jk_p = r$sd_jk_p,
    test_p = r$test_p,
    N = r$N, h = r$h,
    bwselectl = r$opt$bwselectl,
    masspoints_flag = r$opt$masspoints_flag
  )
  # the standalone bandwidth table for the same configuration (data-driven
  # configs only; manual-h configs skip it, as R's rddensity does)
  if (is.null(h)) {
    bw <- suppressWarnings(suppressMessages(rdbwdensity(
      X = x, c = c, p = p, fitselect = fitselect, kernel = kernel,
      vce = vce, massPoints = massPoints, regularize = regularize,
      nLocalMin = if (is.null(nLocalMin)) NULL else nLocalMin,
      nUniqueMin = if (is.null(nUniqueMin)) NULL else nUniqueMin
    )))
    out$bw_table <- list(bw = bw$h[, 1], variance = bw$h[, 2],
                         biassq = bw$h[, 3])
  }
  out
}

golden <- list()

## --- dgp_normal: tie-free bell curve; the primary option sweep --------------
set.seed(42)
x1 <- rnorm(2000, mean = -0.5)
golden$dgp_normal <- list(
  x = x1,
  configs = list(
    default = run_rddensity("default", x1),
    h_scalar = run_rddensity("h_scalar", x1, h = 0.6),
    h_pair = run_rddensity("h_pair", x1, h = c(0.5, 0.7)),
    bw_each = run_rddensity("bw_each", x1, bwselect = "each"),
    bw_diff = run_rddensity("bw_diff", x1, bwselect = "diff"),
    bw_sum = run_rddensity("bw_sum", x1, bwselect = "sum"),
    vce_plugin = run_rddensity("vce_plugin", x1, vce = "plugin"),
    p1 = run_rddensity("p1", x1, p = 1),
    p3 = run_rddensity("p3", x1, p = 3),
    p7 = run_rddensity("p7", x1, p = 7),
    q4 = run_rddensity("q4", x1, q = 4),
    q_equals_p = run_rddensity("q_equals_p", x1, q = 2),
    epanechnikov = run_rddensity("epanechnikov", x1, kernel = "epanechnikov"),
    uniform = run_rddensity("uniform", x1, kernel = "uniform"),
    all_true = run_rddensity("all_true", x1, all = TRUE),
    all_plugin = run_rddensity("all_plugin", x1, all = TRUE, vce = "plugin"),
    plugin_epa = run_rddensity("plugin_epa", x1, vce = "plugin",
                               kernel = "epanechnikov"),
    plugin_uni = run_rddensity("plugin_uni", x1, vce = "plugin",
                               kernel = "uniform"),
    plugin_p3 = run_rddensity("plugin_p3", x1, vce = "plugin", p = 3),
    no_regularize = run_rddensity("no_regularize", x1, regularize = FALSE)
  )
)

## --- dgp_disc: true density discontinuity (the Rd example) ------------------
set.seed(42)
x2 <- rnorm(2000, mean = -0.5)
x2[x2 > 0] <- x2[x2 > 0] * 2
golden$dgp_disc <- list(
  x = x2,
  configs = list(
    default = run_rddensity("disc_default", x2),
    restricted = run_rddensity("restricted", x2, fitselect = "restricted"),
    restricted_sum = run_rddensity("restricted_sum", x2,
                                   fitselect = "restricted", bwselect = "sum"),
    restricted_plugin = run_rddensity("restricted_plugin", x2,
                                      fitselect = "restricted",
                                      vce = "plugin"),
    restricted_all = run_rddensity("restricted_all", x2,
                                   fitselect = "restricted", all = TRUE),
    restricted_p1 = run_rddensity("restricted_p1", x2,
                                  fitselect = "restricted", p = 1)
  )
)
# restricted comb must genuinely pick min(diff, sum): on this fixture the
# p=1 restricted h-table selects the diff row (verified at generation)
stopifnot(abs(golden$dgp_disc$configs$restricted_p1$h$left -
              golden$dgp_disc$configs$restricted_p1$bw_table$bw[3]) < 1e-12)

## --- dgp_masspoints: rounded (tied) running variable ------------------------
set.seed(43)
x3 <- round(rnorm(2000, mean = -0.5), 1)
golden$dgp_masspoints <- list(
  x = x3,
  configs = list(
    default = run_rddensity("mp_default", x3),
    masspoints_off = run_rddensity("mp_off", x3, massPoints = FALSE),
    restricted = run_rddensity("mp_restricted", x3, fitselect = "restricted"),
    floors_50_50 = run_rddensity("floors_50_50", x3, nLocalMin = 50,
                                 nUniqueMin = 50),
    floors_0_0 = run_rddensity("floors_0_0", x3, nLocalMin = 0,
                               nUniqueMin = 0),
    floors_0_50 = run_rddensity("floors_0_50", x3, nLocalMin = 0,
                                nUniqueMin = 50),
    floors_50_0 = run_rddensity("floors_50_0", x3, nLocalMin = 50,
                                nUniqueMin = 0)
  )
)
# the nUniqueMin gate must bind on this fixture
stopifnot(abs(golden$dgp_masspoints$configs$floors_0_0$h$left -
              golden$dgp_masspoints$configs$floors_0_50$h$left) > 1e-10)

## --- dgp_masspoints_small: small tied fixture where nLocalMin binds ---------
set.seed(45)
x4 <- round(rnorm(150, mean = -0.5), 1)
golden$dgp_masspoints_small <- list(
  x = x4,
  configs = list(
    floors_50_50 = run_rddensity("s_floors_50_50", x4, nLocalMin = 50,
                                 nUniqueMin = 50),
    floors_0_0 = run_rddensity("s_floors_0_0", x4, nLocalMin = 0,
                               nUniqueMin = 0),
    floors_0_50 = run_rddensity("s_floors_0_50", x4, nLocalMin = 0,
                                nUniqueMin = 50),
    floors_50_0 = run_rddensity("s_floors_50_0", x4, nLocalMin = 50,
                                nUniqueMin = 0)
  )
)
# the nLocalMin gate must bind on this fixture (independently of nUniqueMin)
stopifnot(abs(golden$dgp_masspoints_small$configs$floors_0_0$h$left -
              golden$dgp_masspoints_small$configs$floors_50_0$h$left) > 1e-10)

## --- dgp_small: regularization floors bind ----------------------------------
set.seed(44)
x5 <- rnorm(120, mean = -0.5)
golden$dgp_small <- list(
  x = x5,
  configs = list(
    default = run_rddensity("small_default", x5)
  )
)

## --- senate: vendored real data (rdrobust senate CSV) -----------------------
senate <- read.csv("benchmarks/data/rdrobust_senate.csv")
xs <- senate$margin[!is.na(senate$margin)]
golden$senate <- list(
  csv = "benchmarks/data/rdrobust_senate.csv",
  column = "margin",
  configs = list(
    default = run_rddensity("senate_default", xs),
    plugin = run_rddensity("senate_plugin", xs, vce = "plugin"),
    restricted = run_rddensity("senate_restricted", xs,
                               fitselect = "restricted")
  )
)

## --- headstart: the CJM 2020 empirical application --------------------------
hs <- read.csv("benchmarks/data/rddensity_headstart.csv")
xh <- hs$povrate60[!is.na(hs$povrate60)]
golden$headstart <- list(
  csv = "benchmarks/data/rddensity_headstart.csv",
  column = "povrate60",
  cutoff = 59.1984,
  configs = list(
    p1_each = run_rddensity("hs_p1_each", xh, c = 59.1984, p = 1,
                            bwselect = "each"),
    p2_each = run_rddensity("hs_p2_each", xh, c = 59.1984, p = 2,
                            bwselect = "each"),
    p3_each = run_rddensity("hs_p3_each", xh, c = 59.1984, p = 3,
                            bwselect = "each"),
    p1_common = run_rddensity("hs_p1_common", xh, c = 59.1984, p = 1,
                              bwselect = "diff"),
    p2_common = run_rddensity("hs_p2_common", xh, c = 59.1984, p = 2,
                              bwselect = "diff"),
    p3_common = run_rddensity("hs_p3_common", xh, c = 59.1984, p = 3,
                              bwselect = "diff")
  )
)

golden$metadata <- list(
  package = "rddensity",
  version = as.character(packageVersion("rddensity")),
  tarball_sha256 = TARBALL_SHA256,
  seeds = list(dgp_normal = 42, dgp_disc = 42, dgp_masspoints = 43,
               dgp_masspoints_small = 45, dgp_small = 44),
  generator = "benchmarks/R/generate_rddensity_golden.R",
  headstart_csv_sha256 = "28f42a04ca7392e786e5f93ba311cdc91489a293c061a3bc59f53f4cfc536ce9",
  algorithm = paste(
    "rddensity() manipulation test (CJM 2020): local polynomial regression",
    "of the rank-based EDF, robust bias-corrected test at order q with",
    "order-p MSE-optimal bandwidths; rdbwdensity() h-table recorded for",
    "every data-driven config. bino windows disabled (out of port scope)."
  ),
  r_version = R.version.string
)

out_path <- "benchmarks/data/rddensity_golden.json"
write_json(golden, out_path, auto_unbox = TRUE, pretty = TRUE, digits = I(17),
           null = "null")
cat("Wrote", out_path, "\n")
