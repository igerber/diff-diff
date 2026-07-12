# Golden-value generator for the diff-diff rdrobust bandwidth-selection port
# (diff_diff/_rdrobust_port.py, sharp-RD path).
#
# Parity target: CRAN rdrobust 4.0.0 (the installed release users get).
#   Source tarball sha256:
#   78f0d6b4bdec4091cc8f42f6f1598704747f95926446d3aaee381ea1d613a36f
# The unreleased GitHub 4.1.0-dev tree behaves differently (nn_tol NN
# tolerance, stdvars=TRUE default, no +1e-8 bwcheck floor) and must NOT be
# installed when regenerating this file.
#
# Outputs benchmarks/data/rdrobust_golden.json:
#   - metadata: package version, tarball sha256, seeds, generator path,
#     algorithm note.
#   - per config: embedded x/y inputs (17 significant digits - exact float64
#     round-trip), the full 10-selector bandwidth matrix from
#     rdbwselect(all=TRUE), and head intermediates (x_iq type-2 IQR, BWp,
#     M_l/M_r unique counts) replicated from rdbwselect.R:116-150 for drift
#     localization. Per-stage pilot internals are deliberately NOT pinned:
#     they are not exposed by rdbwselect, and the 10-selector matrix already
#     triangulates every stage (mserd/msesum/msetwo consume different
#     combinations of the same d/b/h pilot blocks).
#   - The Senate config reads the vendored benchmarks/data/rdrobust_senate.csv
#     (Cattaneo, Frandsen & Titiunik 2015; see benchmarks/R/README.md).
#
# Run from the repo root:  Rscript benchmarks/R/generate_rdrobust_golden.R

library(rdrobust)
library(jsonlite)

stopifnot(packageVersion("rdrobust") == "4.0.0")

TARBALL_SHA256 <- "78f0d6b4bdec4091cc8f42f6f1598704747f95926446d3aaee381ea1d613a36f"

# Head intermediates per rdbwselect.R:106-150 (sort, type-2 IQR, BWp, unique
# counts). Replicated here so the golden pins the pre-linalg chain head.
head_intermediates <- function(y, x, c) {
  ord <- order(x)
  x <- x[ord]; y <- y[ord]
  x_iq <- unname(quantile(x, .75, type = 2) - quantile(x, .25, type = 2))
  BWp <- min(c(sd(x), x_iq / 1.349))
  M_l <- length(unique(x[x < c]))
  M_r <- length(unique(x[x >= c]))
  list(x_iq = x_iq, BWp = BWp, M_l = M_l, M_r = M_r)
}

run_config <- function(y, x, c = 0, masspoints = "adjust", kernel = "tri",
                       p = 1, q = 2, deriv = 0, scaleregul = 1,
                       bwcheck = NULL, bwrestrict = TRUE, stdvars = FALSE,
                       nnmatch = 3) {
  args <- list(y = y, x = x, c = c, masspoints = masspoints, kernel = kernel,
               p = p, q = q, deriv = deriv, scaleregul = scaleregul,
               bwrestrict = bwrestrict, stdvars = stdvars, nnmatch = nnmatch,
               all = TRUE)
  if (!is.null(bwcheck)) args$bwcheck <- bwcheck
  b <- suppressWarnings(do.call(rdbwselect, args))
  sel <- rownames(b$bws)
  bws <- lapply(seq_along(sel), function(i) unname(b$bws[i, ]))
  names(bws) <- sel
  c(list(
    c = c, masspoints = masspoints, kernel = kernel, p = p, q = q,
    deriv = deriv, scaleregul = scaleregul,
    bwcheck = if (is.null(bwcheck)) NA else bwcheck,
    bwrestrict = bwrestrict, stdvars = stdvars, nnmatch = nnmatch,
    N = length(y), bws = bws
  ), head_intermediates(y, x, c))
}

golden <- list()

golden$metadata <- list(
  rdrobust_version = as.character(packageVersion("rdrobust")),
  rdrobust_tarball_sha256 = TARBALL_SHA256,
  seeds = list(dgp_lee_smooth = 42L, dgp_ties_moderate = 123L,
               dgp_asymmetric_scaled = 777L),
  generator = "benchmarks/R/generate_rdrobust_golden.R",
  algorithm = paste(
    "rdbwselect(all=TRUE) 10-selector bandwidth matrices for the sharp-RD",
    "path (vce='nn', no fuzzy/covs/cluster/weights), plus head",
    "intermediates (type-2 IQR, BWp, unique counts) per rdbwselect.R:116-150."
  ),
  r_version = R.version.string
)

# --- DGP 1: smooth Lee-2008-style CEF, continuous running variable --------
set.seed(42)
n <- 1000
x1 <- 2 * rbeta(n, 2, 4) - 1
y1 <- 0.48 + 1.27 * x1 + 7.18 * x1^2 + 20.21 * x1^3 + 21.54 * x1^4 +
  7.33 * x1^5 + 0.04 * (x1 >= 0) + rnorm(n, sd = 0.1295)

golden$dgp_lee_smooth <- list(
  description = "n=1000, x ~ 2*Beta(2,4)-1 (continuous), Lee-2008-style quintic CEF, jump 0.04 at 0",
  x = x1, y = y1,
  configs = list(
    default   = run_config(y1, x1),
    mp_off    = run_config(y1, x1, masspoints = "off"),
    epa       = run_config(y1, x1, kernel = "epa"),
    uni       = run_config(y1, x1, kernel = "uni"),
    p2q3      = run_config(y1, x1, p = 2, q = 3),
    noregul   = run_config(y1, x1, scaleregul = 0),
    deriv1    = run_config(y1, x1, deriv = 1, p = 2, q = 3),
    stdvars   = run_config(y1, x1, stdvars = TRUE),
    norestrict = run_config(y1, x1, bwrestrict = FALSE),
    nn2       = run_config(y1, x1, nnmatch = 2)
  )
)

# --- DGP 2: 2dp-rounded running variable (moderate mass points) ------------
# 1dp rounding crashes rdrobust under masspoints='check'/'off' (svd on too
# few unique support points); 2dp keeps all three modes runnable while the
# >=20% mass share still auto-injects bwcheck=10 under 'adjust'.
set.seed(123)
n2 <- 800
x2 <- round(2 * rbeta(n2, 2, 4) - 1, 2)
y2 <- 0.5 + 0.8 * x2 + (x2 >= 0) * 1.0 + rnorm(n2, sd = 0.3)

golden$dgp_ties_moderate <- list(
  description = "n=800, x rounded to 2dp (153 unique of 800), unit jump at 0",
  x = x2, y = y2,
  configs = list(
    adjust    = run_config(y2, x2, masspoints = "adjust"),
    check     = run_config(y2, x2, masspoints = "check"),
    off       = run_config(y2, x2, masspoints = "off"),
    bwcheck20 = run_config(y2, x2, masspoints = "adjust", bwcheck = 20L)
  )
)

# --- DGP 3: asymmetric density, scaled support, nonzero cutoff -------------
set.seed(777)
n3 <- 300
x3 <- 40 * rbeta(n3, 5, 2)
y3 <- 2 + 0.05 * x3 - 0.001 * x3^2 + 0.8 * (x3 >= 28) + rnorm(n3, sd = 0.5)

golden$dgp_asymmetric_scaled <- list(
  description = "n=300, x ~ 40*Beta(5,2) (support ~(0,40), right-skewed), cutoff 28, jump 0.8",
  x = x3, y = y3,
  configs = list(
    default = run_config(y3, x3, c = 28)
  )
)

# --- Senate real-data anchor (vendored CSV) --------------------------------
senate_path <- "benchmarks/data/rdrobust_senate.csv"
stopifnot(file.exists(senate_path))
senate <- read.csv(senate_path)
ok <- complete.cases(senate$vote, senate$margin)
sv <- senate$vote[ok]
sm <- senate$margin[ok]

golden$senate <- list(
  description = "U.S. Senate elections (Cattaneo-Frandsen-Titiunik 2015); vote on margin, cutoff 0; complete cases only (N=1297)",
  csv = "benchmarks/data/rdrobust_senate.csv",
  configs = list(
    adjust = run_config(sv, sm, masspoints = "adjust"),
    off    = run_config(sv, sm, masspoints = "off")
  )
)

out_path <- "benchmarks/data/rdrobust_golden.json"
write_json(golden, out_path, auto_unbox = TRUE, pretty = TRUE, digits = I(17))
cat("Wrote", out_path, "\n")
