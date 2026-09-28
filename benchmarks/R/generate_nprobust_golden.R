# Generate nprobust mse-dpi golden values for the Phase 1b parity suite.
#
# This script re-implements the mse-dpi algorithm from
# nprobust::lpbwselect.mse.dpi (nprobust 0.5.0, SHA 36e4e53, source at
# npfunctions.R:498-607 of github.com/nppackages/nprobust) so that every
# intermediate quantity the Python port needs to parity-check is exposed.
# Calling lpbwselect() directly would only surface h and b; we also need
# c.bw, bw.mp2, bw.mp3, and the per-stage (V, B1, B2, R) diagnostics.
# We therefore use nprobust::: internals via getFromNamespace().
#
# Usage:
#   Rscript benchmarks/R/generate_nprobust_golden.R
#
# Requirements:
#   nprobust (CRAN), jsonlite
#
# Output:
#   benchmarks/data/nprobust_mse_dpi_golden.json
#
# Phase 1b of the HeterogeneousAdoptionDiD implementation (de Chaisemartin,
# Ciccia, D'Haultfoeuille & Knau 2026, arXiv:2405.04465v6). Python tests at
# tests/test_bandwidth_selector.py and tests/test_nprobust_port.py load
# this JSON and check agreement to 1% relative tolerance.

library(nprobust)
library(jsonlite)

stopifnot(packageVersion("nprobust") == "0.5.0")

# Internal helper re-implementing lpbwselect.mse.dpi while returning every
# stage output.  Mirrors npfunctions.R:498-607 line-by-line (with the even
# / interior branches that apply for p=1, deriv=0, boundary eval).
lprobust_bw <- getFromNamespace("lprobust.bw", "nprobust")

extract_mse_dpi_stages <- function(d, y, kernel = "epa", eval_point = 0.0,
                                    bwcheck = 21, bwregul = 1, vce = "nn") {
  p <- 1L; deriv <- 0L; q <- p + 1L
  # For HAD (p=1, deriv=0), (p-deriv) %% 2 == 1, so `even` is FALSE.
  # nprobust's conditional: if (even==FALSE | interior==TRUE) -> use
  # closed-form lprobust.bw$bw; else -> optimize. For HAD this means
  # every stage bandwidth comes from the closed form, not optimize.
  even <- (p - deriv) %% 2 == 0
  interior <- FALSE

  N <- length(d)
  x_iq <- quantile(d, 0.75) - quantile(d, 0.25)
  x_min <- min(d); x_max <- max(d)
  range_ <- x_max - x_min

  C_c <- switch(kernel, "epa" = 2.34, "uni" = 1.843,
                "tri" = 2.576, "gau" = 1.06,
                stop("unknown kernel: ", kernel))

  c.bw <- C_c * min(sd(d), x_iq / 1.349) * N^(-1/5)
  bw.max <- max(abs(eval_point - x_min), abs(eval_point - x_max))
  c.bw <- min(c.bw, bw.max)

  # Nearest-neighbor precomputation when vce="nn" (npfunctions.R:518-529).
  dups <- dupsid <- NULL
  if (vce == "nn") {
    order_x <- order(d)
    d <- d[order_x]; y <- y[order_x]
    dups <- integer(N)
    for (j in 1:N) dups[j] <- sum(d == d[j])
    dupsid <- integer(N); j <- 1L
    while (j <= N) {
      dupsid[j:(j + dups[j] - 1L)] <- 1:dups[j]
      j <- j + dups[j]
    }
  }

  bw.min <- NULL
  if (!is.null(bwcheck)) {
    bw.min <- sort(abs(d - eval_point))[bwcheck]
    c.bw <- max(c.bw, bw.min)
  }

  # Helper: dispatch between closed-form and optimize exactly as R does.
  select_bw <- function(C, exp_bias, exp_var, scale, range_) {
    if (!even || interior) {
      return(C$bw)
    }
    fn <- function(H) {
      abs(H^exp_bias * (C$B1 + H * C$B2 + scale * C$R)^2 +
          C$V / (N * H^exp_var))
    }
    optimize(fn, interval = c(.Machine$double.eps, range_))$minimum
  }

  # Stage 2: C.d1 -> bw.mp2 (npfunctions.R:539-546)
  C_d1 <- lprobust_bw(y, d, NULL, eval_point, o = q + 1L, nu = q + 1L,
                       o.B = q + 2L, h.V = c.bw, h.B1 = range_,
                       h.B2 = range_, scale = 0, vce = vce, nnmatch = 3L,
                       kernel = kernel, dups = dups, dupsid = dupsid)
  bw.mp2 <- select_bw(C_d1,
                      exp_bias = 2 * (q + 1) + 2 - 2 * (q + 1),
                      exp_var = 1 + 2 * (q + 1),
                      scale = 0, range_ = range_)

  # Stage 2: C.d2 -> bw.mp3 (npfunctions.R:549-556)
  C_d2 <- lprobust_bw(y, d, NULL, eval_point, o = q + 2L, nu = q + 2L,
                       o.B = q + 3L, h.V = c.bw, h.B1 = range_,
                       h.B2 = range_, scale = 0, vce = vce, nnmatch = 3L,
                       kernel = kernel, dups = dups, dupsid = dupsid)
  bw.mp3 <- select_bw(C_d2,
                      exp_bias = 2 * (q + 2) + 2 - 2 * (q + 2),
                      exp_var = 1 + 2 * (q + 2),
                      scale = 0, range_ = range_)

  # Apply clipping (npfunctions.R:559-565)
  bw.mp2 <- min(bw.mp2, bw.max)
  bw.mp3 <- min(bw.mp3, bw.max)
  if (!is.null(bw.min)) {
    bw.mp2 <- max(bw.mp2, bw.min)
    bw.mp3 <- max(bw.mp3, bw.min)
  }

  # Stage 3: C.b -> b.mse.dpi (npfunctions.R:569-580)
  C_b <- lprobust_bw(y, d, NULL, eval_point, o = q, nu = p + 1L,
                     o.B = q + 1L, h.V = c.bw, h.B1 = bw.mp2,
                     h.B2 = bw.mp3, scale = bwregul, vce = vce,
                     nnmatch = 3L, kernel = kernel,
                     dups = dups, dupsid = dupsid)
  b.mse.dpi <- select_bw(C_b,
                         exp_bias = 2 * q + 2 - 2 * (p + 1),
                         exp_var = 1 + 2 * (p + 1),
                         scale = bwregul, range_ = range_)
  b.mse.dpi <- min(b.mse.dpi, bw.max)
  if (!is.null(bw.min)) b.mse.dpi <- max(b.mse.dpi, bw.min)

  # Stage 3 final: C.h -> h.mse.dpi (npfunctions.R:585-595)
  C_h <- lprobust_bw(y, d, NULL, eval_point, o = p, nu = deriv,
                     o.B = q, h.V = c.bw, h.B1 = b.mse.dpi,
                     h.B2 = bw.mp2, scale = bwregul, vce = vce,
                     nnmatch = 3L, kernel = kernel,
                     dups = dups, dupsid = dupsid)
  h.mse.dpi <- select_bw(C_h,
                         exp_bias = 2 * p + 2 - 2 * deriv,
                         exp_var = 1 + 2 * deriv,
                         scale = bwregul, range_ = range_)
  h.mse.dpi <- min(h.mse.dpi, bw.max)
  if (!is.null(bw.min)) h.mse.dpi <- max(h.mse.dpi, bw.min)

  stage_record <- function(C) {
    list(V = as.numeric(C$V), B1 = as.numeric(C$B1),
         B2 = as.numeric(C$B2), R = as.numeric(C$R),
         bw = as.numeric(C$bw))
  }

  list(
    c_bw = as.numeric(c.bw),
    bw_mp2 = as.numeric(bw.mp2),
    bw_mp3 = as.numeric(bw.mp3),
    b_mse_dpi = as.numeric(b.mse.dpi),
    h_mse_dpi = as.numeric(h.mse.dpi),
    bw_min = if (is.null(bw.min)) NA_real_ else as.numeric(bw.min),
    bw_max = as.numeric(bw.max),
    stage_d1 = stage_record(C_d1),
    stage_d2 = stage_record(C_d2),
    stage_b  = stage_record(C_b),
    stage_h  = stage_record(C_h)
  )
}

set.seed(20260419)

# DGP 1: d ~ Uniform(0, 1), y = d + d^2 + N(0, 0.5)
G <- 2000L
d1 <- runif(G, 0, 1)
y1 <- d1 + d1^2 + rnorm(G, 0, 0.5)

# DGP 2: d ~ Beta(2, 2), y = d + d^2 + N(0, 0.5) (f(0) vanishes at boundary)
d2 <- rbeta(G, 2, 2)
y2 <- d2 + d2^2 + rnorm(G, 0, 0.5)

# DGP 3: Half-normal d, y = 0.5 * d^2 + N(0, 1)
d3 <- abs(rnorm(G, 0, 1))
y3 <- 0.5 * d3^2 + rnorm(G, 0, 1)

golden <- list(
  metadata = list(
    nprobust_version = as.character(packageVersion("nprobust")),
    nprobust_sha = "36e4e532d2f7d23d4dc6e162575cca79e0927cda",
    seed = 20260419L,
    generator = "benchmarks/R/generate_nprobust_golden.R",
    algorithm = paste("Port of nprobust::lpbwselect.mse.dpi with all five",
                      "stage bandwidths plus per-stage (V, B1, B2, R).",
                      "Evaluation at boundary eval=0 for HAD use case",
                      "(p=1, deriv=0, interior=FALSE).")
  ),
  dgp1 = c(list(n = G, d = d1, y = y1, kernel = "epa",
                description = "Uniform(0,1), polynomial m(d) = d + d^2"),
           extract_mse_dpi_stages(d1, y1, kernel = "epa")),
  dgp2 = c(list(n = G, d = d2, y = y2, kernel = "epa",
                description = "Beta(2,2) - boundary density vanishes at 0"),
           extract_mse_dpi_stages(d2, y2, kernel = "epa")),
  dgp3 = c(list(n = G, d = d3, y = y3, kernel = "epa",
                description = "Half-normal d, quadratic m(d) with unit noise"),
           extract_mse_dpi_stages(d3, y3, kernel = "epa"))
)

out_path <- "benchmarks/data/nprobust_mse_dpi_golden.json"
dir.create("benchmarks/data", recursive = TRUE, showWarnings = FALSE)
write_json(golden, out_path, auto_unbox = TRUE, pretty = TRUE, digits = 14)
cat("Golden values written to", out_path, "\n")
cat("DGP 1 h.mse.dpi:", golden$dgp1$h_mse_dpi, "\n")
cat("DGP 2 h.mse.dpi:", golden$dgp2$h_mse_dpi, "\n")
cat("DGP 3 h.mse.dpi:", golden$dgp3$h_mse_dpi, "\n")
