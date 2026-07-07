#!/usr/bin/env Rscript
# Generate R HonestDiD parity goldens for diff-diff's Delta^SD optimal FLCI
# (`_compute_optimal_flci`, Rambachan & Roth 2023 Section 4.1). SE-audit B2b.
#
# Requires: R 4.4+, install.packages(c("HonestDiD", "jsonlite")).  HonestDiD 0.2.6.
# Run from the REPO ROOT: `Rscript benchmarks/R/generate_honest_flci_golden.R`
# (the output path below is resolved relative to the working directory).
# Output: benchmarks/data/honest_flci_golden.json
#
# Two tiers, both stored per case:
#   (1) override-R (PRIMARY): R's Monte-Carlo folded-normal quantile
#       `.qfoldednormal` (10^6 draws, seed 0) is replaced by an ANALYTICAL
#       quantile via assignInNamespace, so R solves the SAME deterministic outer
#       problem as diff-diff (whose `_cv_alpha` is analytical). diff-diff matches
#       the override-R center + half-length + optimalVec to ~1e-3 (median ~1e-5).
#   (2) stock-R (REALISM): the unmodified MC `.qfoldednormal`. Because the width
#       surface is near-flat at the optimum, R's MC noise perturbs R's own
#       hstar/center by up to a few e-3 -- diff-diff (analytical) is strictly more
#       accurate, so it matches stock R only to ~1.4e-2 (max observed). Looser tier.
suppressWarnings(suppressMessages({
  library(HonestDiD)
  library(jsonlite)
}))
stopifnot(as.character(packageVersion("HonestDiD")) == "0.2.6")

qfold_analytic <- function(p, mu = 0, sd = 1, ...) sapply(mu, function(m) {
  f <- function(x) (pnorm(x - m, sd = sd) - pnorm(-x - m, sd = sd)) - p
  uniroot(f, lower = 0, upper = abs(m) + sd * 12 + 5, tol = 1e-12)$root
})

# --- Curated stress grid (includes a curved [1,0,1,...] pre-trend, kink-prone
#     for the L1 inner objective) ------------------------------------------------
set.seed(20260707)
cases <- list()
for (npre in c(3, 6, 9)) {
  for (npost in c(1, 4)) {
    n <- npre + npost
    idx <- 0:(n - 1)
    for (rho in c(0.0, 0.5)) {
      corr <- rho^abs(outer(idx, idx, "-"))
      sigma <- corr * 0.01
      base_pre <- rep(c(1, 0), length.out = npre)
      beta <- c(base_pre, seq(1.2, 1.7, length.out = npost))
      lvecs <- list(c(1, rep(0, npost - 1)))
      if (npost > 1) lvecs <- c(lvecs, list(rep(1 / npost, npost)))
      for (l in lvecs) {
        for (M in c(0.0, 0.05, 0.1, 0.2, 0.5)) {
          cases[[length(cases) + 1]] <- list(
            beta = beta, sigma = sigma, num_pre = npre, num_post = npost,
            l_vec = l, M = M, alpha = 0.05
          )
        }
      }
    }
  }
}

run_flci <- function(c) {
  r <- suppressWarnings(findOptimalFLCI(
    betahat = c$beta, sigma = matrix(unlist(c$sigma), nrow = length(c$beta)),
    M = c$M, numPrePeriods = c$num_pre, numPostPeriods = c$num_post,
    l_vec = c$l_vec, alpha = c$alpha
  ))
  list(center = mean(r$FLCI), half_length = (r$FLCI[2] - r$FLCI[1]) / 2,
       optimal_vec = as.numeric(r$optimalVec))
}

# stock-R tier first (unmodified MC .qfoldednormal)
stock <- lapply(cases, function(c) tryCatch(run_flci(c)$center, error = function(e) NA))

# override-R tier (deterministic analytical quantile) -- PRIMARY
assignInNamespace(".qfoldednormal", qfold_analytic, "HonestDiD")
out <- vector("list", length(cases))
for (i in seq_along(cases)) {
  c <- cases[[i]]
  ov <- tryCatch(run_flci(c), error = function(e) NULL)
  out[[i]] <- list(
    beta = c$beta, sigma = c$sigma, num_pre = c$num_pre, num_post = c$num_post,
    l_vec = c$l_vec, M = c$M, alpha = c$alpha,
    center = if (is.null(ov)) NA else ov$center,
    half_length = if (is.null(ov)) NA else ov$half_length,
    optimal_vec = if (is.null(ov)) NA else ov$optimal_vec,
    stock_center = stock[[i]]
  )
}

golden <- list(
  meta = list(
    did_version = as.character(packageVersion("HonestDiD")),
    n_cases = length(cases),
    note = paste0(
      "Delta^SD optimal FLCI. `center`/`half_length`/`optimal_vec` are override-R ",
      "(analytical .qfoldednormal -> deterministic outer problem; diff-diff matches ",
      "to ~1e-3). `stock_center` is unmodified MC-R (diff-diff matches to ~1.4e-2; the ",
      "gap is R's simulation noise on the near-flat width surface)."
    )
  ),
  cases = out
)
out_path <- file.path("benchmarks", "data", "honest_flci_golden.json")
writeLines(toJSON(golden, auto_unbox = TRUE, digits = 14, pretty = TRUE), out_path)
cat("Wrote", out_path, "-", length(cases), "cases\n")
