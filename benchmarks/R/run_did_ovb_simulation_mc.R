#!/usr/bin/env Rscript

# Monte Carlo runner for the Appendix E.1 correctly specified parametric DGP.
# This reports coverage for the short estimand theta_s = 1.7. It is a compact
# reproducibility diagnostic; the paper's full tables additionally use true
# bias factors and several sensitivity-statistic surfaces.

suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args) >= 1) args[[1]] else "benchmarks/data/did_ovb_simulation_mc_r.json"
n <- if (length(args) >= 2) as.integer(args[[2]]) else 500L
reps <- if (length(args) >= 3) as.integer(args[[3]]) else 20L
p <- if (length(args) >= 4) as.numeric(args[[4]]) else 0.5
seed <- if (length(args) >= 5) as.integer(args[[5]]) else 20260928L
alpha <- if (length(args) >= 6) as.numeric(args[[6]]) else 0.05
misspecified <- if (length(args) >= 7) as.logical(as.integer(args[[7]])) else FALSE

one_rep <- function(n, p, seed) {
  set.seed(seed)
  d <- rbinom(n, 1L, p)
  x <- rnorm(n, ifelse(d == 0, 0.3, 0), ifelse(d == 0, sqrt(6), sqrt(3)))
  u <- rnorm(n, ifelse(d == 0, 0.3, 0), ifelse(d == 0, sqrt(6), sqrt(3)))
  dy <- 1 + x + u + 2 * d + rnorm(n, sd = sqrt(2))
  folds <- (seq_len(n) - 1L) %% 10L
  x_fit <- if (misspecified) exp(x / 2) else x
  x_prop <- cbind(1, x_fit, x_fit^2)
  x_out <- cbind(1, x_fit)
  ps <- numeric(n); m <- numeric(n)
  for (fold in 0:9) {
    test <- folds == fold; train <- !test
    ps_fit <- glm.fit(x_prop[train, , drop = FALSE], d[train], family = binomial())
    m_fit <- lm.fit(x_out[train & d == 0, , drop = FALSE], dy[train & d == 0])
    ps[test] <- plogis(x_prop[test, , drop = FALSE] %*% ps_fit$coefficients)
    m[test] <- x_out[test, , drop = FALSE] %*% m_fit$coefficients
  }
  ps <- pmin(pmax(ps, 0.01), 0.99)
  phat <- mean(d); omega <- (ps / (1 - ps)) / (phat / (1 - phat))
  residual <- dy - m
  score <- (d / phat - (1 - d) / (1 - phat) * omega) * residual
  att <- mean(score); se <- sqrt(mean((score - att)^2) / n)
  sigma2 <- mean(residual[d == 0]^2); nu2 <- mean(omega[d == 0]^2)
  scale <- sqrt(sigma2 * nu2)
  theta_if <- score - att
  sigma_if <- (1 - d) / (1 - phat) * (residual^2 - sigma2)
  nu_if <- (1 - d) / (1 - phat) * (omega^2 - nu2)
  scale_if <- (nu2 * sigma_if + sigma2 * nu_if) / (2 * scale)
  z <- qnorm(1 - alpha / 2)
  contains <- function(strength, kind) {
    multiplier <- if (kind == "rv") strength / sqrt(1 - strength) else sqrt(strength / (1 - strength))
    radius <- multiplier * scale
    lower_if <- theta_if - multiplier * scale_if
    upper_if <- theta_if + multiplier * scale_if
    lower_se <- sqrt(mean(lower_if^2) / n); upper_se <- sqrt(mean(upper_if^2) / n)
    att - radius - z * lower_se <= 0 && 0 <= att + radius + z * upper_se
  }
  find_rv <- function(kind) {
    if (!contains(1 - 1e-12, kind)) return(NaN)
    lo <- 0; hi <- 1 - 1e-12
    for (i in seq_len(70)) {
      mid <- (lo + hi) / 2
      if (contains(mid, kind)) hi <- mid else lo <- mid
    }
    hi
  }
  c(att = att, se = se, cover = as.numeric(1.7 >= att - z * se && 1.7 <= att + z * se),
    rv = find_rv("rv"), xrv = find_rv("xrv"))
}

draws <- t(vapply(seq_len(reps), function(i) one_rep(n, p, seed + i - 1L), numeric(5)))
result <- list(
  n = n, reps = reps, p = p, alpha = alpha, misspecified = misspecified,
  seed = seed, true_short_att = 1.7,
  mean_att = mean(draws[, "att"]), sd_att = sd(draws[, "att"]),
  mean_se = mean(draws[, "se"]), bias = mean(draws[, "att"]) - 1.7,
  coverage = mean(draws[, "cover"]),
  mean_rv = mean(draws[, "rv"], na.rm = TRUE),
  mean_xrv = mean(draws[, "xrv"], na.rm = TRUE),
  rv_sd = sd(draws[, "rv"], na.rm = TRUE),
  xrv_sd = sd(draws[, "xrv"], na.rm = TRUE)
)
write_json(result, output, auto_unbox = TRUE, digits = 17, pretty = TRUE)
cat(toJSON(result, auto_unbox = TRUE, digits = 8, pretty = TRUE), "\n")
