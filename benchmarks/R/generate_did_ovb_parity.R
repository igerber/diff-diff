#!/usr/bin/env Rscript

# Independent parity oracle for the canonical DiD OVB implementation.
# The calculations below are written from the paper's displayed formulas;
# this script does not import or call dml.sensemakr.

suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
output_path <- if (length(args) >= 1) args[[1]] else "benchmarks/data/did_ovb_r_results.json"
panel_path <- if (length(args) >= 2) args[[2]] else "benchmarks/data/real/mpdta.csv"

zcrit <- qnorm(0.975)
data <- read.csv(panel_path, check.names = FALSE)
data <- data[data$year %in% c(2006, 2007) & data$`first.treat` %in% c(0, 2007), ]
data <- data[order(data$countyreal, data$year), ]

pre <- data[data$year == 2006, ]
post <- data[data$year == 2007, ]
stopifnot(nrow(pre) == nrow(post), all(pre$countyreal == post$countyreal))

d <- as.numeric(pre$`first.treat` == 2007)
dy <- post$lemp - pre$lemp
x <- pre$lpop
n <- length(d)
n_folds <- 2L
fold_ids <- (seq_len(n) - 1L) %% n_folds
p <- mean(d)

ps <- numeric(n)
m <- numeric(n)
for (fold in 0:(n_folds - 1L)) {
  test <- fold_ids == fold
  train <- !test
  ps_fit <- glm(d ~ x, family = binomial(), subset = train)
  m_fit <- lm(dy ~ x, subset = train & d == 0)
  ps[test] <- predict(ps_fit, newdata = data.frame(x = x[test]), type = "response")
  m[test] <- predict(m_fit, newdata = data.frame(x = x[test]))
}

ps <- pmin(pmax(ps, 0.01), 0.99)
omega <- (ps / (1 - ps)) / (p / (1 - p))
residual <- dy - m
score <- (d / p - (1 - d) / (1 - p) * omega) * residual
short_att <- mean(score)
sigma2 <- mean(residual[d == 0]^2)
nu2 <- mean(omega[d == 0]^2)
scale <- sqrt(sigma2 * nu2)
theta_if <- score - short_att
sigma_if <- (1 - d) / (1 - p) * (residual^2 - sigma2)
nu_if <- (1 - d) / (1 - p) * (omega^2 - nu2)
scale_if <- (nu2 * sigma_if + sigma2 * nu_if) / (2 * scale)
short_se <- sqrt(mean(theta_if^2) / n)

bounds <- function(trend_r2, selection_r2, rho_max = 1, alpha = 0.05) {
  multiplier <- rho_max * sqrt(trend_r2) * sqrt(selection_r2 / (1 - selection_r2))
  radius <- multiplier * scale
  lower_if <- theta_if - multiplier * scale_if
  upper_if <- theta_if + multiplier * scale_if
  lower_se <- sqrt(mean(lower_if^2) / n)
  upper_se <- sqrt(mean(upper_if^2) / n)
  z <- qnorm(1 - alpha / 2)
  list(
    lower = short_att - radius,
    upper = short_att + radius,
    radius = radius,
    lower_se = lower_se,
    upper_se = upper_se,
    lower_ci = short_att - radius - z * lower_se,
    upper_ci = short_att + radius + z * upper_se,
    trend_r2 = trend_r2,
    selection_r2 = selection_r2,
    rho_max = rho_max,
    alpha = alpha
  )
}

robustness <- function(null_value = 0, alpha = 0.05) {
  base <- bounds(0, 0, alpha = alpha)
  if (base$lower_ci <= null_value && null_value <= base$upper_ci) {
    return(list(rv = 0, xrv = 0, rv_bounds = base, xrv_bounds = base))
  }
  search <- function(kind) {
    contains <- function(s) {
      b <- if (kind == "rv") bounds(s, s, alpha = alpha) else bounds(1, s, alpha = alpha)
      b$lower_ci <= null_value && null_value <= b$upper_ci
    }
    lo <- 0
    hi <- 1 - 1e-12
    if (!contains(hi)) {
      b <- if (kind == "rv") bounds(hi, hi, alpha = alpha) else bounds(1, hi, alpha = alpha)
      return(list(value = NaN, bounds = b))
    }
    for (i in seq_len(80)) {
      mid <- (lo + hi) / 2
      if (contains(mid)) hi <- mid else lo <- mid
    }
    b <- if (kind == "rv") bounds(hi, hi, alpha = alpha) else bounds(1, hi, alpha = alpha)
    list(value = hi, bounds = b)
  }
  rv <- search("rv")
  xrv <- search("xrv")
  list(rv = rv$value, xrv = xrv$value, rv_bounds = rv$bounds, xrv_bounds = xrv$bounds)
}

result <- list(
  settings = list(
    panel = panel_path,
    periods = c(2006, 2007),
    treatment_cohort = 2007,
    n_folds = n_folds,
    fold_ids = fold_ids,
    pscore_trim = 0.01,
    alpha = 0.05,
    null_value = -0.1
  ),
  n_obs = n,
  n_treated = sum(d == 1),
  n_control = sum(d == 0),
  short_att = short_att,
  short_se = short_se,
  sigma2_control = sigma2,
  nu2_selection = nu2,
  scale = scale,
  bounds = bounds(1, 0.5),
  robustness = robustness(-0.1, 0.05)
)

write_json(result, output_path, auto_unbox = TRUE, digits = 17, pretty = TRUE)
