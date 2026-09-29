#!/usr/bin/env Rscript

# Generate one draw from Appendix E.1 of Wang et al. and estimate the
# short DiD OVB components using the correctly specified parametric nuisances.
# The CSV is intentionally shared with the Python runner for cross-language
# comparison; it contains the latent U only because this is a simulation.

suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
csv_path <- if (length(args) >= 1) args[[1]] else "benchmarks/data/did_ovb_simulation.csv"
json_path <- if (length(args) >= 2) args[[2]] else "benchmarks/data/did_ovb_simulation_r.json"
n <- if (length(args) >= 3) as.integer(args[[3]]) else 500L
p <- if (length(args) >= 4) as.numeric(args[[4]]) else 0.5
seed <- if (length(args) >= 5) as.integer(args[[5]]) else 20260928L

set.seed(seed)
d <- rbinom(n, 1L, p)
x <- rnorm(n, mean = ifelse(d == 0, 0.3, 0), sd = ifelse(d == 0, sqrt(6), sqrt(3)))
u <- rnorm(n, mean = ifelse(d == 0, 0.3, 0), sd = ifelse(d == 0, sqrt(6), sqrt(3)))
delta_y <- 1 + x + u + 2 * d + rnorm(n, sd = sqrt(2))

fold_ids <- (seq_len(n) - 1L) %% 10L
ps <- numeric(n)
m <- numeric(n)
for (fold in 0:9) {
  test <- fold_ids == fold
  train <- !test
  ps_fit <- glm(d ~ x + I(x^2), family = binomial(), subset = train)
  m_fit <- lm(delta_y ~ x, subset = train & d == 0)
  ps[test] <- predict(ps_fit, newdata = data.frame(x = x[test]), type = "response")
  m[test] <- predict(m_fit, newdata = data.frame(x = x[test]))
}

ps <- pmin(pmax(ps, 0.01), 0.99)
p_hat <- mean(d)
omega <- (ps / (1 - ps)) / (p_hat / (1 - p_hat))
residual <- delta_y - m
score <- (d / p_hat - (1 - d) / (1 - p_hat) * omega) * residual
short_att <- mean(score)
sigma2 <- mean(residual[d == 0]^2)
nu2 <- mean(omega[d == 0]^2)
scale <- sqrt(sigma2 * nu2)

write.csv(
  data.frame(unit = seq_len(n), pre = 0, post = delta_y, treated = d, x = x, u = u),
  csv_path,
  row.names = FALSE
)
write_json(
  list(
    n = n, p = p, seed = seed, n_folds = 10L,
    short_att = short_att, sigma2_control = sigma2,
    nu2_selection = nu2, scale = scale,
    treatment_effect = 2, omitted_bias = -mean(u[d == 1]) + mean(u[d == 0])
  ),
  json_path, auto_unbox = TRUE, digits = 17, pretty = TRUE
)
