#!/usr/bin/env Rscript

# R application oracle for Wang et al.'s minimum-wage example.
# This file uses ranger only as an executed reference implementation; no
# ranger source is copied into diff-diff.

suppressPackageStartupMessages({
  library(jsonlite)
  library(ranger)
})

args <- commandArgs(trailingOnly = TRUE)
input <- if (length(args) >= 1) args[[1]] else "/private/tmp/CS_RR/data/min_wage_CS.rds"
output <- if (length(args) >= 2) args[[2]] else "/private/tmp/did_ovb_minwage_r.json"
fold_output <- if (length(args) >= 3) args[[3]] else "/private/tmp/did_ovb_minwage_folds.csv"
seed <- if (length(args) >= 4) as.integer(args[[4]]) else 42L
n_folds <- if (length(args) >= 5) as.integer(args[[5]]) else 5L
num_trees <- if (length(args) >= 6) as.integer(args[[6]]) else 1000L
mtry <- if (length(args) >= 7) as.integer(args[[7]]) else 2L
min_node_size <- if (length(args) >= 8) as.integer(args[[8]]) else 10L
splitrule <- if (length(args) >= 9) args[[9]] else "variance"

if (grepl("\\.rds$", tolower(input))) {
  raw <- readRDS(input)
} else {
  raw <- read.csv(input, stringsAsFactors = TRUE, check.names = FALSE)
}

dat <- raw[raw$year %in% c(2006, 2007) & raw$first.treat %in% c(0, 2007), , drop = FALSE]
dat$treated <- as.integer(dat$first.treat == 2007)
dat <- dat[order(dat$countyreal, dat$year), , drop = FALSE]

units <- unique(dat$countyreal)
if (length(units) != 1961L || sum(dat$treated[dat$year == 2006]) != 584L) {
  stop("unexpected minimum-wage application sample")
}

wide <- dat[dat$year == 2006, c("countyreal", "treated", "region", "white", "hs", "pov", "lpop", "lmedinc"), drop = FALSE]
post <- dat[dat$year == 2007, c("countyreal", "lemp"), drop = FALSE]
pre <- dat[dat$year == 2006, c("countyreal", "lemp"), drop = FALSE]
wide <- merge(wide, pre, by = "countyreal", suffixes = c("", "_pre"), sort = FALSE)
wide <- merge(wide, post, by = "countyreal", suffixes = c("", "_post"), sort = FALSE)
wide$delta_y <- wide$lemp_post - wide$lemp

set.seed(seed)
folds <- integer(nrow(wide))
for (d in c(0L, 1L)) {
  idx <- which(wide$treated == d)
  folds[idx] <- sample(rep(0:(n_folds - 1L), length.out = length(idx)))
}
write.csv(data.frame(countyreal = wide$countyreal, fold_id = folds), fold_output, row.names = FALSE)

x_names <- c("region", "white", "hs", "pov", "lpop", "lmedinc")
x <- wide[, x_names, drop = FALSE]
# The Python public API currently accepts numeric covariates. Encode region by
# its public integer levels so both application runners consume identical X.
x$region <- as.numeric(x$region)
x <- as.matrix(x)
d <- wide$treated
y <- wide$delta_y
ps <- numeric(nrow(wide))
m <- numeric(nrow(wide))

for (fold in 0:(n_folds - 1L)) {
  test <- folds == fold
  train <- !test
  ps_fit <- ranger(
    x = as.data.frame(x[train, , drop = FALSE]),
    y = d[train],
    num.trees = num_trees,
    mtry = mtry,
    min.node.size = min_node_size,
    splitrule = splitrule,
    seed = seed + fold
  )
  out_fit <- ranger(
    x = as.data.frame(x[train & d == 0, , drop = FALSE]),
    y = y[train & d == 0],
    num.trees = num_trees,
    mtry = mtry,
    min.node.size = min_node_size,
    splitrule = splitrule,
    seed = seed + fold
  )
  ps[test] <- predict(ps_fit, data = as.data.frame(x[test, , drop = FALSE]))$predictions
  m[test] <- predict(out_fit, data = as.data.frame(x[test, , drop = FALSE]))$predictions
}

ps <- pmin(pmax(ps, 0.01), 0.99)
p <- mean(d)
omega <- (ps / (1 - ps)) / (p / (1 - p))
residual <- y - m
score <- (d / p - (1 - d) / (1 - p) * omega) * residual
short_att <- mean(score)
sigma2 <- mean(residual[d == 0]^2)
nu2 <- mean(omega[d == 0]^2)
scale <- sqrt(sigma2 * nu2)
theta_if <- score - short_att
short_se <- sqrt(mean(theta_if^2) / length(y))

# This is the same plug-in/IF root search used by the Python result object.
sigma_if <- (1 - d) / (1 - p) * (residual^2 - sigma2)
nu_if <- (1 - d) / (1 - p) * (omega^2 - nu2)
scale_if <- (nu2 * sigma_if + sigma2 * nu_if) / (2 * scale)
contains <- function(strength, kind, alpha = 0.05) {
  multiplier <- if (kind == "rv") strength / sqrt(1 - strength) else sqrt(strength / (1 - strength))
  radius <- multiplier * scale
  lo_if <- theta_if - multiplier * scale_if
  hi_if <- theta_if + multiplier * scale_if
  z <- qnorm(1 - alpha / 2)
  lo_se <- sqrt(mean(lo_if^2) / length(y))
  hi_se <- sqrt(mean(hi_if^2) / length(y))
  short_att - radius - z * lo_se <= 0 && 0 <= short_att + radius + z * hi_se
}
find_rv <- function(kind) {
  if (!contains(1 - 1e-12, kind)) return(NaN)
  lo <- 0
  hi <- 1 - 1e-12
  for (i in seq_len(70)) {
    mid <- (lo + hi) / 2
    if (contains(mid, kind)) hi <- mid else lo <- mid
  }
  hi
}

result <- list(
  n = nrow(wide), treated = sum(d), control = sum(d == 0), n_folds = n_folds,
  seed = seed, num_trees = num_trees, mtry = mtry,
  min_node_size = min_node_size, splitrule = splitrule,
  short_att = short_att, short_se = short_se, sigma2_control = sigma2,
  nu2_selection = nu2, scale = scale, rv = find_rv("rv"), xrv = find_rv("xrv")
)
write_json(result, output, auto_unbox = TRUE, digits = 17, pretty = TRUE)
cat(toJSON(result, auto_unbox = TRUE, digits = 8, pretty = TRUE), "\n")
