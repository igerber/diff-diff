#!/usr/bin/env Rscript
# Generate the cross-language golden fixture for StackedDiD's covariate-balancing
# (CBWSDID) path against the reference R package `cbwsdid` (Ustyuzhanin 2026).
#
# Unlike generate_stacked_did_golden.R (which operates on a PRE-stacked CSV so the
# R side is independent of Python stacking logic), `cbwsdid` does its OWN stacking
# + balancing, so this harness hands it the raw panel and dumps the dynamic
# event-study ATTs. The Python side (StackedDiD(balance="entropy", ...)) reproduces
# them via its independent entropy-balancing solver + effective-mass W_sa.
#
# Refinement: refinement.method="weightit", method="ebal" = entropy balancing
# (Hainmueller 2012) on covs.formula=~x, matching StackedDiD(balance="entropy",
# covariates=["x"]). Install: remotes::install_github("vadvu/cbwsdid").
#
# Usage: Rscript benchmarks/R/generate_cbwsdid_golden.R

suppressMessages({
  library(cbwsdid)
  library(jsonlite)
})

# Run from the repository root: Rscript benchmarks/R/generate_cbwsdid_golden.R
panel_csv <- "benchmarks/data/cbwsdid_balance_panel.csv"
out_json <- "benchmarks/data/cbwsdid_golden.json"

df <- read.csv(panel_csv)

m <- cbwsdid(
  data = df, y = "y", d = "d", id = c("unit", "time"),
  kappa = c(-2, 2), design = "absorbing", post_path = "stable",
  refinement.method = "weightit", covs.formula = ~x,
  refinement.args = list(method = "ebal"), pooled = TRUE
)
qoi <- cbwsdid_qoi(m, type = "dynamic")

golden <- list(
  meta = list(
    package = "cbwsdid",
    R_version = R.version.string,
    panel = "benchmarks/data/cbwsdid_balance_panel.csv",
    estimator = "cbwsdid(design='absorbing', refinement.method='weightit', method='ebal', covs.formula=~x)",
    kappa = c(-2L, 2L)
  ),
  dynamic = list(
    event_time = as.integer(qoi$et),
    estimate = as.numeric(qoi$estimate),
    std_error = as.numeric(qoi$std.error)
  )
)
write_json(golden, out_json, auto_unbox = TRUE, digits = 15, pretty = TRUE)
cat("wrote", out_json, "\n")
print(data.frame(et = qoi$et, estimate = qoi$estimate, se = qoi$std.error))
