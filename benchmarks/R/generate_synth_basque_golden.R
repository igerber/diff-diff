#!/usr/bin/env Rscript
# Generate the Basque Country (Abadie & Gardeazabal 2003) R `Synth` golden fixture
# for the SyntheticControl estimator's two-tier R-parity test.
#
# Run from the repo root:
#   Rscript benchmarks/R/generate_synth_basque_golden.R
#
# Writes (into tests/data/ so the deterministic Tier-1 parity test runs in
# isolated-install CI without R):
#   tests/data/synth_basque_panel.csv   verbatim Synth::basque, regions != 1
#                                        (Spain aggregate dropped), long format,
#                                        plus an absorbing `treated` indicator.
#   tests/data/synth_basque_golden.json  R Synth solution.v / solution.w, losses,
#                                        the standardization divisor, X1/X0, and
#                                        the treated/synthetic/gap paths.
#
# Provenance: the panel is a verbatim export of R `Synth::basque`; the V-selection
# numerics (standardization divisor, optimizer) are pinned from the `Synth` source,
# not from Abadie-Diamond-Hainmueller (2010) — see docs/methodology/REGISTRY.md.

suppressMessages({
  library(Synth)
  library(jsonlite)
})

data(basque)

predictors <- c(
  "school.illit", "school.prim", "school.med",
  "school.high", "school.post.high", "invest"
)
special <- list(
  list("gdpcap", 1960:1969, "mean"),
  list("sec.agriculture", seq(1961, 1969, 2), "mean"),
  list("sec.energy", seq(1961, 1969, 2), "mean"),
  list("sec.industry", seq(1961, 1969, 2), "mean"),
  list("sec.construction", seq(1961, 1969, 2), "mean"),
  list("sec.services.venta", seq(1961, 1969, 2), "mean"),
  list("sec.services.nonventa", seq(1961, 1969, 2), "mean"),
  list("popdens", 1969, "mean")
)
controls <- c(2:16, 18)

invisible(capture.output({
  dp <- dataprep(
    foo = basque,
    predictors = predictors,
    predictors.op = "mean",
    time.predictors.prior = 1964:1969,
    special.predictors = special,
    dependent = "gdpcap",
    unit.variable = "regionno",
    unit.names.variable = "regionname",
    time.variable = "year",
    treatment.identifier = 17,
    controls.identifier = controls,
    time.optimize.ssr = 1960:1969,
    time.plot = 1955:1997
  )
  so <- synth(dp)
}))

# Standardization divisor exactly as computed inside synth():
#   divisor <- sqrt(apply(cbind(X0, X1), 1, var))
big <- cbind(dp$X0, dp$X1)
divisor <- sqrt(apply(big, 1, var))

pred_names <- rownames(dp$X1)
v <- as.numeric(so$solution.v)
w <- as.numeric(so$solution.w)

# X0 as predictor -> {control -> value} so Python can verify matrix construction.
X0_list <- setNames(
  lapply(seq_len(nrow(dp$X0)), function(i) as.list(setNames(dp$X0[i, ], colnames(dp$X0)))),
  pred_names
)

synthetic_path <- as.numeric(dp$Y0plot %*% so$solution.w)
treated_path <- as.numeric(dp$Y1plot)
years <- as.integer(rownames(dp$Y1plot))

# --- Leave-one-out golden (ADH 2015 §4 donor robustness) ---------------------
# Drop the highest-weight donor (region 10, Cataluna) and re-fit with the
# ORIGINAL solution.v held fixed (custom.v), so the reduced-pool W-solve is
# deterministic and directly comparable to SyntheticControlResults.leave_one_out()
# on a v_method="custom" fit (which likewise reuses the original custom_v on the
# donor pool minus the dropped unit — specs/V are unchanged, only the donors shrink).
loo_drop <- 10L
controls_loo <- controls[controls != loo_drop]
invisible(capture.output({
  dp_loo <- dataprep(
    foo = basque,
    predictors = predictors,
    predictors.op = "mean",
    time.predictors.prior = 1964:1969,
    special.predictors = special,
    dependent = "gdpcap",
    unit.variable = "regionno",
    unit.names.variable = "regionname",
    time.variable = "year",
    treatment.identifier = 17,
    controls.identifier = controls_loo,
    time.optimize.ssr = 1960:1969,
    time.plot = 1955:1997
  )
  so_loo <- synth(dp_loo, custom.v = as.numeric(so$solution.v))
}))
w_loo <- as.numeric(so_loo$solution.w)
synthetic_path_loo <- as.numeric(dp_loo$Y0plot %*% so_loo$solution.w)
gap_loo <- as.numeric(dp_loo$Y1plot) - synthetic_path_loo
att_loo <- mean(gap_loo[years >= 1970])  # mean post-period gap (treatment year 1970)

golden <- list(
  config = list(
    treated_regionno = 17,
    controls = controls,
    treatment_year = 1970,
    predictors = predictors,
    predictors_op = "mean",
    predictor_window = 1964:1969,
    special = lapply(special, function(s) {
      list(var = s[[1]], periods = s[[2]], op = s[[3]])
    }),
    time_optimize_ssr = 1960:1969,
    time_plot = c(1955, 1997)
  ),
  predictor_names = pred_names,
  solution_v = setNames(v, pred_names),
  solution_w = as.list(setNames(w, colnames(dp$X0))),
  loss_v = as.numeric(so$loss.v),
  loss_w = as.numeric(so$loss.w),
  divisor = setNames(as.numeric(divisor), pred_names),
  X1 = setNames(as.numeric(dp$X1), pred_names),
  X0 = X0_list,
  years = years,
  treated_path = treated_path,
  synthetic_path = synthetic_path,
  gap = treated_path - synthetic_path,
  leave_one_out = list(
    dropped_regionno = loo_drop,
    solution_w = as.list(setNames(w_loo, colnames(dp_loo$X0))),
    att = att_loo,
    gap = gap_loo
  )
)

dir.create("tests/data", showWarnings = FALSE, recursive = TRUE)
write_json(
  golden, "tests/data/synth_basque_golden.json",
  auto_unbox = TRUE, digits = 12, pretty = TRUE
)

# Panel CSV: drop region 1 (Spain aggregate); long format + absorbing treated.
panel <- basque[basque$regionno != 1, ]
panel$treated <- as.integer(panel$regionno == 17 & panel$year >= 1970)
stopifnot(!any(is.na(panel$gdpcap)))  # outcome must be complete (balanced panel)
write.csv(panel, "tests/data/synth_basque_panel.csv", row.names = FALSE)

cat("Wrote tests/data/synth_basque_golden.json and synth_basque_panel.csv\n")
cat("nvarsV:", length(v), "  n_controls:", length(w), "\n")
cat("loss.v:", format(so$loss.v, digits = 6), "  loss.w:", format(so$loss.w, digits = 6), "\n")
nz <- setNames(round(w, 4), colnames(dp$X0))
cat("solution.w (nonzero):\n")
print(nz[nz > 1e-4])
