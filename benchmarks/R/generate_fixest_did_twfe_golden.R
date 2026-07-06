# Generate a committed fixest golden for the flagship 2x2 DiD and TWFE
# standard-error parity paths (SE-audit item G2).
#
# The core DiD / TWFE fixest SE parity was previously only checked by
# skip-guarded live-Rscript tests that never run in CI -- and those tests only
# assert `att` (rtol=1e-3), never the SE. This golden materializes deterministic
# panels and fixest's feols() ATT + SE so tests can assert machine-precision SE
# parity WITHOUT R at test time.
#
# Scope (this golden): the classical / iid SE, which Python matches to machine
# precision on both the 2x2 DiD path and the within-transform TWFE path (the
# latter also locks the SE-audit D4 full-K rescale). The cluster-robust ATT is
# stored too; its SE carries the documented CR1 small-sample DOF-convention
# difference vs fixest and is left to a follow-up. (`hetero`/HC1 collapses to iid
# on these balanced 2-group designs, so it is not a distinct target here.)
#
# Regenerate:  Rscript benchmarks/R/generate_fixest_did_twfe_golden.R
# Output:      benchmarks/data/fixest_did_twfe_golden.json
# Environment: R 4.5.2, fixest 0.14.2.

suppressMessages(library(fixest))
suppressMessages(library(jsonlite))

set.seed(12345)

fit_att <- function(model, vcov_spec) {
  ct <- coeftable(model, vcov = vcov_spec)
  idx <- which(rownames(ct) == "treated:post")
  ci <- confint(model, vcov = vcov_spec)[idx, ]
  list(
    att = unbox(ct[idx, "Estimate"]),
    se = unbox(ct[idx, "Std. Error"]),
    t_stat = unbox(ct[idx, "t value"]),
    p_value = unbox(ct[idx, "Pr(>|t|)"]),
    ci_lower = unbox(ci[[1]]),
    ci_upper = unbox(ci[[2]])
  )
}

# ---------------------------------------------------------------------------
# Scenario 1: 2x2 DiD (200 units x 2 periods), outcome ~ treated * post
# ---------------------------------------------------------------------------
n_units_did <- 200
did_rows <- list()
i <- 1
for (unit in 0:(n_units_did - 1)) {
  is_treated <- as.integer(unit < n_units_did %/% 2)
  for (period in c(0, 1)) {
    y <- 10.0 + period * 2.0
    if (is_treated == 1 && period == 1) y <- y + 3.0
    y <- y + rnorm(1, 0, 1)
    did_rows[[i]] <- data.frame(unit = unit, outcome = y, treated = is_treated, post = period)
    i <- i + 1
  }
}
did <- do.call(rbind, did_rows)
did_m <- feols(outcome ~ treated * post, data = did)

did_golden <- list(
  data = list(unit = did$unit, outcome = did$outcome, treated = did$treated, post = did$post),
  n_obs = unbox(nrow(did)),
  iid = fit_att(did_m, "iid"),
  cluster_unit = fit_att(did_m, ~unit)
)

# ---------------------------------------------------------------------------
# Scenario 2: TWFE (50 units x 4 periods), outcome ~ treated:post | unit + post
# ---------------------------------------------------------------------------
n_units_twfe <- 50
n_periods_twfe <- 4
twfe_rows <- list()
i <- 1
for (unit in 0:(n_units_twfe - 1)) {
  is_treated <- as.integer(unit < n_units_twfe %/% 2)
  unit_effect <- unit * 0.2
  for (period in 0:(n_periods_twfe - 1)) {
    post <- as.integer(period >= n_periods_twfe %/% 2)
    y <- 5.0 + unit_effect + period * 1.5
    if (is_treated == 1 && post == 1) y <- y + 2.5
    y <- y + rnorm(1, 0, 1)
    twfe_rows[[i]] <- data.frame(
      unit = unit, period = period, outcome = y, treated = is_treated, post = post
    )
    i <- i + 1
  }
}
twfe <- do.call(rbind, twfe_rows)
twfe_m <- feols(outcome ~ treated:post | unit + post, data = twfe)

twfe_golden <- list(
  data = list(
    unit = twfe$unit, period = twfe$period, outcome = twfe$outcome,
    treated = twfe$treated, post = twfe$post
  ),
  n_obs = unbox(nrow(twfe)),
  iid = fit_att(twfe_m, "iid"),
  cluster_unit = fit_att(twfe_m, ~unit)
)

# ---------------------------------------------------------------------------
golden <- list(
  meta = list(
    generator = unbox("benchmarks/R/generate_fixest_did_twfe_golden.R"),
    r_version = unbox(paste(R.version$major, R.version$minor, sep = ".")),
    fixest_version = unbox(as.character(packageVersion("fixest"))),
    description = unbox(paste(
      "fixest feols() ATT + SE golden for the flagship 2x2 DiD",
      "(outcome ~ treated*post) and TWFE (| unit + post) paths, SE-audit G2."
    ))
  ),
  did = did_golden,
  twfe = twfe_golden
)

out <- "benchmarks/data/fixest_did_twfe_golden.json"
writeLines(toJSON(golden, pretty = TRUE, digits = 16, auto_unbox = FALSE), out)
cat("Wrote", out, "\n")
