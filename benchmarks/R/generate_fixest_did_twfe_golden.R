# Generate a committed fixest golden for the flagship 2x2 DiD and TWFE
# standard-error parity paths (SE-audit item G2).
#
# The core DiD / TWFE fixest SE parity was previously only checked by
# skip-guarded live-Rscript tests that never run in CI -- and those tests only
# assert `att` (rtol=1e-3), never the SE. This golden materializes deterministic
# panels and fixest's feols() ATT + SE so tests can assert machine-precision SE
# parity WITHOUT R at test time.
#
# Scope (this golden): scenarios 1-2 are the original balanced designs — the
# classical / iid SE locks (the TWFE one also pins the SE-audit D4 full-K
# rescale) plus the cluster blocks. Scenarios 3-4 (G2 completion, 2026-07) are
# heteroskedastic + unbalanced so `hetero` (HC1) is a distinct target: the
# plain-OLS DiD path locks hetero AND cluster CR1 at machine precision (the
# CR1 DOF-convention difference vs fixest is absorbed-FE-only); the TWFE
# cluster SE stays band-pinned for that documented non-nested-FE ssc
# deviation, and TWFE hetero has no public unclustered Python surface
# (auto-cluster-at-unit convention), so scenario 4 locks iid on an
# UNBALANCED panel.
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
# Scenario 3: heteroskedastic + unbalanced 2x2 DiD (SE-audit G2 hetero lock).
# Error sd depends on treatment arm and period (so HC1 'hetero' does NOT
# collapse to iid) and ~15% of rows are dropped deterministically-by-draw
# (unbalanced groups). Appended AFTER scenarios 1-2 so their RNG draws (and
# the committed scenario 1-2 golden values) are unchanged on regeneration.
# ---------------------------------------------------------------------------
n_units_h <- 120
did_h_rows <- list()
i <- 1
for (unit in 0:(n_units_h - 1)) {
  is_treated <- as.integer(unit < 45)   # unequal arms: 45 treated / 75 control
  for (period in c(0, 1)) {
    sd_it <- 0.5 + 1.5 * is_treated + 0.8 * period   # heteroskedastic
    y <- 10.0 + period * 2.0
    if (is_treated == 1 && period == 1) y <- y + 3.0
    y <- y + rnorm(1, 0, sd_it)
    keep <- runif(1) > 0.15                          # unbalanced: drop ~15%
    if (keep) {
      did_h_rows[[i]] <- data.frame(unit = unit, outcome = y, treated = is_treated, post = period)
      i <- i + 1
    }
  }
}
did_h <- do.call(rbind, did_h_rows)
did_h_m <- feols(outcome ~ treated * post, data = did_h)

did_hetero_golden <- list(
  data = list(unit = did_h$unit, outcome = did_h$outcome, treated = did_h$treated, post = did_h$post),
  n_obs = unbox(nrow(did_h)),
  iid = fit_att(did_h_m, "iid"),
  hetero = fit_att(did_h_m, "hetero"),
  cluster_unit = fit_att(did_h_m, ~unit)
)

# ---------------------------------------------------------------------------
# Scenario 4: heteroskedastic + unbalanced TWFE. The fixest `hetero` block is
# stored for reference only — Python's TwoWayFixedEffects auto-clusters at
# unit on hc1 (no public unclustered-hetero surface), so the public locks are
# the unbalanced iid/full-K rescale plus the clustered ATT (exact) and SE
# (band, documented non-nested-FE ssc deviation).
# ---------------------------------------------------------------------------
n_units_th <- 40
n_periods_th <- 5
twfe_h_rows <- list()
i <- 1
for (unit in 0:(n_units_th - 1)) {
  is_treated <- as.integer(unit < 15)   # unequal arms
  unit_effect <- unit * 0.2
  for (period in 0:(n_periods_th - 1)) {
    post <- as.integer(period >= 3)
    sd_it <- 0.4 + 1.2 * is_treated + 0.3 * post
    y <- 5.0 + unit_effect + period * 1.5
    if (is_treated == 1 && post == 1) y <- y + 2.5
    y <- y + rnorm(1, 0, sd_it)
    keep <- runif(1) > 0.12
    if (keep) {
      twfe_h_rows[[i]] <- data.frame(
        unit = unit, period = period, outcome = y, treated = is_treated, post = post
      )
      i <- i + 1
    }
  }
}
twfe_h <- do.call(rbind, twfe_h_rows)
twfe_h_m <- feols(outcome ~ treated:post | unit + post, data = twfe_h)

twfe_hetero_golden <- list(
  data = list(
    unit = twfe_h$unit, period = twfe_h$period, outcome = twfe_h$outcome,
    treated = twfe_h$treated, post = twfe_h$post
  ),
  n_obs = unbox(nrow(twfe_h)),
  iid = fit_att(twfe_h_m, "iid"),
  hetero = fit_att(twfe_h_m, "hetero"),
  cluster_unit = fit_att(twfe_h_m, ~unit)
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
  twfe = twfe_golden,
  did_hetero = did_hetero_golden,
  twfe_hetero = twfe_hetero_golden
)

out <- "benchmarks/data/fixest_did_twfe_golden.json"
writeLines(toJSON(golden, pretty = TRUE, digits = 16, auto_unbox = FALSE), out)
cat("Wrote", out, "\n")
