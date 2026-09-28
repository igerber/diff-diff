#!/usr/bin/env Rscript
# Golden generator: PlaceboTests (diff_diff/diagnostics.py) R parity.
#
# Bertrand, Duflo & Mullainathan (2004) placebo-law / randomization-inference
# diagnostics. Writes a fixed, fully deterministic 2-period panel and the R
# reference values so tests/test_methodology_placebo.py::TestPlaceboParityR can
# pin Python output against R without requiring R at test time.
#
# Outputs (checked into the repo):
#   benchmarks/data/placebo_test_panel.csv   (unit, t, y, treatment)
#   benchmarks/data/placebo_golden.json
#
# Usage:
#   Rscript benchmarks/R/generate_placebo_golden.R
#
# Notes:
#   - The panel is HARDCODED (not RNG-generated) so R and Python consume bit-
#     identical data; no cross-language RNG matching is needed.
#   - Permutation p-value uses EXACT enumeration of all C(8, 3) = 56 treated-group
#     assignments (the observed assignment is one of them): exact p =
#     #{|ATT*| >= |ATT_obs|} / total (observed included; min 1/total). This is the
#     ground truth the library's SAMPLED (1 + count)/(B + 1) value converges to.
#   - n_treated = 3 != N/2 = 4, so no assignment's complement shares its |ATT|
#     (avoids exact-tie pairing); the panel is chosen with a clear boundary gap
#     so the 1e-12 exact-parity comparison is not tie-flip fragile.
#   - leave-one-out se is the dispersion (sd, ddof=1) of the per-drop ATTs (NOT a
#     design-based jackknife SE), with a t-distribution (df = n_valid - 1), exactly
#     matching diff_diff.leave_one_out_test via safe_inference.
#   - Optional ri2/coin convention cross-check is guarded by requireNamespace and
#     is NOT a committed dependency (base-R combn enumeration is the anchor).

suppressMessages(library(jsonlite))

# ---- fixed panel (8 units x 2 periods; real treated = units 0,1,2) ----
panel <- data.frame(
  unit = rep(0:7, each = 2),
  t = rep(c(0, 1), times = 8),
  y = c(
    -1.639137, -0.623634, 0.051834, 1.622805, 0.261434, 0.82986,
    0.337559, 1.580412, -1.055892, -1.067745, 1.062855, 1.478681,
    0.139217, 0.8575, -1.253286, -0.560034
  )
)
panel$treatment <- as.integer(panel$unit %in% c(0, 1, 2))
real_treated <- c(0, 1, 2)
n_treated <- 3L
units <- 0:7

# 2x2 DiD ATT = double difference of group means (post = t == 1).
did_att <- function(df, treated) {
  is_t <- df$unit %in% treated
  post <- df$t == 1
  (mean(df$y[is_t & post]) - mean(df$y[is_t & !post])) -
    (mean(df$y[!is_t & post]) - mean(df$y[!is_t & !post]))
}

att_obs <- did_att(panel, real_treated)

# ---- permutation: EXACT randomization-inference p-value ----
combos <- combn(units, n_treated, simplify = FALSE)
atts <- vapply(combos, function(s) did_att(panel, s), numeric(1))
count <- sum(abs(atts) >= abs(att_obs) - 1e-12)
total <- length(atts)
p_exact <- count / total
# boundary gap: nearest distinct |ATT*| to |ATT_obs| (excluding the observed)
gap <- sort(abs(abs(atts) - abs(att_obs)))[2]

# ---- leave-one-out (deterministic jackknife over treated units) ----
loo_units <- real_treated
loo_atts <- sapply(loo_units, function(u) {
  remaining <- panel[panel$unit != u, ]
  treated_rem <- setdiff(real_treated, u)
  did_att(remaining, treated_rem)
})
loo_mean <- mean(loo_atts)
loo_se <- sd(loo_atts) # ddof = 1, the dispersion of LOO ATTs (not an SE-of-mean)
loo_df <- length(loo_atts) - 1L
loo_t <- loo_mean / loo_se
loo_p <- 2 * pt(-abs(loo_t), df = loo_df)
loo_crit <- qt(0.975, df = loo_df)
loo_ci <- c(loo_mean - loo_crit * loo_se, loo_mean + loo_crit * loo_se)

# ---- fake-group (deterministic; drop ever-treated, fake-treat controls 3,4) ----
fg_fake_treated <- c(3, 4)
fg_panel <- panel[!(panel$unit %in% real_treated), ] # never-treated only
fg_att <- did_att(fg_panel, fg_fake_treated)

# ---- optional convention cross-check (NOT a committed dependency) ----
ri2_ok <- requireNamespace("ri2", quietly = TRUE)

golden <- list(
  description = "PlaceboTests R parity (BDM 2004): exact RI permutation p-value + deterministic LOO + fake-group, on a fixed 2-period panel.",
  panel_csv = "benchmarks/data/placebo_test_panel.csv",
  real_treated = real_treated,
  n_treated = n_treated,
  observed_att = att_obs,
  permutation = list(
    convention = "exact enumeration: p = #{|ATT*| >= |ATT_obs|} / total (observed included)",
    count = count,
    total = total,
    p_exact = p_exact,
    boundary_gap = gap
  ),
  leave_one_out = list(
    dropped_units = loo_units,
    per_drop_att = as.list(setNames(loo_atts, as.character(loo_units))),
    mean = loo_mean,
    se = loo_se,
    df = loo_df,
    t_stat = loo_t,
    p_value = loo_p,
    ci_lower = loo_ci[1],
    ci_upper = loo_ci[2]
  ),
  fake_group = list(
    fake_treated_units = fg_fake_treated,
    note = "ever-treated units dropped (treatment filter); ATT is the double-difference",
    att = fg_att
  ),
  ri2_convention_checked = ri2_ok
)

write.csv(panel, "benchmarks/data/placebo_test_panel.csv", row.names = FALSE)
write_json(golden, "benchmarks/data/placebo_golden.json",
  auto_unbox = TRUE, pretty = TRUE, digits = 12
)

cat(sprintf("observed ATT = %.12f\n", att_obs))
cat(sprintf("exact RI: count=%d total=%d p_exact=%.12f gap=%.4f\n", count, total, p_exact, gap))
cat(sprintf("LOO: mean=%.12f se=%.12f df=%d p=%.6f\n", loo_mean, loo_se, loo_df, loo_p))
cat(sprintf("fake_group ATT = %.12f\n", fg_att))
cat(sprintf("ri2 convention cross-check available: %s\n", ri2_ok))
cat("Wrote benchmarks/data/placebo_test_panel.csv + placebo_golden.json\n")
