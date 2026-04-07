#!/usr/bin/env Rscript
# Generate golden values for ported R `did` package tests.
#
# Usage:
#   Rscript benchmarks/R/generate_csdid_test_values.R
#
# Output:
#   benchmarks/data/csdid_golden_values.json
#
# Each scenario exports:
#   - data: the simulated dataset (so Python tests use identical data)
#   - params: att_gt() parameters used
#   - results: ATT(g,t), aggregations, SEs

library(did)
library(jsonlite)

cat("Generating CSDID golden values...\n")

output_path <- file.path("benchmarks", "data", "csdid_golden_values.json")

# Helper: extract group-time results from att_gt output
extract_gt <- function(res) {
  list(
    group = as.numeric(res$group),
    time = as.numeric(res$t),
    att = as.numeric(res$att),
    se = as.numeric(res$se)
  )
}

# Helper: extract aggregation results
extract_agg <- function(agg) {
  list(
    overall_att = agg$overall.att,
    overall_se = agg$overall.se,
    att = as.numeric(agg$att.egt),
    se = as.numeric(agg$se.egt),
    egt = as.numeric(agg$egt)
  )
}

# Helper: convert data to exportable list
export_data <- function(ddf) {
  list(
    unit = as.numeric(ddf$id),
    period = as.numeric(ddf$period),
    first_treat = as.numeric(ddf$G),
    outcome = as.numeric(ddf$Y),
    X = if ("X" %in% names(ddf)) as.numeric(ddf$X) else NULL,
    cluster = if ("cluster" %in% names(ddf)) as.numeric(ddf$cluster) else NULL
  )
}

scenarios <- list()

# Golden value datasets use n=500 to keep the JSON file small (~200KB).
# This is sufficient for exact-match tests (same data in, same results out).
N_GOLDEN <- 500

# ---------------------------------------------------------------------------
# Scenario 1: Default DGP, no covariates, DR
# ---------------------------------------------------------------------------
cat("  Scenario 1: default_no_covariates_dr\n")
set.seed(91420241)
time.periods <- 4
sp <- reset.sim(time.periods = time.periods, n = N_GOLDEN)
sp$bett <- sp$betu <- rep(0, time.periods)
data1 <- build_sim_dataset(sp)

res1 <- att_gt(yname = "Y", xformla = ~1, data = data1, tname = "period",
               idname = "id", gname = "G", est_method = "dr",
               bstrap = FALSE, cband = FALSE)
agg1_simple <- aggte(res1, type = "simple", bstrap = FALSE, cband = FALSE)

scenarios$default_no_covariates_dr <- list(
  data = export_data(data1),
  params = list(est_method = "dr", control_group = "nevertreated",
                xformla = "~1", base_period = "varying"),
  results = list(
    group_time = extract_gt(res1),
    simple = list(att = agg1_simple$overall.att, se = agg1_simple$overall.se)
  )
)

# ---------------------------------------------------------------------------
# Scenario 2: With covariates — DR, REG, IPW
# ---------------------------------------------------------------------------
cat("  Scenario 2: with_covariates\n")
set.seed(91420242)
sp2 <- reset.sim(n = N_GOLDEN)
sp2$ipw <- FALSE  # DR/REG compatible DGP
data2 <- build_sim_dataset(sp2)

res2_dr <- att_gt(yname = "Y", xformla = ~X, data = data2, tname = "period",
                  idname = "id", gname = "G", est_method = "dr",
                  bstrap = FALSE, cband = FALSE)
res2_reg <- att_gt(yname = "Y", xformla = ~X, data = data2, tname = "period",
                   idname = "id", gname = "G", est_method = "reg",
                   bstrap = FALSE, cband = FALSE)

# IPW needs its own DGP
set.seed(91420243)
sp2_ipw <- reset.sim(n = N_GOLDEN)
sp2_ipw$reg <- FALSE
data2_ipw <- build_sim_dataset(sp2_ipw)

res2_ipw <- att_gt(yname = "Y", xformla = ~X, data = data2_ipw, tname = "period",
                   idname = "id", gname = "G", est_method = "ipw",
                   bstrap = FALSE, cband = FALSE)

scenarios$with_covariates_dr <- list(
  data = export_data(data2),
  params = list(est_method = "dr", control_group = "nevertreated",
                xformla = "~X", base_period = "varying"),
  results = list(group_time = extract_gt(res2_dr))
)
scenarios$with_covariates_reg <- list(
  data = export_data(data2),
  params = list(est_method = "reg", control_group = "nevertreated",
                xformla = "~X", base_period = "varying"),
  results = list(group_time = extract_gt(res2_reg))
)
scenarios$with_covariates_ipw <- list(
  data = export_data(data2_ipw),
  params = list(est_method = "ipw", control_group = "nevertreated",
                xformla = "~X", base_period = "varying"),
  results = list(group_time = extract_gt(res2_ipw))
)

# ---------------------------------------------------------------------------
# Scenario 3: Two-period case with all aggregation types
# ---------------------------------------------------------------------------
cat("  Scenario 3: two_period\n")
set.seed(91420244)
sp3 <- reset.sim(time.periods = 2, n = N_GOLDEN * 2)
sp3$ipw <- FALSE
data3 <- build_sim_dataset(sp3)

res3 <- att_gt(yname = "Y", xformla = ~X, data = data3, tname = "period",
               idname = "id", gname = "G", est_method = "reg",
               bstrap = FALSE, cband = FALSE)
agg3_simple <- aggte(res3, type = "simple", bstrap = FALSE, cband = FALSE)
agg3_group <- aggte(res3, type = "group", bstrap = FALSE, cband = FALSE)
agg3_dynamic <- aggte(res3, type = "dynamic", bstrap = FALSE, cband = FALSE)

scenarios$two_period <- list(
  data = export_data(data3),
  params = list(est_method = "reg", control_group = "nevertreated",
                xformla = "~X", base_period = "varying"),
  results = list(
    group_time = extract_gt(res3),
    simple = list(att = agg3_simple$overall.att, se = agg3_simple$overall.se),
    group = extract_agg(agg3_group),
    dynamic = extract_agg(agg3_dynamic)
  )
)

# ---------------------------------------------------------------------------
# Scenario 4: Dynamic effects (te.e = 1:T)
# ---------------------------------------------------------------------------
cat("  Scenario 4: dynamic_effects\n")
set.seed(91420245)
time.periods <- 4
sp4 <- reset.sim(time.periods = time.periods, n = N_GOLDEN)
sp4$te <- 0
sp4$te.e <- 1:time.periods
data4 <- build_sim_dataset(sp4)

res4 <- att_gt(yname = "Y", xformla = ~X, data = data4, tname = "period",
               idname = "id", gname = "G", est_method = "reg",
               bstrap = FALSE, cband = FALSE)
agg4_dynamic <- aggte(res4, type = "dynamic", bstrap = FALSE, cband = FALSE)

scenarios$dynamic_effects <- list(
  data = export_data(data4),
  params = list(est_method = "reg", control_group = "nevertreated",
                xformla = "~X", base_period = "varying"),
  results = list(
    group_time = extract_gt(res4),
    dynamic = extract_agg(agg4_dynamic)
  )
)

# ---------------------------------------------------------------------------
# Scenario 5: Non-consecutive periods [1,2,5,7]
# ---------------------------------------------------------------------------
cat("  Scenario 5: non_consecutive_periods\n")
set.seed(91420246)
time.periods <- 8
sp5 <- reset.sim(time.periods = time.periods, n = N_GOLDEN)
sp5$te <- 0
sp5$te.e <- 1:time.periods
data5 <- build_sim_dataset(sp5)
keep.periods <- c(1, 2, 5, 7)
data5 <- subset(data5, G %in% c(0, keep.periods))
data5 <- subset(data5, period %in% keep.periods)

res5 <- att_gt(yname = "Y", xformla = ~X, data = data5, tname = "period",
               idname = "id", gname = "G", est_method = "reg",
               bstrap = FALSE, cband = FALSE)
agg5_dynamic <- aggte(res5, type = "dynamic", bstrap = FALSE, cband = FALSE)
agg5_balance <- aggte(res5, type = "dynamic", balance_e = 0,
                      bstrap = FALSE, cband = FALSE)

scenarios$non_consecutive_periods <- list(
  data = export_data(data5),
  params = list(est_method = "reg", control_group = "nevertreated",
                xformla = "~X", base_period = "varying"),
  results = list(
    group_time = extract_gt(res5),
    dynamic = extract_agg(agg5_dynamic),
    dynamic_balance_e0 = extract_agg(agg5_balance)
  )
)

# ---------------------------------------------------------------------------
# Scenario 6: Anticipation = 1
# ---------------------------------------------------------------------------
cat("  Scenario 6: anticipation\n")
set.seed(91420247)
time.periods <- 5
sp6 <- reset.sim(time.periods = time.periods, n = N_GOLDEN)
sp6$te <- 0
sp6$te.e <- -1:(time.periods - 2)
data6 <- build_sim_dataset(sp6)
data6$G <- ifelse(data6$G == 0, 0, data6$G + 1)
data6 <- subset(data6, G <= time.periods)

res6 <- att_gt(yname = "Y", xformla = ~X, data = data6, tname = "period",
               idname = "id", gname = "G", est_method = "dr",
               anticipation = 1, bstrap = FALSE, cband = FALSE)
agg6_dynamic <- aggte(res6, type = "dynamic", bstrap = FALSE, cband = FALSE)

scenarios$anticipation <- list(
  data = export_data(data6),
  params = list(est_method = "dr", control_group = "nevertreated",
                xformla = "~X", base_period = "varying", anticipation = 1),
  results = list(
    group_time = extract_gt(res6),
    dynamic = extract_agg(agg6_dynamic)
  )
)

# ---------------------------------------------------------------------------
# Scenario 7: Varying vs universal base period
# ---------------------------------------------------------------------------
cat("  Scenario 7: varying_vs_universal\n")
set.seed(91420248)
time.periods <- 8
sp7 <- reset.sim(time.periods = time.periods, n = N_GOLDEN)
sp7$te <- 0
sp7$te.e <- 1:time.periods
data7 <- build_sim_dataset(sp7)
data7 <- subset(data7, (G <= 5) | G == 0)
data7$G <- ifelse(data7$G == 0, 0, data7$G + 3)

res7_varying <- att_gt(yname = "Y", xformla = ~X, data = data7, tname = "period",
                       idname = "id", gname = "G", est_method = "dr",
                       base_period = "varying", bstrap = FALSE, cband = FALSE)
agg7_varying <- aggte(res7_varying, type = "dynamic", bstrap = FALSE, cband = FALSE)

res7_universal <- att_gt(yname = "Y", xformla = ~X, data = data7, tname = "period",
                         idname = "id", gname = "G", est_method = "dr",
                         base_period = "universal", bstrap = FALSE, cband = FALSE)
agg7_universal <- aggte(res7_universal, type = "dynamic", bstrap = FALSE, cband = FALSE)

scenarios$varying_vs_universal <- list(
  data = export_data(data7),
  params = list(est_method = "dr", control_group = "nevertreated", xformla = "~X"),
  results = list(
    varying_gt = extract_gt(res7_varying),
    varying_dynamic = extract_agg(agg7_varying),
    universal_gt = extract_gt(res7_universal),
    universal_dynamic = extract_agg(agg7_universal)
  )
)

# ---------------------------------------------------------------------------
# Scenario 8: Two-group DGP (ATT=3), 4 specs
# ---------------------------------------------------------------------------
cat("  Scenario 8: two_group\n")
set.seed(20241017)
n <- N_GOLDEN
p3_true <- exp(0.5) / (1 + exp(0.5))
g <- as.numeric(runif(n) <= p3_true)
g <- ifelse(g == 1, 3, 0)
nu <- rnorm(n, mean = g, sd = 1)
att_rand <- rnorm(n, mean = 3, sd = 1)

Yt1_inf <- 1 + nu + rnorm(n)
Yt2_inf <- 2 + nu + rnorm(n)
Yt3_inf <- 3 + nu + rnorm(n)
Yt4_inf <- 4 + nu + rnorm(n)
Yt1_g3 <- 1 + nu + rnorm(n)
Yt2_g3 <- 2 + nu + rnorm(n)
Yt3_g3 <- att_rand + 3 + nu + rnorm(n)
Yt4_g3 <- 1.5 * att_rand + 4 + nu + rnorm(n)

y1 <- (g == 3) * Yt1_g3 + (g == 0) * Yt1_inf
y2 <- (g == 3) * Yt2_g3 + (g == 0) * Yt2_inf
y3 <- (g == 3) * Yt3_g3 + (g == 0) * Yt3_inf
y4 <- (g == 3) * Yt4_g3 + (g == 0) * Yt4_inf

data8 <- data.frame(y1, y2, y3, y4, g)
data8$id <- 1:n
data8 <- tidyr::pivot_longer(data8, cols = starts_with("y"),
                             names_to = "t", names_prefix = "y", values_to = "y")
data8$t <- as.numeric(data8$t)
data8$g <- as.numeric(data8$g)

run_two_group <- function(data, cg, bp) {
  res <- att_gt(yname = "y", idname = "id", gname = "g", tname = "t",
                data = data, control_group = cg, panel = TRUE,
                xformla = ~1, bstrap = FALSE, cband = FALSE,
                est_method = "reg", base_period = bp)
  extract_gt(res)
}

scenarios$two_group <- list(
  data = list(
    unit = as.numeric(data8$id),
    period = as.numeric(data8$t),
    first_treat = as.numeric(data8$g),
    outcome = as.numeric(data8$y)
  ),
  results = list(
    nt_varying = run_two_group(data8, "nevertreated", "varying"),
    nt_universal = run_two_group(data8, "nevertreated", "universal"),
    nyt_varying = run_two_group(data8, "notyettreated", "varying"),
    nyt_universal = run_two_group(data8, "notyettreated", "universal")
  )
)

# ---------------------------------------------------------------------------
# Scenario 9: Fewer time periods than groups
# ---------------------------------------------------------------------------
cat("  Scenario 9: fewer_periods\n")
set.seed(91420249)
time.periods <- 6
sp9 <- reset.sim(time.periods = time.periods, n = N_GOLDEN)
sp9$te <- 0
sp9$te.e <- 1:time.periods
data9 <- build_sim_dataset(sp9)
data9 <- subset(data9, !(period %in% c(2, 5)))

res9 <- att_gt(yname = "Y", xformla = ~X, data = data9, tname = "period",
               idname = "id", gname = "G", est_method = "dr",
               bstrap = FALSE, cband = FALSE)
agg9_dynamic <- aggte(res9, type = "dynamic", bstrap = FALSE, cband = FALSE)
agg9_group <- aggte(res9, type = "group", bstrap = FALSE, cband = FALSE)

scenarios$fewer_periods <- list(
  data = export_data(data9),
  params = list(est_method = "dr", control_group = "nevertreated",
                xformla = "~X", base_period = "varying"),
  results = list(
    group_time = extract_gt(res9),
    dynamic = extract_agg(agg9_dynamic),
    group = extract_agg(agg9_group)
  )
)

# ---------------------------------------------------------------------------
# Scenario 10: Zero pre-treatment outcomes
# ---------------------------------------------------------------------------
cat("  Scenario 10: zero_pretreatment\n")
set.seed(914202410)
sp10 <- reset.sim(time.periods = 10, n = N_GOLDEN)
data10 <- build_sim_dataset(sp10)
data10 <- subset(data10, G != 0)
data10 <- subset(data10, G > 6)
data10 <- subset(data10, period > 5)
data10$Y[data10$period < data10$G] <- 0

res10_uni <- att_gt(yname = "Y", tname = "period", idname = "id", gname = "G",
                    data = data10, control_group = "notyettreated",
                    base_period = "universal", bstrap = FALSE, cband = FALSE)
res10_var <- att_gt(yname = "Y", tname = "period", idname = "id", gname = "G",
                    data = data10, control_group = "notyettreated",
                    base_period = "varying", bstrap = FALSE, cband = FALSE)

scenarios$zero_pretreatment <- list(
  data = export_data(data10),
  params = list(control_group = "notyettreated", xformla = "~1"),
  results = list(
    universal_gt = extract_gt(res10_uni),
    varying_gt = extract_gt(res10_var)
  )
)

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------
dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)
writeLines(toJSON(list(scenarios = scenarios), auto_unbox = TRUE, digits = 10,
                  pretty = TRUE),
           output_path)
cat(sprintf("Golden values written to %s\n", output_path))
cat(sprintf("File size: %.1f KB\n", file.info(output_path)$size / 1024))
