#!/usr/bin/env Rscript
# Benchmark: Real-Data Survey Cross-Validation — California API Dataset
#
# Uses the Academic Performance Index (API) dataset from R's survey package
# to generate golden values for diff-diff's survey variance validation.
#
# Policy context: California's Public Schools Accountability Act (PSAA, 1999)
# created the API score system and the Governor's Performance Award (GPA)
# program, which awarded $150/pupil to schools meeting growth targets.
# The `awards` variable indicates GPA eligibility (real program assignment).
#
# DiD setup: Treatment = awards=="Yes", Pre = api99, Post = api00
# NOTE: This exercises real survey design variables for numerical validation.
# The ATT estimate has limited causal interpretation (not a clean experiment).
#
# Suite A: 7 scenarios testing TSL variance, FPC, subpopulations, covariates,
#          and Fay's BRR replicate weights.
#
# Usage:
#   Rscript benchmark_realdata_api.R
#
# References:
#   Lumley, T. (2004). "Analysis of Complex Survey Samples." JSS 9(8).
#   California Department of Education. Public Schools Accountability Act (1999).

library(survey)
library(jsonlite)

cat("=== Real-Data Survey Benchmark: California API Dataset ===\n\n")

# --- Output directory ---
output_dir <- file.path(dirname(dirname(getwd())), "benchmarks", "data", "real")
if (!dir.exists(output_dir)) {
  output_dir <- "benchmarks/data/real"
}
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

results <- list()

# ---------------------------------------------------------------------------
# Load API dataset and construct DiD panel
# ---------------------------------------------------------------------------

data(api)  # loads apistrat, apiclus1, apiclus2, aipop

cat(sprintf("  apistrat: %d schools, %d variables\n", nrow(apistrat), ncol(apistrat)))
cat(sprintf("  School types: E=%d, M=%d, H=%d\n",
            sum(apistrat$stype == "E"),
            sum(apistrat$stype == "M"),
            sum(apistrat$stype == "H")))
cat(sprintf("  Awards: Yes=%d, No=%d\n",
            sum(apistrat$awards == "Yes"),
            sum(apistrat$awards == "No")))

# Construct 2-period panel: each school appears twice
# Period 0: outcome = api99 (baseline, 1999)
# Period 1: outcome = api00 (post-accountability, 2000)
n_schools <- nrow(apistrat)

panel <- data.frame(
  school_id = rep(1:n_schools, each = 2),
  period = rep(c(0L, 1L), times = n_schools),
  post = rep(c(0L, 1L), times = n_schools),
  treated = rep(as.integer(apistrat$awards == "Yes"), each = 2),
  outcome = c(rbind(apistrat$api99, apistrat$api00)),
  stype = rep(as.character(apistrat$stype), each = 2),
  pw = rep(apistrat$pw, each = 2),
  fpc = rep(apistrat$fpc, each = 2),
  dnum = rep(apistrat$dnum, each = 2),
  meals = rep(apistrat$meals, each = 2),
  ell = rep(apistrat$ell, each = 2)
)

# Drop rows with missing outcome (api99 or api00 might be NA)
panel <- panel[complete.cases(panel[, c("outcome", "treated", "post", "pw")]), ]

cat(sprintf("  Panel: %d rows (%d schools x 2 periods)\n",
            nrow(panel), nrow(panel) / 2))

# ---------------------------------------------------------------------------
# Helper: extract svyglm results for the interaction term
# ---------------------------------------------------------------------------

extract_svyglm_results <- function(model, design, coef_name = "treated:post") {
  coefs <- coef(model)
  ses <- SE(model)
  idx <- which(names(coefs) == coef_name)

  att <- as.numeric(coefs[idx])
  se <- as.numeric(ses[idx])
  t_stat <- att / se
  df <- degf(design)

  # Manual CI using t-distribution with survey df
  t_crit <- qt(0.975, df)
  ci_lower <- att - t_crit * se
  ci_upper <- att + t_crit * se

  list(
    att = att,
    se = se,
    t_stat = t_stat,
    df = df,
    ci_lower = ci_lower,
    ci_upper = ci_upper
  )
}

# ---------------------------------------------------------------------------
# Suite A: 7 test scenarios
# ---------------------------------------------------------------------------

cat("\n--- Suite A: API Real-Data Scenarios ---\n")

# A1: Full design — strata + FPC + weights
cat("  A1: strata + FPC + weights ...\n")
design_a1 <- svydesign(
  id = ~1, strata = ~stype, weights = ~pw, fpc = ~fpc, data = panel
)
model_a1 <- svyglm(outcome ~ treated * post, design = design_a1)
est_a1 <- extract_svyglm_results(model_a1, design_a1)
cat(sprintf("    ATT = %.6f, SE = %.6f, df = %d\n", est_a1$att, est_a1$se, est_a1$df))

results[["a1_strata_fpc_weights"]] <- list(
  scenario = "A1",
  description = "Stratified by stype, FPC, probability weights",
  design_type = "strata_fpc_weights",
  estimator = "survey::svyglm",
  att = est_a1$att,
  se = est_a1$se,
  t_stat = est_a1$t_stat,
  df = est_a1$df,
  ci_lower = est_a1$ci_lower,
  ci_upper = est_a1$ci_upper,
  n_obs = nrow(panel),
  n_strata = length(unique(panel$stype)),
  sum_weights = sum(panel$pw)
)

# A2: Strata + weights, no FPC
cat("  A2: strata + weights (no FPC) ...\n")
design_a2 <- svydesign(
  id = ~1, strata = ~stype, weights = ~pw, data = panel
)
model_a2 <- svyglm(outcome ~ treated * post, design = design_a2)
est_a2 <- extract_svyglm_results(model_a2, design_a2)
cat(sprintf("    ATT = %.6f, SE = %.6f, df = %d\n", est_a2$att, est_a2$se, est_a2$df))

results[["a2_strata_weights"]] <- list(
  scenario = "A2",
  description = "Stratified by stype, no FPC, probability weights",
  design_type = "strata_weights",
  estimator = "survey::svyglm",
  att = est_a2$att,
  se = est_a2$se,
  t_stat = est_a2$t_stat,
  df = est_a2$df,
  ci_lower = est_a2$ci_lower,
  ci_upper = est_a2$ci_upper,
  n_obs = nrow(panel),
  n_strata = length(unique(panel$stype)),
  sum_weights = sum(panel$pw)
)

# A3: Weights only (no strata, no FPC)
cat("  A3: weights only ...\n")
design_a3 <- svydesign(
  id = ~1, weights = ~pw, data = panel
)
model_a3 <- svyglm(outcome ~ treated * post, design = design_a3)
est_a3 <- extract_svyglm_results(model_a3, design_a3)
cat(sprintf("    ATT = %.6f, SE = %.6f, df = %d\n", est_a3$att, est_a3$se, est_a3$df))

results[["a3_weights_only"]] <- list(
  scenario = "A3",
  description = "Weights only, no strata or FPC",
  design_type = "weights_only",
  estimator = "survey::svyglm",
  att = est_a3$att,
  se = est_a3$se,
  t_stat = est_a3$t_stat,
  df = est_a3$df,
  ci_lower = est_a3$ci_lower,
  ci_upper = est_a3$ci_upper,
  n_obs = nrow(panel),
  sum_weights = sum(panel$pw)
)

# A4: TWFE — same full design, but verify TWFE regression matches
cat("  A4: TWFE (strata + FPC + weights) ...\n")
# TWFE: include school and period fixed effects explicitly
# For a 2x2 DiD, svyglm(outcome ~ treated * post) is equivalent to TWFE
# when the data is balanced. We still run it to validate the Python TWFE path.
design_a4 <- svydesign(
  id = ~1, strata = ~stype, weights = ~pw, fpc = ~fpc, data = panel
)
model_a4 <- svyglm(outcome ~ treated * post, design = design_a4)
est_a4 <- extract_svyglm_results(model_a4, design_a4)
cat(sprintf("    ATT = %.6f, SE = %.6f, df = %d\n", est_a4$att, est_a4$se, est_a4$df))

results[["a4_twfe"]] <- list(
  scenario = "A4",
  description = "TWFE plumbing — same design as A1, validates Python TWFE path",
  design_type = "strata_fpc_weights",
  estimator = "survey::svyglm (TWFE)",
  att = est_a4$att,
  se = est_a4$se,
  t_stat = est_a4$t_stat,
  df = est_a4$df,
  ci_lower = est_a4$ci_lower,
  ci_upper = est_a4$ci_upper,
  n_obs = nrow(panel),
  n_strata = length(unique(panel$stype)),
  sum_weights = sum(panel$pw)
)

# A5: Subpopulation — elementary schools only
cat("  A5: subpopulation (elementary schools) ...\n")
design_a5_full <- svydesign(
  id = ~1, strata = ~stype, weights = ~pw, fpc = ~fpc, data = panel
)
design_a5 <- subset(design_a5_full, stype == "E")
model_a5 <- svyglm(outcome ~ treated * post, design = design_a5)
est_a5 <- extract_svyglm_results(model_a5, design_a5)
cat(sprintf("    ATT = %.6f, SE = %.6f, df = %d\n", est_a5$att, est_a5$se, est_a5$df))

n_elem <- sum(panel$stype == "E")
results[["a5_subpop_elementary"]] <- list(
  scenario = "A5",
  description = "Subpopulation: elementary schools only (stype='E')",
  design_type = "strata_fpc_weights_subpop",
  estimator = "survey::svyglm",
  subpopulation = "stype == 'E'",
  att = est_a5$att,
  se = est_a5$se,
  t_stat = est_a5$t_stat,
  df = est_a5$df,
  ci_lower = est_a5$ci_lower,
  ci_upper = est_a5$ci_upper,
  n_obs_subpop = n_elem,
  n_obs_full = nrow(panel),
  n_strata = length(unique(panel$stype)),
  sum_weights = sum(panel$pw)
)

# A6: Covariates — meals + ell
cat("  A6: strata + FPC + covariates (meals, ell) ...\n")
# Drop rows with missing covariate values
panel_cov <- panel[complete.cases(panel[, c("meals", "ell")]), ]
design_a6 <- svydesign(
  id = ~1, strata = ~stype, weights = ~pw, fpc = ~fpc, data = panel_cov
)
model_a6 <- svyglm(outcome ~ treated * post + meals + ell, design = design_a6)
est_a6 <- extract_svyglm_results(model_a6, design_a6)
cat(sprintf("    ATT = %.6f, SE = %.6f, df = %d\n", est_a6$att, est_a6$se, est_a6$df))

results[["a6_covariates"]] <- list(
  scenario = "A6",
  description = "Stratified + FPC + covariates (meals, ell)",
  design_type = "strata_fpc_weights_covariates",
  estimator = "survey::svyglm",
  has_covariates = TRUE,
  covariates = c("meals", "ell"),
  att = est_a6$att,
  se = est_a6$se,
  t_stat = est_a6$t_stat,
  df = est_a6$df,
  ci_lower = est_a6$ci_lower,
  ci_upper = est_a6$ci_upper,
  n_obs = nrow(panel_cov),
  n_strata = length(unique(panel_cov$stype)),
  sum_weights = sum(panel_cov$pw)
)

# A7: Fay's BRR replicate weights (generated from stratified design)
cat("  A7: Fay's BRR replicates (rho=0.3) ...\n")
design_a7_base <- svydesign(
  id = ~1, strata = ~stype, weights = ~pw, fpc = ~fpc, data = panel
)
# Convert to Fay's BRR replicate design
design_a7 <- as.svrepdesign(design_a7_base, type = "Fay", fay.rho = 0.3)
model_a7 <- svyglm(outcome ~ treated * post, design = design_a7)
est_a7 <- extract_svyglm_results(model_a7, design_a7)

# Extract replicate weight info
rep_weights_a7 <- weights(design_a7, type = "analysis")
n_replicates_a7 <- ncol(rep_weights_a7)
cat(sprintf("    ATT = %.6f, SE = %.6f, df = %d, n_replicates = %d\n",
            est_a7$att, est_a7$se, est_a7$df, n_replicates_a7))

# Export replicate weights for Python-side reconstruction
rep_weight_export <- list()
for (r in 1:n_replicates_a7) {
  rep_weight_export[[paste0("rep_", r - 1)]] <- as.list(rep_weights_a7[, r])
}

results[["a7_fay_brr"]] <- list(
  scenario = "A7",
  description = "Fay's BRR replicates (rho=0.3) from stratified design",
  design_type = "fay_brr_replicate",
  estimator = "survey::svyglm (svrepdesign)",
  fay_rho = 0.3,
  att = est_a7$att,
  se = est_a7$se,
  t_stat = est_a7$t_stat,
  df = est_a7$df,
  ci_lower = est_a7$ci_lower,
  ci_upper = est_a7$ci_upper,
  n_obs = nrow(panel),
  n_replicates = n_replicates_a7,
  scale = design_a7$scale,
  rscales = as.list(design_a7$rscales),
  replicate_weights = rep_weight_export,
  sum_weights = sum(panel$pw)
)

# ---------------------------------------------------------------------------
# DEFF diagnostics (for Suite C in Python test)
# ---------------------------------------------------------------------------

cat("\n--- DEFF Diagnostics ---\n")
# Compute DEFF for the interaction term using the full design
# svyglm doesn't expose per-coefficient DEFF directly, but we can compare
# survey SE to naive (SRS) SE for the interaction term
naive_model <- lm(outcome ~ treated * post, data = panel, weights = pw)
naive_se_interaction <- summary(naive_model)$coefficients["treated:post", "Std. Error"]

deff_interaction <- (est_a1$se / naive_se_interaction)^2
cat(sprintf("  DEFF(treated:post) = %.4f (survey SE / naive SE)^2\n", deff_interaction))

results[["deff_diagnostics"]] <- list(
  description = "DEFF for interaction term: (survey_SE / naive_SE)^2",
  survey_se = est_a1$se,
  naive_se = naive_se_interaction,
  deff = deff_interaction
)

# ---------------------------------------------------------------------------
# Embed dataset in JSON
# ---------------------------------------------------------------------------

data_export <- list(
  school_id = as.list(panel$school_id),
  period = as.list(panel$period),
  post = as.list(panel$post),
  treated = as.list(panel$treated),
  outcome = as.list(panel$outcome),
  stype = as.list(panel$stype),
  pw = as.list(panel$pw),
  fpc = as.list(panel$fpc),
  dnum = as.list(panel$dnum),
  meals = as.list(panel$meals),
  ell = as.list(panel$ell)
)

results[["_data"]] <- data_export
results[["_metadata"]] <- list(
  source = "R survey::api dataset (apistrat)",
  policy = "California Public Schools Accountability Act (PSAA, 1999)",
  treatment = "awards == 'Yes' (Governor's Performance Award eligibility)",
  pre_period = "api99 (1999)",
  post_period = "api00 (2000)",
  n_schools = n_schools,
  n_obs = nrow(panel),
  r_version = paste0(R.version$major, ".", R.version$minor),
  survey_version = as.character(packageVersion("survey"))
)

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

json_path <- file.path(output_dir, "api_realdata_golden.json")
writeLines(toJSON(results, auto_unbox = TRUE, pretty = TRUE, digits = 10), json_path)
cat(sprintf("\nResults saved to: %s\n", json_path))

cat("\n=== Summary ===\n")
cat(sprintf("Suite A: 7 scenarios\n"))
cat(sprintf("  A1: strata+FPC+weights    ATT=%.4f SE=%.4f df=%d\n",
            est_a1$att, est_a1$se, est_a1$df))
cat(sprintf("  A2: strata+weights        ATT=%.4f SE=%.4f df=%d\n",
            est_a2$att, est_a2$se, est_a2$df))
cat(sprintf("  A3: weights only          ATT=%.4f SE=%.4f df=%d\n",
            est_a3$att, est_a3$se, est_a3$df))
cat(sprintf("  A4: TWFE                  ATT=%.4f SE=%.4f df=%d\n",
            est_a4$att, est_a4$se, est_a4$df))
cat(sprintf("  A5: subpop (elementary)   ATT=%.4f SE=%.4f df=%d\n",
            est_a5$att, est_a5$se, est_a5$df))
cat(sprintf("  A6: covariates            ATT=%.4f SE=%.4f df=%d\n",
            est_a6$att, est_a6$se, est_a6$df))
cat(sprintf("  A7: Fay BRR (rho=0.3)     ATT=%.4f SE=%.4f df=%d (%d replicates)\n",
            est_a7$att, est_a7$se, est_a7$df, n_replicates_a7))
cat("Done.\n")
