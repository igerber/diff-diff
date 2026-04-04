#!/usr/bin/env Rscript
# Benchmark: Real-Data Survey Cross-Validation — NHANES ACA Coverage
#
# Uses NHANES data (pre-processed CSV) to validate diff-diff's survey
# variance against R's survey::svyglm() with real CDC survey design.
#
# Policy: ACA dependent coverage provision (effective Sept 2010).
# Treatment = adults 19-25 (eligible for parents' insurance).
# Control = adults 27-34 (not eligible).
# Pre = NHANES 2007-2008, Post = NHANES 2015-2016.
# Outcome = insured (0/1), modeled as LPM via svyglm(family=gaussian).
#
# Suite B: 5 scenarios testing TSL variance with strata + PSU + weights,
#          covariates, subpopulation, and CallawaySantAnna RC-DiD.
#
# Usage:
#   python benchmarks/scripts/download_nhanes.py  # first: download data
#   Rscript benchmark_realdata_nhanes.R
#
# References:
#   Sommers, B.D. (2012). "Number of Young Adults Gaining Insurance Due to
#     the ACA." JAMA 307(9): 913-914.
#   Antwi, Y.A., Moriya, A.S., & Simon, K. (2013). "Effects of Federal
#     Policy to Insure Young Adults." AEJ: Economic Policy 5(4).

library(survey)
library(jsonlite)

cat("=== Real-Data Survey Benchmark: NHANES ACA Coverage ===\n\n")

# --- Locate data file ---
data_file <- "benchmarks/data/real/nhanes_aca_subset.csv"
if (!file.exists(data_file)) {
  data_file <- file.path(dirname(dirname(getwd())), data_file)
}
if (!file.exists(data_file)) {
  stop("NHANES CSV not found. Run: python benchmarks/scripts/download_nhanes.py")
}

output_dir <- dirname(data_file)

# --- Read pre-processed data ---
nhanes <- read.csv(data_file)
cat(sprintf("  Loaded: %d rows, %d columns\n", nrow(nhanes), ncol(nhanes)))
cat(sprintf("  Pre-ACA: %d, Post-ACA: %d\n",
            sum(nhanes$period == 0), sum(nhanes$period == 1)))
cat(sprintf("  Treatment (19-25): %d, Control (27-34): %d\n",
            sum(nhanes$treated == 1), sum(nhanes$treated == 0)))
cat(sprintf("  Strata: %d, PSUs per stratum: %d\n",
            length(unique(nhanes$SDMVSTRA)), length(unique(nhanes$SDMVPSU))))

results <- list()

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

extract_svyglm_results <- function(model, design, coef_name = "treated:post") {
  coefs <- coef(model)
  ses <- SE(model)
  idx <- which(names(coefs) == coef_name)

  att <- as.numeric(coefs[idx])
  se <- as.numeric(ses[idx])
  t_stat <- att / se
  df <- degf(design)

  t_crit <- qt(0.975, df)
  ci_lower <- att - t_crit * se
  ci_upper <- att + t_crit * se

  list(
    att = att, se = se, t_stat = t_stat, df = df,
    ci_lower = ci_lower, ci_upper = ci_upper
  )
}

# ---------------------------------------------------------------------------
# B1: Full design — strata + PSU + weights (nest=TRUE)
# ---------------------------------------------------------------------------

cat("\n--- Suite B: NHANES ACA DiD ---\n")

cat("  B1: strata + PSU + weights (nest=TRUE) ...\n")
design_b1 <- svydesign(
  ids = ~SDMVPSU, strata = ~SDMVSTRA, weights = ~WTMEC2YR,
  nest = TRUE, data = nhanes
)
model_b1 <- svyglm(outcome ~ treated * post, design = design_b1)
est_b1 <- extract_svyglm_results(model_b1, design_b1)
cat(sprintf("    ATT = %.6f, SE = %.6f, df = %d\n", est_b1$att, est_b1$se, est_b1$df))

results[["b1_strata_psu_weights"]] <- c(
  list(scenario = "B1", description = "Full TSL: strata + PSU + weights (nest=TRUE)",
       design_type = "strata_psu_weights"),
  est_b1,
  list(n_obs = nrow(nhanes), n_strata = length(unique(nhanes$SDMVSTRA)),
       sum_weights = sum(nhanes$WTMEC2YR))
)

# ---------------------------------------------------------------------------
# B2: Covariates — RIAGENDR + INDFMPIR
# ---------------------------------------------------------------------------

cat("  B2: strata + PSU + covariates (RIAGENDR, INDFMPIR) ...\n")
nhanes_cov <- nhanes[complete.cases(nhanes[, c("RIAGENDR", "INDFMPIR")]), ]
design_b2 <- svydesign(
  ids = ~SDMVPSU, strata = ~SDMVSTRA, weights = ~WTMEC2YR,
  nest = TRUE, data = nhanes_cov
)
model_b2 <- svyglm(outcome ~ treated * post + RIAGENDR + INDFMPIR, design = design_b2)
est_b2 <- extract_svyglm_results(model_b2, design_b2)
cat(sprintf("    ATT = %.6f, SE = %.6f, df = %d\n", est_b2$att, est_b2$se, est_b2$df))

results[["b2_covariates"]] <- c(
  list(scenario = "B2", description = "Covariate-adjusted: RIAGENDR + INDFMPIR",
       design_type = "strata_psu_covariates", has_covariates = TRUE,
       covariates = c("RIAGENDR", "INDFMPIR")),
  est_b2,
  list(n_obs = nrow(nhanes_cov))
)

# ---------------------------------------------------------------------------
# B3: Weights only (no clustering)
# ---------------------------------------------------------------------------

cat("  B3: weights only ...\n")
design_b3 <- svydesign(ids = ~1, weights = ~WTMEC2YR, data = nhanes)
model_b3 <- svyglm(outcome ~ treated * post, design = design_b3)
est_b3 <- extract_svyglm_results(model_b3, design_b3)
cat(sprintf("    ATT = %.6f, SE = %.6f, df = %d\n", est_b3$att, est_b3$se, est_b3$df))

results[["b3_weights_only"]] <- c(
  list(scenario = "B3", description = "Weights only (no strata/PSU)",
       design_type = "weights_only"),
  est_b3,
  list(n_obs = nrow(nhanes), sum_weights = sum(nhanes$WTMEC2YR))
)

# ---------------------------------------------------------------------------
# B4: Subpopulation — female only (RIAGENDR == 2)
# ---------------------------------------------------------------------------

cat("  B4: subpopulation (female, RIAGENDR==2) ...\n")
design_b4_full <- svydesign(
  ids = ~SDMVPSU, strata = ~SDMVSTRA, weights = ~WTMEC2YR,
  nest = TRUE, data = nhanes
)
design_b4 <- subset(design_b4_full, RIAGENDR == 2)
model_b4 <- svyglm(outcome ~ treated * post, design = design_b4)
est_b4 <- extract_svyglm_results(model_b4, design_b4)
cat(sprintf("    ATT = %.6f, SE = %.6f, df = %d\n", est_b4$att, est_b4$se, est_b4$df))

n_female <- sum(nhanes$RIAGENDR == 2)
results[["b4_subpop_female"]] <- c(
  list(scenario = "B4", description = "Subpopulation: female (RIAGENDR==2)",
       design_type = "strata_psu_subpop", subpopulation = "RIAGENDR == 2"),
  est_b4,
  list(n_obs_subpop = n_female, n_obs_full = nrow(nhanes))
)

# ---------------------------------------------------------------------------
# B5: CallawaySantAnna RC-DiD (ATT only — R did is survey-naive for SE)
# ---------------------------------------------------------------------------

cat("  B5: CallawaySantAnna RC-DiD ...\n")

has_did <- requireNamespace("did", quietly = TRUE)
if (has_did) {
  library(did)

  # Remap periods for did::att_gt: period 1 (pre) and 2 (post)
  # first_treat = 2 for treated (first treated at period 2), 0 for never-treated
  nhanes_cs <- nhanes
  nhanes_cs$period_cs <- nhanes_cs$period + 1L  # 0->1, 1->2
  nhanes_cs$first_treat_cs <- ifelse(nhanes_cs$treated == 1L, 2L, 0L)

  out_b5 <- tryCatch({
    att_gt(
      yname = "outcome",
      tname = "period_cs",
      idname = "unit_id",
      gname = "first_treat_cs",
      data = nhanes_cs,
      weightsname = "WTMEC2YR",
      est_method = "reg",
      control_group = "nevertreated",
      base_period = "varying",
      panel = FALSE,
      bstrap = FALSE,
      cband = FALSE
    )
  }, error = function(e) {
    cat(sprintf("    ERROR: %s\n", e$message))
    NULL
  })

  if (!is.null(out_b5)) {
    agg_b5 <- aggte(out_b5, type = "simple")
    cat(sprintf("    Overall ATT = %.6f, SE = %.6f\n",
                agg_b5$overall.att, agg_b5$overall.se))

    results[["b5_cs_rc"]] <- list(
      scenario = "B5",
      description = "CallawaySantAnna RC-DiD (panel=FALSE)",
      estimator = "did::att_gt",
      estimation_method = "reg",
      control_group = "nevertreated",
      panel = FALSE,
      overall_att = agg_b5$overall.att,
      overall_se = agg_b5$overall.se
    )
  }
} else {
  cat("    SKIPPED: 'did' package not available\n")
}

# ---------------------------------------------------------------------------
# Embed dataset and save
# ---------------------------------------------------------------------------

data_export <- list()
for (col in names(nhanes)) {
  data_export[[col]] <- as.list(nhanes[[col]])
}

results[["_data"]] <- data_export
results[["_metadata"]] <- list(
  source = "NHANES (CDC/NCHS) — 2007-2008 and 2015-2016 cycles",
  policy = "ACA dependent coverage provision (effective Sept 2010)",
  treatment = "Adults 19-25 (eligible for parents' insurance)",
  control = "Adults 27-34 (not eligible)",
  outcome = "Health insurance coverage (binary, LPM)",
  n_obs = nrow(nhanes),
  r_version = paste0(R.version$major, ".", R.version$minor),
  survey_version = as.character(packageVersion("survey"))
)

json_path <- file.path(output_dir, "nhanes_realdata_golden.json")
writeLines(toJSON(results, auto_unbox = TRUE, pretty = TRUE, digits = 10), json_path)
cat(sprintf("\nResults saved to: %s\n", json_path))

cat("\n=== Summary ===\n")
cat(sprintf("  B1: strata+PSU+weights   ATT=%.4f SE=%.4f df=%d\n",
            est_b1$att, est_b1$se, est_b1$df))
cat(sprintf("  B2: covariates           ATT=%.4f SE=%.4f df=%d\n",
            est_b2$att, est_b2$se, est_b2$df))
cat(sprintf("  B3: weights only         ATT=%.4f SE=%.4f df=%d\n",
            est_b3$att, est_b3$se, est_b3$df))
cat(sprintf("  B4: subpop (female)      ATT=%.4f SE=%.4f df=%d\n",
            est_b4$att, est_b4$se, est_b4$df))
cat("Done.\n")
