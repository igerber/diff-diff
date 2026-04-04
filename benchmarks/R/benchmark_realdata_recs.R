#!/usr/bin/env Rscript
# Benchmark: Real-Data Survey Cross-Validation — RECS 2020 JK1 Replicates
#
# Uses the RECS 2020 subset (pre-processed CSV) to validate diff-diff's
# JK1 replicate weight variance against R's survey::svrepdesign().
#
# RECS provides 60 pre-computed JK1 (delete-one jackknife) replicate weight
# columns — the actual weights used for official EIA publications.
#
# Suite C: 3 scenarios — simple regression, full regression, DEFF.
# NOTE: This validates survey-weighted regression SEs, not a full DiD.
#
# Usage:
#   python benchmarks/scripts/download_recs.py  # first: download data
#   Rscript benchmark_realdata_recs.R
#
# References:
#   U.S. EIA. Residential Energy Consumption Survey (RECS) 2020.

library(survey)
library(jsonlite)

cat("=== Real-Data Survey Benchmark: RECS 2020 JK1 Replicates ===\n\n")

# --- Locate data file ---
data_file <- "benchmarks/data/real/recs_subset.csv"
if (!file.exists(data_file)) {
  data_file <- file.path(dirname(dirname(getwd())), data_file)
}
if (!file.exists(data_file)) {
  stop("RECS CSV not found. Run: python benchmarks/scripts/download_recs.py")
}

output_dir <- dirname(data_file)

# --- Read pre-processed data ---
recs <- read.csv(data_file)
cat(sprintf("  Loaded: %d rows, %d columns\n", nrow(recs), ncol(recs)))

# Identify replicate weight columns
rep_cols <- grep("^NWEIGHT[0-9]+$", names(recs), value = TRUE)
cat(sprintf("  Replicate weight columns: %d\n", length(rep_cols)))

# Drop rows with missing outcome or weight
recs <- recs[complete.cases(recs[, c("TOTALBTU", "NWEIGHT")]), ]
cat(sprintf("  Rows after NA removal: %d\n", nrow(recs)))

results <- list()

# --- Create JK1 replicate design ---
rep_design <- svrepdesign(
  weights = ~NWEIGHT,
  repweights = recs[, rep_cols],
  type = "JK1",
  data = recs
)
cat(sprintf("  JK1 design: df = %d\n", degf(rep_design)))

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

extract_svyglm_coef <- function(model, design, coef_name) {
  coefs <- coef(model)
  ses <- SE(model)
  idx <- which(names(coefs) == coef_name)

  coef_val <- as.numeric(coefs[idx])
  se <- as.numeric(ses[idx])
  df <- degf(design)

  t_crit <- qt(0.975, df)
  ci_lower <- coef_val - t_crit * se
  ci_upper <- coef_val + t_crit * se

  list(
    coef = coef_val, se = se, df = df,
    ci_lower = ci_lower, ci_upper = ci_upper
  )
}

# ---------------------------------------------------------------------------
# C1: Simple regression — TOTALBTU ~ KOWNRENT
# ---------------------------------------------------------------------------

cat("\n--- Suite C: RECS JK1 Replicate Weights ---\n")

cat("  C1: TOTALBTU ~ KOWNRENT ...\n")
recs_c1 <- recs[complete.cases(recs[, "KOWNRENT"]), ]
design_c1 <- svrepdesign(
  weights = ~NWEIGHT, repweights = recs_c1[, rep_cols],
  type = "JK1", data = recs_c1
)
model_c1 <- svyglm(TOTALBTU ~ KOWNRENT, design = design_c1)
est_c1 <- extract_svyglm_coef(model_c1, design_c1, "KOWNRENT")
cat(sprintf("    KOWNRENT coef = %.4f, SE = %.4f, df = %d\n",
            est_c1$coef, est_c1$se, est_c1$df))

results[["c1_simple"]] <- c(
  list(scenario = "C1", description = "Simple: TOTALBTU ~ KOWNRENT",
       design_type = "jk1_replicate"),
  est_c1,
  list(n_obs = nrow(recs_c1), n_replicates = length(rep_cols))
)

# ---------------------------------------------------------------------------
# C2: Full regression — TOTALBTU ~ KOWNRENT + factor(TYPEHUQ) + factor(REGIONC)
# ---------------------------------------------------------------------------

cat("  C2: TOTALBTU ~ KOWNRENT + TYPEHUQ + REGIONC ...\n")
recs_c2 <- recs[complete.cases(recs[, c("KOWNRENT", "TYPEHUQ", "REGIONC")]), ]
design_c2 <- svrepdesign(
  weights = ~NWEIGHT, repweights = recs_c2[, rep_cols],
  type = "JK1", data = recs_c2
)
# Use factor() for categorical variables to match Python's get_dummies
model_c2 <- svyglm(TOTALBTU ~ KOWNRENT + factor(TYPEHUQ) + factor(REGIONC),
                    design = design_c2)
coefs_c2 <- coef(model_c2)
ses_c2 <- SE(model_c2)
df_c2 <- degf(design_c2)

# Extract KOWNRENT specifically
kownrent_idx <- which(names(coefs_c2) == "KOWNRENT")
kownrent_coef <- as.numeric(coefs_c2[kownrent_idx])
kownrent_se <- as.numeric(ses_c2[kownrent_idx])

t_crit_c2 <- qt(0.975, df_c2)

cat(sprintf("    KOWNRENT coef = %.4f, SE = %.4f, df = %d\n",
            kownrent_coef, kownrent_se, df_c2))

results[["c2_full"]] <- list(
  scenario = "C2",
  description = "Full: TOTALBTU ~ KOWNRENT + factor(TYPEHUQ) + factor(REGIONC)",
  design_type = "jk1_replicate",
  coef_kownrent = kownrent_coef,
  se_kownrent = kownrent_se,
  df = df_c2,
  ci_lower_kownrent = kownrent_coef - t_crit_c2 * kownrent_se,
  ci_upper_kownrent = kownrent_coef + t_crit_c2 * kownrent_se,
  n_obs = nrow(recs_c2),
  n_replicates = length(rep_cols),
  # Also export all coefficients for reference
  all_coef_names = as.list(names(coefs_c2)),
  all_coefs = as.list(as.numeric(coefs_c2)),
  all_ses = as.list(as.numeric(ses_c2))
)

# ---------------------------------------------------------------------------
# Naive (unweighted) SE for DEFF reference
# ---------------------------------------------------------------------------

cat("  DEFF reference: naive (unweighted) regression ...\n")
naive_c1 <- lm(TOTALBTU ~ KOWNRENT, data = recs_c1, weights = NWEIGHT)
naive_se <- summary(naive_c1)$coefficients["KOWNRENT", "Std. Error"]
deff_kownrent <- (est_c1$se / naive_se)^2
cat(sprintf("    DEFF(KOWNRENT) = %.4f\n", deff_kownrent))

results[["deff_diagnostics"]] <- list(
  description = "DEFF for KOWNRENT: (JK1 SE / naive SE)^2",
  survey_se = est_c1$se,
  naive_se = naive_se,
  deff = deff_kownrent
)

# ---------------------------------------------------------------------------
# Embed dataset and save
# ---------------------------------------------------------------------------

data_export <- list()
for (col in names(recs)) {
  data_export[[col]] <- as.list(recs[[col]])
}

results[["_data"]] <- data_export
results[["_metadata"]] <- list(
  source = "RECS 2020 (U.S. EIA) — 2,000-row deterministic subsample",
  replicate_method = "JK1 (delete-one jackknife)",
  n_replicates = length(rep_cols),
  n_obs = nrow(recs),
  r_version = paste0(R.version$major, ".", R.version$minor),
  survey_version = as.character(packageVersion("survey"))
)

json_path <- file.path(output_dir, "recs_realdata_golden.json")
writeLines(toJSON(results, auto_unbox = TRUE, pretty = TRUE, digits = 10), json_path)
cat(sprintf("\nResults saved to: %s\n", json_path))

cat("\n=== Summary ===\n")
cat(sprintf("  C1: simple (KOWNRENT)    coef=%.4f SE=%.4f df=%d\n",
            est_c1$coef, est_c1$se, est_c1$df))
cat(sprintf("  C2: full (+ TYPEHUQ + REGIONC)  KOWNRENT coef=%.4f SE=%.4f df=%d\n",
            kownrent_coef, kownrent_se, df_c2))
cat(sprintf("  DEFF(KOWNRENT) = %.4f\n", deff_kownrent))
cat("Done.\n")
