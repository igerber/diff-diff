#!/usr/bin/env Rscript
# Benchmark: Staggered Triple Difference (R `triplediff` package)
#
# Generates golden values for cross-validation against Python
# StaggeredTripleDifference estimator.
#
# Usage:
#   Rscript benchmark_staggered_triplediff.R

library(triplediff)
library(jsonlite)
library(data.table)

cat("=== Staggered DDD Benchmark Generator ===\n")

output_dir <- file.path(dirname(dirname(getwd())), "benchmarks", "data", "synthetic")
# Handle running from project root or benchmarks/R
if (!dir.exists(output_dir)) {
  output_dir <- "benchmarks/data/synthetic"
}
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

results <- list()

# Scenario definitions
scenarios <- list(
  list(seed=42,  dgp=1, method="dr",  cg="nevertreated",   key="s42_dgp1_dr_nt"),
  list(seed=42,  dgp=1, method="ipw", cg="nevertreated",   key="s42_dgp1_ipw_nt"),
  list(seed=42,  dgp=1, method="reg", cg="nevertreated",   key="s42_dgp1_reg_nt"),
  list(seed=42,  dgp=1, method="dr",  cg="notyettreated",  key="s42_dgp1_dr_nyt"),
  list(seed=42,  dgp=1, method="ipw", cg="notyettreated",  key="s42_dgp1_ipw_nyt"),
  list(seed=42,  dgp=1, method="reg", cg="notyettreated",  key="s42_dgp1_reg_nyt"),
  list(seed=123, dgp=1, method="dr",  cg="nevertreated",   key="s123_dgp1_dr_nt"),
  list(seed=123, dgp=1, method="dr",  cg="notyettreated",  key="s123_dgp1_dr_nyt"),
  list(seed=99,  dgp=1, method="dr",  cg="nevertreated",   key="s99_dgp1_dr_nt"),
  list(seed=99,  dgp=1, method="dr",  cg="notyettreated",  key="s99_dgp1_dr_nyt")
)

for (sc in scenarios) {
  cat(sprintf("  Running scenario: %s ...\n", sc$key))

  set.seed(sc$seed)
  dgp <- gen_dgp_mult_periods(size = 500, dgp_type = sc$dgp)
  data <- dgp$data

  # Save data CSV (one per seed+dgp combo, reused across methods)
  data_key <- sprintf("s%d_dgp%d", sc$seed, sc$dgp)
  csv_path <- file.path(output_dir, sprintf("staggered_ddd_data_%s.csv", data_key))
  if (!file.exists(csv_path)) {
    fwrite(data, csv_path)
    cat(sprintf("    Saved data: %s\n", csv_path))
  }

  # Run DDD estimation
  res <- tryCatch({
    ddd(yname = "y", tname = "time", idname = "id",
        gname = "state", pname = "partition",
        xformla = ~1,  # no covariates for cross-validation
        data = data,
        control_group = sc$cg,
        base_period = "varying",
        est_method = sc$method,
        panel = TRUE)
  }, error = function(e) {
    cat(sprintf("    ERROR: %s\n", e$message))
    return(NULL)
  })

  if (is.null(res)) next

  # Group-time results
  gt_results <- data.frame(
    group = res$groups,
    period = res$periods,
    att = res$ATT,
    se = res$se
  )

  # Event study aggregation
  agg_es <- tryCatch({
    agg_ddd(res, type = "eventstudy")
  }, error = function(e) {
    cat(sprintf("    Event study agg failed: %s\n", e$message))
    NULL
  })

  es_results <- NULL
  overall_att_es <- NA
  overall_se_es <- NA
  if (!is.null(agg_es)) {
    a <- agg_es$aggte_ddd
    es_results <- data.frame(
      event_time = a$egt,
      att = a$att.egt,
      se = a$se.egt
    )
    overall_att_es <- a$overall.att
    overall_se_es <- a$overall.se
  }

  # Simple aggregation
  agg_simple <- tryCatch({
    agg_ddd(res, type = "simple")
  }, error = function(e) {
    cat(sprintf("    Simple agg failed: %s\n", e$message))
    NULL
  })

  overall_att_simple <- NA
  overall_se_simple <- NA
  if (!is.null(agg_simple)) {
    a <- agg_simple$aggte_ddd
    overall_att_simple <- a$overall.att
    overall_se_simple <- a$overall.se
  }

  # Store results
  results[[sc$key]] <- list(
    seed = sc$seed,
    dgp_type = sc$dgp,
    est_method = sc$method,
    control_group = sc$cg,
    n = res$n,
    gt_att = as.list(gt_results$att),
    gt_se = as.list(gt_results$se),
    gt_groups = as.list(gt_results$group),
    gt_periods = as.list(gt_results$period),
    overall_att_simple = overall_att_simple,
    overall_se_simple = overall_se_simple,
    overall_att_es = overall_att_es,
    overall_se_es = overall_se_es,
    es_event_times = if (!is.null(es_results)) as.list(es_results$event_time) else NULL,
    es_att = if (!is.null(es_results)) as.list(es_results$att) else NULL,
    es_se = if (!is.null(es_results)) as.list(es_results$se) else NULL
  )

  cat(sprintf("    GT ATT: %s\n", paste(round(res$ATT, 4), collapse=", ")))
  cat(sprintf("    Overall ATT (simple): %.4f\n", overall_att_simple))
}

# Save all results as JSON
json_path <- file.path(output_dir, "staggered_ddd_r_results.json")
writeLines(toJSON(results, auto_unbox = TRUE, pretty = TRUE, digits = 10), json_path)
cat(sprintf("\nResults saved to: %s\n", json_path))
cat("Done.\n")
