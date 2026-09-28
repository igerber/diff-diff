#!/usr/bin/env Rscript
# Golden generator: TwoStageDiD vs R `did2s::did2s` (Gardner 2022).
#
# Writes a fixed staggered-adoption panel and the R reference estimates so that
# tests/test_methodology_two_stage.py::TestTwoStageDiDParityR can pin Python
# output against R without requiring R at test time.
#
# Outputs (checked into the repo):
#   benchmarks/data/did2s_test_panel.csv   (unit, time, first_treat, y)
#   benchmarks/data/did2s_golden.json
#
# Usage:
#   Rscript benchmarks/R/generate_did2s_golden.R
#
# Notes:
#   - did2s defaults to analytical corrected clustered SEs (`bootstrap = FALSE`),
#     i.e. the Gardner two-stage GMM sandwich with the GLOBAL Jacobian inverse and
#     NO finite-sample multiplier. This is exactly what TwoStageDiD computes (see
#     docs/methodology/REGISTRY.md "## TwoStageDiD"); we cluster on `unit` to match
#     the library default (cluster=None -> unit).
#   - The first stage `~ 0 | unit + time` demeans by unit + time FE (which span the
#     constant); the library's GMM variance re-solves the same exact two-way FE.
#   - Event study: did2s relative-time is `rel_year = time - first_treat` for
#     treated units and `Inf` for never-treated; `i(rel_year, ref = c(-1, Inf))`
#     drops the -1 reference period and excludes never-treated. We compare only the
#     post-treatment horizons (r >= 0), which map 1:1 to the library's
#     event_study_effects keys (h = 0 is the first treated period). The pre-period
#     leads (r < 0) are ~0 under parallel trends and are not part of the library's
#     default (pretrends=False) event study.
#   - Same DGP (seed, cohorts, horizon-heterogeneous effect) as the didimputation
#     golden, so the point estimates coincide (the two-stage and imputation
#     estimands are algebraically identical); only the SEs differ (GMM sandwich vs
#     the imputation Theorem-3 variance).

suppressMessages({
  library(did2s)
  library(jsonlite)
  library(data.table)
})

set.seed(2024)

# ---- Deterministic staggered-adoption panel (parallel trends) ----
# Cohorts 3 and 5 plus a never-treated group (first_treat = 0); 8 periods.
n_per_cohort <- 60L
n_periods <- 8L
cohorts <- c(0L, 3L, 5L)          # 0 = never-treated
tau_h <- function(k) 1.0 + 0.5 * k # heterogeneous-by-horizon effect

rows <- list()
uid <- 0L
for (g in cohorts) {
  for (j in seq_len(n_per_cohort)) {
    c_i <- rnorm(1)
    for (t in seq_len(n_periods)) {
      beta_t <- 0.5 * t
      u <- 0.2 * rnorm(1)
      treated <- (g > 0L) && (t >= g)
      eff <- if (treated) tau_h(t - g) else 0.0
      y <- c_i + beta_t + eff + u
      rows[[length(rows) + 1L]] <- list(
        unit = uid, time = t, first_treat = g, y = y
      )
    }
    uid <- uid + 1L
  }
}
panel <- rbindlist(rows)
panel[, unit := as.integer(unit)]
panel[, time := as.integer(time)]
panel[, first_treat := as.integer(first_treat)]

panel_path <- file.path("benchmarks", "data", "did2s_test_panel.csv")
dir.create(dirname(panel_path), recursive = TRUE, showWarnings = FALSE)
fwrite(panel, panel_path)
message(sprintf("Wrote panel: %s (%d rows)", panel_path, nrow(panel)))

# did2s-specific columns (derived from first_treat; not committed in the CSV).
panel[, treat := (first_treat > 0L) & (time >= first_treat)]
panel[, rel_year := ifelse(first_treat > 0L, time - first_treat, Inf)]

# ---- Overall ATT (static) ----
overall <- did2s(
  panel,
  yname = "y",
  first_stage = ~ 0 | unit + time,
  second_stage = ~ i(treat, ref = FALSE),
  treatment = "treat",
  cluster_var = "unit"
)
oct <- summary(overall)$coeftable
overall_att <- as.numeric(oct[1, 1])
overall_se <- as.numeric(oct[1, 2])
message(sprintf("Overall ATT = %.8f (SE %.8f)", overall_att, overall_se))

# ---- Event study (post-treatment horizons r >= 0) ----
es <- did2s(
  panel,
  yname = "y",
  first_stage = ~ 0 | unit + time,
  second_stage = ~ i(rel_year, ref = c(-1, Inf)),
  treatment = "treat",
  cluster_var = "unit"
)
ect <- summary(es)$coeftable
es_h_all <- as.integer(gsub("rel_year::", "", rownames(ect), fixed = TRUE))
keep <- !is.na(es_h_all) & es_h_all >= 0
ord <- order(es_h_all[keep])
es_h <- es_h_all[keep][ord]
es_att <- as.numeric(ect[keep, 1])[ord]
es_se <- as.numeric(ect[keep, 2])[ord]
for (i in seq_along(es_h)) {
  message(sprintf("  h=%d: ATT=%.6f SE=%.6f", es_h[i], es_att[i], es_se[i]))
}

golden <- list(
  estimator = "did2s::did2s",
  meta = list(
    r_version = R.version.string,
    did2s_version = as.character(packageVersion("did2s")),
    seed = 2024L,
    n_units = length(unique(panel$unit)),
    n_periods = n_periods,
    cohorts = cohorts,
    se_type = "Corrected Clustered (unit); bootstrap = FALSE (analytical GMM sandwich)",
    note = paste(
      "did2s analytical corrected clustered SE = the Gardner (2022) two-stage GMM",
      "sandwich with the global Jacobian inverse and no finite-sample multiplier.",
      "Event study compares post-treatment horizons r >= 0 (i(rel_year, ref=c(-1,Inf)))."
    )
  ),
  overall = list(att = overall_att, se = overall_se),
  event_study = list(horizons = es_h, att = es_att, se = es_se)
)

golden_path <- file.path("benchmarks", "data", "did2s_golden.json")
write_json(golden, golden_path, auto_unbox = TRUE, pretty = TRUE, digits = 12)
message(sprintf("Wrote golden: %s", golden_path))
