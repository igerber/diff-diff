# Generate R-parity goldens for WooldridgeDiD OLS path vcov_type variants.
#
# Phase 1b PR 3/8: pins Python `WooldridgeDiD(method='ols', vcov_type=...)` SE
# output against `lm()` + clubSandwich / sandwich on the fixed-seed staggered
# panel from `benchmarks/data/wooldridge_test_panel.csv`.
#
# Variants generated:
#   - hc1: CR1 Liang-Zeger cluster-robust at unit via clubSandwich `type="CR1S"`
#     (Stata-style G/(G-1) * (n-1)/(n-p) correction on the full-dummy lm design).
#     REFERENCE ONLY — diff-diff's WooldridgeDiD(vcov_type='hc1') uses the
#     within-transformed design and is NOT pinned at parity here. See REGISTRY
#     "Variance families" → "Deviation from R" for the (n-1)/(n-k) factor
#     difference. The hc1 SE in this JSON is for diagnostic comparison only;
#     do NOT add a Python parity test against it.
#   - hc2_bm: CR2 Bell-McCaffrey cluster-robust at unit (per-coef DOF via
#     coef_test()$df_Satt; overall ATT BM contrast DOF via Wald_test(test="HTZ")$df_denom).
#   - classical: lm() summary's homoskedastic OLS SE (no robust correction).
#     Python's vcov_type='classical' drops the unit auto-cluster to match this.
#   - hc2: sandwich::vcovHC type="HC2" with NO clustering. Python's
#     vcov_type='hc2' also drops the unit auto-cluster to match.
#
# clubSandwich >= 0.7.0 required (matches PR #475 / PR #479 pin).

suppressPackageStartupMessages({
  library(clubSandwich)
  library(sandwich)
  library(jsonlite)
  library(etwfe)
})

stopifnot(packageVersion("clubSandwich") >= "0.7.0")
stopifnot(packageVersion("sandwich") >= "3.0.0")
stopifnot(packageVersion("etwfe") >= "0.5.0")

panel_path <- file.path("benchmarks", "data", "wooldridge_test_panel.csv")
out_path <- file.path("benchmarks", "data", "wooldridge_golden.json")

df <- read.csv(panel_path)
stopifnot(all(c("unit", "time", "cohort", "y") %in% names(df)))
# Force integer types on unit/time/cohort so the cluster formula resolves
# cleanly (clubSandwich's `cluster = df$unit` calls `unique(model$unit)` which
# fails on factor-coerced columns from intermediate model frames).
df$unit <- as.integer(df$unit)
df$time <- as.integer(df$time)
df$cohort <- as.integer(df$cohort)

# Build treated (g, t) interaction dummies, matching the Python OLS path's
# `_build_interaction_matrix` (control_group="not_yet_treated", anticipation=0):
# one indicator per treated (g, t) cell with g > 0 and t >= g.
treated_cohorts <- sort(unique(df$cohort[df$cohort > 0]))
times <- sort(unique(df$time))
gt_pairs <- list()
for (g in treated_cohorts) {
  for (t in times) {
    if (t >= g) {
      gt_pairs[[length(gt_pairs) + 1L]] <- c(g, t)
    }
  }
}
gt_names <- vapply(gt_pairs, function(p) sprintf("D_%d_%d", p[1], p[2]), character(1))
for (i in seq_along(gt_pairs)) {
  g <- gt_pairs[[i]][1]
  t <- gt_pairs[[i]][2]
  df[[gt_names[i]]] <- as.integer((df$cohort == g) & (df$time == t))
}
n_int <- length(gt_names)

# Fit lm(y ~ <interactions> + as.factor(unit) + as.factor(time)). The
# `as.factor(...)` form drops the first level of each FE block, matching the
# Python full-dummy build (`drop_first=True` on `pd.get_dummies(unit)` and
# `pd.get_dummies(time)`), and adds a single intercept — matching Python's
# `[intercept, X_design, unit_dummies, time_dummies]`.
formula_str <- paste0(
  "y ~ ", paste(gt_names, collapse = " + "),
  " + as.factor(unit) + as.factor(time)"
)
fit <- lm(as.formula(formula_str), data = df)

# Extract the (interaction) coefficient indices in fit$coefficients. R places
# them right after the intercept (positions 2..(1+n_int) in 1-indexed R).
coef_names <- names(coef(fit))
int_idx <- match(gt_names, coef_names)
stopifnot(!any(is.na(int_idx)))

# Cell weights n_{g,t} for the overall ATT contrast (matches Python's
# `_compute_weighted_agg` with default `weights=n_{g,t}`).
n_gt <- vapply(seq_along(gt_pairs), function(i) {
  g <- gt_pairs[[i]][1]
  t <- gt_pairs[[i]][2]
  sum(df$cohort == g & df$time == t)
}, integer(1))
n_post_total <- sum(n_gt)
contrast_weights <- n_gt / n_post_total  # length n_int

# Build the overall ATT contrast in full-coef space (intercept = 0, then n_int
# weights, then 0 for FE dummies).
n_total_coef <- length(coef_names)
overall_contrast <- numeric(n_total_coef)
overall_contrast[int_idx] <- contrast_weights

# 1. hc1 + CR1S (Stata-style cluster-robust on the full-dummy `lm` design).
#    REFERENCE ONLY — see header: diff-diff's WooldridgeDiD(vcov_type='hc1')
#    uses the within-transformed design with a different (n-1)/(n-k)
#    correction and is NOT pinned at parity against these numbers. The hc1
#    JSON output is retained for diagnostic comparison; tests never assert
#    parity. See REGISTRY "Variance families" → "Deviation from R" for the
#    derivation of the gap.
vcov_cr1s <- vcovCR(fit, cluster = df$unit, type = "CR1S")
se_hc1 <- sqrt(diag(vcov_cr1s)[int_idx])
overall_se_hc1 <- sqrt(
  t(overall_contrast) %*% vcov_cr1s %*% overall_contrast
)[1, 1]

# 2. hc2_bm + CR2 + BM Satterthwaite DOF
vcov_cr2 <- vcovCR(fit, cluster = df$unit, type = "CR2")
se_hc2_bm <- sqrt(diag(vcov_cr2)[int_idx])
coef_test_out <- coef_test(fit, vcov = vcov_cr2, test = "Satterthwaite")
df_satt_hc2_bm <- coef_test_out$df[int_idx]

# Overall ATT BM contrast DOF via Wald_test (HTZ on a 1-row constraint matrix
# reduces to the Satterthwaite t-test; df_denom is the scalar-contrast BM
# DOF). For an arbitrary linear contrast we pass the matrix directly via
# `constraints = matrix(...)`; this is the form Python's
# _compute_cr2_bm_contrast_dof emits for the post-period overall ATT.
constraint_matrix <- matrix(overall_contrast, nrow = 1)
overall_wt <- tryCatch(
  Wald_test(
    fit,
    constraints = constraint_matrix,
    vcov = vcov_cr2,
    test = "HTZ"
  ),
  error = function(e) NULL
)
overall_att_contrast_dof <- if (!is.null(overall_wt)) overall_wt$df_denom else NA_real_

overall_se_hc2_bm <- sqrt(
  t(overall_contrast) %*% vcov_cr2 %*% overall_contrast
)[1, 1]

# 3. classical (lm summary SE — homoskedastic OLS sigma^2 * (X'X)^-1; no
#    robust correction. Python WooldridgeDiD(vcov_type='classical') matches.)
vcov_classical <- vcov(fit)
se_classical <- sqrt(diag(vcov_classical)[int_idx])
overall_se_classical <- sqrt(
  t(overall_contrast) %*% vcov_classical %*% overall_contrast
)[1, 1]

# 4. hc2 (sandwich::vcovHC type="HC2"; no clustering)
vcov_hc2 <- vcovHC(fit, type = "HC2")
se_hc2 <- sqrt(diag(vcov_hc2)[int_idx])
overall_se_hc2 <- sqrt(
  t(overall_contrast) %*% vcov_hc2 %*% overall_contrast
)[1, 1]

# 5. Aggregate hc2_bm BM contrast DOFs for group / calendar / event
# aggregations. These mirror WooldridgeDiDResults.aggregate(...) at fit time:
# each aggregation key gets a 1-row constraint matrix in full-coef space whose
# entries are the per-cell `n_{g,t} / w_total` weights at the (g, t) coefficient
# columns. Compute the BM Satterthwaite DOF via Wald_test(test="HTZ"). diff-diff
# uses lazy contrast-DOF computation in aggregate() with the same algebra;
# pinning here proves R-parity across all three non-simple aggregation surfaces.
build_contrast_for_cells <- function(cells, weights_by_pair) {
  col <- numeric(n_total_coef)
  if (length(cells) == 0L) return(NULL)
  w_total <- sum(vapply(cells, function(p) weights_by_pair[[paste(p, collapse = "_")]], numeric(1)))
  if (w_total == 0) return(NULL)
  for (p in cells) {
    key <- paste(p, collapse = "_")
    cell_w <- weights_by_pair[[key]]
    # find the lm coef index for D_{g}_{t}
    nm <- sprintf("D_%d_%d", p[1], p[2])
    pos <- match(nm, names(coef(fit)))
    if (!is.na(pos)) {
      col[pos] <- cell_w / w_total
    }
  }
  col
}
weights_by_pair <- setNames(as.list(n_gt), vapply(gt_pairs, function(p) paste(p, collapse = "_"), character(1)))

compute_bm_dof_for_contrast <- function(col) {
  if (is.null(col)) return(NA_real_)
  cm <- matrix(col, nrow = 1)
  wt <- tryCatch(
    Wald_test(fit, constraints = cm, vcov = vcov_cr2, test = "HTZ"),
    error = function(e) NULL
  )
  if (is.null(wt)) NA_real_ else wt$df_denom
}

# group: one contrast per treated cohort g, cells = (g, t) for t >= g
agg_group_dofs <- list()
agg_group_keys <- treated_cohorts
for (g in treated_cohorts) {
  cells <- lapply(gt_pairs, function(p) if (p[1] == g && p[2] >= g) p else NULL)
  cells <- Filter(Negate(is.null), cells)
  col <- build_contrast_for_cells(cells, weights_by_pair)
  agg_group_dofs[[as.character(g)]] <- compute_bm_dof_for_contrast(col)
}

# calendar: one contrast per time period t, cells = (g, t) for g > 0 and t >= g
agg_calendar_dofs <- list()
agg_calendar_keys <- times
for (t in times) {
  cells <- lapply(gt_pairs, function(p) if (p[2] == t && p[1] <= t) p else NULL)
  cells <- Filter(Negate(is.null), cells)
  col <- build_contrast_for_cells(cells, weights_by_pair)
  agg_calendar_dofs[[as.character(t)]] <- compute_bm_dof_for_contrast(col)
}

# event: one contrast per relative period k = t - g
all_k <- sort(unique(vapply(gt_pairs, function(p) p[2] - p[1], numeric(1))))
agg_event_dofs <- list()
for (k in all_k) {
  cells <- lapply(gt_pairs, function(p) if ((p[2] - p[1]) == k) p else NULL)
  cells <- Filter(Negate(is.null), cells)
  col <- build_contrast_for_cells(cells, weights_by_pair)
  agg_event_dofs[[as.character(k)]] <- compute_bm_dof_for_contrast(col)
}

# Coefficient point estimates (for cross-check; identical across all 4 variants
# since they share the lm fit).
beta_int <- coef(fit)[int_idx]

golden <- list(
  meta = list(
    panel_csv = panel_path,
    n_obs = nrow(df),
    n_units = length(unique(df$unit)),
    n_periods = length(times),
    cohorts = sort(unique(df$cohort)),
    gt_pairs = lapply(gt_pairs, function(p) list(g = p[1], t = p[2])),
    n_int = n_int,
    n_post_total = n_post_total,
    contrast_weights = contrast_weights,
    clubsandwich_version = as.character(packageVersion("clubSandwich")),
    sandwich_version = as.character(packageVersion("sandwich"))
  ),
  point_estimates = list(
    interaction_coefs = unname(beta_int),
    gt_keys = lapply(gt_pairs, function(p) list(g = p[1], t = p[2]))
  ),
  hc1 = list(
    per_coef_se = unname(se_hc1),
    overall_att_se = overall_se_hc1
  ),
  hc2_bm = list(
    per_coef_se = unname(se_hc2_bm),
    per_coef_df_satt = unname(df_satt_hc2_bm),
    overall_att_se = overall_se_hc2_bm,
    overall_att_contrast_dof = overall_att_contrast_dof,
    aggregate_group_dof = agg_group_dofs,
    aggregate_calendar_dof = agg_calendar_dofs,
    aggregate_event_dof = agg_event_dofs,
    aggregate_event_keys = all_k
  ),
  classical = list(
    per_coef_se = unname(se_classical),
    overall_att_se = overall_se_classical
  ),
  hc2 = list(
    per_coef_se = unname(se_hc2),
    overall_att_se = overall_se_hc2
  )
)

# =============================================================================
# Stage D (PR-B): Poisson + logit R parity via R `etwfe` package.
#
# Generates Poisson + logit outcomes from the existing panel structure with
# a fixed seed, fits `etwfe(family="poisson")` and `etwfe(family="logit")`,
# extracts per-cohort×time ATT coefficients + HC1 SEs, and saves the augmented
# panel back to the same CSV so Python can load the same Y vectors.
#
# Tolerance (Python tests): point ATOL 1e-4, SE ATOL 5e-3. Loose because
# QMLE optimizer paths differ (diff-diff uses direct IRLS via solve_logit /
# solve_poisson; etwfe uses fixest's GLM backend). The HC1 sandwich differs
# by an `(n-1)/(n-k_dm)` vs `(n-1)/(n-k_total)` factor (REGISTRY-documented).
# =============================================================================

set.seed(20260522)
# Treatment indicator: (cohort > 0) & (time >= cohort)
D <- as.integer((df$cohort > 0) & (df$time >= df$cohort))

# Poisson outcome: lambda = exp(0.5 + 0.3 * D)
df$y_pois <- rpois(nrow(df), lambda = exp(0.5 + 0.3 * D))
# Logit outcome: p = plogis(0.0 + 0.8 * D)
df$y_logit <- rbinom(nrow(df), size = 1L, prob = plogis(0.0 + 0.8 * D))

# Save augmented panel back so Python loads the same outcomes. Subset to
# the canonical fixture columns + new nonlinear outcomes to avoid
# polluting the panel CSV with the OLS-stage `D_g_t` working columns
# (codex CI R5 P3 fix — keeps the benchmark input contract narrow).
panel_columns <- c("unit", "time", "cohort", "y", "y_pois", "y_logit")
write.csv(df[, panel_columns], panel_path, row.names = FALSE)
cat(sprintf("Wrote augmented panel with y_pois + y_logit to %s\n", panel_path))

# Fit etwfe(family="poisson")
fit_pois <- etwfe(
  fml = y_pois ~ 1,
  tvar = "time",
  gvar = "cohort",
  data = df,
  family = "poisson",
  vcov = "HC1"
)

# Extract per-cohort-time ATT coefs by name pattern ".Dtreat:cohort::{g}:time::{t}"
extract_etwfe_coefs <- function(fit, gt_pairs) {
  coef_names_fit <- names(coef(fit))
  vcov_fit <- vcov(fit)
  se_diag <- sqrt(diag(vcov_fit))
  out <- list(att = numeric(length(gt_pairs)), se = numeric(length(gt_pairs)),
              gt_keys = list())
  for (i in seq_along(gt_pairs)) {
    g <- gt_pairs[[i]][1]
    t <- gt_pairs[[i]][2]
    nm <- sprintf(".Dtreat:cohort::%d:time::%d", g, t)
    pos <- match(nm, coef_names_fit)
    if (is.na(pos)) {
      # Cell may not be identified (etwfe drops collinear cells)
      out$att[i] <- NA_real_
      out$se[i] <- NA_real_
    } else {
      out$att[i] <- coef(fit)[pos]
      out$se[i] <- se_diag[pos]
    }
    out$gt_keys[[i]] <- list(g = g, t = t)
  }
  out
}

pois_extracted <- extract_etwfe_coefs(fit_pois, gt_pairs)

# Fit etwfe(family="logit")
fit_logit <- etwfe(
  fml = y_logit ~ 1,
  tvar = "time",
  gvar = "cohort",
  data = df,
  family = "logit",
  vcov = "HC1"
)
logit_extracted <- extract_etwfe_coefs(fit_logit, gt_pairs)

golden$poisson <- list(
  per_coef_att = unname(pois_extracted$att),
  per_coef_se = unname(pois_extracted$se),
  gt_keys = pois_extracted$gt_keys,
  etwfe_version = as.character(packageVersion("etwfe"))
)
golden$logit <- list(
  per_coef_att = unname(logit_extracted$att),
  per_coef_se = unname(logit_extracted$se),
  gt_keys = logit_extracted$gt_keys,
  etwfe_version = as.character(packageVersion("etwfe"))
)

write_json(golden, out_path, auto_unbox = TRUE, pretty = TRUE, digits = 18)
cat(sprintf("Wrote %s\n", out_path))
cat(sprintf("  n_obs=%d, n_int=%d, n_units=%d\n",
            nrow(df), n_int, length(unique(df$unit))))
cat(sprintf("  hc1 overall_se=%.10f\n", overall_se_hc1))
cat(sprintf("  hc2_bm overall_se=%.10f, overall_dof=%.4f\n",
            overall_se_hc2_bm, overall_att_contrast_dof))
cat(sprintf("  classical overall_se=%.10f\n", overall_se_classical))
cat(sprintf("  hc2 overall_se=%.10f\n", overall_se_hc2))
cat(sprintf("  poisson ATTs: %s\n",
            paste(round(pois_extracted$att, 4), collapse = ", ")))
cat(sprintf("  logit ATTs: %s\n",
            paste(round(logit_extracted$att, 4), collapse = ", ")))
