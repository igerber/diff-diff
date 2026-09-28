# Generate CR2 Bell-McCaffrey golden values via R clubSandwich.
#
# This script is the parity source for CR2 Bell-McCaffrey cluster-robust
# inference implemented in diff_diff/linalg.py::_compute_cr2_bm.
#
# Usage:
#   Rscript benchmarks/R/generate_clubsandwich_golden.R
#
# Requirements:
#   clubSandwich (CRAN), jsonlite
#
# Output:
#   benchmarks/data/clubsandwich_cr2_golden.json
#
# Phase 1a of the HeterogeneousAdoptionDiD implementation (de Chaisemartin,
# Ciccia, D'Haultfoeuille & Knau 2026, arXiv:2405.04465v6). The parity
# dataset below consists of three small deterministic designs; the Python
# test at tests/test_linalg_hc2_bm.py::TestCR2BMParityClubSandwich loads
# this JSON and checks agreement to 6 digits.

suppressPackageStartupMessages({
  library(clubSandwich)
  library(jsonlite)
})

stopifnot(packageVersion("clubSandwich") >= "0.7.0")
set.seed(20260420)

# --- Three deterministic datasets ---------------------------------------------

make_dataset <- function(name, n_clusters, cluster_sizes, seed) {
  set.seed(seed)
  cluster_ids <- rep(seq_len(n_clusters), times = cluster_sizes)
  n <- length(cluster_ids)
  x <- runif(n, 0, 1)
  # Cluster-level shock to induce within-cluster correlation, plus idiosyncratic noise.
  shock <- rnorm(n_clusters, sd = 0.5)
  y <- 1 + 0.5 * x + shock[cluster_ids] + rnorm(n, sd = 0.2)
  data.frame(name = name, cluster = cluster_ids, x = x, y = y)
}

datasets <- list(
  balanced_small = make_dataset("balanced_small", 5, rep(6, 5), 101),
  unbalanced_medium = make_dataset("unbalanced_medium", 8, c(3, 4, 5, 6, 7, 8, 9, 10), 202),
  singletons_present = make_dataset("singletons_present", 10, c(1, 1, 2, 3, 4, 5, 6, 7, 8, 9), 303)
)

output <- list()

for (nm in names(datasets)) {
  d <- datasets[[nm]]
  fit <- lm(y ~ x, data = d)
  vcov_cr2 <- vcovCR(fit, cluster = d$cluster, type = "CR2")
  # Per-coefficient Bell-McCaffrey Satterthwaite DOF via coef_test()$df_Satt.
  # (clubSandwich 0.7+ removed `Wald_test(..., test="Satterthwaite")`; the
  # `df_Satt` column from coef_test() is the idiomatic per-coefficient form
  # and is numerically identical to the old per-unit-contrast path.)
  ct <- coef_test(fit, vcov = vcov_cr2)
  coef_names <- names(coef(fit))
  output[[nm]] <- list(
    x = d$x,
    y = d$y,
    cluster = d$cluster,
    coef = as.numeric(coef(fit)),
    coef_names = coef_names,
    vcov_cr2 = as.numeric(vcov_cr2),
    vcov_shape = dim(vcov_cr2),
    dof_bm = as.numeric(ct$df_Satt),
    cluster_sizes = as.numeric(table(d$cluster))
  )
}

# --- Absorbed-FE DiD scenario (PR for absorb + hc2/hc2_bm gate lift) ---------
# Canonical DiD-with-FE: y ~ treat_post + factor(unit) + factor(period). The
# Python side uses `DiD(vcov_type="hc2_bm").fit(absorb=["unit","period"])`
# which auto-routes to fixed_effects= internally and builds the same full-
# dummy design as R's `lm()`. R parity targets are computed via the
# singleton-cluster CR2 trick (for HC2-BM one-way) and cluster=unit (for CR2).

make_did_panel <- function(n_units, n_periods, treatment_period, seed) {
  set.seed(seed)
  unit <- rep(seq_len(n_units), each = n_periods)
  period <- rep(seq_len(n_periods), times = n_units)
  treated <- rep(rep(c(0L, 1L), each = n_units / 2L), each = n_periods)
  post <- as.integer(period >= treatment_period)
  treat_post <- treated * post
  unit_fe <- rnorm(n_units, sd = 1.5)
  time_fe <- rnorm(n_periods, sd = 0.5)
  eps <- rnorm(length(unit), sd = 0.3)
  y <- 2.0 * treat_post + unit_fe[unit] + time_fe[period] + eps
  data.frame(unit = factor(unit), period = factor(period),
             treated = treated, post = post, treat_post = treat_post,
             y = y, unit_int = unit, period_int = period)
}

d_did <- make_did_panel(n_units = 8, n_periods = 4, treatment_period = 2, seed = 404)
fit_did <- lm(y ~ treat_post + unit + period, data = d_did)
# HC2 via sandwich::vcovHC(type = "HC2"). Pins the in-tree HC2-parity claim
# the changelog/registry make for the absorb auto-route on the hc2 lane.
vcov_did_hc2 <- sandwich::vcovHC(fit_did, type = "HC2")
# HC2-BM unclustered via singleton-cluster CR2 (PT2018-blessed workaround,
# since clubSandwich::vcovCR requires a cluster arg).
vcov_did_hc2_bm <- vcovCR(fit_did, cluster = seq_len(nrow(d_did)), type = "CR2")
ct_did_hc2_bm <- coef_test(fit_did, vcov = vcov_did_hc2_bm)
# CR2-BM clustered by unit.
vcov_did_cr2 <- vcovCR(fit_did, cluster = d_did$unit, type = "CR2")
ct_did_cr2 <- coef_test(fit_did, vcov = vcov_did_cr2)
output$absorbed_fe_did <- list(
  unit = d_did$unit_int,
  period = d_did$period_int,
  treated = d_did$treated,
  post = d_did$post,
  y = d_did$y,
  coef = as.numeric(coef(fit_did)),
  coef_names = names(coef(fit_did)),
  vcov_hc2 = as.numeric(vcov_did_hc2),
  vcov_hc2_shape = dim(vcov_did_hc2),
  vcov_hc2_bm = as.numeric(vcov_did_hc2_bm),
  vcov_hc2_bm_shape = dim(vcov_did_hc2_bm),
  dof_hc2_bm = as.numeric(ct_did_hc2_bm$df_Satt),
  vcov_cr2 = as.numeric(vcov_did_cr2),
  vcov_cr2_shape = dim(vcov_did_cr2),
  dof_cr2 = as.numeric(ct_did_cr2$df_Satt)
)

# --- Absorbed-FE MultiPeriodDiD event-study scenario (gate lift PR) ----------
# Mirrors MPD(fixed_effects=["unit"]) destination of the absorb auto-route on
# MultiPeriodDiD. MPD parameterization: const + treated + period_f (non-ref)
# + treated:period_X (non-ref) + factor(unit). Build the interaction columns
# explicitly so the R fit's coefficient names match MPD's `treated:period_X`.

make_mpd_panel <- function(n_total, units_per_cohort, n_periods, seed) {
  set.seed(seed)
  d <- expand.grid(unit = seq_len(n_total), period = seq_len(n_periods))
  d$cohort <- ((d$unit - 1L) %/% units_per_cohort) + 1L
  n_cohorts <- n_total %/% units_per_cohort
  # Last cohort is never-treated control; preceding cohorts ever-treated.
  d$treated <- as.integer(d$cohort < n_cohorts)
  d$y <- 1 + 0.5 * d$treated * (d$period >= 3) +
         rnorm(nrow(d), sd = 0.5) +
         0.1 * d$unit + 0.2 * d$period
  d
}

d_mpd <- make_mpd_panel(n_total = 25, units_per_cohort = 5, n_periods = 5,
                        seed = 12345)
d_mpd$period_f <- relevel(factor(d_mpd$period), ref = "1")
# Explicit interaction columns to match MPD's parameterization exactly.
for (p in 2:5) {
  d_mpd[[paste0("treated_period_", p)]] <- d_mpd$treated * (d_mpd$period == p)
}
fit_mpd <- lm(y ~ treated + period_f +
                  treated_period_2 + treated_period_3 +
                  treated_period_4 + treated_period_5 +
                  factor(unit),
              data = d_mpd)
vcov_mpd_hc2 <- sandwich::vcovHC(fit_mpd, type = "HC2")
vcov_mpd_hc2_bm <- vcovCR(fit_mpd, cluster = seq_len(nrow(d_mpd)), type = "CR2")
ct_mpd_hc2_bm <- coef_test(fit_mpd, vcov = vcov_mpd_hc2_bm)
output$mpd_absorbed_fe_did <- list(
  unit = d_mpd$unit,
  period = d_mpd$period,
  treated = d_mpd$treated,
  y = d_mpd$y,
  coef = as.numeric(coef(fit_mpd)),
  coef_names = names(coef(fit_mpd)),
  vcov_hc2 = as.numeric(vcov_mpd_hc2),
  vcov_hc2_shape = dim(vcov_mpd_hc2),
  vcov_hc2_bm = as.numeric(vcov_mpd_hc2_bm),
  vcov_hc2_bm_shape = dim(vcov_mpd_hc2_bm),
  dof_hc2_bm = as.numeric(ct_mpd_hc2_bm$df_Satt),
  reference_period = 1L,
  target_period = 4L
)

# --- MPD clustered avg_att DOF scenario (Gate 6 lift PR) ---------------------
# Pins clubSandwich's compound-contrast Satterthwaite DOF for the post-period-
# average ATT under cluster-robust CR2. Mirrors MultiPeriodDiD(cluster=unit,
# vcov_type='hc2_bm', fixed_effects=['unit']) parameterization. Per-coefficient
# DOFs use coef_test()$df_Satt (the canonical Satterthwaite per-coef API);
# the compound contrast DOF uses Wald_test(constraints=matrix(c_avg, 1),
# test='HTZ')$df_denom — on a 1-row constraint matrix HTZ reduces to a
# Satterthwaite t-test and its df_denom IS the BM Satterthwaite DOF.

d_mpd_cl <- make_mpd_panel(n_total = 15, units_per_cohort = 5, n_periods = 4,
                           seed = 20260517)
d_mpd_cl$period_f <- relevel(factor(d_mpd_cl$period), ref = "1")
for (p in 2:4) {
  d_mpd_cl[[paste0("treated_period_", p)]] <-
    d_mpd_cl$treated * (d_mpd_cl$period == p)
}
fit_mpd_cl <- lm(y ~ treated + period_f +
                     treated_period_2 + treated_period_3 + treated_period_4 +
                     factor(unit),
                 data = d_mpd_cl)
vcov_mpd_cr2 <- vcovCR(fit_mpd_cl, cluster = d_mpd_cl$unit, type = "CR2")
# Per-coefficient DOF via coef_test (canonical Satterthwaite API).
ct_mpd_cr2 <- coef_test(fit_mpd_cl, vcov = vcov_mpd_cr2)
# Compound post-period-average contrast: (1/3) * (e_treated_period_2
# + e_treated_period_3 + e_treated_period_4). Build full-width vector
# matching coef(fit) order, with zeros on the NA-dropped column.
all_coef_names <- names(coef(fit_mpd_cl))
n_coef <- length(all_coef_names)
c_avg_vec <- setNames(rep(0, n_coef), all_coef_names)
post_names <- c("treated_period_2", "treated_period_3", "treated_period_4")
c_avg_vec[post_names] <- 1 / length(post_names)
# Wald_test ignores NA-dropped coefficients; subset the constraint vector
# to the non-NA coefficients (clubSandwich's coef_test convention).
finite_mask <- !is.na(coef(fit_mpd_cl))
c_avg_kept <- c_avg_vec[finite_mask]
dof_avg_compound <- Wald_test(
  fit_mpd_cl,
  constraints = matrix(c_avg_kept, 1),
  vcov = vcov_mpd_cr2,
  test = "HTZ"
)$df_denom
output$mpd_clustered_avg_att_dof <- list(
  unit = d_mpd_cl$unit,
  period = d_mpd_cl$period,
  treated = d_mpd_cl$treated,
  y = d_mpd_cl$y,
  cluster = d_mpd_cl$unit,
  coef = as.numeric(coef(fit_mpd_cl)),
  coef_names = all_coef_names,
  finite_coef_names = all_coef_names[finite_mask],
  vcov_cr2 = as.numeric(vcov_mpd_cr2),
  vcov_cr2_shape = dim(vcov_mpd_cr2),
  dof_per_coef = as.numeric(ct_mpd_cr2$df_Satt),
  c_avg = as.numeric(c_avg_kept),
  dof_avg = unname(dof_avg_compound),
  post_interaction_names = post_names,
  reference_period = 1L,
  n_post_periods = length(post_names)
)

# --- TwoWayFixedEffects HC2 / HC2-BM scenario (Gate 1 lift PR) ---------------
# Mirrors TwoWayFixedEffects(vcov_type in {"hc2","hc2_bm"}) on a 2-period
# panel (binary post indicator). TWFE's `time` parameter is the post
# indicator, so the FE design is factor(unit) + factor(post), NOT
# factor(period). HC2 SE pinned via sandwich::vcovHC; one-way HC2-BM DOF
# via the singleton-cluster CR2 trick (Pustejovsky-Tipton 2018 Section 3.3
# — CR2 with cluster=seq_len(n) reduces to Imbens-Kolesar BM). CR2-BM
# clustered at unit pinned separately for the auto-cluster path.

set.seed(20260518)
n_twfe_units <- 8
n_twfe_periods <- 4
twfe_treated_units <- c(1, 3, 5, 7)
twfe_post_start <- 3
d_twfe <- expand.grid(unit = seq_len(n_twfe_units),
                      period = seq_len(n_twfe_periods))
d_twfe$treated <- as.integer(d_twfe$unit %in% twfe_treated_units)
d_twfe$post <- as.integer(d_twfe$period >= twfe_post_start)
d_twfe$treat_post <- d_twfe$treated * d_twfe$post
twfe_alpha_unit <- rnorm(n_twfe_units, mean = 0, sd = 1)
twfe_gamma_time <- rnorm(n_twfe_periods, mean = 0, sd = 0.5)
d_twfe$y <- 1.0 + 0.7 * d_twfe$treat_post +
            twfe_alpha_unit[d_twfe$unit] +
            twfe_gamma_time[d_twfe$period] +
            rnorm(nrow(d_twfe), sd = 0.4)
fit_twfe <- lm(y ~ treat_post + factor(unit) + factor(post), data = d_twfe)
vcov_twfe_hc2 <- sandwich::vcovHC(fit_twfe, type = "HC2")
# Singleton-cluster CR2 trick for one-way HC2-BM DOF.
vcov_twfe_cr2_one_way <- vcovCR(fit_twfe, cluster = seq_len(nrow(d_twfe)),
                                type = "CR2")
ct_twfe_one_way <- coef_test(fit_twfe, vcov = vcov_twfe_cr2_one_way)
# CR2-BM clustered at unit (the TWFE auto-cluster default).
vcov_twfe_cr2_unit <- vcovCR(fit_twfe, cluster = d_twfe$unit, type = "CR2")
ct_twfe_unit <- coef_test(fit_twfe, vcov = vcov_twfe_cr2_unit)
output$twfe_two_period <- list(
  unit = d_twfe$unit,
  period = d_twfe$period,
  treated = d_twfe$treated,
  post = d_twfe$post,
  treat_post = d_twfe$treat_post,
  y = d_twfe$y,
  coef = as.numeric(coef(fit_twfe)),
  coef_names = names(coef(fit_twfe)),
  vcov_hc2 = as.numeric(vcov_twfe_hc2),
  vcov_hc2_shape = dim(vcov_twfe_hc2),
  vcov_cr2_one_way = as.numeric(vcov_twfe_cr2_one_way),
  dof_bm_one_way = as.numeric(ct_twfe_one_way$df_Satt),
  vcov_cr2_unit = as.numeric(vcov_twfe_cr2_unit),
  dof_bm_unit = as.numeric(ct_twfe_unit$df_Satt)
)

# --- SunAbraham saturated regression HC2 / HC2-BM scenario (Phase 1b PR 1/8) -
# Mirrors SunAbraham(vcov_type in {"classical","hc2","hc2_bm"}) on a
# 5-cohort × 8-period balanced panel. SA's Part G auto-route builds a
# full-dummy saturated design when vcov_type needs the hat matrix —
# matches lm(y ~ D_ge interactions + factor(unit) + factor(period)).
# Targets the (g=4, e=0) cohort × event-time interaction (the canonical
# at-treatment effect of the earliest cohort).

set.seed(42)
n_units_per_cohort <- 8
n_sa_periods <- 8
sa_cohorts <- c(0, 4, 5, 6, 7)  # 0 = never-treated

d_sa <- expand.grid(u_in_cohort = seq_len(n_units_per_cohort),
                    period = seq_len(n_sa_periods),
                    cohort_idx = seq_along(sa_cohorts))
d_sa <- d_sa[order(d_sa$cohort_idx, d_sa$u_in_cohort, d_sa$period), ]
d_sa$unit <- (d_sa$cohort_idx - 1) * n_units_per_cohort + d_sa$u_in_cohort - 1
d_sa$first_treat <- sa_cohorts[d_sa$cohort_idx]
d_sa$time <- d_sa$period
d_sa$rel_time <- ifelse(d_sa$first_treat > 0,
                         d_sa$time - d_sa$first_treat, -999L)
sa_unit_fe <- rnorm(max(d_sa$unit) + 1, mean = 0, sd = 1)
d_sa$treated <- as.integer(d_sa$first_treat > 0 & d_sa$time >= d_sa$first_treat)
d_sa$y <- sa_unit_fe[d_sa$unit + 1] + 0.3 * d_sa$time +
          1.0 * d_sa$treated + rnorm(nrow(d_sa), sd = 0.5)

# Build cohort × event-time interaction columns (excluding ref period -1).
# Sanitize negative event times for R formula compatibility (e=-3 → "n3").
sa_treatment_groups <- sort(unique(d_sa$first_treat[d_sa$first_treat > 0]))
sa_all_rel_times <- sort(unique(d_sa$rel_time[d_sa$first_treat > 0]))
sa_all_rel_times <- sa_all_rel_times[sa_all_rel_times != -1]
sa_interaction_cols <- c()
sa_col_map <- list()
for (g in sa_treatment_groups) {
  for (e in sa_all_rel_times) {
    e_safe <- if (e < 0) paste0("n", abs(e)) else as.character(e)
    col_name <- paste0("D_", g, "_", e_safe)
    original_name <- paste0("D_", g, "_", e)
    ind <- as.integer(d_sa$first_treat == g & d_sa$rel_time == e)
    if (sum(ind) > 0) {
      d_sa[[col_name]] <- ind
      sa_interaction_cols <- c(sa_interaction_cols, col_name)
      sa_col_map[[original_name]] <- col_name
    }
  }
}

sa_target_orig <- "D_4_0"  # the (g=4, e=0) interaction
sa_target_safe <- sa_col_map[[sa_target_orig]]
stopifnot(!is.null(sa_target_safe))

sa_rhs <- paste(c(sa_interaction_cols, "factor(unit)", "factor(time)"),
                collapse = " + ")
fit_sa <- lm(as.formula(paste("y ~", sa_rhs)), data = d_sa)
sa_coef_names <- names(coef(fit_sa))
sa_target_idx <- which(sa_coef_names == sa_target_safe)
stopifnot(length(sa_target_idx) == 1L)

# Extract SE/DOF for the target only (atol=1e-10 pin in Python tests).
sa_classical_se <- summary(fit_sa)$coefficients[sa_target_safe, "Std. Error"]
sa_vcov_hc2 <- sandwich::vcovHC(fit_sa, type = "HC2")
sa_hc2_se <- sqrt(sa_vcov_hc2[sa_target_safe, sa_target_safe])
# Singleton-cluster CR2 reduces to one-way HC2-BM.
sa_vcov_cr2_singleton <- vcovCR(fit_sa, cluster = seq_len(nrow(d_sa)),
                                type = "CR2")
sa_cr2_singleton_se <- sqrt(sa_vcov_cr2_singleton[sa_target_safe,
                                                   sa_target_safe])
sa_ct_singleton <- coef_test(fit_sa, vcov = sa_vcov_cr2_singleton)
sa_dof_bm_singleton <- sa_ct_singleton[sa_target_safe, "df_Satt"]
# CR2-BM clustered at unit (the SA auto-cluster default for hc2_bm).
sa_vcov_cr2_unit <- vcovCR(fit_sa, cluster = d_sa$unit, type = "CR2")
sa_cr2_unit_se <- sqrt(sa_vcov_cr2_unit[sa_target_safe, sa_target_safe])
sa_ct_unit <- coef_test(fit_sa, vcov = sa_vcov_cr2_unit)
sa_dof_bm_unit <- sa_ct_unit[sa_target_safe, "df_Satt"]
# fixest::sunab() parity for SA's HC1 cluster-at-unit default path.
# SA HC1 uses within-transform; fixest also uses within-transform.
# Note: fixest::sunab requires a specific encoding — first_treat=0 means
# never-treated. fixest auto-handles that.
suppressPackageStartupMessages(library(fixest, quietly = TRUE))
fit_sunab <- fixest::feols(
  y ~ sunab(first_treat, time) | unit + time,
  data = d_sa,
  cluster = ~unit
)
# fixest::sunab aggregates to event-study coefficients (IW-aggregated
# across cohorts). The coefficient labels are "time::<event_time>".
# Compare SA's event_study_effects[0] (overall e=0 ATT) against fixest's
# "time::0" event-study SE.
sunab_coef_table <- as.data.frame(summary(fit_sunab)$coeftable)
sunab_target_label <- "time::0"
sunab_hc1_es0_se <- if (sunab_target_label %in% rownames(sunab_coef_table)) {
  sunab_coef_table[sunab_target_label, "Std. Error"]
} else {
  warning("Could not locate fixest sunab event-study target ",
          sunab_target_label)
  NA_real_
}

# CR2-BM Bell-McCaffrey contrast DOF for the IW-aggregated event-time e=0
# effect (under cluster=unit). The contrast at e=0 aggregates all cohorts
# present at relative time 0 with weights w_{g,0} = n_{g,0} / Σ_g n_{g,0}.
# All 4 treated cohorts (g=4,5,6,7) have 8 units each at e=0 → equal
# weights 0.25 each. Build the contrast in full-coef space and call
# Wald_test(test="HTZ") — on a 1-row constraint matrix HTZ reduces to a
# Satterthwaite t-test whose df_denom IS the BM DOF.
sa_all_coef_names <- names(coef(fit_sa))
sa_n_coef <- length(sa_all_coef_names)
sa_es0_contrast <- setNames(rep(0, sa_n_coef), sa_all_coef_names)
sa_es0_cols <- c("D_4_0", "D_5_0", "D_6_0", "D_7_0")
sa_es0_contrast[sa_es0_cols] <- 0.25
# Subset to non-NA coefficients (clubSandwich's convention).
sa_finite_mask <- !is.na(coef(fit_sa))
sa_es0_kept <- sa_es0_contrast[sa_finite_mask]
sa_dof_bm_es0_unit <- Wald_test(
  fit_sa,
  constraints = matrix(sa_es0_kept, 1),
  vcov = sa_vcov_cr2_unit,
  test = "HTZ"
)$df_denom

# CR2-BM Bell-McCaffrey contrast DOF for the IW-aggregated OVERALL ATT.
# SA's overall ATT = Σ_e w_e × Σ_g w_{g,e} × δ_{g,e} where w_e is the
# mass at post-period event-time e and w_{g,e} is the IW cohort share.
# Post-period event-times e ∈ {0, 1, 2, 3} on this panel; n_{g,e} = 8
# for e=0 (all 4 cohorts), 6 for e=1 (3 cohorts), 4 for e=2 (2 cohorts),
# 2 for e=3 (1 cohort) — actually, per fixest::sunab construction:
# cohort g treats at time g; observed event-times for cohort g are
# t - g for t ∈ {1..8}. Compute the cohort × event-time mass matrix
# empirically.
# Post-period event-times: SA includes ALL observed e >= 0, not just
# those where multiple cohorts contribute. For the 4-cohort × 8-period
# panel, max observed e = 8 - 4 = 4 (cohort g=4 at t=8).
sa_post_event_times <- sort(unique(d_sa$rel_time[d_sa$first_treat > 0 & d_sa$rel_time >= 0]))
sa_overall_contrast <- setNames(rep(0, sa_n_coef), sa_all_coef_names)
sa_per_event_mass <- numeric(length(sa_post_event_times))
for (i in seq_along(sa_post_event_times)) {
  e <- sa_post_event_times[i]
  cohorts_at_e <- sort(unique(d_sa$first_treat[d_sa$first_treat > 0 & d_sa$rel_time == e]))
  if (length(cohorts_at_e) == 0) next
  n_per_cohort <- sapply(cohorts_at_e, function(g) sum(d_sa$first_treat == g & d_sa$rel_time == e))
  sa_per_event_mass[i] <- sum(n_per_cohort)
}
sa_post_weights <- sa_per_event_mass / sum(sa_per_event_mass)
for (i in seq_along(sa_post_event_times)) {
  e <- sa_post_event_times[i]
  cohorts_at_e <- sort(unique(d_sa$first_treat[d_sa$first_treat > 0 & d_sa$rel_time == e]))
  if (length(cohorts_at_e) == 0) next
  n_per_cohort <- sapply(cohorts_at_e, function(g) sum(d_sa$first_treat == g & d_sa$rel_time == e))
  iw_weights <- n_per_cohort / sum(n_per_cohort)
  for (j in seq_along(cohorts_at_e)) {
    g <- cohorts_at_e[j]
    e_safe <- if (e < 0) paste0("n", abs(e)) else as.character(e)
    col_name <- paste0("D_", g, "_", e_safe)
    sa_overall_contrast[col_name] <- sa_post_weights[i] * iw_weights[j]
  }
}
sa_overall_kept <- sa_overall_contrast[sa_finite_mask]
sa_dof_bm_overall_unit <- Wald_test(
  fit_sa,
  constraints = matrix(sa_overall_kept, 1),
  vcov = sa_vcov_cr2_unit,
  test = "HTZ"
)$df_denom

output$sun_abraham_two_cohort <- list(
  unit = d_sa$unit,
  time = d_sa$time,
  first_treat = d_sa$first_treat,
  y = d_sa$y,
  target_cohort_g = 4L,
  target_event_time_e = 0L,
  target_col_safe = sa_target_safe,
  classical_se = unname(sa_classical_se),
  hc2_se = unname(sa_hc2_se),
  cr2_bm_singleton_se = unname(sa_cr2_singleton_se),
  dof_bm_singleton = unname(sa_dof_bm_singleton),
  cr2_bm_unit_se = unname(sa_cr2_unit_se),
  dof_bm_unit = unname(sa_dof_bm_unit),
  sunab_hc1_event_study_e0_se = unname(sunab_hc1_es0_se),
  sunab_event_study_target_label = sunab_target_label,
  dof_bm_contrast_es0_unit = unname(sa_dof_bm_es0_unit),
  dof_bm_contrast_overall_unit = unname(sa_dof_bm_overall_unit)
)

# =============================================================================
# Weighted scenarios (clubSandwich WLS-CR2 port)
# =============================================================================
# Pin diff-diff's weighted CR2-BM port against clubSandwich's specific WLS-CR2
# algebra (R/CR-adjustments.R::CR2 + R/get_arrays.R::get_GH + coef_test.R).
# Verified algorithm uses W (not sqrt(W)) in the hat matrix, W^2 in the bias
# correction term, and unweighted residuals in the score construction.

# ---- Scenario: weighted_one_way_no_cluster ----------------------------------
# 12 observations, single "cluster" via vcovCR(cluster=1:n, type="CR2") for the
# one-way HC2-BM reduction. Heteroskedastic weights.
set.seed(50100)
d_w_oneway <- data.frame(
  x = c(-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, -0.8, 0.3, 1.2, -1.2),
  z = c(0.2, -0.3, 0.5, -0.1, 0.4, -0.5, 0.6, -0.2, 0.1, -0.4, 0.3, 0.5)
)
d_w_oneway$y <- 1 + 0.5 * d_w_oneway$x + 0.3 * d_w_oneway$z +
  rnorm(nrow(d_w_oneway), sd = 0.5)
w_oneway <- c(0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2, 2.5, 2.5, 3, 3)
fit_w_oneway <- lm(y ~ x + z, data = d_w_oneway, weights = w_oneway)
vcov_w_oneway <- vcovCR(fit_w_oneway,
                        cluster = seq_len(nrow(d_w_oneway)), type = "CR2")
ct_w_oneway <- coef_test(fit_w_oneway, vcov = vcov_w_oneway)
output$weighted_one_way_no_cluster <- list(
  x = d_w_oneway$x,
  z = d_w_oneway$z,
  y = d_w_oneway$y,
  weights = w_oneway,
  coef = as.numeric(coef(fit_w_oneway)),
  coef_names = names(coef(fit_w_oneway)),
  vcov_hc2_bm = as.numeric(vcov_w_oneway),
  vcov_hc2_bm_shape = dim(vcov_w_oneway),
  dof_bm_one_way = as.numeric(ct_w_oneway$df_Satt),
  se_hc2_bm = as.numeric(ct_w_oneway$SE)
)

# ---- Scenario: weighted_balanced_clusters -----------------------------------
# 20 observations, 4 clusters of 5; weights vary within and across clusters.
set.seed(50200)
n_w_bal <- 20
cluster_w_bal <- rep(1:4, each = 5)
d_w_bal <- data.frame(
  cluster = cluster_w_bal,
  x = rnorm(n_w_bal),
  z = rnorm(n_w_bal)
)
d_w_bal$y <- 1 + 0.5 * d_w_bal$x + 0.3 * d_w_bal$z + rnorm(n_w_bal, sd = 0.5)
w_bal <- runif(n_w_bal, min = 0.5, max = 3.0)
fit_w_bal <- lm(y ~ x + z, data = d_w_bal, weights = w_bal)
vcov_w_bal <- vcovCR(fit_w_bal, cluster = d_w_bal$cluster, type = "CR2")
ct_w_bal <- coef_test(fit_w_bal, vcov = vcov_w_bal)
output$weighted_balanced_clusters <- list(
  x = d_w_bal$x,
  z = d_w_bal$z,
  y = d_w_bal$y,
  weights = w_bal,
  cluster = d_w_bal$cluster,
  coef = as.numeric(coef(fit_w_bal)),
  coef_names = names(coef(fit_w_bal)),
  vcov_cr2 = as.numeric(vcov_w_bal),
  vcov_cr2_shape = dim(vcov_w_bal),
  dof_bm = as.numeric(ct_w_bal$df_Satt),
  se_cr2 = as.numeric(ct_w_bal$SE)
)

# ---- Scenario: weighted_unbalanced_clusters ---------------------------------
# 52 observations, 8 clusters of sizes 3-10. Heteroskedastic weights.
set.seed(50300)
cluster_sizes_unbal <- c(3, 4, 5, 6, 7, 8, 9, 10)
cluster_w_unbal <- rep(1:8, times = cluster_sizes_unbal)
n_w_unbal <- length(cluster_w_unbal)
d_w_unbal <- data.frame(
  cluster = cluster_w_unbal,
  x = rnorm(n_w_unbal),
  z = rnorm(n_w_unbal)
)
shock_unbal <- rnorm(8, sd = 0.3)
d_w_unbal$y <- 1 + 0.5 * d_w_unbal$x + 0.3 * d_w_unbal$z +
  shock_unbal[d_w_unbal$cluster] + rnorm(n_w_unbal, sd = 0.4)
w_unbal <- runif(n_w_unbal, min = 0.3, max = 3.0)
fit_w_unbal <- lm(y ~ x + z, data = d_w_unbal, weights = w_unbal)
vcov_w_unbal <- vcovCR(fit_w_unbal, cluster = d_w_unbal$cluster, type = "CR2")
ct_w_unbal <- coef_test(fit_w_unbal, vcov = vcov_w_unbal)
output$weighted_unbalanced_clusters <- list(
  x = d_w_unbal$x,
  z = d_w_unbal$z,
  y = d_w_unbal$y,
  weights = w_unbal,
  cluster = d_w_unbal$cluster,
  coef = as.numeric(coef(fit_w_unbal)),
  coef_names = names(coef(fit_w_unbal)),
  vcov_cr2 = as.numeric(vcov_w_unbal),
  vcov_cr2_shape = dim(vcov_w_unbal),
  dof_bm = as.numeric(ct_w_unbal$df_Satt),
  se_cr2 = as.numeric(ct_w_unbal$SE),
  cluster_sizes = as.numeric(table(d_w_unbal$cluster))
)

# ---- Scenario: weighted_singletons_present ----------------------------------
# Adversarial: prior PT2018 transform-once derivation hit ~30% gap on
# singleton-cluster scenarios. Verifies the clubSandwich port handles this.
set.seed(50400)
cluster_sizes_sing <- c(1, 1, 2, 3, 4, 5, 6, 6, 4, 3)
cluster_w_sing <- rep(1:10, times = cluster_sizes_sing)
n_w_sing <- length(cluster_w_sing)
d_w_sing <- data.frame(
  cluster = cluster_w_sing,
  x = rnorm(n_w_sing),
  z = rnorm(n_w_sing)
)
shock_sing <- rnorm(10, sd = 0.3)
d_w_sing$y <- 1 + 0.5 * d_w_sing$x + 0.3 * d_w_sing$z +
  shock_sing[d_w_sing$cluster] + rnorm(n_w_sing, sd = 0.4)
w_sing <- runif(n_w_sing, min = 0.3, max = 3.0)
fit_w_sing <- lm(y ~ x + z, data = d_w_sing, weights = w_sing)
vcov_w_sing <- vcovCR(fit_w_sing, cluster = d_w_sing$cluster, type = "CR2")
ct_w_sing <- coef_test(fit_w_sing, vcov = vcov_w_sing)
output$weighted_singletons_present <- list(
  x = d_w_sing$x,
  z = d_w_sing$z,
  y = d_w_sing$y,
  weights = w_sing,
  cluster = d_w_sing$cluster,
  coef = as.numeric(coef(fit_w_sing)),
  coef_names = names(coef(fit_w_sing)),
  vcov_cr2 = as.numeric(vcov_w_sing),
  vcov_cr2_shape = dim(vcov_w_sing),
  dof_bm = as.numeric(ct_w_sing$df_Satt),
  dof_per_coef = as.numeric(ct_w_sing$df_Satt),
  se_cr2 = as.numeric(ct_w_sing$SE),
  cluster_sizes = as.numeric(table(d_w_sing$cluster))
)

# ---- Scenario: weighted_did_absorbed_fe -------------------------------------
# DiD-style analytical CR2 design-matrix parity fixture: 8 units x 4 periods,
# treat_post + unit + period FE, analytics weights varying by unit. Pins the
# clubSandwich WLS-CR2 analytical surface (compute_robust_vcov / solve_ols /
# LinearRegression direct callers) on a DiD-shaped design. NOTE: this is NOT
# a public-estimator `DifferenceInDifferences(survey_design=...)` parity
# fixture — estimator-level survey designs route through the TSL survey
# variance path, which takes precedence over the analytical CR2 sandwich.
set.seed(50500)
d_did_w <- make_did_panel(n_units = 8, n_periods = 4, treatment_period = 2,
                          seed = 50501)
# Unit-level weight (stratum-like): vary by unit, constant within unit-period.
unit_w_did <- runif(8, min = 0.5, max = 2.5)
d_did_w$weights <- unit_w_did[d_did_w$unit_int]
fit_did_w <- lm(y ~ treat_post + unit + period, data = d_did_w,
                weights = d_did_w$weights)
vcov_did_w_cr2 <- vcovCR(fit_did_w, cluster = d_did_w$unit_int, type = "CR2")
ct_did_w_cr2 <- coef_test(fit_did_w, vcov = vcov_did_w_cr2)
output$weighted_did_absorbed_fe <- list(
  unit = d_did_w$unit_int,
  period = d_did_w$period_int,
  treated = d_did_w$treated,
  post = d_did_w$post,
  treat_post = d_did_w$treat_post,
  y = d_did_w$y,
  weights = d_did_w$weights,
  coef = as.numeric(coef(fit_did_w)),
  coef_names = names(coef(fit_did_w)),
  vcov_cr2 = as.numeric(vcov_did_w_cr2),
  vcov_cr2_shape = dim(vcov_did_w_cr2),
  dof_cr2 = as.numeric(ct_did_w_cr2$df_Satt)
)

# ---- Scenario: weighted_mpd_avg_att_dof -------------------------------------
# MPD-style analytical CR2 contrast-DOF parity fixture: 15 units x 4 periods,
# MPD parameterization with analytics weights + cluster=unit. Compound
# contrast = post-period-average ATT. Pins the clubSandwich `Wald_test(test=
# "HTZ")$df_denom` against `_compute_cr2_bm_contrast_dof(weights=)` on an
# MPD-shaped design. NOTE: like `weighted_did_absorbed_fe`, this fixture
# targets the analytical surface only — estimator-level
# `MultiPeriodDiD(survey_design=...)` paths use TSL.
set.seed(50600)
d_mpd_w <- make_mpd_panel(n_total = 15, units_per_cohort = 5, n_periods = 4,
                          seed = 50601)
d_mpd_w$period_f <- relevel(factor(d_mpd_w$period), ref = "1")
for (p in 2:4) {
  d_mpd_w[[paste0("treated_period_", p)]] <-
    d_mpd_w$treated * (d_mpd_w$period == p)
}
unit_w_mpd <- runif(15, min = 0.5, max = 2.5)
d_mpd_w$weights <- unit_w_mpd[d_mpd_w$unit]
fit_mpd_w <- lm(y ~ treated + period_f +
                    treated_period_2 + treated_period_3 + treated_period_4 +
                    factor(unit),
                data = d_mpd_w, weights = d_mpd_w$weights)
vcov_mpd_w_cr2 <- vcovCR(fit_mpd_w, cluster = d_mpd_w$unit, type = "CR2")
ct_mpd_w_cr2 <- coef_test(fit_mpd_w, vcov = vcov_mpd_w_cr2)
# Compound contrast: post-period-average over treated_period_{2,3,4}.
all_coef_names_w <- names(coef(fit_mpd_w))
n_coef_w <- length(all_coef_names_w)
c_avg_vec_w <- setNames(rep(0, n_coef_w), all_coef_names_w)
post_names_w <- c("treated_period_2", "treated_period_3", "treated_period_4")
c_avg_vec_w[post_names_w] <- 1 / length(post_names_w)
finite_mask_w <- !is.na(coef(fit_mpd_w))
c_avg_kept_w <- c_avg_vec_w[finite_mask_w]
dof_avg_w <- Wald_test(
  fit_mpd_w,
  constraints = matrix(c_avg_kept_w, 1),
  vcov = vcov_mpd_w_cr2,
  test = "HTZ"
)$df_denom
output$weighted_mpd_avg_att_dof <- list(
  unit = d_mpd_w$unit,
  period = d_mpd_w$period,
  treated = d_mpd_w$treated,
  y = d_mpd_w$y,
  weights = d_mpd_w$weights,
  cluster = d_mpd_w$unit,
  coef = as.numeric(coef(fit_mpd_w)),
  coef_names = all_coef_names_w,
  finite_coef_names = all_coef_names_w[finite_mask_w],
  vcov_cr2 = as.numeric(vcov_mpd_w_cr2),
  vcov_cr2_shape = dim(vcov_mpd_w_cr2),
  dof_per_coef = as.numeric(ct_mpd_w_cr2$df_Satt),
  c_avg = as.numeric(c_avg_kept_w),
  dof_avg = unname(dof_avg_w),
  post_interaction_names = post_names_w,
  reference_period = 1L,
  n_post_periods = length(post_names_w)
)

output$meta <- list(
  source = "clubSandwich",
  clubSandwich_version = as.character(packageVersion("clubSandwich")),
  R_version = R.version.string,
  generated_at = format(Sys.time(), tz = "UTC", usetz = TRUE),
  note = paste0(
    "CR2 Bell-McCaffrey cluster-robust parity target for ",
    "diff_diff._compute_cr2_bm. Unweighted scenarios pin against ",
    "_compute_cr2_bm / _compute_bm_dof_oneway; weighted scenarios pin ",
    "the clubSandwich WLS-CR2 port (W not sqrt(W), W^2 bias term, ",
    "unweighted residuals)."
  )
)

out_path <- file.path("benchmarks", "data", "clubsandwich_cr2_golden.json")
writeLines(toJSON(output, pretty = TRUE, digits = 15, auto_unbox = TRUE), out_path)
cat("Wrote", out_path, "\n")
