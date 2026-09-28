#!/usr/bin/env Rscript
# benchmark_survey_estimators.R
#
# Generate golden values for 4 survey-enabled estimators:
#   S1: ImputationDiD   — control-only WLS regression
#   S2: StackedDiD      — stacked WLS with Q-weight x survey weight composition
#   S3: SunAbraham      — interaction-weighted ATT
#   S4: TripleDifference — three-way interaction DDD
#
# Usage: Rscript benchmarks/R/benchmark_survey_estimators.R

suppressPackageStartupMessages({
  library(survey)
  library(jsonlite)
})

output_dir <- file.path("benchmarks", "data", "synthetic")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

cat("=== Survey Estimator Validation Benchmark ===\n\n")

# ===========================================================================
# DGP 1: Staggered panel (shared by S1, S2, S3)
# ===========================================================================

generate_staggered_survey_data <- function(seed = 42) {
  set.seed(seed)

  # 4 strata with unequal sizes
  strata_sizes <- c(25L, 30L, 45L, 50L)  # 150 units total
  n_strata <- length(strata_sizes)
  n_units <- sum(strata_sizes)

  # PSU assignment: 2-3 PSUs per stratum
  psu_per_stratum <- c(2L, 2L, 3L, 3L)  # 10 PSUs total
  fpc_values <- c(100, 150, 200, 250)

  # 5 time periods
  T_max <- 5L

  # Build unit-level data
  unit_id <- integer(0)
  stratum_vec <- integer(0)
  psu_vec <- integer(0)
  fpc_vec <- numeric(0)
  weight_vec <- numeric(0)
  first_treat_vec <- numeric(0)
  unit_effect_vec <- numeric(0)

  global_psu_id <- 0L
  unit_counter <- 0L

  for (h in seq_len(n_strata)) {
    n_h <- strata_sizes[h]
    n_psu_h <- psu_per_stratum[h]
    fpc_h <- fpc_values[h]
    w_h <- fpc_h / n_h  # inverse selection probability weight

    # Assign units to PSUs within stratum (roughly equal)
    psu_assignment <- rep(seq_len(n_psu_h), length.out = n_h)

    for (i in seq_len(n_h)) {
      unit_counter <- unit_counter + 1L

      # Staggered treatment: ~30% first_treat=3, ~20% first_treat=4, ~50% never
      frac <- i / n_h
      if (frac <= 0.3) {
        ft <- 3
      } else if (frac <= 0.5) {
        ft <- 4
      } else {
        ft <- Inf
      }

      unit_id <- c(unit_id, unit_counter)
      stratum_vec <- c(stratum_vec, h)
      psu_vec <- c(psu_vec, global_psu_id + psu_assignment[i])
      fpc_vec <- c(fpc_vec, fpc_h)
      weight_vec <- c(weight_vec, w_h)
      first_treat_vec <- c(first_treat_vec, ft)
      unit_effect_vec <- c(unit_effect_vec, rnorm(1, 0, 2))
    }
    global_psu_id <- global_psu_id + n_psu_h
  }

  # Expand to panel: 5 periods
  n_obs <- n_units * T_max
  panel <- data.frame(
    unit     = rep(unit_id, each = T_max),
    period   = rep(seq_len(T_max), times = n_units),
    stratum  = rep(stratum_vec, each = T_max),
    psu      = rep(psu_vec, each = T_max),
    fpc      = rep(fpc_vec, each = T_max),
    weight   = rep(weight_vec, each = T_max),
    first_treat = rep(first_treat_vec, each = T_max)
  )

  # Treatment indicator D_it = 1(period >= first_treat & first_treat < Inf)
  panel$D_it <- as.integer(panel$period >= panel$first_treat & is.finite(panel$first_treat))

  # Covariates: TIME-VARYING (unit-level base + period-specific noise)
  # Time-invariant covariates are collinear with unit FE, so we need variation
  # within units across time for meaningful regression coefficients.
  treated_ever <- as.integer(is.finite(rep(first_treat_vec, each = T_max)))
  stratum_expanded <- rep(stratum_vec, each = T_max)
  panel$x1 <- rnorm(n_obs) + 0.5 * treated_ever + 0.3 * panel$period
  panel$x2 <- as.integer(runif(n_obs) < (0.3 + 0.05 * stratum_expanded + 0.05 * panel$period))

  # Outcome: Y_it = 10 + alpha_i + 5*period + 5*D_it + 2*x1 + 1.5*x2 + noise
  ue_expanded <- rep(unit_effect_vec, each = T_max)
  noise <- rnorm(n_obs, 0, 1)
  panel$outcome <- 10 + ue_expanded + 5 * panel$period +
    5 * panel$D_it + 2 * panel$x1 + 1.5 * panel$x2 + noise

  panel
}

# ===========================================================================
# DGP 2: DDD cross-section (for S4)
# ===========================================================================

generate_ddd_survey_data <- function(seed = 42) {
  set.seed(seed)

  n <- 200L
  n_per_cell <- 25L  # 8 cells x 25 = 200

  # Strata orthogonal to treatment cells
  strata_sizes <- c(40L, 45L, 55L, 60L)
  psu_per_stratum <- c(2L, 2L, 3L, 3L)
  fpc_values <- c(100, 120, 150, 180)

  # Build strata/PSU assignment for all 200 obs (ordered by strata)
  stratum_all <- integer(0)
  psu_all <- integer(0)
  fpc_all <- numeric(0)
  weight_all <- numeric(0)
  global_psu_id <- 0L

  for (h in seq_along(strata_sizes)) {
    n_h <- strata_sizes[h]
    n_psu_h <- psu_per_stratum[h]
    fpc_h <- fpc_values[h]
    w_h <- fpc_h / n_h

    psu_assignment <- rep(seq_len(n_psu_h), length.out = n_h)
    stratum_all <- c(stratum_all, rep(h, n_h))
    psu_all <- c(psu_all, global_psu_id + psu_assignment)
    fpc_all <- c(fpc_all, rep(fpc_h, n_h))
    weight_all <- c(weight_all, rep(w_h, n_h))
    global_psu_id <- global_psu_id + n_psu_h
  }

  # Build treatment cells: G x P x T, 25 per cell
  cells <- expand.grid(G = 0:1, P = 0:1, Ti = 0:1)
  group_vec <- integer(0)
  partition_vec <- integer(0)
  time_vec <- integer(0)
  for (r in seq_len(nrow(cells))) {
    group_vec <- c(group_vec, rep(cells$G[r], n_per_cell))
    partition_vec <- c(partition_vec, rep(cells$P[r], n_per_cell))
    time_vec <- c(time_vec, rep(cells$Ti[r], n_per_cell))
  }

  # Shuffle to break any ordering correlation, then assign strata
  shuffle_idx <- sample(n)
  group_vec <- group_vec[shuffle_idx]
  partition_vec <- partition_vec[shuffle_idx]
  time_vec <- time_vec[shuffle_idx]

  # Outcome: Y = 10 + 2*G + 1.5*P + 3*T + 1*G*T + 1*P*T + 0.5*G*P + 3*G*P*T + noise
  noise <- rnorm(n, 0, 1)
  outcome <- 10 + 2 * group_vec + 1.5 * partition_vec + 3 * time_vec +
    1 * group_vec * time_vec + 1 * partition_vec * time_vec +
    0.5 * group_vec * partition_vec +
    3 * group_vec * partition_vec * time_vec + noise

  data.frame(
    obs_id    = seq_len(n),
    group     = group_vec,
    partition = partition_vec,
    time      = time_vec,
    outcome   = outcome,
    stratum   = stratum_all,
    psu       = psu_all,
    fpc       = fpc_all,
    weight    = weight_all
  )
}

# ===========================================================================
# Helper: extract svyglm results for a named coefficient
# ===========================================================================

extract_svyglm_results <- function(model, design, coef_name) {
  coefs <- coef(model)
  ses <- SE(model)
  idx <- which(names(coefs) == coef_name)
  if (length(idx) == 0) stop(paste0("Coefficient '", coef_name, "' not found in model"))

  att <- as.numeric(coefs[idx])
  se <- as.numeric(ses[idx])
  df <- degf(design)

  # NaN-all-or-nothing guard: if SE is non-finite or non-positive,
  # all derived inference fields must be NaN.
  if (!is.finite(se) || se <= 0) {
    t_stat <- NaN
    ci_lower <- NaN
    ci_upper <- NaN
  } else {
    t_stat <- att / se
    t_crit <- qt(0.975, df)
    ci_lower <- att - t_crit * se
    ci_upper <- att + t_crit * se
  }

  list(att = att, se = se, t_stat = t_stat, df = df,
       ci_lower = ci_lower, ci_upper = ci_upper)
}

# ===========================================================================
# Generate data
# ===========================================================================

cat("Generating staggered panel data...\n")
staggered_data <- generate_staggered_survey_data(seed = 42)
cat(sprintf("  %d obs, %d units, %d periods\n",
            nrow(staggered_data), length(unique(staggered_data$unit)),
            length(unique(staggered_data$period))))
cat(sprintf("  Cohorts: first_treat=3: %d units, first_treat=4: %d units, never: %d units\n",
            length(unique(staggered_data$unit[staggered_data$first_treat == 3])),
            length(unique(staggered_data$unit[staggered_data$first_treat == 4])),
            length(unique(staggered_data$unit[is.infinite(staggered_data$first_treat)]))))

cat("\nGenerating DDD data...\n")
ddd_data <- generate_ddd_survey_data(seed = 42)
cat(sprintf("  %d obs\n", nrow(ddd_data)))

results <- list()

# ===========================================================================
# S1: ImputationDiD — control-only WLS regression
# ===========================================================================

cat("\n--- S1: ImputationDiD control-only regression ---\n")

# Omega_0: all untreated observations
# Never-treated at all periods + pre-treatment obs of eventually-treated units
omega_0 <- staggered_data[is.infinite(staggered_data$first_treat) |
                          staggered_data$period < staggered_data$first_treat, ]
cat(sprintf("  Omega_0: %d observations\n", nrow(omega_0)))

design_s1 <- svydesign(ids = ~psu, strata = ~stratum, fpc = ~fpc,
                        weights = ~weight, data = omega_0)

fit_s1 <- svyglm(outcome ~ factor(unit) + factor(period) + x1 + x2,
                  design = design_s1)

# Extract covariate coefficients
coefs_s1 <- coef(fit_s1)
ses_s1 <- SE(fit_s1)
df_s1 <- degf(design_s1)

results$s1_imputation_did <- list(
  scenario = "S1",
  description = "ImputationDiD: control-only WLS regression on Omega_0",
  estimator = "survey::svyglm",
  coef_x1 = as.numeric(coefs_s1["x1"]),
  se_x1 = as.numeric(ses_s1["x1"]),
  coef_x2 = as.numeric(coefs_s1["x2"]),
  se_x2 = as.numeric(ses_s1["x2"]),
  df = df_s1,
  n_obs_omega0 = nrow(omega_0)
)

cat(sprintf("  coef_x1 = %.6f (SE = %.6f)\n",
            results$s1_imputation_did$coef_x1, results$s1_imputation_did$se_x1))
cat(sprintf("  coef_x2 = %.6f (SE = %.6f)\n",
            results$s1_imputation_did$coef_x2, results$s1_imputation_did$se_x2))
cat(sprintf("  df = %d\n", df_s1))

# ===========================================================================
# S2: StackedDiD — stacked WLS with Q-weight x survey weight composition
# ===========================================================================

cat("\n--- S2: StackedDiD (stacking + Q-weight + composed weights) ---\n")

kappa_pre <- 1L
kappa_post <- 1L
cohorts <- c(3, 4)  # adoption events

# Step 1: Build sub-experiments
sub_experiments <- list()
for (a in cohorts) {
  # Treated units: first_treat == a
  treated_units <- unique(staggered_data$unit[staggered_data$first_treat == a])
  # Control units: never-treated
  control_units <- unique(staggered_data$unit[is.infinite(staggered_data$first_treat)])

  if (length(treated_units) == 0 || length(control_units) == 0) next

  # Time window: [a - kappa_pre, a + kappa_post]
  t_start <- a - kappa_pre
  t_end <- a + kappa_post

  all_units <- c(treated_units, control_units)
  sub_df <- staggered_data[staggered_data$unit %in% all_units &
                           staggered_data$period >= t_start &
                           staggered_data$period <= t_end, ]

  if (nrow(sub_df) == 0) next

  sub_df$sub_exp <- a
  sub_df$event_time <- sub_df$period - a
  sub_df$D_sa <- as.integer(sub_df$unit %in% treated_units)

  sub_experiments[[as.character(a)]] <- sub_df
}

stacked <- do.call(rbind, sub_experiments)
rownames(stacked) <- NULL
n_stacked <- nrow(stacked)
cat(sprintf("  Stacked dataset: %d observations\n", n_stacked))

# Step 2: Compute Q-weights (sample_share)
# Count distinct units per sub-experiment
N_D <- sapply(sub_experiments, function(df) length(unique(df$unit[df$D_sa == 1])))
N_C <- sapply(sub_experiments, function(df) length(unique(df$unit[df$D_sa == 0])))
N_Omega_D <- sum(N_D)
N_Omega_C <- sum(N_C)
N_grand <- N_Omega_D + N_Omega_C

cat(sprintf("  N_D per sub-exp: %s\n", paste(N_D, collapse = ", ")))
cat(sprintf("  N_C per sub-exp: %s\n", paste(N_C, collapse = ", ")))
cat(sprintf("  N_Omega_D = %d, N_Omega_C = %d, N_grand = %d\n",
            N_Omega_D, N_Omega_C, N_grand))

q_control <- setNames(numeric(length(cohorts)), as.character(cohorts))
for (a_str in names(N_D)) {
  n_c <- N_C[a_str]
  n_d <- N_D[a_str]
  if (n_c == 0 || N_Omega_C == 0) {
    q_control[a_str] <- 1.0
  } else {
    control_share <- n_c / N_Omega_C
    sample_share <- (n_d + n_c) / N_grand
    q_control[a_str] <- sample_share / control_share
  }
}
cat(sprintf("  Q-weights (control): %s\n",
            paste(sprintf("%s=%.6f", names(q_control), q_control), collapse = ", ")))

# Assign Q-weights: treated=1, control=q_control[sub_exp]
stacked$Q_weight <- 1.0
for (a_str in names(q_control)) {
  mask <- stacked$sub_exp == as.numeric(a_str) & stacked$D_sa == 0
  stacked$Q_weight[mask] <- q_control[a_str]
}

# Step 3: Compose Q x survey weight, normalize
stacked$composed_weight <- stacked$Q_weight * stacked$weight
w_sum <- sum(stacked$composed_weight)
if (w_sum > 0) {
  stacked$composed_weight <- stacked$composed_weight * (n_stacked / w_sum)
}

# Step 4: Fit svyglm
# Use strata/PSU structure from original data (propagated into stacked data)
# to match Python's TSL variance on the re-resolved stacked survey design.
# Omit FPC — the stacked data inflates the sample size per stratum due to
# unit duplication across sub-experiments, which can make n_h > N_h.
stacked$event_time_fac <- relevel(factor(stacked$event_time), ref = "-1")
design_s2 <- svydesign(ids = ~psu, strata = ~stratum,
                        weights = ~composed_weight, data = stacked)

fit_s2 <- svyglm(outcome ~ D_sa + event_time_fac + D_sa:event_time_fac,
                  design = design_s2)

coefs_s2 <- coef(fit_s2)
ses_s2 <- SE(fit_s2)
V_s2 <- vcov(fit_s2)
df_s2 <- degf(design_s2)

# Extract event-study interaction coefficients (D_sa:event_time_fac*)
interaction_names <- grep("^D_sa:event_time_fac", names(coefs_s2), value = TRUE)
event_study_s2 <- list()
post_coef_indices <- integer(0)

for (nm in interaction_names) {
  # Parse event time from name like "D_sa:event_time_fac0" or "D_sa:event_time_fac1"
  e <- as.integer(sub("^D_sa:event_time_fac", "", nm))
  event_study_s2[[as.character(e)]] <- list(
    coef = as.numeric(coefs_s2[nm]),
    se = as.numeric(ses_s2[nm])
  )
  if (e >= 0) {
    post_coef_indices <- c(post_coef_indices, which(names(coefs_s2) == nm))
  }
}

# ATT = mean of post-treatment interaction coefficients
post_coefs <- coefs_s2[post_coef_indices]
n_post <- length(post_coefs)
att_s2 <- mean(post_coefs)

# SE via delta method: ATT = (1/n_post) * sum(post_coefs)
# var(ATT) = (1/n_post)^2 * ones' V_sub ones
V_sub_s2 <- V_s2[post_coef_indices, post_coef_indices]
ones <- rep(1, n_post)
se_s2 <- sqrt(as.numeric(t(ones) %*% V_sub_s2 %*% ones)) / n_post

results$s2_stacked_did <- list(
  scenario = "S2",
  description = "StackedDiD: stacked WLS with Q-weight x survey weight composition",
  estimator = "survey::svyglm",
  att = att_s2,
  se = se_s2,
  n_stacked = n_stacked,
  df = df_s2,
  event_study = event_study_s2
)

cat(sprintf("  ATT = %.6f (SE = %.6f)\n", att_s2, se_s2))
cat(sprintf("  Event-study effects: %s\n",
            paste(sapply(names(event_study_s2), function(e)
              sprintf("e=%s: %.4f", e, event_study_s2[[e]]$coef)), collapse = ", ")))

# ===========================================================================
# S3: SunAbraham — interaction-weighted ATT
# ===========================================================================

cat("\n--- S3: SunAbraham (IW-aggregated ATT) ---\n")

# Create cohort x relative-time interaction dummies
# Cohort 3: rel_times = {-2, -1, 0, 1, 2} -> exclude -1 -> {-2, 0, 1, 2}
# Cohort 4: rel_times = {-3, -2, -1, 0, 1} -> exclude -1 -> {-3, -2, 0, 1}
staggered_data$rel_time_3 <- ifelse(staggered_data$first_treat == 3,
                                     staggered_data$period - 3, NA)
staggered_data$rel_time_4 <- ifelse(staggered_data$first_treat == 4,
                                     staggered_data$period - 4, NA)

# Build interaction dummies
interaction_info <- list(
  list(g = 3, e = -2, name = "D_3_m2"),
  list(g = 3, e =  0, name = "D_3_0"),
  list(g = 3, e =  1, name = "D_3_1"),
  list(g = 3, e =  2, name = "D_3_2"),
  list(g = 4, e = -3, name = "D_4_m3"),
  list(g = 4, e = -2, name = "D_4_m2"),
  list(g = 4, e =  0, name = "D_4_0"),
  list(g = 4, e =  1, name = "D_4_1")
)

for (info in interaction_info) {
  staggered_data[[info$name]] <- as.integer(
    staggered_data$first_treat == info$g &
    (staggered_data$period - info$g) == info$e
  )
}

# Verify non-zero support for each interaction
for (info in interaction_info) {
  n_ones <- sum(staggered_data[[info$name]])
  cat(sprintf("  %s: %d observations\n", info$name, n_ones))
}

design_s3 <- svydesign(ids = ~psu, strata = ~stratum, fpc = ~fpc,
                        weights = ~weight, data = staggered_data)

# Formula: outcome ~ factor(unit) + factor(period) + all interaction dummies
interaction_names_s3 <- sapply(interaction_info, function(x) x$name)
formula_rhs <- paste(c("factor(unit)", "factor(period)", interaction_names_s3),
                     collapse = " + ")
formula_s3 <- as.formula(paste("outcome ~", formula_rhs))

fit_s3 <- svyglm(formula_s3, design = design_s3)

coefs_s3 <- coef(fit_s3)
ses_s3 <- SE(fit_s3)
V_s3 <- vcov(fit_s3)
df_s3 <- degf(design_s3)

# Extract cohort effects
cohort_effects_s3 <- list()
for (info in interaction_info) {
  nm <- info$name
  cohort_effects_s3[[nm]] <- list(
    coef = as.numeric(coefs_s3[nm]),
    se = as.numeric(ses_s3[nm])
  )
  cat(sprintf("  delta(%d, %d) = %.6f (SE = %.6f)\n",
              info$g, info$e, coefs_s3[nm], ses_s3[nm]))
}

# --- IW Aggregation ---
# Post-treatment interactions (e >= 0)
post_info <- Filter(function(x) x$e >= 0, interaction_info)

# Compute survey-weighted mass n_{g,e} for each post-treatment (g, e) pair
post_masses <- list()
for (info in post_info) {
  mask <- staggered_data$first_treat == info$g &
          (staggered_data$period - info$g) == info$e
  n_ge <- sum(staggered_data$weight[mask])
  post_masses[[info$name]] <- list(g = info$g, e = info$e, mass = n_ge,
                                    coef_name = info$name)
}

# Get unique post-treatment event times
post_e_vals <- sort(unique(sapply(post_info, function(x) x$e)))

# Per-period IW aggregation: beta_e = sum_g w_{g,e} * delta_{g,e}
period_effects <- list()
for (e in post_e_vals) {
  pairs_at_e <- Filter(function(x) x$e == e, post_masses)
  masses_at_e <- sapply(pairs_at_e, function(x) x$mass)
  total_mass_at_e <- sum(masses_at_e)

  if (total_mass_at_e == 0) next

  beta_e <- 0.0
  for (p in pairs_at_e) {
    w_ge <- p$mass / total_mass_at_e
    delta_ge <- as.numeric(coefs_s3[p$coef_name])
    beta_e <- beta_e + w_ge * delta_ge
  }
  period_effects[[as.character(e)]] <- list(effect = beta_e, mass = total_mass_at_e)
}

# Overall ATT = sum_e w_e * beta_e, where w_e proportional to total mass at e
total_post_mass <- sum(sapply(period_effects, function(x) x$mass))
att_s3 <- 0.0
for (e_str in names(period_effects)) {
  w_e <- period_effects[[e_str]]$mass / total_post_mass
  att_s3 <- att_s3 + w_e * period_effects[[e_str]]$effect
}

# SE via delta method: ATT = w' * delta
# Build the full weight vector for each (g, e) in post-treatment
overall_weights <- numeric(length(post_info))
overall_coef_indices <- integer(length(post_info))

for (i in seq_along(post_info)) {
  info <- post_info[[i]]
  nm <- info$name
  e <- info$e
  mass_ge <- post_masses[[nm]]$mass

  # Per-period mass
  e_str <- as.character(e)
  total_at_e <- period_effects[[e_str]]$mass
  w_ge_within_e <- mass_ge / total_at_e  # cohort weight within period

  # Period weight
  w_e <- period_effects[[e_str]]$mass / total_post_mass

  overall_weights[i] <- w_e * w_ge_within_e
  overall_coef_indices[i] <- which(names(coefs_s3) == nm)
}

V_sub_s3 <- V_s3[overall_coef_indices, overall_coef_indices]
var_att_s3 <- as.numeric(t(overall_weights) %*% V_sub_s3 %*% overall_weights)
se_s3 <- sqrt(max(var_att_s3, 0))

cat(sprintf("  IW-aggregated ATT = %.6f (SE = %.6f)\n", att_s3, se_s3))

# Output with g<cohort>_e<rel_time> keys (negative as m<abs>)
cohort_effects_output <- list()
for (info in post_info) {
  e_label <- if (info$e < 0) paste0("m", abs(info$e)) else as.character(info$e)
  key <- paste0("g", info$g, "_e", e_label)
  cohort_effects_output[[key]] <- cohort_effects_s3[[info$name]]
}

results$s3_sun_abraham <- list(
  scenario = "S3",
  description = "SunAbraham: IW-aggregated ATT with survey-weighted cohort masses",
  estimator = "survey::svyglm",
  att = att_s3,
  se = se_s3,
  df = df_s3,
  cohort_effects = cohort_effects_output
)

# ===========================================================================
# S4: TripleDifference — three-way interaction
# ===========================================================================

cat("\n--- S4: TripleDifference (three-way interaction DDD) ---\n")

design_s4 <- svydesign(ids = ~psu, strata = ~stratum, fpc = ~fpc,
                        weights = ~weight, data = ddd_data)

fit_s4 <- svyglm(outcome ~ group * partition * time, design = design_s4)

s4_res <- extract_svyglm_results(fit_s4, design_s4, "group:partition:time")

results$s4_triple_diff <- list(
  scenario = "S4",
  description = "TripleDifference: three-way interaction DDD coefficient",
  estimator = "survey::svyglm",
  att = s4_res$att,
  se = s4_res$se,
  t_stat = s4_res$t_stat,
  df = s4_res$df,
  ci_lower = s4_res$ci_lower,
  ci_upper = s4_res$ci_upper
)

cat(sprintf("  DDD = %.6f (SE = %.6f)\n", s4_res$att, s4_res$se))
cat(sprintf("  df = %d, CI = [%.4f, %.4f]\n", s4_res$df, s4_res$ci_lower, s4_res$ci_upper))

# ===========================================================================
# Embed data and metadata
# ===========================================================================

# Staggered data: export all columns as lists
staggered_export <- as.list(staggered_data[, c("unit", "period", "stratum", "psu", "fpc",
                                                "weight", "first_treat", "x1", "x2",
                                                "outcome")])

# DDD data: export all columns as lists
ddd_export <- as.list(ddd_data[, c("obs_id", "group", "partition", "time",
                                    "outcome", "stratum", "psu", "fpc", "weight")])

results[["_staggered_data"]] <- staggered_export
results[["_ddd_data"]] <- ddd_export
results[["_metadata"]] <- list(
  description = "Golden values for survey-enabled estimator validation",
  r_version = paste0(R.version$major, ".", R.version$minor),
  survey_version = as.character(packageVersion("survey")),
  seed = 42,
  n_units = length(unique(staggered_data$unit)),
  n_periods = length(unique(staggered_data$period)),
  n_obs_staggered = nrow(staggered_data),
  n_obs_ddd = nrow(ddd_data)
)

# ===========================================================================
# Save
# ===========================================================================

json_path <- file.path(output_dir, "survey_estimator_validation_golden.json")
writeLines(toJSON(results, auto_unbox = TRUE, pretty = TRUE, digits = 10), json_path)
cat(sprintf("\nResults saved to: %s\n", json_path))

cat("\n=== Summary ===\n")
cat(sprintf("S1 (ImputationDiD):    coef_x1 = %.6f, SE = %.6f\n",
            results$s1_imputation_did$coef_x1, results$s1_imputation_did$se_x1))
cat(sprintf("S2 (StackedDiD):       ATT = %.6f, SE = %.6f\n",
            results$s2_stacked_did$att, results$s2_stacked_did$se))
cat(sprintf("S3 (SunAbraham):       ATT = %.6f, SE = %.6f\n",
            results$s3_sun_abraham$att, results$s3_sun_abraham$se))
cat(sprintf("S4 (TripleDifference): DDD = %.6f, SE = %.6f\n",
            results$s4_triple_diff$att, results$s4_triple_diff$se))
cat("Done.\n")
