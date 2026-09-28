#!/usr/bin/env Rscript
# Generate R parity goldens for diff-diff PowerAnalysis (analytical path).
#
# Running this script writes ../data/r_power_golden.json. Both the script and
# the JSON are committed in the PowerAnalysis methodology-review PR (PR-B).
#
# Requires:
#   - R 4.4+ (tested on 4.5.2)
#   - install.packages("jsonlite")
#   (base R `qnorm`/`pnorm` only -- no `pwr`, by design; see below.)
#
# PARITY CONTRACT
# ---------------
# diff-diff's analytical PowerAnalysis methods (PowerAnalysis.mde / .power /
# .sample_size and the compute_* convenience wrappers) match the values in this
# JSON to atol=1e-9 on the continuous quantities (variance, se, mde, power) and
# exactly on the integer required_n. The goldens are EXACT closed forms computed
# independently in base R, so the only source of disagreement is qnorm/pnorm vs
# scipy.stats.norm.ppf/cdf (agree to ~1e-15).
#
# Two deliberate choices, both decided with the user during the PR-B methodology
# review (see docs/methodology/REGISTRY.md ## PowerAnalysis and the audits under
# docs/methodology/papers/):
#
#  D1 (z, not t): the MDE multiplier is the NORMAL multiplier
#     M = qnorm(1 - alpha/2) + qnorm(power)        [two-sided]
#     M = qnorm(1 - alpha)   + qnorm(power)        [one-sided]
#     following Bloom (1995). This is why the parity reference is normal-based
#     (qnorm), NOT pwr::pwr.t.test() (noncentral-t) -- pwr.t.test would NOT
#     match the library and is deliberately not used here.
#
#  D4 (panel variance = Burlig Eq. 2, equicorrelated case): the panel-DiD
#     variance is the within-unit equicorrelated special case of Burlig, Preonas
#     & Woerman (2020), Eq. 2:
#         Var(ATT) = sigma^2 (1/n_T + 1/n_C) (1/m + 1/r) (1 - rho)
#     with m = n_pre, r = n_post. Burlig's reference implementation is the Stata
#     package `pcpanel`; a Stata cross-check is intentionally NOT wired here (no
#     Stata in this environment). The independent empirical validation that this
#     closed form is the correct DiD variance under equicorrelated within-unit
#     errors is the literal-equicorrelated Monte-Carlo test in
#     tests/test_methodology_power.py (TestPanelVarianceMonteCarlo); this R file
#     pins the cross-language arithmetic.
#
# The 2x2 (basic_did) variance is the m = r = 1 special case of the same
# equicorrelated formula (Burlig footnote 11): 2 sigma^2 (1/n_T + 1/n_C)(1 - rho),
# reducing to the DiD analog of Bloom (1995) Eq. 1 (2 sigma^2 (1/n_T + 1/n_C)) at
# rho = 0. The router uses basic_did when n_pre + n_post <= 2, panel otherwise; the
# panel form is continuous with basic_did at n_pre = n_post = 1 ((1/m + 1/r) = 2).

suppressWarnings(suppressMessages(library(jsonlite)))

MAX_SAMPLE_SIZE <- 2147483647L  # 2^31 - 1, mirrors diff_diff/power.py MAX_SAMPLE_SIZE

# --- closed-form helpers (mirror diff_diff/power.py exactly) ------------------

crit_values <- function(alpha, power, alternative) {
  z_alpha <- if (alternative == "two-sided") qnorm(1 - alpha / 2) else qnorm(1 - alpha)
  z_beta <- qnorm(power)
  c(z_alpha = z_alpha, z_beta = z_beta)
}

variance_of <- function(n_treated, n_control, n_pre, n_post, sigma, rho, deff) {
  T_tot <- n_pre + n_post
  if (T_tot > 2) {
    # panel: Burlig Eq. 2, equicorrelated
    period_factor <- 1 / n_pre + 1 / n_post
    v <- sigma^2 * (1 / n_treated + 1 / n_control) * period_factor * (1 - rho)
  } else {
    # basic 2x2 DiD = the m = r = 1 equicorrelated case (Burlig footnote 11):
    # 2 sigma^2 (1/n_T + 1/n_C)(1 - rho); reduces to Bloom at rho = 0.
    v <- 2 * sigma^2 * (1 / n_treated + 1 / n_control) * (1 - rho)
  }
  v * deff
}

design_of <- function(n_pre, n_post) if ((n_pre + n_post) > 2) "panel" else "basic_did"

mde_of <- function(se, alpha, power, alternative) {
  cv <- crit_values(alpha, power, alternative)
  unname((cv["z_alpha"] + cv["z_beta"]) * se)
}

power_of <- function(effect_size, se, alpha, alternative) {
  if (alternative == "two-sided") {
    z_alpha <- qnorm(1 - alpha / 2)
    1 - pnorm(z_alpha - effect_size / se) + pnorm(-z_alpha - effect_size / se)
  } else if (alternative == "greater") {
    z_alpha <- qnorm(1 - alpha)
    1 - pnorm(z_alpha - effect_size / se)
  } else { # less
    z_alpha <- qnorm(1 - alpha)
    pnorm(-z_alpha - effect_size / se)
  }
}

required_n_of <- function(effect_size, sigma, n_pre, n_post, rho, alpha, power,
                          alternative, treat_frac, deff) {
  if (effect_size == 0) return(MAX_SAMPLE_SIZE)
  cv <- crit_values(alpha, power, alternative)
  zsum2 <- unname((cv["z_alpha"] + cv["z_beta"])^2)
  if ((n_pre + n_post) > 2) {
    period_factor <- 1 / n_pre + 1 / n_post
    n_total_raw <- sigma^2 * zsum2 * period_factor * (1 - rho) /
      (effect_size^2 * treat_frac * (1 - treat_frac))
  } else {
    # basic 2x2 = m = r = 1 equicorrelated case; (1 - rho) mirrors variance_of.
    n_total_raw <- 2 * sigma^2 * zsum2 * (1 - rho) /
      (effect_size^2 * treat_frac * (1 - treat_frac))
  }
  n_total_raw <- n_total_raw * deff
  if (is.infinite(n_total_raw)) return(MAX_SAMPLE_SIZE)
  # mirror _compute_required_n then sample_size() reassembly:
  n_total <- max(4, ceiling(n_total_raw))
  n_treated <- max(2, ceiling(n_total * treat_frac))
  n_control <- max(2, n_total - n_treated)
  as.integer(n_treated + n_control)
}

# --- fixtures (inputs span design, alpha/power, alternative, rho sign, m != r) -

fixtures <- list(
  list(name = "basic_2x2_two_sided", alpha = 0.05, power = 0.80,
       alternative = "two-sided", n_treated = 50, n_control = 50, sigma = 1.0,
       n_pre = 1, n_post = 1, rho = 0.0, effect_size = 0.5, treat_frac = 0.5),
  list(name = "basic_2x2_rho04", alpha = 0.05, power = 0.80,
       alternative = "two-sided", n_treated = 50, n_control = 50, sigma = 1.0,
       n_pre = 1, n_post = 1, rho = 0.4, effect_size = 0.5, treat_frac = 0.5),
  list(name = "basic_2x2_unbalanced", alpha = 0.05, power = 0.80,
       alternative = "two-sided", n_treated = 30, n_control = 90, sigma = 2.0,
       n_pre = 1, n_post = 1, rho = 0.0, effect_size = 0.8, treat_frac = 0.25),
  list(name = "basic_2x2_one_sided_p90", alpha = 0.05, power = 0.90,
       alternative = "greater", n_treated = 40, n_control = 40, sigma = 1.5,
       n_pre = 1, n_post = 1, rho = 0.0, effect_size = 0.6, treat_frac = 0.5),
  list(name = "panel_rho0", alpha = 0.05, power = 0.80,
       alternative = "two-sided", n_treated = 50, n_control = 50, sigma = 1.0,
       n_pre = 3, n_post = 3, rho = 0.0, effect_size = 0.3, treat_frac = 0.5),
  list(name = "panel_rho03", alpha = 0.05, power = 0.80,
       alternative = "two-sided", n_treated = 50, n_control = 50, sigma = 1.0,
       n_pre = 3, n_post = 3, rho = 0.3, effect_size = 0.3, treat_frac = 0.5),
  list(name = "panel_rho05_asymmetric", alpha = 0.05, power = 0.80,
       alternative = "two-sided", n_treated = 60, n_control = 40, sigma = 1.2,
       n_pre = 2, n_post = 5, rho = 0.5, effect_size = 0.4, treat_frac = 0.6),
  list(name = "panel_one_sided_p90", alpha = 0.10, power = 0.90,
       alternative = "greater", n_treated = 80, n_control = 80, sigma = 2.0,
       n_pre = 4, n_post = 4, rho = 0.2, effect_size = 0.5, treat_frac = 0.5),
  list(name = "panel_negative_rho", alpha = 0.05, power = 0.80,
       alternative = "two-sided", n_treated = 50, n_control = 50, sigma = 1.0,
       n_pre = 3, n_post = 3, rho = -0.1, effect_size = 0.3, treat_frac = 0.5)
)

# Bloom (1995) Table 1 one-sided .05 multipliers (z_{0.95} + z_{power}):
bloom_multipliers <- list(
  list(power = 0.90, expected = 2.93),
  list(power = 0.80, expected = 2.49),
  list(power = 0.70, expected = 2.17)
)
bloom_rows <- lapply(bloom_multipliers, function(b) {
  list(power = b$power,
       multiplier = unname(qnorm(0.95) + qnorm(b$power)),
       bloom_stated = b$expected)
})

results <- lapply(fixtures, function(f) {
  v <- variance_of(f$n_treated, f$n_control, f$n_pre, f$n_post, f$sigma, f$rho, 1.0)
  se <- sqrt(v)
  f$expected <- list(
    design = design_of(f$n_pre, f$n_post),
    variance = v,
    se = se,
    mde = mde_of(se, f$alpha, f$power, f$alternative),
    power = power_of(f$effect_size, se, f$alpha, f$alternative),
    required_n = required_n_of(f$effect_size, f$sigma, f$n_pre, f$n_post, f$rho,
                               f$alpha, f$power, f$alternative, f$treat_frac, 1.0)
  )
  f
})

out <- list(
  meta = list(
    generated_at = as.character(Sys.Date()),
    r_version = R.version.string,
    description = paste(
      "Bloom (1995) + Burlig, Preonas & Woerman (2020) parity goldens for",
      "diff-diff PowerAnalysis (analytical path). Normal-based (qnorm) MDE",
      "multiplier (D1, NOT pwr.t.test); panel variance = Burlig Eq. 2",
      "equicorrelated case sigma^2 (1/n_T+1/n_C)(1/m+1/r)(1-rho) (D4); the 2x2",
      "path is the m=r=1 case 2 sigma^2 (1/n_T+1/n_C)(1-rho), = Bloom at rho=0.",
      "atol=1e-9 on variance/se/mde/power; exact on",
      "required_n. Stata pcpanel cross-check intentionally omitted (no Stata);",
      "the equicorrelated DiD variance is validated empirically by the",
      "Monte-Carlo test in tests/test_methodology_power.py."
    )
  ),
  bloom_table1_one_sided_p05 = bloom_rows,
  fixtures = results
)

# Run from benchmarks/R/ so that ../data/ resolves (mirrors
# generate_pretrends_golden.R).
write_json(out, "../data/r_power_golden.json", auto_unbox = TRUE,
           pretty = TRUE, digits = NA)
cat("Wrote ../data/r_power_golden.json with", length(results), "fixtures.\n")
