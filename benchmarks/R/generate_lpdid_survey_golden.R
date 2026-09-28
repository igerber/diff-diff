#!/usr/bin/env Rscript
# Golden generator: LPDiD COMPLEX-SURVEY path (Dube, Girardi, Jorda & Taylor 2025)
# vs survey::svyglm (Lumley 2004) -- Phase D2, closes Phase D of the LP-DiD salvage.
#
# D1 (#590) added survey support to LPDiD: probability weights + stratified-PSU
# Binder (1983) Taylor-linearization (TSL) sandwich SEs on the variance-weighted
# default path. D1 was smoke-gated against svyglm (~1e-10) on a hand-built single
# horizon. D2 pins the END-TO-END estimator: each LP-DiD horizon builds its own
# clean-control long-difference sample, and the matching svyglm reference must use
# a FRESH svydesign over THAT sample (NOT subset() of a full-panel design -- each
# horizon is a different stacked dataset, not a subpopulation of a fixed design).
#
# Survey variance is validated DIRECTLY against survey::svyglm, the reference
# implementation of the TSL sandwich -- no third-party survey package is needed as
# a gate. The CLEAN-SAMPLE CONSTRUCTION is instead cross-checked independently:
#   * the UNWEIGHTED variance-weighted event study (same prep/clean_h recipe as the
#     absorbing generator) must match alexCardazzi::lpdid() to <1e-8 (fail-closed,
#     pinned commit). Clean-sample selection is weight-independent, so this
#     transitively validates the weighted samples the svyglm goldens are built on.
#   * every svyglm WLS point must equal weighted feols(.. | time, weights, vcov=~unit)
#     on the same sample to <1e-7 (confirms sample/weight wiring before pinning SEs).
#
# Outputs (checked into the repo):
#   benchmarks/data/lpdid_survey_panel.csv    (staggered absorbing survey panel)
#   benchmarks/data/lpdid_survey_golden.json  (svyglm goldens: [est, se, df, n_psu])
#
# Does NOT touch generate_lpdid_golden.R or its panels/goldens (separate seed +
# disjoint output paths => the absorbing / non-absorbing fixtures stay byte-identical).
#
# Usage:
#   Rscript benchmarks/R/generate_lpdid_survey_golden.R

suppressMessages({
  library(dplyr); library(tsibble); library(slider); library(fixest)
  library(survey); library(data.table); library(jsonlite); library(lpdid)
})
setFixest_ssc(ssc(adj = TRUE, cluster.adj = TRUE))  # authors' reghdfe-style convention
options(survey.lonely.psu = "fail")                 # fail-closed: a lonely PSU is an error

SEED <- 20260630L
set.seed(SEED)
PRE <- 3L; POST <- 4L
ALEX_SHA <- "ba64563983861be5e616f6020842c1a1cdf17a27"
TOL_XCHECK <- 1e-8   # unweighted clean-sample cross-check vs alexCardazzi
TOL_POINT  <- 1e-7   # svyglm WLS point == weighted feols point (same sample)

# Verify the installed alexCardazzi/lpdid is exactly the pinned commit so the
# clean-sample cross-check gate is meaningful (fail-closed).
.alex_sha <- packageDescription("lpdid")$RemoteSha
if (is.null(.alex_sha) || is.na(.alex_sha)) {
  stop(sprintf("installed `lpdid` has no RemoteSha; reinstall via remotes::install_github(\"alexCardazzi/lpdid\", ref = \"%s\")",
               ALEX_SHA))
}
if (.alex_sha != ALEX_SHA) {
  stop(sprintf("installed `lpdid` commit %s != pinned ALEX_SHA %s; reinstall the pinned commit",
               .alex_sha, ALEX_SHA))
}

# ============================================================
# 1. Staggered absorbing survey panel. Balanced (gap-free). 4 strata; never-treated
#    distributed ROUND-ROBIN across all PSUs => every per-horizon clean sample keeps
#    all PSUs in every stratum (no lonely PSU; realized df = 14 - 4 = 10 for VW/cov).
#    weight (inverse selection prob) is DECOUPLED from fpc (population PSU count) and
#    dispersed across strata so the survey point clearly differs from the unweighted.
# ============================================================
Tt <- 12L
strata_sizes    <- c(30L, 30L, 40L, 40L)   # 140 units
psu_per_stratum <- c(3L, 3L, 4L, 4L)       # 14 PSUs
weight_h        <- c(5.0, 10.0, 7.5, 15.0) # dispersed inverse-selection weights
fpc_h           <- c(6L, 6L, 8L, 8L)       # population PSU count per stratum (f = n_psu/fpc = 0.5)
cohorts         <- c(5L, 8L)               # ~25% @ g=5, ~25% @ g=8, ~50% never

time_fe <- rnorm(Tt, 0, 1)
rows <- list(); idx <- 0L; unit <- 0L; global_psu <- 0L
for (h in seq_along(strata_sizes)) {
  n_h <- strata_sizes[h]; n_psu_h <- psu_per_stratum[h]
  for (i in 0:(n_h - 1L)) {
    unit <- unit + 1L
    psu_assign <- global_psu + (i %% n_psu_h) + 1L    # round-robin within stratum
    frac <- i / n_h
    g <- if (frac < 0.25) cohorts[1] else if (frac < 0.5) cohorts[2] else 0L
    unit_fe <- rnorm(1, 0, 2); eps_prev <- 0
    for (t in 1:Tt) {
      eps <- 0.4 * eps_prev + rnorm(1); eps_prev <- eps
      treated <- as.integer(g > 0 && t >= g)
      eff <- if (treated == 1) {
        k <- t - g; base <- if (g == 5L) 2.0 else 3.5; slope <- if (g == 5L) 0.5 else 0.2
        base + slope * k
      } else 0
      x <- rnorm(1)
      y <- unit_fe + time_fe[t] + eff + 0.7 * x + eps
      idx <- idx + 1L
      rows[[idx]] <- list(unit = unit, time = t, treat = treated, y = y, x = x,
                          stratum = h, psu = psu_assign, fpc = fpc_h[h], weight = weight_h[h])
    }
  }
  global_psu <- global_psu + n_psu_h
}
panel <- rbindlist(rows)
setorder(panel, unit, time)
panel_path <- file.path("benchmarks", "data", "lpdid_survey_panel.csv")
dir.create(dirname(panel_path), recursive = TRUE, showWarnings = FALSE)
fwrite(panel, panel_path)
message(sprintf("Wrote survey panel: %s (%d rows, %d units, %d PSUs, %d strata)",
                panel_path, nrow(panel), uniqueN(panel$unit),
                uniqueN(panel$psu), uniqueN(panel$stratum)))

# ============================================================
# 2. prep + clean_h: duplicated verbatim from generate_lpdid_golden.R (the recipe
#    the absorbing feols golden uses, alexCardazzi-cross-checked). Survey columns are
#    unit-constant and ride along through fill_gaps (panel is gap-free) into clean_h.
# ============================================================
td <- panel[treat == 1, .(treat_date = min(time)), by = unit]
prep <- panel %>%
  left_join(td, by = "unit") %>%
  as_tsibble(key = unit, index = time) %>%
  fill_gaps() %>%
  group_by(unit) %>% arrange(time, .by_group = TRUE) %>%
  mutate(treat_date = if (all(is.na(treat_date))) NA_real_ else suppressWarnings(min(treat_date, na.rm = TRUE))) %>%
  mutate(
    treat = as.integer(!is.na(treat_date) & time >= treat_date),
    tdiff = treat - lag(treat, 1L),
    Ly    = lag(y, 1L),
    obs   = !is.na(y)
  ) %>% ungroup() %>% as_tibble()
prep$tdiff[is.na(prep$tdiff) | prep$tdiff < 0] <- 0

clean_h <- function(h, base, kind) {
  p <- prep %>% group_by(unit) %>% arrange(time, .by_group = TRUE)
  if (kind == "post") {
    p <- p %>% mutate(Dy = lead(y, h) - .data[[base]], Fh = lead(treat, h)) %>% ungroup() %>%
      filter(obs & !is.na(Dy) & !is.na(tdiff) & !is.na(Fh) & (tdiff == 1 | Fh == 0))
  } else {
    p <- p %>% mutate(Dy = lag(y, h) - .data[[base]]) %>% ungroup() %>%
      filter(obs & !is.na(Dy) & !is.na(tdiff) & !is.na(treat) & (tdiff == 1 | treat == 0))
  }
  p
}
identified <- function(d) nrow(d) > 0 && length(unique(d$time[d$tdiff == 1])) > 0 &&
  any(d$tdiff == 0)

# unweighted feols event study (for the alexCardazzi clean-sample cross-check) ------
feols_es <- function(formula) function(d) {
  if (!identified(d)) return(c(NA_real_, NA_real_))
  m <- feols(formula, data = d, vcov = ~unit, warn = FALSE, notes = FALSE)
  c(coef(m)[["tdiff"]], se(m)[["tdiff"]])
}
post_h <- 0:POST
pre_h  <- 2:PRE
run_es <- function(base, fml) {
  out <- list()
  for (h in post_h) out[[as.character(h)]]  <- feols_es(fml)(clean_h(h, base, "post"))
  for (h in pre_h)  out[[as.character(-h)]] <- feols_es(fml)(clean_h(h, base, "pre"))
  out
}

# ============================================================
# 3. svyglm goldens. Each horizon: a FRESH svydesign over that clean sample.
#    Records c(estimate, SE, df, n_psu) for the tdiff coefficient.
# ============================================================
des_vw     <- function(d) svydesign(ids = ~psu,  strata = ~stratum, fpc = ~fpc, weights = ~weight, data = d)
des_nofpc  <- function(d) svydesign(ids = ~psu,  strata = ~stratum,             weights = ~weight, data = d)
des_inject <- function(d) svydesign(ids = ~unit,                                weights = ~weight, data = d)
fml_main <- Dy ~ tdiff + factor(time)
fml_cov  <- Dy ~ tdiff + x + factor(time)

svy_one <- function(d, formula, design_fn, psu_col, with_cov) {
  if (!identified(d)) return(c(NA_real_, NA_real_, NA_real_, NA_real_))
  if (psu_col == "psu") {                                  # no-lonely-PSU guard (strata designs)
    chk <- d %>% group_by(stratum) %>% summarise(npsu = n_distinct(psu), .groups = "drop")
    if (any(chk$npsu < 2L)) stop(sprintf("lonely PSU in a realized clean sample: %s", deparse(formula)))
  }
  des <- design_fn(d)
  co  <- summary(svyglm(formula, design = des))$coefficients
  est <- unname(co["tdiff", "Estimate"]); se <- unname(co["tdiff", "Std. Error"])
  # internal point gate: same-sample weighted feols WLS point must match svyglm
  ffml <- if (with_cov) (Dy ~ tdiff + x | time) else (Dy ~ tdiff | time)
  pt <- unname(coef(feols(ffml, data = d, weights = ~weight, vcov = ~unit, warn = FALSE, notes = FALSE))["tdiff"])
  if (!is.finite(est) || abs(est - pt) > TOL_POINT)
    stop(sprintf("svyglm WLS point != feols point (%.10f vs %.10f): %s", est, pt, deparse(formula)))
  c(est, se, degf(des), length(unique(d[[psu_col]])))
}

svy_run_es <- function(formula, design_fn, psu_col, with_cov = FALSE) {
  out <- list()
  for (h in post_h) out[[as.character(h)]]  <- svy_one(clean_h(h, "Ly", "post"), formula, design_fn, psu_col, with_cov)
  for (h in pre_h)  out[[as.character(-h)]] <- svy_one(clean_h(h, "Ly", "pre"),  formula, design_fn, psu_col, with_cov)
  out
}

survey_vw_es       <- svy_run_es(fml_main, des_vw,     "psu")
survey_vw_nofpc_es <- svy_run_es(fml_main, des_nofpc,  "psu")
survey_inject_es   <- svy_run_es(fml_main, des_inject, "unit")
survey_cov_es      <- svy_run_es(fml_cov,  des_vw,     "psu", with_cov = TRUE)
message("svyglm event studies built (vw / vw-nofpc / inject / covariate).")

# ---- pooled (authors' fixed-composition window-mean recipe, clean-through-F.POST) ----
pooled_tbl <- prep %>% group_by(unit) %>% arrange(time, .by_group = TRUE) %>%
  mutate(ypost = slide_dbl(y, ~ mean(.x, na.rm = FALSE), .before = 0, .after = POST, .complete = TRUE),
         ypre  = slide_dbl(y, ~ mean(.x, na.rm = FALSE), .before = PRE, .after = -2, .complete = TRUE),
         pooly_post = ypost - Ly, pooly_pre = ypre - Ly,
         Fpost = lead(treat, POST)) %>% ungroup()
pp   <- pooled_tbl %>% filter(obs & !is.na(pooly_post) & !is.na(tdiff) & !is.na(Fpost) & (tdiff == 1 | Fpost == 0))
ppre <- pooled_tbl %>% filter(obs & !is.na(pooly_pre)  & !is.na(tdiff) & !is.na(treat) & (tdiff == 1 | treat == 0))

svy_pooled <- function(d, yvar, design_fn, psu_col, with_cov) {
  if (psu_col == "psu") {
    chk <- d %>% group_by(stratum) %>% summarise(npsu = n_distinct(psu), .groups = "drop")
    if (any(chk$npsu < 2L)) stop(sprintf("lonely PSU in pooled sample (%s)", yvar))
  }
  des <- design_fn(d)
  fml <- as.formula(if (with_cov) sprintf("%s ~ tdiff + x + factor(time)", yvar)
                    else          sprintf("%s ~ tdiff + factor(time)", yvar))
  co  <- summary(svyglm(fml, design = des))$coefficients
  est <- unname(co["tdiff", "Estimate"]); se <- unname(co["tdiff", "Std. Error"])
  ffml <- as.formula(if (with_cov) sprintf("%s ~ tdiff + x | time", yvar) else sprintf("%s ~ tdiff | time", yvar))
  pt <- unname(coef(feols(ffml, data = d, weights = ~weight, vcov = ~unit, warn = FALSE, notes = FALSE))["tdiff"])
  if (!is.finite(est) || abs(est - pt) > TOL_POINT)
    stop(sprintf("pooled svyglm point != feols point (%.10f vs %.10f): %s", est, pt, yvar))
  c(est, se, degf(des), length(unique(d[[psu_col]])))
}

survey_vw_pooled     <- list(post = svy_pooled(pp,   "pooly_post", des_vw,     "psu",  FALSE),
                             pre  = svy_pooled(ppre, "pooly_pre",  des_vw,     "psu",  FALSE))
survey_inject_pooled <- list(post = svy_pooled(pp,   "pooly_post", des_inject, "unit", FALSE),
                             pre  = svy_pooled(ppre, "pooly_pre",  des_inject, "unit", FALSE))
survey_cov_pooled    <- list(post = svy_pooled(pp,   "pooly_post", des_vw,     "psu",  TRUE),
                             pre  = svy_pooled(ppre, "pooly_pre",  des_vw,     "psu",  TRUE))
message("svyglm pooled rows built (vw / inject / covariate).")

# ============================================================
# 4. Independent clean-sample cross-check: UNWEIGHTED VW event study (same recipe)
#    vs alexCardazzi::lpdid() at <1e-8 (fail-closed). Weight-independent selection
#    => transitively validates the weighted svyglm samples above.
# ============================================================
vw_unw_es <- run_es("Ly", Dy ~ tdiff | time)
alex_df <- as.data.frame(panel)
alex_fit <- function(...) tryCatch(
  lpdid(alex_df, window = c(-PRE, POST), y = "y", unit_index = "unit",
        time_index = "time", treat_status = "treat", ...),
  error = function(e) list(error = conditionMessage(e)))
alex_es <- function(fit) {
  if (!is.null(fit$error)) return(list(error = fit$error))
  setNames(lapply(seq_along(fit$window),
                  function(i) c(fit$coeftable$Estimate[i], fit$coeftable$`Std. Error`[i])),
           as.character(fit$window))
}
xcheck <- function(name, ours, theirs) {
  if (!is.null(theirs$error)) stop(sprintf("alexCardazzi cross-check UNAVAILABLE for %s: %s", name, theirs$error))
  hs <- intersect(names(ours), names(theirs))
  if (length(hs) == 0L) stop(sprintf("alexCardazzi cross-check for %s: no overlapping horizons", name))
  diffs <- sapply(hs, function(h) {
    a <- ours[[h]]; b <- theirs[[h]]
    if (any(is.na(a)) || any(is.na(b))) NA_real_ else max(abs(a[1:2] - b[1:2]))
  })
  if (all(is.na(diffs))) stop(sprintf("alexCardazzi cross-check for %s: all horizons non-comparable", name))
  d <- max(diffs, na.rm = TRUE)
  message(sprintf("  [xcheck %s] max|ours - alex| = %.2e over h={%s}", name, d, paste(hs, collapse = ",")))
  if (!is.finite(d) || d > TOL_XCHECK)
    stop(sprintf("alexCardazzi cross-check FAILED for %s (%.2e > %.0e)", name, d, TOL_XCHECK))
}
message("Independent clean-sample cross-check (unweighted VW vs alexCardazzi):")
xcheck("survey_clean_sample_unweighted_vw", vw_unw_es, alex_es(alex_fit()))

# ============================================================
# 5. Write golden JSON
# ============================================================
golden <- list(
  meta = list(
    estimator = "LPDiD (Dube, Girardi, Jorda & Taylor 2025) - complex survey (Phase D2)",
    r_version = R.version.string,
    survey_version = as.character(packageVersion("survey")),
    fixest_version = as.character(packageVersion("fixest")),
    lpdid_alexcardazzi_version = as.character(packageVersion("lpdid")),
    lpdid_alexcardazzi_commit = ALEX_SHA,
    seed = SEED, pre_window = PRE, post_window = POST,
    n_strata = uniqueN(panel$stratum), n_psu_total = uniqueN(panel$psu),
    record_format = "Each horizon / pooled row is c(estimate, SE, df, n_psu) for the tdiff coefficient.",
    se_convention = paste(
      "survey::svyglm Binder (1983) Taylor-linearization sandwich. df = degf(design):",
      "n_PSU - n_strata (strata present, VW/covariate) or n_PSU - 1 (weights-only inject).",
      "Each horizon uses a FRESH svydesign over THAT horizon's clean LP-DiD long-difference",
      "sample (NOT subset() of a full design) - matching the library's per-sample resolution."),
    clean_sample_xcheck = paste(
      "Clean-sample construction independently validated: the UNWEIGHTED VW event study",
      "(same prep/clean_h recipe) matches alexCardazzi::lpdid() to <1e-8. Selection is",
      "weight-independent, so this transitively validates the weighted svyglm samples."),
    point_gate = paste(
      "Every svyglm WLS point validated == weighted feols(.. | time, weights, vcov=~unit)",
      "on the same sample to <1e-7."),
    fpc_note = "fpc is the per-stratum population PSU count (f_h = n_PSU_h / fpc_h); decoupled from the inverse-selection weight."
  ),
  survey_vw_es = survey_vw_es,
  survey_vw_nofpc_es = survey_vw_nofpc_es,
  survey_inject_es = survey_inject_es,
  survey_cov_es = survey_cov_es,
  survey_vw_pooled = survey_vw_pooled,
  survey_inject_pooled = survey_inject_pooled,
  survey_cov_pooled = survey_cov_pooled
)
golden_path <- file.path("benchmarks", "data", "lpdid_survey_golden.json")
write_json(golden, golden_path, auto_unbox = TRUE, pretty = TRUE, digits = 12, na = "null")
message(sprintf("Wrote survey golden: %s", golden_path))
