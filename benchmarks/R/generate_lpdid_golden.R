#!/usr/bin/env Rscript
# Golden generator: LPDiD vs the method authors' reference recipes
# (Dube, Girardi, Jorda & Taylor 2025), with an alexCardazzi/lpdid cross-check.
#
# The paper specifies NO standard-error formula (Section 1 defers to "standard
# techniques"), so every estimate is validated against the reference *software*:
#   - PRIMARY: the authors' own R recipe (danielegirardi/lpdid example scripts) -
#     fixest::feols(long_diff ~ treat_switch | time, vcov = ~unit) with the
#     reghdfe-style small-sample correction setFixest_ssc(adj, cluster.adj), plus
#     the reweighted / premean / pooled / regression-adjustment variants.
#   - CROSS-CHECK GATE: alexCardazzi::lpdid() (third-party, pinned commit) must
#     agree where conditions match. The POOLED cross-check is intentionally skipped
#     (alexCardazzi uses a laxer pooled clean-control window than the authors'
#     clean-through-window-end recipe); alex's pooled value is recorded in `meta`.
#
# Regression-adjustment (RA, variant ra_cov): the canonical RA standard error is
# Stata `teffects ra ... atet vce(cluster)` only - no R package implements it
# (alexCardazzi uses direct covariate inclusion, not RA). We therefore anchor the
# RA POINT estimate here (full-interaction route == teffects point) and pin the
# library's influence-function SE on the Python side as a documented regression
# value; its calibration is validated out-of-band by the ungated Monte-Carlo
# coverage study benchmarks/python/coverage_lpdid_ra.py. See REGISTRY.md "## LPDiD".
#
# Outputs (checked into the repo):
#   benchmarks/data/lpdid_test_panel.csv          (absorbing panel)
#   benchmarks/data/lpdid_golden.json             (absorbing goldens)
#   benchmarks/data/lpdid_nonabsorbing_panel.csv  (non-absorbing panel, Phase C2)
#   benchmarks/data/lpdid_nonabsorbing_golden.json (non-absorbing goldens, Phase C2)
#
# Usage:
#   Rscript benchmarks/R/generate_lpdid_golden.R

suppressMessages({
  library(dplyr); library(tsibble); library(slider); library(fixest)
  library(sandwich); library(data.table); library(jsonlite); library(lpdid)
})
setFixest_ssc(ssc(adj = TRUE, cluster.adj = TRUE))  # authors' reghdfe-style convention

SEED <- 20260629L
set.seed(SEED)
PRE <- 3L; POST <- 4L
ALEX_SHA <- "ba64563983861be5e616f6020842c1a1cdf17a27"
TOL_XCHECK <- 1e-8  # authors' recipe vs alexCardazzi agreement gate

# Verify the installed alexCardazzi/lpdid is exactly the pinned commit, so the
# cross-check gate is meaningful (fail-closed, not just metadata).
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
# 1. Panel: staggered absorbing, ~63% never-treated, heterogeneous + persistent,
#    exogenous confounder x, plus ONE interior-gap unit (pins the reindex / Dev 5).
# ============================================================
Tt <- 12L
cohort <- c(rep(4L, 12), rep(8L, 10), rep(0L, 38))   # 1-12 @ t=4, 13-22 @ t=8, 23-60 never
units <- seq_along(cohort)
unit_fe <- rnorm(length(units), 0, 2)
time_fe <- rnorm(Tt, 0, 1)
rows <- vector("list", length(units) * Tt); idx <- 0L
for (u in units) {
  g <- cohort[u]; eps_prev <- 0
  for (t in 1:Tt) {
    eps <- 0.4 * eps_prev + rnorm(1); eps_prev <- eps
    treated <- as.integer(g > 0 && t >= g)
    eff <- if (treated == 1) {
      k <- t - g; base <- if (g == 4) 2.0 else 3.5; slope <- if (g == 4) 0.5 else 0.2
      base + slope * k
    } else 0
    x <- rnorm(1)
    y <- unit_fe[u] + time_fe[t] + eff + 0.7 * x + eps
    idx <- idx + 1L
    rows[[idx]] <- list(unit = u, time = t, treat = treated, y = y, x = x)
  }
}
panel <- rbindlist(rows)
panel <- panel[!(unit == 60L & time == 7L)]  # interior gap on a never-treated unit
setorder(panel, unit, time)
panel_path <- file.path("benchmarks", "data", "lpdid_test_panel.csv")
dir.create(dirname(panel_path), recursive = TRUE, showWarnings = FALSE)
fwrite(panel, panel_path)
message(sprintf("Wrote panel: %s (%d rows, %d units, 1 interior gap)",
                panel_path, nrow(panel), uniqueN(panel$unit)))

# ============================================================
# 2. Prep on the interior-calendar grid (fill_gaps == our reindex), absorbing-fill
#    treatment on the grid, features computed on the grid, restrict back to observed.
# ============================================================
td <- panel[treat == 1, .(treat_date = min(time)), by = unit]
maxT <- max(panel$time)
prep <- panel %>%
  left_join(td, by = "unit") %>%
  as_tsibble(key = unit, index = time) %>%
  fill_gaps() %>%                                   # interior gaps -> explicit NA rows
  group_by(unit) %>% arrange(time, .by_group = TRUE) %>%
  mutate(treat_date = if (all(is.na(treat_date))) NA_real_ else suppressWarnings(min(treat_date, na.rm = TRUE))) %>%
  mutate(
    treat = as.integer(!is.na(treat_date) & time >= treat_date),  # absorbing-fill on grid
    tdiff = treat - lag(treat, 1L),
    Ly    = lag(y, 1L),
    premean2 = (lag(y, 1L) + lag(y, 2L)) / 2,        # fixed-k=2 premean (pmd variant)
    obs   = !is.na(y)
  ) %>% ungroup() %>% as_tibble()
prep$tdiff[is.na(prep$tdiff) | prep$tdiff < 0] <- 0

# clean sample for one horizon; restrict to OBSERVED rows with a finite long diff.
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
# identified == both treated-switch and clean-control present at >1 time level
identified <- function(d) nrow(d) > 0 && length(unique(d$time[d$tdiff == 1])) > 0 &&
  any(d$tdiff == 0)

feols_es <- function(formula, weights = NULL) function(d) {
  if (!identified(d)) return(c(NA_real_, NA_real_))
  m <- if (is.null(weights)) feols(formula, data = d, vcov = ~unit, warn = FALSE, notes = FALSE)
       else feols(formula, data = d, weights = weights, vcov = ~unit, warn = FALSE, notes = FALSE)
  c(coef(m)[["tdiff"]], se(m)[["tdiff"]])
}

post_h <- 0:POST
pre_h  <- 2:PRE   # h = -2 .. -PRE placebos (h = -1 is the fixed reference)

run_es <- function(base, fml) {
  out <- list()
  for (h in post_h) out[[as.character(h)]]  <- feols_es(fml)(clean_h(h, base, "post"))
  for (h in pre_h)  out[[as.character(-h)]] <- feols_es(fml)(clean_h(h, base, "pre"))
  out
}

# ---- girardi reweighting (FWL inverse weights) for the equally-weighted ATT ----
reweighted_es <- function() {
  get_w <- function(d) {
    mw <- feols(tdiff ~ 1 | time, data = d, vcov = "iid", warn = FALSE, notes = FALSE)
    d$nw <- residuals(mw); d$nw[d$tdiff != 1] <- NA_real_
    den <- sum(d$nw, na.rm = TRUE)
    d <- d %>% group_by(time) %>%
      mutate(w = nw / den, gw = suppressWarnings(max(w, na.rm = TRUE)),
             gw = ifelse(is.infinite(gw), NA_real_, gw),
             w = ifelse(is.na(w), gw, w)) %>% ungroup()
    1 / d$w
  }
  out <- list()
  w0 <- NULL
  for (h in post_h) {
    d <- clean_h(h, "Ly", "post")
    if (!identified(d)) { out[[as.character(h)]] <- c(NA_real_, NA_real_); next }
    d$rw <- get_w(d); if (h == 0) w0 <- d %>% select(unit, time, rw0 = rw)
    d <- d %>% filter(rw > 0)
    out[[as.character(h)]] <- feols_es(Dy ~ tdiff | time, weights = ~rw)(d)
  }
  for (h in pre_h) {                                   # pre uses the h=0 weights (reweight_0)
    d <- clean_h(h, "Ly", "pre") %>% left_join(w0, by = c("unit", "time"))
    d <- d %>% filter(!is.na(rw0) & rw0 > 0)
    out[[as.character(-h)]] <- feols_es(Dy ~ tdiff | time, weights = ~rw0)(d)
  }
  out
}

# ---- RA via full-interaction route (teffects ra ATET POINT; SE pinned in Python) ----
ra_point_es <- function() {
  atet_h <- function(d) {
    if (!identified(d)) return(c(NA_real_, NA_real_, NA_real_))
    d2 <- d %>% transmute(Dy, dtr = factor(tdiff), x, tf = factor(time), unit)
    fi <- lm(Dy ~ dtr * (tf + x), data = d2)
    b <- coef(fi); keep <- !is.na(b); nm <- names(b)[keep]; bk <- b[keep]
    mm <- model.matrix(fi)[, nm, drop = FALSE]
    V0 <- sandwich::vcovCL(fi, cluster = d2$unit, cadjust = FALSE, type = "HC0")
    g <- colMeans(mm[d2$dtr == "1", , drop = FALSE]); g[!grepl("^dtr1", nm)] <- 0
    c(sum(g * bk), sqrt(as.numeric(t(g) %*% V0 %*% g)), NA_real_)  # [att, conditional CR0 se (ref only)]
  }
  out <- list()
  for (h in post_h) out[[as.character(h)]]  <- atet_h(clean_h(h, "Ly", "post"))
  for (h in pre_h)  out[[as.character(-h)]] <- atet_h(clean_h(h, "Ly", "pre"))
  out
}

vw_es     <- run_es("Ly", Dy ~ tdiff | time)
direct_es <- run_es("Ly", Dy ~ tdiff + x | time)
pmd_es    <- run_es("premean2", Dy ~ tdiff | time)
ew_es     <- reweighted_es()
ra_es     <- ra_point_es()

# ---- pooled (authors' recipe): mean(y over [t, t+POST]) - Ly, clean-through-F.POST ----
pooled_tbl <- prep %>% group_by(unit) %>% arrange(time, .by_group = TRUE) %>%
  mutate(ypost = slide_dbl(y, ~ mean(.x, na.rm = FALSE), .before = 0, .after = POST, .complete = TRUE),
         ypre  = slide_dbl(y, ~ mean(.x, na.rm = FALSE), .before = PRE, .after = -2, .complete = TRUE),
         pooly_post = ypost - Ly, pooly_pre = ypre - Ly,
         Fpost = lead(treat, POST)) %>% ungroup()
pp <- pooled_tbl %>% filter(obs & !is.na(pooly_post) & !is.na(tdiff) & !is.na(Fpost) & (tdiff == 1 | Fpost == 0))
mpost <- feols(pooly_post ~ tdiff | time, data = pp, vcov = ~unit, warn = FALSE, notes = FALSE)
ppre <- pooled_tbl %>% filter(obs & !is.na(pooly_pre) & !is.na(tdiff) & !is.na(treat) & (tdiff == 1 | treat == 0))
mpre <- feols(pooly_pre ~ tdiff | time, data = ppre, vcov = ~unit, warn = FALSE, notes = FALSE)
pooled <- list(post = c(coef(mpost)[["tdiff"]], se(mpost)[["tdiff"]]),
               pre  = c(coef(mpre)[["tdiff"]],  se(mpre)[["tdiff"]]))

# ============================================================
# 3. alexCardazzi::lpdid cross-check (VW / direct / pmd / nocomp event-study + pooled-in-meta)
# ============================================================
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
alex_vw     <- alex_es(alex_fit())
alex_direct <- alex_es(alex_fit(controls = ~ x))
alex_pmd    <- alex_es(alex_fit(pmd = TRUE, pmd_lag = 2))
alex_nocomp <- alex_es(alex_fit(nocomp = TRUE))
alex_pooled <- tryCatch({
  f <- lpdid(alex_df, window = c(-PRE, POST), y = "y", unit_index = "unit",
             time_index = "time", treat_status = "treat", pooled = TRUE)
  c(f$coeftable$Estimate[1], f$coeftable$`Std. Error`[1])
}, error = function(e) c(NA_real_, NA_real_))

# cross-check gate: authors' recipe must agree with alexCardazzi where comparable.
# Fail-closed - a cross-check that cannot run (alex error, no overlapping horizons,
# all-NA comparisons, or non-finite difference) is a FAILURE, not a silent pass, so
# the gate can never be bypassed into writing ungated goldens.
xcheck <- function(name, ours, theirs) {
  if (!is.null(theirs$error)) {
    stop(sprintf("alexCardazzi cross-check UNAVAILABLE for %s: %s", name, theirs$error))
  }
  hs <- intersect(names(ours), names(theirs))
  if (length(hs) == 0L) {
    stop(sprintf("alexCardazzi cross-check for %s: no overlapping horizons", name))
  }
  diffs <- sapply(hs, function(h) {
    a <- ours[[h]]; b <- theirs[[h]]
    if (any(is.na(a)) || any(is.na(b))) NA_real_ else max(abs(a[1:2] - b[1:2]))
  })
  if (all(is.na(diffs))) {
    stop(sprintf("alexCardazzi cross-check for %s: all overlapping horizons non-comparable (NA)", name))
  }
  d <- max(diffs, na.rm = TRUE)
  message(sprintf("  [xcheck %s] max|authors - alex| = %.2e over h={%s}",
                  name, d, paste(hs, collapse = ",")))
  if (!is.finite(d) || d > TOL_XCHECK) {
    stop(sprintf("alexCardazzi cross-check FAILED for %s (%.2e > %.0e)", name, d, TOL_XCHECK))
  }
}
message("alexCardazzi cross-check (authors' feols recipe vs the package):")
xcheck("vw", vw_es, alex_vw)
xcheck("direct", direct_es, alex_direct)
xcheck("pmd", pmd_es, alex_pmd)
# nocomp is intentionally NOT a golden parity variant: the library's no_composition
# fixes the realized sample across ALL post horizons (the paper's fixed-composition
# intent, Section 3.6) and excludes cohorts with p_g > T-H, whereas alexCardazzi uses a
# looser per-horizon sample and a stricter treat_date < maxT-POST cutoff - so our path is
# MORE faithful to the paper but has no matching R-package reference. It is validated by
# the pure-Python B1 tests in tests/test_lpdid.py. alexCardazzi's looser-semantics value
# is recorded in meta for transparency only.
message("  [nocomp] no golden parity variant (library is more paper-faithful than any R pkg;")
message("           B1-tested; alexCardazzi looser-semantics value recorded in meta).")

# ============================================================
# 4. Write golden JSON
# ============================================================
golden <- list(
  meta = list(
    estimator = "LPDiD (Dube, Girardi, Jorda & Taylor 2025) - absorbing",
    r_version = R.version.string,
    fixest_version = as.character(packageVersion("fixest")),
    lpdid_alexcardazzi_version = as.character(packageVersion("lpdid")),
    lpdid_alexcardazzi_commit = ALEX_SHA,
    seed = SEED, pre_window = PRE, post_window = POST,
    se_convention = paste(
      "Default/weighted/direct/pmd/pooled SEs: feols cluster-robust at unit with the",
      "reghdfe-style finite-sample factor (G/(G-1))*((n-1)/(n-k)) via",
      "setFixest_ssc(adj=TRUE, cluster.adj=TRUE) - the authors' convention."),
    ra_se_note = paste(
      "ra_cov records the RA POINT estimate (full-interaction == teffects ra atet).",
      "The canonical RA SE is Stata teffects (unconditional IF, NO finite-sample",
      "factor) - no R package computes it; the library influence-function SE is pinned",
      "on the Python side and calibration-validated by benchmarks/python/coverage_lpdid_ra.py.",
      "ra_cov[h] = c(att, conditional_CR0_se_for_reference_only)."),
    pooled_note = paste(
      "Pooled uses the authors' fixed-composition window-mean recipe",
      "(mean(y over [t,t+POST]) - y_{t-1}, clean through F.POST). alexCardazzi's pooled",
      "uses a laxer clean-control window, so its pooled value is recorded here for",
      "transparency but NOT used as the cross-check gate."),
    nocomp_note = paste(
      "no_composition is NOT a golden parity variant: the library fixes the realized",
      "sample across all post horizons (paper Section 3.6) and excludes cohorts p_g > T-H,",
      "more faithful to the paper than alexCardazzi's looser per-horizon version (recorded",
      "below). The library path is validated by the pure-Python B1 tests."),
    alex_pooled_post = alex_pooled,
    alex_nocomp_looser = alex_nocomp
  ),
  vw_es = vw_es, ew_es = ew_es, pmd_es = pmd_es,
  direct_es = direct_es, ra_cov = ra_es, pooled = pooled
)
golden_path <- file.path("benchmarks", "data", "lpdid_golden.json")
# na = "null" so R NA serializes as JSON null (not the string "NA"), which the
# Python loader reads as None and handles in its missing-value branch.
write_json(golden, golden_path, auto_unbox = TRUE, pretty = TRUE, digits = 12, na = "null")
message(sprintf("Wrote golden: %s", golden_path))

# ============================================================
# 5. NON-ABSORBING (Phase C2): independent feols reconstruction of the paper's
#    Eq. 12 (first_entry) and Eq. 13 (effect_stabilization) clean-sample
#    restrictions, anchoring the C1 non-absorbing modes (LPDiD(non_absorbing=...)).
#    EXPLICIT per-window indexing (a different construction than the Python
#    cumulative-lookup; the recipe's independence was demonstrated when an earlier
#    draft's Eq.12 control off-by-one diverged from the already-correct Python and
#    was corrected against the paper). Validated == the library to ~1e-13 on point
#    and ~1e-15 on SE for the variance-weighted variants (reweighted: point ~1e-13,
#    SE has a small feols-weighted-cluster convention difference, pinned on the
#    Python side). alexCardazzi::lpdid()'s `nonabsorbing_lag` is NOT a faithful
#    Eq.13 (it clamps off-switches via treat_diff[<0]<-0 and uses a non-paper
#    boundary/window convention; it diverges ~0.01-0.05 from Eq.13 even on a
#    monotone no-off-switch panel) -> recorded in `meta` as a divergent third-party
#    reference, NOT a parity gate (the B2 alexCardazzi-pooled precedent).
#
#    APPENDED after the absorbing write_json with its OWN set.seed and distinct
#    object names; setFixest_ssc is NOT re-called -> the absorbing panel + golden
#    above are byte-identical.
# ============================================================
NA_L <- 3L; NA_PRE <- 3L; NA_POST <- 4L; NA_TT <- 14L
set.seed(424242)  # isolated; only the rnorm() calls below consume this stream

na_mk <- function(uid, dpath) {
  ufe <- rnorm(1, 0, 1.5)
  data.table(unit = uid, time = 1:NA_TT, treat = as.integer(dpath),
             y = ufe + 0.4 * (1:NA_TT) + 2.0 * as.integer(dpath) + rnorm(NA_TT, 0, 0.3))
}
na_rows <- list(); na_uid <- 0L
na_add <- function(n, gen) for (i in 1:n) { na_uid <<- na_uid + 1L; na_rows[[length(na_rows)+1]] <<- na_mk(na_uid, gen()) }
na_add(18, function() as.integer((1:NA_TT) >= sample(4:8, 1)))                          # enter & stay
na_add(12, function() { g <- sample(3:6, 1); ex <- g + sample(2:3, 1); re <- ex + sample(2:4, 1)
  d <- as.integer((1:NA_TT) >= g); d[(1:NA_TT) >= ex] <- 0L; if (re <= NA_TT) d[(1:NA_TT) >= re] <- 1L; d })  # reversal / re-entry
na_add(10, function() rep(1L, NA_TT))                                                   # stabilized treated (eq13 controls)
na_add(20, function() rep(0L, NA_TT))                                                   # never treated
na_dt <- rbindlist(na_rows); setorder(na_dt, unit, time)
na_panel_path <- file.path("benchmarks", "data", "lpdid_nonabsorbing_panel.csv")
fwrite(na_dt, na_panel_path)
message(sprintf("Wrote non-absorbing panel: %s (%d rows, %d units)",
                na_panel_path, nrow(na_dt), uniqueN(na_dt$unit)))

# ---- explicit per-window mask helpers (independent of the Python) ----
# Each unit is gap-free over 1:NA_TT; positions < 1 are treated as D=0 (the
# "untreated before the first observed period" convention - this is where alex's
# NA-exclude differs from the library's clamp-to-untreated).
na_Dwin  <- function(D,  a, b) { a <- max(a, 1L); if (b < a) integer(0) else D[a:b] }
na_ddwin <- function(dd, a, b) { a <- max(a, 1L); if (b < a) integer(0) else dd[a:b] }
na_clean_sample <- function(dt, mode, h) {
  dt[, {
    D <- treat; n <- length(D)
    dd <- D - c(0L, D[-n])                 # first diff; pre-series untreated -> dd[1]=D[1]
    cumD_before <- cumsum(c(0L, D))[1:n]   # D summed strictly before each position
    cumD_incl   <- cumsum(D)               # D summed THROUGH each position (inclusive)
    cdi <- function(k) if (k < 1L) 0L else cumD_incl[min(k, n)]
    role <- rep(NA_character_, n)
    for (t in 1:n) {
      tgt <- t + h; base <- t - 1L
      if (tgt < 1L || tgt > n || base < 1L) next         # need baseline y_{t-1} and target y_{t+h}
      if (mode == "eq13") {
        if (h >= 0L) {
          treated <- dd[t] == 1L && all(na_Dwin(D, t - NA_L, t - 1L) == 0L) &&
                     all(na_ddwin(dd, t + 1L, t + h) == 0L)       # entry, D=0 on [t-L,t-1], no other change in (t,t+h]
          control <- all(na_ddwin(dd, t - NA_L, t + h) == 0L)     # no treatment change in [t-L,t+h]
        } else {
          m <- max(NA_L, -h)
          treated <- dd[t] == 1L && all(na_Dwin(D, t - m, t - 1L) == 0L)
          control <- all(na_ddwin(dd, t - m, t - 1L) == 0L)       # placebo window backward to t+h
        }
      } else {                                                    # eq12 first_entry
        entry <- dd[t] == 1L && cumD_before[t] == 0L              # first onset (D=0 strictly before t)
        if (h >= 0L) {
          treated <- entry && all(na_ddwin(dd, t + 1L, t + h) == 0L)  # stays treated through t+h
          control <- cdi(t + h) == 0L                            # untreated through t+h INCLUSIVE (Eq.12)
        } else {
          treated <- entry
          control <- cdi(t) == 0L
        }
      }
      if (isTRUE(treated)) role[t] <- "T" else if (isTRUE(control)) role[t] <- "C"
    }
    .(time = time, role = role)
  }, by = unit][!is.na(role)]
}
na_es_one <- function(h, reweight = FALSE, dt = na_dt) {
  cs <- na_clean_sample(dt, na_mode, h)
  if (nrow(cs) == 0L) return(c(NA_real_, NA_real_))
  d <- merge(dt, cs, by = c("unit", "time"))
  d[, `:=`(tb = time - 1L, tt = time + h)]
  d <- merge(d, dt[, .(unit, tb = time, yb = y)], by = c("unit", "tb"))
  d <- merge(d, dt[, .(unit, tt = time, yt = y)], by = c("unit", "tt"))
  d[, Dy := yt - yb]; d[, tr := as.integer(role == "T")]
  d[, has_c := any(tr == 0L), by = time]; d <- d[has_c == TRUE]    # drop event-times with no clean control
  if (uniqueN(d$tr) < 2L) return(c(NA_real_, NA_real_))
  if (reweight) {                                                 # per-event-time N/N_control inverse weights
    w <- d[, .(N = .N, nc = sum(tr == 0L)), by = time][, w := N / nc]
    d <- merge(d, w[, .(time, w)], by = "time")
    m <- feols(Dy ~ tr | time, data = d, weights = ~w, vcov = ~unit, warn = FALSE, notes = FALSE)
  } else {
    m <- feols(Dy ~ tr | time, data = d, vcov = ~unit, warn = FALSE, notes = FALSE)
  }
  c(unname(coef(m)["tr"]), unname(se(m)["tr"]))
}
na_es <- function(mode, reweight = FALSE, dt = na_dt) {
  na_mode <<- mode
  out <- list()
  for (h in 0:NA_POST) out[[as.character(h)]]  <- na_es_one(h, reweight, dt)
  for (h in 2:NA_PRE)  out[[as.character(-h)]] <- na_es_one(-h, reweight, dt)
  out
}

first_entry_es    <- na_es("eq12")
effect_stab_es    <- na_es("eq13")
effect_stab_rw_es <- na_es("eq13", reweight = TRUE)

# ---- monotone (no-off-switch) slice: PIN the "alex diverges even without off-switches"
# claim with committed evidence. On units whose treatment never decreases, alex's
# off-switch clamp is inert, yet alex still diverges from the paper-faithful Eq.13 (its
# non-paper boundary/window convention), so the recorded max post-horizon |alex - Eq.13|
# is well above 0 -> documents that the divergence is NOT only off-switch handling.
na_mono <- na_dt[, if (all(diff(treat) >= 0L)) .SD, by = unit]   # drop units with any turn-off
na_mode <<- "eq13"
mono_ours <- vapply(0:NA_POST, function(h) na_es_one(h, FALSE, na_mono)[1], numeric(1))
mono_alex <- tryCatch({
  am <- lpdid::lpdid(as.data.frame(na_mono), window = c(-NA_PRE, NA_POST), y = "y",
                     unit = "unit", time = "time", treat_status = "treat",
                     cluster = "unit", nonabsorbing_lag = NA_L)
  am$coeftable[(NA_PRE + 1L):(NA_PRE + 1L + NA_POST), "Estimate"]   # rows for h = 0..POST
}, error = function(e) rep(NA_real_, NA_POST + 1L))
alex_monotone_divergence <- max(abs(mono_ours - mono_alex), na.rm = TRUE)
message(sprintf("alex monotone-slice max|alex - Eq.13| (post h) = %.4f", alex_monotone_divergence))

# ---- alexCardazzi nonabsorbing_lag: divergent reference (recorded, NOT gated) ----
na_alex <- tryCatch({
  a <- lpdid::lpdid(as.data.frame(na_dt), window = c(-NA_PRE, NA_POST), y = "y",
                    unit = "unit", time = "time", treat_status = "treat",
                    cluster = "unit", nonabsorbing_lag = NA_L)
  hs <- (-NA_PRE):NA_POST  # alex coeftable rows map to horizons by position
  setNames(lapply(seq_along(hs), function(i)
    c(a$coeftable[i, "Estimate"], a$coeftable[i, "Std. Error"])), as.character(hs))
}, error = function(e) list(error = conditionMessage(e)))

na_golden <- list(
  meta = list(
    estimator = "LPDiD (Dube, Girardi, Jorda & Taylor 2025) - non-absorbing (Phase C2)",
    r_version = R.version.string,
    fixest_version = as.character(packageVersion("fixest")),
    lpdid_alexcardazzi_version = as.character(packageVersion("lpdid")),
    lpdid_alexcardazzi_commit = ALEX_SHA,
    seed = 424242L, L = NA_L, pre_window = NA_PRE, post_window = NA_POST,
    anchor = paste(
      "Both modes are anchored on an INDEPENDENT feols reconstruction of the paper's",
      "Eq. 12 (first_entry) and Eq. 13 (effect_stabilization) clean-sample restrictions",
      "(explicit per-window indexing). Validated == the library to ~1e-13 (point) and",
      "~1e-15 (SE) for the variance-weighted variants."),
    reweight_se_note = paste(
      "effect_stab_rw: POINT matches the library to ~1e-13; the reweighted SE has a",
      "small feols-weighted-cluster convention difference (~5e-5) vs the library, so the",
      "library reweighted SE is pinned as a regression guard on the Python side, not",
      "asserted against feols."),
    first_entry_note = paste(
      "first_entry (Eq. 12) has NO R-package analogue (alexCardazzi only exposes the",
      "effect-stabilization nonabsorbing_lag). It is anchored on the independent feols",
      "Eq. 12 recipe + a hand-computed Python micro-check. Its absorbing reduction",
      "(first_entry == absorbing on absorbing panels) is a separate internal check, NOT",
      "an R anchor for non-absorbing behavior."),
    alex_note = paste(
      "alexCardazzi::lpdid() `nonabsorbing_lag` is NOT a faithful paper Eq. 13: it clamps",
      "treat_diff[<0]<-0 (its clean-control window blocks only treatment turn-ONS, so a",
      "unit turning OFF inside [t-L,t+h] still counts as a control) and reuses a forward",
      "window for placebos. The library uses the literal 'no treatment change' (both",
      "directions) + the backward placebo window [t-max(L,-h),t-1] (more paper-faithful).",
      "alex diverges ~0.01-0.05 from Eq. 13 even on a monotone no-off-switch panel, so it",
      "is recorded below for transparency only, NOT used as a parity gate."),
    boundary_note = paste(
      "Boundary convention (closes the REGISTRY 'confirm vs R in PR-C2' item): periods",
      "before a unit's first observed period are treated as untreated/no-change (the",
      "library clamps pre-min_t to 0); alexCardazzi NA-excludes such rows (first-row lag",
      "is NA). A documented divergence, not a defect."),
    alex_monotone_post_divergence = alex_monotone_divergence  # max|alex - Eq.13| over post h on the monotone (no-off-switch) sub-panel; > 0 shows the divergence is not only off-switch handling
  ),
  first_entry = first_entry_es,
  effect_stab = effect_stab_es,
  effect_stab_rw = effect_stab_rw_es,
  alex_nonabsorbing_lag = na_alex
)
na_golden_path <- file.path("benchmarks", "data", "lpdid_nonabsorbing_golden.json")
write_json(na_golden, na_golden_path, auto_unbox = TRUE, pretty = TRUE, digits = 12, na = "null")
message(sprintf("Wrote non-absorbing golden: %s", na_golden_path))
