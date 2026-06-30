# Practitioner Workflow Scenarios for Performance Benchmarking

This document defines the **realistic practitioner workloads** used to evaluate
diff-diff's end-to-end performance. It is the methodology input for the
per-scenario scripts under `benchmarks/speed_review/` and the findings in
`docs/performance-plan.md`.

## Why this doc exists

The existing `benchmarks/` suite measures **isolated `fit()` calls on synthetic
200-20,000 unit panels** against R packages for accuracy parity. That tells us
whether our point estimates and SEs match `did::att_gt` and `fixest::feols`. It
does **not** tell us what an analyst sees when they run a full 8-step Baker et
al. (2025) workflow on a real BRFSS state-policy panel or a staggered geo
campaign. Without that, any "should we optimize X?" or "should we port X to
Rust?" decision is made on intuition, not data.

The scenarios below are the measurement surface for that decision. They are
chosen to:

1. Cover the six practitioner decision-tree branches in
   `docs/practitioner_decision_tree.rst` (simultaneous, staggered, reversible,
   dose, few-markets, survey).
2. Exercise the code paths added in v3.0-v3.1 that the old `benchmarks/` never
   touched: survey `SurveyDesign` (TSL, replicate weights, PSU-level
   multiplier bootstrap), `aggregate_survey`, dCDH (reversible, `L_max`),
   SyntheticDiD jackknife, ContinuousDiD dose-spline, and the 8-step
   chain (Bacon -> fit -> HonestDiD -> cross-estimator robustness).
3. Use defensibly realistic data shapes anchored to applied-econ paper
   conventions and industry writeups, **not** the 200 x 8 cookie cutter.

This is a **measurement doc**, not a wishlist. It does not propose new
features, does not propose optimizations, and does not propose new estimators.
Anything discovered during measurement that looks like a bug gets flagged
separately and routed to the silent-failures audit, not folded into a perf PR.

## How this doc is used

Each scenario in section 4 defines:

- **Persona / domain** - who runs this and why
- **Data shape** - n_units, n_periods, n_covariates, survey PSUs/strata,
  microdata rows if relevant
- **Estimator + params** - including `covariates`, `n_bootstrap`,
  `survey_design`, `aggregate`, any non-default knobs
- **Operation chain** - fit() is one step; the flow usually includes Bacon
  decomposition, parallel-trends inspection, sensitivity analysis, aggregation,
  and cross-estimator robustness. We time the **chain**, not just fit().
- **Source anchor** - which tutorial, paper, or industry reference the
  shape/workflow comes from

For each scenario, `benchmarks/speed_review/` hosts a script
(`bench_<scenario>.py`) that:

1. Generates (or loads) the data once.
2. Runs the full operation chain under `pyinstrument` and writes a flame HTML
   to `benchmarks/speed_review/baselines/profiles/<scenario>[_<scale>]_<backend>.html`.
3. Writes a wall-clock JSON breakdown (per operation + total) to
   `benchmarks/speed_review/baselines/<scenario>[_<scale>]_<backend>.json`.
   Multi-scale scenarios include the scale segment (`_small`, `_medium`,
   `_large`); single-scale scenarios (dose-response, reversible-dCDH)
   omit it.
4. Runs under both `DIFF_DIFF_BACKEND=python` and `DIFF_DIFF_BACKEND=rust`
   when Rust is available. Scenario 4 (SDiD few markets) skips the
   Python backend at the `large` scale by design because its
   pure-numpy jackknife would exceed 4 minutes per run without adding
   signal; every other (scenario, scale) runs under both backends. The
   Python-vs-Rust gap is the primary input to Rust-expansion decisions.

The scenario scripts are **not** meant to replace `run_benchmarks.py` (which
serves a different purpose: R-parity accuracy). They complement it.

## Ground rules for realism

- **No 200 x 8 synthetic panels.** The existing benchmarks already do that.
  Each scenario below is either a different shape entirely or a 200 x 8 panel
  wrapped in realistic downstream operations (bootstrap, survey, sensitivity).
- **End-to-end, not isolated `fit()`.** Practitioners chain operations. A 50ms
  fit inside a 999-replicate bootstrap wrapped in an 8-M-value HonestDiD loop
  is a ~45-second end-to-end run where 90%+ of time may be outside the fit
  call the old benchmark measured.
- **Cite why the shape is realistic.** Every scenario grounds its data shape
  in an applied-econ paper, a tutorial, an industry writeup, or a bundled
  real dataset. If a scenario cannot cite a source for its shape, it does
  not belong here.
- **Time includes I/O and prep.** The stopwatch starts at the first library
  call a practitioner would write in their notebook and ends at the last
  result-reporting call - `practitioner_next_steps()` or a `summary()`. Data
  generation (synthetic) is outside the stopwatch; data load
  (`load_mpdta()`, CSV read) is inside.

## Scenarios

### 1. Staggered Marketing Campaign - CS + Event Study + HonestDiD

- **Persona / domain.** Growth / performance-marketing data scientist at a
  tech or e-commerce company. A brand campaign rolls out to DMAs in two
  waves; analyst needs overall lift, event-study dynamics, and a sensitivity
  bound for the VP.
- **Data shape (scale sweep).** 26-period weekly panel, ~30% never-treated,
  2 covariates (`log_pop`, `baseline_spend`). Three scales:
    - **small** - 150 units, 2 cohorts (GeoLift DMA-panel analog; US DMAs
      cap at 210).
    - **medium** - 500 units, 3 cohorts (pooled multi-region or multi-year
      DMA panel).
    - **large** - 1,500 units, 3 cohorts (county-level staggered policy
      study; US has ~3,100 counties).
- **Estimator + params.**
  ```python
  CallawaySantAnna(
      control_group="never_treated",
      estimation_method="dr",
      cluster="unit",
      n_bootstrap=999,
  ).fit(data, outcome="y", unit="unit", time="period",
        first_treat="first_treat", covariates=["log_pop", "baseline_spend"],
        aggregate="all")
  ```
- **Operation chain.** (1) `BaconDecomposition.fit()` for TWFE diagnostic;
  (2) CS fit with `aggregate="all"` (populates simple, group, event_study);
  (3) inspect event-study pre-period ATTs for pre-trends; (4)
  `compute_honest_did(results, method="relative_magnitude", M=[0.5, 1.0, 1.5, 2.0])`;
  (5) robustness: refit with `SunAbraham()` and `ImputationDiD()` for
  cross-estimator comparison; (6) refit CS without covariates for the
  Baker-mandated with/without comparison; (7) `practitioner_next_steps()`.
- **Source anchor.** `docs/tutorials/02_staggered_did.ipynb` (staggered DGP
  pattern), `docs/tutorials/18_geo_experiments.ipynb` (DMA framing),
  Callaway & Sant'Anna (2021), Baker et al. (2025) 8-step workflow from
  `diff_diff/guides/llms-practitioner.txt`, GeoLift methodology docs for
  DMA panel conventions.

### 2. Brand Awareness Survey DiD - 2x2 with Survey Design

- **Persona / domain.** Brand / market-research analytics lead at a CPG
  or agency. Runs a pre/post awareness survey across test and control
  markets with complex sampling (strata + PSU clusters + unequal weights).
  Needs design-correct SEs or the CI is too narrow.
- **Data shape (scale sweep).** 12-period quarterly panel, high weight
  variation, JK1 delete-one-PSU replicate weights (replicate count equals
  the PSU count). Three scales:
    - **small** - 200 units, 10 strata × 4 PSUs = 40 replicate columns
      (Tutorial 17 analog).
    - **medium** - 500 units, 15 strata × 6 PSUs = 90 replicate columns
      (typical CPG quarterly brand-tracking wave).
    - **large** - 1,000 units, 20 strata × 8 PSUs = 160 replicate columns
      (multi-region brand tracking at scale, e.g. a national awareness
      study with 50+ sub-markets).
- **Estimator + params.** Two variants in the same script:
  ```python
  # (a) Analytical TSL path
  DifferenceInDifferences(robust=True).fit(
      data, outcome="awareness", treatment="treated", time="post",
      survey_design=SurveyDesign(weights="w", strata="stratum",
                                 psu="cluster", fpc="fpc"),
  )
  # (b) Replicate-weight path (JK1 delete-one-PSU weights produced by
  #     generate_survey_did_data(include_replicate_weights=True))
  SurveyDesign(weights="w", replicate_weights=rep_cols,
               replicate_method="JK1")
  ```
- **Operation chain.** (1) naive `DifferenceInDifferences()` with no survey
  design (for SE-inflation comparison); (2) `SurveyDesign.resolve()`;
  (3) design-aware fit (TSL path); (4) design-aware fit (replicate-weight
  path); (5) three funnel outcomes (awareness, consideration, purchase
  intent) refit in a loop; (6) `check_parallel_trends()` and placebo pre-
  period test; (7) `compute_honest_did()` with default M grid.
- **Source anchor.** `docs/tutorials/17_brand_awareness_survey.ipynb`
  (workflow shape), `docs/tutorials/16_survey_did.ipynb` (SurveyDesign
  API), CDC BRFSS 2024 technical docs (`_STSTR`/`_PSU`/`_LLCPWT`
  variable conventions for the 10-stratum / 40-PSU shape), Rao & Scott
  (1984) for design-effect weighting logic exercised by replicate path.

### 3. BRFSS State-Policy Microdata -> CS Panel

- **Persona / domain.** Health-policy / public-health researcher. Has BRFSS
  respondent-level microdata across 10 years, wants to estimate the effect
  of a staggered state policy (e.g., Medicaid expansion, smoking ban) on
  a design-correct outcome using `aggregate_survey()` to collapse microdata
  to a state-year panel, then a modern staggered estimator.
- **Data shape (scale sweep).** 50 states × 10 years × N respondents per
  state-year cell, 5 adoption cohorts staggered over the window. Three scales:
    - **small** - 50,000 rows (100/cell, 10 strata × 200 PSUs). Narrow
      analytic slice on a state-year grid.
    - **medium** - 250,000 rows (500/cell, 15 strata × 600 PSUs).
      Mid-range analytic slice on the same state-year grid.
    - **large** - 1,000,000 rows (2,000/cell, 20 strata × 1,000 PSUs).
      A realistic pooled 10-year multi-state analysis - comparable to the
      kind of panel built from BRFSS 2024's ~458K-record universe filtered
      and pooled across years. This is where practitioners actually live.
- **Estimator + params.**
  ```python
  panel, stage2 = aggregate_survey(
      microdata, by=["state", "year"], outcomes="y",
      survey_design=SurveyDesign(weights="finalwt", strata="strata", psu="psu"),
  )
  CallawaySantAnna(control_group="never_treated", estimation_method="reg",
                   n_bootstrap=199).fit(
      panel, outcome="y_mean", unit="state", time="year",
      first_treat="first_treat", survey_design=stage2, aggregate="all",
  )
  compute_honest_did(results, method="relative_magnitude", M=[0.5, 1.0, 1.5])
  ```
- **Operation chain.** (1) `aggregate_survey()` - the microdata-to-panel
  collapse; (2) CS fit with the second-stage SurveyDesign returned by
  `aggregate_survey` (pweight + geographic PSU clustering; `aggregate_survey`
  does not stratify the collapsed cell panel) and bootstrap at PSU level;
  (3) event-study pre-trend inspection; (4) HonestDiD sensitivity grid;
  (5) SunAbraham robustness refit using the same second-stage pweight
  SurveyDesign; (6) `practitioner_next_steps()`.
- **Source anchor.** `docs/practitioner_getting_started.rst` ("What If
  You Have Survey Data?" section), CDC BRFSS 2024 overview
  (cdc.gov/brfss/annual_data/2024), `diff_diff.prep.aggregate_survey`
  docstring + `docs/survey-roadmap.md`, CS paper for staggered ATT(g,t)
  inference.

### 4. Geo-Experiment Few Markets - SyntheticDiD + Jackknife

- **Persona / domain.** Growth marketing analyst running a small-market
  campaign test against a pool of control markets. Too few treated for
  asymptotic CS SE; uses SyntheticDiD with jackknife variance and a
  breakdown diagnostic for the VP.
- **Data shape (scale sweep).** 12 weekly periods (6 pre, 6 post),
  2 latent factors. Three scales:
    - **small** - 80 units, 5 treated (Tutorial 18 analog, DMA-scale
      geo-experiment).
    - **medium** - 200 units, 15 treated (zip-cluster-scale or
      multi-DMA geo experiment).
    - **large** - 500 units, 30 treated (zip-level or large-scale geo
      experiment; **Python backend skipped at this scale** because the
      pure-numpy Frank-Wolfe solver plus jackknife would need ~500 per-unit
      refits and exceed 4 minutes per run without adding signal beyond what
      medium scale already shows).
- **Estimator + params.**
  ```python
  SyntheticDiD(variance_method="jackknife", n_bootstrap=0).fit(...)
  # then also variance_method="bootstrap", n_bootstrap=200 for comparison
  # NOTE: bootstrap is now paper-faithful refit (re-estimates ω and λ via
  # Frank-Wolfe per draw); ~5–30× slower than placebo (panel-size dependent;
  # warm-start plumbing reduces the slowdown relative to cold-start). Plan
  # accordingly when timing.
  ```
- **Operation chain.** (1) SDiD fit with `variance_method="jackknife"` -
  exercises the leave-one-out refit loop; (2) SDiD fit with
  `variance_method="bootstrap"`, `n_bootstrap=200` for SE comparison
  (paper-faithful refit; expect order-of-magnitude longer wall-clock
  than jackknife on this scale); (3) `results.in_time_placebo()`;
  (4) `results.get_loo_effects_df()`; (5)
  `results.sensitivity_to_zeta_omega()`; (6)
  `results.get_weight_concentration()`. The bootstrap refit and the
  jackknife loop are now both significant time sinks;
  `sensitivity_to_zeta_omega` also refits.
- **Source anchor.** `docs/tutorials/18_geo_experiments.ipynb`,
  Arkhangelsky et al. (2021), Mercado Libre geo-experiment writeup
  (medium.com/mercadolibre-tech), Meta GeoLift methodology docs
  (facebookincubator.github.io/GeoLift - 10-treated / 10-20-control
  convention).

### 5. Reversible Treatment - dCDH with L_max and Survey TSL

- **Persona / domain.** Marketing analyst measuring an always-on-with-
  dark-periods campaign, or a health-policy researcher studying a policy
  that switches on and off. Reversible treatment breaks the absorbing-only
  staggered estimators; dCDH is the most general fit (LPDiD/TROP `non_absorbing`
  also handle it under stronger assumptions).
- **Data shape.** 120 groups x 10 periods, single-switch pattern per group,
  ~40% always-control, survey-weighted with 8 strata and 24 PSUs. Larger
  than the Tutorial's 80 x 6 demo to expose the `L_max` multi-horizon
  influence-function allocation that was added in v3.1.
- **Estimator + params.**
  ```python
  ChaisemartinDHaultfoeuille().fit(
      data, outcome="y", group="group", time="period", treatment="treated",
      L_max=3,
      survey_design=SurveyDesign(weights="pw", strata="stratum", psu="cluster"),
  )
  ```
- **Operation chain.** (1) dCDH fit with `L_max=3` (computes `DID_l` for
  l=1..3, dynamic placebos, sup-t bands, TWFE diagnostic); (2) snapshot
  `placebo_effect`, `overall_att`, `joiners_att`, `leavers_att` from the
  result object for pre-trend evidence and joiner/leaver inspection;
  (3) `compute_honest_did()` M-grid on the placebo event study;
  (4) heterogeneity refit with `heterogeneity="group"`. The TSL path for
  `L_max >= 1` is newer code (v3.1) and has not been profiled.
- **Source anchor.** `docs/practitioner_decision_tree.rst`
  ("Reversible Treatment (On/Off Cycles)"), de Chaisemartin & D'Haultfoeuille
  (2020), NBER WP 29873 (dynamic companion), R package
  `DIDmultiplegtDYN` as methodological reference, `docs/methodology/REGISTRY.md`
  dCDH section, `project_dcdh_shipped.md` for v3.1 feature set.

### 6. Pricing Dose-Response - ContinuousDiD Cubic Spline

- **Persona / domain.** Pricing / promo analyst at a retailer. Stores
  received varying discount levels; analyst wants the dose-response curve
  ATT(d), not just a binarized average. Requires Strong Parallel Trends.
- **Data shape.** 500 units (stores) x 6 quarterly periods, 1 cohort at
  period 3, dose drawn from log-normal (range 1-12 percentage points off
  baseline price), ~30% untreated (dose = 0). This is the Tutorial 14
  shape scaled from 200 to 500 units to stress the B-spline fitting.
- **Estimator + params.**
  ```python
  ContinuousDiD(degree=3, num_knots=1, n_bootstrap=199).fit(
      data, outcome="y", unit="unit", time="period", first_treat="first_treat",
      dose="dose", aggregate="dose",
  )
  ```
- **Operation chain.** (1) CDiD fit with `aggregate="dose"` - produces
  overall ATT, overall ACRT, and the dose-response curves; (2) extract
  `results.to_dataframe(level="dose_response")` and
  `level="group_time"` (event-study is not populated by a dose-only
  fit, so it is extracted in a separate step); (3) a second CDiD fit
  with `aggregate="eventstudy"` for pre-trend diagnostics (note the
  spelling: `fit(aggregate="eventstudy")` with no underscore, but
  `to_dataframe(level="event_study")` with underscore - see the
  correctness-adjacent observations in `performance-plan.md`);
  (4) compare to a binarized DiD fit on the same data to quantify
  information loss from binarizing; (5) alternate `degree=1` (linear)
  and (6) `num_knots=2` refits for spline-sensitivity. The dose-curve
  bootstrap loop (199 reps x spline refit) is the primary time sink.
- **Source anchor.** `docs/tutorials/14_continuous_did.ipynb`,
  Callaway, Goodman-Bacon & Sant'Anna (2024), `docs/methodology/REGISTRY.md`
  ContinuousDiD section.

## Backend and environment notes

All scenarios run under both backends where available:

```bash
DIFF_DIFF_BACKEND=python python benchmarks/speed_review/bench_<scenario>.py
DIFF_DIFF_BACKEND=rust   python benchmarks/speed_review/bench_<scenario>.py
```

The Python-vs-Rust gap is the primary input to the Rust-expansion decision in
`docs/performance-plan.md`. If Python is already within 2x of Rust for a
scenario, that scenario is a weak Rust-port candidate; if Python is 10x+
slower, it is a strong candidate.

Apple Silicon M4 note per `TODO.md`: a spurious numpy `RuntimeWarning` on
`matmul` for N > 260 does not affect correctness but can clutter profile
output. Scripts filter this warning so profiles stay clean.

## What is explicitly out of scope

- **Optimizations.** This doc defines the measurement surface. Actual
  performance fixes are separate PRs, each citing a specific
  `docs/performance-plan.md` finding.
- **R-parity benchmarking.** That is `benchmarks/run_benchmarks.py`'s job
  and remains valuable; these scenarios complement it.
- **Estimators without realistic practitioner flows.** TROP, EfficientDiD,
  StackedDiD, and BaconDecomposition are exercised via the robustness
  branches of scenarios 1 and 3; they do not get standalone scenarios
  here. If a future practitioner tutorial gives one of them a distinct
  end-to-end flow, a scenario can be added at that point.
- **Rust backend internals.** We measure the Rust backend as a black box
  (backend=rust wall-clock, backend=rust profile breakdown). Optimizing
  inside Rust is a separate concern handled by `rust/` crate owners.

## Pointers

- Scripts: `benchmarks/speed_review/bench_<scenario>.py`
- Raw results: `benchmarks/speed_review/baselines/<scenario>[_<scale>]_<backend>.json`
- Flame profiles: `benchmarks/speed_review/baselines/profiles/<scenario>[_<scale>]_<backend>.html`
  (gitignored; regenerated per run)
- Findings doc: `docs/performance-plan.md` ("Practitioner Workflow Baseline"
  section - per-scenario top-5 hot phases + recommended action category)
