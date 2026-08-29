# diff-diff Tutorials

This directory contains Jupyter notebook tutorials demonstrating the features of the `diff-diff` library.

## Notebooks

### 1. Basic DiD (`01_basic_did.ipynb`)
Introduction to Difference-in-Differences with `diff-diff`:
- Basic 2x2 DiD estimation
- Column-name and formula interfaces
- Adding covariates
- Fixed effects (dummy and absorbed)
- Two-Way Fixed Effects (TWFE)
- Cluster-robust standard errors
- Wild cluster bootstrap

### 2. Staggered DiD (`02_staggered_did.ipynb`)
Handling staggered treatment adoption with the Callaway-Sant'Anna estimator:
- Understanding staggered adoption
- Problems with TWFE in staggered settings
- **Goodman-Bacon decomposition**: Diagnosing *why* TWFE fails
- Group-time effects ATT(g,t)
- Aggregation methods (simple, group, event-study)
- Control group specifications
- Visualization

### 3. Synthetic DiD (`03_synthetic_did.ipynb`)
Synthetic Difference-in-Differences for few treated units:
- When to use Synthetic DiD
- Understanding unit and time weights
- Pre-treatment fit diagnostics
- Inference methods (bootstrap, placebo)
- Regularization tuning
- Comparison with standard DiD

### 4. Parallel Trends (`04_parallel_trends.ipynb`)
Testing assumptions and diagnostics:
- Visual inspection of trends
- Simple parallel trends tests
- Robust Wasserstein-based tests
- Equivalence testing (TOST)
- Placebo tests (timing, group, permutation)
- Event study as a diagnostic
- What to do if parallel trends fails

### 15. Efficient DiD (`15_efficient_did.ipynb`)
Efficient Difference-in-Differences (Chen, Sant'Anna & Xie 2025):
- Optimal weighting across comparison groups and baselines
- PT-All vs PT-Post assumptions
- Efficiency gains vs Callaway-Sant'Anna
- Event study and group-level aggregation
- Bootstrap inference and diagnostics

### 16. Wooldridge ETWFE (`16_wooldridge_etwfe.ipynb`)
Wooldridge Extended Two-Way Fixed Effects (ETWFE) for staggered DiD:
- Basic OLS estimation with cohort x time ATT cells
- Aggregation methods: event-study, group, calendar, simple
- Poisson QMLE for count / non-negative outcomes
- Logit for binary outcomes
- Comparison with Callaway-Sant'Anna
- Delta-method standard errors

### Survey-Aware DiD (`16_survey_did.ipynb`)
Survey-aware DiD with complex sampling designs (strata, PSU, FPC, weights):
- Why survey design matters for DiD inference
- Setting up `SurveyDesign` (weights, strata, PSU, FPC)
- Basic DiD and staggered DiD with survey design
- Replicate weights (JK1, BRR, Fay, JKn)
- Subpopulation analysis
- DEFF diagnostics
- Repeated cross-sections with survey design

### 17. Brand Awareness Survey (`17_brand_awareness_survey.ipynb`)
Practitioner walkthrough for measuring brand-campaign lift on survey data with complex sampling:
- The brand-tracker problem framed for marketing analytics
- Naive vs survey-aware DiD comparison (overconfidence under naive)
- `SurveyDesign` setup (strata, PSU, FPC, weights) wired into the fit
- Funnel-metric extension across awareness / consideration / purchase intent
- Diagnostics (parallel trends, placebo, automated `practitioner_next_steps()`)
- Stakeholder communication template

### 18. Geo-Experiment Analysis with SyntheticDiD (`18_geo_experiments.ipynb`)
Practitioner walkthrough for marketing analytics teams measuring geo-experiment lift:
- The geo-experiment problem framed for marketing analytics
- Synthetic panel of 80 markets with simulated campaign launch
- `SyntheticDiD` fit, diagnostics, and inference (placebo + bootstrap)
- Unit weights and time weights interpretation
- Stakeholder communication template (Tutorial 17 leadership-communication pattern)

### 19. dCDH Marketing Pulse Campaigns (`19_dcdh_marketing_pulse.ipynb`)
Practitioner walkthrough for measuring lift from on/off promotional pulses across markets, where treatment can switch in both directions:
- The marketing-pulse problem framed for reversible (non-absorbing) treatment
- TWFE decomposition diagnostic (`twowayfeweights`) showing why standard regression misleads on reversible panels (de Chaisemartin & D'Haultfoeuille 2020 Theorem 1)
- `DCDH` Phase 1: DID_M, joiners-vs-leavers decomposition, single-lag placebo
- Multi-horizon event study with `L_max` + multiplier bootstrap
- Stakeholder communication template + drift guards

### 20. HAD for National Brand Campaign with Regional Spend Intensity (`20_had_brand_campaign.ipynb`)
Practitioner walkthrough for measuring per-dollar lift when every market is treated at a different dose level and no never-treated unit exists (comparison comes from the dose variation across markets):
- The measurement problem framed for heterogeneous-adoption (no-untreated-control) panels
- `HAD` overall fit on a 2-period collapse, with `design="auto"` resolving to `continuous_near_d_lower` (Design 1) and target `WAS_d_lower` (per-$1K marginal effect above the lightest-touch DMA's spend)
- Multi-week event study showing per-week dynamics with pre-launch placebos
- Stakeholder communication template flagging the Assumption 5/6 identification caveat
- Companion drift-test file (`tests/test_t20_had_brand_campaign_drift.py`)

### 21. HAD Pre-test Workflow (`21_had_pretest_workflow.ipynb`)
Composite pre-test walkthrough for `HeterogeneousAdoptionDiD`, building on Tutorial 20's brand-campaign framing on a panel where the dose distribution has a strictly positive but very near-zero lower bound (so the QUG step fails-to-reject `H0: d_lower = 0`):
- Paper Section 4.2 step taxonomy (QUG support-infimum, parallel pre-trends, linearity)
- `did_had_pretest_workflow` on a two-period collapse (the overall battery, selected from the panel shape): Step 1 + Step 3 only, verdict explicitly flags Step 2 as deferred
- Upgrade to the multi-week panel (the event-study battery): adds the joint pre-trends Stute and joint homogeneity Stute diagnostics (none of the three testable steps reject)
- Side panel comparing `yatchew_hr_test` `null="linearity"` (default, paper Theorem 7) vs `null="mean_independence"` (Phase 4 R-parity with R `YatchewTest::yatchew_test(order=0)`)
- Companion drift-test file (`tests/test_t21_had_pretest_workflow_drift.py`)

### 22. HAD Survey-Weighted Workflow (`22_had_survey_design.ipynb`)
End-to-end HAD walkthrough on a BRFSS-shape stratified survey design (5 strata x 6 PSUs/stratum x 2 states/PSU = 60 states; post-stratification raking weights with CV ~ 0.30; FPC = 30 PSUs/stratum). Demonstrates the `SurveyDesign(strata=...)` path through the Stute pretest family that PR #432 (2026-05-14) unblocked:
- Naive vs survey-aware HAD headline fit on a two-period collapse, with side-by-side ATT / SE / CI table
- Why the SE inflation is modest for HAD (local-linear at d_lower IF concentration vs full-panel regression coefficients)
- Event-study fit with sup-t cband under the survey design
- `did_had_pretest_workflow` on both overall and event-study paths under `survey_design=`, walking the Phase 4.5 C0 QUG-deferred verdict suffix and the stratified-clustered Stute multiplier bootstrap
- Companion drift-test file (`tests/test_t22_had_survey_design_drift.py`)

### 23. SpilloverDiD on a TVA-style Spillover Panel (`23_spillover_tva.ipynb`)
Practitioner workflow for `SpilloverDiD` (Butts 2021 ring-indicator estimator + Gardner 2022 two-stage residualize-then-fit) on a synthetic TVA-style panel (4 periods, 200 units = 25 treated + 120 near-control + 55 far-control) reproducing the Butts §4 Table 1 Panel A ~40% understatement direction:
- When place-based interventions cause geographic spillovers, naive multi-period TWFE on the full sample understates the direct effect because near-controls absorb the spillover (here: ATT recovers as -4.29 vs true `tau_total = -7.4`, a 42% understatement)
- `SpilloverDiD(rings=[0.0, 100.0], conley_coords=("lat", "lon"))` cleanly recovers both `tau_total` (-7.34) and the near-band spillover coefficient `delta_1` (-4.53)
- Choosing the spillover bandwidth via a `rings` sensitivity grid at outer edges 50 / 100 / 150 / 200 km, with the documented "undershooting `d_bar`" failure mode at 50 km
- Conley spatial-HAC variance under `vcov_type="conley", conley_cutoff_km=100, conley_lag_cutoff in {0, 1}` — the cutoff = `d_bar` choice follows Butts §3.1, while the `conley_lag_cutoff` serial extension is the library's documented Wave E.2 follow-up synthesis with Newey-West-style serial Bartlett HAC (per REGISTRY "Variance (Wave E.2 follow-up)")
- Companion drift-test file (`tests/test_t23_spillover_tva_drift.py`)

### 24. Staggered Rollout vs a Collapsed 2×2 (`24_staggered_vs_collapsed_power.ipynb`)
Power-analysis decision guide for geo experiments (framed on a 50-state staggered rollout) on when to use Callaway-Sant'Anna vs collapsing to a familiar pre/post 2×2:
- Why the collapsed 2×2 silently targets a *diluted* estimand (and how often its CI misses the true effect-on-treated)
- The CS event study vs the 2×2's single diluted number
- How the minimum detectable lift (MDE) changes for each estimator as the rollout gets more staggered — the power gap is a *fast-rollout* phenomenon that closes to near parity as staggering increases
- When a clean-tail 2×2 is unbiased, the small-holdout and few-clusters caveats, and a CS-vs-2×2 decision guide
- Fully self-contained: runs live (no committed data files)

### 25. Synthetic Control for a Policy Evaluation: Two Routes to Inference (`25_synthetic_control_policy.ipynb`)
A single state adopts a clean-energy standard and no other state is a clean control - `SyntheticControl` builds a weighted donor-blend counterfactual, and with one treated unit the analytical inference fields are NaN by design, so the tutorial walks both genuine inference routes:
- Philosophy A - compare across regions: the ADH (2010) in-space placebo permutation (`in_space_placebo()`, RMSPE-ratio rank p-value) and the Firpo-Possebom (2018) confidence set
- Philosophy B - compare across time: Chernozhukov-Wuthrich-Zhu (2021) conformal inference, inverting a permutation-over-time test into a p-value for a hypothesized effect path and per-post-period confidence intervals
- The two philosophies side by side: what each does and does not protect against, and how to report them together
- Fully self-contained: runs live (no committed data files)

### 26. Composition Drift & Survey Calibration with balance (`26_composition_drift_calibration.ipynb`)
The failure-mode companion to Meta balance's `balance_diff_diff_brfss` tutorial: when non-response drift correlates with treatment timing, the design-weight DiD itself is biased and calibration becomes essential for the *causal* estimand:
- BRFSS-style smoking-ban DGP with no systematic arm-specific trends (parallel trends hold in expectation; planted ATT -3.0pp, realized -2.98pp) and treatment-correlated non-response drift
- Design-weight Callaway-Sant'Anna overstates the ATT ~35% with *clean pre-trends* (pre-trend tests are not a safety net)
- A per-wave national rake fails (margins satisfied in aggregate, arm-level composition untouched); per-state raking - BRFSS's own granularity - recovers the truth
- The seam both ways: native `SurveyDesign` + `aggregate_survey` 3-liner vs the `balance.interop.diff_diff` adapter (`to_panel_for_did` / `fit_did`), with an exact-parity assert
- Estimator sweep (CS / SunAbraham / ImputationDiD), `survey_metadata` DEFF diagnostics, and `as_balance_diagnostic` cross-package diagnostics
- Requires `pip install "balance>=0.21"` (this tutorial only); fully self-contained data

### 27. When the Average Hides the Action: Distributional DiD with Changes-in-Changes (`27_cic_distributional_effects.ipynb`)
A loyalty program looks dead on mean DiD ($0.22, p = 0.90) but in truth lifted the bottom half of the spend distribution - `ChangesInChanges` (Athey & Imbens 2006) recovers the full quantile-treatment-effect profile:
- Reading a QTE profile, and making joint "which quantiles moved" claims with sup-t uniform bands (they exclude zero for exactly tau = 0.05-0.50 here)
- The interior-range guardrail live: a short-support control sample triggers loud warnings and NaN tail inference instead of silent extrapolation
- The scale-equivariance centerpiece: levels-vs-logs flips mean DiD's verdict and shifts QDiD's profile by dollars, while CiC's counterfactual quantiles agree to floating-point precision (unconditional fits)
- Covariate-composition confounding fixed with `covariates=` (quantile-regression conditioning, qte `xformla` parity), and the `practitioner_next_steps()` close
- Companion drift-test file (`tests/test_t27_cic_distributional_effects_drift.py`); fully self-contained (runs live, no committed data files)

### 29. MMM Calibration with PyMC-Marketing Lift Tests (`29_mmm_calibration_pymc.ipynb`)
Executed end-to-end against the real framework: a staggered geo spend-boost experiment measured with `CallawaySantAnna`, exported via `to_pymc_marketing_lift_test`, and fed to `MMM.add_lift_test_measurements` - the demand-chasing-biased MMM's search-ROI posterior moves to the simulated truth and tightens:
- Grain reconciliation for a national-weekly lift row (`aggregate('simple')` + `scale=G_treated`), the `aggregate('total')` campaign headline, and the double-count guardrail
- With/without-calibration money plot plus in-notebook sampler-health and posterior acceptance asserts
- Requires `pip install "pymc-marketing==1.0.0"` on Python 3.12 (this tutorial only); cannot share an environment with the Meridian tutorial (conflicting `arviz` pins); fully self-contained data
- Companion drift-test file (`tests/test_t29_mmm_calibration_pymc_drift.py`)

### 30. MMM Calibration with Meridian ROI Priors (`30_mmm_calibration_meridian.ipynb`)
Executed end-to-end against the real framework: a geo-holdout launch experiment (the `roi_m` estimand exactly), exported via `to_meridian_roi_prior` from the `aggregate('total')` container, with `meridian_calibration_mask` and the generated `to_code()` snippet run verbatim:
- A national Meridian model that cannot see the geo holdout overcredits the launch; the experiment prior centers its ROI posterior on the truth and tightens it
- Window/spend/mass alignment asserts, default-vs-calibrated money plot, and sampler-health checks
- Requires `pip install "google-meridian==1.8.0"` on Python 3.12 (this tutorial only); cannot share an environment with the PyMC-Marketing tutorial (conflicting `arviz` pins); fully self-contained data
- Companion drift-test file (`tests/test_t30_mmm_calibration_meridian_drift.py`)

### 31. LWDiD Rolling Transformations (`31_lwdid.ipynb`)
Lee & Wooldridge (2026) rolling-transformation DiD: demeaning (Procedure 2.1) and detrending (Procedure 3.1) turn the panel into a cross-section with exact small-sample t inference:
- Prop 99 (common timing, one treated unit) and Walmart entry (staggered) replications
- Cluster-robust and wild-bootstrap inference, randomization tests, transformation diagnostics
- Companion drift-test file (`tests/test_t31_lwdid_drift.py`)

### 32. Double ML DiD (`32_dml_did.ipynb`)
Chang (2020) double/debiased machine learning DiD: cross-fitted ML nuisances on a staggered panel with engineered nonlinear confounding:
- The bias demo: a misspecified linear outcome learner misses by almost five standard errors while the adaptive sieve and a hand-rolled duck-typed `PolynomialRidge` learner recover the truth (double robustness in action)
- Cross-fitting diagnostics, event-study/group aggregation, HonestDiD sensitivity, bootstrap sup-t bands
- The repeated-cross-section lane (`panel=False`) with a `SurveyDesign` (PSU-cohesive folds, design-df t inference)
- Companion drift-test file (`tests/test_t32_dml_did_drift.py`)

## Running the Notebooks

1. Install diff-diff with dependencies:
```bash
pip install diff-diff
pip install matplotlib  # for visualizations
pip install jupyter     # to run notebooks
```

2. Start Jupyter:
```bash
jupyter notebook
```

3. Open any notebook and run the cells.

## Requirements

- Python 3.8+
- diff-diff
- numpy
- pandas
- matplotlib (optional, for visualizations)
