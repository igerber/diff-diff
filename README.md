# diff-diff

<p align="center">
  <img src="https://raw.githubusercontent.com/igerber/diff-diff/main/diff-diff.png"
       alt="diff-diff: Difference-in-Differences causal inference in Python - sklearn-like API with Callaway-Sant'Anna, Synthetic DiD, Honest DiD, and Event Studies"
       width="800">
</p>

[![PyPI version](https://img.shields.io/pypi/v/diff-diff.svg)](https://pypi.org/project/diff-diff/)
[![Python versions](https://img.shields.io/pypi/pyversions/diff-diff.svg)](https://pypi.org/project/diff-diff/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/badge/diff-diff)](https://pepy.tech/projects/diff-diff)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19646175.svg)](https://doi.org/10.5281/zenodo.19646175)

A Python library for Difference-in-Differences (DiD) causal inference - sklearn-like estimators with statsmodels-style outputs, built for econometricians, marketing analysts, and data scientists running campaign-lift, policy, and staggered-rollout analyses.

## Installation

```bash
pip install diff-diff
```

For development:

```bash
git clone https://github.com/igerber/diff-diff.git
cd diff-diff
pip install -e ".[dev]"
```

## Quick Start

```python
import pandas as pd
from diff_diff import DifferenceInDifferences  # or: DiD

data = pd.DataFrame({
    'outcome': [10, 11, 15, 18, 9, 10, 12, 13],
    'treated': [1, 1, 1, 1, 0, 0, 0, 0],
    'post': [0, 0, 1, 1, 0, 0, 1, 1],
})

did = DifferenceInDifferences()
results = did.fit(data, outcome='outcome', treatment='treated', post='post')
print(results)              # DiDResults(ATT=3.0000, SE=1.7321, p=0.1583)
results.print_summary()     # full statsmodels-style table
```

## Documentation

- [Quickstart](https://diff-diff.readthedocs.io/en/stable/quickstart.html) - basic 2x2 DiD with column-name and formula interfaces, covariates, fixed effects, cluster-robust SEs
- [Choosing an Estimator](https://diff-diff.readthedocs.io/en/stable/choosing_estimator.html) - decision flowchart for picking the right estimator
- [Tutorials](https://diff-diff.readthedocs.io/en/stable/tutorials/01_basic_did.html) - hands-on Jupyter notebooks covering every estimator and design pattern
- [Troubleshooting](https://diff-diff.readthedocs.io/en/stable/troubleshooting.html) - common issues and solutions
- [R Comparison](https://diff-diff.readthedocs.io/en/stable/r_comparison.html) | [Python Comparison](https://diff-diff.readthedocs.io/en/stable/python_comparison.html) | [Benchmarks](https://diff-diff.readthedocs.io/en/stable/benchmarks.html) - validation results vs `did`, `synthdid`, `fixest`
- [API Reference](https://diff-diff.readthedocs.io/en/stable/api/index.html) - full API for all estimators, results classes, diagnostics, utilities

## For AI Agents

If you are an AI agent or LLM using this library, call `diff_diff.get_llm_guide()` for a concise API reference with an 8-step practitioner workflow (based on Baker et al. 2025). The workflow ensures rigorous DiD analysis - testing assumptions, running sensitivity analysis, and checking robustness, not just calling `fit()`.

```python
from diff_diff import get_llm_guide

get_llm_guide()                 # concise API reference
get_llm_guide("practitioner")   # 8-step workflow (Baker et al. 2025)
get_llm_guide("full")           # comprehensive documentation
get_llm_guide("autonomous")     # autonomous-agent variant
```

The guides are bundled in the wheel - accessible from a `pip install` with no network access. After estimation, call `practitioner_next_steps(results)` for context-aware guidance on remaining diagnostic steps.

## For Data Scientists

Measuring campaign lift? Evaluating a product launch? Rolling out a policy in waves? diff-diff handles the causal inference so you can focus on the business question.

- [Which method fits my problem?](https://diff-diff.readthedocs.io/en/stable/practitioner_decision_tree.html) - start from your business scenario (campaign in some markets, staggered rollout, survey data) and find the right estimator
- [Getting started for practitioners](https://diff-diff.readthedocs.io/en/stable/practitioner_getting_started.html) - end-to-end walkthrough from marketing campaign to causal estimate to stakeholder-ready result
- [Brand awareness survey tutorial](https://diff-diff.readthedocs.io/en/stable/tutorials/17_brand_awareness_survey.html) - full example with complex survey design, brand funnel analysis, and staggered rollouts
- Have BRFSS/ACS/CPS individual records? Use [`aggregate_survey()`](https://diff-diff.readthedocs.io/en/stable/api/prep.html) to roll respondent-level microdata into a geographic-period panel with inverse-variance precision weights for second-stage DiD

`BusinessReport` and `DiagnosticReport` are experimental preview classes that produce plain-English output and a structured `to_dict()` schema from any fitted result - wording and schema will evolve. See [docs/methodology/REPORTING.md](https://github.com/igerber/diff-diff/blob/main/docs/methodology/REPORTING.md) for usage and stability notes.

## Practitioner Workflow (Baker et al. 2025)

For rigorous DiD analysis, follow these 8 steps. Skipping diagnostic steps produces unreliable results.

1. **Define target parameter** - ATT, group-time ATT(g,t), or event-study ATT_es(e). State whether weighted or unweighted.
2. **State identification assumptions** - which parallel trends variant (unconditional, conditional, PT-GT-Nev, PT-GT-NYT), no-anticipation, overlap.
3. **Test parallel trends** - simple 2x2: `check_parallel_trends()`, `equivalence_test_trends()`; staggered: inspect CS event-study pre-period coefficients (generic PT tests are invalid for staggered designs). Insignificant pre-trends do NOT prove PT holds.
4. **Choose estimator** - staggered adoption -> CS/SA/BJS (NOT plain TWFE); few treated units -> SDiD; factor confounding -> TROP; simple 2x2 -> DiD. Run `BaconDecomposition` to diagnose TWFE bias.
5. **Estimate** - `estimator.fit(data, ...)`. Always print the cluster count first and choose inference method based on the result (cluster-robust if >= 50 clusters, wild bootstrap if fewer - for DifferenceInDifferences pass `cluster=`; TwoWayFixedEffects auto-clusters at unit level).
6. **Sensitivity analysis** - `compute_honest_did(results)` for bounds under PT violations (MultiPeriodDiD, CS, or dCDH natively; the TwoWayFixedEffects `event_study=True` surface and a StackedDiD `results.aggregate('event_study')` container also admit - Stacked needs `kappa_pre >= 2`), `run_all_placebo_tests()` for 2x2 falsification, specification comparisons for staggered designs.
7. **Heterogeneity** - CS: `results.aggregate('group')`/`'event_study'` (post-fit, no refit); SA: `results.event_study_effects` / `to_dataframe(level='cohort')`; Stacked: `results.aggregate('event_study')`/`'simple'` post-fit views (surface always computed since 3.9); EDiD: `results.aggregate(...)` post-fit from retained EIFs (3.9); ImputationDiD/TwoStageDiD: `results.aggregate(...)` post-fit from panel-backed kits (3.9); ContinuousDiD: `results.aggregate('dose'/'simple'/'event_study')` post-fit (3.9; dose/simple are views, event_study recomputes); subgroup re-estimation.
8. **Robustness** - compare 2-3 estimators (CS vs SA vs BJS), report with and without covariates (shows whether conditioning drives identification), present pre-trends and sensitivity bounds.

Full guide: `diff_diff.get_llm_guide("practitioner")`.

## Estimators

- [DifferenceInDifferences](https://diff-diff.readthedocs.io/en/stable/api/estimators.html) - basic 2x2 DiD with robust/cluster-robust SEs, wild bootstrap, formula interface, and fixed effects
- [TwoWayFixedEffects](https://diff-diff.readthedocs.io/en/stable/api/estimators.html) - panel data DiD with unit and time fixed effects via within-transformation or dummies
- [MultiPeriodDiD](https://diff-diff.readthedocs.io/en/stable/api/estimators.html) - event study design with period-specific treatment effects for dynamic analysis (deprecated 3.9 - use TwoWayFixedEffects `event_study=True`)
- [CallawaySantAnna](https://diff-diff.readthedocs.io/en/stable/api/staggered.html) - Callaway & Sant'Anna (2021) group-time ATT estimator for staggered adoption
- [ChaisemartinDHaultfoeuille](https://diff-diff.readthedocs.io/en/stable/api/chaisemartin_dhaultfoeuille.html) - de Chaisemartin & D'Haultfœuille (2020/2022) for **reversible (non-absorbing) treatments** with multi-horizon event study, normalized effects, cost-benefit delta, sup-t bands, and dynamic placebos. The most general option for treatments that switch on AND off (see also `LPDiD`/`TROP` `non_absorbing`). Alias `DCDH`.
- [SunAbraham](https://diff-diff.readthedocs.io/en/stable/api/staggered.html) - Sun & Abraham (2021) interaction-weighted estimator for heterogeneity-robust event studies
- [ImputationDiD](https://diff-diff.readthedocs.io/en/stable/api/imputation.html) - Borusyak, Jaravel & Spiess (2024) imputation estimator, most efficient under homogeneous effects
- [TwoStageDiD](https://diff-diff.readthedocs.io/en/stable/api/two_stage.html) - Gardner (2022) two-stage estimator with GMM sandwich variance
- [SpilloverDiD](https://diff-diff.readthedocs.io/en/stable/api/spillover.html) - Butts (2021) ring-indicator spillover-aware DiD identifying direct effect on treated + per-ring spillover on near-control units; handles non-staggered and staggered timing; supports survey-design variance under `survey_design=` for HC1 / CR1 (Wave E.1 Binder TSL) and Conley (Wave E.2 panel-aware stratified-Conley sandwich on per-period PSU totals; extended in Wave E.2 follow-up to `conley_lag_cutoff > 0` via panel-block composition with within-PSU serial Bartlett HAC — `lag>0` requires an effective PSU via explicit `survey_design.psu` or injected `cluster=<col>`); `SurveyDesign.subpopulation()` preserves full-design `n_psu` / `df_survey` via zero-padded scores (Wave E.3, R `svyrecvar(subset())` form)
- [SyntheticDiD](https://diff-diff.readthedocs.io/en/stable/api/estimators.html) - Synthetic DiD combining standard DiD and synthetic control for few treated units
- [SyntheticControl](https://diff-diff.readthedocs.io/en/stable/api/synthetic_control.html) - Abadie, Diamond & Hainmueller (2010) classic synthetic control for a single treated unit (donor-weight counterfactual, nested/cv/inverse-variance/custom V; in-space placebo permutation inference via `in_space_placebo()`, plus ADH-2015 `leave_one_out()` + `in_time_placebo()` robustness, Firpo-Possebom (2018) test-inversion confidence sets, and Chernozhukov-Wüthrich-Zhu (2021) conformal inference)
- [TripleDifference](https://diff-diff.readthedocs.io/en/stable/api/triple_diff.html) - triple difference (DDD) estimator for designs requiring two criteria for treatment eligibility; serves both the 2x2x2 and the staggered-adoption design from one signature (`fit(..., first_treat=)` selects the staggered engine)
- [ContinuousDiD](https://diff-diff.readthedocs.io/en/stable/api/continuous_did.html) - Callaway, Goodman-Bacon & Sant'Anna (2024) continuous treatment DiD with dose-response curves
- [HeterogeneousAdoptionDiD](https://diff-diff.readthedocs.io/en/stable/api/had.html) - de Chaisemartin, Ciccia, D'Haultfœuille & Knau (2026) for designs where **no unit remains untreated**; local-linear estimator at the dose support boundary returning Weighted Average Slope (WAS) on Design 1' (`d̲ = 0` / QUG) or `WAS_{d̲}` on Design 1 (`d̲ > 0`, continuous-near-d̲ or mass-point), with a multi-period event-study extension (last-treatment cohort, pointwise CIs). **Panel-only** in this release - repeated cross-sections rejected by the validator. Alias `HAD`.
- [RegressionDiscontinuity](https://diff-diff.readthedocs.io/en/stable/api/regression_discontinuity.html) - Calonico, Cattaneo & Titiunik (2014) sharp, fuzzy, AND covariate-adjusted regression discontinuity with robust bias-corrected inference and rdrobust-parity bandwidth selection (all 10 selectors, mass-point handling; fuzzy via `takeup=` with a first-stage block and weak-identification warning; covariates via `covariates=` - CCFT 2019, same estimand, covariate-aware bandwidths). Canonical `att` is the bias-corrected estimate with a coherent robust CI (rdrobust's printed headline is `att_conventional`). Alias `RDD`.
- [StackedDiD](https://diff-diff.readthedocs.io/en/stable/api/stacked_did.html) - Wing, Freedman & Hollingsworth (2024) stacked DiD with Q-weights and sub-experiments; optional covariate balancing (Ustyuzhanin 2026)
- [EfficientDiD](https://diff-diff.readthedocs.io/en/stable/api/efficient_did.html) - Chen, Sant'Anna & Xie (2025) efficient DiD with optimal weighting for tighter SEs
- [TROP](https://diff-diff.readthedocs.io/en/stable/api/trop.html) - Triply Robust Panel estimator (Athey et al. 2025) with nuclear norm factor adjustment
- [StaggeredTripleDifference](https://diff-diff.readthedocs.io/en/stable/api/staggered.html#staggeredtripledifference) - Ortiz-Villavicencio & Sant'Anna (2025) staggered DDD with group-time ATT (deprecated 3.9 - use `TripleDifference` with `first_treat=`)
- [WooldridgeDiD](https://diff-diff.readthedocs.io/en/stable/api/wooldridge_etwfe.html) - Wooldridge (2023, 2025) ETWFE: saturated OLS, logit/Poisson QMLE (ASF-based ATT). Alias `ETWFE`.
- [LPDiD](https://diff-diff.readthedocs.io/en/stable/api/lpdid.html) - Dube, Girardi, Jorda & Taylor (2025) Local Projections DiD: per-horizon long-difference event study on clean controls (no negative weighting), variance- or equally-weighted ATT, for absorbing or non-absorbing (reversible) treatment
- [ChangesInChanges](https://diff-diff.readthedocs.io/en/stable/api/changes_in_changes.html) - Athey & Imbens (2006) nonlinear/distributional DiD for the 2x2 design: full counterfactual distribution and quantile treatment effects via CDF transformation, plus the QDiD comparison estimator via `method="qdid"`; bootstrap inference; R qte parity. Alias `CiC`
- [BaconDecomposition](https://diff-diff.readthedocs.io/en/stable/api/bacon.html) - Goodman-Bacon (2021) decomposition for diagnosing TWFE bias in staggered settings

## Diagnostics & Sensitivity

- [RD Plots](https://diff-diff.readthedocs.io/en/stable/api/regression_discontinuity.html) - Calonico, Cattaneo & Titiunik (2015) optimal data-driven RD plots (`RDPlot`): all 8 rdrobust `binselect` bin selectors, implied-scale/WIMSE-weight reporting, optional matplotlib rendering
- [Manipulation Testing](https://diff-diff.readthedocs.io/en/stable/api/regression_discontinuity.html) - Cattaneo, Jansson & Ma (2020) density-discontinuity test (`RDDensityTest`): rddensity 3.0 parity, robust bias-corrected inference, unrestricted/restricted models, mass-point adjustment
- [Parallel Trends Testing](https://diff-diff.readthedocs.io/en/stable/api/diagnostics.html) - simple and Wasserstein-robust parallel trends tests, equivalence testing (TOST)
- [Placebo Tests](https://diff-diff.readthedocs.io/en/stable/api/diagnostics.html) - placebo timing, group, permutation, leave-one-out
- [Honest DiD](https://diff-diff.readthedocs.io/en/stable/api/honest_did.html) - Rambachan & Roth (2023) sensitivity analysis: robust CI under PT violations, breakdown values
- [Pre-Trends Power Analysis](https://diff-diff.readthedocs.io/en/stable/api/pretrends.html) - Roth (2022) minimum detectable violation and power curves
- [Power Analysis](https://diff-diff.readthedocs.io/en/stable/api/power.html) - analytical and simulation-based MDE, sample size, power curves for study design
- [MMM Calibration Export](https://diff-diff.readthedocs.io/en/stable/api/mmm.html) - convert experiment results into MMM calibration inputs: PyMC-Marketing lift-test frames and Google Meridian lognormal ROI priors
- Conley spatial HAC SE (`vcov_type="conley"`) on cross-sectional `LinearRegression` / `compute_robust_vcov` plus panel `DifferenceInDifferences` / `MultiPeriodDiD` / `TwoWayFixedEffects` (with `conley_lag_cutoff` for within-unit Bartlett temporal HAC) - Conley (1999) spatial-correlation-aware SEs with parity vs R `conleyreg` on cross-sectional + panel fixtures, optional combined spatial + cluster product kernel via explicit `cluster=`, auto-activating sparse k-d-tree fast path for `n > 5_000`

## Survey Support

Most estimators accept an optional `survey_design` parameter (or `survey=` / `weights=` for `HeterogeneousAdoptionDiD`) for design-based variance estimation. Coverage and supported weight types vary by estimator - see the [Survey Design Support compatibility matrix](https://diff-diff.readthedocs.io/en/stable/choosing_estimator.html#survey-design-support) for the per-estimator support table.

- **Design elements available across the supported set**: strata, PSU, FPC, lonely PSU handling, nest. Weight types vary by estimator: some surfaces (e.g. CallawaySantAnna, StackedDiD, the HAD continuous path) accept `pweight` only; others accept `pweight` / `fweight` / `aweight`.
- **Variance methods**: Taylor Series Linearization (TSL via Binder 1983), replicate weights (BRR / Fay / JK1 / JKn / SDR), survey-aware bootstrap
- **Diagnostics**: DEFF per coefficient, effective n, subpopulation analysis, weight trimming, CV on estimates
- **Repeated cross-sections**: `CallawaySantAnna(panel=False)` for BRFSS, ACS, CPS
- **Weight calibration / raking**: upstream by design - pair with Meta's [balance](https://import-balance.org/) package, whose `balance.interop.diff_diff` adapter hands raked samples straight to diff-diff; see the [composition-drift tutorial](https://diff-diff.readthedocs.io/en/stable/tutorials/26_composition_drift_calibration.html)

No other Python or R DiD package offers design-based variance estimation for modern heterogeneity-robust estimators.

## Requirements

- Python 3.9 - 3.14
- numpy >= 1.20
- pandas >= 1.3
- scipy >= 1.10

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black diff_diff tests
ruff check diff_diff tests
```

## References

This library implements methods from a wide body of econometric and causal-inference research. See the full bibliography on [Read the Docs](https://diff-diff.readthedocs.io/en/stable/references.html) for citations spanning DiD foundations, modern staggered estimators, sensitivity analysis, and synthetic controls.

## Citing diff-diff

If you use diff-diff in your research, please cite it:

```bibtex
@software{diff_diff,
  title = {diff-diff: Difference-in-Differences Causal Inference for Python},
  author = {Gerber, Isaac},
  year = {2026},
  url = {https://github.com/igerber/diff-diff},
  doi = {10.5281/zenodo.19646175},
  license = {MIT},
}
```

The DOI above is the Zenodo concept DOI - it always resolves to the latest release. To cite a specific version, look up its versioned DOI on [the Zenodo project page](https://doi.org/10.5281/zenodo.19646175).

See [`CITATION.cff`](https://github.com/igerber/diff-diff/blob/main/CITATION.cff) for the full citation metadata.

**Note on authorship**: academic citation (`CITATION.cff`, the BibTeX above) lists individual authors with ORCIDs per scholarly convention. Package metadata surfaces (`pyproject.toml`, Sphinx docs) list "diff-diff contributors" to acknowledge the collective - see [`CONTRIBUTORS.md`](https://github.com/igerber/diff-diff/blob/main/CONTRIBUTORS.md) for the full list.

## License

MIT License
