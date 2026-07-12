---
title: "diff-diff: Comprehensive Difference-in-Differences Causal Inference for Python"
tags:
  - difference-in-differences
  - causal-inference
  - econometrics
  - Python
  - treatment-effects
  - survey-data
authors:
  - name: Isaac Gerber
    orcid: 0009-0009-3275-5591
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 3 July 2026
bibliography: paper.bib
---

# Summary

`diff-diff` is a Python library for Difference-in-Differences (DiD) causal inference
analysis. It provides 23 estimators covering the full modern DiD toolkit - from classic
two-group/two-period designs through heterogeneity-robust staggered adoption methods,
synthetic control hybrids, and sensitivity analysis - under a consistent scikit-learn-style
API. Most estimators accept an optional `SurveyDesign` object for design-based variance
estimation with complex survey data, a capability absent from existing DiD software in any
language; the underlying design-based variance methodology is derived in the companion
preprint [@Gerber2026]. Point estimates are validated against established R packages to
machine precision, with standard errors matching exactly or to sub-percent relative
differences.

# Statement of Need

Difference-in-differences is the most widely used quasi-experimental research design in
applied economics and the social sciences. Since 2018, a wave of methodological advances
has addressed fundamental limitations of the conventional two-way fixed effects (TWFE)
estimator under staggered treatment adoption and heterogeneous effects [@Roth2023]. These
modern methods - including Callaway and Sant'Anna [-@Callaway2021], Sun and Abraham
[-@Sun2021], Borusyak, Jaravel, and Spiess [-@Borusyak2024], and others - are now standard
practice in applied work.

These methods are well served in R and Stata, but Python lacks a unified DiD library.
Practitioners working in Python-based data science workflows - increasingly common in
industry settings for marketing measurement, product experimentation, and policy
evaluation - must either context-switch to another language, reimplement methods from
scratch, or rely on partial implementations scattered across unrelated packages.

`diff-diff` fills this gap by providing a single-import library that covers 23 estimators
with a consistent API, survey-weighted inference, and numerical validation against R. It
is also the companion software for the design-based variance framework of @Gerber2026,
which establishes design-consistent standard errors for modern DiD estimators under
complex survey designs. It targets both applied researchers who need rigorous econometric
methods and data science practitioners who need accessible causal inference tools
integrated into Python workflows.

# State of the Field

The R ecosystem provides mature implementations across several packages: `did`
[@Callaway2021], `fixest` [@Berge2018], `synthdid` [@Arkhangelsky2021], and `HonestDiD`
[@Rambachan2023]; Stata offers `csdid` and `didregress`. Python coverage is partial and
fragmented. `pyfixest` [@pyfixest] brings `fixest`-style high-dimensional fixed-effects
regression to Python, including Sun-Abraham, two-stage, and local-projections estimators,
but is organized around its regression engine rather than the wider DiD taxonomy;
`differences` implements Callaway-Sant'Anna group-time estimation; `CausalPy` offers
Bayesian analysis of quasi-experiments, including synthetic control, without
staggered-adoption support. General-purpose causal inference toolkits such as `DoWhy` and
`EconML` target other identification strategies.

`diff-diff` was built as a new library, rather than as contributions to these packages,
because its central contribution is cross-cutting: one estimator contract, one shared
inference core, and an influence-function architecture that composes design-based survey
variance across the estimator taxonomy, with per-estimator support documented in a
compatibility matrix and unsupported combinations failing closed. To our knowledge, no existing DiD software
in any language provides design-based variance estimation for complex survey data, and no
Python package covers the modern estimator taxonomy end-to-end; `diff-diff` provides
both, validated against the R reference implementations where they exist.

# Key Features

**Breadth of methods.** `diff-diff` implements 23 estimators organized across the modern
DiD taxonomy. Classic designs include two-group/two-period DiD, two-way fixed effects, and
event-study estimation with period-specific effects. Heterogeneity-robust staggered-adoption
estimators include Callaway-Sant'Anna [@Callaway2021], Sun-Abraham [@Sun2021], imputation
[@Borusyak2024], two-stage [@Gardner2022], stacked [@Wing2024], efficient [@Chen2025], and
local-projections [@Dube2025] approaches, together with reversible-treatment DiD for
non-absorbing interventions [@deChaisemartin2020] and a ring-indicator estimator for
spatial spillovers [@Butts2021]. Synthetic-control hybrids include synthetic DiD
[@Arkhangelsky2021] and the classic synthetic control method [@Abadie2010]. Extended
designs include triple-difference and staggered triple-difference estimators
[@OrtizVillavicencio2025], continuous-treatment DiD with dose-response curves
[@Callaway2024], heterogeneous-adoption designs where no unit remains untreated
[@deChaisemartin2026], nonlinear ETWFE [@Wooldridge2025; @Wooldridge2023], distributional effects via
changes-in-changes and quantile DiD [@Athey2006], triply robust
panel estimation [@Athey2025], and sharp regression-discontinuity estimation with
robust bias-corrected inference [@Calonico2014]. Separate diagnostic and sensitivity tools - outside
the 23 estimators - include Goodman-Bacon decomposition [@GoodmanBacon2021], Honest DiD
sensitivity analysis [@Rambachan2023], placebo tests, and pre-trends power analysis
[@Roth2022].

**Survey-weighted inference.** A `SurveyDesign` class supports stratification, primary
sampling units, finite population corrections, and probability weights. Variance estimation
includes Taylor series linearization, five replicate weight methods (BRR, Fay's BRR, JK1,
JKn, SDR), and survey-aware bootstrap. The design-based variance result - that the
influence functions of modern DiD estimators satisfy the smoothness conditions of
@Binder1983, so stratified-cluster linearization yields design-consistent standard
errors - is derived in @Gerber2026. No other DiD package in any language provides
integrated survey support.

**Practitioner tooling.** Beyond estimation, `diff-diff` includes a practitioner decision
tree for estimator selection, an 8-step diagnostic workflow based on Baker et al.
[-@Baker2025], AI agent integration with structured next-steps guidance, and microdata
aggregation utilities for converting individual-level survey responses into
geographic-period panels suitable for DiD analysis.

# Software Design

Every estimator implements a common contract: a scikit-learn-style `fit()` with
`get_params()`/`set_params()` for configuration and rich results dataclasses with
`summary()`, `to_dict()`, and `to_dataframe()`; the classic regression estimators
additionally accept R-style formulas. Numerical work is deliberately centralized:
estimators solve their least-squares problems through a single shared linear-algebra
core, and analytical robust, cluster-robust, and survey variances route through one
shared sandwich-estimator path, so numerical hardening - rank-deficiency guards,
degrees-of-freedom corrections, small-cluster behavior - lands in one place. Estimators
whose inference is inherently resampling-based - synthetic DiD's placebo and jackknife
variance, for example - use method-specific variance paths. All estimators share one
joint-inference contract: inference fields (standard error, t-statistic, p-value,
confidence interval) are always computed together and become NaN together when inference
is not identified, rather than silently reporting partial results.

Two design choices carry the survey capability and the deployment story. First, the
regression- and influence-function-based estimators compute influence functions for their
target parameters, so design-based variance - Taylor series linearization over strata and
clusters, and replicate weights - routes through one shared survey-variance core rather
than requiring per-estimator derivations; resampling-based estimators such as synthetic
DiD and TROP compose survey designs through documented method-specific bootstrap,
placebo, and jackknife paths. Supported design-estimator combinations are listed in a
per-estimator compatibility matrix, and unsupported ones are rejected explicitly rather
than silently approximated. Second, the runtime
dependency footprint is minimal by policy - numpy, pandas, and scipy only - keeping the
library easy to install in restricted industry environments; high-dimensional fixed
effects are handled by within-transformation rather than by delegating to a heavier
econometrics stack. An optional Rust backend (via PyO3) accelerates compute-intensive
kernels such as synthetic-control weight solving and fixed-effects absorption; the Python
implementation remains canonical, equivalence between backends is enforced by the test
suite, and the library falls back to pure Python automatically when the extension is
unavailable.

# Research Impact Statement

`diff-diff` is the companion software of the design-based variance preprint [@Gerber2026]:
the framework derived there is implemented here, and the preprint's numerical results are
produced with the library. Correctness evidence ships with the repository as reproducible
material. Golden-file benchmarks pin point estimates against R's `did`, `synthdid`, and
`fixest` to machine precision (differences < 1e-10), including the canonical MPDTA
minimum-wage application of Callaway and Sant'Anna [-@Callaway2021], with standard errors
matching exactly for core estimators such as Callaway-Sant'Anna and basic DiD. Survey
variance is validated against R's `survey` package [@Lumley2004] on three real
complex-survey datasets - NHANES, RECS 2020, and the California API school data - with
test gaps below 1e-8 and typically below 1e-10. The library is distributed on PyPI with
tagged releases, has six months of continuous public development history (3,000+
commits), and is exercised by a CI test suite of more than 7,600 tests; 26 tutorial
notebooks and full API documentation are published on Read the Docs, and machine-readable
guides bundled in the wheel (`llms.txt`) make the library directly usable by AI-assisted
analysis workflows.

# AI Usage Disclosure

Generative AI tools were used in developing this software and manuscript. Anthropic's
Claude models (the Opus, Sonnet, and Fable model families, via the Claude Code CLI)
assisted with code generation and refactoring, test scaffolding, documentation, and
drafting and editing of this manuscript. The author reviewed, modified, and validated all
AI-generated code and text and made all primary architectural and methodological
decisions. Numerical results were independently verified against established R reference
packages (`did`, `synthdid`, `fixest`, `survey`) for every estimator with an R
equivalent, and against the author's reference derivations or simulation otherwise. The
author takes full responsibility for the accuracy and integrity of the software and this
paper.

# References
