# Proposal: native Python chained difference-in-differences

Date: 2 November 2026. Status: experimental implementation; qualification incomplete.
See the [validation record](chained-did-validation.md) and
[oracle diagnostic decision](chained-did-oracle-diagnostic.md) for executed
evidence and limits. The restricted contract below defines this contribution.

## Problem and contribution

Add `ChainedDiD` to estimate group-time average treatment effects, ATT(g,t),
from overlapping incomplete panels. Units observed at different pairs of dates
can jointly identify an effect even when no unit spans its reference and target
dates. The contribution is a paper-derived Python implementation, rather than
an R wrapper or a translation of `cdid` source.

Current upstream `CallawaySantAnna` uses direct within-cell differences or an
optional repeated-cross-section route for unbalanced panels. Neither constructs
the chained comparison system. This proposal targets upstream commit
`c5280ae578f1a60fd4f2a38c0ff8f4f7861bac9a` (version 4.0.0). The historical
validation records retain their original source fingerprints; they do not
qualify the revised catalogue policy below.

## First-release contract

| Component | Included |
| --- | --- |
| Design | Binary absorbing treatment, staggered adoption, never-treated controls, zero anticipation |
| Target population | Units represented in the supplied data, including units with fewer than two finite outcomes; explicit missing-outcome rows can retain otherwise unobserved units |
| Covariates | Complete, time-invariant unit attributes; no covariates is the unconditional special case |
| Identification | Conditional PT-post, treatment overlap and the pair-sampling restriction specified in the appendix |
| Treatment adjustment | Normalised inverse probability weighting (IPW), with pair-specific cohort-versus-control logits |
| Comparison system | Every ordered date pair with at least two observed units per arm, including non-adjacent pairs and pre-reference comparisons |
| Combination | Fixed identity-weighted generalised method of moments (GMM); no estimated optimal weights or covariance regularisation |
| Results | Group-time effects, direct universal-reference pre-treatment differentials, overall and event-study summaries, indexed joint covariance, constituent aggregation weights and support diagnostics |
| Inference | Analytical unit-level or one-way clustered covariance; normal pointwise intervals; joint multiplier replay as a validation check |
| Interface | Standalone estimator and result type following the package's fit, parameter, summary and aggregation conventions |

For the covariate-adjusted version, a sufficient sampling restriction is that
observing both endpoints is independent of the joint distribution of covariates
and actual outcome changes, within each cohort, for every used pair. This is
stronger than merely requiring representative unconditional mean changes. It
allows some selection on outcome levels, but does not correct arbitrary attrition.
These assumptions require scientific justification; diagnostics cannot certify them.

PT-post leaves earlier cohort-versus-control trend differences unrestricted.
They must cancel when earlier observations help recover a post-treatment target;
the estimator must not set them to zero. The
[methodological pseudocode](chained-did-pseudocode.md) defines that construction,
the population accounting, influence-function scale and aggregation weights.

Defer PT-all, not-yet-treated controls, non-zero anticipation, outcome regression,
doubly robust methods, conditional observation models, imputation, sampling
weights, complex survey designs, simultaneous bands,
Rust acceleration and custom nuisance learners. These are distinct extensions;
none is required to deliver this first statistical contract.

Pairs below two observed units in either arm are excluded with counts and a
reason. This replication minimum prevents zero singleton residual variance from
entering GMM. Retained pairs below 20 raw units in either arm or control effective
sample size 20 trigger a warning; 20 is an operational threshold, not evidence of
calibration. Identity GMM ignores pair precision. Sparse-pair intervals can
remain materially anti-conservative after singleton exclusion, so statistical
qualification remains open. Changing the GMM metric is outside this revision.

## Validation matrix

The entries below define the acceptance requirements; their executed status is
in the [validation record](chained-did-validation.md). Numerical agreement and
statistical qualification are separate release gates.

| ID | Design or failure mode | Required evidence |
| --- | --- | --- |
| V01 | Two dates, no covariates | Hand-calculated DiD and covariance; agreement with the existing estimator where contracts coincide |
| V02 | Overlapping two-date windows, no long-panel units | Recover the known long effect by chaining; preserve each link's original unit identifiers |
| V03 | Non-adjacent observed dates | Use the long comparison; do not discard it because an intermediate outcome is absent |
| V04 | Earlier bridge under PT-post | Recover a post-treatment effect by cancelling a non-zero unrestricted pre-reference differential; keep direct pre-period reporting separate from GMM nuisance parameters |
| V05 | Unsupported target | Removing a required bridge makes that target unavailable, while independently supported targets remain available |
| V06 | Overidentified and balanced systems | Compare with independent matrix calculations; preserve all comparison rows even when their covariance is singular |
| V07 | Covariates and shared controls | Check the propensity-score Jacobian, normalisation derivatives and cross-link/cross-cohort covariance |
| V08 | Population accounting | Retain zero-/one-outcome units in cohort masses, aggregation influence functions and cluster counts; reject conflicting metadata |
| V09 | Invalid inputs or failed nuisance fits | Reject duplicate unit-date rows, inconsistent supplied cohort dates, missing covariates, rank failure, separation and non-convergence |
| V10 | Unit and cluster inference | Reproduce the declared covariance scaling; match joint multiplier covariance; mark inference unavailable with fewer than two clusters |
| V11 | R reference | Compare supported cells, estimates, full covariance and aggregates with pinned `cdid`, using IPW, PT-post, never-treated controls, no observation correction and identity GMM |
| V12 | Valid simulation designs | Measure bias, empirical standard deviation, standard-error calibration, pointwise coverage and null rejection rates, with Monte Carlo uncertainty |
| V13 | Violated assumptions | Demonstrate the limits under selective trends/effects, misspecified logits and false PT; do not require diagnostics to detect untestable violations |
| V15 | Sparse comparison pairs | Record exclusions below two units per arm; warn below 20 raw units or control ESS 20; evaluate ATT(3,3)=1 at n=400 with 0/1/2/5/20/50 treated units and 19 controls at an extra date, including bias, variance and coverage with Monte Carlo uncertainty |
| V16 | Irregular multi-cohort calendar | Exclude structural reference zeros from mixed diagnostic rows; verify contributing cohorts, weights, counts and covariance while retaining cohort-base provenance |
| V14 | Package integration | Parameter round trips, stable result schemas, plotting/aggregation compatibility, source-to-doc mappings, tutorial and installed-consumer checks |

The appendix specifies reference normalisation, tolerances and simulation gates.
No accuracy, efficiency, interval-coverage or runtime claim follows from this
proposal or from passing a reference fixture alone.

## Delivery and integration

1. Review this methodological contract and its pseudocode before product code.
2. Implement the comparison catalogue, unweighted and IPW link estimates, complete
   unit influence functions and identity-GMM recovery, with V01-V09 fixtures.
3. Complete analytical inference, aggregates, multiplier checks and the external
   reference harness; run V10-V13 before claiming statistical qualification.
4. Complete V14 and submit a bounded upstream contribution.

Candidate new modules are `diff_diff/chained_did.py` and
`diff_diff/chained_did_results.py`. Reuse `linalg.py`, `safe_inference`, the current
aggregation kit and multiplier stream where their contracts fit; do not subclass
`CallawaySantAnna.fit()` or change its estimator behaviour. Production delivery
also reaches exports, tests, R-reference fixtures, API docs, a tutorial,
`docs/references.rst`, methodology registry, bundled guides, estimator catalogues,
`docs/doc-deps.yaml` and a changelog fragment, following upstream contribution
requirements. The experimental implementation includes those production surfaces.

`cdid` is an external validation reference at commit
`1039120668cceabd8111cad82d32ab3927c2b94f`, version 1.1.0.0, with its documented
`did` 2.5.1 / `DRDID` 1.3.0 environment pinned by the external-reference harness. Its planned
Python implementation is not a dependency or a scheduling constraint. The Python
derivation and implementation remain independent of its GPL-3 source.

## Sources

- Bellégo, Benatia and Dortet-Bernadet, *The chained difference-in-differences*,
  [published paper](https://doi.org/10.1016/j.jeconom.2024.105783);
  [accessible mathematical reference, section 2.2 and appendix A.2](https://arxiv.org/html/2301.01085v4).
- [Upstream unbalanced-panel contract](https://github.com/igerber/diff-diff/blob/ba746948b7fc7d06a5e8cdbf9601e21c3eaac6f0/diff_diff/staggered.py#L449)
  and [contribution requirements](https://github.com/igerber/diff-diff/blob/ba746948b7fc7d06a5e8cdbf9601e21c3eaac6f0/CONTRIBUTING.md).
- Pinned `cdid` documentation:
  [GMM](https://github.com/dvbn/cdid/blob/1039120668cceabd8111cad82d32ab3927c2b94f/docs/methods/gmm.md),
  [parallel trends](https://github.com/dvbn/cdid/blob/1039120668cceabd8111cad82d32ab3927c2b94f/docs/methods/parallel-trends.md),
  [treatment adjustment](https://github.com/dvbn/cdid/blob/1039120668cceabd8111cad82d32ab3927c2b94f/docs/methods/treatment.md),
  [missing outcomes](https://github.com/dvbn/cdid/blob/1039120668cceabd8111cad82d32ab3927c2b94f/docs/methods/missing-outcomes.md).
