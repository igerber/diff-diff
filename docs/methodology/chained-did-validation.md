# ChainedDiD validation record

Date: 2 November 2026. Initial milestone: experimental; **not qualified for upstream merge**.

This is the historical first-checkpoint record. The [follow-up](chained-did-validation-followup.md)
contains the executed pinned R comparisons, fresh simulations and current
qualification status; the environment-availability statements below describe
this initial checkpoint. Its original failed outcomes are retained.
The implementation has independent algebraic and integration evidence, but the
pinned R comparison is unrun and three small-sample simulation targets miss the
prespecified limits. Passing engineering tests does not close those gates.

## Scope and reproducibility

The controlling contract is the [proposal](chained-did-proposal.md) and
[methodological pseudocode](chained-did-pseudocode.md). This implementation uses
PT-post, never-treated controls, pair-specific normalised IPW, all supported date
pairs, identity GMM, unit HC0 or one-way cluster CR1 covariance, and normal
pointwise intervals. It neither fits an observation model nor supplies optimal
GMM weights. All changes are uncommitted against upstream base
`ba746948b7fc7d06a5e8cdbf9601e21c3eaac6f0`.

The Monte Carlo runner is
`benchmarks/python/validate_chained_did.py`; the complete machine-readable record
is `benchmarks/data/chained_did/qualification.json`. Run from the repository root:

```bash
PYTHONPATH=. DIFF_DIFF_BACKEND=python OPENBLAS_NUM_THREADS=1 \
  python benchmarks/python/validate_chained_did.py \
  --replications 2000 --workers 4 --seed 20261102 \
  --output /tmp/chained-did-qualification.json
```

The JSON includes source fingerprints, versions, seed derivation, analytic
population targets, individual scenario summaries and failures. The command's
successful exit means the study completed; acceptance is the separate
`core_thresholds_pass` field, which is **false** in this record. Do not treat the
exit status alone as statistical acceptance.

The preliminary study is preserved as `qualification-pre-review.json`. The
final study reruns the same seeds and scenarios after clustering/parameter
boundary fixes and summary-contract fixes. Its fingerprints bind the final
estimator, result module and runner. The final runner separately records
empirical Monte Carlo standard errors for observed coverage/rejection rates and
the nominal binomial standard error used by the unchanged acceptance rule.

## Acceptance matrix

| Gate | Evidence and disposition |
| --- | --- |
| V01 | Two-date unconditional estimate and HC0 variance match hand calculations and `CallawaySantAnna` where the contracts coincide. |
| V02 | Rotating two-date windows recover the known longer effect without long-panel units. |
| V03 | Non-adjacent comparisons retain support when intermediate outcomes are absent. |
| V04 | An earlier bridge cancels a non-zero pre-reference differential; a separate test refuses to publish an indirectly supported nuisance coordinate as a direct pre-period diagnostic. |
| V05 | Disconnected cells are unavailable and invalidate summaries that require them; supported cells remain available. |
| V06 | An overidentified six-link graph matches an independent reference-anchored Laplacian calculation and covariance. Balanced systems also enter the simulation study. |
| V07 | Binary-stratum algebra and an independent continuous-covariate score fit with a finite-difference Jacobian match complete contributions. Shared-control cross-cohort covariance is retained. |
| V08 | Zero-/one-outcome units remain in registry masses; summary contributions include estimated-share uncertainty. Conflicting metadata are rejected. |
| V09 | Duplicate rows, missing/non-finite metadata, conflicting cohort dates, separation and rank failure are rejected. A cohort-date interface cannot observe realised treatment reversals: the caller must verify the supplied absorbing schedule. |
| V10 | Hand calculations and joint cluster multiplier replay reproduce HC0/CR1 scaling. A single cluster preserves points but withholds inference. |
| V11 | **Open.** The pinned R environment is unavailable; no `cdid` numerical agreement is claimed. |
| V12 | **Not passed.** Study complete with zero core fit failures; three targets miss the prespecified calibration limits below. |
| V13 | Negative designs demonstrate bias and coverage loss under selective observation, false parallel trends and propensity misspecification. This is sensitivity evidence, not a diagnostic guarantee. |
| V14 | Public exports, parameter/result/aggregation/plotting contracts, source-to-doc routing and affected tutorial have executed evidence. See engineering checks below for remaining environment limits. |

These are bounded checks, not proofs over every input. In particular, algebraic
agreement does not establish coverage, and a reference implementation would not
certify the observation or causal assumptions.

## Simulation results

There are 13 scenarios with 2,000 replications each: four valid observation
patterns (balanced, rotating, intermittent and cohort-dependent) at n=1,000 and
4,000; rotating null scenarios at both sizes; and three negative designs at
n=1,000. Intermittent core scenarios use 50 and 100 independent clusters. The
DGP includes static covariates affecting treatment and trends, serially
correlated outcomes, heterogeneous effects and unrestricted earlier trend
jumps. Targets are the population overall ATT, ATT(2,3) and ATT(3,4).

Acceptance requires zero failed required fits, absolute bias below 0.1 empirical
standard deviations, estimated-to-empirical variance ratios in [0.9,1.1], and
95% coverage / null rejection within three nominal binomial Monte Carlo
standard errors. For 2,000 replications the coverage acceptance interval is
approximately [0.93538,0.96462]. Thresholds remain as prespecified.

All n=4,000 core and null targets pass these limits. Core fit failures are zero
at both sizes, but the following n=1,000 targets fail:

| Observation / clusters | Target | Bias | Empirical SD | Variance ratio | Coverage | Missed limit |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Rotating / independent units | ATT(3,4) | -0.00201 | 0.26032 | 0.94758 | 0.9340 | Coverage |
| Intermittent / 50 clusters | Overall | -0.00187 | 0.12533 | 0.93308 | 0.9315 | Coverage |
| Intermittent / 50 clusters | ATT(2,3) | -0.00157 | 0.18631 | 0.89639 | 0.9300 | Variance ratio and coverage |

The failure is concentrated in variance/interval calibration rather than mean
recovery in these designs. The larger-sample results support the tested
asymptotic behaviour, but do not erase smaller-sample failures. The contribution
must remain experimental while this qualification gate is unresolved. Changing
finite-sample corrections or interval rules would be a methodological decision
requiring a revised contract and fresh evidence, not a tolerance adjustment.

Under the negative selective-observation design, 63 fits fail and unconditional
overall coverage is 0.263, with bias about 0.311. False parallel trends produce
overall bias about 0.752 and zero coverage. Misspecified logits produce overall
bias about 0.102 and coverage 0.862, although one cell still meets its numerical
limits. These results show why treatment IPW cannot justify arbitrary attrition
or substitute for parallel trends. Failure counts remain in coverage denominators.

## External reference gate

`benchmarks/R/benchmark_chained_did.R` refuses an unmatched environment and never
installs packages. It requires R 4.5.2, `cdid` 1.1.0.0 at commit
`1039120668cceabd8111cad82d32ab3927c2b94f`, `did` 2.5.1 and `DRDID` 1.3.0.
The available environment has R 4.6.1 and lacks those three packages; invoking
the harness correctly refuses that R version. This tests the guard, not the
reference estimator or output schema.

A future matched run must compare cell labels/support, estimates, full joint
covariance, direct earlier diagnostics and aggregates at the declared tolerances.
It must reconcile the reference's sqrt(n) matrix with estimator-scale covariance
and separately reconcile Python's declared K/(K-1) cluster correction. Registry
or catalogue differences must be resolved rather than normalised away.

## Engineering checks

The final focused run passes 1,112 tests, with 14 skips and 65 warnings, across
the new estimator/methodology/harness tests and existing parameter, aggregation,
documentation and changelog contracts. The three new test modules alone pass
48 tests. The configured non-slow suite is recorded separately below.

A fresh whole-change review found no critical issues and three input-boundary
defects: duplicate static/cluster metadata aliases, silent disabling of an
empty-named cluster column, and missing fit-time validation after direct
parameter mutation. Each was reproduced with failing regression tests and fixed.
The simple summary also now reports its canonical unit weight; changing summary
alpha uses the shared analytical refusal message. No review findings are deferred.

The affected notebook executes top to bottom with `nbclient`;
`nbmake` is absent. Sphinx builds with warnings treated as errors and notebook
execution disabled because the affected notebook is executed separately.

A scoped type check on the two new modules passes with installed mypy 1.18.2
and `--follow-imports=silent`. That tool differs from upstream's pinned 2.4.0;
a full imported check reports errors in imported modules, so this is not a
claim that the full upstream type gate passes. Available Black/Ruff also differ
from the pinned development versions. No Rust estimator kernel is added.

The wheel builds and installs into an isolated environment. A consumer outside
the checkout imports both public classes, fits covariate IPW with a shared
cluster/covariate column, aggregates into canonical results, round-trips pickle
and reads the bundled guide. It passes with both Python and Rust backend
settings. This verifies installation and existing backend interoperability;
it does not add or qualify a chained-specific native kernel.


### Broad-suite outcome and baseline comparison

The configured non-slow selection was exercised in two phases. The serial prefix
was interrupted inside `test_methodology_trop.py` after 8,436 passes, 221 skips
and one failure. Its unfinished module and every subsequent collected test file
were rerun using four pytest-xdist workers with work stealing: 6,963 passed,
212 skipped and ten failed. These phase counts overlap and are not a unique
combined test count. The isolated plugin installation changes no project dependency.

Eight contribution-related failing cases across the two phases were fixed and
retested: the snippet harness mistook an exported result class for a fitted
instance; the naming guard needed the new legal calendar-column surface; the
cluster-policy roster/fit recipe needed the new estimator; and four catalogue
guards still expected 23 estimators. The catalogue now explicitly includes the
23 cut estimators plus experimental `ChainedDiD`. These changes preserve the
strict guards instead of bypassing them.

Three remaining cases reproduce identically on a separate untouched checkout of
`ba746948b7fc7d06a5e8cdbf9601e21c3eaac6f0`:

| Case | Observed baseline failure |
| --- | --- |
| `test_naming_guard.py::test_allowlists_are_reachable` | Stale `DMLDiDResults.groups` allowlist entry. |
| `test_rdplot.py::TestRDPlotGoldenParity::test_all_configs` | One senate/esmvpr confidence bound differs by 4.16e-10, exceeding its 1e-9 relative tolerance. |
| `test_rust_backend.py::TestClusterVcovDeterminism::test_raw_kernel_noncontiguous_ids_bit_identical` | The source checkout has no compiled `_rust_backend` module; the raw-kernel test imports it directly even with the Python backend selected. |

Those cases remain unchanged. Therefore the broad run is not a green upstream
suite. The wheel's isolated Rust-enabled smoke check is separate evidence from
the source checkout's missing extension. Final affected-check results are
recorded below; the reproduced stale-allowlist case is explicitly deselected
there rather than reported as a pass.


Final affected run: **1,490 passed, 21 skipped**, 192 warnings, using the Python
backend and four workers. It covers the three new test modules, existing base
and aggregation contracts, source-to-doc routing, changelog, full snippet
harness, naming guard (excluding the separately reproduced stale allowlist
case), cluster policy, diagnostic/estimator catalogues and result serialisation.
All contribution-related failures found by the broad run are resolved.

Final checks also confirm that the installed wheel contains exactly the final
two implementation modules, that every study source fingerprint matches, and
that `git diff --check` is clean. The graphify AST graph was refreshed locally.
No commits, pushes, PRs or standards changes were made. The implementation lives
in the isolated `feature/chained-did` worktree; original checkout edits are preserved.
