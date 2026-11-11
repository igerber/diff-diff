# ChainedDiD review remediation

The estimator remains experimental and statistically unqualified. This revision
addresses sparse-pair catalogue handling, mixed reference rows and reproducible
reference checks. The historical qualification and reference JSON records remain
unchanged; their source hashes identify earlier implementations.

## Pair eligibility and inference limits

A comparison now needs two finite-outcome units in each arm. With a singleton,
the observed within-arm residual is zero regardless of its sampling variance.
Such a comparison must not enter the empirical covariance as if it had measured
that variance. Every excluded candidate retains its cohort, dates, arm counts
and reason in `diagnostics['excluded_links']`. Removing a pair can disconnect a
target, in which case the existing unsupported-target and fail-closed summary
rules apply. Units remain in registry population and cluster accounting.

The replication minimum is a project safeguard, not a theorem establishing
finite-sample validity. Retained comparisons below 20 units per arm or control
effective sample size 20 are flagged as `low_support`. Singleton exclusions or
low retained support trigger one warning per fit. The value 20 is an operational
warning threshold, not a qualification threshold. Identity GMM still ignores
pair precision, so adding a noisy comparison can reduce precision and produce
anti-conservative intervals even after excluding singletons. Precision-aware
weighting remains a separate methodological decision.

## Aggregation and reference checks

On irregular calendars, one cohort's reference can overlap another cohort's
estimated diagnostic horizon. Mixed rows now exclude structural reference zeros
before computing their weights, counts, estimates and covariance. Pure reference
rows remain zero identities. `reference_event_times` retains all cohort bases as
producer provenance, whereas `reference_periods` identifies reference-only rows.
A mixed diagnostic row cannot be selected as a plotting reference.

`aggregation_weights(level)` exposes constituent cohort/date weights, labels and
cell availability for `simple` and `event_study` summaries. Unsupported cells
retain their intended weights; their summary remains unavailable. These weights
are distinct from the simple container's one-row `weight=[1.0]`.

The default test suite compares current fits with the actual frozen public R
outputs. All nine input hashes and each comparison's individual checks are
pinned, including the known incomplete continuous-covariate mismatch. Eight
fixtures still agree; no comparison tolerance was relaxed. These coinciding
fixtures do not exercise singleton exclusions or mixed reference rows, which
have separate deterministic regression tests.

## Sparse-design pilot

`benchmarks/data/chained_did/qualification-sparse.json` records a new 300-replication
pilot per case, seed 20261105, with source hashes and environment versions.
There are 400 units, half never treated and half first treated at date 3.
Every unit has outcomes at dates 2 and 3. At date 1, 19 controls and k treated
units are observed; when k is positive, these extra-date units omit date 4.
Unit intercepts and independent date errors are standard normal, observation
patterns do not depend on outcomes, and ATT(3,3)=1. The grid is
k=0,1,2,5,20,50. All 1,800 fits completed without fit or target-inference failures.

| k | Empirical SD | Mean estimated variance | 95% coverage | Coverage MCSE |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.1292 | 0.01998 | 0.9667 | 0.0104 |
| 1 | 0.1406 | 0.01997 | 0.9433 | 0.0133 |
| 2 | 0.2883 | 0.05016 | 0.7867 | 0.0237 |
| 5 | 0.2169 | 0.04186 | 0.9100 | 0.0165 |
| 20 | 0.1632 | 0.02846 | 0.9400 | 0.0137 |
| 50 | 0.1735 | 0.02545 | 0.9233 | 0.0154 |

Singleton comparisons were excluded in every k=1 replication. All replications
at k>=2 triggered the retained-support diagnostic, including k=20 and k=50
because only 19 controls were observed at the extra date. The two-unit case
still has severe undercoverage, so the safeguard does not resolve the wider
finite-sample inference problem. This pilot's `core_thresholds_pass` is false;
it neither qualifies the revised estimator nor replaces the historical study.
Cases use independent SeedSequence-derived seeds, so differences between rows
are not paired comparisons. The JSON also reports bias, its Monte Carlo standard
error, RMSE, variance ratios, interval width and rejection rates.

To reproduce into a new file from the repository root:

```bash
PYTHONPATH=. DIFF_DIFF_BACKEND=python OPENBLAS_NUM_THREADS=1 \
  python -m benchmarks.python.validate_chained_did \
  --design sparse --replications 300 --workers 3 --seed 20261105 \
  --output /tmp/chained-did-sparse-pilot.json
```

The runner refuses existing output files and the five frozen historical evidence
paths. Exit zero means completion, not statistical acceptance. A larger
prespecified qualification study, hosted CI and a maintainer decision on the
inference policy remain open before merge.

The [oracle covariance diagnostic](chained-did-oracle-diagnostic.md) subsequently
compares known covariance and fixed design precision weighting on 20,000 fresh
native fits. It isolates two failure mechanisms without changing the agreed
production contract or replacing these historical pilot results.

## Prepared-branch review (2 November 2026)

A subsequent user-supplied read-only review examined the prepared contribution
on upstream base `c5280ae5`. It reported 111 passing chained tests with the
pure-Python backend, reproduced the mixed-reference correction, checked all nine
frozen R outputs, and confirmed exposed summary weights and removal of local
process files. It also reported approximately 93% coverage after singleton
exclusion, 78% with two sparse treated units and 91% with five. These are the
reviewer's results, consistent with the recorded pilot, not new runs by the
contribution preparer. The review did not repeat the full suite, wheel,
documentation builds, hosted CI or live R validation.

This review covers the post-initial-review changes to pair eligibility,
support warnings, mixed-reference aggregation and exposed weights. The separate
submission packaging review did not assess numerical methodology. The review
found the contribution suitable for upstream review after correcting the PR
text, but not ready to merge or statistically qualified.

Before merge, maintainers must decide whether to accept experimental identity
GMM with the documented undercoverage, or require precision-aware weighting or
another inference policy with further validation. In the larger recorded
diagnostic, coverage with two fixed sparse treated units is 0.8005 at n=400 and
0.7260 at n=4,000; increasing the main panel does not increase rare-pair support.
Maintainers must also decide whether to accept the native observed-pair logit
criterion despite the known continuous-covariate R difference, or require the
cohort-response-calibrated criterion and its corresponding derivation and
validation. The [validation follow-up](chained-did-validation-followup.md)
describes that criterion difference. Hosted CI remains outstanding, and neither
decision is settled by the packaging checks or experimental label.
