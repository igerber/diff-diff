# ChainedDiD validation follow-up

Date: 2 November 2026. This follows the original qualification record without
replacing it. This document describes historical qualification snapshots, whose
source fingerprints remain unchanged. The current pair eligibility and mixed
reference-row policies are documented in the
[review remediation](chained-did-review-remediation.md); historical experiments
do not establish calibration for that revision.

## Prespecified experiments

The original 13 scenarios are rerun with 10,000 replications and fresh seed
20261103. A separate paired diagnostic study uses 2,000 replications, seed
20261104, rotating panels at n=1,000/4,000, and intermittent panels crossing
those sample sizes with independent outcomes or K=50/100 clusters. Both use a
64-bit SeedSequence-derived replication seed, rather than truncating to the
original runner's 32-bit seed. Cases, targets, baseline hashes, environment,
source hashes, failures and observed-rate Monte Carlo uncertainty are recorded.

Oracle variants know treatment odds and/or population cohort masses. Known
propensity contributions exclude beta estimation uncertainty. Known masses
remove share uncertainty and also change the finite-sample point aggregate.
Each variant retains the same superpopulation target and is assessed against
its own empirical variance. These are benchmark-only diagnostics, not public
estimator options. The original files and their acceptance outcomes are frozen.

The inherited DGP adds a cluster shock to the unit AR errors when K is positive.
Consequently K=50 versus K=100 holds marginal error variance fixed, whereas K=0
versus positive K changes both dependence and variance. The latter comparison
does not isolate dependence alone; within-dataset oracle comparisons remain paired.

## External reference: primitive-score alignment

The exact reference environment is R 4.5.2, cdid 1.1.0.0 at the pinned commit,
did 2.5.1 and DRDID 1.3.0. The image base is digest-pinned; installed package
versions and image identity are recorded with the executed reference evidence.
The R implementation is used only through public inputs/outputs and documentation.

The native contract fits an unweighted logit in each observed pair risk set:

```text
sum_observed Z_i * (D_i - expit(Z_i' beta)) = 0.
```

The reference documentation describes observation-route weights, including
cohort response rates. The independently tested alignment criterion is a
cohort-calibrated score with r_k = observed-pair units / eligible cohort units (at least two observed outcomes):

```text
a_i = D_i/r_g + (1-D_i)/r_0
sum_observed a_i * Z_i * (D_i - expit(Z_i' beta)) = 0.
```

Both use the observed treated mean minus the separately normalised control
odds-weighted mean. Multiplying all control odds by a constant cancels from that
mean. Nevertheless, response-rate calibration can alter fitted logit slopes
with continuous covariates. Under the maintained within-cohort pair sampling
restriction, both logit criteria are correctly specified with different
intercepts; they need not give identical finite-sample points or influence
functions. A saturated binary covariate and equal cohort response rates provide
coinciding cases. The validation-only calibrated-score calculation tests this
hypothesis independently; it supplies point comparisons only, not a new
covariance estimator or a port of R source.

Structural reference rows are also reported differently: the canonical Python
event-study container sets inference to NaN at reference rows; R records zero
SE. Comparison therefore verifies reference roles and exact zero R reference
SE separately, then compares all estimated-row inference and the complete
aggregate covariance. The predeclared CR1 correction is applied once to the R
raw cluster covariance, with its square root applied once to standard errors.
No tolerance or covariance factor is inferred from agreement.

## Executed results and disposition

The fresh qualification record is
`benchmarks/data/chained_did/qualification-followup.json`: 130,000 native fits,
zero core fit failures, all core bias limits satisfied, and all core variance
ratios in [0.95439, 1.03600]. At 10,000 replications the prespecified coverage
acceptance interval is [0.94346, 0.95654]. Four n=1,000 targets miss its lower bound:

| Observation | Target | Coverage | Variance ratio |
| --- | --- | ---: | ---: |
| Balanced | Overall | 0.9429 | 0.95977 |
| Intermittent, 50 clusters | Overall | 0.9425 | 0.98222 |
| Intermittent, 50 clusters | ATT(2,3) | 0.9390 | 0.97469 |
| Cohort-dependent | ATT(2,3) | 0.9410 | 0.96016 |

All larger-sample and null scenarios pass. The original rotating ATT(3,4)
coverage failure and the 0.896 variance ratio are not reproduced in this fresh
study; the original record remains unchanged. The follow-up still has
`core_thresholds_pass = false`, so it does not supersede the failed gate with
an acceptance claim. Independent seeds and greater precision refine the
failure pattern rather than erasing it.

The paired study in `calibration-diagnostics.json` evaluates 16,000 datasets and
four variants per dataset with no fit failures. Knowing cohort masses changes
little in these designs; knowing propensities does not consistently remove
coverage loss. For example, at n=4,000 and K=50, overall absolute standardised
error 95th quantiles are 2.0625 with fitted propensities and 1.9869 with known
propensities, compared with the normal cutoff 1.9600. Fixed population masses
produce 2.0597 and 1.9841 respectively. Other targets still have heavy tails
with oracle propensities. The evidence does not establish an omitted Jacobian
or share term, or justify one universal variance multiplier. Normal/CR1
finite-sample calibration remains a limitation requiring a separate inference
policy decision; production inference is unchanged.

### External agreement

`reference-validation.json` records all nine fixtures, their input hashes,
full public reference outputs, source/environment fingerprints and every
comparison result. Eight fixtures match declared tolerances on registry,
support, all cell estimates, full covariance and simple/event-study aggregates:
two dates, rotating windows, earlier bridges, non-adjacent observations,
balanced binary and continuous covariates, intermittent shared clusters and
zero-/one-outcome registry units.

The incomplete continuous-covariate fixture remains a failing comparison.
Its native ATT(2,2) differs from R by about -0.001018 and ATT(2,3) by -0.001553;
the maximum covariance difference is 0.00087152. The independent calibrated-score
calculation reproduces every reported R point in this fixture within the
original tolerance. The response-rate denominator is the eligible cohort
(at least two observed outcomes), not the entire registry; inspecting the
public custom-callback inputs confirms the supplied weights. The calibrated
point diagnostic does not reproduce or qualify its covariance.

The executable `benchmarks/R/inspect_chained_did_weights.R` probe verifies the
first pair's public callback inputs and stops before estimation. Its captured
output is stored under `public_input_probe` in the reference report, bound to
the input and probe source hashes: eligible counts [365, 418], observed pair
counts [301, 257], and weights [365/301, 418/257]. Maximum weight discrepancy
is 2.22e-16. These are public function inputs, not inspected R estimator source.

Thus V11 has bounded agreement on eight coinciding fixtures and a documented
primitive-score difference on a ninth. It is not blanket parity with `cdid`.
The agreed native risk-set MLE is retained; changing it to the calibrated
criterion would require an explicit methodological revision and its complete
influence-function derivation. No failed comparison was excluded or weakened.

### Reproduction

From the repository root, with single-threaded BLAS:

```bash
DOCKER_BUILDKIT=0 docker build -t diff-diff-cdid-reference:1.1.0.0 \
  benchmarks/environments/chained_did
PYTHONPATH=. DIFF_DIFF_BACKEND=python OPENBLAS_NUM_THREADS=1 \
  python benchmarks/python/compare_chained_did.py \
  --workspace /tmp/chained-did-reference
PYTHONPATH=. DIFF_DIFF_BACKEND=python OPENBLAS_NUM_THREADS=1 \
  python benchmarks/python/diagnose_chained_did.py --mode diagnostic \
  --replications 2000 --seed 20261104 --workers 4 \
  --output /tmp/chained-did-diagnostics.json
PYTHONPATH=. DIFF_DIFF_BACKEND=python OPENBLAS_NUM_THREADS=1 \
  python benchmarks/python/diagnose_chained_did.py --mode qualification \
  --replications 10000 --seed 20261103 --workers 4 \
  --output /tmp/chained-did-followup.json
```

The reference command exits 1 because the ninth comparison fails; this is an
identified criterion difference, not an environment installation failure.
Simulation exit 0 means completion, not acceptance: inspect the manifest flag.
The original record paths are protected against overwrite by the follow-up CLI.
The base image and core R versions are pinned. `packages.json` records the full
used environment; the Dockerfile fetches transitive CRAN dependencies at build
time, so rebuilding later is not a hermetic transitive-package restore.
The comparator checks the selected image's complete package/version/build
inventory and R/platform against that manifest before fitting and records the
verified inventory. A changed rebuild fails before generating reference evidence.

## Engineering verification

The final affected suite passes 1,302 tests with 14 skips, including estimator,
methodology, validation, base/aggregation, catalogue/naming, documentation and
the changed documentation-consumer regression. Scoped naming-guard entries
identify canonical calendar/aggregation uses and a quoted historical failure;
they do not revive removed API surfaces.

The final chained estimator, methodology and validation tests pass on isolated
Python 3.9, 3.14 and stable 3.15 environments: 60 tests per environment. Python
3.12 also supplies the simulation and installed-wheel evidence. The wheel
rebuild and isolated installed-consumer checks pass under Python and Rust
backend settings; the chained estimator itself remains native Python.

Pinned Black 26.10.1 and Ruff 0.17.1 pass across production/tests and the new
validation scripts. Sphinx passes with warnings treated as errors and notebook
execution disabled. Pinned mypy 2.4.0 in an isolated Python 3.14 environment
reports three errors in two unchanged upstream modules; the exact error lines
also occur on the clean target base. The host-inheriting Python 3.12 tool
environment reports 234 errors, likewise identical on that base. Neither
configuration adds errors for the new production modules.

An independent review found no Critical or Important validation defects. Its
minor package-provenance finding was closed with a failing regression followed
by inventory verification and a full reference rerun. The public callback
evidence is now executable and retained. The slow full repository suite was
not rerun in this sprint; the original validation record distinguishes its
previously reproduced baseline failures. Hosted and non-Linux checks remain unrun.
The attempted broader RST-example replay was stopped at an unrelated expensive
estimator example; the affected documentation consumer passes separately.

## Upstream disposition

The native estimator remains experimental. Engineering evidence, matched
reference evidence and statistical qualification are separate gates. The next
methodological decision is whether maintainers want the declared observed-risk
propensity criterion or alignment with the calibrated R criterion, and which
finite-sample inference policy merits a separately prespecified study.
Public OR/DR, attrition corrections and optimal GMM remain outside this work.
The accompanying upstream review draft is prepared locally and has not been posted.
