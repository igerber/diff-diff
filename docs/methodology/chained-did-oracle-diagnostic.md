# ChainedDiD sparse-pair covariance diagnostic

This benchmark separates covariance estimation from the precision of the point
estimator. It does not change the production estimator, its identity GMM metric,
its pair eligibility or its propensity-score criterion. Statistical qualification
is an independent gate. The implementation decision below follows the prespecified experiment.
This is a documented implementation choice within the restricted contract.

## Model and exact covariance

Use the existing sparse Gaussian panel: Y_it = a_i + t + D_it + epsilon_it,
where a_i and epsilon_it are independent standard normal, and date innovations
are independent across units and dates. Treatment begins at date 3 for half the
units and ATT(3,3)=1. Observation patterns are fixed independently of outcomes.
For each retained pair p=(s,t), define c_ip as 1/n_Tp for observed treated units,
-1/n_Cp for observed controls, and zero otherwise. Define the error loading
B_(i,u),p = c_ip * (1[u=t] - 1[u=s]). Then

```text
d_p = sum_i c_ip (Y_it - Y_is)
Omega = B' B
```

The unit intercept cancels because each pair's date coefficients sum to zero.
Omega is the exact conditional covariance under this DGP, on the estimator
scale; it includes shared units, shared controls and shared endpoints. It is
not the empirical residual covariance and is not assumed diagonal.

Anchor the cohort profile at date 2 and form the incidence matrix A. For a fixed
positive diagonal metric H, use the row-space/SVD recovery map

```text
R_H = pinv(sqrt(H) A) sqrt(H)
l_H = contrast(date 3)' R_H
estimate_H = l_H d
variance_oracle_H = l_H Omega l_H'
variance_HC0_H = (Q l_H')' (Q l_H')
```

Q is the original-unit matrix of pair mean residual contributions, with each
arm's own mean and denominator. The native-identity estimate and HC0 variance
must agree with a real ChainedDiD fit, checked during every replication. A
missing earlier coordinate must not alter the supported date 3 contrast.

Compare H=I with H_pp=n_Tp*n_Cp/(n_Tp+n_Cp). The latter uses the fixed design
counts, not observed outcomes or a fitted covariance. In this homoskedastic
Gaussian design it is proportional to inverse diagonal pair variance. Counts
and observation patterns are deterministic within a case, so no random-weight
term is omitted in this benchmark. This argument does not cover empirical
random missingness, treatment IPW, clusters or heterogeneous outcome variances.
The prototype is not a new public estimator or an optimal-GMM implementation.

## Prespecified experiment

Use seed 20261106, SeedSequence([seed,case_index,replication]) and 2,000 replications
per case. Each generated panel is shared by all four variants: identity/HC0,
identity/oracle covariance, design precision/HC0 and design precision/oracle
covariance. Do not compare independently generated samples across variants.

The ten cases are n=400 with k=0,1,2,5,20,50 extra-date treated units and 19
extra-date controls; n=4,000 with fixed k=2,5 and 19 controls; and n=4,000 with
proportionally growing k=20,50 and 190 controls. All units have dates 2 and 3;
extra-date units have date 1 and, when k>0, omit date 4. All six existing evidence
JSON files remain unchanged. The new report has a distinct path.

Use the existing summarize function: count failed replications in unconditional
coverage; report bias/MCSE, RMSE, empirical SD, mean variance, variance ratio,
coverage/MCSE, interval width and zero-null rejection. The declared acceptance
rule remains zero failures, absolute bias below .1 empirical SD, variance ratio
in [.9,1.1], and 95% coverage within three nominal binomial Monte Carlo standard
errors. The oracle covariance is analytically correct here; empirical thresholds
still report finite Monte Carlo fluctuations and do not qualify an empirical
covariance estimator. Record native replay discrepancies and all source hashes.

## Decision rule

If known covariance repairs coverage while HC0 fails on the same estimates,
the experiment supports a covariance-estimation failure in this design. If
design weighting reduces theoretical and empirical variance, that establishes
bounded precision evidence, not calibrated feasible inference. Neither result
alone permits a production change across the covariate/cluster contract or a
claim of general statistical qualification. Prefer the agreed, transparent
restricted estimator over an unqualified correction or an oracle-only solution.

## Executed results

The report `benchmarks/data/chained_did/sparse-covariance-diagnostic.json` records
20,000 native fits and 80,000 paired variant evaluations, with no fit or target
failures. Every native replay agrees with the independent identity calculation:
maximum point discrepancy is 4.22e-15. The report records variance discrepancies,
all comparison metrics, seeds, source hashes and environment versions.

| n | Extra-date treated / controls | Identity HC0 coverage | Identity known-covariance coverage | Design precision HC0 coverage | Design / identity exact variance |
| --- | --- | ---: | ---: | ---: | ---: |
| 400 | 0 / 19 | .9495 | .9495 | .9495 | 1.0000 |
| 400 | 1 / 19 | .9470 | .9470 | .9465 | .9996 |
| 400 | 2 / 19 | .8005 | .9560 | .9520 | .2351 |
| 400 | 5 / 19 | .9080 | .9490 | .9395 | .4184 |
| 400 | 20 / 19 | .9530 | .9560 | .9525 | .6846 |
| 400 | 50 / 19 | .9400 | .9450 | .9395 | .7848 |
| 4000 | 2 / 19, fixed | .7260 | .9535 | .9585 | .0283 |
| 4000 | 5 / 19, fixed | .8880 | .9500 | .9570 | .0602 |
| 4000 | 20 / 190, growing | .9345 | .9460 | .9500 | .2351 |
| 4000 | 50 / 190, growing | .9560 | .9620 | .9605 | .4184 |

At 2,000 replications, the unchanged coverage acceptance interval is
[.93538,.96462]. All oracle and design-precision variants pass the declared
bias, variance and coverage criteria on this ten-case grid. Native identity HC0
fails five cases. Do not widen the threshold to accept the growing k=20 case.
Observed coverage Monte Carlo standard errors and other metrics remain in the
JSON; the nominal 95% binomial standard error is .00487. Since ATT=1, the
reported rejection of a zero-effect null measures power, not test size. The paired variants share
each panel; cases use separate seeds and are not paired across sample sizes.

Known covariance repairs coverage on the identical native estimates, which
isolates empirical covariance estimation in this conditional Gaussian model.
Identity weighting also loses precision: with n=4,000 and 2 sparse treated units,
the theoretical design-weighted variance is 2.83% of identity variance. Increasing
the main panel alone does not fix a rare comparison given substantial weight.
Growing support behaves differently and must not be conflated with fixed counts.

## Implementation decision

Retain the agreed production identity GMM, full propensity-score/mean Jacobian,
HC0/CR1 covariance and documented sparse-pair safeguards. Keep the design-weighted
prototype in the benchmark only. It improves this specific homoskedastic,
no-covariate, independently sampled design; the experiment does not qualify it
for conditional IPW, heterogeneous variances, random observation counts or
clustered inference. Oracle covariance is not available in applications.

There is also an algebraic limit to the proposed weighting remedy. In a balanced
panel each cohort has the same arm counts at every pair, so its count-based
metric is a scalar multiple of identity. That scalar cancels from the recovery
map, leaving estimates and analytical covariance unchanged. Therefore this
weighting change cannot repair the historical balanced-panel coverage failure.
A deterministic regression verifies this invariance. Passing the sparse grid
is insufficient grounds to replace the production estimator or declare a
universal finite-sample correction.

No production source changed and no failed historical evidence was replaced.
A fresh large qualification rerun of the identical production policy would not
supersede its existing high-precision failures. The upstream contribution is
prepared as an experimental restricted estimator with reproducible limits;
statistical qualification remains false.

## Reproduction

```bash
PYTHONPATH=. DIFF_DIFF_BACKEND=python OPENBLAS_NUM_THREADS=1 \
  python -m benchmarks.python.diagnose_chained_did_sparse \
  --replications 2000 --workers 4 --seed 20261106 \
  --output /tmp/chained-did-oracle-diagnostic.json
```

Choose a new output path; the runner refuses existing and frozen evidence files.
Exit zero indicates completion, not statistical qualification. The derivation
uses the linear GMM structure in the paper's
[appendix A.2.6](https://arxiv.org/html/2301.01085v4); known innovation
variance and the fixed-count prototype are benchmark assumptions, not additional
claims about that paper's empirical inference procedure.
