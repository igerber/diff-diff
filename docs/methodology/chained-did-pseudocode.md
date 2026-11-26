# Chained DiD: methodological pseudocode

Date: 2 November 2026. Status: proposed derivation for the
[upstream proposal](chained-did-proposal.md); not executable code or validation
evidence. Notation and equations below are an independent specification based on
the paper, not a transcription of the R implementation. Algorithmic choices such
as identity weighting are proposed implementation decisions.

## 1. Existing Python methodology and the new seam

The inspected upstream commit is
`ba746948b7fc7d06a5e8cdbf9601e21c3eaac6f0`. Its staggered-estimator flow is:

```text
validate input and select panel or repeated-cross-section route
construct the cohort, calendar, outcome and covariate structures
for each requested cohort-time cell:
    select reference and target dates, and eligible controls
    panel route: retain units observed at both selected dates; difference outcomes
    repeated-cross-section route: compare the selected outcome-level samples
    estimate that cell with its selected treatment adjustment
    retain the estimate and its original-unit influence contributions
aggregate cells, including uncertainty in estimated aggregation weights
compute analytical inference and validate joint covariance using multipliers
return results with the retained aggregation inputs
```

The chained estimator replaces reference-to-target cell construction with a
catalogue of date-pair comparisons and a linear recovery step. Unit alignment,
aggregation and uncertainty propagation remain explicit. Reuse numerical
primitives rather than treating the existing private cell estimators as a new
public API. In particular, their fallback, clipping and influence-function
conventions must not enter the new estimator without being reconciled below.

## 2. Inputs, population and assumptions

Let i index n distinct supplied study units; t indexes the strictly ordered stored
calendar. Inputs follow existing column-name conventions: outcome, unit, time,
first treatment date, optional static covariates and an optional cluster column.
For this first version, calendar labels are finite numeric values, treatment dates
must occur in that calendar and 0 denotes never-treated units. A treated cohort
must have a preceding stored date; first-period treatment is unsupported.

Build the calendar from all dated rows, including missing outcomes. A missing
row or a NaN outcome supplies no observed endpoint; infinite outcomes are input
errors. Reject duplicate unit-date rows. Unit, date, cohort and active static
metadata must be complete and consistent. Covariates and cluster assignments must
be constant within units; treatment timing is supplied, never inferred from gaps.
This interface accepts cohort dates, not a separate realised treatment-status
column. It rejects conflicting dates within a unit, but callers must verify
that the realised treatment follows the supplied absorbing schedule. Extra
columns are not interpreted as treatment indicators. Preserve caller-owned data.

The study-unit registry contains every distinct supplied unit, including units
with one or no observed outcomes. A unit absent from the data is outside this
declared population; retaining one dated row with a missing outcome and complete
metadata includes it. Counts and cohort shares use units, not observed rows or
repeat observers. These units represent the cohort population law; the method
does not provide finite-population or survey-design inference.

For cohort g define D_ig = 1{G_i=g}, C_i = 1{G_i=0}, and b_g as the stored date
immediately preceding g. Let R_i,st indicate finite outcomes at both s and t.
Let X_i include a constant and the declared static covariates. The target is:

```text
ATT(g,t) = E[Y_t(1) - Y_t(0) | G=g],  t >= g.
```

Maintain no anticipation, absorbing treatment and conditional parallel untreated
trends from b_g onwards. For every catalogue pair and contributing cohort k,
use the paper's sufficient baseline sampling restriction:

```text
R_st is independent of (X, Y_t - Y_s), conditional on G=k.
```

Here changes are actual outcomes, including treatment effects. Cohort-specific
pair response rates may differ. For no covariates, equality of the relevant mean
changes suffices. With covariates, mean representativeness alone does not preserve
the covariate law needed by IPW. Conditional-on-X missingness without preservation
of that law requires an observation correction, which is outside this release.
Selection on a time-constant outcome-level component is admissible only when it
preserves the required joint restriction. Positive pair response rates, adequate
treatment overlap and correct pair-specific logistic propensity models are also
required. Inference needs independent units or independent clusters, suitable
moments and an increasing number of clusters; few-cluster validity is not claimed.

## 3. Catalogue and the PT-post parameterisation

For each g consider all s<t in the stored calendar, using the same never-treated
control population throughout. Pool all units observed at both endpoints for that
pair; do not create duplicate rows for overlapping observation-pattern categories.
Retain one comparison per (g,s,t) when at least two target units and two control
units have finite outcomes at both endpoints. Record every excluded candidate
with its arm counts and reason: `no_treated_observations`,
`no_control_observations` or `insufficient_pair_replication`. A singleton arm
has zero empirical residual variance, so it cannot enter this replicated
catalogue. This minimum is a project safeguard, not a calibration threshold;
excluding pairs can disconnect a target and change finite-sample GMM estimates.
Keep excluded units in the population registry.

Warn once per fit if singleton pairs were excluded or retained pairs have fewer
than 20 units in either arm or control effective sample size below 20. The value
20 is an operational warning threshold, not a statistical acceptance rule.
Retain these low-count pairs, flag them in link diagnostics and disclose that
identity GMM ignores pair precision. Two or five units can still yield materially
anti-conservative intervals; no sparse-pair calibration is claimed. A supported pair with a
failed propensity fit is an error, not a reason to silently remove that moment.

Define a common population profile anchored at b_g:

```text
theta_g(t) = E[Y_t - Y_b | G=g]
             - E[ E[Y_t - Y_b | X, C=1] | G=g ];  b = b_g
theta_g(b) = 0.
```

For t>=g, the maintained PT-post and no-anticipation assumptions make this profile
ATT(g,t). Before b_g it is an unrestricted trend differential, not a treatment
effect or an imposed zero. The baseline sampling restriction makes every pair
comparison target the same covariate and cohort laws, so:

```text
d_g,st = theta_g(t) - theta_g(s).
```

Build A_g with one row per retained pair and a column for every stored date except
b_g: +1 at t, -1 at s, omitting the anchored column. Stack cohorts block-diagonally
into A. Joint covariance is not block-diagonal because cohorts share controls.

Example: b_g=2, with comparisons 1->2 and 1->4 only. Then
ATT(g,4)=d_g,14-d_g,12. The earlier differential cancels even if it is non-zero.
If 1->2 is absent, 1->4 alone cannot separate the effect from that differential.
This is why earlier parameters remain free under PT-post.

## 4. Normalised IPW link and its complete influence function

For each link e=(g,s,t), restrict the nuisance-fit risk set to R_i,st=1 and
D_ig+C_i=1. Fit the unpenalised logistic model:

```text
p_e(X) = Pr(D_g=1 | X, D_g+C=1, R_st=1) = expit(X' beta_e)
o_e(X) = p_e(X) / (1-p_e(X))
mu_T,e = sum(R_st D_g deltaY_st) / sum(R_st D_g)
mu_C,e = sum(R_st C o_e deltaY_st) / sum(R_st C o_e)
d_hat_e = mu_T,e - mu_C,e
```

This pair-specific propensity accommodates different cohort response rates. An
intercept-only fit gives the ordinary difference in mean observed changes. Reuse
`solve_logit` only with explicit rank/error handling and a checked score residual;
a warning about non-convergence or separation is insufficient for accepting a fit.
Do not fall back to an intercept-only model, clip odds, drop units or trim scores.
Non-finite odds, numerical boundary probabilities, separation, non-convergence
or unresolved singularity must stop the fit. Record propensity range, group counts and control effective
sample size `(sum weights)^2 / sum(weights^2)`; poor overlap remains a limitation.

Derive uncertainty from the joint estimating equations, not fixed fitted weights.
Let r_i=R_i,st (D_ig+C_i), a_e=(beta_e,mu_T,e,mu_C,e), and define:

```text
h_beta,i = r_i X_i (D_ig - p_e(X_i))
h_T,i    = R_i,st D_ig (deltaY_i,st - mu_T,e)
h_C,i    = R_i,st C_i o_e(X_i) (deltaY_i,st - mu_C,e)
H_e      = (1/n) sum_i derivative(h_i, a_e)
IF_a,i,e = -solve(H_e, h_i)
phi_i,e  = IF_mu_T,i,e - IF_mu_C,i,e
```

All sums use the original n-unit registry, with zero score contributions outside
the risk set; compute changes only inside the observation mask to avoid 0*NaN.
The Jacobian includes the logistic-score and odds derivatives and both mean
normalisations. It must reproduce numerical derivatives on well-conditioned
fixtures. No-outcome units have zero link scores but remain in the registry.
Across links, align phi by original unit identifier, never by local row position.
The convention is `d_hat-d = (1/n) sum_i phi_i + o_p(n^-1/2)`; phi is O(1).

## 5. Identity-GMM recovery and target support

Stack d_hat and the n-by-M matrix Phi of link influence functions. Fix the
criterion to the full-catalogue identity metric:

```text
minimise_theta ||d_hat - A theta||^2.
```

Use a rank-revealing SVD, with cutoff `max(A.shape)*machine_epsilon*sigma_max`,
and form A_plus from retained singular directions. For a target contrast c:

```text
if c is not in rowspace(A):
    return unavailable estimate and inference, with a support reason
else:
    ell = c' A_plus
    tau_hat = ell d_hat
    psi_i = ell phi_i
```

Test row-space membership by the projection residual using the same retained
SVD basis and a dimension-scaled roundoff tolerance. An empty catalogue has no
supported non-zero contrast. The minimum-norm parameter solution does not make
unsupported parameters estimable; never publish those arbitrary coordinates.
Reference zeros are exact identities and do not enter hypothesis tests.

Reported pre-treatment cells have a separate, direct construction. For t<b_g,
report `-d_hat_(g,t,b_g)` with influence vector `-phi_(g,t,b_g)` only when that
direct pair is supported. Otherwise report the diagnostic as unavailable, even
if a GMM nuisance coordinate can be recovered indirectly. Keep GMM pre-reference
coordinates internal. Non-zero earlier differentials do not contradict PT-post,
which imposes no zero restriction there. Joint reported covariance includes these
direct diagnostic vectors alongside the recovered post-treatment vectors.

Keep every catalogue row. Do not replace the system with a spanning tree, change
the full comparison metric or invert the moment covariance. Covariance can be
singular on balanced panels even when targets are supported. Identity weighting
avoids covariance inversion; it does not imply optimal efficiency. Cross-target
influence functions retain shared-unit and shared-control dependence.

## 6. Aggregation and inference

Let L_g be the number of stored post-treatment dates for g, and N_g the number
of registry units in that cohort. Define the overall summary over all scheduled
post-treatment cells with weights `N_g / sum_h(N_h L_h)`. For event time e=t-g
in the numeric calendar's original units, weight eligible cohorts by
`N_g / sum_h N_h`. When a horizon mixes cohort reference identities with
estimated diagnostics, exclude the structural reference identities before
computing its weights, counts, estimate and covariance. Pure reference rows
remain zero identities. Preserve all cohort reference event times as producer
provenance, even when they overlap an estimated row. Missing calendar dates do
not become synthetic observations.

For either summary, write weights as `w_j = m_j / D`, with `m_j=E[D_g]` for
cell j and `D=sum_j m_j`. Then:

```text
agg_hat = sum_j w_hat_j tau_hat_j
IF_m,i,j = D_ig - m_hat_j
IF_D,i = sum_j IF_m,i,j
IF_w,i,j = (IF_m,i,j - w_hat_j IF_D,i) / D_hat
psi_agg,i = sum_j(w_hat_j psi_i,j + tau_hat_j IF_w,i,j)
```

Repeated cohort membership terms in the overall cell sum are intentional. Units
without usable changes can contribute through estimated cohort shares. If a
required scheduled cell is unsupported, that summary is unavailable; do not
silently renormalise to the supported subset. Report eligible cohorts, dates and
effective weights through `aggregation_weights(level)`, returning one row per
contributing (cohort, date) with the summary label, intended weight and cell
availability. For an unavailable summary, retain all intended weights rather
than renormalising to available cells. The canonical simple container's
`weight=[1.0]` describes its one summary row, not constituent cell weights.
Pre-treatment event rows are labelled trend differentials.

Let Psi contain the O(1) influence vectors for the requested supported outputs,
centred over the complete unit registry. Use:

```text
unit covariance:    V_hat = Psi' Psi / n^2
cluster sums:      S_k = sum_{i in cluster k} Psi_i
cluster covariance: V_hat = K/(K-1) * sum_k(S_k S_k') / n^2
```

The first convention is HC0; clustered inference uses the stated CR1 factor and
all K study-registry clusters. Cluster membership must be constant within units.
Do not multiply by the number of panel rows. Clustering changes covariance, not
the identity-GMM estimates. Store V_hat on the estimator scale, not the
sqrt(n)-asymptotic scale. Use `safe_inference` with the point estimate, standard
error, alpha and a normal reference to compute pointwise fields together.
With fewer than two clusters or invalid variance, make every corresponding
inference field unavailable while preserving identified point estimates. Handle
exact-zero standard errors through the shared helper's documented convention.
Neither normal intervals nor CR1 promise valid few-cluster coverage.

For validation, reuse a single centred Rademacher multiplier stream across all
outputs. Draw one sign per unit, or per cluster. Perturbations are:

```text
unit:     z_b = sum_i xi_b,i Psi_i / n
cluster:  z_b = sqrt(K/(K-1)) * sum_k xi_b,k S_k / n
```

Their conditional covariance must equal V_hat, up to multiplier Monte Carlo
error. Do not redraw signs by link or cohort. These internal checks supply no simultaneous bands or public bootstrap inference.

## 7. End-to-end pseudocode

```text
fit_chained(data, outcome, unit, time, first_treat, covariates, cluster, alpha):
    validate supported design; build full unit registry and stored calendar
    enumerate all scheduled cohort-time targets and their reference dates
    for each cohort and every ordered stored date pair:
        record endpoint support; retain supported pairs in deterministic order
        estimate normalised IPW link and complete unit-aligned link IF
        construct the incidence row with the reference date anchored
    stack all link estimates, design rows and full-registry IF columns
    identify each target contrast from the design row space
    recover supported post-treatment estimates and IFs using the identity metric
    report direct pre-reference comparisons and IFs, with their own support status
    form declared aggregates and propagate cohort-share uncertainty
    compute full unit/cluster covariance and pointwise inference
    record counts, catalogue, support reasons, weights, rank and nuisance diagnostics
    retain labelled covariance and aggregation inputs using upstream contracts
    return a ChainedDiDResults object; do not modify CallawaySantAnna behaviour
```

Candidate seams are `chained_did.py` and `chained_did_results.py`, shared
`linalg.py` and `utils.py`, and the current `AggregationKit`/staggered aggregation
infrastructure. Existing consumers can expect contributions on another scale:
convert O(1) psi to O(1/n) contributions exactly once at that seam, with a
round-trip covariance test. Do not assume that a compatible result field name
makes inference conventions compatible.

## 8. Numerical and statistical release gates

V01-V10 require small independently calculated fixtures and invalid-input checks.
Include the earlier-bridge example above, disconnected targets, non-adjacent
windows, different cohort response rates, shared controls and zero-/one-outcome
units. Separate rank support, nuisance failure and covariance failure. Algebraic
fixtures use scale-aware tolerances around 1e-10 on well-conditioned unit-scale
data; declare looser tolerances before assessing ill-conditioned fixtures.

For V11 pin `cdid` commit `1039120668cceabd8111cad82d32ab3927c2b94f`, version
1.1.0.0, `did` 2.5.1, `DRDID` 1.3.0 and R 4.5.2. Match IPW, never-treated
controls, zero anticipation, PT-post, universal reference reporting, no
observation correction and identity GMM. Match direct pre-reference diagnostics
separately from the internal GMM nuisance coordinates. Keep propensity scores in the interior
so the reference's probability safeguards are inactive. Supply exactly the same
calendar and unit registry. Compare target labels, availability, links where
exposed, cell estimates, covariance and post-treatment aggregates.

`cdid`'s analytical matrix is on the sqrt(n) scale: divide by n before comparison.
For clusters, also reconcile the proposed K/(K-1) factor explicitly; never tune a
factor to make a fixture pass. Match analytical rather than multiplier-IQR
standard errors. Initial reference tolerances: estimate `rtol=1e-6`,
`atol=1e-7*max(1,outcome_SD)`; covariance `rtol=1e-5`,
`atol=1e-7*max(1,outcome_SD^2)`. Investigate disagreement before changing these
limits. If catalogue/reporting differences prevent a matched comparison, retain
the failing comparison and resolve the specification rather than weakening it.

V12 is a separately run qualification study, not a slow unit test disguised as
proof. Prespecify seed lists, analytic population targets and scenarios before
running: balanced, rotating, intermittent and cohort-dependent observation;
static covariates affecting treatment and trends; heterogeneous cohort/horizon
effects; serial correlation; and independent versus clustered shocks. Use at
least 2,000 independent replications per core scenario, n=1,000 and 4,000, and
50 and 100 independent clusters for clustered scenarios. Report fit failures and
unsupported targets as well as conditional-on-success summaries.

For each prespecified core target report bias, root mean squared error, empirical
standard deviation, mean analytical variance, 95% pointwise coverage and 5% null
rejection. Require absolute bias below 0.1 empirical standard deviations,
mean estimated variance / empirical variance in [0.9,1.1], and coverage/null
rejection within three binomial Monte Carlo standard errors of nominal, assessed
on prespecified targets. Investigate failures; do not pool them away or remove
unsuccessful fits from the headline rate. The larger-n study must support the
claimed asymptotic behaviour. These are proposed project acceptance thresholds,
not universal statistical constants or evidence already obtained.

V13 deliberately breaks PT, the pair sampling restriction or the propensity
model, including selection on heterogeneous effects. Record resulting bias and
coverage loss as assumption sensitivity, not a software defect that must be
repaired by undocumented changes of target. V14 follows current upstream API,
documentation and installed-consumer checks. No new dependencies or backend
acceleration are planned for the first release.

## Sources and derivation boundary

The [proposal's pinned source list](chained-did-proposal.md#sources) is controlling.
The paper's section 2.2.2 supplies the general-pair linear inverse problem and
baseline sampling restriction; appendix A.2 supplies propensity-score and joint
influence-function reasoning. This document spells out an estimating-equation
derivation for normalised IPW and chooses fixed identity weighting for the first
release. The extended PT-post parameterisation preserves earlier differentials
rather than importing PT-all restrictions. These derivations need independent
review and numerical/statistical validation before implementation is qualified.
The R package's documentation informs reference alignment, not the authority of
the Python methodology or a requirement to reproduce its default weighting.

The [validation follow-up](chained-did-validation-followup.md) confirms that
the native unweighted observed-risk logit differs from the reference package's
response-calibrated score on an incomplete continuous-covariate fixture. The
criterion specified here remains unchanged; adopting calibrated scores would
require a methodological revision and a matching influence-function derivation.
