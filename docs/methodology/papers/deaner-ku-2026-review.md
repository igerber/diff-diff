# Paper Review: Causal Duration Analysis with Diff-in-Diff

**Authors:** Ben Deaner and Hyejin Ku

**Citation:** Deaner, B., & Ku, H. (2026). *Causal Duration Analysis with Diff-in-Diff*. arXiv:2405.05220v2, working paper revision. [Version record](https://arxiv.org/abs/2405.05220v2), [original PDF](https://arxiv.org/pdf/2405.05220v2).

**PDF reviewed:** `.workflow/sources/deaner-ku-2026-v2.pdf`

**Review date:** September 5, 2026

The reviewed version was submitted May 25, 2026; its title page is dated May 26,
2026. The original arXiv submission was May 8, 2024. The author's
[research page](https://bendeaner.wordpress.com/research/), checked September 5,
2026, lists the paper as revise and resubmit at *Quantitative Economics*.
This is not evidence of acceptance or journal publication; the citation above
deliberately identifies the reviewed working paper revision.

This document is a methodology foundation, not a shipped estimator. The
maintainer-approved first implementation is a two-group `DurationDiD` with common
treatment timing, common dynamics and proportional hazards, individual bootstrap
inference, and pre-treatment diagnostics. Covariate adjustment and staggered
adoption are deferred from that implementation. They, and all other appendices,
are covered below. Library recommendations are labeled separately from paper
results and reference-software behavior.

## Source provenance and coverage

All 50 PDF pages were read through independent overlapping extractions of
pp.1–20, 16–35, and 31–50. Overlaps were reconciled; no missing or illegible region
was found. Original pages containing figures, tables, algorithms, and ambiguous
equations were also rendered and visually inspected. The page references below
use the PDF's printed page numbers, which match its page order.

| Pages, inclusive | Coverage |
|---|---|
| 1–5 | Title, abstract, introduction, related literature |
| 5–9 | Section 1, population, potential outcomes, ordinary DiD, Figure 1.1 |
| 9–20 | Section 2, hazard restrictions, Theorems 1–3, Proposition 1, specification and censoring |
| 20–24 | Section 3, estimation, weighting, bootstrap, diagnostics, asymptotic qualifications |
| 24–28 | Section 4, application and Figures 4.1–4.4, including conclusions before References on p.28 |
| 28–32 | Complete bibliography, including references before Appendix A on p.32 |
| 32–38 | Appendix A.1–A.4.1: motivation, conditional identification, staggered adoption, semiparametric adjustment |
| 38–40 | Appendix B and Algorithms 1–2, after the estimator on p.38 and before Appendix C on p.40 |
| 40–44 | Appendix C, simulation design, Tables 1–2, Figures C.1–C.4 |
| 45 | Appendix D, proportional-hazards application and Figures D.1–D.2 |
| 46–50 | Appendix E, all identification proofs |

The review snapshot contains seven unchanged originals. The code snapshot is
pinned to commit `202e92ef222bf7d250c26c98e8fb42e151223c71` of the
[authors' repository](https://github.com/ben-deaner-teaching/Duration-DiD/tree/202e92ef222bf7d250c26c98e8fb42e151223c71).
The following paths are relative to `.workflow/sources/`; that local cache and
its seven-entry `.workflow/sources.json` are review evidence, not public package
assets. Each filename links to the canonical original bytes. All seven inventory
digests were verified against the cached files.

| Cached original | SHA-256 |
|---|---|
| [deaner-ku-2026-v2.pdf](https://arxiv.org/pdf/2405.05220v2) | `d40df031669d5ceb2a3fa39c9e904c708fd7f60e7f60f460ab4adab02945f639` |
| [duration-did-reference-202e92ef/README.md](https://raw.githubusercontent.com/ben-deaner-teaching/Duration-DiD/202e92ef222bf7d250c26c98e8fb42e151223c71/README.md) | `d7bcfb98c9498a6701465d620cc2e3c6d1c65e0ce3377245d856228f147b9b7b` |
| [duration-did-reference-202e92ef/UIP_application.do](https://raw.githubusercontent.com/ben-deaner-teaching/Duration-DiD/202e92ef222bf7d250c26c98e8fb42e151223c71/UIP_application.do) | `5a4ee15a3a8cbaabed725ea4973cd64f2c533581db85abb1cde45c907d09be2f` |
| [duration-did-reference-202e92ef/UIP_application_revision_R2.m](https://raw.githubusercontent.com/ben-deaner-teaching/Duration-DiD/202e92ef222bf7d250c26c98e8fb42e151223c71/UIP_application_revision_R2.m) | `092ed6ab096a2127693221708df17f4e56ac52d5f235a82c1d2fffdd0fefa8d5` |
| [duration-did-reference-202e92ef/durationDiD.m](https://raw.githubusercontent.com/ben-deaner-teaching/Duration-DiD/202e92ef222bf7d250c26c98e8fb42e151223c71/durationDiD.m) | `3449d12a4b9e2aca7f943493c9f303130f90d348d089fd88604ac352280f43df` |
| [duration-did-reference-202e92ef/durationdid.ado](https://raw.githubusercontent.com/ben-deaner-teaching/Duration-DiD/202e92ef222bf7d250c26c98e8fb42e151223c71/durationdid.ado) | `6e66e9f19ae3cf384d08209d52abcbaedcc85196f97e524e168ee7eaca58aa0e` |
| [duration-did-reference-202e92ef/montecarlo_revision.m](https://raw.githubusercontent.com/ben-deaner-teaching/Duration-DiD/202e92ef222bf7d250c26c98e8fb42e151223c71/montecarlo_revision.m) | `f7a01d3a46762c6d05eac88014c20dc60da468603367746dcb02da9fa1b5c022` |

## Methodology Registry Entry

This prospective entry follows [REGISTRY.md](../REGISTRY.md)'s format. Keep it
here until the estimator ships; no public API, catalog, or production registry
entry is added by this review.

## DurationDiD

**Primary source:** Deaner and Ku (2026), arXiv:2405.05220v2, cited above.

**Key implementation requirements:**

*Assumption checks / warnings:*

- Absorbing binary outcomes, fixed populations, no anticipation before the common
  intervention, and unaffected controls. The chosen restriction is on untreated
  hazards, not on cumulative outcome levels.
- At least two pre-treatment dates, including the baseline, and one post-date.
  A nontrivial diagnostic requires at least three pre-dates. Proportional hazards
  additionally needs informative positive control hazard increments.
- Preserve the initial treated survival probability and whole-group denominator.
  Check survival/log domains and counterfactual validity; never silently clip.
- Individual bootstrap inference requires independence across individuals and
  regular interior identification. Group intervention identification alone does
  not justify this statistical independence assumption.

*Estimator equation (Theorem 1; Equations 3.1–3.4, with the PH choice below):*

```text
S_hat[k,t] = 1 - mean(Y[i,t] | G[i]=k)
R_hat[k,t] = -log(S_hat[k,t])
D_hat[k,t] = R_hat[k,t] - R_hat[k,1]
H_hat[k,t] = D_hat[k,t] / (t-1), t>1

CD: c_hat = sum_t alpha[t] (H_hat[1,t] - H_hat[2,t])
    R0_hat[1,t] = R_hat[1,1] + D_hat[2,t] + (t-1)c_hat
PH: c_hat = sum_t alpha[t] D_hat[1,t]/D_hat[2,t]
    R0_hat[1,t] = R_hat[1,1] + c_hat D_hat[2,t]

tau_hat[t] = exp(-R0_hat[1,t]) - S_hat[1,t], t>tstar
```

Here `tstar` is the last untreated date, `alpha` sums to one over selected usable
pre-dates, and positive `tau` means increased cumulative absorption/exit.

- **Note:** PH above is the proposed mean-of-ratios estimator, supported by the
  authors' code and Theorem 1. It is not literal Equation 3.5, nor the
  finite-sample PH specialization of Equation 3.6. The distinctions and a
  reproducible reciprocal-coefficient check appear below.
- **Note:** Equal weights on eligible pre-dates and Appendix B's fixed-anchor
  diagnostic are proposed first-version defaults. Paper/application alternatives
  remain documented; they are not silently interchanged.

*With covariates / doubly robust:*

Deferred from the initial estimator. Sections 2.2/3.1 and Appendix A offer
conditional identification, initial-survivor balancing, and semiparametric
adjustment. None is presented as doubly robust. Their distinct assumptions and
estimators are developed below.

*Standard errors (Section 3.2; Appendix B):*

- Resample whole individual histories with replacement from the pooled sample;
  re-estimate all quantities. No wild/multiplier weight distribution applies.
- Bootstrap SD; symmetric bands from centered absolute bootstrap deviations,
  with post-period maxima for simultaneous bands. Recommend 1,000 draws and
  sample SD (`ddof=1`); these are library defaults, not universal paper rules.
- Serial dependence within a person is preserved. Arbitrary higher-level
  clustering, survey inference, and censoring-adjusted inference are outside
  this first version.

*Edge cases:*

- Invalid panel, absorption reversal, missing cells, or unsupported design:
  explicit validation error.
- Zero survival or PH denominators: identify unsupported periods/draws; no
  epsilon replacement or silent sample alteration.
- Zero/invalid SE: joint undefined test statistics, p-values, and intervals;
  retain point estimates and failure diagnostics when meaningful.
- **Note:** A fixed-draw failure policy and simultaneous-family validity must be
  explicit; the conservative first-version recommendation is detailed below.

*Algorithm (Theorem 1 and Algorithms 1–2):*

1. Validate the panel and target; compute group survival and baseline-normalized
   average hazards.
2. Fit the selected hazard relationship on pre-treatment moments; impute
   counterfactual treated survival and post-period absorption ATT.
3. Recompute the complete calculation in each individual bootstrap draw; form
   pointwise and simultaneous effect inference.
4. Separately compare pre-treatment hazard gaps/ratios with their last
   pre-treatment value and use a centered simultaneous bootstrap test.

**Reference implementation(s):** Authors' MATLAB `durationDiD` and Stata
`durationdid` at the pinned commit. Static audit only; executability and exact
numerical equivalence have not been established.

**Requirements checklist:**

- [ ] Balanced individual-panel validation and explicit exit-ATT estimand.
- [ ] CD weighted gap and PH mean-ratio coefficient with baseline normalization.
- [ ] Whole-individual bootstrap with complete nuisance re-estimation.
- [ ] Coherent centered-bootstrap pointwise, simultaneous, and scalar inference.
- [ ] Separate fixed-anchor pre-treatment diagnostics and unavailable-test states.
- [ ] Domain, invalid-curve, failed-draw, and joint-NaN inference handling.
- [ ] `BaseEstimator`/results/serialization/event-study integration and documentation.
- [ ] Regression scenarios and independent reference checks specified below.

## Identification and estimator derivation

### Population, time, and causal assumptions

Section 1 (pp.5–7) follows individuals `i=1,...,n` at dates `t=1,...,T`.
`Y_it=1` means the spell has ended by date `t`; zero means it continues. No new
spells start after the initial observation, but some spells may already have
ended before it. `G_i` is fixed; group 1 receives the intervention strictly after
`tstar`, and group 2 supplies unaffected comparisons. Time can be calendar time
or elapsed spell time. The same calendar event need not occur at the same elapsed
duration for everyone; a common-timing analysis must justify its chosen clock.

The potential outcome `Y_it^(0)` removes the group intervention. The target is

```text
tau[t] = E[Y_it - Y_it^(0) | G_i=1].                    (p.6)
```

It averages over the whole fixed treated group, including baseline-absorbed
people, whose effects are zero under the assumptions. It is neither an ATT
conditional on surviving to treatment nor a hazard ratio nor a mean-duration
effect. If spells begin together, the counterfactual mean is a duration CDF on
the observed grid. A survival-probability effect has the opposite sign. Dropping
baseline-absorbed people changes the population and rescales the estimand.

**Assumption 1** requires binary absorbing factual and counterfactual outcomes:
once one, always one. **Assumption 2** requires factual/untreated equality for
everyone through `tstar` and controls afterward. Within-group spillovers are
permitted by the group-level causal model (p.7); control spillovers are excluded.
Observed monotonicity is checkable, whereas no anticipation, control validity,
and counterfactual hazard restrictions cannot be verified from these checks.

Write `S_kt=1-E[Y_it|G=k]`, `R_kt=-log(S_kt)`, with superscript `(0)` for
counterfactual quantities (Equations 2.6–2.7). The instantaneous hazard is the
limit of exit probability over a short interval conditional on survival,
divided by its length (p.9). With the continuous-duration regularity underlying
the paper's integral identity,

```text
Delta_s R_kt = R_kt - R_k,t-s
Delta R_kt^(0) = integral_(t-1)^t h_k^(0)(r) dr          (2.8)
D_kt^(0) = R_kt^(0)-R_k1^(0) = integral_1^t h_k^(0)(r) dr
H_kt^(0) = D_kt^(0)/(t-1).                             (E.1)
```

Absorption alone does not supply an absolutely continuous hazard representation
for arbitrary mass points. Discrete observation of a continuous duration process
is compatible with these identities; the review does not extend their proof to
every discrete-duration hazard model. Remark 3 permits real observation dates;
elapsed durations must replace `t-1` consistently. The initial survival level is
unrestricted (p.10, footnote 4).

### Two specifications and their identification

Common dynamics (CD) says untreated hazard changes agree across groups
(Equation 2.1), equivalently

```text
h_1^(0)(t) = h_2^(0)(t) + c.                           (2.3)
```

Proportional hazards (PH) says proportional evolution agrees (Equation 2.2),
equivalently

```text
h_1^(0)(t) = c h_2^(0)(t), c>0,                       (2.4)
```

with strictly positive hazards as assumed in the paper. Integrating gives
constant differences of average hazards for CD (2.11), and constant ratios of
long increments for PH (2.12). Equations 2.9–2.10 express related restrictions
on successive integrated hazards. They concern untreated paths; observed treated
post-period hazards need not obey them.

**Theorem 1** (p.13) identifies CD `c=H_1t-H_2t` at every `1<t<=tstar`
(2.13). For PH, `D_1t=c D_2t` (2.15), and at least one nonzero control
pre-increment is required. The respective counterfactuals are

```text
CD: R_1t^(0) = R_11 + D_2t + (t-1)c                   (2.14)
PH: R_1t^(0) = R_11 + c D_2t                         (2.16)
E[Y_it^(0)|G=1] = 1-exp(-R_1t^(0)).                   (2.17)
```

Independent rearrangement makes the normalization and estimand transparent:

```text
CD: S_1t^(0) = S_11 (S_2t/S_21) exp(-(t-1)c)
PH: S_1t^(0) = S_11 (S_2t/S_21)^c
tau[t] = S_1t^(0)-S_1t.
```

The treated baseline factor is outside the PH exponent. Neither equal initial
survival nor a zero initial event fraction is required. Two pre-dates identify
one nuisance coefficient; a third supplies overidentification (Remark 1).
Only an extra known equal-hazards restriction, CD `c=0` or PH `c=1`, permits
identification with just one pre-date. That restricted special case is not the
proposed unrestricted estimator.

Remark 2 offers one-period or final-window CD identification moments. These
agree in population under the restriction, but averaging them gives different
finite-sample estimators. Remark 4 rewrites CD as
`R_kt^(0)=xi_k-c_k t+gamma_t` with `c_1=0` (2.18), PH as
`R_kt^(0)=xi_k+c_k gamma_t` with `c_1=1` (2.19), and equal hazards as additive
group/time effects in `R` (2.20). These group-specific normalizations differ
from the two-group coefficient `c` above.

### Sample estimators and the PH ambiguity

Equation 3.1 replaces each group expectation by its sample mean with fixed
denominator `n_k`. Equations 3.2–3.4 give the CD estimator in the prospective
registry entry. The text calls time weights positive and normalized; the
application explicitly uses zero early weights. Thus nonnegative weights over
a declared selected set reconcile the implementation descriptions. Suggested
alternatives emphasize periods near treatment or minimize asymptotic variance;
the paper supplies no universal optimal-weight procedure.

Equation 3.5 prints, under the natural squared-increment reading,

```text
c_hat_printed = sum_t alpha[t] D_hat[1,t]D_hat[2,t]
                / sum_t alpha[t] D_hat[1,t]^2
R0_hat[1,t] = R_11 + c_hat_printed D_hat[2,t].           (3.5, printed)
```

The exponent placement also leaves squared increment versus increment of a
squared `R` typographically ambiguous. Either way, the expression cannot be
used uncritically. Under exact `D_1t=c D_2t`, the displayed slope reading gives
`1/c`, while (2.16) requires `c`. The empirical baseline also needs a hat.
Replacing the denominator by control squared increments repairs that slope's
direction. It does **not** uniquely select a finite-sample PH estimator:

| PH estimator | Coefficient | Evidence / meaning |
|---|---|---|
| Mean ratios, proposed core choice | `sum alpha[t] D1[t]/D2[t]` | Direct moments from Theorem 1; pinned MATLAB/Stata implement a mean of usable ratios |
| Repaired cumulative-increment slope | `sum alpha[t] D1[t]D2[t] / sum alpha[t] D2[t]^2` | Directional repair of printed 3.5 |
| Average-hazard slope | `sum alpha[t] H1[t]H2[t] / sum alpha[t] H2[t]^2` | PH specialization of 3.6 with zero intercept |

All equal the true ratio under exact population PH and nondegenerate moments.
They differ under sampling noise, weighting, and zero denominators. The second
and third coincide only with suitable elapsed-time-squared reweighting, not
generally with the same `alpha`. The recommended first implementation uses
equal weights over original-sample eligible pre-dates with positive `D2` and
finite logs. Freeze that set for bootstrap draws; a failed denominator then
creates a failed draw, rather than silently changing the estimator in that draw.
This eligibility/failure convention is library policy, not a claim of exact
reference-software parity. Require a positive fitted PH ratio for its strict
positive-hazard interpretation; a zero ratio is a boundary case. Report selected
and excluded fitting dates with their reasons, including unavailable estimation
when the eligible set is empty.

### General affine hazards and constrained estimators

Equation 2.5 allows
`h_1^(0)=beta_1+sum_(k=2)^K beta_k h_k^(0)`, with known constraint set `B`.
**Theorem 2** identifies coefficients from

```text
H_1t = beta_1 + sum_(k=2)^K beta_k H_kt, 1<t<=tstar,    (2.21)
```

provided the solution in `B` is unique. For unrestricted `K` coefficients,
`tstar>K` is necessary and generically sufficient, not a guarantee against
collinearity. Duration triple differences fixes weights `1,1,-1` on three
controls and leaves an intercept; duration synthetic control fixes zero
intercept and nonnegative control weights summing to one (p.10).

The coherent weighted least-squares reading of Equations 3.6–3.7 is

```text
beta_hat = argmin_(beta in B) sum_(t=2)^tstar alpha[t]
  (H_hat[1,t]-beta_1-sum_(k=2)^K beta_k H_hat[k,t])^2
R0_hat[1,t] = R_hat[1,1]+(t-1)beta_hat[1]
             +sum_(k=2)^K beta_hat[k] D_hat[k,t].
```

The printed objective starts at `t=1`, where `H=0/0`. Exclude the row before
evaluation; multiplying it by zero does not define it. These extensions are
outside the core scope. Binding constraints also require separate inference:
ordinary bootstrap normal approximations need not remain valid at their
boundaries (Section 3.3).

## Bootstrap inference and pre-treatment diagnostics

### Algorithm 1: whole-individual bootstrap

Section 3.2 and Appendix B (pp.22–24,38–40) assume independent individuals with
arbitrary serial dependence within each history. For a declared post-period
family `P={tstar+1,...,T}`, the reconciled Algorithm 1 is:

1. Compute `tau_hat[t]` for `t in P` on the original data.
2. For each `b=1,...,B`, independently sample `n` indices uniformly with
   replacement from all individuals. Carry their complete outcomes, groups,
   and any covariates together. Repeated indices represent separate sampled
   individuals, not duplicate records to discard.
3. Recompute group means, log survivals, all coefficients, and effects.
   In adjusted extensions, re-estimate weights or nonlinear first stages too.
4. Calculate `sigma_hat[t]=SD_b(tau_star[b,t])` and centered absolute pivots
   `Z_star[b,t]=abs(tau_star[b,t]-tau_hat[t])/sigma_hat[t]`.
5. Pointwise criticals are `Q_(1-alpha){Z_star[b,t]}`; the simultaneous critical
   is `Q_(1-alpha){max_(t in P) Z_star[b,t]}`.
6. Form symmetric bands `tau_hat[t] +/- q sigma_hat[t]` using the appropriate
   pointwise or simultaneous critical.

There is one bootstrap SD per coordinate, not a nested resampling/studentizing
loop. The intervals are centered-bootstrap absolute-deviation bands, not
percentile intervals of the raw estimates. Original SD cancels from the
pointwise width, but provides the relative scaling in the simultaneous maximum.
The bootstrap is pooled pairs resampling, not a stratified, residual, wild,
multiplier, or group-cluster bootstrap.

Algorithm 1 inconsistently includes `tstar` in some loops/maxima but defines
SEs and pointwise intervals only after it. Table 2 explicitly defines coverage
over `tstar+1,...,T`, supporting the post-only family above. A reference-period
ATT is causally zero and has undefined studentized inference; pre-fit residuals
must not be mixed into this post-period family.

**Proposed numerical conventions:** use `ddof=1`, an explicit inverse empirical
CDF quantile, and `numpy.random.Generator` with recorded seed. For a scalar null
`theta=0`, use the empirical fraction of centered absolute draws at least as
large as `abs(theta_hat)`, with equality counted conservatively. For a
simultaneous null, compare bootstrap maxima with
`max(abs(theta_hat[t]/sigma_hat[t]))`. These p-values are an inversion-style
library companion to the paper's bands, not an additional paper theorem.
Finite-bootstrap discreteness and ties should be reported consistently.

Pooled draws can lose a group, exhaust survival, or eliminate a PH increment.
The paper does not specify a failure/retry rule. The conservative first-version
recommendation is fixed `B` without retries: record each failure and reason;
if any draw needed for an inferential family is invalid, retain meaningful
point estimates but mark that family's inference unavailable. Successful draws
may be retained as diagnostics, without treating a filtered distribution as
the original bootstrap. Use the same complete-draw mask across a simultaneous
family; separate per-column filtering does not define its joint maximum.
Do not silently stratify, redraw, clip, or substitute tiny positive SEs.

### Algorithm 2: separate specification tests

Appendix B's Algorithm 2 (p.40) compares each pre-treatment average-hazard gap
or ratio with the last pre-treatment value:

```text
CD: delta_hat[t] = (H_hat[1,t]-H_hat[2,t])
                  -(H_hat[1,tstar]-H_hat[2,tstar])
PH: delta_hat[t] = D_hat[1,t]/D_hat[2,t]
                  -D_hat[1,tstar]/D_hat[2,tstar].
```

Compute them for `2<=t<=tstar`, but studentize/test only
`J={2,...,tstar-1}`: the final anchor is identically zero and baseline `H` is
undefined. For each pooled individual draw, recompute these contrasts with the
same normalization. Calculate

```text
sigma_delta[t] = SD_b(delta_star[b,t])
M_star[b] = max_(t in J) (abs(delta_star[b,t]-delta_hat[t])/sigma_delta[t])
q = Q_(1-alpha){M_star[b]}
diagnostic_band[t] = delta_hat[t] +/- q sigma_delta[t].
```

Reject if any band excludes zero. An equivalent proposed test statistic is
`M=max_(t in J) abs(delta_hat[t]/sigma_delta[t])`, with the empirical tail
fraction of `M_star` as p-value. Algorithm 2 incorrectly prints `tau_hat` at
the band center and calls the maximum-based critical pointwise; its preceding
steps and rejection target require `delta_hat` and a simultaneous critical.

At least three pre-dates are required for nonempty `J`. PH diagnostic ratios
need positive control increments at every tested date and the anchor, a
stronger requirement than identification from one informative moment. The
recommended behavior is to report the full diagnostic unavailable if those
required ratios or SEs fail, while retaining a separately valid point estimate.
Diagnostic curves have their own units and are not pre-treatment outcome ATTs.
Failure to reject cannot establish post-treatment identification or rule out
low power.

**Main text versus appendix:** Section 3.2, p.23 instead prints contrasts of
the initial window with an equal-length final window ending at `tstar`:

```text
CD: (D_1t-D_2t)/(t-1)
    -[Delta_(t-1)R_1,tstar-Delta_(t-1)R_2,tstar]/(t-1)
PH: D_1t/D_2t
    -Delta_(t-1)R_1,tstar/Delta_(t-1)R_2,tstar.
```

The lag operator on p.12 means the final window starts at `tstar-(t-1)`.
These are also valid population null contrasts, supported by Remark 2; they
are not arithmetic errors. They are distinct finite-sample diagnostics from
Algorithm 2's fixed-baseline anchor. The proposed first implementation chooses
Algorithm 2, matching the application description. Never replace a lagged
final-window numerator with a baseline-to-`tstar` difference while retaining
the shorter elapsed-time denominator.

### Scope of statistical justification

Section 3.3 invokes GMM/sequential-GMM arguments with `n` increasing and `K,T`
fixed, unique identification, and standard regularity. It explicitly omits a
formal asymptotic theorem and proof. The appendix theorems prove identification,
not bootstrap validity, efficiency, or finite-sample unbiasedness. Binding
inequality constraints can break differentiability and bootstrap coverage;
covariate weights must be bounded, requiring overlap bounded away from zero.
Near-zero survival, weak PH denominators, dependence across people, increasing
numbers of tiny cohorts, and censoring require additional work. No general
analytic variance, degrees-of-freedom correction, or universal draw count is
provided by the paper.

## Complete extension and proof coverage

### Appendix A.1: why outcome-level parallel trends can mislead

Equation 1.1 imposes a constant untreated event-probability gap. Appendix A.1
(p.32), for a positive gap `c`, positive control hazard, and
`E[Y_t^(0)|G=2]<1-c`, derives

```text
h_1^(0)(t)/h_2^(0)(t)
  = (1-E[Y_t^(0)|G=2])/(1-E[Y_t^(0)|G=2]-c).
```

As control absorption approaches `1-c`, the hazard ratio diverges. Thus a
constant nonzero CDF gap can impose increasingly different survivor hazards.
The p.8 numerical example increases the required exit-probability ratio from
two to three as outcome shares move from `.8/.6` to `.9/.7`. Figure 1.1 shows
converging untreated outcome means with parallel time-average hazards; naive
DiD can reverse the true effect's sign. This motivates hazard restrictions;
it does not prove that every use of ordinary DiD with a duration outcome fails.

### Covariate-conditional identification: Section 2.2 and Appendix A.2

For time-invariant `X`, Equations 2.22–2.24 allow conditional additive gaps
`c(x)`, ratios `c(x)`, or affine coefficients `beta(x)`. Appendix A.2 repeats
the general restriction as A.1. **Theorem A.1** (pp.32–33) applies Theorem 2
separately at each `x`:

```text
H_1t(x) = beta_1(x)+sum_(k=2)^K beta_k(x) H_kt(x)       (A.2)
R_1t^(0)(x) = R_11(x)+(t-1)beta_1(x)
              +sum_(k=2)^K beta_k(x)D_kt(x).
```

Unique solution in the known set `B` is required per stratum; the theorem
suppresses `x` on some coefficient symbols but does not establish constancy
across strata. Transform to `1-exp(-R_1t^(0)(x))`, then integrate over the
treated covariate distribution to get the unconditional treated mean.
The source omits a separate proof because Theorem 2 applies to the stratum.
Nonparametric conditional-mean estimation is suggested but can be impractical
with high-dimensional covariates or sparse cells. No smoothing/selection
algorithm or special high-dimensional inference guarantee is specified.

### Initial-survivor balancing: Sections 2.2.1 and 3.1

This is a distinct route, balancing covariates among **baseline survivors**:

```text
omega_k(x) = P(X=x|Y_1=0,G=1)/P(X=x|Y_1=0,G=k)        (2.25)
p_k(x) = P(G=k|Y_1=0,X=x)
omega_k(x) = p_1(x)P(G=k|Y_1=0)
             /[p_k(x)P(G=1|Y_1=0)].                  (2.26)
```

Use densities for continuous covariates. The treated weight is one. Overlap
requires positive comparison propensity on relevant treated support; bounded
weights matter for inference. Balancing the full population or current
post-treatment risk sets would define a different procedure.

The weighted hazard (2.27) is the derivative of weighted cumulative event
mass divided by weighted survivor mass. Define

```text
R_tilde[k,t] = -log E[omega_k(X)(1-Y_it)|G=k].
```

The identifying restrictions are directly on weighted untreated hazards:
`h_tilde_1^(0)=h_tilde_2^(0)+c` or `c h_tilde_2^(0)` (2.28), and their affine
generalization (2.29). **Theorem 3** identifies coefficients from (2.30),
`H_1t=beta_1+sum beta_k H_tilde_kt`, and imputes
`R_1t^(0)=R_11+(t-1)beta_1+sum beta_k D_tilde_kt`. The resulting mean
conditions on treated group, having averaged over `X`; it is not a
stratum-specific mean despite the source's occasional terminology.

**Proposition 1** supplies a sufficient primitive condition for weighted CD:
conditional CD with a gap `c(x)=c` constant in `x`. Equal conditional hazards
and certain additive-separable hazard models are examples. It does not show
that arbitrary conditional CD or PH collapses to the same marginal weighted
restriction. Selection into survivor risk sets matters.

For discrete covariates, let `N^S_k(x)` count group-k baseline survivors in
stratum `x`, and `N^S_k=n_k(1-Ybar_k1)`. Equation 3.8 is

```text
omega_hat_k(x) = N^S_k N^S_1(x)/(N^S_1 N^S_k(x)).
```

For continuous/many-valued `X`, estimate `p_k(x)` among baseline survivors,
for example by logistic regression, and use
`omega_hat_k(x)=N^S_k p_hat_1(x)/(N^S_1 p_hat_k(x))`. Estimate survivor mass
as `sum_(i:G=k) omega_hat_k(X_i)(1-Y_it)/n_k`, then substitute its negative
log in the estimator. In general `E[omega(1-Y)]` is not `1-E[omega Y]`;
normalization cannot be dropped casually. A time-constant multiplicative
scaling cancels in long differences, but does not justify inconsistent
baseline treatment or a different estimand. Re-estimate weights in every
bootstrap draw. Missing support is an identification problem; trimming
changes the population and needs an explicit target.

### Appendix A.3: staggered adoption

Pages 33–35 redefine `G=k` as treatment after date `k`; `G=T` includes
never-treated units within the observation window. **Assumption 2*** gives
untreated equality through each group's last pre-date. Equations A.3–A.4
impose common untreated hazard changes or proportional evolution across all
cohorts. For `1<s<=t`, their average hazards obey

```text
CD: H_kt^(0)-H_ks^(0)=c_ts
PH: H_kt^(0)/H_ks^(0)=c_ts.
```

Not-yet-treated cohorts `k>=t` make these observable (A.5–A.6); at least one
such cohort must exist, `P(G>=t)>0`. For treated target `k`, select
`1<s<=k<t`, naturally `s=k`, and impute

```text
CD: R_kt^(0)=R_k1+(t-1)c_ts+(t-1)H_ks
PH: R_kt^(0)=R_k1+(t-1)c_ts H_ks.
```

The PH population line on p.34 prints `R_kt` instead of `R_k1`; A.6 and
the same page's sample formula require the baseline correction shown above.
Estimate each `c_ts` by a normalized weighted average of eligible cohorts'
differences or ratios, then obtain cohort-time absorption ATT. The paper does
not specify an overall aggregation or joint inference across cohort-time cells.

Already-treated cohorts are ineligible controls. No comparison remains once
everyone is treated; `k=1` lacks a usable target pre-increment; PH needs
positive denominators. Nonlinear log transformations create small-cohort bias.
The authors suggest downweighting tiny cohorts and different methods if all
cohorts are small, leaving bias adjustment for future work. This is not a
justification for large-`K` asymptotics with tiny cohorts or for using existing
staggered-DiD aggregators without a new derivation.

### Appendix A.4: semiparametric covariate adjustment

Pages 35–38 impose a known positive link, for example `phi(z)=exp(z)`, and
group-specific coefficients:

```text
h_k^(0)(t,x) = phi(x'gamma_k) hbar_k^(0)(t)             (A.8)
hbar_1^(0)(t) = beta_1+sum_(k=2)^K beta_k hbar_k^(0)(t). (A.9)
```

The unrestricted baseline hazard `hbar` is not the unconditional hazard.
Together these imply the conditional affine relationship (A.7), with
intercept `phi(x'gamma_1)beta_1` and coefficients
`beta_k phi(x'gamma_1)/phi(x'gamma_k)`. For two groups, equal `gamma` and
`beta_2=1` give conditional CD; zero intercept gives conditional PH.

Let `r_kt=integral_1^t hbar_k^(0)(s) ds`. Then

```text
E[Y_it^(0)|Y_i1^(0)=0,G=k,X=x]
  = 1-exp(-phi(x'gamma_k)r_kt).                       (A.10)
```

Equation A.11 makes this observable in untreated cells: all control periods
and all groups' pre-periods. **Theorem 4** assumes a strictly positive,
strictly increasing link, full covariate support, the baseline-hazard relation,
and unique coefficients in `B`. Its statement/proof differ in risk-set support
and use of “and” versus “and/or”; the operative proof uses baseline survivors
and the union of untreated cells. An informative positive increment and
covariate normalization are also essential to recover `gamma`.

The baseline moment system and its implication are

```text
r_1t/(t-1) = beta_1+sum_(k=2)^K beta_k r_kt/(t-1)      (A.12)
r_1t = (t-1)beta_1+sum_(k=2)^K beta_k r_kt.
```

The latter corrects the printed `(1-t)beta_1`. With group conditioning
restored in A.13, the counterfactual mean is

```text
E[Y_it^(0)|G=1] = E[Y_i1|G=1]
 + E[(1-Y_i1){1-exp(-phi(X_i'gamma_1)r_1t)}|G=1].      (A.13)
```

Appendix A.4.1 proposes nonlinear least squares over untreated cells `U`:

```text
min_(gamma,r) sum_((k,t) in U) sum_i
  1{G_i=k}(1-Y_i1)[Y_it-1+exp(-phi(X_i'gamma_k)r_kt)]^2.
```

It mentions Cox partial likelihood, but prefers NLS for ties from discrete
observation. A common `gamma` across groups is optional. Fit baseline-hazard
relations using estimated `r`, then form the sample analogue of A.13. The
corrected empirical ATT is

```text
Ybar_1t-Ybar_11
 -(1/n_1)sum_i 1{G_i=1}(1-Y_i1)
                  [1-exp(-phi(X_i'gamma_hat_1)r0_hat_1t)].
```

The p.38 formula prints current-survivor weight `(1-Y_it)` and an incorrect
summation index. Baseline weighting is forced by A.13 and the proof: current
survival is treatment affected and changes the estimand. No division by the
current number at risk repairs it.

The source does not settle optimizer, starts, tolerances, monotonicity or
nonnegative constraints on fitted increments, or existence of a finite-sample
solution. Covariates should exclude an intercept under the stated baseline
normalization (footnote 10); zero baseline increments cannot identify `gamma`
through the proof's inverse link. A future extension needs these additional
conditions, complete nuisance re-estimation, and separate boundary-inference
analysis. It is not an implementation shortcut for the unadjusted core.

### Censoring, repeated cross-sections, and all Appendix E proofs

Remark 3 (p.14) notes that identification uses only group means, so repeated
cross-sections can suffice if they represent the same populations over time.
That does not supply panel-style resampling for independently changing samples.
Section 2.3 suggests substituting Kaplan–Meier group survival under independent
right-censoring (`C_it` independent of `Y_it` conditional on group), and other
survival estimators for dependent censoring. It provides no complete
censoring-adjusted algorithm or bootstrap proof. Missing panel rows cannot be
silently encoded as continued survival. Administrative end of a complete common
observation window, with remaining survivors, is different from dropout.

Appendix E (pp.46–50) gives the following implementation-relevant arguments:

- **Theorems 1–2:** integrate affine hazards from the initial date (E.1–E.2),
  replace pre-period counterfactuals using no anticipation (E.3), identify the
  unique coefficient vector, and use unaffected controls after treatment.
  Theorem 1 is the restricted two-group case. The treated initial `R_11`
  remains essential; the proof's stray post-period treated equality is not an
  additional no-effect assumption.
- **Proposition 1:** absorption gives
  `(1-Y_t^(0))(1-Y_1^(0))=1-Y_t^(0)`. Initial-survivor density ratios transfer
  covariate expectations; a constant conditional additive gap factors out of
  survival integrals and yields the weighted hazard gap. Positive baseline
  survival is required. The printed `exp(tc)` should be `exp((t-1)c)`; its
  log derivative and thus the final proposition remain unchanged.
- **Theorem 3:** the weighted hazard is minus the derivative of log weighted
  survivor mass. Integrate its assumed affine relation, use `omega_1=1`, and
  apply the same identification logic. This proves the weighted model result,
  not automatic collapsibility of conditional hazards.
- **Theorem 4:** E.4–E.6 connect conditional survival among baseline survivors
  with `phi(x'gamma_k)r_kt`. Evaluate at `x=0` to recover
  `r_kt=-log(E[1-Y_it|G=k,X=0,Y_i1=0])/phi(0)`, then invert the link and use
  support to identify `gamma`. The printed denominator `-phi(0)` has the
  wrong sign. Integrating A.9 and averaging over group-1 baseline survivors
  gives the corrected A.12–A.13 formulas above. No positive increment means
  the inverse-link division cannot identify `gamma`.

## Reported simulations and empirical evidence

The results in this section are transcribed or summarized from the paper.
No Monte Carlo experiment or author application was executed for this review.

### Appendix C: design and complete numerical results

Appendix C (pp.40–44) uses the untreated hazard shape

```text
f(t) = 1+sqrt(t/T)-0.5(t/T-0.5)^2
h_2^(0)(t)=f(t)/(T-1)
h_1^(0)(t)=[f(t)+c]/(T-1)
h_1(t)=[f(t)+c+beta 1{t>=tstar}]/(T-1).
```

Transitions over `(t,t+1]` have probability
`1-exp(-integral_t^(t+1) h_k(s) ds)`, evaluated numerically by the authors.
The actual additive untreated hazard gap is `c/(T-1)`. The instantaneous
change at `tstar` affects subsequent observed transitions, so `tstar+1`
is the first affected discrete outcome. All Table 1 parameters are:

| T | tstar | Initial treated event share | Initial control event share | c | beta |
|---:|---:|---:|---:|---:|---:|
| 20 | 11 | 0.4 | 0.2 | 0.5 | 1 |

The paper uses equal time weights, 1,000 datasets, 1,000 bootstrap draws,
and 95% inference. It labels sample size `n`; its appendix does not specify
allocation, while the pinned script uses the listed count **per group**.
Keep that distinction when designing a future replication. All Table 2 values
are reproduced below:

| Method | n | Absolute bias | Mean squared error | Uniform coverage | Average pointwise coverage | Pre-test rejection |
|---|---:|---:|---:|---:|---:|---:|
| Duration DiD | 100 | 0.00333 | 0.00176 | 0.962 | 0.957 | 0.058 |
| Duration DiD | 500 | 0.00111 | 0.00034 | 0.949 | 0.951 | 0.041 |
| Duration DiD | 1000 | 0.00024 | 0.00017 | 0.955 | 0.953 | 0.051 |
| Duration DiD | 5000 | 0.00008 | 0.00003 | 0.945 | 0.946 | 0.058 |
| Duration DiD | 10000 | 0.00010 | 0.00002 | 0.960 | 0.956 | 0.052 |
| Standard DiD | 100 | 0.070 | 0.008 | 0.680 | 0.739 | 0.038 |
| Standard DiD | 500 | 0.067 | 0.006 | 0.077 | 0.266 | 0.079 |
| Standard DiD | 1000 | 0.068 | 0.005 | 0.004 | 0.083 | 0.166 |
| Standard DiD | 5000 | 0.068 | 0.005 | 0.000 | 0.000 | 0.576 |
| Standard DiD | 10000 | 0.068 | 0.005 | 0.000 | 0.000 | 0.804 |

Uniform coverage means all post-period effects are covered within a simulated
dataset; pointwise coverage is averaged over post-dates. Duration DiD has
small bias, roughly nominal coverage and diagnostic size in its correctly
specified CD design. Standard DiD's persistent bias produces collapsing
coverage; its pretest can have little power at smaller sample sizes. These
results do not validate PH, covariate, staggered, censoring, clustered,
constrained, or boundary variants.

Figure C.1 repeats Figure 1.1's converging untreated CDFs and parallel average
hazards. C.2 shows estimated versus true Duration DiD effects with uniform
bands for example datasets at `n=500,5000,10000`; C.3 shows biased,
sign-reversed standard DiD estimates and narrowing wrong intervals. C.4
shows average hazards and nonrejecting CD diagnostic bands for those examples.
They are illustrations, not extra simulation estimates. The prose calls them
A.1–A.4, while the actual captions are C.1–C.4.

### Section 4 and Appendix D: Austrian unemployment insurance

The August 1, 1989 Austrian reform extended potential benefit duration from
30 weeks/210 days to 39 weeks/273 days for eligible unemployed people aged
40–49 with at least 312 employed weeks in the preceding ten years. Existing
eligible spells were included. The application restricts to people eligible
for this extension but not the simultaneous replacement-rate change.

With spell start `s_i` relative to reform, treated starts satisfy
`-210<s_i<=92`, controls `-575<s_i<=-273`. Figure 4.1 shows these 302-day
windows offset by 365 days. Initial counts are 5,311 treated and 5,390 control.
Analysis time is elapsed unemployment duration; comparison of calendar
cohorts at that duration is not the Appendix A.3 adoption design. To permit
one week of anticipation before day 210, the authors use `tstar=202`.
Earlier anticipation could bias their estimates.

The application balances spell-start day of year via Equation 3.8 and removes
40 of 291 dates with fewer than two untreated people, leaving 4,511 treated
and 5,364 controls. It uses equal weights on the last 50 pre-dates, 153–202,
and zero earlier weights. This does not change the initial origin of cumulative
average hazards. Adding 50 earlier fitting dates reportedly changes little.
The proposed unadjusted core does not automatically inherit this adjusted
application's identifying argument or its trimmed target population.

Figure 4.2 displays cumulative reemployment and weighted time-average hazards;
the control hazard increases sharply at day 210. Figure 4.3 shows the CD
counterfactual following control dynamics with the fitted shift, and the
imputed cumulative exit curve above factual treated exit. Its 95% pointwise
and uniform bands concern **counterfactual means**, which are different from
ATT bands because ATT also includes factual-treated uncertainty/covariance.
The application uses 10,000 whole-history bootstrap draws.

Figure 4.4a reports negative absorption effects after day 210, visually near
`-0.03` around days 240–260 and closer to `-0.01` by day 360; these are
approximate visual readings, not numerical table estimates. Effects attenuate
after treated expiry at day 273. The main-text conclusion on p.28 says both
bands include zero on days 203–209, without significant evidence of one-week
anticipation. Figure 4.4b's **60%** uniform diagnostic band contains zero and
the reported p-value is `0.643`. Coverage 60% is not a significance level of
60%, and nonrejection is not proof of identification. The p.27 sentence
mislabels the solid expiry line: its day 273 corresponds to treated expiry;
controls expire at day 210.

Appendix D (p.45) shows the PH analysis with the same weighted hazards.
Figure D.1 gives imputed hazards and counterfactual cumulative means; D.2(a)
gives negative effects with 95% pointwise and uniform bands. The legend of
D.2(b) specifies **60% uniform diagnostic intervals**, matching the main CD
diagnostic; the band contains zero. The curves are close to CD because
pre-treatment weighted hazards are close in this application. There is no new
numerical effect table or universal
specification selection rule; the two models need not agree in other settings.

## Reference implementation audit

The six pinned originals were inspected, not executed. No `stata`, `stata-mp`,
`stata-se`, MATLAB, or Octave executable was found on PATH; the required
`restud_data.csv` application data are absent. The cache has no LICENSE file,
and no licensing grant was found in its six files. Public availability is not
a copying permission. This review vendors no author code; an implementation
should derive independently from the equations unless reuse rights are settled.

| Evidence in pinned files | Consequence |
|---|---|
| MATLAB `durationDiD.m` begins again in concatenated `endfunction out = durationDiD(...)` at line 187; Stata `durationdid.ado` repeats `program define durationdid` at lines 6 and 616 and duplicates Mata definitions | Static structural defects prevent treating the snapshot as a certified executable oracle; no runtime error or repair was tested |
| MATLAB lines 45/49 and 234, Stata lines 339–353 and 947 | CD averages hazard gaps; PH averages usable ratios, not either printed/repaired WLS slope |
| Stata lines 339–346 sets baseline average hazards to zero and includes them with default `burnin=1`; MATLAB excludes the undefined baseline via `nanmean` | With two informative pre-increments and true gap `c`, Stata's default produces `2c/3`; proposed implementation excludes baseline before constructing moments |
| MATLAB `durationDiD.m` lines 110–124 and 294–308 | SDs and centered absolute deviations with post-only maxima support Appendix B's inference structure |
| Stata lines 495–498 replaces zero SD with `1e-100` | Do not copy this into valid inference; retain zero SE and joint unavailable inference |
| MATLAB lines 132/316; Stata lines 528/1110 use `abs(max(delta/SD))` | Required two-sided statistic is `max(abs(delta/SD))`; `[-4,1]` gives 1 versus 4, and the incorrect statistic is sign asymmetric |
| Reference diagnostics include baseline/reference coordinates; Stata quantiles use `ceil(level*B)` while MATLAB uses its quantile default | Exclude undefined/zero anchors; match time family, resamples, weights, and empirical-quantile convention before expecting numerical parity |
| MATLAB core line 31 uses `periods>=absorbed_time`; Mata survival uses `absorbed_time>t` at `durationdid.ado` lines 305–306 | First absorption at `t` means `Y_t=1`; `UIP_application_revision_R2.m` line 61 instead uses a strict comparison for its raw plot |
| README uses one row per individual and first absorption time, sentinel beyond horizon, and default horizon equal to maximum absorption time | A sentinel can accidentally become the horizon and force absorption; any future duration adapter must require an explicit observation horizon and distinguish censoring |
| MATLAB lines 159–186 and Stata around 218–270 construct discrete covariate weights, drop small control cells, and rescale | Deferred adjustment changes target under trimming; MATLAB's `absorbed_time>=1` at lines 159–160 and Stata's `at :>= 1` at line 218 both include people absorbed exactly at baseline, unlike the required `>1` |
| MATLAB first branch line 23 uses sampling weights but bootstrap line 76 resets to ones; Stata lines 457–467 resamples weights | No consistent survey-inference or sampling-weight parity claim is supported; global normalization can also make group survivor mass exceed one |
| `UIP_application_revision_R2.m` line 86 requests 14 outputs, while cached functions return one struct; cohort construction and draw counts differ from `UIP_application.do` | Cached scripts do not specify one identical verified replication; MATLAB uses treatment cutoff 202 in selection and 1,000 draws, Stata uses initial benefit expiry 210 and 10,000 draws |
| Both application scripts set `burn_in=152`; estimator indexing includes `152:202` | This fits 51 dates, unlike the paper's zero weights on the first 152 dates and equal weights on 153–202; window parity requires an explicit choice |
| `montecarlo_revision.m` has 1,000 simulations, 1,000 bootstrap draws, counts per group, and a 100,000-point integration grid | Static design evidence only; its converted `t_star=floor(50000/5000)=10` differs from Table 1's 11; the standard-DiD bootstrap stratifies groups while duration bootstrap pools them, and comparator diagnostic anchoring differs between original and draws |

README software defaults include CD, `burnin=1`, 1,000 draws, confidence
coverage `.95`, diagnostic coverage `.6`, and seed `12345`. They are not
paper mandates. The first estimator should use explicit library conventions
instead of inheriting the baseline-row problem, 60% diagnostic coverage, or
an implicit observation horizon. Original files remain unchanged, and no
table/figure reproduction or cross-language numerical certification is claimed.

## Implementation Notes

### Data structure, behavior, and result contract

The proposed initial input is a balanced long individual panel with column
arguments `unit`, `time`, `treatment`, and `outcome`; `treatment` is a fixed
0/1 group indicator, not a time-varying received-treatment variable. Require
unique `(unit,time)` cells, both groups, finite binary absorbing outcomes,
the same ordered equally spaced dates for every individual, and an explicitly
identified last pre-date. Normalize the common time origin/scale and use actual
elapsed durations consistently. Accept baseline absorption and survival through
the common administrative horizon. Reject late entry, missing cells, recurrent
spells, covariates, staggered adoption, survey weights, and arbitrary clustering
in this first interface rather than silently interpreting them.

Use `DurationDiD(BaseEstimator)` and `DurationDiDResults(BaseResults)` under
current 3.x conventions; no interface is implemented in this PR. The time path
is primary. A proposed headline scalar is the uniform average of explicitly
reported post-period absorption ATTs, with those same weights applied within
every bootstrap draw. This is an average of probability effects across dates,
not a paper-defined unique overall ATT or a mean-duration effect.

Expose specification, time origin, fitting dates/weights, group counts, fitted
coefficient, observed/counterfactual survival, ATT path, bootstrap covariance,
pointwise intervals, simultaneous bands, and separate diagnostic results/status.
Use canonical `att`, `se`, `t_stat`, `p_value`, `conf_int` semantics for headline
inference, plus `summary()`, `to_dict()`, and `to_dataframe()`. Store the actual
confidence level and replicate/failure counts. When adapting to
`EventStudyResults`, first treated observation has event time zero and the
last untreated date is reference `-1`; reference inference is undefined.
Hazard diagnostic contrasts must not masquerade as pre-treatment ATT rows.

Evaluate logs only where needed. Both groups need positive survival at baseline
and fitted pre-dates; control survival is needed at requested post-dates.
Factual treated post-survival may equal zero: its ATT uses survival directly,
so no treated post-log is required. Do not drop baseline-absorbed individuals.
Near-zero survival should be flagged as weak numerical support, without
inventing a paper-derived cutoff.

CD can extrapolate negative/decreasing cumulative hazards or survival outside
`[0,1]`, especially with a large negative gap and small control hazard
(Section 2.3). Report the offending periods and raw extrapolation as invalid,
with no valid causal inference for that family. Do not silently clip,
monotonize, or change the horizon and still label it the same estimator.
PH with a positive ratio preserves control monotonicity in population, but
sample boundaries and failed resamples remain possible. A shorter supported
horizon must be an explicit refit choice, not automatic outcome-dependent
selection.

### Relation to Existing diff-diff Estimators

`BaseEstimator` already supplies constructor-based parameter introspection and
transactional `set_params`; extend the existing estimator roster tests later.
`results_base.py` supplies results/event-study conventions; serialization and
adapter tests must cover the new type when it ships. Follow `_require_fit_alpha`
where applicable so results cannot silently change fitted inference levels.

Use `safe_inference()`/batch equivalents to compute the joint inference tuple
and enforce its NaN gate, then coherently replace valid p-values and intervals
with centered-bootstrap values. `apply_bootstrap_event_study_overrides` is an
existing pattern for preserving that gate. Do not combine normal p-values
with bootstrap intervals while claiming one inference procedure. Invalid or
zero SE means all downstream t/p/interval/band fields are undefined, not tiny
positive variance.

Existing `stratified_bootstrap_indices` is not the paper's pooled bootstrap;
wild/multiplier generators and influence-function chunking are not drop-in
replacements. `compute_bootstrap_pvalue` uses uncentered sign tails and
`compute_percentile_ci` raw percentile endpoints, so neither directly implements
Appendix B. Failure-rate reporting can be reused. Whole-history index resampling
with NumPy and a dedicated centered-bootstrap summary needs no new dependency.

The core estimates group means and moments, with no least-squares solver or
hazard likelihood. If a later affine extension solves regressions, use shared
`linalg.solve_ols` and its rank policy. Ordinary `DifferenceInDifferences` on
`Y`, or on individual `log(1-Y)`, is inappropriate; the latter is undefined
after absorption. A group-log-survival regression omitting the baseline or
CD time drift is also a different estimator. Existing synthetic, staggered,
GLM, and sensitivity methods have different identifying restrictions; matching
an event-study container does not establish HonestDiD compatibility.

### Computational considerations

For `n` people, `T` dates, and `B` draws, direct panel aggregation costs
`O(nT)` per estimate and `O(BnT)` overall; storing the panel plus replicate
effect paths costs `O(nT+BT)`. These are algorithmic estimates, not paper
benchmarks or measured performance. Draws can run independently, but deterministic
RNG streams, fixed eligible moments, and common simultaneous families must be
preserved. No Rust work is necessary for this documentation foundation.

### Tuning Parameters

All defaults in this table are recommendations for subsequent estimator design,
not parameters currently available in diff-diff.

| Parameter / decision | Type | Proposed default | Basis / restriction |
|---|---|---|---|
| Specification | CD or PH | CD | Explicit choice; author software default, no universal paper preference |
| Last pre-date | Observed date | Required | Intervention occurs afterward; never infer from absorption |
| Fitting dates | Pre-date set | All eligible dates after baseline | Fixed original-sample set; PH requires positive control increments |
| Time weights | Normalized vector | Equal on fitting set | 3.2 permits alternatives; no universal optimal rule |
| PH coefficient | Moment rule | Mean ratios | Theorem 1 and author code; distinguish the two WLS alternatives |
| Bootstrap draws | Integer | 1,000, at least 2 | Simulation/software precedent; application uses 10,000 |
| Confidence significance `alpha` | Float in `(0,1)` | `.05` | 95% effect and diagnostic inference; source application diagnostic differs |
| Seed | Integer or `None` | `None`, recorded when set | Library RNG policy; reproducibility examples must set it |
| Bootstrap SD / quantile | Conventions | `ddof=1` / inverse empirical CDF | Explicit finite-sample policy; source SD divisor unspecified |
| Diagnostic | Contrast family | Algorithm 2 fixed anchor | Interior pre-dates; unavailable if insufficient/invalid |
| Draw failures | Policy | Fixed draws, no retries; suppress affected family inference | Conservative handling outside paper's regular interior argument |
| Headline aggregate | Time weights | Uniform across reported post-dates | Proposed probability-effect average; same weights in every draw |

### Independent checks and future acceptance scenarios

The review executed an independent NumPy algebra script at
`.workflow/validation/algebra_checks.py`; its output is preserved in
`.workflow/validation/algebra-checks.txt`. It checked CD baseline/sign identities,
the PH reciprocal error and three finite-sample choices, max-absolute testing,
fixed-anchor normalization, whole-history resampling, and pointwise/uniform
critical ordering. This is neither estimator testing nor author-code execution.
The following self-contained core calculation preserves the consequential
checks in the published document:

```python
import numpy as np

t = np.arange(1, 6, dtype=float)
elapsed = t - 1
pre, post = np.array([1, 2]), np.array([3, 4])
d2 = np.array([0, .10, .30, .45, .60])
r1_base, r2_base = .50, .20  # unequal initial survival
r2 = r2_base + d2
r1_cd = r1_base + d2 + .04 * elapsed
c = np.mean((r1_cd[pre] - r1_base - d2[pre]) / elapsed[pre])
r0 = r1_base + (r2 - r2_base) + c * elapsed
assert np.allclose(r0, r1_cd)
assert np.max(np.abs(np.exp(-r0) - np.exp(-r1_cd))) < 1e-14
r1_factual = r1_cd.copy()
r1_factual[post] -= .03 * (t[post] - 3)
tau = np.exp(-r0[post]) - np.exp(-r1_factual[post])
assert np.all(tau < 0)

d1 = 2 * d2
ratio = np.mean(d1[pre] / d2[pre])
printed = np.dot(d1[pre], d2[pre]) / np.dot(d1[pre], d1[pre])
assert ratio == 2 and printed == .5
assert np.allclose(r1_base + ratio * (r2 - r2_base), r1_base + d1)
x, y, dt = np.array([.1, .3]), np.array([.2, .9]), np.array([1, 2])
choices = (np.mean(y / x), np.dot(x, y) / np.dot(x, x),
           np.dot(x / dt, y / dt) / np.dot(x / dt, x / dt))
assert np.allclose(choices, [2.5, 2.9, 2.6923076923076925])
z = np.array([-4., 1.])
assert np.max(np.abs(z)) == 4 and abs(np.max(z)) == 1
print(f"CD coefficient={c:.6f}; PH ratio={ratio:.6f}; printed={printed:.6f}")
print("CD post absorption effects:", np.round(tau, 8))
print("PH mean/increment-slope/average-slope:", np.round(choices, 6))
```

```text
CD coefficient=0.040000; PH ratio=2.000000; printed=0.500000
CD post absorption effects: [-0.01044616 -0.01754019]
PH mean/increment-slope/average-slope: [2.5      2.9      2.692308]
```

The later estimator PR should test these identities on admissible individual
panels and add the following behavioral scenarios, with requirements still
unchecked above:

- No treatment effect under both models, unequal initial survival, non-unit
  PH ratio, known exit-effect sign, and finite-sample PH choices that differ.
- Exact timing and denominators, baseline absorption, complete survival to
  horizon, zero treated post-survival, and invariance to consistent time rescaling.
- Invalid binary/absorbing histories, duplicate or missing cells, changing group
  labels, absent groups, insufficient pre-dates, PH support failure, and invalid
  counterfactual curves with the documented outcomes.
- Fixed resample indices preserving complete histories and duplicate draws;
  every nuisance recomputed; seed reproducibility; omitted-group, zero-survival,
  weak-denominator, and failed-family inference behavior.
- Fixed bootstrap draws verifying SD/quantiles, pointwise versus simultaneous
  widths, max-absolute sign symmetry, frozen family endpoints, joint-NaN
  inference, and scalar aggregation using the same weights in every draw.
- Fixed-anchor diagnostics distinct from the main-text moving-window contrast;
  deterministic reference excluded; two pre-dates allow estimation but not a
  nontrivial diagnostic; PH point-estimate support can exceed test support.
- Parameter round trips, transactional invalid updates, results serialization,
  confidence-level consistency, and event-study reference/diagnostic separation.

## Gaps and Uncertainties

The register below separates reproducible internal corrections from unresolved
source choices. Corrections are review derivations supported by specified source
identities, not author-confirmed errata. Literal copied formulas are not a safe
implementation specification.

| Location | Printed/source issue | Resolution and evidence | Confidence / consequence |
|---|---|---|---|
| 2.9–2.10, p.11 | Successive-increment domains include `t=2`, requiring unobserved `R_k0` | Use `t>=3` for successive one-period increments; long-difference identification starts at `t=2` | High; do not invent a pre-baseline observation |
| 2.12, p.12 | Factual symbols used for full-horizon PH restriction | Interpret counterfactuals beyond pre-periods, by 2.4/2.15–2.16 | High; observed post-treatment equality is not required |
| Theorem 2, p.16 | Labels the identified object an increment but displays the level including baseline | Retain `R_11` in the level, consistently with Theorem 1 and proof | High; avoid adding or removing the baseline twice |
| 3.5, p.21 | Squared-treated-increment reading of denominator; missing baseline hat | That reading returns reciprocal under exact PH; use Theorem 1 direction, explicit mean-ratio core choice | High incompatibility; finite-sample choice is policy, not uniquely repaired by typography |
| 3.5 versus 3.6, p.21 | Claimed special case uses cumulative versus average-hazard weights | Elapsed-time-squared reweighting needed; table/check above distinguishes all three choices | High; cannot assert identical finite-sample estimators |
| 3.6, p.21 | Objective begins at baseline `t=1` | Start strictly after baseline, as 2.21 and 3.2 | High; avoids undefined `0/0` |
| p.23 versus Algorithm 2, p.40 | Moving final window versus fixed-baseline anchor | Both valid null contrasts under lag definition/Remark 2; choose Algorithm 2 explicitly | High; documented convention difference, not a source arithmetic failure |
| Algorithm 1, pp.38–39 | Some loops/maxima include `tstar`; SEs only afterward | Post-only family supported by steps 6–7 and Table 2 | High intended endpoint; exclude reference studentization |
| Algorithm 2, p.40 | `tau_hat` band center; maximum labeled pointwise | Center on `delta_hat`, call simultaneous, by steps 1/4/7/9 | High; otherwise tests the wrong object |
| A.3, p.34 | PH population baseline `R_kt` | `R_k1` by A.6 and same-page plug-in | High; avoids factual post-treatment contamination |
| Theorem 4, p.37 | Baseline intercept `(1-t)beta_1` | `+(t-1)beta_1` by A.12 and integrated A.9 | High; sign-consequential deferred correction |
| A.13, p.37 | Final expectation omits treated-group conditioning | Add `G=1`, as proof p.50 and sample group indicator | High; preserves target distribution |
| A.4.1, p.38 | Current-survivor ATT weight and sum index | Use `1-Y_i1`, sum over `i`, by A.13 and proof | High; current risk-set selection changes estimand |
| Theorem 2 proof, p.47 | Stray `R_1t^(0)=R_1t` after control argument | Equality at baseline only; subsequent imputation uses `R_11` | High; does not assume zero post-treatment effects |
| Proposition 1 proof, p.48 | `exp(tc)` with origin 1 | `exp((t-1)c)`; derivative still yields `c` | High normalization correction; proposition unchanged |
| Theorem 4 proof, p.50 | Denominator `-phi(0)` | Positive `phi(0)` by E.6 | High; printed version gives negative integrated hazard |
| Theorem 4, pp.36/50 | Current versus baseline survivor support; untreated intersection versus union | Proof uses baseline survivors and union; retain mismatch for extension review | High algebraic interpretation; stronger implementation conditions must be stated |
| Theorem 4 proof, p.50 | Inverse link divides by possibly zero baseline increment | Require informative positive increment to identify each fitted `gamma` | High nonidentification at zero; missing nondegeneracy detail |
| Figures/prose, pp.13,27,41–43 | Figure 1.1 ATT text says solid red instead of treated blue; treated expiry mislabeled; A.x references for C.x figures | Use treated factual minus treated counterfactual, actual chronology, and captions | High; descriptive correction only |

The core formulas, finite-sample PH recommendation, and diagnostic convention
are sufficiently specified to support the next estimator planning session;
there is no remaining unresolved core algebraic blocker under those explicit
choices. Source uncertainty remains material for the deferred semiparametric
extension, binding-constraint inference, censoring/repeated-cross-section
resampling, and arbitrary dependence. Reference-code licensing, executability,
application data, and numerical reproduction remain unverified limitations.
The bibliography was covered, but cited outside papers were not independently
re-reviewed: statements about Cox, Kaplan–Meier, GMM, or other authors above
are limited to this paper's use of them and the stated mathematical deductions.
