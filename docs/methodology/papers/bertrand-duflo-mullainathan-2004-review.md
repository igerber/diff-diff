# Paper Review: How Much Should We Trust Differences-in-Differences Estimates?

**Authors:** Marianne Bertrand, Esther Duflo, Sendhil Mullainathan
**Citation:** Bertrand, M., Duflo, E., & Mullainathan, S. (2004). How Much Should We Trust Differences-in-Differences Estimates? *The Quarterly Journal of Economics*, 119(1), 249-275. https://doi.org/10.1162/003355304772839588
**PDF reviewed:** QJE published version (DOI: [10.1162/003355304772839588](https://doi.org/10.1162/003355304772839588)), 28 PDF pages = journal pp. 249-275 (content pp. 249-274 incl. the full Conclusion; References pp. 274-275). Local PDFs are gitignored under `/papers/` (`papers/119-1-249.pdf`); the journal/DOI version is the authoritative source.
**Review date:** 2026-06-26

---

> **Scope of this paper (important).** BDM (2004) is an *inference* paper, not an
> estimator. It explicitly "assume[s] away biases in estimating the intervention's effect
> and instead focus[es] on issues relating to the *standard error* of the estimate"
> (p. 250). Its two enduring contributions to a DiD library are (a) the **placebo-law /
> randomization-inference diagnostic** that grounds diff-diff's `PlaceboTests` surface, and
> (b) the demonstration that **serial correlation** makes conventional DD standard errors
> grossly understate sampling variability — motivating serial-correlation-robust inference
> over the conventional OLS/HC1 default: unit-level cluster-robust SEs and block/percentile-t
> bootstrap when the number of groups is large, and time-series aggregation for few groups.

---

## Methodology Registry Entry

*Formatted to match `docs/methodology/REGISTRY.md`. This is a **proposed** entry — it is not yet integrated. Folding it into REGISTRY.md (replacing the current edge-case-only `## PlaceboTests` stub) and promoting the `PlaceboTests` row in `METHODOLOGY_REVIEW.md` from In Progress to Complete are the work of the separate source-validation pass tracked there; this file adds the paper review only.*

## PlaceboTests

**Primary source:** [Bertrand, M., Duflo, E., & Mullainathan, S. (2004). How Much Should We Trust Differences-in-Differences Estimates? *The Quarterly Journal of Economics*, 119(1), 249-275.](https://doi.org/10.1162/003355304772839588) Paper review on file: `docs/methodology/papers/bertrand-duflo-mullainathan-2004-review.md`.

**Scope:** BDM (2004) introduces the **placebo-law experiment** — randomly assign a fake
treatment date and a fake set of treated groups, estimate the DD regression, and ask how
often a (necessarily spurious) "effect" is significant. Because the laws are fictitious,
the true effect is zero, so a correctly-sized 5% test should reject ~5% of the time
(p. 251). The fraction of placebo runs that reject is the test's empirical size. Footnote 11
(p. 256) makes the link to **randomization inference** explicit: "if true laws were
randomly assigned, the distribution of the parameter estimates obtained using these placebo
laws could be used to form a randomization inference test of the significance of the DD
estimate [Rosenbaum 1996]." diff-diff's `PlaceboTests` operationalizes this idea as four
generic diagnostics (`fake_timing`, `fake_group`, `permutation`, `leave_one_out`).

**Key implementation requirements:**

*The placebo-law design (the empirical basis of the diagnostic):*
- **Two-step random assignment per replication (Section III, pp. 255-256):**
  1. Draw a fake treatment year from a **uniform distribution over 1985-1995** (footnote 9:
     bounded to guarantee pre- and post-intervention observations).
  2. Randomly designate **exactly half the groups (25 of 50 states)** as "affected."
  Then `I_st = 1` for units in an affected group after the fake date, else 0.
- **At least 200, typically 400 replications** per cell; the fraction with `|t| > 1.96` is
  the empirical rejection rate (Table II note).
- **Permutation / randomization-inference variant:** randomly reassign treatment over a
  fixed set of outcomes; the empirical distribution of placebo estimates is the reference
  distribution for a randomization test (footnote 11).
- **Serially-uncorrelated placebo (Table II, row 5):** to isolate serial correlation of the
  treatment indicator, set `I_st = 1` only in **ten randomly selected years (1979-1999)** for
  treatment-group states, 0 otherwise (instead of one permanent post-date switch); rejection
  collapses to **5%** (correct size), confirming the serial correlation of `I_st` is the
  driver of over-rejection.
- **Power injection:** to test power against a known alternative, replace the outcome by
  `outcome + I_st × δ` (e.g. `δ = 0.02` for a uniform 2% effect; footnote 16).

*Critical caveat on the reference distribution (footnote 12, p. 256):*
- "we are randomizing the treatment variable while keeping the set of outcomes fixed. In
  general, the distribution of the test statistic induced by such randomization is **not a
  standard normal distribution** and, therefore, the exact rejection rate we should expect
  is not known." **Implication for the diagnostic:** the placebo/permutation distribution
  *is* the correct reference; comparing a permutation estimate to normal critical values is
  not generally valid. This is why a permutation p-value should be read off the empirical
  distribution (proportion of placebo `|estimate|` at least as large as observed), not from
  a t/normal table.

*Estimator equation under test (equation (1), p. 250):*

    Y_ist = A_s + B_t + c·X_ist + β·I_st + ε_ist

where:
- `Y_ist` = outcome for individual `i` in group `s` at time `t`
- `A_s`, `B_t` = group (state) and time (year) fixed effects
- `X_ist` = individual covariates; `c` = their coefficients
- `I_st` = intervention dummy (1 for affected group post-date); `β` = the DD estimate (β̂ via OLS)
- `ε_ist` = error term

*Aggregated (group-time cell) analogue (footnote 14, p. 258):*

    Ȳ_st = α_s + γ_t + β·I_st + ε_st

(residualize the individual outcome on covariates, average to group-year cells `Ȳ_st`, then
run the two-way FE DD on the cell means).

*Edge cases:*
- NaN inference for undefined statistics (diff-diff defensive convention; not from BDM):
  - `permutation_test`: t_stat is NaN when permutation SE is zero (all permutations produce identical estimates).
  - `leave_one_out_test`: t_stat, p_value, CI are NaN when LOO SE is zero (all LOO effects identical).
  - **Note:** Defensive enhancement matching CallawaySantAnna NaN convention.

**Reference implementation(s):**
- No canonical R/Stata package implements BDM's *placebo battery* as a single command; the
  paper's own machinery is custom (block-bootstrap "codes... available upon request",
  footnote 21). BDM's two inference *corrections* that did become standard tooling are:
  - Stata `cluster` (arbitrary/state-level cluster-robust VCV, Section IV.E).
  - Stata `xtgls` (parametric AR(k) corrections, Section IV.A — shown *not* to work).

**Requirements checklist:**

*Directly grounded in BDM (2004):*
- [ ] Placebo-law size experiment — randomly assign a fake treatment date and a fake treated group, estimate the DD, and measure the empirical rejection rate (≈5% under the null when correctly sized) (Section III; `permutation` / `fake_group` operationalize the random-assignment exercise).
- [ ] Permutation/placebo p-value read from the empirical placebo distribution, not from normal/t critical values (footnote 12).
- [ ] Power can be probed by injecting a known additive effect `outcome + I_st × δ` (BDM's `δ = 0.02` exercise, p. 261).
- [ ] The inference lesson holds — serial-correlation-robust SEs (unit-level cluster / block bootstrap for large N; aggregation for small N) rather than OLS/HC1 (Section IV, Conclusion).

*Library extensions of the placebo idea (not prescribed by BDM):*
- [ ] `placebo_timing_test` against a *chosen* pre-treatment period, reframed as a pre-trends diagnostic (BDM draw the fake date at random for a size experiment, not to test a specific pre-period).
- [ ] `leave_one_out` single-unit sensitivity diagnostic.
- [ ] Permutation p-value floor at `1/(n_valid+1)` (finite-draw randomization-inference convention; not stated by BDM).

---

## Implementation Notes

### Data Structure Requirements
- Individual-level micro data pooled across groups (states) and time (years), as repeated
  cross-sections or a panel; or group-time cell means after residualizing on covariates
  (footnote 14).
- Required: an outcome, a group identifier, a time identifier, and a treatment indicator
  `I_st`. Covariates optional (partialled out before aggregation in the cell-mean variant).
- BDM's reference dataset: CPS MORG women aged 25-50, 1979-1999, ~900k observations (~540k
  with positive earnings), 50 × 21 = 1050 state-year cells, outcome `log(weekly earnings)`.

### Computational Considerations
- The placebo / permutation diagnostic re-fits the DD regression once per replication
  (200-400 fits). Cost scales linearly in the number of replications × cost of one DD fit.
- The block bootstrap (Section IV.B) re-fits per bootstrap draw (200 in text / 400 in
  Table V note — see Gaps) and is "not immediate to implement" (one block = one group's
  full time series).

### Tuning Parameters

| Parameter | Type | Default (BDM) | Selection Method |
|-----------|------|---------------|------------------|
| Replications per placebo cell | int | typically 400, ≥200 | More draws → tighter Monte Carlo estimate of size |
| Fake-date window | range | uniform 1985-1995 | Bounded to keep pre/post observations (footnote 9) |
| Fraction of groups treated | float | 0.5 (25 of 50) | Author choice; alternatives tried (footnote 10) |
| Power-test effect size `δ` | float | 0.02 (2%) | Set to the alternative of interest (footnote 16) |
| Block-bootstrap replications | int | 200 / 400 | See Gaps (text/table discrepancy) |
| State resampling probability | float | 1/50 (whole state vectors, w/ replacement) | Preserves within-group autocorrelation (p. 258) |

### Relation to Existing diff-diff Estimators
- **`PlaceboTests` (`diff_diff/diagnostics.py`)** is the direct descendant: `fake_timing`
  ↔ placebo-timing law, `fake_group` ↔ random fake-treated groups, `permutation` ↔ the
  randomization-inference test of footnote 11, plus `leave_one_out` (single-unit
  sensitivity). The module already cites BDM (2004) in its docstrings.
- **Serial-correlation inference** motivates diff-diff's **unit-level cluster-robust SE**
  (the modern form of BDM's "arbitrary variance-covariance" / state-level `cluster`
  correction, Section IV.E) and its **bootstrap** inference paths (BDM's block / percentile-t
  bootstrap, Section IV.B). BDM's headline warning — conventional DD SEs over-reject when
  the outcome and the treatment indicator are both serially correlated — is the reason
  unit-level clustering is the recommended default for panel DD.
- **Distinct from per-estimator placebo/LOO** already documented elsewhere in the registry:
  SyntheticDiD's donor-pool `leave_one_out()` (ADH 2015 §4) and in-time placebo, HAD
  pretests, and DCDH placebo are separate implementations on their own estimators; the
  BDM-grounded `PlaceboTests` is the generic 2×2-DD battery.

---

## Detailed Findings

### 1. The diagnosed problem: serial correlation inflates DD significance

Three reinforcing features of the DD setting make serial correlation severe (p. 251):
1. DD usually uses **long time series** (BDM's survey of 92 papers finds an average of
   16.5 periods, median 11; Table I, p. 253).
2. Common DD outcomes (employment, wages) are **highly positively serially correlated**.
3. The treatment indicator `I_st` is itself **highly serially correlated** (it changes
   rarely within a group).

OLS assumes a diagonal error VCV; clustering on the group-year cell fixes only the
*within-cell* (Moulton [1990]) correlation and leaves the *across-year within-group* serial
correlation untouched — which is why clustering at the state-year level still over-rejects
(p. 257-258).

### 2. Over-rejection magnitudes (Table II, p. 257)

Rejection rate of the (true) null of no effect at the 5% level over placebo laws; correct
value is 0.05.

*Panel A — CPS data:*

| Row | Specification | Reject (no effect) | Reject (2% effect) |
|-----|---------------|--------------------|--------------------|
| 1 | CPS micro, log wage, OLS | **.675** | .855 |
| 2 | + cluster at state-year | **.44** | .74 |
| 3 | Aggregated (cell means), OLS | .435 | .72 |
| 4 | Sampling states w/ replacement (prob 1/50) | .49 | .663 |
| 5 | **Serially-uncorrelated placebo laws** | **.05** | .988 |
| 6 | Employment (ρ̂₁=.470) | .46 | .88 |
| 7 | Hours worked (ρ̂₁=.151) | .265 | .280 |
| 8 | Changes in log wage (ρ̂₁=-.046) | **0** | .978 |

*Panel B — AR(1) Monte Carlo (rejection rises monotonically with ρ):*

| ρ | -.4 | 0 | .2 | .4 | .6 | .8 |
|---|-----|---|----|----|----|----|
| Reject (no effect) | .008 | .053 | .123 | .19 | .333 | .373 |

Takeaways: (a) uncorrected DD rejects a false null **67.5%** of the time (abstract headline
"up to 45 percent" refers to the aggregated/cluster figures); (b) over-rejection tracks the
autocorrelation of the outcome (row 8 with negative autocorrelation *under*-rejects, 0%);
(c) the serially-uncorrelated placebo (row 5) restores correct 5% size, pinpointing the
serial correlation of `I_st` as the mechanism. Magnitude of spurious effects: mean
`|effect|` ≈ .02; ~60% in 1-2%, ~30% in 2-3%, ~10% above 3% (p. 260).

**Scaling with N and T (Table III):** over-rejection is essentially **invariant to the
number of groups N** but **falls slowly as the number of periods T shrinks** — still ~15%
at 7 years and ~8% (CPS) to 17% (AR(1), ρ=.8) at 5 years.

### 3. The four corrections evaluated (Section IV)

| # | Correction | Section / Table | Verdict |
|---|-----------|-----------------|---------|
| 1 | Parametric AR(k) (Stata `xtgls`) | IV.A / Table IV | **Fails** — short-panel downward bias of ρ̂ (CPS ρ̂≈.4 vs true .51; AR(1) true .8 estimated .62) + AR misspecification leave rejection at 16-39%. Works only if true ρ is known and imposed. |
| 2 | Block (percentile-t) bootstrap | IV.B / Table V | **Works at large N only** — correctly sized at N≈50 (6.5% CPS, 5% AR(1)); degrades to 13% (N=20), 23% (N=10), 43.5% (N=6). Asymptotics in N. |
| 3 | Arbitrary / state-level cluster VCV (Stata `cluster`) | IV.E / Table VIII | **Works at large N, best power** — 6.3% at N=50; over-rejects at small N (8% at N=10, 11.5% at N=6). Near oracle power (74% vs 78%; 27.5% vs 32%). |
| 3′ | Empirical VCV | IV.D / Table VII | Similar to cluster but assumes cross-sectional homoskedasticity; 5.5% at N=50, 15.3% at N=6. |
| 4 | Time-series aggregation (collapse to 2 periods) | IV.C / Table VI | **Best small-N size, lowest power** — 5.3% at N=50 and N=10; residual-aggregation variant handles staggered timing (~9% at N=10); needs the Donald-Lang [2001] small-sample t adjustment; power as low as 6.5% at N=10. |

**Block-bootstrap percentile-t procedure (Section IV.B, p. 265):**
1. Compute the observed absolute t-statistic `t = |β̂ / SE(β̂)|`.
2. Resample **with replacement 50 whole-group blocks** `(Ȳ_s, V_s)` (state time-series +
   its design rows), preserving within-group autocorrelation.
3. Refit OLS → `β̂_r`; form the **recentered** statistic `t_r = |(β̂_r − β̂) / SE(β̂_r)|`.
4. Reject `β = 0` at 95% if **95% of the `t_r` are smaller than `t`** (compare `t` to the
   95th percentile of the bootstrap `t_r` distribution).

**Arbitrary / cluster VCV (Section IV.E, p. 271):** a White-type sandwich clustering on
**entire groups** (states), not group-year cells (footnote 24):

    W = (V'V)^(-1) ( Σ_{j=1}^{N} u'_j u_j ) (V'V)^(-1),   with   u_j = Σ_{t=1}^{T} e_{jt} v_{jt}

where `V` is the matrix of independent variables (state dummies, year dummies, treatment),
`e_{jt}` the residual, and `v_{jt}` the row of independent variables for group `j` at time
`t`. Consistent for fixed T as N → ∞ [White 1984; Arellano 1987; Kezdi 2002]; "analogous to
applying the Newey-West correction in the panel context where we allow for all lags"
(footnote 23). VCV is rank `TN − N` (footnote 25).

### 4. Practical recommendations (Section IV.F / Conclusion)

- **Large number of groups (≈50):** arbitrary/state-level cluster-robust SE — good size and
  near-oracle power. Block bootstrap is a reliable alternative.
- **Small number of groups:** collapse the time series (aggregate to before/after) with the
  Donald-Lang [2001] t-adjustment — correct size at the cost of low power. Use residual
  aggregation when laws are staggered.
- **Do not** rely on parametric AR(k) corrections; misspecification yields inconsistent SEs.
- **Core warning:** "conventional DD standard errors may grossly understate the standard
  deviation of the estimated treatment effects, leading to serious overestimation of
  t-statistics... too many false rejections of the null hypothesis of no effect have taken
  place" (Conclusion, p. 273).
- **Make robust inference standard practice (Conclusion, p. 274):** BDM urge practitioners
  to "more carefully examine residuals as well as perform simple tests of serial
  correlation," and — because serial-correlation-robust standard errors are "relatively easy
  to implement in most cases" — argue this "should become standard practice in applied work."
  They flag **GLS** and **GMM estimation of dynamic panel data models** as promising
  directions for inference that is more efficient under serial correlation (consistent with
  footnote 19's deferral of IV/GMM).

---

## Gaps and Uncertainties

- **Bootstrap-replications discrepancy.** The text (p. 265) says "a large number **(200)**"
  of bootstrap samples; the Table V note says "The bootstraps involve **400** repetitions."
  Both appear in the paper; treat 200 as a documented floor and 400 as the typical value.
- **Footnote 12 reference-distribution caveat is load-bearing for the permutation test.**
  Because treatment is randomized over fixed outcomes, the induced statistic is *not*
  standard normal, so the *correct* p-value comes from the empirical placebo distribution,
  not a t/normal table. Any implementation that derives a permutation p-value from normal
  critical values would misstate significance — a property the source-validation pass should
  confirm for `permutation_test` when promoting `PlaceboTests`.
- **No closed-form small-sample correction is proposed.** Wild cluster bootstrap, CR2/CR3,
  and Satterthwaite dof all post-date this paper; BDM endorses only the Donald-Lang (2001)
  threshold adjustment for the aggregation route. Modern small-N fixes are out of scope here.
- **Transcription note (p. 271).** The paper describes `v_{jt}` as "a row vector of
  dependent variables (including the constant)"; in context (`V` is the *independent*-variable
  matrix) this is almost certainly a typo for *independent* variables. Preserved verbatim.
- **No numbered theorems/propositions.** Contributions are empirical/Monte-Carlo (Tables
  I-VIII) plus equation (1); cited asymptotic results (Nickell 1981, Kezdi 2002, Kiefer
  1980, Donald-Lang 2001) are invoked, not derived.
- **Coverage:** the full article text (pp. 249-274), including the complete Conclusion
  (Section V), is reflected above. Only the bibliography (pp. 274-275) is not summarized.
