# Paper Review: Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends

**Authors:** Jonathan Roth
**Citation:** Roth, J. (2022). Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends. *American Economic Review: Insights*, 4(3), 305-322.
**DOI:** https://doi.org/10.1257/aeri.20210236
**Source reviewed:** AER:I 4(3), 305-322 (18 pages, content pages 1-15). PDF was reviewed externally and is not committed to the repository (the `/papers/` working directory is gitignored). Reproduce by downloading the published article via the DOI above or from the author's page at https://www.jonathandroth.com/.
**Review date:** 2026-05-16

---

## Methodology Registry Entry

**Status: proposed replacement text for a future REGISTRY update; this file is a non-authoritative source audit.** The current `## PreTrendsPower` entry in `docs/methodology/REGISTRY.md` is a populated block framed primarily around a joint-Wald pre-trends test; it remains the **sole authoritative methodology contract** until the follow-up audit PR for `compute_pretrends_power` (the helper function), `PreTrendsPower` (the estimator class), and `PreTrendsPowerResults` (the results container) in `diff_diff/pretrends.py` lands and revises it. The follow-up audit will assess which proposed parameters and capabilities below are already in the shipped surfaces. Current signatures (for reference): the helper `compute_pretrends_power(results, M, alpha, target_power, violation_type, pre_periods)` exposes `alpha`, `target_power`, `violation_type`, `pre_periods` (plus the optional violation magnitude `M`); the class `PreTrendsPower(alpha, power, violation_type, violation_weights)` exposes `alpha`, `power`, `violation_type`, `violation_weights`. **Helper/class API gap observed today:** the helper does NOT accept or forward `violation_weights`, so calling `compute_pretrends_power(..., violation_type="custom")` cannot supply a custom weight vector — the `"custom"` path is class-only today. The audit will reconcile these two surfaces with each other and against this proposed contract.

*Formatted to match docs/methodology/REGISTRY.md structure. Heading levels and labels align with existing entries — once the follow-up audit is ready, the `## PreTrendsPower` section below can replace the existing registry entry. The registry-candidate text ends just before `## Implementation Notes`; everything below that boundary is **audit notes / implementation ideas** and is NOT part of the proposed registry replacement (it includes tentative heuristics, provisional R-package surface claims, and library design notes that should NOT be copied into REGISTRY.md as normative requirements).*

## PreTrendsPower

**Primary source:** [Roth, J. (2022). Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends. *American Economic Review: Insights*, 4(3), 305-322.](https://doi.org/10.1257/aeri.20210236)

**Key implementation requirements:**

*Assumption checks / warnings:*
- Input: event-study coefficient vector beta_hat = (beta_hat_pre, beta_hat_post)' that is asymptotically normal under the underlying estimator (Equation 1; Remark 1 lists TWFE, GMM, Freyaldenhoven-Hansen-Shapiro, regression-adjustment/IPW/DR DiD per Sant'Anna-Zhao, Callaway-Sant'Anna, Sun-Abraham)
- Input: estimated variance-covariance matrix Sigma_hat in R^{(K+M) x (K+M)} where K = # pre-period coefficients, M = # post-period coefficients
- **Block decomposition convention (per Roth, Section II.A-B)**: throughout this entry, the variance partition uses Roth's *post-first* ordering for the proofs, i.e., Var[(beta_hat_post, beta_hat_pre)'] = [[Sigma_11, Sigma_12], [Sigma_21, Sigma_22]] where Sigma_11 = Var[beta_hat_post] in R^{M x M} (the post-treatment block), Sigma_22 = Var[beta_hat_pre] in R^{K x K} (the pre-treatment block), Sigma_12 = Cov(beta_hat_post, beta_hat_pre) in R^{M x K}, Sigma_21 = Sigma_12'. The stacked input vector beta_hat is (pre, post)' as stated above; the (post, pre) block ordering is internal to the propositions and matches Roth's paper notation. Implementations must use the post-treatment block Sigma_11 (not the full Sigma_hat) wherever they need Var[beta_hat_post].
- Pre-trend zero-anticipation assumption: tau_pre = 0 (Equation 2) — same identifying convention as Rambachan-Roth (2023)
- Warn if pretest has low power: e.g., if the slope at 80% power (gamma_{0.8}) produces a |bias| > |estimated treatment effect|, the pretest is uninformative for the magnitudes that matter
- Warn that pretest-conditioning distortions are NOT removed by larger samples — they persist as long as the pretest can fail with non-vanishing probability (footnote 12)

*Causal decomposition (Equation 2):*

    beta = (delta_pre, delta_post)' + (0, tau_post)'
              \--------------/         \---------/
                  delta = bias               tau = causal effect
                  from trends

where tau_pre = 0 by the no-anticipation assumption and delta is the bias from a difference in trends. The pretest acts on the random vector beta_hat_pre, whose mean beta_pre equals delta_pre under no anticipation (i.e., the population-mean identity beta_pre = delta_pre follows from Equation 2 with tau_pre = 0; the random draw beta_hat_pre is not itself equal to delta_pre).

*Acceptance region of the standard "no individually significant" (NIS) pretest:*

    B_NIS(Sigma) = { b in R^K : |b_t| <= z_{1-alpha/2} * sigma_t, for all t in {-K, ..., -1} }

where z_{1-alpha/2} is the (1 - alpha/2)-quantile of the standard normal (= 1.96 at Roth's running default alpha = 0.05). This corresponds to checking individual (1 - alpha) CIs of each pre-period coefficient. Section I.B of Roth (2022) reports that all 12 surveyed papers plot pointwise CIs that allow individual-significance inspection, 5 of 12 explicitly discuss individual significance, 1 of 12 reports a joint-significance test, and several rely on visual inspection without specifying a formal criterion; the NIS form is therefore the implicit common denominator across the surveyed papers rather than a literal 11-of-12 explicit-rule count.

Alternative acceptance regions:
- **Joint Wald (chi-squared)**: B_W(Sigma) = { b in R^K : b' Sigma_22^{-1} b <= chi^2_{1-alpha, K} }. **Note:** mentioned in the paper as a less common applied convention (1 of 12 surveyed papers, Section I.B). Propositions 1, 3, 4 apply to this B since it is convex; Roth does NOT separately tabulate power/bias/coverage for the Wald form.
- **Slope-of-best-fit-line t-test**: the paper's Table 1 reports the t-statistic for the slope as an observed property of surveyed papers, but **Note (deviation from paper):** Roth does NOT analyze a slope-t-statistic acceptance region as a pretest framework. Library support for this acceptance form is an extension beyond Roth (2022).
- **Custom user-supplied B(Sigma)**: any (measurable) acceptance set; Propositions 1 and 3 (conditional mean and variance) apply to any B. Proposition 4 (variance reduction / over-coverage under parallel trends) requires B to be **convex** — Roth's statement begins "Suppose that B(Sigma) is a convex set." Nonconvex custom pretests therefore lose Roth's variance-reduction / over-coverage guarantee even though the conditional mean/variance formulas still hold. Proposition 2 (sign of bias under monotone trend) requires the specific NIS form plus Assumption 1.

*Conditional bias after pretesting (Proposition 1):*

    E[beta_hat_post | beta_hat_pre in B(Sigma)]
        = tau_post + delta_post + Sigma_{12} Sigma_{22}^{-1} ( E[beta_hat_pre | beta_hat_pre in B(Sigma)] - beta_pre )

The third (pretest bias) term depends on:
- Sigma_{12} Sigma_{22}^{-1}: the regression coefficient of beta_hat_post on beta_hat_pre (akin to "leakage" from pre to post via the covariance)
- The distortion E[beta_hat_pre | beta_hat_pre in B(Sigma)] - beta_pre: how much pretest conditioning skews the pre-period means

*Sign-of-bias result under monotone trend (Proposition 2 + Assumption 1):*

    Assumption 1: Sigma has a common term sigma^2 on the diagonal and a common term rho > 0 off the diagonal, with sigma^2 > rho.

    If delta_pre < 0 elementwise and delta_post > 0 (upward pretrend), then:
        E[beta_hat_post | beta_hat_pre in B_NIS(Sigma)] > beta_post > tau_post

(Bias is worse after pretesting under monotone violations; symmetric statement for downward pretrend.)

*Variance after pretesting (Proposition 3):*

    Var[beta_hat_post | beta_hat_pre in B(Sigma)]
        = Var[beta_hat_post]
          + (Sigma_{12} Sigma_{22}^{-1}) (Var[beta_hat_pre | beta_hat_pre in B(Sigma)] - Var[beta_hat_pre]) (Sigma_{12} Sigma_{22}^{-1})'

*Convexity gives variance reduction (Proposition 4):*

    If B(Sigma) is a convex set, then Var[beta_hat_post | beta_hat_pre in B(Sigma)] <= Var[beta_hat_post].

Implication (Roth, Section II.C, paragraph after Proposition 4): under parallel trends (delta = 0) and a B(Sigma) symmetric about zero, conventional 95% CIs tend to OVER-cover conditional on passing the pretest (CIs are based on the unconditional variance, which is too large). When parallel trends is violated, conventional 95% CIs tend to UNDER-cover **if the bias is sufficiently large** (i.e., when the bias dominates the variance reduction). The under-coverage direction is therefore contingent on bias magnitude, not universal.

*Target parameter (Section I.C):*

    tau_* = l' tau_post, for some user-specified l in R^M

Defaults Roth uses:
- **Average post-treatment effect**: tau_bar = (1/M)(tau_1 + ... + tau_M), i.e., l = (1/M, ..., 1/M)' (main text emphasis)
- **First-period-after-treatment effect**: tau_1, i.e., l = (1, 0, ..., 0)' (online Appendix)
- **Custom**: any user-specified contrast l

*Plug-in estimator and CI (Section I.C):*

    tau_hat = l' beta_hat_post
    CI_{tau_*} = tau_hat +/- z_{1-alpha/2} * sigma_{tau_hat}, where sigma^2_{tau_hat} = l' Sigma_11 l

(Notes: Sigma_11 is the post-treatment covariance block per the convention above, not the full Sigma_hat. z_{1-alpha/2} is the (1 - alpha/2)-quantile of the standard normal = 1.96 at alpha = 0.05.)

*Power calculation against a linear violation (Section I.C "Power Calculations"):*

For a linear violation with slope gamma (so delta_t = gamma * t with relative time t),
the pretest "passes" probability is

    P( beta_hat_pre in B_NIS(Sigma) ) = P( |beta_hat_pre,t| <= z_{1-alpha/2} * sigma_t, for all t )

where beta_hat_pre ~ N(delta_pre, Sigma_22) with delta_pre,t = gamma * t. The library should
solve for gamma at target power 1 - p in {0.5, 0.8}:

    gamma_{1 - p} = inf{ gamma : P( beta_hat_pre NOT in B_NIS(Sigma) | delta = gamma * t ) >= 1 - p }

These are Roth's gamma_{0.5} and gamma_{0.8} ("the slopes against which pretests have 50%
or 80% power"). Roth uses 80% as a benchmark following Cohen (1988); 50% is supplementary.

*Bias and size calculations against a given gamma (Section I.C):*

- **Unconditional bias**: E[tau_hat - tau_*] = l' delta_post (with delta_t = gamma * t for relevant t)
- **Conditional bias**: E[tau_hat - tau_* | beta_hat_pre in B_NIS(Sigma)] (computed via Proposition 1)
- **Unconditional null rejection**: P(tau_* in CI_{tau_*}^c) under linear trend
- **Conditional null rejection**: P(tau_* in CI_{tau_*}^c | beta_hat_pre in B_NIS(Sigma))

*Computational shortcut (footnote 8):*

Under joint normality, these probabilities and conditional moments can be calculated
ANALYTICALLY using results from Cartinhour (1990) and Manjunath & Wilhelm (2012) — Roth
implements via the R package `tmvtnorm`. Roth verifies simulations yield similar results.
The paper-derived requirement is to compute the correct conditional moments and
probabilities; the specific backend (analytical via `tmvtnorm`-equivalent, an
independent port, a Monte Carlo simulator, etc.) is a library implementation choice —
see "Library design choices" above for the proposed `method` / `n_sim` knobs.

*Standard errors (Section II.C; footnote 7 equivariance):*
- Power calculations are EXACT (no sampling variability — gamma is computed against a hypothesized population trend, not estimated)
- Uncertainty comes entirely from the user-supplied Sigma
- Roth's bias and coverage results have NO dependence on the value of tau_post (footnote 7: the distribution of beta_hat_post conditional on beta_hat_pre passing the pretest is equivariant w.r.t. tau_post)

*Edge cases (paper-stated):*
- **Linear vs nonlinear violations**: paper formally analyzes linear trends; Caveats (Section I.D) note results extend to monotone nonlinear violations under homoskedasticity (Proposition 2); arbitrarily nonlinear violations addressed heuristically — bias is worse for exponentially-growing trends, better for log/shallow trends as pre-periods grow
- **Adding more pretreatment periods**: helps power for linear/log trends; for trends concentrated near treatment (e.g., COVID-19-like shocks), Section I.D notes additional distant pre-periods may not help / may be uninformative — the paper does not assert that they actively *hurt*.
- **K = 1 (single pre-period)**: explicit closed-form intuition via univariate truncated normal in proof of Proposition 2: E[beta_hat_pre | beta_hat_pre in B_NIS] - beta_pre proportional to phi(-z_{1-alpha/2} - beta_pre/sigma) - phi(z_{1-alpha/2} - beta_pre/sigma) (= phi(-1.96 - ...) - phi(1.96 - ...) at the paper's default alpha = 0.05)
- **Symmetric two-sided pretests under parallel trends**: beta_hat_post remains UNBIASED for tau_post (E[beta_hat_pre | beta_hat_pre in B] = 0 if B is symmetric and beta_pre = 0)
- **Heteroskedastic Sigma (off-diagonal not constant)**: Proposition 2 requires Assumption 1; under arbitrary Sigma, sign of pretest-bias term is ambiguous (worked out in Proposition 1's general form)
- **Publication-bias trade-off (Equation 4, Section II.D)**: pretest-as-screen can REDUCE or INCREASE published bias depending on the Bayes-factor of design type vs the bias-given-publication ratio; the net effect is ambiguous (Equation 4). The paper says underpowered pretests are "least effective and potentially harmful" — i.e., they are the worst-case regime, not unambiguously harmful in every parameterization.

*Algorithm (no numbered algorithm in paper; implementation distilled from Section I.C):*

1. Take user-supplied (beta_hat, Sigma, K, M) and a target estimand l in R^M (default: l = uniform 1/M)
2. Compute B_NIS(Sigma) acceptance region using diagonal sigma_t = sqrt(Sigma_22[t, t]) for t in pre periods (Sigma_22 = Var[beta_hat_pre] per the block convention above)
3. **Power**: solve gamma_{1-p} = root of P(reject pretest | gamma) = 1 - p
   - For each candidate gamma, compute P(beta_hat_pre in B_NIS) under beta_hat_pre ~ N(gamma * t_pre, Sigma_22) using `tmvtnorm`-style multivariate normal CDF; or via simulation
4. **Bias**: for gamma in {0, gamma_{0.5}, gamma_{0.8}, user-custom}:
   - Compute unconditional bias = l' delta_post where delta_post,m = gamma * m
   - Compute conditional bias via Proposition 1: requires E[beta_hat_pre | beta_hat_pre in B_NIS] from truncated MVN
5. **Coverage**: for the same gamma values, compute unconditional and conditional null rejection probabilities P(tau_* not in CI):
   - Unconditional: P(|tau_hat - tau_*|/sigma_{tau_hat} > z_{1-alpha/2}) under beta_hat ~ N(beta, Sigma)
   - Conditional: P(|tau_hat - tau_*|/sigma_{tau_hat} > z_{1-alpha/2} | beta_hat_pre in B_NIS) — joint truncated MVN
6. Return a structured summary (Roth's Table 2/Table 3 layout)

**Reference implementation(s):**
- R: [`pretrends`](https://github.com/jonathandroth/pretrends) (Jonathan Roth's own package) and the accompanying Shiny app
- R dependency: [`tmvtnorm`](https://cran.r-project.org/package=tmvtnorm) (Manjunath & Wilhelm 2012) for truncated multivariate normal moments and CDF

### Paper-derived requirements

*Required to remain faithful to Roth (2022). These are the mathematical quantities and conditioning formulas the paper specifies — they do not constrain the numerical backend or the API surface. The follow-up audit PR must verify the library can produce every item below.*

- [ ] NIS acceptance region B_NIS(Sigma) with critical value z_{1-alpha/2} (paper-analyzed in Section I.B + Section II, used for every empirical exercise)
- [ ] Conditional-mean formula (Proposition 1) and conditional-variance formula (Proposition 3) for any measurable B(Sigma)
- [ ] Variance-reduction guarantee (Proposition 4) gated on convex B(Sigma) only
- [ ] Sign-of-bias result under monotone trend (Proposition 2) gated on Assumption 1 (homoskedastic-equicorrelated Sigma)
- [ ] Power calculation against a linear violation with slope gamma — solve for gamma_p at user-specified target power 1 - p (Roth uses gamma_{0.5} and gamma_{0.8})
- [ ] Plug-in estimator tau_hat = l' beta_hat_post and CI tau_hat +/- z_{1-alpha/2} * sqrt(l' Sigma_11 l) for any linear contrast l in R^M
- [ ] Unconditional and conditional bias for the linear contrast
- [ ] Unconditional and conditional null rejection / coverage for the linear contrast

### Library design choices (paper-supported alternatives and extensions beyond Roth 2022)

*These items are diff-diff or R-package conventions, NOT required by the paper. They include both (i) acceptance regions and computations that the paper supports as alternatives without requiring or tabulating them, and (ii) genuine extensions beyond the paper's analysis. The library may keep, drop, or extend each item via the follow-up audit — preserving these items is a library design call, not a methodology requirement.*

- **Joint Wald acceptance region** B_W(Sigma) — paper-supported alternative (convex, so Propositions 1+3+4 all apply), but Roth's empirical exercise uses NIS only and the paper does not separately tabulate Wald-based power/bias/coverage. Library support is a paper-supported alternative, not a Roth-required item.
- **Analytical computational backend** (`tmvtnorm` / Cartinhour-1990 / Manjunath-Wilhelm-2012) — Roth uses this analytical path in the paper, but the requirement is to produce the correct conditional moments and probabilities; any equivalent backend (`tmvtnorm`, a from-scratch port, Monte Carlo simulation, GHK simulator, etc.) is acceptable. The choice of backend is a library implementation decision, not a methodology requirement.
- **Simulation fallback path alongside the analytical computation** — Roth's footnote 8 reports simulation verification yields similar results, but neither the paper nor the R `pretrends` package requires a dual-path implementation. The simulation path is a library robustness choice for cases where the analytical computation is numerically unstable.
- **`method` and `n_sim` API parameters** — proposed knobs to select between analytical and simulation; library design choice.
- **`pretest_form` and `acceptance_region` API surface** — Roth's propositions apply to any (measurable) B(Sigma), so exposing the choice via a typed enum + custom-callable interface is an engineering choice. The enum values mix paper-analyzed forms ("individual" / NIS), paper-supported alternatives ("joint_wald", "custom"), and a non-paper extension ("slope" — Roth tabulates the slope t-stat in Table 1 as an observed property of surveyed papers but does not analyze it as an acceptance region).
- **Non-linear violation parameterizations** ("constant", "last_period", "custom") — Roth Section III endorses power analyses against hypothesized nonlinear trends via the `pretrends` package (applying the same Propositions 1+3, with Proposition 4 conditioned on convex B). The specific named shapes are R-package API conventions, not separately analyzed in the published paper.
- **Figure-1-style plotting interface** — the underlying numerical content (bias and CI by `gamma_p`) is paper-derived; the plotting layout is a library presentation choice.
- **HonestDiD result-object composition / cross-estimator integration** — Roth (2022) is methodology-agnostic about how `(beta_hat, Sigma_hat)` is produced; composition with `HonestDiD`, `CallawaySantAnna`, `SunAbraham`, etc. is a diff-diff design choice.

---

<!-- ============================================================
     END of registry-candidate block. Everything BELOW this
     marker is audit notes / implementation ideas and is NOT
     part of the proposed REGISTRY.md replacement text.
     ============================================================ -->

## Implementation Notes (audit notes — NOT registry-candidate)

### Data Structure Requirements
- **Input**: beta_hat in R^{K+M} (concatenated pre + post event-study coefficients), Sigma_hat in R^{(K+M) x (K+M)} (variance-covariance matrix), integer K (# pre-period coefficients), integer M (# post-period coefficients)
- **Optional input**: linear contrast l in R^M (defaults to uniform 1/M for average post-treatment effect, or e_1 for first-period-only)
- **Optional input**: significance level alpha (default 0.05; critical value z_{1-alpha/2}, equal to 1.96 at the default)
- **Optional input**: target power levels (default {0.5, 0.8} per Roth)
- The pre-period coefficients are typically indexed by relative time t in {-K, -K+1, ..., -1}, with t = 0 omitted as the reference period
- Compatible with the result classes of: MultiPeriodDiD (event study), CallawaySantAnna (staggered), SunAbraham (interaction-weighted), Freyaldenhoven-Hansen-Shapiro (covariate-based)

### Computational Considerations
- **Truncated MVN moments and probabilities**: `scipy.stats.multivariate_normal.cdf` covers MVN box probabilities `P(β̂_pre ∈ B_NIS(Σ))`, but SciPy lacks a `tmvtnorm`-equivalent API for the truncated-MVN moments (`E[β̂_pre | β̂_pre ∈ B(Σ)]` and `Var[β̂_pre | β̂_pre ∈ B(Σ)]`) that Proposition 1's pretest-bias term and Proposition 3's conditional-variance term require. Library options for those moments are (a) port `tmvtnorm` (Manjunath-Wilhelm closed-form for orthant moments + Cartinhour 1990 for the rectangular box), (b) Monte Carlo simulation with rejection sampling. Recommend implementing both paths and validating equivalence at alpha-tol = 1e-3 for small K.
- **Cost**: dominated by the multivariate normal box probability evaluations. As a *tentative heuristic* (not benchmarked in this review and not specified by the paper), analytical methods are typically fast for small K (e.g., K <= 5) and simulation may become preferable for larger K (e.g., K > 10); the follow-up audit should either benchmark these cutoffs locally or replace them with empirically-derived thresholds.
- **Root-finding for gamma_p**: P(reject pretest | gamma) is monotone in |gamma|. Under the normal model power approaches 1 only asymptotically, so there is no finite gamma_max at which power equals 1 exactly. Use a doubling expansion (start with a univariate-derived gamma_high; double until P(reject pretest | gamma_high) >= target_power + tolerance), then bisect over [0, gamma_high] to find gamma_p.
- **Memoization**: power and bias share intermediate quantities (truncated MVN moments); cache by gamma.

### Tuning Parameters

**Note:** The parameters below span both paper-derived requirements (where the paper specifies a fixed default or a free parameter that affects the math) and proposed library extensions (engineering choices for the API surface). The `Source` column makes this distinction explicit. The follow-up audit will reconcile this proposed table against the two current shipped surfaces — the helper `compute_pretrends_power` (exposes `alpha`, `target_power`, `violation_type`, `pre_periods`, plus the optional violation magnitude `M`) and the class `PreTrendsPower` (exposes `alpha`, `power`, `violation_type`, `violation_weights`) — and decide which proposed extensions to keep, rename, unify, or defer.

| Parameter | Type | Default | Source | Selection Method |
|-----------|------|---------|--------|-----------------|
| `alpha` | float in (0, 1) | 0.05 | Paper-derived (free parameter affecting z_{1-alpha/2}) — also currently exposed by `compute_pretrends_power` | Standard significance level for pretest and reporting CI |
| `target_power` | list[float] in (0, 1) | [0.5, 0.8] | Paper-derived defaults (Cohen 1988 benchmark; 0.5 supplementary) — current API exposes scalar `power=0.8` only, so a list-valued knob is a proposed extension | Roth's reported benchmarks |
| `l` (contrast) | array in R^M | uniform 1/M | Paper-derived (free parameter in Section I.C); not in current API as a top-level knob | User-specified linear functional of tau_post |
| `pretest_form` | enum | "individual" (NIS) | **Library extension** (current API uses `violation_type`, a different axis; the paper has no single enum for this) | "individual" (paper-analyzed); "joint_wald" (convex, Propositions 1+3+4 all apply); "custom" (Propositions 1+3 always; Proposition 4 only if user's B is convex); "slope" — deviation beyond Roth (2022) |
| `acceptance_region` | callable or set | B_NIS | **Library extension** (Roth's propositions apply to any measurable B, but the paper does not propose a callable interface) | Custom B(Sigma) for "custom" pretest_form (Propositions 1, 3 apply to any measurable B; Proposition 4 / variance-reduction guarantee additionally requires B to be convex) |
| `method` | enum | "analytical" | **Library extension** (Roth uses analytical via `tmvtnorm`; simulation is a library robustness choice) | "analytical" (`tmvtnorm`-equivalent) or "simulation" |
| `n_sim` | int | 10000 | **Library extension** (only meaningful when `method="simulation"`) | Monte Carlo iterations when method="simulation" |

### Relation to Existing diff-diff Estimators
- **Pre-existing `diff_diff/pretrends.py`** (1133 lines) — implements a Wald-test-based pre-trends MDV/power workflow framed around Roth (2022); the current code path computes Wald power/MDV from the pre-period variance-covariance block rather than the full arbitrary-Sigma Proposition 1 / Proposition 3 / Proposition 4 conditional-moment computations. This paper review's main use is to audit the existing surface against the paper's exact equations and identify which Roth-2022 quantities are missing.
- **Currently composes with** (per the shipped `compute_pretrends_power` adapter in `diff_diff/pretrends.py`): `MultiPeriodDiDResults`, `CallawaySantAnnaResults`, `SunAbrahamResults`. The adapter raises `TypeError` for other result types. Theoretical compatibility extends to any estimator producing an event-study coefficient vector and a consistent variance estimator (e.g., `TwoWayFixedEffects`), but adapters for additional result families are a follow-up audit decision.
- **Note (deviation in current covariance-source):** the shipped adapter uses different sources for the pre-period covariance depending on the result type:
  - `MultiPeriodDiDResults`: extracts the full pre-period sub-block from `results.vcov` when `interaction_indices` is populated, falling back to `diag(ses^2)` otherwise (`pretrends.py:592-601`).
  - `CallawaySantAnnaResults`: hard-codes `vcov = diag(ses^2)` (`pretrends.py:609-652`). Non-bootstrap CS fits persist a full `event_study_vcov` matrix on the result object (`staggered_results.py:126-128`), so the diag fallback is a deliberate choice in that path. Bootstrap CS fits explicitly clear `event_study_vcov` before storing results (`staggered.py:2032-2036`) to prevent mixing analytical VCV with bootstrap SEs, so the full-Σ22 route is not available for bootstrap fits at all. The follow-up audit can extract the full sub-VCV from `event_study_vcov` on the non-bootstrap path; bootstrap CS fits remain on the diag fallback regardless.
  - `SunAbrahamResults`: hard-codes `vcov = diag(ses^2)` (`pretrends.py:660-687`). Unlike CS, `SunAbrahamResults` does NOT currently expose an event-study or cohort covariance matrix (`sun_abraham.py:30-88`), so the diag fallback is *forced* — Roth-faithful off-diagonal support on the SA path first requires extending `SunAbrahamResults` to persist an event-study/cohort covariance matrix, then routing it through the adapter.

  In all three cases the diagonal fallback is a **non-paper approximation**, and the impact differs by which power object the audit chooses to surface:
  - **Roth's NIS power object** (paper-analyzed): the multivariate box probability `P(β̂_pre ∈ B_NIS(Σ))` computed under `β̂_pre ~ N(δ_pre, Σ_22)`. This box probability genuinely depends on the off-diagonals of Σ_22; replacing Σ_22 with `diag(ses^2)` treats the pre-periods as independent and changes the rejection probability in a sign-and-magnitude-dependent way (not provably conservative).
  - **Current library Wald object** (in shipped `compute_pretrends_power`): a Wald / noncentral-χ² calculation involving the quadratic form `w' Σ_22^{-1} w`. Replacing Σ_22 with `diag(ses^2)` ignores the off-diagonal correlations of the score vector and is similarly not provably conservative for MDV/power.

  In neither object is the diag fallback a paper-supported numerical choice. The follow-up audit should either extend full-sub-VCV consumption to all three paths (with SA also requiring upstream surface work) or, if the diag fallback is retained anywhere, add an explicit `REGISTRY.md` Note describing the approximation and its possible miscalibration rather than framing it as conservative.
- **Complement to `HonestDiD` (Rambachan-Roth 2023)**: Roth 2022 asks "what bias survives a pretest under linear violations?"; Rambachan-Roth 2023 asks "what is the identified set of tau_post under bounded violations?" Both use the same (beta_hat, Sigma_hat) input contract — the library should expose a unified entry-point that can produce both Roth-2022 and HonestDiD reports from one event-study result object.
- **Shares zero-anticipation convention with HonestDiD**: tau_pre = 0, so beta_pre = delta_pre. Cross-reference the existing `diff_diff/honest_did.py` for the contract.

---

## Key Theorems / Propositions

| # | Statement | Implementation use |
|---|-----------|---------------------|
| **Proposition 1** | For any B(Sigma): E[beta_hat_post | beta_hat_pre in B] = tau_post + delta_post + Sigma_{12} Sigma_{22}^{-1} (E[beta_hat_pre | beta_hat_pre in B] - beta_pre) | The main bias decomposition formula. Drives the conditional-bias computation in step 4 of the algorithm. |
| **Proposition 2** | Under Assumption 1 (homoskedastic-equicorrelated Sigma) and monotone trend (delta_pre < 0, delta_post > 0): E[beta_hat_post | beta_hat_pre in B_NIS] > beta_post > tau_post | Justifies a WARN that conditional bias is worse than unconditional bias under monotone trends — applicable in many but not all empirical settings. Assumption 1 in the paper is a condition on the *model* covariance Σ (the population variance-covariance of β̂ in Equation 1), not on design metadata. Software can only inspect the *estimated* Σ̂, so any direct numerical check (e.g., approximately-constant diagonal entries + approximately-constant positive off-diagonal entries below the diagonal) is a heuristic implementation aid, not the paper's assumption itself. A library that surfaces a sharper warning based on Σ̂ should label it as a heuristic; without such a check, the library should issue only the generic caveat that the sign-of-bias result is ambiguous outside Assumption 1. |
| **Proposition 3** | Var[beta_hat_post | beta_hat_pre in B] = Var[beta_hat_post] + (Sigma_{12} Sigma_{22}^{-1}) (Var[beta_hat_pre | beta_hat_pre in B] - Var[beta_hat_pre]) (Sigma_{12} Sigma_{22}^{-1})' | The conditional-variance formula; drives the over/under-coverage analysis. |
| **Proposition 4** | If B(Sigma) is convex: Var[beta_hat_post | beta_hat_pre in B] <= Var[beta_hat_post] (variance-reduction guarantee, conditional on convex B only). | Justifies the "do not interpret a wide CI as ample power" warning. Implication for CI coverage (Section II.C paragraph after Prop 4): CIs based on unconditional Sigma tend to OVER-cover under parallel trends with symmetric B; under violations they tend to UNDER-cover *only if the bias is sufficiently large* to outweigh the variance reduction — the under-coverage direction is contingent on bias magnitude, not universal. |

No formal theorems are stated for the publication-rules analysis (Section II.D); Equation 4 is the operational result.

---

## Calibrated DGP for Simulations (Section I.C "Calibrating the Model")

For each paper in Roth's empirical survey:

1. Calibrate finite-sample normal model (Equation 1): beta_hat ~ N(beta, Sigma) with K pre-periods + M post-periods matching the original paper
2. Set Sigma = estimated variance-covariance matrix from the original paper (using whatever clustering method the authors specified)
3. Set tau_post = original paper's beta_hat_post (footnote 7: has no impact on bias/coverage results by equivariance)
4. Calibrate delta to a linear trend with slope gamma_{0.5} or gamma_{0.8}
5. Re-compute power, bias, and coverage analytically (or by simulation)

**Test fixture suggestion for the library**: a Roth-2022 parity test against one of the 12 papers in Table 1 (e.g., Bailey & Goodman-Bacon 2015 has 5 pre-periods + a clean calibrated VCV available in his replication data — `https://doi.org/10.3886/E151982V1`).

---

## Empirical Findings (Section I.C "Results"; Tables 2-3)

Quoting Roth's key empirical results (for cross-validation):

- **Power**: in the most extreme paper (Deryugina 2017), an unconditional bias of magnitude comparable to the estimated effect is detected only 50% of the time
- **Coverage**: under gamma_{0.8} (80%-power slope), unconditional null rejection rates of 95% CIs for tau_bar range from 14% (Deschenes et al. 2017) to 98% (Lafortune et al. 2017; Markevich & Zhuravskaya 2018) across the 12 papers in Table 2
- **Pretest bias**: percent additional bias from pretest conditioning (Table 3, gamma_{0.8}, tau_1): from -34% (Bosch-Campos-Vazquez 2014, beneficial — rare) to +120% (Deryugina 2017, harmful — common); paper-aggregate finding is that conditional bias EXCEEDS unconditional bias in 9 of 12 papers for tau_1 and in 10 of 12 for tau_bar
- **Equation 4 sign**: the relative-fraction term is < 1 (pretest helps screen out biased designs); the conditional-bias term is typically > 1 (pretest amplifies bias when a biased design is published); net sign depends on which dominates — the paper does not provide closed-form criteria

---

## Gaps and Uncertainties

- **Joint Wald acceptance region**: paper mentions joint tests only briefly (Section I.B notes 1 of 12 papers uses one). Power, bias, and coverage formulas all apply by replacing B_NIS with the joint Wald acceptance region B_W (convex, so Propositions 1+3+4 all hold), but Roth does not work out a separate table. Joint Wald is theoretically admissible under the paper's propositions, but the published R `pretrends` package surface (`github.com/jonathandroth/pretrends` at commit `122731d082` / package version 0.1.0, verified in PR-C parity goldens) is NIS-based (`pretrends()`, `slope_for_power()`, `*_NIS` helpers) and does NOT expose a joint-Wald parity target. Any library implementation of joint-Wald PreTrendsPower will need an independent fixture or derivation rather than direct R-package parity.
- **"Slope-of-best-fit-line t-test" acceptance region**: Table 1 column shows the t-stat for the slope of the linear pre-trend. Paper does not analyze pretests based on this t-stat as a separate acceptance region; library should NOT extrapolate without further reading the `pretrends` package source.
- **Nonlinear violations**: Section I.C formally tabulates power only against linear violations; Section I.D extends the sign-of-bias result (Proposition 2) to monotone violations under homoskedasticity. Section III ("Practical Recommendations") explicitly endorses power analyses against hypothesized nonlinear trends via the `pretrends` package, so the general nonlinear capability is paper-supported even though the paper does not separately tabulate it. The specific named shapes the library exposes ("constant", "last_period") are R-package API conventions, not separately analyzed in the paper.
- **Custom delta vector interface**: paper Section III endorses "power analyses for the types of violations of parallel trends deemed to be most relevant in their context," which is the paper-level framing for a user-supplied delta vector; the specific `violation_weights`-style INTERFACE used in the library and the R `pretrends` package is a package-API convention layered on top of that paper-level framework.
- **Choice of contrast l**: paper highlights l = uniform 1/M (average post-treatment) and l = e_1 (first period after treatment). No guidance on other contrasts (e.g., long-run effect l = e_M, dynamic-weighted contrast) — library should document defaults and warn that bias and coverage depend on l.
- **K = 0 (no pre-periods)**: trivially no pretest possible; library should error.
- **Heteroskedastic Sigma**: Proposition 2 requires Assumption 1. The current `diff_diff/pretrends.py` implements NIS and Wald power/MDV (PR-B 2026-05-18; NIS is the default per `pretest_form='nis'`), but NOT the full conditional-moment path (Propositions 1 / 3 / 4 numerical evaluation). Once those are added, the library will be able to operate under arbitrary Sigma — but at that point the sign of the bias-amplification effect is NOT guaranteed without Assumption 1. The library should NOT print "pretest amplifies bias under monotone trends" unless Assumption 1 is approximately satisfied (or just always issue the conditional warning).
- **Equation 4 publication-rules analysis**: not standardly implemented in PreTrendsPower-style tools. Roth notes it as part of the discussion (Section II.D) but does not provide a numerical workflow for users. Library should NOT attempt to implement Equation 4 unless requested.
- **Connection to `compute_pretrends_power` library helper**: the paper review confirms that "minimum slope detectable at 80% power" is exactly Roth's gamma_{0.8}, and the library helper should compute and surface this. Need to verify the existing helper's calling convention against the paper's framework when auditing `diff_diff/pretrends.py`.
- **Compatibility with multi-cohort estimators**: Remark 1 lists Callaway-Sant'Anna, Sun-Abraham, etc. as compatible. The paper does not detail how to construct (beta_hat, Sigma_hat) from those estimators when the event-study output is multi-cohort (e.g., cohort × event-time matrix). Library should document the aggregation convention (per Sun-Abraham overall ATT or per Callaway-Sant'Anna `results.aggregate('event_study')`).
