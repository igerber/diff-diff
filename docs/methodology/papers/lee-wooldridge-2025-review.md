# Paper Review: A Simple Transformation Approach to Difference-in-Differences Estimation for Panel Data

**Authors:** Soo Jeong Lee (Southern Illinois University Carbondale), Jeffrey M. Wooldridge (Michigan State University)
**Citation:** Lee, S.J., & Wooldridge, J.M. (2025). A Simple Transformation Approach to Difference-in-Differences Estimation for Panel Data. SSRN Working Paper No. 4516518. First posted 27 Jul 2023; version reviewed dated April 26, 2026, last revised June 8, 2026. https://ssrn.com/abstract=4516518
**PDF reviewed:** SSRN download of abstract 4516518 (61 pages; cover page dated June 8, 2026; SHA-256 `78460841def3f15fdac6a2c6b04bc0c80ecc192493b9aa441e465a81f6846ea0`). https://ssrn.com/abstract=4516518 | DOI: https://doi.org/10.2139/ssrn.4516518
**SSRN metadata (verified live 2026-07-13):** 61 pages; Posted: 27 Jul 2023; Last revised: 8 Jun 2026; Date Written: April 26, 2026. Note: stale caches/mirrors of the SSRN page may still show the superseded December 25, 2025 revision (52 pages).
**Review date:** 2026-07-11

---

## Methodology Registry Entry

*Formatted to match docs/methodology/REGISTRY.md structure. Heading levels and labels align with existing entries - copy the `## LWDiD` section into the appropriate category in the registry.*

## LWDiD

**Primary source:** Lee, S.J., & Wooldridge, J.M. (2025). A Simple Transformation Approach to Difference-in-Differences Estimation for Panel Data. SSRN Working Paper No. 4516518 (revision of June 8, 2026). https://ssrn.com/abstract=4516518

**Key implementation requirements:**

*Assumption checks / warnings:*
- **Common timing** (Section 2): all treated units start treatment at date S with `1 < S <= T` (at least one pre-treatment period); treatment stays in place through T.
  - **Assumption NAC** (No Anticipation, Common Timing; Equation (2.7)): `E[Y_t(1) - Y_t(0) | D = 1] = 0` for `t = 1, ..., S-1`. Implied by the stronger `Y_t(1) = Y_t(0)` for pre-periods (implicit in Heckman, Ichimura and Todd 1997; explicit in Abadie 2005). If announcement effects are suspected, drop a period or two just prior to the intervention (Section 4.4 robustness check).
  - **Assumption CPTC** (Conditional Parallel Trends, Common Timing; Equation (2.10)): `E[Y_t(0) - Y_1(0) | D, X] = E[Y_t(0) - Y_1(0) | X]`, `t = 2, ..., T`. PT holds within sub-partitions defined by X; PT need not hold across the entire population.
  - **Assumption OVLC** (Overlap, Common Timing; Equations (2.14)-(2.15)): propensity score `p(x) = P(D = 1 | X = x) < 1` for all x in Supp(X).
  - Theorem 2.1: NAC + CPTC + OVLC identify `tau_r`, r = S, ..., T.
- **Staggered interventions** (Section 4): cohorts defined by first treatment period `g in {S, ..., T, infinity}`; treatment is absorbing (no reversibility); cohort indicators `D_S, ..., D_T, D_infinity` mutually exclusive and exhaustive.
  - **Assumption CNAS** (Conditional No Anticipation, Staggered; Equation (4.4)): `E[Y_t(g) | D_g = 1, X] = E[Y_t(infinity) | D_g = 1, X]` for `g in {S,...,T}`, `t in {1,...,g-1}`. If only the never-treated (NT) group is used as control for each (g, r), the conditioning on X can be dropped (p. 17).
  - **Assumption CPTS** (Conditional PT, Staggered; Equation (4.6)): `E[Y_t(infinity) - Y_1(infinity) | D, X] = E[Y_t(infinity) - Y_1(infinity) | X]`, `t = 2, ..., T`, for `D = (D_S, ..., D_T)`.
  - **Assumption OVLS** (Overlap, Staggered; Equation (4.10)): `P(D_g = 1 | D_g + A_{r+1} = 1, X = x) < 1` for all x in Supp(X), where `A_{r+1} = D_{r+1} + ... + D_T + D_infinity` is the legitimate control-pool indicator at (g, r).
  - Theorem 4.1: under CNAS, Equation (4.5) holds; adding CPTS makes cohort assignments D unconfounded with respect to `Y_dot_rg(infinity)` (conditional mean sense) given X.
- **Heterogeneous linear trends** (Section 5): **Assumption CHT** (Conditional Heterogeneous Trends; Equation (5.3)) allows a separate linear trend in the never-treated state per treatment cohort: `E[Y_t(infinity) | D, X] = eta_S*(D_S*t) + ... + eta_T*(D_T*t) + q_infinity(X) + sum_g D_g*q_g(X) + m_t(X)`. Under CHT, PT fails even conditional on X (Equation (5.4)) and the Section 3/4 demeaning estimators are inconsistent; detrending (Procedure 5.1) restores consistency under CNAS + CHT + OVLS.
- **Minimum pre-treatment periods:** demeaning requires >= 1 pre-treatment period per cohort; detrending requires >= 2 (`g >= 3` in Appendix B). Cohorts/cells failing the minimum are not estimable.
- IPWRA consistency requires correct specification of either the propensity score model or the outcome model (doubly robust; p. 14).

*Estimator equation (Equations (3.2), (4.11) in paper, as implemented):*

The rolling transformation converts each (cohort, period) analysis into a standard cross-sectional treatment-effects problem. Common timing (Equation (3.2)):

    Y_dot_ir = Y_ir - (1/(S-1)) * sum_{q=1}^{S-1} Y_iq  =  Y_ir - Ybar_{i,pre}

Staggered (Equation (4.11)):

    Y_dot_irg = Y_ir - (1/(g-1)) * sum_{s=1}^{g-1} Y_is  =  Y_ir - Ybar_{i,pre(g)}

where:
- `Y_ir` = outcome for unit i in calendar period r
- `S` = common intervention date; `g` = first treatment period of unit i's cohort
- `Ybar_{i,pre}` / `Ybar_{i,pre(g)}` = unit-level average of pre-treatment outcomes
- `D_i` (common timing) / `D_ig` (staggered) = treatment / cohort indicator
- `X_i` = time-constant pre-treatment covariates
- Target parameters: `tau_r = E[Y_r(1) - Y_r(0) | D = 1]` (Equation (2.1)); `tau_gr = E[Y_r(g) - Y_r(infinity) | D_g = 1]` (Equation (4.2))

Linear RA, single regression per period (Equation (3.3)); `tau_hat_r` = coefficient on `D_i`:

    Y_dot_ir  on  1, D_i, X_i, D_i*(X_i - Xbar_1),   i = 1, ..., N

with `Xbar_1 = N_1^{-1} sum_i D_i X_i` the covariate average over treated units (centering at `Xbar_1` ensures the coefficient on `D_i` recovers the ATT). No-covariate special case (Equation (3.4)) is plain DiD:

    tau_hat_r = (Ybar_1r - Ybar_0r) - (Ybar_{1,pre} - Ybar_{0,pre})

Detrending variant (Section 5 / Appendix B): replace demeaning with unit-specific linear detrending. Run unit-specific regressions `Y_it on 1, t` over `t = 1, ..., g-1` (Equation (5.6)); for `r >= g` compute out-of-sample predictions `Yhat_irg` and residuals `Yddot_irg = Y_ir - Yhat_irg`. In the simplest case (T=3, S=3, no covariates) the estimator of `tau_3` is the difference-in-difference-in-differences (Equation (5.7)):

    [(Ybar_13 - Ybar_12) - (Ybar_03 - Ybar_02)] - [(Ybar_12 - Ybar_11) - (Ybar_02 - Ybar_01)]

Event-study / placebo transformations over ALL periods (Appendix D): demeaning (Equation (D.1)) `Y_dot_itg = Y_it - (1/(g-1)) sum_{q=1}^{g-1} Y_iq` for `t = 1,...,T`, `t != g-1`; detrending (Equation (D.2)) `Y_ddot_itg = Y_it - Yhat_itg` for `t = 1,...,T`, `t not in {g-2, g-1}`. Anchor periods (event time r = -1 under demeaning; r = -2, -1 under detrending) are excluded.

*With covariates / doubly robust (Equations (E.1), IPW weights p. 55):*

After the transformation, ANY standard cross-sectional TE estimator applies: linear RA, IPW, IPWRA (doubly robust), propensity-score/covariate matching, causal ML for high-dimensional X (Belloni et al. 2014; Chernozhukov et al. 2018). Per-(g,t) cell RA regression (Equation (E.1)):

    Y_dot_{i,g,t} = alpha_{g,t} + theta_{g,t} D_{i,g} + X_i' gamma_{g,t} + D_{i,g} (X_i - Xbar_g)' delta_{g,t} + eps_{i,g,t}

IPWRA (workhorse; two-step per Wooldridge 2025a, Section 19.4): (1) logit propensity score `p_{g,t}(X_i; gamma) = Lambda(X_i' gamma_{g,t})` estimated by ML on the cell sample `A_{g,t}`; (2) weighted least squares of (E.1) with weights

    w_{i,g,t}(gamma) = D_{i,g} + (1 - D_{i,g}) * p_{g,t}(X_i; gamma) / (1 - p_{g,t}(X_i; gamma))

ATT(g,t) = coefficient on `D_{i,g}` (`theta_hat_{g,t} = e_2' eta_hat_{g,t}`). IPW is the special case omitting the outcome-regression component (estimating equation in Appendix E.4). The abstract states the doubly robust IPWRA "works particularly well in terms of bias and efficiency."

*Standard errors (Appendix E):*
- Default: influence-function-based **multiplier bootstrap** for SEs and **simultaneous (sup-t) confidence bands** over the event-study path `{WATT(r) : r in R}` (Algorithm 1). Influence functions come from the stacked estimating equations; no re-estimation per bootstrap replication.
- Influence functions: RA (Appendix E.2) `IF_hat^{RA}_{i,g,t} = e_2' (Z'Z)^{-1} Z_i u_hat_{i,g,t}` - exact in finite samples; IPWRA (Appendix E.3) `IF_{eta,i,g,t} = Q_{w,g,t}^{-1} (w_{i,g,t} Z_{i,g} u_{i,g,t} + H_{g,t} IF_{gamma,i,g,t})` with block upper-triangular stacked Jacobian; IPW (Appendix E.4) `IF^{IPW}_{i,g,t} = psi_{i,g,t} - Gamma_{g,t}' IF_{gamma,i,g,t}`. IPW/IPWRA IFs include first-stage logit-score corrections `IF_{gamma,i,g,t} = A_{g,t}^{-1} X_i (D_{i,g} - p_{g,t}(X_i; gamma_{g,t}))` where `A_{g,t}` here is the logit information matrix.
- Bootstrap: **Rademacher multipliers** `xi_i in {-1, +1}` drawn **per unit i** (one multiplier per unit across all its cells/event times - unit-level clustering by construction, preserving within-unit dependence). Iterations: Algorithm 1 leaves B generic; B = 999 used in the Walmart application (p. 58). Bootstrap applied only to reported event-time coefficients, excluding anchor periods "set to zero by construction" (p. 50). IFs are centered across observations before multiplier draws for finite-sample stability.
- Clustering: unit (panel) level via the per-unit multiplier; no other clustering options, degrees-of-freedom, or t-vs-normal small-sample conventions discussed in the paper (small-sample inference is the subject of the companion paper, SSRN 5325686 - reviewed separately).
- Single-cell inference: for one cohort-time effect, standard cross-sectional TE inference applies directly (e.g., heteroskedasticity-robust SEs from (3.3); Stata `teffects`); IPWRA is a two-step M-estimator with standard asymptotics (Wooldridge 2007; Newey and McFadden 1994). Matching-based variants: standard matching software SEs reflect matching uncertainty (p. 20).

*Aggregation (Section 6.2 / Appendix D.3; unnumbered equation):*

    WATT(r) = sum_{g in G_r} w(g,r) * ATT(g, g+r),   w(g,r) = N_g / N_{G_r},   N_{G_r} = sum_{g in G_r} N_g

contributing-treated-unit weighted average over cohorts `G_r` for which `ATT(g, g+r)` is identified at event time `r = t - g`: the operative weight (Appendix E.1) is the number of treated units in cohort g contributing at event time r over the total treated units contributing at that event time, which simplifies to the cohort-size form `N_g / N_{G_r}` above in balanced panels. Estimable event times: `r = -1` excluded under demeaning; `r = -2, -1` excluded under detrending. Aggregated IF: `IF_{i,r} = sum_g omega_{g,r} IF_{i,g,g+r}`.

*Edge cases:*
- Cohort with zero pre-treatment periods (demeaning) or fewer than two (detrending): transformation undefined -> cell not estimable; detrending rank condition `(J'_{g-1} J_{g-1})` invertible iff >= 2 distinct pre-treatment time points (Appendix B, Equations (B.1)-(B.3)).
- Anchor periods: event-study plots omit `r = -1` (demeaning) / `r = -2, -1` (detrending); bootstrap excludes them (p. 50).
- All units eventually treated (Section 4.3): no NT group -> state CPT relative to `Y_t(T)`; effects defined as `E[Y_r(g) - Y_r(T) | D_g = 1]` for `g in {S,...,T-1}`; no effect estimable for the final cohort; compute `Y_dot_irg` only for `g in {S,...,T-1}` and drop `D_infinity` from Steps 2-3 of Procedure 4.1; when `r = T`, cohort `D_T = 1` is the only control. By no anticipation, for `r < T` the ATTs coincide with the NT-referenced ones.
- Unbalanced panels (Section 4.4): apply demeaning/detrending to the observed data; a (g, r) cell is usable only if `Y_ir` is observed and there are enough pre-g observations (one for demeaning, two for detrending). Selection may depend on unobserved time-constant heterogeneity (like FE); detrending additionally allows trend heterogeneity to correlate with selection; selection must not be systematically related to shocks to `Y_it(infinity)`.
- Suspected anticipation: drop one or more periods just prior to the intervention from the pre-period average/trend and redo the analysis (Section 4.4); procedures apply unchanged with skipped periods. No anticipation is load-bearing for detrending: the second term of (B.8) vanishes only under no anticipation.
- Control-group choice: default control pool at (g, r) is `A_{i,r+1} = 1` (never-treated plus not-yet-treated as of r); an NT-only subset is allowed and lets the X-conditioning in CNAS be dropped. Appendix D.3 generalizes to pre- and post-treatment cells: comparison sample `A_{g,t} = {G_i = g} ∪ {G_i = 0} ∪ {G_i > max(g,t)}` (G_i = 0 denoting never treated), so pre-treatment placebo cells use cohorts treated after g and post-treatment cells cohorts still untreated at t.
- Incidental parameters: the unit-specific trend regressions (5.6) do NOT create an incidental parameters problem with small T; only removal of the CHT heterogeneity is needed, not "good" unit-level trend estimates (p. 28; logic of Wooldridge 2010, Section 11.7.2).
- Normalization differences vs CS: CS (2021) event studies are normalized to `r = -1`, while demeaned rolling estimates are deviations from each unit's own pre-treatment average - event-study points are not directly numerically comparable across the two (p. 30).
- Efficiency: when `r = g` the rolling RA with all controls reproduces the POLS/ETWFE estimator of Wooldridge (2025b); when `r > g` it does not, and under standard assumptions the rolling approach is inefficient for dynamic effects - the trade-off is applicability of doubly robust and matching estimators (p. 19). In the common-timing case, Theorem 3.1 gives full numerical equivalence between the per-period regressions (3.3) and pooled OLS (3.6).

*Algorithm (Procedures 3.1, 4.1, 5.1; Algorithm 1 in paper):*

Procedure 3.1 (Rolling Methods, Common Timing):
1. For a given time period `r in {S, ..., T}` and each unit i, compute `Y_dot_ir` as in (3.2).
2. Using all of the units, apply standard TE methods - such as linear RA, IPW, IPWRA, matching - to the cross section `{(Y_dot_ir, D_i, X_i) : i = 1, ..., N}`.

Procedure 4.1 (Rolling Methods, Staggered Interventions):
1. For a given `g in {S,...,T}` and time period `r in {g,...,T}`, compute `Y_dot_irg = Y_ir - Ybar_{i,pre(g)}` as in (4.11).
2. Choose as the control group the units with `A_{i,r+1} = 1` (or, if desired, a subset, such as the NT group).
3. Using the subset of data with `D_ig + A_{i,r+1} = 1`, apply standard TE methods - such as linear RA, IPW, IPWRA, matching - to the cross section `{(Y_dot_irg, D_ig, X_i) : i = 1,...,N}`, with `D_ig` acting as the treatment indicator.

Procedure 5.1 (Staggered Entry, detrending):
1. For a specified cohort `g in {S,...,T}`, run the unit-specific regressions `Y_it on 1, t`, `t = 1,...,g-1` (Equation (5.6)). For `r in {g,...,T}`, compute out-of-sample predictions `Yhat_irg` and residuals `Yddot_irg = Y_ir - Yhat_irg`. (Not needed for units treated prior to period g.)
2. Same as Procedure 4.1.
3. Identical to Procedure 4.1 with `Yddot_irg` replacing `Y_dot_irg`.

Algorithm 1 (Multiplier Bootstrap for Simultaneous Inference on WATT_hat(r); Appendix E, p. 52):
1. For each group-time cell (g,t), compute `ATT_hat(g,t)` and observation-level influence contributions `IF_hat_{i,g,t}`.
2. Aggregate: `WATT_hat(r) = sum_g omega_{g,r} ATT_hat(g, g+r)`.
3. Construct event-time influence contributions `IF_hat_{i,r} = sum_g omega_{g,r} IF_hat_{i,g,g+r}`; center: `IF_tilde_{i,r} = IF_hat_{i,r} - (1/N) sum_j IF_hat_{j,r}`.
4. For each `b = 1,...,B`, draw independent unit-level Rademacher multipliers `xi_i^(b) in {-1,+1}` and compute `WATT_hat^{*(b)}(r) = sum_i xi_i^(b) IF_tilde_{i,r}` (IFs already contain sample-size scaling; no additional normalization).
5. Compute bootstrap SEs `se_hat_boot(r)` from the empirical variance of `{WATT_hat^{*(b)}(r)}`.
6. For each b, compute the supremum statistic `T*_b = sup_{r in R} |WATT_hat^{*(b)}(r) / se_hat_boot(r)|`.
7. With `c_hat^{sup}_{1-alpha}` the empirical (1-alpha)-quantile of `{T*_b}`, the simultaneous band is `WATT_hat(r) ± c_hat^{sup}_{1-alpha} * se_hat_boot(r)`, `r in R`.

**Reference implementation(s):**
- Stata: user-written command `lwdid` - "implements the full procedure and multiplier-bootstrap inference"; see Hur, Lee and Wooldridge (2026) for details (pp. 30, 32). Walmart results replicable with `lwdid`.
- R: none mentioned in the paper.

**Requirements checklist:**
- [ ] Rolling demeaning transformation (3.2)/(4.11) computed per (cohort g, period r) using ALL pre-g periods
- [ ] Rolling detrending transformation (5.6)/(D.2) via unit-specific OLS on (1, t) over pre-g periods, out-of-sample prediction for r >= g
- [ ] Minimum pre-period enforcement: >= 1 (demeaning), >= 2 (detrending); cells failing are dropped/not estimable
- [ ] Control pool per (g, r): NT + not-yet-treated (`A_{i,r+1} = 1`) by default; NT-only option
- [ ] Pre-treatment placebo cells use the Appendix D.3 comparison rule (`G_i > max(g,t)`)
- [ ] RA estimator (E.1) with treated-cohort-centered covariate interactions `D_{i,g}(X_i - Xbar_g)`
- [ ] IPWRA: logit PS by ML per cell + WLS of (E.1) with weights `w_{i,g,t}`; IPW special case
- [ ] Influence functions per E.2 (RA, exact), E.3 (IPWRA), E.4 (IPW) including first-stage logit-score corrections
- [ ] WATT(r) contributing-treated-unit weighted aggregation over identified (g, g+r) cells (cohort-size weights in balanced panels)
- [ ] Multiplier bootstrap (Algorithm 1): unit-level Rademacher draws, centered IFs, sup-t simultaneous bands; anchor periods excluded
- [ ] Anchor-period exclusion: r = -1 (demeaning); r = -2, -1 (detrending)
- [ ] Section 4.3 support: all-eventually-treated panels (drop D_infinity; last cohort as control at r = T; no effect for last cohort)
- [ ] Section 4.4 support: unbalanced panels (transform on observed data; per-cell observability checks) and anticipation-robustness period dropping
- [ ] Common-timing rolling RA reproduces plain DiD (3.4) with no covariates and matches pooled OLS (3.6) per Theorem 3.1

---

## Implementation Notes

### Data Structure Requirements
- Panel data: random sample of N units observed for t = 1, ..., T periods; balanced panel is the baseline, unbalanced supported per Section 4.4 (transformation applied to observed data with per-cell observability requirements).
- Required variables: outcome `Y_it`; first-treatment period / cohort variable (`g in {S,...,T}`, never-treated coded separately - Appendix D.3 uses `G_i = 0` for never treated); time-constant pre-treatment covariates `X_i` (optional).
- Treatment must be absorbing (no reversibility) with common or staggered entry; at least one pre-treatment period overall (`1 < S <= T`).
- Identification treats each (g, r) analysis as a cross-sectional treatment-effects problem after the transformation; no unit fixed effects are estimated (unit heterogeneity removed by the transformation).

### Computational Considerations
- Per-(g,r)-cell estimation: one transformation pass plus one cross-sectional fit (OLS, or logit-ML + WLS for IPWRA) per cell; cells are independent given the transformed data, so cell-level parallelization is natural.
- Detrending: N unit-specific 2-parameter OLS regressions per cohort (constant + trend on pre-g periods) - vectorizable; no incidental parameters problem with small T (p. 28).
- Multiplier bootstrap requires NO re-estimation per replication: perturb centered influence contributions only ("computationally efficient"; Appendix E). Storage: N x |R| matrix of aggregated IF contributions; B draws of N Rademacher signs.
- Monte Carlo evidence (Appendix C, common timing, T=6, S=4): works down to N = 100 with modest bias (average propensity score ~0.16, i.e. ~16 treated units); rolling IPWRA gives up at most ~2% SD relative to POLS/RA when everything is correctly specified (Table A2) and is uniformly the most efficient consistent estimator under conditional-mean misspecification (Table A3), with the smallest RMSE in all but one design.

### Tuning Parameters

| Parameter | Type | Default | Selection Method |
|-----------|------|---------|-----------------|
| Transformation (demean vs detrend) | choice | demeaning (Procedure 4.1) | Detrend (Procedure 5.1) when pre-treatment event-study estimates show unit-specific linear trends (upward/downward pre-trend pattern); costs event time r = -2 and requires >= 2 pre-periods per cohort |
| Control group | choice | all `A_{i,r+1} = 1` (NT + not-yet-treated) | NT-only subset optional; NT-only lets CNAS drop conditioning on X |
| TE estimator per cell | choice | none prescribed; IPWRA recommended | RA / IPW / IPWRA / matching / causal ML; IPWRA "works particularly well in terms of bias and efficiency" (abstract; Appendix C) |
| Propensity score model | model | logit (estimated by ML per cell) | Fixed in the paper's derivations (Appendix E.3/E.4) |
| Bootstrap repetitions B | int | none stated (generic in Algorithm 1) | B = 999 used in the application (p. 58) |
| Confidence level 1 - alpha | float | 95% bands shown in application | User choice; sup-t simultaneous bands are the default inference mode |
| Anticipation-robustness window | int | 0 (use all pre-g periods) | Drop 1-2 periods just before g as robustness check when no-anticipation is doubtful (Section 4.4); "recommendation is context specific" |

### Relation to Existing diff-diff Estimators
- **CallawaySantAnna**: same (g, t) cell structure, same control pools (NT or NT + not-yet-treated), same RA/IPW/doubly-robust menu, and same multiplier-bootstrap-with-sup-t-bands inference style - but a different outcome transformation: CS uses the long difference `Y_ring_irg = Y_ir - Y_{i,g-1}` (Equation (4.12)), ignoring pre-(g-1) periods, while LWDiD's rolling demeaning (4.11) weights all pre-treatment periods by `1/(g-1)`. Both are consistent under no anticipation + conditional PT; the paper positions them as complementary ("it makes sense to try both", p. 4) with different biases under PT failure and neither uniformly more efficient (CS can win under strong positive serial correlation, being similar to first-differencing). Table 1 (p. 22) contrasts pre-treatment information use. CS is also normalized differently in event-study plots (to r = -1). diff-diff's `CallawaySantAnna` machinery (per-cell estimation loops, IPW/OR/DR components, multiplier bootstrap, event-study aggregation) is the closest structural template for an LWDiD implementation.
- **WooldridgeDiD (ETWFE)**: in the common-timing case, rolling RA is numerically equivalent to Wooldridge (2025b) POLS/ETWFE (Theorem 3.1, proof in Appendix A); in the staggered case equivalence holds only at r = g (instantaneous effects). Under standard error-component assumptions the (3.6) estimators are BLUE (Wooldridge 2025b, Theorem 6.2), so the transformation discards no useful information; the rolling approach trades some efficiency at r > g for doubly-robust/matching applicability. Theorem 3.1-type equivalence is a natural cross-estimator test target against diff-diff's existing `WooldridgeDiD`.
- **DifferenceInDifferences**: the no-covariate common-timing special case (3.4) is plain 2x2 DiD on (pre-average, period-r) means; with T=2, `Y_dot_2 = Y_2 - Y_1` and LWDiD coincides with the canonical estimator - another equivalence test target.
- Detrending (Procedure 5.1) has no current diff-diff analogue; (5.7) shows it is a difference-in-difference-in-differences in the simplest case.
- Reusable diff-diff infrastructure: `linalg.solve_ols()` for all per-cell OLS/WLS fits, `safe_inference()` for joint inference fields, existing logit-IPW propensity code and multiplier-bootstrap/sup-t band code from the CallawaySantAnna family.

---

## Gaps and Uncertainties

1. **Citation-year consistency (RESOLVED 2026-07-11 - use "Lee & Wooldridge (2025)"):** the SSRN working paper was first posted 27 Jul 2023 and the revision reviewed here is dated April 26, 2026 (last revised June 8, 2026), but the companion inference paper (SSRN 5325686, Section 9, p31) cites this paper as "Lee and Wooldridge (2025)" - the authors' own year handle. All PR surfaces (REGISTRY.md, docstrings, llms.txt, docs/references.rst) should cite "Lee & Wooldridge (2025)" with the SSRN 4516518 link, optionally noting the revision date.
2. **Small-sample inference deferred to companion paper:** no degrees-of-freedom corrections, t-vs-normal conventions, or clustering options beyond the unit-level multiplier are discussed anywhere in this paper (flagged by the estimation extractor for pp. 16-32 and the appendix extractor for Appendix E). This is the subject of the separate small-sample-inference companion paper (SSRN 5325686), whose review is being written separately - cross-reference that review for small-sample conventions.
3. **No overall-ATT or per-cohort aggregation:** only the event-study aggregation WATT(r) is defined (Section 6.2 / Appendix D.3; unnumbered equation). No overall ATT, per-cohort, or calendar-time aggregation appears in any extraction. An implementation offering such aggregations goes beyond the paper.
4. **Bootstrap defaults:** Algorithm 1 leaves B generic; the only concrete value is B = 999 in the application (p. 58). No recommended default is stated. Only Rademacher multipliers are specified - no alternative weight distributions are discussed.
5. **Anchor-period rationale (as printed, two phrasings):** p. 29 motivates excluded event times via degrees of freedom ("fitting a unit-specific mean and a unit-specific linear trend requires at least one and two pre-treatment periods, respectively"), while p. 50 refers to "anchor periods that are set to zero by construction" excluded from the bootstrap. Both extractors report their page's phrasing; whether an implementation should display r = -1 (and r = -2 under detrending) as hard zeros or omit them entirely should be settled against the `lwdid` Stata behavior.
6. **Notation collision on `A_{g,t}` (as printed):** the symbol is used for (i) the comparison sample of cell (g,t) (Appendix D.3/E.2), (ii) the stacked-moment Jacobian in E.3, and (iii) the logit information matrix in E.4 (flagged by the appendix extractor: "same symbol ... but a different object"). Also `A_{r+1}` in the main text is the control-pool indicator. Implementations should not conflate these.
7. **E.2 comparison-sample nuance:** Appendix E.2 defines `A_{g,t}` with an additional intersection `{i : T_i = t}`, indicating estimation on the cross-section of observations at calendar time t - notation not fully unpacked in the extraction; relevant for unbalanced-panel handling of the IF sample.
8. **Simulation coverage gaps:** results tables for Scenarios 2C and 4C (defined in Table A1: PS misspecified with correct mean; both misspecified) were not in any extractor's page range - only Table A2 (Scenario 1C) and Table A3 (Scenario 3C) are extracted. There are **no staggered-adoption Monte Carlos** in the paper (Appendix C is common-timing only), and no simulations exercising the detrending estimator were extracted.
9. **WATT weight phrasing:** Section 6.2 / Appendix D.3 give `w(g,r) = N_g / N_{G_r}` (cohort sizes), while Appendix E.1 phrases `omega_{g,r}` as the number of treated units in cohort g *contributing at event time r* over the total contributing at that event time. These coincide in a balanced panel; under unbalanced panels the E.1 phrasing (contributing units) is the operative one. Not a contradiction, but the balanced-panel shorthand should not be hard-coded.
10. **Minor page-boundary fragment:** a sentence on selection into the sample begins on p. 27 and ends on p. 28 (flagged by the edge-cases extractor); its content appears consistent with the Section 4.4 selection discussion (selection may depend on time-constant heterogeneity, and additionally on trend heterogeneity under detrending) but the exact sentence was split across extraction ranges.
11. **Reference implementation details:** `lwdid` internals are documented in Hur, Lee and Wooldridge (2026), which was not reviewed here; parity targets (e.g., exact placebo construction beyond Appendix D, default B, display conventions) require consulting that reference or the command itself.
12. **No contradictions between extraction files were found.** All overlapping content (Procedure 4.1 transcriptions, WATT(r) definitions, anchor-period exclusions, Walmart headline numbers WATT(1) = 0.032 (SE 0.005) vs Table A4 row r = 1, minimum pre-period requirements, CS comparison) is mutually consistent; apparent scope differences (e.g., the edge-cases extractor noting unbalanced panels "not addressed" in its pages while the estimation extractor covers Section 4.4) are page-range artifacts resolved in this synthesis.
