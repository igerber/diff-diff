# Paper Review: Simple Approaches to Inference with Difference-in-Differences Estimators with Small Cross-Sectional Sample Sizes

**Authors:** Soo Jeong Lee (Southern Illinois University Carbondale), Jeffrey M. Wooldridge (Michigan State University)
**Citation:** Lee, S.J., & Wooldridge, J.M. (2026). Simple Approaches to Inference with Difference-in-Differences Estimators with Small Cross-Sectional Sample Sizes. SSRN Working Paper No. 5325686, dated February 3, 2026. https://ssrn.com/abstract=5325686
**PDF reviewed:** SSRN download of abstract 5325686 (36 pages; cover page dated February 3, 2026; SHA-256 `30b2b9bcd09ce63981671624daccc04deed9350cd98e06a6b402325c2eccc145`). https://ssrn.com/abstract=5325686 | DOI: https://doi.org/10.2139/ssrn.5325686
**SSRN metadata (verified live 2026-07-13):** 36 pages; Posted: 27 Jun 2025; Last revised: 13 Jun 2026; SSRN "Date Written" field: January 03, 2026 (the author-entered metadata field lags the delivered PDF's cover date of February 3, 2026 - cite the cover date).
**Review date:** 2026-07-11

---

## Methodology Registry Entry

*Formatted to match docs/methodology/REGISTRY.md structure. This paper supplies the small-sample inference layer of the LWDiD estimator (whose estimation core is documented in lee-wooldridge-2025-review.md) - the section below is written as ADDITIONS to the LWDiD registry entry, not a separate estimator.*

## LWDiD - small-sample inference layer

**Primary source:** Lee, S.J., & Wooldridge, J.M. (2026). Simple Approaches to Inference with Difference-in-Differences Estimators with Small Cross-Sectional Sample Sizes. SSRN Working Paper No. 5325686. https://ssrn.com/abstract=5325686

Companion estimation core: Lee & Wooldridge (2025) [LW (2025)] - panel DiD estimators are obtainable via cross-sectional regressions after unit-level time-series ("rolling") transformations; this paper adds exact small-sample inference by collapsing the panel to `{(ΔȲ_i, D_i) : i = 1, ..., N}` in the spirit of Donald and Lang (2007).

**Key implementation requirements:**

*Assumption checks / warnings:*
- Sampling: data draws are i.i.d. across `i`; general dependence and changing distributions allowed across `t` (stated after eq. 2.2). Because inference is based on a cross-sectional regression, NO adjustment for serial correlation is needed, even under strong time-series dependence (unit-root-like processes need not push `ΔȲ_i` far from normality); large `T` requires no modification.
- No anticipation (NA), weakest version (eq. 2.14): `E[Y_it(1) − Y_it(0) | D_i = 1] = 0, t = 1, ..., S−1`. Sufficient: `Y_it(1) = Y_it(0), t = 1, ..., S−1`.
- Parallel trends (PT) (eq. 2.15): `E[Y_it(0) − Y_i1(0) | D_i] = E[Y_it(0) − Y_i1(0)] ≡ δ_t` for all `t = 2, ..., T`; `δ_t` unrestricted over time. Allows `D_i` correlated with the level `Y_i1(0)`; rules out assignment based on differential trends. PT implies (eq. 2.16, from LW 2025): `E[ΔȲ_i(0) | D_i] = α`; with NA this identifies `τ` and makes `τ̂_DD` conditionally unbiased. With small `N1` or `N0` the paper relies on unbiasedness, not consistency.
- Conditional PT with time-constant controls `X_i` (1×K) (eq. 2.17): `E[Y_it(0) − Y_i1(0) | D_i, X_i] = α_t + X_i β_t` (linearity imposed "because we have little choice in a small-N setting"; unconfoundedness in differences, not levels).
- Classical linear model (CLM) assumptions for EXACT inference (eqs. 2.7-2.9): `ΔȲ_i = α + τ D_i + U_i`; `E(U_i | D_i) = 0`; `U_i | D_i ~ Normal(0, σ_U²)`. Eq. (2.9) bundles conditional normality AND homoskedasticity - the paper flags homoskedasticity as a (technical) drawback; HC3 is the relaxation when N is not too small. With controls, the CLM version is (2.18)-(2.19).
- Normality justification (p32): because the method averages across time, the CLT across the time dimension often justifies the CLM normality assumption; works best with large `T0` and `T1`, but under exact joint normality and homoskedasticity applies with few periods and few units (even `N = 3`: two controls, one treated). A log outcome transform can make normality of the transformed outcome a better approximation (used in the Prop 99 application, p19).
- Sample-size checks: `N0 ≥ 1`, `N1 ≥ 1`, `N = N0 + N1 ≥ 3` (no controls); `N > K + 2` with K controls. Interacted-controls regression requires `N0 > K + 1` AND `N1 > K + 1` - NOT feasible with small `N1`.
- Staggered case: `N_∞ ≥ 2` needed when only never-treated units serve as controls; with `N_∞ ≥ 2`, treated cohorts may have as few as one unit (p26). Not-yet-treated (NYT) units are also valid controls under suitable NA + PT assumptions from LW (2025), per (7.7).
- No incidental-parameters problem: `ΔȲ_i` and `Ȳ̈_i` are linear functions of `{Y_it}` computed unit-by-unit; independence across i delivers independent cross-sectional observations.
- Caveat (p19, p32): the detrending advantage relies on unit-specific *linear* trends; with pre-trend patterns too complex for linear detrending the method "will not always work better than existing alternatives."

*Inference procedures (with equation/procedure numbers):*

    Exact t (core result, eq. 2.10): under (2.7) + (2.9), conditional on D_i,
    tau-hat_DD is exactly normal and

        (tau-hat_DD − tau) / se(tau-hat_DD) ~ T_{N−2}          (2.10)

    - Regression (2.5): DeltaYbar_i on 1, D_i  (i = 1, ..., N); usual OLS t
      statistic and standard error; exact tests of any null; CIs from T_{N−2}
      percentiles have exact coverage. Valid for any N0 >= 1, N1 >= 1, N >= 3.
    - With K controls (2.18)-(2.19): OLS on DeltaYbar_i on 1, D_i, X_i requires
      N > K + 2; t statistic is exactly T_{N−K−2}.
    - Preferred full regression when N0 > K+1 and N1 > K+1 (unnumbered, p9):
      DeltaYbar_i on 1, D_i, X_i, D_i·(X_i − Xbar_1), identical to separate
      regressions for D_i = 0 and D_i = 1.
    - N1 = 1 (single treated unit): the t statistic tau-hat_DD / se(tau-hat_DD)
      is the "studentized residual" from outlier analysis (Wooldridge 2025a,
      Section 9.5) - the test asks whether the single treated unit is an
      "outlier" relative to controls. No particular number of time periods
      required provided (2.9) holds (contrast Hagemann 2025).
    - Detrending (Section 3): identical CLM logic for tau-hat_DT from (3.4),
      valid even with N1 = 1: Ybar-ddot_i = alpha + tau_DT·D_i + U_i,
      U_i | D_i ~ Normal(0, sigma^2) (unnumbered display, p11).
    - Per-period effects (2.20): Y-dot_it on 1, D_i gives tau-hat_{t,DD} for
      t = S, ..., T; identical to a standard panel DiD on periods
      {1, ..., S−1, t}. Exact CI per tau_t under CLM. CAUTION (p10): SEs for
      linear combinations of the tau-hat_{t,DD} are not easily obtained
      (serial-correlation-induced dependence across per-period estimates).
    - Alternative baselines: CS-style long difference (2.21)
      Y-ring_it = Y_it − Y_{i,S−1}; anticipation guard (2.22) uses
      Ybar_{i,S0} = (1/S0)·sum_{r=1}^{S0} Y_ir with S0 < S−1.
    - Second differencing (unnumbered, p12):
      Y-dddot_it = (Y_it − Y_{i,S−1}) − (Y_{i,S−1} − Y_iR), 1 <= R < S−1;
      a triple-difference form, no pre/post averaging, so likely more
      sensitive to normality violations.

    Staggered rollout (Section 7.1): cohorts g in {S, ..., T} with indicators
    D_ig; never-treated D_{i,infinity}; tau_gt (7.1), cohort averages tau_g (7.2).
    Transformations use only data through g−1: demeaning regressions (7.3) then
    Y-dot_itg = Y_it − Ybar_{i,pre(g)} (7.4); detrending regressions (7.5) then
    Y-ddot_itg = Y_it − A-hat_ig − B-hat_ig·t (7.6). Valid controls at (g,t) per
    (7.7): D_{i,t+1} + ... + D_iT + D_{i,infinity} = 1 (any subset valid; all
    controls preferred for efficiency).

        tau-hat_gt: Y-dot_itg on 1, D_ig  using D_ig + C_{i,t+1} = 1     (7.8)
        tau-hat_g:  Ybar-dot_ig on 1, D_ig using D_ig + D_{i,inf} = 1    (7.10)

    with Ybar-dot_ig (and Ybar-ddot_ig) the post-treatment time averages (7.9).
    Under homoskedastic normality, exact inference applies in (7.8)/(7.10) even
    if N_g = 1 and the number of control units is small ("We can appeal to
    exact theory if N_g is small, including N_g = 1", p28).

    Aggregated effect tau-hat_omega (7.11) with cohort-share weights
    omega-hat_g = N_g / (N_S + ... + N_T) (7.12). Via the difference-in-means
    representation (7.13) and algebra (7.14)-(7.17), define the composite
    outcome (7.18):

        Ybar-dot-bar_i = D_iS·Ybar-dot_iS + ... + D_iT·Ybar-dot_iT
                       + D_{i,infinity}·( sum_{g=S}^{T} omega-hat_g·Ybar-dot_ig )   (7.18)

    (eventually-treated unit: its own cohort's post-average residual;
    never-treated unit: the omega-hat_g-weighted average of its cohort-specific
    post-average residuals across all treated cohorts). With the ever-treated
    indicator D_i = D_iS + ... + D_iT (7.15), run the single cross-sectional
    regression

        Ybar-dot-bar_i on 1, D_i,   i = 1, ..., N                        (7.19)

    tau-hat_omega is the coefficient on D_i. Replace Ybar-dot_ig with
    Ybar-ddot_ig everywhere for the detrended version. Advantages: (i)
    automatically accounts for correlations among the tau-hat_g; (ii) usable
    whether N_treat / N_control are small or large; (iii) if both are even
    moderately large, HC3 from (7.19) is justified asymptotically.

where:
- `N`, `N0`, `N1` = total, control, treated cross-sectional unit counts; `T0`, `T1` = pre/post period counts; `S` = first treated period
- `D_i` = time-constant treatment-group indicator (2.1); `W_it = D_i · post_t` (2.2)
- `ΔȲ_i ≡ Ȳ_{i,post} − Ȳ_{i,pre}` = the rolling transformation (2.4); `Ẏ_it` = demeaned out-of-sample residual (2.12); `Ÿ_it` = detrended outcome (3.2); `Y̊_it` = long difference vs period S−1 (2.21)
- `U_i` = cross-sectional regression error with variance `σ_U²`; `T_{N−2}`, `T_{N−K−2}` = t distributions with the indicated df
- `τ̂_DD` / `τ̂_DM` = demeaning estimator (Procedure 2.1); `τ̂_DT` = detrending estimator (Procedure 3.1); `τ̂_{t,DD}`, `τ̂_{t,DT}` = per-period versions
- Staggered: `g` = cohort (time of first treatment); `D_ig` = cohort indicator; `D_{i,∞}` = never-treated indicator; `N_g`, `N_∞` = cohort sizes; `N_treat = N_S + ... + N_T`; `ω̂_g` = cohort share (7.12); `Ȳ̇_ig` / `Ȳ̈_ig` = cohort-specific post-average transformed outcomes (7.9); composite outcome per (7.18)
- `Â_i, B̂_i` = unit-specific pre-period intercept and linear-trend slope (Procedure 3.1); `R` = reference period for second differencing; `S0` = anticipation-guard pre-period count (2.22)

*Standard errors:*
- Default: usual OLS standard error from the collapsed cross-sectional regression, with exact `T_{N−2}` (no controls) or `T_{N−K−2}` (K controls) reference distribution under CLM assumptions (2.7)-(2.9) / (2.18)-(2.19). Exact coverage for CIs; valid down to `N = 3` and `N1 = 1` (or `N_g = 1` in the staggered case).
- Alternative: HC3 heteroskedasticity-robust SEs (the MacKinnon and White (1985) estimator, per Simonsohn (2021) commenting on Young (2019)) when N is "not too small" - the paper's operational phrasing is "provided there are at least a handful of treated units" (p32). Exact homoskedastic-normal inference is the fallback when treated units are too few for HC3.
- Randomization inference: exact p-values for the sharp null that all treatment effects are zero; two-sided RI p-value = `c/N` where `c` = number of permutation test statistics as or more extreme than observed and `N` = total number of permutations (p7 - note this `N` is the permutation count, an overload of the sample-size symbol). Does not rely on normality. Stata: `ritest`, or the Stata `lwdid` command's `ri` option. In the Prop 99 application, Procedure 3.1's exact p-value (0.021) and RI p-value with 1,000 replications (0.020) are nearly identical (Table 3, Note 2, p23).
- Clustering (Section 8.2, pp30-31): because the final regressions are cross-sectional, SEs can be clustered at a level higher than i (e.g., i = county, policy at state level, cluster at state, given sufficiently many treated and control states) - cites Abadie, Athey, Imbens and Wooldridge (2023). Clustering can also be done separately by time period. With spatially correlated treatment assignment, use heteroskedasticity-and-spatial-correlation robust ("SHAC") standard errors (Conley 1999). It does not matter how large T0 and T1 are.

*Edge cases:*
- `N1 = 1` (single treated unit): detection trivial from treatment counts -> exact t inference still valid; the t statistic is the studentized residual / outlier statistic (p6-7). Same for the detrending estimator (p11).
- `N_g = 1` (single treated unit per cohort, staggered): -> exact inference in (7.8)/(7.10) under homoskedastic normality (pp27-28).
- Never-treated group too small: NT-only control strategy requires `N_∞ ≥ 2` (pp26-27) -> otherwise use not-yet-treated controls per (7.7).
- Periods with no new treated units (staggered): -> no treatment effects estimated in those periods (p26).
- Control-group choice: never-treated only, or any NYT unit under NA + PT per (7.7); all possible controls preferred for efficiency but any subset (including NT-only) valid (p27).
- Heteroskedasticity: "always an issue in cross-sectional regressions" -> use HC3 provided at least a handful of treated units (p32); with too few treated units, fall back to exact homoskedastic-normal inference.
- Too few observations for controls: `N ≤ K + 2` -> the with-controls exact result unavailable; interacted regression additionally needs `N0 > K + 1` and `N1 > K + 1` (p9).
- Complex nonlinear pre-trends: linear detrending inadequate -> method may be more biased than SC/SDiD; not universal (p19, p32). With large `S`, higher powers (`t²`, `t³`) can be added in step 1, but removing too much variation hurts detection power (p12).
- Anticipation: -> replace `Ȳ_{i,pre}` with `Ȳ_{i,S0}` averaging only the first `S0 < S − 1` pre-periods (2.22), or use a long difference `Y_it − Y_{i,S0}`.
- Seasonality (quarterly/monthly/weekly data): -> include seasonal dummies in step 1 of Procedure 2.1 or 3.1 to deseasonalize/detrend; afterwards the procedure is exactly the same (p12).
- Linear combinations of per-period effects: -> SEs not easily obtained because of serial-correlation-induced dependence across per-period estimates (p10); the aggregated regression (7.19) is the paper's device for a valid SE on the weighted average.
- Choosing the number of pre-treatment periods (Section 8.1, p30): robustness via varying the data's starting point (varying `T0`), in the spirit of Rambachan and Roth (2023); the approach does not rely on large `T0` or `T1` (but does rely on normality). Fewer pre-treatment periods may suffice when policy is based on past outcomes.
- Repeated cross-sections (pp13-14): aggregate micro outcomes to the assignment level, `Ȳ_st = Σ_{i∈(s,t)} w_ist Y_ist` with weights summing to 1 (equal weights `1/n_st` allowed; survey/design weights usable); random sampling within `(s,t)` cells plus large cell sizes justify treating the cell means as the population means (Donald and Lang 2007) with no adjustment for sampling error; then Sections 2-3 procedures apply unmodified. Sub-annual frequency uses `Ȳ_stq` with quarter dummies `q ∈ {q1, ..., q4}` in the transformation step.

*Procedures (Procedure 2.1 / 3.1 in paper):*

Procedure 2.1 (Unit-Specific Demeaning), p8:
1. Obtain `Ȳ_{i,pre}` for each i from the pre-intervention regression `Y_it on 1, t = 1, ..., S−1` (2.11) (the lone coefficient is `Ȳ_{i,pre}`).
2. In each post-intervention period, form out-of-sample residuals (prediction errors): `Ẏ_it = Y_it − Ȳ_{i,pre}, t = S, ..., T` (2.12).
3. Average the out-of-sample residuals to obtain the same regressand as in (2.5): `Ȳ̇_i ≡ (1/(T−S+1)) Σ_{t=S}^{T} Ẏ_it = Ȳ_{i,post} − Ȳ_{i,pre} = ΔȲ_i`.
4. Obtain the average effect `τ̂_DM`, its standard error and confidence interval, from `Ȳ̇_i on 1, D_i, i = 1, ..., N` (2.13).

Procedure 3.1 (Unit-Specific Detrending), p11 (easily modified for higher-order trends):
1. For each i, obtain `Â_i, B̂_i` from the pre-treatment periods by regressing `Y_it on 1, t, t = 1, ..., S−1` (3.1).
2. For the post-treatment periods, remove the pre-treatment trends: `Ÿ_it = Y_it − Ŷ_it ≡ Y_it − Â_i − B̂_i·t, t = S, ..., T` (3.2), where `Ŷ_it ≡ Â_i + B̂_i·t` is the projected value.
3. For each unit, average the adjusted outcomes: `Ȳ̈_i ≡ (1/(T−S+1)) Σ_{t=S}^{T} Ÿ_it` (3.3).
4. Obtain the average effect `τ̂_DT`, its standard error and confidence interval, from `Ȳ̈_i on 1, D_i, i = 1, ..., N` (3.4).

Staggered analogue (Section 7.1): per-cohort transformations (7.3)-(7.6) using only data through g−1 (compute for every unit, sort out valid controls afterwards); per-(g,t) regressions (7.8); cohort-average regressions (7.10); aggregate via composite-outcome regression (7.19).

Synthetic-control diagnostic (Section 4, eqs. 4.1-4.2): after transformation, the cross-sectional average of control residuals `{N0^{-1} Σ_{i=1}^{N0} Ẏ_it : t = 1, ..., T}` (4.1) acts as the synthetic control for the average of treated-unit residuals `{N1^{-1} Σ_{i=N0}^{N0+N1} Ẏ_it : t = 1, ..., T}` (4.2) (as printed, the lower summation index in (4.2) is `i = N0`; treated units are the `N1` units indexed after the `N0` controls). With detrending, `Ÿ_it` replaces `Ẏ_it`. Success = close agreement of the two average residual series over `t = 1, ..., S−1`; estimated treatment effects are the differences for `t ≥ S` and are exactly the coefficients in (2.5) or (3.4).

**Reference implementation(s):**
- Stata: `lwdid` (publicly available user-written command; implements Procedure 2.1 and Procedure 3.1; randomization-inference p-values via its `ri` option). Stata `ritest` cited for RI generally. Stata 18 `sdid` package used for the SDiD comparisons in the castle-laws application.

**Requirements checklist:**
- [ ] Exact t inference on the collapsed cross-sectional regression: `T_{N−2}` df without controls, `T_{N−K−2}` with K controls (eq. 2.10; p9)
- [ ] Validity down to `N0 ≥ 1, N1 ≥ 1, N ≥ 3`; guard `N > K + 2` with controls; interacted controls require `N0 > K + 1` and `N1 > K + 1`
- [ ] `N1 = 1` supported (studentized-residual interpretation); `N_g = 1` supported in the staggered case
- [ ] Per-period effects via (2.20) with exact CIs; per-period equivalence to panel DiD on `{1, ..., S−1, t}`
- [ ] Alternative baselines: CS-style long difference (2.21); anticipation guard `Ȳ_{i,S0}` (2.22)
- [ ] Detrending Procedure 3.1 with the same exact-inference layer (3.4); optional higher-order trends and seasonal dummies
- [ ] Staggered aggregation via composite outcome (7.18) and single regression (7.19) with cohort-share weights (7.12)
- [ ] Staggered control-group options: never-treated (requires `N_∞ ≥ 2`) or not-yet-treated per (7.7)
- [ ] HC3 SEs offered when there are at least a handful of treated units
- [ ] Randomization inference for the sharp null (two-sided p = c / #permutations)
- [ ] Higher-level clustering and SHAC (Conley 1999) SEs available for larger cross sections (Section 8.2)
- [ ] Repeated cross-sections handled by cell-level aggregation with weights summing to 1 (Donald-Lang)
- [ ] Synthetic-control-style diagnostic plot of (4.1) vs (4.2) residual averages

---

## Implementation Notes

### Data Structure Requirements

- Balanced panel of `N` cross-sectional units over `T` periods, intervention at `S ∈ {2, ..., T}` remaining in place through `T` (common timing, Sections 2-6). Pre-periods `1, ..., S−1`; post periods `S, ..., T`.
- Staggered rollout (Section 7): cohorts `g ∈ {S, ..., T}` = time of first treatment; cohort indicators `D_ig`; never-treated indicator `D_{i,∞}`. Transformations for cohort g use only data through `g−1`. Easiest to compute the transformation for every unit, then sort out valid control units afterwards (p26-27).
- Time-constant controls `X_i` (1×K) enter the collapsed cross-sectional regression when N is large enough.
- Repeated cross-sections: aggregate micro-level outcomes `Y_ist` to the assignment level (e.g., state-year) with weights `w_ist` summing to 1 within each `(s,t)` cell (equal weights `1/n_st` allowed; survey/design weights usable); quarterly-or-higher frequency uses state-year-quarter means `Ȳ_stq` with quarter dummies (pp13-14).

### Computational Considerations

- All estimation reduces to unit-by-unit OLS on pre-periods (a constant, or constant + trend, optionally + seasonal dummies) followed by ONE cross-sectional OLS regression; no incidental-parameters problem (transformed outcomes are linear functions of `{Y_it}` computed per unit; independence across i gives independent observations).
- No serial-correlation correction of any kind: strong dependence (e.g., time-constant unobserved effects) is eliminated by differencing post- and pre-averages; large `T` requires no modification.
- The demeaning estimator reproduces the standard DiD/TWFE estimator: pooled regression (2.3) `Y_it on 1, D_i, post_t, W_it` (equivalently full unit and time FE, per Wooldridge 2025b, 2010) equals the collapsed cross-sectional regression (2.5) via (2.6) `τ̂_DD = ΔȲ_treat − ΔȲ_control`.
- Per-period coefficients from (2.20) are identical to running a standard panel DiD on time periods `{1, ..., S−1, t}`.
- Staggered aggregation needs no covariance matrix across `τ̂_g`: the composite-outcome regression (7.19) delivers `τ̂_ω` and its SE in one pass (the difference-in-means algebra (7.13)-(7.17) uses `D_i · D_ig = D_ig` and `D_i + D_{i,∞} = 1`).
- Randomization inference used 1,000 replications in the Prop 99 application (Table 3, Note 2).

### Tuning Parameters

| Parameter | Type | Default | Selection Method |
|-----------|------|---------|-----------------|
| Transformation | categorical: demean (Procedure 2.1) / detrend (Procedure 3.1) | none stated | Detrend when unit-specific trends plausible/assignment depends on trends; demean is badly biased under heterogeneous trends (Table 2); detrend caveat for complex nonlinear pre-trends (p19) |
| Trend order | integer (1 = linear; `t²`, `t³` possible with large `S`) | linear | Higher powers only with large `S`; removing too much variation hurts detection power (p12) |
| Seasonal dummies | boolean/set | off | Include in step 1 for quarterly/monthly/weekly data (p12) |
| Pre-treatment baseline | full pre-average (2.4) / period S−1 long difference (2.21) / first `S0` periods (2.22) | full pre-average | `S0 < S−1` guards against anticipation; any (even weighted) average of pre-periods valid |
| `R` (second differencing) | integer `1 ≤ R < S−1` | not used | Only for the triple-difference variant (p12); more sensitive to normality violations |
| Control group (staggered) | never-treated only / any not-yet-treated subset per (7.7) | not stated | All possible controls preferred for efficiency; NT-only requires `N_∞ ≥ 2` |
| SE type | exact-normal OLS t / HC3 / RI / clustered / SHAC | exact-normal OLS t | HC3 with "at least a handful of treated units" (p32); RI needs no normality; clustering/SHAC for larger N (Section 8.2) |
| RI permutations | integer | not stated (1,000 used in application) | Two-sided p = c / #permutations |
| `T0` (pre-period span) | data window choice | full window | Sensitivity analysis by varying the starting point, Rambachan-Roth (2023) spirit (Section 8.1) |
| Donor pool | subset of controls | all `N0` controls | Paper deliberately uses ALL donors via transformation (contrast SC's sparse weights); subset robustness in Sections 6.2 / Appendix A |
| Aggregation weights `ω̂_g` | cohort shares (7.12) | `N_g / N_treat` | Fixed by the target parameter `τ_ω` (7.11) |
| RC cell weights `w_ist` | weights summing to 1 per cell | equal `1/n_st` | Survey/design weights allowed (pp13-14) |

### Relation to Existing diff-diff Estimators

- This paper is the inference companion to LW (2025), the estimation core of the LWDiD estimator under evaluation in third-party PR #588; the registry additions above extend that entry rather than defining a new estimator.
- Equivalences stated in the paper: the demeaning estimator equals the standard 2x2/pooled DiD (2.3) and the TWFE estimator (Wooldridge 2025b, 2010) - i.e., the point estimates coincide with `DifferenceInDifferences` / `TwoWayFixedEffects` on collapsed data; only the inference layer differs (exact t on the collapsed cross-section vs. cluster-robust panel inference, which "performs poorly" with small N or N1).
- The long-difference baseline (2.21) `Y̊_it = Y_it − Y_{i,S−1}` is explicitly identified as the transformation used by Callaway and Sant'Anna (2021) (`CallawaySantAnna` in diff-diff).
- Section 4 positions the method against synthetic control (ADH 2010) and synthetic DiD (AAHIW 2021) (`SyntheticDiD` in diff-diff): SDiD requires weakly dependent (I(0)) series and large `N0, T0, T1`; the exact approach has minimal restrictions on `N0, N1, T0, T1`, gives per-period effects with CIs ("a feature not allowed by SDiD methods", p32), and accommodates heterogeneous trends - but can be more biased than SDiD when pre-trend differences are complicated, and can be considerably less efficient. For staggered designs, AAHIW (2021)'s appendix suggestion (split by adoption date, SDiD per cohort with never-treated donors) is "exactly what we propose with our unit-specific demeaning or detrending" (p29), making the two directly comparable.
- The second-differencing variant (p12) yields a difference-in-difference-in-differences estimator (cf. `TripleDifference`).
- Randomization inference for the sharp null and Conley SHAC standard errors connect to diff-diff's existing placebo/RI and spatial-SE machinery.

### Replication Targets

All California numbers use log per-capita cigarette sales (ADH used levels), California as the single treated state (`N1 = 1`), 19 pre-treatment years (1970-1988), 12 treatment years (1989-2000).

**Table 3 (p23): Estimated ATTs using 38 states as the donor pool**

|  | Procedure 2.1 (DiD) | Procedure 3.1 (Unit-Specific Detrending) | SC | Synthetic DiD |
|---|---|---|---|---|
| Average Effect | -0.422*** (0.121) | -0.227** (0.094) | -0.304*** (0.112) | -0.286*** (0.097) |
| tau_1989 | -0.168* (0.096) | -0.043 (0.059) | | |
| tau_1995 | -0.484*** (0.137) | -0.282** (0.112) | | |
| tau_2000 | -0.667*** (0.164) | -0.403** (0.152) | | |

Note 1: Standard errors in parentheses. \*\*\* p < 0.01, \*\* p < 0.05, \* p < 0.1.
Note 2: For Procedure 3.1, the exact-inference p-value (under normality) is **0.021**, and the randomization-inference p-value (1,000 replications) is **0.020**. The two are nearly identical.

The table reports per-period effects only for 1989, 1995, 2000; the framework estimates all 12 post-period effects.

**PR #588 cross-check:** PR #588 claims "demeaning ATT = -0.422, detrending ATT = -0.227" for Table 3. This MATCHES the extracted Table 3 Average Effect row exactly (Procedure 2.1 = -0.422, Procedure 3.1 = -0.227).

**Table 4 (p25): Estimated ATTs using AL, AR, LA, MS as the donor pool** (four Southern states a priori NOT similar to California; Section 6.2)

|  | Procedure 2.1 (DiD) | Procedure 3.1 (Unit-Specific Detrending) | SC | Synthetic DiD |
|---|---|---|---|---|
| Average | -0.556*** (0.080) | -0.215** (0.039) | -0.571*** (0.034) | -0.392*** (0.030) |
| 1989 | -0.247 (0.107) | -0.027 (0.052) | | |
| 1995 | -0.611*** (0.077) | -0.259** (0.055) | | |
| 2000 | -0.839*** (0.032) | -0.377** (0.115) | | |

Note: standard errors in parentheses: \*\*\* p<0.01, \*\* p<0.05, \* p<0.1. Key finding (p25): Procedure 3.1 with the 4-state pool (-0.215) closely mirrors the all-38-state result (-0.227); SC and SDiD yield larger-magnitude estimates, reflecting poorer pre-treatment fits (Figure 6).

**Table A1 (p35): Estimated ATTs using IL, IA, MN and OH as the donor pool** (Appendix A, Midwestern pool)

|  | Procedure 2.1 (DiD) | Procedure 3.1 (Unit-Specific Detrending) | SC | Synthetic DiD |
|---|---|---|---|---|
| Average | -0.413** (0.118) | -0.198* (0.079) | -0.437** (0.184) | -0.275* (0.154) |
| 1989 | -0.178* (0.071) | -0.040 (0.045) | | |
| 1995 | -0.462** (0.133) | -0.239* (0.088) | | |
| 2000 | -0.655** (0.183) | -0.363* (0.136) | | |

Note: standard errors in parentheses: \*\*\* p<0.01, \*\* p<0.05, \* p<0.1.

**Castle-laws application (Section 7.2, pp29-30), staggered:** data from Cunningham (2021); all 50 U.S. states, 2000-2010; adoptions: 1 state in 2005, 13 in 2006, 4 in 2007, 2 in 2008, 1 in 2009; `N_treat = 21` eventually treated, `N_control = 29` never-treated; outcome = log of annual homicides per 100,000 state residents. Using regression (7.19) with the composite outcome (7.18):
- Demeaning: **τ̂_ω ≈ 0.092** (about 9.2% more homicides); usual OLS SE = 0.057, t ≈ 1.61 (not quite significant at 10%, two-sided); HC3 t statistic = 1.50.
- Detrending (Ȳ̈ composite): estimate decreases to **0.067**; HC3 SE 0.055, t ≈ 1.21.
- SDiD (Stata 18 `sdid`): estimate **0.099**, placebo-method SE 0.069 (t = 1.41) - close agreement with the demeaning method.

**Monte Carlo Table 2 (p18), headline cells:** common timing, N = 20, T = 20, treatment at t = 11, 1,000 replications, true average post-treatment effect = 2 by construction. Full table as extracted:

Scenario 1, P(D_i = 1) = 0.32 (Sample ATT 1.991):

| Estimator | Average Effect | Bias | SD | RMSE | Coverage | Avg SE |
|---|---|---|---|---|---|---|
| Demeaning | 3.905 | 1.914 | 5.10 | 5.445 | 0.94 | 4.96 |
| Detrending | 2.000 | 0.009 | 1.73 | 1.734 | 0.96 | 1.74 |
| Detrending (HC3) | 2.000 | 0.009 | 1.73 | 1.734 | 0.96 | 1.87 |
| SC | 1.616 | -0.375 | 2.14 | 2.177 | 0.97 | 3.27 |
| SDiD | 2.383 | 0.392 | 1.77 | 1.808 | 0.96 | 2.56 |

Scenario 2, P(D_i = 1) = 0.24 (Sample ATT 2.011):

| Estimator | Average Effect | Bias | SD | RMSE | Coverage | Avg SE |
|---|---|---|---|---|---|---|
| Demeaning | 4.264 | 2.253 | 5.67 | 6.101 | 0.93 | 5.48 |
| Detrending | 1.969 | -0.042 | 1.89 | 1.892 | 0.95 | 1.91 |
| Detrending (HC3) | 1.969 | -0.042 | 1.89 | 1.892 | 0.93 | 2.04 |
| SC | 1.622 | -0.389 | 2.35 | 2.384 | 0.94 | 2.73 |
| SDiD | 2.395 | 0.384 | 1.89 | 1.925 | 0.95 | 2.25 |

Scenario 3, P(D_i = 1) = 0.17 (Sample ATT 1.996):

| Estimator | Average Effect | Bias | SD | RMSE | Coverage | Avg SE |
|---|---|---|---|---|---|---|
| Demeaning | 4.566 | 2.570 | 6.78 | 7.254 | 0.94 | 6.43 |
| Detrending | 2.161 | 0.165 | 2.37 | 2.380 | 0.95 | 2.26 |
| Detrending (HC3) | 2.161 | 0.165 | 2.37 | 2.380 | 0.91 | 2.60 |
| SC | 2.962 | 0.966 | 2.92 | 3.078 | 0.92 | 2.85 |
| SDiD | 2.615 | 0.619 | 2.35 | 2.435 | 0.94 | 2.33 |

DGP for regenerating (Section 5.1, Table 1): `C_i ~ N(0, σ_C²)`, `G_i ~ N(1, σ_G²)`, σ_C = 2, σ_G = 1; AR(1) errors `U_i1 ~ N(0, sqrt(2/(1−ρ²)))`, `U_it = ρ·U_{i,t−1} + A_it`, `A_it ~ N(0,2)`, ρ = 0.75; `Y_it(0) = λ_t·fs_t − C_i + G_i·t + U_it`, `Y_it(1) = Y_it(0) + δ_t·fs_t + V_it`, `V_it ~ N(0, √2)`; logistic assignment `Pr(D_i = 1) = 1[α_0 − α_1·C_i + α_2·G_i + A_i > 0]`, `A_i ~ Logistic(0,1)`; treatment-rule parameters (α_0, α_1, α_2): Scenario 1 = (−1, −1/3, 1/4), Scenario 2 = (−1.5, 1/3, 1/4), Scenario 3 = (−2, 1/3, 1/4); time FE (λ_1..λ_20) = (0, 0, 0, 0, 0.2, 0.6, 0.7, 0.8, 0.6, 0.9, 0.9, 1, 1.1, 1.3, 1.2, 1.5, 0.6, 1.4, 1.8, 1.9); treatment effects (δ_11..δ_20) = (1, 2, 3, 3, 3, 2, 2, 2, 1, 1); δ_t = 0 for t < 11. Demeaning is badly biased here because the DGP has unit-specific trends `G_i·t` correlated with assignment.

**SC unit weights (Figure 4, p22, 38-state pool):** positive weights on only six donor states - Nevada (0.31), Montana (0.30), Utah (0.28), Connecticut (0.06), Colorado (0.03), New Hampshire (0.02). Useful for validating an SC comparison harness.

---

## Gaps and Uncertainties

**As-printed internal inconsistencies in the paper (flagged by the second extractor; carried verbatim):**

1. (p22 vs Table 3, p23): the p22 text says the estimated effect grows over time, "starting off small but and reaching -0.403 (t = -1.52) by the year 2000" - but Table 3's tau_2000 for Procedure 3.1 is -0.403 with SE 0.152, which gives t ≈ -2.65, not -1.52. Extractor note as printed: "the -0.403 with SE 0.152 in Table 3 gives t ≈ -2.65; the t = -1.52 in the text appears to refer to a different specification or is possibly an inconsistency in the working paper - transcribed as printed."
2. (p17 vs Table 2, p18): the p17 text says "the SD for SC is 1.73, while the average standard error is 2.66," which does not match Table 2's SC row for Scenario 1 (SD 2.14, Avg SE 3.27); transcribed as printed, likely a working-paper inconsistency - the 1.73/1.74 pair belongs to the Detrending row.

**Contradictions between extraction files (both resolved against the PDF, 2026-07-11):**

3. Working-paper date: RESOLVED - the PDF cover page (p1) reads "February 3, 2026". The citation above uses that date. (An earlier metadata source said January 3; that was wrong.)
4. Section numbering for the closing material: RESOLVED - the PDF contains Section 8.1 (choosing the pre-treatment window, p30), Section 8.2 "Clustering and Spatial Correlation with Larger Cross Sections" (pp30-31), and Section 9 "Concluding Remarks" (pp31-32). The first extraction's "Section 8 (conclusion)" description was an inference from the intro, not a read of those pages. Note: Section 9's opening sentence cites the companion transformation paper as "Lee and Wooldridge (2025)" - the authors' own year handle for SSRN 4516518.

**PR #588 cross-check result (no discrepancy):** Table 3 as extracted shows Procedure 2.1 (demeaning) Average Effect = -0.422 and Procedure 3.1 (detrending) Average Effect = -0.227, matching PR #588's claimed reproduction exactly. No flag needed.

**Other gaps / uncertainties:**

- The paper contains NO numbered theorems, propositions, or lemmas on pages 1-19; formal results are inline distributional claims (eq. 2.10 and variants) plus Procedures 2.1 and 3.1. The formal CLM underpinnings invoked for the staggered "exact theory" claims (p28) are the Section 2 assumptions; no separate staggered-case theorem is stated.
- Randomization-inference mechanics are described only briefly (p7: two-sided p = c/N with N = number of permutations; note the symbol overload with sample size N). No further RI mechanics are given in the later page range; the permutation scheme (which units are permuted, whether the full assignment vector is permuted) is not detailed in the extractions.
- Table 3, Note 2 attributes the 0.021 / 0.020 p-values to "Procedure 3.1" without stating explicitly that they refer to the Average Effect row (the natural reading, consistent with its ** star).
- Tables 3, 4, and A1 report per-period effects only for 1989, 1995, 2000; the remaining 9 post-period estimates (and all SC/SDiD per-period cells) are not available from the extractions.
- Pages 33-34 (references) were skipped per extraction instructions; the exact reference-list entries are not transcribed.
- Figures 1-6 and A1 are described qualitatively only; no underlying series values are available for pinning plot-level goldens.
- The default behavior of the Stata `lwdid` command (e.g., default SE type, default transformation) is not described beyond it implementing Procedures 2.1/3.1 and offering the `ri` option; parity testing against `lwdid` will need its documentation or source.
- The Hagemann (2025) comparison (single treated cluster, unequal variances) is mentioned only in passing (p7); how its variance-heterogeneity robustness relates to the homoskedasticity requirement (2.9) is not elaborated.
- Guidance thresholds are qualitative: HC3 requires "at least a handful of treated units" / N "not too small" - no numeric cutoff is given.
