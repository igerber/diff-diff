# Paper Review: Estimating Difference-in-Differences in the Presence of Spillovers

**Authors:** Damian Clarke
**Citation:** Clarke, D. (2017). Estimating Difference-in-Differences in the Presence of Spillovers. MPRA Paper No. 81604. https://mpra.ub.uni-muenchen.de/81604/
**PDF reviewed:** papers/MPRA_paper_81604.pdf
**Review date:** 2026-05-09

---

## Methodology Registry Entry

*Formatted to match docs/methodology/REGISTRY.md structure. Heading levels and labels align with existing entries.*

## SpilloverRobustDiD

**Primary source:** Clarke, D. (2017). Estimating Difference-in-Differences in the Presence of Spillovers. MPRA Paper No. 81604.

**Scope:** Difference-in-Differences estimation when SUTVA fails locally - i.e., when the treatment status of one unit (or one cluster) leaks into outcomes of nearby "close to treated" units. The estimator augments a standard two-period DD with one or more "close to treatment" indicators `R^k(i,t)` defined over distance bins `[(k-1)h, kh)` of an observable `X(i,t)` (geographic distance, network distance, ethnic distance, etc.). It separately recovers (i) the average treatment effect on the treated (`ATT = α`) and (ii) the average treatment effect on the close-to-treated (`ATC = β_k`), under monotonicity and a fixed bandwidth `h`. The optimal bandwidth `h*` and maximum spillover distance `d = kh` are chosen by data-driven RMSE-minimising leave-one-out (or k-fold) cross-validation. A multidimensional generalisation (Section 3.3) lets `R(i,t)` depend on a vector `X(i,t)` (e.g. distance interacted with vehicle ownership).

**Key implementation requirements:**

*Assumption checks / warnings:*
- Two-period panel `t in {0, 1}`. Treatment occurs only between periods; `D(i,0) = 0` for all `i`. Multi-period and staggered settings are NOT formally treated in this paper - the propositions are stated for the two-period case (Equation 1, Equation 7). For staggered designs, downstream applications add unit and time fixed effects to a binary `T_{it}` (see Equation 22 in the empirical example), but no theoretical results extend Proposition 1 to the staggered case.
- `D(i,t) = 1` and `R(i,t) != 0` are mutually exclusive (a unit cannot be both treated and "close to treated"). Implementations must enforce `R(i,t) = 0` whenever `D(i,1) = 1`.
- Assumption 1 (parallel trends in treatment and control): `E[Y^0(i,1) - Y^0(i,0) | D(i,1)=1, R(i,1)=0] = E[Y^0(i,1) - Y^0(i,0) | D(i,1)=0, R(i,1)=0]`.
- Assumption 2 (parallel trends in close and control): `E[Y^0(i,1) - Y^0(i,0) | D(i,1)=0, R(i,1)=1] = E[Y^0(i,1) - Y^0(i,0) | D(i,1)=0, R(i,1)=0]`.
- Assumption 3 (SUTVA holds for some units): there exists a non-empty subset `j in J subset N` for whom potential outcomes `(Y_j^0, Y_j^1)` are independent of `D = {0,1}` for all `i != j`. Practically: at least some "far" units must be unaffected by spillovers, otherwise `ATT` is not identified by this method (Footnote 5).
- Assumption 4A (assignment to close-to-treatment depends on observable `X(i,t)`): there exists a deterministic rule `δ(X(i,t)) = 1{X(i,t) < d}` with `d` a fixed scalar cutoff. Loosened in Section 3.3 to multidimensional `δ : X -> {0,1}`.
- Assumption 5 (monotonicity of spillovers in distance `X(i,t)`): the parameters on `R^k(i,t)` for all `k in 1,...,K` behave monotonely in distance. This is required for the bias arguments around Equation 19 to attenuate `|E[β_j^k]| <= |β_j|`.
- The propagation function need NOT be parametric (Proposition 1 says LS controlling "parametrically or non-parametrically" for `R(i,t) = 1{X(i,t) <= d}` is consistent), but the `R^k(i,t)` partition (Equation 11) IS the paper's parametric workhorse.
- Warn when no observations satisfy `D(i,t) = 1` AND `R(i,t) = 1` simultaneously (mutual exclusivity); warn when `N_{R_kT} = 0` for a candidate bin (no close-to-treated observations - that bin is uninformative).
- Warn when treated and close-to-treated zones overlap (e.g. proximity-based `X(i,t)` for treated units is undefined). Treated units have `R(i,t) = 0` by construction; this must be enforced upstream of regression.

*Estimating equation (Equation 7 - single binary `R`):*

    Y(i,t) = μ + τ·D(i,1) + γ·R(i,1) + δ·t + α·D(i,t) + β·R(i,t) + ε(i,t)

where:
- `D(i,t)`: treatment indicator (1 if treated in period `t`).
- `R(i,t)`: "close to treated" indicator (1 if untreated but lives within distance `d` of nearest treated unit in period `t`; `R(i,0) = 0` by construction).
- `μ`: intercept; `τ`: fixed effect for treated units; `γ`: fixed effect for close-to-treated units; `δ`: time trend.
- `α`: ATT (direct treatment effect).
- `β`: ATC (average treatment effect on the close-to-treated).
- `ε(i,t) = η(i) - E[η(i)|D(i,1), R(i,1)] + v(i,t)` with `E[ε(i,t) · {1, D(i,1), R(i,1), D(i,t), R(i,t)}] = 0` from Assumptions 2 and 5.

*Estimating equation generalised to `K` distance bins (Equation 12 - the operational form):*

    Y(i,t) = μ + τ·D(i,1)
           + γ_1·R^1(i,1) + ... + γ_K·R^K(i,1)
           + δ·t + α·D(i,t)
           + β_1·R^1(i,t) + ... + β_K·R^K(i,t) + ε(i,t)

Each `β_k` captures the spillover effect on units in distance bin `k`. The `γ_k` are pre-period close-fixed-effects.

*Distance-bin (exposure) construction (Equations 10-11):*

Given an observable distance `X_i` to the nearest treated unit and a bandwidth `h`, define mutually exclusive indicators

    R(i,t) = R^1(i,t) + R^2(i,t) + ... + R^K(i,t)                                  (10)

with

    R^k(i,t) = { 1   if X_i >= (k-1)·h  and  X_i < k·h         for all k in 1,...,K   (11)
               { 0   otherwise

`X_i` is treated as time-invariant in the empirical example (county-to-treated-state-border distance) but the framework allows time-varying `X(i,t)`. The summed indicator `R(i,t)` corresponds to the binary "close" variable in Equation 7; the matrix `R_M(i,t)` (full set of `K` indicators) corresponds to Equation 12.

*Kernel/weight choices (Section 3 and Section 4.1):*
- Clarke does NOT propose continuous kernels (Gaussian, Epanechnikov, exponential, inverse-distance) as the operational construction. The workhorse exposure mapping in Equations 10-11 is a STEPWISE / RING (donut) partition: contiguous, mutually-exclusive distance bins of fixed width `h`. Inside each ring the spillover effect `β_k` is constant; across rings it can vary monotonically.
- Section 3.3 discusses multidimensional exposure mappings via an assignment set `T = {x in X : δ(x) = 1}` (Equation 21), permitting interactions of distance with binary covariates (e.g. vehicle ownership). The functional form is "context-specific, ideally driven by economic theory" (page 17).
- Section 4.1 Monte Carlo Model 3 generates spillovers via an exponential function `γ · exp(-dist)` for `0 < dist <= 10` (page 20). The paper shows that the stepwise bin estimator still gives correct test size even when the true DGP is exponential (model misspecification robustness, Section 4.1, Table 1) - so the ring partition is the proposed estimator and continuous kernels appear only inside the simulation DGPs against which the estimator is evaluated.
- Adjacency / network / ethnic distance / message-strength: any univariate `X(i,t)` measure plugs into Equations 10-11. The Introduction (page 3) explicitly lists "euclidean space, ethnic distance, edges between nodes in a network, strength of messaging transmission, travel time" as legitimate `X` candidates.

**For the diff-diff Phase 3 implementation note:** the user-facing API (`spillover_kernel=`, `spillover_distance=`, `spillover_bandwidth=`) is consistent with this paper IF `spillover_kernel="ring"` is the default and continuous kernels (`"exponential"`, `"inverse_distance"`, `"adjacency"`) are documented as DEVIATIONS supported as engineering convenience. Clarke's paper does not endorse continuous kernels as the primary construction.

*Bandwidth `h` and maximum distance `d` selection (Section 3.2, Equation 20):*

Optimal bandwidth `h*` minimises the leave-one-out cross-validation criterion

    CV(h) = (1/N) · Σ_{i=1}^N ( Y_i - Ŷ*(X_i(h); h, θ̂_{-i}) )²                       (20)

with

    h*_{CV} = argmin_h CV(h)

`Ŷ*` depends explicitly on `h` because the matrix of regressors `X_i(h)` includes the `R^k(i,t)` indicators, which themselves depend on `h`. The procedure is:

1. For each candidate `h` in a discrete grid (e.g. `2km, 4km, ..., 40km` in the text-messaging example):
   a. Build `R^k(i,t)` for `k = 1, ..., K(h)` where `K(h) = ceil(max(X)/h)` or the user's chosen ceiling.
   b. Run the iterative procedure (Section 3.1, page 14) to determine the smallest `K` such that the marginal coefficient `β_K` is statistically zero.
   c. Compute `CV(h)` as the LOOCV RMSE of the chosen specification.
2. Pick `h* = argmin_h CV(h)`.

For large `N`, LOOCV is computationally infeasible; a `k`-fold variant (10-fold in the application, page 29) is recommended. Appendix Figure A2 / Appendix Table A1 documents that `k`-fold and LOOCV select identical `h*` in simulation; LOOCV reports lower RMSE values but identical argmin.

**Default bandwidth?** Clarke does NOT recommend a numerical default. The whole point of the cross-validated procedure is to remove the researcher's degree of freedom. Instead, the implementation should expose the search grid and a CV mode (LOOCV vs k-fold) and surface the chosen `h*` in the result object.

*Maximum spillover distance `d` (iterative procedure, Section 3.1, pages 13-14):*
1. Estimate Equation 7 with a single close-to-treatment indicator `R^1(i,t)` (and pre-period `R^1(i,1)`).
2. Test `H_0: β_1 = 0` vs `H_1: β_1 != 0`. The t-statistic is `t_{β̂_1^1} = (β̂_1^1 - β_1) / s.e.(β̂_1^1)` and is asymptotically standard normal under the null.
3. If rejected (spillovers present at distance `<= h`), augment with `R^2(i,t)` and test `H_0: β_2 = 0`. Continue until the marginal `β̂_{k+1}` fails to reject zero.
4. Conclude `d = k·h` where `k` is the last ring with rejected null. Equivalently, `R(i,t) = 1{X(i,t) <= k·h}`.

Appendix C ("Spillovers as a Nuisance Parameter") provides an alternative iterative procedure that compares successive treatment-effect estimates `α̂^k` rather than spillover coefficients `β̂_k`, using a Zellner (1962) seemingly-unrelated-regression Chi-squared test (`H_0: α^{k-1} = α^k`). This variant is appropriate when spillovers are nuisance parameters and only `α` is of empirical interest.

*Bias of naive DD that ignores spillovers (Equation 15):*

    Bias(α̂) = E[α̂|X] - α = -β · ( N_{R_T} / (N_T - N_{D_T}) )                         (15)

where `N_T` is observations with `t = 1`, `N_{D_T}` is treated observations at `t = 1`, and `N_{R_T}` is close-to-treated observations at `t = 1`. The bias is proportional to `β` (the spillover effect) times the fraction of the control group contaminated by spillovers. With `K` rings the bias generalises to (Equation 16):

    E[α̂|X] = α - Σ_{k=1}^K β_k · ( N_{R_kT} / (N_T - N_{D_T}) )                         (16)

When `j` rings are included and `K - j` are omitted, the residual biases on the included `α̂^j` and `β̂_j^k` are (Equations 18-19):

    Bias(α̂^k) = -β_{k+1} · ( N_{R_{k+1}T} / (N_T - N_{D_T} - Σ_{l=1}^k N_{R_lT}) ) - ... - β_K · ( N_{R_KT} / (N_T - N_{D_T} - Σ_{l=1}^k N_{R_lT}) )
    Bias(β̂_j^k) = same denominator, applied to omitted-ring coefficients

**Practical use (REGISTRY-relevant):** `Bias(α̂)` is signed; if spillovers `β_k` are same-sign as `α`, the naive estimator UNDERESTIMATES `|α|` (attenuation). If opposite-sign, it OVERESTIMATES `|α|`. Naive DD is unbiased in only two cases: (i) `β_k = 0` for all `k` (no spillovers); (ii) `N_{R_kT} = 0` for all `k` (no close-to-treated units exist).

*Identification (Proposition 1, page 9):*

> Under Assumptions 1 to 4A, the ATT and ATC can be consistently estimated by least squares when controlling, parametrically or non-parametrically, for `R(i,t) = 1{X(i,t) <= d}`.

Proof in Appendix B pages 44-45. Key identifying conditions beyond standard DD:
1. SUTVA holds on a non-empty subset (Assumption 3) - i.e. some "far" units are unaffected by spillovers. This replaces the standard DD requirement that SUTVA holds GLOBALLY.
2. Close-to-treatment status `R(i,t)` is determined by an observable rule on `X(i,t)` (Assumption 4A or 4B) - violations of SUTVA must be observable.
3. Close-to-treatment status is mutually exclusive with treatment.
4. Assumptions 1 and 2: parallel trends extend to BOTH treated and close-to-treated cohorts vs. far-untreated controls. Note that no parallel-trends assumption is needed BETWEEN treated and close-to-treated (Footnote 4) - they may have direct interactions.
5. Maximum spillover distance `d` is correctly identified (via the iterative procedure or CV) - misspecification of `d` propagates as omitted-variable bias per Equations 18-19.

Proposition 2 (page 17) generalises Proposition 1 to multidimensional `X` under Assumption 4B; the proof reduces to Proposition 1 once `δ(x) = 1_{x in T}` provides the close-to-treatment indicator.

*Decomposition: direct vs. indirect (spillover) effects:*

ATT (direct effect, Equation 8):

    ATT = E[Y^1(i,1) - Y^0(i,1) | D(i,1) = 1] = α

ATC (indirect/spillover effect on close-to-treated, Equation 9):

    ATC = E[Y^1(i,1) - Y^0(i,1) | R(i,1) = 1] = β        (single-bin form)
        = β_k                                              (for the `k`-th distance bin under Equation 12)

These are recovered by the SAME OLS regression of Equation 12. There is no separate identifying step for `α` vs. `β_k` - both are read directly off the coefficient vector, exploiting the mutual exclusivity of `D(i,1)` and `R^k(i,1)` in the design matrix.

For diff-diff's Phase 3 result object: `result.direct_effect = α` (coefficient on `D(i,t)`) and `result.spillover_effect` should expose the FULL VECTOR `[β_1, ..., β_K]` (Equation 12), not a single scalar `β`. A scalar `result.spillover_effect = β` is only correct when the true `K = 1`, which is the case the binary Equation 7 covers but is the EXCEPTION rather than the typical operational specification. The empirical illustration uses `K = 4` (Table 2) and `K in {1, 2}` (Table 3, depending on ban type).

*Standard errors (paper-level recommendation):*
- Clarke does NOT propose Conley (1999) HAC standard errors as the primary inference tool. Equation 22 (the empirical specification) and Tables 2-3 use STATE-LEVEL CLUSTERED standard errors (page 26: "Standard errors are clustered by state, and observations are weighted by county population").
- The Conclusion (page 30) cites Bertrand, Duflo and Mullainathan (2004), Cameron, Gelbach and Miller (2008), and Cameron and Miller (2015) as the inference literature relevant to DD - all CLUSTER-ROBUST, not Conley HAC.
- For the spillover-augmented spec, two-way clustering (state × time) or unit + time clustering is sensible because the close-to-treated units in a state are spatially correlated by construction. The paper does not formalise this; it is an implementation choice.
- Conley (1999) HAC SE on top of Clarke-style exposure regressors is consistent with the spirit of the paper (spatial correlation in residuals after controlling for spillovers) but is NOT what Clarke himself recommends. **For diff-diff Phase 3:** if Conley SE is offered alongside the Clarke exposure regressor, it should be advertised as a COMPLEMENTARY method (Phase 1 inference) rather than Clarke's prescription. Document this clearly.
- The iterative test of `β_{k+1} = 0` uses the standard t-statistic with cluster-robust SE (page 14, no explicit small-sample correction).
- The `α^{k-1} = α^k` SUR test in Appendix C uses the Zellner (1962) Chi-squared distribution.

*Edge cases:*
- **No close-to-treated units (`N_{R_kT} = 0` for all k):** naive DD is unbiased per Equation 15. Detection: count observations satisfying `R(i,t) = 1` post-period; if zero, fall back to standard DD with a warning.
- **No far units (Assumption 3 fails):** ATT is NOT identified by this method (Footnote 5). Detection: if the iterative procedure runs to `K = K_max` without ever failing to reject `β_k = 0`, the maximum spillover distance has not been bounded. Emit error.
- **Treated and close-to-treated zones overlap:** mathematically excluded (`D(i,1) = 1 implies R(i,1) = 0`). If user data has overlap (e.g. a county that is both treated and within `h` of another treated state), enforce `R = 0` on `D = 1` units and warn.
- **Non-monotonic spillovers (Assumption 5 failure):** the bias bounds `0 <= |E[β̂_j^k]| <= |β_j|` (page 14, Footnote 9) no longer hold; the iterative test of `β_{k+1} = 0` may falsely terminate. Clarke notes that monotonicity can be loosened to "spillovers do not fade out at a certain distance and then reappear at a greater distance" (page 12). Detection is hard in finite samples; document as a maintained assumption.
- **Multidimensional spillovers (Section 3.3):** the `R(i,t) = f(X_1, X_2)` parameterisation (e.g. Equation in Section 3.3) avoids the curse of dimensionality only when `f` is parametric. The paper uses `R(i,t) = X_1 · [β_{0,1} X_2^1 + ... + β_{0,K} X_2^K] + (1 - X_1) · [β_{1,1} X_2^1 + ... + β_{1,K} X_2^K]` - separate distance bins for binary `X_1 in {0,1}`. The iterative procedure runs separately on each `X_1` slice.
- **Sparse spillover region (small `N_{R_kT}`):** estimates of `β_k` are imprecise; the iterative test of `β_k = 0` may fail to reject the null even when true `β_k != 0` (Type II error). The CV procedure may select a smaller `h` than the true bandwidth in this regime. Page 22 (Footnote 12 on Table 1, Model 3): when spillovers reach 5% of the population, average `h*` underestimates the true 10-unit DGP cutoff.
- **Bandwidth grid does not bracket optimal `h`:** if `argmin CV(h)` is at the boundary of the grid, the search space is too narrow. Re-run with extended grid.
- **Time-varying `X(i,t)`:** the framework permits it (Assumption 4A is stated for `X(i,t)`), but the empirical example treats `X` as time-invariant. Implementation can support either; document that `R(i,0) = 0` is fixed by construction (treatment hasn't occurred yet), so time variation only affects `R(i,1)`.
- **Computational scaling for LOOCV with large N:** Section 3.2 (page 16) acknowledges `O(N²)` complexity if a vector `h_CV*` is searched (different `h` per iteration). Defaults: scalar `h` constant across iterations, k-fold (10-fold) substitute for LOOCV.

*Algorithm (Section 3.1 stepwise + Section 3.2 bandwidth optimisation):*
1. Build distance variable `X_i` from raw geographic / network coordinates. For geographic, "average distance from county to nearest treated state border" (page 25) is the recipe used in the empirical example. Treated units have `X_i = 0` and `R(i,t) = 0` by construction.
2. Choose bandwidth grid (e.g. `2km, 4km, ..., 40km`). For each `h`:
   a. Build `R^k(i,t)` for `k = 1, 2, ..., K_max(h)` per Equation 11.
   b. Initialise `K = 1`. Fit Equation 12 with `R^1(i,t)` only (plus pre-period `R^1(i,1)` fixed effects).
   c. Test `H_0: β_K = 0`. If rejected, set `K <- K + 1` and refit with the new ring; loop. Otherwise stop and record `K(h)`.
   d. Compute LOOCV (or k-fold) RMSE `CV(h)` with the chosen `K(h)` rings.
3. Select `h* = argmin_h CV(h)`.
4. Refit Equation 12 at `h = h*`, `K = K(h*)`. Output:
   - `α̂` = direct ATT (coefficient on `D(i,t)`).
   - `β̂_1, ..., β̂_K` = spillover/ATC by distance bin.
   - `d = K(h*) · h*` = maximum spillover distance.
   - Standard errors clustered (Clarke uses state-level in the empirical example).

*Algorithm variant (Appendix C - spillovers as nuisance):*
1. Run Step 2 above but compare successive `α̂^{k-1}` vs `α̂^k` instead of `β̂_k`.
2. Use Zellner (1962) SUR Chi-squared test for `H_0: α^{k-1} = α^k`.
3. Stop when the test fails to reject; report `α̂^k`.
4. This variant does NOT report individual `β_k`; appropriate when spillovers are uninteresting per se.

**Reference implementation(s):**
- Stata, Matlab, R: code at `https://github.com/damianclarke/cdifdif` (footnote 1 of paper). Companion command name: `cdifdif`.
- The implementation is described as "[automating] this methodology in various languages" (Conclusion, page 31). diff-diff's Phase 3 is the Python equivalent.

**Requirements checklist:**
- [ ] Two-period DiD baseline that supports an extra design-matrix column for `R^k(i,t)` indicators.
- [ ] Distance / network / arbitrary-`X(i,t)` input support: a `spillover_distance=` array (N x 1 for unidimensional, N x P for multidimensional).
- [ ] Ring/bin partition (`spillover_kernel="ring"` or equivalent) per Equations 10-11. This is the paper's PRIMARY construction.
- [ ] Bandwidth selection via LOOCV (small N) and k-fold CV (large N) per Equation 20. Expose `bandwidth_grid=`, `cv_folds=` (None for LOOCV).
- [ ] Iterative `β_K = 0` test loop with cluster-robust t-statistic (Section 3.1).
- [ ] Result fields: `direct_effect = α̂`, `spillover_effect = [β̂_1, ..., β̂_K]` (vector), `optimal_bandwidth = h*`, `max_spillover_distance = K · h*`, `cv_rmse = CV(h*)`.
- [ ] Cluster-robust SE by state / unit / two-way as the DEFAULT inference (Clarke's own choice).
- [ ] Mutual-exclusivity enforcement: warn / coerce if a unit appears in both `D(i,1) = 1` and `R(i,1) = 1` rows.
- [ ] Warning for `K_max` reached without test failure (Assumption 3 likely violated).
- [ ] Multidimensional `R(i,t) = f(X_1, X_2)` extension (Section 3.3) - low priority; document as future work for Phase 3.
- [ ] Optional: SUR Chi-squared test variant from Appendix C (`spillover_target="alpha_only"`).
- [ ] If continuous kernels (`exponential`, `inverse_distance`, `adjacency`) are exposed in the API, document them as Clarke-DEVIATIONS in REGISTRY.md - they are NOT in the paper's primary construction.
- [ ] If Conley (1999) HAC SE is exposed, document it as a Phase 1 inference complement, NOT as Clarke's prescription.

---

## Implementation Notes

### Data Structure Requirements
- Two-period panel: unit id, time id (`t in {0, 1}`), outcome `Y`, treatment `D in {0, 1}`. Treatment status is panel-invariant per unit-period within `t = 1`.
- Distance / exposure source: an `(N, 1)` array of distances (or higher-dimensional `(N, P)` for the Section 3.3 generalisation). Required at `t = 1`; can be time-varying. Treated units must have `R = 0`.
- Optional weights (the empirical example uses county-population weights, page 26).
- Recommended schema (sklearn-style):
    - `did = SpilloverRobustDiD(spillover_distance=X, bandwidth_grid=[2,4,...,40], cv_folds=10)`
    - `did.fit(data, formula="Y ~ D | unit + time")` -> result has `direct_effect`, `spillover_effect` (vector), `optimal_bandwidth`, `max_spillover_distance`.

### Computational Considerations
- Building the `R^k(i,t)` matrix: O(N) per bin per bandwidth candidate; O(N · K(h) · |H_grid|) total.
- LOOCV: O(N²) in the worst case (N regression refits per bandwidth). 10-fold CV reduces this to O(10 · N · |H_grid|).
- The iterative `β_K = 0` test adds a small constant factor per `K`.
- The empirical example uses `N = 149,328` observations (3,111 counties × 48 months) and reports 10-fold CV taking minutes (page 29 - computationally demanding given LOOCV was infeasible).
- For multidimensional spillovers (Section 3.3): curse of dimensionality. Don't expose this in the first pass.

### Tuning Parameters

| Parameter | Type | Default | Selection Method |
|-----------|------|---------|-----------------|
| `spillover_kernel` | string | `"ring"` (paper-faithful) | `"ring"` per Equations 10-11. Continuous kernels (`"exponential"`, `"inverse_distance"`, `"adjacency"`) are deviations - document accordingly. |
| `spillover_distance` | array `(N,)` or `(N, P)` | required | User input - geographic distance, network distance, etc. Must be non-negative and finite for non-treated units; ignored for treated units. |
| `spillover_bandwidth` | float or "auto" | `"auto"` (CV) | Equation 20 LOOCV or k-fold CV when `"auto"`. User-supplied scalar overrides CV (e.g. for replication). |
| `bandwidth_grid` | list[float] | data-derived | Default to a uniform grid spanning `[X.min(), X.max()]` with ~20 points; user-overridable. Paper uses `2km..40km` step `2km` for the empirical example. |
| `cv_folds` | int or None | 10 | None -> LOOCV; recommended 10-fold for `N > 1000`. |
| `iterative_test_alpha` | float | 0.05 | t-test significance level for `β_k = 0` rings. |
| `cluster_var` | string | `"unit"` | Cluster-robust SE; user can specify a different clustering var (state in the empirical example). Two-way `["unit", "time"]` accepted. |
| `max_iterations` | int | `len(bandwidth_grid)` | Cap to detect Assumption 3 failure. |

### Relation to Existing diff-diff Estimators
- `TwoWayFixedEffects` is the natural host for Phase 3. `SpilloverRobustDiD` is conceptually `TwoWayFixedEffects` + an extra group of design-matrix columns (`R^k(i,t)` and `R^k(i,1)` fixed effects) + a CV-based bandwidth selector + result-object spillover fields.
- `result.direct_effect = α̂` (scalar coefficient on `D(i,t)`) and `result.spillover_effect` is a `K`-length vector (or named dict by ring) - the multi-bin form is the operational case in Clarke's empirical example, NOT the scalar binary case.
- `Conley HAC` (Phase 1 deliverable) is COMPLEMENTARY but not Clarke-prescribed. The paper uses cluster-robust SE. Document Conley HAC as an alternative-inference layer.
- `CallawaySantAnna`, `SunAbraham`, `MultiPeriodDiD` - none of Clarke's results extend formally to staggered designs. The empirical example (Equation 22) uses `T_{im}` (binary contemporaneous treatment) with month and county fixed effects, but no propositions formalise the staggered case. **Phase 3 should ship two-period only**; staggered support requires a follow-up methodology contribution beyond this paper.
- `BaconDecomposition` and `SyntheticDiD` are unrelated.

### T22 tutorial design hints
The empirical example (Section 4.2, Aboukand Adams 2013 text-messaging bans) is a near-ideal tutorial:
- 49 states, 3,111 counties, 48 months (Jan 2007 - Dec 2010). Outcome: log fatal SVSO accidents + 1.
- Three ban types: strong (primary enforcement), weak (secondary enforcement), handheld. Treatment indicator switches on at state-month of enactment.
- Distance: county-centroid to nearest treated-state border (km); Section 4.2 also offers travel-time-over-roads.
- Optimal bandwidths: 30km (weak ban, 30km spillover distance), 6km (handheld ban, 12km spillover distance), no spillover (strong ban). Table 3 reports.
- Magnitudes: weak-ban ATT = 7.6%, ATC[0-30km] = 5.4%; handheld ATT = -7.7% (n.s.), ATC[0-6km] = -11.1%, ATC[6-12km] = -5.3%.
- Pre-trend testing: not formally extended in the paper (two-period model). The MC simulations in Section 4.1 evaluate test SIZE under correctly- and mis-specified spillover bins; that infrastructure can support a tutorial demonstration of selection-via-CV.

**Synthetic DGP for tutorial (Monte Carlo Model 1, Section 4.1):**

    y_{it} = α + β·T_{it} + Σ_{j=1}^{4} γ_j·close_{it,((j-1)×5,j×5]} + φ_t + λ_i + ε_{it}

with 5km bins, `θ = (β, γ_1, γ_2, γ_3, γ_4) = (10, 5, 4, 3, 2)`, `ε ~ N(0, σ)` for `σ in {1, 2, 5}`, treatment switched on for 20% of sample in period 2, spillovers reaching 5%/10%/25% of population. Naive DD recovers `β̂ ≈ 9.56` (10% spillover), spillover-robust DD recovers `β̂ ≈ 10.00`. Closed-form bias check (Footnote 12): `E[β̂] = 10 - 5(0.025/0.8) - 4(0.025/0.8) - 3(0.025/0.8) - 2(0.025/0.8) = 9.5625`, matching simulation to 2 decimal places.

### Critique / limitations Clarke himself acknowledges
1. **Two-period only.** No formal extension to staggered or multi-period DD (Section 2 specifies `t in {0, 1}`). The empirical example sidesteps this by using county-month panel data with `T_{it}` and standard FEs.
2. **Maintained Assumption 3 (some far units unaffected).** If all control units are spilled into, ATT is unidentified. The iterative procedure cannot detect this directly; it would simply run to `K_max` without failing to reject.
3. **Maintained Assumption 5 (monotonicity).** Loosened mildly (page 12) but still required. Non-monotonic spillovers (e.g. tipping-point effects) violate the iterative-stopping logic.
4. **Bin partition is arbitrary at small scales.** A bandwidth `h = 5km` forces all units in `[0, 5km)` to share the same `β_1`. The CV procedure picks `h*` by minimising prediction error, but does not address the fundamental discreteness of the bin model. Section 4.1 Models 2 and 3 (irregular and exponential DGPs) demonstrate the estimator's robustness, but the construction is still piecewise-constant.
5. **Multidimensional `X` requires functional-form choice (Section 3.3).** Nonparametric multivariate `R(i,t)` is not provided; user must specify a parametric form.
6. **Standard error inference is not a contribution of this paper.** The Conclusion (page 30) defers to Bertrand-Duflo-Mullainathan / Cameron-Gelbach-Miller / Cameron-Miller for cluster-robust SE; nothing is proven about valid inference under the specific spillover-augmented spec. The companion Phase 1 Conley (1999) HAC work fills part of this gap but is independent of Clarke.
7. **Application-specific spillover magnitudes are small in absolute terms.** Even when statistically significant, the spillover-corrected ATT (Table 3) is "not statistically distinguishable" from the naive ATT (Section 4.2, page 27) when spillovers reach a small fraction of the control group. The bias formula (Equation 15) explains this: small `N_{R_T} / (N_T - N_{D_T})` shrinks the bias regardless of `β`.

---

## Gaps and Uncertainties

1. **Inference under the iterative procedure.** Clarke uses a sequence of t-tests to determine `K`. Sequential testing inflates Type I error in principle (the chance that SOME ring fails to reject is higher than `α`). The paper does not propose a multiple-testing correction. The CV-based approach (Equation 20) is a partial remedy but is itself a model-selection procedure with no formal post-selection inference guarantee.
2. **Conley (1999) HAC vs cluster-robust.** Clarke chooses cluster-robust SE in the empirical example without explicit comparison to spatial HAC. For the diff-diff Phase 3 deliverable that BUNDLES Conley HAC (Phase 1) with the Clarke exposure regressor (Phase 3), there is no theoretical guarantee in either Conley (1999) or Clarke (2017) that the combination is valid. Both pieces are jointly used by the practitioner literature (e.g. Almond-Edlund-Palme 2009, cited at page 3) but a clean theoretical statement is absent. Document as a maintained-assumption in REGISTRY.md.
3. **Choice of distance metric.** The paper says distance can be "euclidean, ethnic, network, messaging-strength, travel-time" but does not provide guidance on which metric to choose for a given application. Practitioners must justify on substantive grounds.
4. **`d̲` for the bandwidth grid.** No formula or default for the grid. Practitioners default to `h_min = 2km` and `h_max ~= mean distance to nearest treated unit` based on the empirical example's `2..40 km` grid. The PYTHON IMPLEMENTATION should derive a sensible default grid from the data range and document the heuristic.
5. **Parallel trends in the spillover spec.** Assumptions 1 and 2 are stated for the binary `R(i,t)`. The K-bin generalisation (Equation 12) implicitly extends parallel trends to each `R^k(i,t)` group: `E[Y^0(i,1) - Y^0(i,0) | R^k(i,1) = 1] = E[Y^0(i,1) - Y^0(i,0) | far-control]` for every `k`. This is a STRONGER assumption than the binary case (parallel trends must hold separately at every distance). The paper does not flag this explicitly.
6. **Pre-trend testing in the spillover-augmented spec.** No formal pre-trend test is proposed. The iterative `β_K = 0` test is forward-looking (post-period spillovers) and does not test parallel-trends. Implementations should clearly distinguish "spillover-iteration test" (Clarke) from "event-study placebo test" (standard DD pre-trends).
7. **Multidimensional bandwidth grid (Section 3.3).** Curse of dimensionality is acknowledged but no remedy is proposed. Future-work flag.
8. **Heterogeneous spillover effects.** The paper assumes `β_k` is constant within ring `k`. Spillover heterogeneity by unit characteristic (e.g. urban vs rural counties, demographics) is not formalised; the multidimensional Section 3.3 example introduces ONE interaction (`X_1` binary). Practitioners interested in spillover heterogeneity beyond a single binary stratifier are on their own.
9. **Continuous kernel implementation choice.** If diff-diff Phase 3 ships `spillover_kernel="exponential"` for engineering convenience (Phase 3 plan mentions exponential, ring/donut, inverse-distance, adjacency), this DEVIATES from Clarke's proposed estimator. Justify in REGISTRY.md as `**Note (deviation from Clarke 2017):**` since exponential / inverse-distance are operational shortcuts for the multi-bin partition - not formally derived by the paper. The `"ring"` default is paper-faithful.
