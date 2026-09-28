# Paper Review: An Exact and Robust Conformal Inference Method for Counterfactual and Synthetic Controls

**Authors:** Victor Chernozhukov, Kaspar Wüthrich, Yinchu Zhu
**Citation:** Chernozhukov, V., Wüthrich, K., & Zhu, Y. (2021). "An Exact and Robust Conformal Inference Method for Counterfactual and Synthetic Controls." *Journal of the American Statistical Association*, 116(536), 1849–1864.
**PDF reviewed:** https://doi.org/10.1080/01621459.2021.1920957 (JASA; arXiv:1712.09089 v10). PDF is 102 pages: main body printed pp. 1–31, references pp. 31–35, then a ~60-page online supplemental appendix.
**Review date:** 2026-05-29

> Scope note: this paper provides an **inference layer** (valid p-values and confidence intervals) for synthetic-control and related counterfactual estimators — it is the basis for the planned conformal-inference path (PR-3) on top of the classic `SyntheticControl` estimator. It is **not** a new point estimator. Everything below is sourced from this paper; the canonical SC estimator it nests is the constrained-least-squares form (its own §2.3), which differs from the classic ADH V-matrix estimator in ways flagged under Gaps.

---

## Methodology Registry Entry

*Formatted to match docs/methodology/REGISTRY.md. This documents the conformal-inference layer for `## SyntheticControl`.*

## SyntheticControl — Conformal Inference (Chernozhukov-Wüthrich-Zhu)

**Primary source:** Chernozhukov, V., Wüthrich, K., & Zhu, Y. (2021). "An Exact and Robust Conformal Inference Method for Counterfactual and Synthetic Controls." *JASA*, 116(536), 1849–1864. https://doi.org/10.1080/01621459.2021.1920957

**Key implementation requirements:**

*Setting & notation (Section 1, 2.1):*
- One treated unit `j=1`, observed for `T0` pre-intervention periods and `T_* = T − T0` post periods (`T_*` typically **short** relative to `T0`). `J ≥ 1` control units `j=2,…,J+1`, observed all `T` periods; optional covariates `X_jt`. (Multiple treated units → Appendix A.2.)
- Potential outcomes `Y^I_{1t}`, `Y^N_{1t}`; policy effect `θ_t = Y^I_{1t} − Y^N_{1t}`. Control outcomes equal their no-intervention values: `Y_jt = Y^N_jt`, `j≥2`.

*Counterfactual Model (Assumption 1, "CMF" — the fundamental identifying assumption):*

    Y^N_{1t} = P^N_t + u_t
    Y^I_{1t} = P^N_t + θ_t + u_t        E(u_t) = 0,   t = 1,…,T

where `{P^N_t}` is a **mean-unbiased proxy** for the counterfactual (`E[P^N_t] = E[Y^N_{1t}]`), and `{u_t}` is a centered stationary stochastic process **whose distribution is invariant under the intervention**. Observed: `Y_{1t} = Y^N_{1t} + D_t(Y^I_{1t} − Y^N_{1t})`, `D_t = 1{t>T0}`. No restriction on the dependence between `{P^N_t}` and `{u_t}`.

*Hypotheses (Section 2.2):* the **sharp null** over the post-period trajectory `θ = (θ_{T0+1},…,θ_T)'`:

    H0:  θ = θ0 = (θ0_{T0+1}, …, θ0_T)'        (eq. 1)

It fully pins the counterfactual: under H0, `Y^N_{1t} = Y_{1t} − θ0_t` for `t>T0`. (Per-period `H0: θ_t = θ0_t` is used for CIs; average-effect nulls in Appendix A.1.)

*Algorithm (the conformal test):*
1. **Build data under the null** `Z(θ0)`: subtract `θ0_t` from the **post-period** treated outcomes (pre-period unchanged); keep controls and covariates.

       Z_t = (Y^N_{1t}, Y_{2t},…,Y_{J+1,t}, X'_{·t})'        t ≤ T0
       Z_t = (Y_{1t} − θ0_t, Y_{2t},…,Y_{J+1,t}, X'_{·t})'   t > T0

2. **Estimate the proxy `P̂^N_t` UNDER THE NULL on ALL `T` periods** (not pre-period only) using any nested estimator (SC, constrained Lasso, DiD, factor/MC, …). *Estimating under the null is essential for good small-sample size and for exact validity.*
3. **Residuals:** `û_t = Y^N_{1t} − P̂^N_t`, `t = 1,…,T` (`Y^N_{1t}` here means the null-imputed value).
4. **Test statistic** (high → reject):

       S_q(û) = ( (1/√T_*) · Σ_{t=T0+1}^{T} |û_t|^q )^{1/q}

   `q=1` (S1) is the application default (robust to heavy tails); `q=2` (S2) for permanent effects; `q=∞` for large temporary effects. For an **average** effect, `S(û) = T_*^{-1/2} |Σ_{t>T0} û_t|` (Remark 1).
5. **Permutation p-value (eq. 2):**

       p̂ = 1 − F̂(S(û)),   F̂(x) = (1/|Π|) · Σ_{π∈Π} 1{ S(û_π) < x }

   where `û_π = (û_{π(1)},…,û_{π(T)})'`. *Footnote 7:* if the proxy estimator is **invariant to time permutations of the data** (true for SC, constrained Lasso, DiD, factor/PCA, matrix completion — NOT for AR/time-series proxies), then permuting residuals ≡ permuting data, so the proxy is fit **once** and only residuals are permuted.

*Permutation schemes (`Π`):* (always includes the identity)
- **i.i.d. permutations `Π_all`** — all `T!` permutations of `{1,…,T}`; use when `{u_t}` is i.i.d. (Assumption 2.1). Gives precise p-values / low significance levels; sample randomly (e.g., 10,000 draws) if `T!` is large.
- **Moving-block permutations `Π_→`** — `T` **cyclic shifts** (wrap-around), indexed `j=0,…,T−1`:

      π_j(i) = i + j           if i + j ≤ T
      π_j(i) = i + j − T       otherwise

  use when `{u_t}` is stationary, strongly mixing (Assumption 2.2; ARMA/GARCH). `|Π_→| = T`.
- (i.i.d. block permutations `Π_mb` over a partition into blocks — footnote 6; secondary.)

*Confidence intervals — Algorithm 1 (pointwise, by test inversion):*

    (i)   choose a fine grid Θ̃_t = {θ̃0_{1t},…,θ̃0_{Gt}} of candidate values
    (ii)  for each θ̃0_t: build Z under H0: θ_t = θ̃0_t, recompute p̂(θ̃0_t) via eq. (2)
    (iii) C_{1−α}(t) = { θ̃0_t ∈ Θ̃_t : p̂(θ̃0_t) > α }

**Cost:** one proxy re-fit per grid value (each `θ̃0_t` defines a different `Z`). One-sided variants and average/aggregate effects (collapse to non-overlapping `T_*`-blocks, effective sample `T/T_*`) in Appendix A.1.

*Synthetic-control proxy (Section 2.3; eqs. 3–4, 13):*

    P^N_t = Σ_{j=2}^{J+1} w_j Y^N_jt,   w ≥ 0,  Σ_{j=2}^{J+1} w_j = 1        (3)
    (SC)  E( u_t Y^N_jt ) = 0,  j = 2,…,J+1                                 (identification)
    ŵ = argmin_w  Σ_{t=1}^{T} ( Y^N_{1t} − Σ_{j} w_j Y^N_jt )^2   s.t. simplex   (4)

Eq. (4)'s objective sums over **all `t = 1,…,T`**, and **footnote 9** states it explicitly: *"unlike Doudchenko and Imbens (2016), we estimate `w` under the null hypothesis based on all the data"* — i.e. §2.3's canonical SC estimator is fit on the null-imputed `Z(θ0)` over **all `T` periods**, *not* pre-period-only (the classic-ADH convention). Covariates may be folded in (the ADH 2010/2015 versions), and the method also works with modified SC (e.g. augmented SC, Ben-Michael et al. 2018). **Constrained Lasso** (eqs. 5–6, 14) generalizes with an optional intercept `μ` and `‖w‖_1 ≤ K` (natural `K=1`); it is essentially **tuning-free** and **nests DiD** (`w_j = 1/J`) **and canonical SC** (`μ=0, w≥0`). General penalized form (§2.3.3) allows Lasso/Elastic-Net penalties toward a focal `w0`.

*Validity (two routes; finite-sample bounds → exact as `T0→∞`):*
- **Route (i) — consistency:** Assumption 3 (proxy MSE and pointwise error `→0` under the null) + Assumption 2 (`{u_t}` i.i.d. → `Π_all`, or stationary strongly-mixing → `Π_→`). **Theorem 1:** `|P(p̂ ≤ α) − α| ≤ C(δ̃_T + δ_T + √δ_T + γ_T)`, `δ̃_T = (T_*/T0)^{1/4} log T` (with `T_*` fixed). **Lemma 1** verifies Assumption 3 for constrained-LS/SC/Lasso, allowing `J` large (`log J = o(T^c)`) and requiring **no sparsity**.
- **Route (ii) — stability (misspecification-robust):** Assumption 4 (estimator stable under perturbing a few observations; `β̂` need NOT converge) + Assumption 5 (β-mixing data). **Theorem 2** bounds size with `Π_→`. Verified for constrained Lasso and Ridge (Appendices E–F).
- **Exact finite-sample validity under exchangeability** (Appendix D): imposing the null ⇒ permutation-invariant estimator ⇒ exchangeable residuals ⇒ exact size, **model-free** (e.g., DiD differencing makes residuals exchangeable even when data are not).

*Edge cases / conditions:*
- `T0` must be **large** (drives exactness); `T_*` may be small/fixed. Imposing the null is what rescues small-`T0` size (empirically excellent at `T0=19`).
- **Remark 2:** conditional heteroscedasticity in `{u_t}` is allowed; **unconditional** heteroscedasticity in `{u_t}` is **not** — apply an extra filter to get standardized residuals if suspected.
- If the **shock distribution changes** under the intervention (Assumption 1 invariance fails), the test becomes a test of "no impact whatsoever" (Appendix B, structural-break interpretation); or treat `θ_t` as random → valid **prediction sets** (Appendix C).
- For **time-series (AR) proxies**, residual permutation ≠ data permutation: estimate the AR parameters on residuals and permute the **innovations** (Lemmas 5–7).

*Placebo diagnostic (Appendix A.3):* an **in-time placebo** — test `H0: θ_{T0−τ+1}=⋯=θ_{T0}=0` on **pre-period data only**, treating the last `τ` pre-periods as a pseudo-post-period; rejection undermines credibility. Useful to compare credibility across DiD vs SC vs constrained Lasso. Cannot test Assumption 1's shock-invariance.

**Reference implementation(s):**
- Authors state "all computations were performed in R"; **no named package/repo** is cited in the article body. (Abadie 2021 refers to this as CWZ with available software; verify the package name separately if needed.)

**Requirements checklist (conformal layer for `SyntheticControl`):**
- [ ] Build `Z(θ0)` (subtract post-period `θ0`); estimate proxy **under the null on all `T` periods**.
- [ ] Residuals + `S_q` statistic (`q=1` default; expose `q∈{1,2,∞}`).
- [ ] Permutation p-value (eq. 2); both `Π_all` (i.i.d.; random sampling fallback) and `Π_→` (moving-block cyclic shifts) schemes.
- [ ] Pointwise CIs via Algorithm 1 (grid + test inversion); one-sided + average-effect (block-collapse) variants.
- [ ] Reuse residual-permutation shortcut only for permutation-invariant proxies (SC/Lasso/DiD); AR path needs innovation permutation.
- [ ] Document `T0`-large / `T_*`-small requirement and the unconditional-heteroscedasticity caveat.

---

## Implementation Notes

### Data Structure Requirements
- Single treated unit, block timing, balanced time series with **large `T0`**, short `T_*`; control outcomes (and covariates) over all `T`. Multiple treated units handled by per-unit application or cross-unit averaging (A.2).

### Computational Considerations
- **Fit-once** then permute residuals for SC/Lasso/DiD (footnote 7) → the p-value for a single null is cheap (`|Π|` statistic evaluations, no re-fit). `Π_→` has only `T` elements; `Π_all` is sampled (e.g., 10k draws).
- **CIs are the cost driver:** Algorithm 1 re-estimates the proxy once per grid value per period (each `θ̃0_t` changes `Z`). Warm-starting across the grid and bounding the grid are natural optimizations (this matches the plan's PR-3 "test-inversion cost" risk).

### Tuning Parameters

| Parameter | Type | Default | Selection |
|-----------|------|---------|-----------|
| `q` (statistic norm) | int/∞ | `1` (S1) | `1`/`2` permanent effects, `∞` large temporary |
| Permutation set `Π` | scheme | `Π_→` (moving block) if serial dependence; `Π_all` if i.i.d. | by error-dependence assumption |
| `K` (constrained Lasso ℓ1 bound) | float | `1` | tuning-free at `K=1`; SC ⇔ `K=1,w≥0,μ=0` |
| CI grid `Θ̃_t`, `α` | grid, level | fine grid | resolution vs cost |
| `R`, `k` (Theorem 2) | — | — | **theory only — NOT exposed** |

### Relation to Existing diff-diff Estimators
- This is the **PR-3 inference layer** for `SyntheticControl`. It yields what ADH placebo inference cannot: **valid p-values for the effect trajectory and pointwise confidence intervals**.
- Its **constrained-LS SC** (no `V`-matrix, weights on all periods under the null) differs from the classic ADH V-matrix estimator. The conformal *machinery* (build `Z(θ0)` → fit proxy under null → residuals → `S_q` → permute → invert) is estimator-agnostic; layering it onto the classic ADH SC is a design choice (see Gaps).
- The moving-block permutation + test-inversion CI is **greenfield** in `diff_diff` (no existing conformal code); the `1/(n+1)`-style discreteness of permutation p-values resembles `diagnostics.py`'s permutation p-value floor, but the residual-permutation/test-inversion mechanics are new.
- Constrained Lasso **nests DiD and SC**, mirroring the factor-model→DiD reduction noted in the ADH reviews.

---

## Gaps and Uncertainties

- **Which proxy to pair with the conformal layer.** CWZ's exact/robust guarantees are derived for its **constrained-LS SC weights estimated under the null on all `T` periods**. The classic ADH `SyntheticControl` (PR-1) uses **V-matrix weights on pre-period predictors only**. Applying CWZ conformal inference to the *classic* estimator means either (a) re-estimating ADH weights under the null on all periods per grid value (faithful to CWZ, but changes the estimator's weight semantics and is costly), or (b) treating ADH weights as the proxy and accepting that the exactness theory (Lemma 1) was proven for the constrained-LS form. **This is the central PR-3 design decision** and must be resolved against this paper + the planned estimator API.
- **No named software.** The article cites only "computations in R"; the implementing package name/repo is not in the body (Abadie 2021 references CWZ's software as available — locate separately if a reference implementation is wanted for parity).
- **Permutation-invariance requirement.** The fit-once shortcut (footnote 7) holds only for time-permutation-invariant proxies. If `SyntheticControl` ever incorporates time-ordered components, the residual-permutation equivalence breaks (use the AR-style innovation permutation, Lemmas 5–7).
- **Unconditional heteroscedasticity / non-stationarity of `{u_t}`** invalidates the basic procedure (Remark 2) — needs a standardizing filter; no default filter is prescribed.
- **CI grid specification** (`Θ̃_t` range/resolution) and the choice between `S1`/`S2`/`S∞` and `Π_all`/`Π_→` are left to the analyst; defaults must be chosen (application used `S1` + both schemes; `Π_all` sampled at 10k).
- **Appendix-level material** (proofs in Appendix H; stability sufficient conditions in E–F; simulation design in G) was summarized, not transcribed; consult the supplement if the exactness proof or stability conditions are needed verbatim during PR-3.
