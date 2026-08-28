# Paper Review: Semiparametric Difference-in-Differences with Potentially Many Control Variables

**Authors:** Neng-Chieh Chang
**Citation:** Chang, N.-C. (2020). Double/debiased machine learning for difference-in-differences models. *The Econometrics Journal*, 23(2), 177-191. https://doi.org/10.1093/ectj/utaa001
**PDF reviewed:** https://arxiv.org/pdf/1812.10846v3 (63 pages; SHA-256 `23e92b74c393e17e6f66d362536b48deb4b76a76149af72ab8ac9b070a362757`)
**Review date:** 2026-08-22

> **Numbering convention.** This review was conducted against **arXiv:1812.10846v3** (the
> working-paper version, titled "Semiparametric Difference-in-Differences with Potentially
> Many Control Variables"), which was published as the *Econometrics Journal* article above.
> **All equation, theorem, lemma, assumption, figure, and page references below are pinned to
> the arXiv v3 layout**, not the published article. Cross-check against the published version
> before citing section/equation numbers in user-facing docs.

---

## Methodology Registry Entry

*Formatted to match docs/methodology/REGISTRY.md structure. Heading levels and labels align with existing entries — copy the `## DMLDiD` section into the appropriate category in the registry.*

## DMLDiD

**Primary source:** Chang, N.-C. (2020). Double/debiased machine learning for difference-in-differences models. *The Econometrics Journal*, 23(2), 177-191. https://doi.org/10.1093/ectj/utaa001 (reviewed as arXiv:1812.10846v3; numbering below follows the arXiv version)

The paper extends Abadie (2005)'s semiparametric DiD to settings where the control-variable
dimension `d` may exceed the sample size `N`, by constructing **Neyman-orthogonal scores**
(each = Abadie's score + a mean-zero adjustment term) and estimating with the
**Chernozhukov et al. (2018) DML cross-fitting algorithm (DML2 variant)**. Three data
structures are covered, each with its own score:

1. **Case 1 — repeated outcomes (panel):** observe `{Y_i(0), Y_i(1), D_i, X_i}` — the primary implementation target.
2. **Case 2 — repeated cross sections:** observe `{Y_i, D_i, T_i, X_i}` with `T_i` the post-period indicator, `Y_i = Y_i(0) + T_i (Y_i(1) - Y_i(0))`.
3. **Case 3 — multilevel treatment:** `W ∈ {0, w_1, ..., w_J}`, a separate ATT `θ_0^w` per level `w` against the never-treated `W = 0`.

**Key implementation requirements:**

*Assumption checks / warnings:*
- **Conditional parallel trends (Assumption 2.1, Abadie 2005):** `E[Y^0(1) - Y^0(0) | X, D = 1] = E[Y^0(1) - Y^0(0) | X, D = 0]`. Untestable; the new estimators add **no identification assumptions beyond Abadie (2005)** (Section 2, p. 8).
- **Overlap (Assumption 2.2):** `P(D = 1) > 0` and `P(D = 1 | X) < 1` a.s. Regularity Assumptions 3.1(a)/3.2(a) strengthen this to **strict overlap**: `Pr(κ ≤ g_0(X) ≤ 1 - κ) = 1` for some fixed `κ > 0`, and the nuisance realization set `T_N` imposes the same band on the **estimated** propensity score (`‖ĝ - 1/2‖_{P,∞} ≤ 1/2 - κ`, pp. 28, 37). The theory does not cover fitted propensities approaching 0 or 1 — an implementation must clip/trim ĝ to `[κ, 1 - κ]` or refuse (a deviation to document; the paper gives no trimming rule).
- **Repeated cross sections sampling (Assumption 2.3):** conditional on `T = 0` (resp. `T = 1`), data are i.i.d. draws from the distribution of `(Y(0), D, X)` (resp. `(Y(1), D, X)`).
- **Case 3 versions** (p. 7): Assumptions 2.1-2.2 hold for each treatment level `w`: `E[Y^0(1) - Y^0(0) | X, W = w] = E[Y^0(1) - Y^0(0) | X, W = 0]`; `P(W = w) > 0` and `P(W = w | X) < 1` a.s. **Caution — the printed condition is not sufficient:** with more than one positive treatment level, `P(W = w | X) < 1` does NOT imply `P(W = 0 | X) > 0`, yet Equation 3.3 and the appendix algorithm divide by `g_0z(X) = P(W = 0 | X)` (and fit `ℓ_30` on the `W = 0` subsample). The paper prints no Case 3 regularity assumptions (p. 13: multilevel results "can be proven using the same arguments"); by analogy with Assumptions 3.1(a)/3.2(a) an implementation must additionally require and enforce strict comparison-group overlap `P(W = 0 | X) ≥ κ_z > 0` on the support of each analyzed arm `w` (clip/refuse on fitted `ĝ_z` accordingly), require positive `W = w` and `W = 0` counts in every fold and auxiliary sample, and define `p̂_w` as the sample mean of `I(W = w)`. Not stated in the paper — an implementation-required condition; see Gaps.
- **First-stage rate condition (Assumptions 3.1(f)/3.2(h) + Theorem 1):** BOTH conditions hold jointly — the bundle envelope `‖η̂_k - η_0‖_{P,2} ≤ ε_N` with `ε_N = o(N^{-1/4})` (so **each** nuisance component must meet the `o(N^{-1/4})` rate; a fast learner cannot compensate a slow one under the printed assumptions), AND additionally the product bound `‖ĝ - g_0‖²_{P,2} + ‖ĝ - g_0‖_{P,2}·‖ℓ̂ - ℓ_0‖_{P,2} ≤ ε_N²`.
- Two periods only; treatment occurs only at `t = 1` (`D_i(0) = 0` for all `i`).

*Identification (Abadie 2005, restated; Equations 2.1-2.3 in paper):*

Target: ATT `θ_0 := E[Y^1(1) - Y^0(1) | D = 1]` (Cases 1-2); `θ_0^w := E[Y^w(1) - Y^0(1) | W = w]` (Case 3).

    (2.1)  θ_0   = E[ (Y(1) - Y(0)) / P(D=1) · (D - P(D=1|X)) / (1 - P(D=1|X)) ]
    (2.2)  θ_0   = E[ (T - λ_0) / (λ_0(1-λ_0)) · Y / P(D=1) · (D - P(D=1|X)) / (1 - P(D=1|X)) ],   λ_0 := P(T=1)
    (2.3)  θ_0^w = E[ (Y(1) - Y(0)) / P(W=w) · ( I(W=w)·P(W=0|X) - I(W=0)·P(W=w|X) ) / P(W=0|X) ]

The conventional plug-in of (2.1) fails with ML first stages (p. 8): the score has a non-zero
Gateaux derivative in `g_0`, and ML nuisances converge slower than `N^{-1/2}` due to
regularization bias.

*Estimator equation — orthogonal scores (Equations 3.1-3.3 in paper; Cases 1-2 SHIPPED as the staggered `DMLDiD` estimator, `panel=True`/`False` — see REGISTRY.md "DMLDiD"; Case 3 remains unimplemented):*

Each score = Abadie's score + a mean-zero adjustment term (`c_1`, `c_2`, `c_w`), so the same ATT is identified.

Case 1, repeated outcomes (Equation 3.1):

    ψ_1(W, θ_0, p_0, η_10) = (Y(1) - Y(0)) / p_0 · (D - g_0(X)) / (1 - g_0(X)) - θ_0
                             - [ (D - g_0(X)) / ( p_0 (1 - g_0(X)) ) ] · ℓ_10(X)

Case 2, repeated cross sections (Equation 3.2):

    ψ_2(W, θ_0, p_0, λ_0, η_20) = (T - λ_0) / (λ_0(1-λ_0)) · Y / p_0 · (D - g_0(X)) / (1 - g_0(X)) - θ_0 - c_2,
    c_2 = [ (D - g_0(X)) / ( λ_0(1-λ_0) · p_0 · (1 - g_0(X)) ) ] · ℓ_20(X)

Case 3, multilevel treatment, for each w (Equation 3.3):

    ψ_w(W, θ_w0, p_w0, η_w0) = (Y(1) - Y(0)) / p_w0 · ( I(W=w)·g_0z(X) - I(W=0)·g_0w(X) ) / g_0z(X) - θ_w0 - c_w,
    c_w = [ ( I(W=w)·g_0z(X) - I(W=0)·g_0w(X) ) / ( p_w0 · g_0z(X) ) ] · ℓ_30(X)

where:
- `p_0 = P(D = 1)` — unconditional treated share (finite-dimensional nuisance, sample mean)
- `λ_0 = P(T = 1)` — post-period sampling share (Case 2 only, finite-dimensional, sample mean)
- `g_0(X) = P(D = 1 | X)` — propensity score (infinite-dimensional)
- `ℓ_10(X) = E[Y(1) - Y(0) | X, D = 0]` — control-group outcome-trend regression (Case 1)
- `ℓ_20(X) = E[(T - λ_0) Y | X, D = 0]` — Case 2 outcome nuisance
- `p_w0 = P(W = w)`; `g_0w(X) = P(W = w | X)`; `g_0z(X) = P(W = 0 | X)`; `ℓ_30(X) = E[Y(1) - Y(0) | X, I(W=0) = 1]` (Case 3)
- `η_10 = (g_0, ℓ_10)`, `η_20 = (g_0, ℓ_20)`, `η_w0 = (g_0w, g_0z, ℓ_30)` — infinite-dimensional nuisance bundles

**Lemma 1** (Section 3.2.1, p. 13): scores (3.1)-(3.3) are Neyman-orthogonal — with respect
to the **infinite-dimensional nuisances only** (a deliberate difference from Chernozhukov et
al. 2018); the finite-dimensional nuisances `p_0`, `λ_0` are handled instead by correction
terms in the variance estimator.

*Asymptotics (Theorem 1, p. 14):* under Assumptions 2.1-2.2 + 3.1 (Case 1) or 2.1-2.3 + 3.2
(Case 2), and `ε_N = o(N^{-1/4})`:

    √N (θ̃ - θ_0) → N(0, Σ),   Σ = Σ_10 (Case 1) or Σ_20 (Case 2)

Even when first stages are slower than `N^{-1/4}`, `θ̃` still has smaller bias than the
plug-in because orthogonality removes the first-order bias (p. 15). Influence-function form
(proof of Theorem 1, pp. 31, 39):

    √N(θ̃ - θ_0) = N^{-1/2} Σ_i [ ψ_1(W_i, θ_0, p_0, η_10) + G_1p0 (D_i - p_0) ] + o_P(1)                            (Case 1)
    √N(θ̃ - θ_0) = N^{-1/2} Σ_i [ ψ_2(W_i, θ_0, p_0, λ_0, η_20) + G_2p0 (D_i - p_0) + G_2λ0 (T_i - λ_0) ] + o_P(1)   (Case 2)

with `G_1p0 = E_P[∂_p ψ_1(·)]`, `G_2p0 = E_P[∂_p ψ_2(·)]`, `G_2λ0 = E_P[∂_λ ψ_2(·)]`.

*Kernel first stages — small bias property (Theorems 3-4, Assumption 3.3, pp. 16-17):*
with classical kernel first stages (Assumption 3.3(1): kernel `K(u)` of order `m`,
`s`-times differentiable with bounded derivatives, **zero outside a bounded set** — i.e.
compactly supported; smoothness `s`, `d = dim(X)`,
`inf_x f_0(x) > 0`, `log N / (√N h^{d+2s}) → 0`), the orthogonal estimator needs only
`√N h^{2m} → 0` — second-order bias `h^{2m}` instead of Abadie's `h^m` — so **no
undersmoothing is required**: the CV/MSE-optimal bandwidth `h = N^{-1/(d+2s+2m)}` is valid
for `θ̃` (feasible if `2m > d + 2s`) but invalid for Abadie's plug-in (Figure 2). Theorem 4:
the same variance estimators remain consistent under kernel first stages.

*Standard errors (Theorems 2 and 4, pp. 14-15, 17; proof pp. 46-56):*
- Default: **analytical plug-in variance from the augmented (corrected) score** — the only inference method in the paper; no bootstrap is proposed. SE = `sqrt(Σ̂ / N)` with normal critical values.
- Case 1 (repeated outcomes):

      Σ̂_1 = (1/K) Σ_{k=1}^K E_{n,k}[ ( ψ_1(W, θ̃, p̂_k, η̂_1k) + Ĝ_1p (D - p̂_k) )² ],   Ĝ_1p = -θ̃ / p̂_k

  Equivalently (proof of Theorem 2, p. 47) the p-correction folds **exactly** into the combined score

      ψ̄_1(W, θ, p, η_1) := (1/p) · (D - g(X))/(1 - g(X)) · (Y(1) - Y(0) - ℓ_1(X)) - Dθ/p,
      Σ̂_1 = (1/K) Σ_k E_{n,k}[ ψ̄_1(W, θ̃, p̂_k, η̂_1k)² ],   E_P[ψ̄_1(·, θ_0, p_0, η_10)²] = Σ_10

  Implementations can use the augmented score directly or add `Ĝ_1p (D - p̂_k)` — algebraically identical.
- Case 2 (repeated cross sections) — needs **BOTH** correction terms:

      Σ̂_2 = (1/K) Σ_{k=1}^K E_{n,k}[ ( ψ_2(W, θ̃, p̂_k, λ̂_k, η̂_2k) + Ĝ_2p (D - p̂_k) + Ĝ_2λ (T - λ̂_k) )² ],
      Ĝ_2p = -θ̃ / p̂_k;   Ĝ_2λ = consistent estimator of G_2λ0 (only consistency needed, no rate)

  The p-correction folds into `ψ̄_2 = 1/(λ(1-λ)p) · (D-g)/(1-g) · ((T-λ)Y - ℓ_2(X)) - Dθ/p + G_2λ(T-λ)`, but the λ-correction stays as an explicit extra term inside the square. Closed form for the λ-slope (from the |G_2λ0| display, p. 55; sample analogue with cross-fitted nuisances gives Ĝ_2λ):

      G_2λ0 = E_P[ -(1-2λ_0)/(λ_0²(1-λ_0)² p_0) · (D-g_0)/(1-g_0) · ((T-λ_0)Y - ℓ_20)
                   - Y/(λ_0(1-λ_0) p_0) · (D-g_0)/(1-g_0) ]

  **Omitting the λ-correction term is a plausible implementation bug the proof structure warns against.**
- Squaring the bare `ψ` alone is NOT the estimator — the augmentation terms account for the estimated finite-dimensional nuisances `p_0`, `λ_0` (which sit outside the orthogonality).
- No degrees-of-freedom or small-sample correction: plain `1/n` fold averages, `1/K` across folds.
- Clustering: not discussed in the paper (i.i.d. sampling assumed).
- Multilevel case: Theorem 2/4 as printed cover only `Σ̂_1`, `Σ̂_2`; the multilevel variance estimator is not printed in the extracted range (by analogy the correction would be `-θ̃_w / p̂_w · (I(W=w) - p̂_w)`, but that formula is NOT in the paper text reviewed — see Gaps).
- Consistency: `Σ̂_1 = Σ_10 + o_P(1)`, `Σ̂_2 = Σ_20 + o_P(1)`; only consistency is claimed, no rate for the variance estimator itself.

*Edge cases (constraints derived from the proofs, pp. 31-56):*
- Propensity near 0/1: ruled out by assumption (`κ ≤ g ≤ 1-κ` for BOTH `g_0` and every admissible estimate); error bounds blow up as `κ^{-2}`, and the 4th-moment constant needed for variance consistency scales like `C/(p_0 κ) + C/(p_0² κ)` -> clip fitted ĝ to `[κ, 1-κ]` or refuse (implementation choice; paper is silent on how).
- Few treated units: all bounds carry powers of `1/p_0` (up to `1/p̄⁴`); `p̂_k` is a fold mean of `D`, so a fold with zero/very few treated units makes `1/p̂_k` explode -> detect and error/warn per fold.
- Unbalanced periods (Case 2 only): bounds carry `1/(λ_0(1-λ_0))` through `1/(λ³(1-λ)³)`; `λ̂_k` near 0 or 1 (a fold with almost no pre- or post-period observations) is the cross-section analogue of few-treated -> detect and error/warn per fold.
- Empty untreated auxiliary subset (Cases 1-2): `ℓ̂_1k`/`ℓ̂_2k` train only on `I_kz^c = I_k^c ∩ {D = 0}`; a random fold can leave this empty or learner-infeasibly small (Case 2 additionally needs untreated auxiliary rows in both periods) -> detect and fail with a targeted error before invoking the learner.
- Slow ML learners: valid normal inference requires the FULL Assumption 3.1(f)/3.2(h) package — each nuisance bundle at `‖η̂-η_0‖_{P,2} ≤ ε_N = o(N^{-1/4})` AND the product bound `‖ĝ-g_0‖² + ‖ĝ-g_0‖·‖ℓ̂-ℓ_0‖ ≤ ε_N² = o(N^{-1/2})`; if either fails, the `√n ε_N²` bias term does not vanish and normal inference fails (not detectable at runtime; document).
- Heavy-tailed outcomes — case-specific moment conditions: Case 1 (Assumption 3.1(b)-(d)): `E[U²|X] ≤ C`, `E[V_1²|X] ≤ C`, `‖UV_1‖_{P,4} ≤ C`. Case 2 (Assumption 3.2(b)-(f)): the analogous `V_2` conditions plus `E[Y²|X] ≤ C` and `|E[YU]| ≤ C` — the level-outcome moment bounds belong to the repeated-cross-section case ONLY; do not impose them on the panel path. Variance consistency additionally needs a fourth moment of the full augmented score (A.13/A.15). Heavy tails void inference guarantees before consistency.
- Kernel path only: requires `inf_x f_0(x) > 0` (covariate density bounded away from zero on the support; stated twice on p. 57) — a condition the Lasso/ML path does not need.
- Case 3 (multilevel) only: a fold or auxiliary sample with no `W = 0` controls, or fitted `ĝ_z(X)` near 0, makes the score denominator `p̂_w ĝ_zk(X)` undefined/explosive, and `ℓ̂_3k` cannot be fit at all without `W = 0` rows -> require per-fold positive `W = w` and `W = 0` counts and enforce `ĝ_z ≥ κ_z` (implementation-required; the paper prints no Case 3 regularity conditions — see the Case 3 caution above and Gaps).
- Missing data: not addressed in the paper.

*Algorithm (Algorithm 1 in paper, Section 3.1, pp. 10-11; multilevel variant in appendix, pp. 25-26) — DML2 cross-fitting:*
1. Take a K-fold **random partition** `(I_k)_{k=1}^K` of `{1, ..., N}`, each fold of size `n = N/K` (K a fixed integer ≥ 2, independent of N; no specific K is prescribed — see Tuning Parameters). Define the auxiliary sample `I_k^c := {1,...,N} \ I_k` (size `M = N - n`).
2. For each `k`, fit the infinite-dimensional nuisances on the **auxiliary sample** `I_k^c`: `ĝ_k` (any ML method — Logit Lasso given explicitly, Equation 3.4; or kernels), and `ℓ̂_1k` / `ℓ̂_2k` on the **untreated subsample** `I_kz^c := I_k^c ∩ {D = 0}` only, and `ℓ̂_3k` on the multilevel comparison group `I_k^c ∩ {W = 0}` (modified Lasso of Belloni et al. 2012, or random forest, or kernel). Compute the scalar nuisances `p̂_k`, `λ̂_k` as sample means (see the p̂_k contradiction in Gaps: both printed algorithms (pp. 11, 25) show a suspect `1/n` factor over `I_k^c`, while the Theorem 1/2 proofs use fold means over `I_k`).
3. For each `k`, construct the intermediate ATT estimator on fold `I_k` (empirical solution of `E_n[ψ] = 0`; scores are linear in θ so these are closed-form averages):

       (repeated outcomes)       θ̃_k = (1/n) Σ_{i∈I_k} [ (D_i - ĝ_k(X_i)) / ( p̂_k (1 - ĝ_k(X_i)) ) ] · ( Y_i(1) - Y_i(0) - ℓ̂_1k(X_i) )
       (repeated cross sections) θ̃_k = (1/n) Σ_{i∈I_k} [ (D_i - ĝ_k(X_i)) / ( p̂_k λ̂_k (1-λ̂_k)(1 - ĝ_k(X_i)) ) ] · ( (T_i - λ̂_k) Y_i - ℓ̂_2k(X_i) )
       (multilevel, per level w) θ̃_wk = (1/n) Σ_{i∈I_k} [ ( I(W_i=w) ĝ_zk(X_i) - I(W_i=0) ĝ_wk(X_i) ) / ( p̂_w ĝ_zk(X_i) ) ] · ( Y_i(1) - Y_i(0) - ℓ̂_3k(X_i) )

4. Final estimator: simple unweighted average `θ̃ = (1/K) Σ_{k=1}^K θ̃_k` (equal fold sizes make this the pooled average). The proof of Theorem 1 (p. 31) is "essentially identical to Step 3 in the proof of Theorem 3.1 **(DML2)** in Chernozhukov et al. (2018)".
5. Variance per Theorems 2/4 above: per-fold empirical mean of the squared augmented score (fold-k's own `p̂_k`, `λ̂_k`, auxiliary-sample nuisances, and the final `θ̃` plugged in), averaged over K.

*Analytic derivative identities (proofs, pp. 35, 41-42, 45) — usable as implementation checks for the correction-term plumbing:*

    ∂_p ψ_1 = -(1/p)(ψ_1 + θ);           ∂_p² ψ_1 = (2/p²)(ψ_1 + θ)
    ∂_p ψ_2 = -(1/p)(ψ_2 + θ)
    ∂_λ ψ_2 = -[(1-2λ)/(λ²(1-λ)²)] · [(D-g)/(p(1-g))] · ((T-λ)Y - ℓ_2) - [Y/(pλ(1-λ))] · [(D-g)/(1-g)]
    ∂_p² ψ_2 = [2/(p³λ(1-λ))] · [(D-g)/(1-g)] · ((T-λ)Y - ℓ_2)
    ∂_λ∂_p ψ_2 = [(1-2λ)/(p²λ²(1-λ)²)] · [(D-g)/(1-g)] · ((T-λ)Y - ℓ_2) + [Y/(p²λ(1-λ))] · [(D-g)/(1-g)]

**Reference implementation(s):**
- R: none — the paper ships no companion package.
- Stata: none.
- Python (validation oracle — **panel lane only**): `doubleml.DoubleMLDID` implements a Chang/Zimmert-style orthogonal panel score and is the closest parity anchor for Case 1 — but only under a specific configuration (its `in_sample_normalization` option changes the score's normalization; pin the config and verify score equivalence at the equation level before treating any run as an oracle). **Version caveat:** `DoubleMLDID` is deprecated upstream ("will be removed with version 0.12.0. Please use DoubleMLDIDBinary instead", verified 2026-08-22) — pin the exact DoubleML version used for golden-fixture generation, archive the fixtures in-repo, and verify `DoubleMLDIDBinary`'s score/normalization equivalence separately before adopting it as the replacement anchor. **Scope caveat:** treat DoubleML as an **equation-level score oracle**, not a full-estimator finite-sample oracle, unless the fixture supplies identical fold assignments and scalar-nuisance normalization: upstream uses the global treated share and treatment-stratified sample splitting, while Chang specifies random folds and leaves the fold-level `p̂_k` convention ambiguous (see Gaps). Full-estimator comparisons under differing conventions are approximate/asymptotic — a finite-sample mismatch there does not falsify a paper-faithful implementation, and exact-parity fixtures must not silently import DoubleML's normalization as if it were Chang's. Add independent fixtures for whichever `p̂_k` convention is selected.
- **Not oracles:** `doubleml.DoubleMLDIDCS` uses a Sant'Anna-Zhao-style repeated-cross-section score with four treatment-by-period outcome regressions and different normalizations — a *related* estimator, not an implementation of Chang's single-`ℓ_20` Equation 3.2 score or its `λ`-corrected variance. `DoubleMLDIDMulti` handles **staggered treatment timing**, not Chang's Case 3 multilevel treatment *intensity* — it is not a Case 3 anchor. For Cases 2-3, validation must rest on independent equation-level fixtures (Equations 3.2/3.3 and their variance corrections) plus recovery tests on the paper's simulation DGPs (Section 4).

**Requirements checklist** (Case 1-2 items SHIPPED as `DMLDiD` — the
per-item conventions/deviations are the REGISTRY "DMLDiD" Notes; Case 3
items remain open, tracked in DEFERRED.md):
- [x] Neyman-orthogonal Case 1 score (3.1) implemented exactly (Abadie score + mean-zero adjustment); [x] score (3.2) — SHIPPED as `chang_rcs_score` (`DMLDiD(panel=False)`); [ ] score (3.3) — Case 3 open
- [x] DML2 cross-fitting: K-fold partition (D-stratified — documented deviation; PSU-cohesive instead under a coarser-than-unit survey/cluster design, a further documented library extension), nuisances fit on fold complements, never on the evaluation fold
- [x] Outcome nuisance `ℓ̂` fit on the UNTREATED subsample of the auxiliary fold only (`I_kz^c`)
- [x] Scalar nuisance p̂: the global (full-sample-within-cell) convention adopted and documented (the I_k vs I_k^c printing contradiction is thereby sidestepped — see Gaps); [x] `λ̂` — same global convention (`mean(T)` within cell; REGISTRY Note)
- [x] Final estimator: pooled mean (equals the paper's `1/K` average at equal fold sizes — documented deviation)
- [x] Variance from the AUGMENTED score: `Ĝ_1p = -θ̃/p̂` folded in; [x] `Ĝ_2λ (T - λ̂)` — explicit term in `chang_rcs_score_augmented` (`Ĝ_2λ` = sample mean of the closed-form `∂λψ₂`, `chang_rcs_lambda_slope`; the paper prints no estimator — REGISTRY Note)
- [x] Strict-overlap enforcement: fitted propensities clipped to `[trim, 1-trim]` (documented deviation; paper gives no rule)
- [x] Per-fold/per-cell degenerate guards (zero treated/control, cell < K, singleton stratum) — closed skip vocabulary; [x] Case 2 pre/post-share guards — four-group guard + `λ̂` extremeness warning + D×T-stratified folds (control rows in both periods per training complement — by construction under stratified folds; preserved by an explicit per-complement composition guard under PSU folds / weighted fits, REGISTRY Note)
- [x] Auxiliary-sample feasibility: an empty untreated training complement raises `DegenerateFoldError` (targeted, before the learner) → `cross_fit_degenerate` cell
- [x] Normal-approximation inference via `safe_inference()` (no-design fits; survey/bare-cluster fits use finite-df t inference via `df=df_survey` — library extension, not from the paper)
- [ ] Multilevel treatment (Case 3): open — DEFERRED
- [ ] Case 3 guards: open — DEFERRED (see the Case 3 caution above)
- [x] Validation (Case 1): `doubleml.DoubleMLDID` (2-period) + `DoubleMLDIDBinary` (staggered per-cell, end-to-end public fit) parity spikes, doubleml==0.11.4 pinned, golden literals in-repo; [x] Case 2 equation-level fixtures — SHIPPED (closed-form/oracle fixtures, derivative-identity checks, DR both directions, `DoubleMLDIDCSBinary` characterization spike — no parity oracle exists). CAVEAT: the paper's own §4 RCS simulation DGPs (pp. 17-21) are NOT replicated — the shipped recovery/coverage tests use a library-authored RCS design; replication is a tracked TODO.md row (needs the paper PDF); [ ] Case 3 fixtures — open

---

## Implementation Notes

### Data Structure Requirements
- **Repeated outcomes (panel, primary target — SHIPPED as `DMLDiD`):** one row per unit with `(Y(0), Y(1), D, X)` — equivalently a 2-period panel converted to differences `ΔY = Y(1) - Y(0)`. The paper's design is 2-period with treatment only at `t = 1`; the shipped estimator applies it per Callaway-Sant'Anna (g, t) cell over staggered timing (a documented library extension — REGISTRY.md "DMLDiD").
- **Repeated cross sections (SHIPPED as `DMLDiD(panel=False)`):** pooled rows `(Y, D, T, X)` with `T ∈ {0,1}` the post-period sampling indicator; i.i.d. within each period (Assumption 2.3). The shipped lane applies Equation 3.2 per Callaway-Sant'Anna (g, t) cell over staggered timing with row-unique unit IDs (the same documented library extension as the panel lane).
- **Multilevel treatment:** `(Y(0), Y(1), W, X)` with `W ∈ {0, w_1, ..., w_J}`; `W = 0` is the comparison group for every level.
- Covariates `X ∈ R^d` may have `d > N` (the headline setting); no factor-structure or panel-index requirements beyond the above.

### Computational Considerations
- The estimator is a closed-form sample average once nuisances are estimated; cost is dominated by the K nuisance fits (Logit Lasso + outcome learner per fold). Computational complexity is not discussed in the paper.
- Per-fold nuisance fits are embarrassingly parallel across folds (and across treatment levels in Case 3).
- Cross-fitting device: conditional on the auxiliary sample, fitted nuisances are treated as fixed (Lemma A.1 = Lemma 6.1 of Chernozhukov et al. 2018 lifts conditional rates to unconditional). **Never fit nuisances on the fold they are evaluated on.**
- Memory: no special considerations noted; high-dimensional `X` implies the usual Lasso design-matrix footprint per fold.

### Tuning Parameters

| Parameter | Type | Default | Selection Method |
|-----------|------|---------|-----------------|
| Cross-fitting folds `K` | int ≥ 2, fixed (independent of N) | none prescribed in the paper | fixed by user; theory covers any fixed K ≥ 2 (pp. 10, 31-32) |
| Propensity learner | model | none prescribed — "any ML methods or classical estimators such as kernel or series estimators" (p. 11); paper's example/simulation choice: Logit Lasso (Multi-Logit Lasso for multilevel) (Eq. 3.4) | penalty by K-fold CV with K = 10 (Van de Geer 2008) or Belloni, Chernozhukov, Chetverikov & Wei (2018) (Sec. 4, p. 17) |
| Outcome learner (`ℓ` nuisances) | model | none prescribed; paper's example/simulation choices: modified Lasso (Belloni et al. 2012), random forest (500 trees in simulations) | Lasso penalty: iterated plug-in loadings, appendix p. 26 (below); RF: fixed 500 trees |
| Lasso penalty loadings `(λ_k, Υ̂_k)` | diagonal matrix + scalar | `λ_k = 2c·√M_k·Φ^{-1}(1 - γ/(2p))`, `c > 1`, `γ → 0` | **As printed (appendix "The Lasso Penalty", p. 26):** "Let `y_i` denote `Y_i(1) - Y_i(0)` or `(T_i - λ̂_k)`"; initial `γ̂_kj = sqrt((1/M_k) Σ_{i∈I_kz^c} q_ij²(y_i - ȳ_k)²)`; refined `γ̂_kj = sqrt((1/M_k) Σ_{i∈I_kz^c} q_ij² ε̂_i²)` with `ε̂_i = y_i - q_i'β*_k`, repeated B > 0 times; `ȳ_k = M^{-1} Σ_{i∈I_k^c} y_i`. `M = N - n` = full auxiliary-sample size; `M_k = |I_kz^c|` = its untreated subset. **Proposed implementation (corrections; see Suspected typos and Gaps):** use the same response the modified-Lasso objective (p. 12) minimizes — `y_i = Y_i(1) - Y_i(0)` (Case 1), `y_i = (T_i - λ̂_k)·Y_i` (Case 2) — pending comparison with the published article. (No defaults for c, γ, B printed; Belloni et al. 2012 conventions, e.g. c = 1.1, are the natural reference) |
| Kernel (alternative first stage) | kernel fn | none prescribed by theory — Assumption 3.3(1) requires a **compactly supported** order-`m` kernel; simulations use a standard Gaussian (violates 3.3's compact support) | Theorems 3-4 cover only 3.3-compatible kernels; using Gaussian must be documented as a deviation (Sec. 4, p. 17) |
| Kernel bandwidth `h` | float | data-driven CV | CV/MSE-minimizing `h = N^{-1/(d+2s+2m)}` is VALID (no undersmoothing needed — SBP, Theorem 3); requires `2m > d + 2s` |
| Overlap clip `κ` | float | not specified | implementation choice; theory requires fitted `g ∈ [κ, 1-κ]` |

### Relation to Existing diff-diff Estimators
- **Abadie (2005) IPW-DiD lineage:** the scores are orthogonalized versions of the Abadie IPW score; the same `(D - g(X)) / (p(1 - g(X)))` weighting with a regression adjustment `ℓ` appears in the Sant'Anna-Zhao (2020) doubly robust scores already implemented in `CallawaySantAnna` (`diff_diff/staggered.py`) and in the DRDID-parity cell score `drdid_panel_inf_func` in `diff_diff/_dr_scores.py` (relocated from `ContinuousDiD` in PR-B0, oracle-pinned) — those are the closest existing score implementations and natural code-sharing targets. Note the structural difference: Chang's Case-1 score normalizes by the unconditional `p_0` (with an explicit `G_1p (D - p̂)` variance correction), whereas the SZ DR score is self-normalized; the two are not interchangeable.
- **Nuisance estimation:** `solve_logit` in `diff_diff/linalg.py` provides the (unpenalized) logistic propensity fit; the sieve nuisance machinery in `diff_diff/efficient_did_covariates.py` is the existing precedent for flexible outcome-regression nuisances. Lasso/Logit-Lasso penalized fits do not yet exist in the library (scipy-only constraint — coordinate descent or proximal gradient would need to be written, or the learner made pluggable).
- **Cross-fitting machinery now exists as private infrastructure** (PR-B0): `diff_diff/_crossfit.py` provides the replayable fold partition and per-fold out-of-fold nuisance prediction, with the Chang scores in `diff_diff/_dr_scores.py` and pluggable learners in `diff_diff/_learners.py` — see the REGISTRY "Cross-fitting, DR-score, and ridge infrastructure (DML)" section. The shipped `DMLDiD` estimator (PR-B1) consumes it — see REGISTRY.md "DMLDiD" and the parity spikes in `benchmarks/doubleml/`.
- Inference should route through `safe_inference()` (`diff_diff/utils.py`) per library convention; the augmented-score variance is a per-fold empirical second moment, structurally similar to the influence-function variances used across the staggered family.
- The PAPER's two-period design sits next to `DifferenceInDifferences` in scope, but with ML/high-dimensional covariates; the shipped `DMLDiD` applies it per (g, t) cell over staggered timing (documented library extension), so in practice it sits beside the staggered estimators.

---

## Gaps and Uncertainties

**Suspected typos in the arXiv v3 PDF** (carry to any implementation cross-check against the published *Econometrics Journal* version):

1. **`p̂_k` factor (Algorithm 1, main text, p. 11):** printed as `p̂_k = (1/n) Σ_{i∈I_k^c} D_i` — a `1/n` factor over the auxiliary sample `I_k^c` (whose size is `M = N - n`, not `n`) — likely intended as the auxiliary-sample average; verify against the published version. (Same issue for `λ̂_k`.)
2. **`ψ_1` inside `Σ_20` (Assumption 3.2, p. 14):** the definition prints `Σ_20 := E_P[(ψ_1(W,θ_0,p_0,η_10) + G_2p0(D - p_0) + G_2λ0(T - λ_0))²]` — the leading `ψ_1(W, θ_0, p_0, η_10)` appears to be a typo for `ψ_2(W, θ_0, p_0, λ_0, η_20)`; check the published version.
3. **`I_{4,k} = O_P(n^{1/2})` (proof of Theorem 2, p. 48):** the PDF text prints "O_P(n^{1/2})" — from the Chebyshev step and the final display `I_k = O_P(N^{-1/2}) + O_P(N^{-1/2} + ε_N)` this is evidently a typo for `n^{-1/2}`.
4. **(A.14) outer exponent (p. 51):** as printed the sup-norm display appears with `(...)² ≤ ε_N`; by parallel with (A.12) it should read as the `^{1/2}` L2 norm.
5. **Undefined `D_i` in the multilevel `p̂_w` (appendix algorithm, p. 25, step 2):** prints `p̂_w = (1/n) Σ_{i∈I_k^c} D_i`, but `D` is not defined in the multilevel notation (treatment is `W`); evidently intended as `I(W_i = w)` (verified against the PDF page 2026-08-22).
6. **Stray `λ_0` in the multilevel algorithm (p. 25, step 2):** the step says "construct the estimator of `p_0` and `λ_0`", but no `λ_0` exists in Case 3 (there is no time-sampling share in the repeated-outcomes multilevel setting); evidently copied from the Case 2 algorithm (verified against the PDF page 2026-08-22).
7. **Case 2 response in the Lasso penalty-loading recipe (appendix "The Lasso Penalty", p. 26):** the recipe literally prints "Let `y_i` denote `Y_i(1) - Y_i(0)` or `(T_i - λ̂_k)`" — the Case 2 response omits the `Y_i` multiplier (verified against the PDF page 2026-08-22), while the modified-Lasso objective it feeds (p. 12) minimizes over `(T_i - λ̂_k)·Y_i - q_i'β`. Loadings computed on a bare `(T_i - λ̂_k)` would produce a different first-stage estimator; implement loadings on the objective's response, `(T_i - λ̂_k)·Y_i`, pending comparison with the published article.
8. **`ȳ_k` population/denominator inconsistency (p. 26):** `ȳ_k = M^{-1} Σ_{i∈I_k^c} y_i` is defined as the mean over the FULL auxiliary sample (size `M`), while both loading sums run over the untreated subset `I_kz^c` with normalizer `1/M_k` (verified against the PDF page 2026-08-22). Whether the centering mean was intended over `I_k^c` or `I_kz^c` is not resolvable from the arXiv text; record the choice made at implementation time and compare with the published article.

**Contradiction — where `p̂_k` (and `λ̂_k`) is computed (algorithms vs proofs):**
- Both printed algorithm statements agree with each other and were verified against the PDF pages directly (2026-08-22): main-text Algorithm 1 (p. 11) prints `p̂_k = (1/n) Σ_{i∈I_k^c} D_i` and the appendix multilevel algorithm (p. 25, step 2) prints `p̂_w = (1/n) Σ_{i∈I_k^c} D_i` — in both cases a `1/n` normalizer (n = |I_k|) over the **auxiliary sample `I_k^c`** (size `M = N - n`), which is not a valid mean of anything as printed (typo 1 above).
- The proofs of Theorems 1-2, however, use `p̂_k = E_{n,k}[D]` (fold mean over the **main fold `I_k`**, pp. 31, 33, 37), i.e. `p̂_k - p_0 = E_{n,k}[D - p_0]`, which is what generates the `G_1p0 (D - p_0)` influence-correction term in the variance algebra.
- So the two self-consistent readings of the printed formula are: `(1/M) Σ_{i∈I_k^c}` (auxiliary-sample mean — fix the normalizer, standard DML convention of fitting all nuisances on the complement) or `(1/n) Σ_{i∈I_k}` (main-fold mean — fix the index set, matching the proofs). Both are √N-consistent means of `D`, so the estimators are asymptotically equivalent; finite-sample values and the exact variance-correction algebra differ. Do not guess: consult the published *Econometrics Journal* version and DoubleML's implementation before fixing a convention. Pages: 11, 25, 31-37.
- **Library convention adopted (PR-B0):** DoubleML's implementation was consulted (it uses the GLOBAL full-sample treated share — verified to machine precision in the committed spike `benchmarks/doubleml/chang_case1_parity.py`), and the library adopts that third self-consistent reading; recorded as a `- **Note:**` deviation in REGISTRY's "Cross-fitting, DR-score, and ridge infrastructure (DML)" section. The published-version cross-check of the printed typo remains outstanding (DEFERRED.md, "Needs external reference").

**Content not covered by the paper / open items:**
- No companion software package; no coverage/RMSE tables — simulation evidence is Monte Carlo histograms only (Figures 3-20, pp. 58-63; the "New" vs "HD" right-panel labeling is inferred to distinguish kernel-variant from Lasso-variant designs, but the figure section carries no prose confirming it).
- No recommended number of cross-fitting folds K anywhere in the paper.
- No trimming/clipping rule for propensity scores (theory assumes the band; implementation must choose and document).
- No explicit consistent formula for `Ĝ_2λ` is given in the theorem statements — the closed form for `G_2λ0` recovered from the proof (p. 55) is the natural sample analogue.
- Multilevel treatment: variance estimator and full theorem statements are not printed in the reviewed text (Theorems 2/4 cover Cases 1-2 only; multilevel results "can be proven using the same arguments", p. 13). The analogy `Ĝ_wp = -θ̃_w/p̂_w` is plausible but NOT stated in the paper.
- Theorem 2 proof for the repeated-cross-sections case concludes within the reviewed range (pp. 51-56); no gaps remain in the proofs of Theorems 1-4 as reviewed, but the appendix's stated Lasso-penalty constants `c`, `γ`, `B` have no defaults (p. 26).
- No clustering, no bootstrap, no missing-data handling, no aggregation across multilevel treatment arms.
- Empirical application (Section 5, pp. 21-22, repeated cross sections via Eqs. 2.2/3.2): Sequeira (2016) tariff/bribery data, N = 1084. Table 1 exact estimate (SE) pairs (verified against the PDF 2026-08-22): Sequeira (2016) TWFE -2.928 (0.944); Abadie kernel -7.986 (3.028); orthogonal θ̃ kernel -8.670 (3.643); Abadie Lasso -7.499 (2.746); orthogonal θ̃ Lasso -9.191 (4.854) — usable as a rough replication target only if the Sequeira data is obtainable.
- Simulation DGPs (Section 4, pp. 17-21) are fully specified and are the recommended validation fixtures: e.g. 4.1.1 repeated outcomes ML: `N ∈ {200, 500}`, `p ∈ {100, 300}`, `X ~ N(0, I_p)`, `γ_0 = (1, 1/2, 1/3, 1/4, 1/5, 0, ...)`, logistic PS, `β_0 = γ_0 + 0.5`, `θ_0 = 3`, errors N(0, 0.1) (variance/SD as printed "N(0,0.1)" — ambiguous); 4.3.1 multilevel: `W ∈ {0,1,2}`, shares (0.3, 0.3, 0.4), `θ_10 = 3`, `θ_20 = 6`.
- Finding to preserve for docs: in the repeated-cross-section simulations the orthogonal estimator appears well centered on the truth but visibly NOISIER than Abadie's plug-in at small N (Figures 9-14) — expect larger SEs. Orthogonality removes first-order nuisance-estimation bias (an asymptotic property under the Theorem 1 rates); it does NOT guarantee finite-sample unbiasedness, and histogram centering in simulations cannot establish it.
