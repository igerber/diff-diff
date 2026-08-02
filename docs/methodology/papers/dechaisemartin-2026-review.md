# Paper Review: Difference-in-Differences Estimators When No Unit Remains Untreated

**Authors:** Clément de Chaisemartin, Diego Ciccia, Xavier D'Haultfœuille, Felix Knau
**Citation:** de Chaisemartin, C., Ciccia, D., D'Haultfœuille, X., & Knau, F. (2026). Difference-in-Differences Estimators When No Unit Remains Untreated. arXiv:2405.04465v6.
**PDF reviewed:** papers/Difference-in-Differences Estimators When No Unit Remains Untreated.pdf
**Review date:** 2026-04-18

---

## Methodology Registry Entry

*Formatted to match docs/methodology/REGISTRY.md structure. Heading levels and labels align with existing entries - copy the `## {EstimatorName}` section into the appropriate category in the registry.*

## {EstimatorName}

**Primary source:** de Chaisemartin, C., Ciccia, D., D'Haultfœuille, X., & Knau, F. (2026). Difference-in-Differences Estimators When No Unit Remains Untreated. arXiv:2405.04465v6.

**Scope:** Heterogeneous Adoption Design (HAD): a single-date, two-period DiD setting in which no unit is treated at period one and at period two all units receive strictly positive, heterogeneous treatment doses `D_{g,2} >= 0`. The estimator targets a Weighted Average Slope (WAS) when no genuinely untreated group exists. Extensions cover multiple periods without variation in treatment timing (Appendix B.2) and covariate-adjusted identification (Appendix B.1).

**Key implementation requirements:**

*Assumption checks / warnings:*
- Data must be panel (or repeated cross-section) with `D_{g,1} = 0` for all `g` (nobody treated in period one). **Phase 2a implementation note:** `diff_diff.HeterogeneousAdoptionDiD` ships panel-only; repeated-cross-section inputs are rejected by the balanced-panel validator. RCS support is queued for a follow-up PR.
- Treatment dose `D_{g,2} >= 0`. For Design 1' (the QUG case) the support infimum `d̲ := inf Supp(D_{g,2})` must equal 0; for Design 1 (no QUG) `d̲ > 0` and Assumption 5 or 6 must be invoked.
- Assumption 1 (i.i.d. sample): `(Y_{g,1}, Y_{g,2}, D_{g,1}, D_{g,2})_{g=1,...,G}` i.i.d.
- Assumption 2 (parallel trends for the least-treated): `lim_{d ↓ d̲} E[ΔY(0) | D_2 ≤ d] = E[ΔY(0)]`. Testable with pre-trends when a pre-treatment period `t=0` exists. Reduces to standard parallel trends when treatment is binary.
- Assumption 3 (uniform continuity of `d → Y_2(d)` at zero): excludes extensive-margin effects; holds if `d → Y_2(d)` is Lipschitz. Not testable.
- Assumption 4 (regularity for nonparametric estimation): positive density at boundary (`lim_{d ↓ 0} f_{D_2}(d) > 0`), twice-differentiable `m(d) := E[ΔY | D_2 = d]` near 0, continuous `σ²(d) := V(ΔY | D_2 = d)` with `lim_{d ↓ 0} σ²(d) > 0`, bounded kernel, bandwidth `h_G → 0` with `G h_G → ∞`.
- Assumption 5 (for Design 1 sign identification): `lim_{d ↓ d̲} E(TE_2 | D_2 ≤ d) / WAS < E(D_2) / d̲`. Not testable via pre-trends. Sufficient version Equation 9: `0 ≤ E(TE_2 | D_2 = d) / E(TE_2 | D_2 = d') < E(D_2) / d̲` for all `(d, d')` in `Supp(D_2)²`.
- Assumption 6 (for Design 1 WAS_{d̲} identification): `lim_{d ↓ d̲} E[Y_2(d̲) - Y_2(0) | D_2 ≤ d] = E[Y_2(d̲) - Y_2(0)]`. Not testable.
- Warn (do NOT fit silently) when: staggered treatment timing is detected - "in designs with variation in treatment timing, there must be an untreated group, at least till the period where the last cohort gets treated" (Appendix B.2). `did_had` covers only the no-untreated case without staggered timing (or, in a staggered setting, treatment effects for the last cohort only).
- Warn when Assumption 5/6 is invoked that these are not testable via pre-trends.
- With Design 1 (no QUG) WAS is NOT point-identified under Assumptions 1-3 alone (Proposition 1); only sign identification (Theorem 2) or the alternative target WAS_{d̲} (Theorem 3) is available.

*Target parameter - Weighted Average Slope (WAS, Equation 2):*

    WAS := E[(D_2 / E[D_2]) · TE_2]
         = E[Y_2(D_2) - Y_2(0)] / E[D_2]

where `TE_2 := (Y_2(D_2) - Y_2(0)) / D_2` is the per-unit slope relative to "no treatment". Authors prefer WAS over the unweighted Average Slope `AS := E[TE_2]` because AS suffers a small-denominator problem near `D_2 = 0` that prevents `√G`-rate estimation.

Alternative target (Design 1 under Assumption 6):

    WAS_{d̲} := E[(D_2 - d̲) / E[D_2 - d̲] · TE_{2,d̲}]

where `TE_{2,d̲} := (Y_2(D_2) - Y_2(d̲)) / (D_2 - d̲)`. Compares to a counterfactual where every unit gets the lowest dose, not zero; authors describe it as "less policy-relevant" than WAS.

*Estimator equations:*

Design 1' identification (Theorem 1, Equation 3):

    WAS = (E[ΔY] - lim_{d ↓ 0} E[ΔY | D_2 ≤ d]) / E[D_2]

Nonparametric local-linear estimator (Equation 7):

    β̂_{h*_G}^{np} := ((1/G) Σ_{g=1}^G ΔY_g  -  μ̂_{h*_G}) / ((1/G) Σ_{g=1}^G D_{g,2})

where `μ̂_h` is the intercept from a local-linear regression of `ΔY_g` on `D_{g,2}` using weights `k(D_{g,2}/h)/h`. This estimates the conditional mean `m(0) = lim_{d ↓ 0} E[ΔY | D_2 ≤ d]`.

Design 1 mass-point case (Section 3.2.4, discrete bunching at `d̲`):

    target = (E[ΔY] - E[ΔY | D_2 = d̲]) / E[D_2 - d̲]
           = (E[ΔY | D_2 > d̲] - E[ΔY | D_2 = d̲]) / (E[D_2 | D_2 > d̲] - E[D_2 | D_2 = d̲])

Compute via sample averages or a 2SLS of `ΔY` on `D_2` with instrument `1{D_2 > d̲}`. Convergence rate is `√G`.

Design 1 continuous-near-`d̲` case: use the same kernel construction as Equation 7 with 0 replaced by `d̲` and `D_2` replaced by `D_2 - d̲`. `d̲` is estimated by `min_g D_{g,2}`, which converges at rate `G` (asymptotically negligible versus the `G^{2/5}` nonparametric rate of `β̂_{h*_G}^{np}`).

Sign identification for Design 1 (Theorem 2, Equation 10):

    WAS ≥ 0  ⟺  (E[ΔY] - lim_{d ↓ d̲} E[ΔY | D_2 ≤ d]) / E[D_2 - d̲] ≥ 0

WAS_{d̲} identification (Theorem 3, Equation 11):

    WAS_{d̲} = (E[ΔY] - lim_{d ↓ d̲} E[ΔY | D_2 ≤ d]) / E[D_2 - d̲]

*With covariates / conditional identification (Equation 19, Appendix B.1):*

Assumption 9 (conditional parallel trends): almost surely, `lim_{d ↓ 0} E[ΔY(0) | D_2 ≤ d, X] = E[ΔY(0) | X]`.

Theorem 6 (Design 1' + Assumptions 3 and 9):

    WAS = (E[ΔY] - E[ lim_{d ↓ 0} E[ΔY | D_2 ≤ d, X] ]) / E[D_2]

Implementing Equation 19 requires MULTIVARIATE nonparametric regression `E[ΔY | D_2, X]`; Calonico et al. (2018) covers only the univariate case, so the authors leave this extension to future work.

TWFE-with-covariates (Appendix B.1, Equations 20-21): under linearity Assumption 10 (`E[ΔY(0) | D_2, X] = X' γ_0`) and homogeneity `E[TE_2 | D_2, X] = X' δ_0`,

    E[ΔY | D_2, X] = X' γ_0 + D_2 X' δ_0    (21)

so `δ_0` is recovered by OLS of `ΔY` on `X` and `D_2 * X`; Average Slope is `((1/n) Σ X_i)' δ̂^X`.

*Standard errors (Section 3.1.3-3.1.4, 4):*

- Nonparametric estimator (Design 1' and Design 1 continuous-near-`d̲`): bias-corrected Calonico-Cattaneo-Farrell (2018, 2019) 95% CI (Equation 8):

      [ β̂_{ĥ*_G}^{np} + M̂_{ĥ*_G} / ((1/G) Σ D_{g,2})  ±  q_{1-α/2} sqrt(V̂_{ĥ*_G} / (G ĥ*_G)) / ((1/G) Σ D_{g,2}) ]

  The procedure relies on Calonico et al. `nprobust`: estimate optimal bandwidth `ĥ*_G`, compute `μ̂_{ĥ*_G}`, the first-order bias estimator `M̂_{ĥ*_G}`, and the variance estimator `V̂_{ĥ*_G}`.
- 2SLS (Design 1 mass-point case): standard 2SLS inference (details not elaborated in the paper).
- TWFE with small `G`: HC2 standard errors with Bell-McCaffrey (2002) degrees-of-freedom correction, following Imbens and Kolesar (2016). Used in the Pierce and Schott (2016) application with `G=103`.
- Bootstrap: wild bootstrap with Mammen (1993) two-point weights is used for the Stute test (see Diagnostics below), NOT for the main WAS estimator.
- Clustering: no explicit clustering formulas in the paper's core equations.

*Convergence rates:*
- Design 1' nonparametric estimator: `G^{2/5}` (univariate nonparametric rate; Equations 5-6).
- Design 1 discrete-mass-point case: `√G` (parametric rate).
- Estimate of `d̲` via `min_g D_{g,2}`: rate `G` (asymptotically negligible).

*Asymptotic distributions (Equations 5-6):*
- Equation 5: `√(G h_G) (β̂_{h_G}^{np} - WAS - h_G² · C m''(0) / (2 E[D_2])) →^d N(0, σ²(0) ∫_0^∞ k*(u)² du / (E[D_2]² f_{D_2}(0)))`
- Equation 6 (optimal rate, `G^{1/5} h_G → c > 0`): `G^{2/5} (β̂_{h_G}^{np} - WAS) →^d N(c² C m''(0) / (2 E[D_2]), σ²(0) ∫_0^∞ k*(u)² du / (c E[D_2]² f_{D_2}(0)))`
- Kernel constants: `κ_k := ∫_0^∞ t^k k(t) dt`, `k*(t) := (κ_2 - κ_1 t) / (κ_0 κ_2 - κ_1²) · k(t)`, `C := (κ_2² - κ_1 κ_3) / (κ_0 κ_2 - κ_1²)`.

*Edge cases:*
- **No genuinely untreated units, D_2 continuous with `d̲ = 0` (Design 1')**: use `β̂_{h*_G}^{np}` (Equation 7) with bias-corrected CI (Equation 8).
- **No untreated units, `d̲ > 0`, `D_2` has mass point at `d̲`**: use 2SLS of `ΔY` on `D_2` with instrument `1{D_2 > d̲}`, or equivalent sample-average formula. Identifies WAS_{d̲} under Assumption 6 (Theorem 3) or the sign of WAS under Assumption 5 (Theorem 2).
- **No untreated units, `d̲ > 0`, `D_2` continuous near `d̲`**: replace 0 by `d̲` and `D_2` by `D_2 - d̲` in Equation 7; estimate `d̲` by `min_g D_{g,2}`.
- **Genuinely untreated units present but a small share**: Authors do NOT require untreated units to be dropped. In the Garrett et al. (2020) bonus-depreciation application with 12 untreated counties out of 2,954, they keep the untreated subsample. Simulations (DGP 2, DGP 3) suggest CIs retain close-to-nominal coverage even when `f_{D_2}(0) = 0`.
- **WAS is not point-identified without a QUG (Proposition 1, proof C.1)**: the proof explicitly constructs `tilde-Y_2(d) := Y_2(d) + (c / d̲) · E[D_2] · (d - d̲)` for any `c ∈ R`, compatible with the data under Assumptions 2 and 3 but with `tilde-WAS = WAS + c`. Practical consequence: do NOT report a point estimate of WAS under Design 1 without Assumption 5 or 6; fall back to Theorem 2 (sign) or Theorem 3 (WAS_{d̲}).
- **Extensive-margin effects**: ruled out by Assumption 3. If a jump `Y_2(0) ≠ Y_2(0+)` is suspected, the target parameter and estimator are not appropriate.
- **Partial identification of WAS_{d̲}**: only identified up to a positive constant offset `≤ ε` by the bound in Equation 22 (Jensen inequality argument in Appendix C.3).
- **Density at boundary**: Assumption 4 requires `f_{D_2}(0) > 0`. This is a non-trivial assumption since 0 is on the boundary of `Supp(D_2)`.
- **Variation in treatment timing**: Appendix B.2 - "in designs with variation in treatment timing, there must be an untreated group, at least till the period where the last cohort gets treated." `did_had` may be used only for the last treatment cohort in a staggered design; otherwise use `did_multiplegt_dyn`.
- **Mechanical zero at reference period under linear trends (Footnote 13, main text p. 31)**: with industry/unit-specific linear trends, the pre-trends estimator is mechanically zero in the second-to-last pre-period (the slope anchor year). Practical consequence: that year is not an informative placebo check.

*Algorithm (Design 1' nonparametric - summarized from Section 3.1.3-3.1.4 and Equations 7-8):*
1. Compute bandwidth `ĥ*_G` via Calonico et al. (2018) `nprobust` optimal-bandwidth selector on the local-linear regression of `ΔY_g` on `D_{g,2}` with kernel weights `k(D_{g,2}/h)/h`.
2. Fit the local-linear regression at bandwidth `ĥ*_G`; read off the intercept `μ̂_{ĥ*_G}`.
3. Compute `β̂_{ĥ*_G}^{np} = ((1/G) Σ ΔY_g - μ̂_{ĥ*_G}) / ((1/G) Σ D_{g,2})` (Equation 7).
4. Compute the first-order bias estimator `M̂_{ĥ*_G}` and the variance estimator `V̂_{ĥ*_G}` (Calonico et al. 2018, 2019).
5. Form the bias-corrected 95% CI by Equation 8.

*Algorithm variant - Design 1 mass-point 2SLS (Section 3.2.4):*
1. Detect a mass point at `d̲`: either user-supplied `d̲` or detected automatically (e.g., the modal minimum value of `D_{g,2}`).
2. Either compute `(Ȳ_{D_2 > d̲} - Ȳ_{D_2 = d̲}) / (D̄_{D_2 > d̲} - D̄_{D_2 = d̲})` (sample averages), or run 2SLS of `ΔY_g` on `D_{g,2}` with instrument `1{D_{g,2} > d̲}`.
3. Report the estimate as WAS_{d̲} under Assumption 6 or as the sign-identifying quantity under Assumption 5.

*Algorithm variant - QUG null test (Theorem 4, Section 3.3):*
Tuning-parameter-free test of `H_0: d̲ = 0` versus `H_1: d̲ > 0`.
1. Sort `D_{2,g}` ascending to obtain order statistics `D_{2,(1)} ≤ D_{2,(2)} ≤ ... ≤ D_{2,(G)}`.
2. Compute test statistic `T := D_{2,(1)} / (D_{2,(2)} - D_{2,(1)})`.
3. Reject `H_0` if `T > 1/α - 1`.
4. Theorem 4 establishes:
   - Asymptotic size: `lim sup_{G→∞} sup_{F ∈ F^{0,d̄}_{m,K}} P_F(W_α) = α`.
   - Uniform consistency: `lim inf_{G→∞} inf_{F ∈ F^{d̲,d̄}_{m,K}} P_F(W_α) = 1`.
   - Local power at rate `G`: for any sequence `(d̲_G)` with `lim inf G · d̲_G > 0`, `lim inf_{G→∞} inf_{F ∈ F^{d̲_G,d̄}_{m,K}} P_F(W_α) > α`.
   - Class: `F^{d̲,d̄}_{m,K} := { F : F differentiable on [d̲, d̄], F(d̲) = 0, F'(d) ≥ m, |F'(d) - F'(d_1)| ≤ K |d - d_1| }`.
5. Li et al. (2024, Theorem 2.4) result on asymptotic independence of extreme order statistics and sample averages implies the QUG test is asymptotically independent of the WAS / TWFE estimator, so conditional inference on WAS given non-rejection of the pre-test does not distort inference (asymptotically; extension to triangular arrays is conjectured but not proven - Footnote 8 / page 21 top).

*Algorithm variant - TWFE linearity test via Stute (1997) Cramér-von Mises with wild bootstrap (Section 4.3, Appendix D):*
Used to test whether `E(ΔY | D_2)` is linear, which is the testable implication of TWFE's homogeneity assumption (Assumption 8) in HADs.
1. Fit linear regression of `ΔY_g` on constant and `D_{g,2}`; collect residuals `ε̂_{lin,g}`.
2. Form cusum process `c_G(d) := G^{-1/2} Σ_{g=1}^G 1{D_{g,2} ≤ d} · ε̂_{lin,g}`.
3. Compute Cramér-von Mises statistic `S := (1/G) Σ_{g=1}^G c_G²(D_{g,2})`. Equivalently, after sorting by `D_{g,2}`: `S = Σ_{g=1}^G (g/G)² · ((1/g) Σ_{h=1}^g ε̂_{lin,(h)})²`.
4. Wild bootstrap for p-value (Stute, Manteiga, Quindimil 1998; Algorithm in main text p. 25 and vectorized form in Appendix D):
   - Draw `(η_g)_{g=1,...,G}` i.i.d. from the Mammen two-point distribution: `η_g = (1+√5)/2` with probability `(√5-1)/(2√5)`, else `η_g = (1-√5)/2`.
   - Set `ε̂*_{lin,g} := ε̂_{lin,g} · η_g`.
   - Compute `ΔY*_g = β̂_0 + ΔD_g · β̂_{fe} + ε̂*_{lin,g}` (Page 25, Footnote 7 weights). The paper uses the Δ first-difference operator; since `D_{g,1} = 0` for all `g`, `ΔD_g ≡ D_{g,2}`, so the bootstrap DGP equals `β̂_0 + D_{g,2} · β̂_{fe} + ε̂*_{lin,g}` in this setup. Implementations can code either form.
   - Re-fit OLS on the bootstrap sample to get `ε̂*_{lin,g}`, compute `S*`.
   - Repeat B times; the p-value is the fraction of `S*` exceeding `S`.
5. Properties (page 26): asymptotic size, consistency under any fixed alternative, non-trivial local power at rate `G^{-1/2}`.
6. Vectorized implementation (Appendix D, Online Appendix p. 1): with `L` a `G × G` lower-triangular matrix of ones and `I` a `1 × G` row of ones, `S = (1/G²) · I · (L · E)^{∘2}`. Bootstrap uses a `G × G` realization matrix `H` of Mammen weights; memory-bounded at `G ≈ 100,000`.

*Algorithm variant - Yatchew (1997) heteroskedasticity-robust linearity test (Appendix E, Theorem 7):*
Alternative to Stute when `G` is large or heteroskedasticity is suspected.
1. Sort `(D_{g,2}, ΔY_g)` by `D_{g,2}`.
2. Compute difference-based variance estimator: `σ̂²_{diff} := (1/(2G)) Σ_{g=2}^G [(Y_{2,(g)} - Y_{1,(g)}) - (Y_{2,(g-1)} - Y_{1,(g-1)})]²`.
3. Fit linear regression; compute residual variance `σ̂²_{lin}`.
4. Naive test statistic: `T := √G · (σ̂²_{lin} / σ̂²_{diff} - 1) →^d N(0, 1)` under homoskedasticity. NOT valid under heteroskedasticity (over-rejects).
5. Heteroskedasticity-robust variance: `σ̂⁴_W := (1/(G-1)) Σ_{g=2}^G ε̂²_{lin,(g)} ε̂²_{lin,(g-1)}`.
6. Robust test statistic: `T_{hr} := √G · (σ̂²_{lin} - σ̂²_{diff}) / σ̂²_W`. Reject linearity if `T_{hr} ≥ q_{1-α}` (Equation 29 and downstream in Theorem 7).
7. Theorem 7: under `H_0`, `lim E[φ_α] = α`; under fixed alternative, `lim E[φ_α] = 1`; local power against alternatives at rate `G^{-1/4}` (slower than Stute's `G^{-1/2}` rate, but scales to `G ≥ 10⁵`).
8. Key result: inference on `β̂_{fe}` conditional on accepting the linearity test is asymptotically valid (Theorem 7, Point 1; citing de Chaisemartin and D'Haultfœuille 2024 arXiv:2407.03725).

**Reference implementation(s):**
- R: `did_had` (de Chaisemartin, Ciccia, D'Haultfœuille, Knau 2024a); `stute_test` (2024c); `yatchew_test` (Online Appendix, Table 3).
- Stata: `did_had` (2024b); `stute_test` (2024d); `yatchew_test`. Also `twowayfeweights` (de Chaisemartin, D'Haultfœuille, Deeb 2019) for negative-weight diagnostics.
- Underlying bias-correction machinery: Calonico, Cattaneo, Farrell (2018, 2019) `nprobust`.

**Requirements checklist:**
- [x] Panel data loader verifies `D_{g,1} = 0` for all units. **Phase 1c implementation:** `_validate_had_inputs` in `diff_diff/had.py:1029-1042` rejects panels where the pre-period does not have all-zero dose (HAD pre-period contract violation).
- [x] Separate code paths for Design 1' (`d̲ = 0`), Design 1 mass-point (`d̲ > 0` discrete), and Design 1 continuous-near-`d̲`. **Phase 2a implementation:** three dispatch paths `continuous_at_zero` / `continuous_near_d_lower` / `mass_point` with `design="auto"` resolving via `_detect_design()`. Mismatched overrides raise `ValueError` rather than silently identifying a different estimand. See `HeterogeneousAdoptionDiD` class docstring at `diff_diff/had.py:2531-2560`.
- [x] Local-linear regression backend (kernel weights, bandwidth selector). **Phase 1a/1b implementation:** full `nprobust` (Calonico-Cattaneo-Farrell) port in `diff_diff/_nprobust_port`. `tests/test_nprobust_port.py` (~789 LoC) validates the port at `atol=1e-14` machine precision against golden fixtures.
- [x] Integration with bias-corrected CI from Calonico-Cattaneo-Farrell. **Phase 1c implementation:** `bias_corrected_local_linear()` returns the CCF bias-corrected point + robust-bias-corrected CI. `tests/test_bias_corrected_lprobust.py` (~44 tests) validates parity against hand-derived R reference at `atol=1e-12` including the weighted path.
- [x] QUG null test (`T = D_{2,(1)} / (D_{2,(2)} - D_{2,(1)})`, rejection region `{T > 1/α - 1}`). **Phase 3 implementation (2026-04):** `qug_test()` in `diff_diff/had_pretests.py`. Asymptotic p-value `1/(1+T)` under Exp(1)/Exp(1) limit law. Zero-dose observations filtered upfront with `UserWarning`; tie-break `D_{(1)} == D_{(2)}` returns all-NaN inference. Tight closed-form parity at `atol=1e-12`.
- [x] Stute Cramér-von Mises test with Mammen wild bootstrap. **Phase 3 implementation (2026-04):** `stute_test()` in `diff_diff/had_pretests.py`. Literal per-iteration OLS refit per paper Appendix D Algorithm. `n_bootstrap=999` default, `n_bootstrap >= 99` validated.
- [x] Yatchew heteroskedasticity-robust linearity test. **Phase 3 implementation (2026-04):** `yatchew_hr_test()` in `diff_diff/had_pretests.py`. Test statistic `T_hr = sqrt(G)·(σ²_lin - σ²_diff)/σ²_W` from paper Equation 29. `σ²_diff` normalizes by `2G` (paper-literal), NOT `2(G-1)` (finite-sample equivalent but tests pin the paper-literal form). Standard-normal critical value, one-sided.
- [x] Composite workflow `did_had_pretest_workflow()` (paper Section 4.2-4.3). **Phase 3 implementation (2026-04):** `aggregate="overall"` (default, two-period) runs QUG + Stute + Yatchew on a two-period panel; step 2 is NOT run on this path because a two-period panel has no pre-period placebo horizon. **Phase 3 follow-up (2026-04):** `aggregate="event_study"` (multi-period) runs QUG at F + joint pre-trends Stute + joint homogeneity-linearity Stute; closes the paper step-2 gap.
- [x] Warnings for staggered treatment timing (direct users to existing `ChaisemartinDHaultfoeuille` in diff-diff). **Phase 4 closure (2026-05-20):** fail-closed `ValueError` at `diff_diff/had.py:1511` when multiple first-treat cohorts are detected without `first_treat`; the error message directs the user to either supply `first_treat` (which activates the last-cohort + never-treated auto-filter per Appendix B.2) or to use `ChaisemartinDHaultfoeuille` (`did_multiplegt_dyn`) for full staggered support. The fail-closed choice (over `UserWarning`) is documented in REGISTRY Deviations § "Staggered-timing fail-closed" as a library extension toward stricter safety than the paper's "Warn" prescription.
- [x] Warnings for extensive-margin effects / positive mass of untreated (not fatal; suggests running existing DiD). **Closed 2026-06-01:** `HeterogeneousAdoptionDiD.fit()` now emits a fit-time `UserWarning` on the **overall** path when `>= 10%` of units have an exactly-zero post-period dose — pointing the user to a standard DiD per the Section 4 recommendation. The 10% cutoff is a library convention (the paper prescribes "warn" with NO numeric threshold and explicitly retains small untreated shares, e.g. Garrett et al.'s 12/2954 ≈ 0.4% with close-to-nominal coverage), chosen ~25× above that kept example. Overall-path-only because the event-study path *requires* never-treated units per Appendix B.2 (so an untreated mass is expected there, not a misuse signal). This complements the pre-existing `qug_test()` zero-dose `UserWarning`, which surfaces the *presence* of extensive-margin / positive-mass-of-untreated units only when the user runs the pre-tests. Documented in REGISTRY § HeterogeneousAdoptionDiD ("Note (Extensive-margin / positive-untreated-mass fit-time warning)"); locked by `tests/test_methodology_had.py::TestHADDeviations::test_extensive_margin_warning_is_10pct_library_convention`.
- [x] Documentation of non-testability of Assumptions 5 and 6. **Phase 4 closure (2026-05-20):** `HeterogeneousAdoptionDiD.fit()` emits a `UserWarning` at fit time when `resolved_design ∈ {continuous_near_d_lower, mass_point}` (Design 1 family) explicitly flagging that point identification of `WAS_{d_lower}` requires Assumption 6, sign identification requires Assumption 5, and NEITHER is testable via pre-trends (`diff_diff/had.py`, search for "---- Assumption 5/6 warning on Design 1 paths ----"). The `HeterogeneousAdoptionDiD` class docstring + `qug_test` / `stute_test` / `yatchew_hr_test` / `did_had_pretest_workflow` Notes sections cross-reference this and explicitly state that the available pre-tests verify ADJACENT identifying conditions: QUG tests the Theorem 4 / Design 1' support-infimum null `d_lower = 0` — adjacent evidence on the `d_lower = 0` clause of Assumption 4 only, NOT a test of full Assumption 4's boundary-density / conditional-mean smoothness / variance regularity statement; the raw `stute_test` / `yatchew_hr_test` helpers test Assumption 8 linearity (residuals from `dy ~ 1 + d`); `joint_pretrends_test` tests Assumption 7 mean-independence (intercept-only residuals via `null_form="mean_independence"`). None of these test Assumptions 5 or 6 directly. The composite workflow verdict string does NOT mention Assumptions 5 or 6 — it only flags the Assumption 7 step-2 gap on the two-period `aggregate="overall"` path. The Assumption 5/6 caveat is surfaced separately by the Design 1 fit-time `UserWarning` and by T21 tutorial prose.
- [x] Multi-period event-study extension (Appendix B.2). **Phase 2b implementation (2026-04):** `aggregate="event_study"` returns per-event-time WAS estimates using uniform `F-1` anchor. Staggered-timing contract (see L190 closure for full statement): when `first_treat` is supplied, the panel auto-filters to last-cohort + never-treated units with a `UserWarning` per Appendix B.2 prescription; when omitted on a multi-cohort panel, the estimator raises `ValueError` (fail-closed, see REGISTRY § "Library extension: Staggered-timing fail-closed"). Pointwise CIs per horizon (no joint cross-horizon covariance; matches paper's Pierce-Schott Figure 2). Pre-period placebos at `e <= -2`; the anchor `e = -1` is skipped since `ΔY = 0` there by construction.
- [x] Joint Stute tests (paper Section 4.2 step 2 + Section 4.3 joint extension, pages 23-25 + 32). **Phase 3 follow-up (2026-04):** `stute_joint_pretest()` (residuals-in core) + `joint_pretrends_test()` (mean-independence null) + `joint_homogeneity_test()` (linearity null) in `diff_diff/had_pretests.py`. Sum-of-CvMs aggregation, shared-η Mammen wild bootstrap across horizons (Delgado-Manteiga 2001), per-horizon exact-linear short-circuit. **Eq (18) linear-trend detrending variant SHIPPED (PR #389):** the `trends_lin: bool = False` keyword-only kwarg on `HeterogeneousAdoptionDiD.fit(aggregate="event_study")`, `joint_pretrends_test`, and `joint_homogeneity_test` applies the per-group linear-trend slope `Y[g, F-1] - Y[g, F-2]` adjustment. R parity validated against `DIDHAD::did_had(..., trends_lin=TRUE)` v2.0.0 (`Credible-Answers/did_had`) — see REGISTRY § "Note (Phase 4 — Eq 17 / Eq 18 linear-trend detrending shipped)". The Pierce-Schott (2016) NUMERICAL REPLICATION against the published p=0.51 anchor on the LBD-restricted panel is waived per REGISTRY Deviations Note #3.

**Eq (18) transcription (paper page 31):** The Pierce-Schott linear-trend-detrended joint Stute test of pre-trends reads
```
E( Y_{g,t} − Y_{g,1999} − (t − 1999)·(Y_{g,2000} − Y_{g,1999}) | D_{g,2001} ) = μ_t  ∀ t ∈ {1998, 1997}
```
The paper reports p=0.51 on US-China tariff data. The detrended outcome replaces a raw first-difference: `Y_{g,t}` is first linearly-detrended using the `(Y_{g,2000} − Y_{g,1999})` pre-period slope per unit, then tested against `D_{g,2001}` via the joint CvM. The Phase 3 follow-up ships the simpler mean-independence joint Stute (no detrending); Phase 4 extends it with the Eq (18) detrending wired to the Pierce-Schott replication.

**Joint Stute construction (paper Section 4.2-4.3 non-linear-trend variant, Phase 3 follow-up delivery):** For a set of horizons `{t_1, ..., t_K}` with residuals `{ε̂_{g,k}}_{k}` per unit and shared doses `D_g`:
1. Per-horizon CvM `S_k = (1/G²) · Σ_g (Σ_{h ≤ g(dose-order)} ε̂_{(h),k})²` (tie-safe via block-collapsed cumsum).
2. Joint statistic `S_joint = Σ_k S_k`.
3. Wild bootstrap p-value `p = (1 + #{S*_b ≥ S_joint}) / (B+1)`, with `η_g` drawn once per iteration and applied SHARED across horizons per unit (vector-valued empirical-process convention). Per-horizon OLS refit on the same design matrix each iteration; `(X'X)^{-1}X'` precomputed.
The paper's text does not prescribe the joint aggregation rule; sum-of-CvMs is the standard joint specification-test construction (Delgado 1993; Escanciano 2006).

---

## Implementation Notes

### Candidate class names (to choose later)

The paper does not prescribe a class name. Reasonable candidates:
- `HeterogeneousAdoptionDiD` - closest to the paper's own terminology (HAD).
- `DiDNoUntreated` - describes the problem setting from the practitioner's angle.
- `WeightedAverageSlopeDiD` / `WASDiD` - names the target parameter.
- `DidHad` - mirrors the Stata/R command name.

The authors identify the design as a "Heterogeneous Adoption Design", so `HeterogeneousAdoptionDiD` is the most faithful to the paper; `DidHad` is the name their reference implementation uses. Pick one after surveying the library's existing naming conventions.

### Relation to Existing diff-diff Estimators

This estimator solves the INVERSE of the few-treated-many-donors problem that motivates synthetic-control-style methods: here, the entire population is treated with heterogeneous dose and there is no genuine control group. A quasi-untreated group (QUG) of units with `D_2` local to zero serves as the control via local-linear regression at the boundary - effectively borrowing the RDD identification strategy for DiD.

Known overlap and distinctions:
- **`ChaisemartinDHaultfoeuille`** (de Chaisemartin, D'Haultfœuille 2020, AER 110(9)) - addresses heterogeneous treatment effects in TWFE with an untreated group available. The new paper is complementary: it targets the case `ChaisemartinDHaultfoeuille` cannot handle (no untreated group at all). Theorem 5 and the TWFE decomposition (Equation 14) directly generalize the dCDH 2020 weight analysis to HADs.
- **`ContinuousDiD`** (Callaway, Goodman-Bacon, Sant'Anna 2024) - addresses continuous treatment but assumes an untreated group exists. This paper's contribution is specifically removing that assumption.
- **`TripleDifference`** - unrelated; triple-diff assumes an untreated subgroup exists inside the treatment group.
- **`SyntheticDiD`** - targets single-treated-unit / few-treated-units designs with many donors; this paper targets the opposite regime (all units treated, donors approximated via local kernel).
- **`MultiPeriodDiD`** (simultaneous event study) - conceptually closest for the multi-period extension (Appendix B.2), where all treated units start at a common date `F` and results apply to every `t ≥ F` with periods redefined as `F-1` and `t`. The joint Stute test across `t ≥ F` (to avoid multiple-testing) is the natural multi-period diagnostic.

Code reuse opportunities:
- Local-linear regression backend / kernel utilities: if diff-diff does not already have them, wrap Calonico et al. (2018) bias-correction code paths via a Python port or external dependency.
- Mammen wild-bootstrap weights: `DifferenceInDifferences` already supports Rademacher, Mammen, Webb weight distributions - reuse the Mammen path.
- HC2 SE with Bell-McCaffrey correction: shared infrastructure with `DifferenceInDifferences` / `TwoWayFixedEffects` small-cluster inference.
- Stute test and Yatchew test are SEPARABLE diagnostics that could ship as standalone utilities (e.g., `diff_diff.diagnostics.stute_test`, `diff_diff.diagnostics.yatchew_test`) - not only embedded inside the WAS estimator. They test a hypothesis about `E(ΔY | D_2)` that is applicable beyond this specific design (e.g., as a TWFE pre-test in other continuous-dose settings).

Empirical-validation anchor (roadmap commit criterion):
- The paper uses Pierce and Schott (2016) "The surprisingly swift decline of US manufacturing employment" (AER 106(7), 1632-62) - China PNTR tariff data, 103 US industries, 1997-2005 - as a main empirical application (Section 5.2). Replicating the paper's Figure 2 with the new estimator is the roadmap validation target.
- The second application is Garrett, Ohrn, Suárez Serrato (2020) "Tax Policy and Local Labor Market Behavior" (AER: Insights), 2,954 US counties, 1997-2012 (bonus depreciation). Nonparametric results "are close to" the authors' TWFE results, demonstrating robustness to heterogeneous effects (Conclusion, p. 33).

### Data Structure Requirements
- Panel with at least two time periods: `t=1` (pre, all units have `D=0`) and `t=2` (post, heterogeneous doses). Multi-period extension (Appendix B.2) accepts any panel with a common treatment date `F`.
- Required columns: unit id, time id, outcome `Y`, dose `D`. Optional: covariates `X` for the (future-work) Theorem 6 extension.
- For the multi-period event-study extension, the panel should support differencing `Y_{g,t} - Y_{g,F-1}` for all post-periods `t ≥ F`, and `Y_{g,t} - Y_{g,1}` for pre-periods.

### Computational Considerations
- Nonparametric WAS estimator `β̂_{h*_G}^{np}` via local-linear regression: O(G) per bandwidth evaluation; bandwidth selection adds constant-factor overhead from Calonico et al. pilot regressions.
- QUG null test: O(G) for the min/second-min (or `O(G log G)` with standard sort). Tuning-parameter-free; only a single critical value `1/α - 1`.
- Stute test: vectorized form in Appendix D uses a `G × G` lower-triangular cusum matrix. Runtime benchmark (Stata, Table 3 in Online Appendix):

  | G       | stute_test  | yatchew_test |
  |---------|-------------|--------------|
  | 50      | 0.021 s     | 0.309 s      |
  | 500     | 0.022 s     | 0.186 s      |
  | 5,000   | 0.945 s     | 0.192 s      |
  | 50,000  | 113.923 s   | 0.419 s      |
  | 500,000 | memory fail | 0.379 s      |
  | 5 M     | memory fail | 2.250 s      |
  | 50 M    | memory fail | 24.200 s     |

  Stute fails memory allocation around `G = 100,000`. Yatchew-HR scales sub-linearly to `G = 50 M`.
- Recommended: switch from Stute to Yatchew-HR for `G ≥ 100,000` or whenever heteroskedasticity is plausible.
- 2SLS in the mass-point case: closed-form (standard 2SLS).
- Multivariate covariate extension (Equation 19 / Theorem 6): would require multivariate nonparametric regression; not yet available in Calonico et al. (2018). Treat as FUTURE WORK; flagged explicitly by the authors.

### Tuning Parameters

| Parameter        | Type     | Default                            | Selection Method |
|------------------|----------|-------------------------------------|------------------|
| `h_G` (bandwidth) | float    | data-driven (Calonico et al. 2018) | Optimal-MSE bandwidth selector; plug-in from `nprobust`. |
| `kernel`         | string   | Epanechnikov                        | Paper uses Epanechnikov; bounded-support kernels required (Assumption 4 point 4). |
| `α` (level)       | float    | 0.05                                | Standard. QUG test rejection region is `{T > 1/α - 1}`; Stute/Yatchew use `α` for the critical value. |
| `B` (bootstrap reps for Stute) | int | Paper does not specify; typical 499 or 999 | User choice. Paper gives vectorized implementation but does not recommend a count. |
| Linearity test choice | enum | Stute for `G < 100k` and homoskedastic; Yatchew-HR otherwise | Runtime/heteroskedasticity-driven. |
| `d̲` (for Design 1 continuous) | float | `min_g D_{g,2}` | Rate `G` estimation is asymptotically negligible versus `G^{2/5}` nonparametric rate. |

### Pre-testing workflow (Section 4.2-4.3)

The authors propose a four-step decision rule for TWFE reliability in HADs:
1. Test the null of a QUG (`H_0: d̲ = 0`) using the Theorem 4 order-statistic test.
2. Run a pre-trends test of Assumption 7 (requires a pre-period `t=0`).
3. Test that `E(ΔY | D_2)` is linear (Stute or Yatchew-HR).
4. If NONE of these tests is rejected, `β̂_{fe}` from TWFE may be used to estimate the treatment effect.

Post-test inference validity:
- Under Theorem 5 and de Chaisemartin and D'Haultfœuille (2024, arXiv:2407.03725), if `E(ΔY | D_2)` is linear, `E[Y_1 - Y_0 | D_2] = E[Y_1 - Y_0]`, and Assumptions 3 and 7 hold, then inference on AS conditional on accepting the two tests remains valid.
- Li et al. (2024, Theorem 2.4) implies the QUG test is asymptotically independent of the TWFE estimator; inference conditional on accepting the QUG test is asymptotically valid. The authors CONJECTURE (but do not prove) that joint conditioning on all three tests preserves valid inference.
- For Yatchew-HR: Theorem 7 Point 1 states inference on `β̂_{fe}` conditional on accepting the linearity test is asymptotically valid.

### TWFE connections (Theorem 5 and Equation 14)

- TWFE slope: `β_{fe} = E[(D_2 - E(D_2)) ΔY] / E[(D_2 - E(D_2)) D_2]` under Assumption 1.
- Weighted-CAS decomposition (Equation 14, under Assumption 7):

      β_{fe} = E{ [(D_2 - E(D_2)) D_2 / E((D_2 - E(D_2)) D_2)] · E(TE_2 | D_2) }

  Weights are proportional to `(d - E(D_2)) · d`; some are necessarily negative when `P(0 < D_2 < E(D_2)) > 0`.
- Theorem 5 (page 23):
  1. Design 1 + Assumptions 7 and 8 ⟹ `E(ΔY | D_2) = β_0 + β_{fe} · D_2`.
  2. Design 1' + Assumptions 3 and 7, and `E(ΔY | D_2) = β_0 + β_{fe} · D_2` ⟹ Assumption 8 holds and `β_{fe} = WAS`.
  3. Design 1 + Assumptions 3 and 7, and `E(Y_2(d̲) - Y_2(0) | D_2) = δ_0`, and linearity ⟹ `β_{fe} = WAS_{d̲}`.
- Diagnostic consequence (Pierce-Schott, Section 5.2): with industry-specific linear trends (Equation 17), twowayfeweights reports 62 positive and 41 negative weights summing to `-0.32` - far from a convex combination. With the Stute test of homogeneity not rejected (p = 0.40), the authors conclude the TWFE estimate is PROBABLY reliable despite the weights, but warn the test may lack power with `G = 103` (Section 5.2 caveat and main text p. 32).

### Numerical details from the applications (Sections 5.1, 5.2)

**Bonus depreciation (Garrett et al. 2020)**:
- `G = 2,954` US counties, `1997-2012`, `T=16`.
- 12 units with `D_g = 0`; kept in sample. After excluding them, the QUG test gives `D_{2,(1)} = 0.044`, `D_{2,(2)} = 0.069`, `T = 1.77`, p-value `= 0.361` - null of `d̲ = 0` NOT rejected.
- Nonparametric event-study estimators via Equation 7, bias-corrected CIs via Equation 8.
- Multi-period via Appendix B.2: one estimator per post-period `t ∈ {2002, 2003, 2004, ...}` using outcome `Y_{g,t} - Y_{g,2001}`.

**PNTR (Pierce and Schott 2016)**:
- `G = 103` US industries, `1997-2002` and `2004-2005` (2003 dropped from data).
- `D_{g,t}` = potential tariff spike eliminated by PNTR, mean 30pp, SD 14pp, zero for `t < 2001`.
- No untreated group. QUG test: `D_{2,(1)} = 0.020`, `D_{2,(2)} = 0.024`, `T = 6.150`, p-value `= 0.140` - null not rejected.
- Nonparametric event-study (Equation 7 with Equation 8 CIs); TWFE with HC2 + Bell-McCaffrey DOF; TWFE with industry-specific linear trends (Equation 17).
- Joint Stute test (Equation 18) of the linear-trends Assumption: p-value `= 0.51`.
- Homogeneity Stute test (main text p. 32): p-value `= 0.40` (non-rejection attributed partly to low power with `G = 103`, four years of data).
- Under Appendix B.2, the WAS estimator is computed per post-period; Figure 2 reports nonparametric pointwise CIs alongside TWFE and TWFE-with-linear-trends. Paper acknowledges "NP estimators are too noisy to be informative in this application" (GAP: Figure 2 reading in extraction notes).

---

## Gaps and Uncertainties

**1. Equation 7 / Equation 8 construction details**
The extraction files consistently reference Equations 7 and 8 but the explicit construction of `μ̂_h` (the intercept of the local-linear regression) and the full bias-correction machinery is in Section 3.1.3-3.1.4 of the paper. Agent 1 covered pages 1-20 and thus has Equations 7-8, but the bias-correction mechanics (Calonico-Cattaneo-Farrell) are referenced as imported from `nprobust`. Implementers should consult Calonico et al. (2018, 2019) directly for `M̂_h`, `V̂_h` formulas.

**2. Stute wild-bootstrap notation (page 25) — resolved on re-read**
Agent 2 flagged `ΔD_g` versus `D_{g,2}` in the bootstrap DGP as a potential typo. Re-reading confirms it is **not a typo**: the paper uses the first-difference operator `Δ` throughout, and because all units are untreated at period one (`D_{g,1} = 0`), `ΔD_g = D_{g,2} - D_{g,1} = D_{g,2}` identically. The expression `β̂_0 + ΔD_g · β̂_{fe} + ε̂*_{lin,g}` and `β̂_0 + D_{g,2} · β̂_{fe} + ε̂*_{lin,g}` are equivalent in this setting. Implementations can use either form; prefer `D_{g,2}` for clarity.

**3. Bootstrap iteration count for Stute**
Not specified in the extraction pages. Standard practice (499 or 999 Mammen draws) is reasonable default.

**4. Conditional inference after the QUG pre-test**
The authors CONJECTURE asymptotic independence between the QUG test and downstream WAS inference (Li et al. 2024 result on extreme order statistics and sample averages), but state explicitly that extending this to triangular arrays is NOT proven (page 21 top, Footnote 8). Document this as a "best-effort" assumption in the implementation.

**5. Multivariate covariate extension (Equation 19, Theorem 6)**
Authors acknowledge that implementing Equation 19 requires multivariate nonparametric regression and that Calonico et al. (2018) covers only univariate. Explicitly FUTURE WORK. Implementation should either defer this or warn users about the lack of bias-correction machinery.

**6. The `f_{D_2}(0) > 0` density assumption**
Assumption 4 Point 1 requires positive density at the boundary, which is nontrivial. Simulations (DGP 2, DGP 3 in Section 3.1.5) suggest CIs retain close-to-nominal coverage even when `f_{D_2}(0) = 0`, but this is an empirical observation, not a theoretical guarantee. Flag for users whose `D_2` distribution has a density vanishing at zero.

**7. Variation-in-treatment-timing caveat**
Appendix B.2 is unambiguous that "in designs with variation in treatment timing, there must be an untreated group". For staggered designs without untreated units, `did_had` covers only the LAST treated cohort. The implementation should raise an error (or very loud warning) if users invoke this estimator with staggered timing and no untreated subgroup - redirecting them to `ChaisemartinDHaultfoeuille` / `did_multiplegt_dyn`.

**8. Proof details (Appendix C.4 Theorem 4)**
The Theorem 4 proof relies on the Shorack-Wellner (1986) spacings representation (Equation 23), the extended continuous mapping theorem (van der Vaart 2000 Theorem 18.11), and Arzela-Ascoli. These are well-known tools; no re-derivation required in implementation, but if a Python port of the test is written, validating on the limit law `T_λ = (λ + E_1) / E_2` (with `E_i` i.i.d. Exponential(1)) is straightforward and recommended.

**9. Theorem 7 (Yatchew-HR) proof details (Appendix E.1)**
The proof uses concomitants of order statistics, Lindeberg CLT (Hall and Heyde 2014 Corollary 3.1), Gut (1992) for conditional Lindeberg, and a martingale filtration `F_g = σ(D_2, (ε_{(g')})_{g' < g})`. Equations 30-46 establish asymptotics for `σ̂²_{lin}`, `σ̂²_{diff}`, `σ̂⁴_W`. Not load-bearing for implementation but useful for a unit-test that validates the `N(0, 1)` limit.

**10. Simulation study DGP specifications (Section 3.1.5, page 15)**
Section 3.1.5 (page 15) defines three DGPs used for Table 1's 2,000 simulations at `G ∈ {100, 500, 2500}`. These are in the paper and directly usable as a regression-test harness when implementing:
- **DGP 1:** `D_2 ~ Uniform(0,1)`; `ΔY(0) ~ N(0,1)` independent of `D_2`; `ΔY_2(D_2) = D_2 + D_2² + ΔY(0)`. Implies `WAS = 5/3`. Assumptions 2, 3, 4 all hold. Coverage of 95% BCCI: 89% at G=100, 93% at G=500, 95% at G=2500.
- **DGP 2:** Same as DGP 1 but `D_2 ~ Beta(2,2)`. Implies `WAS = 8/5`. Assumption 4 fails (`f_{D_2}(0) = 0`). Coverage: 90% at G=100, ~95% at G=2500.
- **DGP 3:** `D_2` drawn without replacement from the empirical distribution of Pierce-Schott (2016); `ΔY(0)` drawn without replacement from the empirical `(Y_{g,2} - Y_{g,1})` of Pierce-Schott; `ΔY_2(D_2) = ΔY(0)`. Implies `WAS = 0`. Assumption 4 fails. Coverage: 92% at G=100, ~95% at G=2500.
These three DGPs constitute a natural validation harness: reproduce Table 1's point-estimate column and coverage column to within Monte Carlo error (2,000 sims per cell).

**11. Garrett et al. (2020) replication details**
Beyond the QUG test numbers (T = 1.77, p = 0.361), the nonparametric event-study results and their comparison to TWFE are referenced in the Conclusion (p. 33) but not detailed in the extracted pages. Implementers validating against this application should consult Section 5.1 directly.
