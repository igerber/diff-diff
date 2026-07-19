# Paper Review: Simple approaches to nonlinear difference-in-differences with panel data

**Authors:** Jeffrey M. Wooldridge
**Citation:** Wooldridge, J. M. (2023). Simple approaches to nonlinear difference-in-differences with panel data. *The Econometrics Journal*, 26(3), C31–C66. https://doi.org/10.1093/ectj/utad016
**PDF reviewed:** https://doi.org/10.1093/ectj/utad016 (Econometrics Journal — official article URL)
**Review date:** 2026-03-21

---

## Methodology Registry Entry

*Formatted to match docs/methodology/REGISTRY.md structure.*

## WooldridgeDiD (ETWFE)

**Primary source:** Wooldridge, J. M. (2023). Simple approaches to nonlinear difference-in-differences with panel data. *The Econometrics Journal*, 26(3), C31–C66. https://doi.org/10.1093/ectj/utad016

**Secondary source:** Wooldridge, J. M. (2021). Two-way fixed effects, the two-way Mundlak regression, and difference-in-differences estimators. SSRN Working Paper. https://doi.org/10.2139/ssrn.3906345

**Key implementation requirements:**

*Assumption checks / warnings:*
- **No Anticipation (NA):** `E[Y_1(1) - Y_1(0) | D = 1] = 0` (Eq 2.2). Pre-treatment outcomes unaffected by future treatment.
- **Linear Parallel Trends (LPT):** `E[Y_2(0)|D] - E[Y_1(0)|D] = γ_2` (Eq 2.5). Trend in untreated PO is the same across groups. Only valid for continuous/unbounded outcomes.
- **Index Parallel Trends (IPT):** For a known strictly increasing `G(·)`, `E[Y_t(0)|D] = G(α + βD + γ_t)` (Eq 2.6-2.7). PT holds on the index inside `G(·)`, not on levels. This is the key nonlinear extension.
- **Conditional NA, Staggered (CNAS):** `E[Y_t(g)|D_g = 1, X] = E[Y_t(∞)|D_g = 1, X]` for `t < g` (Eq 3.3).
- **Conditional IPT, Staggered (CIPTS):** `E[Y_t(∞)|D_q,...,D_T, X] = G(α + Σ β_g D_g + Xκ + Σ (D_g · X) η_g + γ_t + Xπ_t)` (Eq 3.4). The change in the index does not depend on cohort assignment `D`, conditional on `X`.
- **Overlap:** `P(D = 1|X = x) < 1` for all `x ∈ Supp(X)`. Equivalently, `Supp(X|D=1) ⊂ Supp(X|D=0)`.

*Target parameter — two-period case (Equation 2.1):*

    τ_2 = E[Y_2(1) - Y_2(0) | D = 1]

*Target parameter — staggered case (Equation 3.2):*

    τ_{gr} = E[Y_r(g) - Y_r(∞) | D_g = 1], r = g,...,T; g = q,...,T

where `g` is the cohort (first treatment period), `r` is calendar time, `D_g` are cohort indicators, `D_∞ = 1` indicates never-treated.

*Main estimator — two-period, no covariates (Equation 2.17):*

    τ̂_2 = Ȳ_12 - G(α̂ + β̂ + γ̂_2)
         = Ȳ_12 - G(G⁻¹(Ȳ_11) + (G⁻¹(Ȳ_02) - G⁻¹(Ȳ_01)))

where `Ȳ_{dt}` is the sample average for group `d` in period `t`. Special cases:
- Linear (`G(z) = z`): reduces to standard DiD `(Ȳ_12 - Ȳ_11) - (Ȳ_02 - Ȳ_01)` (Eq 2.18)
- Exponential (`G(z) = exp(z)`): `τ̂_2 = Ȳ_12 - Ȳ_11 · (Ȳ_02/Ȳ_01)` (Eq 2.19)

*Index-scale effect (Equation 2.20):*

    δ_2 ≡ G⁻¹(E[Y_2(1)|D=1]) - G⁻¹(E[Y_2(0)|D=1])
        = G⁻¹(E[Y_2(1)|D=1]) - (α + β + γ_2)

Interpretation depends on the link function:
- Exponential mean (`G(z) = exp(z)`, Poisson): `δ_2` is a log difference, i.e., a proportional/log rate effect on `E[Y|...]`.
- Logistic mean (`G(z) = Λ(z) = 1/(1 + exp(-z))`, logit): `δ_2` is a change in log-odds of `E[Y|...]`.

*Imputation estimator — staggered with covariates (Procedure 1, Eq 3.10-3.11):*

    Step 1: Using W_{it} = 0 observations, estimate (α, β_q,...,β_T, κ, η_q,...,η_T, γ_2,...,γ_T, π_2,...,π_T) by pooled QMLE in the LEF.

    Step 2: For cohort g, impute counterfactual:
        Ŷ_{igr}(∞) ≡ G(α̂ + β̂_g + X_i κ̂ + X_i η̂_g + γ̂_r + X_i π̂_r), r = g,...,T     (3.10)

    Step 3: ATT estimator:
        τ̂_{gr} = N_g⁻¹ Σ_i D_{ig} [Y_{ir} - Ŷ_{igr}(∞)]                                (3.11)

*Pooled QMLE estimator — staggered with covariates (Procedure 2, Eq 3.12-3.15):*

    E(Y_{it}|D_{iq},...,D_{iT}, X_i, W_i) = G[α + Σ β_g D_{ig} + X_i κ + Σ (D_{ig} · X_i) η_g
        + Σ γ_s f s_t + Σ (f s_t · X_i) π_s
        + ΣΣ δ_{gs} (W_{it} · D_{ig} · f s_t)
        + ΣΣ (W_{it} · D_{ig} · f s_t · X̂_{ig}) ξ_{gs}]                                  (3.12)

where `X̂_{ig} = X_i - X̄_g` are cohort-centred covariates, `W_{it} = D_{ig} · (f q_t + ... + f T_t)` is the time-varying treatment indicator, and `f s_t` are time period dummies.

    ATT estimator from pooled method:
        τ̃_{gr} = N_g⁻¹ Σ_i D_{ig} [G(α̃ + β̃_g + X_i κ̃ + X_i η̃_g + γ̃_r + X_i π̃_r + δ̃_{gr} + X̂_{ig} ξ̃_{gr})
            - G(α̃ + β̃_g + X_i κ̃ + X_i η̃_g + γ̃_r + X_i π̃_r)]                         (3.15)

This is the Average Structural Function (ASF) approach: `ATT(g,r) = mean_i[G(η_i + δ_{gr}) - G(η_i)]`.

*Equivalence result (Proposition 3.1):*

When `G⁻¹(·)` is the canonical link function for the chosen LEF density, and the pooled QMLE solution is unique:
- Parameter estimates from imputation (Procedure 1) and pooled (Procedure 2) are identical
- ATT estimates are identical: `τ̃_{gr} = τ̂_{gr}`

This holds for: OLS (identity link), logit (logistic link with Bernoulli), Poisson QMLE (log link with Poisson). Proven in Appendix A.

*Common timing simplification (Equation 3.17):*

When all units are treated at the same time `g`, the model simplifies to:

    E(Y_{it}|D_i, X_i, W_i) = G[α + βD_i + X_i κ + (D_i · X_i) η
        + Σ γ_s f s_t + Σ (f s_t · X_i) π_s
        + Σ δ_s (W_{it} · D_i · f s_t) + Σ (W_{it} · D_i · f s_t · X̂_{ig}) ξ_{gs}]       (3.17)

*Standard errors (Section 3, p. C47):*
- Delta method for individual `τ̂_{gr}` and aggregated ATTs (Wooldridge 2010, problem 12.17)
- Panel bootstrap (resampling units) is also valid in the paper's framework — allows for serial dependence and model misspecification
- Cluster-robust SEs at unit level for pooled estimation
- For nonlinear models: standard sandwich `(X'WX)⁻¹ meat (X'WX)⁻¹` where `W = diag(μ_i(1-μ_i))` for logit or `W = diag(μ_i)` for Poisson
- **Note (shipped API restriction):** in `WooldridgeDiD`, `n_bootstrap > 0` is currently OLS-only (rejected for logit/Poisson at `diff_diff/wooldridge.py:432-437`) and rejected when `survey_design` is set (`diff_diff/wooldridge.py:441-444`). Use analytical SEs for nonlinear or survey paths.

*Aggregation (Section 3.1, p. C50):*
- **Simple (static):** weighted average of immediate effects `τ̃_{sg}` across cohorts `g = q,...,T`, weights = cohort proportions among all eventually treated
- **Dynamic (event study):** weighted average of `τ̃_{g,g+h}` for horizon `h ≥ 0` across cohorts, weights = cohort proportions
- **Calendar time:** weighted average across cohorts for each calendar period
- **Group:** weighted average across periods for each cohort
- SEs for all aggregations via delta method or panel bootstrap
- **Note (current implementation deviation):** the shipped `WooldridgeDiD` aggregations use cell-level observation-count weights `n_{g,t}` (matching Stata `jwdid_estat`) rather than the cohort-share weights described conceptually in Section 3.1. The 2023 paper does not provide explicit aggregation-weight equations; the formal cohort-share equations referenced in `docs/methodology/REGISTRY.md` ("W2025 Eqs. 7.2-7.4") are from a later Wooldridge ETWFE source. See `docs/methodology/REGISTRY.md` "Aggregations" under WooldridgeDiD and the corresponding line in `DEFERRED.md` ("Needs external reference (R / Stata / Julia)" → WooldridgeDiD follow-up cluster) for the tracked deviation.

*Testing parallel trends (Section 4):*

Two approaches:
1. **Event-study test (Section 4.1):** Include pre-treatment interactions `D_{ig} · f s_t` for `s = {2,...,g-1}` and test joint significance. With canonical link, test is the same whether using `W_{it} = 0` subsample or pooling all observations.
2. **Heterogeneous linear trends (Section 4.2):** Add cohort-specific linear trends `D_{ig} · t`. Conserves degrees of freedom vs event-study approach. Tests via cluster-robust Wald statistic.
- Adding cohort-specific linear trends `D_{ig} · t` to the expanded equation (Eq 4.1) provides a sensible correction when PT is violated

*Edge cases:*
- **Binary outcomes:** Use logistic function `G(z) = Λ(z) = exp(z)/[1 + exp(z)]` with Bernoulli QLL. Fractional outcomes use same setup (Papke & Wooldridge 1996, 2008).
- **Count/nonnegative outcomes:** Use exponential `G(z) = exp(z)` with Poisson QLL. Does not require Poisson distribution — only exponential conditional mean.
- **Corner solutions (`Y ≥ 0` with mass at zero):** Poisson QMLE still valid. Exponential mean handles zeros naturally (p. C56-C57).
- **All units eventually treated (Section 7.1):** Methods work with minor modification. Last-treated cohort (`g = T`) serves as control. ATTs defined relative to `Y_t(T)` rather than `Y_t(∞)`.
- **Treatment exit (Section 7.2, extension):** Expand cohort definition to `D_{gh}` where `g` = start period, `h` = end period. ATTs `τ_{ghr}` defined for `r = g,...,T`. The paper notes this extension requires the additional restriction that future shocks to untreated potential outcomes do not drive exit; absent that condition, exit timing becomes endogenous and the standard ETWFE identification argument no longer carries over directly.
- **Time-varying covariates (Section 7.3):** Replace `X_i` with `X_{it}` in Eq 3.12. Only valid if covariates are not influenced by the intervention.
- **Multiple treatment levels (Section 7.4, extension):** Replace binary `W_{it}` with a set of treatment-level indicators; define cohorts by `D_{iga}` (first period `g`, initial level `a`). The paper describes this only as relatively straightforward, not fully general, and leaves the precise multi-level estimand to future work — so the exact ATT object under non-binary treatment is not pinned down in Wooldridge (2023).
- **Incidental parameters:** For logit, unit-specific dummies cause incidental parameters problem. Pooled QMLE with cohort dummies (not unit dummies) avoids this. For Poisson with exponential mean, FE Poisson estimator does NOT suffer from incidental parameters (Wooldridge 1999) — this is a unique exception.
- **No never-treated group:** When `g = T` is the last cohort, it serves as control. Cannot estimate ATT for this last cohort.

*Algorithm (Procedure 2 — Pooled Estimation, recommended):*
1. Pool all observations (treated and untreated, all periods)
2. Construct design matrix: cohort dummies `D_{ig}`, time dummies `f s_t`, covariates `X_i`, cohort×covariate interactions `D_{ig} · X_i`, time×covariate interactions `f s_t · X_i`, treatment indicators `W_{it} · D_{ig} · f s_t`, treatment×centred-covariate interactions `W_{it} · D_{ig} · f s_t · X̂_{ig}`
3. Estimate by pooled QMLE in the LEF (OLS for linear, Bernoulli QLL for logit, Poisson QLL for exponential)
4. For each treated cell (g, r): compute ASF-based ATT via Eq 3.15 — average `G(η̂_i + δ̂_{gr}) - G(η̂_i)` over units in cohort `g`
5. For linear case: `δ̂_{gr}` coefficients on treatment interactions are directly the ATTs
6. Aggregate as desired (simple, dynamic/event-study, calendar, group)
7. SEs via delta method or panel bootstrap (cluster at unit level)

**Table 1. Canonical link and log likelihood pairings (p. C44):**

| Conditional Mean | LEF Density | Use Case |
|-----------------|-------------|----------|
| Linear | Normal | Any response; leads to OLS |
| Logistic | Bernoulli | Binary or fractional response |
| Logistic | Binomial | Nonnegative response with known upper bound |
| Logistic | Multinomial | Multinomial or multiple fractional response |
| Exponential | Poisson | Nonnegative response (count, corner), no natural upper bound |

**Reference implementation(s):**
- Stata: `jwdid` package (Rios-Avila, 2021)
- R: `etwfe` package (McDermott, 2023)
- Simulations in paper performed in Stata 17

**Requirements checklist:**
- [ ] Pooled QMLE estimation with full saturated design matrix (Eq 3.12)
- [ ] Three estimation methods: OLS (identity link), logit (Bernoulli QLL), Poisson (Poisson QLL)
- [ ] ASF-based ATT computation for nonlinear models (Eq 3.15)
- [ ] Imputation-based ATT for comparison/equivalence check (Eq 3.10-3.11)
- [ ] Equivalence between imputation and pooled when using canonical link (Proposition 3.1)
- [ ] Cohort×time interaction design matrix with centred covariates
- [ ] Unit-level cluster-robust SEs
- [ ] Delta-method SEs for aggregated ATTs
- [ ] Panel bootstrap (unit resampling) as alternative inference
- [ ] Four aggregation types: simple (static), dynamic (event study), calendar, group
- [ ] Pre-treatment testing: event-study and heterogeneous trends approaches (Section 4)
- [ ] Support for both never-treated and all-eventually-treated designs (Section 7.1)
- [ ] Treatment exit support (Section 7.2)
- [ ] Time-varying covariates (Section 7.3)
- [ ] Index-scale effects `δ_{gr}` reported alongside level-scale ATTs `τ_{gr}` (Eq 3.16)

---

## Implementation Notes

### Data Structure Requirements

*Paper notation:* `Y_{it}` (outcome), `D_g` (cohort indicator), `W_{it}` (time-varying treatment), `X_i` (time-invariant covariates).

*Shipped API (`diff_diff/wooldridge.py:394-411`):* users provide outcome, unit ID, time, and `cohort` (or `first_treat`). The model derives `W_{it}` internally from `cohort` and `time` via `_build_interaction_matrix` (`diff_diff/wooldridge.py:165-189`) — users do NOT pass `W_{it}` as a column.

*Covariates (richer than paper notation):* `exovar` (time-invariant, no interaction or demeaning), `xtvar` (time-varying, demeaned within cohort×period cells when `demean_covariates=True`), `xgvar` (covariates interacted with each cohort indicator). See `docs/methodology/REGISTRY.md` under WooldridgeDiD "Covariates".

*Other contracts:*
- Balanced or unbalanced panel: N units observed over T fixed time periods.
- Treatment is absorbing: once treated, always treated (no exit unless using Section 7.2 extension).
- Cohorts defined by first treatment period `g ∈ {q, q+1, ..., T, ∞}`.

### Computational Considerations
- Pooled estimation is a single regression over all N×T observations — O(N·T·K²) where K is number of parameters
- K grows as (number of cohorts) × (number of post-treatment periods) for the interaction terms
- For nonlinear models, IRLS convergence typically fast (< 25 iterations for logit/Poisson)
- ASF computation requires averaging over all units in each cohort — O(N_g) per (g,r) cell
- Delta-method gradient for nonlinear ATTs requires `G'(η̂_i + δ̂_{gr})` and `G'(η̂_i)` per unit
- Parallelization: bootstrap iterations are embarrassingly parallel; ASF computation per cell is independent

### Tuning Parameters

| Parameter | Type | Default | Selection Method |
|-----------|------|---------|-----------------|
| `G(·)` | function | identity (OLS) | Dictated by outcome type: logistic for binary/fractional, exponential for counts/nonneg |
| control_group | str | "not_yet_treated" (matches `diff_diff/wooldridge.py:305`) | Use "never_treated" only when a pure never-treated group is available and required |
| anticipation | int | 0 | Domain knowledge; test with pre-treatment indicators |
| exovar / xtvar / xgvar | list / list / list | None / None / None | `exovar` for time-invariant, `xtvar` for time-varying (Section 7.3; demeaned via `demean_covariates=True`, default), `xgvar` for cohort-interacted. See `diff_diff/wooldridge.py:394-411`. |

### Relation to Existing diff-diff Estimators

- **CallawaySantAnna:** OLS ETWFE ATT(g,t) estimates are equivalent to CS ATT(g,t) under linear PT with never-treated control (proven in Wooldridge 2021). However: CS uses only the period just prior to treatment as comparison, while ETWFE uses all pre-treatment periods — stronger assumption but more efficient. Key difference: ETWFE extends naturally to nonlinear outcomes; CS does not.
- **ImputationDiD (BJS):** In the linear case, the imputation procedure (Procedure 1) is numerically identical to BJS imputation (Section 3.2, Proposition 3.1). With canonical link in LEF, pooled and imputation are equivalent.
- **TwoStageDiD (Gardner):** Similar two-step logic (estimate FE on controls, impute counterfactuals). ETWFE pooled approach is a single-step alternative.
- **SunAbraham:** Both produce interaction-weighted estimates. SA uses TWFE with saturated interactions; ETWFE generalises to nonlinear.
- **EfficientDiD:** Chen, Sant'Anna & Xie (2025) achieve the semiparametric efficiency bound. ETWFE does not claim efficiency — but the pooled QMLE is typically more efficient than CS in simulations (SEs 45-69% smaller than CS for logit; 31-57% smaller for Poisson — Section 5).
- **Solvers in `linalg.py`:** `solve_logit` is reused for the logit outcome path; `solve_poisson` (`diff_diff/linalg.py:3431`) is the IRLS solver used by the Poisson path (`diff_diff/wooldridge.py:1085-1124`).
- **`within_transform` in `utils.py`:** Can be used for the OLS path to absorb unit+time FE. NOT suitable for nonlinear paths — logit/Poisson require explicit cohort+time dummy columns (no within-transformation for nonlinear models due to incidental parameters).

---

## Simulation Findings (Section 5)

### Binary outcomes with common timing (Section 5.1)
- N = 500, T = 6, q = 4 (3 pre-treatment periods), 5000 MC replications
- **Logit PMLE essentially unbiased** for all three SATTs regardless of trend strength
- **POLS (linear) shows large downward bias** with strong aggregate trends (biases -0.15 to -0.29 vs true SATTs of 0.06-0.16)
- **CS (2021) also shows nontrivial negative bias** in strong-trend scenarios
- POLS SEs at least 45% larger than logit PMLE SEs; CS SEs at least 69% larger
- PT tests (event-study and heterogeneous trends) fail to detect misspecification of the linear model — rejection rates around 5% even when LPT is violated

### Nonnegative outcomes with staggered intervention (Section 5.2)
- Poisson QMLE is clear winner when exponential mean is correct
- **Poisson QMLE essentially unbiased** for all six SATTs
- **Pooled OLS shows downward biases over 30%** in all cases, over 50% in some
- **CS has less bias than POLS** but still very different from true SATTs
- Poisson QMLE precision notably better than both alternatives (SDs 31-57% smaller than CS)
- Event-study PT test rejects linear model 99.5% of the time; heterogeneous trends test rejects 99.9%
- Corner solution case (`P(Y = 0) ≈ 0.37`): Poisson QMLE still essentially unbiased despite mass at zero

### Key takeaway
The simulations demonstrate that **when the outcome is binary or count-valued, using the correct nonlinear link function produces dramatically better estimates** than linear methods. The improvement is both in bias (nonlinear: ~0; linear: up to 50% bias) and precision (nonlinear SEs 30-70% smaller).

---

## Gaps and Uncertainties

1. **Fractional logit implementation details.** The paper mentions fractional responses (Papke & Wooldridge 1996, 2008) and the Bernoulli QLL with known upper bound `B_{it}`, but does not provide explicit formulas for the bounded case. Implementation should follow standard fractional logit/probit references.

2. **Aggregation weight formulas not fully explicit.** Section 3.1 (p. C50) describes aggregation conceptually ("weighted averages... where the weights are the proportions of the treated cohorts") but does not provide explicit weight formulas with equation numbers. The `jwdid_estat` Stata command documentation should be consulted for exact weight definitions.

3. **Delta-method gradient for nonlinear ATTs.** The paper states "by applying the delta method with averaging" (p. C47, citing Wooldridge 2010 problem 12.17) but does not write out the explicit gradient. For implementation:
   - For `τ̃_{gr}` from Eq 3.15: `∂τ̃_{gr}/∂δ_{gr} = N_g⁻¹ Σ_i D_{ig} G'(η̂_i + δ̂_{gr})`
   - For other parameters `θ_k`: `∂τ̃_{gr}/∂θ_k = N_g⁻¹ Σ_i D_{ig} [G'(η̂_i + δ̂_{gr}) · ∂η̂_i/∂θ_k - G'(η̂_i) · ∂η̂_i/∂θ_k]`
   - Verify these against numerical gradients during implementation.

4. **Covariate centring.** The paper centres covariates at cohort means `X̂_{ig} = X_i - X̄_g` where `X̄_g = E(X_i|D_{ig} = 1)` (p. C48). In practice, use sample cohort means. The centring affects interpretation of `δ_{gs}` (makes it the ATT on the log-odds/log scale) but does not affect `τ_{gr}` estimates.

5. **Poisson FE vs pooled QMLE.** Section 3.3 (p. C51-C52) notes that FE Poisson (with unit dummies) does NOT suffer from incidental parameters — a unique property of the exponential mean. The paper notes that without covariates, FE Poisson and pooled QMLE give identical `γ_s` and `δ_{gs}` estimates, but with covariates they differ. The pooled approach is recommended for simplicity.

6. **Online Appendix.** The paper references an Online Appendix with detailed simulation results (p. C56: "From the findings reported in the Online Appendix..."). This appendix is not included in the main PDF and may contain additional implementation-relevant details.
