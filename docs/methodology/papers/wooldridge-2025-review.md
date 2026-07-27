# Paper Review: Two-way fixed effects, the two-way mundlak regression, and difference-in-differences estimators

**Authors:** Jeffrey M. Wooldridge
**Citation:** Wooldridge, J. M. (2025). Two-way fixed effects, the two-way mundlak regression, and difference-in-differences estimators. *Empirical Economics*, 69(5), 2545-2587. DOI: [10.1007/s00181-025-02807-z](https://doi.org/10.1007/s00181-025-02807-z). Received 27 Dec 2024 / Accepted 30 Jul 2025 / Published 27 Aug 2025.
**PDF reviewed:** Local PDFs are gitignored under `/papers/`; the journal/DOI version (https://doi.org/10.1007/s00181-025-02807-z) is the authoritative source. Published version of the 2021 SSRN working paper (also circulated as NBER WP 29154).
**Review date:** 2026-05-21

---

## Methodology Registry Entry

*Formatted to match docs/methodology/REGISTRY.md `## WooldridgeDiD (ETWFE)` section (currently at REGISTRY:1431-1547). This review is the **Primary source** for the linear / Mundlak / aggregation / equivalence-chain surface; the existing `docs/methodology/papers/wooldridge-2023-review.md` review is the **Secondary source** for nonlinear extensions (logit / Poisson / ASF counterfactual / canonical-link LEF density). The 2025 paper itself defers nonlinear extensions to the 2023 companion paper (p. 2554, Section 11 p. 2580).*

## WooldridgeDiD (ETWFE)

**Primary source:** Wooldridge, J. M. (2025). Two-way fixed effects, the two-way mundlak regression, and difference-in-differences estimators. *Empirical Economics*, 69(5), 2545-2587. https://doi.org/10.1007/s00181-025-02807-z

**Secondary source (cross-link):** Wooldridge, J. M. (2023). Simple approaches to nonlinear difference-in-differences with panel data. *The Econometrics Journal*, 26(3), C31-C66. https://doi.org/10.1093/ectj/utad016 — reviewed at `docs/methodology/papers/wooldridge-2023-review.md`. The 2025 paper defers nonlinear extensions (logit / Poisson / fractional outcomes) to this companion paper (p. 2554, p. 2580 — "Pooled OLS using cohort indicators in a flexible way extends to nonlinear models, and I have developed those in Wooldridge (2023)").

**Application reference:** Section 9 of the 2025 paper applies the method to Walmart store openings on US county log retail employment (N = 1,288 counties, 1977-1999 panel, 14 cohorts, three time-constant controls dated 1980).

**Reference implementations:**
- Stata: `jwdid` package (Rios-Avila, 2021)
- R: `etwfe` package (McDermott, 2023)

**Key implementation requirements:**

### Data Structure & Notation

*Paper notation (Sections 2-3):*

- Unit `i = 1, ..., N`; time `t = 1, ..., T`. Small-`T`, large-`N` asymptotics (`N → ∞`, `T` fixed) — the framework also accommodates large `T`.
- `y_it` = scalar outcome; `x_it` = `1 × K` row vector of variables varying across both `i` and `t` (after Section 3 redefinition, no time-period dummies inside `x_it`).
- `c_i` = unobserved unit-specific effect; `f_t` = time-specific effect; `u_it` = idiosyncratic error.
- `ch_i` = unit dummies (`ch_i = 1` if `h = i`); `fs_t` = time dummies (`fs_t = 1` if `s = t`); first-period dummy dropped.
- `dg_i` = treatment-cohort dummy (`dg_i = 1` if unit `i` is first treated in period `g`), `g ∈ {q, ..., T}`; `q ≥ 2` is the earliest treatment-onset period.
- `w_it` = binary treatment indicator. `{w_it : t = 1, ..., T}` is a sequence of zeros followed by ones; the first one appears for cohort `g` at `t = g`. Constructed as `w_it = dq·(fq_t + ... + fT_t) + d(q+1)·[f(q+1)_t + ... + fT_t] + ... + dT·fT_t`.
- `ps_t = fs_t + ... + fT_t` = post-intervention indicator for treatment starting in period `s`.
- `z_i` = time-constant variables; `m_t` = unit-constant variables (no `i` variation).
- `y_t(g)` = potential outcome in period `t` if a unit is first treated in period `g`; `y_t(∞)` = never-treated potential outcome.
- `te_t(g) = y_t(g) − y_t(∞)` = unit-level treatment effect (random variable) — Eq. 4.1.
- `τ_gt ≡ E[te_t(g) | dg = 1]` = ATT for cohort `g` at time `t` — Eq. 4.2.
- `N_g = Σ_i dg_i` = number of units in cohort `g`.
- `x = (x_1, ..., x_K)` = time-constant controls (`x ≡ x(∞)` after NBC).

*Within transformations / averages (Eqs. 2.3-2.5):*

```
x̄_i. = T⁻¹ Σ_{t=1}^T x_it                                       (2.3)
x̄_.t = N⁻¹ Σ_{i=1}^N x_it                                       (2.4)
x̄    = (NT)⁻¹ Σ_i Σ_t x_it
ẍ_it = (x_it − x̄_i.) − N⁻¹ Σ_i (x_it − x̄_i.)
     = x_it − x̄_i. − x̄_.t + x̄                                  (2.5)
```

The outcome transform is `ÿ_it = y_it − ȳ_i. − ȳ_.t + ȳ`.

### Identifying Assumptions (Section 4)

Five numbered assumptions:

- **SUTVA (Stable Unit Treatment Value Assumption).** Potential outcomes of each unit do not depend on other units' treatment assignments. Independent cross-sectional sampling implies SUTVA, though SUTVA can hold without independent sampling.

- **NBC (No Bad Controls).** Let `x(g)` be the covariates when the treatment cohort is `g`. Then `x(g) = x(∞)` for all `g ∈ {q, ..., T}`. Time-constancy of `x` (used throughout Sections 4-8) operationalizes this; covariates whose values change due to the intervention are excluded.

- **NA (No Anticipation), Eq. 4.3.** For `t < g`:

  ```
  E[y_t(g) − y_t(∞) | d, x] = 0,   t < g.                       (4.3)
  ```

  A stronger pointwise form (used elsewhere) is `y_t(g) = y_t(∞)` for `t < g`.

- **CPT (Conditional Parallel Trends), Eq. 4.7.** For `t = 2, ..., T` and time-constant controls `x`,

  ```
  E[y_t(∞) − y_1(∞) | d, x] = E[y_t(∞) − y_1(∞) | x].            (4.7)
  ```

  Conditional on `x`, cohort assignment `d` is unrelated to the trend in the never-treated state.

- **LIN (Linearity), Eqs. 4.8-4.9.** For all cohort indicators `dg`, `g ∈ {q, ..., T}`,

  ```
  E[y_1(∞) | d, x] = α + Σ_{g=q}^T β_g dg + xκ + Σ_{g=q}^T (dg·x) ξ_g            (4.8)

  E[y_t(∞) | d, x] − E[y_1(∞) | d, x] = Σ_{s=2}^T γ_s fs_t
                                          + Σ_{s=2}^T (fs_t · x) π_s,
                                          t = 2, ..., T.                          (4.9)
  ```

  Without controls, (4.8) is definitional; with controls it imposes a functional-form restriction. Equation (4.9) implies CPT because `d` does not appear on its right-hand side. `x` may contain squares, interactions, logs, etc.

CNA (Conditional No Anticipation) is implicit in Eq. 4.11 and used for identification of `τ_gt` — a stronger form than NA.

Combining (4.8) and (4.9) gives the regression basis (Eq. 4.10):

```
E[y_t(∞) | d, x] = α + Σ_{g=q}^T β_g dg + xκ + Σ_{g=q}^T (dg·x) ξ_g
                    + Σ_{s=2}^T γ_s fs_t + Σ_{s=2}^T (fs_t · x) π_s,
                    t = 1, ..., T.                                     (4.10)
```

ATT identification (Eq. 4.11):

```
τ_gt = E[y_t | dg = 1]
        − [α + β_g + γ_t + E(x | dg = 1) · (κ + ξ_g + π_t)].   (4.11)
```

`E[y_t | dg = 1]` is identified by the cohort-period sample mean; `E(x | dg = 1)` by the cohort sample average.

### Target Estimand

```
τ_gt ≡ E[te_t(g) | dg = 1] = E[y_t(g) − y_t(∞) | dg = 1],
                                g = q, ..., T;  t = g, ..., T.          (4.2)
```

ATTs are identified only in periods `t ≥ g`. Section 7 discusses aggregating these to a single overall effect or to event-time effects.

### Main Estimator Equations — Five Algebraically Equivalent Forms

The paper introduces five estimators that are algebraically identical for `τ_gt`. They share a common conditional-mean specification but differ in regressor labels.

**(A) TWFE estimator — Eqs. 2.1-2.6.**

Motivational equation (Eq. 2.1, "motivational only"):

```
y_it = x_it β + c_i + f_t + u_it,   t = 1, ..., T;  i = 1, ..., N.      (2.1)
```

TWFE regression (Eq. 2.2):

```
y_it on x_it, c1_i, c2_i, ..., cN_i, f2_t, ..., fT_t,
       t = 1, ..., T;  i = 1, ..., N.                                   (2.2)
```

Equivalent large-`T` "double demeaning" form (Eq. 2.6, Baltagi 2021):

```
y_it on ẍ_it,   t = 1, ..., T;  i = 1, ..., N.                          (2.6)
```

Same estimates if `y_it` is replaced with `ÿ_it`.

**(B) Two-way Mundlak (TWM) regression — Section 3, Eqs. 3.1-3.2.**

One-way Mundlak (Wooldridge 2019), Eq. 3.1:

```
y_it on 1, x_it, x̄_i.,   t = 1, ..., T;  i = 1, ..., N.                (3.1)
```

Two-way Mundlak (Eq. 3.2):

```
y_it on 1, x_it, x̄_i., x̄_.t, z_i, m_t,
       t = 1, ..., T;  i = 1, ..., N.                                   (3.2)
```

`x_it` here contains only variables with variation across both `i` and `t`; time-period dummies are separated out.

**Interactive-form lemma (Eqs. 3.4-3.5).** If `x_itj = z_ij · m_tj` (Eq. 3.4), then `x̄_i.j = z_ij · m̄_j` and `x̄_.tj = z̄_j · m_tj` (Eq. 3.5). Including `(z_ij, m_tj)` separately is equivalent to including the averages, so POLS and RE coincide in models whose only time-varying variable has this interactive form.

**(C) Cohort imputation estimator — Procedure 4.1, Eqs. 4.13-4.20.**

Step (i): On `w_it = 0` observations only, run pooled OLS (Eq. 4.14):

```
y_it on 1, dq_i, ..., dT_i, x_i, dq_i·x_i, ..., dT_i·x_i,
       f2_t, ..., fT_t, f2_t·x_i, ..., fT_t·x_i,                         (4.14)
```

obtaining `(α̃, β̃_q, ..., β̃_T, κ̃, ξ̃_q, ..., ξ̃_T, γ̃_2, ..., γ̃_T, π̃_2, ..., π̃_T)` (Eq. 4.15).

Step (ii): Impute (Eq. 4.16):

```
ỹ_it(∞) = α̃ + Σ_{g=q}^T β̃_g dg_i + x_i κ̃ + Σ_{g=q}^T (dg_i·x_i) ξ̃_g
           + Σ_{s=2}^T γ̃_s fs_t + Σ_{s=2}^T (fs_t·x_i) π̃_s              (4.16)
```

and form `t̃e_it = y_it − ỹ_it(∞)` (Eq. 4.17).

Step (iii) — cohort imputation ATT (Eq. 4.18):

```
τ̃_gt = N_g⁻¹ Σ_i dg_i · t̃e_it
      = ȳ_gt − [(α̃ + β̃_g + γ̃_t) + x̄_g · (κ̃ + ξ̃_g + π̃_t)],            (4.18)
```

where `ȳ_gt = N_g⁻¹ Σ_i dg_i · y_it` (Eq. 4.13) and `x̄_g = N_g⁻¹ Σ_i dg_i · x_i` (Eq. 4.19).

Consistency (Eq. 4.20): `(α̃ + β̃_g + γ̃_t) + x̄_g · (κ̃ + ξ̃_g + π̃_t) →ᵖ E[y_t(∞) | dg = 1]`.

Without covariates, this reduces to Gardner (2022)'s two-stage DiD.

**(D) Pooled OLS on cohort dummies (POLS / ETWFE without unit dummies) — Procedure 5.1, Eqs. 5.1-5.5.**

Master conditional-mean specification (Eq. 5.1):

```
E(y_it | dq_i, ..., dT_i, x_i) = α + Σ_{g=q}^T β_g dg_i + x_i κ
                                  + Σ_{g=q}^T (dg_i·x_i) ξ_g
                                  + Σ_{s=2}^T γ_s fs_t + Σ_{s=2}^T (fs_t·x_i) π_s
                                  + Σ_{g=q}^T Σ_{s=g}^T τ_gs (w_it · dg_i · fs_t)
                                  + Σ_{g=q}^T Σ_{s=g}^T (w_it · dg_i · fs_t · ẍ_ig) δ_gs.    (5.1)
```

Here `ẍ_ig ≡ x_i − E(x_i | dg = 1)`; sample analog (Eq. 5.2):

```
ẍ_ig = x_i − x̄_g,   g = q, ..., T.                                       (5.2)
```

Procedure 5.1 runs the pooled OLS regression (Eq. 5.3):

```
y_it on  w_it·dq_i·fq_t, ..., w_it·dq_i·fT_t,
         w_it·d(q+1)_i·f(q+1)_t, ..., w_it·d(q+1)_i·fT_t,
         ...,
         w_it·dT_i·fT_t,
         w_it·dq_i·fq_t·ẍ_iq, ..., w_it·dq_i·fT_t·ẍ_iq,
         w_it·d(q+1)_i·f(q+1)_t·ẍ_{i,q+1}, ..., w_it·d(q+1)_i·fT_t·ẍ_{i,q+1},
         ...,
         w_it·dT_i·fT_t·ẍ_iT,
         1, f2_t, ..., fT_t, f2_t·x_i, ..., fT_t·x_i,
         dq_i, ..., dT_i, dq_i·x_i, ..., dT_i·x_i.                       (5.3)
```

`τ̂_gs` is the OLS coefficient on `w_it · dg_i · fs_t`. Shorthand `dgfs_it ≡ dg_i · fs_t` (Eq. 5.5).

**(E) Extended TWFE estimator (ETWFE) — Eqs. 5.6-5.7.**

Unobserved-effects model (Eq. 5.6):

```
y_it = Σ_{g=q}^T Σ_{s=g}^T τ_gs (w_it·dg_i·fs_t)
        + Σ_{g=q}^T Σ_{s=g}^T (w_it·dg_i·fs_t·ẍ_ig) δ_gs                 (5.6)
        + Σ_{s=2}^T γ_s fs_t + Σ_{s=2}^T (fs_t·x_i) π_s + c_i + u_it,
        t = 1, ..., T.
```

Fixed-effects estimation of (5.6) — regression (Eq. 5.7):

```
y_it on  w_it·dq_i·fq_t, ..., w_it·dq_i·fT_t, ..., w_it·dT_i·fT_t,
         w_it·dq_i·fq_t·ẍ_iq, ..., w_it·dT_i·fT_t·ẍ_iT,
         f2_t·x_i, ..., fT_t·x_i,
         f2_t, ..., fT_t, c1_i, c2_i, ..., cN_i.                         (5.7)
```

By Theorem 3.1, (5.7) and (5.3) yield identical estimates for the `τ_gs`, the `δ_gs` heterogeneous-effect interactions, and the heterogeneous-trend `(fs_t · x_i)` coefficients.

**Restrictive "lags-only" TWFE — Eq. 5.8 (the version critiqued in the literature).**

```
y_it = τ · w_it + Σ_{s=2}^T γ_s fs_t + c_i + u_it,    t = 1, ..., T.   (5.8)
```

This imposes a constant treatment effect. The paper's point: (5.8) is the problem, not TWFE per se. The De Chaisemartin–d'Haultfœuille (2020), Callaway–Sant'Anna (2021), Goodman-Bacon (2021), Sun-Abraham (2021) negative-weighting results stem from estimating (5.8). Note `w_it = w_it · (Σ_{g=q}^T Σ_{s=g}^T dg_i · fs_t)`, showing that (5.8) collapses `(T-q+1)(T-q+2)/2` cohort-time treatment indicators in (5.9) into a single `w_it`.

**Intermediate flexible TWFE — Eq. 5.9.**

```
y_it = Σ_{g=q}^T Σ_{s=g}^T τ_gs (w_it·dg_i·fs_t) + Σ_{s=2}^T γ_s fs_t + c_i + u_it.   (5.9)
```

Under random sampling, NA, and unconditional PT, TWFE estimation of (5.9) is unbiased and consistent for the `τ_gt`.

**Common-timing simplification — Eq. 5.10.** With a single treated cohort:

```
y_it = Σ_{s=q}^T τ_s (w_it·d_i·fs_t) + Σ_{s=q}^T (w_it·d_i·fs_t)·ẍ_i1·δ_s
        + Σ_{s=2}^T γ_s fs_t + Σ_{s=2}^T (fs_t·x_i)·π_s + c_i + u_it,
        t = 1, ..., T,                                                  (5.10)
```

with `ẍ_i1 = x_i − x̄_1`, `x̄_1 = N_1⁻¹ Σ_i d_i · x_i`.

### Key Theorems / Propositions

**Theorem 3.1 (Two-way Mundlak ≡ TWFE) — p. 2549.**

> Let `β̂_TWM` be the `K × 1` vector of coefficients on `x_it` from the two-way Mundlak regression in (3.2), and let `β̂_TWFE` be the TWFE estimator from the regression in (2.2) [equivalently, (2.6)], assuming that `Σ_{i=1}^N Σ_{t=1}^T ẍ_it' ẍ_it` is nonsingular (3.3). Then, `β̂_TWM = β̂_TWFE`. Moreover, the coefficients on `x_it` do not change when any subset of `(z_i, m_t)` is dropped from (3.2). ∎

Proof (Appendix A, pp. 2581-2582) uses Frisch-Waugh partialling-out repeatedly: the residuals from regressing `x_it` on `(1, x̄_i., x̄_.t)` are `ẍ_it` (Eq. A4). The result extends Wooldridge (2019, Proposition 2.1) to the two-way FE setting; concurrent and independent work by Yang (2022) reached the same conclusion. Extension with `(z_i, m_t)`: by repeated F-W partialling, the matrix coefficients on `(z_i − z̄)` and `(m_t − m̄)` are identically zero, so the centering conditions `T⁻¹ Σ m_t = m̄` and `N⁻¹ Σ z_i = z̄` matter.

**Corollary 3.2 (RE GLS ≡ POLS in the TWM regression) — p. 2550.**

> The pooled OLS estimator `β̂_TWM` from (3.2) is identical to the one-way random effects (RE) GLS estimator (with a cross-sectional "random effect") using the same regressors as in (3.2). Moreover, the remaining coefficients from the POLS and RE estimators are identical. ∎

Implication: POLS and RE give numerically identical estimates of *all* coefficients in (3.2), not just those on `x_it`. POLS is best linear unbiased under standard random-effects assumptions and asymptotically efficient under fixed `T`, `N → ∞`.

**Proposition 5.2 (Equivalence between cohort imputation and POLS on cohort dummies) — p. 2558.**

> Assuming no perfect collinearity among the regressors in (4.14),
>
> ```
> τ̂_gt = τ̃_gt,   g = q, ..., T;  t = g, ..., T.                       (5.4)
> ```
>
> Moreover, the coefficients on the control variables `1, f2_t, ..., fT_t, f2_t·x_i, ..., fT_t·x_i, dq_i, ..., dT_i, dq_i·x_i, ..., dT_i·x_i` are identical across the two procedures. ∎

**Proposition A.1 (general POLS↔imputation equivalence, Appendix A, pp. 2582-2585).**

> Consider a panel `{(y_it, g_it, h_it, w_it) : t = 1, ..., T; i = 1, ..., N}` with `g_it` (`1 × K`), `h_it` (`1 × L`), `w_it ∈ {0, 1}`, and `w_it h_it = h_it` (so `(1 − w_it) h_it = 0`). Let `(β̂, γ̂)` be POLS from `y_it on g_it, h_it` (Eq. A7); let `β̃` be POLS from `y_it on g_it` using the `w_it = 0` sample (Eq. A8); let `ṽ_it = y_it − g_it β̃` (Eq. A9); let `γ̃` be from `ṽ_it on h_it` using `w_it = 1` (Eq. A10). If the OLS rank condition holds and, for all `(i, t)`, `w_it g_it = h_it A` (Eq. A11) for an `L × K` matrix `A`, then `β̃ = β̂` and `γ̃ = γ̂` (Eq. A12). ∎

Application to Section 5: Setting `g_it = (1, dq_i, ..., dT_i, f2_t, ..., fT_t, x_it, dg_i·x_it, fs_t·x_it)` (Eq. A17) and `h_it = (dg_i·fs_t, dg_i·fs_t·ẋ_itgs)` (Eq. A18) verifies `w_it h_it = h_it` because `w_it = Σ Σ dg_i · fs_t` is a sum of mutually exclusive treatment dummies. The mutually exclusive structure plus cohort-centering orthogonality (Σ Σ (dg_i · fs_t) · (dg_i · fs_t · ẍ_ig) = 0) makes the second-step regression separate cleanly into the cohort-time ATT coefficients.

**Extensions to leads-and-lags and heterogeneous trends (Appendix A, Eq. A21 + final paragraph p. 2585):**

- For leads-and-lags: add `dg_i · fs_t, dg_i · fs_t · ẋ_itgs` for `s = 1, ..., g-2` to `g_it`; multiplication by `w_it` zeros these out, so Eq. A11 holds trivially.
- For cohort-specific linear trends `dg_i · t` and `dg_i · t · x_it`: added to `g_it`. Equivalence holds because `w_it · dg_i · t` is a linear combination of `dg_i · fg_t, ..., dg_i · fT_t` with coefficients `g, g+1, ..., T` — all already in `h_it`.

### Composite Equivalence Chain — Eq. 5.16 (VERBATIM)

```
Cohort Imputation = POLS = TWFE = RE = BJS Imputation    (5.16)
```

Five algebraically identical estimators for the `τ_gt`:

1. **Cohort imputation** (Procedure 4.1, Eqs. 4.14-4.18).
2. **POLS on cohort dummies** with centered interactions (Procedure 5.1, Eq. 5.3).
3. **TWFE** applied to the flexible model (5.6), via regression (5.7).
4. **Random effects (RE)** using cohort dummies (via Corollary 3.2 applied to (5.6) — see decomposition Eqs. 5.11-5.12).
5. **BJS imputation** (Borusyak, Jaravel & Spiess 2024): cohort dummies replaced by unit dummies in (4.14). Out-of-sample residuals `t̃e_it` will generally differ from those in Procedure 4.1, but the cohort-time ATT averages coincide.

Wooldridge's framing (Section 5.5 / Section 11, p. 2580): "the imputation methods use only the control observations and TWFE sweeps away all time-constant variables. For general panel data applications, POLS, RE, and TWFE are all different. That they are the same in the staggered DiD setting when applied to an equation derived under standard no anticipation, parallel trends, and linear conditional expectations assumptions is compelling. These equivalences also hold in the absence of a never treated group."

### Event Study (Section 6)

**Eq. 6.1 — Sun-Abraham-style leads-and-lags (no controls).** Pre-treatment dummies `dg_i · fs_t` for `s < g`, excluding `dg_i · f(g-1)_t` so that `s = g − 1` is the reference period.

**Eq. 6.2 — lags-only (no leads).** Drops the `s < g` interactions.

**Eq. 6.4 — full regression with covariates and ALL `dg · fs · ẍ_g` interactions** — natural extension of (5.3) that includes treatment "leads" and lead-control interactions. Fully saturated; equivalent to a collection of 2×2 DiDs using `g − 1` as reference.

**Eq. 6.5 — equivalent cross-sectional long-differences (CS 2021) representation:**

```
y_is − y_{i,g-1}  on  1, dg_i, ẍ_{ig}, dg_i · ẍ_{ig}
                      using  dg_i = 1 or d∞_i = 1.                    (6.5)
```

For `s ≥ g`, the coefficient on `dg_i` is `τ̃_gs`; for `s ≤ g − 2`, (6.4) recovers the pre-trend `θ̆_gs`. The regressions in (6.5) are identical to the CS (2021) linear regression-adjustment estimators when never-treated units are used as controls.

> Harmon (2024) shows that the CS (2021) regression estimator is efficient if the idiosyncratic errors `u_it` follow a random walk and are conditionally homoskedastic.

**Eq. 6.6 — simplest pre-trend t-test** (`T=3, q=3`, no controls):

```
y_it on 1, d_i, f2_t, f3_t, d_i · f1_t, d_i · f3_t,  t = 1, 2, 3;  i = 1, ..., N.   (6.6)
```

Test statistic is the cluster-robust `t` on `d_i · f1_t`. The same `t`-statistic is obtained regardless of which pre-period is used as reference (Section 6.1).

**Eq. 6.7 — relaxed-PT-pre-intervention model.** Adds `Σ Σ θ_gs (dg · fs_t)` and `Σ Σ (dg · fs_t · ẍ_g) λ_gs` for `s = 1, ..., g-2`. Important caveat (p. ~2569): "If the violation of PT carries into the treated periods, including the pre-treatment indicators can actually exacerbate the bias compared with not including them."

### Aggregation (Section 7) — KEY FINDING / LOAD-BEARING DEVIATION

This is the central comparison point with the shipped `WooldridgeDiD` implementation. The paper defines population aggregation weights based on **cohort shares**, not cell-level observation counts.

**Eq. 7.1 — straight (unweighted) average:**

```
τ̄ ≡ [1 / ((T-q+1)(T-q+2)/2)] · Σ_{g=q}^T Σ_{t=g}^T τ_gt   (7.1)
```

**Eq. 7.2 — cohort-share weighted population parameter (VERBATIM):**

```
τ̄_ω ≡ Σ_{g=q}^T Σ_{t=g}^T ω_g · τ_gt    (7.2)
```

> "Typically, one might prefer a weighted average of the `τ_gt`, where the weights are the cohort shares in the population."

**Eq. 7.3 — cohort-share weighted estimate (VERBATIM):**

```
τ̂̄_ω = Σ_{g=q}^T Σ_{t=g}^T ω̂_g · τ̂_gt    (7.3)
```

**Eq. 7.4 — cohort-share weights, exact formula (VERBATIM):**

> "where the weights are the same within a treated cohort:"

```
ω̂_g ≡ N_g / [(T − q + 1) N_q + ··· + 2 N_{T-1} + N_T],
       t = g, ..., T;  g = q, ..., T.                                   (7.4)
```

**Key properties of Eq. 7.4:**

- Numerator `N_g` = count of units in cohort `g` (cohort size).
- Denominator is a decreasing-coefficient sum `(T-q+1) N_q + (T-q) N_{q+1} + ··· + 2 N_{T-1} + N_T`, where earlier cohorts (treated longer) receive higher multiplicative weight reflecting their `(T − g + 1)` post-treatment periods.
- Weights `ω̂_g` do NOT depend on time `t` within a cohort — they are **cohort-level constants** applied to every `τ̂_gt` for that cohort.
- Equivalently, the denominator is the total number of post-treatment observations across all cohorts (`Σ_g N_g · (T-g+1)`).

> "Stata provides a simple way to obtain (7.3) and obtain a valid standard error that accounts for the sampling error not only in the `x̄_g` but also in the cohort shares, `ω̂_g`. After running the regression (5.3), compute the partial effect with respect to `w_it`, and then average over the treated observations `w_it = 1`. This average partial effect (APE) over the treated units automatically weights the `τ̂_gt` by cohort shares."

**Eq. 7.5 — weighted average of exposure-time effects (VERBATIM):**

```
τ̂_{ω,e} = Σ_{g=q}^{T-e} ω̂_{ge} · τ̂_{g, g+e}    (7.5)
```

**Eq. 7.6 — exposure-time weights (VERBATIM):**

```
ω̂_{ge} = N_g / (N_q + ··· + N_{T-e})    (7.6)
```

> "these are positive and sum to one for each exposure time."

Worked exposure-time weights:

- `τ̂_{ω,0} = ω̂_{q0} τ̂_qq + ω̂_{q+1,0} τ̂_{q+1,q+1} + ··· + ω̂_T0 τ̂_TT`, with `ω̂_{g0} = N_g / (N_q + ··· + N_T)`.
- `τ̂_{ω,1} = ω̂_{q1} τ̂_{q,q+1} + ω̂_{q+1,1} τ̂_{q+1,q+2} + ··· + ω̂_{T-1,0} τ̂_{T-1,T}`, with `ω̂_{g1} = N_g / (N_q + ··· + N_{T-1})`.

**Eq. 7.7 — leads-and-lags exposure-time trick.** Define `nw_it ≡ 1 − w_it`, run regression (6.4) recast with `nw_it` interacted on leads and `w_it` on lags. Compute APE w.r.t. `nw_it` (pre-treatment effects, weighted by time-until-exposure) and w.r.t. `w_it` (post-treatment lags). In the common-timing case there is no weighting because there is only one treated cohort.

### Heterogeneous Cohort Trends (Section 8)

**Eq. 8.1 — model with linear-in-`t` heterogeneous cohort trends:**

```
E[y_t(∞) | d, x] = α + Σ_{g=q}^T β_g dg + xκ + Σ_{g=q}^T (dg·x) ξ_g
                    + Σ_{s=2}^T γ_s fs_t + Σ_{s=2}^T (fs_t · x) π_s
                    + Σ_{g=q}^T η_g (dg · t),    t = 1, ..., T.    (8.1)
```

> "The new terms in (8.1), `η_g · (dg · t)`, allow for trends to differ from the baseline in a linear fashion, with a different slope for each treated cohort `g`. Unlike the final line in (6.7), Eq. (8.1) allows violation of CPT into the treatment periods."

Practical implementation: "In the (long) regression (5.3), simply add the interactions `dg_i · t`, `g = q, ..., T`. It is shown in the appendix that all of the coefficients and the ATT estimates, and the moderating effects, are the same as the imputation approach that only uses the `w_it = 0` observations in the first stage."

Requirements:

- Need at least two pre-treatment periods per cohort to identify separate linear trends.
- Higher-order polynomials (`t²`, `t³`) can be added if there are many pre-treatment periods, at a cost to precision.
- Collinearity with the treatment indicators `w_it · dg_i · fs_t = dg_i · fs_t` is induced — handled by the saturated-interaction structure.

**Eqs. 8.2-8.3 — `T=3, q=3` worked DDD example:**

```
y_it on 1, d_i, f2_t, f3_t, d_i · t, d_i · f3_t,  t = 1, 2, 3;  i = 1, ..., N.   (8.2)

τ̂_{3,ddd} = N_1⁻¹ Σ_i d_i · Δ² y_i3  −  N_0⁻¹ Σ_i (1 − d_i) · Δ² y_i3
          = [(ȳ_13 − ȳ_12) − (ȳ_03 − ȳ_02)]
            − [(ȳ_12 − ȳ_11) − (ȳ_02 − ȳ_01)]                          (8.3)
```

> "Note that `τ̂_{3,ddd}` is a difference-in-difference-in-differences estimator, where the double differencing in this case is across time. The first term in (8.3) is the usual DiD estimator using periods two and three. The second term is a DiD estimator using periods two and one, where measures the difference in trends before the intervention. Often `(ȳ_12 − ȳ_11) − (ȳ_02 − ȳ_01)` is used as a placebo check, but in (8.3) it is used to adjust the 2×2 DiD estimator for the presence of pre-trends."

The cluster-robust `t` statistics on `d_i · t`, `d_i · f1_t`, and `d_i · f2_t` are all the same in absolute value (so the test of parallel trends is unchanged), but the estimates of `τ_3` can be very different.

### All Units Eventually Treated (Section 5.4, Eqs. 5.13-5.15)

When there is no never-treated group, the last cohort `T` plays its role:

```
te(g:T) = y_t(g) − y_t(T),    g = q, ..., T-1;  t = g, ..., T.     (5.13)
τ_{(g:T),t} ≡ E[y_t(g) − y_t(T) | dg = 1],
                                   g = q,...,T-1;  t = g,...,T.    (5.14)
```

Under NA: `E[y_t(T) | dg = 1] = E[y_t(∞) | dg = 1]` for `t < T`, so:

```
τ_{(g:T),t} = τ_gt,  g = q, ..., T-1;  t = g, ..., T-1.            (5.15)
```

In the final period `T`, the ATT for cohort `T` is unidentified. Mechanically, all `dT_i`-containing regressors in (5.3) get dropped, effectively forcing cohort `T` to serve as the never-treated reference.

### Time-Varying Covariates (Section 10.1)

Extended NBC assumption (Eq. 10.1):

```
x_t(g) = x_t(∞),   g = q, ..., T;   t = 1, ..., T.        (10.1)
```

Cohort-period demeaning (Eq. 10.2-10.3):

```
ẋ_itgs ≡ x_it − x̄_gs                                       (10.2)

x̄_gs ≡ N_gs⁻¹ Σ_i Σ_t dg_i · fs_t · x_it,
N_gs ≡ Σ_i Σ_t dg_i · fs_t.                                  (10.3)
```

Implementation note (p. 2578): "in (5.3), the interactions for the moderating terms become `w_it · dg_i · fs_t · ẋ_itgs`, and `dg_i · fs_t = 0` unless `s = t`. In practice, it is easiest to define `ẋ_itgs` even when `s ≠ t` and then interact these with the appropriate treatment dummy `dg_i · fs_t`."

**Key cautions:**

- "With time-varying covariates, POLS on cohort dummies and POLS with unit dummies (TWFE) are no longer the same."
- "Technically, with fixed-`T` asymptotics, consistency of the TWFE estimator requires that the sequence `{x_it : t = 1, ..., T}` is strictly exogenous with respect to the implicit idiosyncratic shocks, `u_it`."
- "Cases where `x_it` is contemporaneously exogenous, does not react to treatment assignment, but is not strictly exogenous would be fairly rare."
- Pre-treatment leads use `(g, s)` averages: "When interacted with the pre-treatment indicators `dg_i · fs_t` for `s < g − 1`, the covariates are centered around the `(g, s)` averages."

CS (2021) comparison: "Incidentally, with current event study methods, particularly Callaway and Sant'Anna (2021), it is not clear how time-varying controls dated in the same period as the outcome should be handled. CS (2021) recommend using only the covariates dated in the earlier time period, although this makes less sense in estimating the pre-treatment effects."

### Unbalanced Panels (Section 10.2)

"When a panel dataset is unbalanced, the equivalences derived earlier no longer hold. The POLS regression are still consistent if the missingness is related to `(dq_i, ..., dT_i, x_i)` (or replace `x_i` with `x̄_i`). TWFE has an advantage with missing data because the missingness can be related to the unobserved effect, `c_i`, implicitly appearing in the equation."

Underlying equation (common-timing case, Eq. 10.4):

```
y_it = τ_q (w_it · fq_t) + ··· + τ_T (w_it · fT_t)
        + θ_2 f2_t + ··· + θ_T fT_t + c_i + u_it             (10.4)
```

With complete-cases indicator `s_it`, define `fr̄_i ≡ T_i⁻¹ Σ_t s_it · fr_t` where `T_i = Σ s_it`. The POLS regression (Eqs. 10.5-10.6) regresses `y_it` on:

```
1, (d_i · fq_t), ..., (d_i · fT_t), f2_t, ..., fT_t,
d_i · fq̄_i, ..., d_i · fT̄_i, f2̄_i, ..., fT̄_i                 (10.6)
```

> "In effect, including the `fs̄_i` along with `d_i · fs̄_i` accounts for selection issues related to additive unobserved heterogeneity. Of course, it is easier to use TWFE on the unbalanced panel once the interaction terms `w_it · fs_t = w_it · d_i · fs_t` have been created."

**Operational notes:**

- "Any unit with `T_i = 0` has no usable data, and `T_i = 1` units do not contribute to the estimation."
- "With time-varying controls, one would demean the covariates using only the complete cases by `(g, s)` pair."

### Standard Errors / Inference

- **Cluster-robust SEs at the unit level** are the default for Procedures 4.1 and 5.1. "One might cluster at a level higher than `i` if the data were obtained via cluster sampling or, more likely, the intervention is assigned at a higher level of aggregation; see Abadie, Athey, Imbens and Wooldridge (2023)."
- **Aggregate inference via APE.** "popular statistical packages, such as Stata, provide a simple way to obtain (7.3) and obtain a valid standard error that accounts for the sampling error not only in the `x̄_g` but also in the cohort shares, `ω̂_g`. … Often there is an option that applies formulas from generalized method of moments (GMM) estimation to account for all estimation uncertainty."
- **Walmart application SEs (Section 9):** "The standard errors are clustered at the county level, and the sampling variation in the controls and the weights used to obtain the exposure time effects have been accounted for using the margins command in Stata 18."
- **Imputation-step inference (Section 5.5):** "Inference is complicated, and BJS (2024) only provide conservative standard errors. Inference is made easier by implementing the estimator as pooled OLS, random effects (equivalently TWFE)."
- **Bootstrap usage (Section 9):** For the heterogeneous-cohort-trends event study plot (Figure 2): "The standard errors used to obtain the 95% confidence are obtained using 1,000 bootstrap replications (rather than working through the more complicated analytical standard errors)." No specifics on bootstrap type (pairs / wild / cluster) or weight distribution are provided.

### Aggregations Comparison Table (default + opt-in surfaces)

| Aggregation | Paper (Eqs. 7.2-7.4 / 7.6) | Shipped `WooldridgeDiD` |
|-------------|----------------------|-------------------------|
| `simple` / overall | Cohort-share `ω̂_g = N_g / Σ_g (T-g+1) N_g`, constant within cohort, decreasing-coefficient denominator | Default `weights="cell"`: cell-level `n_{g,t}` observation counts (matches Stata `jwdid_estat`). Opt-in `weights="cohort_share"`: paper Eq. 7.4 form (shipped in PR-B). |
| `event` / exposure-time | Eq. 7.6: `ω̂_{ge} = N_g / (N_q + ··· + N_{T-e})` | Default `weights="cell"`: cell counts (placebo leads included). Opt-in `weights="cohort_share"`: paper Eq. 7.6 form, restricted to `k >= 0` (shipped in PR-B). |
| `group` | Not explicitly given in paper as a closed-form weight | Weighted average across `t` for each cohort `g` (cell-count weights; `weights="cohort_share"` raises by design — no paper closed-form). |
| `calendar` | Not explicitly given in paper as a closed-form weight | Weighted average across `g` for each calendar time `t` (cell-count weights; `weights="cohort_share"` raises by design — no paper closed-form). |

### Covariates

*Paper covariate types:*

- **Time-constant `x_i`** (Sections 4-5): added as `x_i`, `dg_i · x_i`, `fs_t · x_i`, `w_it · dg_i · fs_t · ẍ_ig` per Eq. 5.3.
- **Time-varying `x_it`** (Section 10.1): replaces `x_i` with `x_it`; demeaned at the cohort-period level (`ẋ_itgs = x_it − x̄_gs`, Eq. 10.2).

*Shipped `WooldridgeDiD` covariate categories:*

- `exovar`: time-invariant, added without demeaning (paper Eq. 5.2 `x_i`).
- `xtvar`: time-varying, demeaned within cohort×period cells when `demean_covariates=True` (paper Eq. 10.2).
- `xgvar`: covariates interacted with each cohort indicator.

### Control Groups

- **Never-treated present (default in Sections 4-5):** `T − q + 2` total treatment levels.
- **All units eventually treated (Section 5.4):** Last cohort `T` plays the never-treated role; ATT for cohort `T` is unidentified.
- Shipped library options: `not_yet_treated` (default) and `never_treated`.

### Centering Conventions (Section 5, pp. 2559-2560)

- **Cohort centering `ẍ_ig = x_i − x̄_g`:** required on `w_it · dg_i · fs_t · ẍ_ig` interactions so that the `w_it · dg_i · fs_t` coefficients estimate `τ_gt` at the cohort mean. Without centering they would estimate ATT at `x = 0`.
- **Optional centering on `dg_i · x_i`:** "Often, one might want to use the same centering when interacting with just the cohort dummies, `dg_i`. Replacing `dg_i · x_i` with `dg_i · ẍ_ig` will make the coefficients on `dg_i` more meaningful. In effect, the coefficients on the `dg_i` will be estimates of an average 'selection effect.'" Does not affect `τ_gt`.
- **Alternative centering for `fs_t · x_i`:** "to obtain an estimate of the average trend in the never treated state, one can use `fs_t · (x_i − x̄)` in place of `fs_t · x_i`, where `x̄` is the overall sample average … Alternatively, one might use `x̄_∞`, the average of the never treated units."

### Edge Cases

- **Single cohort (no staggered adoption):** Reduces to standard 2×2 DiD via the common-timing simplification (Eq. 5.10).
- **Missing cohorts:** Only cohorts observed in the data are included in interactions.
- **Anticipation:** When the paper's NA is violated, the relaxed-PT model (Eq. 6.7) can absorb pre-treatment effects; the shipped library exposes this via `anticipation > 0`.
- **Never-treated control only:** Pre-treatment placebo ATTs remain estimable; ATT for the last cohort under all-treated remains unidentified.
- **Unbalanced panel — `T_i = 0` or `T_i = 1`:** No usable data; do not contribute to estimation (p. 2579).
- **Heterogeneous-trends residual deviations near sample edges (Walmart, Section 9):** "Going back 20 periods before the intervention, the deviations from zero are small, although some are statistically different from zero. At lags 21 and 22, the estimated pre-treatment effects are larger in magnitude." Robustness check: drop earliest few years.
- **Rank condition for general POLS↔imputation (Proposition A.1, Eq. A11):** `w_it g_it = h_it A` for an `L × K` matrix `A`. If the rank condition fails, the equivalence is not guaranteed.

### Practical Recommendations

- **Bootstrap for SEs:** 1,000 replications used in Walmart Section 9 for the cohort-specific-trends event study (Figure 2).
- **Pre-trend diagnostics with cohort trends:** Use the imputation characterization to plot residuals across exposure time (pre-treatment from first-step imputation regression; post-treatment from second imputation step / long regression).
- **Robustness for cohort-trend specifications:** Drop the earliest few years of the dataset to test sensitivity at long lags.
- **Time-varying controls dating:** Wooldridge prefers contemporaneously-dated controls (Eq. 10.1); CS (2021) recommend pre-treatment dating, which "makes less sense in estimating the pre-treatment effects."
- **POLS vs TWFE on unbalanced panels:** "easier to use TWFE on the unbalanced panel once the interaction terms `w_it · fs_t = w_it · d_i · fs_t` have been created."
- **Heterogeneous trends — when to use:** "heterogeneous trends can be used within the general framework when PT is clearly violated. The estimated effects are much smaller than either the lags only or leads and lags estimates."
- **Bad-controls warning:** "covariates whose values change due to the intervention" should be excluded; "if the controls are included to partially account for selection into treatment, then they should be dated before the treatment period."

### Requirements Checklist

- [x] Saturated cohort×time interaction design matrix (paper Eq. 5.3 / 5.7)
- [x] Unit + time FE absorption (within-transformation or as dummies)
- [x] OLS path with Theorem 3.1 equivalence (TWM ≡ TWFE)
- [x] Cohort imputation equivalence (Procedure 4.1, Proposition 5.2)
- [x] BJS imputation equivalence (Section 5.5)
- [x] Cluster-robust SEs at unit level (Section 7.5)
- [x] Cohort centering `ẍ_ig = x_i − x̄_g` for treatment interactions
- [x] Three control covariate categories (exovar / xtvar / xgvar)
- [x] Time-varying covariates with cohort×period demeaning (Eq. 10.2)
- [x] Unbalanced-panel handling via TWFE (Eq. 10.4-10.6)
- [x] Anticipation support
- [x] **Heterogeneous cohort trends via `dg_i · t` (Section 8 / Eq. 8.1).** **Status:** Closed in PR-B via `WooldridgeDiD(cohort_trends=True)` (OLS path only; auto-routes to full-dummy mode regardless of `vcov_type`). The estimator surface now exposes the paper Eq. 8.1 contract; rejects on `method ∈ {"logit","poisson"}` per paper Section 8 OLS-only scope.
- [x] **Cohort-share aggregation weights — both simple-overall (Eqs. 7.2-7.4) and event-time (Eq. 7.6) paths.** **Status:** Closed in PR-B via opt-in `aggregate(weights="cohort_share")` on `WooldridgeDiDResults` (paper Eq. 7.4 for `type="simple"` and Eq. 7.6 for `type="event"`; raises on `type ∈ {"group","calendar"}` per the no-paper-formula contract). Default stays `weights="cell"` for `jwdid_estat` back-compat; the two coincide on balanced panels with uniform within-cohort cell counts (paper Section 7.5 footnote).

**Library-extension items (not paper requirements):**

- [x] *Library extension:* Multiplier bootstrap (Rademacher / Webb / Mammen) for overall OLS SE. The 2025 paper uses 1,000 bootstrap replications for the Walmart Figure 2 cohort-trends CIs but **does not specify** the bootstrap variant (pairs vs wild vs cluster), weight distribution, or resampling unit — so the library's multiplier-bootstrap implementation is an implementation choice consistent with Wooldridge's general framing rather than a 2025-paper-mandated requirement.

### Deviations

These deviations are recognized via a labeled note or deviation entry in `docs/methodology/REGISTRY.md` per the project's documented-deviation convention — accepted label forms include plain `- **Note:** <text>`, `- **Note (deviation from R/Stata):** <text>`, and `- **Deviation from R:** <text>`. The aggregation Note for this estimator at `REGISTRY.md` § Aggregations uses the plain `**Note:** <text>` form.

- **Aggregation weighting (simple-overall via paper Eqs. 7.2-7.4; event-time via paper Eq. 7.6):** **Status:** Closed in PR-B via the opt-in `aggregate(weights="cohort_share")` parameter on `WooldridgeDiDResults` (paper Eq. 7.4 for `type="simple"` and Eq. 7.6 for `type="event"`; the cohort-share event path is restricted to post-treatment `k >= 0` exposure times per the paper's Eq. 7.6 scope, with leads using the default `weights="cell"` placebo path). Default stays `weights="cell"` for `jwdid_estat` back-compat. The two schemes coincide on balanced panels with uniform within-cohort cell counts (paper Section 7.5 footnote). Remaining follow-ups: design-consistent cohort totals for survey-weighted fits (raises `ValueError` today) and unconditional cohort-share inference accounting for sampling uncertainty in `ω̂_g` / `ω̂_{ge}` (t-stat / p-value / conf-int fail-closed to NaN today); both tracked in `DEFERRED.md`.
- **Deviation from R `etwfe`:** R's `etwfe` package uses `fixest` for nonlinear paths; this implementation uses direct QMLE via `compute_robust_vcov` to avoid a statsmodels/fixest dependency. (Carried from existing REGISTRY.)
- **Note (deviation from Stata `jwdid` small-sample correction):** QMLE sandwich uses `weight_type="aweight"` (applies `(G/(G-1)) · ((n-1)/(n-k))` small-sample adjustment); Stata `jwdid` uses `G/(G-1)` only. The `(n-1)/(n-k)` term is conservative (inflates SEs slightly); for typical ETWFE panels where `n >> k`, the difference is negligible. (Carried from existing REGISTRY.)
- **Note (silent NaN-coercion now warned):** NaN values in the `cohort` column are filled with 0 (treated as never-treated) in `_filter_sample` and `fit()`; this recategorization emits a `UserWarning` reporting the affected row count (axis-E warning under the Phase 2 audit). Pass `0` directly for never-treated units to avoid the warning. (Carried from existing REGISTRY.)

### Walmart Application (Section 9) — Reference Empirical Findings

Setting: 1977-1999 panel of `N = 1,288` US counties; outcome `log(retail employment)`; first Walmart in 1986; 893/1,288 counties had a Walmart by 1999. Fourteen treated cohorts. Three time-constant controls dated 1980 (manufacturing-employment share, above-poverty share, high-school-degree share). County-level cluster-robust SEs; 1,000 bootstrap replications for Figure 2.

**Table 1 — selected values:**

| Horizon | (1) Lags Only | (2) Leads and Lags | (3) Heterogeneous Trends |
|---------|---------------|--------------------|--------------------------|
| τ_ω,0   | 0.0414 (0.0057) | 0.0232 (0.0028) | 0.0060 (0.0039) |
| τ_ω,1   | 0.0732 (0.0066) | 0.0543 (0.0041) | 0.0315 (0.0052) |
| τ_ω,7   | 0.1192 (0.0129) | 0.0954 (0.0117) | 0.0431 (0.0138) |
| τ_ω,13  | 0.2062 (0.0432) | 0.1910 (0.0345) | 0.0282 (0.0515) |
| τ_ω (overall) | 0.0935 (0.0098) | 0.0728 (0.0073) | 0.0260 (0.0096) |

Key findings (Section 9, pp. 2575-2577):

- Many pre-treatment estimates in column (2) leads-and-lags are large (>10% going back 15+ years), consistent with retail employment trending higher in counties that eventually got a Walmart.
- Heterogeneous trends (column 3): "the picture is now much different, with the estimated ATTs being much more modest. Except for the immediate effect, the effects in the first six years range between two and three percent. The effect peaks at just under 5% nine years out, but then effectively becomes zero. The average effect across all horizons is about 2.6% (`t ≈ 2.71`)."
- Precision: "the leads and lags estimator has uniformly smaller standard errors, probably because there is substantial positive serial correlation in the underlying idiosyncratic errors. Interestingly, for the short exposure times, the standard errors for the heterogeneous trends estimates are actually below those for the lags only, perhaps reflecting the fact that the cohort-specific trends explain nontrivial variation in log retail sales."

### Acronyms

- **TWFE** = Two-way Fixed Effects; **ETWFE** = Extended TWFE (TWFE applied to flexible model (5.6)).
- **TWM** = Two-way Mundlak; **POLS** = Pooled OLS; **RE** = Random Effects; **OWFE** = One-way Fixed Effects.
- **SUTVA** = Stable Unit Treatment Value Assumption; **NBC** = No Bad Controls; **NA** = No Anticipation; **CNA** = Conditional No Anticipation; **CPT** = Conditional Parallel Trends; **PT** = (unconditional) Parallel Trends; **LIN** = Linearity (functional form).
- **ATT** = Average Treatment Effect on the Treated; **DiD** = Difference-in-Differences; **APE** = Average Partial Effect; **GMM** = Generalized Method of Moments.
- **BJS 2024** = Borusyak, Jaravel & Spiess (2024); **CS 2021** = Callaway & Sant'Anna; **SA 2021** = Sun & Abraham; **DCDH** = De Chaisemartin & d'Haultfœuille.

---

## Implementation Notes

### Why TWFE Is Not the Problem — It's the Specification

The paper's central interpretive contribution (Sections 5, 11) is to distinguish the *estimator* (TWFE) from the *model* it's applied to:

- The DCDH 2020 / CS 2021 / Goodman-Bacon 2021 / SA 2021 negative-weighting results arise from estimating Eq. 5.8 (`y_it = τ · w_it + Σ γ_s fs_t + c_i + u_it`), which imposes a constant treatment effect across all cohorts and time.
- Applying TWFE to the flexible Eq. 5.6 (with saturated `w_it · dg_i · fs_t` interactions) — i.e., ETWFE — recovers the correct `τ_gt`. The five-way equivalence chain (Eq. 5.16) shows this recovers exactly the same point estimates as cohort imputation, CS 2021's regression-adjustment estimator (Eq. 6.5), BJS imputation, and RE.

Quote from Section 11 (p. 2580): "provided one allows treatment effects to be suitably heterogeneous, there is nothing inherently wrong with using TWFE — a conclusion reached by Sun and Abraham (2021) for the leads and lags estimator without controls."

### Why Cohort Dummies Suffice (vs. Unit Dummies)

The equivalence between (5.3) and (5.7) means that one need not include `N` unit dummies (`c1_i, ..., cN_i`) to control for unit heterogeneity if one includes cohort dummies (`dg_i`), controls `x_i`, and their interactions:

> "With `N = 1,000`, 5 cohorts, 10 controls, regression (5.3) has 60 time-constant controls vs. (5.7)'s 1,000+ — the ATT estimates and moderating effects (`fs_t · x_i`) are identical (p. 2561). This is essentially a consequence of the parallel trends assumption because the trend is identified through cohort-level (not unit-level) variation."

For the shipped library, this means OLS-path ETWFE can choose between explicit unit-FE absorption (within-transformation) and the lighter cohort-dummy formulation, with identical point estimates. The shipped path uses within-transformation by default; this matches Eq. 5.7 but the alternative (cohort-dummy POLS per Eq. 5.3) would give identical results.

### Relationship to Existing diff-diff Estimators

- **`CallawaySantAnna`**: Eq. 6.5 shows that the leads-and-lags variant of ETWFE (Eq. 6.4) recovers CS (2021)'s linear regression-adjustment estimator when never-treated units are controls. Harmon (2024) gives an efficiency condition (CS is efficient if `u_it` is a conditionally homoskedastic random walk).
- **`ImputationDiD`** (BJS 2024): Eq. 5.16 establishes algebraic identity. BJS uses unit dummies; ETWFE/cohort imputation uses cohort dummies; cohort-time ATT averages coincide.
- **`TwoStageDiD`** (Gardner 2022): Without covariates, Procedure 4.1 (cohort imputation) reduces to Gardner's two-stage DiD. With covariates, Procedure 4.1 interacts `x_i` with both `dg_i` and `fs_t`, allowing heterogeneous selection and trends.
- **`SunAbraham`**: Eq. 6.1 is exactly the SA (2021) leads-and-lags TWFE regression; Eq. 6.4 generalizes to include controls and cohort-control interactions.
- **`TripleDifference`**: Eq. 8.3 shows the heterogeneous-cohort-trends specification (with `dg_i · t` added) reduces to a DDD estimator in the `T=3, q=3` worked case.

### Equivalence Conditions

The five-way equivalence (Eq. 5.16) requires:

1. No perfect collinearity in the (5.3) regressor set.
2. Mutually exclusive cohort indicators (each unit in exactly one cohort or never-treated).
3. Treatment is absorbing (zeros followed by ones).
4. Either a never-treated group OR all units eventually treated (with cohort `T` serving as control).
5. Balanced panel for the strict equivalence (Section 10.2: equivalences break on unbalanced panels, though all approaches remain consistent if missingness satisfies appropriate conditions).
6. Time-constant covariates for the strict POLS-TWFE equivalence (Section 10.1: with time-varying covariates, POLS on cohort dummies and POLS with unit dummies are no longer the same).

The general statement (Proposition A.1) requires the rank condition `w_it g_it = h_it A` for some `L × K` matrix `A`.

### Computational Considerations

- POLS on Eq. 5.3 is a single OLS over all `N × T` observations — `O(N · T · K²)` where `K` is the number of parameters.
- `K` grows as `(number of cohorts) × (number of post-treatment periods)` for the interaction terms `w_it · dg_i · fs_t`, plus `(number of cohorts) × (T − 1) × dim(x)` for the heterogeneous-trend interactions.
- The within-transformation (Eq. 2.5 / 2.6) is preferred for high-dimensional `N` to avoid carrying `N` unit dummies in memory; for `N = 1,000` and `dim(x) = 10`, this saves ~1,000 columns.
- Bootstrap iterations (`n_bootstrap > 0`) are embarrassingly parallel.

### Tuning Parameters

The paper has minimal tuning:

| Parameter | Default | Selection Method |
|-----------|---------|-----------------|
| Reference period for event study | `g − 1` for each cohort | Section 6.1: "any set of pre-treatment periods can be used." |
| Order of polynomial trend (Section 8) | Linear `dg_i · t` | Higher-order `t²`, `t³` available with enough pre-treatment periods |
| Bootstrap replications | 1,000 (Walmart) | Convention for plot CIs |

No cross-validation or information-criterion tuning is involved.

### Reference Implementations

- **Stata `jwdid`** (Rios-Avila 2021): Implements Procedure 5.1 (POLS on the (5.3) regressor set). The `jwdid_estat` post-estimation command computes aggregations. The shipped `WooldridgeDiD` matches `jwdid_estat` cell-count weights as the **default** `aggregate(weights="cell")` surface; the opt-in `aggregate(weights="cohort_share")` exposes the paper W2025 Eq. 7.4 / 7.6 cohort-share forms for `type ∈ {"simple", "event"}` (shipped in PR-B).
- **R `etwfe`** (McDermott 2023): Implements ETWFE for OLS, logit, and Poisson via `fixest`. The shipped `WooldridgeDiD` uses direct QMLE via `compute_robust_vcov` to avoid a `fixest`/statsmodels dependency.

### Cross-link to 2023 Companion Review

For nonlinear extensions (logit, Poisson, fractional outcomes, ASF counterfactual computation, canonical-link LEF density theory, simulation evidence on linear-vs-nonlinear bias), see `docs/methodology/papers/wooldridge-2023-review.md`. The 2025 paper deliberately defers these to the 2023 paper (p. 2554) — "Pooled OLS using cohort indicators in a flexible way extends to nonlinear models, and I have developed those in Wooldridge (2023)." The two reviews together cover the full WooldridgeDiD methodology surface: 2025 is authoritative for the linear / Mundlak / equivalence / aggregation / heterogeneous-trends material, and 2023 is authoritative for nonlinear link functions and ASF.

---

## Gaps and Uncertainties

1. **Aggregation deviation framing.** **Status:** Resolved in PR-B by taking option (a) — `WooldridgeDiD` now exposes opt-in `aggregate(weights="cohort_share")` alongside the default `weights="cell"`. Paper Eqs. 7.4 (simple) and 7.6 (event, restricted to `k >= 0`) are implemented; default `weights="cell"` matches `jwdid_estat` back-compat. The `group`/`calendar` paths still use cell-count weights (no paper closed-form). Two remaining sub-followups: survey-weighted cohort totals (raises `ValueError` on `survey_design`) and unconditional inference accounting for sampling uncertainty in the cohort shares (t-stat / p-value / conf-int fail-closed to NaN today; tracked in `DEFERRED.md`).

2. **Proposition / theorem labeling cross-check.** Propositions 5.1 and 5.2 are introduced on pages 1-15 (Proposition 5.2 statement on p. 2558); Sections 5.5-5.6 (pages 16-35) discuss the equivalence chain referring back to these propositions. No contradiction.

3. **Bootstrap specifics.** The paper uses 1,000 bootstrap replications for the Walmart Figure 2 cohort-trends CIs but does not specify pairs vs. wild vs. cluster bootstrap, weight distribution (Rademacher / Mammen / Webb), or whether resampling is at the unit level. The shipped library supports multiplier bootstrap (Rademacher / Webb / Mammen) for the OLS overall SE — this is consistent with Wooldridge's general framing but not directly cited from the 2025 paper.

4. **Cluster level for cluster-robust SEs.** Section 7.5: "One might cluster at a level higher than `i` if the data were obtained via cluster sampling or, more likely, the intervention is assigned at a higher level of aggregation; see Abadie, Athey, Imbens and Wooldridge (2023)." The shipped library defaults to unit-level clustering; multi-way clustering for cohort-and-unit is not in the paper's recommendation.

5. **Repeated cross sections extension.** Section 11 (p. 2581) points to Deb et al. (2024) for the extension; not covered in the 2025 paper's main body. Not currently supported in `WooldridgeDiD`.

6. **Treatment exit / non-absorbing treatment.** The 2025 paper assumes absorbing treatment throughout. The 2023 paper Section 7.2 sketches an extension to treatment exit with a `D_{gh}` cohort definition, which is also not currently supported in `WooldridgeDiD`. No conflict between papers.

7. **Time-varying covariate strict exogeneity.** Section 10.1 requires strict exogeneity of `{x_it : t = 1, ..., T}` for the TWFE estimator under fixed-`T` asymptotics. The shipped library's `xtvar` accepts time-varying covariates without verifying strict exogeneity; this is a user-responsibility assumption. Consider adding a documentation note in REGISTRY about this requirement.

8. **Application reference in current REGISTRY.** The REGISTRY currently lists Nagengast-Rios-Avila-Yotov (2026) "European single market and intra-EU trade" as an application reference. This is an external application by one of the package authors (Rios-Avila) — not from the 2025 paper itself. The 2025 paper's empirical application is the Walmart-on-county-retail-employment study (Section 9). Both references are valid and serve different purposes (Walmart = original empirical illustration; Nagengast et al. = published empirical application of `jwdid`).

9. **Reference period sensitivity.** Section 6.1 notes that "any set of pre-treatment periods can be used" for the reference period, and the pre-trend `t`-test is numerically identical. This is a useful documentation point but is not currently captured in the shipped library docstrings.

10. **Heterogeneous-trends interpretation when pre-trends bleed into post-treatment.** The paper warns (Eq. 6.7 commentary): "If the violation of PT carries into the treated periods, including the pre-treatment indicators can actually exacerbate the bias compared with not including them." This is a methodological caveat about both leads-and-lags and the heterogeneous-trends Section 8 specification, not currently surfaced in REGISTRY.

11. **All-cohort-treated mechanics.** Section 5.4 explains that when all units are eventually treated and the last cohort `T` serves as control, "all variables in regression (5.3) (or its TWFE version) involving `dT_i` get dropped." That sentence governs four regressor families: the cells `dT·f_t`, the cell x covariate block `dT·f_t·ẋ`, the cohort trend `dT·t`, and the cohort x covariate block `D_T × X`. The shipped library applies it to the first three: comparison-support filtering removes the periods at which no unit is untreated, so cohort `T` receives no cells and no trend column, and `control_group="not_yet_treated"` on an all-eventually-treated panel ESTIMATES (verified against Stata `jwdid` on the `mpdta` panel with never-treated counties dropped: identical cell set, identical `N` = 764 of 955, ATTs agreeing to ~1e-15). The `D_{G_max} × X` covariate normalization is NOT applied — the remaining open work, which surfaces only when covariates are supplied.

12. **Cohort centering for `dg_i · x_i` interactions.** The paper notes (p. 2560) that centering `dg_i · x_i` around `x̄_g` makes the `dg_i` coefficient an "average selection effect" estimate. This is currently not exposed as a user-facing parameter in the shipped library; the existing implementation does NOT center the `dg_i · x_i` interactions by default (the cohort-centering applies only to the `w_it · dg_i · fs_t · ẍ_ig` treatment-interaction term per Eq. 5.2). This is an interpretive nuance, not a numerical issue for ATT estimates.

13. **No formal Monte Carlo simulation in the 2025 paper.** Unlike the 2023 paper (which has a substantial Section 5 simulation study), the 2025 paper's only empirical evidence is the Walmart application. The simulation-based comparisons against CS / DCDH / SA reside in the 2023 paper and the cited literature.


---

## Cross-References

- **Companion paper for nonlinear extensions:** `docs/methodology/papers/wooldridge-2023-review.md` (Wooldridge, 2023, *The Econometrics Journal*).
- **Related estimator reviews** (cross-cited in Sections 5, 6, 11 of the 2025 paper):
  - `docs/methodology/papers/goodman-bacon-2021-review.md` — TWFE decomposition that motivates the (5.8)-vs-(5.6) framing.
  - `docs/methodology/papers/dechaisemartin-dhaultfoeuille-2020-review.md` / `dechaisemartin-dhaultfoeuille-2022-review.md` / `dechaisemartin-2026-review.md` — negative-weighting and heterogeneity literature.
- **Registry entry:** `docs/methodology/REGISTRY.md` `## WooldridgeDiD (ETWFE)` (REGISTRY:1431-1547 at time of writing).
- **References file:** `docs/references.rst:25` already cites the 2025 *Empirical Economics* publication form.
- **Implementation:** `diff_diff/wooldridge.py` — main `WooldridgeDiD` estimator; covariate handling at lines 165-189 (`_build_interaction_matrix`) and 394-411 (covariate parameters); aggregation at the `WooldridgeDiDResults` level (`diff_diff/wooldridge_results.py`).
