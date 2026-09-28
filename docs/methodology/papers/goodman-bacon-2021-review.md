# Paper Review: Difference-in-differences with variation in treatment timing

**Authors:** Andrew Goodman-Bacon
**Citation:** Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing. *Journal of Econometrics*, 225(2), 254-277. DOI: [10.1016/j.jeconom.2021.03.014](https://doi.org/10.1016/j.jeconom.2021.03.014)
**PDF reviewed:** Journal of Econometrics version (DOI: [10.1016/j.jeconom.2021.03.014](https://doi.org/10.1016/j.jeconom.2021.03.014)) — 24 pages total: 14 main-text pages (Sections 1-5.1), 6 appendix/figure pages (Section 5.2 DD with controls, Section 6 Conclusion, Appendix A proof of Theorem 1), 4 reference pages. Local PDFs are gitignored under `/papers/`; the journal/DOI version is the authoritative source.
**Review date:** 2026-05-16

---

## Methodology Registry Entry

*Formatted to match `docs/methodology/REGISTRY.md` structure. Heading levels and labels align with existing entries.*

**Status: landed.** The `diff_diff/bacon.py` paper-vs-code audit has since completed: the `## BaconDecomposition` section in `docs/methodology/REGISTRY.md` is the authoritative methodology contract, with R `bacondecomp::bacon()` parity at `atol=1e-6` (`tests/test_methodology_bacon.py::TestBaconParityR`) and the always-treated-remap PR-B audit item checked, and `METHODOLOGY_REVIEW.md` records Bacon as **Complete** (2026-05-16). The block below is the paper-review record that informed that section; consult REGISTRY for the current authoritative text.

## BaconDecomposition

**Primary source:** [Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing. *Journal of Econometrics*, 225(2), 254-277.](https://doi.org/10.1016/j.jeconom.2021.03.014)

**Scope:** Decomposes the two-way fixed-effects DD (TWFEDD) estimator in Equation (2) when treatment timing varies across units. Theorem 1 expresses `β̂^DD` as a positively-weighted average of all possible 2x2 DD estimators in the data, with weights summing to 1. The decomposition is a **diagnostic tool**, not a treatment-effect estimator: it explains *which comparisons drive* the TWFEDD coefficient and *why* the estimator can fail to identify an interpretable causal parameter when treatment effects vary over time.

The decomposition identifies three types of 2x2 comparisons:

1. **Treated vs. untreated** (`β̂_{kU}^{2x2}`): timing group `k` compared to the always-untreated group `U` (or never-treated / always-treated control).
2. **Earlier-treated vs. later-treated, before ℓ** (`β̂_{kℓ}^{2x2,k}`): early group `k` as treatment, late group `ℓ` as control during `ℓ`'s pre-period.
3. **Later-treated vs. earlier-treated, after k** (`β̂_{kℓ}^{2x2,ℓ}`): late group `ℓ` as treatment, early group `k` as **already-treated** control after `k`'s treatment date.

The third type is the source of "negative weights" in TWFEDD (see Borusyak-Jaravel 2017; de Chaisemartin-D'Haultfœuille 2020; Sun-Abraham 2020): when treatment effects in the early group are changing during `ℓ`'s treatment window, the early group's outcome is no longer a clean counterfactual.

**Key implementation requirements:**

*Notation (Section 2, pp. 256-257):*
- `k ∈ {1, ..., K}`: timing groups ordered by first-treatment time. May include `U` (never-treated or always-treated, encoded as `t_i = ∞` for never-treated or `t_i < 1` for always-treated — i.e., treated before the first observable period, per paper footnote 11).
- `T`: total number of periods.
- `N`: number of cross-sectional units; `n_k = (count of units in group k) / N`: sample share.
- `n_{kℓ} = n_k / (n_k + n_ℓ)`: relative size of timing group `k` in pair `(k, ℓ)`.
- `D_{it}`: binary treatment indicator (1 if unit `i` treated by period `t`, else 0).
- `D̄_k = (1/T) Σ_t 1{t ≥ k}`: share of periods that group `k` spends treated.
- `ȳ_k^W = (1/(T_W · |k|)) Σ_{a ∈ W} Σ_{i ∈ k} y_{ia}`: average outcome for group `k` in period window `W` (one of `PRE(k)`, `MID(k,ℓ)`, `POST(k)`, `POST(ℓ)`).
- Period windows in the three-group case (Fig. 1):
  - `PRE(k) = [1, k-1]`
  - `MID(k, ℓ) = [k, ℓ-1]`
  - `POST(ℓ) = [ℓ, T]`

*TWFEDD specification (Equation 2):*
```
y_{it} = α_i + α_t + β^DD · D_{it} + e_{it}
```

*Theorem 1 — DD Decomposition (Equation 10a; full statement on p. 258):*
```
β̂^DD = Σ_{k ≠ U} s_{kU} · β̂_{kU}^{2x2}
      + Σ_{k ≠ U} Σ_{ℓ > k} [ s_{kℓ}^k · β̂_{kℓ}^{2x2,k} + s_{kℓ}^ℓ · β̂_{kℓ}^{2x2,ℓ} ]
```

The three 2x2 estimators are:
```
β̂_{kU}^{2x2}   = (ȳ_k^POST(k)   - ȳ_k^PRE(k))    - (ȳ_U^POST(k)   - ȳ_U^PRE(k))         (Eq. 10b)
β̂_{kℓ}^{2x2,k} = (ȳ_k^MID(k,ℓ) - ȳ_k^PRE(k))    - (ȳ_ℓ^MID(k,ℓ) - ȳ_ℓ^PRE(k))         (Eq. 10c)
β̂_{kℓ}^{2x2,ℓ} = (ȳ_ℓ^POST(ℓ)  - ȳ_ℓ^MID(k,ℓ)) - (ȳ_k^POST(ℓ)  - ȳ_k^MID(k,ℓ))        (Eq. 10d)
```

*Weight construction (Equations 7-9 for variance, 10e-g for the weights themselves):*

Fixed-effects-adjusted treatment-dummy variances (the "amount of identifying variation" in each subsample):
```
V̂_{kU}^D     = n_{kU}(1 - n_{kU}) · D̄_k(1 - D̄_k)                              (Eq. 7)
V̂_{kℓ}^{D,k} = n_{kℓ}(1 - n_{kℓ}) · (D̄_k - D̄_ℓ)/(1 - D̄_ℓ) · (1 - D̄_k)/(1 - D̄_ℓ)   (Eq. 8)
V̂_{kℓ}^{D,ℓ} = n_{kℓ}(1 - n_{kℓ}) · D̄_ℓ/D̄_k · (D̄_k - D̄_ℓ)/D̄_k                  (Eq. 9)
```

Decomposition weights:
```
s_{kU}   = ((n_k + n_U)^2 · V̂_{kU}^D)        / V̂^D                                  (Eq. 10e)
s_{kℓ}^k = ((n_k + n_ℓ)(1 - D̄_ℓ))^2 · V̂_{kℓ}^{D,k} / V̂^D                            (Eq. 10f)
s_{kℓ}^ℓ = ((n_k + n_ℓ) · D̄_k)^2 · V̂_{kℓ}^{D,ℓ}    / V̂^D                            (Eq. 10g)
```

Where `V̂^D` is the variance of the FE-adjusted treatment dummy `D̃_{it} = D_{it} - D̄_i - D̄_t + D̄̄` across the **full** sample, and equals the sum of the numerators above (so weights sum to 1):
```
Σ_{k ≠ U} s_{kU} + Σ_{k ≠ U} Σ_{ℓ > k} [s_{kℓ}^k + s_{kℓ}^ℓ] = 1
```

*Timing-only pair decomposition (Equation 11):*

A 2x2 DD between two timing groups is itself a weighted average of the two "directional" 2x2s:
```
β̂_{kℓ}^{2x2} ≡ μ_{kℓ} · β̂_{kℓ}^{2x2,k} + (1 - μ_{kℓ}) · β̂_{kℓ}^{2x2,ℓ}              (Eq. 11)

μ_{kℓ} = ((1 - D̄_ℓ) · V̂_{kℓ}^{D,k})
       / ((1 - D̄_ℓ) · V̂_{kℓ}^{D,k} + D̄_k · V̂_{kℓ}^{D,ℓ})

Simplification: μ_{kℓ} = (1 - D̄_k) / (1 - D̄_k + D̄_ℓ)
```

The timing group treated closest to the **middle** of the panel gets more weight.

*Probability-limit decomposition (Equations 14a-c, 15; pp. 260-261):*

Substituting potential-outcome definitions into the 2x2 DDs:
```
β_{kU}^{2x2}   = ATT_k(POST(k)) + [ΔY_k^0(POST(k), PRE(k)) - ΔY_U^0(POST(k), PRE(k))]      (Eq. 14a)
β_{kℓ}^{2x2,k} = ATT_k(MID(k,ℓ)) + [ΔY_k^0(MID(k,ℓ), PRE(k)) - ΔY_ℓ^0(MID(k,ℓ), PRE(k))]  (Eq. 14b)
β_{kℓ}^{2x2,ℓ} = ATT_ℓ(POST(ℓ)) + [ΔY_ℓ^0(POST(ℓ), MID(k,ℓ)) - ΔY_k^0(POST(ℓ), MID(k,ℓ))]
              - [ATT_k(POST(ℓ)) - ATT_k(MID(k,ℓ))]                                          (Eq. 14c)
```

The first term in each is an ATT; the second is a differential-trend bias; the **third term in Eq. 14c is unique** — it picks up the change in `k`'s treatment effect between `MID(k,ℓ)` and `POST(ℓ)`. When `k`'s treatment effect is constant over time this term is zero; when it grows or shrinks, this term biases `β̂_{kℓ}^{2x2,ℓ}` away from `ATT_ℓ(POST(ℓ))`.

*Estimand decomposition (Equation 15):*
```
plim β̂^DD = VWATT + VWCT - ΔATT
```

Where (Equations 15a, 15b, 15c):

`VWATT` — **Variance-Weighted ATT**, the only interpretable causal piece:
```
VWATT ≡ Σ_{k ≠ U} σ_{kU} · ATT_k(POST(k))
      + Σ_{k ≠ U} Σ_{ℓ > k} [ σ_{kℓ}^k · ATT_k(MID(k,ℓ)) + σ_{kℓ}^ℓ · ATT_ℓ(POST(ℓ)) ]   (Eq. 15a)
```
(The `σ` terms are the population analogues of `s` from Eq. 10e-g.)

`VWCT` — **Variance-Weighted Common Trends**, the differential-trend bias generalized to timing:
```
VWCT ≡ Σ_{k ≠ U} σ_{kU} · [ΔY_k^0(POST(k), PRE(k)) - ΔY_U^0(POST(k), PRE(k))]
     + Σ_{k ≠ U} Σ_{ℓ > k} σ_{kℓ}^k · [ΔY_k^0(MID(k,ℓ), PRE(k)) - ΔY_ℓ^0(MID(k,ℓ), PRE(k))]
     + Σ_{k ≠ U} Σ_{ℓ > k} σ_{kℓ}^ℓ · [ΔY_ℓ^0(POST(ℓ), MID(k,ℓ)) - ΔY_k^0(POST(ℓ), MID(k,ℓ))]   (Eq. 15b)
```
Zero when potential-untreated outcome trends are parallel across timing groups.

`ΔATT` — **change in ATT within already-treated groups**, source of negative-weight pathology:
```
ΔATT ≡ Σ_{k ≠ U} Σ_{ℓ > k} σ_{kℓ}^ℓ · [ATT_k(POST(ℓ)) - ATT_k(MID(k,ℓ))]                  (Eq. 15c)
```
Zero when treatment effects in each timing group don't change over time; nonzero when they do (e.g., dynamic effects, learning, depreciation).

*VWCT approximation (Equation 19, p. 263):*

If average untreated potential-outcome trends are **linear** (`ΔY_k^0 ≡ ΔY_k^0(t)` independent of `t`):
```
VWCT ≈ Σ_k ΔY_k^0 · [w_k^T - w_k^C]                                                       (Eq. 19)
```
Where the superscript on each `σ` term identifies which group plays the **treatment** role in the corresponding 2x2 DD:
- `w_k^T = σ_{kU} + Σ_{j < k} σ_{jk}^k + Σ_{j > k} σ_{kj}^k` — total decomposition weight on `k` as **treatment**. All superscripts equal `k` because `k` is the treatment group in each contributing 2x2.
- `w_k^C = Σ_{j < k} σ_{jk}^j + Σ_{j > k} σ_{kj}^j` — total weight on `k` as **control**. All superscripts equal the *other* group `j` (the one playing the treatment role), because `k` is the control in those 2x2 DDs.

Bias from a positive differential trend in group `k` equals `ΔY_k^0 · (w_k^T - w_k^C)`. **No bias when `w_k^T = w_k^C`** — extreme-timed groups can have `w_k^C > w_k^T`, in which case a positive trend in their counterfactual induces *negative* bias.

*Linear-trend-break wrong-sign case (Equations 17-18, p. 262):*

If `Y_{it}(1) = Y_{it}(0) + φ · (t - t_i + 1)` (treatment effect grows linearly post-adoption) with parallel `Y(0)`, then:
```
β̂_{kℓ}^{2x2,ℓ} = ATT_ℓ(POST(ℓ)) - φ · (ℓ - k) / 2   ≤ 0 when φ > 0                       (Eq. 17)

β̂_{kℓ}^{2x2}   = φ · [(σ_{kℓ}^k - σ_{kℓ}^ℓ)(ℓ - k + 1)] / 2                              (Eq. 18)
```
The two-group timing estimate can be **wrong-signed** if there is more weight on `β̂_{kℓ}^{2x2,ℓ}` than `β̂_{kℓ}^{2x2,k}` (i.e., `σ_{kℓ}^ℓ > σ_{kℓ}^k`). Summarizing time-varying effects using Eq. (2) yields estimates that are "too small or even wrong-signed, and should not be used to judge the meaning or plausibility of effect sizes" (p. 262).

*Identification under heterogeneous-but-time-constant ATT (Equation 16, p. 261):*

If `ATT` varies across units but **not** over time, `ΔATT = 0` and (assuming `VWCT = 0`):
```
plim β̂^DD = Σ_{k ≠ U} ATT_k · [ σ_{kU} + Σ_{j < k} σ_{jk}^k + Σ_{j > k} σ_{kj}^k ]
          = Σ_{k ≠ U} ATT_k · w_k^T                                                       (Eq. 16)
```
TWFEDD identifies a variance-weighted ATT with weights `w_k^T`, generally **not equal** to the sample share `n_k / (1 - n_U)`. The discrepancy depends on the relationship between treatment effect heterogeneity and treatment timing — Roy-style selection (largest effects roll out first) means TWFEDD under-weights early-treated groups and overstates the sample-ATT; "reverse Roy" (smallest effects first) yields the opposite.

*Algorithm (per Theorem 1, pp. 257-258):*

1. **Identify K timing groups** based on first-treatment time `t_i ∈ {1, ..., T} ∪ {∞}`. Always-treated units (`t_i < 1`, i.e., treated before the first observable period) enter as part of `U` (paper footnote 11, p. 259).
2. **Compute sample shares** `n_k`, treatment shares `D̄_k`, and the pairwise relative sizes `n_{kU}`, `n_{kℓ}`.
3. **Compute the full-sample FE-adjusted treatment variance** `V̂^D` = `Var(D̃_{it})` where `D̃_{it} = D_{it} - D̄_{i·} - D̄_{·t} + D̄̄`.
4. **For each comparison** `(k, ℓ)` with `ℓ > k` and each `(k, U)` with `k ≠ U`:
   - Compute the 2x2 DD via Eqs. 10b-d using only the relevant period windows.
   - Compute the subsample variance via Eqs. 7-9.
   - Compute the weight via Eqs. 10e-g.
5. **Verify the contract**: `Σ s = 1` and `β̂^DD = Σ s · β̂^{2x2}` (matches the OLS coefficient from the TWFEDD regression).
6. **Report** by comparison type (treated/untreated, earlier/later, later/earlier) with weights, estimates, and weighted contributions.

*Diagnostic interpretation (Section 4, pp. 264-266):*

- **Plot** each 2x2 DD against its weight (scatter plot). Three marker types (treated/untreated, earlier/later, later/earlier) reveal which type of variation drives the overall estimate.
- **Sum weights by type** to quantify the share of identifying variation that comes from timing-only comparisons. Stevenson-Wolfers replication: 37% from timing comparisons, with the late-as-control 2x2s biasing the overall estimate.
- **Average within type**: if later-vs-earlier 2x2s average to a different value than treated-vs-untreated 2x2s, treatment effects are likely changing over time and TWFEDD is biased relative to `VWATT`.

*Standard errors / variance:*

The decomposition itself is a deterministic algebraic identity of sample moments. Standard errors are **not the focus of the paper** — the 2x2 DDs, weights, and total are point identities given the data, and `β̂^DD` from the decomposition matches the OLS coefficient from TWFEDD by construction.

Inference for the TWFEDD coefficient itself is typically clustered at the unit level (cluster-robust SE per the empirical convention; the Stevenson-Wolfers replication on p. 265 uses heteroskedasticity-robust SE = 1.13 (or 1.27 with original Stevenson-Wolfers data)). The decomposition operates pre-inference: it asks "what does the OLS coefficient equal?", not "what is its sampling distribution?".

*Edge cases:*

- **Always-treated units** (paper footnote 11, p. 259): Units treated before `t = 1` (i.e., `t_i < 1`) enter the decomposition just like untreated units — they only ever serve as controls in `β̂_{kU}^{2x2}`-type terms with weights `s_{kU}`. The decomposition implicitly groups them into `U`.

  **Library handling (PR-B audit — landed):** `bacon.py` automatically remaps genuinely always-treated units — those whose `first_treat` is at or before the first observable period (`first_treat <= min(time)`, excluding the never-treated sentinels `0` and `np.inf`; detection uses ordered-time logic, so negative / zero-crossing `time` axes are handled correctly) — into the `U` bucket per footnote 11 (option (a) of the original audit question), via an internal column with a `UserWarning`; the count is surfaced as `BaconDecompositionResults.n_always_treated_remapped` and the user's original `first_treat` column is preserved. The library uses the **inclusive** boundary `first_treat <= min(time)` (also folding in `first_treat == min(time)` cohorts, which have no untreated cell in-window) — a documented extension of the paper's strict `t_i < 1`. See REGISTRY `## BaconDecomposition` → **Note (always-treated remap)** and **Deviation (first-period boundary extension on always-treated remap)** for the authoritative contract, including the R `bacondecomp` per-component parity-convention divergence.
- **Never-treated units**: Same as always-treated — enter as untreated controls; only `s_{kU}`-type terms involve them.
- **No untreated group**: The `s_{kU}` terms drop; only timing-only comparisons (`s_{kℓ}^k`, `s_{kℓ}^ℓ`) remain, with weights rescaled to sum to 1. **Both VWCT and ΔATT can still introduce bias** — VWCT from differential trends *between* timing groups (the `σ_{kℓ}^k` and `σ_{kℓ}^ℓ` terms in Eq. 15b are not eliminated when `s_{kU}` drops), and ΔATT from time-varying treatment effects in already-treated controls. Footnote 15 (p. 261) emphasizes that ΔATT is the bias source unique to designs with timing variation; it does not say VWCT vanishes.
- **D̄_k → 0 or 1**: A timing group with no within-window treatment variation has `V̂^D = 0` and contributes zero weight (Eqs. 7-9 go to zero).
- **n_{kU} → 0 or 1**: Same — no variation in group membership within the pair means no weight.
- **Timing at panel boundaries**: Groups treated near `t=1` or `t=T` have low `D̄_k(1-D̄_k)` variance and thus low *total* weight but also low `w_k^T - w_k^C` (Fig. 4, p. 264) — they tend to act as controls more than as treatments.
- **Time-varying treatment effects**: `ΔATT ≠ 0` → TWFEDD does not identify `VWATT`. The diagnostic flags this through divergence between later-vs-earlier 2x2s and treated-vs-untreated 2x2s.
- **Linear trend break**: Eq. 17-18 — entire two-group timing DD can be wrong-signed.
- **Constant ATT across units AND time**: `ATT_k(W) = ATT` for all `k, W` → `VWATT = ATT`, `ΔATT = 0`, and TWFEDD identifies the constant treatment effect exactly.

*Algorithm extensions — controls and weighting (Section 5):*

**Oaxaca-Blinder-Kitagawa decomposition for specification comparison (Equation 20, p. 267):**

The DD decomposition can be written as `β̂^DD = s' · β̂^{2x2}`. Comparing an alternative specification `β̂_{alt}^DD = s'_{alt} · β̂_{alt}^{2x2}`:
```
β̂_{alt}^DD - β̂^DD = s' · (β̂_{alt}^{2x2} - β̂^{2x2})              (due to 2x2 DDs)
                  + (s'_{alt} - s') · β̂^{2x2}                    (due to weights)
                  + (s'_{alt} - s') · (β̂_{alt}^{2x2} - β̂^{2x2})   (interaction)             (Eq. 20)
```
This makes it possible to attribute specification-induced changes in `β̂^DD` to changes in the underlying 2x2 DDs, changes in weights, or their interaction.

**Population-weighted (WLS) vs. unweighted (OLS) (Section 5.1, p. 267):**

Population weighting affects both the 2x2 DD components (via cell-level weighted means in `ȳ`) and the decomposition weights (via population shares replacing sample shares in `n_k`). Eq. 20 isolates which channel dominates the discrepancy.

**DD with time-varying controls (Section 5.2, Equations 21-27):**

Specification with covariates `X_{it}` (Eq. 21):
```
y_{it} = α_i + α_t + Φ · X_{it} + β^{DD|X} · D_{it} + e_{it}
```

Partialing `X_{it}` out of `D_{it}` via Frisch-Waugh-Lovell (Eq. 22):
```
D̃_{it} = Γ̂ · X̃_{it} + d̃_{it}      (with p̃_{it} ≡ Γ̂ · X̃_{it})
```

Yielding (Eq. 23):
```
β̂^{DD|X} = Cov(y_{it}, d̃_{it}) / V̂^d = Cov(y_{it}, D̃_{it} - p̃_{it}) / V̂^d
```

Splitting `d̃_{it}` into within-timing-group and between-timing-group components (Eq. 24 → Eq. 25), the controlled estimator becomes:
```
β̂^{DD|X} = Ω · β̂_w^d + (1 - Ω) · β̂_b^d                                                  (Eq. 25, 27)
```
Where `Ω = V̂_w^d / V̂^d` is the share of identifying variation from within-timing-group covariate movements (only present with controls), and `β̂_b^d` decomposes via a modified Theorem 1 (Eq. 26) into adjusted 2x2 DDs `β̂_{b,kℓ}^{2x2|d}` with weights `s_{kℓ}^{b|X}`.

**Implication:** adding controls introduces a "within-timing-group" identifying variation source that is **not in** the uncontrolled decomposition. Controlled DD coefficients are not directly comparable to the uncontrolled DD decomposition; the Section 5.2 derivation provides the bridging machinery.

*Practitioner guidance (Section 6 Conclusion, p. 272):*

- TWFEDD identifies VWATT (an interpretable parameter) only under both `VWCT = 0` *and* `ΔATT = 0` — the latter requires **constant treatment effects within each timing group over time**.
- "Researchers seeking to exploit variation in treatment timing to estimate causal effects should use TWFEDD with caution. The TWFEDD estimator only has a meaningful causal interpretation under strong assumptions on treatment effects and even then it yields a parameter that may differ from what researchers have in mind."
- "If treatment effects are likely to vary over time one should not use TWFEDD to summarize the estimated effects. If the variance-weighted average of treatment effects is not of interest one should not use TWFEDD either."
- Alternatives that "carefully construct control groups to address the bias from time-varying treatment effects": Borusyak-Jaravel (2017), de Chaisemartin-D'Haultfœuille (2020), Sun-Abraham (2020), Ben-Michael-Feller-Rothstein (2019). Plus reweighting estimators like Callaway-Sant'Anna (2020) "which allow control over the target parameter".

**Reference implementation(s):**

- **Stata:** `bacondecomp` (SSC). Authors: Goodman-Bacon, Goldring, Nichols (2019). Available via `ssc install bacondecomp`.
- **R:** `bacondecomp` (CRAN). Implements `bacon()` with the same algorithmic specification.
- The paper does not cite a Python implementation; this library's `diff_diff/bacon.py` is independent.

**Requirements checklist:**

- [ ] Implements Theorem 1 weighting algorithm exactly (Eqs. 7-9 for variances, Eqs. 10b-g for 2x2 DDs and weights)
- [ ] Computes all three comparison types: treated/untreated, earlier/later (`β̂_{kℓ}^{2x2,k}`), later/earlier (`β̂_{kℓ}^{2x2,ℓ}`)
- [ ] Period windows `PRE(k)`, `MID(k, ℓ)`, `POST(k)`, `POST(ℓ)` defined per Section 2
- [ ] Always-treated units correctly grouped into `U` (footnote 11): they only ever serve as controls in `β̂_{kU}^{2x2}` terms
- [ ] Never-treated units handled equivalently to always-treated (both enter `U`)
- [ ] Weights sum to 1 (numerical contract; assert exact within float tolerance)
- [ ] `β̂^DD` from the decomposition matches the TWFEDD OLS coefficient from Eq. (2)
- [ ] Reports weight × estimate by comparison type (sum of weights, weighted-average estimate per type)
- [ ] Visualization: scatter 2x2 DDs against weights, distinguishing the three comparison types (Fig. 6 of paper)
- [ ] Handles `D̄_k → 0/1` and `n_{kℓ} → 0/1` boundary cases (zero weight, no contribution)
- [ ] Documents that the diagnostic is **not** an estimator of treatment effects — it explains the OLS coefficient

---

## Implementation Notes

### Data Structure Requirements

- **Panel data**: `(unit_id, time, treatment_indicator, outcome)`.
- **Treatment**: binary, absorbing (irreversible). Treatment timing variable derived from first period in which `D_{it} = 1` per unit.
- **Time periods**: discrete and ordered; the decomposition uses `t ∈ {1, ..., T}` after relabeling.
- **No covariates** are required for the basic Theorem-1 decomposition. Controlled extension (Section 5.2) requires `X_{it}` matrix.
- **Paper assumption — balanced panel**: The Theorem 1 derivation in Appendix A (Lemma 1, Eqs. A.1-A.18) assumes the variables `z_{kt}` and `x_{kt}` are observed in every `(k, t)` cell. The paper does not derive the decomposition for unbalanced panels.
- **Library deviation — unbalanced panels accepted with a warning**: `diff_diff/bacon.py:491-499` emits a `UserWarning` ("Unbalanced panel detected. Bacon decomposition assumes balanced panels. Results may be inaccurate.") rather than raising. Users with unbalanced panels see *approximate* Theorem 1 weights; the exact paper contract holds only on balanced panels. Document this as a **Note (library extension):** in the REGISTRY entry.

### Computational Considerations

- **Complexity:** For `K` timing groups and `T` periods, the algorithm computes `O(K^2)` 2x2 DDs (one per ordered pair `(k, ℓ)` with `ℓ > k`) plus `K` treated-vs-untreated comparisons. Each 2x2 DD is `O(N_pair · T_window)` for sample sums. Total dominant cost: `O(K^2 · N · T)`.
- **Memory:** The window means `ȳ_k^PRE(j)`, `ȳ_k^MID(j,ℓ)`, `ȳ_k^POST(j)` are **pair-specific** (`MID(k,ℓ)` and `POST(ℓ)` depend on both `k` and `ℓ`), so the natural caching scale is `O(K^2)` window-means, not `O(K)`. Two strategies: (i) precompute per-group-per-time cell means `ȳ_{kt}` (`O(K · T)` storage) and then compute any window-mean via cumulative-sum lookup in `O(1)` per 2x2; (ii) accept the `O(K^2)` cache of window means computed directly per ordered pair. The variance scalars `D̄_k`, `n_{kU}`, `n_{kℓ}` remain `O(K)` and `O(K^2)` respectively.
- **Parallelization:** Each `(k, ℓ)` and `(k, U)` 2x2 computation is independent — embarrassingly parallel.
- **Numerical contract:** The identity `β̂^DD = Σ s · β̂^{2x2}` must hold to within float64 tolerance (`atol ≈ 1e-10` is reasonable). Weights summing to 1 should also be exact within tolerance.

### Tuning Parameters

| Parameter | Type | Default | Selection Method |
|-----------|------|---------|-----------------|
| (none — algorithmic) | — | — | The decomposition is parameter-free. The only user choices are: (a) the time panel, (b) the treatment-timing column, (c) optional controls/weights for Section 5 extensions. |

### Relation to Existing diff-diff Estimators

- **TwoWayFixedEffects** (`twfe.py`): produces the `β̂^DD` that this decomposition explains. The decomposition is the diagnostic tool *for* the TWFE estimator. The existing `TwoWayFixedEffects.decompose()` integration should call `BaconDecomposition` and return a `BaconDecompositionResults` object.
- **CallawaySantAnna** (`staggered.py`): An alternative estimator that targets `ATT(g, t)` directly and constructs an aggregation with explicit user-chosen weights — addressing the `ΔATT` bias.
- **SunAbraham** (`sun_abraham.py`): Saturated cohort × relative-time interactions; interaction-weighted aggregation avoids the negative-weight pathology by construction.
- **ImputationDiD** (`imputation.py`), **TwoStageDiD** (`two_stage.py`), **WooldridgeDiD** (`wooldridge.py`): Each uses an explicit pre-/post-period imputation or saturated cohort regression to sidestep the ΔATT bias the Bacon decomposition diagnoses.
- **ChaisemartinDHaultfoeuille** (`chaisemartin_dhaultfoeuille.py`): Provides the "TWFE weights diagnostic" (a related but distinct decomposition; cited in the paper as Borusyak-Jaravel / dCDH-style estimand-level weights, vs. Theorem 1's estimator-level weights). The two diagnostics complement each other.

### Related Decompositions (Section 2 footnote 7 + p. 260)

- **Borusyak-Jaravel (2017) / de Chaisemartin-D'Haultfœuille (2020):** Decompose the TWFEDD *estimand* into a weighted average of ATTs with potentially **negative** weights. Theorem 1 here decomposes the *estimator* into a weighted average of simpler estimators with strictly **positive** weights. The two are connected (Section 2): the negative-weight pathology in BJ/dCDH corresponds to the `−ΔATT` term in Eq. (15).
- **Athey-Imbens (2018, eq. 4.3):** Decompose `β̂^DD` into terms representing causal effects over different time horizons (pre/post, treatment/control). The Bacon decomposition groups 2x2 components by the identifying variation; AI groups them by causal-effect horizon. Both unify the same estimator.
- **Strezhnev (2018):** Expresses `β̂^DD` as an unweighted average of comparisons between all units and all time periods. Weights across comparison types (2x2 DDs) are only implicitly defined in this decomposition.

---

## Gaps and Uncertainties

1. **Appendix B and Appendix C**: The published paper says (footnote 36, p. 272) that Appendix B "analyzes triple-difference models and shows that they also have a weighted average form" and Appendix C "briefly discusses treatment variables that turn off". These are not in the published Journal of Econometrics article; the paper cites an external appendix at http://goodman-bacon.com/pdfs/ddtiming_appendix.pdf. **For implementation purposes, only Appendix A (proof of Theorem 1) is available in the PDF reviewed here.** If the library wants to extend to triple-difference or non-absorbing treatment, the external appendix is needed.

2. **Variance/SE for the decomposition components**: The paper does not provide standard errors for individual `β̂_{kℓ}^{2x2}` or `s_{kℓ}` terms. The decomposition is treated as a deterministic algebraic identity. If users want SEs for the *components*, that requires a separate derivation — e.g., bootstrap. The paper's empirical Stevenson-Wolfers replication reports a single TWFEDD SE (=1.13 unweighted, =1.27 from the original paper) for the overall coefficient.

3. **Equation 7 simplification with population weighting**: Section 5.1 says population weighting "increases the influence of large units ... and large groups by basing the decomposition weights on population rather than sample shares." The exact updated form of Eq. 10e-g under WLS is not transcribed in the main text but follows from substituting weighted sample shares into the same algebraic structure. The R `bacondecomp` package implements this; comparing the Python implementation to R is the best parity check.

4. **Boundary case: K = 1 (only one timing group, no untreated)**: The paper assumes `K ≥ 1` and `K = 1` requires an untreated group `U`. With `K = 1` and no `U`, there is no variation in `D_{it}` and TWFEDD is undefined. The library should raise an explicit error rather than return zero or NaN.

5. **Eq. 19 linearity assumption**: VWCT ≈ formula assumes linear average untreated trends. The paper notes this is "a common starting point, [but] it approximates non-linear pre-trends, and it provides a simple way to increase the power of such pre-tests" (footnote 21, p. 264). For non-linear trends, this approximation breaks down — but the underlying decomposition still holds; only the heuristic mapping from `(w_k^T - w_k^C)` to bias requires the linearity assumption.

6. **Treatment-effect heterogeneity within a single timing group**: The paper assumes `ATT_k(W)` is well-defined as an average over units in group `k` over window `W`. If unit-level treatment effects differ within `k`, this is absorbed into the average — the decomposition does not surface within-timing-group heterogeneity. (This is intentional; the Callaway-Sant'Anna and Sun-Abraham frameworks make within-cohort heterogeneity a first-class concern.)

7. **Continuous or non-binary treatment**: The paper assumes binary treatment. Extension to continuous treatment requires a different decomposition (the dCDH and Sun-Abraham frameworks address some of this; not covered here).

8. **`stacked_did` connection**: The paper's footnote 15 (p. 261) mentions "stacked DD" (Cengiz et al. 2019, Deshpande-Li 2019, Fadlon-Nielsen 2015) as one of several alternatives that "address the bias from time-varying treatment effects". The Bacon decomposition does not directly diagnose stacked DD; it diagnoses the unstacked TWFEDD coefficient.
