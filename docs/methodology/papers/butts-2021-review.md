# Paper Review: Difference-in-Differences with Spatial Spillovers

**Authors:** Kyle Butts
**Citation:** Butts, K. (2023). Difference-in-Differences with Spatial Spillovers. arXiv:2105.03737v3 (originally posted 2021; v3 dated June 13, 2023). https://arxiv.org/abs/2105.03737
**PDF reviewed:** papers/2105.03737v3.pdf (35 pages)
**Review date:** 2026-05-09

---

## CRITICAL FINDING — Paper Does Not Use Kernel-Weighted Exposure

The Phase 3 brief describes the canonical estimator as a kernel-weighted exposure regressor `E_it = Sum_{j != i} w_ij * D_jt` added to the design matrix. **Butts (2021) v3 does NOT actually propose this form.** Butts proposes a **ring / distance-bin indicator** estimator and references kernel/spatial-decay exposure only in a footnote as one example of how a generic exposure mapping `h_i(D-vector)` could be defined.

The paper's headline estimators are:
- Equation (5): single "near treatment" ring indicator `S_i` interacted with `(1 - D_i)`.
- Equation (6): multiple concentric ring indicators `Ring_{ij}` interacted with `(1 - D_it)`.
- Equation (8): TVA application uses 4 distance bins `{(0,50], (50,100], (100,150], (150,200]}` miles with ring indicators.

The text on page 6, footnote 8, says only: *"For example, h_i(D-vector) could be an indicator for being within a certain distance of a treated unit. Additionally, exposure could be defined as a spatial decay function where exposure is declining in the distance of treated units."* This is a generic remark about possible exposure-mapping families inside the potential-outcomes framework (Section 2), not a proposed estimator. The paper's identification result (Proposition 2.3) and all empirical specifications use **ring/indicator** rather than continuous kernel exposure.

**Implications for diff-diff Phase 3:**
- If Phase 3 implements the ring-indicator estimator, Butts (2021) IS the canonical source. The decomposition `tau_total - tau_spill(0)` (Equations (1)-(2)), Proposition 2.3, and the rings construction (Equation (6)) are exactly what to cite.
- If Phase 3 implements a continuous kernel-weighted `E_it = Sum_{j != i} w_ij * D_jt`, that specification is **NOT** from this paper. It is closer to Clarke (2017) MPRA paper or to design-based / linear-in-means peer-effects literature (Manski 1993, Goldsmith-Pinkham and Imbens 2013) referenced in Butts' Section 1.1. A separate paper review (Clarke 2017) is needed to source the kernel form, OR the diff-diff API should be reframed around ring/distance bins as the primary spec with kernel as an extension.
- The user-facing API design should be reconsidered. `spillover_kernel="exp"|"inverse_distance"|"power"|callable` does not map to anything in Butts (2021). What maps is `spillover_rings=[(0, 50), (50, 100), (100, 150), (150, 200)]` (or a single ring for a one-indicator spec) plus the "far-away" cutoff `d-bar`.
- The `direct_effect` and `spillover_effect` terminology can still be derived from Butts: `tau_total` (direct, from Proposition 2.3) and `tau_spill(s)` for s in {0, 1} or per-ring. Butts uses `tau_total` for treated-unit effects and `gamma_0` for the spillover-on-control coefficient (Equation 5).

The remainder of this review documents what Butts ACTUALLY proposes (the ring-indicator estimator) so the diff-diff REGISTRY entry for Phase 3 is accurate.

---

## Methodology Registry Entry

*Formatted to match docs/methodology/REGISTRY.md structure. Heading levels and labels align with existing entries — copy the `## SpilloverDiD` section into the appropriate category in the registry.*

## SpilloverDiD (ring-indicator)

**Primary source:** Butts, K. (2023). Difference-in-Differences with Spatial Spillovers. arXiv:2105.03737v3.

**Scope:** A two-period (Section 2-3) and staggered (Section 5) DiD setting in which treatment is assigned by an administrative boundary but treatment effects spill over onto nearby control units. The estimator separately identifies (i) the **total effect on the treated** `tau_total = E[Y_i1(1, h_i(D-vector)) - Y_i1(0, 0-vector) | D_i = 1]` and (ii) **spillover effects on control units** `tau_spill(0)`, by adding "near-treatment" ring indicators interacted with `(1 - D_it)` to the canonical TWFE regression. Identification is non-parametric in the spillover function — the researcher does not need to specify how spillovers decay over space, only the maximum distance `d-bar` past which spillovers do not occur.

**Key implementation requirements:**

*Notation (Section 2, page 6):*
- `D_i` ∈ {0,1}: treatment status; `D_it = D_i * 1{t=1}`.
- `D-vector` ∈ {0,1}^n: full vector of treatment statuses across all units.
- `h_i(D-vector)`: non-negative scalar- or vector-valued **exposure mapping** — generic representation of how unit `i` is affected by spillovers from the other units' treatments. `h_i(0-vector) = 0-vector` by definition.
- Potential outcome: `Y_it(D_i, h_i(D-vector))` — depends on own treatment AND on the exposure mapping.
- `S_i`: indicator for `min_{j: D_j=1} d(i,j) <= d-bar` (i.e., `i` is within `d-bar` miles of the nearest treated unit). Note `D_i = 1` implies `S_i = 1`.
- `Ring_{ij}`: indicator for unit `i` being in the `j`-th distance ring from treatment.

*Assumption checks / warnings:*
- **Assumption 1 (Random Sampling)**: `{Y_i0, Y_i1}_{i=1..n}` is i.i.d. panel.
- **Assumption 2 (No Anticipation)**: `Y_i0(D, h) = Y_i0(0, 0)` for all D and h. The pre-treatment outcome does not depend on future treatment / exposure.
- **Assumption 3 (Parallel Counterfactual Trends)**: counterfactual trends do NOT depend on `D_i`:
  `E[Y_i1(0, 0-vector) - Y_i0(0, 0-vector) | D_i = 1] = E[Y_i1(0, 0-vector) - Y_i0(0, 0-vector) | D_i = 0]`.
  (Note: this is a STRONGER assumption than ordinary parallel trends — it requires parallel trends in the absence of all treatment AND zero exposure, not merely absence of own treatment. Reduces to standard parallel trends when SUTVA holds, since then every unit has zero exposure.)
- **Assumption 5 (Spillovers Are Local)**: there exists `d-bar` such that
  (i) `min_{j: D_j=1} d(i,j) > d-bar => h_i(D-vector) = 0-vector` (spillovers vanish past `d-bar`); and
  (ii) there exist treated and control units with `min_{j: D_j=1} d(i,j) > d-bar` (i.e., the sample contains far-away units that can serve as a clean control group).
- **Assumption 6 (Total Effect Parallel Trends)**: counterfactual trends do not depend on `D_i` AND `S_i`:
  `E[Y_i1(0, 0-vector) - Y_i0(0, 0-vector) | D_i = 1] = E[Y_i1(0, 0-vector) - Y_i0(0, 0-vector) | D_i = 0, S_i = 0]`.
- **Assumption 7 (Spillover Effect Parallel Trends)**: for `s = 0, 1`,
  `E[Y_i1(0, 0-vector) - Y_i0(0, 0-vector) | D_i = 1] = E[Y_i1(0, 0-vector) - Y_i0(0, 0-vector) | D_i = 0, S_i = s]`.
  Required to estimate `gamma_0` (spillover on control) from Equation (5). STRONGER than Assumption 6.
- **Assumption 8 (Parallel Counterfactual Trends, Staggered, Equation 9)**: for all `i, t`, `Y_it(0, 0-vector) = mu_i + lambda_t + epsilon_it` with `E[epsilon_it] = 0`. Imposes a unit + time additive structure on untreated/unexposed potential outcomes (Section 5). Stronger than Assumption 3.
- **Assumption 9 (No Anticipation, Staggered)**: for all `(i,t)` with `D_it = 0` and `h_i(D-vector_t) = 0-vector`, `Y_it(D, h) = Y_it(0, 0)`.
- Warn that choice of `d-bar` is a researcher decision — Butts notes (page 13) "each value of `d-bar` corresponds to a different effective control group and hence a different parallel trends assumption" — and that data-driven selection under stricter assumptions is in the companion paper Butts (2021b) "Difference-in-Differences with Geocoded Microdata" (the "ring method" paper).
- Warn that using `S_i = 1` close-to-treatment as the control group (which is common in border-RD applied work) magnifies spillover bias, since these units experience the largest spillover effects (Section 4.1 / page 22).

*Treatment effects definitions (Section 2.1, page 7-8):*

Switching effect:
```
tau_{i,switch}(h-vector) := Y_i1(1, h-vector) - Y_i1(0, h-vector)
tau_switch(h-vector) := E[ Y_i1(1, h_i(D-vector)) - Y_i1(0, h_i(D-vector)) | D_i = 1, h_i(D-vector) = h-vector ]
```
Effect of changing only `i`'s treatment, holding exposure fixed at `h-vector`. Policy-relevant for local policymakers.

Total effect on the treated (the diff-diff Phase 3 `direct_effect`):
```
tau_total := E[ Y_i1(1, h_i(D-vector)) - Y_i1(0, 0-vector) | D_i = 1 ]
```
Effect of going from no-treatment-no-exposure world to enacted treatment vector. Policy-relevant for national policymakers.

Spillover effect:
```
tau_{i,spill}(D_i, h_i(D-vector)) := Y_i1(D_i, h_i(D-vector)) - Y_i1(D_i, 0-vector)
tau_spill(D) := E[ Y_i1(D, h_i(D-vector)) - Y_i1(D, 0-vector) | D_i = D ]
```
Average over all treated/control units regardless of whether they actually experience non-zero exposure.

Algebraic identity (page 9, unnumbered):
```
[Y_i1(1, h_i(D-vector)) - Y_i1(0, h_i(D-vector))]   [SWITCHING EFFECT]
= [Y_i1(1, 0-vector) - Y_i1(0, 0-vector)]           [DIRECT EFFECT]
+ [Y_i1(1, h_i(D-vector)) - Y_i1(1, 0-vector)]      [SPILLOVER ON TREATED]
- [Y_i1(0, h_i(D-vector)) - Y_i1(0, 0-vector)]      [SPILLOVER ON CONTROL]
```
The first two terms together equal the TOTAL EFFECT ON TREATED.

*Bias decomposition of the canonical DiD estimand (Proposition 2.1, Equations 1-2):*

Under Assumptions 1, 2, and 3 (Parallel Counterfactual Trends):
```
E[Y_i1 - Y_i0 | D_i = 1] - E[Y_i1 - Y_i0 | D_i = 0]   (1)
                                                        (Difference-in-Differences)
= tau_total - tau_spill(0)                              (2)
```
The standard DiD estimator is biased for `tau_total` by `-tau_spill(0)` — the average spillover effect onto control units. If spillovers and total effect have the same sign, the canonical DiD is **attenuated**; opposite signs, **inflated**. Butts then estimates the canonical TWFE (Equation 3):
```
y_it = tau * D_it + mu_i + lambda_t + epsilon_it     (3)
```
and shows `tau-hat` is the sample analog of (1), hence biased for `tau_total` whenever spillovers exist.

*Identification of the total effect (Proposition 2.3, Equation 4):*

Under Assumptions 1, 2 (random sampling), Assumption 5 (Spillovers Are Local) and Assumption 6 (Total Effect Parallel Trends):
```
E[Y_i1 - Y_i0 | D_i = 1] - E[Y_i1 - Y_i0 | D_i = 0, S_i = 0] = tau_total      (4)
```
Conditioning on `S_i = 0` uses only "far-away" control units, which by Assumption 5 have `h_i(D-vector) = 0-vector` and so identify the counterfactual trend cleanly. **Crucially, the researcher does NOT need to know the form of the exposure mapping** — only an indicator for "close to treatment" within distance `d-bar`.

*Estimator equation — single ring (Equation 5):*

```
Y_it = tau * D_it + gamma_0 * (1 - D_it) * S_i + mu_i + lambda_t + epsilon_it     (5)
```
- `tau-hat` consistent for `tau_total` under Assumption 7 (Spillover Effect Parallel Trends).
- `gamma_0-hat` averages spillover effects across **all** units with `S_i = 1` (treated AND untreated), specifically `E[gamma_0-hat] = E[tau_{i,spill}(0) | S_i = 1, D_i = 0]`. This is NOT `tau_spill(0)` — and may be attenuated towards zero if the indicator captures many units with no actual spillover exposure.

*Estimator equation — multiple rings (Equation 6):*

```
y_it = tau * D_it + sum_{j=1..n_rings} (1 - D_it) * Ring_{ij} * delta_j + mu_i + lambda_t + epsilon_it     (6)
```
- Each `delta_j` estimates the average `tau_{i,spill}(0)` for control units inside ring `j`.
- `tau-hat` continues to identify `tau_total` because the rings are collinear with the "single big S_i" indicator.
- Multiple rings better trace the spatial decay function as the number of rings grows and ring widths shrink — semi-parametric (Section 3.2, page 15).
- Bias-variance trade-off: more rings = less bias on the spillover function but smaller cell counts and noisier estimates. Clarke (2017) proposes cross-validation; Butts (2021b) proposes data-driven selection under a stricter parallel-trends assumption (footnote 14, 15).
- Caveat (page 16, end of Section 3): "spillover effects are additive in the number of nearby treated units... summarizing exposure by the distance to the closest treated unit fails to capture important information." If spillovers are additive in nearby treated units, count of treated units within each ring is preferred, but then bias is fully removed only if the exposure mapping is correctly specified — which the paper otherwise does not require.

*Estimator equation — staggered TWFE / event-study (Section 5, Table 2):*

Two-stage imputation estimator following Gardner (2021):
1. Estimate `Y_it = mu_i + lambda_t + u_it` on observations with `D_it = 0` AND `S_it = 0` (untreated AND unexposed). Compute residuals `Y-tilde_it := Y_it - mu-hat_i - lambda-hat_t`.
2. Regress `Y-tilde_it` on treatment + spillover dummies.

Table 2 (page 24) gives the second-stage variables for each estimand:

| Estimand | Included Variables |
|----------|-------------------|
| Total Effect | `D_it` |
| Total Effect (Event Study) | `D^k_{it}` dummies |
| Spillover Effect on Control | `S_it (1 - D_it)` or `Ring_{it,j} (1 - D_it)` |
| Spillover Effect on Control (Event Study) | `S^k_{it} (1 - D_it)` or `Ring^k_{it,j} (1 - D_it)` |

where `D^k_{it} := D_i * 1{K_it = k}` and `K_it` is years since treatment turned on.

**TWFE bias under staggered + spillover (page 22):** TWFE is a weighted sum of 2x2 DiDs (Goodman-Bacon 2018, Sun-Abraham 2020, de Chaisemartin-D'Haultfœuille 2019). Spillover bias enters each 2x2 with the same sign, but Goodman-Bacon weights can be negative — so the SIGN of `tau_spill` no longer determines the sign of the bias on the staggered TWFE. This makes spillover bias under staggered timing harder to sign than in the 2x2 case.

*Standard errors (Section 3.1, page 13):*

- **Cluster by unit `i`** to allow for serial correlation across periods.
- **Conley spatial HAC** (Conley 1999): "since assumption 5 is predicated on the fact that nearby places affect one another, we should account for such spatial correlation by allowing for spatial correlations following Conley (1999). More recent work by Ferman (2020) shows that for large-n asymptotics to be used, the structure of spatial correlation must be limited in that errors are assumed to be uncorrelated after a certain cutoff distance. A natural candidate for this cutoff would be `d-bar` used for the creation of `S_i`."
- TVA application (Table 1, page 19) uses Conley SEs with cutoff 200 miles.
- Two-way clustering (unit + time) is NOT explicitly recommended — Butts goes straight from clustering by unit to Conley spatial-HAC.
- For the staggered two-stage estimator, inference accounts for the first-stage estimation following Gardner (2021); implemented in `did2s` R/Stata package (Butts 2021a).

*Edge cases:*
- **No nearby control units (Assumption 5(ii) fails)**: `tau_total` cannot be identified from Proposition 2.3. Detection: count units with `S_i = 0` and `D_i = 0`; if zero, error. Handling: error or warn; recommend smaller `d-bar` if the user has chosen too large a cutoff.
- **`d-bar` too small** (some spillover-affected units classified as `S_i = 0`): residual bias remains in `tau-hat`. Detection: not detectable from data alone without an explicit decay model; Butts argues bias is small because spillovers decay over distance. Handling: sensitivity analysis across multiple `d-bar` values.
- **`d-bar` too large** (`S_i = 0` units share fewer characteristics with treated units): increases variance and may worsen parallel-trends on the control group. Detection: parallel-pre-trends test on the chosen `S_i = 0` group. Handling: bias-variance trade-off; smaller `d-bar` reduces variance.
- **Single-ring `S_i` indicator covers many unaffected units**: `gamma_0-hat` is attenuated towards zero. Detection: consistent near-zero estimates for the spillover coefficient when the ring is wide. Handling: switch to multiple concentric rings (Equation 6).
- **Spillovers extend past the largest ring**: `tau-hat` from Equation (6) remains biased. Detection: the outermost ring's `delta_j` is statistically different from zero. Handling: extend the outermost ring or signal to user.
- **Additive spillovers in number of treated neighbors**: distance-to-nearest-treated rings under-identify; recommend count-of-treated-in-ring instead, but this re-introduces functional-form dependence (Section 3.2 end / page 16).
- **Staggered timing with negative Goodman-Bacon weights**: ordinary TWFE can flip the sign of the spillover bias; use the two-stage Gardner-style estimator (Section 5).

*Algorithm (two-period, multiple rings):*
1. Compute distance from every unit `i` to the nearest treated unit: `d_i := min_{j: D_j = 1} d(i, j)`.
2. User supplies `d-bar` (max spillover distance) and a list of inner ring breakpoints `[r_0=0 < r_1 < ... < r_K = d-bar]`.
3. Build `Ring_{ij} := 1{r_{j-1} <= d_i <= r_j}` for `j = 1, ..., K`. Treated units have `d_i = 0` and are excluded from all rings (since the regressor multiplies by `(1 - D_it)`).
4. Augment design matrix with treatment dummy `D_it` and `(1 - D_it) * Ring_{ij}` for `j = 1, ..., K`.
5. Fit TWFE via partial-out / FW projection.
6. Read `tau-hat` (direct/total effect) and `delta_j-hat` (per-ring spillover-on-control effects).
7. Compute Conley spatial-HAC SEs with cutoff `d-bar` (recommended) or larger.
8. Optional pre-trends test: regress pre-period outcome differences on `D_i` and the rings to verify parallel trends on each subgroup.

*Algorithm (staggered, two-stage following Gardner 2021):*
1. Build `D_it` (treatment indicator) and `S_it` (within `d-bar` of nearest treated unit at time `t`).
2. Subset to `D_it = 0` AND `S_it = 0`. Estimate `Y_it = mu_i + lambda_t + u_it` on this subsample.
3. Compute residuals `Y-tilde_it := Y_it - mu-hat_i - lambda-hat_t` for all observations.
4. Regress `Y-tilde_it` on the second-stage variables in Table 2.
5. Compute SEs accounting for the first-stage estimation (GMM-style; see Gardner 2021).

**Reference implementation(s):**
- R: companion package `did2s` for the two-stage staggered estimator (Butts 2021a). https://github.com/kylebutts/did2s (cited in references and footnote 22).
- Stata: `did2s` Stata port also referenced.
- Software for the two-period ring estimator (Equations 5-6) is not separately distributed; it can be implemented directly via `lm` / `feols` after constructing ring indicators.

**Requirements checklist (for diff-diff Phase 3 mapping to Butts 2021):**
- [ ] Distance computation utility: `d(i, j)` between every pair (or at least `min_{j: D_j=1} d(i, j)` for each `i`).
- [ ] Ring-indicator builder taking `(distance_array, ring_breakpoints, d_bar)` and producing `(Ring_{ij})` matrix and `S_i` indicator.
- [ ] TWFE / two-way FE estimator that accepts arbitrary additional regressors interacted with `(1 - D_it)`.
- [ ] Result object exposing `direct_effect = tau_total`, per-ring `spillover_effects[j]`, `d_bar` cutoff used.
- [ ] Conley spatial-HAC SE (Phase 1 deliverable) integrated as the recommended SE option.
- [ ] Pre-trends test on the chosen `S_i = 0` control group.
- [ ] Two-stage staggered variant (deferrable to Phase 3+).

---

## Implementation Notes

### Data Structure Requirements
- **Distance input**: either a precomputed `d_ij` matrix (NxN, dense or sparse), OR per-unit `(latitude, longitude)` columns from which great-circle distance is computed. The user-facing API should accept either: `spillover_distance="dist_matrix"` (column-name pointing to a precomputed array) OR `spillover_coords=("lat", "lon")` (paired column names with on-the-fly Haversine).
- **Ring breakpoints**: a sorted list of distance bin edges; defaults must be context-dependent (Butts uses 50-mile bins for the TVA application but cautions this is application-specific).
- **`d-bar`**: scalar; defaults should default to `max(ring_breakpoints)`.
- Treatment indicator `D_it` and panel structure unchanged from base TWFE.

### Computational Considerations
- Construction of `Ring_{ij}` is O(n_treated * n) per period if computed via `min_{j: D_j=1} d(i, j)`. For panel data with time-varying treatment, recompute at each `t`.
- For large `n` (~10^5+), a k-d tree / ball tree on treated-unit coordinates yields `min_{j: D_j=1} d(i, j)` in O(n log n_treated) — Phase 2 sparse k-d-tree fast path (Conley spatial-HAC) can be reused.
- Once ring indicators are built, the augmented TWFE adds at most K + 1 columns (`D`, plus K ring interactions), so the Frisch-Waugh / partial-out workflow is O(n_obs * (K + 1)). Should be a thin wrapper around existing `TwoWayFixedEffects`.

### Tuning Parameters

| Parameter | Type | Default | Selection Method |
|-----------|------|---------|-----------------|
| `spillover_d_bar` | float | None (required) | User-supplied. Bias-variance trade-off (Butts pages 13-14); Butts (2021b) gives data-driven CV under stricter assumptions. Default could be the largest `ring_breakpoints` value. |
| `spillover_rings` | list of floats OR int | `[d_bar]` | User-supplied breakpoints, OR an int that auto-builds equal-width rings. Single-ring (Equation 5) by default; multi-ring (Equation 6) when list has 2+ values. |
| `spillover_distance` | str (column name) OR ndarray | None | Either precomputed nearest-treated distance per unit, OR a full distance matrix. |
| `spillover_coords` | tuple (str, str) | None | Pair of column names for on-the-fly Haversine if `spillover_distance` not supplied. |
| `spillover_ring_method` | "nearest" or "count" | "nearest" | "nearest": indicator per nearest-treated ring (Butts default). "count": count of treated units within each ring (re-introduces functional-form dependence; flagged in Section 3.2 end). |
| `vcov` | str | "conley" | Recommended. Cluster-by-unit also supported (Section 3.1). Cutoff defaults to `spillover_d_bar`. |

### Relation to Existing diff-diff Estimators
- **Builds on** `TwoWayFixedEffects`. The augmented model is just TWFE with extra ring-indicator columns interacted with `(1 - D_it)`.
- **Result object**: should expose `direct_effect = tau_total = tau-hat` (treatment dummy coefficient), `spillover_effects = {ring_label: delta_j-hat}` dict (or a results table), and `d_bar` actually used. Standard errors should attach Conley spatial-HAC by default.
- **Conley SE (Phase 1)** is a hard prerequisite — Butts explicitly recommends it.
- **Sparse k-d-tree (Phase 2)** is reusable for ring-indicator construction.
- **Staggered variant** can be added later as a wrapper around an existing two-stage / imputation estimator (e.g., `ImputationDiD` in the diff-diff catalog corresponds to Borusyak-Jaravel-Spiess 2021 / Gardner 2021); the interaction with spillover indicators is described in Table 2.

### Relation to Other Reviews in this Initiative
- **vs Clarke 2017 (clarke-2017-review.md, if produced):** Butts cites Clarke (2017) in his Section 1.1 as proposing a similar ring-method spec and referenced for the cross-validation technique for choosing the number of rings (footnote 15). Butts' contribution is the formal non-parametric **identification result** (Proposition 2.3) tying rings to a well-defined target estimand `tau_total`, plus the staggered extension (Section 5). If Phase 3 wants to support continuous-kernel exposure, that is closer to Clarke's stepwise `R^k(i,t) = 1{(k-1) h <= X_i < k h}` indicator construction OR to the spatial-decay-function suggestion in Butts' footnote 8 — but neither paper provides a full identification result for a continuous kernel `E_it = sum_j w_ij D_jt`.
- **vs Butts 2023 (butts-2023-review.md, the JUE Insight on geocoded microdata):** the JUE paper (Butts 2021b in this references list) is **complementary** to the present paper. The 2021b paper is the "ring method" paper proper — it adds a data-driven CV procedure for choosing the number / width of rings under a stricter parallel-trends assumption, in the single-treatment-point setting with continuous distance. The present paper (Butts 2021/2023, arXiv:2105.03737) is the **identification + multi-treated-unit + staggered** paper. They are cited together in published applications and in the `did2s` package documentation. Phase 3's REGISTRY entry should cite both.
- **vs Conley 1999 (conley-1999-review.md):** Butts (Section 3.1 page 13) explicitly recommends Conley spatial-HAC SEs in conjunction with the augmented TWFE, with cutoff equal to `d-bar`. He cites Ferman (2020) for the requirement that errors be uncorrelated past a finite cutoff for large-n asymptotics. The diff-diff Phase 1 Conley implementation should be the default `vcov` for SpilloverDiD.
- **vs Borusyak-Hull 2023 (design-based formula-instrument critique):** Butts does NOT directly anticipate the Borusyak-Hull (2023) critique (Borusyak-Hull is not in the references). However, Butts' framework is **model-based** (parallel trends on potential outcomes) rather than **design-based** (assumptions on treatment-assignment mechanism). The closest discussion is page 5: *"Those papers' identification results rely on design-based assumptions around the treatment-assignment mechanism, while this paper relies on model-based assumptions, based on a modified parallel-trends assumption, for identification in non-experimental settings."* So Butts is aware of the design-based / model-based dichotomy and positions his work in the model-based camp; whether the Borusyak-Hull critique applies to the ring-indicator spec is a separate question that the diff-diff REGISTRY entry should flag.

### Empirical illustration — Tennessee Valley Authority (Section 4)
- Application: revisits Kline and Moretti (2014) on the long-run effects of the 1934-WWII TVA federal investment program on agriculture and manufacturing employment.
- Data: county-level decadal outcomes 1940-2000 (long run, Panel A) and 1940-1960 (short run, Panel B). 5/4 column specification.
- Specification (Equation 8): first-differenced two-period DiD with rings at `{(0, 50], (50, 100], (100, 150], (150, 200]}` miles from the TVA boundary, X covariates from Kline and Moretti (1940 controls).
- SEs: Conley with 200-mile cutoff (Conley 1999).
- Findings (Table 1, Panel A long run):
  - Agriculture employment: standard DiD = -5.1% per decade. With spillovers controlled, total effect = -7.4% per decade. **Spillover bias = +2.3 pp; canonical DiD UNDERESTIMATED the agricultural decline by ~40%.** Spillover-on-control coefficients are negative (-3.7%, -1.6%, -3.0%, -1.6% per decade across the 4 distance bins) — consistent with farm-worker out-migration to higher-paying TVA manufacturing jobs.
  - Manufacturing employment: standard DiD = +5.6% per decade. With spillovers controlled, total effect = +3.5% per decade. **Spillover bias = +2.1 pp; canonical DiD OVERESTIMATED the manufacturing gain by ~40%.** Spillover coefficients are negative (-2.0%, -2.5%, -3.3%, -3.0%), consistent with "urban shadow" effects whereby firms relocate INTO the TVA from neighboring areas.
- Interpretation (page 21): "the long-run spillovers cause the original estimates to be about 40 percent too small for agriculture employment and 40 percent too large for manufacturing employment."
- Useful design choice for diff-diff T23 tutorial DGP: a TVA-style bias-correction percentage of ~40% is large enough to be visible without being implausible. (Tutorial slot 22 went to `22_had_survey_design.ipynb`; SpilloverDiD landed in slot 23 — `docs/tutorials/23_spillover_tva.ipynb`.)

### Empirical illustration — Opportunity Zones (Appendix B)
- Application: revisits Chen, Glaeser, and Wessel (2021) on the 2017 federal Opportunity Zone program's effect on home prices.
- Two competing identification strategies in the literature:
  - "Not-selected" (eligible-but-rejected as control, Equation B.1): Treat x Post = 0.30%* (s.e. 0.17%).
  - "Neighboring" (geographically nearby Census tracts as control, Equation B.2): Treat x Post = 0.65%*** (s.e. 0.25%).
- Augmented spec (Equation B.3) adds Within-1/2-mi and Within-1-mi indicators, recovering 0.18% (s.e. 0.17%) for treatment + -1.06%*** for <1/2 mi spillover and -0.74%*** for 1/2-1 mi spillover.
- Reconciles the two literatures: the neighboring spec is biased upward by negative spillovers on adjacent tracts. Page 31 footnote 25: upper-bound effect size lowered from ~1.15% to ~0.65%.

### Empirical illustration — Community Health Centers (Appendix C)
- Application: revisits Bailey and Goodman-Bacon (2015) on 1965-1974 federal community health center mortality effects.
- Uses the staggered two-stage estimator from Section 5 with `did2s`.
- Spillover indicator: within 25 miles of a treated county, time-varying.
- Result (Figure C1): "no spillover effect is estimated to be significantly different from zero which suggests that the effects of community health centers are very local. Since there are near zero spillover effects, the total effect estimates marked in Figure C1 as diamonds maintain the same shape as the author's original estimates with estimates between 15-30 fewer deaths per 100,000 persons" (page 34).
- Useful for the diff-diff tutorial as a NEGATIVE example (spillover NOT contaminating original estimate) — completes the asymmetry between the TVA case (large bias) and the CHC case (no bias).

### Critique / limitations Butts acknowledges
- "A limitation of this research is in deciding how wide and how many rings to include in estimation. Concurrent work by Butts (2021b) discusses data-driven ring selection under a more stringent parallel trends assumption that does not readily apply in the context of large geographic units such as counties." (page 25, Discussion section).
- Stronger parallel-trends assumption: Assumption 3 / 6 / 7 require parallel trends not just in absence of own treatment but in absence of all treatment AND zero exposure (page 7) — this is fundamentally untestable without stronger structure.
- Trade-off between `d-bar` choice and quality of the `S_i = 0` control group (page 12-14): a wider `d-bar` reduces spillover-bias but may worsen parallel trends on the remaining control group ("each value of `d-bar` corresponds to a different effective control group and hence a different parallel trends assumption").
- Identification for the "switching effect" `tau_switch(h-vector)` (Proposition 2.2, page 11) requires either parametrization of the spillover function or constraints on heterogeneity — Butts argues this is far harder than identifying `tau_total`.
- Counts-in-ring vs nearest-treated-ring: counts re-introduce functional-form dependence (page 16, end of Section 3); the no-functional-form claim of the paper depends on using nearest-treated rings.

---

## Gaps and Uncertainties

- **No continuous-kernel exposure regressor in the paper.** The Phase 3 brief assumes `E_it = sum_{j != i} w_ij D_jt` with a kernel `w_ij = K(d_ij / h)` is the canonical Butts spec. It is NOT — the paper uses ring/distance-bin indicators throughout. Footnote 8 (page 6) is the only reference to a "spatial decay function" exposure mapping, and it is illustrative of the abstract framework, not a proposed estimator. If Phase 3 wants the kernel form, the source must be either (a) Clarke (2017) MPRA, (b) the design-based / linear-in-means peer-effects literature (Manski 1993, Goldsmith-Pinkham and Imbens 2013), or (c) the Phase 3 plan should fold the kernel form into a future-work section while implementing the ring spec from Butts 2021.

- **Bandwidth selection guidance is sparse.** Butts (2021/2023) does not give a default `d-bar` or default ring breakpoints — these are contextual. He defers data-driven selection to Butts (2021b) under stronger parallel-trends. For Phase 3, the diff-diff API will need explicit user-supplied defaults (e.g., quantiles of the distance distribution, application examples).

- **Counts-in-ring vs nearest-treated-ring is under-specified.** Butts mentions both forms in passing but the algorithmic prescription is for the nearest-treated case. The Phase 3 API should either (i) support only nearest-treated rings and document the limitation, or (ii) support both and warn that counts re-impose functional-form constraints.

- **Equation (8) is two-period first-differenced**, not a panel TWFE. Phase 3 may want to support both: (a) panel + time FEs + ring indicators (Equation 6), and (b) two-period first-differenced + ring indicators (Equation 8). Butts uses (b) for the TVA application but states (a) is equivalent.

- **Ring construction across panel periods**: ring membership is computed at the unit-level in the two-period spec (Section 3) but at the unit-period level in the staggered spec (Section 5). The diff-diff implementation will need a clear convention for "is unit `i` within `d-bar` of a unit treated by time `t`?" vs "is unit `i` within `d-bar` of a unit ever treated?". Butts uses the time-varying definition for the staggered case (Table 2 caption / page 24).

- **No published code for the two-period ring estimator.** `did2s` covers the staggered two-stage estimator only. The two-period ring spec (Equations 5-6) is shown in worked applications but not packaged as a standalone routine — Phase 3 will need to implement it from scratch.

- **Borusyak-Hull (2023) connection**: not cited in this paper. The diff-diff REGISTRY caveats section should add a note on whether the ring-indicator approach inherits the formula-instrument / exposure-mapping concerns of Borusyak-Hull when researchers parametrize the rings or move to count-of-treated-neighbors.

- **Pre-trends testing protocol**: Butts does not give an explicit pre-trends test for the augmented spec, but the standard event-study extension is straightforward — replace `D_it` and `(1 - D_it) Ring_{it,j}` by their event-time interactions and inspect pre-period leads (analog of Section 5 Table 2 staggered approach). Worth flagging in the REGISTRY entry as the standard diagnostic.

- **Page-reference precision check.** Equation numbers preserved verbatim from PDF: (1)-(2) Proposition 2.1 page 9, (3) canonical TWFE page 9, (4) Proposition 2.3 page 12, (5) single-ring estimator page 14, (6) multi-ring estimator page 15, (7) Kline-Moretti baseline page 17, (8) TVA augmented spec page 19, (9) staggered parallel-trends page 23, (B.1)-(B.3) Opportunity Zones pages 31-32, (C.1) CHC event study page 32. Theorem / proposition numbers: Proposition 2.1, 2.2, 2.3 (no Theorem labels in this paper). Assumptions: 1-9 sequential.
