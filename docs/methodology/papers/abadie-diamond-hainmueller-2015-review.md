# Paper Review: Comparative Politics and the Synthetic Control Method

**Authors:** Alberto Abadie, Alexis Diamond, Jens Hainmueller
**Citation:** Abadie, A., Diamond, A., & Hainmueller, J. (2015). "Comparative Politics and the Synthetic Control Method." *American Journal of Political Science*, 59(2), 495–510.
**PDF reviewed:** https://doi.org/10.1111/ajps.12116 (published AJPS version)
**Review date:** 2026-05-29

> Scope note: this review captures only ADH (2015). The synthetic-control *estimator* itself (weights, `V`-matrix) is stated here but **attributed by the paper to Abadie & Gardeazabal (2003) and ADH (2010)**. This paper's own contributions are (a) the **diagnostics / robustness layer** — in-time placebos, leave-one-out donor removal, and the post/pre-RMSPE-ratio permutation test; (b) **out-of-sample cross-validation** for choosing the predictor-importance weights `V`; and (c) the **regression-vs-synthetic-control extrapolation** result. Results the paper merely *cites* to 2003/2010 are flagged as such; nothing here is sourced from outside this paper.

---

## Methodology Registry Entry

*Formatted to match docs/methodology/REGISTRY.md. This complements the ADH-2010 entry for `## SyntheticControl`; here the focus is the diagnostics layer, CV-based `V` selection, and the extrapolation result.*

## SyntheticControl

**Primary source (this document):** Abadie, A., Diamond, A., & Hainmueller, J. (2015). "Comparative Politics and the Synthetic Control Method." *AJPS*, 59(2), 495–510. https://doi.org/10.1111/ajps.12116

**Key implementation requirements:**

*Notation (Section "Synthetic Control Method", journal pp. 497–498):*
- `J+1` units; `j=1` treated; donors `j=2,…,J+1` (the "donor pool", `J` units). Balanced panel `t=1,…,T`; `T0` pre-periods, `T1` post-periods, `T=T0+T1`. No effect in `1,…,T0`.
- `Y_1` = `(T1×1)` post-period outcomes of the treated unit; `Y_0` = `(T1×J)` post-period outcomes of donors.
- `X_1` = `(k×1)` pre-intervention characteristics of the treated unit (**may include pre-intervention outcome values**); `X_0` = `(k×J)` for donors; row `m` = predictor `m`. Predictors are "not affected by the intervention."

*Weights and the simplex constraint (journal p. 497):*

    W = (w_2, …, w_{J+1})',   0 ≤ w_j ≤ 1,   Σ_{j=2}^{J+1} w_j = 1

*Weight optimization (Equation 1; attributed to AG 2003 / ADH 2010):*

    W* = argmin_W  Σ_{m=1}^{k}  v_m · (X_{1m} − X_{0m}·W)^2     s.t. simplex

where `v_m ≥ 0` reflects the predictive importance of predictor `m`. Seminorm form (footnote 5): `‖u‖ = sqrt(u'Vu)` for PSD `V`; with `V` diagonal `= diag(v_1,…,v_k)`, minimizing `‖X_1 − X_0 W‖` equals minimizing Equation (1). `W*` is **invariant to the scale of `(v_1,…,v_k)`**, so `V`'s diagonal can be normalized to sum to one.

*Estimator (journal p. 498):*

    τ̂_{1t} = Y_{1t} − Σ_{j=2}^{J+1} w_j*·Y_{jt}     (post periods);   vector form  Y_1 − Y_0·W*

*`V` selection — out-of-sample cross-validation (this paper's method; journal pp. 501–502):*
1. Split the pre-period into a **training** period and a **validation** period (application: training 1971–1980, validation 1981–1990).
2. For each candidate `V`, compute weights `W̃(V)` using **training-period** predictor data.
3. Choose `V*` to minimize the **RMSPE over the validation period**: `Σ_{t∈valid} (Y_{1t} − Σ_j w̃_j(V)·Y_{jt})²`.
4. Re-estimate `W* = W(V*)` using predictor data from the **last part of the pre-period** (validation-window predictors).
- **Footnote 17 (deviation note):** AG 2003 / ADH 2010 instead choose `V` so the synthetic control best fits the **pre-intervention outcome path**; for the German example this produces "almost identical" results to the CV method used here.

*Standard errors / inference (journal pp. 499–505):*
- **No standard errors, confidence intervals, or posterior distributions** (explicit, journal p. 500). Inference is restricted to "whether the estimated effect of the actual intervention is large relative to the distribution of placebo effects."
- **In-space placebo / permutation test:** apply SCM treating *each* donor as the pseudo-treated unit; build the distribution of placebo effects. p-value = **fraction of units whose (placebo) effect ≥ the treated unit's effect**; reduces to classical randomization inference under randomization (Rosenbaum 2005).
- **RMSPE-ratio test statistic (preferred):** with pre-period RMSPE (footnote 16)

      RMSPE = ( (1/T0) · Σ_{t=1}^{T0} ( Y_{1t} − Σ_{j=2}^{J+1} w_j*·Y_{jt} )^2 )^{1/2}

  compute `ratio = post-period RMSPE / pre-period RMSPE` for every unit; rank the treated unit's ratio. The ratio "avoid[s] having to discard countries with pre-period values that cannot be approximated" (footnote 19, crediting ADH 2010). Application: West Germany's ratio is the largest of 17 units → permutation p-value `1/17 ≈ 0.059`.
- **In-time placebo (this paper's diagnostic):** reassign the intervention to an earlier date in the pre-period, re-estimate the synthetic control with the **same CV technique and predictors lagged accordingly**, and check whether a spurious effect appears. The application reassigns reunification to **1975** (~15 yrs before the actual 1990 date — **Figure 4 is titled "Placebo Reunification 1975"**) and finds no perceptible placebo gap; **footnote 18** reports the same exercise reassigned to **1970 and 1980** ("similar to the results for 1975").
- **Leave-one-out / iterative donor removal (this paper's diagnostic):** re-estimate the synthetic control **omitting, one at a time, each donor that received positive weight**; overlay the leave-one-out counterfactual trajectories to gauge how much results depend on any single donor.

*Regression vs. synthetic control — extrapolation (journal pp. 498–499, 503):*
The regression-based counterfactual `B̂'X_1` with `B̂=(X_0 X_0')^{-1}X_0 Y_0'` equals `Y_0·W^{reg}` with

    W^{reg} = X_0'(X_0 X_0')^{-1} X_1

If a constant is included, `ι'W^{reg}=1` — i.e., regression is *also* a weighting estimator summing to one, but with **unrestricted weights** (can be negative or >1), so it **extrapolates** outside the donor convex hull. Detectable by computing `W^{reg}` and observing weights outside `[0,1]`. (In the application, regression assigned negative weights to Greece/Italy/Portugal/Spain.)

*Edge cases / practical guidance:*
- **Convex hull / no extrapolation:** the simplex constraint keeps the synthetic control inside the donors' convex hull (no model-dependent extrapolation; King & Zeng 2006 cited, journal p. 496).
- **Interpolation bias + non-uniqueness (footnote 10, journal p. 499):** even with no extrapolation, interpolation bias can be severe if donors have very different characteristics → **restrict the donor pool to similar units**; the `‖X_1−X_0W‖` objective can be **augmented with penalty terms** on discrepancies, which *also* help **select among multiple solutions** when `X_1` lies inside the convex hull (the objective then has non-unique minimizers).
- **Donor-pool curation (journal p. 500):** (1) exclude units affected by the same/similar intervention; (2) exclude units with large idiosyncratic shocks not shared by the treated unit; (3) restrict to units with characteristics similar to the treated unit (interpolation-bias control).
- **Overfitting (journal p. 500):** a large donor pool can artificially match the treated unit by combining idiosyncratic variation → restrict donor-pool size; motivates the CV `V`-selection.
- **Sparsity vs. fit (journal pp. 506–507):** synthetic controls are typically sparse; reducing to `l` contributing units (`l=4,3,2`) degrades fit "moderately"; `l=1` (single match) is much worse and "comes close to a difference-in-differences design" (footnote 23, citing ADH 2010 for the SC↔DiD relationship).
- **No interference among contributing donors:** spillovers onto donors *with positive weight* bias the estimate; spillovers onto zero-weight donors do not affect estimates (journal p. 504).

*Algorithm (in-time placebo + leave-one-out + RMSPE-ratio, reconstructed):*
1. Estimate the baseline synthetic control (weights `W*`, `V` via CV); record pre/post RMSPE and the gap path.
2. **In-space placebo:** for each donor `j`, treat `j` as pseudo-treated (donors = all other `J` units), re-estimate, record `r_j = postRMSPE_j/preRMSPE_j`. p-value = `(#{j: r_j ≥ r_1})/(J+1)`.
3. **In-time placebo:** re-estimate with the intervention date moved into the pre-period (lag predictors accordingly); confirm no spurious gap.
4. **Leave-one-out:** for each donor with `w_j*>0`, drop it, re-estimate, overlay trajectories.

**Reference implementation(s):**
- Authors' `Synth` package for **R, MATLAB, and Stata** (footnote 7). R: `Synth` (CRAN), documented in Abadie, Diamond & Hainmueller (2011), *J. Stat. Software* 42(13). Stata: `ssc install synth`.

**Requirements checklist (this paper's additions):**
- [ ] Out-of-sample cross-validation option for `V` (training/validation split), in addition to the pre-period-outcome-fit method.
- [ ] In-time placebo (date reassignment with predictor lagging).
- [ ] Leave-one-out donor robustness (drop each positively-weighted donor).
- [ ] Post/pre-RMSPE-ratio permutation p-value `(#{r_j ≥ r_1})/(J+1)`.
- [x] Regression-weight (`W^{reg}`) extrapolation diagnostic (flag weights outside `[0,1]`) — implemented as `SyntheticControlResults.regression_weights()`.
- [x] Sparse-SC subset search (`l<J` contributing units, `V` held fixed at the baseline, footnote 20) — implemented as `SyntheticControlResults.sparse_synthetic_control()`.
- [ ] Donor-pool curation + size limit (overfitting guard); optional penalty terms for interpolation bias / tie-breaking.

---

## Implementation Notes

### Data Structure Requirements
- Balanced panel, single treated unit, block treatment after `T0`; donor pool curated per the three rules above. Predictors `X` may mix covariates and pre-period outcomes (or pre-period outcome summaries).

### Computational Considerations
- The inner weight solve (Equation 1) is **constrained quadratic optimization** over the simplex (the paper calls it that; it does not specify the solver here).
- CV `V`-selection adds an outer search over `V` evaluated on a validation window.
- Inference loops: in-space placebo re-fits once per donor (`J` fits); leave-one-out re-fits once per positively-weighted donor; in-time placebo is a handful of re-fits at alternative dates. Sparse-SC subset search (`l<J`) is combinatorial — the paper holds `V` fixed at the baseline to limit cost (footnote 20).

### Tuning Parameters

| Parameter | Type | Default (this paper) | Selection Method |
|-----------|------|----------------------|------------------|
| `V` (predictor importance, diag) | nonneg vector | data-driven | **Out-of-sample CV** (training/validation split, this paper); alternative: pre-period outcome-path fit (AG 2003/ADH 2010, footnote 17) |
| Donor pool | set | curated | Exclude treated-like / shocked units; restrict to similar units; limit size (overfitting) |
| Predictors `X` | matrix | covariates + pre-period outcome summary | Analyst choice; summaries increase weight sparsity |
| In-time placebo date | period | application-specific | Paper's example reassigns to **1975** (Figure 4, "Placebo Reunification 1975"); footnote 18 repeats for 1970 and 1980 |

### Relation to Existing diff-diff Estimators
- Same `SyntheticControl` estimator as the ADH-2010 review; this paper adds the **diagnostics suite** (PR-2 in the implementation plan) and the **CV `V`-selection** option.
- The **leave-one-out** and **in-space-placebo** loops resemble (in spirit) `diff_diff/diagnostics.py` `leave_one_out_test` / `permutation_test`, but the **RMSPE-ratio statistic** and the `(#{r_j ≥ r_1})/(J+1)` p-value are specific here.
- The `l=1` single-match reducing toward DiD echoes Equation (1)'s factor-model generalization of DiD noted in the 2010 review.

---

## Gaps and Uncertainties

- **Inner-solver and `V`-search numerics are not given here.** Equation (1) is called "constrained quadratic optimization" but no algorithm/solver, starting values, or `V`-grid is specified — these are referenced to AG 2003 / ADH 2010 and the `Synth` software.
- **CV split lengths are application-specific.** The 1971–1980 / 1981–1990 split is illustrative; the paper gives no general rule for the training/validation boundary (cf. Abadie 2021, which formalizes `t0 = T0/2` as a concrete-but-heuristic default — captured in that review).
- **Penalty terms** for interpolation bias / tie-breaking are recommended (footnote 10) but **not formalized** (no penalty functional or weight given) — anticipates penalized-SC work, out of scope here.
- **CV-weight non-uniqueness:** the paper notes (footnote 10) that minimizing `‖X_1−X_0W‖` can have multiple solutions when `X_1` is interior to the convex hull; it suggests penalties but does not prescribe a canonical selection — an implementation must choose a deterministic tie-break.
- **p-value granularity:** as in 2010, the permutation p-value is `rank/(J+1)`; the smallest attainable here is `1/17 ≈ 0.059` (16-donor pool). No CIs.
- **In-time placebo predictor lagging** ("lag the predictor variables accordingly") is described qualitatively; the exact re-indexing of training/validation windows for an arbitrary placebo date needs a concrete convention at implementation.
