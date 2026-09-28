# Paper Review: Regression Discontinuity Designs Using Covariates

**Authors:** Sebastian Calonico, Matias D. Cattaneo, Max H. Farrell, and Rocío Titiunik
**Citation:** Calonico, S., Cattaneo, M. D., Farrell, M. H., & Titiunik, R. (2019). Regression Discontinuity Designs Using Covariates. *The Review of Economics and Statistics*, 101(3), 442-451. DOI: 10.1162/rest_a_00760
**PDF reviewed:** papers/Calonico-Cattaneo-Farrell-Titiunik_2019_RESTAT.pdf
**Review date:** 2026-07-12

---

## Methodology Registry Entry

*Formatted to match docs/methodology/REGISTRY.md structure. This covers the covariate-adjustment extension of the RD estimator (a fast-follow after sharp-RD v1); merge into the RegressionDiscontinuity registry section when covariates ship.*

## RegressionDiscontinuity (covariate adjustment)

**Primary source:** Calonico, S., Cattaneo, M. D., Farrell, M. H., & Titiunik, R. (2019). Regression Discontinuity Designs Using Covariates. *REStat*, 101(3), 442-451. https://doi.org/10.1162/rest_a_00760

**Key implementation requirements:**

*Assumption checks / warnings:*
- **Assumption SRD** (Section III.A), for `t in {0,1}` and all `x` in `[x_l, x_u]` around the cutoff: (a) density `f(x)` of `X_i` continuous and bounded away from zero; (b) `mu_Y-(x) = E[Y_i(0)|X_i=x]`, `mu_Y+(x) = E[Y_i(1)|X_i=x]` thrice continuously differentiable; (c) `mu_Z-(x) = E[Z_i(0)|X_i=x]`, `mu_Z+(x) = E[Z_i(1)|X_i=x]` thrice continuously differentiable and `E[Y_i(t) Z_i(t)'|X_i=x]` continuously differentiable; (d) `V[(Y_i(t), Z_i(t)')'|X_i=x]` continuously differentiable and **invertible**; (e) fourth moments `E[|(Y_i(t), Z_i(t)')|^4 | X_i=x]` continuous.
- **Covariate balance at the cutoff is the operative identification condition**: Lemma 1 gives the exact probability limit `tau_tilde ->_p tau - [mu_Z+ - mu_Z-]' gamma_Y`, and the paper's stated **sufficient** condition for consistency is zero RD treatment effect on the covariates, `tau_Z := mu_Z+ - mu_Z- = 0` ("It follows that a sufficient condition...", Section III / Lemma 1 discussion). (Algebraically, consistency would also survive an imbalance orthogonal to `gamma_Y`, but that is not an interpretable design condition - balance is the condition the paper recommends and tests.) Balance is *weaker* than requiring equal marginal distributions of `Z_i(0)`, `Z_i(1)` near the cutoff. It is testable - the paper ran "placebo tests" on the covariates (Section V.A) - so the implementation should offer/encourage a covariate-balance diagnostic (each covariate as outcome in a sharp RD) and warn when it rejects.
- **Invertibility of the local covariate Gram matrix** (Assumption SRD(d) sample analogue): rank-check the joint design; collinear covariates or covariates constant near the cutoff must error clearly.
- **Kink designs need more**: first-derivative balance `mu^(1)_Z+ = mu^(1)_Z-` (Section III end; details in supplemental appendix). Warn if covariate-adjusted kink is ever supported.

*Estimator equation (Equations 1-2; the paper's recommended specification):*

Standard (unadjusted) local-linear RD estimator, observations `X_i in [-h, h]`, weights `K(X_i/h)`:

    tau_hat :  Y_i = alpha_hat + T_i tau_hat + X_i beta_hat_- + T_i X_i beta_hat_+        (Equation 1)

**Covariate-adjusted RD estimator** (Equation 2) - one joint weighted least squares fit with covariates entering additively with a **common coefficient across sides** (no `T_i Z_i` interaction):

    tau_tilde :  Y_i = alpha_tilde + T_i tau_tilde + X_i beta_tilde_- + T_i X_i beta_tilde_+ + Z_i' gamma_tilde   (Equation 2)

Five specifications are analyzed (Equations 2-6: additive; treatment-interacted; common demeaning; demeaned+interacted; group-demeaned+interacted). **Lemma 1** (Sharp RD with covariates) derives all five probability limits:
- `tau_tilde ->_p tau - [mu_Z+ - mu_Z-]' gamma_Y` -> the recommended sufficient (and testable) condition for consistency is balance, `mu_Z+ = mu_Z-`, with `gamma_Y = (sigma^2_Z- + sigma^2_Z+)^{-1} E[(Z_i(0) - mu_Z-(X_i)) Y_i(0) + (Z_i(1) - mu_Z+(X_i)) Y_i(1) | X_i = x_bar]`.
- The interacted version `tau_check` (Equation 3, equivalent to two separate per-side fits with covariates): the paper's stated necessary-and-sufficient condition for `tau_check ->_p tau` is the product equality `mu'_Z+ gamma_Y+ = mu'_Z- gamma_Y-`; the interpretable sufficient route is balance PLUS `gamma_Y+ = gamma_Y-` - strictly stronger and "harder to justify in practice" than the no-interaction condition (Section III). Do NOT default to per-side covariate coefficients.
- The three demeaning-based estimators (Equations 4-6) implicitly kernel-regress covariate means with a uniform kernel and are "severely affected: these estimators exhibit slower convergence rates, new misspecification biases, and additional asymptotic variability" (Section III end). Do not implement.
- Covariates may be continuous, discrete, or mixed; linearity is a *local linear projection*, not a functional-form assumption - "we allow for complete misspecification of `E[Y_i(t)|X_i, Z_i(t)]` in any finite sample" (Section III.A). No extra bandwidths or kernels over `Z` (avoids the curse of dimensionality of the localize-then-smooth approach of Frolich and Huber 2018).

*MSE expansion and bandwidth (Theorem 1, Section IV.A):*

    MSE[tau_tilde(h)] = h^4 B_tautilde(h)^2 {1 + o_P(1)} + (1/(nh)) V_tautilde(h)
    h_tautilde = [ (V_tautilde / n) / (4 B_tautilde^2) ]^{1/5}

- Preasymptotic (fixed-n) bias and variance forms; exact expressions in the supplemental appendix. `B = B_+ - B_-`, `V = V_- + V_+` per side. Bias/variance formulas *differ* from the no-covariate case (Imbens-Kalyanaraman 2012; CCT 2014; Arai-Ichimura 2018), so covariate-aware bandwidth selection is required - selecting `h` without covariates and then adding covariates is not MSE-optimal.
- The convergence rate is unaffected by `d = dim(Z)` (no nonparametric smoothing over covariates); optimal bandwidth decays as `n^{-1/(3+2p)}` for generic degree `p >= 1`.
- Feasible plug-in selectors `h_tilde = [(V_hat(v)/n)/(4 B_hat(b)^2)]^{1/5}` with pilot bandwidths; consistent, `h_tilde/h ->_p 1`. Per-side, sum-form, and regularized variants in the supplemental appendix.
- Asymptotic representation used throughout (Section IV): `tau_tilde(h) - tau = s(h)'[tau_hat(h) - tau; tau_hat_Z(h)] {1 + o_P(1)}` with `s(h) = (1, -gamma_tilde_Y(h)')'` - i.e., the covariate-adjusted estimator is the unadjusted RD estimator on `Y` minus a `gamma`-weighted combination of RD estimators on each covariate (`tau_hat_Z` = vector of standard RD estimators with each covariate as outcome). This partial-out representation is a good implementation/testing identity.

*Efficiency (Section IV.B):*

    V_tautilde / V_tauhat = ( V[Y(0) - Z(0)'gamma_Y | X = x_bar] + V[Y(1) - Z(1)'gamma_Y | X = x_bar] )
                          / ( V[Y(0) | X = x_bar] + V[Y(1) | X = x_bar] )

Efficiency gains are guaranteed when `gamma_Y = gamma_Y- = gamma_Y+` (the best linear effect of `Z` on `Y` at the cutoff is equal across groups); in general no definitive ranking. Parallels the Lin (2013)/Freedman (2008) covariate-adjustment results for randomized experiments. Practical upshot quoted in Section V: point estimates should be stable; CIs shrink (Head Start example: ~10% CI length reduction).

*Robust bias-corrected inference (Section IV.C, Equation 7, Theorem 2):*

    tau_tilde^bc(h, b) = tau_tilde(h) - h^2 * B_tilde_tautilde(b)        (Equation 7)

with the bias estimator from a higher-order (local quadratic) covariate-adjusted fit at pilot bandwidth `b` (`b = h` is an "empirically useful choice," allowed by the theory, citing CCT 2014 Remark 7 and CCF 2018/2019). Fixed-n robust variance:

    V^bc_tautilde(h,b) = [s' ⊗ P^bc_-(h,b)] SIGMA_S- [s' ⊗ P^bc_-(h,b)]' + [s' ⊗ P^bc_+(h,b)] SIGMA_S+ [s' ⊗ P^bc_+(h,b)]'

where `P^bc_+/-` are computable from the data and the only unknowns are the `n(1+d) x n(1+d)` variance-covariance matrices `SIGMA_S-`, `SIGMA_S+` (joint over `(Y, Z')`) - estimated by **nearest-neighbor (NN) or plug-in residual (PR)** approaches "covering both conditional heteroskedasticity and clustered data" (exact formulas in the supplemental appendix). **Theorem 2** (asymptotic normality): if `n h^7 -> 0` and `limsup(h/b) < infinity` and `tau_Z = 0`, then `(tau_tilde^bc - tau)/sqrt((nh)^{-1} V^bc) ->_d N(0,1)` and the feasible variance is consistent. Valid with the MSE-optimal bandwidth. 95% CI (common `h = b`): `tau_tilde^bc ± 1.96 sqrt(V_hat^bc/(nh))`.

*CER-optimal bandwidth (Section IV.C):* `h_hat_CER,tautilde = n^{-1/20} * h_hat_tautilde` for `p = 1` (generic-`p` rate scaling in the supplemental appendix; cites CCF 2018/2019 Edgeworth expansions). Preferred when the goal is inference.

*Clustered data (Section IV.C end):* Theorems 1-2 remain valid under clustered sampling with (a) each unit in exactly one of `G` clusters, (b) `G -> infinity` and `G h -> infinity` (cites Cameron-Miller 2015; Bartalotti-Brummet 2017). Variance formulas change with the clustering form; "conceptually straightforward but notationally cumbersome," deferred to the supplemental appendix. Companion software includes cluster-robust bandwidth selection, MSE-optimal estimation, and robust bias-corrected inference.

*Edge cases:*
- Covariate imbalance at the cutoff (`tau_Z != 0`): the adjusted estimator is generically inconsistent for `tau` (plim shifted by `[mu_Z+ - mu_Z-]' gamma_Y`, Lemma 1); "adjusting for imbalanced covariates in order to restore identification is not possible without functional form assumptions" (Section II). Detection: RD placebo test on each covariate -> warn, do not "fix."
- Irrelevant covariates: harmless - "including an irrelevant covariate hardly changes empirical results and conclusions" (Section V.B, model 1).
- Largest gains when residual outcome-covariate correlation at the cutoff is strong (Section V.B models 2/4); zero residual correlation (model 3) -> no gain, as theory predicts.
- Treatment-interacted or demeaned specifications requested by users: not supported / documented as inconsistent-or-inferior per Lemma 1.

*Algorithm (covariate-adjusted sharp RD, assembled from Sections III-IV):*
1. Run the covariate-balance placebo RDs on each `Z` column (optional but recommended diagnostic).
2. Select covariate-aware MSE-optimal `h` (and pilot `b`) via the Theorem 1 plug-in constants (or CER-rescale for inference-first use).
3. One joint WLS on `[-h, h]`: `Y ~ 1 + T + X + T:X + Z` with kernel weights; `tau_tilde` = coefficient on `T`.
4. Bias-correct with the order-(p+1) covariate-adjusted fit at `b` (Equation 7).
5. Robust variance via the `s' ⊗ P^bc` sandwich with NN (default) or PR residuals on the joint `(Y, Z)` system; normal-quantile CIs (Theorem 2).

**Reference implementation(s):**
- R: `rdrobust::rdrobust(y, x, covs = Z)`, `rdrobust::rdbwselect(..., covs = Z)` (companion software, Calonico et al. 2017; replication files at https://sites.google.com/site/rdpackages/replication)
- Stata: `rdrobust ..., covs(z1 z2 ...)`

**Requirements checklist:**
- [x] `covariates=` accepting continuous/discrete/mixed columns; additive-with-common-gamma specification ONLY (Equation 2)
- [x] Covariate-aware MSE-optimal bandwidth constants (NOT the no-covariate constants with covariates bolted on)
- [x] Joint `(Y, Z)` NN / plug-in-residual variance for the `s' ⊗ P^bc` sandwich (heteroskedastic NN form; cluster variance remains a documented v1 seam alongside the RD estimator's other cluster paths)
- [x] Partial-out identity test: `tau_tilde = tau_hat - gamma_tilde' tau_hat_Z` (up to the WLS algebra) as an internal consistency check (`tests/test_rdd_methodology.py::TestCovariates::test_partial_out_identity_exact` - exact at common manual (h, b), both conventional and bias-corrected rows)
- [x] Covariate balance placebo diagnostic + warning on rejection - the RECIPE is documented (module docstring + REGISTRY: fit each covariate as `outcome`); a packaged `covariate_balance` helper with automatic warning stays a named follow-up (diagnostics wave), matching rdrobust's scope (R does not auto-test balance either)
- [x] CER rescaling `n^{-1/20}` (p = 1) applies unchanged to the covariate-adjusted bandwidth (the `cer*` selectors rescale the covariate-aware MSE `h`; `covs_cercomb2` golden config)
- [ ] Head Start numbers as parity smoke test: standard `tau_hat = -2.41` (h = 6.81, b = 10.72, n- = 234, n+ = 180); covariate-adjusted with covariate-aware bandwidths `tau_tilde = -2.47`, robust 95% CI `[-5.21, -0.37]`, h = 6.98, b = 11.64, n- = 240, n+ = 184 (Table 1; triangular kernel, NN het-robust variance, 9 Census covariates) - NOT shipped: needs the external replication dataset; the library's parity policy prefers live-R end-to-end goldens (9 covariate configs vs installed rdrobust 4.0.0) over published-number replication

---

## Implementation Notes

### Data Structure Requirements
- Cross-section: outcome `Y`, running variable `X`, cutoff `c`, plus a covariate matrix `Z` (n x d), continuous/discrete/mixed, no transformations imposed (users may pass expansions/interactions themselves).
- Fuzzy and kink extensions exist in the supplemental appendix only (main text is sharp local-linear).

### Computational Considerations
- The joint WLS is `(4 + d)`-column (sharp local-linear case: constant, T, X, TX, Z) on the in-bandwidth sample - still trivial.
- The robust variance needs joint residual covariance across `(Y, Z)` - the `SIGMA_S` blocks - so the NN machinery must return residual *products* across outcome and covariates, not just squared outcome residuals (cross-covariance analogue of what CCT 2014 Section 5 already defines for fuzzy designs).
- Partial-out representation gives an O(d) way to sanity-check the point estimate from d+1 unadjusted RD fits.

### Tuning Parameters

| Parameter | Type | Default | Selection Method |
|-----------|------|---------|-----------------|
| `covariates` | matrix | none | User-supplied; additive common-gamma only (Equation 2 / Lemma 1) |
| `h`, `b` | float | covariate-aware MSE-optimal | Theorem 1 plug-ins; `b = h` allowed (Section IV.C) |
| CER option | flag | off | `h_CER = n^{-1/20} h_MSE` for p = 1 (Section IV.C) |
| `vce` | str | NN | NN or plug-in residuals, joint over `(Y, Z)`; cluster variants (supplemental appendix) |
| `p` | int | 1 | Main text is local-linear; any `p >= 1` in supplemental appendix |

### Relation to Existing diff-diff Estimators
- Extends the CCT 2014 sharp RD machinery (same fixed-n RBC template the `_nprobust_port` implements) with a wider design matrix and a joint-outcome variance; the partial-out representation means the covariate path can reuse the unadjusted RD internals for testing.
- The covariate-balance placebo diagnostic is itself just the sharp RD estimator applied to each covariate - free once v1 ships.
- Deferral note for v1 (sharp, no covariates): this paper documents exactly what `rdrobust`'s `covs()` option computes, so the v1 REGISTRY entry should mark covariates as a documented follow-up citing this review.

---

## Gaps and Uncertainties

1. **All exact constants live in the supplemental appendix** (not reviewed): precise `B_tautilde`/`V_tautilde` expressions, feasible plug-in construction, per-side/sum/regularized bandwidth variants, NN and PR variance formulas for the joint system, cluster-robust forms, fuzzy/kink extensions, generic-`p` results, and the kink covariate-balance specification tests. The supplemental appendix is at http://www.mitpressjournals.org/doi/suppl/10.1162/rest_a_00760 - fetch before implementing covariates; the rdrobust R source is the operational reference meanwhile.
2. **`gamma_Y` in Lemma 1**: the displayed formula (page 445) sets `gamma_Y` as the inverse of the *sum* of the two one-sided covariate variance matrices times a *sum* of one-sided cross moments; transcription verified visually but re-check against the supplemental appendix before using it in code or docs (dense notation, small type).
3. **Equation-3 estimator equivalence**: the paper states the interacted specification "corresponds to fitting two separate weighted linear regressions on each side" (Section III) - i.e., what a naive per-side implementation would produce. Worth a unit test asserting our implementation does NOT equal that under covariate adjustment.
4. **Clustered asymptotics conditions** (`G -> infinity`, `Gh -> infinity`) are stated but the variance estimator itself is deferred (Section IV.C end); parity-target rdrobust's `vce(nncluster ...)` behavior.
5. Table 1 transcription (Head Start numbers above) re-verified once; verify against replication output when the smoke test is written (the replication code is public at the rdpackages site).
