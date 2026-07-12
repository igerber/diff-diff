# Paper Review: Robust Nonparametric Confidence Intervals for Regression-Discontinuity Designs

**Authors:** Sebastian Calonico, Matias D. Cattaneo, and Rocio Titiunik
**Citation:** Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). Robust Nonparametric Confidence Intervals for Regression-Discontinuity Designs. *Econometrica*, 82(6), 2295-2326. DOI: 10.3982/ECTA11757
**PDF reviewed:** papers/Calonico-Cattaneo-Titiunik_2014_ECMA.pdf
**Review date:** 2026-07-12

---

## Methodology Registry Entry

*Formatted to match docs/methodology/REGISTRY.md structure. Heading levels and labels align with existing entries - copy the `## {EstimatorName}` section into the appropriate category in the registry.*

## RegressionDiscontinuity

**Primary source:** Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). Robust Nonparametric Confidence Intervals for Regression-Discontinuity Designs. *Econometrica*, 82(6), 2295-2326. https://doi.org/10.3982/ECTA11757

**Key implementation requirements:**

*Assumption checks / warnings:*
- **Continuous running variable** (Assumption 1(a)): `X_i` must have a continuous density `f(x)` bounded away from zero near the cutoff. Remark 1 explicitly rules out discrete running variables: with many mass points near the cutoff the results "may still give a good approximation," but with few mass points "our results do not apply directly" and other methods (local randomization, Cattaneo-Frandsen-Titiunik 2014) are more appropriate. Implementation should detect mass points in `X` near the cutoff and warn.
- **Smoothness** (Assumption 1(b)): `mu_-(x) = E[Y_i(0)|X_i=x]` and `mu_+(x) = E[Y_i(1)|X_i=x]` must be `S` times continuously differentiable in a neighborhood `(-kappa_0, kappa_0)` of the cutoff. Theorem 1 (local-linear point estimate, local-quadratic bias correction) requires `S >= 3`. Not directly checkable; document as an identifying assumption.
- **Moments and variances** (Assumption 1(a), (c)): `E[Y_i^4|X_i=x]` bounded; conditional variances `sigma_+^2(x)`, `sigma_-^2(x)` continuous and bounded away from zero (they may differ across sides).
- **Kernel regularity** (Assumption 2): kernel `k(.): [0, kappa] -> R` bounded, nonnegative, zero outside its support, positive and continuous on `(0, kappa)`. Triangular `k(u) = (1-u)1(0 <= u <= 1)` and uniform kernels both satisfy this. The paper uses the same kernel on both sides: `K(u) = k(-u)1(u < 0) + k(u)1(u >= 0)`.
- **Bandwidth positivity / sample support**: each side of the cutoff must have enough observations inside its bandwidth window to identify the local polynomial (at least `p + 1` distinct support points per side; rank checks on the weighted design matrix).
- **Fuzzy designs only** (Theorem 3): `tau_T,SRD != 0` (a discontinuity in treatment take-up must exist); implementation must guard the denominator.

*Estimator equation (Section 2.1, sharp RD, local-linear case):*

The sharp RD estimand is the average treatment effect at the cutoff (`x_bar = 0` WLOG):

    tau_SRD = E[Y_i(1) - Y_i(0) | X_i = x_bar] = mu_+ - mu_-,
    mu_+ = lim_{x->0+} mu(x),  mu_- = lim_{x->0-} mu(x),  mu(x) = E[Y_i | X_i = x]

(nonparametric identification per Hahn, Todd, and van der Klaauw 2001, cited in Section 2). The local-linear point estimator with bandwidth `h_n` is the difference in intercepts of two one-sided kernel-weighted regressions:

    tau_hat_SRD(h_n) = mu_hat_{+,1}(h_n) - mu_hat_{-,1}(h_n)

    (mu_hat_{+,1}(h_n), mu_hat^(1)_{+,1}(h_n))' = argmin_{b0,b1} sum_i 1(X_i >= 0)(Y_i - b0 - X_i b1)^2 K(X_i / h_n)
    (mu_hat_{-,1}(h_n), mu_hat^(1)_{-,1}(h_n))' = argmin_{b0,b1} sum_i 1(X_i <  0)(Y_i - b0 - X_i b1)^2 K(X_i / h_n)

*General p-th order form (Appendix A.1):* for derivative order `nu <= p`,

    mu_hat^(nu)_{+,p}(h_n) = nu! e_nu' beta_hat_{+,p}(h_n),
    beta_hat_{+,p}(h_n) = argmin_{beta in R^{p+1}} sum_i 1(X_i >= 0)(Y_i - r_p(X_i)'beta)^2 K_{h_n}(X_i)

with `r_p(x) = (1, x, ..., x^p)'`, `K_h(u) = K(u/h)/h`, and closed form `beta_hat_{+,p}(h_n) = H_p(h_n) Gamma^{-1}_{+,p}(h_n) X_p(h_n)' W_+(h_n) Y / n`, `H_p(h) = diag(1, h^{-1}, ..., h^{-p})`. The estimand family is `tau_nu = mu_+^(nu) - mu_-^(nu)`; `tau_SRD = tau_0` (levels, `p = 1` recommended), `tau_SKRD = tau_1` (kink, `p = 2` recommended).

*Bias correction (Section 2.2):* the leading conditional bias is

    E[tau_hat_SRD(h_n) | X_n] - tau_SRD = h_n^2 B_SRD(h_n) {1 + o_p(1)},
    B_SRD(h_n) = (mu_+^(2)/2!) B_{+,SRD}(h_n) - (mu_-^(2)/2!) B_{-,SRD}(h_n)

where `B_{+/-,SRD}(h_n)` are observed quantities (Lemma A.1(B): `B_{+/-,nu,p,r}(h) = nu! e_nu' Gamma^{-1}_{+/-,p}(h) theta_{+/-,p,r}(h)`). The plug-in bias-corrected estimator uses local-quadratic (`q = 2`) curvature estimates at pilot bandwidth `b_n`:

    tau_hat^bc_SRD(h_n, b_n) = tau_hat_SRD(h_n) - h_n^2 * B_hat_SRD(h_n, b_n),
    B_hat_SRD(h_n, b_n) = (mu_hat^(2)_{+,2}(b_n)/2!) B_{+,SRD}(h_n) - (mu_hat^(2)_{-,2}(b_n)/2!) B_{-,SRD}(h_n)

General `q`-th order form (Appendix, before Theorem A.1): `tau_hat^bc_{nu,p,q}(h_n, b_n) = tau_hat_{nu,p}(h_n) - h_n^{p+1-nu} B_hat_{nu,p,q}(h_n, b_n)`.

*Robust bias-corrected inference (Theorem 1 - the paper's central result):*

Let `S >= 3`. If `n min{h_n^5, b_n^5} max{h_n^2, b_n^2} -> 0` and `n min{h_n, b_n} -> infinity`, then

    T^rbc_SRD(h_n, b_n) = (tau_hat^bc_SRD(h_n, b_n) - tau_SRD) / sqrt(V^bc_SRD(h_n, b_n)) ->_d N(0, 1),
    V^bc_SRD(h_n, b_n) = V_SRD(h_n) + C^bc_SRD(h_n, b_n)

provided `kappa max{h_n, b_n} < kappa_0`. The robust variance **adds a correction term** `C^bc_SRD` capturing the variability of the estimated bias and its covariance with the point estimate. The robust 100(1-alpha)% CI is

    I^rbc_SRD(h_n, b_n) = [ tau_hat^bc_SRD(h_n, b_n) +/- Phi^{-1}_{1-alpha/2} sqrt(V_SRD(h_n) + C^bc_SRD(h_n, b_n)) ]

Crucially this remains valid when `rho_n = h_n / b_n -> rho in [0, infinity]` - including `b_n = h_n` - whereas conventional bias-corrected inference requires `h_n / b_n -> 0` (Section 2.2, Remark 2, Remark 3). This is what makes MSE-optimal bandwidths usable for inference.

*Exact variance decomposition (Theorem A.1(V)):* `V^bc_{nu,p,q}(h_n, b_n) = V^bc_{+,nu,p,q}(h_n, b_n) + V^bc_{-,nu,p,q}(h_n, b_n)` with (per side; `+` shown)

    V^bc_{+,nu,p,q}(h, b) = V_{+,nu,p}(h)
                            - 2 h^{p+1-nu} C_{+,nu,p,q}(h, b) B_{+,nu,p,p+1}(h) / (p+1)!
                            + h^{2(p+1-nu)} V_{+,p+1,q}(b) B^2_{+,nu,p,p+1}(h) / ((p+1)!)^2

    V_{+,nu,p}(h)   = n^{-1} h^{-2nu} nu!^2 e_nu' Gamma^{-1}_{+,p}(h) Psi_{+,p}(h) Gamma^{-1}_{+,p}(h) e_nu        [Lemma A.1(V)]
    C_{+,nu,p,q}(h, b) = n^{-1} h^{-nu} b^{-p-1} nu! (p+1)! e_nu' Gamma^{-1}_{+,p}(h) Psi_{+,p,q}(h, b) Gamma^{-1}_{+,q}(b) e_{p+1}   [Theorem A.1(V)]

In Theorem 1's notation: `V^bc_SRD(h_n,b_n) = V^bc_{0,1,2}(h_n,b_n)`, `V_SRD(h_n) = V_{+,0,1}(h_n) + V_{-,0,1}(h_n)`, and `C^bc_SRD = V^bc_{0,1,2} - V_{0,1}` (stated after Theorem A.1).

*Standard errors (Section 5):*
- All variance formulas are sandwich forms; the only unknowns are the middle matrices (generalizations of the Huber-Eicker-White meat):

      Psi_{UV+,p,q}(h_n, b_n) = (1/n) sum_i 1(X_i >= 0) K_{h_n}(X_i) K_{b_n}(X_i) r_p(X_i/h_n) r_q(X_i/b_n)' sigma^2_{UV+}(X_i)

  (and the `-` analogue), where `sigma^2_{UV+}(x) = Cov[U(1), V(1) | X = x]`, with `U, V` placeholders for `Y` (and `T` in fuzzy designs).
- **Default (recommended): nearest-neighbor (NN) variance estimator** (Section 5, following Abadie and Imbens 2006):

      sigma_hat^2_{UV+}(X_i) = 1(X_i >= 0) * (J/(J+1)) * (U_i - (1/J) sum_{j=1}^J U_{l+_j(i)}) (V_i - (1/J) sum_{j=1}^J V_{l+_j(i)})

  where `l+_j(i)` is the j-th closest unit to `i` *within the same side* `{X_i >= 0}` and `J` is the fixed number of neighbors (simulations use `J = 3`). Valid for any fixed `J in N_+`: approximately conditionally unbiased, though inconsistent for fixed `J` (Supplemental Material S.2.4). Feasible CI: plug `Psi_hat` matrices into `V_hat^bc_SRD`; for sharp RD this requires the 8 matrices `Psi_hat_{YY+,1,1}, Psi_hat_{YY+,1,2}, Psi_hat_{YY+,2,1}, Psi_hat_{YY+,2,2}` and their `-` analogues (page 2314).
- Alternative: "plug-in estimated residuals" from the local polynomial fits themselves (Supplemental Material S.2.4); the paper warns it "may not perform well in finite samples because it implicitly employs the bandwidth choices used to construct the estimates" and simulations confirm more undercoverage (Section 6).
- Bootstrap: not recommended as a fix for conventional CIs - "Bootstrapping tau_hat_SRD(h_n) or T_SRD(h_n) will not improve the performance of the conventional confidence intervals because the bootstrap distribution is centered at E[tau_hat_SRD(h_n)|X_n]" (Remark 6). Bootstrapping the asymptotically pivotal `T^rbc_SRD` is possible as an alternative to the Gaussian approximation.
- Clustering: not covered in this paper (i.i.d. random sampling assumed).

*MSE-optimal bandwidths (Section 4, Lemma 1):*

For the generic estimand/estimator pair `(nu, p)` with `s` indexing sum-vs-difference of one-sided biases:

    MSE_{nu,p,s}(h) = h^{2(p+1-nu)} [B^2_{nu,p,p+1,s} + o_p(1)] + (1/(n h^{1+2nu})) [V_{nu,p} + o_p(1)]

    h_MSE,nu,p,s = C^{1/(2p+3)}_{MSE,nu,p,s} n^{-1/(2p+3)},   C_MSE,nu,p,s = (1 + 2nu) V_{nu,p} / (2(p + 1 - nu) B^2_{nu,p,p+1,s})

with `V_{nu,p} = (sigma_-^2 + sigma_+^2) nu!^2 e_nu' Gamma_p^{-1} Psi_p Gamma_p^{-1} e_nu / f` and `B_{nu,p,r,s} = ((mu_+^(r) - (-1)^{nu+r+s} mu_-^(r)) / r!) nu! e_nu' Gamma_p^{-1} theta_{p,r}` (Section 4.1), where `Gamma_p = int_0^inf K r_p r_p'`, `theta_{p,q} = int_0^inf K u^q r_p`, `Psi_p = int_0^inf K^2 r_p r_p'`. For sharp RD inference per Theorem 1: main bandwidth `h_n = h_MSE,0,1,0` (for `tau_hat_SRD`) and pilot `b_n = h_MSE,2,2,2` (for the bias estimate). These MSE-optimal rates violate the conventional CI's bias condition but are **fully compatible with the robust CIs** (Remark 10). Feasible direct plug-in (DPI) selectors are developed in the Supplemental Material (S.2.6, cited via Remark 11); they differ from Imbens-Kalyanaraman (2012) in that (i) `V_{nu,p}` is estimated without separately estimating `sigma^2_{+/-}` and `f`, and (ii) pilot bandwidths are themselves MSE-optimal (l-stage DPI, Wand and Jones 1995), plus IK-style "regularization" to avoid small denominators.

*Edge cases:*
- Discrete running variable / mass points near cutoff (Remark 1): detect -> warn; few mass points invalidate the asymptotics.
- `h_n = b_n` with the same kernel (Remark 7): `tau_hat^bc_SRD(h_n, h_n)` is **numerically identical** to the (not bias-corrected) local-quadratic estimator, and `V^bc_SRD(h_n, h_n)` coincides with the local-quadratic variance. Holds for any `p` (order-`p` estimate + manual bias correction = order-`p+1` estimate). Useful as an internal consistency test and as the `rho = 1` implementation shortcut.
- Different bandwidths on the two sides of the cutoff (Remark 9): all results extend, provided each sequence satisfies the theorem conditions.
- Fuzzy RD denominator `tau_hat_T,SRD ~ 0` (weak identification): Theorem 3 requires `tau_T,SRD != 0`; the paper cites Marmer, Feir, and Lemieux (2014) for weak-IV-robust fuzzy RD inference (Section 1). Guard and warn.
- Fuzzy RD bias correction (Section 3.2): bias-correct the **first-order linear approximation** of the ratio, NOT the numerator and denominator separately - "The former approach seems more intuitive, as it captures the leading bias of the actual estimator of interest."
- Coverage failure of conventional CIs (Section 6, Table I): with IK bandwidth, conventional 95% CIs cover as little as 27-31% (Model 2); robust CIs restore ~89-95%. This motivates defaulting to robust inference.

*Algorithm (sharp RD, p = 1, q = 2; assembled from Sections 2, 4, 5):*
1. Normalize the running variable: `X_i <- X_i - c` so the cutoff is 0.
2. Choose bandwidths: main `h_n` (MSE-optimal for `tau_hat_SRD`, i.e. `h_MSE,0,1,0`) and pilot `b_n` (MSE-optimal for the curvature estimate, `h_MSE,2,2,2`), via DPI selectors (Remark 11 / Supplement S.2.6).
3. Fit weighted local-linear regressions separately on each side with kernel weights `K(X_i/h_n)`; point estimate `tau_hat_SRD(h_n)` = difference in intercepts.
4. Fit weighted local-quadratic regressions on each side at pilot bandwidth `b_n`; extract curvatures `mu_hat^(2)_{+/-,2}(b_n)`; form `B_hat_SRD(h_n, b_n)` using the observed `B_{+/-,SRD}(h_n)` design constants; bias-corrected estimate `tau_hat^bc = tau_hat - h_n^2 B_hat`.
5. Estimate `sigma_hat^2` at each observation by same-side nearest-neighbor residuals (J = 3 default); assemble the `Psi_hat` matrices; compute `V_hat_SRD(h_n)` (conventional) and `V_hat^bc_SRD(h_n, b_n)` (robust, Theorem A.1(V) three-term formula).
6. Report three inference flavors (as in the companion software, Section 1 / references to CCT 2014b, 2014d): conventional (`tau_hat`, `V_hat_SRD`), bias-corrected (`tau_hat^bc`, `V_hat_SRD`), robust (`tau_hat^bc`, `V_hat^bc_SRD`). Default inference = robust.

**Reference implementation(s):**
- R: `rdrobust::rdrobust()`, `rdrobust::rdbwselect()` (Calonico, Cattaneo, and Titiunik 2014d, "rdrobust: An R Package for Robust Inference in Regression-Discontinuity Designs")
- Stata: `rdrobust` (Calonico, Cattaneo, and Titiunik 2014b, *Stata Journal*)

**Requirements checklist:**
- [ ] Sharp RD local-polynomial point estimator, general `p`, default `p = 1`, triangular kernel default
- [ ] Bias correction via order-`q` pilot fit, default `q = p + 1 = 2`, pilot bandwidth `b_n`
- [ ] Three inference flavors: conventional / bias-corrected / robust; robust as default
- [ ] Robust variance per Theorem A.1(V) (three-term per-side decomposition with covariance term `C_{+/-}`)
- [ ] NN variance estimator with fixed `J` (default 3), same-side neighbors; plug-in residual variance as opt-in alternative
- [ ] MSE-optimal bandwidth selection (`h` and `b` jointly, DPI with regularization; Supplement S.2.6 / software papers for feasible formulas)
- [ ] `rho = h/b` reporting; support user-specified `h`, `b`, and `b = h`
- [ ] Mass-point detection warning for discrete running variables (Remark 1)
- [ ] Optional different bandwidths per side (Remark 9)
- [ ] (Deferred, fuzzy) linearization-based bias correction, denominator guard (Theorems 3-4)
- [ ] (Deferred, kink) `nu = 1` estimands via the general `(nu, p, q)` machinery (Theorem 2)

---

## Implementation Notes

### Data Structure Requirements
- Cross-sectional random sample `(Y_i, X_i)`, `i = 1..n`: outcome column + running-variable column + scalar cutoff `c` (paper normalizes `c = 0` WLOG).
- Treatment assignment is derived, not supplied: `T_i = 1(X_i >= c)` (sharp). Fuzzy designs additionally need an observed take-up column `T_i in {0, 1}` (Section 3.2).
- No panel structure, no time dimension, no fixed effects - this is a boundary-nonparametrics estimator.

### Computational Considerations
- Each fit is a tiny weighted least squares problem on the observations inside the bandwidth window (per side): `(p+1) x (p+1)` normal equations; trivially cheap. Bandwidth selection dominates runtime (multiple pilot fits).
- NN variance estimation requires same-side J-nearest-neighbor lookups in the (sorted) running variable: `O(n log n)` sort then linear scans - no KD-tree needed in 1D.
- All estimators are deterministic given `(h_n, b_n, J, kernel)` - no simulation/bootstrap in the default path, so golden-file parity vs `rdrobust` at `atol/rtol` levels is feasible.
- Numerical care: `Gamma^{-1}` inversions on kernel-weighted design matrices can be ill-conditioned for tiny effective samples; guard rank and condition, mirror the library's `solve_ols` conventions.

### Tuning Parameters

| Parameter | Type | Default | Selection Method |
|-----------|------|---------|-----------------|
| `p` (point estimator order) | int | 1 (local-linear) | Paper's main case; Section 2.1. Kink designs use `p = 2` with `nu = 1` (Section 3.1) |
| `q` (bias estimator order) | int | 2 (= `p + 1`) | Section 2.2; Appendix requires `nu <= p < q` |
| `h_n` (main bandwidth) | float | MSE-optimal `h_MSE,0,1,0` | Lemma 1 / DPI selector (Remark 11, Supplement S.2.6); user-overridable |
| `b_n` (pilot bandwidth) | float | MSE-optimal `h_MSE,2,2,2` | Same; `b_n = h_n` is a valid, theory-backed choice (Remark 7; performed well in simulations, Section 6) |
| `kernel` | str | triangular `k(u) = (1-u)1(0<=u<=1)` | Assumption 2; uniform also discussed; same kernel both sides in the paper |
| `J` (NN variance neighbors) | int | 3 | Section 5 / Section 6 simulations; "asymptotically valid for any choice of J" |
| `vce` | str | nearest-neighbor | Section 5; plug-in residuals as alternative (warned slower-coverage) |
| `alpha` | float | 0.05 | Standard; CIs use `Phi^{-1}_{1-alpha/2}` (normal quantiles throughout - no t correction in the paper) |

### Relation to Existing diff-diff Estimators
- **`diff_diff/_nprobust_port.py` is a port of this paper's machinery** (as extended by Calonico-Cattaneo-Farrell 2018 for interior/boundary nonparametric regression): `lprobust_res`/`lprobust_vce` implement the NN and HC variance meats, `lprobust_bw`/`lpbwselect_mse_dpi` implement DPI bandwidth selection, and `local_linear.bias_corrected_local_linear` already computes CCT-2014-style robust bias-corrected CIs at a single boundary point. The HAD estimator consumes exactly this stack.
- **What RD adds on top:** (i) *two* one-sided boundary fits at the cutoff (left and right limits - the cutoff lies in the interior of X's support, but each side is a boundary-point estimation problem) rather than HAD's single support-boundary fit; (ii) the estimand is the *difference*, so bias constants and MSE-optimal bandwidths target `mu_+^(r) - (-1)^{...} mu_-^(r)` combinations (the `s` index in Lemma 1) - this is why `rdbwselect`-style joint selectors differ from `lpbwselect` per-side selectors; (iii) the variance is the sum of two independent one-sided sandwiches (disjoint samples, no cross-covariance); (iv) NN variance neighbors must be restricted to the same side of the cutoff.
- Kernels: `local_linear.py`'s one-sided `[0, 1]` kernels are exactly the `k(.)` of Assumption 2; RD applies them to `|X_i|/h` on each side.
- Remark 7 (`h = b` <=> higher-order fit) gives a free internal consistency test analogous to the library's existing cross-surface parity tests.

---

## Gaps and Uncertainties

1. **Feasible bandwidth selector formulas are not in this paper.** Lemma 1 gives infeasible MSE-optimal bandwidths; Remark 11 defers the data-driven DPI construction (including the regularization terms and the estimation of `V_{nu,p}` without density/variance plug-ins) to the Supplemental Material Section S.2.6 (CCT 2014c). The supplement PDF has been downloaded alongside (papers/Calonico-Cattaneo-Titiunik_2014_ECMA_Supplemental.pdf) but is NOT covered by this review. The companion software papers (CCT 2014b Stata Journal; CCFT 2017 Stata Journal for the updated package) document what `rdbwselect` actually computes - the implementation should parity-target those.
2. **Plug-in residual variance details** are in Supplement S.2.4, not the main text (page 2313 cites it). Main text only warns it may underperform NN.
3. **`rho_n` optimality is explicitly open** (Remark 12): MSE-optimal bandwidths imply `rho_n -> 0`; whether that is optimal for distributional approximation is deferred to Calonico-Cattaneo-Farrell (2014/2018). The modern rdrobust defaults (`rho = h/b` from separate MSE-optimal selectors) follow the later papers - review of CCF 2018 covers this.
4. **No covariates, no clustering, no weights** anywhere in this paper - i.i.d. sampling, outcome-and-running-variable only. Covariate adjustment is Calonico-Cattaneo-Farrell-Titiunik (2019); any `cluster=`/`weights=` surface in our implementation exceeds this paper's theory and must cite the software papers or later work.
5. **Kink designs scale factor**: Section 3.1 notes `tau_SKRD = mu_+^(1) - mu_-^(1)` is the estimand "up to a known scale factor" - the scale (from the assignment rule's slope change) is not developed in the paper; deferred if kink support is ever added.
6. **Table I values transcription**: simulation coverage numbers cited above (27-31% conventional IK coverage in Model 2, etc.) were read from Table I (pages 2316-2317); the table is dense - re-verify against the PDF before quoting in docs.
7. The paper's CIs use **normal quantiles** exclusively; any finite-sample t-style adjustment in our implementation would be a deviation requiring a REGISTRY.md note.
