# Paper Review: On the Effect of Bias Estimation on Coverage Accuracy in Nonparametric Inference

**Authors:** Sebastian Calonico, Matias D. Cattaneo, and Max H. Farrell
**Citation:** Calonico, S., Cattaneo, M. D., & Farrell, M. H. (2018). On the Effect of Bias Estimation on Coverage Accuracy in Nonparametric Inference. *Journal of the American Statistical Association*, 113(522), 767-779. DOI: 10.1080/01621459.2017.1285776
**PDF reviewed:** papers/Calonico-Cattaneo-Farrell_2018_JASA.pdf
**Review date:** 2026-07-12

---

## Methodology Registry Entry

*This is a theory paper, not an estimator paper: it supplies the coverage-error theory behind robust bias-corrected (RBC) inference defaults and the coverage-error-rate (CER) optimal bandwidths that `rdbwselect`'s `cer*` options and `nprobust` implement. Formatted to be merged into the RegressionDiscontinuity registry section as supporting theory.*

## RegressionDiscontinuity (supporting theory: RBC coverage and CER-optimal bandwidths)

**Primary source:** Calonico, S., Cattaneo, M. D., & Farrell, M. H. (2018). On the Effect of Bias Estimation on Coverage Accuracy in Nonparametric Inference. *JASA*, 113(522), 767-779. https://doi.org/10.1080/01621459.2017.1285776

**Key implementation requirements:**

*Assumption checks / warnings:*
- Density case (Assumption 1): iid sample, absolutely continuous distribution, `f > 0` at the evaluation point, `f` S-times continuously differentiable with bounded derivatives, `f^(S)` Hölder continuous with exponent `varsigma`.
- Local polynomial case (Assumption 3): `E[Y^{8+delta}|X] < infinity` for some `delta > 0`; `f` and the conditional variance `v(x) = V[Y|X=x]` continuous and bounded away from zero near the evaluation point; `m(x) = E[Y|X=x]` is `S > q + 2` times continuously differentiable (strict inequality; for integer `S`, at least `q + 3` derivatives) with bounded derivatives, and `m^(S)` is Hölder continuous with exponent `varsigma`.
- Kernels (Assumptions 2, 4): `K` and `L` bounded, even, positive, compact support; density case additionally requires kernel orders `k >= 2`, `l >= 2` (even integers).
- These are inference-theory assumptions - nothing new to check at runtime beyond what CCT 2014 already imposes; the deliverable is *which defaults to pick*, below.

*Core result (why robust bias correction is the default inference):*

Three Wald statistics for a nonparametric point estimate `m_hat` (Section 3.2; density analogues in Section 2):

    T_us  = sqrt(nh) (m_hat - m) / sigma_hat_us                      [undersmoothing]
    T_bc  = sqrt(nh) (m_hat - B_hat_m - m) / sigma_hat_us            [traditional bias correction]
    T_rbc = sqrt(nh) (m_hat - B_hat_m - m) / sigma_hat_rbc           [robust bias correction]

with CIs `I_us`, `I_bc`, `I_rbc` formed with normal quantiles (Equation 5). Theorems 1-2 (Edgeworth expansions, density and local polynomial cases) show:
- `I_bc` (bias-correct but keep the old standard error) is **inferior**: its coverage error carries extra variance/covariance terms `rho^{p+2}(Omega_1,bc + rho^{p+1} Omega_2,bc)` from ignoring the variability of `B_hat_m` - this formalizes why plain bias correction performs poorly (Hall 1992b's negative finding).
- `I_rbc` (bias-correct AND Studentize by the fixed-n variance of `m_hat - B_hat_m`) is as good as undersmoothing in coverage-error decay under identical assumptions and **strictly better when additional smoothness exists** (Section 2.3 worked rates: for `S = 4`, second-order kernels: `|coverage error of I_us| ~ 1/(nh) + nh^5 + h^2` vs `|coverage error of I_rbc| ~ 1/(nh) + nh^9 + h^4`).
- `I_rbc` remains valid for a wide range of bandwidths including MSE-optimal ones, is "more robust to bandwidth choice in applications," and delivers shorter intervals at larger valid bandwidths (Section 4, Figure 2).
- Implementation default: **robust bias-corrected CIs with fixed-n standard errors** - matches the CCT 2014 RD default and the existing `_nprobust_port` behavior.

*Fixed-n (nonasymptotic) variance formulas (Section 3.1; `_nprobust_port.lprobust_vce` implements this family - see the HCk paper-vs-parity caveat below):*

    sigma^2_us  = (h/n) e_0' Gamma_p^{-1} (R_p' W_p SIGMA W_p R_p) Gamma_p^{-1} e_0        (Equation 7)
    sigma^2_rbc = (h/n) e_0' Gamma_p^{-1} (XI_{p,q} SIGMA XI_{p,q}') Gamma_p^{-1} e_0      (Equation 8)
    XI_{p,q} = R_p' W_p - rho^{p+2} LAMBDA_p e_{p+1}' Gamma_q^{-1} R_q' W_q,   rho = h/b

  (transcribed exactly as printed in the published version, page 774). **Note (source-internal discrepancy):** composing Equation 8 from the same page's own building blocks - `B_hat_m = h^{p+1} m_hat^{(p+1)} (1/(p+1)!) e_0' Gamma_p^{-1} LAMBDA_p` with `m_hat^{(p+1)} = b^{-p-1} (p+1)! e_{p+1}' Gamma_q^{-1} R_q' W_q Y / n` - yields `sigma^2_rbc = (nh) V[m_hat - B_hat_m | X]` with a `rho^{p+1}` factor, not the printed `rho^{p+2}`; the published display and its own definitions disagree by one factor of `rho`. Resolve against the nprobust source (the operational reference; its bias-corrected weighting matrix is already ported as the `Q.q` construction in `diff_diff/_nprobust_port.py`) before using this line as an implementation spec.

where `R_p`, `W_p`, `Gamma_p = R_p'W_p R_p/n`, `LAMBDA_p = R_p'W_p[((X_i-x)/h)^{p+1}]'/n` are the usual local-polynomial design objects and `SIGMA = diag(v(X_i))`. Key points:
- **Fixed-n Studentization is essential**: using first-order asymptotic variance approximations instead "can introduce further errors in coverage probability, with particularly negative consequences at boundary points" (Section 1; Section 3.1 discussion of Chen and Qin 2002). Boundary adaptivity ("boundary carpentry") holds only with the fixed-n form - directly relevant to RD, where everything is at a boundary.
- Feasible `SIGMA`: plug in `v_hat(X_i) = (Y_i - r_p(X_i - x)' beta_hat_p)^2` for `sigma^2_us`, and `v_hat(X_i) = (Y_i - r_q(X_i - x)' beta_hat_q)^2` (residuals from the *higher-order* fit - bias-reduced) for `sigma^2_rbc`.
- **HCk weighting** (Section 3.1): HC0 = raw squared residuals; HC1 divides by `(n - 2tr(Q_p) + tr(Q_p'Q_p))/n`; HC2 by `(1 - Q_{p,ii})`; HC3 by `(1 - Q_{p,ii})^2`, where `Q_p = R_p Gamma_p^{-1} R_p' W_p / n` is the (kernel-weighted) projection matrix. "These estimators may perform better for small sample sizes." Theory uses HC0; all HCk share the rates.
  - **Paper/theory target for the RBC variance:** "the corresponding estimators ... are the same, but with q in place of p" (p. 775) - i.e., q-fit residuals AND q-fit hat leverage.
  - **Current nprobust/R parity behavior (differs):** R's `lprobust` (lprobust.R:229-241) - and therefore our parity port `diff_diff/_nprobust_port.py` - reuses the **p-fit** hat-matrix leverage `hii` with the q-fit residuals for the bias-corrected HC2/HC3 variance. This is already documented in REGISTRY.md's HeterogeneousAdoptionDiD Phase 1c `**Note (public-API surface restriction):**` (hc modes not separately parity-tested). An RD implementation that parity-targets rdrobust should follow the R behavior and keep the deviation under the registry's `Note`/`Deviation` labels; a from-paper implementation would use q-fit leverage.
- **NN variance** (Section 3.1): nearest-neighbor residuals with fixed neighbor count (Muller-Stadtmuller 1987; Abadie-Imbens 2008) as an alternative; none of these assume local/global homoskedasticity nor add tuning parameters (details in supplement S.II.2.3, Table S.II.9).

*Bias estimation (Section 3):* `B_hat_m = h^{p+1} m_hat^{(p+1)} (1/(p+1)!) e_0' Gamma_p^{-1} LAMBDA_p` (from Equation 6's conditional bias), with `m_hat^{(p+1)} = b^{-p-1}(p+1)! e_{p+1}' Gamma_q^{-1} R_q' W_q Y / n` from a second local polynomial regression of degree `q > p` (kernel `L`, bandwidth `b`).

*CER-optimal bandwidths (Corollaries 3-5 - the basis of rdbwselect's `cer*` options):*
- MSE-optimal bandwidths are **never CER-optimal**: "an MSE-optimal bandwidth never delivers optimal coverage error decay, even for local linear regression: `h*_mse ∝ n^{-1/(2p+3)} >> h*_rbc ∝ n^{-1/(p+3)}`" (Section 3.3, boundary case). Interior case: `h*_rbc ∝ n^{-1/(p+4)}` (Corollary 5(a)); boundary case: `h*_rbc ∝ n^{-1/(p+3)}` (Corollary 5(b)). RD cutoffs are boundary points -> the RD-relevant rate is `n^{-1/(p+3)}`.
- Consequently the CER bandwidth can be obtained by **rescaling an MSE-optimal bandwidth**: `h_cer = h_mse * n^{-(p/((p+3)(2p+3)))}` (rate arithmetic from the two exponents; for `p = 1`: multiply by `n^{-1/20}`). The paper offers both a fully data-driven DPI selector (`h_hat_dpi = H_hat_dpi n^{-1/(p+3)}` at the boundary, complete steps in the supplement) and this "second data-driven bandwidth choice, based on rescaling already-available MSE-optimal bandwidths" (Section 3.3).
- `H*_rbc(rho_bar)` optimizes the absolute leading coverage-error term (Corollary 5; density analogue Corollary 3) - the `q_k` polynomials are known, odd, kernel-dependent-only functions. The exponents differ by case, and **RD cutoffs are boundary points, so Corollary 5(b) is the RD-operative form**:
  - Interior (Corollary 5(a), `h ∝ n^{-1/(p+4)}`): `argmin_H | H^{-1} q_1,rbc + H^{1+2(p+3)} (eta~^int_bc)^2 q_2,rbc + H^{p+3} (eta~^int_bc) q_3,rbc |`
  - Boundary (Corollary 5(b), `h ∝ n^{-1/(p+3)}`): `argmin_H | H^{-1} q_1,rbc + H^{1+2(p+2)} (eta~^bnd_bc)^2 q_2,rbc + H^{p+2} (eta~^bnd_bc) q_3,rbc |`

*Choice of pilot bandwidth / rho (Sections 2.4, 3.3):*
- Recommended default `rho = h/b = 1` (i.e., `b = h`): "setting rho = 1 has good theoretical properties, minimizing interval length of I_rbc or the MSE of f_hat - B_hat_f, depending on the conditions imposed... we found that rho = 1 performed well. As a result, from the practitioner's point of view, the choice of b (or rho) is completely automatic."
- `rho -> 0` required by traditional bias correction is exactly what wastes the variance correction; bounded positive `rho` capitalizes on it. `rho_bar = infinity` cannot reduce bias and inflates variance.
- With `q = p + 1`, `K = L`, `rho = 1`: `m_hat - B_hat_m` is identical to the order-`q` local polynomial estimator (cites CCT 2014 Remark 7).

*Edge cases:*
- Boundary points: for `p` odd (local-linear), the *undersmoothing* coverage rate is unchanged at the boundary, but the *RBC* rate differs interior-vs-boundary (`sqrt(nh) h^{p+3}` interior vs `sqrt(nh) h^{p+2}` boundary for odd `q`) - bandwidth selectors must know whether the point is interior or boundary (RD: always boundary).
- `q` even vs odd changes boundary rates (Section 3.2 end); default `q = p + 1` with `p` odd keeps the standard case.
- Ad hoc undersmoothing of `h_hat_mse` (common practice) is "no panacea" (Section 4 / supplement Table S.II.8).
- Kernel choice: second-order minimum-variance (interval length) or MSE-optimal kernels recommended; triangular, Epanechnikov, uniform all standard (Section 2.4 / 3.3).

*Algorithm (RBC inference at a point, assembled from Sections 3.1-3.3):*
1. Choose `p` (default 1), `q = p + 1`, kernel; select `h` (CER-optimal for inference-first use; MSE-optimal remains valid for `I_rbc`), set `b = h` (`rho = 1`) unless the user overrides.
2. Fit order-`p` local polynomial at the point -> `m_hat`; fit order-`q` at bandwidth `b` -> `m_hat^{(p+1)}` -> `B_hat_m`.
3. Compute `sigma_hat^2_rbc` by the fixed-n formula (Equation 8) with HC0-HC3 or NN residual plug-ins.
4. `I_rbc = [m_hat - B_hat_m ± z_{1-alpha/2} sigma_hat_rbc / sqrt(nh)]`.

**Reference implementation(s):**
- R/Stata: `nprobust` (Calonico, Cattaneo, and Farrell 2017) - already ported as `diff_diff/_nprobust_port.py`
- R/Stata: `rdrobust` `cer*` bandwidth options implement the CER logic at RD boundary points

**Requirements checklist:**
- [ ] RBC (not conventional-bc) as the default inference flavor, fixed-n Studentization
- [ ] `rho = 1` / `b = h` as the automatic pilot choice applies to standalone CCF/nprobust-style point inference (this paper's recommendation); the RD estimator's parity default follows the 2017 rdrobust behavior instead - data-driven `b` from the bandwidth selectors unless the user supplies `rho` (then `b = h/rho`)
- [ ] CER bandwidth option via MSE-rescaling `h_cer = h_mse * n^{-p/((p+3)(2p+3))}` (boundary exponents, since RD is a boundary problem)
- [ ] HC0-HC3 and NN variance options with residuals from the order-`q` fit for the RBC variance
- [ ] Boundary (not interior) rate constants everywhere in RD bandwidth selection

---

## Implementation Notes

### Data Structure Requirements
- Same as the RD estimator: iid `(Y_i, X_i)` cross-section; this paper adds no data surface.

### Computational Considerations
- All quantities are small weighted least-squares objects; the `XI_{p,q}` sandwich needs the cross design matrices `R_p' W_p` and `R_q' W_q` at both bandwidths - already the shape of `_nprobust_port.lprobust_vce`.
- The CER rescaling route (`h_cer` from `h_mse`) is one multiplication - prefer it over a separate DPI chain for v1, matching rdrobust's `cerrd` construction.

### Tuning Parameters

| Parameter | Type | Default | Selection Method |
|-----------|------|---------|-----------------|
| `p` | int | 1 | Section 3; odd degree keeps boundary adaptivity |
| `q` | int | `p + 1` | Sections 3.2-3.3 |
| `rho = h/b` | float | 1 (when only `h` given) | Sections 2.4/3.3: "completely automatic"; minimizes interval length |
| `h` | float | CER- or MSE-optimal | Corollary 5 (CER); MSE-optimal valid but coverage-suboptimal |
| `vce` | str | fixed-n; HC0 (theory) / NN or HCk (practice) | Section 3.1 |
| kernel | str | triangular / Epanechnikov / uniform | Section 3.3; second-order kernels |

### Relation to Existing diff-diff Estimators
- This is the paper the existing `_nprobust_port.py` already credits for its MSE-DPI selector; this review documents the *other half* of the paper (coverage theory, CER bandwidths, HCk small-sample weightings, rho = 1 guidance) that the RD estimator will additionally draw on.
- For RD: everything at the cutoff is a *boundary* point, so the boundary-case corollaries (5(b), `n^{-1/(p+3)}`) are the operative ones; `rdbwselect`'s `cerrd`/`certwo`/`cersum`/`cercomb*` options are rescalings of the corresponding `mse*` selectors using exactly these rates.
- The fixed-n Studentization message reinforces the library convention that Rust/Python parity must be checked against the fixed-n formulas, not asymptotic simplifications.

---

## Gaps and Uncertainties

1. **Exact CER constants and DPI implementation steps are in the online supplement**, not the paper (Corollary 3/5 constants `Omega_1`, `Omega_2`, the `q_k,rbc` polynomials, and the complete DPI recipe: "we give precise implementation details" - supplement Sections S.I.3, S.II). The practical rdrobust route (rescaling MSE bandwidths) does not need them; a from-scratch CER-DPI selector would. The supplement was not downloaded.
2. **The `rho = 1` default vs rdrobust behavior**: this paper recommends `rho = 1`; rdrobust's default selectors instead produce `h` and `b` from separate MSE criteria (rho != 1 in general) per CCT 2014 / CCFT 2017. The implementation must pick rdrobust parity (separate `h`, `b`) as default and note `b = h` as the supported special case - resolve in the CCFT 2017 review/implementation plan.
3. **Exact exponent bookkeeping for `h_cer`**: the multiplicative rescaling `n^{-p/((p+3)(2p+3))}` is derived here from the two rates (boundary case); verify the constant against `rdbwselect`'s source before hardcoding (the software may also rescale `b`).
4. Coverage-error expansions assume **normal quantiles and iid data**; nothing here about clustering - cluster-robust CER guidance comes from later work (CCFT 2017 software paper documents what the package does).
5. Table 1 (page 777) transcription of simulation coverage/length used for context only; re-verify before quoting.
