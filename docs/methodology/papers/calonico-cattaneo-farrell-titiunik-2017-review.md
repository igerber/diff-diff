# Paper Review: rdrobust: Software for regression-discontinuity designs

**Authors:** Sebastian Calonico, Matias D. Cattaneo, Max H. Farrell, and Rocío Titiunik
**Citation:** Calonico, S., Cattaneo, M. D., Farrell, M. H., & Titiunik, R. (2017). rdrobust: Software for regression-discontinuity designs. *The Stata Journal*, 17(2), 372-404.
**PDF reviewed:** papers/Calonico-Cattaneo-Farrell-Titiunik_2017_Stata.pdf
**Review date:** 2026-07-12

---

**Notational note (applies throughout):** In this paper all displayed equations are UNNUMBERED — there are no "Equation (N)" labels anywhere in pages 372-403. Formulas below are referenced by journal page number and section only. The paper repeatedly defers "implementation and numerical details" to **CCFT** = Calonico, Cattaneo, Farrell, and Titiunik (2016b, "Regression discontinuity designs using covariates") and its supplemental appendix (published as CCFT 2019 REStat), plus Calonico, Cattaneo, Farrell (2016a / Forthcoming JASA) for CER-optimal bandwidths.

---

## Methodology Registry Entry

*Formatted to match docs/methodology/REGISTRY.md structure. Heading levels and labels align with existing entries — copy the `## RegressionDiscontinuity (rdrobust software reference)` section into the appropriate category in the registry.*

## RegressionDiscontinuity (rdrobust software reference)

**Primary source:** Calonico, S., Cattaneo, M. D., Farrell, M. H., & Titiunik, R. (2017). rdrobust: Software for regression-discontinuity designs. *The Stata Journal*, 17(2), 372-404. https://doi.org/10.1177/1536867X1701700208
- Methodology companions: Calonico, Cattaneo, Titiunik (2014b, *Econometrica*) for robust bias-corrected inference; Calonico, Cattaneo, Farrell (2016a / Forthcoming JASA) for CER-optimal bandwidths; CCFT (2016b; published 2019 *REStat*) + supplemental appendix for covariate-adjusted/cluster formulas.

**Key implementation requirements:**

*Assumption checks / warnings:*
- Treatment assignment is deterministic in the score: `T_i = 1(X_i >= x̄)` — units at exactly the cutoff are TREATED (Section 2.1, p. 375).
- Covariate balance is an ASSUMPTION, not automatic (Section 2.1, p. 376; Section 7, p. 397): the covariate-adjusted estimator is consistent for `tau` under the balance condition, and unbalanced covariates generically break consistency (CCFT 2019 Lemma 1 gives the exact probability limit; zero RD effect on each covariate is the recommended sufficient, testable condition). For sharp and fuzzy RD, balanced = equal conditional expectations of the potential covariates at the cutoff (zero RD effect on covariates — the usual falsification condition); for sharp and fuzzy KINK RD, balanced = equal FIRST DERIVATIVES of the covariate conditional-expectation functions at the cutoff. Testable by running the estimator with each covariate as the outcome variable.
- Covariates may LENGTHEN CIs (irrelevant-covariate example, p. 397, +0.28% CI length) — documented as expected behavior; no warning issued by rdrobust.
- Missing covariate values drop observations (Senate example: N falls 1297 -> 1108, pp. 395-396), so covariate-adjusted runs are not on the same sample as unadjusted runs.
- MSE bandwidth closed form assumes nonzero bias constant (`B_j != 0`, p. 380); IK-style regularization (default on, `scaleregul`) exists partly to handle small-bias denominators.

*Model / data structure (Section 2.1, p. 375):*
- Random cross-section sample `(Y_i, T_i, X_i, Z_i')'`, `i = 1, ..., n`; `X_i` = score/running variable; `x̄` = known cutoff; `Z_i` = d-dimensional preintervention covariates (continuous, discrete, or mixed); optional unit weights `w_i`; optional cluster ID.
- Potential outcomes: `Y_i = Y_i(0)*(1 - T_i) + Y_i(1)*T_i`.
- Sharp RD estimand (p. 375, unnumbered display):

      tau = tau(x̄) = E{ Y_i(1) - Y_i(0) | X_i = x̄ }

- Software covers sharp, fuzzy, sharp kink, and fuzzy kink RD plus higher-order derivatives via `deriv(dvalue)` (fuzzy kink = `fuzzy(fuzzyvar)` + `deriv(1)`); exposition is sharp-RD only, with kink identification cited to Card et al. (2015). Covariate adjustment targets the SAME `tau` (efficiency improvement, not a different estimand).

*Estimator equation (Section 2.1, p. 376, unnumbered displays, as implemented):*

Polynomial bases and kernel:

    r_{-,p}(x) = 1(x < 0) * (1, x, ..., x^p)'
    r_{+,p}(x) = 1(x >= 0) * (1, x, ..., x^p)'
    e_0 = (p+1)-vector with 1 first, 0s elsewhere
    K_h(u) = K(u/h)/h

Covariate-adjusted RD estimator (headline feature):

    tau~(h) = e_0' * beta~_{Y+,p}(h) - e_0' * beta~_{Y-,p}(h)

where the coefficients come from the SINGLE JOINT weighted least-squares regression:

    theta~_{Y,p}(h) = argmin_{beta_-, beta_+, gamma}
        sum_{i=1}^{n} { Y_i - r_{-,p}(X_i - x̄)' beta_-
                            - r_{+,p}(X_i - x̄)' beta_+
                            - Z_i' gamma }^2 * K_h(X_i - x̄)

    theta~_{Y,p}(h) = { beta~_{Y-,p}(h)', beta~_{Y+,p}(h)', gamma~_{Y,p}(h)' }'
    beta_-, beta_+ in R^{p+1},   gamma in R^d

where:
- `p` = polynomial order for point estimation (preferred choice `p = 1`, the standard local linear RD estimator; default `p = 1`, `q = 2`)
- `q` = polynomial order for bias correction, `q > p`
- `h` = main bandwidth; `b` = bias (preliminary) bandwidth; `rho = h/b`
- `gamma` is COMMON to both sides — this common-gamma restriction is what makes `tau~(h)` consistent for the standard RD effect `tau` (p. 376)
- Estimation COMBINES units from both sides in one regression when covariates are included; without covariates, two separate one-sided fits (standard), and the pre-2017 rdrobust estimator is recovered EXACTLY (backward compatibility, p. 376)
- Covariates enter additively-separably and linear-in-parameters (mimicking experimental covariate-adjustment practice)
- Consistency `tau~(h) ->_P tau` holds under the balance condition on the covariates at the cutoff (sufficient, testable; see assumption checks above)

*With covariates:* see the joint regression above — covariate adjustment IS the main displayed estimator; setting `d = 0` recovers the unadjusted two-sided fit.

*With user weights (p. 376):* if unit weights `w_i` are provided, ALL estimation and inference procedures replace `K_h(X_i - x̄)` with `w_i * K_h(X_i - x̄)`.

*Three inference flavors (Section 2.1, p. 377):*

1. **Conventional**: the point estimator `tau~(h)`; leading misspecification bias `h^{p+1} * B`, where `B` depends on the curvature of `E{Y_i(t)|X_i = x}` AND (with covariates) on the curvature of the covariate conditional expectations.
2. **Bias-corrected**:

       tau~^bc(h, b) := tau~(h) - h^{p+1} * B_hat(b)

   with `B_hat(b)` an estimator of `B` on a possibly different preliminary bandwidth `b`.
3. **Robust bias-corrected**: bias-corrected point estimate + variance `V^bc(h, b)` capturing BOTH the RD estimator's variability AND the bias correction's variability. Robust t statistic and 100*(1-alpha)% CI (p. 377, unnumbered displays):

       sqrt(n*h) * { tau~^bc(h,b) - tau } / sqrt( V_hat^bc(h,b) )

       [ tau~^bc(h,b) - (Phi_{1-alpha/2} / sqrt(n*h)) * sqrt( V_hat^bc(h,b) ) ,
         tau~^bc(h,b) + (Phi_{1-alpha/2} / sqrt(n*h)) * sqrt( V_hat^bc(h,b) ) ]

   where `Phi_alpha` = alpha percentile of the standard normal. `V^bc(h,b)` is derived with a FIXED-n (preasymptotic) approach, CONDITIONAL on the scores `X_1, ..., X_n` (Section 2.2, p. 378); with covariates it depends on the covariates and "is necessarily different from prior work."

Output structure (pp. 392-399): two inference rows — `Conventional` (point estimate + conventional SE + z + p + CI) and `Robust` (bias-corrected robust z, p, CI only; no Coef./Std. Err. printed for the Robust row). Optimality pairing (Section 1, p. 373): MSE-optimal bandwidths for an MSE-optimal POINT estimator; CER-optimal bandwidths for CER-optimal robust bias-corrected CIs.

*Standard errors (Section 2.2, pp. 378-379; option catalog pp. 385-386):*
- Default: `vce(nn 3)` — heteroskedasticity-robust nearest-neighbor (NN) variance estimator, minimum `J = 3` neighbors. Sample covariance estimators from the `J` nearest neighbors of unit i among units on the SAME side of the cutoff, neighbors by Euclidean distance on the score (Müller and Stadtmüller 1987; Abadie and Imbens 2008). With covariates, the required unknowns include the outcome-covariate conditional covariances `sigma_{Y Z_k -, i} = Cov{Y_i(0), Z_ki(0) | X_i}` and `sigma_{Y Z_k +, i} = Cov{Y_i(1), Z_ki(1) | X_i}` (new here).
- Alternative: `hc0` / `hc1` / `hc2` / `hc3` — heteroskedasticity-robust plug-in residuals variance estimators (MacKinnon 2013; Cameron and Miller 2015): products of side-specific plug-in residuals from local polynomial regressions of order `q` using the outcome and each covariate as dependent variables, with finite-sample weights `omega_{-,i}`, `omega_{+,i}` for HCk (p. 378, unnumbered displays):

      1(X_i < x̄)  * omega_{-,i} * { Y_i   - r_q(X_i - x̄)' beta_hat_{Y-,q}(h) }
                                * { Z_ki - r_q(X_i - x̄)' beta_hat_{Z_k-,q}(h) }

      1(X_i >= x̄) * omega_{+,i} * { Y_i   - r_q(X_i - x̄)' beta_hat_{Y+,q}(h) }
                                * { Z_ki - r_q(X_i - x̄)' beta_hat_{Z_k+,q}(h) }

  Precise formulas deferred to the CCFT supplemental appendix.
- Clustering: `nncluster clustervar [nnmatch]` (cluster-robust NN) and `cluster clustervar` (cluster-robust plug-in residuals) — one-way clustering "following the same logic" as the heteroskedastic forms (p. 379; Bartalotti and Brummet 2017). Explicit cluster formulas not printed in this paper (CCFT supplemental appendix). With cluster VCE, per-side cluster counts are reported and can differ across sides (48/50 in the msetwo example, pp. 398-399) because clusters are counted within each side's bandwidth window.
- The paper describes the resulting menu as 10 distinct variance estimators (p. 379) — NN or plug-in residuals with HC0-HC3, plus NN-adjusted or degrees-of-freedom-adjusted clustering (previous version had only 2: NN, and HC0-weighted plug-in). These variance estimators are used to Studentize statistics, form CIs, AND inside the data-driven bandwidth selectors.
- Stata equivalence (p. 401): `vce(cluster ...)` + `kernel(uniform)` is closest to Stata's built-in `regress` with clustering; without clustering, `vce(hc1)` is closest to `regress` — useful for validation tests.

*Bandwidth selection (Section 2.3, pp. 379-381; option catalog p. 385):*
- Asymptotic MSE expansion (p. 380, unnumbered display), for an estimator `theta_hat(h)`:

      MSE{ theta_hat(h) } ~ h^{2p+2} * B + (1/(n*h)) * V

  with four sharp-RD expansions by choice of estimator (p. 380): (1) RD estimator `tau~(h)`; (2) left-side `e_0' beta~_{Y-,p}(h)`; (3) right-side `e_0' beta~_{Y+,p}(h)`; (4) sum `e_0' beta~_{Y+,p}(h) + e_0' beta~_{Y-,p}(h)` (the sum is "mostly useful for regularization purposes"). Both `B` and `V` depend on whether covariates are included.
- Population MSE-optimal bandwidth (assuming nonzero denominator, p. 380) and feasible plug-in:

      h_mse,j     = { (V_j / n)     / (2*(1+p) * B_j)     }^{1/(3+2p)},   j in {rd, l, r, sum}
      h_hat_mse,j = { (V_hat_j / n) / (2*(1+p) * B_hat_j) }^{1/(3+2p)},   j in {rd, l, r, sum}

  Preliminary estimates `V_hat_j`, `B_hat_j` depend on covariate inclusion and the heteroskedasticity/clustering assumption; exact forms are in the CCFT supplemental appendix (not in this paper).
- CER-optimal plug-in selectors (p. 381; Calonico, Cattaneo, Farrell 2016a / Forthcoming):

      h_hat_cer,j = n^{ -p / ((3+p)*(3+2p)) } * h_hat_mse,j,   j in {rd, l, r, sum}

  These minimize the coverage error rate of the robust bias-corrected CI — preferable for inference. Structural fact from the worked example (p. 400): CER selectors shrink only `h`; the bias bandwidth `b` is IDENTICAL to the corresponding MSE selector's `b` (e.g., cerrd b = 27.984 = mserd b). A parity implementation must reproduce this.
- Combined selectors "with better rate properties" (p. 381):

      h_hat_comb,l = median{ h_hat_mse,l, h_hat_mse,rd, h_hat_mse,sum }     (per-side; comb2)
      h_hat_comb,r = median{ h_hat_mse,r, h_hat_mse,rd, h_hat_mse,sum }
      h_hat_comb   = min{ h_hat_mse,rd, h_hat_mse,sum }                     (common; comb1)

  (similarly for the CER-optimal versions).
- The 10 `bwselect()` option strings (verbatim definitions, p. 385):

  | Option | Definition |
  |---|---|
  | `mserd` | One common MSE-optimal bandwidth for the RD treatment-effect estimator. **THE DEFAULT.** |
  | `msetwo` | Two different MSE-optimal bandwidths (below and above the cutoff) |
  | `msesum` | One common MSE-optimal bandwidth for the SUM of regression estimates (not the difference) |
  | `msecomb1` | `min(mserd, msesum)` |
  | `msecomb2` | `median(msetwo, mserd, msesum)` for each side separately |
  | `cerrd` | One common CER-optimal bandwidth for the RD treatment-effect estimator |
  | `certwo` | Two different CER-optimal bandwidths (below and above the cutoff) |
  | `cersum` | One common CER-optimal bandwidth for the sum of regression estimates |
  | `cercomb1` | `min(cerrd, cersum)` |
  | `cercomb2` | `median(certwo, cerrd, cersum)` for each side separately |

  Guidance (p. 400): the most useful in practice are `mserd`, `msetwo`, `cerrd`, `certwo`; "the other options are useful for regularization and sensitivity analysis purposes." The paper also counts the underlying menu as 8 MSE-optimal choices (rd/l/r/sum, each with and without regularization) + 8 CER-optimal + the two combined selectors (p. 381).
- Regularization (pp. 381, 400): Imbens and Kalyanaraman (2012)-style regularization, implemented as in Calonico, Cattaneo, Titiunik (2014a, 2014b, 2015b) and their supplemental appendices, is INCLUDED BY DEFAULT and "always leads to smaller bandwidths." Modified or removed via `scaleregul(scaleregulvalue)`; `scaleregul(0)` removes it. Exact regularization formula not printed in this paper.
- rho handling (Section 3.2, p. 383; p. 392): `bwselect` computes both `h` and `b` by default UNLESS `rho(rhovalue)` is specified, in which case only `h` is computed and `b = h / rho`. `rho = h/b` is printed per side in every output header; data-driven defaults yield rho < 1 (0.56-0.65 in the examples); `rho(1)` forces h = b.
- Covariates/clustering/weights propagate into bandwidth selection (pp. 396, 398-400): data-driven selectors "account for the additional covariates and the clustering structure of the matrix of variances and covariances"; cluster VCE alone changes selected bandwidths (mserd h: 17.708 -> 17.509 with `vce(nncluster state)`).
- Fuzzy bandwidth logic (Section 6, p. 389; option p. 385): two approaches — (1) select bandwidth(s) for the sharp intention-to-treat estimator in the NUMERATOR of the fuzzy estimator (only approach in the 2014 version); (2) select bandwidth(s) for the actual fuzzy RD treatment-effect estimator ("the ratio of reduced-form RD estimators"). Default: approach 2 whenever TWO-SIDED imperfect compliance is present; otherwise approach 1. `sharpbw` inside `fuzzy()` forces approach 1; it is automatically selected when there is perfect compliance at either side of the threshold — an implementation must detect one-sided perfect compliance and switch.
- Deprecated/removed selectors (Section 2.3, p. 381): `ik`, `cct`, `cv` are no longer supported. `h_hat_mse,rd` is an upgraded version of BOTH the `ik` (Imbens-Kalyanaraman 2012) and `cct` (CCT 2014b) implementations. Cross-validation (`cv`) was removed: less popular, not theoretically justified, not portable to covariates/clustering.

*Edge cases:*
- Mass points / discrete running variable in NN variance (Section 2.2, p. 379): number of eligible EQUIDISTANT nearest neighbors can strictly exceed the requested `J` (default minimum 3). OLD (2014) behavior: ties broken AT RANDOM to hit exactly J (non-replicable). NEW behavior: use ALL equidistant neighbors even if the total exceeds the requested number — fully replicable and more efficient. `nnmatch` is therefore documented as the MINIMUM number of neighbors.
- Perfect one-sided compliance in fuzzy RD (p. 385): detected -> fuzzy bandwidth selection automatically switches to the sharp-model procedure (approach 1 / `sharpbw` behavior).
- Zero/small bias constant in MSE bandwidth (p. 380): closed form assumes `B_j != 0` -> IK-style regularization (default on) and the `sum` expansion exist partly for this; `scaleregul` controls it.
- Missing covariate values (pp. 395-396): observations dropped -> effective sample changes vs the unadjusted run (1297 -> 1108 in the example).
- Unbalanced covariates (pp. 376, 397): consistency for `tau` breaks -> test empirically by running the estimator with each covariate as outcome (`e(tau_cl)` / `e(pv_rb)`).
- Irrelevant covariates (p. 397): may lengthen CIs (+0.28% example) -> documented expected behavior, no warning.
- msetwo divergent side bandwidths (p. 399): one-sided MSE bandwidths can be "quite distinct" (14.661 vs 20.893) while the point estimate stays stable (6.807 vs 6.851) -> offered as a sensitivity-analysis pattern, not an error.
- rdplot per-bin CIs with IMSE-optimal evenly spaced bins (pp. 391-392): "may exhibit a first-order smoothing bias" -> remedies: `binselect(esmv)` (mimicking-variance bins), or undersmooth by scaling up bins via `scale()` / setting `nbins()`.
- No mass-points handling for estimation itself, degenerate-sample warnings, or weak-first-stage diagnostics are discussed anywhere in the paper.

*Algorithm (sharp RD with covariates, as the paper's surface implies):*
1. Center the score: `x_i = X_i - x̄`; classify sides (`x_i < 0` control, `x_i >= 0` treated).
2. If `h`/`b` not user-supplied, run the bandwidth selector (`bwselect`, default `mserd`), accounting for covariates, clustering, and weights; if `rho` given, compute only `h` and set `b = h/rho`. CER selectors: compute the MSE bandwidths, shrink `h` by `n^{-p/((3+p)(3+2p))}`, keep the MSE pilot `b`.
3. Fit the order-`p` local polynomial (joint regression with common `gamma` if covariates present) with kernel weights `K_h(x_i)` (times `w_i` if given) -> conventional point estimate `tau~(h)`.
4. Fit the order-`q` (`q > p`) local polynomial on bandwidth `b` to construct the bias estimate `B_hat(b)`; form `tau~^bc(h,b) = tau~(h) - h^{p+1} * B_hat(b)`.
5. Estimate the variance menu element requested by `vce()` (NN default; HCk plug-in residuals evaluated at the cutoff; cluster-robust variants) — for the conventional variance and the robust `V_hat^bc(h,b)` capturing bias-correction variability.
6. Report the Conventional row (point estimate, SE, z, p, CI from normal quantiles) and the Robust row (bias-corrected robust z, p, CI only).
7. Fuzzy RD: repeat with the reduced-form (outcome) and first-stage (take-up) estimators; point estimator = ratio of reduced-form RD estimators; bandwidth per the fuzzy logic above. (Linearization/bias-correction details for fuzzy CIs are in CCFT, not this paper.)

**Reference implementation(s):**
- R: `rdrobust::rdrobust()`, `rdrobust::rdbwselect()`, `rdrobust::rdplot()` ("A companion R package with the same functionality and syntax is also available," Section 8, p. 402)
- Stata: `rdrobust`, `rdbwselect`, `rdplot` (plus bundled `rdbwselect_2014` for legacy bandwidths; replication file `rdrobust_illustration.do`)

**Requirements checklist:**
- [ ] Units at exactly the cutoff are treated (`X_i >= x̄`)
- [ ] Conventional, bias-corrected, and robust bias-corrected inference all computed; Robust row reports z/p/CI only
- [ ] `p = 1`, `q = 2`, triangular kernel, `mserd`, `vce(nn 3)` defaults reproduced
- [ ] Covariate adjustment uses the single joint regression with common `gamma`; recovers the unadjusted estimator exactly when `d = 0`
- [ ] Covariates, clustering, and weights propagate into bandwidth selection, not just variance
- [ ] CER selectors shrink `h` only and reuse the corresponding MSE selector's `b`
- [ ] `rho()` computes `h` only and sets `b = h/rho`
- [ ] IK-style regularization on by default; `scaleregul(0)` removes it
- [ ] NN variance uses ALL equidistant neighbors under ties (minimum-J semantics)
- [ ] Fuzzy bandwidth approach auto-switches to sharp under one-sided perfect compliance
- [ ] Senate worked-example smoke-test targets reproduced (see below)

---

## Implementation Notes

### Data Structure Requirements
- Cross-sectional random sample `(Y_i, T_i, X_i, Z_i')'` — outcome, treatment take-up (fuzzy only), running variable, optional d-dimensional covariate vector (continuous, discrete, or mixed).
- Known scalar cutoff `x̄` (Stata default `c(0)`).
- Optional unit-specific (frequency) weights variable — multiplies the kernel function in ALL estimation and inference procedures.
- Optional one-way cluster ID variable.
- Missing covariate values reduce the estimation sample (observations dropped).

### Computational Considerations
- The 2016/2017 implementation is dramatically faster than the 2014 version (Section 1, pp. 373-374, Table 1: n = 50,000: 95.651s old vs 1.148s new = 83.32x; n = 100,000: 183.04x) and was tested on 30+ million observations (~5 min default options; ~16 min with a cluster-robust option, Stata/SE 14). Speed gains are exclusively implementation, not methodology; the old code "does not scale well with n."
- Plug-in residuals are computed ALL at the cutoff point in the new version ("mimics exactly linear least-squares methods," much faster); the 2014 version evaluated predicted values nonparametrically at points near the cutoff — the source of the plug-in variance backward-incompatibility.
- The optimized NN procedure under ties is deterministic (all equidistant neighbors used), replacing the 2014 random tie-breaking — fully replicable.
- Variance estimators are shared machinery: they Studentize the test statistics, build CIs, and sit inside the data-driven bandwidth selectors, so a single variance module should serve all three consumers.

### Tuning Parameters

| Parameter | Type | Default | Selection Method |
|-----------|------|---------|-----------------|
| `c` (cutoff) | float | 0 | Known by design |
| `p` (estimation order) | int | 1 | Preferred choice p=1 (local linear); user override |
| `q` (bias order) | int | 2 | q > p; user override |
| `deriv` | int | 0 | 0 sharp/fuzzy levels; 1 for kink designs |
| `kernel` (estimation) | choice | triangular | triangular (recommended), uniform, epanechnikov |
| `h` (main bandwidth) | float or (L, R) pair | data-driven | `bwselect` (default `mserd`) |
| `b` (bias bandwidth) | float or (L, R) pair | data-driven | `bwselect`; or `b = h/rho` when `rho` given |
| `rho` | float | unset | If set, only `h` selected and `b = h/rho` |
| `bwselect` | choice (10) | `mserd` | mserd/msetwo/msesum/msecomb1/msecomb2 + cer* analogs |
| `scaleregul` | float | 1 (regularization on) | 0 removes IK-style regularization; always-on shrinks bandwidths |
| `scalepar` | float | (see help files) | Scaling of reported estimand (option listed in syntax, p. 382; semantics not detailed in paper) |
| `vce` | choice | `nn 3` | nn [J] / hc0-hc3 / nncluster / cluster |
| `nnmatch` (J) | int | 3 | MINIMUM number of NN neighbors (all equidistant ties used) |
| `level` | float | 95 | CI coverage level |
| `fuzzy` / `sharpbw` | var / flag | off (sharp) | sharpbw forces sharp-model bandwidths; auto under one-sided perfect compliance |
| `covs` | var list | none | User-chosen; must be balanced at cutoff (testable) |
| `weights` | var | none | User-supplied; multiplies kernel |
| rdplot `kernel` | choice | uniform | Unlike estimation (triangular preferred) |
| rdplot `p` | int | 4 | Global-fit order (evident from examples) |
| rdplot `nbins` | int or (L, R) | data-driven | `binselect()` method (es/esmv etc.; formulas in CCT 2015a) |
| rdplot `scale` | (L, R) | (1, 1) | Multiplies optimal bin counts (undersmoothing knob) |
| rdplot `h` | float or (L, R) | full support | Restrict to match rdrobust bandwidth for exact point-estimate display |
| rdplot `support` | (L, R) | sample range | Extended support for bin construction |
| rdplot `ci` | float | off | Per-bin CI coverage level |

### Command surface (rdrobust / rdbwselect / rdplot)

*rdrobust syntax (Section 3.1, p. 382):*

    rdrobust depvar runvar [if] [in] [, c(cutoff) p(pvalue) q(qvalue)
        deriv(dvalue) fuzzy(fuzzyvar [sharpbw]) covs(covars) kernel(kernelfn)
        weights(weightsvar) h(hvalueL hvalueR) b(bvalueL bvalueR) rho(rhovalue)
        scalepar(scaleparvalue) bwselect(bwmethod) scaleregul(scaleregulvalue)
        vce(vcemethod) level(level) all]

Key option semantics (Sections 3.2 / 4.2, pp. 382-386):
- `fuzzy(fuzzyvar [sharpbw])`: treatment take-up variable for fuzzy RD (fuzzy kink with `deriv(1)`); default sharp. `sharpbw` = fuzzy estimation with sharp-RD bandwidth selection; auto-selected under perfect compliance on either side.
- `covs(covars)`: additional covariates for estimation AND inference (and, when bandwidths are data-driven, bandwidth selection).
- `weights(weightsvar)`: unit-specific weights; multiply the kernel function.
- `h(hvalueL hvalueR)` / `b(bvalueL bvalueR)`: one value applies to both sides; two values = below/above the cutoff. If unspecified, computed by rdbwselect.
- `bwselect(bwmethod)`: 10-method menu (see registry table above); computes both `h` and `b` unless `rho` is given.
- `vce(vcemethod)` full catalog (pp. 385-386):

  | vce string | Meaning |
  |---|---|
  | `nn [nnmatch]` | Heteroskedasticity-robust NN variance; *nnmatch* = minimum neighbors. **DEFAULT `vce(nn 3)`** |
  | `hc0` / `hc1` / `hc2` / `hc3` | Heteroskedasticity-robust plug-in residuals variance (HCk weighting) |
  | `nncluster clustervar [nnmatch]` | Cluster-robust NN variance |
  | `cluster clustervar` | Cluster-robust plug-in residuals variance |

- `all`: report all three inference flavors (conventional/bias-corrected/robust).
- Reported header quantities (default run, p. 392): Number of obs; Eff. number of obs (within h per side); Order est. (p); Order bias (q); BW est. (h) L/R; BW bias (b) L/R; rho (h/b) per side; BW type; Kernel; VCE method; per-side Number of clusters when clustering (p. 398).
- Returned e() results observed in use (pp. 395, 397): `e(h_l)`, `e(h_r)`, `e(b_l)`, `e(b_r)`, `e(ci_l_rb)`, `e(ci_r_rb)` (robust CI endpoints), `e(tau_cl)` (conventional point estimate), `e(pv_rb)` (robust p-value).
- Removed options (Section 3.3): `delta()`, `cvgrid_min()`, `cvgrid_max()`, `cvgrid_length()`, `cvplot`, `matches()`.

*rdbwselect syntax (Section 4, pp. 384-386):*

    rdbwselect depvar runvar [if] [in] [, c(cutoff) p(pvalue) q(qvalue)
        deriv(dvalue) fuzzy(fuzzyvar [sharpbw]) covs(covars) kernel(kernelfn)
        weights(weightsvar) bwselect(bwmethod) scaleregul(scaleregulvalue)
        vce(vcemethod) all]

Options mirror rdrobust (same `bwselect` and `vce` menus and defaults). `all` reports every selector at once (used p. 400). Same removed options as rdrobust (Section 4.3, p. 386).

*rdplot syntax (Section 5, pp. 386-388):*

    rdplot depvar runvar [if] [in] [, c(cutoff) p(pvalue) kernel(kernelfn)
        weights(weightsvar) h(hvalueL hvalueR) nbins(nbinsvalueL nbinsvalueR)
        binselect(binmethod) scale(scalevalueL scalevalueR) ci(cilevel) shade
        support(supportvalueL supportvalueR) genvars graph_options(gphopts) hide]

| Option | Default | Meaning |
|---|---|---|
| `kernel(kernelfn)` | `uniform` | Kernel for the global polynomial fits; triangular, uniform, or epanechnikov (uniform = equal weighting) — NEW: kernel-weighted global fits |
| `weights(weightsvar)` | none | Unit weights multiplying the kernel |
| `h(hvalueL hvalueR)` | full support of data | Main bandwidth L/R for the global fits; restricting to `[-h, h]` enables exact display of the RD point estimator |
| `nbins(nbinsvalueL nbinsvalueR)` | estimated via `binselect()` | Number of bins J_- / J_+ |
| `binselect(binmethod)` | (carried over; es/esmv evident from examples) | Bin-number selection method (formulas in CCT 2015a, not this paper) |
| `scale(scalevalueL scalevalueR)` | `scale(1 1)` | Bins used = round(scaleL x Jhat_{-,n}), round(scaleR x Jhat_{+,n}) |
| `ci(cilevel)` | off | NEW: per-bin CIs at *cilevel* coverage |
| `shade` | off | Replace CIs with shaded areas |
| `support(supportvalueL supportvalueR)` | sample range | Extended support for bin construction |
| `genvars` | off | Generate per-observation result variables |
| `graph_options(gphopts)` | — | Pass-through graph options |
| `hide` | off | Suppress the plot |

Per-bin CI formula (p. 382, unnumbered display), for bins `j = 1, ..., J_n`:

    CI_j = [ Xbar_j - T_{1-alpha/2} * sqrt( S_j^2 / N_j ) ,
             Xbar_j + T_{1-alpha/2} * sqrt( S_j^2 / N_j ) ]

with `Xbar_j` = bin sample mean (the paper's generic notation; the binned variable in rdplot is the OUTCOME, cf. `rdplot_mean_y`/`rdplot_se_y` below), `S_j^2` = bin sample variance, `N_j` = bin sample size, `T_alpha` = Student's t quantile with `N_j - 1` degrees of freedom; justified preasymptotically, valid up to smoothing bias (Cattaneo and Farrell 2013, partitioning regression).

`genvars` output variables (pp. 387-388): `rdplot_id` (bin ID; negative naturals left of cutoff, positive right), `rdplot_N`, `rdplot_min_bin`, `rdplot_max_bin`, `rdplot_mean_bin` (bin midpoint), `rdplot_mean_x`, `rdplot_mean_y`, `rdplot_se_y` (SD of the bin mean of the outcome), `rdplot_ci_l`, `rdplot_ci_r`, `rdplot_hat_y` (global-polynomial predicted value).

Removed rdplot options (Section 5.3, p. 388): `numbinl()`, `numbinr()`, `scalel()`, `scaler()`, `generate()`, `lowerend()`, `upperend()` — replaced by the paired-argument forms `nbins()`, `scale()`, `genvars`, `support()`.

### Backward compatibility notes (2014 -> 2017 package versions)
- **rdrobust (Section 6, p. 388):** for a GIVEN bandwidth choice and NO covariates, backward compatible by default. POINT estimators identical in all cases; VARIANCE estimators may differ:
  - NN variance (default): identical to the 2014 version in the ABSENCE of ties in `X_i`; slightly different with ties (new deterministic all-equidistant-neighbors procedure vs old random tie-breaking). Truly continuous running variable (no ties) = fully backward compatible.
  - Plug-in residuals variance: NOT backward compatible — new version computes ALL residuals at the cutoff (fast, mimics linear least squares); 2014 version evaluated predictions nonparametrically at points near the cutoff. The 2014 version offered only NN and unweighted plug-in (HC0).
- **rdbwselect (Section 6, p. 389):** NOT backward compatible at all ("we were forced to redo the command completely"). The 2014 version ships as `rdbwselect_2014` inside the upgraded package; its bandwidths can be fed manually into rdrobust (`h()`, `b()`) to reproduce old results — smoke-test target 2 below does exactly this.
- **rdplot:** FULLY backward compatible; options only reorganized for homogeneity with rdrobust/rdbwselect.
- **Deprecated selectors:** `ik`, `cct` superseded by `mserd` (an upgraded version of both); `cv` removed outright.
- **Parity-target implication:** parity-target the 2017 surface (`mserd`, deterministic NN ties, cutoff-evaluated plug-in residuals); the 2014 behaviors are reachable only via manual bandwidths and are covered by smoke-test target 2.

### Smoke-test targets (Senate example)

Dataset: `rdrobust_senate.dta` (Cattaneo, Frandsen, Titiunik 2015). U.S. Senate elections 1914-2010; unit = state at a given election; outcome `vote` (Democratic vote share next election, 0-100), running variable `margin` (Democratic margin of victory, -100 to 100), cutoff c = 0. Covariates: `class` (electoral class 1-3), `termshouse`, `termssenate`, `population`. Also `state`, `year` variables.

Summary stats (p. 390): vote N=1297 mean 52.66627 sd 18.12219; margin N=1390 mean 7.171159 sd 34.32488; class N=1390 mean 2.023022 sd .8231983; termshouse N=1108 mean 1.436823 sd 2.357133 (0-16); termssenate N=1108 mean 4.555957 sd 3.720294 (1-20); population N=1390 mean 3827919 sd 4436950 (78000-3.73e7).

1. **Default** `rdrobust vote margin` (p. 392): N=1297 (595 L / 702 R), eff. 359/322, p=1 q=2, mserd h=17.708, b=27.984, rho=0.633, triangular, vce NN.
   - Conventional: coef **7.416**, SE 1.4604, z 5.0782, p 0.000, 95% CI [4.55378, 10.2783]
   - Robust: z 4.3095, p 0.000, CI **[4.09441, 10.9255]**
   - NOT numerically identical to CCT 2014a/2015b (new bandwidth selector).
2. **Manual 2014 bandwidths** `h(16.79369) b(27.43745)` (p. 393): eff. 343/310, rho=0.612. Conventional 7.4253, SE 1.4954, z 4.9656, CI [4.49446, 10.3561]; Robust z 4.2675, CI [4.06975, 10.9833]. Reproduces CCT 2014a/2015b exactly.
3. **Covariates, fixed bw** `covs(class termshouse termssenate) h(17.708...) b(27.984...)` (p. 395): N=1108, eff. 309/280. Conventional **6.8595**, SE 1.4165, z 4.8426, CI [4.08322, 9.63574]; Robust z 4.1911, CI [3.75238, 10.345]. Robust CI length change: **-3.49%**.
4. **Covariates, data-driven mserd** `covs(class termshouse termssenate)` (p. 396): h=17.987, b=28.943, rho=0.621, eff. 313/283. Conventional 6.8514, SE 1.4081, z 4.8656, CI [4.09148, 9.61125]; Robust z 4.1999, CI [3.72856, 10.2537]. CI length change: **-4.48%**.
5. **Irrelevant covariate** `covs(population)` (p. 397): N=1297, h=17.585, b=27.857, rho=0.631, eff. 359/320. Conventional 7.4376, SE 1.4654, z 5.0754, CI [4.56545, 10.3097]; Robust z 4.3102, CI [4.10714, 10.9573]. CI length change: **+0.28%** (covariates can lengthen CIs).
6. **Covariate balance** (p. 397, rdrobust of each covariate on margin): RD effect / robust p-value: class -.021 / .897; termshouse -.173 / .561; termssenate -.192 / .901; population -318455.26 / .634.
7. **Cluster-robust NN** `vce(nncluster state)` (p. 398): mserd h=17.509, b=27.032, rho=0.648, eff. 359/320, clusters 50/50. Conventional 7.4221, SE 1.5225, z 4.8750, CI [4.43811, 10.4061]; Robust z 4.2659, CI **[4.09109, 11.0456]** (vs het-robust [4.094, 10.926]).
8. **Covariates + msetwo + nncluster** (p. 399): N=1108, h_l=14.661, h_r=20.893, b_l=24.458, b_r=37.338, rho 0.599/0.560, clusters 48/50, eff. 274/310. Conventional **6.8072**, SE 1.3696, z 4.9704, CI [4.12297, 9.49153]; Robust z 4.6153, CI [4.13522, 10.2398].
9. **`rdbwselect vote margin, all`** (p. 400; triangular, vce NN, p=1, q=2, N=595/702) — h(L/R), b(L/R):

   ```
   mserd     17.708/17.708   27.984/27.984
   msetwo    16.154/18.009   27.096/29.205
   msesum    18.326/18.326   31.280/31.280
   msecomb1  17.708/17.708   27.984/27.984
   msecomb2  17.708/18.009   27.984/29.205
   cerrd     12.374/12.374   27.984/27.984
   certwo    11.288/12.585   27.096/29.205
   cersum    12.806/12.806   31.280/31.280
   cercomb1  12.374/12.374   27.984/27.984
   cercomb2  12.374/12.585   27.984/29.205
   ```

   (Header also reports Min/Max of margin per side: left [-100.000, -0.079], right [0.036, 100.000].)
10. **rdplot Figure 1** `binselect(es) ci(95)` (p. 391): uniform kernel, p=4, h=100 both sides; bins selected 8 L / 9 R (= IMSE-optimal); avg bin length 12.5/11.111; mimicking-variance bins 15/35; WIMSE var weight 0.500, bias weight 0.500.
11. **rdplot Figure 2** (p. 394): support restricted to [-17.708, 17.708], `binselect(esmv) kernel(triangular) p(1)`, N=681 (359/322); bins selected 16/18; IMSE-optimal bins 7/7; mimicking-variance bins 16/18; implied scale 2.286/2.571; WIMSE var weight .077/.056, bias weight .923/.944. Gap between fits = exactly 7.416.

Additional parity anchors from the worked example:
- RD plot as exact point-estimator display (Section 7.3, pp. 393-395): restricting rdplot's support to `[-h, h]` with `p(1)`, `kernel(triangular)`, and `h()` equal to the rdrobust bandwidth makes the global polynomial fit reproduce the local-polynomial RD point estimator exactly — "the vertical distance between the two weighted linear polynomial fits is exactly 7.416."
- Fuzzy usage recipes (p. 401): `rdrobust y x, fuzzy(t) covs(z) vce(nncluster cid)` (fuzzy RD); `rdrobust y x, fuzzy(t) deriv(1) covs(z) vce(nncluster cid)` (fuzzy kink); sharp kink = `deriv(1)` without `fuzzy()`.
- Stata `regress` equivalence (p. 401): `kernel(uniform)` + `vce(cluster ...)` closest to clustered `regress`; `vce(hc1)` closest to unclustered `regress`.
- Replication file: `rdrobust_illustration.do` ships with the package (p. 401).

### Relation to Existing diff-diff Estimators
- The library already ports the sibling nprobust package as `diff_diff/_nprobust_port.py` (CCF 2018 MSE-DPI selector + CCT 2014 robust bias correction, NN/HC variance, cluster, weights) used by the HAD estimator; the planned RegressionDiscontinuity estimator (sharp RD v1: p=1, triangular kernel, mserd, vce=nn, three inference flavors) will parity-target THIS paper's package surface.
- The paper's variance menu (NN default, HC0-HC3 plug-in residuals, one-way cluster variants) structurally mirrors the NN/HCk/cluster machinery already in `_nprobust_port.py` — the RD implementation should share or parallel that module rather than re-derive it.
- The recommended covariate balance falsification (run the estimator with each covariate as outcome; smoke-test target 6 gives golden numbers) is the RD analog of the library's placebo-test pattern (PlaceboTests) and can seed an RD balance-check helper.
- The Stata `regress` equivalence notes (uniform kernel + hc1/cluster, p. 401) give a statsmodels-free validation path against diff-diff's existing `solve_ols()` / `compute_robust_vcov()` OLS machinery for the degenerate global-fit case.
- User weights semantics (weights multiply the kernel) match the `weights`-times-kernel convention already used in the nprobust port.

---

## Gaps and Uncertainties

**Deferred to CCFT (2016b; published CCFT 2019 REStat) and its supplemental appendix — NOT printed in this paper:**
- Exact two-bandwidth (different h/b per side) estimation formulas: the p. 376 display assumes equal bandwidths on both sides only to simplify exposition; the general formulas are in the CCFT supplemental appendix (p. 377).
- Exact forms of the preliminary estimates `V_hat_j`, `B_hat_j` entering the plug-in MSE bandwidth selectors, and their asymptotic properties (p. 380).
- Precise HCk plug-in residual variance formulas (finite-sample weights `omega_{-,i}`, `omega_{+,i}`) and ALL explicit cluster-robust variance formulas (pp. 378-379).
- Bias-correction/linearization details for fuzzy RD CIs; the paper characterizes the fuzzy point estimator only as "the ratio of reduced-form RD estimators" (Section 6, p. 389).
- The exact IK-style regularization term: "as introduced in Imbens and Kalyanaraman (2012) but implemented as discussed in Calonico, Cattaneo, and Titiunik (2014a, 2014b, 2015b) and the corresponding supplemental appendix" (p. 400) — formula not printed.
- Fuzzy/kink estimand definitions: "we do not spell out the details for fuzzy and kink RD designs beyond giving a few generic examples at the end of section 7" — details live in the help files and CCFT.

**Unnumbered equations:** all displayed equations in this paper are unnumbered; every formula above is anchored only by journal page and section number. Cross-referencing against R/Stata source must go through those anchors or through CCFT's numbered results.

**Outside the paper entirely:**
- rdplot bin-number selection formulas (evenly-spaced vs quantile-spaced, IMSE-optimal and mimicking-variance counts) are in Calonico, Cattaneo, Titiunik (2015a, JASA) and the earlier software articles; only `binselect(binmethod)` appears in the syntax here. Defaults `c(0)`, `p(4)` and methods `es`/`esmv` are evident only from the examples.
- Default rho handling formula (how data-driven b relates to h when neither is user-set) is not stated; only the observed rho values (0.56-0.65) and the `rho()` override semantics are given.
- `scalepar()` semantics are listed in the syntax (p. 382) but not explained in the reviewed pages.

**Counting tension (not a contradiction, both from the paper):** Section 2.2 (p. 379, extraction 01) reports "10 distinct variance estimators" (NN or plug-in residuals with HC0-HC3, plus NN-adjusted or df-adjusted clustering), while the `vce()` option surface (pp. 385-386, extraction 02) exposes 7 option strings (`nn`, `hc0`-`hc3`, `nncluster`, `cluster`). The mapping from the 10-count to the 7 option strings (e.g., whether cluster forms carry their own HCk/df sub-variants) is not spelled out in the paper — resolve against the R/Stata source when implementing. Similarly, Section 2.3 (p. 381) counts "8 MSE-optimal + 8 CER-optimal choices + two combined selectors" at the methodology level, while the option surface exposes exactly 10 `bwselect()` strings; these are different counting schemes (with/without regularization variants vs option strings), not conflicting catalogs.

**Extraction overlap resolution (pages 385-389 covered by both extractions):** fuzzy option semantics, backward-compatibility notes, removed options, and rdplot syntax appear in both extraction files with consistent content; no contradictions were found between the two extractions. The perfect-compliance auto-switch is worded as "perfect compliance at either side of the threshold" in both (pp. 385, 389).

**Not discussed in paper:** mass-points handling for estimation itself (only the NN-variance tie behavior is covered), degenerate-sample warnings, weak-first-stage diagnostics for fuzzy RD, and any small-sample/degrees-of-freedom inference corrections beyond what HC1-HC3 imply.

**Worked-example pages:** the empirical illustration (Section 7) spans journal pages 389-401; extraction 01 covered through p. 389 and extraction 02 covered pp. 385-403, so the Senate numbers above come solely from extraction 02 (single-sourced, not independently cross-checked).
