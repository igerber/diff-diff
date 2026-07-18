# Paper Review: Optimal Data-Driven Regression Discontinuity Plots

**Authors:** Sebastian Calonico, Matias D. Cattaneo, and Rocio Titiunik
**Citation:** Calonico, S., Cattaneo, M. D., & Titiunik, R. (2015). Optimal Data-Driven Regression Discontinuity Plots. *Journal of the American Statistical Association*, 110(512), 1753-1769. DOI: 10.1080/01621459.2015.1017578
**PDFs reviewed:** papers/Calonico-Cattaneo-Titiunik_2015_JASA.pdf (main paper, 17 pp) AND papers/Calonico-Cattaneo-Titiunik_2015_JASA_Supplemental.pdf (supplemental appendix, dated November 25, 2015). Supplement-derived content carries S-prefixed citations (e.g., "Supplement S.1"); supplement page numbers are the printed ones (printed page = PDF page - 1).
**Review date:** 2026-07-18

---

## Methodology Registry Entry

*Formatted to match docs/methodology/REGISTRY.md structure. Heading levels and labels align with existing entries - copy the `## {EstimatorName}` section into the appropriate category in the registry. The public API name (function vs. result-class) is implementation (PR-B) scope; "RDPlot" is used as a placeholder heading.*

## RDPlot

**Primary source:** Calonico, S., Cattaneo, M. D., & Titiunik, R. (2015). Optimal Data-Driven Regression Discontinuity Plots. *JASA*, 110(512), 1753-1769. https://doi.org/10.1080/01621459.2015.1017578 (+ Supplemental Appendix)

**What this is:** NOT a treatment-effect estimator - an exploratory/graphical companion to RD estimation. An RD plot has two ingredients (Section 1, p. 1753; Section 2.1): (i) two global k-th order polynomial fits of `Y` on `X`, estimated separately with control (`X_i < xbar`) and treatment (`X_i >= xbar`) observations; and (ii) binned local sample means of `Y` over a disjoint partition of the support of `X` on each side of the cutoff. The paper's contribution is the formal, data-driven selection of the number of bins `(J_-, J_+)`, targeted at two distinct goals (Section 1, pp. 1754-1755):

1. **Detection of discontinuities** - IMSE-optimal `J` traces out the underlying regression functions (validating the smooth global fit and highlighting discontinuities away from the cutoff that would cast doubt on the design).
2. **Representation of variability** - larger (undersmoothing) `J` choices so the binned means' variability mimics the raw-data scatter, giving a disciplined "cloud of points."

**Key implementation requirements:**

*Assumption checks / warnings (Assumption 1, Section 2, p. 1756):*
- For `x_l < xbar < x_u` and all `x in [x_l, x_u]`: (a) `E[Y_i^4 | X_i = x]` bounded, and the running variable `X_i` continuously distributed with density `f(x)` continuous and bounded away from zero; (b) `mu_-(x) = E[Y_i(0)|X_i = x]` and `mu_+(x) = E[Y_i(1)|X_i = x]` are `S` times continuously differentiable (`S >= 1`); (c) conditional variances `sigma2_-(x)`, `sigma2_+(x)` continuous and bounded away from zero. Theorems 1-2 require `S >= 2`; the consistency results for the data-driven selectors (Theorems 3-4, Supplement Theorems SA1-SA2) require `S >= 5`.
- **Continuously distributed outcome** required for the spacings-based (concomitant) selectors (Theorems 3-4; Remark 1, p. 1763): with discrete/binary `Y` the concomitant-based estimation "becomes invalid" and the series (polynomial) estimators must be used instead. Implementation should detect (or document) discrete outcomes and route to the series variants.
- Continuity of `X` (Assumption 1(a)) underlies the spacings estimators as well: ties produce zero spacings. The paper does not treat mass points; handling is a software-level seam (see the merged 2017 review for rdrobust's masspoints machinery).
- Rate conditions: ES needs `J log(J)/n -> 0` and `J -> infinity` (Theorem 1; Lemma SA1); QS additionally needs `J/log(n) -> infinity` (Theorem 2; Lemma SA2 - the stronger rate ensures sample-quantile consistency). ES partitions "could lead to empty bins in finite samples (this possibility disappears asymptotically; see Lemma SA1 in the supplemental appendix)" while QS guarantees roughly `N_-/J_-` and `N_+/J_+` observations per bin (Section 4.1, p. 1761).
- Global polynomial order: `k = 4` or `k = 5` "are almost always the preferred choices" in practice (Section 2.1.1, p. 1756); the paper's own figures use "fourth-order polynomial fits using control and treated units separately" (Figure 1 notes, p. 1754; repeated in Figures 2-3 and Supplement Figures SA-1 to SA-6). Nonparametric consistency of the selectors treats `k = k_n -> infinity` with `k_n^7/n -> 0` (Theorems 3-4).

*Partition schemes (Sections 2.1.2, 3, 4):*

Two generic disjoint partitions `P_{-,n} = {P_{-,j}: j = 1, ..., J_{-,n}}` of `[x_l, xbar)` and `P_{+,n} = {P_{+,j}: j = 1, ..., J_{+,n}}` of `[xbar, x_u]`, with

    P_-,j = [x_l, p_-,1)                    j = 1
            [p_-,j-1, p_-,j)                j = 2, ..., J_-,n - 1
            [p_-,J_-,n - 1, xbar)           j = J_-,n
    P_+,j = [xbar, p_+,1)                   j = 1
            [p_+,j-1, p_+,j)                j = 2, ..., J_+,n - 1
            [p_+,J_+,n - 1, x_u]            j = J_+,n

- **Evenly spaced (ES) partition (Section 3, p. 1757):** `p_-,j = x_l + j * (xbar - x_l)/J_-,n` and `p_+,j = xbar + j * (x_u - xbar)/J_+,n`. Nonrandom; every ES bin has length `(xbar - x_l)/J_-,n` (left) or `(x_u - xbar)/J_+,n` (right) - this is the "average bin length" a software implementation reports for ES bins.
- **Quantile spaced (QS) partition (Section 4, p. 1760):** `p_-,j = Fhat_-^{-1}(j/J_-,n)` and `p_+,j = Fhat_+^{-1}(j/J_+,n)`, with the side-specific empirical distribution functions and generalized inverse

      Fhat_-(x) = (1/N_-) * sum_i 1(X_i < xbar) 1(X_i <= x),
      Fhat_+(x) = (1/N_+) * sum_i 1(X_i >= xbar) 1(X_i <= x),
      Fhat^{-1}(y) = inf{x : Fhat(x) >= y}.

  Random partition; bin lengths vary (approximately equal bin *counts* instead). QS "may be interpreted as covariate design adaptive" and is advantageous under sparse data (Section 1, p. 1755).

*Binned (partitioning) estimator (Section 2.1.2, p. 1757):*

    muhat_-(x; J_-,n) = sum_{j=1}^{J_-,n} 1_{P_-,j}(x) * Ybar_-,j,
    Ybar_-,j = (1(N_-,j > 0)/N_-,j) * sum_i 1_{P_-,j}(X_i) Y_i,
    N_-,j = sum_i 1_{P_-,j}(X_i),  N_- = sum_j N_-,j     (analogously for +)

The `1(N_-,j > 0)/N_-,j` convention makes an empty bin's mean zero rather than 0/0 - implementation must handle empty ES bins explicitly (plot-level treatment of an empty bin is a software decision; see Gaps).

*Global polynomial estimation (Section 2.1.1, p. 1756):* with `r_k(x) = (1, x, x^2, ..., x^k)'`,

    muhat_-,k(x) = r_k(x)' betahat_-,k,
    betahat_-,k = argmin_beta sum_i 1(X_i < xbar) (Y_i - r_k(X_i)' beta)^2    (and analogously for +)

*Variance/bias constants - ES (Theorem 1, Section 3.1, p. 1758):* under Assumption 1 with `S >= 2`, `w: [x_l, x_u] -> R_+` continuous, `J_-,n log(J_-,n)/n -> 0`, `J_-,n -> infinity`:

    var_ES,-(J_-,n)  = (J_-,n / n) * V_ES,- * {1 + o_P(1)},
    V_ES,-  = (1/(xbar - x_l)) * integral_{x_l}^{xbar} (sigma2_-(x)/f(x)) w(x) dx,
    Bias_ES,-(J_-,n) = (1/J_-,n^2) * B_ES,- * {1 + o_P(1)},
    B_ES,-  = ((xbar - x_l)^2 / 12) * integral_{x_l}^{xbar} (mu1_-(x))^2 w(x) dx

(`mu1` = first derivative of `mu`; analogous `+` expressions with `x_u - xbar` and integrals over `[xbar, x_u]`). The weighting function may be discontinuous at the cutoff: "All the results presented in this article remain valid if `w(x) = w_+(x) 1(x >= xbar) + w_-(x) 1(x < xbar)`" (p. 1758).

*Variance/bias constants - QS (Theorem 2, Section 4.1, pp. 1760-1761):* under the additional `J/log(n) -> infinity`:

    V_QS,- = (1/P_-) * integral_{x_l}^{xbar} sigma2_-(x) w(x) dx,        P_- = P[X_i < xbar],
    B_QS,- = (P_-^2 / 12) * integral_{x_l}^{xbar} (mu1_-(x)/f(x))^2 w(x) dx

(analogous `+` with `P_+ = P[X_i >= xbar]`). Rates are the same as ES; only the fixed constants differ (Section 4.1, p. 1761).

*Population number-of-bins selectors (Equations 1-6):* all selectors use the ceiling operator `ceil(y)` = smallest integer `>= y`.

- **IMSE-optimal, ES (Equation 1, p. 1758):** minimizing `IMSE_ES,-(J) = (J/n) V_ES,- + (1/J^2) B_ES,-`:

      J_ES-mu,-,n = ceil( (2 B_ES,- / V_ES,-)^{1/3} * n^{1/3} ),
      J_ES-mu,+,n = ceil( (2 B_ES,+ / V_ES,+)^{1/3} * n^{1/3} )

- **Weighted IMSE (WIMSE), ES (Section 3.3.1, Equation 2, p. 1759):** for fixed weights `omega_V,- + omega_B,- = 1` (both > 0), minimizing `WIMSE_ES,-(J) = omega_V,- var_ES,-(J) + omega_B,- Bias_ES,-(J)`:

      J_ES-omega,-,n = ceil( omega_- * (2 B_ES,- / V_ES,-)^{1/3} * n^{1/3} ),   omega_- = (omega_B,- / omega_V,-)^{1/3}
      J_ES-omega,+,n = ceil( omega_+ * (2 B_ES,+ / V_ES,+)^{1/3} * n^{1/3} ),   omega_+ = (omega_B,+ / omega_V,+)^{1/3}

  Equal weights recover Equation 1: `J_ES-omega = J_ES-mu` when `omega_V = omega_B = 1/2`. Equivalently `J_ES-omega,-,n = ceil(omega_- * J_ES-mu,-,n)` (Supplement S.1, p. 2).

  **Implied-weights inverse map (p. 1759; Supplement S.1, p. 2)** - the formula behind the "var weight / bias weight" pair reported by the rdrobust software for ANY scale factor: for each choice of rescaling constants `omega_-, omega_+ > 0` there exists a unique compatible weighting scheme

      (omega_V,-, omega_B,-) = ( 1/(1 + omega_-^3),  omega_-^3/(1 + omega_-^3) ),
      (omega_V,+, omega_B,+) = ( 1/(1 + omega_+^3),  omega_+^3/(1 + omega_+^3) )

  "which rationalizes the resulting choices of number of bins as optimal in the sense of minimizing the WIMSE loss function" (p. 1759). This also gives an objective interpretation to any ad hoc bin choice: given a used `J` and the IMSE-optimal `J_ES-mu`, the implied scale is `omega = J / J_ES-mu` and the implied WIMSE weights follow from the map ("for any initial ad hoc choice of number of bins used to construct the ES RD plot, the above logic can be used to find an IMSE-optimal choice and a scaling factor that is consistent with the ad hoc choice," p. 1759). Supplement S.1 (p. 2) tabulates the implied weights for common scales and confirms "Our software implementations in R and Stata compute this weights explicitly as part of the standard output":

      omega:    0.1     0.2     0.5     1       2       5       10
      omega_V:  0.999   0.992   0.889   0.500   0.111   0.008   0.001
      omega_B:  0.001   0.008   0.111   0.500   0.889   0.992   0.999

- **Mimicking variance, ES (Section 3.3.2, Equation 3, p. 1760):** choose `J` so the integrated variability of the binned means matches the raw-data variability: with `V_-`, `V_+` = the sample variances of the outcome subsamples `{Y_i : X_i < xbar}` and `{Y_i : X_i >= xbar}`, solving `var_ES,-(J_-,n) = V_-` gives "optimal" choice `(V_-/V_ES,-) n`; to satisfy the rate conditions of Theorem 1 it is modified to

      J_ES-theta,-,n = ceil( (V_- / V_ES,-) * n / (log n)^2 ),
      J_ES-theta,+,n = ceil( (V_+ / V_ES,+) * n / (log n)^2 )

  which "mimics the overall variability of the data, up to a log(n) factor" and is in general larger than Equation 1 (undersmoothing; Section 3.3, p. 1760). (The paper's subscript is the Greek letter vartheta; transliterated `theta` here.)

- **QS analogs (Equations 4, 5, 6, p. 1761):** identical structure with `QS` constants: `J_QS-mu = ceil((2 B_QS/V_QS)^{1/3} n^{1/3})` (Eq 4), `J_QS-omega = ceil(omega * (2 B_QS/V_QS)^{1/3} n^{1/3})` with the same `omega = (omega_B/omega_V)^{1/3}` and inverse map (Eq 5), `J_QS-theta = ceil((V/V_QS) * n/(log n)^2)` (Eq 6).

*Data-driven implementation with recommended `w(x) = f(x)` (Section 5, pp. 1761-1764):* the recommended weighting "removes a density from the denominators of the unknown constants and leads to particularly simple and intuitive data-driven rules" (p. 1762). Simplified targets (Section 5.1, p. 1762):

    V_ES,- = (1/(xbar - x_l)) * integral sigma2_-(x) dx,
    B_ES,- = ((xbar - x_l)^2 / 12) * E[ 1(X_i < xbar) (mu1_-(X_i))^2 ]      (analogous +)

Order-statistics / concomitants notation (Section 5, p. 1762): within each side's subsample, `X_-,(i)` is the i-th order statistic of the control-side running variable and `Y_-,[i]` its concomitant (the `Y` paired with `X_-,(i)`); midpoints `Xbar_-,(i) = (X_-,(i) + X_-,(i-1))/2, i = 2, ..., N_-`; derivative regressor `r1_k(x) = d r_k(x)/dx = (0, 1, 2x, 3x^2, ..., k x^{k-1})'` and `mu1hat_-,k(x) = r1_k(x)' betahat_-,k` (analogous +).

- **ES spacings estimators (Equations 7-10, p. 1762):**

      Vhat_ES,- = (1/(xbar - x_l)) * (1/2) * sum_{i=2}^{N_-} (X_-,(i) - X_-,(i-1)) * (Y_-,[i] - Y_-,[i-1])^2      (7)
      Bhat_ES,- = ((xbar - x_l)^2 / (12 n)) * sum_{i=1}^{n} 1(X_i < xbar) * (mu1hat_-,k(X_i))^2                   (8)
      Vhat_ES,+ = (1/(x_u - xbar)) * (1/2) * sum_{i=2}^{N_+} (X_+,(i) - X_+,(i-1)) * (Y_+,[i] - Y_+,[i-1])^2      (9)
      Bhat_ES,+ = ((x_u - xbar)^2 / (12 n)) * sum_{i=1}^{n} 1(X_i >= xbar) * (mu1hat_+,k(X_i))^2                  (10)

  Spacings estimators are preferred "whenever possible" because they (i) avoid explicit estimation of the density `f(x)` and (ii) require no additional tuning-parameter choices (p. 1762).

- **ES spacings selectors (Equations 11-14, p. 1763, left column):**

      Jhat_ES-mu,-,n    = ceil( (2 Bhat_ES,- / Vhat_ES,-)^{1/3} * n^{1/3} )        (11, with + analog)
      Jhat_ES-omega,-,n = ceil( omega_- * (2 Bhat_ES,- / Vhat_ES,-)^{1/3} * n^{1/3} )   (12)
      Jhat_ES-omega,+,n = ceil( omega_+ * (2 Bhat_ES,+ / Vhat_ES,+)^{1/3} * n^{1/3} )   (13)
      Jhat_ES-theta,-,n = ceil( (Vhat_- / Vhat_ES,-) * n / (log n)^2 )             (14, left column; with + analog)

  where `Vhat_-`, `Vhat_+` are consistent estimators of `V_-`, `V_+` (the side-wise outcome sample variances).

- **ES series (polynomial) estimators for noncontinuous outcomes (Remark 1, p. 1763):** when `Y_i(0), Y_i(1)` are not continuously distributed the concomitant-based estimation is invalid; assuming `E[Y_i(t)^2 | X_i = x]`, `t = 0, 1`, twice continuously differentiable, use the variance-function series approximation

      sigma2hat_-,k(x) = muhat_-,k,2(x) - (muhat_-,k,1(x))^2,
      muhat_-,k,p(x) = r_k(x)' betahat_-,k,p,
      betahat_-,k,p = argmin_beta sum_i 1(X_i < xbar) (Y_i^p - r_k(X_i)' beta)^2,   p = 1, 2     (analogous +)

  (`muhat_-,k(x) = muhat_-,k,1(x)` in the base notation), giving

      Vcheck_ES,- = (1/(xbar - x_l)) * sum_{i=2}^{N_-} (X_-,(i) - X_-,(i-1)) * sigma2hat_-,k(Xbar_-,(i))     (analogous +)

  and series selectors `Jcheck_ES-mu` / `Jcheck_ES-omega` / `Jcheck_ES-theta` of identical form to (11)-(14) with `Vcheck` in place of `Vhat` - printed as Equations (14)-(16) on p. 1763, right column. **Published numbering quirk:** the equation label "(14)" appears TWICE on p. 1763 - once for the spacings `Jhat_ES-theta` pair (left column) and once for the series `Jcheck_ES-mu` pair (right column). Cite these as "(14, left col.)" and "(14, right col.)".

- **Consistency, ES (Theorem 3, p. 1763):** under Assumption 1 with `S >= 5`, `Y(0), Y(1)` continuously distributed, `k_n^7/n -> 0`, `k_n -> infinity`, and `Vhat_+/- ->_P V_+/-`: `Jhat_ES-omega/J_ES-omega ->_P 1`, `Jhat_ES-theta/J_ES-theta ->_P 1` (both sides; `Jhat_ES-mu = Jhat_ES-omega` at equal weights). The series selectors `Jcheck` are consistent in the same sense under the same conditions (p. 1763).

- **QS spacings estimators (Equations 17-20, p. 1764):**

      Vhat_QS,- = (1/(2 N_-)) * sum_{i=2}^{N_-} (Y_-,[i] - Y_-,[i-1])^2                                          (17)
      Bhat_QS,- = (N_-^2 / (24 n)) * sum_{i=2}^{N_-} (X_-,(i) - X_-,(i-1))^2 * (mu1hat_-,k(Xbar_-,(i)))^2        (18)
      Vhat_QS,+ = (1/(2 N_+)) * sum_{i=2}^{N_+} (Y_+,[i] - Y_+,[i-1])^2                                          (19)
      Bhat_QS,+ = (N_+^2 / (24 n)) * sum_{i=2}^{N_+} (X_+,(i) - X_+,(i-1))^2 * (mu1hat_+,k(Xbar_+,(i)))^2        (20)

  (note the QS variance estimator uses NO spacings - just successive concomitant differences - while the QS bias estimator uses squared spacings evaluated at midpoints.)

- **QS selectors (Equations 21-23, p. 1764):** same form as ES: `Jhat_QS-mu` (21), `Jhat_QS-omega` (22), `Jhat_QS-theta = ceil((Vhat/Vhat_QS) n/(log n)^2)` (23).

- **QS series variants (Remark 2, p. 1764):** for noncontinuous outcomes, `Vcheck_QS,- = (1/N_-) * sum_i 1(X_i < xbar) sigma2hat_-,k(X_i)` (analogous +), with selectors `Jcheck_QS-mu` / `Jcheck_QS-omega` / `Jcheck_QS-theta` printed as Equations (24)-(26).

- **Consistency, QS (Theorem 4, p. 1764):** same conditions as Theorem 3; the QS spacings and series selectors' ratios `->_P 1`.

*Arbitrary known `w(x)` (Supplement S.3, pp. 12-18):* generic-weight versions of every estimator, with DIFFERENT powers-of-spacings and constants than the `w = f` main-text forms (the Lemma SA3 change-of-variables absorbs different powers of `f`):

    Vhat_ES,- = (1/(xbar - x_l)) * (n/4) * sum_{i=2}^{N_-} (X_-,(i) - X_-,(i-1))^2 (Y_-,[i] - Y_-,[i-1])^2 w(Xbar_-,(i))   (SA-1)
    Bhat_ES,- = ((xbar - x_l)^2 / 12) * sum_{i=2}^{N_-} (X_-,(i) - X_-,(i-1)) (mu1hat_-,k(Xbar_-,[i]))^2 w(Xbar_-,(i))     (SA-2)

(SA-3/SA-4 = `+` analogs; selectors SA-5 to SA-7 and Theorem SA1 for ES spacings; series versions with `sigma2hat` and selectors SA-8 to SA-10; QS: `Vhat_QS,- = (n/(2 N_-)) sum spacing * (delta Y)^2 * w` (SA-11), `Bhat_QS,- = (N_-^2/72) sum spacing^3 * (mu1hat(Xbar))^2 * w` (SA-12), SA-13/SA-14 `+` analogs, selectors SA-15 to SA-17, Theorem SA2, series QS and SA-18 to SA-20.) "In practice, the choice w(x) = 1 is arguably the simplest one" (Supplement S.3.2, p. 17). The spacings machinery rests on Lemma SA3 (Supplement S.2.3, p. 6): for `l in Z_+`, continuous `g`,

    N_+^{l-1} * sum_{i=2}^{N_+} (X_+,(i) - X_+,(i-1))^l g(Xbar_+,(i))  ->_P  l! * P_+^{l-1} * integral f(x)^{1-l} g(x) dx     (i)
    N_+^{l-1} * sum (spacing)^l (Y_+,[i] - Y_+,[i-1])^2 g(Xbar_+,(i))  ->_P  l! * P_+^{l-1} * 2 * integral f(x)^{1-l} sigma2_+(x) g(x) dx   (ii)

- the `l!` and the factor 2 in (ii) are what generate the 1/2, /12, /24, /72 constants across (7)-(20) and (SA-1)-(SA-14).

*Edge cases:*
- **Empty ES bins** (finite samples): possible by construction; `Ybar` convention zeroes the mean; Lemma SA1(i) (Supplement S.2.1, pp. 3-4) shows `P(N_+,j > 0) -> 1` uniformly in `j` under the ES rate conditions -> handle/flag empty bins in implementation.
- **Discrete/noncontinuous outcomes:** spacings (concomitant) selectors invalid -> series selectors (Remarks 1-2); requires `E[Y^2|X]` twice continuously differentiable.
- **ES vs QS:** "Our proposed selection rules for ES and QS partitioning coincide when the underlying distribution of X_i is uniform. In general, however, neither partitioning approach dominates the other in terms of asymptotic variability or IMSE" (Section 6.3, p. 1768); QS tends to do better "when the density f(x) is low in some regions of the support." ES and QS "do not necessarily deliver different number of bins" (Supplement S.4.4, pp. 20-21: Head Start mimicking-variance choices essentially identical).
- **Fuzzy RD (Section 7.1, p. 1768):** all results apply unchanged to the reduced form (`Y` on `X`); imposing Assumption-1-analog conditions on `E[T_i(t)|X_i = x]`, they also apply to the first stage - "simply changing the outcome variable used (Y_i or T_i)". The first-stage plot's outcome is binary, so its spacings selectors are invalid -> use the series (Remark 1/2) selectors for a first-stage RD plot.
- **Covariates (Section 7.2, p. 1768):** two discussion-level approaches for sharp designs - (i) conditioning (apply the results within subsamples defined by discrete covariates; continuous conditioning "is hard"), (ii) GAM/backfitting removal of additive covariate effects, then RD-plot the adjusted outcome. No formal results; not part of the proposed selectors.
- **Weight discontinuity at the cutoff** is explicitly allowed (p. 1758), consistent with the everything-is-per-side structure.

*Not in this paper (cross-references):* per-bin confidence intervals for the binned means, the `binselect` menu naming (es/espr/esmv/qs/qspr/qsmv and the `*mvpr` combinations), `genvars`-style output fields, bin-count reporting conventions, and support-restriction options are software-surface features documented in the merged 2017 Stata Journal review (`calonico-cattaneo-farrell-titiunik-2017-review.md`); the per-bin CI formula appears there (Cattaneo-Farrell partitioning justification), not in this 2015 paper. This paper supplies the selector formulas that the 2017 paper's software section defers to ("formulas in CCT 2015a").

*Empirical smoke targets (transcribe-exact; for future golden anchoring):*

1. **U.S. House (Lee 2008) data (Sections 1, 6.1; Figures 1-3):** `n = 6558`, `N_- = 2740`, `N_+ = 3818`, cutoff `xbar = 0`; forcing variable = Democratic margin of victory, outcome = Democratic vote share in the following election; support stated as `x_l = -100, x_u = 100` in the Section 2.1.2 example (figure axes rendered on a [-1, 1] scale; see Gaps). Figure 1's ad hoc 250/100/20-bins-per-side variants (p. 1754) are motivation only - context, not selector output.
   - **Figure 2 (ES, p. 1766):** (b) mimicking variance, spacings: `Jhat_ES-theta,- = 84`, `Jhat_ES-theta,+ = 130`; (c) IMSE-optimal, spacings: `Jhat_ES-mu,- = 20`, `Jhat_ES-mu,+ = 17`; (e) mimicking variance, series: `Jcheck_ES-theta,- = 87`, `Jcheck_ES-theta,+ = 145`; (f) IMSE-optimal, series: `Jcheck_ES-mu,- = 20`, `Jcheck_ES-mu,+ = 17`.
   - **Figure 3 (QS, p. 1766):** (b) MV spacings: `119 / 144`; (c) IMSE spacings: `48 / 19`; (e) MV series: `118 / 137`; (f) IMSE series: `48 / 19`.
2. **U.S. Senate data (Supplement S.4.1, pp. 18-19; Figures SA-1/SA-2, pp. 22-23):** `n = 1297` state-year observations (1914-2010), `N_- = 595`, `N_+ = 702`; running variable = state-level Democratic margin of victory for a Senate seat, outcome = Democratic vote share in the following election for the same seat (six years later), `xbar = 0`. **This is the same Cattaneo-Frandsen-Titiunik Senate dataset already vendored in `benchmarks/data/` for the RegressionDiscontinuity goldens.**
   - **Figure SA-1 (ES):** (b) MV spacings: `Jhat_ES-theta,- = 15`, `Jhat_ES-theta,+ = 35`; (c) IMSE spacings: `Jhat_ES-mu,- = 8`, `Jhat_ES-mu,+ = 9`; (e) MV series: `21 / 36`; (f) IMSE series: `8 / 9`. (Consistent with the 2017 paper's Figure 1 as recorded in the 2017 review: IMSE-optimal 8/9, mimicking-variance 15/35.)
   - **Figure SA-2 (QS):** (b) MV spacings: `28 / 49`; (c) IMSE spacings: `21 / 16`; (e) MV series: `29 / 50`; (f) IMSE series: `21 / 16`.
3. **Progresa/Oportunidades (Supplement S.4.2, pp. 19-20; Figures SA-3/SA-4, pp. 24-25):** urban RD, `N_- = 691` control households, `N_+ = 2118` intention-to-treat households, `n = 2809`, `X in [-2.25, 4.11]` (poverty index, centered cutoff 0), outcome = per capita non-food consumption two years post.
   - **Figure SA-3 (ES):** MV spacings `69 / 59`; IMSE spacings `7 / 9`; MV series `77 / 67`; IMSE series `7 / 9`.
   - **Figure SA-4 (QS):** MV spacings `45 / 46`; IMSE spacings `36 / 14`; MV series `46 / 47`; IMSE series `36 / 14`.
4. **Head Start (Supplement S.4.3, p. 20; Figures SA-5/SA-6, pp. 26-27):** county 1960 poverty rate, cutoff `xbar = 59.198` (300th poorest county), outcome = child mortality per 100,000 (ages 5-9, Head Start-related causes, 1973-1983, Ludwig-Miller 2007); `N_- = 2810`, `N_+ = 294`, `n = 3104`.
   - **Figure SA-5 (ES):** MV spacings `45 / 42`; IMSE spacings `6 / 6`; MV series `39 / 52`; IMSE series `6 / 6`.
   - **Figure SA-6 (QS):** MV spacings `48 / 47`; IMSE spacings `31 / 6`; MV series `49 / 49`; IMSE series `31 / 6`.

*Simulation smoke targets (Section 6.2, pp. 1765-1768; Table 1, p. 1767):* model `Y_i = mu(X_i) + eps_i`, `X_i ~ Uniform(-1, 1)`, `eps_i ~ Normal(0, 1)` iid, with the Lee-data-derived quintic

    mu(x) = 0.48 + 1.27 x + 7.18 x^2 + 20.21 x^3 + 21.54 x^4 + 7.33 x^5     if x < 0
            0.52 + 0.84 x - 3.00 x^2 + 7.99 x^3 - 9.01 x^4 + 3.56 x^5      if x >= 0

(simplest of 16 DGPs; full set in Supplement S.5.) Key Table 1 facts:
- Panel A: the finite-sample IMSE over the bins grid is minimized exactly at the theoretical IMSE-optimal choices; normalized IMSE = 1.000 at `J_-,n = 24, 25` and `J_+,n = 16` for both ES and QS panels. At the estimated selectors: `Jhat_ES-mu,-` ratio 1.033 (spacings) / `Jcheck` 1.034 (polynomial); `+` side 0.9435 / 0.9428; QS: 1.072 / 1.073 and 0.9351 / 0.9347.
- Panel B (population parameters `J_ES-mu,- = 25`, `J_ES-theta,- = 118`, `J_ES-mu,+ = 16`, `J_ES-theta,+ = 116`; identical values for QS): the data-driven selectors are well centered and concentrated, e.g. `Jhat_ES-mu,-`: min 22, 1st qu. 25, median 26, mean 25.95, 3rd qu. 27, max 29, sd 0.93; `Jcheck_ES-mu,-`: 23 / 25 / 26 / 25.93 / 26 / 29 / 0.87 (text confirms "sample means are 25.95 and 25.93 ... standard deviation are 0.93 and 0.87", pp. 1767-1768).

**Reference implementation(s):**
- R: `rdrobust::rdplot()` - "All results are readily available in R and STATA using the software packages described in Calonico, Cattaneo, and Titiunik" (abstract, p. 1753; Section 6, p. 1765, citing the 2014a Stata Journal and 2015 R Journal software papers)
- Stata: `rdplot`

**Requirements checklist:**
- [ ] ES and QS partition construction (exact endpoint formulas; `Fhat^{-1}` = inf-form generalized inverse for QS)
- [ ] Per-side binned means with the `1(N_j > 0)/N_j` empty-bin convention
- [ ] Global polynomial fits per side (default order 4)
- [ ] Spacings estimators (7)-(10) and (17)-(20) with order statistics + concomitants + midpoints
- [ ] Series estimators (Remarks 1-2: regress `Y` and `Y^2` on `r_k`, variance-function difference)
- [ ] All selector variants: {ES, QS} x {IMSE-optimal, mimicking-variance} x {spacings, series}, plus the omega/WIMSE generalization with the implied-weights inverse map
- [ ] Scale-factor reporting: implied `omega`, implied `(omega_V, omega_B)` per side
- [ ] Discrete-outcome detection or routing note (spacings invalid for noncontinuous `Y`)
- [ ] Golden anchoring vs the paper's Senate figures (vendored dataset) and Lee/Progresa/Head Start numbers where data are available

---

## Implementation Notes

### Data Structure Requirements
- Cross-section `(Y_i, X_i)` with scalar continuous running variable and known cutoff (same input surface as `RegressionDiscontinuity`)
- Per-side subsamples; order statistics of `X` with concomitant `Y` (sort once per side); midpoints of successive order statistics
- Optional fuzzy first-stage plot: same machinery with `T_i` as outcome (series selectors, since `T` is binary)

### Computational Considerations
- Sorting dominates: O(n log n) per side; all spacings sums are O(n)
- Global polynomial fits: two OLS problems with `k + 1` parameters (Vandermonde design; the `Y^2` regression in the series variance path still uses the same degree-`k` basis `r_k(x)` - it is the squared fitted mean `(muhat_-,k,1(x))^2` inside `sigma2hat` that is a degree-`2k` polynomial when evaluated - numerically prefer a scaled/centered basis internally if conditioning warrants, while matching reference output)
- No simulation/bootstrap; everything is closed-form given the two polynomial fits

### Tuning Parameters

| Parameter | Type | Default | Selection Method |
|-----------|------|---------|-----------------|
| `J_-, J_+` (bins per side) | int | data-driven | Equations (11)-(26): IMSE-optimal or mimicking-variance, spacings or series, ES or QS |
| `k` (global poly order) | int | 4 | "k = 4 or k = 5 are almost always the preferred choices" (Section 2.1.1, p. 1756); figures use 4 |
| `omega_-, omega_+` (scale) | float > 0 | 1 | WIMSE rescaling of the IMSE-optimal choice; implied weights via the S.1 inverse map |
| `w(x)` (weighting) | function | `f(x)` implicitly | Section 5 recommendation (spacings absorb the density); arbitrary known `w` via Supplement S.3; `w = 1` simplest explicit choice |
| partition scheme | ES / QS | ES (most common in the literature, Section 3) | neither dominates (Section 6.3); QS advantageous under sparse regions |

### Relation to Existing diff-diff Estimators
- Companion diagnostic to `RegressionDiscontinuity` (`diff_diff/rdd.py`) - the REGISTRY v1 scope-seams Note already names the rdplot diagnostic as a documented follow-up
- The vendored Senate CSV in `benchmarks/data/` is the S.4.1 dataset: the supplement's Figure SA-1/SA-2 bin counts are direct golden anchors requiring no new data
- Global polynomial fits can reuse the library's OLS core (`solve_ols`); the sorted-side/order-statistics utilities are new but small
- Rendering should follow the established lazy-matplotlib pattern (`honest_did.py` / `pretrends.py` `.plot(ax=...)`), keeping numpy/pandas/scipy-only hard dependencies

---

## Gaps and Uncertainties

1. **Duplicate equation label "(14)" (p. 1763).** The published paper labels both the spacings `Jhat_ES-theta` display (left column) and the series `Jcheck_ES-mu` display (right column) as Equation (14); the series ES selectors then run (14)-(16). Any citation of "Equation 14" must disambiguate by column. Transcribed as printed.
2. **Supplement SA-2 subscript inconsistency (Supplement S.3.1, p. 13).** (SA-2) prints the derivative-fit argument as `mu1hat_-,k(Xbar_-,[i])` (concomitant-style square bracket on an X-quantity) where every neighboring display uses `Xbar_-,(i)`; almost surely a typo for `(i)`. Transcribed as printed above with this flag.
3. **Two consistent forms of the ES bias estimator.** Main text (8)/(10) (with `w = f`) average `(mu1hat)^2` over the raw observations (`(1/n) sum 1(side) (mu1hat(X_i))^2`), while the generic-`w` supplement version (SA-2)/(SA-4) uses spacing-weighted midpoint evaluation. Both estimate the same constant (Lemma SA3); they are numerically different estimators in finite samples. Which form the reference software uses is a PR-B verification item against the rdrobust 4.0.0 source - do not assume either from this paper.
4. **Lee-data support units.** Section 2.1.2's worked example states `x_l = -100, x_u = 100` (percentage points, 200 half-point ad hoc bins), while Figures 1-3 render the running variable on a `[-1, 1]` axis. The paper does not reconcile the scaling; record both and resolve against the software/dataset in PR-B (bin counts are scale-invariant for ES only if the support endpoints are consistently scaled).
5. **Per-bin confidence intervals are absent here.** The 2015 paper's plots contain binned means and global fits only; the per-bin CI construction shipped in rdrobust is documented in the 2017 Stata Journal paper (see the merged 2017 review). A registry entry for the implementation must source CIs from the 2017 paper, not this one.
6. **Supplement coverage.** Reviewed in full: S.1 (implied weights), S.2 lemma statements (SA1, SA2, SA3), S.3 (arbitrary-`w` implementations, Theorems SA1-SA2, Equations SA-1 to SA-20), S.4 (empirical applications and Figures SA-1 to SA-6). Skimmed (proof detail only, no additional implementable content transcribed): S.2 proofs (printed pp. 4-12). NOT reviewed in detail: S.5 "Complete Simulation Results" (16 DGPs, printed pp. 28-47) and S.6 "Numerical Comparison of Partitioning Schemes" (exact finite-sample IMSE formulas for ES vs QS, printed p. 48 ff.) - the main text's Section 6.2-6.3 summaries and Table 1 are transcribed above; consult S.5/S.6 only if PR-B needs additional simulation benchmarks.
7. **Bin-count echo conventions** (what "average bin length" means for QS output, `binselect` naming, `genvars` fields, scale reporting format) are software-surface items outside this paper - the 2017 review documents them; the formulas here are what they compute.
