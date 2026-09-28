# Paper Review: Simple Local Polynomial Density Estimators

**Authors:** Matias D. Cattaneo, Michael Jansson, Xinwei Ma
**Citation:** Cattaneo, M. D., Jansson, M., & Ma, X. (2020). Simple Local
Polynomial Density Estimators. *Journal of the American Statistical
Association*, 115(531), 1449-1455. https://doi.org/10.1080/01621459.2019.1635480
**PDFs reviewed:** `papers/Cattaneo-Jansson-Ma_2020_JASA.pdf` (main article,
7 pp.) and `papers/Cattaneo-Jansson-Ma_2020_JASA--Supplement.pdf` (supplemental
appendix, 44 printed pp.: full theory, proofs, and simulations; dated
June 7, 2019)
**Review date:** 2026-08-15

**Numbering convention:** the SA does NOT use "SA-" prefixes. Its results are
plain-numbered Lemma 1-14, Theorem 1-3, Corollary 1-2, Remark 1-6, Sections 1-7,
and exactly one numbered display (equation (1), inside Lemma 12). Labels below
cite "main text" or "SA" explicitly. The SA writes the empirical distribution
function as `F_tilde` (an F with a tilde); the main text writes `F_hat` for
the same object. This review uses the SA's `F_tilde` throughout.

---

## Methodology Registry Entry

*Formatted to match docs/methodology/REGISTRY.md structure. The class name
`RDDensityTest` is a placeholder - the implementation PR decides final naming
(diagnostic family, like `RDPlot`).*

## RDDensityTest (manipulation testing via density discontinuity)

**Primary source:** [Cattaneo, M. D., Jansson, M., & Ma, X. (2020). Simple
Local Polynomial Density Estimators. *JASA*, 115(531),
1449-1455.](https://doi.org/10.1080/01621459.2019.1635480) (+ Supplemental
Appendix). Full formula extraction in
`docs/methodology/papers/cattaneo-jansson-ma-2020-review.md`.

**What it is:** two nested contributions. (1) A boundary-adaptive
nonparametric **density estimator**: instead of smoothing a histogram, run a
kernel-weighted local polynomial regression of the *empirical distribution
function* on a polynomial in `(x_i - x)`; the density estimate is the slope
coefficient. No prebinning, no boundary-specific transformation, one tuning
parameter (the bandwidth). (2) A **manipulation test** for RD designs
(McCrary 2008's question): estimate the density just below and just above the
cutoff with this estimator and t-test the difference, with robust
bias-corrected (RBC) inference. This is the methodology behind R/Stata
`rddensity`.

**Key implementation requirements:**

*Assumption checks / warnings:*
- **Continuous running variable (no mass points)**: SA Assumption 1's
  smooth-CDF-with-positive-density requirement excludes discrete
  distributions and point masses, and neither reviewed document provides a
  mass-point adjustment. An implementation must detect duplicated support
  values (rounded/discrete running variables) and FAIL CLOSED - refuse
  Theorem 2 / RBC inference on such data until a reviewed
  rddensity-compatible mass-point procedure exists. (The
  RegressionDiscontinuity registry section already treats mass points as a
  first-class inference concern; this diagnostic must not be the one RD
  surface that silently ignores them.)
- **i.i.d. random sample**: both Assumption 1 versions open with `{x_i}`
  being a random sample - cross-observation independence is what Theorem 2's
  variance estimator and the standard-normal calibration of the manipulation
  tests (Corollaries 1-2) rest on. Clustered, survey-weighted, or
  repeated-observation dependence is UNSUPPORTED by the reviewed theory; the
  eventual API must reject such inference requests rather than silently
  apply the unclustered Theorem 2 variance.
- DGP smoothness/positivity, two strengths: **main text Assumption 1** is
  local - `F` is `p+1` times continuously differentiable *in a neighborhood
  of the evaluation point* and `f` is positive *at* that point. **SA Assumption 1**
  is the stronger global condition the SA theorems actually use -
  `F in C^{alpha_x}(X)` on the whole support `X = [x_L, x_U]` with
  `f(x) = F^{(1)}(x) > 0` for all `x in X` (the SA also notes infinite
  support endpoints are possible in principle; finite endpoints are adopted
  for the boundary exposition). `alpha_x >= p+1` for the main results;
  bandwidth selection additionally needs `alpha_x >= p+2` (the two-term bias
  expansion of SA Lemma 5).
- Main text Assumption 2 / SA Assumption 2 (kernel): nonnegative, symmetric,
  continuous on compact support `[-1, 1]`, integrates to one. Unbounded-support
  kernels (Gaussian) are excluded by construction.
- Manipulation testing: Assumptions 1-2 hold **separately** on
  `X_- = [x_L, x_bar)` and `X_+ = [x_bar, x_U]` - derivatives of `F` need not
  be continuous across the cutoff (SA Section 4).
- Bandwidth rate conditions: estimation `h -> 0`, `nh^2 -> inf`,
  `nh^{2p+1} = O(1)` (SA Theorem 1); the undersmoothed test additionally needs
  `n min{h_-^2, h_+^2} -> inf` and `n max{h_-^{1+2p}, h_+^{1+2p}} -> 0`
  (main text Section 4; SA Corollaries 1-2 with a common `h`).

*Density estimator (main text Section 2; SA Section 1):*

    F_tilde(x)   = (1/n) sum_i 1[x_i <= x]                       (EDF)
    beta_hat_p(x) = argmin_{b in R^{p+1}}
                      sum_i ( F_tilde(x_i) - r_p(x_i - x)' b )^2 K((x_i - x)/h)
    f_hat(x)     = e_1' beta_hat_p(x)

with `r_p(u) = (1, u, ..., u^p)'`, `e_v` the `(v+1)`-th unit vector, `p >= 1`.
The same fit delivers the smoothed CDF (`v = 0`) and higher density
derivatives (`F_hat_p^(v)(x) = v! e_v' beta_hat_p(x)`, `v <= p`). Recommended
order: **`p = 2`** ("analogous to local linear regression"; main text
Section 3 end, SA Section 6).

*Asymptotics (SA Theorem 1; main text Theorem 1):* bias
`B_{p,v}(x) = v! [F^{(p+1)}(x)/(p+1)!] e_v' S_{p,x}^{-1} c_{p,x}`, variance
`V_{p,v}(x) = (v!)^2 f(x) e_v' S_{p,x}^{-1} Gamma_{p,x} S_{p,x}^{-1} e_v`
(`v >= 1`), where `S_{p,x}`, `c_{p,x}`, `Gamma_{p,x}` are kernel moment
matrices integrated over `[(x_L - x)/h, (x_U - x)/h]` - the truncation of the
integration region is the entire boundary adaptation. At interior points with
`p - v` even the leading bias vanishes and the `h^{p+2-v}` second-order term
(SA Lemma 5) takes over; **at boundary points the leading bias never
vanishes** (SA Section 3 bias-order table).

*Variance estimation (main text Theorem 2; SA Theorem 2):* fully automatic,
boundary-adaptive, no knowledge of boundary location needed:

    S_hat_{p,x}     = (1/n) sum_i r_p(x_check_i) r_p(x_check_i)' K_h(x_i - x)
    Gamma_hat_{p,x} = (1/n^3) sum_{i,j,k} r_p(x_check_j) r_p(x_check_k)'
                        K_h(x_j - x) K_h(x_k - x)
                        (1[x_i <= x_j] - F_tilde(x_j)) (1[x_i <= x_k] - F_tilde(x_k))
    V_hat_{p,v}(x)  = (v!)^2 e_v' N_x S_hat_{p,x}^{-1} Gamma_hat_{p,x}
                        S_hat_{p,x}^{-1} N_x e_v   ->_P   V_{p,v}(x)

with `x_check_i = (x_i - x)/h`, the scaled-kernel convention
`K_h(u) = h^{-1} K(u/h)`, and the interior/boundary scaling matrix from
SA Lemma 3:

    N_x = diag(1,        h^{-1/2}, ..., h^{-1/2})    x interior
    N_x = diag(h^{-1/2}, h^{-1/2}, ..., h^{-1/2})    x in a boundary region

(the intercept coordinate scales differently because the smoothed CDF is
sqrt(n)-estimable in the interior but super-consistent at the boundary). For
`v >= 1` the finite-sample standard error is
`sigma_hat_{p,v}(x) = v! sqrt( e_v' S_hat^{-1} Gamma_hat S_hat^{-1} e_v / (n h^{2v}) )`
(SA Theorem 2). SA Section 5 gives two alternatives: a
**plug-in** variance `(v!)^2 f_hat_p(x) e_v' S^{-1} Gamma S^{-1} e_v` (the
`S`, `Gamma` matrices are DGP-free - construct analytically or numerically;
requires knowing the boundary), and a **jackknife** SE built from the
Hoeffding projection of the underlying second-order U-statistic
(`Gamma_hat^JK` leave-one-out formula, SA Section 5.2), consistent under the
same conditions as Theorem 2.

*Manipulation test (main text Section 4; SA Section 4):* with subsamples
below/above `x_bar` (sizes `n_-`, `n_+`, `n = n_- + n_+`) and per-side
bandwidths `h_-`, `h_+`:

    T_p(h) = ( (n_+/n) f_hat_+(x_bar) - (n_-/n) f_hat_-(x_bar) )
             / sqrt( (n_+/n) V_hat_+(x_bar)/(n h_+) + (n_-/n) V_hat_-(x_bar)/(n h_-) )

    H0: lim_{x -> x_bar-} f(x) = lim_{x -> x_bar+} f(x)
    reject iff |T_p(h)| >= Phi_{1-alpha/2}

(main text Section 4 display; the SA works with a joint whole-sample-EDF
formulation whose estimates map to the per-side ones by exact least-squares
identities, SA Remarks 5-6 - `f_hat_{p,-}(x_bar) = (n/n_-) f_hat_p(x_bar-)`
etc., so joint and separate implementations must agree numerically.) Two
model variants:
- **Unrestricted** (SA Section 4.1): duplicated one-sided basis in
  `R^{2p+2}`; the two sides are asymptotically independent, variance is
  additive across sides (SA Lemma 9, Corollary 1).
- **Restricted** (SA Section 4.2): basis
  `(1, u*1(u<0), u*1(u>=0), u^2, ..., u^p)' in R^{p+2}` - CDF and derivatives
  other than the first are continuous at `x_bar`, only the density may jump.
  Improves power when the restriction holds. Its asymptotic variance couples
  the sides through the density-weighted combined Gram matrix
  `f(x_bar+) S_{+,p} + f(x_bar-) S_{-,p}` (SA Lemma 13, Corollary 2) and uses
  the reflection matrix `Psi` (see edge cases).

*Robust bias correction (main text Section 4, final paragraph):* the
uncorrected `T_p(h_p)` at the MSE-optimal bandwidth over-rejects (a
first-order bias survives; undersmoothing - shrinking `h` below the
MSE-optimal rate per SA Corollaries 1-2 - and RBC are the two distinct
remedies). The recommended data-driven test is

    T_{p+1}(h_hat_p)   - reject H0 iff |T_{p+1}(h_hat_p)| >= Phi_{1-alpha/2}

i.e. construct the statistic with order-`(p+1)` density estimators but the
bandwidth `h_hat_p` that is MSE-optimal for the order-`p` point estimator
(of `f_hat_+ - f_hat_-`), "a special case of manual bias-correction together
with the corresponding adjustment of Studentization" following CCT 2014 and
CCF 2018. **Default: `p = 2`, i.e. the reported test is `T_3(h_hat_2)`**
(main text Sections 4-5), with either common (`h_- = h_+`) or distinct
per-side bandwidths.

*Bandwidth selection (main text Section 3; SA Section 3 + Theorem 3):* for
the density (`v = 1`) **when the leading bias is nonzero** - i.e. `x` in a
boundary region or `p - v` odd, and `F^{(p+1)}(x) != 0`:

    h_MSE(x) = ( V(x) / (2 p B(x)^2) )^{1/(1+2p)} * n^{-1/(1+2p)}

(the main text's display). At **interior** points with `p - v` even the
leading bias vanishes (`e_v' S_{p,x}^{-1} c_{p,x} = 0`; Fan-Gijbels 1996) -
the formula above degenerates (division by zero) and the second-order
`h^{p+2-v}` bias drives the trade-off instead, giving rate `n^{-1/(2p+3)}`
and requiring `F^{(p+2)}(x) != 0` (SA Section 3 order table). Bandwidth
selection therefore needs `alpha_x >= p+2` (SA Lemma 5's two-term expansion).
Implemented by plugging preliminary consistent `B_hat(x)`, `V_hat(x)` (SA
Lemma 6 gives the `(S^{-1}c)_hat` estimators at a pilot bandwidth `l`;
`F_hat^{(p+1)}`, `F_hat^{(p+2)}` from a higher-order fit of the same type or a
reference model). SA Theorem 3 splits on the same parity condition: consistent
in rate and constant when `x` is a boundary point or `p - v` is odd (with
`F^{(p+1)}(x) != 0`); interior with `p - v` even is consistent in rate up to a
constant (with `F^{(p+2)}(x) != 0` and the MSE optimum well-defined). NOTE the
manipulation test always evaluates at per-side boundary points, so the
nonzero-leading-bias case is the operative one there. For the test, the main text selects
`h_hat_p` as MSE-optimal for the *difference* `f_hat_+(x_bar) - f_hat_-(x_bar)`
and also offers per-side MSE-optimal selectors (each side is a boundary point
of its own subsample, so the single-density theory applies side-by-side).
For the CDF (`v = 0`) at **interior** evaluation points, a bandwidth
trade-off only exists after including the quadratic-term variance (SA
Section 3.2); at boundary points the leading variance is proportional to
`h/n + 1/n^2`, the population MSE-optimal bandwidth is explicitly undefined,
and the SA's boundary construction is an empirical pseudo-selector (its rate
depends on the pilot bandwidth), not a consistent selector for a well-defined
population optimum. Not needed for the manipulation test but part of the
estimator surface.

*Edge cases:*
- Boundary points: detection is automatic (the kernel moment matrices
  truncate); the leading bias never vanishes at a boundary -> the `p - v`
  even shortcut must not be applied there (SA Section 3 order tables).
- One-sided Gram matrices in the two-sample bases are singular in the full
  space; the SA inverts them as **Moore-Penrose pseudo-inverses**
  (SA proof 7.12).
- Restricted model: only the +side kernel matrices need be computed; the
  -side follows from the reflection identity
  `Gamma_{-,p} = Psi Gamma_{+,p} Psi + S_{-,p} e_{1,-} e_0' S_{-,p}
  + S_{-,p} e_0 e_{1,-}' S_{-,p}` with `S_{+,p} e_{1,-} = S_{-,p} e_{1,+} = 0`
  (SA proof 7.17). `Psi` is the `(p+2) x (p+2)` sign matrix with `(-1)^0` at
  (1,1), the two `(-1)^1` entries OFF-diagonal (swapping the `e_{1,-}` and
  `e_{1,+}` slope coordinates), then diagonal `(-1)^2, ..., (-1)^p`.
- The restricted-model variance requires the density-weighted combined Gram
  matrix `f(x_bar+) S_{+,p} + f(x_bar-) S_{-,p}` to be invertible; one-sided
  densities enter as cubes (`f(.)^3` weights) - a (near-)zero one-sided
  density at the cutoff degenerates the expression (SA Lemma 13 / proof
  p. 30).
- Mass points / discrete running variables: outside the reviewed theory
  (see the assumption bullet above); detect duplicated support values and
  fail closed rather than report unadjusted inference.
- Estimated bandwidth inflates test size in finite samples: SA Tables 1-14
  show fixed-`h_MSE` empirical size ~4.1-5.4% everywhere, while the
  data-driven `h_hat` rows reach 19.0% (truncated normal, interior `x`,
  `p = 2`, n = 1000) without improving at n = 2000. `p = 3` largely repairs
  it at boundary points (5.1% vs 8.2%). NOTE these sizes isolate the SE
  estimator (statistics are centered at `E[f_hat_p]` per the tables'
  footnote); bias handling must come from RBC or undersmoothing - which is
  exactly why the recommended test is `T_{p+1}(h_hat_p)`, not `T_p(h_hat_p)`.
- The `p = 3` bandwidth selector undersmooths heavily at boundary points
  (median `h_hat/h_MSE` as low as 0.36-0.40; SA Tables 2, 4, 10).

*Algorithm (manipulation test, as composed from main text Sections 4-5):*
1. Split the sample at the cutoff: `{x_i < x_bar}` (n_-) and
   `{x_i >= x_bar}` (n_+).
2. Select bandwidth(s): `h_hat_p` MSE-optimal for the density difference at
   `x_bar` (common), or per-side MSE-optimal `h_-`, `h_+` (each side treats
   `x_bar` as a boundary point).
3. Fit at order `q = p+1`, **by model**:
   - *Unrestricted*: fit each side separately (side-specific EDFs), or
     jointly with the duplicated `R^{2q+2}` basis - equivalent AFTER the
     SA Remarks 5-6 rescaling maps (`f_hat_{q,-} = (n/n_-) f_hat_q(x_bar-)`
     etc.; Section 4.1 / unrestricted results). Raw outputs differ by
     scale: separate fits estimate *conditional* densities, the joint fit
     whole-sample-scale ones - the main text's `T_p(h)` display already
     embeds the matching `(n_-/n)`, `(n_+/n)` weights, so be explicit about
     which scale the extracted `f_hat_-`, `f_hat_+` carry.
   - *Restricted*: a **joint** fit with the `R^{q+2}` shared basis is
     required - separate per-side fits cannot impose the shared
     CDF/higher-derivative coefficients, and the identity above does NOT
     carry over.
   Extract `f_hat_+`, `f_hat_-`.
4. Compute the variance at the same order `q`, by model: unrestricted -
   additive per-side variances (SA Lemma 9 / Corollary 1; Theorem 2 form,
   plug-in, or jackknife per SA Section 5); restricted - the coupled
   sandwich of SA Lemma 13 built from the joint fit (the density-weighted
   combined Gram matrix; not a sum of per-side variances).
5. Form `T_q(h_hat_p)` and the two-sided p-value against `N(0,1)`.

**Reference implementation(s):**
- R / Stata: `rddensity` (manipulation testing; Cattaneo, Jansson & Ma 2018,
  *Stata Journal* 18(1), 234-261 - cited as the software companion, not
  reviewed here)
- R / Stata: `lpdensity` (generic density estimation over the support;
  Cattaneo, Jansson & Ma 2019, cited companion)

**Requirements checklist (for the implementation PR):**
- [x] EDF-based local polynomial fit with one-sided/two-sample bases
      (unrestricted `R^{2p+2}` and restricted `R^{p+2}`)
- [x] Boundary-adaptive variance estimation - shipped as the jackknife
      (default) and plug-in variants, the surface R rddensity 3.0
      implements; the Theorem-2 automatic triple-sum estimator is NOT
      implemented (documented Note in REGISTRY.md, see the Implementation
      status block below)
- [x] MSE-optimal bandwidth plug-in (difference objective + per-side),
      pilot-bandwidth chain per SA Lemma 6 / Theorem 3
- [x] RBC test `T_{p+1}(h_hat_p)` with default `p = 2` (`T_3(h_hat_2)`),
      common and distinct bandwidths
- [x] Joint-vs-separate estimation identity (SA Remarks 5-6) as a test anchor
- [x] Singular-Gram handling - resolved as not-applicable-to-R: the SA's
      Moore-Penrose pseudo-inverse is a proof device on the one-sided
      embedded matrices, while R's joint bases use plain `solve(tol=0)`;
      the port uses ordinary solves behind fail-loud design guards
      (documented Deviation in REGISTRY.md). Reflection identity shipped
      as a generator-level test anchor
- [x] Fail-closed mass-point gate (duplicated-support detection before any
      inference; tests must cover a rounded running variable and a side of
      only repeated values)
- [x] Head Start empirical anchors (Table 1 of the main text; below)
- [x] R `rddensity` golden parity (generator + committed JSON, per the
      rdrobust/rdplot precedent) - requires consulting the rddensity software
      paper/source, out of scope for this review

---

## Implementation Notes

### Data Structure Requirements
- A single numeric vector (the running variable) and a scalar cutoff - same
  input surface as `RDPlot`. No outcome variable is involved.
- The estimator itself additionally supports arbitrary evaluation points and
  derivative orders (`v <= p`), but the manipulation test only needs `v = 1`
  at `x = x_bar` per side.

### Computational Considerations
- The Theorem 2 variance `Gamma_hat_{p,x}` is a triple sum (O(n^3) written
  literally). Two reductions are available: (a) only observations within the
  bandwidth window contribute (kernel support), so sums run over effective
  windows; (b) [our observation, not the paper's] the inner sum over `i`
  factorizes through `sum_i 1[x_i <= x_j] 1[x_i <= x_k] = n F_tilde(min(x_j, x_k))`
  (keeping the full centered covariance
  `n [F_tilde(min(x_j, x_k)) - F_tilde(x_j) F_tilde(x_k)]`),
  reducing the computation to O(n_h^2) pairs after sorting - verify against
  the reference implementation at porting time.
- The jackknife `Gamma_hat^JK` (SA Section 5.2) is a double sum over pairs,
  but it CANNOT be restricted to pairs lying entirely inside the kernel
  window: `U_hat(x_i, x_j)` is nonzero whenever EITHER member is localized
  (the localized member's term still involves the other observation through
  its indicator `1(x_j <= x_i)`), so every observation stays eligible as the
  non-localized member of a pair. Direct localized cost is O(n * n_h); any
  faster exact aggregation (rank/cumulative-sum tricks) must be parity-tested
  against the literal formula.
- The DGP-free matrices `S_{p,x}`, `Gamma_{p,x}`, `c_{p,x}` can be computed
  by closed-form integration for the triangular/uniform/epanechnikov kernels
  or by quadrature (SA Section 5.1 explicitly allows either).

### Tuning Parameters

| Parameter | Type | Default | Selection Method |
|-----------|------|---------|-----------------|
| `p` (point-estimation order) | int >= 1 | 2 | main text Sections 4-5: "a natural choice is p = 2 and this is the default in our companion Stata and R software"; Fan-Gijbels odd `p - v` logic (SA Section 6) |
| `q` (test/inference order) | int | `p + 1` | RBC construction `T_{p+1}(h_hat_p)` (main text Section 4) |
| kernel | enum | not specified by the article/SA - verify against the companion software | compact support required (Assumption 2); triangular is the SA Section 6 *simulation configuration*, with the note "the choice of kernel function is usually not very important" |
| `h` (bandwidth) | float or per-side pair | data-driven `h_hat_p` | MSE-optimal for the density difference (common) or per side (main text Section 4; SA Section 3 / Theorem 3) |
| pilot bandwidth `l` | float | unspecified in the paper | enters the preliminary variance/bias estimates (SA Lemma 6, Remark 3); concrete recipe is a software-level choice (gap) |
| model | enum | not specified by the article/SA - verify against the companion software | two variants developed: unrestricted (SA Section 4.1) vs restricted (SA Section 4.2, improves power when higher derivatives are continuous at the cutoff) |
| SE flavor | enum | Theorem 2 automatic | alternatives: plug-in, jackknife (SA Section 5) |

### Relation to Existing diff-diff Estimators
- This is the missing member of the RD validity toolkit that
  `RegressionDiscontinuity`'s docs and Tutorial 28 explicitly disclose as not
  yet packaged (REGISTRY RegressionDiscontinuity "not in v1" seams).
- Diagnostic-family object like `RDPlot` (no treatment effect; a test
  statistic + p-value + per-side densities/bandwidths/effective Ns), with the
  same `cutoff` constructor surface and `running=` fit surface.
- The RBC logic (estimate at order `q = p+1` with a `p`-optimal bandwidth,
  adjust the Studentization) is the same philosophy as the rdrobust port in
  `diff_diff/_rdrobust_port.py`, but the machinery is NOT shared: here the
  regression is of the EDF on the running variable (density estimation), not
  of an outcome. Reusable pieces are the kernel definitions and the
  validation/report conventions, not the fitting code.
- The Head Start application (Ludwig & Miller 2007 data) is the same dataset
  family already used in the CCFT 2019 covariate review's smoke targets.

### Empirical anchors (main text Table 1 and Section 5; Head Start data)

Setting: `x` = county poverty index, cutoff `x_bar = 59.1984`,
`n_- = 2504`, `n_+ = 300`. All rows: manipulation test `T_q(h_hat_p)` with
q-order density estimators at bandwidths MSE-optimal for the p-order
estimator; first block distinct per-side bandwidths (`h_- != h_+`, selected
for the difference or per side as described in the paper's notes), second
block common bandwidth (`h_- = h_+`, MSE-optimal for the difference); last
row = McCrary (2008) original implementation (76/60 prebins).

| Test | h_- | h_+ | eff. n_- | eff. n_+ | T | p-value |
|------|------|------|------|------|------|---------|
| `T_2(h_hat_1)`, distinct | 15.771 | 2.326 | 581 | 65 | 0.024 | 0.981 |
| `T_3(h_hat_2)`, distinct | 19.776 | 8.296 | 762 | 210 | -1.146 | 0.252 |
| `T_4(h_hat_3)`, distinct | 32.487 | 10.808 | 1598 | 232 | -1.083 | 0.279 |
| `T_2(h_hat_1)`, common | 3.274 | 3.274 | 99 | 95 | -1.355 | 0.175 |
| `T_3(h_hat_2)`, common | 9.213 | 9.213 | 316 | 221 | -0.515 | 0.607 |
| `T_4(h_hat_3)`, common | 12.270 | 12.270 | 419 | 243 | -0.712 | 0.477 |
| McCrary | 13.950 | 13.950 | 24 | 24 | 0.142 | 0.887 |

Recommended headline: `T_3(h_hat_2)`. Two theory-based observations the
implementation should preserve (main text Section 5): RBC confidence
intervals are asymmetric (not centered at the density point estimate), and
the effective sample size of McCrary's prebinned test is far smaller
(24/24 bins vs 762/210 observations), costing power.

---

## Gaps and Uncertainties

1. **The RBC test's formal construction is not written out anywhere in the
   pair of documents.** The SA's Corollaries 1-2 prove validity only for the
   *undersmoothed* `T_p(h)` (`n h^{2p+1} -> 0`); the recommended
   `T_{p+1}(h_hat_p)` appears only as a prescription in the main text
   (Section 4, final paragraph), justified by analogy to CCT 2014 / CCF 2018.
   The implementation should treat "order q = p+1 point estimate + order-q
   variance + p-optimal bandwidth" as the definition and verify the exact
   variance choice against the `rddensity` source at porting time.
2. **The difference-objective bandwidth selector is described, not
   displayed.** The main text says `h_hat_p` implements the approximate
   MSE-optimal choice for `f_hat_+(x_bar) - f_hat_-(x_bar)` "in an automatic
   and data-driven way" with details in the SA, but the SA's Section 3 /
   Theorem 3 develops the *single-density* selector; no explicit
   difference-objective display was found in either document. The plug-in
   composition (difference bias = per-side biases with opposite signs;
   difference variance = sum) is implied but its exact plug-in recipe (and
   the pilot bandwidth `l` choice) is a software-level decision - consult the
   rddensity companion paper (Cattaneo, Jansson & Ma 2018, Stata Journal) for
   PR-B.
3. **Transcription ambiguities in the SA scan** (all flagged in place during
   extraction): (a) SA Theorem 3's second bullet prints side condition
   `nh^3 -> 0` where Lemma 5's two-term bias expansion requires
   `nh^3 -> inf` - one of the two is a typo; resolve against the published
   version or the software. (b) The `^{-1}` superscripts on the outer
   `S_hat` factors in the two `V_hat_{p,1}(x_bar)` displays (SA Sections
   4.1-4.2) are below scan resolution; the sandwich form
   `S_hat^{-1} Gamma_hat S_hat^{-1}` is certain from Theorem 2's referenced
   construction. (c) SA Section 4.2's restricted objective prints "arg max"
   for a least-squares minimand - clearly "arg min".
4. **Simulation "size" columns validate the SE only.** Every SA table centers
   the t-statistic at `E[f_hat_p]` (footnote), so the 4-5% fixed-bandwidth
   sizes say nothing about bias handling; do not cite them as end-to-end test
   validity. The end-to-end statement is main text Section 4's asymptotic
   result plus the RBC prescription.
5. **Estimated-bandwidth size distortion** (SA Tables 3, 5, 7, 11, 13; up to
   19% at nominal 5%) is a real finite-sample phenomenon that remains
   unresolved within the reviewed documents. Because the SA's simulation
   statistics are centered at `E[f_hat_p]`, the distortion reflects
   bandwidth-estimation randomness (the empirical sd rises at `h_hat` while
   the mean estimated SE does not), NOT smoothing bias - so RBC, which
   targets first-order smoothing bias and its Studentization, is separately
   required for bias-valid inference but is not established by these
   simulations as a fix for the bandwidth-randomness distortion. Any
   simulation-based test of our implementation should expect fixed-bandwidth
   behavior to be cleaner than data-driven-bandwidth behavior.
6. **No power comparisons vs McCrary / Otsu-Xu-Matsushita / binomial tests**
   appear in either document beyond the Head Start effective-sample-size
   remark; comparative claims should not be sourced from this review.
7. **`lpdensity` vs `rddensity` division of labor**: the main text assigns
   generic density estimation to `lpdensity` and manipulation testing to
   `rddensity`. Whether diff-diff should expose the generic estimator surface
   (evaluation grids, CDF/derivatives) or only the test is a scoping decision
   for the implementation plan, not settled by the paper.


---

## Implementation status (PR-B) - *this block is implementation commentary, not paper-sourced*

The methodology above is implemented as `RDDensityTest`
(`diff_diff/rddensity.py`), parity-targeting R `rddensity` 3.0 (CRAN tarball
sha256 `a9c45ab0f6b86ead4d91084db16513d4156b7f59b0472510b63deb5dee6f305d`).
Software-level behaviors the paper does not specify are documented from the
pinned R source in `docs/methodology/rddensity-source-notes.md`, and every
implementation choice carries a REGISTRY Note/Deviation label (REGISTRY
section "RDDensityTest"). Resolutions of this review's flagged items:

- **The fail-closed mass-point gate** (assumption checks + checklist): its
  condition - "until a reviewed rddensity-compatible mass-point procedure
  exists" - is now met. The R 3.0 adjustment (EDF on unique values,
  replication via last-occurrence indices, first-occurrence replication in
  the jackknife) is documented in the source-notes and shipped with a
  fit-time warning and an RD-family `masspoints="adjust"/"check"/"off"`
  surface.
- **The Theorem-2 "automatic" SE** (tuning table row "SE flavor"): R 3.0
  implements only the jackknife (its default) and the plug-in variance; the
  triple-sum automatic estimator is not in the parity target. The port
  matches R (REGISTRY Note).
- **The jackknife localization note** (Computational Considerations): R's
  construction is WINDOW-RESTRICTED - it drops the out-of-window
  non-localized pair members the SA literal formula keeps. The port ships
  R's construction; the divergence from the SA formula is locked by
  `test_jackknife_window_vs_literal` and documented in the source-notes.
- **The Moore-Penrose checklist item**: resolved as not-applicable-to-R -
  the SA's pseudo-inverse is a proof device on the one-sided embedded
  matrices; R's joint bases use plain `solve(..., tol=0)`. The port replaces
  R's silent degenerate modes with fail-loud design guards (REGISTRY
  Deviations).
- **The difference-objective bandwidth recipe** (Gaps item 2): R selects via
  the `each`/`diff`/`sum` h-table rows and the `comb` combination rules
  (median per side unrestricted; min(diff, sum) restricted), documented in
  the source-notes and pinned by the golden suite.
- **The rank-based EDF** (estimator display above): R implements
  `(0:(N-1))/(N-1)`, not the displayed `(1/n)*sum` form; see the
  source-notes and the REGISTRY Note. The SA Remarks 5-6 rescaling factors
  become `(N-1)/(n_side-1)` under R's EDF.
- **Head Start Table 1**: R 3.0 reproduces the published p=1/p=2 rows at
  display precision; the p=3 bandwidths drifted across package versions
  (published h_left=32.487 vs 3.0's 22.135). The anchor tests pin the
  published p=1/p=2 values and the R-3.0 p=3 behavior, drift documented.
