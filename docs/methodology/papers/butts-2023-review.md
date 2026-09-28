# Paper Review: JUE Insight: Difference-in-Differences with Geocoded Microdata

**Authors:** Kyle Butts
**Citation:** Butts, K. (2023). JUE Insight: Difference-in-Differences with Geocoded Microdata. *Journal of Urban Economics*, 133, 103493. DOI: 10.1016/j.jue.2022.103493
**PDF reviewed:** papers/1-s2.0-S0094119022000705-main.pdf
**Review date:** 2026-05-09

---

## Methodology Registry Entry

*Formatted to match docs/methodology/REGISTRY.md structure.*

## SpilloverDiD-GeocodedMicrodata (Ring Method + Nonparametric Treatment-Effect Curve)

**Primary source:** Butts, K. (2023). JUE Insight: Difference-in-Differences with Geocoded Microdata. *Journal of Urban Economics*, 133, 103493.

**Scope:** Spatially-targeted DiD with geocoded microdata. Treatment occurs at a *point* in space (e.g. a foreclosed home, a new transit stop, a sex-offender residence) and the researcher observes panel or cross-sectional outcomes at neighboring locations indexed by their distance `Dist_i = d(theta_i, theta-bar)`. The paper formalizes the "ring method" used by applied work (Currie et al. 2015; Linden and Rockoff 2008; Gerardi et al. 2015; Campbell et al. 2011) that compares an inner "treated" ring to an outer "control" ring, derives the assumptions under which it identifies the average treatment effect on the treated, exposes the bias when these assumptions fail (Fig. 2 panels b-d), and proposes a nonparametric partitioning-based-least-squares estimator of the entire treatment-effect curve `tau(Dist_i)` based on Cattaneo et al. (2019a, 2019b). Paper is 8 pages and treats the no-staggered-timing single-shock case.

**Note (scope vs. diff-diff Phase 3 plan):** This paper does NOT use the kernel-weighted-exposure regressor `E_it = sum_{j != i} w_ij D_jt` of the broader spillover-DiD literature. The estimand is `tau(Dist_i)`, a function of distance from the *one* treatment point, identified ring-by-ring under "Local Parallel Trends" (Assumption 2). Diff-diff's Phase 3 kernel choices (`exp`, `inverse_distance`, `adjacency`, `power`) correspond to this paper's "ring" indicator (`adjacency` with one cutoff, or a partition of cutoffs); the nonparametric estimator is conceptually a binsreg-style flexible ring kernel rather than a continuous-decay weighting. The "indirect / spillover ATT" decomposition language we plan to expose belongs to Butts (2021) "Difference-in-differences estimation with spatial spillovers" (arXiv:2105.03737, cited in this paper's references) - NOT this JUE Insight paper. See "Gaps and Uncertainties" below.

**Key implementation requirements:**

*Assumption checks / warnings:*
- **Assumption 1 (Random Sampling):** observed data `{Y_{i,1}, Y_{i,0}, Dist_i}` is i.i.d. (paper p. 3).
- **Assumption 2 (Local Parallel Trends, p. 3):** for a maximum distance `d-bar`, `lambda(d) = lambda(d')` for all positive `d, d' <= d-bar`. That is, in the absence of treatment, outcomes evolve identically at every distance from treatment within `d-bar`. Stronger than the standard parallel-trends assumption (which only requires equality between treated and control rings *on average*).
- **Assumption 3 (Average Parallel Trends, p. 3):** `E[lambda_d | 0 <= d <= d_t] = E[lambda_d | d_t < d <= d_c]`. Holds *between* the treated and control rings on average. Weaker than Assumption 2.
- **Assumption 4 (Correct d_t, p. 3):** the chosen treatment-ring outer-edge `d_t` satisfies (i) `tau(d) > 0` for all `d <= d_t` and `tau(d) = 0` for all `d > d_t`, AND (ii) `F(d_c) - F(d_t) > 0` (positive density of control units beyond `d_t`).
- **Assumption 5 (d_t is within d_c, p. 4):** there exists `d_t` with `0 < d_t < d_c` such that Assumption 4 holds AND `F(d_t) - F(d_t) > 0` (positive mass between any candidate `d_t` and `d_c`).
- Failure modes that MUST be warned about (Fig. 2 / Proposition 1 part (ii)):
  - **Treated ring too wide:** `d_t > d_t-true` -> control units inside `d_t` average a zero treatment effect into the "treated" mean, biasing `tau-hat` *toward zero* (attenuation). Fig. 2(b).
  - **Treated ring too narrow:** `d_t < d_t-true` -> the "control" ring `d_c \ d_t` contains units that ARE treated, so the counterfactual trend `lambda` is contaminated upward by treatment effects, biasing `tau-hat` *upward* in absolute magnitude. Fig. 2(c).
  - **Robustness check via different rings:** Fig. 2(d) shows wider-ring and narrower-ring "robustness checks" can BOTH return the same biased estimate as the original mis-specified spec. Wider rings are NOT a robustness check.
- Warn when the user supplies only one ring (single `d_t`, single `d_c`) without justification: paper recommends nonparametric estimation when `d_t` is not known a priori from theory.

*Estimator equation (Equation 1 - underlying outcome model, p. 3):*

    Y_it = mu_i + tau_i * 1{t=1} + lambda_i * 1{t=1} + u_it

where:
- `mu_i` is unit-specific time-invariant fixed effect.
- `tau_i = tau(Dist_i) + tilde-tau_i` is unit `i`'s treatment effect, with `tau(d)` the systematic component varying with distance and `tilde-tau_i = tau_i - tau(Dist_i)` an idiosyncratic deviation.
- `lambda_i = lambda(Dist_i) + tilde-lambda_i` is unit `i`'s common-trend component, also split into a distance-dependent systematic part and an idiosyncratic deviation.
- `u_it` is an idiosyncratic error term.

*Equation 2 (rewritten model, p. 3):*

    Y_it = mu_i + tau(Dist_i) * 1{t=1} + lambda(Dist_i) * 1{t=1} + epsilon_it

where `epsilon = u_it + tilde-tau_i + tilde-lambda_i` is uncorrelated with `Dist_i` by construction.

*Ring-method estimator (Equation 3, p. 3):*

For a chosen treated cutoff `d_t` and control outer-edge `d_c`, define `D_i = {i : 0 <= Dist_i <= d_t}` and `D_c = {i : d_t < Dist_i <= d_c}`. On the subsample `D = D_i union D_c`, estimate:

    Delta Y_it = beta_0 + beta_1 * 1_{i in D_i} + u_it     (3)

`beta_1-hat` is the difference-in-differences estimator with expectation:

    E[beta_1-hat] = E[Delta Y_it | D_i] - E[Delta Y_it | D_c]

*Decomposition of beta_1 (Proposition 1, p. 4):*

Under model (2), `beta_1-hat` decomposes into:

    E[beta_1-hat] = (E[tau(Dist_i) | D_i] - E[tau(Dist_i) | D_c])
                 + (E[lambda(Dist_i) | D_i] - E[lambda(Dist_i) | D_c])
                  ^^^ Difference in Treatment Effect ^^^         ^^^ Difference in Trends ^^^

- (i) ALWAYS holds (algebraic decomposition).
- (ii) Under Local Parallel Trends (Assumption 2) OR Average Parallel Trends between `D_i` and `D_c` (Assumption 3), the "Difference in Trends" term collapses to zero, leaving:

      E[beta_1-hat] = E[tau(Dist_i) | D_i] - E[tau(Dist_i) | D_c]

- (iii) If additionally `d_t` satisfies Assumption 4, then `E[tau(Dist_i) | D_c] = 0` and `E[beta_1-hat] = tau-bar` (average treatment effect on the affected).

*Nonparametric partitioning-based estimator (Section 4, p. 4-5):*

Following Cattaneo et al. (2019a, 2019b), partition the support `[0, d_c]` into `L` quantile-spaced intervals `D_1, ..., D_L` of `Dist_i`. Per-interval mean:

    bar-Delta-Y_j := (1 / n_j) * sum_{i in D_j} Delta Y_it

Estimator for `E[Delta Y_it | Dist_i]`:

    bar-Delta-Y_it-hat := sum_{j=1}^{L} 1_{i in D_j} * bar-Delta-Y_j

This paper uses degree-0 polynomials (constant within each interval), with `n_j ~ n / L`.

*Treatment-effect-curve estimator under Assumption 5 (Section 4, p. 4):*

Within the "control" interval `D_L` (the outermost ring), the average is:

    bar-Delta-Y_L  ->^p  lambda     as L -> infinity, n -> infinity

(under Local Parallel Trends + Assumption 5: the last bin is left-bounded by some `d_t' > d_t-true`, so `tau(Dist) = 0` in `D_L`).

Per-interval treatment-effect estimator:

    tau-hat_j := bar-Delta-Y_j - bar-Delta-Y_L

with population limit:

    tau-hat_j  ->^p  E[tau(Dist) | Dist in D_j] + lambda - lambda
                 = E[tau(Dist) | Dist in D_j]

*Proposition 2 (Consistency of Nonparametric Estimator, p. 4):*

Given units follow model (2) and `d_c` satisfies Local Parallel Trends and Assumption 5, as `n -> infinity` and `L -> infinity`:

    tau-hat = sum_{j=1}^{L} tau-hat_j * 1_{i in D_j}  ->^{unif}  tau(Dist)

i.e. uniform convergence to the treatment-effect curve. Proof Appendix A.2 invokes Cattaneo et al. (2019b) for uniform convergence and underlying smoothness conditions.

*Standard errors (Section 4, p. 5; Footnote 10):*

- For `bar-Delta-Y_j`, Cattaneo et al. (2019a) provide robust standard errors that account for the additional randomness of *quantile-estimated* bin endpoints. Implemented in Stata/R `binsreg`.
- For `tau-hat_j = bar-Delta-Y_j - bar-Delta-Y_L`: the SE on the *difference of means* across two disjoint intervals is `sqrt(sigma_j^2 + sigma_L^2)`, where each `sigma_j` is the Cattaneo et al. (2019a) `binsreg` SE for the corresponding bin.
- Inference: form `t-stat = tau-hat_j / SE(tau-hat_j)` and use the standard normal distribution.
- **Footnote 10:** "There may be concerned that the standard errors need to adjust for spatial correlation. However, this is not the case under Assumption 2 as this implies the error term is uncorrelated with distance." So the paper does NOT recommend Conley spatial HAC SEs *for this estimator under Assumption 2*. (This is in tension with diff-diff Phase 1 Conley SE guidance for spillover settings - see "Gaps and Uncertainties" below.)

*Remark 1 (Overall Average Treatment Effect, p. 5):*

A practitioner may wish to "pool" the significant `tau-hat_j` rings into a single average. But inference on this back-of-envelope average is NOT valid because the number of significant rings is itself a random variable - "model selection makes inference a very difficult problem (Leeb and Ptscher, 2005)". A potential workaround is sample-splitting cross-validation: half the data picks the inner ring, the other half estimates the average effect.

*Remark 2 (Covariates, p. 5):*

`binsreg` allows for covariates `X` in the model with valid inference. The "common neighborhood trends" assumption then must hold conditional on `X` (Sant'Anna and Zhao, 2020).

*Remark 3 (Choosing d_c, p. 5):*

The method still requires the researcher to specify `d_c` (the outer-edge / sample boundary). Recommendation: use pre-treatment periods (`t = -2, -1`) on a large sample to estimate `tau(Dist)` under the null and choose `d_c` as the largest distance where the estimated curve is approximately flat. Functions as a pre-trends test.

*Edge cases:*
- **Knife-edge ring choice (Fig. 2a):** when `d_t = d_t-true` exactly, ring estimator is unbiased - in practice this is unlikely without prior theory.
- **`tau(d)` non-monotonic / sign-changing (e.g. negative hyper-local + positive at intermediate distance):** paper p. 4-5 - "the average effect could be near zero across signs". Pooled ring estimate masks heterogeneity. Nonparametric curve is essential here.
- **`tau(d)` exactly cancels `lambda(d)`:** Fig. 4 shows visual pre-trends-like check using `tau-hat_j` for `j` close to `L`. If the bins past the outer `d_t` are all near zero, this is *suggestive* (not conclusive) evidence that Local Parallel Trends holds.
- **No untreated mass at large `d` (`F(d_c) - F(d_t) = 0`):** Assumption 4 part (ii) fails; cannot identify `tau-bar`.
- **Cross-sectional data (Linden-Rockoff application):** identification requires the alternative assumption that the *composition* of homes at a given distance does not change over time. First-differencing replaced by separate before/after nonparametric estimators differenced (Online Appendix).
- **Density of `Dist`:** quantile-spaced bins automatically allocate `n_j ~ n/L`; no special handling needed for sparse-distance regions.

*Algorithm:*
1. Construct `Dist_i = d(theta_i, theta-bar)` (Euclidean distance from each unit's location to the treatment point).
2. Choose outer-edge `d_c` from theory or via Remark 3 (largest distance where pre-treatment `tau(Dist)` is flat).
3. Restrict sample to `{i : Dist_i <= d_c}`.
4. Compute first differences `Delta Y_i = Y_{i,1} - Y_{i,0}` (cross-sectional case: separate before/after estimators).
5. Choose `L` via Cattaneo et al. (2019a) `binsreg` data-driven optimal-`L` selector (variance-bias trade-off integrated over the quantile distribution).
6. Compute quantile bin edges of `Dist_i` at probabilities `(1/L), (2/L), ..., 1`.
7. Compute per-bin `bar-Delta-Y_j = mean(Delta Y_i | i in D_j)` and per-bin SE via `binsreg` formulas.
8. Set the outermost bin `D_L` as the local "control": `tau-hat_j = bar-Delta-Y_j - bar-Delta-Y_L` for `j = 1, ..., L-1`.
9. SE: `sqrt(sigma_j^2 + sigma_L^2)`.
10. Plot `(D_j, tau-hat_j)` with confidence bands; visual pre-trends check is implicit in bins near `D_L`.

**Reference implementation(s):**
- Stata/R: `binsreg` (Cattaneo, Crump, Farrell, Feng) - https://nppackages.github.io/binsreg/. Used directly per Section 4.
- R: Butts maintains `did2s` (cited in Phase 3 plan); for the JUE Insight estimator, `binsreg` is the working tool.
- Code/data for the JUE Insight paper: not explicitly cited in the paper text. Butts personal page (https://kylebutts.com/) typically hosts replication code; not verifiable from the PDF alone.

**Requirements checklist:**
- [ ] Geocoded microdata: each unit has a `(lat, lon)` or projected `(x, y)`.
- [ ] A single treatment point `theta-bar`. (Multiple treatment points = the broader spillover-exposure case in Butts 2021, NOT this paper.)
- [ ] Panel data with two periods (or cross-sectional with unchanging composition).
- [ ] Sufficient density of units across `[0, d_c]` for `binsreg` quantile bins to be well-populated.
- [ ] Either prior theory pinning `d_t` (parametric ring) OR a flat region in pre-treatment `tau(Dist)` (nonparametric ring).

---

## Implementation Notes

### Data Structure Requirements

- Inputs: per-unit `(Y_{i,0}, Y_{i,1}, lat_i, lon_i)` plus a treatment point `(lat_T, lon_T)`. Compute `Dist_i = sqrt((lat_i - lat_T)^2 + (lon_i - lon_T)^2)` (or great-circle / projected). The paper uses Euclidean distance on the unit circle in the Monte Carlo (Equation 4) and presumably miles in the application.
- Output: a `tau-hat(Dist)` step function over `L` bins plus the bin endpoints, plus pointwise SE per bin.
- Cross-sectional case: needs unchanging composition of units at each distance over time (Online Appendix). Not generally available in the typical diff-diff panel test fixtures.

### Computational Considerations

- The paper's nonparametric estimator is `O(n)` for binning + `O(L)` for bin means + `O(n)` for inference. Total `O(n + L) = O(n)`. Optimal `L*` chosen by `binsreg` is typically `L* ~ n^{1/3}` to `n^{1/5}` depending on smoothness of `tau`.
- Quantile bin construction is `O(n log n)` (sort).
- Compared to the "kernel-weighted exposure" approach in Phase 3 (cost `O(n^2)` for the full pairwise weight matrix `w_ij`), the ring/binsreg approach is much cheaper for large `n` because it never builds a pairwise object.

### Tuning Parameters

| Parameter | Type | Default | Selection Method |
|-----------|------|---------|-----------------|
| `d_c` (outer-edge) | float (distance units) | None - REQUIRED | Theory, OR Remark 3 (largest `d` with flat pre-treatment curve). |
| `L` (number of bins) | int | None - data-driven | `binsreg` optimal-`L` selector (Cattaneo et al. 2019a). Typically 5-30. |
| `polynomial_degree` | int | 0 (constant within bin) | Section 4: degree 0 is sufficient under uniform consistency (Footnote 8). Higher degrees reduce bias but increase variance. |
| `kernel` | enum | "uniform" (rings) | Paper does NOT use kernels in the smooth-decay sense. The closest mapping in diff-diff Phase 3 is `adjacency` (uniform indicator inside each ring). Diff-diff's `exp` / `inverse_distance` / `power` choices come from Butts 2021, not this paper. |
| `bandwidth h` | float | n/a (uses bins, not h) | This estimator does NOT have a single bandwidth `h`. The "bandwidth" is the bin width, set automatically by quantile spacing once `L` is chosen. |

### Relation to Existing diff-diff Estimators

- **Phase 3 mapping:** the "ring method" with one inner ring + one outer ring is exactly the case `kernel="adjacency"` with two cutoffs `(d_t, d_c)` over a single treatment point. The nonparametric estimator extends this to `L` rings. The paper's strongest contribution is the warning that single-ring spec is only unbiased at the knife-edge `d_t = d_t-true` and that wider/narrower "robustness checks" can replicate the same bias.
- **Direct ATT / spillover ATT decomposition:** the paper's `tau(Dist)` curve is a *direct-effect-as-a-function-of-distance* curve. There is no separate "indirect / spillover ATT" parameter in this paper because the treatment is a *point*, not a discrete set of treated units with neighbors. The Butts-2023 setup is the limit of a Butts-2021 setup with a single treated `j*` and infinite neighbors. The parameter that diff-diff Phase 3 calls "indirect / spillover ATT" is from the Butts (2021) "Difference-in-differences estimation with spatial spillovers" working paper (arXiv:2105.03737), not this JUE Insight.
- **Conley SE (Phase 1):** Footnote 10 explicitly notes that Conley spatial HAC is NOT needed when Local Parallel Trends holds, because the assumption itself implies the error is uncorrelated with distance. Diff-diff users who *suspect* spatial autocorrelation in the residuals (i.e. who do NOT trust Local Parallel Trends fully) should still combine `binsreg` SEs with Conley-style spatial HAC. The two are not mutually exclusive.
- **Tutorial T22:** the paper's empirical illustration (Linden-Rockoff 2008 sex-offender arrival, Fig. 4) is an excellent T22 anchor:
  - Single point treatment.
  - `tau(d)` is large negative within ~0.1 mi, then noise around zero out to ~0.3 mi.
  - The naive "1/10 mi treated, 1/10-1/3 mi control" ring spec yields ~`-7.5%` (Fig. 4a).
  - The nonparametric `binsreg` estimator yields ~`-20%` in the very nearest bins (Fig. 4b, panel "Nonparametric Approach").
  - This is the canonical "ring is too wide -> attenuated estimate" lesson.
- **DGP for T22:** Equation 4 of the paper:

      p_{it} = 1 + tau(Dist_i) * 1_{t=1} + beta_Lat * Lat_i * 1_{t=1} + beta_Lon * Lon_i * 1_{t=1} + epsilon_it

  with `Lat_i, Lon_i ~ N(0, 0.036)` (units on the unit circle), `beta_Lat, beta_Lon ~ N(0, 0.036)` determining how the price levels evolve, `lambda ~ N(0, 0.025)` the constant common trend, `epsilon_it ~ N(0, 0.036)` idiosyncratic. Treatment-effect curves used in Monte Carlo (Table 1):
  - `tau_1(Dist) = 0.15 * 1_{Dist<0.4}` (constant within ring - favorable to ring method).
  - `tau_2(Dist) = (0.5 * (0.8 - Dist)^2) * 1_{Dist<0.8}` (smooth decline to zero).
  - `tau_3(Dist) = (-0.15 + 1.2875*Dist - 1.375*Dist^2) * 1_{Dist<0.8}` (negative-then-positive: ring averages near zero despite real heterogeneity).
  - `tau_4(Dist) = (0.5 * (0.8 - Dist)^2) * 1_{Dist<0.25}` (very-narrow effect; many unaffected units).
- **Diff-diff coverage of binsreg:** diff-diff currently has no `binsreg` integration. Implementing the full Butts (2023) nonparametric estimator would require either bundling Cattaneo et al.'s `binsreg` SE formulas or providing a simplified equal-bin-spacing approximation with bootstrap SEs.

---

## Gaps and Uncertainties

- **This paper does NOT provide the exposure regressor formulation.** Diff-diff Phase 3's plan to add `E_it = sum_{j != i} w_ij * D_jt` with kernel choices (`exp`, `inverse_distance`, `adjacency`, `power`) traces to **Butts (2021) "Difference-in-differences estimation with spatial spillovers" (arXiv:2105.03737)**, which is cited in this JUE Insight's references but is a SEPARATE paper. Phase 3 should also pull and review Butts (2021) for the direct-vs-indirect ATT decomposition we plan to expose. The JUE Insight version of the decomposition is "treatment effect as a function of distance from a single point" - not "direct ATT vs spillover ATT" in the multi-treated-unit sense.
- **Bandwidth `h` selection:** the paper does NOT use a kernel bandwidth `h` (the "bandwidth" mentioned in Fig. 3 is for the *graphical visualization* via Local Polynomial Kernel Density, not for the Section-4 estimator). Diff-diff Phase 3 needs separate guidance on selecting `h` for the smooth-kernel weights `w_ij = exp(-d_ij/h)` etc.; that guidance is NOT in this paper.
- **Identification with multiple treatment points:** the paper assumes ONE treatment point `theta-bar` (Section 3, paragraph 1: "Treatment occurs at a location theta-bar"). If multiple points exist, distances `Dist_i` to the *nearest* treated point cannot resolve compound exposure; the estimand becomes ambiguous. Phase 3's target case (multiple treated units with overlapping neighborhoods) is OUT OF SCOPE for this JUE Insight.
- **Conley SE recommendation:** Footnote 10 says spatial HAC is unnecessary under Assumption 2. But Phase 1 of the spillover-Conley initiative is *adding* Conley SE as a robustness option. The two are not contradictory: Conley SE is for the case where the user is *not certain* that the error term is distance-uncorrelated. We should document Footnote 10 as the "default off" rationale: under strict Local Parallel Trends, classic SEs suffice; users who suspect residual spatial correlation can opt into Conley.
- **No formal Hausman-style test for `d_t`:** the paper proposes a *visual* pre-trends-style check via Fig. 4(b) (bins past the true `d_t-true` should be flat near zero). There is no formal test statistic, p-value, or critical value. Diff-diff could expose a "flat-tail" diagnostic that bootstraps the joint hypothesis "`tau-hat_j = 0` for `j` in the outer K bins" but this would extrapolate beyond the paper.
- **No staggered treatment timing:** the paper is single-shock, two-period. Combining with multi-period staggered designs (de Chaisemartin-D'Haultfœuille / Callaway-Sant'Anna / Sun-Abraham) is not addressed. Phase 3 must decide whether to support `binsreg` per cohort, or per (cohort, post-period) cell, or use a single pooled cross-section.
- **Reference implementation:** the paper does not cite a specific GitHub repo or replication archive in the main text. JUE Insight policy generally requires data + code on JUE's online supplement; the citation lists DOI 10.1016/j.jue.2022.103493 and the paper says supplementary material is available "in the online version". Diff-diff implementation should not rely on Butts' code being available; the equations in Section 3-4 plus `binsreg` documentation are sufficient.
- **`binsreg` Python equivalent:** as of this paper's publication, there was no first-class Python implementation of Cattaneo et al. (2019a) `binsreg`. Diff-diff's options are (a) call the R/Stata `binsreg` via a subprocess (heavy dependency), (b) re-implement the bin-mean + per-bin SE manually (loses Cattaneo et al.'s data-driven `L*` and quantile-randomness adjustment), or (c) defer to Butts (2021) exposure-regressor formulation as Phase 3's primary entry point and mark the binsreg-style nonparametric ring method as a "Phase 3.5" follow-up.
- **Density assumption:** unlike de Chaisemartin et al. 2026's `f_{D_2}(0) > 0` assumption, this paper does not formalize a positive-density-at-boundary requirement on `f_{Dist}(0)`. In practice, very few units near `Dist = 0` will inflate the SE on `tau-hat_1` but should not bias the estimate.
- **Cross-sectional support:** the Linden-Rockoff application uses cross-sectional data, but the paper relegates the alternative identification argument (composition unchanged over time, separate-then-differenced nonparametric estimators) to the Online Appendix not visible in the main 8-page PDF. Diff-diff should treat panel as primary; cross-sectional as a follow-up.
