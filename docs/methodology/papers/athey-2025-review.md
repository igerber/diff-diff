# Paper Review: Triply Robust Panel Estimators

**Authors:** Susan Athey, Guido Imbens, Zhaonan Qu, Davide Viviano
**Citation:** Athey, S., Imbens, G.W., Qu, Z., & Viviano, D. (2025). Triply Robust Panel Estimators. *arXiv preprint arXiv:2508.21536v2*.
**PDF reviewed:** https://arxiv.org/abs/2508.21536v2 (version-pinned arXiv abstract for v2)
**Review date:** 2026-02-08

**Version-pinning note (2026-05-25):** The current arXiv version of arXiv:2508.21536 is **v3** (submitted 2026-02-09). The 2026-05-24 methodology promotion ships against this v2-pinned review; a formal v2-vs-v3 delta-check against the v3 PDF for TROP-relevant methodology changes (Eqs. 2-3, Algorithms 1-3, Section 2.2, Section 5.2-5.3, Section 6.1-6.2, Theorem 5.1, Corollary 1, Appendix Theorem 8.1) has **NOT** been performed in full. **Update (non-absorbing support work):** the v3 PDF was consulted for the treatment-assignment-pattern sections (§2.1 general assignment, §2.2 Eq. 2 masking, §6.1 Eq. 12 / Algorithm 2, Assumption 1(i), Theorem 5.1) and confirms the general-assignment scope on which `TROP(non_absorbing=True)` is built; the remaining sections of the delta-check stay deferred.

**Action item**: before the next paper-author reference implementation or substantive v3 release, refresh this review against the most recent arXiv version, perform a real v2→v3 PDF delta audit, and re-validate that the verified-component checklist still maps cleanly. Pending that refresh, the methodology promotion is anchored on v2 as documented here.

---

## Methodology Registry Entry

*Working draft formatted to match docs/methodology/REGISTRY.md structure. Heading levels and labels align with existing entries.*

*Resolution status (post-promotion, 2026-05-24):* Gap #5 (weight normalization) is resolved as a **library-side choice**, not a source-side clarification: the shipped implementation uses unnormalized exponential weights matching Eq. 2, and the methodology-promotion PR documents this choice as a deliberate deviation from the Section 5 sum-to-one statement (see the weight-normalization note in the `## TROP` block of `docs/methodology/REGISTRY.md` and the Deviations subsection of the `#### TROP` block in `METHODOLOGY_REVIEW.md`). The deviation is locked by `tests/test_methodology_trop.py::TestTROPDeviations::test_unnormalized_weights_match_eq2` via direct kernel-weight inspection. See Gap #5 below for the original source-side ambiguity.

## TROP

**Primary source:** Athey, S., Imbens, G.W., Qu, Z., & Viviano, D. (2025). Triply Robust Panel Estimators. arXiv:2508.21536. https://arxiv.org/abs/2508.21536

**Key implementation requirements:**

*Assumption checks / warnings:*
- **Assumption 1** (Section 5, pages 20-21): Factor model structure on potential outcomes: `Y_it(0) = L_it + epsilon_it` where `L_it = Gamma_i^T Lambda_t` and `E[epsilon_it | L] = 0`. K (number of factors) is arbitrary and may grow with sample size.
- **Assumption 2** (page 22): Regression adjustment estimator class satisfies `E[L_hat | L] = Gamma (I + B) Lambda^T` where B is a K x K bias matrix. Satisfied by PCA/truncated SVD, nuclear-norm penalized least squares, and general singular-value shrinkage estimators. B = 0 means correctly specified.
- **Assumption 3** (Appendix, page 36): Factor model with covariates: `Y_it(0) = R_it + X_it beta + epsilon_it` where `R_it = Gamma Lambda^T`.
- **Assumption 4** (Appendix, page 36): Estimator class with covariates admits bias decomposition into factor bias, coefficient bias, and vanishing approximation error.
- Block treatment assignment: `W_it = 1{i > N_0} * 1{t > T_0}` (Assumption 1(i), expositional; general assignment patterns supported via Section 6).
- No spillovers and no dynamic effects (Section 5.3).
- Weight estimation error negligible when `N_0 >> N_1` and `T_0 >> T_1` (Section 5.3).
- Weights sufficiently dispersed: `||theta||_2, ||omega||_2 = o(sigma_NT)` (Section 5.3).

*Target parameter (Equation 1):*

```
           sum_{i=1}^{N} sum_{t=1}^{T} W_it (Y_it(1) - Y_it(0))
tau = ---------------------------------------------------------------
              sum_{i=1}^{N} sum_{t=1}^{T} W_it
```

This is the Average Treatment Effect on the Treated (ATT), averaged over all treated unit-period pairs.

*Working model (Section 2.2, page 6):*

```
Y_it(0) = alpha_i + beta_t + L_it + epsilon_it,    E[epsilon_it | L] = 0
```

where:
- `alpha_i` = unit fixed effects
- `beta_t` = time fixed effects
- `L_it = Gamma_i^T Lambda_t` = low-rank factor structure component
- `epsilon_it` = idiosyncratic shocks

*Estimator equation -- single treated unit (Equation 2, page 6):*

```
(alpha_hat, beta_hat, L_hat) = arg min_{alpha, beta, L}
    sum_{j=1}^{N} sum_{s=1}^{T} theta_s^{i,t} omega_j^{i,t} (1 - W_{js}) (Y_{js} - alpha_j - beta_s - L_{js})^2
    + lambda_nn ||L||_*
```

where `||L||_*` is the nuclear norm of L. The treatment effect estimate is then:

```
tau_hat_it = Y_it - alpha_hat_i - beta_hat_t - L_hat_it
```

*Per-observation weights (Equation 3, page 7):*

Time weights (exponential decay in temporal distance):
```
theta_s^{i,t}(lambda) = exp(-lambda_time * dist_time(s, t))
```
where `dist_time(s, t) = |t - s|`.

Unit weights (exponential decay in unit distance):
```
omega_j^{i,t}(lambda) = exp(-lambda_unit * dist_unit_{-t}(j, i))
```
where the unit distance is the RMSE of outcome differences over shared control periods:
```
dist_unit_{-t}(j, i) = sqrt(
    sum_{u=1}^{T} 1{u != t} (1 - W_iu)(1 - W_ju)(Y_iu - Y_ju)^2
    / sum_{u=1}^{T} 1{u != t} (1 - W_iu)(1 - W_ju)
)
```

*General case -- multiple treated units (Equation 13, Section 6.1, pages 26-27):*

Each treated unit-period (i, t) is estimated individually, as if it were the only treated unit:

```
tau_hat_it(lambda) = arg min_tau min_{alpha, beta, L}
    sum_{j=1}^{N} sum_{s=1}^{T}
    [(1 - W_{js}) + W_{js} I_{js}^{i,t}] omega_j^{i,t}(lambda) theta_s^{i,t}(lambda)
    (Y_{js} - alpha_j - beta_s - L_{js} - tau_it * I_{js}^{i,t})^2
    + lambda_nn ||L||_*
```

where `I_{js}^{i,t} = 1{(j,s) = (i,t)}`. The ATT is then:

```
tau_hat = (1 / sum_{i,t} W_it) * sum_{i,t} W_it tau_hat_it(lambda_hat)
```

*Balancing representation (Equation 10, page 22):*

The estimated counterfactual can be decomposed as:
```
tau_hat_TROP(0, theta, omega) = L_hat_NT
    + sum_{t<=T_0} theta_t (Y_Nt - L_hat_Nt)
    + sum_{i<=N_0} omega_i (Y_iT - L_hat_iT)
    - sum_{t<=T_0} sum_{i<=N_0} theta_t omega_i (Y_it - L_hat_it)
```

*With covariates (Equation 14, Section 6.2, page 28):*

Parametrize `L_{js} = X_{js} beta_coef + R_{js}` where R is low-rank. The objective becomes:

```
tau_hat_{i*,t*} = arg min_tau min_{alpha, beta, beta_coef, R}
    sum_{j=1}^{N} sum_{s=1}^{T} theta_s^{i*,t*} omega_j^{i*,t*}
    (Y_{js} - alpha_j - beta_s - X_{js} beta_coef - R_{js} - tau * W_{js})^2
    + lambda_nn ||R||_*
```

This relaxes the low-rank assumption on L, requiring only that the residual after covariates is low-rank.

*Special cases (Section 2.2, page 6):*
- `lambda_nn = infinity`, `omega_j = theta_s = 1` (uniform weights, no factor model) --> DID/TWFE
- `omega_j = theta_s = 1`, `lambda_nn < infinity` (uniform weights, with factor model) --> Matrix Completion (MC)
- `lambda_nn = infinity` with specific choices of unit and time weights --> SC and SDID

*Standard errors (Section 5.3, pages 25-26):*
- Default: Bootstrap variance estimation (Algorithm 3, page 27)
- **No analytical standard errors** for the general case
- For single treated unit: inference driven by variance of idiosyncratic shock `sigma_NT^2` (multiple procedures available in literature)
- Bootstrap: Non-parametric (pairs) stratified bootstrap, resampling entire unit-level rows
- Clustering: Implicitly at the unit level (resamples full unit trajectories)
- Weight distribution: Not applicable (pairs bootstrap, not multiplier bootstrap)
- Recommended iterations: Not explicitly stated; simulations use 1000 replications for Monte Carlo RMSE

*Triple robustness property (Theorem 5.1, page 23):*

Under Assumption 1, for fixed (non-data-dependent) weights theta, omega:
```
|E[tau_hat - tau | L]| <= ||Delta_u(omega, Gamma)||_2 * ||Delta_t(theta, Lambda)||_2 * ||B||_*
```
where:
- `Delta_u(omega, Gamma) = Gamma_bar_0(omega) - Gamma_N` (unit imbalance in loadings)
- `Delta_t(theta, Lambda) = Lambda_bar_0(theta) - Lambda_T` (time imbalance in factors)
- `||B||_*` = spectral norm of regression adjustment bias matrix

The bias is bounded by the **product** of three components. Strictly tighter than bounds for DID, SC, and SDID.

*Corollary 1 (page 23):*

Estimator is unbiased (`E[tau_hat - tau | L] = 0`) if **any one** of:
- (a) Unit balance: `sum_{i<=N_0} omega_i Gamma_i = Gamma_N`
- (b) Time balance: `sum_{s<=T_0} theta_s Lambda_s = Lambda_T`
- (c) Correct regression adjustment: `B = 0_K`

*Bias comparison with existing estimators (Equation 11, Section 5.2, page 24):*
```
B_DID  = (Gamma_N - Gamma_bar_0)^T (Lambda_T - Lambda_bar_0)       [no robustness]
B_SC   = (Gamma_bar_0(omega) - Gamma_N)^T Lambda_T                  [singly robust]
B_SDID = (Gamma_bar_0(omega) - Gamma_N)^T (Lambda_bar_0(theta) - Lambda_T)  [doubly robust]
B_TROP <= ||unit_imbalance||_2 * ||time_imbalance||_2 * ||regression_bias||_*  [triply robust]
```

*Triple robustness with covariates (Theorem 8.1, Appendix, pages 36-37):*

Under Assumptions 3 and 4:
```
|E[tau_hat_TROP - Y_NT(0) | X, R]| <= ||Delta_u||_2 * ||B||_* * ||Delta_t||_2
                                      + ||g(theta, omega)||_2 * ||delta_beta||_2 + o(eta)
```
where `g(theta, omega) = sum_{i,t} M(theta, omega)_it X_it` is the covariate contrast vector and `delta_beta = E[beta_hat | X, R] - beta` is the coefficient estimation bias.

*Convergence rates (Section 5.3, page 25):*
- Weight estimation error: `epsilon_bar_0(theta, omega) = O_p(min{||theta||_2, ||omega||_2})`
- When weights sufficiently dispersed: `tau_hat - E[tau | L] = epsilon_NT(1) + o_p(sigma_NT^2)`
- With multiple post-treatment periods and CLT: `epsilon_NT(1) ~ N(0, sigma_NT^2)`

*Edge cases:*
- Single treated unit, single period: TROP outperforms competitors in about half of cases, within 25% in others (Table 2)
- Very few control units (N_co ~ 10): TROP, SDID, and DIFP perform similarly; advantage emerges with larger control pools (Figure 2)
- No interactive effects (only additive FE): DID performs competitively with TROP (Table 3, "No M" row); TROP adapts without underperforming
- No additive fixed effects (only interactive): SC performs slightly better in some specs (Table 3, "No F" row)
- Only noise (no factors): MC and DID can slightly outperform TROP due to regularization overhead (Table 3, "Only Noise" row)
- Non-random treatment assignment: TWFE shows noticeable bias; TROP and SDID show substantially smaller/no bias (Appendix Table A2)
- Bootstrap validity requires growing number of treated units (Algorithm 3)
- Increasing pre-treatment periods can increase bias for DID-type estimators when treatment is at end of panel (Figure 2 right)

*LOOCV tuning parameter selection (Equations 4-5, pages 7-8):*

For each control observation (i, t) with W_it = 0, compute pseudo-treatment effect:
```
tau_hat_it^{loocv}(lambda) = arg min_tau min_{alpha, beta, L}
    sum_{j,s: W_{js}=0} omega_j^{i,t}(lambda) theta_s^{i,t}(lambda)
    (Y_{js} - alpha_j - beta_s - L_{js} - tau * I_{js}^{i,t})^2
    + lambda_nn ||L||_*                                                   (Equation 4)
```

Select tuning parameters by minimizing:
```
Q(lambda) = sum_{i=1}^{N} sum_{t=1}^{T} (1 - W_it) (tau_hat_it(lambda))^2   (Equation 5)
```

Practical LOOCV procedure (footnote 2, page 8):
1. Fix `lambda_nn = infinity`, `lambda_unit = 0`; optimize `lambda_time` over grid
2. Fix `lambda_time = optimal`, `lambda_nn = infinity`; optimize `lambda_unit` over grid
3. Fix `lambda_time = optimal`, `lambda_unit = optimal`; optimize `lambda_nn` over grid
4. Use these as upper bounds for a finer joint grid
5. Cycle through parameters, updating each while holding others at most recent optima

*Algorithm 1 -- Single treated unit (page 9):*
```
Input: Grid G for (lambda_time, lambda_unit, lambda_nn), treatments W, outcomes Y
1. For each lambda in G:
   a. For each (i,t) with W_it = 0, estimate tau_hat_it(lambda) via Equation (4)
   b. Compute Q(lambda) via Equation (5)
2. Find lambda_hat = arg min_{lambda in G} Q(lambda)
3. Compute tau_hat(lambda_hat) via Equation (2) with selected lambda_hat
```

*Algorithm 2 -- Multiple treated units (page 27):*
```
Input: Grid G for (lambda_time, lambda_unit, lambda_nn), treatments W, outcomes Y
1. For each lambda in G:
   a. For each (i,t) with W_it = 0, estimate tau_hat_it(lambda) via Equation (13)
   b. Compute Q(lambda) = sum_{i,t} (1 - W_it)(tau_hat_it(lambda))^2
2. Find lambda_hat = arg min_{lambda in G} Q(lambda)
3. Compute tau_hat = (1/sum W_it) * sum_{i,t} W_it tau_hat_it(lambda_hat)
```

*Algorithm 3 -- Bootstrap variance estimation (page 27):*
```
Input: Y, W, B (number of bootstrap iterations)
1. For b = 1 to B:
   a. Construct bootstrap dataset (Y^(b), W^(b)) by:
      - Sampling N_0 rows of (Y, W) WITH REPLACEMENT from control units
      - Sampling N_1 rows of (Y, W) WITH REPLACEMENT from treated units
   b. Compute TROP estimator tau_hat^(b) from (Y^(b), W^(b))
2. Variance estimator:
   V_hat_tau = (1/B) sum_{b=1}^{B} (tau_hat^(b))^2 - ((1/B) sum_{b=1}^{B} tau_hat^(b))^2
```

Note: Stratified bootstrap -- control and treated units resampled separately. Preserves within-unit temporal correlation. Validity requires growing number of treated units.

**Reference implementation(s):**
- Authors' replication code (forthcoming as of review date)
- No specific software package mentioned in the paper
- Simulation designs reference Arkhangelsky et al. (2019) SDID replication infrastructure
- diff-diff: `diff_diff/trop.py` (existing implementation)

**Requirements checklist:**
- [ ] Factor model estimated via nuclear-norm penalized least squares with soft-threshold SVD (Equation 2)
- [ ] Unit weights: `exp(-lambda_unit * RMSE_distance)` per Equation 3
- [ ] Time weights: `exp(-lambda_time * |t - s|)` per Equation 3
- [ ] LOOCV implemented for tuning parameter selection via Equation 5
- [ ] LOOCV uses SUM of squared pseudo-treatment effects on control observations
- [ ] Coordinate-wise grid search followed by cycling (footnote 2)
- [ ] Per-observation treatment effect estimation for multiple treated units (Equation 13, Algorithm 2)
- [ ] ATT averages over all treated unit-period pairs with equal weight
- [ ] Stratified bootstrap preserving unit-level structure (Algorithm 3)
- [ ] Covariate extension supported (Equation 14)
- [ ] Special cases recoverable: DID, MC, SC, SDID via tuning parameters
- Weight normalization (`1^T omega = 1^T theta = 1`, Section 5, page 20) — **resolved as library-side choice (2026-05-24)**. Section 5 states sum-to-one, Equation 3 / Equation 2 use unnormalized exponential weights, and the shipped implementation matches Equation 2 (unnormalized). The methodology-promotion PR documents this as a deliberate deviation from the Section 5 sum-to-one statement, locked by direct kernel inspection (see Gap #5 below for original source-side ambiguity).
- [ ] Heterogeneous treatment effects supported via per-observation estimation (Remark 6.1)

---

## Implementation Notes

### Data Structure Requirements
- **Paper assumption — balanced panel:** N units x T time periods.
- **Shipped implementation:** `diff_diff/trop.py` accepts unbalanced panels with structural gaps (see the "Unbalanced panels" item under Gaps and Uncertainties below and the corresponding section of `docs/methodology/REGISTRY.md` under TROP).
- **Outcome matrix**: Y (N x T), observed outcomes
- **Treatment matrix**: W (N x T), binary treatment assignments where `W_it in {0, 1}`
- **Covariates** (optional): X_it, observed covariates for each unit-period pair
- Treatment is an absorbing state for standard block assignment (W_it = 1{i > N_0} * 1{t > T_0}); this is the default mode.
- **Paper scope (Equation 13 / Section 6.1):** the paper extends TROP to general assignment patterns including treatment switching on/off (§2.1: "units moving into and out of treatment").
- **Shipped implementation:** `diff_diff/trop.py` accepts general (on/off) assignment via the opt-in `TROP(non_absorbing=True)` (`method='local'` only), matching the paper's scope. The default `non_absorbing=False` retains the absorbing-state monotonicity gate (in `diff_diff/trop_local.py::_setup_trop_data`, around `trop_local.py:131-144`) as a defensive guard against event-style mis-encoding; it rejects non-monotonic D with a `ValueError` that also points to the opt-in. See `docs/methodology/REGISTRY.md` under TROP for the no-dynamic-effects requirement and the block-only inference caveat (Theorem 5.1 is proven under Assumption 1(i) block assignment only). Removing the opt-in restriction *narrows* a prior implementation over-restriction; the global method still requires block assignment and rejects `non_absorbing=True`.

### Computational Considerations
- **Main bottleneck**: LOOCV grid search -- for each grid point, every control observation requires a separate nuclear-norm penalized weighted least squares solve
- **Per-observation estimation**: With multiple treated units, each `tau_hat_it` estimated separately (Algorithm 2). Computationally expensive with many treated observations.
- **Coordinate-wise search** (footnote 2) reduces grid search from cubic to approximately linear in grid points per parameter
- **Unit distance matrix**: Computing pairwise RMSE distances between all units requires O(N^2 T) operations. Can be parallelized (existing Rust acceleration: `compute_unit_distance_matrix()`, 4-8x speedup).
- **LOOCV parallelization**: Each control observation's pseudo-treatment effect is independent, enabling parallelization across observations and grid points (existing Rust acceleration: `loocv_grid_search()`, 10-50x speedup).
- **Bootstrap parallelization**: Each bootstrap replicate is independent (existing Rust acceleration: `bootstrap_trop_variance()`, 5-15x speedup).

### Tuning Parameters

| Parameter | Type | Default | Selection Method |
|-----------|------|---------|-----------------|
| `lambda_time` | float >= 0 | Data-driven via LOOCV | Exponential decay rate for time weights. 0 = uniform time weights. |
| `lambda_unit` | float >= 0 | Data-driven via LOOCV | Exponential decay rate for unit weights. 0 = uniform unit weights. |
| `lambda_nn` | float > 0 or infinity | Data-driven via LOOCV | Nuclear norm penalty. infinity = no factor model (DID/TWFE). |

Empirical guidance from Table 2 (cross-validated values, T_post=10, N_tr=10):

| Dataset | lambda_unit | lambda_time | lambda_nn |
|---------|-------------|-------------|-----------|
| CPS logwage | 0 | 0.1 | 0.9 |
| CPS urate | 1.6 | 0.35 | 0.011 |
| PWT | 0.3 | 0.4 | 0.006 |
| Germany | 1.2 | 0.2 | 0.011 |
| Basque | 0 | 0.35 | 0.006 |
| Smoking | 0.25 | 0.4 | 0.011 |
| Boatlift | 0.2 | 0.2 | 0.151 |

Key finding (Section 4.3, pages 18-19): Substantial heterogeneity across applications in optimal tuning parameters. Removing regression adjustment (lambda_nn = infinity) or time weights (lambda_time = 0) increases RMSE significantly (up to 85% and 91% respectively). Removing unit weights (lambda_unit = 0) increases RMSE only modestly (max 10%).

### Relation to Existing diff-diff Estimators
- **Direct implementation**: `diff_diff/trop.py` implements the TROP estimator with two public methods (`diff_diff/trop.py:64-78`, validated at `diff_diff/trop.py:909`): `method="local"` is the per-treated-cell estimator (Algorithm 2), and `method="global"` fits a single weighted model on control observations and averages residual-based treated-cell effects into the ATT (`diff_diff/trop_global.py:554-585`).
- **Existing Rust backend**: `rust/src/trop.rs` provides accelerated distance matrix computation, LOOCV grid search, and bootstrap variance estimation.
- **Encompasses SyntheticDiD**: TROP with `lambda_nn = infinity` and specific weight choices reduces to SDID (`diff_diff/synthetic_did.py`).
- **Encompasses TWFE**: TROP with `lambda_nn = infinity` and uniform weights reduces to DID/TWFE (`diff_diff/twfe.py`).
- **Factor estimation**: The nuclear-norm penalized SVD in TROP is related to factor model estimation used in `SyntheticDiD`.
- **Bootstrap structure**: The stratified pairs bootstrap (Algorithm 3) differs from the multiplier bootstrap used in `CallawaySantAnna` and `SunAbraham` -- it resamples entire unit rows rather than applying random weights.

---

## Simulation Design Details

The paper uses semi-synthetic simulations (Section 3.1, pages 9-11) based on 6 real datasets (7 simulation applications, since CPS is used for two outcomes: logwage and urate):
1. CPS (Current Population Survey): N=50, T=40 — used for both CPS logwage and CPS urate applications
2. PWT (Penn World Table): N=111, T=48
3. Germany reunification: N=17, T=44
4. Basque country: N=18, T=43
5. Smoking: N=39, T=31
6. Boatlift: N=44, T=19

**Outcome DGP (Equation 6):** Rank-4 factor model with AR(2) autocorrelated errors, normalized to mean zero and unit variance. True treatment effects are zero (placebo studies).

**Treatment assignment (Equation 8):** Logistic regression on latent factors, inducing confoundedness.

**Key results (Table 1):** TROP outperforms all competitors in 20 of 21 specifications. Competitor RMSE inflation vs. best: DID up to 900%, SC up to 580%, MC up to 300%, SDID up to 90%.

---

## Gaps and Uncertainties

1. **Analytical standard errors not provided**: The paper provides bootstrap inference (Algorithm 3) but no closed-form standard errors for the general case. For single treated units, the paper references "existing literature" for variance estimation without specifying which procedures (Section 5.3, page 25).

2. **Bootstrap iteration count**: The paper does not recommend a specific number of bootstrap iterations. Simulations use 1000 replications for Monte Carlo RMSE, but this is for the simulation study, not a bootstrap recommendation (Section 3, page 10).

3. **Efficient estimation under homogeneity**: Remark 6.1 (page 27) notes that under homogeneous treatment effects, alternative weighting of per-unit estimates could improve precision, but leaves this to future research.

4. **Computational complexity**: Not explicitly discussed. The LOOCV grid search is described as the bottleneck, but no formal complexity analysis is provided.

5. **Weight normalization** (*resolved 2026-05-24*): Section 5 (page 20) states weights sum to one (`1^T omega = 1^T theta = 1`), but the weight specification in Equation 3 (page 7) uses unnormalized exponential weights. It is unclear whether normalization is applied before or after the optimization, or whether the theoretical results in Section 5 assume normalized weights while the practical algorithm uses unnormalized weights. **Resolution**: the shipped implementation uses unnormalized weights matching Equation 2. The methodology-promotion PR adopts this as a deliberate **library-side choice / deviation from the Section 5 sum-to-one statement**, locked by `tests/test_methodology_trop.py::TestTROPDeviations::test_unnormalized_weights_match_eq2` which directly inspects the per-(i, t) weight matrix at `lambda_unit = lambda_time = 0` and asserts every entry equals 1.0 (sum = N*T, not 1). The source-side ambiguity remains open — clarification from the paper authors / forthcoming reference implementation would let the library either confirm the unnormalized choice or migrate to the normalized form; for now the unnormalized form is the documented library contract.

6. **Nuclear norm penalty in Equation 13** (resolved): the source uses the same unsquared nuclear-norm penalty `lambda_nn ||L||_*` in Equation 13 as in Equation 2 (consistent with the rest of the draft and confirmed against the paper text). The shipped implementation matches.

7. **General assignment patterns**: Section 6.1 generalizes beyond block assignment, but the inference results (Section 5.3) and theoretical guarantees (Theorem 5.1) are stated under block assignment. The extent to which theoretical results carry over to general patterns is not fully characterized.

8. **Rank selection**: The paper uses a fixed rank-4 approximation in simulations (Equation 6) but the theoretical framework allows arbitrary K. No guidance is provided for choosing K in practice beyond the nuclear norm penalty (which implicitly selects rank via soft-thresholding).

9. **Unbalanced panels**: The paper assumes balanced panel data (N units x T periods). Extension to unbalanced panels is not discussed but is supported in the existing diff-diff implementation.
