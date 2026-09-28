# Paper Review: A More Credible Approach to Parallel Trends

**Authors:** Ashesh Rambachan, Jonathan Roth
**Citation:** Rambachan, A., & Roth, J. (2023). A More Credible Approach to Parallel Trends. *Review of Economic Studies*, 90(5), 2555-2591.
**PDF reviewed:** Rambachan & Roth (2023), accessed via doi:10.1093/restud/rdad018
**Review date:** 2026-03-31

---

## Methodology Registry Entry

*Formatted to match docs/methodology/REGISTRY.md structure.*

## HonestDiD

**Primary source:** [Rambachan, A., & Roth, J. (2023). A More Credible Approach to Parallel Trends. *Review of Economic Studies*, 90(5), 2555-2591.](https://doi.org/10.1093/restud/rdad018)

**Key implementation requirements:**

*Assumption checks / warnings:*
- Requires event-study estimates with pre-treatment coefficients (beta_hat_pre) and post-treatment coefficients (beta_hat_post)
- Requires consistent variance-covariance estimator Sigma_hat_n for beta_hat_n
- No anticipation: tau_pre = 0 (pre-treatment coefficients estimate delta_pre directly)
- Normalization: delta_0 = 0 (reference period)
- Sigma_n must have eigenvalues bounded away from zero (Assumption 3)

*Causal decomposition (Assumption 1, Equation 3):*
```
beta = (0, tau_post)' + (delta_pre, delta_post)'
       -----------      --------------------
          tau                  delta
```
where tau_pre = 0 (no anticipation), so beta_pre = delta_pre.

*Target parameter:*
```
theta = l' tau_post
```
where l is a known T_bar-vector. Special cases: l = e_t (period-t ATT), l = (1/T_bar,...,1/T_bar) (average ATT).

*Identified set (Lemma 2.1, Equations 5-6):*

If Delta is closed and convex, S(beta, Delta) = [theta^lb, theta^ub] where:
```
theta^lb = l' beta_post - max{ l' delta_post : delta in Delta, delta_pre = beta_pre }    (Eq 5)
theta^ub = l' beta_post - min{ l' delta_post : delta in Delta, delta_pre = beta_pre }    (Eq 6)
```

CRITICAL: The constraint delta_pre = beta_pre pins the pre-treatment violations to the
observed pre-treatment coefficients. The LP optimizes over delta_post only (given the
coupling through Delta).

For unions of sets (Equation 7):
```
S(beta, Delta) = union_{k=1}^{K} S(beta, Delta_k)
```

*Delta restriction classes:*

**Delta^SD(M) -- Smoothness (Equation 8):**
```
Delta^SD(M) = { delta : |(delta_{t+1} - delta_t) - (delta_t - delta_{t-1})| <= M, for all t }
```
Constrains SECOND DIFFERENCES (changes in slope). M=0 requires linear trends.
All periods (pre+post) are constrained, with delta_0 = 0 at the boundary.
Structure: polyhedral.

**Delta^RM(Mbar) -- Relative Magnitudes:**
```
Delta^RM(Mbar) = { delta : |delta_{t+1} - delta_t| <= Mbar * max_{s<0} |delta_{s+1} - delta_s|,
                   for all t >= 0 }
```
Constrains post-treatment FIRST DIFFERENCES relative to max pre-treatment first difference.
With delta_0 = 0: post constraints include |delta_1|, |delta_2 - delta_1|, etc.
Pre-treatment max includes |delta_{-1}| (boundary through delta_0 = 0).
Structure: union of polyhedra (one per max location).

**Delta^SDRM(Mbar) -- Smoothness + Relative Magnitudes:**
```
Delta^SDRM(Mbar) = { delta : |(delta_{t+1} - delta_t) - (delta_t - delta_{t-1})|
                      <= Mbar * max_{s<0} |(delta_{s+1} - delta_s) - (delta_s - delta_{s-1})|,
                      for all t >= 0 }
```
Post-treatment second differences bounded by Mbar times max pre-treatment second difference.
Structure: union of polyhedra.

**Delta^SDPB(M) -- Smoothness + Positive Bias:**
```
Delta^SDPB(M) = Delta^SD(M) intersect { delta : delta_t >= 0 for all t >= 0 }
```

*Inference -- two recommended approaches:*

**Approach 1: Optimal FLCI (Section 4.1, for Delta^SD):**
```
CI = (a + v' beta_hat) +/- chi
chi(a, v; alpha) = sigma_{v,n} * cv_alpha(b_tilde(a,v) / sigma_{v,n})     (Eq 18)
```
where:
- sigma_{v,n} = sqrt(v' Sigma_n v)
- cv_alpha(t) = (1-alpha) quantile of |N(t,1)| (folded normal)
- b_tilde(a,v) = sup_{delta in Delta, tau_post} |a + v'(delta + L_post tau_post) - l'tau_post|  (Eq 17)
- Optimize (a,v) to minimize chi (convex optimization)

The paper proves FLCIs are consistent and finite-sample near-optimal for Delta^SD (Proposition 4.1).
FLCIs are NOT consistent for Delta^RM, Delta^SDPB, or Delta^SDRM (Proposition 4.2).

**Approach 2: ARP Conditional/Hybrid (Sections 3.2.1-3.2.2, for general Delta):**

Profiled test statistic via dual LP (Equation 15):
```
eta_hat = max_gamma  gamma' Y_tilde_n(theta_bar)
  s.t.  gamma' X_tilde = 0
        gamma' sigma_tilde_n = 1
        gamma >= 0
```

Conditional test: reject if eta_hat > max{0, c_{C,alpha}} where c_{C,alpha} is from
a truncated normal distribution conditional on optimal vertex gamma_* and sufficient
statistic S_n.

Hybrid test (recommended): two-stage with kappa = alpha/10.
1. Stage 1: size-kappa LF test (reject if eta_hat > c_{LF,kappa})
2. Stage 2: conditional test at modified size (alpha-kappa)/(1-kappa) with
   v^up_H = min(v^up, c_{LF,kappa})

Confidence set by test inversion:
```
C^{C-LF} = { theta_bar : hybrid test does not reject H0: theta = theta_bar }
```

*Standard errors:*
- Inherits Sigma_hat_n from underlying event-study estimation
- Any consistent variance estimator (HC, cluster-robust) is valid
- The HonestDiD framework takes (beta_hat, Sigma_hat) as inputs

*Edge cases:*
- M=0 for Delta^SD: linear extrapolation, FLCI near-optimal, LICQ may fail (conditional test ~50% efficient)
- Mbar=0 for Delta^RM: post-treatment first differences = 0, point identification
- LICQ failure: conditional test still controls size but may lose optimal power (Proposition 3.3)
- FLCI inconsistency for non-SD restrictions: only valid for Delta^SD (Proposition 4.2)
- Very large M: identified set approaches [-inf, inf]

**Reference implementation(s):**
- R: `HonestDiD` package (http://github.com/asheshrambachan/HonestDiD)
- Stata: `stata-honestdid` (https://github.com/mcaceresb/stata-honestdid/)

**Requirements checklist:**
- [ ] Delta^SD constrains second differences with delta_0 = 0 boundary handling
- [ ] Delta^RM constrains first differences (not levels), union of polyhedra
- [ ] Identified set LP pins delta_pre = beta_pre
- [ ] Optimal FLCI for Delta^SD (convex optimization, folded normal quantile)
- [ ] ARP hybrid confidence sets for Delta^RM (vertex enumeration, truncated normal)
- [ ] Sensitivity analysis over M/Mbar grid
- [ ] Breakdown value (smallest M where CI includes zero)

---

## Implementation Notes

### Data Structure Requirements
- Input: event-study coefficient vector beta_hat = (beta_pre, beta_post) in R^{T+T_bar}
- Input: variance-covariance matrix Sigma_hat in R^{(T+T_bar) x (T+T_bar)}
- Input: number of pre-periods T, number of post-periods T_bar
- The delta vector is (delta_{-T}, ..., delta_{-1}, delta_1, ..., delta_{T_bar}) -- delta_0 = 0 omitted

### Computational Considerations
- Identified set: LP solves via scipy.optimize.linprog (HiGHS solver)
- Optimal FLCI: nested convex optimization (outer: scipy.optimize.minimize, inner: LP for bias)
- ARP hybrid: vertex enumeration via basis enumeration (C(n, T_bar+1) systems), simulation for c_LF
- DeltaRM: union of polyhedra requires O(T) LP solves per bound
- Test inversion for ARP CI: grid search + bisection refinement

### Tuning Parameters

| Parameter | Type | Default | Selection Method |
|-----------|------|---------|-----------------|
| M | float >= 0 | None | Domain knowledge; M=0 = linear trends |
| Mbar | float >= 0 | None | Domain knowledge; Mbar=1 = post <= max pre |
| alpha | float (0,1) | 0.05 | Standard significance level |
| l | vector | uniform | Target parameter weights |
| kappa | float | alpha/10 | Hybrid test first-stage size (ARP recommendation) |

### Relation to Existing diff-diff Estimators
- Operates on results from MultiPeriodDiD (event study) or CallawaySantAnna (staggered)
- Takes beta_hat and Sigma_hat as inputs -- agnostic to the underlying estimator
- For staggered designs: use estimators robust to treatment effect heterogeneity (Sun & Abraham, Callaway & Sant'Anna)
- Existing _extract_event_study_params handles both result types

---

## Key Theorems

**Lemma 2.1:** If Delta is closed/convex, identified set is an interval with LP bounds.

**Lemma 2.2:** For unions of sets, CI = union of component CIs. Valid for Delta^RM.

**Proposition 3.1:** Conditional and hybrid tests uniformly control size (Assumptions 2-5).

**Proposition 3.2:** Conditional and hybrid tests are uniformly consistent (Assumptions 4-7).

**Proposition 3.3:** Under LICQ, conditional test achieves optimal local asymptotic power.

**Proposition 4.1:** Optimal FLCI achieves finite-sample near-optimality for Delta^SD.

**Proposition 4.2:** FLCI is consistent iff the identified set length is maximal. Fails for Delta^SDPB, Delta^RM, Delta^SDRM.

---

## Gaps and Uncertainties

- The exact algorithm for computing FLCI truncation bounds v^lo, v^up relies on polyhedral conditioning (Lee et al. 2016) -- details in the ARP paper, not fully spelled out here.
- The optimal FLCI reformulation as a single convex program vs nested optimization is not detailed in the paper. The R package implementation provides the concrete algorithm.
- For Delta^RM with many pre-periods, the union of polyhedra has 2T components (T locations x 2 signs). Complexity is manageable for typical T but scales linearly.
- The paper's simulation results (Section 5) are calibrated to 12 specific papers -- generalization to other settings is not formally established.
