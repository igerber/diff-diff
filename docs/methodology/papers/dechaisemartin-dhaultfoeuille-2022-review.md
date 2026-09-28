# Paper Review: Difference-in-Differences Estimators of Intertemporal Treatment Effects

**Authors:** Clément de Chaisemartin and Xavier D'Haultfœuille
**Citation:** de Chaisemartin, C., & D'Haultfœuille, X. (2022). Difference-in-Differences Estimators of Intertemporal Treatment Effects. *NBER Working Paper 29873*, March 2022 (revised July 2023). URL: https://www.nber.org/papers/w29873.
**PDF reviewed:** NBER Working Paper 29873 ([https://www.nber.org/papers/w29873](https://www.nber.org/papers/w29873)), March 2022 (revised July 2023). Local PDFs are gitignored under `/papers/`; the NBER landing page is the authoritative source.
**Review date:** 2026-05-21

---

## Methodology Registry Entry

*Formatted to match `docs/methodology/REGISTRY.md` structure. The new content below extends the existing `## ChaisemartinDHaultfoeuille` entry in REGISTRY with the dynamic-effects (`DID_l`) machinery introduced in this paper. The companion AER 2020 review at `docs/methodology/papers/dechaisemartin-dhaultfoeuille-2020-review.md` covers the static `DID_M` half of the same library entry.*

## ChaisemartinDHaultfoeuille (2022 NBER WP 29873 component - dynamic effects)

**Primary source:** de Chaisemartin & D'Haultfœuille (2022), *Difference-in-Differences Estimators of Intertemporal Treatment Effects*, NBER Working Paper 29873. https://www.nber.org/papers/w29873.

**Scope:** The dynamic-effects extension of DCDH. Introduces `DID_l` (event-study at horizon `l`), `DID_l^pl` (placebo), `DID^n_l` (normalized), `DID_l^X` (covariate-adjusted), `DID^{fd}_{g,l}` (first-differenced for group-specific linear trends), cohort-aggregation of per-group building blocks, the **cohort-recentered plug-in analytical variance** (Section 3.4 + Web Appendix Section 3.7.3), joint heterogeneity testing across cohorts and horizons, and the literature comparison with TWFE / local-projection / distributed-lag / Callaway-Sant'Anna / Sun-Abraham (Section 4). The companion AER 2020 paper (`docs/methodology/papers/dechaisemartin-dhaultfoeuille-2020-review.md`) introduces the foundational DCDH framework and the static `DID_M` estimator; this 2022 paper picks up the dynamic event-study extension that footnote 13 of the AER 2020 paper deferred.

**Key implementation requirements:**

*Assumption checks / warnings:*

The dynamic paper formalizes Assumptions 1-6 as its core identification + inference set, with further numbered assumptions (covering extensions like non-binary / covariate-adjusted estimation, group-specific linear trends, set-of-groups trends, and additional regularity conditions) introduced through the body and the Web Appendix; the precise main-text-vs-appendix split varies by topic and is not catalogued in detail here. The implementation-relevant subset:

- **Assumption 1 (Restriction on the design, page 9):** there exists a pair `(g, g')` such that `D_{g,1} = D_{g',1}` AND `F_g != F_{g'}`. Two groups must share the period-one treatment but switch at different first-change dates. Fails when period-one treatment is continuous i.i.d., when all groups change simultaneously, or when no group ever changes. For consistency and asymptotic normality, the number of such qualifying pairs must go to infinity as `G -> infinity` (Section 3.4). Implementation: pre-fit validator can detect "no qualifying pair" panels and refuse to fit.

- **Assumption 2 (Zero treatment at baseline, page 10):** `D_{g,1} = 0` for all `g`. Singles out designs where every group is untreated at period one. The estimator runs without this; under Assumption 2, the period-one-treatment-stratified control set `D_1^r` collapses to `{0}` and Assumption 4 reduces to a parallel-trends-on-the-never-treated-outcome statement (matching Callaway-Sant'Anna and Sun-Abraham). The library treats non-zero baselines via per-cohort stratification on `D_{g,1}`.

- **Assumption 3 (No Anticipation, page 11):** `Y_{g,t}(d_1, ..., d_T) = Y_{g,t}(d_1, ..., d_t)`. A group's period-`t` outcome does not depend on future treatments. Adopts the Robins (1986) dynamic-potential-outcome framework; same usage as Malani-Reif (2015), Botosaru-Gutierrez (2018), Callaway-Sant'Anna (2021), and Sun-Abraham (2021). Library does not validate Assumption 3 (untestable in isolation) but exposes the placebo test as a joint A3+A4 diagnostic.

- **Assumption 4 (Conditional parallel trends for the status-quo outcome, page 12):** the CORE identifying restriction. For any `(g, g')` with `D_{g,1} = D_{g',1} in D_1^r` and any `t >= 2`,
  ```
  E[Y_{g,t}(D_{g,1,t}) - Y_{g,t-1}(D_{g,1,t-1}) | D]
    = E[Y_{g',t}(D_{g',1,t}) - Y_{g',t-1}(D_{g',1,t-1}) | D].
  ```
  Two groups with the same period-one treatment must have the same expected evolution of their status-quo (period-one-treatment-held-constant) potential outcome. Generalizes Abadie (2005) to dynamic effects and complicated designs. Restricts ONE potential outcome per group; alone places no restriction on treatment effects. Testable via placebo (Section 3.5; Lemma 5).

- **Assumption 5 (No-crossing condition, page 14):** for every group, either `D_{g,t} >= D_{g,1}` for all `t`, or `D_{g,t} <= D_{g,1}` for all `t`. Implied by Assumption 2; holds automatically for binary treatments or single-change designs. When violated, the `did_multiplegt_dyn` Stata package drops affected `(g, t)` cells via the `drop_larger_lower` option - which is the library's default. Library Note: multi-switch groups are dropped before estimation; see REGISTRY.

- **Assumption 6 (Unconditional parallel trends, page 17):** stronger than Assumption 4 - imposes that ALL groups (not just same-period-one-treatment groups) experience the same status-quo evolution. Required by alternative estimators that binarize the treatment and apply Callaway-Sant'Anna or Sun-Abraham machinery in non-binary designs. Explicitly criticized in the paper for ruling out lagged-treatment effects and time-varying treatment effects in non-binary designs (page 17, footnote: "under Assumption 6, for binary treatment, the effect of being treated for `t` periods equals the effect of being treated for `t-1` periods"). The library does NOT impose Assumption 6.

- **Assumption 7 (Cost-benefit aggregation):** `D_{g,t} >= D_{g,1}` for all `(g, t)`. Required for the single-sign cost-benefit interpretation in Section 3.3. When leavers are present (binary `1 -> 0` groups violate A7), the library emits a `UserWarning` and exposes `delta_joiners` / `delta_leavers` separately.

- **Assumption 8 (Independent groups, Section 3.4 page 26):** `(D_g, Y_g)_g` are mutually independent conditional on the design `D`. Permits within-group serial correlation (treatments and outcomes). The CI is asymptotically conservative (`liminf >= 1 - alpha`) under Assumption 8; equality (`lim = 1 - alpha`) holds under the stronger i.i.d. condition.

- **Assumption 9 (Finite cohort count, page 27):** the number of distinct triples `(D_{g,1}, F_g, S_g)` over groups with `D_{g,1} in D_1^r` is finite, equal to `K`. Defines the cohort partition `{C_k}_{k=1,...,K}`.

- **Assumption 10 (Moment bound):** `sup_{g,t} E[|Y_{g,t}|^{2+delta} | D] < infinity` for some `delta > 0`. Lyapunov CLT prerequisite.

- **Web Appendix Assumption 11 (Parallel trends with covariates, Section 1.2):** invoked by the `DID_l^X` covariate-adjusted path.

- **Web Appendix Assumption 12 (Parallel trends for first-differenced status-quo outcome, Section 1.3):** invoked by the `DID^{fd}_{g,l}` group-specific-linear-trends path.

- Library warns (does NOT fit silently) when fewer than 50% of `l=1` switchers contribute at a far horizon `l`. This is the direct empirical rationale from the banking-deregulation application (Section 5): at horizon `l=9` only 357/905 ~ 39% of original switchers remain, and the authors explicitly drop the report at that horizon (page 39).

*Target parameters:*

- **Per-group AVSQ (actual-versus-status-quo) effect** for any `g` with `F_g <= T_g` and any `l in {1, ..., T_g - F_g + 1}` (Equation 2):
  ```
  delta_{g,l} = E[ Y_{g, F_g - 1 + l} - Y_{g, F_g - 1 + l}(D_{g,1}, ..., D_{g,1}) | D ]    (2)
  ```
  the difference between `g`'s actual outcome at horizon `l` after the first switch and its counterfactual under status-quo (period-one treatment held constant from period 1 through `F_g - 1 + l`).

- **Aggregate event-study (non-normalized), Equation 4:**
  ```
  delta_l = (1/N_l) * sum_{g : F_g - 1 + l <= T_g} S_g * delta_{g,l}                  (4)
  ```
  where `S_g in {-1, +1}` is the sign of `g`'s first treatment change, and `N_l = #{g : F_g - 1 + l <= T_g}`.

- **Normalized AVSQ effect** (Equations 9 + 10):
  ```
  delta^D_{g,l} = sum_{k=0}^{l-1} (D_{g, F_g + k} - D_{g,1})                          (9)
  delta^n_{g,l} = delta_{g,l} / delta^D_{g,l}                                          (10)
  ```
  Per-group normalization by the cumulative-dose deviation. For binary staggered designs, `delta^D_{g,l} = l` so normalization divides by `l`.

- **Normalized event-study effect** (Equation 13):
  ```
  delta^n_l = (1/N_l) * sum_{g : F_g - 1 + l <= T_g} (|delta^D_{g,l}| / delta^D_l) * delta^n_{g,l}    (13)
  ```
  where `delta^D_l = (1/N_l) * sum_g |delta^D_{g,l}|`. Equivalent algebraic identity (Equation 14): `delta^n_l = delta_l / delta^D_l` since `delta^D_l` is design-only (deterministic given `D`).

- **Cost-benefit aggregate `delta`** (Section 3.3, Equation 17):
  ```
  delta = (sum_{g:F_g<=T_g} sum_{l=1}^{T_g-F_g+1} delta_{g,l})
        / (sum_{g:F_g<=T_g} sum_{l=1}^{T_g-F_g+1} (D_{g,F_g-1+l} - D_{g,1}))
  ```
  Lemma 4 (Equation 21): under Assumptions 1 + 7, `delta = sum_{l=1}^L w_l * delta_l` is a non-negative weighted average of the event-study parameters.

*Estimator equations:*

**Per-group building block `DID_{g,l}` (Equation 3, page 13):**
```
DID_{g,l} = Y_{g, F_g - 1 + l} - Y_{g, F_g - 1}
          - (1 / N^g_{F_g - 1 + l}) * sum_{g' : D_{g',1} = D_{g,1}, F_{g'} > F_g - 1 + l}
                                      (Y_{g', F_g - 1 + l} - Y_{g', F_g - 1})        (3)
```
The DID comparison between `g`'s outcome change from `F_g - 1` to `F_g - 1 + l` and the average change over **same-baseline-treatment, not-yet-treated control groups** (stratified not-yet-treated). Footnote 9: the reference period is `F_g - 1`; an alternative averages over the entire pre-`F_g` baseline.

**Lemma 1 (page 13):** Under Assumptions 3 + 4, `E[DID_{g,l} | D] = delta_{g,l}` for every `(g, l)` with `1 <= l <= T_g - F_g + 1`. Consequently `DID_l` is conditionally unbiased for `delta_l` under Assumptions 1, 3, 4.

**Aggregate event-study estimator `DID_l` (Equation 5, page 15):**
```
DID_l = (1 / N_l) * sum_{g : F_g - 1 + l <= T_g} S_g * DID_{g,l}                       (5)
```

**Lemma 2 (page 19):** Under Assumption 5, the normalized AVSQ effect decomposes as a CONVEX (non-negative-weighted, sums-to-1) weighted average of slopes of `g`'s potential outcome at `F_g - 1 + l` with respect to its `l - 1` first treatment lags:
```
delta^n_{g,l} = sum_{k=0}^{l-1} w_{g,l,k} * s_{g,l,k}
```
with `w_{g,l,k} = (D_{g, F_g - 1 + l - k} - D_{g,1}) / delta^D_{g,l} >= 0` and `sum_k w_{g,l,k} = 1`. Establishes the no-sign-reversal / monotonicity property: the normalized effect averages slopes that all have the same sign under A5.

**Normalized event-study estimator `DID^n_l` (Equation 15, page 20):**
```
DID^n_l = (1 / N_l) * sum_{g : F_g - 1 + l <= T_g} (|delta^D_{g,l}| / delta^D_l) * (DID_{g,l} / delta^D_{g,l})    (15)
```
or equivalently `DID^n_l = DID_l / delta^D_l` (Equation 14). Since `delta^D_l` is a function of the design `D` only (deterministic given `D`), the inference results for `DID_l` extend directly to `DID^n_l` with no additional delta method.

**Cohort definition (Section 3.4, page 27-28):** Under Assumption 9, the set of distinct values of `(D_{g,1}, F_g, S_g)` across groups with `D_{g,1} in D_1^r` is finite, equal to `{(d_k, f_k, s_k) : k = 1, ..., K}`. Then:
```
C_k   = { g >= 1 : D_{g,1} = d_k, F_g = f_k, S_g = s_k }
C^G_k = C_k intersect {1, ..., G}
```
The `(C^G_k)_{k=1,...,K}` are the **cohorts** that anchor the variance recentering. Membership is determined by the triple `(baseline_treatment, first_switch_period, sign_of_change)`.

**U-form rewrite (Equation 22, page 27):** `DID_l` is a sample average of per-group scores:
```
DID_l = (1 / N_l) * sum_{g=1}^G U^G_{g,l}                                              (22)
```
where `U^G_{g,l} = (lambda^G_{g,l})' * Y_g`. The score `U^G_{g,l}` is the per-group influence-function contribution and is linear in `g`'s outcome vector. Component-by-component:
```
lambda^G_{g,l,t} = S_g * 1{F_g <= T_g - l + 1} * (1{F_g = t - l + 1} - 1{F_g = t + 1})
                 - (N^g_{t,l} / N^g_l) * 1{F_g > t}
                 + (N^g_{t+l,l} / N^g_{t+l}) * 1{F_g > t + l}
```
with `N^g_{t,l} = sum_{g' <= G : D_{g',1} = D_{g,1}} S_{g'} * 1{F_{g'} = t - l + 1}` (count of switchers at the appropriate horizon among groups sharing `g`'s baseline treatment), convention `0/0 = 0`.

This rewrite anchors the variance: every `(g, l)` contribution to `DID_l` is a clean linear functional of `Y_g`.

*Cohort-recentered plug-in variance (Section 3.4 + Web Appendix Section 3.7.3, pages 25-28 and 67-70):*

Under Assumption 8 (independent groups conditional on `D`), the conditional variance of the scaled estimator is (Equation 23, page 27):
```
V(N_l^{1/2} * DID_l | D) = (1/N_l) * sum_{g=1}^G E[(U^G_{g,l} - E[U^G_{g,l} | D])^2 | D]    (23)
```

The cohort-recentered plug-in estimator subtracts the cohort-specific mean rather than the (unidentifiable) per-group mean. Define for each cohort `k` and each `g in C^G_k`:
```
U_k_bar = (1 / #C^G_k) * sum_{g' in C^G_k} U^G_{g', l}
```
Then the **feasible plug-in variance estimator** is (page 28; Web Appendix Section 3.7.3, page 67):

    sigma_hat_l^2 = (1/N_l) * sum_{g=1}^G U^{G,2}_{g,l} - sum_{k=1}^K (#C^G_k / N_l) * U_k_bar^2

Equivalently (centered form):
```
sigma_hat_l^2 = (1/N_l) * sum_{k=1}^K sum_{g in C^G_k} (U^G_{g,l} - U_k_bar)^2
```

The Web Appendix derivation (page 67-69) confirms the algebraic identity via Equation (49):
```
sigma_hat_l^2 - sigma_bar^2_{G,l}  --> 0  in probability, almost surely
```
where `sigma_bar^2_{G,l}` is the population analogue (Equation (49)). Convergence is proved via two sub-claims:
```
(1/N_l) * sum_g (U^{G,2}_{g,l} - E[U^{G,2}_{g,l} | D]) --> 0          ... (50)
sum_k (#C^G_k / N_l) * (U_k_bar^2 - E[U_k_bar | D]^2) --> 0           ... (51)
```

**Confidence interval (page 28):**
```
CI_{1-alpha} = [ DID_l +/- z_{1-alpha/2} * sigma_hat_l / sqrt(N_l) ]
```
where `z_{1-alpha/2}` is the standard normal quantile.

**Theorem 1 (page 28):** Under Assumptions 3, 4, and 8-10, for all `l` in the set `L` such that `lim_G N_l = infinity` a.s., conditional on `(D_g)_{g >= 1}` and almost surely:
```
DID_l - delta_l  -->P 0
sqrt(N_l) * (DID_l - delta_l) / ((1/N_l) sum_g V(U^G_{g,l} | D))^{1/2}  -->d  N(0, 1)
liminf_{G->infinity} Pr[delta_l in CI_{1-alpha} | D]  >=  1 - alpha
```
with equality under the additional i.i.d. assumption on `(D_g, Y_g)_{g>=1}`.

**Key inference properties:**
- The estimator is asymptotically normal almost surely conditional on the design.
- **Inference is analytical (Theorem 1), NOT bootstrap-based.** No bootstrap procedure is proposed or transcribed in the main paper for the dynamic estimators.
- The CI is asymptotically **conservative** in the i.n.i.d. case (one-sided coverage inequality) and **exact** under i.i.d. groups (equality).
- Clustering is at the **group** level via Assumption 8. Asymptotics are taken as `G -> infinity`.
- No finite-sample df correction: the CI uses the standard-normal quantile directly. Conservatism in the i.n.i.d. setting partially substitutes for any small-sample adjustment.

**Web Appendix Section 3.7.3 implementation guidance (page 65-66):**
- `||lambda^G_{g,l}||^2 <= T * (1 + max_{t} N^k_{t,l} / N^k_t)` (Equation 46), bounding each group's IF norm.
- The per-cohort sub-statistic `T^G_k = (sum_{g in C^G_k} W^G_{g,l}) / (sum_{g in C^G_k} V(W^G_{g,l} | D))^{1/2}` admits the cohort-weighted decomposition `T^G = sum_k omega^G_k * T^G_k` (Equation 39), where `omega^G_k` is the within-cohort variance share. Cohorts that stay bounded contribute `O_p(1)` to `T^G_k` but get weighted by `omega^G_k --> 0`; cohorts that diverge contribute their CLT-normal limits (Equation 48, page 66).

**Conservative-vs-exact trade-off (Web Appendix Equation 54, page 69):**
```
sigma_bar^2_{G,l} >= (1/N_l) * sum_g V(U^G_{g,l} | D)
```
with equality iff data are i.i.d. The plug-in CI is asymptotically conservative when cohort means are non-trivially correlated within cohort and exact under i.i.d. cohorts.

*Aggregation weights:*

- `DID_l` aggregates per-group `S_g * DID_{g,l}` with **uniform** `1/N_l` weights across the `N_l` eligible groups (with sign `S_g`).
- `DID^n_l` aggregates `delta^n_{g,l}` with weights `|delta^D_{g,l}| / delta^D_l` proportional to absolute dose deviation (NOT uniform).
- **Total-lag weights** `w_{l,k}` (Section 3.2, page 20): `delta^n_l` is a weighted average of effects of the `k`th treatment lag with total weight
  ```
  w_{l,k} = (1/N_l) * sum_g |D_{g, F_g - 1 + l - k} - D_{g,1}| / delta^D_l
  ```
  In one-shot designs `w_{l, l-1} = 1`; under single-change treatment `w_{l,k} = 1/l`.

*Placebo estimator (Section 3.5 + Web Appendix Section 1.1):*

For any `g` with `3 <= F_g <= T_g` and `l in {1, ..., min(T_g - F_g + 1, F_g - 2)}` (Web Appendix Section 1.1, page 46):
```
DID^pl_{g,l} = Y_{g, F_g - 1 - l} - Y_{g, F_g - 1}
              - (1 / N^g_{F_g - 1 + l}) * sum_{g' : D_{g',1} = D_{g,1}, F_{g'} > F_g - 1 + l}
                                          (Y_{g', F_g - 1 - l} - Y_{g', F_g - 1})
```
The placebo mirrors `DID_{g,l}` but runs BACKWARDS in time from `F_g - 1` to `F_g - 1 - l`. Aggregator:
```
DID^pl_l = (1 / N^pl_l) * sum_{g : 1 <= F_g - 1 - l, F_g - 1 + l <= T_g} S_g * DID^pl_{g,l}
```
with `L^pl = max_g {min(T_g - F_g + 1, F_g - 2)}` and `N^pl_l = #{g : 1 <= F_g - 1 - l, F_g - 1 + l <= T_g}`.

**Lemma 5:** Under Assumptions 1, 3, 4, with `L^pl >= 1`, `E[DID^pl_l | D] = 0` for all `l in {1, ..., L^pl}`. Non-zero placebo signs the bias of `DID_l`: under a monotone-difference-in-trends condition, the sign of bias of `DID_l` equals the sign of `-E[DID^pl_l | D]`.

**Boundary condition:** `L^pl = -1` (no placebos computable) if all switchers first switch at period 2. Library detects this and skips the placebo step rather than erroring.

*Empirical application (Section 5 - Banking deregulation panel):*

Favara and Imbs (2015) revisit: 12-year panel (1994-2005) of US counties; treatment is the number of restrictions lifted by their state under the Interstate Banking and Branching Efficiency Act (IBBEA). Outcomes are log growth rate of mortgage volume and log growth rate of house prices. Design statistics (page 39): 8 states never deregulate; 33 states deregulate once; 8 deregulate twice; 1 deregulates three times. 38 of 42 deregulating states first do so in 1995-1998. Max informative horizon `l = 2005 - 1998 + 1 = 8`.

**Switcher attrition at long horizons (page 39):**
```
At l = 1: 905 counties.
At l = 8: 773 counties (85% of original).
At l = 9: 357 counties (39% of original) -- AUTHORS DO NOT REPORT.
At l = 10: 238 counties.
At l = 11: 1 county.
```

> "Hence, we do not report estimates of those parameters." (page 39)

**This is the direct empirical rationale for the diff-diff library's `<50%-switcher` warning at long horizons (REGISTRY ChaisemartinDHaultfoeuille section).**

Similarly for placebos: "five placebo estimators can be computed, but only three apply to more than 50% of the 905 counties whose treatment changes at least once. The four and fifth placebos only apply to 128 and 120 counties respectively, so we do not report them" (page 39).

**Loan-volume results (page 39, Figure 1 top-center):**
- `DID_1 = 0.043` (s.e. 0.035), insignificant.
- `DID_3 = 0.081` (s.e. 0.049), significant at 10%.
- `DID_5 = 0.148` (s.e. 0.064), significant at 5%.
- Placebos jointly insignificant (F-test p-value = 0.400).

**Normalized loan-volume (page 40, Figure 1 top-right):**
- Restricted to 632 counties (773 with `DID_{g,8}` computable, minus 141 multi-switch counties).
- Lemma 3 Point 1 test of `l -> DID^n_l` constancy: not rejected (p-value = 0.328).
- Authors' interpretation: local-projection's "deregulations only have short-lived effects on mortgage volume" is an artifact of declining weights `sum w^{lp,l}_{g,k}`, NOT declining effects.

**House-prices results (page 40, Figure 1 bottom):**
- `DID_1 = 0.003` (s.e. 0.004), insignificant.
- `DID_4 = 0.016` (s.e. 0.009), significant at 10%.
- `DID_5 = 0.026` (s.e. 0.010), significant at 5%.
- `DID^n_l` increases with `l` - deregulations have long-lasting effects on house prices.
- Placebos jointly insignificant (F-test p-value = 0.703).

**Inference:** 95% CIs from normal approximation with standard errors clustered at the state level (Figure 1 caption).

*Extensions (Section 4 + Web Appendix Section 1):*

- **Covariate adjustment `DID_l^X`** (Web Appendix Section 1.2, Assumption 11):
  ```
  E[Y_{g,t}(D_{g,1,t}) - Y_{g,t-1}(D_{g,1,t-1}) - (X_{g,t} - X_{g,t-1})' theta_{D_{g,1}} | D, X]
    = E[Y_{g',t}(...) - Y_{g',t-1}(...) - (X_{g',t} - X_{g',t-1})' theta_{D_{g',1}} | D, X]
  ```
  Two-step estimator: per-baseline-treatment OLS of `Delta Y` on `Delta X` and time FE in not-yet-treated subsample yields `theta_hat_d`; then substitute residual differences into `DID_{g,l}`.

- **Group-specific linear trends `DID^{fd}_{g,l}`** (Web Appendix Section 1.3, Assumption 12). Generalizes Mora-Reggio (2019) to staggered timing. Web Appendix §1.3 (PDF page 50) gives:
  ```
  DID^{fd}_{g,l} = Y_{g, F_g - 1 + l} - Y_{g, F_g - 1 + l - 1} - (Y_{g, F_g - 1} - Y_{g, F_g - 2})
                  - (1 / N^g_{F_g - 1 + l}) * sum_{g' : D_{g',1} = D_{g,1}, F_{g'} > F_g - 1 + l}
                      (Y_{g', F_g - 1 + l} - Y_{g', F_g - 1 + l - 1} - (Y_{g', F_g - 1} - Y_{g', F_g - 2}))
  ```
  The construction is a DID on **first-differenced outcomes**: take group `g`'s outcome change from period `F_g - 1 + l - 1` to `F_g - 1 + l` (the `l`-th post-switch first difference), subtract the baseline first difference `Y_{g, F_g - 1} - Y_{g, F_g - 2}`, and subtract the same-baseline-not-yet-switched groups' analogue. **Lemma 6**: under Assumptions 3 + 12, `E[DID^{fd}_{g,l} | D] = delta_{g,l} - delta_{g,l-1}` with convention `delta_{g,0} = 0`. Identification floor: requires `F_g >= 3` (two pre-periods to construct the baseline first-difference).

- **State-set-specific trends** (Web Appendix Section 1.4, Assumptions 13 + 14): partition groups by a time-invariant covariate `s(g)` (e.g., state membership for county-level panel); restrict controls to groups in the same set. Stata `trends_nonparam` option.

- **Heterogeneity-of-treatment-effects testing via covariate predictors** (Web Appendix Section 1.5). Estimator `beta_hat^het_l` regresses `S_g * (Y_{g, F_g-1+l} - Y_{g, F_g-1})` on `X_g` and indicators for `F_g x D_{g,1} x S_g` in subsample `{g : F_g - 1 + l <= T_g}`. **Lemma 7**: `E[beta_hat^het_l | D, X] = beta^het_l` under Assumptions 1, 3, 15. **Paper-backed inference claim:** standard OLS inference on the regression is valid (NOT the cohort-clustered variance); the paper notes (page 51, appendix p. 6) that this works because the regression outcome is `S_g * (Y - Y)` not `S_g * DID_{g,l}`, so the inference does not need to account for estimation noise in the per-group `DID_{g,l}`. **Library implementation choice:** the dCDH heterogeneity-test SE is computed via the library's default `solve_ols(..., vcov_type="hc1")` HC1 heteroskedasticity-robust variance (`diff_diff/linalg.py` + the dCDH dispatch at `chaisemartin_dhaultfoeuille.py:5231-5261`); the paper does not specify a heteroskedasticity-robust variant, so HC1 is a library decision rather than a paper-backed default.

- **Design 2 switching-in/out** (Web Appendix Section 1.6, Assumption 16). Treatment binary, units join and leave. Separate `delta^+_l` (effect among groups that have NOT yet switched out `l` periods after switching in) from `delta^-_{g,l}` (switching-out effect). The switching-out estimator uses controls `{g' : F_{g'} = F_g, E_{g'} >= E_g + l}` - same switch-in date as `g`, never left.

- **Fuzzy designs** (Web Appendix Section 1.7). Cell-level `D_{g,t}` and `Y_{g,t}` become unit-averages within `(g, t)`. Estimators remain unbiased under Assumptions 3 + 4 PROVIDED no group is partly treated at period one.

- **Lag-`k` restriction** (Web Appendix Section 1.8, Assumption 17-`k`): rules out the effect of past treatments beyond `k` lags. Used as solution to the initial-conditions problem when pre-panel treatments may affect outcomes.

*Comparison with current practice (Section 4):*

The paper devotes Section 4 (pages 30-36) to decomposing three TWFE-derived practices into the dynamic per-group ATEs. All three suffer different forms of contamination relative to `DID_l`:

- **TWFE event-study with interactions (Regression 1, Proposition 1, Equation 25):** decomposes into a weighted sum of effects with weights `w^{fe}_g = I_g (I_g - I_bar) / sum_{g'} I_{g'} (I_{g'} - I_bar)`. Weights sum to 1 but groups with `I_g < I_bar` get NEGATIVE weight. The pre-period coefficients (`l < 0`) are valid placebos for joint A3+A4 testing (Proposition 1 Point 2, Equation 26).

- **Local-projection panel regressions (Regression 2, Proposition 2, page 36):** binary-staggered case. The local-projection coefficient at horizon `l` regresses `Delta Y_{g, t+l}` on `D_{g, t-1}` (or `D_{g, t}`), county and year FEs, and controls. In the Favara-Imbs revisit (Section 5.2), `beta_hat^{lp}_1` is a weighted sum of 7,626 effects with 4,670 positive (+1.067) and 2,956 negative (-0.125) weights; net 0.942 (not 1). For `l in {2, ..., underline F}`, weights sum strictly less than 1 -> downward bias even under homogeneity. `beta_hat^{lp}_4` has weights summing to -0.018, so the local-projection coefficient at horizon 4 has expectation of opposite sign from the true effect.

- **Distributed-lag regressions (Regression 3, Proposition 3, Equation 27):** decomposes into the `l`th-lag effect (weights sum to 1, possibly negative) plus `K` contamination terms from other lags (weights sum to 0). Application of Corollary 1 in de Chaisemartin & D'Haultfœuille (2023).

*Edge cases (full list):*

- **Long horizons `l`:** few switchers contribute. Library emits `<50%-switcher warning` when `N_l / N_1 < 0.5`. The paper's banking-deregulation application is the canonical citation.

- **Singleton-cohort degeneracy:** a cohort `C^G_k` with `#C^G_k = 1` has `U_k_bar = U^G_{g,l}` (the lone group's own score), so the within-cohort centered term `(U^G_{g,l} - U_k_bar)^2 = 0` contributes nothing to `sigma_hat_l^2`. The library handles this per footnote 15 of the dynamic paper: singleton cohorts are excluded from variance computation. When EVERY variance-eligible group forms a singleton cohort, `sigma_hat_l^2 = 0` and the library returns `overall_se = NaN` with a `UserWarning` rather than collapsing to `0.0`. The Web Appendix proof handles this via the convention `T^G_k = 0` and `omega^G_k = 0` for cohorts with `sum_{g in C^G_k} V(W^G_{g,l} | D) = 0` (page 65).

- **Never-switching groups:** `F_g > T` so `S_g = 0` and the score `U^G_{g,l}` carries zero estimand loading. **But never-switching groups appear in `N^g_{F_g-1+l}` denominators as stable controls for switchers in their `D_{g,1}` cohort.** Hence `lambda^G_{g,l}` for never-switchers is NON-ZERO via the stable-control role. The Web Appendix derivation (page 66) makes this explicit: `N^k_t >= #C^G_k` if `f_k > t`, so the stable-control pool is at least as large as the cohort cardinality. **This is the library's "Round 2 full-IF fix"** (`chaisemartin_dhaultfoeuille_results.py:355-358`) - never-switching groups participate in `sigma_hat_l^2` not via their own treatment effect (which does not exist) but via the perturbation they contribute to the comparison-cohort mean for each switcher. The `n_groups_dropped_never_switching` field is retained for backwards compatibility but no longer represents an actual exclusion.

- **Anticipation (paper concept, not exposed by the library):** the paper's Section 3.5 "Other extensions" framework discusses shifting placebo definitions and eligible horizon ranges backwards by `K` periods to accommodate treatment anticipation. The library does NOT currently expose an `anticipation` parameter; the fix uses `F_g - 1` as the canonical pre-period anchor (no shift). Anticipation-aware analyses would require a manual horizon-window re-mapping by the caller. Tracked as a potential future extension.

- **Terminal missingness:** library retains terminal-missing rows via the per-period `present = (N_mat[:, t] > 0) & (N_mat[:, t-1] > 0)` mask. This is a deviation from R `DIDmultiplegtDYN` - documented in REGISTRY.

- **Identification floor for Lemma 6 (`DID^{fd}`):** requires `F_g >= 3` (two pre-periods to construct `Y_{g, F_g - 1} - Y_{g, F_g - 2}`). Groups with `F_g in {1, 2}` drop out of the `DID^{fd}` path.

- **Lyapunov CLT moment condition (Web Appendix Equation 40):** convergence requires `2 + delta` moments of `Y_{g, t}`, satisfied under Assumption 10.

- **Convergence rate:** `sqrt(N_l)`, driven by **switcher count** at horizon `l`, NOT total panel size `G`. Long horizons with few switchers converge slowly.

**Reference implementation(s):**
- R: `DIDmultiplegtDYN::did_multiplegt_dyn()` (the dynamic-effects successor to the AER 2020 paper's `DIDmultiplegt`); covers placebos, covariates (`controls` option), set-of-groups trends (`trends_nonparam` option), and switching-out estimation in Design 2.
- Stata: `did_multiplegt_dyn` (SSC).
- Python: `diff_diff.ChaisemartinDHaultfoeuille` with the multi-horizon `DID_{g,l}` path enabled via `L_max >= 1` (the analytical SE path matches Section 3.4 / Web Appendix Section 3.7.3). Setting `L_max = None` falls back to the AER 2020 per-period `DID_M` path covered in the companion review.

---

## Implementation Notes

### Data Structure Requirements

- Panel of `G` groups by `T` periods. The paper allows ragged panels (page 8); the library extension requires a balanced baseline (every group observed at the first global period) and supports terminal missingness via the per-period `present` mask (REGISTRY).
- **Treatment scope (paper vs library):** The paper's main setup (Section 2) defines treatment as a general variable `D_{g,t}` allowing ordinal / continuous values; the binary case is a special case (REGISTRY `## ChaisemartinDHaultfoeuille` L495 captures this explicitly). The **library** has two paths with different scopes: Phase 1 `DID_M` (`L_max=None`) uses binary joiner/leaver categorization and requires binary treatment; Phase 2 multi-horizon `DID_{g,l}` (`L_max >= 1`) follows the paper's general non-binary setup, with baselines `D_{g,1}` (float), cohorts defined by `(D_{g,1}, F_g, S_g)` where `S_g = sign(D_{g,F_g} - D_{g,1})`, and same-baseline / same-timing / same-sign pooling within cohorts.
- Group-level panel - typically geographic entities (states, counties, municipalities) but the framework accepts individuals or firms.
- May be constructed from individual-level / repeated-cross-section data by `(g, t)` aggregation.
- Period-one treatment `D_{g,1}` defines cohorts (combined with `F_g` and `S_g`).

### Computational Considerations

- `DID_l` is O(`G * L`) for `L` horizons (per-group building block plus aggregation).
- Cohort-recentered variance is O(`G * K`) where `K` is the number of distinct cohorts. The per-cohort mean `U_k_bar` is reused across all groups in the cohort.
- **No bootstrap required for the default analytical SE path** - a significant performance advantage. The library exposes an opt-in PSU-level Hall-Mammen wild bootstrap for survey-design extensions (REGISTRY) but the default inference is analytical.
- Memory is dominated by the per-group score vector `lambda^G_{g,l}` (O(`G * T * L`)) and the cohort-recentered IF matrix. For `G` in the tens of thousands and `L < 20`, this fits comfortably in RAM.
- The Lyapunov CLT convergence rate scales as `sqrt(N_l)` where `N_l` is the **switcher count at horizon `l`**, not the panel size. Implementations should expose `N_l` per horizon so users can spot horizons with few switchers.

### Tuning Parameters

| Parameter | Library API | Type | Default | Selection method |
|-----------|------|------|---------|-----------------|
| Max horizon (paper: `L`; R: `effects`) | `L_max` on `fit()` | `Optional[int]` | `None` | `None` falls back to AER 2020 per-period `DID_M`; `>= 1` activates the dynamic event study. Library emits a `<50%-switcher` warning when `N_l / N_1 < 0.5` per the Favara-Imbs application (page 39). |
| Placebo gate (paper: `L^pl`) | `placebo` on `__init__` | `bool` | `True` | Boolean gate, not an integer count. The library computes the single-lag `DID_M^pl` on Phase 1 (`L_max=None`) and all `L_max` backward placebos on Phase 2; per-horizon placebo eligibility is bounded above by `max_g {min(T_g - F_g + 1, F_g - 2)}` automatically. Set to `False` to skip placebo computation; results still expose `placebo_*` fields with NaN values. |
| Cluster level | `cluster` on `__init__` | `Optional[Any]` | `None` (only supported value) | The estimator clusters at the group level by default (or PSU-level under `survey_design`). The `cluster=` kwarg rejects any non-`None` value via `NotImplementedError` — custom user-specified clustering is reserved for a future phase (see REGISTRY's cluster-contract Note). |
| Bootstrap iteration count | `n_bootstrap` on `__init__` | `int` | `0` (analytical default) | `0` uses the cohort-recentered analytical plug-in variance (Section 3.4 / Web Appendix Section 3.7.3). `>= 199` activates the multiplier bootstrap; PSU-level Hall-Mammen wild bootstrap is automatic under `survey_design`. |
| Bootstrap weight distribution | `bootstrap_weights` on `__init__` | str | `"rademacher"` | `"rademacher"`, `"mammen"`, or `"webb"`. Ignored when `n_bootstrap = 0`. |

### Relation to Existing diff-diff Estimators

- **Implemented as the multi-horizon path of `diff_diff.ChaisemartinDHaultfoeuille`** (the same class that hosts the AER 2020 `DID_M` static path; see companion review).
- The library's analytical SE for `DID_l` uses the Section 3.4 plug-in variance formula directly. The Web Appendix Section 3.7.3 derivation guides the cohort recentering, with the per-cohort centered IF squared and summed.
- **Round 2 full-IF fix** (`chaisemartin_dhaultfoeuille_results.py:355-358`) implements the "never-switching groups participate in variance via stable-control roles" behavior derived in Web Appendix Section 3.7.3 (page 66, `N^k_t >= #C^G_k`).
- **6 documented deviations from R `DIDmultiplegtDYN`** (see REGISTRY ChaisemartinDHaultfoeuille section for the full text and citations):
  - **Equal-cell weighting (Python) vs cell-size weighting (R):** the paper's main-text equations use observation counts `N_{g,t}` as weights. The library aggregates to cell means up-front and weights each cell equally. Carries forward to all Phase 2 estimands (`DID_l`, `DID^{pl}_l`, `DID^n_l`, `delta`).
  - **Period-based vs cohort-based stable controls:** Python uses period-based `stable_0(t)`, `stable_1(t)` per Theorem 3 of AER 2020. R additionally conditions on `D_{g,1}`. The two agree on pure-direction panels; they disagree by O(1%) on mixed-direction panels.
  - **`<50%-switcher` warning at far horizons:** Phase 2 extension; warns when fewer than 50% of `l=1` switchers contribute at horizon `l`. Direct citation: this paper's Section 5 (page 39).
  - **Terminal-missingness retention:** groups missing one or more later periods are kept (the `present` mask handles the missing transitions), instead of being dropped wholesale.
  - **SE normalization ~4% smaller than R:** Python implements the paper's Section 3.7.3 plug-in formula verbatim with `SE = sigma_hat / sqrt(N_l)`. R normalizes the influence function by `G` (total number of groups including never-switchers and stable controls). Both converge to the same asymptotic variance as `G -> infinity`. In finite samples R's formula produces slightly larger (more conservative) SEs (~3.5-5.1% gap, deterministic on identical data). Since the paper's formula is already an upper bound on the true variance (Web Appendix Equation 54, Jensen under Assumption 8), Python's tighter SE remains conservative.
  - **Singleton-cohort degeneracy NaN handling:** singleton-baseline groups are excluded from the variance computation only (per footnote 15 of the dynamic paper) but retained in the point-estimate sample as period-based stable controls. When every variance-eligible group forms its own cohort, Python returns `overall_se = NaN` with a `UserWarning` rather than silently collapsing to `0.0`. R returns a non-zero SE via small-sample sandwich machinery the library does not implement.
- The Section 4 negative-weights diagnostic (`twfe_diagnostic = True` in `fit()`) extends the AER 2020 Theorem 1 decomposition to the staggered design case; covers Regression 1, Regression 2, Regression 3 contamination patterns documented in Propositions 1-3.

---

## Gaps and Uncertainties

- **Sections of the paper NOT covered by the extraction passes:**
  - **Section 3.6** ("Test of treatment-effect homogeneity"): joint testing across cohorts likely covered here. Not in the four extraction files.
  - **Section 3.8** (whatever its content): not extracted.
  - **Sections 3.6, 3.8 transcriptions are deferred** to a future paper-review pass if needed.

- **Web Appendix sections beyond Section 3.7.3 (page 67-70) that ARE covered:** the four extension subsections (1.1-1.8) plus the cohort decomposition (Section 3.7.2, page 65) plus the variance derivation (Section 3.7.3, pages 67-70). Sections of the appendix NOT covered:
  - Detailed proofs of Lemmas 1-7 and Propositions 1-2 - high-level structure captured, line-by-line algebraic steps deferred.
  - Section 2 (literature review table of 25 highly-cited AER 2015-2019 papers) - census statistics summarized but the table itself not transcribed.
  - Heterogeneity test asymptotic distribution beyond the unbiasedness statement of Lemma 7.

- **Bootstrap procedure (or lack thereof) in this paper:** pages 16-35 do NOT propose or transcribe a bootstrap procedure for the dynamic estimators. The inference apparatus in Section 3.4 is fully analytical (cohort-recentered plug-in variance + normal-quantile CI). **The library's "bootstrap SE inherits analytical IF" framing is consistent** with the paper's analytical-only inference: when a bootstrap SE is computed, it operates on the same influence-function representation as the analytical SE, so any deviations from R documented in REGISTRY apply equally to both the analytical and bootstrap paths.

- **Per-group score `U^G_{g,l}` definition:** transcribed at Equation 22 (page 27) as `U^G_{g,l} = (lambda^G_{g,l})' * Y_g` with the component-by-component formula for `lambda^G_{g,l,t}` (above). This is consistent with the Section 3.7.2 cohort decomposition (page 65) and the influence-function bound `||lambda^G_{k,l}||^2 <= T * (1 + max_t N^k_{t,l} / N^k_t)` in Equation 46.

- **Equation-numbering reconciliation across the four extraction files:** the per-group `DID_{g,l}` is consistently labeled Equation 3 (in Sections 1 and 3 of the extraction files). The aggregate `DID_l` is consistently labeled Equation 5. The normalized `DID^n_l` is labeled Equation 15 in Section 2 and Equation 13 (estimand `delta^n_l`) in Section 1. The cost-benefit aggregate `delta` is consistently labeled Equation 17 (estimand) with the sample analogue in the text rather than as a separately numbered equation. No contradictions in equation labels detected; the cohort-recentered variance does NOT have a single equation number - it is presented across page 28 (feasible plug-in) and Web Appendix Equation 49 (consistency statement).

- **`DID^{fd}_{g,l}` transcription source-locked:** The NBER WP 29873 Web Appendix Section 1.3 (July 2023 revision, Web Appendix page 4) gives the canonical form with `Y_{g, F_g - 1 + l - 1}` (i.e., the `(l-1)`-th post-switch period) as the lagged term in the outcome first-difference. Earlier extraction passes by sub-agents produced two slightly different transcriptions (one with `F_g - 1 - l - 1`, another with `F_g - 1 + l - 1`); the latter was verified against the PDF and is the one transcribed in the main "Extensions" section above. The Lemma 6 characterization `E[DID^{fd}_{g,l} | D] = delta_{g,l} - delta_{g,l-1}` is the unambiguous identification statement and is the appropriate target for implementation tests.

- **No Monte Carlo / simulation evidence in the paper:** the paper relies on the Favara-Imbs (2015) banking-deregulation empirical application as its sole validation surface (Section 5). The Web Appendix has no simulation studies. Implementations should be tested against R `DIDmultiplegtDYN` on the same Favara-Imbs panel, which is the canonical parity anchor.

- **Revision of record:** this review covers the "March 2022, revised July 2023" version of NBER WP 29873. Local PDFs are gitignored under `/papers/`; the NBER landing page (https://www.nber.org/papers/w29873) is the authoritative source. The same revision string is used in `docs/references.rst:199`, `docs/methodology/REGISTRY.md:488`, and the code docstring at `diff_diff/chaisemartin_dhaultfoeuille_results.py:16`, aligned in the same PR that adds this review. If a later NBER revision is posted, a follow-up paper-review pass against that revision would extend this surface; the methodology transcribed above (six numbered assumptions, cohort-recentered plug-in variance, the `DID_l` / `DID^n_l` / `DID_l^pl` / `DID^X` / `DID^{fd}` constructions) is keyed to the July 2023 revision and any future-revision diffs are out of scope for this review.

For cross-reference with library implementation deviations, consult `docs/methodology/REGISTRY.md` ChaisemartinDHaultfoeuille section. For the AER 2020 static `DID_M` half of the same library entry, see `docs/methodology/papers/dechaisemartin-dhaultfoeuille-2020-review.md`.
