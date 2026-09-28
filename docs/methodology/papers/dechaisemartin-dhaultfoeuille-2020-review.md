# Paper Review: Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects

**Authors:** Clément de Chaisemartin and Xavier D'Haultfœuille
**Citation:** de Chaisemartin, C., & D'Haultfœuille, X. (2020). Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects. *American Economic Review*, 110(9), 2964-2996. DOI: 10.1257/aer.20181169.
**PDF reviewed:** American Economic Review version (DOI: [10.1257/aer.20181169](https://doi.org/10.1257/aer.20181169)). Local PDFs are gitignored under `/papers/`; the journal/DOI version is the authoritative source.
**Review date:** 2026-05-21

---

## Methodology Registry Entry

*Formatted to match `docs/methodology/REGISTRY.md` structure. Heading levels and labels align with existing entries - copy the `## ChaisemartinDHaultfoeuille` section into the registry if you're extending the existing entry.*

## ChaisemartinDHaultfoeuille (2020 AER component)

**Primary source:** de Chaisemartin & D'Haultfœuille (2020), *Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects*, AER 110(9): 2964-2996. DOI 10.1257/aer.20181169.

**Scope:** The foundational paper introducing the DCDH framework. Establishes (a) the TWFE-decomposition diagnostic - showing OLS two-way fixed effects on a treatment indicator estimates a weighted average of treatment effects with potentially negative weights (Theorem 1 / Equation 1); and (b) the `DID_M`, `DID_{+,t}`, `DID_{-,t}` estimators (Theorem 3, page 2978) and their multi-lag placebos (Theorem 4, page 2980) as bias-robust alternatives. The 2022, revised July 2023 NBER WP 29873 companion paper adds dynamic event-study estimators (`DID_l`, `DID_l^pl`) and cohort-recentered plug-in variance. The 2020 paper covers (i) individual-level panel, (ii) repeated cross-section (where groups are e.g. counties of birth), (iii) cross-section with cohort-of-birth playing the role of time (e.g. Duflo 2001 Indonesia schooling), and (iv) the `N_{g,t} = 1` per-cell special case, all under the same `G x T` cell-level setup.

**Key implementation requirements:**

*Assumption checks / warnings (the 13 numbered assumptions):*

- **Assumption 1 (Balanced Panel of Groups, page 2967):** For all `(g,t)`, `N_{g,t} > 0`. Relaxable but complicates denominators. Implementation should validate every group has at least one observation in every period (or accept ragged panels under documented terminal-missingness conventions).
- **Assumption 2 (Sharp Design, page 2968):** For all `(g,t)` and `i`, `D_{i,g,t} = D_{g,t}` (no within-cell treatment variation). Fuzzy designs are deferred to the online Appendix. Implementations should validate that cell-min equals cell-max on the treatment column and raise an error if the data is fuzzy (since the sharp-design `DID_M` is no longer the correct estimator).
- **Assumption 3 (Independent Groups):** The vectors `(Y_{g,t}(0), Y_{g,t}(1), D_{g,t})_{1<=t<=T}` are mutually independent across `g`. This is the standard panel-DiD independence assumption; bootstrap inference clusters at the group level to respect it.
- **Assumption 4 (Strong Exogeneity for Y(0)):** `E(Y_{g,t}(0) - Y_{g,t-1}(0) | D_{g,1}, ..., D_{g,T}) = E(Y_{g,t}(0) - Y_{g,t-1}(0))`. Rules out Ashenfelter's dip / treatment-driven adjustment of pre-period outcomes.
- **Assumption 5 (Common Trends for Y(0)):** For `t >= 2`, `E(Y_{g,t}(0) - Y_{g,t-1}(0))` does not vary across `g`. The standard parallel-trends assumption on the untreated potential outcome path.
- **Assumption 6 (Staggered Adoption Designs):** `D_{g,t} >= D_{g,t-1}` for all `t >= 2`. Used only for sub-results (Proposition 2). Implies `DID_M = DID_+` (no leavers).
- **Assumption 7 (`w` Uncorrelated with `Delta-bar`):** `E[ sum_{(g,t):D_{g,t}=1} (N_{g,t}/N_1) (w_{g,t} - 1) * (Delta-bar_{g,t} - Delta-bar^{TR}) ] = 0`. If this holds, `beta_fe = delta^{TR}` even with negative weights (Corollary 2). Not testable directly; usually invoked as a defense of TWFE when negative-weight diagnostics are bad.
- **Assumption 8 (`w_{fd}` Uncorrelated with `Delta-bar`):** First-difference analogue of Assumption 7 (page 2976). Required for `beta_fd = delta^{TR}` under negative FD weights.
- **Assumption 9 (Strong Exogeneity for Y(1)):** `E(Y_{g,t}(1) - Y_{g,t-1}(1) | D_{g,1}, ..., D_{g,T}) = E(Y_{g,t}(1) - Y_{g,t-1}(1))`. Required for "leavers" (`DID_-`). Symmetric to Assumption 4 but on the treated potential outcome.
- **Assumption 10 (Common Trends for Y(1)):** For `t >= 2`, `E(Y_{g,t}(1) - Y_{g,t-1}(1))` does not vary across `g`. The Y(1) analog of Assumption 5; required for the leavers' component.
- **Assumption 11 (Existence of "Stable" Groups, page 2977):** For all `t >= 2`:
  - (i) If at least one group is a "joiner" (`D_{g,t-1}=0, D_{g,t}=1`), then there exists another group `g'` with `D_{g',t-1} = D_{g',t} = 0`.
  - (ii) If at least one group is a "leaver" (`D_{g,t-1}=1, D_{g,t}=0`), then there exists another group `g'` with `D_{g',t-1} = D_{g',t} = 1`.
  Implementations must detect per-period A11 violations and either drop that period's contribution (set the relevant `DID_{+,t}` / `DID_{-,t}` to 0) or emit a warning. The paper's AER survey (Section V.A) finds ~80% of applications satisfy A11.
- **Assumption 12 (Mean Independence Between a Group's Outcome and Other Groups' Treatments):** For all `g`, `t`: `E(Y_{g,t}(0) | D) = E(Y_{g,t}(0) | D_g)` and `E(Y_{g,t}(1) | D) = E(Y_{g,t}(1) | D_g)`. Weaker than Assumption 3, used jointly with A11 (which makes treatments not independent across groups by construction).
- **Assumption 13 (Existence of "Stable" Groups for the Placebo Test, page 2979):** Analog of A11 extended one period back (requires `t-2, t-1, t` stable groups). Used for `DID_M^pl`. Each additional placebo lag requires deeper stable-group support.

Treatment is assumed binary throughout the main paper; ordered/nonbinary treatment extension is in online Appendix Section 3.2 (page 2983, footnote 18 calls out that Application 1's newspaper count is technically nonbinary and the parameter is the average causal response).

*Estimator equations:*

**Regression 1 (TWFE, page 2968):** OLS regression of `Y_{i,g,t}` on group FE, period FE, and `D_{g,t}`, giving coefficient `hat{beta}_fe`. Footnote 5 (page 2968): this is equivalent to a `(g,t)`-level regression of `Y_{g,t}` on group + period FE + `D_{g,t}`, weighted by `N_{g,t}`. Auxiliary FE regression of treatment on FEs (page 2969):

    D_{g,t} = alpha + gamma_g + lambda_t + epsilon_{g,t}

defines the residual `epsilon_{g,t}` used in the weight formulas.

**TWFE weights (page 2970):**

    w_{g,t} = epsilon_{g,t} / sum_{(g',t'): D_{g',t'}=1} (N_{g',t'}/N_1) * epsilon_{g',t'}

**Theorem 1 - headline TWFE decomposition (Equation 1, page 2970):** Under Assumptions 1-5,

    beta_fe = E[ sum_{(g,t): D_{g,t}=1} (N_{g,t}/N_1) * w_{g,t} * Delta_{g,t} ]

Footnote 7 strengthens to the conditional version:

    E[ hat{beta}_fe | D ] = sum_{(g,t): D_{g,t}=1} (N_{g,t}/N_1) * w_{g,t} * E[Delta_{g,t} | D]

The weights `(N_{g,t}/N_1) * w_{g,t}` sum to 1 but can individually be negative.

**Regression 2 (First-Difference, page 2975):** OLS of `Y_{g,t} - Y_{g,t-1}` on period FE and `D_{g,t} - D_{g,t-1}` (for `t >= 2`). With `epsilon_{fd,g,t}` the residual of `D_{g,t} - D_{g,t-1}` on period FE and the convention `epsilon_{fd,g,1} = epsilon_{fd,g,T+1} = 0`:

    w_{fd,g,t} = (epsilon_{fd,g,t} - (N_{g,t+1}/N_{g,t}) * epsilon_{fd,g,t+1})
               / sum_{(g',t'): D_{g',t'}=1} (N_{g',t'}/N_1) * (epsilon_{fd,g',t'} - (N_{g',t'+1}/N_{g',t'}) * epsilon_{fd,g',t'+1})

**Theorem 2 - FD decomposition (page 2975):** Under Assumptions 1-5,

    beta_fd = E[ sum_{(g,t): D_{g,t}=1} (N_{g,t}/N_1) * w_{fd,g,t} * Delta_{g,t} ]

Same negative-weights pathology as `beta_fe`.

**DID_M building blocks (Equation 3 and surrounding text, page 2978):** Define cell counts

    N_{d,d',t} = sum_{g: D_{g,t}=d, D_{g,t-1}=d'} N_{g,t}

so `N_{1,0,t}` = observations that joined at `t`, `N_{0,1,t}` = observations that left at `t`, `N_{0,0,t}` = stayed untreated, `N_{1,1,t}` = stayed treated. Then for `t in {2, ..., T}`:

    DID_{+,t} =   sum_{g: D_{g,t}=1, D_{g,t-1}=0} (N_{g,t} / N_{1,0,t}) * (Y_{g,t} - Y_{g,t-1})
                - sum_{g: D_{g,t}=D_{g,t-1}=0}    (N_{g,t} / N_{0,0,t}) * (Y_{g,t} - Y_{g,t-1})

    DID_{-,t} =   sum_{g: D_{g,t}=D_{g,t-1}=1}    (N_{g,t} / N_{1,1,t}) * (Y_{g,t} - Y_{g,t-1})
                - sum_{g: D_{g,t}=0, D_{g,t-1}=1} (N_{g,t} / N_{0,1,t}) * (Y_{g,t} - Y_{g,t-1})

Boundary conventions: `DID_{+,t} = 0` when there is no joiner OR no fully-untreated control at `t`; `DID_{-,t} = 0` when there is no leaver OR no fully-treated control at `t`. The paper's notation uses observation counts `N_{g,t}` / `N_{a,b,t}` as weights (cell-size weighting); see the library deviation note below.

**`DID_M` aggregation (page 2978):**

    N_S = sum_{(g,t): t>=2, D_{g,t} != D_{g,t-1}} N_{g,t}

    DID_M = sum_{t=2}^{T} ( (N_{1,0,t} / N_S) * DID_{+,t}  +  (N_{0,1,t} / N_S) * DID_{-,t} )

Joiners-only and leavers-only variants are constructed by reweighting just one component (the paper notes this on page 2978 lower; explicit formula is the natural cell-share weighting on `N_{1,0,t}` / `sum_t N_{1,0,t}` for `DID_+` and symmetrically for `DID_-`). In a staggered-adoption design no group leaves treatment, so `DID_M = DID_+`.

**Theorem 3 (page 2978):** Under Assumptions 1, 2, 4, 5, 9-12, `E[DID_M] = delta^S` where the switching-cell ATE is

    delta^S = E[ (1/N_S) * sum_{(i,g,t): t>=2, D_{g,t} != D_{g,t-1}} (Y_{i,g,t}(1) - Y_{i,g,t}(0)) ]

In a staggered-adoption design, `delta^S` is the average treatment effect at the moment a group starts receiving treatment, across all groups that ever become treated. Asymptotic normality of `DID_M` as `G -> infinity` is established in online Appendix Section 5 (outside the main text).

**Pretrends placebo `DID_M^pl` (page 2980):** Let `N_{d,d',d'',t} = sum_{g: D_{g,t}=d, D_{g,t-1}=d', D_{g,t-2}=d''} N_{g,t}` and `N_S^pl = sum_{(g,t): t>=3, D_{g,t} != D_{g,t-1} = D_{g,t-2}} N_{g,t}`. Then:

    DID_{+,t}^pl =   sum_{g: D_{g,t}=1, D_{g,t-1}=D_{g,t-2}=0} (N_{g,t}/N_{1,0,0,t}) * (Y_{g,t-1} - Y_{g,t-2})
                  - sum_{g: D_{g,t}=D_{g,t-1}=D_{g,t-2}=0}    (N_{g,t}/N_{0,0,0,t}) * (Y_{g,t-1} - Y_{g,t-2})

    DID_{-,t}^pl =   sum_{g: D_{g,t}=D_{g,t-1}=D_{g,t-2}=1}    (N_{g,t}/N_{1,1,1,t}) * (Y_{g,t-1} - Y_{g,t-2})
                  - sum_{g: D_{g,t}=0, D_{g,t-1}=D_{g,t-2}=1} (N_{g,t}/N_{0,1,1,t}) * (Y_{g,t-1} - Y_{g,t-2})

    DID_M^pl = sum_{t=3}^{T} ( (N_{1,0,0,t} / N_S^pl) * DID_{+,t}^pl  +  (N_{0,1,1,t} / N_S^pl) * DID_{-,t}^pl )

**Theorem 4 (page 2980):** Under Assumptions 1, 2, 4, 5, 9, 10, 12, 13, `E[DID_M^pl] = 0`. This is the paper's recommended pretrends test; it differs from the Autor (2003) event-study pretrends because, as Abraham and Sun (2018) showed, the latter is invalid under heterogeneous effects. Multi-lag placebos `DID_M^{pl,2}, DID_M^{pl,3}, ...` are constructed by extending the "stable group" requirement deeper (each lag drops observations - Application 2 sees `DID_M^pl` use 3,101 obs and `DID_M^{pl,3}` use 1,881 obs).

where:

- `G` = number of groups
- `T` = number of periods
- `N_{g,t}` = number of observations in group `g` at period `t`
- `N = sum_{g,t} N_{g,t}` = total observations
- `D_{i,g,t}` = treatment status of unit `i` in group `g` at period `t`
- `D_{g,t} = (1/N_{g,t}) sum_i D_{i,g,t}` = cell-average treatment
- `Y_{i,g,t}(0), Y_{i,g,t}(1)` = potential outcomes
- `Y_{g,t}(0), Y_{g,t}(1), Y_{g,t}` = cell averages of potential / observed outcomes
- `N_1 = sum_{i,g,t} D_{i,g,t}` = total number of treated units
- `Delta_{g,t} = (1/N_{g,t}) sum_i [Y_{i,g,t}(1) - Y_{i,g,t}(0)]` = ATE in cell `(g,t)`
- `Delta^{TR} = (1/N_1) sum_{(i,g,t): D=1} [Y(1) - Y(0)]` = ATE across all treated units
- `delta^{TR} = E[Delta^{TR}]` = ATT
- `delta^S` = switching-cell ATE (target of `DID_M`)
- `beta_fe = E[hat{beta}_fe]`, `beta_fd = E[hat{beta}_fd]` = population OLS coefficients
- `epsilon_{g,t}` = residual of `D_{g,t}` on group + period FE
- `w_{g,t}` = TWFE weight (formula above)
- `epsilon_{fd,g,t}` = residual of `D_{g,t} - D_{g,t-1}` on period FE
- `w_{fd,g,t}` = first-difference weight (formula above)
- `Delta-bar_{g,t} = E[Delta_{g,t} | D]`, `Delta-bar^{TR} = E[Delta^{TR} | D]` = conditional ATEs
- `beta-bar_fe = E[hat{beta}_fe | D]` = conditional expectation
- `n = #{(g,t): D_{g,t}=1}` = number of treated cells
- `w_{(1)} >= w_{(2)} >= ... >= w_{(n)}` = order statistics of treated-cell weights

*TWFE weights diagnostic (Equation 1 / Theorem 1):*

The headline negative-weights diagnostic is the decomposition

    beta_fe = E[ sum_{(g,t): D_{g,t}=1} (N_{g,t}/N_1) * w_{g,t} * Delta_{g,t} ]

with `w_{g,t}` defined above. The weights `(N_{g,t}/N_1) * w_{g,t}` sum to 1 but can be negative.

**Proposition 1 (page 2971-2972):** Suppose Assumption 1 holds and for all `t >= 2`, `N_{g,t}/N_{g,t-1}` does not vary across `g`. Then:
- For `(g, t, t')` with `D_{g,t} = D_{g,t'} = 1` and `D_{.,t} > D_{.,t'}`, we have `w_{g,t} < w_{g,t'}`. (Periods with many treated groups get more-negative weights.)
- For `(g, g', t)` with `D_{g,t} = D_{g',t} = 1` and `D_{g,..} > D_{g',..}`, we have `w_{g,t} < w_{g',t}`. (Groups treated for many periods get more-negative weights.)

Footnote 10 specializes to staggered designs: `w_{g,t}` is decreasing in `t`, so long-treated groups in later periods are most likely to carry negative weights. The intuition is that OLS implicitly uses long-treated cells as "controls" for short-treated cells, differencing out the former's treatment effect.

**Proposition 2 (page 2975):** Under Assumptions 1, 2, and 6 (staggered adoption) plus stationary `N_{g,t}`: for `(g,t)` such that `D_{g,t} = 1`, `w_{fd,g,t} < 0` iff `D_{g,t-1} = 1` and `D_{.,t} - D_{.,t-1} > D_{.,t+1} - D_{.,t}` (convention: `D_{.,T+1} = D_{.,T}`). **Implication:** in staggered designs, period-`t` ATEs of groups already treated in `t-1` (i.e., long-run effects) get negative weights when the treatment adoption pace decelerates. Negative weights are much more prevalent in the "more early adopters" case than "more late adopters" case. The Proposition 2 proof (page 2992) makes the endpoint conventions explicit:
- For `1 <= t <= T-1`: `w_{fd,g,t}` is strictly negative iff `D_{g,t-1} = 1` and `2*D_{.,t} - D_{.,t-1} - D_{.,t+1} > 0`.
- When `t = T`: `w_{fd,g,T}` is strictly negative iff `D_{g,T-1} = 1` and `D_{.,T} - D_{.,T-1} > 0`.
- When `t = 1`: under Assumption 6, `D_{g,1} = 1 => D_{g,2} = 1`, so `w_{fd,g,1}` has the sign of `D_{.,2} - D_{.,1} > 0` (positive).

*Robustness measures (Corollary 1, Corollary 2):*

**Corollary 1 (page 2973):** Under Assumptions 1-5, define

    sigma(Delta-bar) = ( sum_{(g,t):D_{g,t}=1} (N_{g,t}/N_1) * (Delta-bar_{g,t} - Delta-bar^{TR})^2 )^{1/2}
    sigma(w)        = ( sum_{(g,t):D_{g,t}=1} (N_{g,t}/N_1) * (w_{g,t} - 1)^2 )^{1/2}

- (i) If `sigma(w) > 0`, the minimal `sigma(Delta-bar)` compatible with `beta-bar_fe` and `Delta-bar^{TR} = 0` is

      sigma-underbar_fe = |beta-bar_fe| / sigma(w)

- (ii) If `beta-bar_fe != 0` AND at least one `w_{g,t}` is strictly negative, the minimal `sigma(Delta-bar)` compatible with `beta-bar_fe` and sign-flipped `Delta-bar_{g,t}` for every `(g,t)` is

      s_underbar_fe = |beta-bar_fe| / [T_s + S_s^2 / (1 - P_s)]^{1/2}

  where `s = min{i in {1,...,n}: w_{(i)} < -S_{(i)}/(1 - P_{(i)})}`, with `P_k = sum_{i>=k} N_{(i)}/N_1`, `S_k = sum_{i>=k} (N_{(i)}/N_1) * w_{(i)}`, `T_k = sum_{i>=k} (N_{(i)}/N_1) * w_{(i)}^2`.

A small `sigma-underbar_fe` indicates `beta_fe` could be sign-opposite to the ATT under modest treatment-effect heterogeneity. The numerical illustration on page 2971 (`1.5 * 1 - 0.5 * 4 = -0.5`) shows how a 1.5/-0.5 weight combination can deliver a negative `beta_fe` even when both cell ATEs are positive.

**Corollary 2 (page 2974):** Under Assumptions 1-5 and 7, `beta_fe = delta^{TR}`. When weights are uncorrelated with cell ATEs, TWFE is unbiased for the ATT even if individual weights are negative.

The FD analogues `sigma-underbar_fd` and `s_underbar_fd` are constructed identically with `w_{fd,g,t}` in place of `w_{g,t}` (page 2976 references both quantities and Assumption 8 is the FD analog of Assumption 7).

*Empirical applications (Section V, pages 2981-2986):*

- **Gentzkow-Shapiro-Sinkinson (newspapers, 1868-1928 US presidential turnout):** county-level FD design with state-year FE; `beta_fd = 0.0026` (SE 0.0009), `beta_fe = -0.0011` (SE 0.0011), `DID_M = 0.0043` (SE 0.0014). `DID_M` is 66% larger than `beta_fd` and OPPOSITE sign from `beta_fe`. Placebo `DID_M^pl = -0.0009` (not significant) supports parallel trends. The `beta_fe`/`beta_fd` divergence is significant (t = 2.86), implying A7 and A8 cannot jointly hold. For `beta_fe`: 40% of weights strictly negative, summing to -0.53. `sigma-underbar_fe = 3e-4`.
- **Vella-Verbeek (NLSY union premium, 1980-1987):** `beta_fe = 0.107` (SE 0.030), `beta_fd = 0.060`, `DID_M = 0.041` - `DID_M` differs significantly from both. Joiners' effect 0.059, leavers' effect 0.021 (insignificant difference, t = 0.55). Joiners' placebo `DID_M^pl = 0.119` (SE 0.051) reveals a differential positive pretrend, suggesting even the modest `DID_M` may overstate the premium. Conclusion: there may not be a significant union wage premium. Application 2 also demonstrates the recommended measurement-error cleanup: replace `D_{i,t}` for sub-sequences `D_{i,t-1}=0, D_{i,t}=1, D_{i,t+1}=0` by 0 (and symmetrically for the `1,0,1` pattern), discarding half the union-status changes.

*Applicability survey (Section V.A, pages 2981-2983):*
- Table 1: 33 papers using two-way FE regressions published in the AER 2010-2012 (9.8% of all papers; 19.1% of empirical papers excluding lab experiments).
- Table 2 Panel A: 13 FE OLS, 6 FD OLS, 6 FE/FD with several treatment variables, 3 FE/FD 2SLS, 5 other.
- Table 2 Panel B: 26 sharp designs, 7 fuzzy.
- Table 2 Panel C (A11 stable groups present?): 12 yes, 14 presumably yes, 5 presumably no, 2 no - so roughly 80% of the AER applications either definitely or presumably satisfy A11.

*Standard errors:*

The paper does not provide an explicit closed-form analytical variance for `DID_M` in the main body; the asymptotic normality result is deferred to online Appendix Section 5. The empirical applications use:
- Application 1 (page 2983-2984): standard errors **clustered by county** for the `DID_M` table entries.
- Application 2 (page 2985): standard errors **clustered at the worker level** for the `DID_M` table entries.
- Cross-estimator t-statistics (footnotes 23-24, pages 2985-2986): "The standard errors of `beta_hat_fe - DID_M` and `beta_hat_fd - DID_M` are computed with a worker-level clustered bootstrap." Clustered bootstrap is invoked specifically for the **cross-estimator difference tests** (`beta_hat_fe - DID_M`, `beta_hat_fd - DID_M`), not as the paper's recommended default for plain `DID_M` SEs.

The paper does not prescribe a bootstrap weight distribution (Rademacher / Mammen / Webb), an iteration count, or a step-by-step algorithm for `DID_M` SE construction. The **library's default Phase 1 inference** (`n_bootstrap=0`) is the dynamic companion (NBER WP 29873, revised July 2023) cohort-recentered analytical plug-in variance evaluated at horizon `l = 1` (see REGISTRY `## ChaisemartinDHaultfoeuille` Standard-errors block at L572-L598 and the per-cell `Lambda^G_{g,l=1}` weight derivation). `DID_M = DID_{l=1}` under the dynamic-companion notation, so the Section 3.7.3 plug-in covers it as a special case. **Library override:** setting `n_bootstrap > 0` replaces the analytical SE/CI/p-value surface with the corresponding multiplier-bootstrap percentile inference (REGISTRY L598 + L618). Both paths are library extensions over what the AER 2020 paper itself recommends (cluster-robust SEs for `DID_M` tables + clustered bootstrap for cross-estimator contrasts only).

*Edge cases:*

- **No joiners or no fully-untreated controls at some `t`:** `DID_{+,t} = 0`. **No leavers or no fully-treated controls at some `t`:** `DID_{-,t} = 0`. (Page 2978.)
- **Staggered-adoption design:** no group leaves treatment, so `DID_M = DID_+`. The leavers' component is identically zero and Assumptions 9-10 are not needed.
- **A11 violation:** if no stable-control group exists at some `t`, the corresponding switching observations contribute 0 to the aggregate. The paper notes (page 2977) that Assumption 11 is needed for *unbiasedness* but not for *consistency*; online Appendix Section 5 establishes consistency under only Assumption 3. The AER survey finds 7 of 33 papers either presumably do not or do not satisfy A11.
- **Bias-variance trade-off (page 2979):** under a correctly-specified Regression 1 with constant `delta` and homoskedastic, uncorrelated errors, `beta_hat_fe` is the Gauss-Markov efficient OLS estimator and `Var(beta_hat_fe) <= Var(DID_M)`. With heteroskedastic or correlated errors there are examples where `Var(beta_hat_fe) > Var(DID_M)`, but `DID_M` may often have larger variance than `beta_hat_fe`. This is empirically confirmed in both applications.
- **Common-trends violation diagnosed via placebo:** if `DID_M^pl != 0` significantly, parallel trends is suspect. Each additional placebo lag drops observations - implementations should report placebo-subsample sizes alongside placebo point estimates.
- **Treatment reversibility (leavers):** Assumptions 9 and 10 (Strong Exogeneity and Common Trends for `Y(1)`) are required only when leavers exist. They are not needed in staggered-adoption designs.
- **Measurement error in treatment:** Application 2 demonstrates the recommended cleanup - discard the middle observation of sub-sequences `0, 1, 0` and `1, 0, 1` over three consecutive years (replace `D_{i,t}` by `D_{i,t-1}`). Footnote 22 reports that keeping the original noisy data does not change the qualitative results much, except that `DID_M^{pl,2}` becomes significant.
- **Multi-period placebos require deeper stable-group support:** Application 2 shows the observation count sequence 3,815 (`DID_M`) -> 3,101 (`DID_M^pl`) -> 2,458 (`DID_M^{pl,2}`) -> 1,881 (`DID_M^{pl,3}`).
- **Nonbinary treatment:** the main paper assumes binary treatment; the average-causal-response extension is in online Appendix Section 3.2. Application 1's newspaper count is technically nonbinary; footnote 18 notes the parameter of interest is the average causal response rather than the ATT.

*Extensions in this paper (Section IV, page 2981):*

The paper sketches four extensions covered in the online Appendix:
- **Fuzzy DID / nonbinary treatment (online Appendix Sec 3.2):** Theorem 1 / 2 decomposition extends to fuzzy designs (within-`(g,t)` treatment variation) and to nonbinary treatments. Footnote 16 confirms the FD decomposition Theorem 2 also extends.
- **TWFE with covariates (Equations 20-21 territory in the appendix):** Replace `epsilon_{g,t}` by `epsilon_{g,t}^X`, the residual from regressing `D_{g,t}` on group + period FE *and* covariates `X_{g,t}`. Identification requires the modified parallel-trends assumption

      E(Y_{g,t}(0) | D_g, X_g) - E(Y_{g,t-1}(0) | D_g, X_g) = (X_{g,t} - X_{g,t-1})' * gamma + lambda_t

  for some `gamma` and `lambda_t`. With group-specific linear trends, this is equivalent to `gamma_g + lambda_t`. The paper warns: "two-way fixed effects regressions with covariates may rely on a more plausible common trends assumption than those without covariates, but they still require that the treatment effect be homogeneous, across time and between groups."
- **Identification under common trends and constant per-cell ATE:** Under common trends + constant ATE within `(g,t)` cells, `beta_fe` and `beta_fd` identify weighted sums of cell-switching ATEs. In sharp designs, the FD weights are all positive, while the FE weights are all positive only under staggered adoption.
- **Nonbinary discrete treatments (online Appendix Section 4):** `DID_M` extends as a weighted average of DID terms comparing groups moving from `d` to `d'` between `t-1` and `t` against groups stable at `d`, summed across all `(d, d', t)`. Implemented by the Stata `did_multiplegt` package.

The Section IV extensions are not the primary scope of the library's `ChaisemartinDHaultfoeuille` estimator; fuzzy DID is a separate dCDH 2018 paper, and the covariate-adjusted path is partial.

**Reference implementation(s):**
- R: `DIDmultiplegt::did_multiplegt()`, `DIDmultiplegtDYN` (the dynamic-effects successor), `twowayfeweights` package.
- Stata: `did_multiplegt` (`DID_M`, `DID_+`, `DID_-`, `DID_M^pl`, longer-lag placebos), `twowayfeweights` (weight diagnostics + `sigma-underbar` / `s-underbar` summary statistics), `fuzzydid` (fuzzy / nonbinary extensions). All available from SSC.
- Python: `diff_diff.ChaisemartinDHaultfoeuille` (this library).

Footnote 15 cross-references Callaway and Sant'Anna (2018), later published 2021 *JoE*, for an alternative placebo test in staggered adoption designs.

**Requirements checklist:**
- [x] Cell-level aggregation step: collapse panel to `(g, t)` cells via `groupby([group, time]).agg(y_gt=mean)`; reject within-cell-varying treatment with a clear error.
- [x] Per-period `DID_{+,t}` / `DID_{-,t}` construction with the boundary conventions on missing joiners / leavers / stable groups.
- [x] `DID_M` aggregation with switching-cell weights `N_{1,0,t}/N_S` and `N_{0,1,t}/N_S`.
- [x] `DID_M^pl` single-lag placebo. **Deferred / out of scope on Phase 1:** longer backward placebo lags require the dynamic companion's `DID^{pl}_l` machinery (Phase 2, `L_max >= 1`); the Phase 1 (`L_max=None`) per-period aggregation path supports only the single-lag placebo.
- [x] TWFE weights diagnostic (Equation 1 / Theorem 1) — `twfe_diagnostic = True` in `fit()` exposes per-cell weights `w_{g,t}` plus the summary scalars surfaced on the top-level `ChaisemartinDHaultfoeuilleResults` dataclass as `twfe_fraction_negative`, `twfe_sigma_fe`, and `twfe_beta_fe` (`chaisemartin_dhaultfoeuille_results.py:300-321`). The same scalars are also returned on the standalone `TWFEWeightsResult` helper (under the bare names `fraction_negative`, `sigma_fe`, `beta_fe`) when callers run the diagnostic in isolation via the module-level helper. The `twfe_sigma_fe` field IS the Corollary 1 sign-flip threshold `sigma-underbar_fe`. **Deferred:** the related Corollary 1 ratio statistic `s-underbar` from `twowayfeweights` is not exposed as a result field. Contributors needing it must compute by hand from `result.twfe_weights`.
- [x] Joiners-only / leavers-only views (`DID_+` and `DID_-`) returned alongside the aggregate `DID_M`.
- [x] Inference for `DID_M`, `DID_+`, and `DID_-`: **library default (`n_bootstrap=0`)** is analytical SE from the dynamic companion's cohort-recentered plug-in variance evaluated at horizon `l = 1` (REGISTRY `## ChaisemartinDHaultfoeuille` Standard-errors block L572-L598; `DID_M = DID_{l=1}` under the companion's notation). **Library override (`n_bootstrap > 0`)** swaps the analytical SE/CI/p-value surface for percentile-based multiplier-bootstrap inference (REGISTRY L598 + L618). Both paths are library extensions over the AER 2020 paper's recommendation — the paper itself uses cluster-robust SEs for `DID_M` tables (Applications 1-2, county- and worker-level clustering) and reserves clustered bootstrap for cross-estimator difference tests only (footnotes 23-24). **Deferred:** the single-period placebo `DID_M^pl` has no analytical-IF derivation in the AER 2020 paper, so on the Phase 1 (`L_max=None`) path the library populates the top-level `ChaisemartinDHaultfoeuilleResults` placebo-inference fields with explicit NaN placeholders: `placebo_se = NaN`, `placebo_t_stat = NaN`, `placebo_p_value = NaN`, `placebo_conf_int = (NaN, NaN)` (runtime assignment at `chaisemartin_dhaultfoeuille.py:2772-2775`; field docstrings at `chaisemartin_dhaultfoeuille_results.py:204-219`). A `UserWarning` is also emitted. Placebo inference is materialized only on the Phase 2 multi-horizon path via the dynamic-companion `DID_l^pl` machinery. The standalone `DCDHBootstrapResults` helper carries a separate per-bootstrap placebo field set under the bare names `placebo_se` / `placebo_ci` / `placebo_p_value` that is only populated when the bootstrap path is invoked. **Deferred:** cross-estimator t-tests (e.g., `DID_+` vs `DID_-` joint test, or TWFE-vs-DID_M Hausman-style comparison) are not implemented as first-class library outputs, even though the paper proposes the clustered-bootstrap recipe for the `beta_fe - DID_M` / `beta_fd - DID_M` contrasts.
- [x] Warn on A11 violations and document the per-period zero-retention behavior.
- [x] Document A7 / A8 non-testability when invoked as a defense of TWFE under bad weight diagnostics.

---

## Implementation Notes

### Data Structure Requirements
- Panel-shaped or repeated-cross-section input with columns `(group, time, outcome, treatment)`.
- Treatment is assumed binary in the main paper. Library's `ChaisemartinDHaultfoeuille` accepts both binary and non-binary treatment with the multi-horizon `DID_{g,l}` path (REGISTRY § ChaisemartinDHaultfoeuille).
- Balanced panel ideal (Assumption 1). Implementations should reject NaN in outcome / treatment columns. Library extension retains terminal-missing groups via the per-period `present` mask.
- Within-cell treatment must be constant (Assumption 2 / sharp design). Cell-min != cell-max should raise an error.
- The paper's general `G x T` setup nests panel, repeated cross-section, cross-section with cohort-as-time, and the `N_{g,t} = 1` per-cell special case; implementations targeting only one of these should document the restriction explicitly.

### Computational Considerations
- Aggregation step is O(N) where `N` is the total observation count; per-period DiDs are then O(`G * T`).
- Memory is dominated by the per-period role-weight tensor and the cohort-recentered influence-function matrix (dynamic companion paper, Section 3.7.3); for `G` in the tens of thousands the IF matrix sits in RAM comfortably.
- Bootstrap inference: clustered at the group level, multipliers applied per-group. Recommend `B in {199, 499, 999}` (paper does not specify in the main text).

### Tuning Parameters

| Parameter (paper concept) | Library API | Type | Default | Selection method |
|---|---|---|---|---|
| Treatment direction (joiners / leavers / both) | Not user-settable; inferred from the fitted sample. Joiners-only and leavers-only views are exposed as `results.joiners_*` / `results.leavers_*` alongside the aggregate `DID_M`. | n/a | n/a | Staggered-adoption designs auto-restrict to joiners. |
| Max horizon (dynamic companion: `L`) | `L_max` on `fit()` | `Optional[int]` | `None` | `None` selects Phase 1 per-period `DID_M`; `>= 1` activates the Phase 2 dynamic `DID_l` event-study path (NBER WP 29873, revised July 2023). |
| Bootstrap iteration count | `n_bootstrap` on `__init__` | `int` | `0` (analytical) | `>= 199` recommended for percentile bootstrap CIs; analytical SE remains the library default. |
| Cluster level | `cluster` on `__init__` | `Optional[Any]` | `None` (only supported value) | Clustering is fixed at the group level (or PSU level under `survey_design`). User-specified cluster columns raise `NotImplementedError`; the paper's `DID_M` table SEs in Applications 1-2 are county- and worker-level cluster-robust respectively, which the library mirrors by clustering at the group level. |
| Placebo gate | `placebo` on `__init__` | `bool` | `True` | Boolean gate, not an integer lag count. Library computes the single-lag `DID_M^pl` on Phase 1; longer backward placebos require the Phase 2 `DID^{pl}_l` path (`L_max >= 1`). Set `placebo=False` to skip computation for speed. |
| Bootstrap weight distribution | `bootstrap_weights` on `__init__` | str | `"rademacher"` | `"rademacher"`, `"mammen"`, or `"webb"`. Ignored when `n_bootstrap = 0`. |

### Relation to Existing diff-diff Estimators
- Already implemented as `diff_diff.ChaisemartinDHaultfoeuille` (8,783 LoC).
- Library covers binary `DID_M` / `DID_+` / `DID_-` from the 2020 AER paper. The single-lag placebo `DID_M^pl` is shipped on the Phase 1 (`L_max=None`) path with point estimate only (placebo inference fields are NaN — see the Requirements checklist above). Multi-lag placebos require the Phase 2 dynamic-companion `DID^{pl}_l` machinery (`L_max >= 1`), covered in the separate 2022 paper review.
- The Phase 2 event-study path (`DID_l`, `DID_{g,l}`, `DID^{pl}_l`, normalized `DID^n_l`, cost-benefit `delta`, sup-t simultaneous bands) implements the 2022, revised July 2023 NBER WP 29873 dynamic companion - see the separate review for that paper.
- Library does NOT cover all Section IV extensions in this repository: fuzzy DID is a separate dCDH 2018 paper (not in scope here), and the covariate-adjusted path is partial.
- 6 documented deviations from R `DIDmultiplegtDYN` (see REGISTRY § ChaisemartinDHaultfoeuille for the full text and citations):
  - **Equal-cell weighting (Python) vs cell-size weighting (R, absent explicit weights):** the paper's main-text equations (transcribed above at the `DID_{+,t}` / `DID_{-,t}` definitions) use observation counts `N_{g,t}` as weights and aggregate at the individual-row level — i.e., cell-size weighting. The library's documented choice is to aggregate the panel to cell means up-front and then weight each cell equally; this differs from R's individual-row weighting on unbalanced inputs but matches the paper's formulas exactly when each `(g,t)` cell has the same observation count (which the parity-test generators enforce). The library deviation is intentional and documented, not a re-reading of the paper's weighting semantics.
  - **Period-based vs cohort-based stable controls:** Python uses period-based `stable_0(t) = {g : D_{g,t-1} = D_{g,t} = 0}` and `stable_1(t)` per Theorem 3 of this paper. R `DIDmultiplegtDYN` additionally conditions on the baseline `D_{g,1}` (cohort-based) per the dynamic companion paper's `(D_{g,1}, F_g, S_g)` framework. The two definitions agree on pure-direction panels (joiners-only or leavers-only) and on the worked-example 4-group case; they disagree by O(1%) on the point estimate when joiners and leavers coexist and some joiner post-switch cells could serve as leaver controls. After the Round 2 full-IF fix, the SE parity gap on pure-direction scenarios narrowed from ~18% to ~3%. Mixed-direction SE parity is not asserted (R parity tests skip the SE check in mixed scenarios — see REGISTRY's tolerance table).
  - **`<50%` switcher warning at far horizons** (Phase 2 dynamic-companion extension; warns when fewer than 50% of the `l=1` switchers contribute at a far horizon `l`, per the Favara-Imbs application reported in the 2022/2023 NBER WP companion).
  - **Terminal-missingness retention:** groups missing one or more *later* periods are kept (the per-period `present` mask handles the missing transitions), instead of being dropped wholesale.
  - **SE normalization ~4% smaller than R:** documented difference in the variance denominator on cohort-recentered IF aggregation; both converge to the same asymptotic variance as `G -> infinity`, R is slightly more conservative in finite samples.
  - **Singleton-cohort degeneracy NaN handling:** singleton-baseline groups (those with a `D_{g,1}` value unique in the post-drop panel) are excluded from the variance computation only — per footnote 15 of the dynamic companion paper — but retained in the point-estimate sample as period-based stable controls. When every variance-eligible group forms its own cohort, Python returns `overall_se = NaN` with a `UserWarning`; R returns a non-zero SE via small-sample sandwich machinery the library does not implement.
- Round 2 full-IF fix: never-switching groups participate in variance via stable-control roles in the full influence function (see `chaisemartin_dhaultfoeuille_results.py:355-358`). The `n_groups_dropped_never_switching` field is retained for backward compatibility but no longer represents an actual exclusion.

---

## Gaps and Uncertainties

- **Web Appendix references:** the main paper relies on the online Appendix for several load-bearing results:
  - **Online Appendix Section 5:** asymptotic consistency of `DID_M` under Assumption 3, plus asymptotic normality as `G -> infinity`. Referenced at pages 2977 and 2978.
  - **Online Appendix Section 3.2:** nonbinary-treatment extension (average causal response parameter). Referenced at page 2983 footnote 18 and page 2984.
  - **Online Appendix Section 3.4:** FE or FD 2SLS regressions. Referenced at page 2982.
  - **Online Appendix Section 4:** `DID_M` extension to nonbinary discrete treatments. Referenced at page 2979.
  - **Online Appendix Section 6:** paper-by-paper review of the 33 AER 2010-2012 papers. Referenced at page 2983.
  - **Online Appendix Section 2:** fuzzy-design extensions. Referenced at page 2981.
- **Closed-form analytical variance for `DID_M`:** Not given in the AER 2020 main text. The dynamic companion (NBER WP 29873, revised July 2023) materializes the cohort-recentered plug-in variance (Web Appendix Section 3.7.3); the AER 2020 paper's empirical applications themselves report cluster-robust SEs (county-level in Application 1, worker-level in Application 2) for the `DID_M` table entries, with clustered bootstrap reserved for the cross-estimator difference tests `beta_hat_fe - DID_M` / `beta_hat_fd - DID_M` (footnotes 23-24). See the Standard Errors section above for the paper-vs-library SE breakdown.
- **Bootstrap iteration count:** Not specified in the main paper. Standard practice (199, 499, or 999) is reasonable default. The Stata `did_multiplegt` package help files presumably specify a default.
- **Bootstrap weight distribution:** Not specified in the main paper. The accompanying Stata implementation uses clustered bootstrap; whether it is Rademacher, Mammen, Webb, or pairs is not documented in the extracted main-text range.
- **Relation to Imai & Kim (2018) multiperiod DID:** Discussed on page 2979 - `DID_M` is "related to" the Imai-Kim multiperiod DID, but the Imai-Kim estimator is a weighted average of `DID_{+,t}` only (no leavers component) and does not extend to nonbinary treatments. The Imai-Kim authors do not establish properties of their estimator. The relation is detailed in online Appendix Section 4.
- **Relation to Wald-TC (dCDH 2018):** Page 2978 notes `DID_M` is related to the Wald-TC estimator in point 2 of Theorem S1 in the online Appendix of de Chaisemartin and D'Haultfœuille (2018), but the weighting of `DID_{+,t}` and `DID_{-,t}` differs and `DID_M` identifies `delta^S` under weaker assumptions.
- **Section V applicability statistics qualifications:** Table 2 Panel C reports A11 as definitively or presumably satisfied for ~80% of the 33 AER 2010-2012 papers; the "presumably yes / presumably no" categories are best-effort assessments by the authors without raw data access (page 2982-2983).
- **Section IV extensions are sketched, not derived in the main text:** The full extensions to fuzzy DID, covariates, and nonbinary treatment require reading the online Appendix sections referenced above.
- **No simulation studies** are reported in the main text - the paper relies on two real-data empirical applications.
- **`DID_l` dynamic event-study is NOT in this paper.** Footnote 13 (page 2977) is explicit: "It should be possible to weaken Assumptions 9-10, in particular to account for dynamic effects where `Delta_{g,t}` may depend on `(D_{g,1}, ..., D_{g,t-1})`. This introduces complications that are beyond the scope of this paper, but that we address in de Chaisemartin and D'Haultfœuille (2020a)." The 2020a companion is the 2022, revised July 2023 NBER WP 29873.

For complete cross-reference with library implementation deviations, consult `docs/methodology/REGISTRY.md` § ChaisemartinDHaultfoeuille and the `chaisemartin_dhaultfoeuille_results.py` docstrings.
