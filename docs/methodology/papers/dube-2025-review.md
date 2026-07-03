# Paper Review: A Local Projections Approach to Difference-in-Differences

**Authors:** Arindrajit Dube, Daniele Girardi, Òscar Jordà, Alan M. Taylor
**Citation:** Dube, A., Girardi, D., Jordà, Ò., & Taylor, A. M. (2025). A Local Projections Approach to Difference-in-Differences. *Journal of Applied Econometrics*, 40(7), 741-758. https://doi.org/10.1002/jae.70000 (Open Access, CC BY)
**PDF reviewed:** papers/J of Applied Econometrics - 2025 - Dube - A Local Projections Approach to Difference-in-Differences.pdf (main article, 18 pp) **+** papers/lpdid_online_appendix.pdf (official Wiley online appendix, 24 pp; Appendices A-C reviewed in full, D-F = simulation/empirical replication, skimmed). Both gitignored under top-level `papers/`; do NOT commit the PDFs.
**Review date:** 2026-06-28

---

## Methodology Registry Entry

*Formatted to match docs/methodology/REGISTRY.md structure. The corresponding registry entry is mirrored in `docs/methodology/REGISTRY.md` (`## LPDiD`, under "Modern Staggered Estimators").*

## LPDiD

**Primary source:** Dube, Girardi, Jordà & Taylor (2025), *Journal of Applied Econometrics* 40(7):741-758, https://doi.org/10.1002/jae.70000. NBER WP 31184; FRBSF WP 2023-12. Reference Stata package `lpdid` (SSC s459273); authors' example repo https://github.com/danielegirardi/lpdid (R + Stata).

**Key implementation requirements:**

*Assumption checks / warnings:*
- **Assumption 1 (No anticipation):** `E[y_it(p) - y_it(0)] = 0` for all `t < p`. Units do not respond before treatment.
- **Assumption 2 (Parallel trends):** `E[y_it(0) - y_{i1}(0) | p_i = p] = E[y_it(0) - y_{i1}(0)]` for all `t in {2..T}`, `p in {1..T, inf}` - untreated potential-outcome trends are common across cohorts, stated relative to the first period (`t = 1`). The base period for LP-DiD's long difference (first-lag `t-1` vs premean) is a separate efficiency/robustness choice, not part of this assumption (see PMD edge case below; first-lag relies on weaker PT than premean per Marcus & Sant'Anna 2021).
- Treatment is **binary**. The Section 3 main path assumes **absorbing** treatment (`D_{is} <= D_{it}` for `s < t`); non-absorbing handled by the Section 4.2 extension with a modified clean-control window.
- Identification is **model-based** (DGP `E[y_it(0)|i,t] = alpha_i + delta_t`, Equation 1). Treatment effects may be **dynamic and heterogeneous** across cohorts.

*Target parameter (Section 2.1):*

Cohort-specific dynamic ATT at horizon `h` for group `g`:

    tau_h^g = E[ y_{i,p_g+h}(p_g) - y_{i,p_g+h}(0) | p_i = p_g ]

`p_g` = period group `g` first enters treatment; `g = 0` is never-treated.

*Estimator equation (Equation 4 restricted by Equation 8, as implemented):*

The LP regression is run **separately for each horizon `h`** on a long-differenced outcome:

    y_{i,t+h} - y_{i,t-1} = beta_h^{LP-DiD} * Delta_D_it + delta_t^h + e_it^h

estimated on the **restricted (clean) sample** of observations that are either:

    newly treated:  Delta_D_it = 1            (Delta_D_it = D_it - D_{i,t-1})
    clean control:  D_{i,t+h} = 0             (absorbing case; Equation 8)

where:
- `y_{i,t+h} - y_{i,t-1}` = long difference (horizon-`h` outcome minus base period `t-1`)
- `Delta_D_it` = first difference of the treatment indicator (1 in the period a unit switches on)
- `delta_t^h` = time (calendar-period) fixed effects, horizon-specific
- **No unit fixed effects** (differenced out by construction)
- `h in {-Q, ..., 0, ..., H}`, `h != -1` (the base period). Negative `h` give pre-trend / placebo coefficients.

`beta_h^{LP-DiD}` consistently estimates a **variance-weighted ATT** (VWATT, Goodman-Bacon 2021 terminology):

    E(beta_h^{LP-DiD}) = sum_{g != 0} omega_{g,h}^{LP-DiD} * tau_h^g                              (Equation 9)
    omega_{g,h}^{LP-DiD} = [N_CCS_{g,h} * n_{g,h} * (1 - n_{g,h})] / sum_{g != 0}[ N_CCS_{g,h} * n_{g,h} * (1 - n_{g,h}) ]   (Equation 10)

`N_CCS_{g,h}` = # obs in the clean-control sample for `(g,h)`; `n_{g,h} = N_g / N_CCS_{g,h}` = treated share. **Weights are always non-negative** (this is the central result — no negative weighting). Derived via Frisch-Waugh-Lovell in online Appendix B.

*Equally weighted ATT — two numerically equivalent routes (Section 3.3):*

1. **Reweighted regression (`reweight=True`):** assign each obs in `CCS_{g,h}` weight `(omega_{g,h}^{LP-DiD} / N_g)^{-1}`. In practice obtained via an auxiliary regression of `Delta_D` on time indicators in the Equation-8 sample (FWL). Reweighted LP-DiD is **numerically equivalent to Callaway-Sant'Anna (2020)**.
2. **Regression adjustment (RA):** fit the long-difference on time indicators using **clean controls only**, predict the counterfactual for each treated obs, average the residuals:

       beta_h^{LP-DiD,RA} = N_TR^{-1} * sum_{(i,t) in TR} [ (y_{i,t+h} - y_{i,t-1}) - Ehat(y_{i,t+h} - y_{i,t-1} | D_{i,t+h}=0) ]

   `TR` = set of newly-treated obs (`Delta_D_it = 1`). This RA form is an **imputation estimator in the sense of Borusyak-Jaravel-Spiess (2024)**.

*With covariates / doubly robust (Section 4.1):*

Under conditional PT (Assumptions 3-5, linear CEF), the **recommended** covariate path is RA (Equation 11):

    beta_{h,x}^{LP-DiD,RA} = N_TR^{-1} * sum_{(i,t) in TR} [ (y_{i,t+h} - y_{i,t-1}) - gammahat^h * x_i - deltahat_t^h ]

where `gammahat^h, deltahat^h` come from the clean-control-only regression `y_{i,t+h} - y_{i,t-1} = delta_t^h + gamma^h x_i + u_it^h`. Generalizes to non-linear/semiparametric first-stage CEF (Equation 11 still holds).

**Direct covariate inclusion vs RA - positivity of weights (settled in online Appendix B.2):**
- **Linear + homogeneous covariate effects (App. B.2.1):** adding `x_i` directly to the Equation-8 OLS leaves the cohort weights **unchanged** (Equation B.6 still holds) -> weights stay non-negative. Safe.
- **General (heterogeneous/non-linear) covariate effects (App. B.2.2):** weights become proportional to the residual of `Delta_D` on time indicators *and* the covariates (Equations B.10-B.11); the authors state it is **"not possible to ensure that they are always positive."** Negative weighting can return.
- **=> Recommendation (authors, Section 4.1.1 + App. B.2.2): use the RA covariate path**, which is unbiased for the equally-weighted ATT under the weaker Assumptions 3-5. The direct-inclusion path is only safe under the strong homogeneity assumption (main-text Assumption 6 / App. B.2.1). **Implementation: the simple-covariate path should at minimum carry a docstring/warning that it assumes homogeneous covariate effects; default practitioners toward RA.**

*Premean-differenced base period (PMD, Section 3.4):*

Replace the single base period `t-1` with the mean of the last `k` pretreatment periods:

    y_{i,t+h} - (1/k) * sum_{tau=t-k}^{t-1} y_{i,tau} = beta_h^{PMD LP-DiD} * Delta_D_it + delta_t^h + e_it^h

`k = t-1` uses all pretreatment obs. Same weights as Section 3.2. PMD LP-DiD with `k=t-1` and a single treated group is **numerically equal to BJS** (footnotes 10-11); with multiple groups, very close but not identical.

*Pooled estimand (Section 3.5):*

Single overall ATT over the posttreatment window `h in {0..H}` by using the posttreatment mean long-difference as the dependent variable:

    (1/(H+1)) * sum_{h=0}^{H} y_{i,t+h} - y_{i,t-1}

(can be combined with PMD on the base period). Pre-window pooling is analogous over negative horizons.

*Standard errors (Section 1 - NOT specified by the paper):*
- **The paper deliberately gives no SE formula.** Section 1: "all the estimators ... allow for standard statistical inference using well-understood techniques ... For this reason, we do not discuss statistical inference here."
- Default in the reference Stata implementation: **cluster-robust at the unit level** (`vce(cluster unit)`, footnote 9).
- Pooled / joint tests across horizons: stack the per-horizon regressions (Stata `suest`).
- Bootstrap: not discussed in the paper.
- => Any analytical SE we ship (and especially the influence-function variance for the RA path) is an **implementation choice to be validated against the reference package**, not against the paper. Document under "Deviations from the paper".

*Edge cases:*
- **Composition effects (Section 3.6):** the treated/clean-control set can change across horizons `h`. To hold the **control** set fixed: tighten the clean-control condition to `D_{i,t+H} = 0` at all horizons (`H` = max horizon) -> the `no_composition` / `nocomp` option. To hold the **treated** set fixed: exclude cohorts with `p_g > T - H`. Costs statistical power.
- **Bias-variance trade-off (Sections 3.3, 5.3):** variance weighting (default) -> lower variance, some bias; equal weighting (`reweight`) -> unbiased, higher variance. Variance weighting won lower RMSE at short horizons (h=0,1), equal weighting at long horizons in the simulation.
- **PMD vs first-lag (Section 3.4):** PMD gains efficiency under low autocorrelation but can **amplify bias** if PT holds only in some pretreatment periods; first-lag differencing relies on a **weaker** PT assumption (Marcus & Sant'Anna 2021).
- **Few/no never-treated units (non-absorbing, Section 4.2.3):** use the effect-stabilization assumption (Assumption 9, parameter `L`) so recently-but-not-currently-changing units can serve as clean controls.

*Algorithm (per horizon `h`):*
1. Build `Delta_D_it = D_it - D_{i,t-1}`; identify newly-treated obs (`Delta_D_it = 1`).
2. Restrict to the clean sample: newly treated OR clean control (`D_{i,t+h} = 0` absorbing; Equation 13 window for non-absorbing).
3. Form the long difference `y_{i,t+h} - y_{i,t-1}` (or PMD base; or pooled posttreatment mean).
4. Run OLS of the long difference on `Delta_D_it` + time FE (+ direct covariates, or RA first-stage on controls; + reweighting weights for equal-weight estimand).
5. `beta_h` is the event-study coefficient at horizon `h`. Cluster-robust SE at unit level.
6. Repeat across `h in {-Q..H}, h != -1`; `h = -1` is the (zero) reference.

**Reference implementation(s):**
- Stata: `lpdid` (SSC `s459273`) - the authors' reference implementation.
- R: `alexCardazzi/lpdid` (third-party, A. Cardazzi & Z. Porreca; covers absorbing AND non-absorbing); authors' own R example scripts in `danielegirardi/lpdid`.
- Stata RA syntax (footnote 9): `teffects ra (Dhy i.time) (dtreat) if D.treat==1 | Fh.treat==0, atet vce(cluster unit)`.

**Requirements checklist:**
- [ ] Long-difference dependent variable with selectable base period (`t-1` default, PMD `k`)
- [ ] Clean-control sample restriction (absorbing: `D_{i,t+h}=0`)
- [ ] Per-horizon OLS with time FE, no unit FE; `h=-1` reference fixed at 0
- [ ] Variance-weighted (default) and reweighted (equal-weight) estimands
- [ ] Regression-adjustment path for covariates (recommended) + direct-inclusion path
- [ ] Pooled pre/post estimands
- [ ] `no_composition` option (fixed control/treated set across horizons)
- [ ] Cluster-robust SE at unit level by default (implementation choice; validate vs package)
- [ ] Non-absorbing extension (Section 4.2) - phased follow-up

---

## Implementation Notes

### Data Structure Requirements
- Balanced or unbalanced **panel**: unit `i`, time `t`, outcome `y_it`, binary treatment `D_it`.
- Absorbing main path needs a well-defined entry period `p_i` per unit (`inf` for never-treated).
- Long differences require observing `y` at both `t-1` (base) and `t+h` (target) for each used obs -> missing target/base rows drop out at each horizon.

### Computational Considerations
- **Speed is the headline advantage.** Table 2: at `N=184, T=54, 26 events`, LP-DiD ~0.16s vs CS ~137.5s, SA ~105.5s, BJS ~0.54s. LP-DiD is a stack of small OLS fits -> trivially parallelizable across horizons.
- Memory is modest: each horizon is an independent regression on a sample subset; no large group-time matrix as in CS.

### Tuning Parameters

| Parameter | Type | Default | Selection Method |
|-----------|------|---------|-----------------|
| `pre_window` (Q) | int | application-specific | # pre-treatment horizons (placebos) to estimate |
| `post_window` (H) | int | application-specific | # post-treatment horizons |
| `reweight` | bool | False (variance-weighted) | True -> equally weighted ATT (= CS) |
| `pmd` (k) | None / int / "max" | None (`t-1` base) | average over `k` pretreatment periods; choose ex-ante |
| `no_composition` | bool | False | True -> fixed control/treated set across horizons |
| covariates / RA | columns | None | conditional-PT settings; RA recommended over direct inclusion |
| cluster | column | unit | cluster-robust SE level (implementation default) |

### Relation to Existing diff-diff Estimators
The paper *proves* numerical equivalences we can exploit for **internal cross-validation** (no external package needed):
- **Reweighted LP-DiD == CallawaySantAnna (2020)** [Section 3.7]. Strongest cross-check; we have `CallawaySantAnna`.
- **PMD LP-DiD (k=t-1, single cohort) == ImputationDiD/BJS** [Section 3.4, footnotes 10-11]; we have `ImputationDiD`.
- **Variance-weighted LP-DiD == Cengiz et al. (2019) stacked regression** [Section 3.7]; we have `StackedDiD`.
- **2x2, h=0 == first-difference / static TWFE == plain DiD** [Section 2.2]; we have `DifferenceInDifferences` / `TwoWayFixedEffects`.
- RA LP-DiD is an imputation estimator -> shares machinery conceptually with `ImputationDiD`.
- Backend reuse: `linalg.solve_ols` (with `weights`, `cluster_ids`) covers every estimation path; `survey.py` (`_resolve_survey_for_fit`, `_compute_stratified_psu_meat`) is the template for the Phase D survey extension.

---

## Appendix Derivations (Online Appendix A-C, reviewed)

### A - LP/DiD equivalence and the identification insight
- **A.1 (2x2):** the LP regression at `h=0` is *exactly* a first-difference regression, and `beta_0^LP = beta^STWFE = beta^2x2 = ATT`. (Confirms the 2x2 cross-validation test against `DifferenceInDifferences`.)
- **A.2 (single cohort, multiple periods):** `beta_h^LP` recovers the dynamic DiD estimand `tau_h` exactly. **Key identification fact:** the LP regression is equivalent to a *cross-sectional* regression run only on entry-period rows (`t = s`) - "observations with `t != s` do not contribute to the estimated coefficient `beta_h^LP`." Only the period in which a unit switches on identifies the horizon-`h` coefficient. (Mirrors the absorbing clean-control restriction; useful when reasoning about which rows drive each estimate.)
- **A.3 (staggered, homogeneous effects):** an LP with leads *and* lags (Appendix Eq A.2: `y_{i,t+h}-y_{i,t-1} = delta_t^h + beta_h^LP Delta_D_it + sum_{j != 0} theta_j^h Delta_D_{i,t-j} + e`) recovers `tau_h`. LP differs from dynamic TWFE only in how unit FE are removed (LP differences around treatment times rather than full-sample mean-differencing), which can reduce bias under partial PT violations.

### B - Weights (the FWL derivation, App. B)
- **DGP (Eq B.1):** `y_{i,t+h} - y_{i,t-1} = delta_t^h + tau_{i,t+h} D_{i,t+h} - tau_{i,t-1} D_{i,t-1} + e_it^h`.
- Each obs satisfying the clean-control condition belongs to exactly **one** clean-control sample `CCS_{g,h}` (taken at `t = p_g`) -> LP-DiD == Cengiz stacked.
- **FWL (Eq B.4):** `E(beta_h^{LP-DiD}) = [sum_g sum_{i in CCS} DeltaDtilde_{i,pj} * E(y_{i,pj+h}-y_{i,pj-1})] / [sum_g sum_{i in CCS} DeltaDtilde_{i,pj}^2]`, where `DeltaDtilde` is the residual of `Delta_D` on time indicators within the clean sample.
- **Residualized treatment dummy (Eq B.5):** for every unit in group `g`, `DeltaDtilde_{i,pg} = DeltaDtilde_{g,pg} = 1 - N_g / N_{CCS_{g,h}}` (a per-group constant). This is the practical hook for the **reweighting** implementation (equal-weight estimand = weight obs by the inverse of this).
- **Weight (Eq B.6) = main-text Eq 10**, re-expressed: `omega_{g,h} = N_{CCS_{g,h}} n_{g,h}(1-n_{g,h}) / sum_{g != 0}[...]`, `n_{g,h} = N_g/N_{CCS_{g,h}}`. Always non-negative.
- **B.2 covariate weights:** see the "Direct covariate inclusion vs RA" block above - B.2.1 (homogeneous) leaves weights unchanged; B.2.2 (general) gives `omega^c` proportional to the residual of `Delta_D` on time indicators **and** covariates (Eq B.10-B.11), not guaranteed positive.

### C - Non-absorbing weights (App. C, for Phase C)
- Introduces **exit events** (`Delta_D_{g,j} = -1`), exit periods `q_g^n`, and exit-event dynamic effects `eta_h^{g,n}`. Assumption 9 (effect stabilization) applies to exit events too.
- Eq C.1 decomposes `E[y_{t+h}-y_{t-1}]` over treatment AND exit events within the `[t-L, t+h]` window -> justifies the modified clean-control condition of main-text **Equation 13**: clean controls = units with no treatment change in `[t-L, t+h]`.
- **Non-absorbing weight:** `omega_{g,n,h}^{LP-DiD''} = nbar_{g,n,h} N_{CCS_{g,n,h}} [nhat_{g,n,h}(1-nhat_{g,n,h})] / sum[...]`, where `nhat` = share of newly-treated and `nbar` = share of group `g` among newly-treated, in `CCS_{p_g^n, h}`. Non-negative. (Phase C will need both the "first-time entry" estimand of Eq 12 and this effect-stabilization estimand of Eq 13 - they are distinct.)

### D - dynamic-selection simulation (App. D, reference for Phase C tests)
- DGP calibrated to ANRR democracy: 184 units x 51 periods, `rho=0.98`, treatment entry on `psi*Delta_y_{i,t-1} + (1-psi)u_i <= theta` (dynamic selection), reversals allowed after >=5 periods, effect stabilizes after 9 periods, estimators include `Delta_y_{i,t-1}` as a control. Conditional (not unconditional) PT. Flags **Nickell bias** from the lagged-outcome control (only material under high autocorrelation + short T). A candidate template for a Phase C non-absorbing test DGP.

---

## Gaps and Uncertainties

1. **No SE formula in the paper.** Inference is deferred to "standard techniques" (Section 1, p.742). The reference Stata uses `vce(cluster unit)`. Our analytical SEs - and especially the contributor's hand-rolled influence-function cluster variance for the RA path - are **unspecified by the paper** and must be validated against the `lpdid` package / R reference, then documented as an implementation choice under "Deviations". (Pages 742, footnote 9.)
2. ~~Weight derivations are in the unreviewed online appendix.~~ **RESOLVED** - the official `lpdid_online_appendix.pdf` (Appendices A-C) has now been reviewed and the FWL weight derivations (App. B), covariate-weight positivity conditions (App. B.2.1/B.2.2), and non-absorbing weights (App. C) are captured in the "Appendix Derivations" section above. Remaining appendix material not transcribed: D (dynamic-selection simulation, skimmed), E (LW banking-dereg replication), F (ANRR replication) - empirical/simulation reproduction, not needed for the estimator contract.
3. **Non-absorbing extension (Section 4.2)** is presented as illustrative, not exhaustive ("a comprehensive discussion ... would indeed require a whole article", p.751). The effect-stabilization window `L` (Assumption 9, Equation 13) and the "first-time entry" estimand (Equation 12) are two distinct estimands; a Phase C design will need to choose/expose both deliberately. The contributor's scaffold rejects non-absorbing entirely.
4. **PMD multi-cohort vs BJS** are "very similar (although not identical)" (p.747) - do not assert exact equality in tests except in the single-cohort `k=t-1` case.
5. **`pmd="max"` semantics:** the paper's `k=t-1` "use all pretreatment periods" is per-observation (expanding window). The contributor implemented `pmd="max"` as the expanding mean of all prior periods and integer `pmd=k` as the trailing-`k` mean - verify this matches the package's `pmd` option exactly (the paper fixes the formula but not the option's edge behavior at the panel start).
6. **Pooled SE / joint tests** use stacking (`suest`); the variance of the pooled estimand under clustering should be cross-checked against the package rather than assembled ad hoc.
