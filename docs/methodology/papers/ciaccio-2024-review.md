# Paper Review: Distributional Difference-in-Differences Models with Multiple Time Periods

**Author:** Andrea Ciaccio
**Citation:** Ciaccio, A. (2024). Distributional Difference-in-Differences Models with Multiple Time Periods. *arXiv preprint* arXiv:2408.01208.
**PDF reviewed:** **arXiv:2408.01208v2** (https://arxiv.org/abs/2408.01208v2, v2 submitted 21 May 2025; v1 was 2 Aug 2024; 44 PDF pages, printed page = PDF page; main text pp. 1-32, References pp. 33-36, Appendix A proofs pp. 37-40, Appendix B repeated cross sections pp. 41-42, Appendix C additional simulations pp. 43-44). Per the project's PDFs-never-committed convention the local PDF is kept outside the repository (gitignored `papers/ciaccio-2408.01208v2.pdf`). This is an unpublished preprint as of the review date (2026-07-12) - check for a published version (which may renumber results) before citing it in shipped docs. All numbers below are pinned to arXiv v2.
**Review date:** 2026-07-12

---

## Methodology Registry Entry

*Drafted in docs/methodology/REGISTRY.md format for FUTURE use. Do not copy this section into the registry now - see the status line below.*

## Staggered Distributional DiD (Ciaccio)

**Status: NOT shipping in diff-diff CiC/QDiD v1 (scope decision 2026-07-12).** The staggered/multi-period distributional extension is reviewed-but-deferred; this review un-parks the ROADMAP's "Distributional DiD for staggered timing (Ciaccio)" row from "commit when a user reports need" to "reviewed, implementation deferred pending demand". Any future implementation would be a separate scoped PR.

**Primary source:** Ciaccio, A. (2024). Distributional Difference-in-Differences Models with Multiple Time Periods. arXiv:2408.01208 (reviewed at v2, 21 May 2025).
- Preprint-status note (p. 1 footnote, quoted as printed): "This is a preliminary version of the paper where most of the comments and feedback received still need to be addressed. All the Monte Carlo simulations were run in STATA. The ado files of the command used for implementing the methodology presented in the paper, `qtt`, are available upon request at the time of writing."
- **This paper version has NO empirical application - simulations only.** Section 4 (Monte Carlo, pp. 23-30) is followed directly by Section 5 (Discussion, pp. 31-32) and Section 6 (Conclusion, p. 32); References begin on p. 33. The section numbering 2 -> 3 -> 4 -> 5 -> 6 leaves no room for an application anywhere in the paper.
- The paper is the staggered/multi-period generalization of Callaway and Li (2019) (2-period/2-group panel QTT), using the copula-invariance-across-groups device of Callaway, Li, and Oka (2018) and the group(cohort)-time parameter + aggregation architecture of Callaway and Sant'Anna (2021).

**Model (Section 2.1, pp. 6-8):**

- `T` available periods, `t = 1, ..., T`; classic 2x2 DiD is `T = 2` with nobody treated in `t = 1`; this paper allows `T > 2`. `D_{i,t}` is a binary indicator = 1 if unit `i` (of `i = 1, ..., N`) is treated in period `t`. `q >= 2` is the first period the policy is implemented; staggered rollout starts at `q`.
- **Cohorts** (p. 7): assuming a never-treated group exists, define `T - q + 2` mutually exclusive cohort dummies `d_r` (`r = q, ..., T`) denoting the period in which unit `i` FIRST receives treatment. `C` is a dummy = 1 for never-treated groups. `d-bar = max_i d_{i,r}` is the maximum period where a unit may become treated. Always-treated units are dropped (footnote 3, p. 7: "there is no pre-treatment period for these units").
- **Comparison group**: main results use the NEVER-TREATED group; results "can be generalized to cases where not-yet-treated units serve as the comparison group, as they follow from symmetric arguments (Callaway and Sant'Anna, 2021)" (p. 7). Trade-off (pp. 5, 7): not-yet-treated expands the pool of valid comparisons and can improve inference, but if later-treated units adjust behavior in response to the policy's introduction, never-treated is the more reliable counterfactual (footnote 5 cites Ciaccio (2023) for an empirical application).
- **Covariates** (p. 7): a complete set of PRE-TREATMENT covariates `X` with support `chi = supp(X) subset R^K`, `dim(X) = k`.
- **Generalized propensity score** (p. 7): `P_{r,t}(X) = P(d_r = 1 | X, d_r + C = 1)` - the probability of being first treated in period `r`, conditional on covariates and on being either in cohort `r` or never-treated (pairwise cohort-vs-never-treated conditioning, CS-2021 style, NOT multinomial across cohorts). Footnote 6 (p. 7) gives the not-yet-treated version `P_{r,s}(X) = P(d_r = 1 | X, d_r + (1 - D_s)(1 - d_r) = 1)`.
- **Potential outcomes** (p. 7): dynamic potential outcomes (Robins 1986, 1987) combined with dynamic treatment adoption (Heckman et al., 2016). `Y_{i,t}(r)` = outcome at `t` had the policy been introduced by period `r`; `Y_{i,t}(0)` = outcome had treatment never been received. Observed outcome (Eq. (1), p. 7):

```
Y_{i,t} = Y_{i,t}(0) + sum_{r=q}^{T} d_{i,r} * (Y_{i,t}(r) - Y_{i,t}(0))        (1)
```

- Observation rule (pp. 7-8): never-treated units reveal `Y(0)` for all `t`; units first treated in `r` reveal `Y(0)` for `t < r` and `Y(r)` for `t >= r`.
- Quantile notation (p. 8): `q_tau = F_T^{-1}(tau) := inf{ t : F_T(t) >= tau }` (inf-based, weak inequality). Long difference (p. 16): `Delta_{[r-rho-1, t]} Y(0) = Y_t(0) - Y_{r-rho-1}(0)`. Short difference (p. 12): `Delta Y_t = Y_t - Y_{t-1}`.

**Target parameters (Sections 2.2 and 2.4, pp. 10-11, 18-20):**

Cohort-time distributional/quantile treatment effect on the treated (name borrowed from Callaway and Sant'Anna 2021's "group-time" parameters, p. 11). Eq. (2), p. 10:

```
QTT_{r,t}(tau) = F^{-1}_{Y_t(r)|d_r=1}(tau) - F^{-1}_{Y_t(0)|d_r=1}(tau),
    tau in [0,1], r = q, ..., T, and t >= r        (2)
```

`F_{Y_t(r)|d_r=1}` is directly identified from treated outcomes of cohort `r` at `t >= r`; the whole identification problem is the counterfactual `F_{Y_t(0)|d_r=1}` (p. 11). Heterogeneity is allowed across `tau`, across cohorts `r`, and over time `t`.

Post-identification version with anticipation (Section 2.4.1, p. 18):

```
QTT_{r,t,rho}(tau) = F^{-1}_{Y_t(r)|d_r=1}(tau) - F^{-1}_{Y_t(inf)|d_r=1}(tau)
    for all r, t in {q, ..., T}, t >= r - rho
```

(As printed on p. 18 the subtrahend is typeset with what appears to be `Y_t(infinity)` - the never-treated potential outcome, used interchangeably with `Y_t(0)` elsewhere; the p. 19 restatement uses `Y_t(0)`. See Gaps and Uncertainties.) Conditional version (p. 18): `QTT_{r,t,rho}(tau; x) = F^{-1}_{Y_t(r)|X,d_r=1}(tau|x) - F^{-1}_{Y_t(0)|X,d_r=1}(tau|x)` for all `r, t in {q, ..., T}` and `t >= r - rho`. Rank invariance is required to interpret the QTT as the quantile treatment effect for specific units.

Stochastic-dominance parameters (Sections 2.2 and 2.4.2, pp. 8-11, 20): for each post-treatment `t` and cohort `r`, test whether `Y_t(r)|d_r=1` FSD or SSD `Y_t(0)|d_r=1`. These avoid the rank-invariance assumption implicit in quantile-by-quantile QTT comparisons (Maasoumi and Wang 2019 show data often reject rank invariance). With `U_1` the class of increasing von Neumann-Morgenstern utilities and `U_2` its concave subclass, FSD holds iff any of (pp. 9-10, restated p. 20): (1) `E u(Y_t(r)|d_r=1) >= E u(Y_t(0)|d_r=1)` for all `u in U_1`, strict for some; (2) `F_{Y_t(r)|d_r=1}(y) <= F_{Y_t(0)|d_r=1}(y)` for all `Y`, strict for some values; (3) quantile-by-quantile dominance at all points of the support. SSD analogously with `U_2` / integrated CDFs / integrated quantiles; FSD implies SSD (p. 10).

ATT as by-product: footnote 12 (p. 13) - once the counterfactual distribution is estimated, `E(Y(0)|d_r=1) = int_0^1 F^{-1}_{Y(0)|d_r=1}(tau) d tau`, so the ATT is straightforwardly retrieved. Footnote 14 (p. 16): under Assumptions 1-4 alone the ATT is identified in the staggered multi-period context.

**Assumptions (exact statements):**

*Assumption 1 (Irreversibility of Treatment), p. 7:*

```
D_j = 0 for all j = 1, ..., (q-1) almost surely (a.s.).
For t = q, ..., T,  D_{t-1} = 1 implies that D_t = 1 a.s.
```

No unit is treated before `q`; once treated, always treated ("staggered treatment adoption"). Interpreted, following Callaway and Sant'Anna (2021) and Sun and Abraham (2021), as units changing behavior "forever" once treated. Assumption 1 automatically defines each unit's cohort.

*Assumption 2 (Random Sampling), p. 8:*

```
{Y_{i,1}, Y_{i,2}, ..., Y_{i,t}, X_i, D_{i,1}, D_{i,2}, ..., D_{i,tau}}_{i=1}^{n}  is independent and identically distributed.
```

(Subscripts as printed; the last treatment subscript is typeset `tau` in the paper.) Requires panel data; Appendix B extends to repeated cross sections. Neither rules out time-series dependence nor restricts the relation between `D_{i,t}` and `(Y(0), Y(r))`. iid sampling rules out interference; together with treatment consistency this constitutes SUTVA (p. 8).

*Assumption 3 (Limited Treatment Anticipation), p. 11:*

```
There is a known rho >= 0 such that
P{Y_t(r) <= y | X, d_r = 1} = P{Y_t(0) <= y | X, d_r = 1}  a.s.
for all r = q, ..., T and t = 1, ..., T such that t < r - rho.
```

Extension from the mean to the ENTIRE DISTRIBUTION of the Limited Treatment Anticipation assumption of Callaway and Sant'Anna (2021). `rho = 0` reduces to No Anticipation. Implies the distributional treatment effect is 0 for all `t < r - rho`.

*Assumption 4 (Conditional Distributional Parallel Trends based on a "Never-treated" Group), p. 12:*

```
Let rho be defined as in Assumption 3. For each r, t in {q, ..., T} such that t >= q - rho,
P(Delta Y_t(0) <= Delta y | X, d_r = 1) = P(Delta Y_t(0) <= Delta y | X, C = 1)   a.s.
where Delta Y_t = Y_t - Y_{t-1}.
```

(The time restriction is printed as `t >= q - rho` - see Gaps and Uncertainties.) In the absence of treatment, the counterfactual distribution for cohort `r` would have evolved in parallel to the never-treated units. Extends common trends from the average to the entire distribution (cites Fan and Yu 2012; Callaway and Li 2019; Miller 2023). Weaker than selection-on-observables (Firpo 2007) - unobservable confounders may differ between treated and untreated cohorts - and weaker than Assumptions 1-2 of Bonhomme and Sauder (2011) (p. 12). Testable under strict stationarity of the changes in untreated potential outcomes if the pre-treatment period is long (Callaway et al. 2018) (p. 13). Footnote 10 (p. 12): if it fails, the bias varies across quantiles.

*Assumption 5 (Conditional Copula Invariance based on a "Never-treated" Group), p. 13:*

```
For all x in chi and for all (u, v) in [0,1]^2
C_{Delta Y_t(0), Y_{t-1}(0) | X, d_r = 1}(u, v) = C_{Delta Y_t(0), Y_{t-1}(0) | X, C = 1}(u, v)
```

Background (p. 13): Assumption 4 alone is NOT sufficient to point-identify `F_{Y_t(0)|d_r=1}` (Fan and Yu 2012 - partial identification only). The missing object is the dependence (copula) between `Y_{t-1}(0)` and `Delta Y_t(0)`; via Sklar's theorem, `F_{T,W} = C_{T,W}(F_T(t), F_W(w))`. Assumption 5 replaces the UNKNOWN copula for the treated cohort with the OBSERVED dependence of the never-treated group. Key contrasts (pp. 13-14):
- **vs Callaway and Li (2019) copula STABILITY**: CL19 instead impose that the copula between `Y_{t-1}(0)` and `Delta Y_t(0)` is stable ACROSS TIME, which requires panel data with at least two pre-treatment periods. Copula INVARIANCE (across GROUPS, following Callaway et al. 2018) needs neither, and permits repeated cross sections (Appendix B).
- Assumption 5 imposes NO restriction on marginal distributions - initial distributions can differ between treated and never-treated groups; only the dependence is restricted.
- Assumption 4 does not imply Assumption 5, nor the reverse (p. 14; noted by Callaway et al. 2018 and Callaway and Li 2019). Joint-normal illustration (p. 14): if `corr(Y_{t-1}(0), Delta Y_t(0)) = rho_t`, Assumption 5 requires `rho_t` be independent of treatment assignment, conditional on X and on being in cohort r or never-treated; means/variances are unrestricted.
- No formal test exists; Callaway and Li (2019) suggest rank-correlation measures (Kendall's tau) in pre-treatment periods to assess plausibility (p. 14).

*Assumption 6 (Continuity), p. 15:*

```
The random variables Y_{t-1}(0) and Delta Y_t(0) have continuous distribution conditional on either
being part of the treated cohort r (i.e., d_r = 1) or being never-treated (i.e., C = 1), and
Y_t(r)|d_r = 1 is also continuously distributed on its support. Moreover, each of these distributions
has a compact support with marginal distributions, which are (uniformly) bounded away from 0 and 1
over their respective support.
```

Needed because the copula representation is not unique unless the random variables are continuous (Joe 1997; Nelsen 2006); follows Callaway et al. (2018) and Callaway and Li (2019).

*Assumption 7 (Overlap), p. 15:*

```
For each r, t in {q, ..., T}, there exists some epsilon > 0 s.t.
P(d_r = 1) > epsilon  and  P_{r,t}(X) < 1 - epsilon  a.s.
```

Required ONLY for the covariate-conditional results (Propositions 1-2). Generalizes overlap in Firpo (2007), Callaway et al. (2018), Callaway and Li (2019) to multiple periods/groups.

**Identification (Theorem 1, p. 15):**

```
Theorem 1. Suppose Assumptions 1, 2, 6, and unconditional version of Assumptions 3-5 hold.
Then F_{Y_t(0)|d_r=1}(tau) is identified:

F_{Y_t(0)|d_r=1} = P(Y_t(0) <= y | d_r = 1)
  = E[ 1( Delta_{[r-rho-1, t]} Y(0) <= y
          - F^{-1}_{Y_{r-rho-1}(0)|d_r=1} ( F_{Y_{r-rho-1}(0)|C=1} ( Y_{r-rho-1}(0) ) ) ) | C = 1 ]

where Delta_{[r-rho-1, t]} Y(0) = Y_t(0) - Y_{r-rho-1}(0) is the long difference.
```

Reading: the expectation runs over never-treated units (`C = 1`). For each never-treated unit, take its long difference `Y_t(0) - Y_{r-rho-1}(0)` and its base-period rank `F_{Y_{r-rho-1}(0)|C=1}(Y_{r-rho-1}(0))`; map that rank into the TREATED cohort's observed base-period marginal via `F^{-1}_{Y_{r-rho-1}(0)|d_r=1}(.)`. This quantile-quantile transform is where the copula-invariance dependence structure of the never-treated is transplanted onto the treated cohort's marginals. The base period is `r - rho - 1`, the last period unaffected by anticipation.

Mechanism (p. 16): `P(Y_t(0) <= y | d_r = 1) = E[ 1( Delta_{[r-rho-1,t]} Y(0) + Y_{r-rho-1}(0) <= y ) | d_r = 1 ]` is an integral over the JOINT distribution of the pre-treatment level and the change in the untreated potential outcome. Assumption 5 identifies this joint by replacing the unknown treated copula with the observed never-treated one; since `Delta_{[r-rho-1,t]} Y(0)` is not observed for treated cohort `r`, Assumption 4 replaces `F^{-1}_{Delta_{[r-rho-1,t]} Y(0)|d_r=1}(.)` with `F^{-1}_{Delta_{[r-rho-1,t]} Y(0)|C=1}(.)`. The copula is used IMPLICITLY - it is never estimated; the formula involves only marginal CDFs/quantile functions and the observed joint behavior of never-treated units. Once `F_{Y_t(0)|d_r=1}` is identified, so is its inverse (p. 16). Full proof in Appendix A (below).

*Example 1 (p. 16)* - TWFE DGP satisfying the assumptions: `Y_it(0) = alpha_t + eta_i + u_it`, `rho = 0`, panel data. For the CS-2021 ATT it suffices that `E(Delta u_it | d_r = 1) = E(Delta u_it | C = 1)` (mean version of Assumption 4). For this paper's method, two sufficient conditions: (i) `Delta u_it` independent of `D`, (ii) `C_{Delta u_t, u_{t-1} | d_r = 1} = C_{Delta u_t, u_{t-1} | C = 1}`. These allow the time-varying shock's distribution to vary over time (serial correlation allowed) and `u_it` correlated with `eta_i`, in contrast to Bonhomme and Sauder (2011). Only the UNTREATED potential outcome's generation is restricted; nothing is said about the treated potential outcome (pp. 16-17).

**Covariate scenarios (p. 17):** it is "highly unlikely" Assumptions 3-5 hold unconditionally. Three identification scenarios (cf. p. 5): unconditional (Theorem 1); (i) Assumptions 3-4 conditional on covariates, Assumption 5 unconditional -> Proposition 1 (IPW); (ii) Assumptions 3-5 all conditional -> Proposition 2.

**Proposition 1 (conditional PT, unconditional copula invariance; IPW), p. 17:**

```
Proposition 1. Under Assumptions 1-4, 6, 7, and unconditional version of Assumption 5 hold,
F_{Y_t(0)|d_r=1}(tau) is identified:

F_{Y_t(0)|d_r=1} = E[ 1[ F^{p,-1}_{(Y_t(0) - Y_{r-rho-1}(0))|d_r=1} ( F_{(Y_t(0) - Y_{r-rho-1}(0))|C=1} ( Y_t(0) - Y_{r-rho-1}(0) ) )
                        <= y - F^{-1}_{Y_{r-rho-1}(0)|d_r=1} ( F_{Y_{r-rho-1}(0)|C=1} ( Y_{r-rho-1}(0) ) ) ] | C = 1 ]

where

F^{p}_{Delta_{[r-rho-1,t]} Y(0)|d_r=1}(delta)
    = E[ (C / p_r) * ( p_r(x) / (1 - p_r(x)) ) * 1{ Y_t - Y_{r-rho-1} <= delta } ]        (3)

which is identified.
```

Notation as printed: `p_r(x)` is the generalized propensity score (the `P_{r,t}(X)` object) and `p_r = p(d_r = 1)` its unconditional cohort probability normalizer; `C` is the never-treated dummy, so the expectation runs over never-treated units reweighted by the odds `p_r(x)/(1 - p_r(x))` - the implementable IPW weight on never-treated units is `C * p_r(X) / (p_r * (1 - p_r(X)))` (Abadie 2005 / CS-2021 style reweighting of the never-treated change distribution to match cohort r's covariate distribution; the Appendix A proof makes this explicit). The superscript `p` marks the propensity-reweighted distribution of the long difference; `F^{p,-1}` its inverse. The only part of Theorem 1 needing modification is the identification of `F_{Delta Y(0)|d_r=1}`; the rest stays valid because Assumption 5 holds unconditionally. Generalizes Firpo (2007)'s reweighting (as readapted by Callaway and Li 2019) to staggered adoption; "almost identical to the one proposed by Abadie (2005) and Firpo (2007) with some minor changes" (p. 17). Requires a parametric specification of `P_{r,t}(X)` (no guarantee it is correct). Footnote 15 (p. 17): a doubly-robust estimator of `F_{Y(0)|d_r=1}` (as in Miller 2023) would relax reliance on correct parametric specification but is beyond the paper's scope.

*Example 2 (p. 18)* - conditional-validity DGP: `Y_it(0) = alpha_t + eta_i + X'_it beta + u_it`, `X` possibly distributed differently between treated and never-treated units; `u_it = rho u_{i,t-1} + epsilon_it` with `epsilon ~ WNN process`, `|rho| < 1`. Sufficient conditions: (i) `Delta u_it` independent of `D | X`, (ii) `C_{Delta u_t, u_{t-1}|d_r=1} = C_{Delta u_t, u_{t-1}|C=1}`.

**Proposition 2 (both assumptions conditional on covariates), p. 18:**

```
Proposition 2. Suppose that the random variables Y_{t-1}(0) and Delta Y_t(0) are continuously
distributed conditionally on x, and that this is true for all x in chi and both group r or the
never-treated units. Then, under Assumptions 1-7, F_{Y_t(0)|d_r=1}(tau) is identified:

P(Y_t(0) <= y | X = x, d_r = 1)
  = E[ 1[ F^{-1}_{Delta_{[r-rho-1,t]} Y(0)|X,C=1} ( F_{Delta_{r-rho-1,t} Y(0)|X,C=1} ( Delta_{[r-rho-1,t]} Y(0) ) | X )
          <= y - F^{-1}_{Y_{r-rho-1}(0)|X,d_r=1} ( F_{Y_{r-rho-1}(0)|X,C=1} ( Y_{r-rho-1}(0) ) | X ) ] | X = x, C = 1 ]

and

P(Y_t(0) <= y | d_r = 1) = int_{chi} P(Y_t(0) <= y | X = x, d_r = 1) dF_{X|d_r=1}(x)
```

(Transcription note: in the first inner composed term both the outer inverse CDF and the inner CDF of the long difference are printed with conditioning `|X, C=1`; under conditional Assumption 4 the `d_r=1` and `C=1` conditional distributions of the long difference coincide, so the composition is written entirely with never-treated conditionals.) The only difference vs Theorem 1: compute CONDITIONAL distributions first, then integrate covariates out against `dF_{X|d_r=1}` to obtain the unconditional counterfactual (p. 18). Computational caveat (p. 5): estimating the QTT under Conditional Copula Invariance requires estimating FIVE conditional distributions (as noted in Callaway and Li 2019), often infeasible; the practical implementation (following Callaway et al. 2018) assumes DISCRETE covariates and computes the conditional counterfactual for each covariate value. Note also that Callaway et al. (2018)'s conditional copula invariance results pertain to the CONDITIONAL QTT; this paper additionally delivers the unconditional QTT by integrating out X (p. 5).

**Aggregation (Section 2.4.1, pp. 19-20; following Callaway and Sant'Anna 2021):**

Generic weighted form, Eq. (4), p. 19:

```
theta(tau) = sum_{t=q}^{T} sum_{r=q}^{T} w(r,t) * QTT_{r,t,rho}(tau),    r in {q, ..., T}        (4)
```

with researcher-chosen weighting functions `w(r,t)` answering, e.g.: how the QTT varies across groups; with length of exposure; cumulative distributional effect across all groups until `t`; overall impact.

*Event-study (exposure) aggregation* (p. 19), with event time `e = t - r`:

```
theta^{e}_{exp}(tau) = sum_{r=q}^{T} 1{r + e <= T} P(d_r = 1 | r + e <= T) QTT_{r, r+e, rho}(tau),
    r in {q, ..., T}
```

`theta^e_exp(tau)` = effect, on units at the tau-th quantile of the outcome distribution, of being exposed to treatment for `e` periods; computed from all units exactly `e` periods past initial treatment (dynamic event-study analog, cf. Sun and Abraham 2021; de Chaisemartin and d'Haultfoeuille 2022).

*Overall aggregation* (p. 20):

```
theta^{o}_{weight}(tau) = (1/kappa) sum_{t=q}^{T} sum_{r=q}^{T} 1{t >= r} P(d_r = 1 | r <= T) QTT_{r,t,rho}(tau),
    r in {q, ..., T}
where kappa = sum_{t=q}^{T} sum_{r=q}^{T} 1{t >= r} P(d_r = 1 | r <= T)
```

Weighted average putting more weight on QTTs from larger cohorts; drawback: systematically overweights groups participating longer in the treatment (Callaway and Sant'Anna, 2021) (p. 20). Weighting functions are estimated by sample analogs (analogy principle, Manski 1994) (p. 23). The paper notes richer aggregations along the distribution of Y are conceivable (e.g., cumulative effect on the bottom two deciles for a minimum-wage evaluation) but deliberately limits itself to generalizing the CS-2021 schemes (p. 19).

*Stochastic dominance rankings* (Section 2.4.2, p. 20): if rank invariance is implausible (or `QTT_{r,t,rho}(tau)` changes sign in `tau`), use stochastic-dominance tests or inequality measures instead of the QTT, comparing `F_{Y_t(r)|d_r=1}` vs `F_{Y_t(0)|d_r=1}` for treated group `r`.

**Estimation (Section 3, pp. 20-23):**

Plug-in via empirical distribution functions (EDFs), by the analogy principle (Manski 1994), as in Callaway and Li (2019).

*Case A - no covariates (Theorem 1) (p. 21):*

1. Estimate the treated-outcome quantile directly by EDF + left-continuous (inf-based) inversion:
   ```
   QTT-hat_{r,t,rho}(tau) = F-hat^{-1}_{Y_t(r)|d_r=1}(tau) - F-hat^{-1}_{Y_t(0)|d_r=1}(tau)
   F-hat^{-1}_{Y_t(r)|d_r=1}(tau) = inf{ y : F-hat_{Y_t(r)|d_r=1}(y) >= tau }
   ```
2. Estimate the counterfactual CDF by the sample analog of Theorem 1 (p. 21):
   ```
   F-hat_{Y_t(0)|d_r=1}(y) = (1/n_0) sum_{i in 0} 1[ F-hat^{-1}_{Delta_{[r-rho-1,t]}Y|C=1}( F-hat_{Delta_{[r-rho-1,t]}Y|C=1}( Delta_{[r-rho-1,t]}Y ) )
                                    <= y - F-hat^{-1}_{Y_{r-rho-1}|d_r=1}( F-hat_{Y_{r-rho-1}|C=1}( Y_{r-rho-1} ) ) ]
   ```
   where `n_0` = number of never-treated units in periods `r-rho-1` and `t`; the sum runs over never-treated units. (As printed, the first composed term is `F-hat^{-1}_{DeltaY|C=1}(F-hat_{DeltaY|C=1}(.))` - both conditionals `C=1`, i.e. the never-treated quantile substituted for the treated one per Assumption 4, mirroring the Theorem 1 statement; under continuity this composition is the identity, matching the (A.6) -> (A.7) simplification in the proof.)
3. Invert the counterfactual CDF: `F-hat^{-1}_{Y_t(0)|d_r=1}(tau) = inf{ y : F-hat_{Y_t(0)|d_r=1}(y) >= tau }`.
4. Difference of quantiles gives `QTT-hat_{r,t,rho}(tau)`; aggregate with sample-analog weights (p. 23).

Footnote 16 (p. 21): under Theorem 1's assumptions plus compact support of `Y(a)`, `a in {r, 0}`, Callaway and Li (2019) show the EDF-based estimator is consistent and, via a functional central limit theorem, converges uniformly to a Gaussian process. "These results can be readily extended to the current context, as the only difference lies in the construction of the benchmark, not in the estimator itself." No new asymptotic theorem is stated in this paper.

*Case B - covariates via IPW (Proposition 1) (p. 22):*

The only additional term to estimate (shown in Appendix A) is `F^{p,-1}_{Delta_{[r-rho-1,t]}Y(0)|d_r=1}(delta)`, computed by a generalization of Firpo (2007) and Callaway and Li (2019):

```
F^{p,-1}_{Delta_{[r-rho-1,t]}Y(0)|d_r=1}(delta) =
    [ (1/n_{r,t}) sum_{i in r} (C/p_r) ( p-hat_r(x_i) / (1 - p-hat_r(x_i)) ) 1{ DeltaY_{t,r-rho-1} <= delta } ]
  / [ (1/n_{r,t}) sum_{i in r} (C/p_r) ( p-hat_r(x_i) / (1 - p-hat_r(x_i)) ) ]
```

(Two as-printed notation issues here - see Gaps item 17. First, the LHS carries the `p,-1` superscript although the display is the reweighted CDF of Eq. (3), i.e. a Hajek-normalized estimator; inversion then yields the quantile. Second, both sums are written `sum_{i in r}` while numerator and denominator carry the never-treated dummy `C` - over literal cohort-r units every term would be zero. Read `i in r` as the (r,t) estimation subsample - cohort r plus never-treated - with `C_i` selecting the never-treated contributions. The implementable form is a Hajek-normalized CDF over never-treated units with weight `w_i = C_i * p-hat_r(x_i) / (p-hat_r * (1 - p-hat_r(x_i)))`: compute `F-hat^p(delta) = sum_i w_i * 1{DeltaY_i <= delta} / sum_i w_i`, then invert.) `n_{r,t}` = number of units used to compute `QTT-hat_{r,t,rho}(tau)`; `p-hat(X)` is an estimator of the generalized propensity score; the denominator normalizes weights to sum to 1 in finite samples, ensuring `F^p(.)` is a proper distribution function. The propensity score can be estimated parametrically or non-parametrically (both Firpo 2007 and Callaway-Li 2019 show this).

*Case C - fully conditional (Proposition 2) (p. 22):*

Computationally demanding: FIVE conditional distributions, possibly infeasible when `dim(X)` is large and `n` small. Remedies:
- quantile regression for the relevant conditional quantities (as Melly and Santangelo 2015 do for CiC with covariates), or
- for discrete covariates, the cell-by-cell method of Callaway, Li, and Oka (2018): estimate `F-hat_{Y_t(r)|X,d_r=1}` by EDF for each value of `x`, invert; similarly the counterfactual per `x` (p. 22):
  ```
  F-hat_{Y_t(0)|X=x,d_r=1}(y) = (1/n_{0,x}) sum_{i in 0} 1[ (Delta_{[r-rho-1,t]}Y)
        <= y - F-hat^{-1}_{Y_{r-rho-1}|X=x,C=1}( F-hat_{Y_{r-rho-1}|X=x,C=1}( Y_{r-rho-1} ) ) | X = x ]
  ```
  (as printed; each quantity estimated by EDFs), valid for each `tau in (0,1)` and each `x in chi`. Then compute the conditional QTT and, if the unconditional parameter is wanted, integrate x out against `F-hat_{X|d_r=1}`.

**Inference (pp. 21-23):**

The choice of procedure should be guided by the dependence structure of the application.

1. *No serial correlation / clustering unlikely: empirical bootstrap of Callaway and Li (2019)* (p. 21), yielding uniform confidence bands covering `QTT(tau)` with fixed probability for all `tau in [eps, 1-eps] subset (0,1)` for some small `eps > 0`:
   - Let `QTT-hat*(tau)` be a bootstrap estimate computed with the same plug-in procedure on a bootstrap sample; `B` total iterations, `b = 1, ..., B`.
   - For each `b` compute the sup-t statistic:
     ```
     I^b = sup_{tau in T} Sigma-hat(tau)^{-1/2} | sqrt(n) ( QTT-hat^b(tau) - QTT-hat(tau) ) |
     ```
     with the bootstrap robust variance scale
     ```
     Sigma-hat(tau)^{1/2} = (q_{0.75}(tau) - q_{0.25}(tau)) / (z_{0.75} - z_{0.25})
     ```
     - the bootstrap interquartile range of the estimator divided by the IQR of a standard normal (p. 21).
   - The (1-alpha) uniform band (p. 22):
     ```
     C-hat_{QTT(tau)} = QTT-hat*(tau) +/- c^B_{1-alpha} Sigma-hat(tau)^{1/2} / sqrt(n)
     ```
     where `c^B_{1-alpha}` is the (1-alpha) empirical quantile of `{I^b}_{b=1}^B`. **Centering discrepancy (flag for any implementation):** the paper prints the band centered at `QTT-hat*(tau)` (the bootstrap estimate); in Callaway-Li (2019) the band is centered at the original-sample estimate `QTT-hat(tau)`. See Gaps and Uncertainties.
   - Footnote 17 (p. 22): for extreme quantiles, use alternative inference procedures (Chernozhukov et al., 2016 - extremal quantile regression).
2. *Clustered dependence likely*: adapt the wild cluster bootstrap to the QTT setting "with minor modifications"; with few treated clusters use the subcluster wild bootstrap (MacKinnon and Webb, 2018). Reviews: Cameron and Miller (2015); MacKinnon, Nielsen, and Webb (2023). (p. 22; no algorithmic detail given.)
3. *Stochastic dominance tests* (p. 23): generalized Kolmogorov-Smirnov statistics of Linton, Maasoumi, and Whang (2005) for FSD and SSD:
   ```
   d_{r,t,rho} = sqrt( (n_{r,t} * n_{0,t}) / (n_{r,t} + n_{0,t}) ) min sup ( F-hat_{Y_t(r)|d_r=1}(y) - F-hat_{Y_t(0)|d_r=1}(y) )

   s_{r,t,rho} = sqrt( (n_{r,t} * n_{0,t}) / (n_{r,t} + n_{0,t}) ) min sup int_{-inf}^{y} ( F-hat_{Y_t(r)|d_r=1}(z) - F-hat_{Y_t(0)|d_r=1}(z) ) dz
   ```
   where `n_{r,t}`, `n_{0,t}` are the sample sizes used to estimate the two EDFs. Following Maasoumi and Wang (2019), a PAIR BOOTSTRAP test for iid samples gives the probability of either SD statistic falling in a given interval and a p-value; e.g., if `P(d <= 0)` is large (above .90) and `d` is non-positive, conclude `Y_t(r)|d_r=1` FSD `Y_t(0)|d_r=1` with high confidence. More sophisticated tests are context-dependent (survey: Maasoumi 2003); detailed testing strategies are declared beyond the paper's scope.

Iterations: no `B` is fixed in the algorithm text; the Monte Carlo table notes (Tables 1-4, pp. 26, 27, 29, 30, and Tables C.1-C.4) state "Each Monte Carlo simulation uses 2,000 bootstrap replications"; all simulation results are based on 2,000 Monte Carlo simulations (p. 25).

**Monte Carlo evidence (Section 4, pp. 23-30; Tables 1-4):**

Common setup: no anticipation (`rho = 0`), policy starts from period 2 (`q = 2`), panel data, cohorts `r in {2, ..., T}`, `T = R` (positive treatment probability every period from 2 on); performance measured as average (simulated) bias and root-MSE over 2,000 Monte Carlo simulations; quantiles `tau in {.25, .50, .75}`. The main text reports the counterfactual quantile `F^{-1}_{Y_2(0)|d_2=1,rho=0}(tau)` only; the focus is recovering `F_{Y(0)|d_r=1}`, not the QTT itself (footnote 18, p. 23: rank invariance may plausibly hold since nothing is assumed about `Y_t(r)|d_r=1`). "Unc" rows = estimator based on the unconditional distributional PT ignoring covariates (Theorem 1); "Cond" rows = covariate-based estimator.

*Section 4.1 - varying N and T (pp. 23-27):*

- **DGP 1** (Eq. (5), p. 24): `Y_it(0) = alpha_t + eta_i + u_it` with `alpha_t = t`, `eta|d_r=1 ~ N(r,1)` for `r in {2,...,T}` (selection on unobservables), `u_t ~ N(0,1)`; `P(d_r=1) = 1/T` (no covariates). Then `Y_it(0) ~ N(alpha_t + r, 2)`, true quantile `(alpha_t + r) + sqrt(2) Phi^{-1}(tau)`.
- **DGP 2** (Eqs. (6)-(7), pp. 24-25): TWFE with one covariate, mimicking Callaway-Sant'Anna (2021); selection on observables via the generalized propensity score:
  ```
  P(d_r = 1 | X) = exp(X'gamma_r) / (1 + sum_r exp(X'gamma_r))     (6)      gamma_r = 0.5 r / T
  Y_it(0) = alpha_t + eta_i + X_it + u_it                          (7)
  ```
  with `X ~ N(0,1)`, `alpha_t = t`, `eta|d_r=1 ~ N(r,1)`, `u_t ~ N(0,1)`; `Y_it(0) ~ N(alpha_t + r, 3)`, true quantile `(alpha_t + r) + sqrt(3) Phi^{-1}(tau)`. Appendix C variant replaces `alpha_t = t` with the quadratic trend `alpha_t = t + t^2` (Table C.4).
- **DGP 3** (Eq. (8), p. 25): panel quantile regression, re-adaptation of Machado and Santos Silva (2019):
  ```
  Y_it(0) = alpha_t + eta_i + X_it + (1 + X_it) u_it     (8)
  ```
  treatment probability still Eq. (6); `u_t ~ N(0,1)`, `alpha_t = t`, `eta|d_r=1 ~ Z_r`, `gamma_r = r/(4T)`, `X_it = (1/8) chi_it`, with `Z_r = r + T`, `T ~ chi-squared(1)` and `chi_it ~ chi-squared(1)` (note: this `T` is a chi-squared random variable - a notation clash with the number of periods). No analytical quantile formula; population quantiles approximated by simulating Eq. (8) with 1 million observations.

All three DGPs satisfy the identifying assumptions (footnote 19, p. 25: iid errors per period suffice in the linear models; in the nonlinear DGP the term `(1+X_it)u_it` means distributional PT holds conditionally, and copula invariance holds unconditionally as long as X and u are independent of treatment assignment; footnote 20: copula invariance stays unconditional because no RHS random variable is distributed conditionally on X - estimation gets more demanding when the copula holds only conditionally).

Scenarios: (i) `T = R = 4` and (ii) `T = R = 10`; `n in {100, 1000}`.

**Table 1 (p. 26), T = R = 4, selected values as extracted** (bias / RMSE for `F^{-1}_{Y_2(0)|d_2=1}`):

| DGP | Est | n | tau=.25 | tau=.50 | tau=.75 |
|-----|-----|---|-------|-------|-------|
| DGP 1 | Unc | 100 | 0.112 / 0.525 | 0.097 / 0.489 | 0.122 / 0.509 |
| DGP 2 | Unc | 100 | 0.331 / 0.695 | 0.341 / 0.684 | 0.391 / 0.714 |
| DGP 2 | Cond | 100 | 0.086 / 0.531 | 0.091 / 0.515 | 0.146 / 0.541 |
| DGP 3 | Unc | 100 | 0.178 / 0.550 | 0.244 / 0.583 | 0.361 / 0.721 |
| DGP 3 | Cond | 100 | 0.114 / 0.571 | 0.203 / 0.590 | 0.335 / 0.730 |
| DGP 1 | Unc | 1000 | 0.011 / 0.152 | 0.007 / 0.150 | 0.013 / 0.157 |
| DGP 2 | Unc | 1000 | 0.258 / 0.314 | 0.250 / 0.304 | 0.257 / 0.315 |
| DGP 2 | Cond | 1000 | 0.018 / 0.154 | 0.011 / 0.147 | 0.018 / 0.158 |
| DGP 3 | Unc | 1000 | 0.061 / 0.167 | 0.085 / 0.180 | 0.132 / 0.230 |
| DGP 3 | Cond | 1000 | -0.005 / 0.163 | 0.040 / 0.169 | 0.101 / 0.218 |

Findings (pp. 25-26): performance is poor with few units per cohort (n = 100), especially for the nonlinear DGP 3; improves substantially with n (largest bias-reduction gain for DGP 3, attributed to fewer observations per quantile under a nonlinear DGP); ignoring covariates ("Unc" for DGPs 2-3) substantially worsens bias/RMSE and leads to unreliable inference, as the theory predicts.

**Table 2 (p. 27), T = R = 10, selected values as extracted**: DGP 1 Unc n=100 tau=.25: bias 0.047 / RMSE 0.858; n=1000: 0.013 / 0.245. DGP 2 Cond n=1000 tau=.25: 0.008 / 0.247; DGP 3 Cond n=1000 tau=.25: -0.005 / 0.261. Findings (p. 27): average biases mostly smaller in magnitude than T=4 (except tau=.75, n=100), but RMSEs almost always LARGER (much smaller cohorts, imprecise estimates); with `T = R = 10` and small n, DGP 2/DGP 3 parameters could not be estimated in many simulations due to too few treated and/or never-treated units (results not shown). Overall performance does not necessarily improve relative to T = R = 4.

Cross-cutting results (pp. 27-28): all results align with theory and with simulations in Callaway et al. (2018), Callaway-Li (2019), Callaway-Sant'Anna (2021); RMSE shrinks by ~`1/sqrt(10)` when n goes 100 -> 1000 across DGPs and conditioning, supporting sqrt(n)-consistency of the estimator of `F^{-1}_{Y_t(0)|d_r=1,rho}(tau)`. Power is relatively larger at the .50 quantile (p. 28, in line with Callaway et al. 2018).

Robustness discussion (pp. 26-27): the Table C.4 quadratic-trend variant gives nearly identical bias/RMSE - the key requirement of Assumption 4 is that untreated-outcome TRENDS be independent of treatment assignment, not linear. The same conclusion would hold for heteroskedastic or within-cluster-correlated errors, as long as the heteroskedasticity/clustering structure is unaffected by treatment assignment (e.g., a skedastic function `exp{X beta}` with the same functional form for treated and never-treated); Copula Invariance requires the dependence between the pre-treatment error level `u_{ir'}` and its change `Delta_{r-1,t}u` to be unaffected by treatment assignment.

*Section 4.2 - violations of the main identifying assumptions (pp. 28-30):*

Fixed setup: n = 1000, T = R = 4, `X ~ N(0,1)`, cohort probabilities via Eq. (6) with `gamma_r = 0.5r/T`. Reported parameter: `F^{-1}_{Y_2(0)|d_2=1,rho=0}(tau)`.

- **DGP 4** (Eq. (9), p. 28) - violation of (Conditional) Distributional Parallel Trends (Assumption 4), copula invariance intact:
  ```
  Y_it(0) = alpha_{t,d} + eta_i + X_it + u_it     (9)
  ```
  `eta|d_r=1 ~ N(r,1)`, `r = {2,...,4}`, `u_t ~ N(0,1)`, and `alpha_{t,d} = t(1 + eps-bar d)` with d = 0 for never-treated, d = 1 for treated cohorts; `eps-bar` = degree of violation (`eps-bar = 0` -> Assumption 4 holds; larger `eps-bar d` -> larger deviation; no cohort-specific violations, to keep the analysis simple). Then `Y_it(0) ~ N(t(1+eps-bar d) + r, 3)`, true quantile `t(1+eps-bar d) + r + sqrt(3) Phi^{-1}(tau)`. Grid: `eps-bar in {0.00, 0.05, 0.10, 0.50}`.

  **Table 3 (p. 29)** - selected Cond rows at tau=.50: bias 0.012 (eps-bar=0), -0.037 (0.05), -0.086 (0.10), -0.488 (0.50); RMSE 0.15 -> 0.158 -> 0.166 -> 0.51. Unc tau=.50 bias: 0.253, 0.203, 0.155, -0.248. Findings (pp. 28-29): SMALL violations of Assumption 4 -> only minor increases in bias/RMSE; large violation (eps-bar=0.50) -> substantial bias. Same pattern at .25/.75. Ignoring covariates significantly worsens things EXCEPT when the Assumption-4 violation is large: with a big trend deviation, the single covariate (which by construction has a time-constant effect, unrelated to the trend in Y(0)) cannot capture it, and conditioning on X merely adds estimation noise within a quantile - so "Cond" is worse than "Unc" at eps-bar=0.50.

- **DGP 5** (pp. 29-30) - violation of unconditional Copula Invariance (Assumption 5), Assumption 4 intact; similar to DGP 2 of Callaway et al. (2018). Untreated potential outcome follows Eq. (7) with
  ```
  (eta_i, u_{i,t}, u_{i,r-1}) | d_r = 1 ~ N(mu_r, V_r),   mu_r = [r, 0, 0]^T

  V_r = [ [1, rho_{r,u_t}, rho_{r,u_{r-1}}],
          [rho_{r,u_t}, 1, rho_{u_t,u_{r-1}}],
          [rho_{r,u_{r-1}}, rho_{u_t,u_{r-1}}, 1] ]      r in {2,3,4}, t in {1,...,4}
  ```
  `V_r` symmetric; `(Delta_{r-1,t}y(0), Y_{r-1})` is bivariate normal with correlation parameter `rho_{u_t,u_{r-1}} + rho_{r,u_t} - rho_{r,u_{r-1}} - 2`; for a bivariate normal the copula is Gaussian with dependence = correlation coefficient (Callaway et al. 2018). Violation device (p. 30): set `rho_{u_t,u_{r-1}} = 0.5` for all treated and never-treated units and
  ```
  rho_{r,u_t} = rho-bar d   for all t >= r
              = 0           for all t < r
  ```
  with d = 0 for never-treated, d = 1 for treated (any cohort); `rho-bar` = degree of copula-invariance violation. Grid: `rho-bar in {0.00, 0.05, 0.10, 0.50}`.

  **Table 4 (p. 30)** - selected Cond rows at tau=.25: bias 0.023 (rho-bar=0), 0.042 (0.05), 0.061 (0.10), 0.203 (0.50). Findings (p. 30): small copula violations -> minimal bias/RMSE increases; substantial violation (rho-bar=0.50) -> pronounced increase (Cond .25 bias 0.061 at rho-bar=0.10 rises to 0.203 at rho-bar=0.50; similar at .75). The **.50 quantile is insensitive** to copula violations: the deviation only affects the variance of `Y_it(0)`, and `Phi^{-1}(0.50) = 0` so `F^{-1}_{Y_t(0)|d_r=1}(tau) = (alpha_t + r)` - aligning with Callaway et al. (2018). Ignoring covariates significantly increases bias/RMSE in nearly all cases; the single exception is tau=.75 at rho-bar=0.50, where the unconditional bias is lower - flagged by the author as "a puzzling result."

**Appendix A - proof structure (pp. 37-40):**

*Proof of Theorem 1, route 1 (copula-pdf route, pp. 37-38):* adapted from Callaway and Li (2019); uses two Sklar's-Theorem results, Lemma A.1 and Lemma A.2 in Appendix A of Callaway and Li (2019) (deliberately NOT reproduced in this paper: "To save space, I refer the reader to their paper," p. 37 - external cross-reference). Setup assumes no anticipation (`rho = 0`); abbreviations `f_{t|d_r=1}`, `f_{t|C=1}` for the joint pdfs of `(Y_t(0)-Y_{r-rho-1}(0), Y_{r-rho-1}(0))` and `c_{t|d_r=1}`, `c_{t|C=1}` for the corresponding copula pdfs; supports `DeltaY` for the long difference and `Y` for the pre-period level (support notation printed as `Y_{t-1}(0)` in one place and `Y_{r-1}(0)` elsewhere, as extracted). Chain: write `F_{Y_t(0)|d_r=1}` as a double integral over the joint pdf; **(A.1)** rewrites the joint via the copula pdf (CL19 Lemma A.1); **(A.2)** substitutes the never-treated copula `c_{t|C=1}` for the unobservable treated copula by Assumption 5; **(A.3)** rewrites the copula pdf as the never-treated joint distribution in density-ratio form (CL19 Lemma A.2). A change of variables `u = F^{-1}_{Delta|C=1}(F_{Delta|d_r=1}(delta))`, `v = F^{-1}_{Y_{r-1}(0)|C=1}(F_{Y_{r-1}(0)|d_r=1}(y'))` (with implied equalities numbered 1-4 in the paper for `delta`, `y'`, `d delta/d u`, `d y'/d v`) turns (A.3) into **(A.4)** (integral over the never-treated joint), **(A.5)** (definition of expectation), **(A.6)** (replace the unknown `F^{-1}_{Delta_{[r-1,t]}Y(0)|d_r=1}` with the never-treated change distribution by Assumption 4 holding unconditionally), and **(A.7)**:

```
F_{Y_t(0)|d_r=1} = E[ 1( Delta_{[r-1,t]}Y(0) <= y
    - F^{-1}_{Y_{r-1}(0)|d_r=1}( F_{Y_{r-1}(0)|C=1}( Y_{r-1}(0) ) ) ) | C=1 ]        (A.7)
```

which proves the result since each remaining distribution is identified by its sample counterpart. QED.

*Proof of Theorem 1, route 2 (Sklar route, pp. 38-39):* "an alternative and more direct approach" exploiting Assumption 6 (continuity) and Sklar's Theorem, as in Callaway et al. (2018). Write **(A.8)** `F_{Y_t(0)|d_r=1} = P(Delta_{[r-1,t]}Y_i(0) + Y_{i,r-1}(0) <= y | d_r=1)`; under continuity, rank representations **(A.9)** for cohort-r units (`Delta = F^{-1}_{Delta|d_r=1}(u_i^r)`, `Y_{r-1} = F^{-1}_{Y_{r-1}|d_r=1}(v_i^r)` with `u_i^r`, `v_i^r` the treated ranks) and **(A.10)** for never-treated units (ranks `u_i^0`, `v_i^0`). Substituting into (A.8) gives **(A.11)** `F_{Y_t(0)|d_r=1} = P( F^{-1}_{Delta_{[r-1,t]}Y(0)|d_r=1}(u_i^r) + F^{-1}_{Y_{r-1}(0)|d_r=1}(v_i^r) <= y | d_r=1 )`. The joint distribution of `(u_i^r, v_i^r)` is the unknown treated copula; by Sklar's Theorem and the unconditional version of Assumption 5, replace it with the never-treated copula (conditioning switches to `C=1`, ranks to `u_i^0, v_i^0`); `F^{-1}_{Y_{r-1}(0)|d_r=1}` is observable but `F^{-1}_{Delta_{[r-1,t]}Y(0)|d_r=1}` is not, so Assumption 4 replaces it with the never-treated change distribution, giving the final estimable form `F_{Y_t(0)|d_r=1} = P( Delta_{[r-1,t]}Y_i(0) + F^{-1}_{Y_{r-1}(0)|d_r=1}( F_{Y_{r-1}(0)|C=1}( Y_{i,r-1}(0) ) ) <= y | C=1 )`. QED.

*Implementation-relevant constructions revealed by the proofs:* the final identification formula requires estimating only (1) the never-treated pre-treatment CDF `F_{Y_{r-1}(0)|C=1}`, (2) the treated cohort-r pre-treatment quantile function `F^{-1}_{Y_{r-1}(0)|d_r=1}` (composed as a quantile-quantile transform of never-treated pre-period ranks), and (3) the observed changes `Y_{i,t} - Y_{i,r-1}` among never-treated units. The counterfactual CDF is the never-treated empirical mean of `1{ Delta_i + QQ(Y_{i,r-1}) <= y }` - one `F^{-1} o F` composition per unit; NO density or copula estimation at the estimation stage (densities/copula pdfs appear only in the proof).

*Proof of Proposition 1 (pp. 39-40):* all Theorem 1 steps remain valid; the only change is step **(A.6)**, which used unconditional Assumption 4 to identify `F_{Delta Y(0)|d_r=1}`. With covariates this object is instead identified by the reweighted distribution of Eq. (3). It suffices to prove `F_{Delta Y(0)|d_r=1} = F^p_{Delta Y(0)|d_r=1}`:

```
F_{Delta Y(0)|d_r=1} = P(Delta Y_t(0) <= delta | d_r=1)
  = P(Delta Y_t(0) <= delta, d_r=1) / p_r                                                 (A.12)
  = E( P(Delta Y_t(0) <= delta, d_r=1 | X) / p_r )
  = E( (P_r(X)/p_r) * P(Delta Y_t(0) <= delta | d_r=1, X) )
  = E( (P_r(X)/p_r) * P(Delta Y_t(0) <= delta | X, C=1) )                                 (A.13)
  = E( (P_r(X)/p_r) * E[ C * 1{Delta Y <= delta} | X, C=1 ] )                             (A.14)
  = E( (P_r(X) / (p_r (1 - p_r(X)))) * E[ C * 1{Delta Y_t <= delta} | X ] )
  = E( (C p_r(X) / (p_r (1 - p_r(X)))) * 1{Delta Y_t <= delta} )                          (A.11)
```

Step logic: (A.12) definition of conditional probability; **(A.13) holds by Assumption 4** (conditional distributional PT); (A.14) definition of probability, then multiplication by `C` (valid since the inner expectation conditions on `C=1`); conditioning on `C=1` lets the potential outcome be rewritten as the observed outcome; the last equality uses the Law of Iterated Expectations. QED. **Labeling quirk (p. 40): the last display of this chain is printed with tag "(A.11)", duplicating the (A.11) of the Alternative Proof on p. 39** - presumably a typo (should be (A.15) or a back-reference to Eq. (3)'s weight); transcribed as printed. Also note the capitalization inconsistency `P_r(X)` vs `p_r(X)` within the chain (as printed). Definitions (p. 40): `p_r = p(d_r = 1)` (fraction of treated units in a given cohort); `p_r(X) = P(d_r = 1 | X, d_r + C = 1)`, the generalized propensity score as defined in Assumption 7, which "models the conditional probability of belonging to cohort r and either being part of cohort r or the never-treated group."

*Proof of Proposition 2 (p. 40):* "follows directly from Theorem 1, where now all the steps hold after conditioning on covariates." QED (one line; the outcome-regression/conditional analogue).

**Appendix B - repeated cross sections (pp. 41-42):**

Setting: extends identification to repeated cross sections (RCS), following Callaway et al. (2018) and CS (2021). For each cross-sectional unit the researcher observes `(Y, d_r, ..., d_T, C, t, X)` where `r = q, ..., T` and `t = 1, ..., T` is the period in which unit i is observed; `S_t` is a dummy = 1 if the observation is observed in period t; a random sample is available in each period.

*Assumption B.1 (p. 41, exact statement):* "Conditional on period t, `(Y, d_r, ..., d_T, C, X)` are cross sectionally independent and identically distributed for all `t in {1, ..., T}`, where `(d_r, ..., d_T, C, X)` is invariant to t."

Stated to be identical to the corresponding assumption in CS (2021); the pooled cross section is composed of independent draws from the mixture distribution

```
F_M(Y, d_r, ..., d_T, c, t, x) = sum_{t=1}^{T} P(S_t = 1) * F_{Y, d_r, ..., d_T, C, t, X}(Y, d_r, ..., d_T, c, x | t)
```

Also similar to Abadie (2005) and Sant'Anna and Zhao (2020); **excludes the possibility of compositional changes over time**.

Key problem: even if Assumptions 1, B.1, 6, and unconditional 3-5 hold, the difference `Y_it - Y_{i,r-1}` is NOT observed for the same unit in a pooled cross section (setup: covariates play no role; `rho = 0` WLOG). Following Callaway et al. (2018), impose **rank invariance on potential outcomes over time** within the never-treated group - units preserve their relative position in the distribution of Y over time; the paper flags it as "a strong assumption and is often rejected in empirical data (as discussed in the main text), [but] it is necessary to recover the unknown distribution at time t."

*Corollary 1 (p. 41, exact statement):* "Suppose we have access to repeated cross sectional data, specifically `{(Y, d_ir, C_i)}_{i=1}^{n^s}` for period `s in {r-1, t}` where `r = q, ..., T`, `t = 1, ..., T`, and `n^s` denotes the sample size of the cross section. Suppose further that Assumptions 1, 6, B1, and the unconditional version of Assumptions 3-5 hold. If the copula of `(Y_{i,r-1}(0), Y_{i,t}(0) | C=1)` satisfies rank invariance, then for every `(u,v) in [0,1]^2`

```
C_{Y_{i,r-1}(0), Y_{i,t}(0) | C=1}(u, v) = min{u, v}
```

Thus, for `y in supp(Y_{i,t}(0)|d_r=1)`, we obtain

```
F_{Y_t(0)|d_r=1}(y) = P{ Delta~_{[r-1,t]}Y(0)
                         + F^{-1}_{Y_{r-1}(0)|d_r=1}( F_{Y_{r-1}(0)|C=1}( Y_{r-1}(0) ) ) <= y | C=1 }

where Delta~_{[r-1,t]}Y(0) := F^{-1}_{Y_t(0)|C=1}( F_{Y_{r-1}(0)|C=1}( Y_{r-1}(0) ) ) - Y_{r-1}
```

(`Delta~` denotes the tilde-Delta imputed change; the trailing term is printed as `Y_{r-1}`, without the `(0)`.) `min{u,v}` is the comonotone (Frechet upper bound) copula. Proof (p. 42): under the stated assumptions Theorem 1's result holds, but with RCS `Delta_{[r-1,t]}Y(0)` cannot be identified from observed never-treated outcomes; rank invariance yields `F_{Y_t(0)|C=1}(Y_t(0)) = F_{Y_{r-1}(0)|C=1}(Y_{r-1}(0))`, and since both CDFs are identifiable from observed outcomes, the never-treated change is imputable as `Delta~` above. QED.

*Caveat (p. 42, exact):* "As noted in Callaway et al. (2018), the rank invariance assumption on the copula of `(Y_{i,r-1}(0), Y_{i,t}(0)|C=1)` **neither implies nor is implied by Assumption 5**." (Rank invariance here is a within-never-treated-group over-time copula condition; Assumption 5 is copula invariance across treatment groups.)

*Covariates in RCS (p. 42):* Proposition 2's result generalizes by making all steps conditional on X; extending Proposition 1 requires a MODIFIED definition of the generalized propensity score, as outlined in Appendix B of Callaway and Sant'Anna (2021) (external reference; not derived in this paper).

**Appendix C - additional simulation results (pp. 43-44; Tables C.1-C.4):**

All tables: `T = R = 4`, `tau = .5` (except C.4, which reports tau = .25/.50/.75), 2,000 bootstrap replications per Monte Carlo simulation. Six group-time parameters per table: `F^{-1}_{Y_t(0)|d_g=1}` for (g,t) in {(2,2),(2,3),(2,4),(3,3),(3,4),(4,4)}.

*Table C.1 - DGP 1 (Eq. (5)), tau = .5 (p. 43):* no Unc/Cond split (DGP 1 is the no-covariate baseline; note the table-notes boilerplate still mentions rows labeled 'UNC' although none appear - minor inconsistency).

| Parameter | True | n=100 Bias | n=100 RMSE | n=1,000 Bias | n=1,000 RMSE |
|---|---|---|---|---|---|
| F^{-1}_{Y_2(0)\|d_2=1} | 4 | 0.097 | 0.489 | 0.007 | 0.15 |
| F^{-1}_{Y_3(0)\|d_2=1} | 5 | 0.089 | 0.491 | 0.009 | 0.148 |
| F^{-1}_{Y_4(0)\|d_2=1} | 6 | 0.097 | 0.492 | 0.008 | 0.148 |
| F^{-1}_{Y_3(0)\|d_3=1} | 6 | 0.087 | 0.451 | 0.013 | 0.133 |
| F^{-1}_{Y_4(0)\|d_3=1} | 7 | 0.095 | 0.451 | 0.012 | 0.133 |
| F^{-1}_{Y_4(0)\|d_4=1} | 8 | 0.101 | 0.416 | 0.015 | 0.126 |

Pattern: small-sample bias ~0.09-0.10 shrinking to ~0.01 at n=1,000; RMSE roughly one third (consistent estimator under correct specification).

*Table C.2 - DGP 2 (Eq. (7)), tau = .5 (p. 43):*

| Block | Parameter | True | n=100 Bias | n=100 RMSE | n=1,000 Bias | n=1,000 RMSE |
|---|---|---|---|---|---|---|
| Unc | Y_2\|d_2 | 4 | 0.341 | 0.684 | 0.25 | 0.304 |
| Unc | Y_3\|d_2 | 5 | 0.344 | 0.686 | 0.248 | 0.303 |
| Unc | Y_4\|d_2 | 6 | 0.341 | 0.689 | 0.248 | 0.302 |
| Unc | Y_3\|d_3 | 6 | 0.469 | 0.723 | 0.371 | 0.41 |
| Unc | Y_4\|d_3 | 7 | 0.47 | 0.728 | 0.372 | 0.41 |
| Unc | Y_4\|d_4 | 8 | 0.571 | 0.794 | 0.479 | 0.509 |
| Cond | Y_2\|d_2 | 4 | 0.091 | 0.515 | 0.011 | 0.147 |
| Cond | Y_3\|d_2 | 5 | 0.099 | 0.522 | 0.007 | 0.147 |
| Cond | Y_4\|d_2 | 6 | 0.091 | 0.519 | 0.005 | 0.145 |
| Cond | Y_3\|d_3 | 6 | 0.099 | 0.508 | 0.014 | 0.156 |
| Cond | Y_4\|d_3 | 7 | 0.098 | 0.515 | 0.012 | 0.151 |
| Cond | Y_4\|d_4 | 8 | 0.09 | 0.507 | 0.007 | 0.153 |

Pattern: the unconditional estimator's bias does NOT vanish with n (0.25-0.48 at n=1,000) and GROWS with later-treated cohorts (0.25 for d_2 -> 0.371 for d_3 -> 0.479 for d_4); Cond bias vanishes (<= 0.014 at n=1,000). A clear demonstration that when trends depend on covariates the unconditional estimator is inconsistent.

*Table C.3 - DGP 3 (Eq. (8)), tau = .5 (p. 44):*

| Block | Parameter | True | n=100 Bias | n=100 RMSE | n=1,000 Bias | n=1,000 RMSE |
|---|---|---|---|---|---|---|
| Unc | Y_2\|d_2 | 4.839 | 0.244 | 0.583 | 0.085 | 0.18 |
| Unc | Y_3\|d_2 | 5.841 | 0.239 | 0.571 | 0.083 | 0.179 |
| Unc | Y_4\|d_2 | 6.867 | 0.2 | 0.55 | 0.061 | 0.171 |
| Unc | Y_3\|d_3 | 6.837 | 0.239 | 0.561 | 0.092 | 0.178 |
| Unc | Y_4\|d_3 | 7.869 | 0.206 | 0.544 | 0.064 | 0.163 |
| Unc | Y_4\|d_4 | 8.867 | 0.206 | 0.527 | 0.065 | 0.162 |
| Cond | Y_2\|d_2 | 4.839 | 0.203 | 0.59 | 0.04 | 0.169 |
| Cond | Y_3\|d_2 | 5.841 | 0.195 | 0.576 | 0.036 | 0.168 |
| Cond | Y_4\|d_2 | 6.867 | 0.16 | 0.56 | 0.015 | 0.167 |
| Cond | Y_3\|d_3 | 6.837 | 0.191 | 0.573 | 0.041 | 0.168 |
| Cond | Y_4\|d_3 | 7.869 | 0.16 | 0.565 | 0.012 | 0.163 |
| Cond | Y_4\|d_4 | 8.867 | 0.157 | 0.545 | 0.017 | 0.158 |

Pattern: under the nonlinear DGP 3 (distributional PT holds conditionally per footnote 19), both estimators carry modest residual bias at n=1,000 (Unc 0.06-0.09, Cond 0.01-0.04); Cond still dominates but the gap is smaller than DGP 2. Non-integer true values here (4.839, 5.841, ...) vs integer true values in C.1/C.2, since DGP 3 has no analytical quantile formula.

*Table C.4 - DGP 2 with quadratic trend `alpha_t = t + t^2`, parameter `F^{-1}_{Y_2(0)|d_2=1, rho=0}(tau)` (p. 44):*

| n | Est | tau=.25 True | Bias | RMSE | tau=.50 True | Bias | RMSE | tau=.75 True | Bias | RMSE |
|---|---|---|---|---|---|---|---|---|---|---|
| 100 | Unc | 6.832 | 0.331 | 0.684 | 8 | 0.344 | 0.672 | 9.168 | 0.392 | 0.715 |
| 100 | Cond | 6.832 | 0.082 | 0.526 | 8 | 0.09 | 0.508 | 9.168 | 0.152 | 0.553 |
| 1000 | Unc | 6.832 | 0.256 | 0.315 | 8 | 0.252 | 0.305 | 9.168 | 0.25 | 0.31 |
| 1000 | Cond | 6.832 | 0.017 | 0.158 | 8 | 0.013 | 0.149 | 9.168 | 0.011 | 0.156 |

Nonlinear-trend result: with a quadratic covariate-dependent trend, Unc bias persists at ~0.25 for all three quantiles at n=1,000 (inconsistent), while Cond bias falls to 0.011-0.017 - the method does not rely on linearity of trends, only on conditioning on the covariates that drive them. At n=100, Cond bias is mildly larger in the upper tail (0.152 at tau=.75 vs 0.082 at tau=.25).

**Relation to other methods:**

- **Callaway and Li (2019)** (2-period/2-group QTT under distributional PT + copula STABILITY): this paper is its generalization to multiple periods and staggered adoption, applying the CL19 machinery "to each pair of treated cohorts and never-treated units" (p. 3). Key substitution: copula stability over time (needs panel + >= 2 pre-treatment periods) is replaced by copula invariance across groups (Assumption 5), which needs neither and extends to repeated cross sections (pp. 3, 13). CL19's empirical bootstrap is retained for QTT inference (p. 4); CL19's IPW (after Firpo 2007) is generalized for the conditional-PT case (p. 17); the asymptotics of the EDF plug-in are inherited from CL19 (footnote 16, p. 21).
- **Callaway, Li, and Oka (2018)** ("Callaway et al. (2018)"): source of the copula-invariance-across-groups idea and of the repeated cross-sections / discrete-covariates conditional implementation (pp. 3, 5, 13). CLO18's conditional copula invariance results pertain to the CONDITIONAL QTT; this paper also delivers the unconditional QTT (Proposition 2, integrating out X) (p. 5).
- **Callaway and Sant'Anna (2021)**: supplies the group(cohort)-time parameter architecture, never-treated vs not-yet-treated comparison-group logic, the Limited Treatment Anticipation extension, and the aggregation schemes (Eq. (4), event-time and overall weights) - all generalized from means (ATT(g,t)) to distributions/quantiles (pp. 3-4, 7, 11, 19-20). Assumptions 1-3 closely align with CS (2021) (p. 32).
- **Athey and Imbens (2006) CiC** (pp. 2, 6, 31): relates the untreated outcome to group, time, and unobservables via a monotonic production function; allows selection on unobservables but assumes the unobservables' distribution within a group is stable over time. This paper does NOT restrict the functional form relating groups, time, and covariates to outcomes - but, unlike CiC, it is **not scale-invariant** (p. 31). Melly and Santangelo (2015) extend CiC to covariates via quantile regression (cited as the Case C remedy, p. 22). All the alternative counterfactual-distribution methods in Section 5 are declared extendable to staggered adoption using the paper's intuition (p. 31).
- **Bonhomme and Sauder (2011)** (pp. 12, 16, 31): requires the production function mapping groups, time, and covariates into outcomes to be ADDITIVE. This paper's assumptions are weaker on serial correlation (allowed) and on correlation between time-varying shocks and unobserved heterogeneity (allowed), and it does not require the time-varying unobservable to be independent of treatment assignment; but unlike Bonhomme-Sauder it does NOT allow returns to unobserved skills to change after the policy (p. 31). The generality "comes at the cost of an additional assumption regarding the missing dependence (copula) between the change in the untreated potential outcome and its pre-treatment level" (p. 31).
- **Li and Lin (2024)** (concurrent; the only other QTT identification in staggered DiD known to the author, pp. 3, 5-6): both use distributional PT + copula invariance and extend to repeated cross sections, but (a) Li-Lin extend Callaway et al. (2018) whereas this paper generalizes Callaway-Li (2019); (b) Li-Lin use a NOT-YET-TREATED comparison group vs never-treated here; (c) Li-Lin offer NO estimator for the QTT and do not specify covariate nature, leaving unconditional QTT estimation under conditional copula invariance unclear; this paper provides estimators, addresses the five-conditional-distributions challenge via three identification scenarios, and adds aggregation schemes.
- **Fan and Yu (2012)** (pp. 3, 11, 13): distributional PT alone only PARTIALLY identifies the distributional effect - motivates the copula assumption.
- **Firpo (2007)** / **Abadie (2005)** (pp. 3, 12, 17): selection-on-observables IPW quantile treatment effects; distributional PT is weaker; the reweighting lineage generalized in Proposition 1.
- **Miller (2023)** (footnote 15, p. 17): distributional PT usage; the doubly-robust route not pursued here.
- **Maasoumi and Wang (2019)** (pp. 4, 8-10, 20, 23): evidence against rank invariance; source of the FSD/SSD framework and the pair-bootstrap SD test adapted here. Linton, Maasoumi, and Whang (2005): the generalized KS statistics.
- **Sensitivity caveat vs CS-2021 (Discussion, p. 32)**: the estimator "is sensitive to strong violations of the Copula Invariance assumption (not required in Callaway and Sant'Anna, 2021) and may underperform, compared to the method proposed by Callaway and Sant'Anna (2021), when the PT assumption holds on average but not across the entire distribution" - researchers should weigh assumption plausibility against other approaches. Footnote 21: Imbens and Wooldridge (2009) for a review of common counterfactual-estimation assumptions; footnote 22: Wooldridge (2021) for a comparison of CS 2021 / Sun-Abraham 2021 / Wooldridge 2021. Use-case example (pp. 31-32): minimum wage / teen employment as in CS (2021), distributional effects "particularly on the lower deciles of the income distribution" - relevant for equity assessments.
- Broader staggered-ATT literature cited for context (p. 3): Borusyak et al. (2021), Sun and Abraham (2021), Wooldridge (2021), de Chaisemartin and d'Haultfoeuille (2022); Roth et al. (2023) review.

**Reference implementation(s):**
- **The initiative knows of NO verified reference implementation of this paper's estimator.** The paper's own p. 1 footnote states the Monte Carlo simulations were run in STATA and that "the ado files of the command used for implementing the methodology presented in the paper, `qtt`, are available upon request at the time of writing" - i.e., a private Stata implementation exists but is not published; no software, package, or replication code is mentioned anywhere in pp. 16-32 (verified during extraction).
- **Do NOT conflate with the R `ecic` package (Kluser)**: `ecic` implements a RELATED but methodologically DIFFERENT staggered event-study CiC (the Athey-Imbens transformation applied per cohort-period), not this paper's copula-based Callaway-Li-style method. Verify any candidate parity target's methodology against Theorem 1 / Proposition 1 before a future implementation PR.

**Requirements checklist (for the future scoped PR, not v1):**
- [ ] Identify or build a parity target first: request the author's Stata `qtt` ado, or construct simulation-based golden values from the paper's DGPs 1-5 (analytical true quantiles exist for DGPs 1, 2, 4); there is no published reference implementation to test against
- [ ] Re-verify all equation/assumption/table numbers against any published version of record (this review is pinned to arXiv v2)
- [ ] Resolve the Assumption 4 `t >= q - rho` restriction and the uniform-band centering discrepancy (see Gaps and Uncertainties) before writing code
- [ ] (g,t)-cell architecture: reuse diff-diff's CallawaySantAnna cohort-time cell and aggregation infrastructure (Eq. (4); event-time and overall weights with sample-analog estimation; base period `r - rho - 1`; anticipation parameter `rho`; never-treated default with a not-yet-treated option)
- [ ] EDF plug-in estimation per Section 3 Case A; inf-based quantile inversion; no smoothing anywhere
- [ ] IPW path (Proposition 1): parametric generalized propensity score `P_{r,t}(X)` with pairwise cohort-vs-never-treated conditioning; Hajek weight normalization; overlap/trimming warnings as `p_r(X) -> 1` (Assumption 7)
- [ ] Fully conditional path (Proposition 2): discrete-covariate cells per CLO18, warn on the five-conditional-distributions cost; integrate out against `F-hat_{X|d_r=1}` for the unconditional QTT
- [ ] Empirical bootstrap sup-t uniform bands with the IQR-based variance scale; tau grid restricted to a compact `[eps, 1-eps]`; extreme-quantile warning (footnote 17); wild/subcluster options documented for clustered data
- [ ] Stochastic dominance tests (`d_{r,t,rho}`, `s_{r,t,rho}`) with pair bootstrap as the rank-invariance-free summary
- [ ] Continuity/ties check (Assumption 6): warn on heavy ties or atoms - the `F^{-1} o F` compositions and copula uniqueness break on discrete outcomes
- [ ] Small-cohort guards: the paper's own simulations failed to compute for T = R = 10 with n = 100; warn on tiny cohort/never-treated cells
- [ ] Document tail-quantile sensitivity to copula-invariance violations (Table 4: Cond .25 bias 0.023 -> 0.203 as rho-bar goes 0 -> 0.5) and median robustness; document the DGP 4 caveat that conditioning can be worse than not conditioning under large PT violations
- [ ] RCS mode only via Corollary 1 with an explicit rank-invariance warning (strong, often rejected empirically; neither implies nor is implied by Assumption 5); RCS-with-covariates IPW needs the modified propensity score from CS (2021) Appendix B, which this paper does not derive

---

## Implementation Notes

**Relevance to diff-diff CiC/QDiD v1 (2026-07-12):**
- (i) This review documents the deferred staggered extension, so the ROADMAP's "Distributional DiD for staggered timing (Ciaccio)" row is now backed by a reviewed scope rather than a bare citation - the row moves from "commit when a user reports need" to "reviewed, implementation deferred pending demand".
- (ii) The Assumption 4 (distributional parallel trends) / Assumption 5 (copula invariance) structure is the multi-period generalization of the Callaway-Li-Oka (2018) machinery also reviewed today - together the three reviews map the whole panel-copula design space: CLO18 (2 periods, copula invariance across groups), CL19 (>= 3 periods, copula stability across time), Ciaccio (staggered multi-period, copula invariance across groups per cohort-time cell).
- (iii) The Unc-vs-Cond Monte Carlo results are quantitative evidence for why v1's covariates deferral must be documented loudly rather than silently: the unconditional estimator's bias does not vanish with n and GROWS across later-treated cohorts (Table C.2 at n=1,000: 0.25 for d_2, 0.371 for d_3, 0.479 for d_4) and persists under quadratic trends (Table C.4: ~0.25 at all three quantiles at n=1,000). Ignoring relevant covariates is an inconsistency, not a finite-sample nuisance.
- (iv) The CS-2021-style aggregation architecture (cohort-time cells, Eq. (4) weights, event-time and overall schemes, sample-analog weight estimation) means a future staggered distributional PR can reuse diff-diff's existing CallawaySantAnna (g,t) cell/aggregation infrastructure rather than building parallel plumbing.

### Data Structure Requirements
- Balanced-in-spirit panel with `T > 2` periods, unit-level first-treatment cohort `d_r`, and a never-treated group (`C = 1`); always-treated units must be dropped (footnote 3, p. 7). Not-yet-treated comparisons are possible by symmetry but not spelled out.
- The estimator consumes, per (r,t) cell: never-treated units' long differences `Y_t - Y_{r-rho-1}` and base-period levels `Y_{r-rho-1}`, plus treated cohort r's base-period marginal (for the quantile-quantile transform) and post-treatment outcomes (for the treated quantile).
- Covariates must be PRE-TREATMENT; the fully conditional path effectively requires discrete covariates (cell-by-cell EDFs) or a quantile-regression approximation.
- Continuous outcomes (Assumption 6): ties/atoms break copula uniqueness and every `F^{-1} o F` composition.
- Repeated cross sections: supported only via Corollary 1 at the cost of over-time rank invariance within the never-treated group (comonotone copula), compositional stability (Assumption B.1), and - with covariates on the IPW path - a modified propensity score this paper does not derive.

### Computational Considerations
- Pure ECDF plug-in per (r,t) cell: sorting dominates, O(n log n) per cell; the quantile-quantile transform evaluates via two searchsorted passes over sorted arrays. No kernels, bandwidths, or density estimation for point estimates or bootstrap inference (densities/copula pdfs appear only in the Appendix A proofs).
- The copula is never estimated explicitly - Theorem 1 handles it implicitly through each never-treated unit's own (long difference, base level) pairing. An implementation that separately estimates a copula object is over-engineering the estimator.
- The number of (r,t) cells grows as O(T^2); each bootstrap iteration re-runs all cells, and the paper's convention is B = 2,000 - budget accordingly. Replicates are embarrassingly parallel.
- The IPW path adds one propensity-score fit per cohort r (pairwise cohort-vs-never-treated, NOT one multinomial fit) plus Hajek reweighting of the never-treated change distribution.
- Small cells are the binding constraint in practice: with T = R = 10 and n = 100 the paper's own simulations frequently failed to produce estimates (p. 27). More periods shrink cohorts; guard and warn.

### Tuning Parameters

| Parameter | Type | Default | Selection Method |
|-----------|------|---------|-----------------|
| `rho` (anticipation) | int >= 0 | 0 (no anticipation) | Known from institutional context (Assumption 3 requires a KNOWN rho); shifts the base period to `r - rho - 1` |
| `n_bootstrap` | int | 2,000 (paper's Monte Carlo convention; no B fixed in the algorithm text) | Band stability |
| `quantiles` (tau grid) | floats in `[eps, 1-eps] subset (0,1)` | paper reports .25/.50/.75 in simulations | Compact grid strictly inside (0,1); avoid extreme quantiles (footnote 17) |
| comparison group | {never-treated, not-yet-treated} | never-treated | Not-yet-treated by symmetric arguments (CS 2021); prefer never-treated if later-treated units anticipate (pp. 5, 7) |
| covariate path | {unconditional, IPW (Prop 1), fully conditional (Prop 2)} | unconditional only if Assumptions 3-5 plausibly hold unconditionally ("highly unlikely", p. 17) | IPW when only PT is conditional; Prop 2 (discrete cells) when the copula is also conditional |
| propensity-score model | parametric spec of `P_{r,t}(X)` | logit-style (Eq. (6) used in simulations) | Misspecification risk; doubly-robust alternative (Miller 2023) noted but out of scope |
| aggregation | {none (cell-level), event-time, overall} | cell-level QTT_{r,t,rho}(tau) | Eq. (4) weights, sample-analog estimation (p. 23) |

No smoothing parameters exist anywhere in the method.

### Relation to Existing diff-diff Estimators
- `CallawaySantAnna`: the mean-ATT sibling and architectural template - cohort definitions, limited anticipation, never-/not-yet-treated comparison logic, and the aggregation schemes are lifted from CS (2021) and generalized to quantiles. A future implementation should reuse the CallawaySantAnna (g,t) cell enumeration and aggregation weight code paths. Note the CS-2021 sensitivity caveat (p. 32): when PT holds on average but not distributionally, CS-2021 is the safer tool - the docs comparison table should carry this.
- `ChangesInChanges` / `QDiD` (v1): 2x2 distributional siblings with a DIFFERENT identification strategy (monotone production function vs distributional PT + copula invariance). The Discussion's trade-offs belong in the choosing-an-estimator docs: this method is not scale-invariant (CiC is), does not allow post-policy changes in returns to unobserved skills (Bonhomme-Sauder does), but imposes no functional form and tolerates serial correlation and shock-heterogeneity correlation.
- The ATT is recoverable as a by-product by integrating the counterfactual quantile function (footnote 12, p. 13) - a natural cross-check against `CallawaySantAnna` output in any future test suite (under Assumptions 1-4 alone the ATT is identified, footnote 14, p. 16).
- Bootstrap utilities: the sup-t band with the IQR-based variance scale is the same Callaway-Li (2019) construction the CiC/QDiD v1 bootstrap docs cite; any implementation must reuse `safe_inference()` joint-NaN conventions from `diff_diff.utils` and resolve the band-centering discrepancy (Gaps item 1) before shipping bands.

---

## Gaps and Uncertainties

**Contradictions and as-printed flags (both versions preserved; do NOT resolve by guessing - check the PDF and any published version):**

1. **Uniform-band centering (p. 22).** The paper prints the (1-alpha) uniform band as `C-hat_{QTT(tau)} = QTT-hat*(tau) +/- c^B_{1-alpha} Sigma-hat(tau)^{1/2} / sqrt(n)`, centered on `QTT-hat*(tau)` - the BOOTSTRAP estimate. In Callaway-Li (2019), whose empirical bootstrap this section adopts, the band is centered on the ORIGINAL-SAMPLE estimate `QTT-hat(tau)`. Either the printed `*` is a typo or the author intends a bootstrap-centered band; a future implementation must pick the CL19 centering deliberately and document the deviation, or reconcile against a published version.
2. **Assumption 4's time restriction as printed (p. 12).** The statement reads "For each r, t in {q, ..., T} such that `t >= q - rho`". The parallel objects elsewhere in the paper (the QTT definitions on pp. 18-19, Assumption 3's `t < r - rho` cutoff, and the base period `r - rho - 1`) are all expressed relative to the cohort date `r`, so `q` may be a typo for `r` - but the extraction transcribed the printed `q`, and this review does not resolve it. Verify against any published version before implementing the assumption's period-eligibility logic.
3. **`Y_t(inf)` notation in the p. 18 QTT definition.** The post-identification `QTT_{r,t,rho}(tau)` on p. 18 is printed with what appears to be `F^{-1}_{Y_t(infinity)|d_r=1}(tau)` as the subtrahend (the never-treated potential outcome), while the p. 19 restatement of the same object uses `F^{-1}_{Y_t(0)|d_r=1}(tau)`. The two notations are used interchangeably for the never-treated potential outcome; transcribed as printed, not normalized.
4. **Duplicate "(A.11)" display label (p. 40).** The final display of the Proposition 1 proof chain (the IPW weight form `E[(C p_r(X) / (p_r (1 - p_r(X)))) 1{Delta Y_t <= delta}]`) is tagged "(A.11)", duplicating the (A.11) of the Alternative Proof on p. 39 - presumably a typo for (A.15) or a back-reference to Eq. (3). Do not cite "(A.11)" without page context. The same chain also mixes `P_r(X)` and `p_r(X)` capitalization (as printed).

**Verified quirks (not contradictions, but easy to trip over):**

5. Assumption 2's last treatment subscript is typeset `tau` (`D_{i,tau}`) in the printed statement - transcribed as printed.
6. The Case A step-2 estimator (p. 21) prints the first composed term as `F-hat^{-1}_{DeltaY|C=1}(F-hat_{DeltaY|C=1}(.))` - both conditionals never-treated, an identity composition under continuity. This mirrors the (A.6) -> (A.7) simplification (Assumption 4 substitutes the never-treated change quantile for the treated one); an implementation can skip the redundant composition but should document why.
7. Appendix A notation slips (as printed): the support of the pre-period level is written `Y_{t-1}(0)` in one place and `Y_{r-1}(0)` elsewhere (p. 37); the (A.9)/(A.10) rank definitions carry redundant conditioning in the printed `v_i^r`/`v_i^0` definitions; Corollary 1's imputed change ends with `- Y_{r-1}` (no `(0)`).
8. Table C.1's notes boilerplate references 'UNC' rows that do not exist in the no-covariate DGP 1 table (p. 43).
9. DGP 3 notation clash (p. 25): `T` denotes both the number of periods and a chi-squared(1) random variable in `Z_r = r + T`.

**Version and coverage gaps:**

10. **Preprint status.** arXiv v2 is explicitly "a preliminary version" (p. 1 footnote); no published version of record exists as of 2026-07-12. Any published version may renumber assumptions/equations/tables and may resolve items 1-4 above; re-verify before citing in shipped docs.
11. **No empirical application** exists in this version (simulations only) - there is no published-number replication target; parity work must go through the author's unreleased Stata `qtt` ado or simulation-based golden values.
12. **No new asymptotic theory.** Section 3 states no theorem; consistency and the functional CLT are inherited from Callaway and Li (2019) via footnote 16 with the claim that "the only difference lies in the construction of the benchmark, not in the estimator itself." Hadamard-differentiability machinery is not restated; formal proofs of the staggered extension's asymptotics are not in the paper.
13. **External proof dependencies.** Lemma A.1 and Lemma A.2 (Sklar's-theorem results used in the Theorem 1 proof) live in Appendix A of Callaway and Li (2019) and are deliberately not reproduced (p. 37). The RCS-with-covariates IPW path defers the modified generalized propensity score to Appendix B of Callaway and Sant'Anna (2021).
14. **Wild/subcluster bootstrap is a pointer, not a procedure** (p. 22): the clustered-inference route is described as adapting MacKinnon-Webb (2018) "with minor modifications" with no algorithmic detail; a future implementation would have to derive the QTT adaptation itself.
15. **Extraction coverage.** This review synthesizes three extraction passes (pp. 1-20, pp. 16-35, pp. 31-44; printed page = PDF page) with overlaps on pp. 16-20 and 31-35 reconciled as documented; References (pp. 33-36) were skipped by design. Tables 2-4 numbers above are the selected values transcribed by the extraction, not full tables.
16. **No verified reference implementation** (see Reference implementation(s) above); in particular the R `ecic` package is NOT an implementation of this paper despite the overlapping "staggered distributional DiD" label.
17. **IPW display's sum index vs `C` indicator (p. 22).** The Proposition 1 estimator display prints `sum_{i in r}` in both numerator and denominator while each summand carries the never-treated dummy `C` - a literal cohort-r sum would be identically zero. The intended reading (consistent with the (A.11)-(A.14) proof chain and the `n_{r,t}` definition) is a sum over the (r,t) estimation subsample - cohort r plus never-treated - with `C_i` selecting never-treated units. Implement as a Hajek-normalized CDF over never-treated observations with weight `C_i * p_r(X_i) / (p_r * (1 - p_r(X_i)))`, then invert; do not copy the printed display literally.
