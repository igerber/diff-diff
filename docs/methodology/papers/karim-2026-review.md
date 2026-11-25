# Paper Review: Improved Inference for CSDID Using the Cluster Jackknife

**Authors:** Sunny R. Karim, Morten Ørregaard Nielsen, James G. MacKinnon, Matthew D. Webb (authors in certified random order)
**Citation:** Karim, S. R., Nielsen, M. Ø., MacKinnon, J. G., & Webb, M. D. (2026). Improved Inference for CSDID Using the Cluster Jackknife. arXiv:2602.12043v1 [econ.EM], dated February 13, 2026.
**URL:** https://arxiv.org/abs/2602.12043
**Source reviewed:** arXiv:2602.12043v1 (21 pages; references begin on p. 20; numbered equations (1)-(20), all legible). The PDF was reviewed externally and is not committed to the repository. Reproduce by downloading v1 from the arXiv link above. Greek letters and mathematical symbols inside quoted passages are transliterated to ASCII (for example `tau`, `rho`, `+/-`, `x` for the multiplication sign).
**Review date:** 2026-11-22

> **Personal verification stamp (load-bearing items spot-checked against the source).** Beyond the multi-agent extraction, the reviewer read pp. 11-12 directly and verified verbatim: equation (20), its full-sample centering and `(H - 1)/H` factor, and all three jackknife caveats. Also checked against the source: every rejection frequency in Tables 2 and 3 (pp. 15-16; 48 rows); the cluster-jackknife rows of Tables 4 and 5 (pp. 17-18); the identifying-assumptions sentence (p. 4); and the three-regression equivalence sentence (p. 7). The normal-reference P value for Table 5 Panel B was recomputed independently. Every quoted passage of 40 or more characters was additionally run through a mechanical word-sequence comparison against the PDF text layer.

---

## Methodology Registry Entry

**Status: implemented as `CallawaySantAnna(vcov_type="jackknife")`; this file is a non-authoritative source audit written before the implementation.** The `## CallawaySantAnna` REGISTRY section ("Cluster jackknife") is the **sole authoritative methodology contract**, including the library choices this review lists as open. Statements below about what the library lacks ("no full-re-estimation cluster jackknife today", the open items under *Relation to Existing diff-diff Estimators*) describe the library at review time. Every statement about the paper in this file is sourced from the paper only.

*Formatted to match the REGISTRY structure. The registry-candidate text ends just before `## Monte Carlo Evidence`; the sections after that boundary are evidence summaries and audit notes and are NOT part of the proposed registry text.*

### CallawaySantAnna - cluster jackknife inference (proposed addition)

**Primary source:** [Karim, S. R., Nielsen, M. Ø., MacKinnon, J. G., & Webb, M. D. (2026). Improved Inference for CSDID Using the Cluster Jackknife. arXiv:2602.12043v1 [econ.EM].](https://arxiv.org/abs/2602.12043)

**Stated contribution (abstract, p. 1; Section 1, p. 2):** the paper finds that "the issues of over-rejection with few clusters and/or few treated clusters are at least as severe for CSDID as for traditional DiD methods" and proposes "using a cluster jackknife for inference with CSDID, which simulations suggest greatly improves inference." It adds: "The jackknife substantially improves the reliability of inferences, but the problem of few treated clusters remains in some settings." (p. 2)

**Key implementation requirements:**

*Setting and notation (Section 2, p. 3):*

The paper does NOT use `G` for the number of clusters and has no symbol for the number of treated clusters. Its symbols are:

| Paper symbol | Meaning | Where |
|---|---|---|
| `H` | number of clusters | Section 3.1 (p. 10), Section 4 (p. 11) |
| `R` | number of regions (states in the simulations); `H = R` in all experiments | Section 2 (p. 3), Section 5 (p. 14) |
| `R_1`, `R_0` | sets of treated / never-treated regions | Section 2 (p. 3) |
| `S_r` | first treatment period of region `r`; `S_r = 0` if never treated | Section 2 (p. 3) |
| `S = (S_1, ..., S_R)` | `1 x R` timing vector | Section 2 (p. 3) |
| `calG` (script G in the paper) | set of unique non-zero values in `S`, i.e. the treated cohorts | Section 2 (p. 3) |
| bold `G_g` | `1 x R` binary cohort-membership vector, `G_{g,r} = 1` iff `S_r = g` | Section 2 (p. 3) |
| bold `G_infinity` | never-treated group, `G_{infinity,r} = 1` iff `S_r = 0` | Section 2 (p. 3) |
| `g` | cohort index = period of first treatment | Section 2 (p. 3) |
| `calG_comp(g,t)` | comparison group for a given ATT(g,t); variants `NT` (never-treated) and `NY` (not-yet-treated) | Section 2.2 (p. 6) |
| `N_{r,t}`, `N_{g,t}`, `N_{calG_comp,t}` | observation counts: region-period, cohort-period, comparison-group-period | Section 2.2 (p. 6) |
| `J`, `L` | number of early-cohort / late-cohort treated states in the simulations; treated clusters total `J + L` | Section 5 (p. 13) |

- Data level: persons `i` nested in regions `r` observed in periods `t in {1, ..., T}`. Treatment and cohort are defined at the region level, and the paper clusters at the region level.
- Target parameter:

```
ATT = E[ Y_{i,r,t}(1) - Y_{i,r,t}(0) | D_r = 1 ]                                   (1)
```

  where `D_r` is a binary treatment indicator "assigned at the region level" and only realized outcomes are observed.
- Absorbing treatment: "once a region is treated, it remains treated for all subsequent periods t >= S_r." (p. 3)
- Staggered adoption: `S` has multiple unique non-zero entries; common adoption: a single unique non-zero value.

*CSDID estimator as the paper defines it (Section 2.2, p. 5-9):*

- Two steps: "The first step estimates a series of 2 x 2 treatment effects, and the second step aggregates them." (p. 5-6). The first step "enumerates all the two-group, two-period differences that could estimate a treatment effect, excluding any forbidden comparisons."
- Base period: fixed at `g - 1` for every ATT(g,t) of cohort `g`: "the time periods used for the control group are the same as for the treated group, specifically, t and g - 1. Recall that g - 1 is the last pre-adoption period for cohort g." (p. 5-6). A varying base period, anticipation periods, and pre-treatment ATT(g,t) with `t < g`: Not discussed in paper.
- Group-time ATT, without covariates (p. 6):

```
ATT(g,t) = E[ Y_{r,t} - Y_{r,g-1} | S_r = g ] - E[ Y_{r,t} - Y_{r,g-1} | r in calG_comp ]      (6)
```

- Which (g,t): "For each g in calG, there is an ATT(g,t) for all g <= t <= T. Thus there is an ATT(g,t) for each timing cohort, for all post-treatment periods in which there exists an untreated comparison group." (p. 6)
- Estimator, stated for exogenous treatment and a fully-balanced panel (p. 6):

```
ATT-hat(g,t) = (1 / N_{g,t}) * SUM_{r: S_r = g} SUM_{i=1}^{N_{r,t}} ( Y_{i,r,t} - Y_{i,r,g-1} )
             - (1 / N_{calG_comp,t}) * SUM_{r in calG_comp} SUM_{i=1}^{N_{r,t}} ( Y_{i,r,t} - Y_{i,r,g-1} )      (7)
```

  with `N_{g,t} = SUM_{r: S_r = g} N_{r,t}` and `N_{calG_comp,t} = SUM_{r in calG_comp} N_{r,t}`. As written, each cohort mean in (7) is pooled over individuals across regions (observation-weighted), not an equal-weighted average of region means.
- Unbalanced panels and repeated cross-sections: "In repeated cross-sectional datasets, or in panel datasets with attrition, the number of observations in each cohort can differ across time periods, and (7) would have to be replaced by a more complicated equation." (p. 6-7). The replacement equation is not given.
- Comparison group (p. 6-8):
  - Never-treated (`NT`): "comprised of the cohort G_infinity, that is, the regions with S_r = 0."
  - Not-yet-treated (`NY`): "comprised of the G_infinity cohort and cohorts with g > t, that is, regions that were treated after period t, for which S_r > t."
  - Default as the paper states it: "By default, the csdid package uses only never-treated units when that is feasible." (p. 7)
  - Not-yet-treated "may provide more efficient estimates of the ATT, but it requires a no-anticipation assumption for those units." (p. 8)
  - "Sometimes, all groups are eventually treated. In that case, R_0 is empty, and only the not-yet-treated control group is used". (p. 6)
  - With more than one region in a comparison group, the regions enter with a weight. For Example 2 under not-yet-treated controls (p. 8):

```
ATT(2,2) = ( E[Y_{A,2} - Y_{A,1}] ) - ( omega * E[Y_{B,2} - Y_{B,1}] + (1 - omega) * E[Y_{C,2} - Y_{C,1}] )      (12)
```

    "where omega is a weight. It could be 1/2, or it could equal the average number of individuals in region B in the two periods divided by the average number in both regions B and C." The paper does not fix `omega`.
- Covariates: (6) and (7) are stated "Without covariates". Covariate-adjusted ATT(g,t) estimators (outcome regression, IPW, doubly robust): no formulas given. Not discussed in paper beyond the facts that the simulations and both empirical examples are run with covariates and that the R usage example passes `est_method = "reg"` (p. 20).
- Aggregation: the general aggregation equation (p. 7), with worked examples in equations (9) and (11) on the same page:

```
ATT-hat = SUM_{g in calG} SUM_{t=g}^{T} w_{g,t} * ATT-hat(g,t)                     (8)
```

  - "The weights w_{g,t} sum to one and are chosen by the user, but the default is equal weighting (referred to as "simple" weighting), so that ATT-hat is simply the average of the ATT-hat(g,t)." (p. 7)
  - Section 5 repeats this description: "we use simple aggregation (i.e. unweighted averaging) to estimate the ATT". (p. 13)
  - No cohort-size-based weight formula appears anywhere in the paper.
  - Group / cohort aggregation: mentioned only as "cohort-specific effects ATT(g)" (p. 3) and as an aggregation scheme the Stata package supports (p. 19). Weight formula: Not discussed in paper.
  - Calendar-time aggregation: mentioned only as an aggregation scheme the Stata package supports (p. 19). Weight formula: Not discussed in paper.
  - Event-study / dynamic aggregation: Not discussed in paper.
- Worked examples the jackknife caveats refer back to:

| Example | Layout (equation) | Cohorts | Estimable ATT(g,t) |
|---|---|---|---|
| Example 1 / Example 2 (p. 5, 7-8) | 3 regions x 3 periods, adoption matrix (5): A treated in period 2, B in period 3, C never treated | `calG = {2, 3}` | ATT(2,2), ATT(2,3), ATT(3,3), averaged with weight 1/3 each in (9) and (11) |
| Example 3 (p. 8-9) | 6 regions x 3 periods, matrix (13): A and D treated in period 2, B and E in period 3, C and F never treated; `S = (2, 3, 0, 2, 3, 0)` | `calG = {2, 3}`, two regions per cohort, two never-treated | same three terms |
| Example 4 (p. 9) | 3 regions x 4 periods, matrix (17): `S = (2, 3, 4)`, `R_0` empty | all regions eventually treated | "it is still only possible to estimate ATT(2,2), ATT(2,3), and ATT(3,3). We cannot estimate any ATT(g,4) because there are no untreated units in the fourth period." |

  - Without covariates, "a numerically identical estimate of the ATT can be obtained by using three individual 2 x 2 DiD models" (p. 7); for Example 3 the paper writes the 2 x 2 regression for ATT(3,3) on the subsample in cohorts `G_3` or `G_infinity` and periods 2 or 3 as equation (16) (p. 9).

*Existing inference procedures (Section 3, p. 9-11):*

Why clustering needs care for CSDID (p. 9): "observations from a single region may be used to estimate multiple ATT(g,t) terms", so "A valid routine for cluster-robust inference has to account for the resulting correlations."

Procedure 1 - recentered influence function (RIF) standard errors (Section 3.1, p. 10):

- "The default method for obtaining standard errors in Callaway and Sant'Anna (2021) uses the contributions to the influence functions by each of the observations. The expressions for the influence functions are quite long, so we omit them".
- Simplified statement of the asymptotic result the paper relies on:

```
sqrt(n) * ( ATT-hat_{g<=t} - ATT_{g<=t} )  -->_d  N(0, Sigma)                      (18)

Sigma = E[ Psi_{g<=t}(W_i) * Psi_{g<=t}(W_i)^T ]                                   (unnumbered, p. 10)
```

  where the bold `ATT_{g<=t}` and `ATT-hat_{g<=t}` stack ATT(g,t) and ATT-hat(g,t), `W_i` is the data vector for observation `i`, and `Psi_{g<=t}(W_i)` are the influence functions.
- (18) "is used to compute asymptotic RIF standard errors based on Sigma-hat, the sample analog of Sigma." The clustered form of `Sigma-hat`: Not discussed in paper (no formula).
- Reference distribution, as stated: "These RIF standard errors are then used with the standard normal distribution to calculate P values and confidence intervals." No degrees-of-freedom adjustment.
- The paper's critique (p. 10): "Almost all empirical work that uses cluster-robust t-statistics for linear regression models assumes that these are distributed as t(H - 1), where H is the number of clusters. ... Thus, it seems highly unlikely that the default RIF method for inference will yield reliable results when there is clustering and H is not large."

Procedure 2 - multiplier bootstrap (Section 3.2, p. 11):

```
ATT-hat*_{b,(g<=t)} = ATT-hat_{g<=t} + (1/n) * SUM_{i=1}^{n} V*_{b,i} * Psi-hat_{g<=t}(W_i)      (19)
```

- `b` indexes `B` bootstrap samples; `Psi-hat` is the sample analog of the influence function; `V*_{b,i}` are "random draws from an auxiliary distribution with mean 0 and variance 1, typically the Mammen distribution. When the data are clustered, these draws are constant within each cluster".
- The `B` vectors from (19) "are then used to form confidence intervals for the individual ATT(g,t) or for the ATT."
- No re-estimation: "With the multiplier bootstrap, however, these are not re-estimated. The bootstrap draws simply add noise to the original estimate ... Since the model is never re-estimated, this is just a convenient way to estimate Sigma."
- How the CI is built from the `B` draws, the number of replications `B`, any degrees-of-freedom choice: Not discussed in paper.
- P value: "the bootstrap procedure in csdid does not provide a P value" (p. 13).

*Cluster jackknife (Section 4, p. 11-12):*

- Proposal (p. 11): "we propose to use the cluster jackknife to calculate standard errors for ATT-hat. ... Another attractive feature of the jackknife is that no analytical expressions for the influence functions are needed. Instead, the cluster jackknife simply omits each cluster h in turn, estimates each of the ATT(g,t), and then uses those estimates to re-estimate the ATT. The standard error is then simply the square root of"

```
CV_3( ATT-hat ) = ((H - 1) / H) * SUM_{h=1}^{H} ( ATT-hat^{(h)} - ATT-hat )^2      (20)
```

  "where ATT-hat is the original CSDID estimate of the ATT, and ATT-hat^{(h)} is the estimate when cluster h is omitted."
- Name: `CV_3`, "because it generalizes the jackknife HC_3 variance estimator". The paper defines no other jackknife variant.
- Centering: on the full-sample estimate `ATT-hat`, not on the mean of the delete-one estimates.
- Scale factor: `(H - 1) / H`, with `H` the number of clusters.
- Range of the sum: all clusters `h = 1, ..., H`, treated and control alike.
- Standard error: the square root of (20).
- Scope of the formula: the scalar aggregate `ATT-hat` only. No vector or matrix version is given.
- What is deleted: one whole cluster at a time (all of its observations).
- What is re-estimated on each delete-one-cluster subsample: every ATT(g,t), then the aggregate ATT formed from them. Implementation is by full re-estimation (p. 12): "The jackknife we propose needs to call whatever command implements CSDID H + 1 times, once for the full sample estimate ATT-hat, and H additional times for each of the ATT-hat^{(h)} estimates."
- Aggregation weights in each subsample: the paper gives no explicit rule. Its second caveat (below) states that the weights given to the ATT(g,t) terms change across jackknife replications when a deletion removes terms, so the subsample aggregate is formed from whatever ATT(g,t) terms are estimable in that subsample rather than with weights held at full-sample values. Whether cohort-size-dependent weights are recomputed from subsample cohort sizes is not stated separately (reviewer inference: it follows only implicitly from re-running the estimation command on the subsample).
- Cluster level: in every simulation and example the cluster is the region / state, which is also the level at which treatment is assigned.
- Caveat 1 - deleted cohorts (p. 12): "the jackknife is not amenable to all staggered adoption settings. ... consider Example 1 with clustering at the region level. In this case, some of the ATT(g,t) terms cannot be calculated when certain regions are dropped. The ATT(2,2) and ATT(2,3) terms cannot be calculated when region A is dropped, the ATT(3,3) term cannot be calculated when region B is dropped, and no ATT terms can be calculated when region C is dropped. In contrast, all of the ATT(g,t) terms can be estimated for each jackknife sample for a setup like the one in Example 3. For that reason, we designed the Monte Carlo simulations described in Section 5 so that there are always multiple treated clusters and untreated clusters. However, in simulations with one early adopter and one late adopter, there are not multiple regions in each cohort."
- Caveat 2 - weights change across replications (p. 12): "some staggered adoption settings will alter the weights given to different ATT(g,t) across jackknife replications. Consider a modification of Example 1, where there are now two control groups, C and D. In this case, it would be possible to estimate an ATT for each jackknife replication. Dropping either region C or D would still allow all three ATT(g,t) terms to be calculated. However, dropping region A would mean that the ATT would be based solely on ATT(3,3). Likewise, dropping region B would mean that the ATT would just be a weighted average of ATT(2,2) and ATT(2,3). If the ATT is sensitive to the regions used to compute it, this variation in weights may highlight that fact and thus provide valuable information. However, the ATT-hat^{(h)} may now vary across jackknife samples in ways that would not happen for other applications of the jackknife. Thus, unlike with other inference procedures, care must be taken to assess whether the jackknife is a sensible tool to use for a given treatment design."
  - So, per the paper: when a deletion eliminates a whole cohort, `ATT-hat^{(h)}` is computed from the surviving ATT(g,t) terms only, and that value still enters (20). The paper treats this as something to be aware of, not as an error condition, and offers no correction.
- Caveat 3 - cost (p. 12): "the jackknife procedure can be slow. Many of the modern DiD routines, including CSDID, can be slow to calculate when samples are large and there are many ATT(g,t) terms to estimate."
- Diagnostic use of the delete-one estimates (Section 7.2, p. 18): "it is highly instructive to inspect the full vector of omit-one-cluster estimates to ensure that results are not driven by a single cluster."
- Theory: no asymptotic theory, consistency proof, or validity conditions for the jackknife applied to CSDID are given. Support is simulation evidence plus citations to results for the jackknife in other settings.

*Procedure (reconstruction - the paper has no algorithm box, pseudo-code, or numbered procedure; the steps below arrange the paper's own sentences from p. 11-12):*

1. Run CSDID on the full sample to obtain `ATT-hat` (p. 12: "once for the full sample estimate ATT-hat").
2. For each cluster `h = 1, ..., H`: omit cluster `h`; estimate each of the ATT(g,t) on the remaining data; use those estimates to re-estimate the ATT, giving `ATT-hat^{(h)}` (p. 11).
3. Compute `CV_3(ATT-hat)` from equation (20) and take the square root for the standard error (p. 11).

*Reference distribution:*

- Section 4 does NOT state the reference distribution or the degrees of freedom for jackknife t-tests or confidence intervals.
- The paper makes two explicit distributional statements, both in Section 3.1 (p. 10): RIF standard errors are used "with the standard normal distribution" (quoted under Procedure 1 above), and the `t(H - 1)` sentence, which concerns cluster-robust inference in linear regression generally. Neither specifies the jackknife reference distribution: the paper does not say that jackknife t-statistics for CSDID are compared with `t(H - 1)`.
- Section 5 (p. 13) says only: "The final rejection frequency is for t-statistics based on cluster-jackknife standard errors discussed in Section 4 and calculated using our csdidjack package."
- Section 8.1 (p. 19) says the Stata command "returns CV_3 standard errors, t-statistics, P values, and confidence intervals" without naming the distribution.
- **Reviewer arithmetic (NOT a statement in the paper).** Backing out the implied critical value `(CI upper - CI lower) / (2 * SE)` and recomputing P values from the reported ATT and SE in Tables 4 and 5:

| Row | H (states) | Implied critical value | t(H-1) 0.975 quantile | N(0,1) 0.975 quantile | Reported P | P under t(H-1) | P under N(0,1) |
|---|---|---|---|---|---|---|---|
| Table 4, jackknife | 34 | 2.027 | 2.0345 | 1.960 | 0.069 | 0.0695 | 0.0606 |
| Table 4, asymptotic | 34 | 1.966 | 2.0345 | 1.960 | 0.038 | 0.0455 | 0.0377 |
| Table 5 Panel A, jackknife | 29 | 2.0482 | 2.0484 | 1.960 | 0.8045 | 0.8044 | 0.8026 |
| Table 5 Panel A, asymptotic | 29 | 1.962 | 2.0484 | 1.960 | 0.0040 | 0.0078 | 0.0042 |
| Table 5 Panel B, jackknife | 28 | 2.0516 | 2.0518 | 1.960 | 0.7770 | 0.7770 | 0.7748 |
| Table 5 Panel B, asymptotic | 28 | 1.960 | 2.0518 | 1.960 | 0.2860 | 0.2960 | 0.2866 |

  Reviewer inference: the reported jackknife rows are numerically consistent with a `t(H - 1)` reference distribution, with `H` the number of clusters in the estimation sample, and the reported asymptotic rows broadly resemble `N(0,1)` inference. This rests on rounded table values (Table 4 is reported to 3 decimals) and on taking `H = 34 / 29 / 28` from the paper's text. It is not asserted anywhere in the paper. One asymptotic row does not reproduce: for Table 5 Panel B the reported P value is 0.2860, while a normal Wald calculation from the printed ATT (0.1476) and standard error (0.1385) gives 0.2866, and letting both vary over their four-decimal rounding intervals gives 0.2862 to 0.2869, which excludes 0.2860 (reviewer arithmetic). This is an apparent inconsistency in the source table, not a transcription difference.

*Edge cases (case: detection -> handling per paper):*

- **Deleting the only cluster of a treated cohort** (Example 1, drop A or B; p. 12): detection - Not discussed in paper -> handling - that cohort's ATT(g,t) terms "cannot be calculated"; per caveat 2 the subsample ATT is then "based solely on" / "a weighted average of" the surviving terms and still enters (20). No renormalization formula and no validity threshold; only "care must be taken".
- **Deleting the only never-treated cluster, so no ATT(g,t) is estimable** (Example 1, drop C; p. 12): detection - Not discussed in paper -> handling - Not discussed in paper. The paper states only that "no ATT terms can be calculated". Whether the jackknife is then undefined, whether cluster `h` is skipped, or whether `H` and the `(H - 1)/H` factor are adjusted: Not discussed in paper.
- **Every cohort and the never-treated group contain more than one cluster** (Example 3; p. 12): detection - Not discussed in paper -> handling - "all of the ATT(g,t) terms can be estimated for each jackknife sample". The simulations are designed "so that there are always multiple treated clusters and untreated clusters", but the paper adds that "in simulations with one early adopter and one late adopter, there are not multiple regions in each cohort" (p. 12): the `J = L = 1` designs therefore do NOT preserve every ATT(g,t) after deletion.
- **One early and one late adopter** (`J = L = 1`; p. 12): detection - Not discussed in paper -> handling - no special handling; the paper notes "there are not multiple regions in each cohort" and still reports jackknife results for these cells (the largest over-rejection in Tables 2-3).
- **Few treated clusters relative to total clusters** (p. 15-16): detection - no rule or threshold given -> handling - none; "there can still be noticeable over-rejection when the number of treated clusters is small relative to the total number of clusters."
- **Few clusters overall** (p. 20): detection - no threshold given -> handling - none; "there are still cases (such as few clusters and/or few treated clusters) where inference can be quite unreliable".
- **Unbalanced cluster sizes** (p. 11, 17): detection - Not discussed in paper -> handling - no special handling stated; "This is a setting in which we might expect the jackknife to be more reliable".
- **A single influential cluster** (p. 18): detection - inspect "the full vector of omit-one-cluster estimates" (the paper plots them against cluster size in Figure 1) -> handling - in the minimum-wage example the authors drop the outlier state and repeat the analysis; no general rule.
- **All regions eventually treated** (Example 4, p. 9): for the estimator, "the never-treated option is infeasible, and the not-yet-treated set of comparison groups has to be used" and no ATT(g,T) for the final period is estimable. Under the jackknife: Not discussed in paper.
- **Not-yet-treated comparison group under the jackknife**: Not discussed in paper.
- **Repeated cross-sections or panels with attrition** (p. 6-7): equation (7) "would have to be replaced by a more complicated equation"; the equation is not given. Jackknife-specific handling: Not discussed in paper.
- **Covariate-specific failures in a delete-one subsample** (collinearity, propensity-score overlap or non-convergence): Not discussed in paper.
- **Cluster level finer or coarser than the treatment-assignment level, clusters spanning cohorts, unit-level clustering**: Not discussed in paper.
- **Identifying assumptions, pre-treatment (`t < g`) ATT(g,t), varying base period, anticipation**: the identification discussion states that "the two most critical are the parallel trends assumption and the no-anticipation assumption" (p. 4), and not-yet-treated controls require "a no-anticipation assumption for those units" (p. 8). Anticipation allowances (an anticipation window), varying base periods, pre-treatment ATT(g,t), and any jackknife-specific treatment of these: Not discussed in paper.

**Reference implementation(s):** the paper states that it provides a Stata package `csdidjack` and an R package `didjack` (Section 8). See `## Software (Section 8)` below for exactly what the paper says about them; the paper pins no version or commit.

### Paper-derived requirements

*Only what the paper specifies. These do not constrain the API surface.*

- [ ] Jackknife variance for the overall aggregate ATT computed by equation (20): sum over all `H` clusters, treated and control alike, with scale factor `(H - 1) / H`
- [ ] Deviations in (20) centered on the full-sample CSDID estimate `ATT-hat`, not on the mean of the delete-one estimates
- [ ] Standard error = square root of (20)
- [ ] Each delete-one replication omits one whole cluster, re-estimates every ATT(g,t) on the remaining data, and re-forms the aggregate ATT from those estimates (full re-estimation; `H + 1` estimations in total)
- [ ] No influence-function expressions enter the jackknife variance
- [ ] When a deletion removes an entire cohort but some ATT(g,t) remain estimable, the subsample ATT is the aggregate of the surviving terms and enters (20) (caveat 2, p. 12); the paper pairs this with the instruction that "care must be taken to assess whether the jackknife is a sensible tool to use for a given treatment design"
- [ ] The vector of omit-one-cluster ATT estimates is available for inspection (p. 18: "highly instructive"; the paper states its Stata package makes them available)
- [ ] Scope: equation (20) gives the jackknife variance for the scalar aggregate ATT of equation (8), whose weights `w_{g,t}` are user-chosen. The paper's simulations and empirical examples all use simple aggregation as the paper describes it, with clusters equal to the level of treatment assignment

### What the paper does NOT provide

Any item below, if present in a library implementation, would be a **library design choice** that needs its own source or an explicit documented choice. None of them can be attributed to this paper.

- **No per-ATT(g,t) jackknife**: no formula and no evidence for jackknife standard errors of individual ATT(g,t) terms, even though each is re-estimated in every subsample (p. 11).
- **No group, calendar, or event-study jackknife formula or evidence**: group and calendar aggregation appear only as schemes the Stata package is said to support (p. 19); event-study aggregation is not mentioned at all. Whether (20) is applied element-wise to each reported aggregate parameter is not stated, nor what happens when one parameter is undefined in some subsamples while the overall ATT is not.
- **No jackknife covariance matrix**: (20) is scalar; no matrix or vector version, and no jackknife covariances between ATT(g,t) terms.
- **No uniform (simultaneous) confidence bands and no joint (Wald) tests** under the jackknife.
- **No stated reference distribution or degrees of freedom** for jackknife t-tests and confidence intervals (see *Reference distribution* above; the `t(H - 1)` reading is reviewer inference from the tables).
- **No rule for a delete-one subsample in which no ATT(g,t) is estimable**: no detection criterion, and no statement on whether the SE is undefined, the cluster is skipped, or `H` is adjusted (p. 12).
- **No renormalization rule for partially estimable subsamples**: no explicit weight formula for the surviving terms, no threshold at which the practice becomes invalid, and no statement on whether software warns (p. 12).
- **No rule on recomputing aggregation weights** from subsample cohort sizes versus holding them at full-sample values (p. 11-12).
- **No alternative centering**: the mean-of-replicates centering is not mentioned, nor when it would differ from full-sample centering.
- **No theory**: no formal validity result for the cluster jackknife applied to CSDID.
- **No power, coverage, or CI-length results**: the Monte Carlo experiments report only rejection frequencies under a zero-effect null, with two cohorts, `J = L`, and `T = 8`; nothing with non-zero or heterogeneous effects or unequal `J` and `L` is simulated. The two empirical examples cover other time spans (2005-2018 and 2001-2007) and, in the prenatal-policy example, more than two adoption cohorts (p. 17), but they are single applications, not simulation evidence.
- **Covariates under the jackknife**: no covariate-adjusted ATT(g,t) formula at all, and no discussion of covariate-related subsample failures.
- **Estimation method under the jackknife** (outcome regression / IPW / doubly robust): the only information is the R usage example, which applies the jackknife after estimation with `est_method = "reg"` (p. 20). The method behind the "with covariates" simulations is not stated, methods are not compared, and no covariate-adjusted formula is given.
- **Not-yet-treated controls under the jackknife**: not discussed, including the all-eventually-treated case.
- **Varying base period, pre-treatment ATT(g,t), anticipation under the jackknife**: not discussed.
- **Unit-level clustering, or any cluster level other than the treatment-assignment level, under the jackknife**: not discussed.
- **The repeated-cross-section / unbalanced-panel estimator formula**: said to be "more complicated" than (7) and not given.
- **Any closed-form or shortcut jackknife**: none given; no timing numbers reported.
- **Software arguments and internals**: the paper gives a minimal invocation for each package, shows `res <- didjack(ag)` followed by `summary(res)`, and names the Stata command's outputs ("CV_3 standard errors, t-statistics, P values, and confidence intervals", p. 19). It does not document further arguments, options, defaults, or the structure of return values, and says nothing on how the packages re-invoke the underlying command, handle failed subsamples, or choose the reference distribution (p. 19-20).
- **Any remedy for the remaining few-treated-clusters over-rejection**: none is proposed or evaluated for CSDID.

---

<!-- ============================================================
     END of registry-candidate block. Everything BELOW this
     marker is evidence summary / audit notes and is NOT part
     of the proposed REGISTRY.md text.
     ============================================================ -->

## Monte Carlo Evidence (Sections 5-6)

### Design (Section 5, p. 12-14)

There is no parametric DGP in the paper: no DGP equations, no error-correlation parameter, no simulated cluster sizes. The experiments are "placebo laws" on real data.

- Data (p. 13): "the same dataset used for the placebo-laws simulations in MacKinnon and Webb (2017). Our dataset is extracted from the Merged Outgoing Rotation Group (MORG) of the CPS. It includes individual-level information on weekly earnings, state, and year, covering the period from 1979 to 1999. The dependent variable is the log of women's earnings. We exclude observations with earnings less than $20, which may be erroneous. The total sample, from all years and states, has 547,818 observations."
- Sampling per replication (p. 13): "Across replications, we randomly pick 8 consecutive years of data and R states from the full sample." So `T = 8`.
- Treatment assignment (p. 13): "treatment is assigned in two cohorts, an early cohort of J states treated in year 4, and a late cohort of L states treated in year 6. This results in J + L total treated states, and we set J = L." Treatment applies to all individuals in a state and is absorbing.
- Never-treated states per replication: `R - 2J` (reviewer arithmetic from the stated design; at least 2 in every design cell).
- Number of ATT(g,t) terms per replication: not stated by the paper. Reviewer arithmetic from the paper's definitions (cohorts `g = 4` and `g = 6`, `T = 8`, `g <= t <= T`): 5 + 3 = 8.
- True effect: zero; the hypothesis tested is `ATT = 0`. Nominal level 5%.
- Estimation (p. 13): "we estimate the ATT using the csdid package, both with and without covariates. In all simulations, we use simple aggregation (i.e. unweighted averaging) to estimate the ATT".
- Covariates (p. 13): "age, age squared, and a set of four education dummies. These covariates, and the outcome variable, have all been demeaned at the state level using the full set of data."
- Clustering (p. 13-14): "always clustering at the state level"; "Since all three inference methods use states as clusters, H = R in all our experiments."
- Replications (p. 14): "Our experiments have 2,400 replications. This number is quite a bit smaller than we would like, but the simulations are very slow, particularly for the jackknife."
- Software (p. 12): "Our simulations were conducted in Stata, using both the csdid package and our new csdidjack package."

Grid (Table 1, p. 13: "Values of R and J = L studied in our experiments"):

| | R = 8 | R = 16 | R = 24 | R = 32 |
|---|---|---|---|---|
| J = L = 1 | yes | yes | yes | yes |
| J = L = 2 | yes | yes | yes | yes |
| J = L = 3 | yes | yes | yes | yes |
| J = L = 4 | | yes | yes | yes |
| J = L = 6 | | | yes | yes |
| J = L = 8 | | | yes | yes |
| J = L = 10 | | | | yes |
| J = L = 12 | | | | yes |

(21 design cells by reviewer count of the table; the same 21 cells are populated in each panel of Tables 2 and 3.)

The three tests compared (p. 13):

1. "t-statistics based on the "cluster" option of csdid, which uses the recentered influence functions discussed in Section 3.1."
2. "the multiplier bootstrap discussed in Section 3.2. Since the bootstrap procedure in csdid does not provide a P value, we calculate the rejection rate based on whether the 95% confidence interval covers the true value of 0."
3. "t-statistics based on cluster-jackknife standard errors discussed in Section 4 and calculated using our csdidjack package."

Design details the paper does not report: state sample sizes and their heterogeneity; whether states are sampled with or without replacement and how the treated states are chosen beyond "randomly pick"; the comparison group used; the `csdid` estimation method with covariates; the number of bootstrap replications `B` and the weight distribution; the critical value used for the jackknife test; seed and software versions.

### Results (Section 6, p. 14-16)

All results are in Tables 2 and 3; there are no simulation figures. Rows are `J = L`; columns are `H = R`. Entries are rejection frequencies at nominal 5% for the test of `ATT = 0`. A blank cell is not in the design.

**Table 2 (p. 15): "Rejection frequencies for models with covariates"**

Panel A: Asymptotic (RIF) using `csdid`

| J = L | H = R = 8 | H = R = 16 | H = R = 24 | H = R = 32 |
|---|---|---|---|---|
| 1 | 0.3046 | 0.3333 | 0.3558 | 0.3800 |
| 2 | 0.1779 | 0.1867 | 0.1750 | 0.1979 |
| 3 | 0.2146 | 0.1245 | 0.1371 | 0.1388 |
| 4 | | 0.1132 | 0.1204 | 0.1187 |
| 6 | | | 0.0829 | 0.0954 |
| 8 | | | 0.0950 | 0.0850 |
| 10 | | | | 0.0779 |
| 12 | | | | 0.0925 |

Panel B: Multiplier bootstrap using `csdid`

| J = L | H = R = 8 | H = R = 16 | H = R = 24 | H = R = 32 |
|---|---|---|---|---|
| 1 | 0.3079 | 0.3458 | 0.3725 | 0.3950 |
| 2 | 0.1808 | 0.1887 | 0.1762 | 0.2004 |
| 3 | 0.2125 | 0.1233 | 0.1408 | 0.1388 |
| 4 | | 0.1161 | 0.1200 | 0.1204 |
| 6 | | | 0.0846 | 0.0967 |
| 8 | | | 0.0929 | 0.0842 |
| 10 | | | | 0.0787 |
| 12 | | | | 0.0938 |

Panel C: Cluster jackknife using `csdidjack`

| J = L | H = R = 8 | H = R = 16 | H = R = 24 | H = R = 32 |
|---|---|---|---|---|
| 1 | 0.0938 | 0.1279 | 0.1450 | 0.1642 |
| 2 | 0.0379 | 0.0763 | 0.0796 | 0.0929 |
| 3 | 0.0533 | 0.0488 | 0.0600 | 0.0688 |
| 4 | | 0.0505 | 0.0621 | 0.0571 |
| 6 | | | 0.0450 | 0.0571 |
| 8 | | | 0.0425 | 0.0483 |
| 10 | | | | 0.0450 |
| 12 | | | | 0.0500 |

**Table 3 (p. 16): "Rejection frequencies for models without covariates"**

Panel A: Asymptotic (RIF) using `csdid`

| J = L | H = R = 8 | H = R = 16 | H = R = 24 | H = R = 32 |
|---|---|---|---|---|
| 1 | 0.3150 | 0.3383 | 0.3683 | 0.3725 |
| 2 | 0.1717 | 0.1937 | 0.1813 | 0.2042 |
| 3 | 0.2233 | 0.1279 | 0.1392 | 0.1504 |
| 4 | | 0.1144 | 0.1225 | 0.1208 |
| 6 | | | 0.0892 | 0.0975 |
| 8 | | | 0.0929 | 0.0821 |
| 10 | | | | 0.0763 |
| 12 | | | | 0.0921 |

Panel B: Multiplier bootstrap using `csdid`

| J = L | H = R = 8 | H = R = 16 | H = R = 24 | H = R = 32 |
|---|---|---|---|---|
| 1 | 0.3171 | 0.3492 | 0.3808 | 0.3879 |
| 2 | 0.1750 | 0.1954 | 0.1821 | 0.2079 |
| 3 | 0.2208 | 0.1308 | 0.1396 | 0.1483 |
| 4 | | 0.1165 | 0.1258 | 0.1192 |
| 6 | | | 0.0887 | 0.0983 |
| 8 | | | 0.0938 | 0.0829 |
| 10 | | | | 0.0804 |
| 12 | | | | 0.0950 |

Panel C: Cluster jackknife using `csdidjack`

| J = L | H = R = 8 | H = R = 16 | H = R = 24 | H = R = 32 |
|---|---|---|---|---|
| 1 | 0.0942 | 0.1279 | 0.1475 | 0.1550 |
| 2 | 0.0358 | 0.0733 | 0.0775 | 0.1013 |
| 3 | 0.0571 | 0.0505 | 0.0646 | 0.0679 |
| 4 | | 0.0538 | 0.0662 | 0.0658 |
| 6 | | | 0.0487 | 0.0546 |
| 8 | | | 0.0467 | 0.0475 |
| 10 | | | | 0.0442 |
| 12 | | | | 0.0575 |

### The authors' reading (p. 14-16)

- "The results for the two procedures proposed in Callaway and Sant'Anna (2021) in Panels A and B generally resemble each other greatly, while those for our cluster-jackknife procedure in Panel C are always much closer to the nominal level of 0.05."
- "For every method, there are serious inferential problems when there are few treated clusters. The top row of each panel, where J = L = 1, always contains the largest rejection frequencies. The asymptotic and bootstrap procedures reject between 30% and 40% of the time. In contrast, the jackknife procedure never rejects much more than 16% of the time."
- "The jackknife is always reliable when J = L is large enough. Interestingly, the number of treated clusters that are needed for the jackknife to be reliable depends on the total number of clusters." (p. 14)
- Asymptotic versus bootstrap: "At most, they differ by about 2% (0.3333 vs 0.3458) ... It is clear that the multiplier bootstrap does not improve inference much at all ... For that reason, we will not focus on the multiplier bootstrap in the empirical examples." (p. 14)
- Table 3: "These results are broadly similar to the results with covariates. ... The jackknife tests generally provide much more reliable inferences, but they can still over-reject substantially when there are few treated clusters." (p. 14-16)
- The four takeaways, verbatim (p. 15-16):
  1. "The CSDID estimator with asymptotic or multiplier bootstrap standard errors is prone to over-reject, often very severely, when there are few clusters."
  2. "The CSDID estimator with asymptotic or multiplier bootstrap standard errors is especially prone to over-reject when there are few treated clusters."
  3. "The asymptotic and multiplier bootstrap standard errors proposed in Callaway and Sant'Anna (2021) are generally extremely similar for inference about the ATT, as the discussion in Section 3.2 suggests."
  4. "Inference based on cluster-jackknife standard errors is always much better than inference based on asymptotic or multiplier bootstrap standard errors. However, there can still be noticeable over-rejection when the number of treated clusters is small relative to the total number of clusters."

### Where the jackknife still over-rejects (Panel C of Tables 2 and 3)

| J = L (treated clusters) | Covariates | H = 8 | H = 16 | H = 24 | H = 32 |
|---|---|---|---|---|---|
| 1 (2 treated) | with | 0.0938 | 0.1279 | 0.1450 | 0.1642 |
| 1 (2 treated) | without | 0.0942 | 0.1279 | 0.1475 | 0.1550 |
| 2 (4 treated) | with | 0.0379 | 0.0763 | 0.0796 | 0.0929 |
| 2 (4 treated) | without | 0.0358 | 0.0733 | 0.0775 | 0.1013 |
| 3 (6 treated) | with | 0.0533 | 0.0488 | 0.0600 | 0.0688 |
| 3 (6 treated) | without | 0.0571 | 0.0505 | 0.0646 | 0.0679 |
| 4 (8 treated) | with | | 0.0505 | 0.0621 | 0.0571 |
| 4 (8 treated) | without | | 0.0538 | 0.0662 | 0.0658 |

Reviewer observations on these numbers (not statements in the paper unless quoted):

- With `J = L = 1`, jackknife over-rejection increases with `H`: from 0.0938 to 0.1642 with covariates and from 0.0942 to 0.1550 without. This is also the case the paper flags (p. 12) as one where "there are not multiple regions in each cohort", so deleting a treated cluster removes an entire cohort from that jackknife sample. The paper does not attribute the `J = L = 1` over-rejection specifically to this.
- With `J = L = 2`, the jackknife under-rejects at `H = 8` (the half-treated case `H = 4J`) and over-rejects for `H >= 16`, reaching 0.0929 (with covariates) and 0.1013 (without) at `H = 32`.
- With `J = L >= 6`, every jackknife cell lies between 0.0425 and 0.0575.
- The largest jackknife rejection frequency in either table is 0.1642 (Table 2, `J = L = 1`, `H = 32`).
- Reviewer arithmetic: with 2,400 replications the simulation standard error of a rejection frequency near 0.05 is about 0.0044 (`sqrt(0.05 * 0.95 / 2400)`). The paper reports no simulation standard errors.
- Reviewer arithmetic: the text's "At most ... about 2% (0.3333 vs 0.3458)" is a gap of 0.0125; the largest Panel A versus Panel B gap in Table 2 is 0.0167 (`J = L = 1`, `H = 24`: 0.3558 vs 0.3725), still "about 2%".

---

## Empirical Examples (Section 7)

Framing (p. 16-17): "two brief empirical examples to demonstrate that there can be meaningful differences between inferences based on asymptotic and cluster-jackknife standard errors. Both examples use American data clustered by state. The first example uses individual-level data, and the second uses county-level data."

### Prenatal substance-use policies and mental health (Section 7.1, p. 17)

- Data: Behavioral Risk Factor Surveillance System, 2005-2018; "women aged 18-44 with at least one child, resulting in a sample of 440,446 observations from 34 U.S. states."
- Outcome: "the number of days in the past month that a respondent reports having good mental health."
- Treatment: "Thirteen states adopted punitive prenatal substance-use policies during the sample period. ... The never-treated group consists of 21 states that never adopted such policies during the sample period."
- Covariates: age, race, education, and the log of income.
- Specification: "the CSDID estimator with simple aggregation and clustered at the state level, using the csdid package"; jackknife via `csdidjack`.
- Cluster balance: "The sample is relatively unbalanced across the 34 states, with the largest states having nearly 5 times as many observations as the smallest ones."

Table 4 (p. 17): "Estimates of ATT for good mental health days"

| Method | ATT | Std. error | P value | 95% CI lower | 95% CI upper |
|---|---|---|---|---|---|
| CSDID asymptotic | -0.212 | 0.102 | 0.038 | -0.412 | -0.011 |
| Cluster jackknife (CV_3) | -0.212 | 0.113 | 0.069 | -0.441 | 0.017 |

- Paper's reading: "this standard approach yields a statistically significant estimate at the 5% level. However, the cluster jackknife (using csdidjack) leads to a larger standard error and a P value greater than 0.05."
- Reviewer arithmetic: jackknife SE / asymptotic SE = 0.113 / 0.102 = 1.11.

### Minimum wage and teen employment (Section 7.2, p. 17-19)

- "an empirical example adapted from Callaway and Sant'Anna (2021)", using county-level Quarterly Workforce Indicators data for 2001-2007.
- Sample statement: "We focus on 29 states that changed their minimum wage during the sample period."
- Outcome: "the log of teen employment (lemp)".
- Covariates: "region indicators, percent white, percent with a high school diploma, poverty rate, and both county population and median income as quadratics."
- Specification: "Estimation is performed using the csdid command with simple aggregation, clustered at the state level."

Table 5 (p. 18): "Effect of minimum wage on teen employment"

| Panel | Method | ATT | Std. error | P value | CI lower | CI upper |
|---|---|---|---|---|---|---|
| A: Original estimation sample | CSDID asymptotic | -0.0490 | 0.0171 | 0.0040 | -0.0825 | -0.0154 |
| A: Original estimation sample | Cluster jackknife (CV_3) | -0.0490 | 0.1960 | 0.8045 | -0.4504 | 0.3525 |
| B: Excluding Indiana | CSDID asymptotic | 0.1476 | 0.1385 | 0.2860 | -0.1238 | 0.4190 |
| B: Excluding Indiana | Cluster jackknife (CV_3) | 0.1476 | 0.5159 | 0.7770 | -0.9108 | 1.2060 |

- Panel A, paper's reading: "The ATT is -0.0490, with an asymptotic standard error of 0.0171 (P = 0.004), suggesting a statistically very significant reduction in teen employment. In contrast, the cluster-jackknife standard error is 0.1960 (P = 0.8045)."
- Influence analysis (p. 18): "we investigate the influence of each state on the aggregate effect. In Figure 1, we plot these estimates against the number of observations in the state/cluster. ... It is evident from this figure that one control state is an extreme outlier. The outlier state is ID 6 (Indiana). Although Indiana has an intermediate number of observations, dropping it results in a subsample ATT estimate of 0.1476, whereas all other leave-one-out ATT estimates are negative and close in magnitude to the full sample estimate."
- Panel B (p. 18): "We therefore drop Indiana and repeat the analysis with the remaining 28 states. ... With Indiana excluded, the ATT is 0.1476. The asymptotic standard error increases to 0.1385, and the jackknife standard error increases to 0.5159; in both cases, the coefficient is no longer statistically significant."
- Figure 1 (p. 19), "ATT^{(h)} vs. cluster size for minimum wage example": scatter of the omit-one-cluster ATT against the number of observations per state, treated states in blue and control states in red. Read approximately off the plot: one control point near x = 600, y = 0.147; all other points between roughly -0.06 and -0.03.
- Reviewer arithmetic: jackknife SE / asymptotic SE = 0.1960 / 0.0171 = 11.5 in Panel A and 0.5159 / 0.1385 = 3.7 in Panel B.
- Reviewer consistency check: the Panel B full-sample ATT (0.1476) equals the Panel A leave-one-out estimate with Indiana omitted (0.1476), as it must.

---

## Software (Section 8)

Only what the paper states. "We provide open-source software implementations of the cluster jackknife (CV_3) inference procedure for both Stata and R users." (p. 18)

### Stata: `csdidjack` (Section 8.1, p. 19)

- "The csdidjack package is a post-estimation command for use with the csdid command (Rios-Avila, Callaway, and Sant'Anna, 2021) and with the built-in Stata command hdidregress."
- "The package supports the agg(simple), agg(group), and agg(calendar) aggregation schemes." Event-study aggregation is not in this list.
- Install, as printed (including the line continuation shown in the PDF):

```
net install csdidjack, \\\
from("https://raw.githubusercontent.com/liu-yunhan/csdidjack/main/") replace
```

- Usage: "Once a model has been estimated using csdid or hdidregress and estat aggregation, users can run:"

```
csdidjack
```

- Output: "This command returns CV_3 standard errors, t-statistics, P values, and confidence intervals."
- The paper also says the package makes "the full vector of omit-one-cluster estimates" available for CSDID estimation of the ATT (p. 18).
- "Source code is available at https://github.com/liu-yunhan/csdidjack."
- Arguments, options, defaults, how the original command is recovered and re-invoked per subsample, how the cluster variable is identified, handling of failed subsamples, stored results: Not discussed in paper (the command is shown with no arguments).
- `csdid` options the paper mentions in passing: `cluster` (default clustered RIF inference, p. 13), `wboot` (multiplier bootstrap, p. 13), `notyet` (not-yet-treated controls, p. 8), `gvar` (cohort variable, "an observation-level variable that is constant within region", p. 8), `agg(simple)` / `agg(group)` / `agg(calendar)` (p. 19).

### R: `didjack` (Section 8.2, p. 19-20)

- "The didjack package provides CV_3 jackknife inference for users of the did package by Callaway and Sant'Anna in R."
- Install:

```
remotes::install_github("liu-yunhan/didjack")
```

- Usage example, verbatim (p. 20):

```
library(did)
library(didjack)
gt  <- att_gt(yname = "log_wage_dm", tname = "year", gname = "first_treat",
data = df, clustervars = "state", panel = FALSE,
est_method = "reg", bstrap = FALSE)
ag  <- aggte(gt)
res <- didjack(ag)
summary(res)
```

- "Full documentation and source code are available at https://github.com/liu-yunhan/didjack."
- What the example fixes: `didjack()` takes the object returned by `aggte()`, itself built from `att_gt()`; the example uses `clustervars = "state"`, repeated cross-sections (`panel = FALSE`), `est_method = "reg"`, `bstrap = FALSE`, and `aggte(gt)` with no `type` argument; a `summary()` method exists for the result.
- Other `didjack()` arguments, defaults, supported aggregation types, return structure, error handling: Not discussed in paper.
- CRAN / SSC availability: Not discussed in paper. A Python implementation of the jackknife: Not discussed in paper (the paper only notes that `csdid` exists as a Python package, p. 5).

---

## Implementation Notes (audit notes - NOT registry-candidate)

### Data Structure Requirements

- Observations nested in clusters, with a cluster identifier. In the paper the cluster is the region / state, which is also the level at which treatment timing is assigned (p. 3, 13).
- A first-treatment-period variable that is constant within region (`S_r`, with 0 for never-treated; p. 3, 8) and absorbing treatment.
- Equation (7) is written for a fully-balanced panel. The paper's simulations use individual-level CPS data and its R example passes `panel = FALSE`; the estimator formula for that case is not written in the paper (p. 6-7, 20).
- For every ATT(g,t) to be estimable in every delete-one subsample, the paper's own contrast between Example 1 and Example 3 (p. 12) points to more than one cluster in each treated cohort and more than one cluster in the comparison group. The paper states this by example, not as a formal condition.
- The paper's evidence covers `H` from 8 to 32 in simulations and `H = 34`, `29`, `28` in the examples.

### Computational Considerations

- Cost is `H + 1` full CSDID estimations: one on the full sample and one per deleted cluster (p. 12). Each replication re-estimates every ATT(g,t) and the aggregate.
- "the jackknife procedure can be slow" (p. 12); "the simulations are very slow, particularly for the jackknife" (p. 14). No timing numbers are reported.
- No closed-form, influence-function-based, or algebraic leave-one-out shortcut is given. Any shortcut in a library implementation would be a library design choice that must be shown to reproduce full re-estimation.
- The paper's only stated speed-up idea concerns its own simulations: "We hope to speed them up in the future using a methodology similar to the one in the fastdid package in R. This package is up to 100x faster than the did package in R, but it requires panel data rather than cross-sectional data." (p. 14)
- Reviewer inference: with unit-level clustering `H` would equal the number of units, so the `H + 1` cost statement would apply with `H` that large. The paper says nothing about this case.
- Equation (20) involves no random draws, so the jackknife itself needs no seed or replication count.

### Tuning Parameters

The paper's jackknife has no tuning parameters beyond the choice of cluster variable: no bandwidth, no replication count, no auxiliary distribution, no seed.

| Parameter | Type | Default | Source | Selection Method |
|-----------|------|---------|--------|-----------------|
| cluster variable | column identifying clusters | none stated; region / state in all of the paper's evidence | Paper-derived (defines `H` and the deletion units in (20)) | Chosen by the analyst; the paper only studies clusters equal to the treatment-assignment level |
| significance level | float | 5% nominal level and 95% intervals in all reported results | Paper-reported setting, not a tuning parameter of (20) | Analyst choice |
| reference distribution / degrees of freedom | - | Not discussed in paper | Not paper-derived | Must be an explicit, documented library choice (see *Reference distribution*) |

### Relation to Existing diff-diff Estimators

Library facts below are limited to the current `CallawaySantAnna` surface and the `## CallawaySantAnna` REGISTRY section. Each point of contact is listed as an open item for an implementation plan; none is resolved here.

- **Current constructor and fit surface.** `CallawaySantAnna.__init__` parameters include `control_group` (`"never_treated"` default, or `"not_yet_treated"`), `estimation_method` (default `"dr"`), `alpha`, `cluster`, `n_bootstrap` (default 0), `bootstrap_weights` (default resolves to `"rademacher"`), `seed`, `base_period` (default `"varying"`), `cband`, `panel`, and `vcov_type` (default `"hc1"`, currently the only accepted value). The fit signature is `fit(data, outcome, unit, time, first_treat, covariates=None, survey_design=None)`.
- **Current inference paths.** Per REGISTRY, default standard errors are analytical and influence-function based; with `cluster=X` the standard error is cluster-robust CR1 (Liang-Zeger) on the influence function; `n_bootstrap > 0` selects the multiplier bootstrap. These are the same two kinds of procedure the paper reviews in Sections 3.1 and 3.2. The paper's simulation evidence is for the Stata `csdid` implementations of them; whether the library's finite-sample conventions on these paths (for example the reference distribution on its clustered path) match those is not audited here. Results carry `cluster_name` and `n_clusters`; `H` in equation (20) is a cluster count.
- **No full-re-estimation cluster jackknife today.** `CallawaySantAnna` has no inference path that implements equation (20), i.e. deleting each cluster, re-estimating every ATT(g,t), and re-aggregating. The library's survey machinery does accept jackknife replicate weights (`replicate_method` values include `"JK1"` and `"JKn"` in `diff_diff/survey.py`), and `CallawaySantAnna` accepts replicate-weight survey designs; how that replicate-weight variance relates to the paper's full-re-estimation jackknife has not been examined in this review and is an open item for the implementation plan. Wild cluster bootstrap inference is exposed as `inference="wild_bootstrap"` on `DifferenceInDifferences` / `TwoWayFixedEffects`, and the library also has an LWDiD-facing `wild_cluster_bootstrap` utility (`diff_diff/lwdid_wild_bootstrap.py`); neither applies to `CallawaySantAnna`.
- **Simple-aggregation weights differ between the paper's description and the library - must be resolved by the implementation plan.** Per REGISTRY, the library's simple aggregation has the weighted-sum form of equation (8) with the weights `w_{g,t}` set by group size, and its aggregation standard errors include the weight influence function. The paper describes simple aggregation as "equal weighting", "simply the average of the ATT-hat(g,t)", "i.e. unweighted averaging" (p. 7, 13), and gives no cohort-size-based weight formula. Whether the paper's wording is exact or a simplification of what its wrapped software computes cannot be determined from the paper. An implementation plan must decide which overall ATT the jackknife targets and state the weights explicitly. This review does not resolve it.
- **Subsample weights.** The paper's procedure re-forms the aggregate in each delete-one subsample from whatever ATT(g,t) terms are estimable there (caveat 2, p. 12). How that interacts with the library's group-size weights and with its weight-influence-function term is not addressed by the paper and is a plan decision.
- **Parameters whose non-default or default values fall outside the paper's evidence.** Each is a library design choice if supported under a jackknife:
  - `control_group="not_yet_treated"`: jackknife behavior not discussed in paper.
  - `estimation_method` (default `"dr"`): the paper gives no covariate-adjusted formula; the method behind its covariate simulations is not stated and its R example uses `est_method = "reg"`.
  - `base_period` (default `"varying"`): the paper uses base period `g - 1` and post-treatment `t >= g` only; pre-treatment cells are outside what it covers.
  - `cband`: the paper defines no jackknife covariance matrix and no uniform bands.
  - `panel`: the paper's simulation evidence is on repeated cross-sections; whether the county-level example is run as a panel is not stated.
  - `survey_design`: survey weights or survey designs under the jackknife: Not discussed in paper.
  - `cluster` at the unit level, or at any level other than treatment assignment: Not discussed in paper.
- **Public spelling is a plan decision, not made here.** The library already uses the spelling `variance_method="jackknife"` on `SyntheticDiD`, and `vcov_type` is the library-wide argument for choosing a variance estimator. The library has a standing constraint to reuse existing method / argument / result-field names, so the public spelling for a CSDID cluster jackknife should be chosen from existing conventions in the implementation plan.

---

## Gaps and Uncertainties

1. **Reference distribution / degrees of freedom for the jackknife (p. 10, 11, 13, 19).** Section 4 never states it. `t(H - 1)` appears only in Section 3.1 as what "Almost all empirical work" assumes for linear-regression cluster-robust t-statistics. The jackknife P values and CIs in Tables 4 and 5 are numerically consistent with `t(H - 1)` (reviewer arithmetic, from rounded numbers); the paper does not say so. It would have to be confirmed from the package sources, for which the paper pins no version or commit.
2. **Subsample in which no ATT(g,t) is estimable (p. 12).** The paper notes that "no ATT terms can be calculated when region C is dropped" in Example 1 but gives no rule: no detection criterion, and no statement on whether the jackknife SE is then undefined, whether cluster `h` is skipped, or whether `H` and the `(H - 1)/H` factor are adjusted.
3. **Subsample in which only some ATT(g,t) are estimable (p. 12).** The subsample ATT is then "based solely on" / "a weighted average of" the surviving terms and still enters the jackknife, but there is no explicit weight formula for the renormalization and no threshold at which the practice becomes invalid; only "care must be taken to assess whether the jackknife is a sensible tool". Whether the software warns is not stated.
4. **Aggregation weights - definition (p. 7, 13).** The default "simple" weighting is described as "equal weighting", "simply the average of the ATT-hat(g,t)", "i.e. unweighted averaging". No cohort-size-based weight formula appears. Whether this description is exact or a simplification of what the wrapped software applies is not determinable from the paper.
5. **Aggregation weights - treatment in each subsample (p. 11-12).** No explicit statement on whether weights are recomputed from subsample cohort sizes or held at full-sample values. Re-running the estimation command per subsample and caveat 2 imply recomputation (reviewer inference); for group / calendar aggregation this is never spelled out.
6. **Scope beyond the overall simple ATT (p. 11, 19).** Equation (20) is scalar and for the overall ATT only. Individual ATT(g,t), per-group or per-calendar-period aggregates, and event-study coefficients have no formulas and no simulation evidence. The Stata package is said to support group and calendar aggregation; event-study is not listed. Whether (20) is applied element-wise is not stated, nor what happens when a parameter is undefined in some subsamples but the overall ATT is not.
7. **Simultaneous / uniform bands and joint tests.** Not discussed. No jackknife covariance matrix is defined.
8. **Centering (p. 11).** The paper centers on the full-sample estimate. It does not mention centering on the mean of the delete-one estimates or when the two differ (reviewer note: they can differ here in particular because subsample estimand composition can change, per caveat 2).
9. **Covariate-adjusted estimators (p. 6, 13, 20).** Only the no-covariate estimator (7) is given. Which estimation method underlies the "with covariates" simulations is not stated; the R example uses `est_method = "reg"`. Covariate-specific subsample failures are not discussed.
10. **Panel vs repeated cross-sections (p. 6-7, 14, 20).** Equation (7) is for a fully-balanced panel; the "more complicated equation" for repeated cross-sections or attrition is not given. The simulations use individual-level CPS data (the remark that `fastdid` "requires panel data rather than cross-sectional data" and the R example's `panel = FALSE` indicate repeated cross-sections), so the reported evidence is for a case whose estimator formula is not written in the paper. Evidence for true panel data with units nested in clusters: only the county-level minimum-wage example, and whether it is run as a panel is not stated.
11. **Cluster level vs treatment-assignment level.** All evidence has cluster = state = level of treatment assignment, with many individuals or counties per cluster. Unit-level clustering, clusters that span cohorts, and clusters nested within treatment groups are not discussed.
12. **Comparison group (p. 6-8, 12).** The jackknife discussion uses never-treated controls in its examples. Behavior under not-yet-treated controls (where deleting a later-treated cluster changes the comparison group of earlier cohorts) and in the all-eventually-treated case (Example 4) is not discussed. The comparison group used in the simulations is not stated explicitly; the only relevant statement is that `csdid` by default uses never-treated units when feasible (p. 7).
13. **Base period / pre-treatment estimates / anticipation (p. 5-6).** The paper uses base period `g - 1` and only post-treatment `t >= g`. Jackknife inference for pre-trend estimates is outside what the paper covers.
14. **Minimum-wage example sample description (p. 17-19).** The text says "We focus on 29 states that changed their minimum wage during the sample period", yet Figure 1 and the text distinguish "control states" from "treated states" within these states, and the outlier Indiana is called a "control state". The treated / control split, cohort structure, comparison group, and number of counties or observations are not reported. The label "ID 6 (Indiana)" is given without explanation of the ID coding.
15. **Simulation details needed for replication (p. 13-14).** Not reported: state sample sizes and size heterogeneity, sampling with vs without replacement, how treated states are selected, bootstrap `B` and weight distribution, estimation method, comparison group, the critical value used for the jackknife test, software versions, seed. Only 2,400 replications; no simulation standard errors are reported.
16. **No DGP (p. 12-13).** The Monte Carlo is a placebo-law design on real CPS MORG data, so there is no parametric DGP, serial-correlation parameter, or cluster-size parameter to port. Reproducing the tables requires that dataset.
17. **Only rejection frequencies under the null (p. 14-16).** No power, coverage, or CI-length results; no results with non-zero or heterogeneous treatment effects; none for unequal `J` and `L`, more than two cohorts, or `T` other than 8.
18. **No theory.** No formal validity result for the cluster jackknife applied to the two-step, aggregated CSDID estimator; the support is simulation evidence and citations to results for other estimators.
19. **Software internals (p. 19-20).** Beyond a minimal invocation for each package and the named outputs of the Stata command, no further arguments, options, defaults, or return-value structure are given for `csdidjack` / `didjack`, and there is no description of how they re-invoke the underlying command, handle subsample failures, or choose the reference distribution. The paper does not pin a version or commit.
20. **Jackknife variant naming (p. 11).** The paper uses only the name `CV_3` and defines it by (20) with full-sample centering. It names no other jackknife variant.
21. **No remedy where the jackknife still over-rejects (p. 2, 10, 16, 20).** The paper recommends no alternative for the few-treated-clusters cases. Other procedures are mentioned only in its literature summary and introduction; none is developed or evaluated for CSDID. The only future work stated is speeding up the simulations (p. 14).
22. **Minor notational inconsistencies in the paper.** (a) Equation (16) uses `Y_{i,g,t}` on the left but `eps_{i,r,t}` on the right (p. 9). (b) The text around (18) describes the ATT vectors as covering "all g = 2, ..., t and t = 2, ..., T" (p. 10), whereas Section 2.2 defines ATT(g,t) for `g in calG`, `g <= t <= T`. (c) The title page is dated February 13, 2026; the arXiv margin stamp reads 12 Feb 2026. (d) Section 2 says "Consider a panel setting" while the data are individuals within regions and the simulations use repeated cross-sections. (e) In Example 1 (p. 5), AB31 is first listed among the comparisons that "do not provide any information" ("AB31 has no within-time variation in treatment") and then among the valid comparisons ("Both AB21 and AB31 are valid comparisons that use information from B before it was treated").
