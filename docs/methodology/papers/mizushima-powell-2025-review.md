# Paper Review: Inference with Modern Difference-in-Differences Methods

**Authors:** Yuji Mizushima, David Powell (both RAND)
**Citation:** Mizushima, Y., & Powell, D. (2025). Inference with Modern Difference-in-Differences Methods. SSRN working paper 5221387 (RAND).
**URL:** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5221387
**Source reviewed:** SSRN 5221387, 32-page PDF (title page and abstract p.1; main text pp.2-25; references pp.26-28; Tables 1-3 pp.29-31; Appendix Table A1 p.32; no figures). This is an unpublished working paper. The PDF was reviewed externally and is not committed to the repository. Reproduce by downloading the working paper via the SSRN link above. Version: the title page prints no date, version label or series number; the year 2025 in the citation comes from the SSRN listing, the latest year in the paper's own reference list is 2024, and the PDF file's embedded creation date is 2025-04-04 (a file property, not printed on any page). All page citations below are PRINTED page numbers, which coincide with the PDF page numbers. Numbered equations: (1) and (2) only, both legible; one further equation (the TWFE specification, p.6) is unnumbered. The four tables are images in the PDF; every cell was legible and nothing is marked unreadable. Transcription conventions: the paper's dashes are rendered as hyphens, typographic quotes as straight quotes, and Greek letters, mathematical symbols and diacritics in ASCII. Greek letters and mathematical symbols inside quoted passages are transliterated to ASCII (for example `tau`, `rho`, `+/-`, `x` for the multiplication sign).
**Review date:** 2026-11-22

> **Personal verification stamp (load-bearing items spot-checked against the source).** Beyond the multi-agent extraction, the reviewer read p.16 and p.29 directly and verified equation (2) symbol for symbol and every cell of Table 1. For Tables 1-3 and Appendix Table A1 (pp.29-32), which are embedded as images, every column's Mean Rejection Rate and RMSE row was recomputed from the transcribed cells and matches the printed summary rows, with the single exception of the JWDID RMSE cell that the paper itself misprints in all four tables. Also checked against the source: the randomization / t-test and wild bootstrap procedure text (pp.16-17); the CPS assignment sentence and the 1000-simulation statement (pp.13, 17); and the heterogeneous-effect construction (pp.13-14). Every quoted passage of 40 or more characters was additionally run through a mechanical word-sequence comparison against the PDF text layer.

---

## Methodology Registry Entry

**Status: proposed text for a future REGISTRY section; no such section exists in `docs/methodology/REGISTRY.md` today; this file is a non-authoritative source audit.** The paper is the candidate source for (1) a MULTI-ESTIMATOR, STAGGERED-ADOPTION, REJECTION-RATE simulation design for a planned "design diagnosis" capability (placebo assignments on a user's own untreated data, with per-estimator performance reported before any real analysis), and (2) a few-treated-unit inference procedure for the imputation estimator. It is reviewed as a set with the companion reviews of Coglianese and Fang (2026), Blair, Cooper, Coppock and Humphreys (2019) and Karim, Nielsen, MacKinnon and Webb (2026). Nothing below is a methodology contract until a REGISTRY section is written and reviewed alongside an implementation.

*This is an INFERENCE-EVALUATION paper, not an estimator paper: it proposes no new point estimator, states no theorem, and prints two numbered equations (one data generating process, one test statistic). The standard registry headings (estimator equation, standard errors) therefore do not apply and are replaced by the two simulation designs, the test and rejection-rate computation, the roster of estimators with their paired inference procedures, and the two inference procedures the paper brings to this setting. The registry-candidate text ends at the marker before `## Results`; everything below that marker is audit material (transcribed tables, quotes, implementation notes, gaps) and is NOT part of the proposed registry text.*

## Placebo-assignment rejection-rate simulation for staggered-adoption estimators (working title)

*The heading above is provisional. The paper's own terms are "placebo policies" / "CPS-based simulations", "rejection rates", "imputation bootstrap" and "t-test/randomization test"; the name of any library surface is to be decided in the implementation plan.*

**Primary source:** [Mizushima, Y., & Powell, D. (2025). Inference with Modern Difference-in-Differences Methods. SSRN working paper 5221387 (RAND).](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5221387)

**Key implementation requirements:**

*Scope of the source:*
- "This paper examines the finite-sample performance of seven leading DiD estimators under various simulated conditions." (abstract, p.1)
- What is measured: the rejection rate of a nominal 5% test of a zero average treatment effect when the true average effect is zero. Every table is headed "Rejection Rates for Average Treatment Effect Estimate" (pp.29-32). Nothing else is evaluated - see "What the paper does NOT provide" below.
- Two exercises: "First, we generate 'placebo' policy adoptions in real data from the Current Population Survey (CPS). Second, we design our own data generating process and provide results with and without treatment heterogeneity." (p.12)
- Setting: staggered adoption of an absorbing binary treatment assigned at the unit (state) level, with a never-treated group always present, no covariates ("Because there are no covariates", p.15; "this paper does not focus on issues related to the inclusion of covariates", p.7).
- Design range: 5, 10 or 25 treated units out of 30, 50, 51 or 100 total units; annual data for 1978-2019 (CPS exercise) or 10 or 30 periods (synthetic exercise).
- Stated purpose: "We provide these results to inform the credibility of the inference associated with each estimator with less emphasis on comparing across different estimators." (p.6)

*Rationale (Sections 1-2, pp.2-12):*
- "The literature on these new estimators typically focuses on their ability to accurately recover treatment effects. The reliability of their standard errors remains less understood, particularly in settings where the number of treated units is limited. Many of the new methods rely on asymptotic arguments that assume a sufficiently large number of both treated and untreated units. In practice, policy adoption can be concentrated in a small subset of units." (p.3)
- "When the number of treated units is small, cluster-robust variance estimators can produce severe biases." (p.3)
- Why two exercises: "First, we generate placebo policies using state-level unemployment rates from the Current Population Survey (CPS), following the approach in Bertrand et al. (2004). Second, to help isolate specific factors that might lead to over- or under-rejection, we simulate panel data under controlled conditions, varying the number of time periods, total units, and proportion of treated units. We compare scenarios with and without treatment heterogeneity." (pp.3-4)
- "The motivation of this paper is to test the rejection rates of the methods used by researchers." (p.14)
- The two competing biases (p.5): "First, as documented in the literature with TWFE, proper inference often requires a large number of both treated and untreated units. When there are a limited number of treated units, many inference methods tend to over-reject. Second, with treatment heterogeneity, the true value of any unit-specific estimate is unknown and must be estimated using values from other units (e.g., the average estimate by time of adoption). The resulting variance estimate reflects both noise and treatment heterogeneity. The inclusion of treatment heterogeneity inflates the estimated variance and should reduce rejection rates, leading to conservative inference. There is little reason to believe that these competing biases will equilibrate."
- Positioning against the concurrent simulation studies summarized in Section 2.3 (pp.10-11; not transcribed here): "In contrast to this concurrent literature, we place more emphasis on understanding how the total number of units affects proper inference, and we vary the number of treated units to more of an extreme given our interest in applications with few treated units. We also evaluate inference for stacked differences-in-differences, which has not been included in simulations in the literature." (p.11)

*Simulation design A - CPS placebo policies on real outcome data (Section 3.1, p.13):*

The paper prints no numbered algorithm. The steps below are distilled from the prose; each step cites the page it comes from.

1. **Outcome data (real).** "Following methods inspired by Bertrand et al. (2004), we take publicly available monthly CPS data for 1978-2019 and construct annual unemployment rates for adults aged 25-54 by state and year." (p.13). Outcome: annual unemployment rate. Unit: state. Sample: "the 50 states plus Washington D.C." (51 units). Periods: annual, 1978-2019 (reviewer arithmetic: 42 periods; the paper does not print the count).
2. **Number of units.** Either all 51 units, or a 30-unit subsample: "We also perform simulations in which we randomly select (without replacement) 30 units to test the implications of having fewer total units." (p.13). Whether the 30-unit subsample is redrawn in every simulation is not stated.
3. **Which units are treated.** "we randomly assign treatment to a subset of 5, 10, or 25 states" (p.13). Selection without replacement is implied by "subset"; the phrase "without replacement" is printed only for the 30-unit subsample. The selection probabilities are not stated.
4. **Adoption timing (staggered).** "with start years selected randomly between 1980 and 2019" (p.13). Each treated state receives a start year; cohorts are whatever groups of states happen to share a start year. The paper confirms that cohorts are not controlled: "our simulations do not guarantee that there is more than one unit per cohort" (p.20). The sampling distribution over years, whether the endpoints are inclusive, and whether draws are independent across states are not stated.
5. **Treatment path.** "States are deemed 'treated' starting in that year, while other states remain never-treated. Treatment is considered absorbing." (p.13)
6. **No effect is injected.** No treatment effect is added to the outcome; only a treatment indicator is assigned. The paper calls the assignments "placebo policies" (p.3) and "'placebo' policy adoptions in real data" (p.12), computes "rejection rates (Type I error) under a true null hypothesis" (pp.12-13), and states that "the CPS simulations impose treatment homogeneity" (p.20). It does not print a sentence saying that the outcome is left unmodified; that is the reviewer's reading of these statements.
7. **Estimate and test.** "For each set of simulations, we implement each DiD estimator, discussed below, using available Stata commands and packages. Additionally, we explore inference methods that pair imputation with the wild bootstrap and t-test/randomization inference. For each simulation, we record whether the 95% confidence interval excludes zero, thereby calculating rejection rates (Type I error) under a true null hypothesis." (pp.12-13)
8. **Replications and reporting.** "We do not vary sample length in these simulations. We perform 1000 simulations for each combination of total units and treated units, and then report the share of simulations in which the null hypothesis was rejected." (p.13)

Design cells: 2 total-unit values x 3 treated-unit values = 6 cells (Table 1).

*Simulation design B - synthetic panel data (Section 3.2, pp.13-14):*

1. **Outcome data (generated).** "The baseline model can be written as" (p.13):

   ```
   y_st = alpha_s + t + beta_s * D_st + eps_st                         (1)

   alpha_s ~ U(0,1)
   eps_st  = 0.8 * eps_{s,t-1} + omega_st,     omega_st ~ N(0,1)
   ```

   "D_st is the policy variable equal to 1 for treated states after treatment" (p.13). The linear term `t` is common to all units. The initial condition of the AR(1) error is not stated.
2. **Treatment effect, case 1 - homogeneous.** "Homogeneous Treatment Effects: beta_s = 0 for all s." (p.14). The homogeneous case is a zero effect for every unit: "Table 2 summarizes rejection rates under the assumption that all treated units have the same (zero) treatment effect." (p.21), and Table 2's note reads "These simulations assume no treatment effect for any unit." No non-zero homogeneous effect is simulated.
3. **Treatment effect, case 2 - heterogeneous across units, zero on average.** "Heterogeneous Treatment Effects: We draw a unit-specific treatment effect from a standard normal distribution but constrain the average treatment effect to equal 0 in each simulation. Note that we only have treatment heterogeneity at the unit level." (p.14). The effect enters equation (1) additively as `beta_s * D_st`: it is constant within a unit over all of its treated periods, does not vary with time since adoption or with cohort except through the unit draw, and is not proportional to the outcome. How the zero-average constraint is imposed (for example, by demeaning the draws) and whether the average is taken over treated units or over treated unit-periods are not stated.
4. **Design parameters.** "In all scenarios, we systematically vary: Number of Treated Units: 5, 10, or 25. Total Number of Units: 30, 50, or 100. Number of Time Periods: 10 or 30." (p.14)
5. **Adoption timing.** Not discussed in Section 3.2. That adoption is staggered with uncontrolled cohort sizes is implied by the appendix variant ("We forced each adopting unit to adopt at the same time as at least one other unit", Table A1 note, p.32) and by p.22 ("we implement the same simulations but require that each cohort have at least two units"). The range and distribution of adoption periods in the synthetic panels are not stated.
6. **Estimate, test, replicate.** "We again implement all DiD methods and report the fraction of simulations (out of 1,000) in which the true null of zero average effect is rejected." (p.14)

Design cells: 3 x 3 x 2 = 18 cells per effect case (Tables 2 and 3), plus the 18-cell cohort-size variant below (Table A1).

*Design variants and parameter values (complete list):*

| Exhibit | Outcome data | Total units | Treated units | Periods | Effect | Cohort rule | Cells | Simulations per cell |
|---|---|---|---|---|---|---|---|---|
| Table 1 (p.29) | Real: CPS state unemployment rate, 1978-2019 | 30, 51 | 5, 10, 25 | 1978-2019 (fixed) | None injected | None | 6 | 1000 |
| Table 2 (p.30) | Generated, equation (1) | 30, 50, 100 | 5, 10, 25 | 10, 30 | `beta_s = 0` for all s | None | 18 | 1000 |
| Table 3 (p.31) | Generated, equation (1) | 30, 50, 100 | 5, 10, 25 | 10, 30 | `beta_s` standard normal, average constrained to 0 | None | 18 | 1000 |
| Table A1 (p.32) | Generated, equation (1) | 30, 50, 100 | 5, 10, 25 | 10, 30 | As Table 3 | Each adopting unit adopts "at the same time as at least one other unit" | 18 | 1000 |

Every design has a true average effect of zero. There is no design with a common adoption date, with a non-zero average effect, with effects that vary over time since adoption, or with covariates.

*Test and rejection-rate computation (pp.12-13, 17-18):*
- Null hypothesis: zero average treatment effect ("the true null of zero average effect", p.14).
- Level: 5%. "For each estimator, we report the fraction of simulations in which the null hypothesis is rejected at the 5% significance level." (pp.17-18)
- Decision rule: for the packaged estimators, "whether the 95% confidence interval excludes zero" (pp.12-13); for the two procedures the paper adds, a p-value from "the placement of t" among placebo statistics, or t critical values (see below).
- Rejection rate: "the share of simulations in which the null hypothesis was rejected" (p.13), out of 1000 per design cell.

  **Reviewer reconstruction (NOT printed in the paper).** With `r = 1..1000` indexing simulations in design cell `c` and procedure `m`:

  ```
  RejRate_{m,c} = (1/1000) * sum_r 1{ procedure m rejects H0 at the 5% level in simulation r of cell c }
  ```

- Whether every procedure is applied to the same simulated assignments (a paired comparison) is not stated; the only relevant sentence is "For each set of simulations, we implement each DiD estimator" (p.12).
- Failed or non-estimable simulations, and how they would enter the denominator: Not discussed in paper.
- Monte Carlo uncertainty of the reported rates: Not discussed in paper.

*Summary metrics across design cells (p.18; table notes pp.29-32):*

> "To further summarize the results, we report the average rejection rate across all simulations for each approach. We also report the mean squared error relative to the expected rejection rate of 0.05. While both of these metrics obscure important variation in behavior - such as which approaches perform better when the number of treated units is small - they provide useful summary metrics given the large number of results reported in this paper." (p.18)

Each table note states: "RMSE reports the root mean squared difference relative to 0.05."

**Reviewer reconstruction (NOT printed in the paper).** With `C` the number of design cells in a table (6 or 18):

```
MeanRejRate_m = (1/C) * sum_c RejRate_{m,c}
RMSE_m        = sqrt( (1/C) * sum_c ( RejRate_{m,c} - 0.05 )^2 )
```

Reviewer check: these two formulas reproduce every printed "Mean Rejection Rate" and "RMSE (relative to 0.05)" value in all four tables to three decimals, with one exception (the JWDID RMSE cell; see Gaps and Uncertainties). This RMSE is a summary of REJECTION RATES across design cells. It is not an RMSE of point estimates.

*Estimators and paired inference procedures (Section 2.2, pp.8-10; Section 3.3, pp.14-17; list on p.18):*

The paper's list of reported methods (p.18) has ten entries, and the tables have twelve columns because two entries carry two inference variants each: "1. TWFE 2. Callaway and Sant'Anna (Asymptotic and Wild Bootstrap) 3. De Chaisemartin and d'Haultfoeuille (DCDH) 4. Stacked DiD 5. Sun and Abraham 6. Wooldridge (JWDID) 7. Borusyak et al. (Asymptotic and Small) 8. Gardner's 2SDID 9. Imputation Bootstrap 10. t-test/Randomization Test". The paper never enumerates which seven are the "seven leading DiD estimators" of the abstract. Reviewer reading: entries 2-8 (seven), with TWFE as the traditional benchmark and entries 9-10 as the two added inference procedures.

Software: "We rely on publicly available Stata packages for each new DiD estimator. These packages often provide several options. Where options were ambiguous, we used the recommended default parameters for each package." (p.14)

| Table column | Estimator and citation as the paper gives it | Stata command (citation) | Estimand / aggregation tested | Specification choices stated | Inference as run |
|---|---|---|---|---|---|
| TWFE | "a traditional two-way fixed effects estimator" (p.4); no citation; `y_st = alpha_s + gamma_t + beta * D_st + eps_st` (p.6, unnumbered) | Not named | Coefficient `beta` on the treatment indicator | Unit and time fixed effects, treatment indicator (p.6) | "TWFE with cluster-robust standard errors" (p.16). Clustering level, small-sample correction and reference distribution not stated |
| Callaway-Sant'Anna: Asymptotic | Callaway and Sant'Anna (2021); "a doubly-robust DiD approach" building on Sant'Anna and Zhao (2020) (p.8, footnote 4) | `csdid` (Rios-Avila et al., 2023) | Average treatment effect (table header); the aggregation scheme used to form it is not stated | "Because there are no covariates, the command automatically uses the 'reg' option. We used the 'notyet' option so counterfactuals are produced using both never-treated and not-yet-treated units." (p.15). Base period not stated | "The first provides 'asymptotic' standard errors" (p.15), from "an asymptotic linear representation of the estimator" that "can be consistently estimated" (p.8) |
| Callaway-Sant'Anna: Wild Bootstrap | Same | `csdid`, second inference option | Same | Same | "the second uses a wild bootstrap and should have better properties when the number of treated units is small" (p.15). Background (p.8): "Callaway and Sant'Anna (2021) propose perturbing the representation by Mammen (1993) weights to simulate the distribution." The weight distribution actually used in the runs, the number of bootstrap draws and the clustering level are not stated |
| DCDH | de Chaisemartin and d'Haultfoeuille (2020, 2024) | `did_multiplegt_dyn` (de Chaisemartin et al., 2023) | Average treatment effect (table header); which output of the command is tested is not stated | "comparing outcome changes in treated units to outcome changes in not-yet-treated units, and then aggregates those estimates. The baseline period is the period immediately prior to treatment." (p.8) | "Variance estimation requires a plugin for the true value of the treatment effect for each unit. Since this does not exist, the code constructs a variance estimate by plugging in the average estimate by 'cohort' using cohorts with more than 1 unit (de Chaisemartin et al., 2024)." (p.15) |
| Stacked DiD | "implemented in Deshpande and Li (2019), Cengiz et al. (2019), and others" (p.9) | Not named ("we implemented a stacked difference-in-differences method", p.16) | Single treatment coefficient: "The treatment dummy is allowed to have a common effect across all stacks." (p.9) | "constructing a separate sample for each 'cohort' and its comparison groups. These samples are stacked and unit\*stack and time\*stack interactions are included in the specification." Comparisons are "to appropriate units, such as never-treated units" (pp.9-10). Event window, stack weights and the exact comparison set not stated | "traditional cluster-robust standard errors (clustering by unit)" (p.16); "clusters are defined as units (not unit\*stacks)" (p.10) |
| Sun-Abraham | Sun and Abraham (2021) | `eventstudyinteract` (Sun, 2022), then `lincom` | "We calculate the implied average treatment effect by weighting each estimate by the number of observations in each time-relative-to-treatment cell." (p.15) | "interacting the time-relative-to-treatment indicators with cohort indicators" (p.8). Control cohort, reference period and set of leads and lags not stated | "We use lincom to perform inference on the average effect." (p.15). Described as one of the methods that "use traditional cluster-robust variance estimators" (p.19) |
| JWDID | Wooldridge (2024) | `jwdid` (Rios-Avila et al., 2024) | "The package interacts the treatment dummy with cohort indicators and then aggregates the estimates." (p.16). Aggregation weights not stated | "a Mundlak-style regression" with a first stage "estimated using only untreated observations" that "includes cohort (based on time of adoption) fixed effects, time fixed effects" (p.9). Command options not stated | "Inference is based on a traditional cluster-robust variance estimator, similar to Sun and Abraham (2021)." (p.16) |
| Borusyak et al.: Asymptotic | Borusyak et al. (2024) | `did_imputation` (Borusyak, 2023) | Average treatment effect (table header) | "We forced the command to provide coefficients and standard errors even when the number of treated observations was small through the option 'minn(0).'" (p.15) | Package default: "The default is to average treatment effects by cohort for the variance estimation" (pp.15-16). Section 2.2: "They propose using the overall average effect or cohort-specific estimates as the plug-in values. This approach leads to conservative inference." (p.9) |
| Borusyak et al.: Small | Same | `did_imputation` | Same | Same | "We provide separate results using the leave-one-out approach, which is recommended for cases with a small number of cohorts, paired with 'averageeffectsby(D)' (where 'D' represents the treatment indicator)." This option "uses all treated observations and is more appropriate when the number of treated units is small" (pp.15-16) |
| Gardner | Gardner (2022), "two-stage DiD (2SDID)" (p.9) | `did2s` (Butts, 2023) | Average treatment effect (table header) | "first estimates unit- and time-level components (and covariates, if present) to predict counterfactual outcomes for treated units" (p.9) | "The inference procedure adjusts for the two-stage process in a GMM framework." (p.15) |
| Imputation Bootstrap | New to this setting: "pair the imputation approach with a wild bootstrap" (p.4) | Not named | Test statistic of equation (2) | Imputation first step, then unit-specific effect estimates | Wild bootstrap with the null imposed and Rademacher weights independent across units; see the procedure below |
| Randomization Test | New to this setting: Ibragimov and Muller (2010) t-test with 5 treated units, Canay et al. (2017) randomization test otherwise | Not named | Test statistic of equation (2) | Same | Sign-flip randomization of unit-specific estimates, or the t-distribution with `q - 1` degrees of freedom; see the procedure below |

Points that apply across the roster:
- **Labels.** The text says of the two `csdid` variants "We refer to these as CS 'asymptotic' and 'small,' respectively" (p.15), but the tables label them "Asymptotic" and "Wild Bootstrap". "We refer to these estimates as the BJS 'asymptotic' and 'small,' respectively" (p.16) matches the table labels.
- **Forced output.** "In some cases (e.g., the Borusyak et al. (2024) estimator with few treated units), we forced the code to output estimates and standard errors despite the authors' caution that inference may be unreliable in such contexts. Our interest is in the consequences of making similar decisions in applied work, regardless of whether the original paper recommends use of the method in those contexts." (p.14)
- **Coinciding point estimates.** For Callaway and Sant'Anna: "In the absence of covariates, the point estimates generated from this method coincide with Sun and Abraham (2021)." (p.8). For `jwdid`: "Without covariates, the point estimates are identical to those generated by Sun and Abraham (2021) and Callaway and Sant'Anna (2021)." (p.16)
- **Baseline period.** "Some estimators use the period before adoption as the baseline while other use the full period prior to treatment." (p.10)
- **TWFE under heterogeneity.** "When the simulations include treatment heterogeneity, the TWFE results are likely less comparable to the others since the point estimates will be biased." (p.16)
- **Block bootstrap (not reported).** Footnote 8 (p.15): "An early version of this paper used did_multiplegt (de Chaisemartin et al., 2019), which implements a block bootstrap. This technique tended to over-reject, though we do not provide the results here."
- **Not stated for any packaged estimator:** package or Stata version, the full command line, the reference distribution and degrees of freedom behind the 95% confidence interval, and any small-sample correction to the cluster-robust variance. Footnote 2 (p.3) notes that modifications to the traditional cluster-robust variance estimator "have also been proposed"; none is evaluated.

*Inference procedures new to this setting (Section 2.4, pp.11-12; Section 3.3, pp.16-17):*

"To assess alternative inference procedures, we also implemented the following methods which are new to this setting. Each method uses an imputation approach and then applies an existing inference method" (p.16). Both use the same test statistic, the paper's equation (2) (p.16):

```
t = | sqrt(q) * [ (1/q) * sum_{i in Treated} beta_hat_i ]
              / sqrt( (1/(q-1)) * sum_{i in Treated} ( beta_hat_i - beta_hat )^2 ) |          (2)
```

"where beta_hat represents the average treatment effect and q is the number of treated units" (p.16). `beta_hat_i` is a unit-specific treatment estimate: "Following the first-step of the imputation approach, the residuals are then used to construct q unit-specific treatment estimates (beta_hat_i), the average of the difference between the observed and predicted counterfactual values, for each treated unit." (p.16)

**Procedure 1 - Randomization/t-statistic Test (pp.16-17).** As printed, in order:

1. Run the first step of the imputation approach and form the `q` unit-specific estimates `beta_hat_i` as each treated unit's average of observed minus predicted counterfactual outcomes (p.16).
2. Compute `t` by equation (2) (p.16).
3. If the number of treated units is 10 or more ("we use the t-test approach when there are 5 treated units and the randomization test when there are 10 or more", p.12): "We then follow the Canay et al. (2017) approach and perturb the unit-specific estimates with independent Rademacher weights, equal to 1 with a 50% chance and -1 otherwise. The test statistic is re-calculated with these perturbed estimates. The placement of t among these placebo test statistics defines a p-value." (pp.16-17)
4. Ties: "Canay et al. (2017) break ties at random, but we do not, given our skepticism that researchers would do so in practice. As such, inference will be conservative." (p.17)
5. If the number of treated units is 5: "we do not perturb the unit-specific estimate since we would only be able to generate 16 unique placebo values (Webb, 2023). Instead, we apply Ibragimov and Muller (2010) and refer to the t-distribution with q - 1 degrees of freedom for significance thresholds." (p.17)

Not stated: the number of perturbation draws (or whether all sign patterns are enumerated), and the exact p-value formula. Reviewer arithmetic: with `q = 5` there are `2^5 = 32` sign patterns, and because equation (2) takes an absolute value each pattern and its negation give the same statistic, leaving 16 distinct values, which matches the paper's count.

**Procedure 2 - Wild Bootstrap on the imputation estimator (p.17).** This is the paper's headline recommendation. The paper prints it as one paragraph with no equations of its own; the only equation it references is (2). The sentences, in order:

1. "We apply a wild bootstrap procedure to the residuals in the imputation framework,"
2. "imposing the null of no treatment effect for all treated units"
3. "and perturbing the residuals using Rademacher weights (which are independent across units)."
4. "This generates placebo outcomes and the imputation approach is then applied to these outcomes, generating placebo treatment effect estimates."
5. "The test statistic is the same as expressed in equation (2) above."
6. "The wild bootstrap generates placebo test statistics, and a p-value can be calculated using the placement of t in the distribution of the placebos."

**Reviewer reconstruction (NOT printed in the paper; bracketed items are choices the paper leaves open).**

```
Inputs: panel y_st, treatment indicator D_st, q treated units.

(a) Imputation first step: fit the untreated-outcome model and obtain residuals e_st.
    [The model is not written out for this procedure; Section 2.2 describes the first step as
     estimating "unit- and time-level components". Which observations the residuals are taken
     over, and how the null is imposed on treated observations, are not stated.]
(b) Compute the unit-specific estimates beta_hat_i and the statistic t by equation (2).
(c) For b = 1..B:                                   [B is not stated]
      draw w_s in {+1, -1} with probability 1/2 each, independently across units s
                                                    [one weight per unit is the reviewer's reading of
                                                     "independent across units"; not stated explicitly]
      form placebo outcomes y*_st from the null-imposed fit and w_s * e_st
                                                    [the formula is not printed]
      re-apply the imputation approach to y*_st, form beta*_i, and compute t*_b by equation (2)
(d) p-value = placement of t in the distribution of { t*_b }.
                                                    [formula, tie handling and inequality not stated]
(e) Reject at the 5% level.                         [stated only as the overall 5% level, pp.17-18]
```

Stated properties and caveats of both procedures:
- "We highlight that the randomization (and t-statistic) test and wild bootstrap approaches should provide conservative inference since both assume treatment homogeneity. We study them because we presume that they are less likely to suffer from small-sample issues when there are few treated units." (p.17)
- "We extend the wild bootstrap approach to the imputation DiD method. This method assumes treatment homogeneity and should provide conservative inference otherwise." (p.11)
- "The wild bootstrap gained favor in difference-in-differences settings due to its performance in simulations (Cameron et al., 2008), though it has also been shown to perform poorly when the number of treated units is small (MacKinnon & Webb, 2017, 2018) and may not be theoretically justified (Canay et al., 2021)." (p.11)
- Footnote 3 (p.5), on the wild bootstrap in the TWFE context: "Conceptually, recent work finds that the implied assumptions of this approach are restrictive and unlikely to hold in difference-in-differences settings (Canay et al., 2021)."
- Footnote 7 (p.11): "Webb (2023) highlights that the use of Rademacher weights when the number of treated units is small may not be appropriate since it provides a limited number of placebo values."
- Footnote 9 (p.25): "With a small number of treated units and a large number of untreated units, the wild bootstrap will produce a large set of unique placebo values. However, they are not necessarily independent since the perturbations related to the treated units will dominate."
- On the unit-level t-test and randomization test: "Ibragimov and Muller (2010) and Ibragimov and Muller (2016) show that using the mean and variance of the unit-specific estimates to construct a t-statistic permits conservative inference in certain contexts. Canay et al. (2017) propose a similar 'randomization inference' approach in which the distribution of the t-statistic is simulated by perturbing the unit-specific estimates with Rademacher weights. This approach improves power. These methods also assume treatment homogeneity and will likely provide conservative inference in the presence of treatment heterogeneity." (p.12)
- The paper gives no proof or formal justification for either procedure in the imputation setting. The source of the imputation point estimator is given only as "the 'imputation' approach discussed in Gardner (2022) and Borusyak et al. (2024)" (p.4).

*Edge cases:*
- **Singleton cohorts.** Cohort sizes are not controlled in Tables 1-3: "the DCDH variance estimate plugs-in the cohort-specific average, but our simulations do not guarantee that there is more than one unit per cohort" (p.20). Table A1 repeats the heterogeneous-effect design with every adopting unit sharing its adoption time with at least one other unit; "this substantially drives down rejection rates for the asymptotic Borusyak et al. (2024) and the Sun and Abraham (2021) approaches, suggesting that these may be important conditions to check" (p.22).
- **Few treated observations in `did_imputation`.** Output is forced with "minn(0)"; "The documentation specifically highlights that inference is unreliable when the number of observations is small." (p.15)
- **Five treated units in the randomization test.** Only 16 unique placebo values, so the t-distribution with `q - 1` degrees of freedom is used instead (p.17).
- **Ties in the randomization test** are not broken at random; the authors expect this to be conservative (pp.12, 17).
- **Few never-treated units.** The 30-unit designs with 25 treated units leave 5 never-treated units (reviewer arithmetic). The paper does not discuss this corner separately.
- **Adoption near the panel edges.** In the CPS design a start year of 1980 leaves two pre-treatment years and a start year of 2019 leaves a single treated year (reviewer arithmetic from the stated ranges). No exclusion of early or late start years is described.
- **TWFE with heterogeneous effects** is reported but flagged as less comparable because its point estimates are biased (p.16).
- **Real policy changes in the CPS outcome series, missing data, unbalanced panels, estimation failures, dropped simulations:** Not discussed in paper.

**Reference implementation(s):** Stata. Named commands: `did2s` (Butts, 2023), `csdid` (Rios-Avila et al., 2023), `did_multiplegt_dyn` (de Chaisemartin et al., 2023), `eventstudyinteract` (Sun, 2022), `did_imputation` (Borusyak, 2023), `jwdid` (Rios-Avila et al., 2024), and `lincom`. No command or code is named for TWFE, stacked DiD, the imputation bootstrap or the t-test/randomization test, and no replication code is referenced.

### Paper-derived requirements

*Only items the paper specifies for its own exercise or explicitly recommends to practitioners. These constrain what may be attributed to this source; they do not constrain the API surface.*

- [ ] Placebo treatment is assigned on real outcome data with no effect added (CPS exercise, p.13); the null of zero average effect is true by construction (pp.12-13)
- [ ] Treatment is binary, absorbing, and assigned at the unit level; untreated units remain never-treated (p.13)
- [ ] Adoption is staggered: each treated unit receives a randomly selected start period (CPS: a start year between 1980 and 2019) (p.13)
- [ ] The number of treated units is a design parameter (5, 10, 25) and so is the total number of units (CPS: 30 by random subsampling without replacement, or all 51) (p.13)
- [ ] The estimand tested is the average treatment effect for every estimator (table headers, pp.29-32)
- [ ] The test is at the 5% level; for packaged estimators the decision is whether the 95% confidence interval excludes zero (pp.12-13, 17-18). The paper does not use the word "two-sided"; a confidence-interval decision and the absolute value in equation (2) both imply it (reviewer note)
- [ ] The rejection rate is the share of simulations in which the null is rejected, with 1000 simulations per design cell (pp.13-14)
- [ ] Results are reported per estimator AND per inference variant (two variants each for Callaway-Sant'Anna and Borusyak et al.) (pp.4, 18)
- [ ] Results are reported per design cell, with two cross-cell summaries: the mean rejection rate and the RMSE of rejection rates relative to 0.05 (p.18; table notes)
- [ ] Cohort sizes are a condition to check: singleton cohorts affect some variance estimators (pp.20, 22; Table A1)
- [ ] Effect heterogeneity in the synthetic exercise is unit-level only, additive, standard normal, with the average constrained to zero (p.14)
- [ ] Imputation-based test statistic: equation (2), built from unit-specific averages of observed minus imputed outcomes (p.16)
- [ ] Imputation wild bootstrap: residual perturbation with the null imposed for all treated units, Rademacher weights independent across units, imputation re-applied to each placebo outcome set, p-value from the placement of `t` among placebo statistics (p.17)
- [ ] Unit-level t-test with `q - 1` degrees of freedom when there are 5 treated units; sign-flip randomization test without random tie-breaking when there are 10 or more (pp.12, 16-17)
- [ ] The imputation bootstrap and the randomization test assume treatment homogeneity and are expected to be conservative otherwise (pp.11, 17)
- [ ] Stated to practitioners: the estimator "should be selected based on the plausibility of its assumptions for the specific research context", not "primarily on the behavior of its inference procedure" (p.25)

### What the paper does NOT provide

**The paper evaluates null rejection rates and nothing else.** None of the following appears anywhere in pp.1-32:

- **No power.** Every design has a true average effect of zero; no rejection rate under a non-zero average effect is reported. The word "power" occurs twice, neither time for a result of this paper: once for a study summarized in Section 2.3 ("Their main focus is to consider power, not rejection rates when the null hypothesis is true", p.11) and once for the Canay et al. (2017) approach ("This approach improves power", p.12).
- **No bias or RMSE of point estimates.** No point estimate, mean estimate, bias, standard deviation or RMSE of any estimator is reported. The only RMSE in the paper is the root mean squared difference of rejection rates from 0.05 across design cells. TWFE point estimates are said to be biased under heterogeneity (p.16) without any number.
- **No confidence-interval coverage.** The word "coverage" does not occur. Rejection is recorded from whether the 95% interval excludes zero; coverage is never reported as such.
- **No minimum detectable effect.** Not computed, not mentioned.
- **No cluster jackknife.** The word "jackknife" does not occur. The only leave-one-out construct is the `did_imputation` leave-one-out variance option (pp.9, 15), described no further than quoted above.
- **No comprehensive ordered ranking of estimators and no selection rule.** The paper does make comparative performance judgments within its simulations (the imputation bootstrap is "the most accurate across the entire set of estimators", p.20; "Among the existing estimators, the other estimators perform noticeably worse", p.21), but it disclaims estimator selection on that basis: results are provided "with less emphasis on comparing across different estimators" (p.6), and "We do not recommend selecting a difference-in-differences estimator based primarily on the behavior of its inference procedure." (p.25). The two summary rows permit a reader to compare columns, and the text calls the imputation bootstrap "the most accurate across the entire set of estimators" (p.20), but no complete ordering, threshold or decision rule is given.
- **No recipe for a practitioner to run the exercise on their own data.** The procedural content is the description of the authors' own two exercises in Section 3. The paper does not tell practitioners to repeat the placebo exercise on their data, and gives no guidance on choosing the number of placebo-treated units, the timing distribution or the number of replications for an application.
- **No held-out periods.** No split-sample, pre-period-only or out-of-sample design is used.
- **No software or replication code of its own.** Existing Stata packages are named, but no code, command, package or repository is provided or referenced for the simulations, the stacked DiD implementation, the imputation bootstrap or the t-test/randomization test.
- **No injected non-zero AVERAGE effect.** No design has a non-zero average treatment effect: there is no additive constant effect, no proportional effect, and no effect that varies with time since adoption or by cohort. The CPS exercise injects nothing. The heterogeneous synthetic designs DO inject non-zero unit-specific additive effects (`beta_s * D_st` in equation (1), with `beta_s` drawn from a standard normal), constrained so that their average is zero in each simulation (pp.13-14).
- **No variation in the degree of heterogeneity.** A single distribution (standard normal) is used; the authors note that "simulations with different amounts of treatment heterogeneity will likely produce different results" (p.22).
- **No event-study, pre-trend or cohort-specific tests.** Only the average treatment effect is tested.
- **No covariates and no common-adoption-date design.**
- **No bootstrap or randomization draw counts**, and no weight distribution for the `csdid` wild bootstrap as run.
- **No reference distribution or degrees of freedom** for the packaged estimators' confidence intervals, and no small-sample-corrected cluster-robust variance among the evaluated procedures.
- **No Monte Carlo uncertainty** on any rejection rate, no random seed, and no statement on failed simulations.
- **No formal justification of the imputation wild bootstrap.** The authors state that the wild bootstrap "is not a conceptually valid approach in a traditional two-way fixed effect framework (Canay et al., 2021), and these concerns may apply to its application with imputation approach as well" (pp.24-25).
- **No formula for the wild bootstrap's placebo outcomes or p-value, and no numbered algorithm.**

**Any of these in a library implementation would be a library design choice requiring its own source. It would NOT be derived from this paper and must not be attributed to it.**

---

<!-- ============================================================
     END of registry-candidate block. Everything BELOW this
     marker is audit material (transcribed tables, quotes,
     implementation notes, gaps) and is NOT part of the proposed
     REGISTRY.md text.
     ============================================================ -->

## Results

The paper has four tables and no figures. All four share one layout: a spanning header "Rejection Rates for Average Treatment Effect Estimate", twelve procedure columns, and two summary rows. In the transcriptions below "CS:" stands for the printed group header "Callaway-Sant'Anna" and "BJS:" for the printed group header "Borusyak et al."; all other column labels are as printed. Every value is transcribed from the printed table, not read off a chart.

Exhibit map:

| Exhibit | Page | Data | Effect | Rows |
|---|---|---|---|---|
| Table 1 | p.29 | CPS unemployment rates, placebo treatment | none | total units x treated units (6) |
| Table 2 | p.30 | synthetic, equation (1) | zero for every unit | total units x treated units x periods (18) |
| Table 3 | p.31 | synthetic, equation (1) | unit-level, zero on average | same 18 |
| Appendix Table A1 | p.32 | synthetic, equation (1) | unit-level, zero on average; cohorts of at least two | same 18 |

Reviewer arithmetic for reading the tables (the paper reports no Monte Carlo error): with 1000 simulations a true rejection probability of 0.05 has a binomial standard error of about 0.007, so a single cell between roughly 0.036 and 0.064 is within two standard errors of nominal.

### Table 1 (p.29): "CPS Simulation Results"

| Total Units | Treated Units | TWFE | CS: Asymptotic | CS: Wild Bootstrap | DCDH | Stacked DiD | Sun-Abraham | JWDID | BJS: Asymptotic | BJS: Small | Gardner | Imputation Bootstrap | Randomization Test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30 | 5 | 0.122 | 0.185 | 0.191 | 0.028 | 0.117 | 0.510 | 0.513 | 0.523 | 0.106 | 0.192 | 0.036 | 0.062 |
| 30 | 10 | 0.075 | 0.115 | 0.121 | 0.029 | 0.080 | 0.379 | 0.326 | 0.343 | 0.071 | 0.096 | 0.058 | 0.073 |
| 30 | 25 | 0.047 | 0.066 | 0.067 | 0.024 | 0.068 | 0.058 | 0.093 | 0.154 | 0.061 | 0.070 | 0.068 | 0.130 |
| 51 | 5 | 0.129 | 0.206 | 0.213 | 0.025 | 0.127 | 0.599 | 0.611 | 0.607 | 0.090 | 0.179 | 0.046 | 0.042 |
| 51 | 10 | 0.087 | 0.131 | 0.136 | 0.016 | 0.085 | 0.458 | 0.500 | 0.456 | 0.087 | 0.120 | 0.058 | 0.068 |
| 51 | 25 | 0.045 | 0.069 | 0.069 | 0.017 | 0.056 | 0.175 | 0.265 | 0.241 | 0.053 | 0.065 | 0.054 | 0.086 |
| **Mean Rejection Rate** |  | 0.084 | 0.129 | 0.133 | 0.023 | 0.089 | 0.363 | 0.385 | 0.387 | 0.078 | 0.120 | 0.053 | 0.077 |
| **RMSE (relative to 0.05)** |  | 0.047 | 0.095 | 0.100 | 0.027 | 0.046 | 0.366 | 0.372 | 0.372 | 0.033 | 0.086 | 0.011 | 0.038 |

Table note, verbatim: "Results based on 1000 simulations using unemployment rates generated from the CPS for 1978-2019 with randomly-assigned treatment. RMSE reports the root mean squared difference relative to 0.05. DCDH = de Chaisemartin and d'Haultfoeuille (2024); JWDID = Wooldridge (2024). When the number of treated units is 5, the randomization test is the t-test approach discussed in Ibragimov and Muller (2010). Otherwise, we use Canay et al. (2017)."

**Reviewer reconstruction (not printed): mean rejection rate by number of treated units, Table 1.** Reconstructed means in this review are computed from the printed three-decimal cells and rounded half-up to three decimals.

| Treated units | TWFE | CS: Asymptotic | CS: Wild Bootstrap | DCDH | Stacked DiD | Sun-Abraham | JWDID | BJS: Asymptotic | BJS: Small | Gardner | Imputation Bootstrap | Randomization Test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 (2 cells) | 0.126 | 0.196 | 0.202 | 0.027 | 0.122 | 0.555 | 0.562 | 0.565 | 0.098 | 0.186 | 0.041 | 0.052 |
| 10 (2 cells) | 0.081 | 0.123 | 0.129 | 0.023 | 0.083 | 0.419 | 0.413 | 0.400 | 0.079 | 0.108 | 0.058 | 0.071 |
| 25 (2 cells) | 0.046 | 0.068 | 0.068 | 0.021 | 0.062 | 0.117 | 0.179 | 0.198 | 0.057 | 0.068 | 0.061 | 0.108 |

**Reviewer reconstruction (not printed): mean rejection rate by total units, Table 1.**

| Total units | TWFE | CS: Asymptotic | CS: Wild Bootstrap | DCDH | Stacked DiD | Sun-Abraham | JWDID | BJS: Asymptotic | BJS: Small | Gardner | Imputation Bootstrap | Randomization Test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30 (3 cells) | 0.081 | 0.122 | 0.126 | 0.027 | 0.088 | 0.316 | 0.311 | 0.340 | 0.079 | 0.119 | 0.054 | 0.088 |
| 51 (3 cells) | 0.087 | 0.135 | 0.139 | 0.019 | 0.089 | 0.411 | 0.459 | 0.435 | 0.077 | 0.121 | 0.053 | 0.065 |

### Table 2 (p.30): "Simulation Results with Treatment Effect Homogeneity"

| Total Units | Treated Units | Time Periods | TWFE | CS: Asymptotic | CS: Wild Bootstrap | DCDH | Stacked DiD | Sun-Abraham | JWDID | BJS: Asymptotic | BJS: Small | Gardner | Imputation Bootstrap | Randomization Test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30 | 5 | 10 | 0.117 | 0.181 | 0.180 | 0.069 | 0.117 | 0.285 | 0.468 | 0.390 | 0.099 | 0.166 | 0.047 | 0.039 |
| 30 | 5 | 30 | 0.126 | 0.189 | 0.190 | 0.065 | 0.127 | 0.445 | 0.457 | 0.465 | 0.099 | 0.160 | 0.046 | 0.038 |
| 30 | 10 | 10 | 0.071 | 0.112 | 0.112 | 0.060 | 0.073 | 0.146 | 0.296 | 0.209 | 0.061 | 0.094 | 0.059 | 0.080 |
| 30 | 10 | 30 | 0.064 | 0.105 | 0.105 | 0.034 | 0.063 | 0.260 | 0.322 | 0.330 | 0.066 | 0.102 | 0.042 | 0.064 |
| 30 | 25 | 10 | 0.040 | 0.074 | 0.071 | 0.060 | 0.074 | 0.063 | 0.074 | 0.090 | 0.066 | 0.087 | 0.075 | 0.167 |
| 30 | 25 | 30 | 0.045 | 0.064 | 0.066 | 0.036 | 0.078 | 0.047 | 0.077 | 0.138 | 0.063 | 0.082 | 0.075 | 0.168 |
| 50 | 5 | 10 | 0.116 | 0.188 | 0.194 | 0.061 | 0.119 | 0.332 | 0.585 | 0.441 | 0.086 | 0.158 | 0.051 | 0.031 |
| 50 | 5 | 30 | 0.150 | 0.190 | 0.197 | 0.070 | 0.153 | 0.513 | 0.588 | 0.545 | 0.101 | 0.184 | 0.043 | 0.048 |
| 50 | 10 | 10 | 0.068 | 0.091 | 0.091 | 0.055 | 0.067 | 0.166 | 0.422 | 0.217 | 0.063 | 0.101 | 0.043 | 0.057 |
| 50 | 10 | 30 | 0.089 | 0.122 | 0.118 | 0.041 | 0.084 | 0.352 | 0.463 | 0.389 | 0.070 | 0.097 | 0.049 | 0.059 |
| 50 | 25 | 10 | 0.039 | 0.055 | 0.057 | 0.054 | 0.041 | 0.058 | 0.168 | 0.084 | 0.050 | 0.066 | 0.048 | 0.085 |
| 50 | 25 | 30 | 0.049 | 0.058 | 0.059 | 0.046 | 0.049 | 0.111 | 0.211 | 0.174 | 0.061 | 0.074 | 0.063 | 0.090 |
| 100 | 5 | 10 | 0.129 | 0.182 | 0.188 | 0.067 | 0.133 | 0.334 | 0.700 | 0.525 | 0.099 | 0.168 | 0.036 | 0.030 |
| 100 | 5 | 30 | 0.172 | 0.206 | 0.212 | 0.059 | 0.174 | 0.551 | 0.734 | 0.670 | 0.110 | 0.200 | 0.055 | 0.043 |
| 100 | 10 | 10 | 0.095 | 0.092 | 0.095 | 0.082 | 0.101 | 0.191 | 0.580 | 0.258 | 0.072 | 0.116 | 0.049 | 0.055 |
| 100 | 10 | 30 | 0.094 | 0.115 | 0.114 | 0.036 | 0.098 | 0.431 | 0.614 | 0.475 | 0.073 | 0.109 | 0.046 | 0.054 |
| 100 | 25 | 10 | 0.064 | 0.060 | 0.062 | 0.047 | 0.063 | 0.076 | 0.391 | 0.109 | 0.055 | 0.073 | 0.048 | 0.066 |
| 100 | 25 | 30 | 0.071 | 0.074 | 0.075 | 0.028 | 0.068 | 0.223 | 0.435 | 0.253 | 0.062 | 0.074 | 0.058 | 0.067 |
| **Mean Rejection Rate** |  |  | 0.089 | 0.120 | 0.121 | 0.054 | 0.093 | 0.255 | 0.421 | 0.320 | 0.075 | 0.117 | 0.052 | 0.069 |
| **RMSE (relative to 0.05)** |  |  | 0.054 | 0.088 | 0.090 | 0.015 | 0.056 | 0.258 | 0.319 | 0.319 | 0.031 | 0.079 | 0.011 | 0.043 |

Table note, verbatim: "Results based on 1000 simulations using simulated data discussed in Section 3.2. These simulations assume no treatment effect for any unit. RMSE reports the root mean squared difference relative to 0.05. DCDH = de Chaisemartin and d'Haultfoeuille (2024); JWDID = Wooldridge (2024). When the number of treated units is 5, the randomization test is the t-test approach discussed in Ibragimov and Muller (2010). Otherwise, we use Canay et al. (2017)."

**Reviewer reconstruction (not printed): mean rejection rate by number of treated units, Table 2.**

| Treated units | TWFE | CS: Asymptotic | CS: Wild Bootstrap | DCDH | Stacked DiD | Sun-Abraham | JWDID | BJS: Asymptotic | BJS: Small | Gardner | Imputation Bootstrap | Randomization Test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 (6 cells) | 0.135 | 0.189 | 0.194 | 0.065 | 0.137 | 0.410 | 0.589 | 0.506 | 0.099 | 0.173 | 0.046 | 0.038 |
| 10 (6 cells) | 0.080 | 0.106 | 0.106 | 0.051 | 0.081 | 0.258 | 0.450 | 0.313 | 0.068 | 0.103 | 0.048 | 0.062 |
| 25 (6 cells) | 0.051 | 0.064 | 0.065 | 0.045 | 0.062 | 0.096 | 0.226 | 0.141 | 0.060 | 0.076 | 0.061 | 0.107 |

**Reviewer reconstruction (not printed): mean rejection rate by total units, Table 2.**

| Total units | TWFE | CS: Asymptotic | CS: Wild Bootstrap | DCDH | Stacked DiD | Sun-Abraham | JWDID | BJS: Asymptotic | BJS: Small | Gardner | Imputation Bootstrap | Randomization Test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30 (6 cells) | 0.077 | 0.121 | 0.121 | 0.054 | 0.089 | 0.208 | 0.282 | 0.270 | 0.076 | 0.115 | 0.057 | 0.093 |
| 50 (6 cells) | 0.085 | 0.117 | 0.119 | 0.055 | 0.086 | 0.255 | 0.406 | 0.308 | 0.072 | 0.113 | 0.050 | 0.062 |
| 100 (6 cells) | 0.104 | 0.122 | 0.124 | 0.053 | 0.106 | 0.301 | 0.576 | 0.382 | 0.079 | 0.123 | 0.049 | 0.053 |

**Reviewer reconstruction (not printed): mean rejection rate by number of time periods, Table 2.**

| Time periods | TWFE | CS: Asymptotic | CS: Wild Bootstrap | DCDH | Stacked DiD | Sun-Abraham | JWDID | BJS: Asymptotic | BJS: Small | Gardner | Imputation Bootstrap | Randomization Test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 (9 cells) | 0.082 | 0.115 | 0.117 | 0.062 | 0.088 | 0.183 | 0.409 | 0.258 | 0.072 | 0.114 | 0.051 | 0.068 |
| 30 (9 cells) | 0.096 | 0.125 | 0.126 | 0.046 | 0.099 | 0.326 | 0.433 | 0.382 | 0.078 | 0.120 | 0.053 | 0.070 |

### Table 3 (p.31): "Simulation Results with Treatment Effect Heterogeneity"

| Total Units | Treated Units | Time Periods | TWFE | CS: Asymptotic | CS: Wild Bootstrap | DCDH | Stacked DiD | Sun-Abraham | JWDID | BJS: Asymptotic | BJS: Small | Gardner | Imputation Bootstrap | Randomization Test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30 | 5 | 10 | 0.044 | 0.069 | 0.071 | 0.040 | 0.054 | 0.181 | 0.477 | 0.330 | 0.043 | 0.069 | 0.011 | 0.015 |
| 30 | 5 | 30 | 0.061 | 0.101 | 0.108 | 0.022 | 0.072 | 0.401 | 0.497 | 0.488 | 0.041 | 0.078 | 0.019 | 0.016 |
| 30 | 10 | 10 | 0.014 | 0.035 | 0.035 | 0.016 | 0.020 | 0.058 | 0.262 | 0.097 | 0.012 | 0.017 | 0.010 | 0.013 |
| 30 | 10 | 30 | 0.029 | 0.047 | 0.050 | 0.017 | 0.033 | 0.204 | 0.307 | 0.260 | 0.023 | 0.037 | 0.019 | 0.025 |
| 30 | 25 | 10 | 0.010 | 0.028 | 0.030 | 0.016 | 0.035 | 0.022 | 0.070 | 0.046 | 0.025 | 0.032 | 0.024 | 0.063 |
| 30 | 25 | 30 | 0.018 | 0.036 | 0.035 | 0.007 | 0.067 | 0.034 | 0.077 | 0.101 | 0.026 | 0.033 | 0.031 | 0.067 |
| 50 | 5 | 10 | 0.067 | 0.084 | 0.081 | 0.035 | 0.066 | 0.194 | 0.586 | 0.339 | 0.048 | 0.078 | 0.018 | 0.014 |
| 50 | 5 | 30 | 0.078 | 0.112 | 0.112 | 0.020 | 0.078 | 0.423 | 0.631 | 0.583 | 0.049 | 0.096 | 0.020 | 0.017 |
| 50 | 10 | 10 | 0.025 | 0.037 | 0.039 | 0.019 | 0.030 | 0.093 | 0.437 | 0.149 | 0.032 | 0.043 | 0.019 | 0.022 |
| 50 | 10 | 30 | 0.042 | 0.061 | 0.059 | 0.010 | 0.040 | 0.271 | 0.488 | 0.369 | 0.021 | 0.036 | 0.013 | 0.019 |
| 50 | 25 | 10 | 0.011 | 0.024 | 0.026 | 0.014 | 0.014 | 0.020 | 0.173 | 0.035 | 0.014 | 0.018 | 0.013 | 0.026 |
| 50 | 25 | 30 | 0.020 | 0.026 | 0.026 | 0.009 | 0.024 | 0.056 | 0.208 | 0.119 | 0.017 | 0.021 | 0.014 | 0.021 |
| 100 | 5 | 10 | 0.071 | 0.097 | 0.103 | 0.031 | 0.073 | 0.225 | 0.704 | 0.413 | 0.050 | 0.085 | 0.018 | 0.017 |
| 100 | 5 | 30 | 0.083 | 0.111 | 0.113 | 0.019 | 0.084 | 0.481 | 0.728 | 0.629 | 0.056 | 0.090 | 0.028 | 0.019 |
| 100 | 10 | 10 | 0.034 | 0.040 | 0.041 | 0.015 | 0.034 | 0.102 | 0.601 | 0.160 | 0.022 | 0.040 | 0.012 | 0.014 |
| 100 | 10 | 30 | 0.043 | 0.053 | 0.053 | 0.008 | 0.048 | 0.311 | 0.634 | 0.399 | 0.030 | 0.040 | 0.017 | 0.019 |
| 100 | 25 | 10 | 0.015 | 0.021 | 0.022 | 0.012 | 0.017 | 0.032 | 0.370 | 0.047 | 0.013 | 0.017 | 0.009 | 0.013 |
| 100 | 25 | 30 | 0.017 | 0.026 | 0.026 | 0.008 | 0.018 | 0.127 | 0.406 | 0.127 | 0.012 | 0.017 | 0.012 | 0.014 |
| **Mean Rejection Rate** |  |  | 0.038 | 0.056 | 0.057 | 0.018 | 0.045 | 0.180 | 0.425 | 0.261 | 0.030 | 0.047 | 0.017 | 0.023 |
| **RMSE (relative to 0.05)** |  |  | 0.027 | 0.032 | 0.033 | 0.034 | 0.023 | 0.193 | 0.280 | 0.280 | 0.025 | 0.027 | 0.033 | 0.031 |

Table note, verbatim: "Results based on 1000 simulations using simulated data discussed in Section 3.2. These simulations permit each treated unit to have a unique treatment effect, but the average treatment effect is equal to zero in each simulation. RMSE reports the root mean squared difference relative to 0.05. DCDH = de Chaisemartin and d'Haultfoeuille (2024); JWDID = Wooldridge (2024). When the number of treated units is 5, the randomization test is the t-test approach discussed in Ibragimov and Muller (2010). Otherwise, we use Canay et al. (2017)."

**Reviewer reconstruction (not printed): mean rejection rate by number of treated units, Table 3.**

| Treated units | TWFE | CS: Asymptotic | CS: Wild Bootstrap | DCDH | Stacked DiD | Sun-Abraham | JWDID | BJS: Asymptotic | BJS: Small | Gardner | Imputation Bootstrap | Randomization Test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 (6 cells) | 0.067 | 0.096 | 0.098 | 0.028 | 0.071 | 0.318 | 0.604 | 0.464 | 0.048 | 0.083 | 0.019 | 0.016 |
| 10 (6 cells) | 0.031 | 0.046 | 0.046 | 0.014 | 0.034 | 0.173 | 0.455 | 0.239 | 0.023 | 0.036 | 0.015 | 0.019 |
| 25 (6 cells) | 0.015 | 0.027 | 0.028 | 0.011 | 0.029 | 0.049 | 0.217 | 0.079 | 0.018 | 0.023 | 0.017 | 0.034 |

### Appendix Table A1 (p.32): "Simulation Results with Treatment Effect Heterogeneity + Requiring that Each Treated Unit is Part of a Cohort"

| Total Units | Treated Units | Time Periods | TWFE | CS: Asymptotic | CS: Wild Bootstrap | DCDH | Stacked DiD | Sun-Abraham | JWDID | BJS: Asymptotic | BJS: Small | Gardner | Imputation Bootstrap | Randomization Test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30 | 5 | 10 | 0.032 | 0.044 | 0.043 | 0.030 | 0.041 | 0.051 | 0.423 | 0.074 | 0.030 | 0.048 | 0.018 | 0.014 |
| 30 | 5 | 30 | 0.058 | 0.072 | 0.073 | 0.044 | 0.063 | 0.086 | 0.446 | 0.107 | 0.040 | 0.079 | 0.025 | 0.010 |
| 30 | 10 | 10 | 0.024 | 0.019 | 0.020 | 0.018 | 0.030 | 0.017 | 0.224 | 0.032 | 0.018 | 0.028 | 0.022 | 0.034 |
| 30 | 10 | 30 | 0.020 | 0.046 | 0.048 | 0.015 | 0.022 | 0.044 | 0.276 | 0.038 | 0.025 | 0.030 | 0.019 | 0.025 |
| 30 | 25 | 10 | 0.033 | 0.050 | 0.050 | 0.027 | 0.065 | 0.051 | 0.081 | 0.067 | 0.057 | 0.063 | 0.072 | 0.182 |
| 30 | 25 | 30 | 0.031 | 0.059 | 0.061 | 0.018 | 0.063 | 0.060 | 0.083 | 0.060 | 0.055 | 0.059 | 0.067 | 0.162 |
| 50 | 5 | 10 | 0.050 | 0.060 | 0.063 | 0.047 | 0.054 | 0.083 | 0.532 | 0.089 | 0.025 | 0.059 | 0.011 | 0.009 |
| 50 | 5 | 30 | 0.058 | 0.092 | 0.099 | 0.045 | 0.061 | 0.126 | 0.572 | 0.106 | 0.043 | 0.071 | 0.017 | 0.009 |
| 50 | 10 | 10 | 0.025 | 0.022 | 0.021 | 0.014 | 0.028 | 0.020 | 0.359 | 0.042 | 0.023 | 0.032 | 0.014 | 0.021 |
| 50 | 10 | 30 | 0.029 | 0.046 | 0.046 | 0.017 | 0.033 | 0.044 | 0.411 | 0.038 | 0.022 | 0.032 | 0.017 | 0.020 |
| 50 | 25 | 10 | 0.020 | 0.026 | 0.026 | 0.009 | 0.021 | 0.019 | 0.133 | 0.030 | 0.023 | 0.028 | 0.024 | 0.042 |
| 50 | 25 | 30 | 0.018 | 0.030 | 0.028 | 0.012 | 0.026 | 0.028 | 0.142 | 0.022 | 0.015 | 0.020 | 0.018 | 0.039 |
| 100 | 5 | 10 | 0.057 | 0.079 | 0.080 | 0.039 | 0.058 | 0.092 | 0.657 | 0.102 | 0.036 | 0.065 | 0.013 | 0.009 |
| 100 | 5 | 30 | 0.076 | 0.101 | 0.103 | 0.062 | 0.077 | 0.127 | 0.682 | 0.116 | 0.045 | 0.087 | 0.018 | 0.008 |
| 100 | 10 | 10 | 0.035 | 0.034 | 0.035 | 0.011 | 0.039 | 0.035 | 0.538 | 0.042 | 0.022 | 0.034 | 0.016 | 0.018 |
| 100 | 10 | 30 | 0.033 | 0.048 | 0.050 | 0.013 | 0.035 | 0.056 | 0.566 | 0.043 | 0.020 | 0.034 | 0.014 | 0.015 |
| 100 | 25 | 10 | 0.010 | 0.013 | 0.011 | 0.011 | 0.011 | 0.012 | 0.315 | 0.017 | 0.011 | 0.015 | 0.011 | 0.019 |
| 100 | 25 | 30 | 0.022 | 0.033 | 0.030 | 0.007 | 0.023 | 0.032 | 0.335 | 0.024 | 0.019 | 0.022 | 0.018 | 0.020 |
| **Mean Rejection Rate** |  |  | 0.035 | 0.049 | 0.049 | 0.024 | 0.042 | 0.055 | 0.376 | 0.058 | 0.029 | 0.045 | 0.023 | 0.036 |
| **RMSE (relative to 0.05)** |  |  | 0.023 | 0.024 | 0.026 | 0.030 | 0.020 | 0.035 | 0.033 | 0.033 | 0.024 | 0.022 | 0.032 | 0.051 |

Table note, verbatim: "Results based on 1000 simulations using simulated data discussed in Section 3.2. These simulations permit each treated unit to have a unique treatment effect, but the average treatment effect is equal to zero in each simulation. We forced each adopting unit to adopt at the same time as at least one other unit. RMSE reports the root mean squared difference relative to 0.05. DCDH = de Chaisemartin and d'Haultfoeuille (2024); JWDID = Wooldridge (2024). When the number of treated units is 5, the randomization test is the t-test approach discussed in Ibragimov and Muller (2010). Otherwise, we use Canay et al. (2017)."

**Reviewer reconstruction (not printed): mean rejection rate by number of treated units, Table A1.**

| Treated units | TWFE | CS: Asymptotic | CS: Wild Bootstrap | DCDH | Stacked DiD | Sun-Abraham | JWDID | BJS: Asymptotic | BJS: Small | Gardner | Imputation Bootstrap | Randomization Test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 (6 cells) | 0.055 | 0.075 | 0.077 | 0.045 | 0.059 | 0.094 | 0.552 | 0.099 | 0.037 | 0.068 | 0.017 | 0.010 |
| 10 (6 cells) | 0.028 | 0.036 | 0.037 | 0.015 | 0.031 | 0.036 | 0.396 | 0.039 | 0.022 | 0.032 | 0.017 | 0.022 |
| 25 (6 cells) | 0.022 | 0.035 | 0.034 | 0.014 | 0.035 | 0.034 | 0.182 | 0.037 | 0.030 | 0.035 | 0.035 | 0.077 |

### Summary rows side by side (printed values)

Printed "Mean Rejection Rate" row of each table:

| Table | TWFE | CS: Asymptotic | CS: Wild Bootstrap | DCDH | Stacked DiD | Sun-Abraham | JWDID | BJS: Asymptotic | BJS: Small | Gardner | Imputation Bootstrap | Randomization Test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Table 1 (CPS) | 0.084 | 0.129 | 0.133 | 0.023 | 0.089 | 0.363 | 0.385 | 0.387 | 0.078 | 0.120 | 0.053 | 0.077 |
| Table 2 (homogeneous) | 0.089 | 0.120 | 0.121 | 0.054 | 0.093 | 0.255 | 0.421 | 0.320 | 0.075 | 0.117 | 0.052 | 0.069 |
| Table 3 (heterogeneous) | 0.038 | 0.056 | 0.057 | 0.018 | 0.045 | 0.180 | 0.425 | 0.261 | 0.030 | 0.047 | 0.017 | 0.023 |
| Table A1 (heterogeneous, cohorts of 2+) | 0.035 | 0.049 | 0.049 | 0.024 | 0.042 | 0.055 | 0.376 | 0.058 | 0.029 | 0.045 | 0.023 | 0.036 |

Printed "RMSE (relative to 0.05)" row of each table (the JWDID entries are as printed; see Gaps and Uncertainties):

| Table | TWFE | CS: Asymptotic | CS: Wild Bootstrap | DCDH | Stacked DiD | Sun-Abraham | JWDID | BJS: Asymptotic | BJS: Small | Gardner | Imputation Bootstrap | Randomization Test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Table 1 (CPS) | 0.047 | 0.095 | 0.100 | 0.027 | 0.046 | 0.366 | 0.372 | 0.372 | 0.033 | 0.086 | 0.011 | 0.038 |
| Table 2 (homogeneous) | 0.054 | 0.088 | 0.090 | 0.015 | 0.056 | 0.258 | 0.319 | 0.319 | 0.031 | 0.079 | 0.011 | 0.043 |
| Table 3 (heterogeneous) | 0.027 | 0.032 | 0.033 | 0.034 | 0.023 | 0.193 | 0.280 | 0.280 | 0.025 | 0.027 | 0.033 | 0.031 |
| Table A1 (heterogeneous, cohorts of 2+) | 0.023 | 0.024 | 0.026 | 0.030 | 0.020 | 0.035 | 0.033 | 0.033 | 0.024 | 0.022 | 0.032 | 0.051 |

### Findings: few treated units

Stated by the authors:
- "Most estimators tend to over-reject when the number of treated units is small." (abstract, p.1)
- "many (but not all) new DiD estimators over-reject the null in small-sample settings, particularly when 10 or fewer units adopt treatment. This behavior parallels earlier critiques of cluster-robust inference in DiD." (p.5)
- CPS (p.19): "most methods substantially over-reject in small-sample settings, especially when the number of treated units is very small (i.e., 5 treated states). For instance, Sun and Abraham (2021) and Wooldridge (2024) reject over 50% of the time when there are 5 treated units (for total number of units equal to 30 or 51). They improve as the number of treated units increases."
- CPS (p.19): "Stacked difference-in-differences rejects in 12-13% of simulations with 5 treated units while both Callaway and Sant'Anna (2021) inference methods reject 18+% of the time. The leave-one-out approach for Borusyak et al. (2024) dominates the asymptotic estimator across all simulations, rejecting at 9-10% for the simulations with 5 treated units. Gardner (2022) rejects in 18-19% of simulations with 5 treated units. By contrast, DCDH under-rejects, yielding rejection rates below 3% when the number of treated units is small."
- CPS, 25 treated units (pp.19-20): "When the number of treated units is 25, stacked difference-in-differences, the Borusyak et al. (2024) leave-one-out approach, and both Callaway and Sant'Anna (2021) methods are closer to the expected rejection rate. Other estimators improve as the number of treated units increases but still fare poorly. Even with 25 treated units, some estimators still reject at high rates. DCDH remains the exception - consistently rejecting under 3% of the time and even under 2% when the number of total units is 51 and the number of treated units is 10+."
- CPS overall (p.20): "for the CPS design, DCDH is extremely conservative, while most of the popular new DiD estimators - especially their asymptotic implementations - severely over-reject with limited treated units. Pairing imputation with a wild bootstrap comes closest to nominal rejection rates across the board."
- Synthetic, homogeneous (p.21): "approaches such as the Sun and Abraham (2021) and Wooldridge (2024) methods considerably over-reject with few treated units - reaching rejection rates of 55-70%. All approaches tend to over-reject with only 5 treated units with the exception of DCDH which approaches 5% rejection rates."
- Synthetic, homogeneous, 25 treated units (p.21): "Almost all of the approaches perform better with 25 treated units compared to fewer treated units. The Callaway and Sant'Anna (2021) wild bootstrap version rejects at a rate of 6-7%, stacked difference-in-differences at a rate of 4-8%, and Gardner (2022) at a rate of 6-9%."
- Summary (p.23): "Small Number of Treated Units is Especially Problematic. Nearly every estimator suffers size distortions when only 5-10 treated units are available. Many new DiD methods rely on large-cohort asymptotics and substantially over-reject when there are only a few treated units."

Headline cells with 5 treated units, from the tables (CPS with 30 / 51 total units; synthetic homogeneous range over the six 5-treated cells of Table 2):

| Procedure | Table 1, 30 units | Table 1, 51 units | Table 2, range over 5-treated cells |
|---|---|---|---|
| TWFE | 0.122 | 0.129 | 0.116-0.172 |
| CS: Asymptotic | 0.185 | 0.206 | 0.181-0.206 |
| CS: Wild Bootstrap | 0.191 | 0.213 | 0.180-0.212 |
| DCDH | 0.028 | 0.025 | 0.059-0.070 |
| Stacked DiD | 0.117 | 0.127 | 0.117-0.174 |
| Sun-Abraham | 0.510 | 0.599 | 0.285-0.551 |
| JWDID | 0.513 | 0.611 | 0.457-0.734 |
| BJS: Asymptotic | 0.523 | 0.607 | 0.390-0.670 |
| BJS: Small | 0.106 | 0.090 | 0.086-0.110 |
| Gardner | 0.192 | 0.179 | 0.158-0.200 |
| Imputation Bootstrap | 0.036 | 0.046 | 0.036-0.055 |
| Randomization Test (t-test at 5 treated) | 0.062 | 0.042 | 0.030-0.048 |

Reviewer notes (read directly off the tables, not stated by the paper):
- The two `csdid` columns are nearly identical: they differ by at most 0.007 in every cell of every table, and in the CPS table the wild bootstrap column is at least as high as the asymptotic column in every cell (higher in five of six). The text's expectation that the wild bootstrap "should have better properties when the number of treated units is small" (p.15) is not visible in the tables.
- The imputation bootstrap's largest rejection rates are in the 30-unit, 25-treated designs: 0.068 (Table 1), 0.075 and 0.075 (Table 2), 0.072 and 0.067 (Table A1).
- The randomization test over-rejects most in the same 30-unit, 25-treated designs: 0.130 (Table 1), 0.167 and 0.168 (Table 2), 0.182 and 0.162 (Table A1), against 0.063 and 0.067 in Table 3. The paper does not comment on this corner, which has 5 never-treated units.

### Findings: more untreated units and more time periods

- "Interestingly, increasing the number of total units while holding the number of treated units constant actually increases the rejection rates for both Sun and Abraham (2021) and Wooldridge (2024). Notably, these methods use traditional cluster-robust variance estimators." (p.19)
- "Interestingly, many of the rejection rates worsen (or at least do not improve) as the number of time periods and number of total units increase. As before, Sun and Abraham (2021) and Wooldridge (2024) perform considerably worse as the number of total units increases (holding the number of treated units constant)." (p.21)
- "Third, over-rejection rates are exacerbated for some methods as the number of untreated units increases." (pp.5-6)
- Summary (p.23): "More Time Periods and/or Untreated Units Can Actually Worsen Rejection Rates. Estimators relying on traditional cluster-robust variance estimates are especially prone to this problem."

The reviewer reconstructions under Tables 1 and 2 give the corresponding column means by total units and by number of periods (for example, Table 2 JWDID: 0.282, 0.406, 0.576 at 30, 50, 100 total units; Table 2 Sun-Abraham: 0.183 at 10 periods, 0.326 at 30 periods).

### Findings: treatment-effect heterogeneity

What "induces its own bias" means. The abstract says "Treatment heterogeneity induces its own bias, reducing rejection rates." (p.1). The bias meant is a bias of the INFERENCE (an inflated variance estimate that lowers rejection rates), not a bias of the point estimate. The mechanism as stated (p.5): "with treatment heterogeneity, the true value of any unit-specific estimate is unknown and must be estimated using values from other units (e.g., the average estimate by time of adoption). The resulting variance estimate reflects both noise and treatment heterogeneity. The inclusion of treatment heterogeneity inflates the estimated variance and should reduce rejection rates, leading to conservative inference." In Section 4.2.2 (p.22): "This heterogeneity can induce two competing forces for inference: small samples lead to over-rejection while the added variation from heterogeneous effects can inflate standard errors, sometimes lowering rejection rates below the nominal 5%."

Stated results:
- "Comparing Table 3 to Table 2, rejection rates drop considerably in almost all cases. As a result, many methods' rejection rates actually move closer to 5%. For instance, Gardner (2022), Callaway and Sant'Anna (2021), and stacked DiD now, on average, reject at close to the expect rate while DCDH and the Borusyak et al. (2024) leave-one-out approach tends to under-reject." (p.22)
- "The imputation bootstrap and t-test/randomization test also now under-reject. The comparison with Table 2 is important because the results in Table 3 help isolate the balancing of the two competing biases. There is little reason to believe that these biases will perfectly cancel out, and simulations with different amounts of treatment heterogeneity will likely produce different results." (p.22)
- "We find that treatment heterogeneity does reduce rejection rates; however, most procedures still tend to over-reject in our specific simulations. Notably, as the degree of treatment heterogeneity increases, this over-rejection bias may diminish, and in some cases, the procedures could even under-reject." (p.5)
- Summary (p.23): "Treatment Heterogeneity Reduces Rejection Rates. Given that most approaches over-reject, this property could be seen as beneficial. However, there is little reason to believe that this bias properly cancels out the tendency to over-reject in the presence of few treated units."
- Conclusion (p.24): "As noted in the literature, treatment heterogeneity will lead to conservative inference. Our simulations confirm this as rejection rates consistently decrease across all methods and simulation parameters when treatment effects are permitted to vary across units. There is no reason to believe that this bias will perfectly cancel out the small-sample biases explored in the simulations with treatment homogeneity. Instead, rejection rates are likely a function of the magnitude of the heterogeneity, and it is difficult to predict for a specific application how these biases will balance."

The numbers (printed mean rejection rates, Table 2 to Table 3): TWFE 0.089 to 0.038; CS asymptotic 0.120 to 0.056; CS wild bootstrap 0.121 to 0.057; DCDH 0.054 to 0.018; stacked DiD 0.093 to 0.045; Sun-Abraham 0.255 to 0.180; JWDID 0.421 to 0.425; BJS asymptotic 0.320 to 0.261; BJS small 0.075 to 0.030; Gardner 0.117 to 0.047; imputation bootstrap 0.052 to 0.017; randomization test 0.069 to 0.023.

Reviewer note (cell-by-cell comparison of Table 3 with Table 2, not stated by the paper). The rejection rate falls in all 18 cells for ten of the twelve columns. The two exceptions: JWDID rises in 10 cells, is unchanged in 1 and falls in 7 (its mean rises from 0.421 to 0.425), and BJS asymptotic rises in 2 cells (30 units / 5 treated / 30 periods: 0.465 to 0.488; 50 / 5 / 30: 0.545 to 0.583). The conclusion's "consistently decrease across all methods and simulation parameters" (p.24) is therefore not exact for the JWDID column as printed; Section 4.2.2's "almost all cases" (p.22) is the accurate wording.

Cohort size (Table A1, p.22): "Some of the inference procedures benefit from having multiple units in a 'cohort.' To explore this, we implement the same simulations but require that each cohort have at least two units. Appendix Table A1 provides the results to help understand the extent to which some procedures struggled with the lack of cohorts populated with multiple adopters in the above simulations. Notably, this substantially drives down rejection rates for the asymptotic Borusyak et al. (2024) and the Sun and Abraham (2021) approaches, suggesting that these may be important conditions to check." Printed means, Table 3 to Table A1: Sun-Abraham 0.180 to 0.055; BJS asymptotic 0.261 to 0.058; JWDID 0.425 to 0.376. The cohort-size requirement is applied only to the heterogeneous-effect design; no homogeneous-effect or CPS counterpart is reported.

### Findings: stacked DiD

- Novelty claim: "We also evaluate inference for stacked differences-in-differences, which has not been included in simulations in the literature." (p.11)
- CPS: "Stacked difference-in-differences rejects in 12-13% of simulations with 5 treated units" (p.19); with 25 treated units it is among the methods "closer to the expected rejection rate" (p.19). Table 1: 0.117 / 0.080 / 0.068 with 30 units and 0.127 / 0.085 / 0.056 with 51 units at 5 / 10 / 25 treated; mean 0.089, RMSE 0.046.
- Synthetic, homogeneous: "stacked difference-in-differences at a rate of 4-8%" with 25 treated units (p.21). Table 2 mean 0.093, RMSE 0.056.
- Synthetic, heterogeneous: stacked DiD is among the methods that "now, on average, reject at close to the expect rate" (p.22). Table 3 mean 0.045, RMSE 0.023; Table A1 mean 0.042, RMSE 0.020.
- Reviewer notes (read off the tables): the stacked DiD column tracks the TWFE column closely in every table (means 0.089 vs 0.084, 0.093 vs 0.089, 0.045 vs 0.038, 0.042 vs 0.035), and stacked DiD has the smallest printed RMSE in Table 3 and in Table A1. The paper does not report how the stacks were built (event window, comparison units, weights), so these numbers cannot be tied to a specific stacked specification.

### Findings: DCDH

- CPS: "DCDH is extremely conservative" (p.20), every cell at or below 0.029.
- Synthetic, homogeneous: "we find that the DCDH approach performs well relative to the other existing estimators across all simulations, rejecting at a 5.4% rate on average with an RMSE of 0.015. The Borusyak et al. (2024) leave-one-out approach rejects, on average, at a 7.5% rate across all simulations with an RMSE of 0.031. Among the existing estimators, the other estimators perform noticeably worse." (p.21)
- Conclusion: "There are exceptions, such as the de Chaisemartin et al. (2024) estimator and inference procedure, which tends to under-reject across most simulation parameters." (p.24)

### Findings: the imputation bootstrap and the t-test/randomization test

- "One of the most notable findings in Table 1 is how inference procedures paired with the imputation approach - particularly the imputation bootstrap - tend to perform. In the CPS simulations, the imputation bootstrap yields mean rejection rates very close to 5%, with an RMSE of 0.011 - the most accurate across the entire set of estimators. t-test and randomization inference also perform well with an RMSE of 0.038." (p.20)
- "Among the new approaches studied in this paper, the imputation bootstrap again fares well, rejecting at a 5.2% rate across all simulations with the lowest RMSE of all the approaches. It performs especially well with only 5 treated units, rejecting 4-6% of the time. The t-test/randomization test also performs reasonably well." (p.21)
- Under heterogeneity both under-reject (p.22): Table 3 means 0.017 (imputation bootstrap) and 0.023 (randomization test); every imputation bootstrap cell in Table 3 is at or below 0.031.
- Summary (p.23): "The Imputation Bootstrap Tends to Perform Well. Both in the CPS design and in simulated environments without treatment heterogeneity, pairing an imputation-style estimator with a wild bootstrap yields rejection rates closest to 5%. This echoes prior TWFE studies showing that wild-type bootstrap methods can help mitigate small-sample biases."
- Abstract (p.1): "An imputation-based estimator paired with a wild bootstrap performs well across simulation designs, maintaining rejection rates close to nominal levels when there is treatment homogeneity and providing conservative inference otherwise."

### Power results

None. See "What the paper does NOT provide".

### Text statements checked against the tables (reviewer notes)

| Statement | Page | Table values | Consistent? |
|---|---|---|---|
| Sun-Abraham and Wooldridge "reject over 50% of the time" with 5 treated units, CPS | p.19 | 0.510, 0.599; 0.513, 0.611 | Yes |
| Stacked DiD "12-13%" with 5 treated units, CPS | p.19 | 0.117, 0.127 | Yes |
| Both Callaway-Sant'Anna methods "18+%" with 5 treated units, CPS | p.19 | 0.185, 0.206; 0.191, 0.213 | Yes |
| BJS leave-one-out "9-10%" with 5 treated units, CPS | p.19 | 0.106, 0.090 | Approximately (0.106 rounds to 11%) |
| Gardner "18-19%" with 5 treated units, CPS | p.19 | 0.192, 0.179 | Yes |
| DCDH "below 3%" with few treated units; "under 2%" with 51 units and 10+ treated, CPS | pp.19-20 | all six cells 0.016-0.029; 0.016, 0.017 | Yes |
| Imputation bootstrap RMSE 0.011; t-test/randomization RMSE 0.038, CPS | p.20 | 0.011; 0.038 | Yes |
| Sun-Abraham and Wooldridge "reaching rejection rates of 55-70%", homogeneous | p.21 | Sun-Abraham maximum 0.551; JWDID maximum 0.734 | Approximately (JWDID exceeds 70% in one cell) |
| "All approaches tend to over-reject with only 5 treated units with the exception of DCDH", homogeneous | p.21 | DCDH 0.059-0.070; imputation bootstrap 0.036-0.055; randomization test 0.030-0.048 | Holds for the existing procedures only; the two added procedures do not over-reject with 5 treated units |
| DCDH mean 5.4%, RMSE 0.015; BJS leave-one-out mean 7.5%, RMSE 0.031, homogeneous | p.21 | 0.054, 0.015; 0.075, 0.031 | Yes |
| With 25 treated units: Callaway-Sant'Anna wild bootstrap "6-7%", stacked DiD "4-8%", Gardner "6-9%", homogeneous | p.21 | 0.057-0.075; 0.041-0.078; 0.066-0.087 | Approximately |
| Imputation bootstrap mean 5.2%, lowest RMSE, "4-6%" with 5 treated units, homogeneous | p.21 | 0.052; 0.011; 0.036-0.055 | Yes |
| Rejection rates "consistently decrease across all methods and simulation parameters" under heterogeneity | p.24 | JWDID rises in 10 of 18 cells; BJS asymptotic rises in 2 | No for JWDID (see above) |

---

## Practical Recommendations and Limitations (verbatim)

All quotes verbatim with printed page numbers.

**Recommendations**

1. **Choose the estimator on its assumptions, not on its inference behavior.**
   - "We do not recommend selecting a difference-in-differences estimator based primarily on the behavior of its inference procedure. The estimator should be selected based on the plausibility of its assumptions for the specific research context. However, our results suggest that the accompanying inference procedures often behave poorly in common applied settings." (p.25)
   - "The choice of estimator should primarily depend on its appropriateness for the application and the assumptions required for the estimator." (p.10)
   - "We highlight that the decision to use a specific estimator and inference procedure should depend on several considerations given that the assumptions required for different estimators may hold in specific contexts and not others." (p.6)

2. **Consider alternative inference tools with few treated units.**
   - "These findings delineate the circumstances in which practitioners should adopt alternative inference tools when employing these new DiD methods and motivate the need for new inference methods to accompany the new difference-in-differences estimators." (p.25)
   - "By mapping out the performance of the inference methods for these estimators, our study aims to equip applied researchers with clearer guidance for appropriate inference with policy evaluations that involve modest sample sizes." (p.6)

3. **The imputation estimator with a wild bootstrap, absent heterogeneity.**
   - "Second, an imputation estimator combined with a wild bootstrap maintains rejection rates closest to the nominal 5% in the absence of treatment heterogeneity, even with limited treated units." (pp.5-6)
   - "Without treatment heterogeneity (i.e., in our CPS simulations and our simulated data without treatment heterogeneity), we find that the imputation approach paired with a wild bootstrap fares well, rejecting at close to the expected rate across many settings." (p.24)

4. **Check cohort sizes.**
   - Requiring at least two units per cohort "substantially drives down rejection rates for the asymptotic Borusyak et al. (2024) and the Sun and Abraham (2021) approaches, suggesting that these may be important conditions to check." (p.22)

The paper gives no checklist, decision rule or step-by-step guidance beyond these statements.

**Limitations and caveats stated by the authors**

1. **The wild bootstrap's validity.**
   - "The caveat to this result is that the wild bootstrap is not a conceptually valid approach in a traditional two-way fixed effect framework (Canay et al., 2021), and these concerns may apply to its application with imputation approach as well. Moreover, while we find that the wild bootstrap works well given only five treated units, the concerns discussed in MacKinnon and Webb (2017) about pairing the wild bootstrap with TWFE given a small number of treated units potentially apply too." (pp.24-25)
   - Footnote 9 (p.25), quoted in full under the procedure above.

2. **Results depend on the amount of heterogeneity.**
   - "There is little reason to believe that these biases will perfectly cancel out, and simulations with different amounts of treatment heterogeneity will likely produce different results." (p.22)
   - "Instead, rejection rates are likely a function of the magnitude of the heterogeneity, and it is difficult to predict for a specific application how these biases will balance." (p.24)
   - "Note that we only have treatment heterogeneity at the unit level." (p.14)

3. **Results are specific to the simulations run.**
   - "most procedures still tend to over-reject in our specific simulations" (p.5)
   - "Given the subsequent exercise with simulated data, we highlight that the CPS simulations impose treatment homogeneity, and we will find that the homogeneous treatment effect results in Section 4.2 are more comparable to the CPS results than the heterogeneous treatment effect results in Section 4.3 are." (p.20)

4. **Summary metrics hide variation.**
   - "While both of these metrics obscure important variation in behavior - such as which approaches perform better when the number of treated units is small - they provide useful summary metrics given the large number of results reported in this paper." (p.18)

5. **Packages were pushed beyond their authors' advice.**
   - "we forced the code to output estimates and standard errors despite the authors' caution that inference may be unreliable in such contexts" (p.14)

6. **Singleton cohorts were allowed.**
   - "our simulations do not guarantee that there is more than one unit per cohort" (p.20)

7. **TWFE comparability.**
   - "When the simulations include treatment heterogeneity, the TWFE results are likely less comparable to the others since the point estimates will be biased." (p.16)

8. **Scope of the contribution.**
   - "In this paper, we begin to address these limitations by examining how current approaches fare in typical applied contexts, and by starting to identify existing and promising alternatives better suited to these settings." (p.24)

---

## Citations as Printed for Estimators, Software and Inference Methods (reference list, pp.26-28)

Entries are transcribed in the paper's own form; several carry no venue, volume or version.

*Estimators:*
- Borusyak, K., Jaravel, X., & Spiess, J. (2024). Revisiting event study designs: Robust and efficient estimation. Review of Economic Studies.
- Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences with multiple time periods. Journal of Econometrics, 225(2), 200-230.
- Cengiz, D., Dube, A., Lindner, A., & Zipperer, B. (2019). The effect of minimum wages on low-wage jobs. The Quarterly Journal of Economics, 134(3), 1405-1454.
- de Chaisemartin, C., & d'Haultfoeuille, X. (2020). Two-way fixed effects estimators with heterogeneous treatment effects. American Economic Review, 110(9), 2964-2996.
- de Chaisemartin, C., & d'Haultfoeuille, X. (2024). Difference-in-differences estimators of intertemporal treatment effects. Review of Economics and Statistics, 1-45.
- Deshpande, M., & Li, Y. (2019). Who is screened out? Application costs and the targeting of disability programs. American Economic Journal: Economic Policy, 11(4), 213-248.
- Gardner, J. (2022). Two-stage differences in differences. arXiv preprint arXiv:2207.05943.
- Sant'Anna, P. H., & Zhao, J. (2020). Doubly robust difference-in-differences estimators. Journal of Econometrics, 219(1), 101-122.
- Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects in event studies with heterogeneous treatment effects. Journal of Econometrics, 225(2), 175-199.
- Wooldridge, J. M. (2024). Two-way fixed effects, the two-way mundlak regression, and difference-in-differences estimators. Available at SSRN 3906345. (A file-sharing URL is also printed.)

*Software:*
- Borusyak, K. (2023). DID_IMPUTATION: Stata module to perform treatment effect estimation and pre-trend testing in event studies.
- Butts, K. (2023). Did2s: Stata module to estimate a twfe model using the two-stage difference-in-differences approach.
- de Chaisemartin, C., Ciccia, D., D'Haultfoeuille, X., Knau, F., Malezieux, M., & Sow, D. (2024). Event-Study Estimators and Variance Estimators Computed by the did_multiplegt_dyn Command. Available at SSRN.
- de Chaisemartin, C., d'Haultfoeuille, X., & Guyonvarch, Y. (2019). DID_MULTIPLEGT: Stata module to estimate sharp Difference-in-Difference designs with multiple groups and periods.
- de Chaisemartin, C., D'Haultfoeuille, X., Malezieux, M., & Sow, D. (2023). DID_MULTIPLEGT_DYN: Stata module to estimate event-study Difference-in-Difference (DID) estimators in designs with multiple groups and periods, with a potentially non-binary treatment that may increase or decrease multiple times.
- Rios-Avila, F., Nagengast, A., & Yotov, Y. (2024). JWDID: Stata module to estimate Difference-in-Difference models using Mundlak approach.
- Rios-Avila, F., Sant'Anna, P., & Callaway, B. (2023). CSDID: Stata module for the estimation of Difference-in-Difference models with multiple time periods.
- Sun, L. (2022). EVENTSTUDYINTERACT: Stata module to implement the interaction weighted estimator for an event study.

*Inference methods:*
- Brewer, M., Crossley, T. F., & Joyce, R. (2018). Inference with difference-in-differences revisited. Journal of Econometric Methods, 7(1), 20170005.
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). Bootstrap-based improvements for inference with clustered errors. The review of economics and statistics, 90(3), 414-427.
- Canay, I. A., Romano, J. P., & Shaikh, A. M. (2017). Randomization tests under an approximate symmetry assumption. Econometrica, 85(3), 1013-1030.
- Canay, I. A., Santos, A., & Shaikh, A. M. (2021). The wild bootstrap with a "small" number of "large" clusters. Review of Economics and Statistics, 103(2), 346-363.
- Hansen, L. P. (1982). Large sample properties of generalized method of moments estimators. Econometrica: Journal of the econometric society, 1029-1054.
- Ibragimov, R., & Muller, U. K. (2010). t-Statistic based correlation and heterogeneity robust inference. Journal of Business & Economic Statistics, 28(4), 453-468.
- Ibragimov, R., & Muller, U. K. (2016). Inference with few heterogeneous clusters. Review of Economics and Statistics, 98(1), 83-96.
- MacKinnon, J. G., & Webb, M. D. (2017). Wild bootstrap inference for wildly different cluster sizes. Journal of Applied Econometrics, 32(2), 233-254.
- MacKinnon, J. G., & Webb, M. D. (2018). The wild bootstrap for few (treated) clusters. The Econometrics Journal, 21(2), 114-135.
- Mammen, E. (1993). Bootstrap and wild bootstrap for high dimensional linear models. The Annals of Statistics, 21(1), 255-285.
- Webb, M. D. (2023). Reworking wild bootstrap-based inference for clustered errors. Canadian Journal of Economics/Revue canadienne d'economique, 56(3), 839-858.

---

## Implementation Notes (audit notes - NOT registry-candidate)

### Data Structure Requirements

What the paper's exercises use (not a proposed library contract):
- A panel of units by periods with one outcome and no covariates. CPS exercise: 51 states (or a 30-state random subsample) by annual periods 1978-2019. Synthetic exercise: 30, 50 or 100 units by 10 or 30 periods.
- The CPS outcome is a real series; the paper does not screen it for real policy changes.
- A binary, absorbing treatment indicator assigned by the simulation at the unit level; a never-treated group in every design (at least 5 units, in the 30-unit / 25-treated designs).
- Staggered start periods with uncontrolled cohort sizes (Tables 1-3) or cohorts of at least two units (Table A1).
- The unit is both the level of treatment assignment and the cluster for the cluster-robust procedures where a level is stated (stacked DiD, p.16).
- The two added procedures need unit-specific effect estimates `beta_hat_i` for every treated unit, and at least two treated units for the denominator of equation (2) (reviewer note).
- Balanced vs unbalanced panels, missing observations, sampling weights, repeated cross-sections: Not discussed in paper.

### Computational Considerations

- **Reviewer arithmetic (the paper does not print these totals).** Design cells: 6 (Table 1) + 18 x 3 (Tables 2, 3, A1) = 60. Each cell has 1000 simulations and 12 procedure columns, so the four tables report 720 rejection rates from 60,000 simulated data sets.
- Three of the twelve columns are resampling procedures run inside every simulation (the `csdid` wild bootstrap, the imputation bootstrap, and the randomization test at 10 or more treated units). The number of draws is not stated for any of them.
- The imputation bootstrap re-applies the imputation approach to every placebo outcome set (p.17), so its cost is one imputation fit per bootstrap draw per simulation.
- Reviewer arithmetic: the binomial standard error of a rejection rate estimated from 1000 simulations is about 0.007 at a true rate of 0.05 and about 0.016 at 0.5. The paper attaches no uncertainty to any rate and tests no difference between columns.
- Whether all procedures see the same simulated assignments (a paired comparison) is not stated (see the test and rejection-rate notes above). The 60,000 figure above assumes they do.
- Runtime, parallelization, random seeds, software versions: Not discussed in paper.

### Parameters reported and not reported

| Parameter | Value in the paper | Where |
|---|---|---|
| Simulations per design cell | 1000 | pp.13, 14, 17; table notes |
| Test level | 5% (95% confidence interval) | pp.12-13, 17-18 |
| Treated units | 5, 10, 25 | pp.13, 14 |
| Total units | CPS: 30, 51; synthetic: 30, 50, 100 | pp.13, 14 |
| Periods | CPS: annual 1978-2019; synthetic: 10, 30 | pp.13, 14 |
| CPS start years | "selected randomly between 1980 and 2019" | p.13 |
| Unit effect | `alpha_s ~ U(0,1)` | p.13 |
| Time effect | linear `t` | equation (1), p.13 |
| Error process | AR(1), coefficient 0.8, standard normal innovations | p.13 |
| Heterogeneous effects | unit-specific, standard normal, average constrained to 0 | p.14 |
| Bootstrap / randomization weights for the added procedures | Rademacher (+1 or -1 with probability 50%) | pp.16-17 |
| Randomization vs t-test switch | t-test at 5 treated units, randomization at 10 or more | p.12 |
| t-test degrees of freedom | `q - 1` | p.17 |
| Number of bootstrap draws (`csdid` wild bootstrap, imputation bootstrap) | Not stated | - |
| Number of randomization draws | Not stated | - |
| Weight distribution for the `csdid` wild bootstrap as run | Not stated (Mammen weights are mentioned as the original proposal, p.8) | - |
| Adoption-period distribution, synthetic design | Not stated | - |
| Stacked DiD window and comparison set | Not stated | - |
| Random seed | Not stated | - |

### Relation to Existing diff-diff Estimators

Library facts in this section are limited to class names, inference-related constructor arguments, and the existing simulation result fields. Every correspondence is by cited source or by name; implementation correspondence was not audited.

- **Estimator roster.** The library ships these related DiD estimator classes: `TwoWayFixedEffects`, `CallawaySantAnna`, `SunAbraham`, `ImputationDiD`, `TwoStageDiD`, `StackedDiD`, `ChaisemartinDHaultfoeuille`, `EfficientDiD`, `WooldridgeDiD`, `LPDiD`, and `SyntheticDiD`. Of these, `SyntheticDiD` requires block treatment (all treated units treated at the same time) and does not accept staggered adoption.

  | Paper column | Source the paper attributes it to | Library class (by cited source; implementation correspondence not audited) |
  |---|---|---|
  | TWFE | None cited ("a traditional two-way fixed effects estimator", p.4) | `TwoWayFixedEffects` (matched by estimator name; the paper cites no source) |
  | Callaway-Sant'Anna (Asymptotic, Wild Bootstrap) | Callaway and Sant'Anna (2021) | `CallawaySantAnna` |
  | DCDH | de Chaisemartin and d'Haultfoeuille (2020, 2024) | `ChaisemartinDHaultfoeuille` |
  | Stacked DiD | "implemented in Deshpande and Li (2019), Cengiz et al. (2019), and others" (p.9) | `StackedDiD` (matched by method name only; the paper attributes stacked DiD to applied implementations, not to a methodological source, so a source-paper match cannot be established from this paper) |
  | Sun-Abraham | Sun and Abraham (2021) | `SunAbraham` |
  | JWDID | Wooldridge (2024) | `WooldridgeDiD` |
  | Borusyak et al. (Asymptotic, Small) | Borusyak et al. (2024) | `ImputationDiD` |
  | Gardner | Gardner (2022) | `TwoStageDiD` |
  | Imputation Bootstrap | Point estimator: "the 'imputation' approach discussed in Gardner (2022) and Borusyak et al. (2024)" (p.4). Inference: new in this paper | Point estimator: `ImputationDiD` / `TwoStageDiD`. Inference procedure: open item (see below) |
  | Randomization Test | Ibragimov and Muller (2010); Canay et al. (2017), applied to unit-specific imputation estimates | none |
  | (not evaluated in the paper) | - | `EfficientDiD`, `LPDiD`, `SyntheticDiD` |

- **Inference-related constructor arguments (verified) against the paper's paired inference.**

  | Library class | Inference-related constructor arguments | Inference the paper pairs with the matching estimator |
  |---|---|---|
  | `TwoWayFixedEffects` | `inference="wild_bootstrap"` exists (wild cluster restricted bootstrap) | Cluster-robust standard errors (p.16) |
  | `CallawaySantAnna` | `cluster`, `n_bootstrap`, `bootstrap_weights`, `seed`, `vcov_type` | `csdid` "asymptotic" standard errors; `csdid` wild bootstrap (p.15) |
  | `ChaisemartinDHaultfoeuille` | `cluster`, `n_bootstrap`, `bootstrap_weights`, `seed` | `did_multiplegt_dyn` variance with a cohort-average plug-in (p.15) |
  | `StackedDiD` | `cluster`, `vcov_type` | Cluster-robust, clustering by unit, not unit-by-stack (pp.10, 16) |
  | `SunAbraham` | `cluster`, `n_bootstrap`, `seed`, `vcov_type` | `lincom` on an observation-count-weighted average; cluster-robust (pp.15, 19) |
  | `WooldridgeDiD` | `cluster`, `n_bootstrap`, `bootstrap_weights`, `seed`, `vcov_type` | "traditional cluster-robust variance estimator" (p.16) |
  | `ImputationDiD` | `cluster`, `vcov_type`, `n_bootstrap`, `bootstrap_weights`, `seed` | `did_imputation` default variance ("asymptotic"); leave-one-out with `averageeffectsby(D)` ("small"); the paper's imputation wild bootstrap; the t-test/randomization test |
  | `TwoStageDiD` | `cluster`, `n_bootstrap`, `bootstrap_weights`, `seed`, `vcov_type` | `did2s` GMM inference (p.15) |
  | `EfficientDiD` | `cluster`, `vcov_type`, `n_bootstrap`, `bootstrap_weights`, `seed` | Not evaluated in the paper |
  | `LPDiD` | `cluster` | Not evaluated in the paper |

- **Wild bootstrap today.** `inference="wild_bootstrap"` (wild cluster restricted bootstrap) is a constructor argument of `DifferenceInDifferences` and `TwoWayFixedEffects`; the library also has an LWDiD-facing `wild_cluster_bootstrap` utility (`diff_diff/lwdid_wild_bootstrap.py`). None of `CallawaySantAnna`, `SunAbraham`, `ImputationDiD`, `TwoStageDiD`, `StackedDiD`, `ChaisemartinDHaultfoeuille`, `EfficientDiD`, `WooldridgeDiD` or `LPDiD` has an `inference` constructor argument.
- **Open items for the implementation plan (not checked in this review; neither asserted nor denied):**
  - What kind of bootstrap `n_bootstrap` selects on each staggered estimator.
  - Whether any existing `n_bootstrap` path corresponds to the wild bootstrap procedure this paper recommends for the imputation estimator (null imposed for all treated units, Rademacher weights independent across units, imputation re-applied to each placebo outcome set, the equation (2) statistic).
  - Whether `ImputationDiD` has a counterpart to the `did_imputation` leave-one-out variance with `averageeffectsby(D)` (the paper's "BJS small"), and whether its default variance corresponds to the paper's "BJS asymptotic".
  - Whether `ImputationDiD` / `TwoStageDiD` expose the unit-specific effect estimates `beta_hat_i` that equation (2) needs.
  - Whether the library's default aggregation for each class matches the "average treatment effect" each Stata command was asked for (the paper itself leaves the aggregation unstated for `csdid`, `did_multiplegt_dyn`, `jwdid` and `did2s`, and uses an observation-count-weighted average for `eventstudyinteract`).
  - Whether any library default (comparison group, base period, variance type, reference distribution) matches the Stata defaults behind the paper's numbers. The paper's rejection rates are evidence about those Stata procedures, not about the library's.
- **Existing simulation tools.** `diff_diff/power.py` has `simulate_power`, `simulate_mde`, and `simulate_sample_size`, which by default draw from an internal per-estimator registry of SYNTHETIC data generators and also accept a custom `data_generator` / `data_generator_kwargs` that bypasses the registry. `SimulationPowerResults` carries `power`, `rejection_rate`, `mean_estimate`, `std_estimate`, `mean_se`, `coverage`, `bias`, `rmse`, `true_effect`, `alpha`, `n_simulations`.
- **Quantity-to-field map.** Matching is by concept and name only; whether each field's computation matches the paper's definition was not checked in this review.

  | Paper quantity | Where defined | Existing library field |
  |---|---|---|
  | Rejection rate: share of simulations in which the null is rejected | pp.12-13, 14, 17-18; every table | `rejection_rate` |
  | Nominal level (5%) | pp.12-13, 17-18 | `alpha` |
  | Number of simulations per design cell (1000) | pp.13, 14, 17 | `n_simulations` |
  | True average effect (zero in every design) | pp.12-13, 14 | `true_effect` |
  | Mean rejection rate across design cells | p.18; "Mean Rejection Rate" rows | none |
  | RMSE of rejection rates relative to 0.05 across design cells | p.18; "RMSE (relative to 0.05)" rows | none (the existing field is named `rmse`, but the paper's quantity is a cross-cell summary of rejection rates, not of point estimates; the name collision is an open naming question) |
  | Number of treated units, total units, periods (design-cell labels) | pp.13-14; table row labels | none among the fields listed above |
  | Procedure label (estimator by inference variant) | p.18; table column labels | none among the fields listed above |
  | Unit-specific effect estimates `beta_hat_i` and the statistic `t` | equation (2), p.16 | none |
  | Unit-specific true effects `beta_s` in the heterogeneous design | p.14 | none |
  | Power under a non-zero average effect | not reported | `power` exists; the paper supplies nothing for it |
  | Bias, RMSE, mean and standard deviation of point estimates | not reported | `bias`, `rmse`, `mean_estimate`, `std_estimate` exist; the paper supplies nothing for them |
  | Mean reported standard error | not reported | `mean_se` exists; the paper supplies nothing for it |
  | Confidence-interval coverage | not reported | `coverage` exists; the paper supplies nothing for it |

- **Reviewer note on how this source sits in the set.** The companion reviews found a placebo point-estimate MSE summary on real data for a single treated unit with no inference (Coglianese and Fang 2026), a framework vocabulary on fully simulated data (Blair, Cooper, Coppock and Humphreys 2019), and a cluster jackknife for one estimator's overall ATT (Karim, Nielsen, MacKinnon and Webb 2026). This paper adds what those lack on one axis only: a multi-estimator, staggered-adoption design with placebo assignment on real outcome data (its CPS exercise) and null rejection rates per estimator and inference variant by number of treated units and total units. It supplies size only. It supplies nothing for power, bias, coverage, or minimum detectable effect, no injection of a non-zero average effect (its heterogeneous synthetic designs inject unit-specific effects that average to zero), and no recipe for a user's own data.
- **Reviewer note on the paper's two exercises.** The CPS exercise (placebo assignment on real outcomes) and the synthetic exercise (assumed data generating process) correspond respectively to a placebo-on-the-user's-own-data capability and to the existing synthetic-generator simulation functions.
- **Naming constraint.** The library has a standing constraint to reuse existing method / argument / result-field names wherever a concept already exists, so the "none" rows above are open naming questions and the matched rows are not. The API itself is to be decided in the implementation plan, not here.

---

## Gaps and Uncertainties

**Checked questions (each checked against the text of pp.1-32)**

| Question | Answer | Evidence |
|---|---|---|
| (a) Reports power (rejection under a true non-zero average effect)? | No | Every design has a zero average effect (pp.13-14; table notes); "rejection rates (Type I error) under a true null hypothesis" (pp.12-13) |
| (b) Reports bias or RMSE of point estimates? | No | The only RMSE is of rejection rates relative to 0.05 (table notes); no point estimate is reported |
| (c) Reports confidence-interval coverage? | No | The word does not occur; only rejection is recorded |
| (d) Computes a minimum detectable effect? | No | Not mentioned |
| (e) Uses real outcome data? | Yes, in one of two exercises | CPS state unemployment rates 1978-2019 (p.13; Table 1); the other exercise uses generated data (equation (1); Tables 2, 3, A1) |
| (f) Evaluates a cluster jackknife? | No | The word does not occur |
| (g) Ranks estimators or gives a selection rule? | No complete ordering and no selection rule; it does make comparative judgments (pp.20-21) | "with less emphasis on comparing across different estimators" (p.6); "We do not recommend selecting a difference-in-differences estimator based primarily on the behavior of its inference procedure." (p.25) |
| (h) Gives a recipe for a practitioner's own data? | No | Section 3 describes the authors' exercises only |
| (i) Uses held-out periods? | No | Not mentioned |
| (j) Provides software or replication code? | No | Existing Stata packages are named (pp.15-16); no code is provided or referenced for the simulations or for the two added procedures |

**Version and provenance**
- **No date on the document.** The title page prints no date, version or series number. The year 2025 is from the SSRN listing; the PDF file's embedded creation date is 2025-04-04. A later revision may differ.
- **Unpublished working paper.** No journal, acceptance or peer-review status is printed.
- **An earlier version existed** with different DCDH software (footnote 8, p.15).

**Simulation design**
- **Adoption timing in the synthetic design is never described.** Section 3.2 gives no range or distribution of adoption periods; staggering is inferred from Table A1's note and p.22.
- **CPS start-year distribution** is "selected randomly between 1980 and 2019" (p.13) with no distribution, independence statement or endpoint convention. Reviewer arithmetic, conditional on an assumption the paper does not state: if start years were independent and uniform over the 40 years 1980-2019, all five start years of a 5-treated-unit design would be distinct in about 77% of simulations, so singleton cohorts would be the norm at 5 treated units.
- **Selection of treated units:** probabilities not stated; "without replacement" printed only for the 30-unit subsample.
- **Whether the 30-unit CPS subsample is redrawn per simulation** is not stated.
- **How the zero-average constraint on heterogeneous effects is imposed** is not stated, nor whether the average is over treated units or over treated unit-periods. Reviewer note: under staggered adoption these two averages differ, and the estimators in the roster weight unit-period effects differently, so "the true null of zero average effect" (p.14) need not be exactly true for every estimator's estimand. The paper acknowledges this only for TWFE (p.16).
- **AR(1) initialization** of `eps_st` is not stated.
- **Edge start years** (two pre-periods at 1980, one treated period at 2019 in the CPS design) are not discussed.
- **Real policy variation in the CPS unemployment series** is not discussed.
- **Section cross-reference.** p.20 refers to "the homogeneous treatment effect results in Section 4.2" and "the heterogeneous treatment effect results in Section 4.3"; the printed headings are 4.2.1 (homogeneous), 4.2.2 (heterogeneous) and 4.3 "Summary of Findings".
- **"proportion of treated units"** (p.4) vs the number of treated units actually varied (p.14): wording only.

**Estimators and specifications**
- **"Seven" is never enumerated.** The abstract says "seven leading DiD estimators"; p.4 says "seven new and widely used difference-in-differences methods"; the list on p.18 has ten entries and the tables twelve columns. The reading used in this review (entries 2-8) is the reviewer's.
- **Aggregation to the "average treatment effect"** is stated only for `eventstudyinteract` (observation-count weights across relative-time cells, p.15). It is not stated for `csdid`, `did_multiplegt_dyn`, `jwdid`, `did2s`, or the default `did_imputation` run.
- **Stacked DiD** has no command, no event window, no stack weights and no explicit comparison set; only "such as never-treated units" (pp.9-10).
- **TWFE** has no command and no stated clustering level.
- **Command lines, package versions and Stata version** are not given for any estimator.
- **Reference distribution and degrees of freedom** behind each package's 95% confidence interval are not stated.
- **`minn(0)`**: whether it is applied to both `did_imputation` variants is not stated explicitly.
- **Label mismatch.** "CS 'asymptotic' and 'small'" (p.15) vs the table labels "Asymptotic" and "Wild Bootstrap".
- **Point estimates said to coincide, rejection rates that do not.** The paper says the Callaway-Sant'Anna, Sun-Abraham and `jwdid` point estimates coincide without covariates (pp.8, 16) and that `jwdid` inference is "similar to Sun and Abraham (2021)" (p.16), yet the Sun-Abraham and JWDID columns differ widely in the synthetic tables (for example Table 2, 100 units / 25 treated / 10 periods: Sun-Abraham 0.076, JWDID 0.391, Callaway-Sant'Anna 0.060 and 0.062). The paper offers no explanation. Reviewer note: the aggregation weights are stated for Sun-Abraham only.
- **DCDH citation.** The table notes define "DCDH = de Chaisemartin and d'Haultfoeuille (2024)"; the conclusion calls it "the de Chaisemartin et al. (2024) estimator and inference procedure" (p.24).

**The two added inference procedures**
- **No formula for the bootstrap placebo outcomes.** How the null is imposed, which residuals are perturbed, and over which observations, are given in words only (p.17).
- **Weight level.** "Rademacher weights (which are independent across units)" (p.17) is read here as one weight per unit; the paper does not say that the weight is constant within a unit over time.
- **Number of bootstrap and randomization draws:** not stated.
- **p-value definition.** "the placement of t in the distribution of the placebos" (p.17); no formula, no statement on ties for the bootstrap, no statement on whether the original statistic is counted among the placebos.
- **`beta_hat` in equation (2)** "represents the average treatment effect" (p.16). The numerator is printed as the simple mean of the unit-specific estimates; whether `beta_hat` is that same unit-weighted mean or an observation-weighted average is not stated (reviewer note: the two differ when treated units have different numbers of treated periods).
- **First-step model.** The imputation first step is not written out for these procedures; Section 2.2 describes it in words for Gardner (2022) (p.9).
- **Which package, if any, computes the first step** for the two added procedures is not stated.
- **No theory.** Validity is argued by analogy to the TWFE wild bootstrap and qualified by the authors (pp.24-25; footnotes 3, 7, 9).
- **Randomization test at exactly 5 treated units** is replaced by a t-test; the paper's designs have no treated-unit counts between 5 and 10, so the switch point is defined only as "5" vs "10 or more" (p.12).

**Tables**
- **JWDID RMSE cells (reviewer finding).** In all four tables the printed JWDID RMSE equals the printed Borusyak et al. asymptotic RMSE (0.372, 0.319, 0.280, 0.033). Recomputing from the printed JWDID rows with the formula that reproduces every other column gives 0.378 (Table 1), 0.419 (Table 2), 0.426 (Table 3) and 0.376 (Table A1). In Table A1 the printed 0.033 is impossible for a column whose mean rejection rate is 0.376, since the root mean squared difference from 0.05 cannot be smaller than the absolute difference of the mean from 0.05 (0.326). The JWDID RMSE cells should be treated as misprints; the JWDID rejection-rate rows and the JWDID mean are internally consistent.
- **"mean squared error" vs "RMSE".** The text says "mean squared error relative to the expected rejection rate of 0.05" (p.18); the tables and their notes report the root ("RMSE", "root mean squared difference"). The printed values match the root form (reviewer check).
- **No Monte Carlo standard errors** and no tests of differences between columns.
- **Conclusion vs Table 3.** "rejection rates consistently decrease across all methods and simulation parameters" (p.24) does not hold for the JWDID column (see Results).
- **"with the exception of DCDH"** (p.21) omits the two added procedures, which also do not over-reject with 5 treated units in Table 2.
- **Table A1 has no homogeneous-effect or CPS counterpart**, so the effect of singleton cohorts is shown only under heterogeneous effects.

**Not developed by the paper**

*Each item below means the paper gives no procedure, formula, or evidence for it. It does not mean the words never occur: passing mentions are noted where found.*
- Power, bias, coverage, minimum detectable effect, and any design with a non-zero average effect.
- Effects that vary over time since adoption or by cohort; proportional effects; more than one degree of heterogeneity.
- Event-study coefficients, pre-trend tests, cohort-specific effects, uniform bands.
- Covariates (p.7), common adoption dates, non-absorbing treatments, unbalanced panels, repeated cross-sections, sampling weights.
- Treated-unit counts below 5 or between the three values used; total units above 100.
- Clustering above the unit level; small-sample corrections to the cluster-robust variance (mentioned in footnote 2, p.3, not evaluated).
- The block bootstrap of the earlier DCDH command (footnote 8, p.15: results not provided).
- Guidance for running the exercise on a practitioner's own data.

**Software and replication materials**
- Stata commands are named (pp.15-16) without versions, options beyond those quoted, or command lines.
- No code availability statement, data availability statement, repository or replication-package reference appears anywhere in pp.1-32. The CPS data are described only as "publicly available monthly CPS data" (p.13).
- The reference list (pp.26-28) is transcribed above only for the entries the paper cites for its estimators, software and inference methods.
