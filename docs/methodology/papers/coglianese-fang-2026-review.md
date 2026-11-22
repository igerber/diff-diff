# Paper Review: An Evaluation of Difference-in-Differences Methods Using Placebo Event Studies

**Authors:** John Coglianese, Jade A. Fang (authors listed in certified random order)
**Citation:** Coglianese, J., & Fang, J. A. (2026). An Evaluation of Difference-in-Differences Methods Using Placebo Event Studies. Finance and Economics Discussion Series 2026-045. Washington: Board of Governors of the Federal Reserve System. Paper dated June 1, 2026.
**DOI:** https://doi.org/10.17016/FEDS.2026.045
**Source reviewed:** FEDS 2026-045, 34-page PDF (cover sheet + printed pp. 1-33; bibliography from printed p. 30; no appendix). The PDF was reviewed externally and is not committed to the repository. Reproduce by downloading the working paper via the DOI above. All page citations below are PRINTED page numbers. Greek letters and mathematical symbols inside quoted passages are transliterated to ASCII (for example `tau`, `rho`, `+/-`, `x` for the multiplication sign).
**Review date:** 2026-11-22

> **Personal verification stamp (load-bearing items spot-checked against the source).** Beyond the multi-agent extraction, the reviewer checked directly against the source PDF: every cell of Table 2 (p.19); the "TWFE-like estimators all perform identically to TWFE" passage (p.17); the Section 3 draw - estimate - repeat - pool description (p.7) and the Section 3.5 design statements, including "the true 'effect' is zero on average" (p.16); the synthetic difference-in-differences weight objectives (pp.14-15); the "giving equal weight to each estimator" normalization aim (p.25); footnote 12 (p.23); and the absence of any code or data availability statement (full-text search of all 34 PDF pages). Every quoted passage of 40 or more characters was additionally run through a mechanical word-sequence comparison against the PDF text layer.

---

## Methodology Registry Entry

**Status: proposed text for a future REGISTRY section; no such section exists in `docs/methodology/REGISTRY.md` today; this file is a non-authoritative source audit.** The paper is the candidate primary source for a planned "design diagnosis" capability (run placebo assignments on a user's own untreated data and report estimator performance). Nothing below is a methodology contract until a REGISTRY section is written and reviewed alongside an implementation.

*This is an EVALUATION paper, not an estimator paper: it proposes no new estimator, states no theorem, and prints no numbered equation. The standard registry headings (estimator equation, standard errors) therefore do not apply and are replaced by a procedure, a performance metric, and the roster of estimators evaluated. The registry-candidate text ends at the marker before `## Results`; everything below that marker is audit material (transcribed results, quotes, implementation notes, gaps) and is NOT part of the proposed registry text.*

## Placebo event-study design diagnosis (working title)

*The heading above is provisional. The paper's own term for the exercise is "placebo tests" / "placebo event studies"; the name of any library surface is to be decided in the implementation plan.*

**Primary source:** [Coglianese, J., & Fang, J. A. (2026). An Evaluation of Difference-in-Differences Methods Using Placebo Event Studies. Finance and Economics Discussion Series 2026-045. Washington: Board of Governors of the Federal Reserve System.](https://doi.org/10.17016/FEDS.2026.045)

**Key implementation requirements:**

*Scope of the source:*
- Setting evaluated: "state-level labor market event studies in the United States" (p.6) on "a monthly, state-level panel dataset spanning 1990-2024" (p.15), 50 states and the District of Columbia (p.16).
- Single-unit event study only: "given a panel dataset with multiple units i and time periods t, exactly one unit receives treatment at a specific point in time" (p.9).
- What is measured: the spread of placebo POINT ESTIMATES around zero, summarized as mean squared error (MSE). MSE is the paper's only estimator-performance metric; it also reports descriptive summaries by horizon and by treated state and examines control-weight patterns. No inference quantity is evaluated - see "What the paper does NOT provide" below.

*Rationale (Sections 1-2, pp.2-7):*
- "Placebo tests are a form of randomization inference, as the random sampling of placebo events reveals the distribution of event study estimates underlying the null hypothesis. In this way, placebo tests measure the efficiency of estimators through the spread of the distribution of placebo estimates." (p.2)
- "A distribution centered at zero indicates no bias, and a lower variance indicates better efficiency." (p.3)
- "Placebo tests are complementary to Monte Carlo simulations. Both methodologies use randomization to reveal the finite-sample properties of estimators (Bertrand et al., 2004; Arkhangelsky et al., 2021; Borusyak et al., 2024). However, the two differ in their approaches: placebo tests leverage real-world data, while Monte Carlo simulations assume specific data generating processes. Monte Carlo simulations are advantageous because they give the analyst full control over the data generating process, but the results are specific to said data generating process and may not necessarily be representative of real-life dynamics." (p.6)
- "A narrower distribution of estimates implies a lower standard error under randomization inference, which is indicative of a more efficient estimator for the outcome of interest." (p.6). This is offered as an interpretation only; no standard error is computed anywhere in the paper.

*Placebo procedure (Section 3.5, pp.15-16; high-level statement in Section 3, p.7):*

The paper prints no numbered algorithm. The steps below are distilled from the prose; each step cites the page it comes from.

1. **Panel and outcomes.** Start from a real-data panel: monthly, state-level, 1990-2024 (p.15). Five outcomes are used, each run as a separate placebo exercise (p.15): unemployment rate (Current Population Survey); labor force participation rate (Current Population Survey); seasonally adjusted employment growth rate (Local Area Unemployment Statistics, LAUS); employment growth rate (Quarterly Census of Employment and Wages, QCEW); house price index growth rate (Federal Housing Finance Agency).
2. **Enumerate the placebo-treated unit.** Units are NOT sampled. Every one of the "50 states and the District of Columbia" (51 units) is taken in turn as the placebo-treated unit (p.16).
3. **Randomize the event date.** The only randomized dimension is the month within each calendar year: "31 year-months selected by randomly sampling one month in each calendar year from 1992 to 2022" (p.16). Every calendar year 1992-2022 is used.
4. **Edge exclusion rule.** "We exclude the first and last 24 months of data from sampling to ensure that we can produce point estimates over the entirety of the 24-month window of each event study." (footnote 6, p.16). With data spanning 1990-2024 this yields the 1992-2022 sampling range.
5. **Define the event.** An "event" is a (state, year-month) pair, for example "Nebraska for October 1998" (pp.3, 7). The design is "the Cartesian product between 13 estimators, two covariate options, five outcomes, 50 states and the District of Columbia, and 31 year-months" (p.16).
6. **Single treated unit, single cohort, absorbing path.** Exactly one unit is treated per placebo (p.9); "in our placebo tests we will only have a single 'treated' cohort" (p.10); "in our placebo tests we will only have movements from untreated to treated" (p.11). The treated state is compared "to control states" (pp.3, 7), described elsewhere as "the set of remaining control states" (p.4). The paper gives no formal one-line definition of the control pool.
7. **Estimate an event study "as if some treatment truly occurred during that event"** (p.7), "isolating dynamic treatment effects at each time horizon ranging from 12 months before to 12 months after the event" (p.16). The baseline specification (Section 3.1, p.9; unnumbered in the paper) is

   ```
   y_{i,t} = alpha_i + gamma_t + sum_{tau=-T}^{T} beta^tau * delta_{i,t-tau} + epsilon_{i,t}
   ```

   with `alpha_i` "unit-fixed effects", `gamma_t` "time-fixed effects", `delta_{i,t-tau}` "leads and lags of the event occurrence for unit i at tau periods before/after time period t", and `beta^tau` "the coefficient of interest, reflecting the dynamic treatment effect at tau periods after the event" (p.9). The paper never writes `T = 12` as an equality; the 12-months-each-side window is given in prose only (p.16).
8. **Normalization.** "For methods where it is possible, we normalize effects during the estimation process based on the latest pre-treatment period." (p.16). The latest pre-treatment period is `tau = -1` (pp.10, 13). For most estimators "the point estimate for the month preceding treatment is zero by construction" (p.18). Exceptions are listed under Edge cases.
9. **Covariate option.** Placebo tests are run "both with and without time-invariant covariates" (p.15), using "foreign-born share as a representative covariate" (p.15). "Not all estimators take time-invariant covariates, which we denote in Table 1 with 'n.a.' and do not include in our covariate specifications." (footnote 5, p.15)
10. **Reuse the same events across estimators.** "We iteratively estimate event studies over the same set of randomly selected 'events'" (p.16). Every estimator and specification sees the same placebo events (a paired comparison).
11. **Pool into one distribution per specification.** "We aggregate point estimates across time horizons and placebo events to produce a unique distribution for each specification." (p.16). A specification is a combination of estimator, covariate option, and outcome (pp.15-16); distributions are produced "for each estimator and outcome separately" (p.7).
12. **Summarize** each distribution with the performance metric below and compare against the baseline estimator.

Counts as stated by the paper: "134,000+ state-level placebo event studies across 13 estimators" (abstract, p.1); "over 134,000 placebo event study estimates across all specifications of interest" (p.16). The paper prints no per-specification count; see Computational Considerations for reviewer arithmetic.

No exclusion of real policy events is applied in the paper's own exercise (see Edge cases, footnote 2).

*Performance metric (Section 4, p.17; Table 2, p.19):*

The paper prints NO formula for its metric and NO numbered equations anywhere. The metric is defined in prose only:

> "We focus on **mean squared error (MSE)** to evaluate the performance of each difference-in-differences estimator. For each estimator, we calculate the average MSE across placebo events, including all time horizons post-event-date. Ranking estimators by MSE for each outcome reveals which estimators perform best, in the sense of having the lowest variance in practice, for the specific outcome." (p.17)

> "We display the MSE for TWFE in the first row, and in subsequent rows we provide the MSE difference relative to TWFE for each estimator, normalized as a percentage of the TWFE MSE to facilitate comparison." (p.17)

**Reviewer reconstruction (NOT printed in the paper).** With `e` indexing placebo events for estimator `m` and outcome `o`, and `beta_hat^tau_{e,m,o}` the placebo point estimate at horizon `tau`, the prose is consistent with:

```
MSE_{m,o}      = mean over { placebo events e, post-event horizons tau } of ( beta_hat^tau_{e,m,o} - 0 )^2
PctDiff_{m,o}  = 100 * ( MSE_{m,o} - MSE_{TWFE,o} ) / MSE_{TWFE,o}
```

Negative `PctDiff` = lower MSE than TWFE (shaded blue in Table 2); positive = higher (shaded red).

Ambiguities the paper leaves open (an implementation must make and document a choice for each):
- **Pooled vs per-event averaging.** "the average MSE across placebo events, including all time horizons post-event-date" (p.17) can be read as one mean over all (event, horizon) pairs, or as a per-event MSE over post horizons that is then averaged across events. The two coincide when every event contributes the same number of horizons.
- **Whether `tau = 0` is post.** "post-event-date" is not defined; the paper does not say whether the event month itself enters the MSE.
- **Scope of the footnote 9 decile trimming.** "We exclude the top and bottom deciles at each time horizon to prevent severe outliers from skewing MSE calculations." (footnote 9, p.20). The footnote is attached to the Section 5.1 normalization comparison; whether it also applies to Table 2 and Figures 2-4 is not stated.
- **Pre-event horizons.** The text says post-event horizons only. Pre-event horizons appear in Figure 1, but as percentiles, not as an MSE.
- **Units.** Table 2 prints bare numbers (for example 1.556) with no units.
- **Table 2 note wording.** The table note says subsequent rows report "the MSE for each estimator normalized to be a percentage of the TWFE MSE", while the row-group header reads "Percent difference relative to TWFE" and the text (p.17) says "MSE difference relative to TWFE". Reviewer note: the printed zeros for the TWFE-like rows are only consistent with the difference reading used in the reconstruction above.

Ranking and comparison rules as stated:
- Estimators are ranked by MSE within each outcome (p.17). No cross-outcome ranking or composite score is constructed.
- TWFE is the benchmark (pp.8-9); every other estimator is expressed as a percent MSE difference from TWFE (Table 2).
- No uncertainty measure is attached to any MSE number and no statistical test of MSE differences between estimators is reported.

Other descriptive summaries used in the paper (none is a second performance metric):
- **Percentiles by horizon** (Figure 1, p.21): 10th, 25th, 50th, 75th and 90th percentiles of placebo estimates at each horizon from -12 to +12 months. This is the only exhibit that shows the distribution by horizon.
- **Normalized MSE by treated unit** (Section 6, p.25; Figure 4, p.26): "For each combination of estimator, outcome, and covariate choice, we average MSE by state across placebo events and convert these state-level averages to a unit scale so that the minimum value equals 0 and the maximum value equals 1. We then average the normalized MSE for each state across specifications." (p.25). Figure 4 plots the percentile of this average. Reviewer reconstruction (described in words only, not printed):

  ```
  For each specification k = (estimator, outcome, covariate choice):
      MSE_{s,k}      = average MSE of placebo estimates when state s is the placebo-treated unit
      NormMSE_{s,k}  = ( MSE_{s,k} - min_s MSE_{s,k} ) / ( max_s MSE_{s,k} - min_s MSE_{s,k} )     in [0, 1]

  NormMSE_s = average over specifications k of NormMSE_{s,k}
  ```

  The paper states the aim of this normalization as "giving equal weight to each estimator" (p.25). It does not explain how averaging across specifications gives equal estimator weight when estimators have different numbers of applicable covariate options, so the specification weights in the reconstruction above are unresolved; equal specification weights should not be assumed to be equal estimator weights.

  "The two covariate options are none or foreign-born share. Also, we exclude generalized synthetic control and elastic net synthetic control because they are extreme outliers." (footnote 13, p.25)
- **Average control weight by control unit** (Section 6, pp.26-27; Figure 5, p.28): for the four estimators that weight control states, "the average control weight assigned to each state by method, coloring by percentile based on averages across combinations of placebo events, covariates, and outcomes" (pp.26-27). No numeric weights are reported.

Bias is discussed conceptually but never computed: "For each distribution, the mean of placebo estimates represents the probability limit of the estimator and the spread reflects its efficiency. Since we have randomly chosen our 'events,' all estimators should have the same probability limit of zero; however, estimators may vary in terms of their efficiency." (pp.7-8). Footnote 8 (p.17): "Since MSE measures the difference from zero, it captures both bias and variance; however, as these are placebo estimates the bias term is zero by definition (since the true estimate is zero), therefore our measured MSE reflects only the variance term." No empirical mean of any placebo distribution is reported.

*Estimators evaluated (Table 1, p.8: "Difference-in-differences methods of interest"):*

Selection rule: "selecting 13 methods used in recent literature" (p.8). Grouping rule: "We choose two-way fixed effects (TWFE) as the baseline and assign each of the other twelve estimators to one of the following groups: TWFE-like, matching, and synthetic-control-like." (p.8). Table 1 note: "The table describes our implementation of each estimator in our baseline specification. n.a. means not applicable."

The three columns headed "T1" transcribe Table 1 (Covariates / Optimize on pre-treatment outcomes / Normalization). The remaining columns are collected from Sections 3.1-3.4 (pp.9-15).

| Estimator | Family | Citation as the paper gives it | T1: Cov. | T1: Pre-outcome opt. | T1: Norm. | Key specification choices | Normalization handling |
|---|---|---|---|---|---|---|---|
| Two-way fixed effects | Baseline | None given; "the traditional method for estimating event studies" (p.9) | n.a. | n.a. | Yes | Unit and time fixed effects with leads and lags of the event (Section 3.1, p.9); benchmark for all comparisons | Relative to the latest pre-treatment period (p.11) |
| Callaway and Sant'Anna | TWFE-like | Callaway and Sant'Anna (2021); "doubly robust estimation method of Sant'Anna and Zhao (2020)" (p.10) | No | n.a. | Yes | Doubly robust; "in some specifications we include time-invariant covariates"; comparison group as printed: not yet treated at time t+tau (p.10) | "relative to tau = -1" (p.10) |
| Imputation | TWFE-like | Borusyak et al. (2024) (p.10) | n.a. | n.a. | Yes | Unit and time fixed effects fit on never-treated or not-yet-treated observations; effect is the average treated-minus-imputed outcome (p.10) | "relative to the latest pre-treatment period, tau = -1" (p.10) |
| Wooldridge difference-in-differences | TWFE-like | Wooldridge (2025) (pp.10-11) | No | n.a. | Yes | "extends TWFE to include fully saturated interactions between the treatment-time indicator and group dummies" (p.10) | "As with baseline TWFE, we normalize estimates relative to the latest pre-treatment period." (p.11) |
| De Chaisemartin & D'Haultfoeuille | TWFE-like | de Chaisemartin and D'Haultfoeuille (2020); "estimation method of de Chaisemartin and D'Haultfoeuille (2024)" (p.11) | n.a. | n.a. | Yes | First-difference comparison of newly treated units against units untreated in t and t-1 (p.11) | Relative to the latest pre-treatment period (p.11) |
| Local projections | TWFE-like | Dube et al. (2025) (p.11) | No | n.a. | Yes | "estimates difference-in-differences separately by event time tau"; time-invariant covariates account "for differential trends across units" (p.11) | Through the differenced outcome: "the coefficient estimates represent the effect normalized relative to this period" (p.11) |
| Propensity score matching | Matching | Rosenbaum and Rubin (1983) (p.12) | Yes | n.a. | Yes | Keep "the units with the five highest propensity scores as our controls", drop the rest, then estimate baseline TWFE (p.12; footnote 3) | Inherits baseline TWFE (Table 1: Yes) |
| Coarsened exact matching | Matching | Iacus et al. (2012) (p.12) | Yes | n.a. | Yes | Sturges method for the number of bins per covariate; controls must share the treated unit's bin "for all covariates" (footnote 4, p.12); then baseline TWFE | Inherits baseline TWFE (Table 1: Yes) |
| Synthetic control | Synthetic-control-like | Abadie et al. (2010) (p.13) | No | Yes | Yes | Weights "constrained to be non-negative and sum to one"; X is "a vector of pre-treatment outcomes and/or covariates" (p.13) | Estimates normalized "relative to tau = -1" (p.13) |
| Elastic net synthetic control | Synthetic-control-like | Doudchenko and Imbens (2016) (p.13) | n.a. | Yes | No | Allows "negative weights, weights that do not sum to one, and a permanent additive difference"; elastic net regularization; cross validation tunes the hyperparameters (p.13) | Outcome pre-differenced for each unit relative to the last pre-treatment period before fitting; tau = -1 estimate "close to, but not exactly equal to, zero" (p.13) |
| Generalized synthetic control | Synthetic-control-like | Xu (2017) (pp.13-14) | n.a. | Yes | No | Factor model on control units only; loadings for the treated unit from the pre-treatment period only; "Cross validation is used to select the optimal number of factors." (p.14) | Same pre-differencing; "some non-zero values for tau = -1" (p.14) |
| Matrix completion | Synthetic-control-like | Athey et al. (2021) (p.14) | n.a. | Yes | No | Untreated potential outcomes treated as a matrix with missing entries that are filled in; hyperparameters chosen "using cross-validation" (p.14) | Same pre-differencing; "some non-zero values for tau = -1" (p.14) |
| Synthetic difference-in-differences | Synthetic-control-like | Arkhangelsky et al. (2021); event-study estimates "follow the procedure detailed in Clarke et al. (2023)"; staggered adaptation attributed to Ben-Michael et al. (2022) (pp.14-15) | n.a. | Yes | No | Chooses unit weights for controls and time weights for periods: "The unit weights are chosen to minimize differences (up to a constant) between treated and control units during the pre-treatment period, while the time weights are chosen to minimize pre-treatment and post-treatment differences (up to a constant) among control units" (pp.14-15) | None: "since this method estimates time period weights, we do not normalize estimates to the latest pre-treatment period" (p.15) |

Family definitions as stated:
- **TWFE-like:** "We define TWFE-like estimators as variations of TWFE that are robust to the challenges of staggered adoption settings. In our single-event research setting, these distinctions are irrelevant: TWFE-like estimators all converge to the canonical TWFE estimator since we only have a single treated unit. However, these methods are slightly different in their implementation, so we still evaluate these methods to determine whether they perform any worse than TWFE under finite-sample randomization inference." (p.9). They "compare treated units to all control units available in the data" (p.11) and "assign equal weight to all control units" (p.13).
- **Matching:** "Matching estimators work by comparing treated units to a subset of control units that are sufficiently similar based on a set of observable covariates." (p.11). Footnote 3 (p.12): "The matching process selects which units to compare, not how, so practitioners could theoretically implement any difference-in-differences method on the subsetted data. In our approach, we intentionally estimate TWFE after matching to offer a comparison that highlights potential gains from the matching process itself."
- **Synthetic-control-like:** "These estimators are typically fitted on covariates and pre-treatment data in order to construct a close counterfactual for each treated unit, and tend to feature more complex statistical algorithms than the methods described above." (p.12). They "typically use the pre-treatment outcomes in assigning weight to control units" (p.13).

Estimators that accept time-invariant covariates (six, per Section 5.3 and Figure 3): coarsened exact matching, Callaway and Sant'Anna, local projections, propensity score matching, synthetic control, Wooldridge difference-in-differences. The two matching estimators require covariates (footnote 12, p.23).

*Estimator formulas as printed (Sections 3.2-3.4, pp.10-14; all unnumbered; the baseline TWFE equation is in procedure step 7):*

```
Callaway and Sant'Anna (p.10):
    beta_g^tau = E[ y_{i,t+tau} - y_{i,t-1} | Treatment cohort g ]
               - E[ y_{i,t+tau} - y_{i,t-1} | Not yet treated at time t+tau ]

Imputation (p.10):
    y_{i,t} = alpha_i + gamma_t + epsilon_{i,t}          where i,t are never-treated or not-yet-treated
    beta^tau = (1/N) * sum_i ( y_{i,t+tau} - y_hat_{i,t+tau} )     where i is treated

Wooldridge difference-in-differences (pp.10-11):
    y_{i,t} = alpha_i + gamma_t
            + sum_{tau=-T}^{T} sum_g beta_g^tau * delta_{i,t-tau} * 1(treatment cohort g) + epsilon_{i,t}

de Chaisemartin and D'Haultfoeuille (p.11):
    beta = E[ y_{i,t} - y_{i,t-1} | Newly treated in t ] - E[ y_{i,t} - y_{i,t-1} | Untreated in t, t-1 ]

Local projections (p.11):
    y_{i,t+tau} - y_{i,t} = beta^tau * delta_{i,t} + theta * X_{i,t} + epsilon_{i,t}

Propensity score matching (p.12):
    Pr(Treatment_i | X) = theta * X_{i,t} + epsilon_{i,t}

Synthetic control (p.13):
    beta^tau = y_{Treated,t+tau} - sum_{i in {Control}} w_hat_i * y_{i,t+tau}
    w_hat == argmin_w || X_Treated - sum_{i in {Control}} w_i * X_i ||

Generalized synthetic control (pp.13-14):
    beta_i^tau = y_{i,t+tau} - lambda_i * F_hat_t     where i is treated

Synthetic difference-in-differences (p.14):
    beta^tau = sum_{i in Treated} (1/N_{Tr}) * ( y_{i,t+tau} - sum_{t'=-T}^{-1} phi_{t'} * y_{i,t'} )
             - sum_{j in Control} w_j        * ( y_{j,t+tau} - sum_{t'=-T}^{-1} phi_{t'} * y_{j,t'} )
```

No equation is displayed for coarsened exact matching, elastic net synthetic control, or matrix completion. Transcription notes (as printed, not reconciled by the paper): the de Chaisemartin and D'Haultfoeuille `beta` carries no `tau` superscript; the local-projections left-hand side subtracts `y_{i,t}` while the prose says the outcome is differenced relative to the latest pre-treatment period; the generalized synthetic control factor is indexed `F_hat_t`, not `F_hat_{t+tau}`; the synthetic control norm has no subscript or weighting matrix; the synthetic difference-in-differences prose names unit weights `w_i` while the equation uses `w_j`; `N`, `N_{Tr}` and `epsilon_{i,t}` are not defined in words.

*Edge cases:*
- **Single treated unit collapses the TWFE-like estimators to TWFE.** Table 2 (p.19) prints a percent difference of 0 for all five TWFE-like estimators (Callaway and Sant'Anna, imputation, Wooldridge, de Chaisemartin and D'Haultfoeuille, local projections) on all five outcomes. The text: "TWFE-like estimators all perform identically to TWFE across the five economic outcomes of interest. Although these methods differ in how they estimate treatment effects with multiple cohorts of treated units, having only one treated unit in our setting simplifies them all to TWFE, which explains the identical performance." (p.17). Reviewer note: Table 2 reports whole-number percentages, so the table alone cannot distinguish exact numerical equality from differences that round to 0; the claim of identity rests on the quoted text. Footnote 1 (p.3): "Note that we focus on the single event study setting in our analysis; conducting placebo tests in staggered adoption settings may result in more discernable differences in performance for the TWFE-like estimators." The paper does not evaluate that case.
- **Outlier estimators are excluded from some exhibits.** Generalized synthetic control is omitted from Figure 2 because "this estimator is a severe outlier, and showing it would limit comparability" (footnote 10, p.21). Generalized synthetic control and elastic net synthetic control are excluded from the normalized-MSE-by-state measure and Figure 4 "because they are extreme outliers" (footnote 13, p.25). All 13 estimators remain in Table 2. Separately, the top and bottom deciles of estimates at each horizon are trimmed in (at least) the Section 5.1 normalization comparison (footnote 9, p.20).
- **Real policy events contaminate the null.** Footnote 2 (p.6): "As in usual randomization inference, this exercise is informative about the variance under the null hypothesis only for studying policies that do not change during our sample period. For policies that change during our sample period, our variance estimate also potentially includes the treatment effect, but one could conduct a similar exercise excluding the treated units to estimate the variance under the null hypothesis." The paper's own exercise applies no such exclusion.
- **Normalization cannot be imposed exactly for every estimator.** Footnote 7 (p.16): "This normalization practice standardizes a reference point across methods, which facilitates clear, concrete interpretations of point estimates. However, as mentioned in the methodology section, we do not obtain perfectly normalized estimates for elastic net synthetic control, generalized synthetic control, and matrix completion; additionally, we do not impose any normalization for synthetic difference-in-differences." For the first three, normalization is applied to the DATA (each unit's outcome is differenced relative to its latest pre-treatment period before fitting), not to the estimates, leaving `tau = -1` non-zero (pp.13-14).
- **Coarsened exact matching can lose its sample.** It "is the most sensitive to covariate choice because it strictly defines treatment and control comparisons based on the covariate cells, and sample size can shrink rapidly as more covariates are included in the matching process with the curse of dimensionality" (p.23).
- **Covariates are time-invariant only.** Footnote 11 (p.22): "Time-invariant controls account for persistent differences between treatment and control units. Some methods also allow for controlling for time-varying covariates, which we leave to future work to examine."
- **Convergence failures, non-estimable events, missing data, unbalanced panels:** Not discussed in paper.

**Reference implementation(s):** None. No software package, function, or programming language is named in the paper, and no replication code is referenced (see Gaps and Uncertainties for the scope of that check).

### Paper-derived requirements

*Only items the paper specifies for its own exercise or explicitly recommends to practitioners. These constrain what may be attributed to this source; they do not constrain the API surface.*

- [ ] Placebo events are drawn on real outcome data with no assumed data generating process (pp.2, 6)
- [ ] An event is a (unit, period) pair; exactly one unit is placebo-treated per event; treatment is absorbing with a single treated cohort (pp.7, 9-11)
- [ ] Placebo-treated units are enumerated, not sampled; the event date is the randomized dimension (one random month per calendar year in the paper) (p.16)
- [ ] Event dates are excluded near the panel edges so that the full event window can be estimated (footnote 6, p.16: first and last 24 months excluded for a 12-before / 12-after window)
- [ ] Event-study point estimates are produced at every horizon of the window (12 before to 12 after in the paper) (p.16)
- [ ] Estimates are normalized to the latest pre-treatment period wherever the estimator allows, and estimators that cannot be normalized exactly are identified as such (p.16; footnote 7)
- [ ] The same set of placebo events is reused across every estimator and specification (p.16)
- [ ] One distribution of placebo estimates is formed per specification, pooling across events and horizons (p.16)
- [ ] The performance summary is the MSE of placebo point estimates around zero over placebo events and post-event horizons (p.17)
- [ ] Each estimator's MSE is reported as a percent difference relative to the TWFE baseline, and estimators are ranked within each outcome (p.17; Table 2, p.19)
- [ ] MSE is interpreted as variance only: bias is asserted zero by definition, not measured (footnote 8, p.17). Reviewer note: this is the authors' asserted interpretation. A zero true placebo effect does not by itself establish that an estimator is unbiased (`MSE = variance + bias^2`), and the paper reports no empirical mean of any placebo distribution, so the variance-only reading should not be promoted to a guarantee
- [ ] For policies that change during the sample period, the placebo variance "potentially includes the treatment effect"; the stated remedy is to repeat the exercise "excluding the treated units" (footnote 2, p.6)
- [ ] Recommended to practitioners: conduct placebo tests in their own setting (pp.1, 2, 18, 27, 29); normalize and seasonally adjust outcomes where possible (pp.21-22, 29)
- [ ] Optional descriptive summaries used by the paper: percentiles of placebo estimates by horizon (Figure 1); min-max normalized MSE by placebo-treated unit (Figure 4); average control weight by control unit (Figure 5)

### What the paper does NOT provide

**The paper's only estimator-performance metric is the MSE of placebo point estimates.** It also reports descriptive summaries by horizon and by treated state and examines which control states each estimator weights (Section 6). It does not evaluate power, rejection rates, or confidence-interval coverage. None of the following appears anywhere in printed pp. 1-33:

- **No effect injection.** No artificial or simulated treatment effect is ever added to the data. The authors describe the true placebo effect as zero ON AVERAGE: "Since we randomize placebo events, the true 'effect' is zero on average" (p.16). Footnote 2 (p.6) warns that real policy changes in the sample can contribute treatment effects to the estimated variance.
- **No power.** Power is not computed and is not used as a metric.
- **No minimum detectable effect.** Not computed, not mentioned.
- **No size or rejection rates.** No test is run on any placebo estimate; no empirical rejection frequency is reported.
- **No confidence-interval coverage.** No confidence interval is constructed.
- **No standard errors or p-values of any kind.** No estimator-reported standard error, no placebo or permutation p-value, no randomization-inference p-value for a real event. The only mention of a standard error is the interpretive sentence on p.6 quoted under Rationale.
- **No bias metric.** Bias is asserted to be zero by definition (footnote 8, p.17); no empirical mean of a placebo distribution is reported.
- **No RMSE or mean absolute deviation.** MSE is the only summary of spread.
- **No uncertainty on the metric itself.** No interval or test accompanies any MSE number or any MSE difference between estimators.
- **No multiple-treated-unit or staggered-adoption placebo design.** The design has exactly one treated unit and one cohort (pp.9-10); footnote 1 (p.3) explicitly leaves staggered adoption out.
- **No standalone practitioner algorithm or checklist.** Practitioners are told to run their own placebo tests. The procedural content is a high-level draw - estimate - repeat - pool description in Section 3 ("We then repeat this process many times with additional randomly drawn events and pool the estimates together to produce a distribution of placebo estimates", p.7) and the exercise-specific details of the authors' own design in Section 3.5.
- **No decision rule for choosing an estimator.** The selection criterion is stated only as "identify and implement the estimator that performs the best" (p.27), with best meaning lowest MSE (p.17); no threshold, tie-break, or minimum gap is given.
- **No recommended number of draws.** The 31 dates x 51 units grid is fixed and no sensitivity to the number of placebo events is examined.
- **No held-out validation.** No out-of-sample or split-sample check of an estimator chosen by placebo MSE is described.
- **No guidance on restricting placebo events in an application with a real treatment** beyond the one-sentence remedy in footnote 2 (p.6).
- **No software or replication code.** No package, function, or language is named and no replication material is referenced.
- **No formula for the metric and no numbered equations.**

**Any of these in a library implementation would be a library design choice requiring its own source. It would NOT be derived from this paper and must not be attributed to it.**

---

<!-- ============================================================
     END of registry-candidate block. Everything BELOW this
     marker is audit material (transcribed results, quotes,
     implementation notes, gaps) and is NOT part of the proposed
     REGISTRY.md text.
     ============================================================ -->

## Results

Exhibit map (what each exhibit aggregates over and splits by):

| Exhibit | Page | Aggregated over | Split by |
|---|---|---|---|
| Table 2 | p.19 | placebo events, post-event horizons | estimator x outcome |
| Figure 1 | p.21 | placebo events | horizon (-12 to +12) x normalization on/off; synthetic control only; seasonally adjusted LAUS employment growth only |
| Figure 2 | p.22 | placebo events, horizons | estimator (12; generalized synthetic control excluded) x seasonal adjustment; LAUS employment growth only |
| Figure 3 | p.24 | placebo events, horizons | estimator (6) x covariate specification (14; 13 for the matching estimators); seasonally adjusted LAUS employment growth only |
| Figure 4 | p.26 | placebo events, horizons, then specifications after min-max scaling | placebo-treated state |
| Figure 5 | p.28 | placebo events, covariates, outcomes | control state x estimator (4) |

### Table 2 (p.19): "Mean squared error of estimates by method and outcome"

Held fixed: default normalization; "Matching estimators use foreign-born share as a covariate; all other methods exclude covariates" (table note).

| | RU | LFPR | LAUS Emp. (SA) | QCEW Emp. (NSA) | HPI |
|---|---|---|---|---|---|
| **Baseline mean squared error of estimates** | | | | | |
| Two-way fixed effects | 1.556 | 2.18 | 1.57 | 3.189 | 5.426 |
| **Percent difference relative to TWFE** | | | | | |
| Callaway and Sant'Anna | 0 | 0 | 0 | 0 | 0 |
| Imputation | 0 | 0 | 0 | 0 | 0 |
| Wooldridge difference-in-differences | 0 | 0 | 0 | 0 | 0 |
| De Chaisemartin & D'Haultfoeuille | 0 | 0 | 0 | 0 | 0 |
| Local projections | 0 | 0 | 0 | 0 | 0 |
| Propensity score matching | 12 | 19 | -1 | 9 | -9 |
| Coarsened exact matching | 14 | 19 | -1 | 9 | -7 |
| Synthetic control | -16 | -10 | 8 | -4 | -60 |
| Elastic net synthetic control | 248 | 13 | 78 | -9 | -65 |
| Generalized synthetic control | 95 | 164 | 10662 | 216 | 2278 |
| Matrix completion | -29 | -28 | 27 | -35 | 52 |
| Synthetic difference-in-differences | -20 | -11 | 8 | -8 | 25 |

Table note, verbatim: "First row reports mean squared error (MSE) for TWFE placebo estimates. Subsequent rows report the MSE for each estimator normalized to be a percentage of the TWFE MSE. Cells are colored based on the MSE relative to TWFE, with blue representing improvement and red representing worse performance; more extreme values are darker. Matching estimators use foreign-born share as a covariate; all other methods exclude covariates. RU refers to the unemployment rate. LFPR refers to the labor force participation rate. LAUS Emp. (SA) refers to the Local Area Unemployment Statistics employment growth rate that is seasonally adjusted. QCEW Emp. (NSA) refers to the Quarterly Census of Employment and Wages employment growth rate that is not seasonally adjusted. HPI refers to the house price index growth rate. Source: Bureau of Labor Statistics; Census Bureau; Federal Housing Finance Agency."

**TWFE baseline MSE levels** (first row of Table 2) range from 1.556 (RU) to 5.426 (HPI), a ratio of about 3.5 between the largest and smallest (reviewer arithmetic). Units are not stated.

**Findings stated in the text (Section 4, pp.17-18):**
- "Overall, we find that there is no single 'superior' method." (p.17)
- "Matching estimators based on TWFE also perform similarly: up to about 10 percent better than TWFE when the outcome is LAUS employment or house price index and up to around 20 percent worse for the other outcomes." (p.17)
- "Performance varies substantially for synthetic-control-like estimators across outcomes, outperforming TWFE in some cases but dramatically underperforming in other cases. These differences in MSE are nontrivial. Certain combinations of estimator and outcome produce variance that is 30-60 percent lower than TWFE, but in the worst cases other combinations yield variance that is multiples of that of TWFE." (p.17)
- "Even within the set of synthetic-control-like methods, performance is not strictly ordered across estimators. For example, matrix completion performs best for three of the five outcomes but worse than other methods for the two remaining outcomes. The one exception is generalized synthetic control, which performs substantially worse compared to TWFE for every outcome." (pp.17-18)
- "In sum, we do not find that there is a single method that performs best for every outcome, nor even that there is a general ordering of methods." (p.18)
- "the ranking of estimators by efficiency also varies across outcomes" (p.28)

**Reviewer note (read directly off Table 2, not stated by the paper).** Lowest-MSE estimator per outcome: RU - matrix completion (-29); LFPR - matrix completion (-28); LAUS Emp. (SA) - propensity score matching and coarsened exact matching (-1 each; every synthetic-control-like method is worse than TWFE on this outcome); QCEW Emp. (NSA) - matrix completion (-35); HPI - elastic net synthetic control (-65), then synthetic control (-60). Elastic net synthetic control ranges from -65 (HPI) to +248 (RU). The -65 cell lies slightly outside the "30-60 percent lower" range quoted from p.17.

### Authors' explanation for synthetic-control-like variability

> "The variation in performance of synthetic-control-like methods illustrates a bias-variance trade-off inherent to using these methods. Unlike TWFE or matching, synthetic-control-like methods use information about pre-treatment outcomes to construct the counterfactual for treated units, but this poses a risk of overfitting. In some settings, using pre-treatment outcomes helps deliver lower variance of placebo estimates, but in other settings these methods overfit on the pre-treatment period, resulting in higher variance of placebo estimates." (p.18)

Restated in the conclusion: "the practitioner must grapple with a bias-variance trade-off, in which optimizing on pre-treatment outcomes helps capture heterogeneous dynamics between units in certain cases but may lead to overfitting in others." (p.29). The introduction frames the same point: such methods "can sometimes construct better counterfactuals by picking up on heterogeneous dynamics across control units. However, the flexibility of these models in forming counterfactuals can also lead to overfitting in other cases." (pp.3-4)

The paper gives no outcome-specific explanation of why a given synthetic-control-like method wins on one outcome and loses on another, beyond this general argument.

### Normalization (Section 5.1, pp.18, 20; Figure 1, p.21)

- "We find that normalization vastly reduces the variance of estimates relative to estimating unnormalized specifications." (p.20)
- Single worked example: synthetic control on seasonally adjusted LAUS employment growth, where the paper reports normalization "reducing the MSE of the placebo estimates by 69 percent" (p.20), computed with the top and bottom deciles excluded at each horizon (footnote 9, p.20).
- "This example is representative of the gains from normalization in other combinations of estimators and outcomes, although the magnitude of gains may vary." (p.20). No table of the other combinations is given.
- "we find improvement from normalizing in all settings in our analysis" (p.20).
- Link to the synthetic-control-like advantage: "The benefits of normalization provide an explanation for how synthetic-control-like methods are able to outperform TWFE in some cases. Normalizing by the latest pre-treatment period is another way of using pre-treatment data to improve counterfactual estimation, as the synthetic-control-like methods do in various ways. Synthetic difference-in-differences notably estimates different weights for different pre-treatment periods, which nests our approach of normalizing by the latest pre-treatment period as a special case." (p.20)
- Figure 1, "Synthetic control estimates without normalization (left) vs. with (right)": percentile lines against "Time to treatment (months)" from -12 to 12. **Visual approximation (the paper prints no numbers):** without normalization the 10th/90th percentiles are about -0.9/+0.8 at tau = -12, narrow to roughly +/-0.2 near tau = -6, then fan out to about -2.2/+2.6 at tau = +12 (25th/75th about -1.1/+1.2); with normalization they are about -0.4/+0.4 at tau = -12, pinch to 0 just before the event, and fan out to about -1.6/+1.7 at tau = +12 (25th/75th about -0.75/+0.8). The median is near 0 throughout, and in both panels the spread grows monotonically with the post-event horizon.

### Seasonal adjustment (Section 5.2, pp.21-22; Figure 2, p.22)

- "seasonal adjustment of LAUS employment data decreases MSE by an average of 23 percent across the estimators shown. The magnitude of reductions varies across these methods, but the difference between seasonally adjusted and not seasonally adjusted data is generally larger than the differences across estimators." (p.21)
- Evidence is from one outcome only: LAUS employment growth "is the only outcome variable with both seasonally adjusted and not seasonally adjusted data that are readily available at the state-level for our time range of interest" (footnote 10, p.21).
- The seasonal adjustment method is not described; no in-house filter is described.
- Figure 2, "Mean squared error of estimates by method and seasonal adjustment": paired bars for 12 estimators (generalized synthetic control excluded). **Visual approximation:** not-seasonally-adjusted bars about 2.1-2.2 for most estimators, about 2.4 for synthetic control, about 3.1 for elastic net synthetic control; seasonally adjusted bars about 1.55-1.6 for TWFE, the TWFE-like and the matching estimators, about 1.7 for synthetic control and synthetic difference-in-differences, about 2.0 for matrix completion (smallest reduction), about 2.8 for elastic net synthetic control. The seasonally adjusted bar is lower for every estimator shown.

### Covariates (Section 5.3, pp.22-24; Figure 3, p.24)

- Covariate pool: "average age, average income, Black population share, college-educated share, female employment rate, foreign-born share, manufacturing share of employment, and poverty rate, each calculated from the 1990 Census." (p.23)
- Specifications in Figure 3 (figure note, p.24): none (not available for the matching estimators); each of the eight covariates alone; four two-covariate sets ({average income, average age}; {college-educated share, female employment rate}; {foreign-born share, Black population share}; {poverty rate, female employment rate}); one four-covariate set ({average income, college-educated share, Black population share, manufacturing share of employment}). The two- and four-covariate sets are "randomly selected" (p.23).
- "the choice of at least one covariate generally has much less of an impact on precision than the choices for normalization or seasonal adjustment. Among the combinations of estimators and sets of at least one covariate, the vast majority result in MSEs within +/-10 percent of their respective estimator's baseline of no covariates. These minor differences in performance between choices of covariates are representative of other choices of the outcome variable." (p.23)
- Exception: "One exception to this pattern is coarsened exact matching, where we see larger differences across different covariates." (p.23)
- Figure 3, "Mean squared error of estimates by method and covariate specification". **Visual approximation:** coarsened exact matching about 1.25 to about 2.2 (the four-covariate bar is tallest); Callaway and Sant'Anna about 1.05-1.27, with the four-covariate bar lower at about 0.85; local projections essentially flat at about 1.15-1.18; propensity score matching about 1.25-1.53; synthetic control about 1.05-1.45; Wooldridge about 1.07-1.22. These levels do not match Table 2 for the same outcome - see Gaps and Uncertainties.

### Treated-unit heterogeneity (Section 6, pp.25-28; Figures 4-5)

- "We show that MSE is systematically lower when the 'treated' state is populous and diverse, and the variation across states is often much larger than variation across estimators." (p.25)
- "Overall, estimates are the most precise in populous, diverse states and least so in smaller and more distinctive ones. The five states with the highest precision are Pennsylvania, Ohio, Illinois, Texas, and North Carolina; the five states with the lowest precision are Nevada, Hawaii, Alaska, DC, and Arizona. These differences are substantial: the normalized MSE of Nevada is over 26 times that of Pennsylvania. In unreported results, we have confirmed that these patterns are common across estimators and are not skewed by any single estimator or specification in particular." (p.25)
- Explanation offered: "Treated states that are populous and diverse resemble the contiguous United States more so than their idiosyncratic counterparts, thus yielding more precise estimates." (p.26)
- "which unit is 'treated' matters just as much as which estimator is chosen: large, diverse states systematically yield lower variance in placebo estimates than their idiosyncratic counterparts." (p.28). Abstract: "Performance also varies across states at least as much as it does across estimators." (p.1)
- Control weights (Figure 5, four estimators: propensity score matching, synthetic control, elastic net synthetic control, synthetic difference-in-differences): "Unlike for the precision of estimates, we find that geographic patterns in control weights are not common across methods. For example, propensity score matching favors states in New England and the Great Plains, while synthetic difference-in-differences favors states in the Southwest and the Rust Belt." (p.27)
- "While differences in MSE across treated states align for most estimators, the optimal choice of control states varies substantially across estimators, since estimators form counterfactuals from control states differently." (p.25). "When only a subset of states is available, some estimators may be able to attain close to their optimal set of states, while other estimators' preferred states may not be in the subset." (p.27)
- Hedged: "Much of the difference in control weights likely stem from technical differences between these estimators" (p.27).

### Relative importance of design choices, as ranked by the authors

- "other choices that researchers make in setting up their analysis matter at least as much as the choice of estimator" (p.18)
- "We find meaningful improvements from normalizing estimates relative to the pre-treatment period or removing seasonal patterns in time series data. In contrast, we find much less of an impact on the variance of estimates from controlling for different sets of time-invariant covariates." (p.18)
- "While researchers typically cannot choose the setting for a natural experiment, it may nonetheless matter as much as or more than the choices under researchers' control." (p.25)

---

## Practical Recommendations

All quotes verbatim with printed page numbers.

1. **Run placebo tests in your own setting.**
   - "Our results highlight the need for practitioners to conduct placebo tests to understand the performance of methods in their research context." (abstract, p.1)
   - "Given such context dependence, we urge practitioners to conduct placebo tests to understand the performance of their methods in their unique setting." (p.2)
   - "These findings highlight that the research context matters greatly in determining the precision of event study estimates, and researchers would benefit from examining performance of placebo estimates in their setting to determine the 'best' method to use." (p.18)
   - "To construct the best possible counterfactual under data constraints, researchers should conduct placebo tests to identify and implement the estimator that performs the best given the particular set of control states represented in the data." (p.27)
   - "Most importantly, we urge practitioners to conduct their own placebo tests to understand the practical performance of their methods in their unique setting." (p.29)

2. **Normalize and seasonally adjust.**
   - "The systematic pattern of lower variance across estimators with seasonally adjusted data suggests that researchers should use seasonally adjusted data when possible." (pp.21-22)
   - "We encourage analysts to normalize and seasonally adjust their outcome variables where possible." (p.29)

3. **Be aware of coarsened exact matching's covariate sensitivity.**
   - "Practitioners using this method should be aware of the sensitivity of this estimator to the choice of covariates." (p.23)

4. **Be mindful of which unit is treated.**
   - "The heterogeneity in performance across states highlights that practitioners should be mindful of where their event of interest occurred." (p.26)
   - "If the event of interest is in a particularly distinctive state, such as Nevada, we encourage researchers to be mindful that estimates may vary more than in a typical setting." (p.26)

5. **Be mindful of which control units are available.**
   - "The geographic variation in control weights conditional on estimator emphasizes that practitioners should be mindful of which states are represented in the data when considering which estimator to use." (p.27)

6. **Interpret and communicate carefully.**
   - "Researchers should be aware of the context dependence of difference-in-differences methods and strategize accordingly." (p.29)
   - "We also advise researchers to interpret results and communicate findings carefully, given that difference-in-differences estimates may vary significantly depending on which estimator is chosen, which outcome is selected, which state is treated, and which control states are represented in the data." (p.29)

7. **When real policy changes occur in the sample.**
   - "one could conduct a similar exercise excluding the treated units to estimate the variance under the null hypothesis." (footnote 2, p.6)

Beyond these statements and the high-level draw - estimate - repeat - pool description in Section 3 (p.7), the paper gives no step-by-step guidance, checklist, or decision rule.

---

## Implementation Notes (audit notes - NOT registry-candidate)

### Data Structure Requirements

What the paper's exercise uses (not a proposed library contract):
- A real-data panel of units by periods with at least one outcome: 51 units, monthly, 1990-2024 (pp.15-16). The paper does not screen the panel for real policy events (footnote 2, p.6).
- Enough periods before the first and after the last admissible event date to estimate the full event window; the paper excludes the first and last 24 months for a 12-before / 12-after window (footnote 6, p.16).
- One placebo-treated unit per event; the comparison units are the remaining units (pp.3, 4, 7), with matching estimators restricting that pool (p.12).
- Optional time-invariant unit covariates (p.15); time-varying covariates are out of scope (footnote 11, p.22).
- Balanced vs unbalanced panels, missing observations, and minimum numbers of units or periods: Not discussed in paper.
- Whether each placebo fit uses the full panel or only the event window: Not discussed in paper.

### Computational Considerations

- **Reviewer arithmetic (the paper does not print these numbers).** Placebo events per specification: 51 units x 31 year-months = 1,581. Horizons per event study: tau = -12 to +12 inclusive is 25 point estimates, of which tau = -1 is zero by construction for the exactly normalized estimators (p.18).
- **Reviewer arithmetic - count reconciliation.** The literal Cartesian product stated on p.16 is 13 x 2 x 5 x 51 x 31 = 205,530. The phrase "over 134,000" is a lower bound that 205,530 also satisfies, so the two are not strictly incompatible; but applying the covariate-applicability restrictions gives a count much closer to the stated figure. That count comes from 17 estimator-covariate cells: the 7 estimators marked "n.a." under Covariates in Table 1 run once, the 4 marked "No" run twice (without and with a covariate), and the 2 matching estimators run once (they require a covariate, footnote 12). Equivalently, 11 no-covariate runs plus 6 with-covariate runs. Then 17 x 5 outcomes x 51 x 31 = 134,385, consistent with "134,000+" (p.1) and "over 134,000" (p.16). This decomposition is an inference; the paper does not print it. It implies the 134,000 figure counts event studies (each yielding estimates at 25 horizons), which matches the abstract's wording.
- The same event list is reused for every estimator (p.16), so the event draw is made once and estimator comparisons are paired.
- Three estimators tune by cross-validation inside every fit (elastic net synthetic control, generalized synthetic control, matrix completion; pp.13-14). No tuning values, grids, or fold counts are given.
- Runtime, parallelization, random seed, and the sampling distribution over months: Not discussed in paper.

### Relation to Existing diff-diff Estimators

- **Estimator coverage.** Of the 13 evaluated estimators, the library ships classes for eight: TWFE (`TwoWayFixedEffects`), Callaway-Sant'Anna (`CallawaySantAnna`, doubly robust via `estimation_method="dr"`), imputation (`ImputationDiD`), Wooldridge (`WooldridgeDiD`), de Chaisemartin-D'Haultfoeuille (`ChaisemartinDHaultfoeuille`), local projections (`LPDiD`), synthetic control (`SyntheticControl`), and synthetic DiD (`SyntheticDiD`). It does not ship propensity-score-matching-then-TWFE, coarsened exact matching, elastic net synthetic control, generalized synthetic control, or matrix completion estimators.
- **Reviewer note on the paper's single-treated-unit finding.** Five of the eight shipped classes fall in the paper's TWFE-like family, for which Table 2 prints a 0 percent difference from TWFE in the single-treated-unit design. Whether the library's implementations coincide in such a design has not been checked in this review.
- **Existing placebo diagnostics.** `diff_diff/diagnostics.py` already has placebo diagnostics for the basic DiD estimator - individual fake-timing and fake-group placebo fits, a permutation procedure that repeatedly reassigns treatment and returns a permutation distribution, and a leave-one-out procedure that refits after dropping each treated unit: `run_placebo_test`, `placebo_timing_test`, `placebo_group_test`, `permutation_test`, and `leave_one_out_test`, each returning `PlaceboTestResults`, plus `run_all_placebo_tests`, which returns a dictionary of those results (or structured error dictionaries) keyed by test. Existing argument spellings there: `data`, `outcome`, `treatment`, `time`, `unit`, `post_periods`, `pre_periods`, `fake_treatment_period`, `fake_treated_units`, `n_permutations`, `alpha`, `seed`, `**estimator_kwargs`.
- **Existing simulation tools.** `diff_diff/power.py` has `simulate_power`, `simulate_mde`, and `simulate_sample_size` (results `SimulationPowerResults`, `SimulationMDEResults`, `SimulationSampleSizeResults`), which by default draw from an internal per-estimator registry of SYNTHETIC data generators and also accept a custom `data_generator` / `data_generator_kwargs` that bypasses the registry. Existing spellings: `estimator`, `n_simulations`, `alpha`, `effect_sizes`, `seed`, `estimator_kwargs`, `result_extractor`, `progress`.
- **Reviewer note on how the paper positions the two approaches.** The paper's distinction between Monte Carlo simulation (assumed data generating process) and placebo tests (real-world data), which it calls complementary (p.6), corresponds to the distinction between the existing simulation functions run on their default synthetic generators and a placebo-on-the-user's-own-data capability. (Those functions also accept a custom `data_generator`; whether a real-data placebo exercise can be expressed through it is a question for the implementation plan, not settled here.) The paper supplies the placebo MSE summary only; it supplies nothing for the power, minimum-detectable-effect, or sample-size quantities those functions report.
- **Naming constraint.** The library has a standing constraint to reuse existing method / argument / result-field names wherever a concept already exists, so an implementation should start from the spellings listed above. The API itself is to be decided in the implementation plan, not here.

---

## Gaps and Uncertainties

**Metric definition**
- **No printed formula.** MSE and the percent difference are defined in prose only (pp.3, 17); the formulas in this review are reviewer reconstructions.
- **Pooled vs per-event averaging** is not specified (p.17).
- **Whether tau = 0 counts as "post-event-date"** is not specified (p.17).
- **Scope of the footnote 9 decile trimming** (p.20) - that comparison only, or all MSE calculations including Table 2 and Figures 2-4 - is not stated.
- **Units of the MSE values** in Table 2 are not stated (p.19).
- **Table 2 note wording** ("normalized to be a percentage of the TWFE MSE") differs from the row-group header and the text ("Percent difference relative to TWFE", "MSE difference relative to TWFE", p.17). The printed values support the difference reading (reviewer note).
- **"30-60 percent lower"** (p.17) vs a Table 2 cell of -65 (elastic net synthetic control, HPI, p.19): minor mismatch between text and table, not addressed by the paper (reviewer note).
- **No uncertainty on any MSE number** and no test of MSE differences.

**Level inconsistency between exhibits**
- **Figure 3 vs Table 2 (reviewer note, visual read).** The no-covariate bars for Callaway and Sant'Anna, local projections and Wooldridge in Figure 3 (p.24) sit at about 1.17, whereas Table 2 (p.19) reports 1.57 for TWFE with a 0 percent difference for those three estimators on the same outcome (seasonally adjusted LAUS employment growth), and the seasonally adjusted bars in Figure 2 (p.22) sit at about 1.57. The paper gives no explanation. Candidate explanations, such as the footnote 9 trimming being applied in one exhibit and not another, are NOT stated by the paper. Absolute MSE levels should not be relied on until this is checked against the authors' materials.

**Counts**
- **205,530 vs 134,385 (reviewer arithmetic).** The literal Cartesian product on p.16 gives 205,530; the stated "over 134,000" (a lower-bound phrase, which 205,530 also satisfies) is closely matched by 17 estimator-covariate cells x 5 x 51 x 31 = 134,385. The paper does not print the breakdown. See Computational Considerations.
- **What the 134,000 counts.** The abstract says "134,000+ state-level placebo event studies" (p.1); Section 3.5 says "over 134,000 placebo event study estimates" (p.16). The arithmetic above matches a count of event studies, not of horizon-level point estimates.

**Placebo design**
- **Whether the 31 sampled year-months are shared across all states** is not explicit. "Cartesian product" and "the same set of randomly selected 'events'" (p.16) read as one shared list; "randomizing placebo events in each state and year" (p.6) does not settle it.
- **High-level vs exact description.** Section 3 describes "randomly drawing an 'event'" from "the set of all possible events" (p.7), while Section 3.5 enumerates all states and all calendar years and randomizes only the month within the year (p.16).
- **24-month edge exclusion vs a 12-month-each-side window.** Footnote 6 (p.16) gives only the quoted reason and calls the window "the 24-month window"; why 24 months are dropped at each end is not explained.
- **Estimation sample per placebo** (full 1990-2024 panel vs a window around the event) is not stated. `T` is never assigned a value in an equation, including the pre-period limit of the synthetic difference-in-differences sum (p.14).
- **Control pool composition** is never formally defined (pp.3, 4, 7, 11, 13).
- **Balanced vs unbalanced panel, random seed, sampling distribution over months:** not stated.
- **Sensitivity to the number of placebo events:** not examined; the 31 x 51 grid is fixed.

**Data**
- **Growth-rate definitions** (1-month vs 12-month, log vs percent) are not stated (p.15).
- **Construction and temporal alignment of the QCEW employment-growth outcome** within the monthly panel are not described (p.15). "Quarterly" appears in the program's name (Quarterly Census of Employment and Wages); the paper does not separately state the frequency of the series it uses.
- **Seasonal adjustment status** is given only for LAUS (seasonally adjusted, p.15) and QCEW (not seasonally adjusted, Table 2 note, p.19); it is not stated for the unemployment rate, labor force participation rate, or house price index. The adjustment method is not described.
- **Foreign-born share source:** not stated where it is introduced (p.15); Section 5.3 lists it among covariates "each calculated from the 1990 Census" (p.23).
- **Introduction vs Section 3.5 outcome lists:** the introduction names "the unemployment rate, payroll employment, and a house price index" (p.3); Section 3.5 lists five outcomes (p.15).

**Estimator descriptions**
- **Table 1 "Covariates" column.** Callaway and Sant'Anna, Wooldridge, local projections and synthetic control are marked "No" in Table 1 (p.8) yet accept covariates in Section 5.3 (p.23). Reviewer reading: per the table note the column describes the baseline specification, so "No" means not used at baseline and "n.a." means not accepted (footnote 5, p.15). The paper does not spell this out.
- **Propensity score model:** functional form / link not stated beyond the printed equation (p.12).
- **Synthetic control:** which pre-treatment outcomes or lags enter `X` is not stated; the norm has no weighting matrix (p.13).
- **Cross-validated estimators:** no scheme, grid, fold count, or tuning value is given for elastic net synthetic control, generalized synthetic control, or matrix completion (pp.13-14).
- **Printed-equation oddities** (local projections left-hand side, missing `tau` on the de Chaisemartin and D'Haultfoeuille `beta`, `F_hat_t` subscript, `w_i` vs `w_j`) are transcribed as printed and not reconciled by the paper (pp.11, 14).
- No abbreviation is given for generalized synthetic control in the figure notes.

**Evidence base for the secondary findings**
- **Normalization:** shown for one example (synthetic control x seasonally adjusted LAUS employment growth, Figure 1); other combinations are asserted representative "although the magnitude of gains may vary" (p.20).
- **Seasonal adjustment:** one outcome only (footnote 10, p.21).
- **By-state patterns common across estimators:** rests on "unreported results" (p.25).
- **Control-weight differences:** explanation is hedged as "likely" (p.27).
- **Residual heterogeneity:** "the significance of heterogeneous economic dynamics across units, which the analyst often cannot fully account for" (p.29).

**Not developed by the paper**

*Each item below means the paper gives no procedure, formula, or evidence for it. It does not mean the words never occur: passing mentions are noted where found.*
- Staggered adoption or multiple simultaneously treated units (footnote 1, p.3, defers this).
- No formal validity analysis of placebo-based inference, no power or detectable-effect calculation, and no experiment with an injected non-zero effect. Footnote 2 (p.6) does state one limitation of the null-variance interpretation - it holds "only for studying policies that do not change during our sample period" - and suggests excluding the treated units.
- Interference, anticipation, serial correlation, or exchangeability of units.
- Time-varying covariates (footnote 11, p.22, defers this).
- Unit types other than states, non-monthly frequencies, event windows other than 12 months each side.
- Generalization beyond the stated setting: "our placebo tests can compare estimator performance in only a specific setting: state-level labor market event studies in the United States." (p.6)

**Software and replication materials**
- No software package, function, or programming language is named in printed pp. 1-33, and there is no appendix after the bibliography (pp.30-33).
- No code availability statement, data availability statement, repository URL, or replication-package reference appears on the cover sheet or in printed pp. 1-33. Data sources are named in Section 3.5 (the outcome list, p.15), in Section 5.3 (covariates "each calculated from the 1990 Census", p.23), and in exhibit source lines ("Bureau of Labor Statistics; Census Bureau; Federal Housing Finance Agency").
- The bibliography entries (pp.30-33) were not transcribed; citations in this review are given in the short form the paper uses in its text.
