# Paper Review: Designing Experiments with Synthetic Controls

**Authors:** Nick Doudchenko (Google), David Gilinson (Facebook), Sean Taylor (Lyft), Nils Wernerfelt (Calibra, Facebook)
**Citation:** Doudchenko, N., Gilinson, D., Taylor, S., & Wernerfelt, N. (2019). Designing Experiments with Synthetic Controls. Working paper.
**Source reviewed:** the 21-page working-paper PDF hosted by the Mack Institute (Wharton) ([PDF](https://mackinstitute.wharton.upenn.edu/wp-content/uploads/2020/03/Wernerfelt-Nils-Doudchenko-Nick-Gilinson-David-and-Taylor-Sean_Designing-Experiments-with-Synthetic-Controls.pdf)): title page and abstract (p.1), text (pp.2-19), references (pp.20-21). The PDF carries no date on its pages; its file metadata gives a creation date of December 20, 2019, and Doudchenko et al. (2021) cite it as "(2019) ... Working paper"; Abadie and Zhao (2026) cite it as "n.d.". The PDF was reviewed externally and is not committed to the repository. All page citations are the page numbers printed in the PDF. References in the paper are numbered; bracketed numbers in quotes are the paper's own. Inside quoted passages the paper's dashes are written as spaced hyphens, its typographic apostrophes as straight ones, and its plus-or-minus sign as `+/-`.
**Review date:** 2026-11-27

> **Personal verification stamp.** The reviewer read the full 21-page text and checked directly against it: the Lasso weight problem (Equation 3, p.5); the description of the confidence intervals (p.5); the one-region power procedure (p.6); the simulated-annealing procedure and the sentence on the objective and on discarding high-error countries (pp.7-8); the three external-validity options (p.8); the three pragmatic constraints (p.9); the `power.analysis` arguments and printed output (pp.10-12); every cell of Table 1 and its caption (p.18); and the Section 5 numbers (pp.17-18). Terms reported as absent were searched in the PDF text layer.

---

## Methodology Registry Entry

**Status: proposed text for a possible future REGISTRY section; no such section exists in `docs/methodology/REGISTRY.md` today; this file is a non-authoritative source audit.** The paper is a candidate source for a planned unit-selection capability: it is the written source for choosing test units by simulated power on historical data. Nothing below is a methodology contract until a REGISTRY section is written and reviewed alongside an implementation. The registry-candidate text ends at the marker before `## Results`.

*This is a procedure paper with one numbered optimization problem (Equation 3) and no theorem. The standard headings are replaced by the procedure, its objective, and the evidence.*

## Unit selection by simulated power (working title)

**Primary source:** Doudchenko, N., Gilinson, D., Taylor, S., & Wernerfelt, N. (2019). Designing Experiments with Synthetic Controls. Working paper.

**Key implementation requirements:**

*Scope of the source:*
- The question: "given a set of possible test units, how can a researcher select the best one(s) to experiment on?" (abstract, p.1). The paper describes an approach "leveraging simulated interventions and permutation tests across candidate test units" (abstract, p.1).
- Setting: few treated aggregate units because "such large scale interventions are quite costly, meaning that only a few markets may be able to get the treatment and the researcher has to select which ones" (p.2).
- Stated criteria for a good choice (pp.2-3): "if the aim of the experiment is to assess the intervention for a potential global launch, the selected units should be representative of the broader population"; "if the intervention does affect the outcome of interest at the aggregate level, the experiment would maximize the probability of detecting it; similarly, if the intervention has no effect, the experiment should minimize the probability of detecting one."
- The estimator is taken as given: "instead of optimizing features of the synthetic control itself, we aim to optimize the overall observational study conditional on a preferred synthetic control methodology" (p.5).

*Estimator used in the paper's implementation (Section 3.1, p.5).* For a candidate test unit with outcome `y_t` over training periods `t = 1, ..., T0` and `p` potential control units with outcomes `x_tj`:

    beta_hat^Lasso = argmin over beta of  (1/2) * sum_{t=1..T0} ( y_t - beta_0 - sum_{j=1..p} x_tj beta_j )^2  +  lambda * sum_{j=1..p} |beta_j|        (Equation 3)

Equation 2 (p.4) gives the generic synthetic-control estimate, the treated outcome minus the weighted control outcomes, `tau_hat_1t = Y_1t - sum_{j=2..J+1} w_j Y_jt`. Reviewer reconstruction from Equation 3 for the Lasso implementation: the fitted counterfactual includes the fitted intercept, `beta_hat_0 + sum_j x_tj beta_hat_j`, and is subtracted from the observed treated outcome. In Equation 3 the weights are not constrained to be non-negative or to sum to one. "In our R package, we somewhat arbitrarily select the model from Lasso where we restrict the maximum number of coefficients to be 12." (p.5). Two reasons are given: "This number was chosen as empirically it performed well with internal Facebook data, but more importantly" the procedure "runs a number of computationally expensive calculations to both conduct power calculations and analyze tests" (p.5).

*Confidence intervals (p.5):* "we generate the confidence intervals via permutation over past dates. Specifically, for a researcher-specified training window, we retrain Lasso (and, as a result, the controls and weights for a given test unit will vary across each iteration) starting at all possible historical dates, and then compare the predicted value against the observed data at each horizon. This generates an empirical distribution of residuals at each horizon, which is then used to produce confidence intervals." The authors attribute validity to Chernozhukov, Wuthrich and Zhu ("[14] provide formal conditions under which confidence intervals generated in this manner will be exact", p.5). The paper gives no argument of its own that residuals from fits re-trained at each historical date meet those conditions. Reviewer note: such residuals are not exchangeable in general (a one-period rolling mean, for instance, gives residuals `Y_t - Y_{t-1}`, which are correlated for adjacent dates even when the outcomes are independent over time), so the interval needs its own calibration argument, and the rejection rate at a zero-size effect is best read as a historical diagnostic of the whole procedure and not as a guaranteed size.

*Power for one candidate unit (Section 3.2.1, p.6).* Distilled from the prose; the paper prints no numbered algorithm.

1. Take the historical series of the metric in the candidate unit.
2. "select a random date and take all data after that day and shift it up by a specified effect size". The example shifts all data after `t = 550` up by 20%.
3. Run the synthetic-control procedure on the shifted data to get a forecast counterfactual and a confidence interval: the fit uses the pre-launch training window of the candidate unit and of the untreated potential controls, and the controls' post-launch outcomes give the counterfactual (Section 2, p.4). Units in a proposed treated set are not available as controls ("as the number of test units gets large, there are both fewer possible control countries", p.17). The paper does not say which historical dates' residuals calibrate the interval for a given simulated launch. Reviewer note: for the replay to mimic a real launch, those residuals have to come from forecast windows that end before the simulated launch, so that injected outcomes do not enter the reference distribution used to detect them. The target statistic is fixed beforehand: in the worked example it is "the average effect from launch to day 21" (p.11; "The 'cumulative' comes from averaging the effect over the whole horizon", footnote 12, p.12); a `min.horizon` moves the start of the window, and the approach can also target the effect at a specific period (footnote 3, p.4).
4. "If the shifted data falls outside the confidence interval, then we flag that this effect size was detected in this potential test unit."
5. "We repeat this procedure many times - over different dates and effect sizes - and use the average empirical rejection rates of the null hypothesis to generate both a power curve and a reported Type 1 Error Rate."

The effect is multiplicative in the package ("Currently, our R package only supports multiplicative interventions", footnote 6, p.6); "if the researcher believes the true effect is not modeled as a level shift but instead some other functional form (e.g. a trend shift), the procedure can be reproduced assuming such an intervention" (p.6). The horizon may be a sub-window after launch, for example days 21 to 51, to skip novelty effects (footnote 4, p.6).

Outcome: "this approach can be used to empirically rank regions by the power and Type 1 error rates they will have. This is useful if we want to launch in a single region and believe the true effect is homogenous." (p.6)

*Several treated units (Section 3.2.2, pp.7-8).* "When combining treatment effect estimates across regions, the combined variance must take into account the covariance across regions. From an ex ante perspective, this makes selection of test regions an optimal subset selection problem, which is computationally very involved." (p.7). For a pre-specified set no search is needed: "analyzing each country individually and then estimating the combined power still yields valid results; the point here is that there might exist another subset with better experimental properties" (p.7). The combination rule (p.17): each unit is "an equally weighted observation. The combined average treatment effect is the average of that from the treated units, and the variance is given by their sums, including the covariance terms." Reviewer reconstruction of the scale: for `m` equally weighted unit estimates the variance of their mean is `( sum_i Var(tau_hat_i) + 2 sum_{i<j} Cov(tau_hat_i, tau_hat_j) ) / m^2`. The paper gives the rule in prose only and does not say how the covariances are estimated.

Search by simulated annealing (p.7), implemented as `experimental.unit.set.search`:

1. Choose a temperature `T_0 > 0`, a cooling factor `kappa < 1`, and "an objective function f (e.g. the minimal detectable effect obtained in a power analysis for a set of countries)".
2. "Start with an initial guess of the set of test units", `S_0^test`, with `f_{-1}` set to infinity.
3. At iteration `j = 0, 1, 2, ...`, evaluate `f_j` and `delta_j = f_j - f_{j-1}`. Switch to the candidate set with probability 1 if `delta_j <= 0` and `exp(-delta_j / T_j)` if `delta_j > 0`.
4. Set `T_{j+1} = kappa * T_j` and propose a new candidate set `S_{j+1}^test` by "randomly replacing one unit".
5. Repeat "until the maximum number of iterations is reached".

*The objective and the screen, as implemented (p.8):* "In our implementation of this package, we throw out countries that have high Type 1 Error Rates and set the objective function to minimize as the width of the power curve at 80%. This is due to the fact that empirically the kinds of interventions we have studied are better to estimate imprecisely than with bias. However, any objective function could be plugged into this simulated annealing process to be maximized (e.g. MSE, minimum detectable effect, etc.)."

That is: the package discards countries with a high empirical Type I error rate and then searches over subsets of the remaining countries to minimize "the width of the power curve at 80%". Reviewer interpretation: with `e_L` and `e_R` the negative and positive effects at which power reaches 80%, the width is `e_R - e_L`. The annealing passage states no cutoff for "high"; the worked example's output uses 5% ("This is OK compared to the acceptable rate of 5%", p.12). Section 5 separately defines the MDE it reports as the average of the absolute values of the left and right interpolated effects, `(|e_L| + |e_R|) / 2` (Table 1 caption, p.18), which is half the width when `e_L <= 0 <= e_R`, as in the worked example; the paper notes the curve can be off-center when the historical predictions are biased (p.12).

*How the false-positive rate is computed (p.12):* "the Type I error rate comes from launching fake interventions of different sizes on historical data and reporting the empirical rejection rates for the 95% confidence interval at the horizon of interest. If this number is above 5%, the experimenter should be worried that the proposed country-metric pair is biased at that horizon and consider alternatives." The power curve is read for both signs and need not be symmetric: "due to the permutation test, there is no guarantee the upper and lower envelopes of the power curve are symmetric" (p.12); "bias in the historical predictions will manifest in an off-center curve" (p.12). In the worked example the effect grid includes zero and the rejection rate at a zero-size effect agrees with the printed 0% Type I error rate; the paper does not give the general calculation (see Gaps).

*Which dates are simulated (p.11):* the power analysis uses `n.simulated.launch.dates` start dates taken from the most recent data: "It leverages the most recent possible data for this (i.e., starting at max(wide.dt[ds < launch.date]$ds) - max.horizon and including the previous n.simulated.launch.dates days)." Footnote 10: "We commonly use between 30 and 90 days; if the answers are very different between these, that suggests there may be a lack of stationarity in the data, which is concerning." The simulated dates are consecutive days, so the simulated experiments overlap. The printed example reports "30 days between 2017-10-26 and 2017-11-24" (p.12). The paper does not show the last pre-launch date, so the example settles neither whether the rule yields `n` or `n + 1` dates nor which horizon (the 21 days of the call or the 31 of the printed output) produced it.

*External validity (Section 3.3, p.8).* Three options "we have seen discussed in practice":
1. Choose the launch time and the unit at random ("in theory that would provide valid inference for the question of a global launch"). "empirically we have found massive heterogeneity in confidence interval size across candidate regions. Hence, choosing a region purely at random, while reducing the bias, likely has a massive variance effect, making it suboptimal from an MSE perspective." "empirically we are clearly limited in terms of choosing the timing at random, and it may well be worth it to run the same experiment at different time periods to understand if there is any heterogeneity across time".
2. Stratify on observables along which effects are believed to differ, "optimize within each strata and then reweigh accordingly"; in the paper's example the strata estimates are weighted "according to the share of markets that are rainy versus sunny", i.e. by the strata's shares of the target population.
3. "partial randomization, whereby the experimenter first limits the set of possible countries down, and then randomizes within them ... If the experimenter feels the treatment effect does not correlate with the power too substantially, then doing a first pass screen based on power and a randomization within that set may be optimal from an MSE perspective." The paper notes that exogenous constraints usually "put a substantial fraction of the world off limits for the experimenter, thereby already being a threat to estimating a global average treatment effect".

*Pragmatic constraints (Section 3.4, p.9):* (i) "some regions may be off limits to experiment in"; (ii) "SUTVA violations caused by underlying market structure", which is "very market and metric-specific" and is handled by "excluding as potential controls markets that we believe are likely contaminated" (in the authors' country tests such exclusions had "a minimal effect on power"); (iii) "time shocks - either historical or in the future - that are outside of the experimenter's control", for which "the best an experimenter can do is research possible future events in different candidate regions, check historical data for such anomalies, and then drop outliers, winsorize, use moving averages, or remove candidate test or control units as appropriate". At design time the paper also recommends "taking the most recent data, and plotting for each potential test country the trends in it and the control units", noting that "the control units (and weights) for a given country on one day will not necessarily equal those on another day" (p.13).

*Edge cases the paper addresses:*
- Many treated units: "if the option exists to experiment on half of all countries in the world, then a matching analysis or pure randomization may both provide better power and more reliable estimates" (p.17). Reasons: fewer controls remain, and controls shared by several test units raise "the risk of a correlated shock" (p.17).
- Minimum detectable effects beyond the simulated grid are returned as NA (p.18).
- Missing data: none allowed in the input; "we recommend using interpolations or only starting the historical data after problematic stretches" (footnote 9, p.10).
- Non-stationarity: varying the number of simulated launch dates is offered as a sensitivity check; if the answers are very different, "that suggests there may be a lack of stationarity in the data, which is concerning" (footnote 10, p.11); relatively stable power curves are described as reassuring (p.13).

**Reference implementation(s):** the R package `countrytestr`, described as "being released in conjunction with this paper" (p.3), with functions `power.analysis`, `experimental.unit.set.search`, `countrytestr`, `control.plot` and `unit.plot`. The paper gives no repository URL. Lasso is run through `glmnet` (p.10).

### What the paper does NOT provide

- **No evaluation, in Section 5, of the selected unit on data other than those that selected it.** In Section 5 the selected country's MDE, bias and false-positive rate are the values from the same `power.analysis` runs that selected it (p.17). The separate Section 4 example does set one actual launch against its design-time prediction: the design step uses only pre-launch data (p.11), and afterwards "the post-launch confidence interval size is consistent with that predicted by the power calculation" (p.14). The text layer has no match for "holdout", "hold-out", "validation", "cross-validation", "split", "overfit" or "out-of-sample". The paper does not discuss whether the reported power of the selected unit is optimistic because it was selected.
- **No formal statement of the objective or of the screen.** The annealing objective and the false-positive screen are described in the one passage quoted above (p.8). The procedure is given in form (geometric cooling, an initial guess, a maximum number of iterations; p.7), but not the cutoff for the screen, the numerical values of `T_0` and `kappa`, the iteration cap, how the initial set is chosen, or any restart policy, and there is no evidence on how close the search gets to the best subset.
- **No theory.** No theorem, no consistency or validity result of its own; validity of the intervals is by reference to [14] ("these are based on historical data and permutation tests in such settings are only valid under conditions laid out in [14]", p.13).
- **No evaluation of the multi-unit search.** Section 5 evaluates the single-unit case only.
- **No difference-in-differences estimator.** The estimator is a Lasso synthetic control. "parallel trends" is used for the visual check of test and control series ("to see if anything suspicious is happening that might violate the parallel trends assumption", p.13; "This helps provide assurance on the parallel trends assumption", p.16).
- **No formal cost or budget constraint in the search** (search: "budget" has no match; "costs per intervention" and "costly" appear as motivation on p.2, and p.9 describes the decision to open an otherwise off-limits region as "a cost-benefit analysis at the firm level").
- **No covariates in the estimator** (search: "covariate"; no match): the Lasso uses the other units' outcomes as predictors. Observable characteristics enter in two other places: Section 3.3, as a way to stratify the choice of units (p.8), and Section 3.4, where markets believed to be contaminated are excluded as potential controls after "looking at our metrics of interest" (p.9).
- **No random seed or reproducibility statement for the simulations**; country names, dates and the metric are anonymized or rescaled (footnote 7, p.9).

---

**The registry-candidate text ends here. Everything below is audit material.**

## Results

### Section 5 (pp.17-18): single test country, 14-day horizon, 101 candidate countries

Design: "a multiplicative treatment effect that is homogenous across countries"; the authors ran `power.analysis` for each of 101 candidate countries and recorded, for each, the false positive rate (FPR), the interpolated upper and lower bounds of the power curve and the average bias (p.17). The number of simulated launch dates used in this section is not stated. The power analysis "was run with effect sizes ranging from -10% to +10%, meaning that MDE's greater than 10% in absolute value were entered as NA values ... the NA's were replaced with +/-10%", so the gains "are actually underestimates" (p.18). Reviewer reconstruction: a missing left effect is set to -10% and a missing right effect to +10% before the absolute values are averaged; the paper does not say how a country with only one missing effect was handled.

Table 1 (p.18). Columns to the left of the double line are statistics across all countries; columns to the right are the values for one country chosen by the stated rule.

| | Average | Min | Max | Min(MSE) | Min(MDE given FPR <= 5%) | Min(Abs(Bias) given FPR <= 5%) |
|---|---|---|---|---|---|---|
| FPR | 0.0422 | 0 | 0.5 | 0 | 0 | 0 |
| Abs(Bias) | 0.0135 | 0.0001 | 0.1001 | 0.0034 | 0.0034 | 0.0001 |
| MDE | 0.0613 | .031 | 0.1 | 0.0363 | 0.0325 | 0.0567 |

Caption: "Analysis of False Positive Rate (FPR), bias, and Minimum Detectable Effects (MDE) at 80% power across 101 countries with Facebook data ... The MDE was calculated by taking the average of the absolute value of the left and right interpolated effects from power.analysis."

Table 1 has a `Min(MSE)` column, but the paper gives no formula for the MSE or its inputs (the caption defines only the MDE).

Authors' reading (p.18):
- "the expected MDE and bias from selecting a random country are 6.13% and 1.35%, respectively. In contrast, selecting the country with the lowest MSE has MDE and bias of 3.63% and 0.34%. (The FPR also decreases from 4.22% to 0%.) Hence, leveraging this approach in this example, we can reduce the size of the MDE by 41% and the bias by 75%."
- "19/101 countries had FPR's greater than 5%. In other words, selecting a country at random to experiment in will yield a country with an overly high FPR about 20% of the time. This is of course conditional on our method of analysis, but we think still is informative about risks from synthetic control analyses where the unit is exogenously set."
- In the Introduction: "close to 20% of the countries had false positive rates in excess of 5% (with some as large as 50%), which we interpreted as a cautionary note against analyses where the experimenter has to take the treated unit as a given." (p.3)

Reviewer arithmetic: (0.0613 - 0.0363) / 0.0613 = 0.408, which the paper rounds to 41%; (0.0135 - 0.0034) / 0.0135 = 0.748, which the paper rounds to 75%.

These comparisons are in-sample: the selected country's MDE, bias and FPR are the values from the same power analysis that selected it. The paper reports no figure for the selected country on later data in Section 5.

### Worked example (Section 4, pp.9-16)

- Power analysis for one country over 30 simulated launch dates between 2017-10-26 and 2017-11-24: 80% power for effects of about -5% (-4.545% interpolated) and +5% (4.429% interpolated) at 95% confidence, Type I error rate 0% (p.12).
- The printed output reads "Test of cumulative ATE from horizon 0 to 31" (p.12) although the call shown sets `max.horizon = 21` (p.10) and the text describes a 21-day window (pp.11-12). See Gaps.
- Post-launch analysis with `max.horizon = 31`: "Average Estimated Treatment Effect: -13.7% [-17.3%, -10.2%] with 95% confidence" (p.14); "the post-launch confidence interval size is consistent with that predicted by the power calculation" (p.14). The interval half-width is about 3.5 percentage points (reviewer arithmetic from the printed interval).
- Control weights for the example unit, "generated by the Lasso run at launch day" (p.16): eleven controls, the largest 30.3%; the printed weights sum to 100% (reviewer arithmetic: 30.3 + 13.85 + 8.94 + 8.82 + 8.2 + 7.18 + 7.14 + 6.75 + 4.32 + 3.15 + 1.35 = 100.0).

---

## Implementation Notes (audit notes - NOT registry-candidate)

### Data Structure Requirements
- A wide table sorted by a date column (`ds`): one row per period (daily in the example, from June 2015), one numeric column per unit, with "no missing, NA, or infinite values" (p.10). For the design step the example restricts the table to dates before the launch (`wide.dt[ds < launch.date]`) "to mirror the information set we would have in designing the experiment" (p.11). No standardization is needed: `glmnet` "standardizes the variables for us" (p.10).
- Design call (p.10): the pre-launch table, the test units, the number of simulated launch dates, the horizon (`max.horizon`) and a grid of effect sizes. Post-launch call (p.14): the full table, pre- and post-launch, with an explicit launch date (`launch.ds`), the test units and `max.horizon`. Both calls accept a `min.horizon` to restrict the window (pp.11, 14).

### Computational Considerations
- Each candidate is evaluated across simulated dates and effect sizes ("the runtime is largely determined by the number of effect sizes and simulated launch dates", p.11), and the interval construction retrains the Lasso "starting at all possible historical dates" (p.5). The paper gives no count of distinct fits and does not say which are reused. One call "will run for a few minutes" (p.11); speed is the stated reason for capping the Lasso at 12 coefficients (p.5).
- Subset search repeats the objective evaluation at every annealing iteration; no runtime is reported for it.
- Finer effect grids "take longer but also give more precise output since we interpolate between entered effects to estimate the exact power" (p.11).

### Tuning Parameters

| Parameter | Type | Setting reported in the paper | Selection method |
|---|---|---|---|
| Lasso `lambda` / model size | penalty | number of coefficients capped at 12; how the fit is chosen among eligible ones is not stated | "somewhat arbitrarily" (p.5) |
| training window | periods | "researcher-specified" (p.5); no value appears in the example calls | analyst |
| `n.simulated.launch.dates` | integer | 30 in the example; "between 30 and 90" in practice (footnote 10) | compare results across values as a stationarity check |
| `effect.sizes` | grid | -10% to +10% in steps of 1% in the example call | analyst |
| `max.horizon`, `min.horizon` | days | 21 in the example call, 14 in Section 5 | analyst; a burn-in window is suggested |
| acceptable Type I error rate | proportion | 5% in the worked example's output | not discussed |
| annealing `T_0 > 0`, `kappa < 1`, iteration cap, initial set | - | form given (p.7), values not given | not given |
| annealing objective | function | width of the power curve at 80% | "any objective function could be plugged" (p.8) |

### Relation to Existing diff-diff Estimators
- The paper takes "a preferred synthetic control methodology" as given (p.5) and illustrates with a Lasso fit and intervals from historical dates, whose validity the authors rest on the conditions of [14] (pp.5, 13). Reviewer note: the steps of the power procedure themselves (move a fake launch date through history, inject an effect, record whether the null of no effect is rejected, i.e. whether the shifted data fall outside the interval around the forecast) do not depend on the estimator being a synthetic control. The paper itself suggests the same exercise for matching: "evaluating matching techniques akin to what we are doing in power.analysis - apply to several different historical dates and see what the empirical power has been" (p.17).
- `diagnose_design` (`diff_diff/design.py`) injects an effect into placebo-treated observations and reports, per estimator, a rejection rate under no effect and optionally a minimum detectable effect. It differs in what is varied: `diagnose_design` draws the placebo-treated units at random and assigns them the declared adoption period (when several periods are declared, each placebo-treated unit draws its own period from the list independently), whereas this paper fixes the candidate unit and moves the fake launch date.
- The paper's screen-then-minimize rule (discard countries with a high empirical false-positive rate, then minimize the width of the power curve at 80%) has the same two ingredients as the library's per-estimator size check followed by a minimum detectable effect for estimators that pass it. In the paper the screen is applied to candidate countries; in the library, to estimators.

---

## Gaps and Uncertainties

1. **Date and status.** Undated working paper; the year 2019 comes from file metadata and from citations in later papers. No journal version is known to the reviewer.
2. **Selection and evaluation on the same data.** The reported MDE, bias and FPR of a selected unit come from the history that selected it. The 41% and 75% comparisons therefore reuse the simulations that informed the selection. Reviewer inference: this can make the selected unit look better than it would on later data; Section 5 does not establish the direction or size of any such discrepancy for the reported metrics.
3. **Overlapping simulated experiments (reviewer inference).** The worked example reports 30 consecutive daily launch dates; the call and Figure 2 give a 21-day horizon and the printed output gives horizon 0 to 31. Under either, nearby simulated launches share outcome observations and their rejection indicators may be dependent. The paper gives no effective sample size and reports no uncertainty for the empirical false-positive rate. With 30 scored dates and one rejection indicator per date the rate moves in steps of 1/30. Section 5 does not state how many dates it used.
4. **How the Type I error rate is formed.** It is described as coming from "fake interventions of different sizes" (p.12). The worked grid (`-10:10 / 100`) includes zero; the printed rate is 0% (p.12), and Figure 2 (p.13) plots the power curve for that example, whose value at a zero-size effect is the rejection rate under no effect. The prose does not give the general calculation or say that zero must be in the grid.
5. **Horizon mismatch in the printed output** (21 in the call and text, 31 in the output on p.12, 31 in the post-launch call on p.14).
6. **Annealing details.** Geometric cooling, an initial guess and a maximum number of iterations are given; the numerical values, the choice of the initial set, restarts and any evidence of search quality are not. The acceptance rule as printed compares `f_j` with `f_{j-1}`; whether the best set seen is retained is not stated. The rule treats smaller values as better; the text also speaks of other objectives "to be maximized (e.g. MSE, minimum detectable effect, etc.)" (p.8), which have to be read as losses to be minimized for the printed rule to apply (reviewer interpretation). Nor does the paper say from which set the next one-unit replacement is drawn after a rejected proposal, or whether `f_{j-1}` then refers to the last evaluated set or to the retained one.
7. **Interpretation of a Type I error rate above 5%.** The authors treat it as a reason to "be worried that the proposed country-metric pair is biased at that horizon and consider alternatives" (p.12). Reviewer note: intervals that are too narrow would also raise the rate; Section 5 records the rate and the average bias as separate quantities (p.17).
8. **External validity is discussed qualitatively only.** None of the three options is implemented or evaluated in the paper, and representativeness is not part of the objective.
9. **The package.** `countrytestr` is described but no location is given; its current availability was not checked.
