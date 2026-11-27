# Paper Review: Trimmed Match Design for Randomized Paired Geo Experiments

**Authors:** Aiyou Chen, Marco Longfils, Nicolas Remy (Google LLC)
**Citation:** Chen, A., Longfils, M., & Remy, N. (2021). Trimmed Match Design for Randomized Paired Geo Experiments. arXiv:2105.07060.
**Source reviewed:** arXiv:2105.07060v2 ([PDF](https://arxiv.org/pdf/2105.07060v2); dated May 21, 2021 on the first page; arXiv stamp May 19, 2021), a 19-page PDF: text pp.1-17, references pp.17-19. The PDF was reviewed externally and is not committed to the repository. All page citations are the printed page numbers, which coincide with the PDF page numbers (page 1 is numbered in its footer, later pages in the header). Mathematical symbols are transliterated to ASCII (`theta`, `lambda`, `eps`, `sum_i`); inside quoted passages the paper's dashes are written as spaced hyphens and footnote markers are omitted.
**Review date:** 2026-11-27

> **Personal verification stamp.** The reviewer read the full text and checked directly against it: the hypothesis (1.1) (p.2); the model (3.1)-(3.3) (pp.4-5); the trimmed mean equation and the simplified variance (pp.5-6); the loss (3.4) and the distance (4.1) (pp.6-7); Algorithm 1 (p.8); the minimum-detectable-effect relation (4.2) and Remark 2 (p.9); the two statements on reusing pairing data (pp.3, 9); the five-step summary (p.10); the simulation data generation and configuration (pp.10-11) and every number quoted from Sections 5 and 6 (pp.11-16). Terms reported as absent were searched in the PDF text layer.

---

## Methodology Registry Entry

**Status: proposed text for a possible future REGISTRY section; no such section exists in `docs/methodology/REGISTRY.md` today; this file is a non-authoritative source audit.** The paper is a candidate source for a planned unit-selection capability: it is a written source for building a design on one part of the pre-treatment data and computing its power on another. Nothing below is a methodology contract until a REGISTRY section is written and reviewed alongside an implementation. The registry-candidate text ends at the marker before `## Results`.

## Matched-pair design with held-out power (working title)

*The paper's own name is "Trimmed Match Design (TMD)" (abstract, p.1).*

**Primary source:** Chen, A., Longfils, M., & Remy, N. (2021). Trimmed Match Design for Randomized Paired Geo Experiments. https://arxiv.org/abs/2105.07060

**Key implementation requirements:**

*Scope of the source:*
- Randomized paired geo experiments: "experimental units are non-overlapping ad-targetable geographical areas" (abstract, p.1). Pairs are constructed from data; treatment is then randomized within each pair.
- Stated difficulties (abstract, p.1): "1) the number of geos is often small, 2) the response metric (e.g. revenue) across geos can be very heavy-tailed due to geo heterogeneity, and furthermore 3) the response metric can vary dramatically over time."
- Estimand: the incremental return on ad spend,

      theta = sum_{g in G} (R^T_g - R^C_g) / sum_{g in G} (S^T_g - S^C_g)

  with `R` potential responses and `S` potential ad spends under treatment and control (p.2). It is a ratio of two causal effects, not a mean effect.
- Hypotheses (Equation 1.1, p.2): `H0: theta = 0` against `H1: theta >= theta_0`, with `theta_0` "a user-specified minimum detectable iROAS".
- Contributions as stated (p.3): "formulation of the design problem under a nonparametric iROAS model, and proposal of the use of Trimmed Match and cross validation (a technique commonly used in model fitting and machine learning, but not for experimental design yet) for robust power analysis."

*The central statement on reusing data (p.3):* "By cross validation, we split the pretest data into two non-overlapping time periods, where one is used for constructing matched pairs and the other for power analysis. If the same data is used for both pairing and power analysis, it can lead to overestimating the power of the design due to overfitting."

And again (p.9): "It is important to emphasize that if the pairing data during T_p is reused for the RMSE evaluation, RMSE may be under-estimated, similar to the overfitting problem in machine learning. On the other hand, with cross validation, the RMSE evaluation can give a reliable assessment as long as the distribution of the underlying responses and spends during the test period is similar to that during the evaluation period. When there is a systematic difference between the evaluation period and the test period, say due to the business growth, one may need to apply some adjustment to RMSE."

*Model (Section 3.1, pp.4-5).* With `n` pairs, `Y_i` the response difference and `X_i` the spend difference (treatment geo minus control geo) in pair `i`:

    Y_i = theta * X_i + eps_i,   eps_i symmetric around 0            (Equation 3.1)

`theta` is assumed the same across geos (p.4). With `Z_ij = R_ij - theta * S_ij` the "uninfluenced" response of geo `j` in pair `i` (Equation 3.2) and `A_i` in `{1, -1}` the within-pair assignment with probability 1/2 each,

    Y_i - theta * X_i  ==  A_i * (Z_i1 - Z_i2)                       (Equation 3.3; `==` stands for the paper's identity sign)

"where both Z_i1 and Z_i2 are invariant to geo assignment" (p.5); the paper writes (3.3) as an identity. The paper takes the invariance from the common `theta` ("if theta is the true iROAS for each geo, the Z value for a specific geo would be the same no matter whether that geo would be treated or controlled (see Lemma 1 of Chen and Au [2019])", p.4); given it, the fair sign draw `A_i` makes `eps_i` symmetric. Reviewer note: the invariance is an assumption in its own right, and it concerns `Z = R - theta * S`, the response net of the spend effect, with respect to the whole assignment. A geo's response may depend on other geos' assignments through its own spend, as it does under the fixed budget of Section 4.2, where spend moves with the whole assignment while the baseline responses, and so `Z`, stay fixed (p.9); that is compatible with the model. A spillover that changes `Z` itself is not: both geos of a pair can then share the same `theta` while `Z_i1 - Z_i2` changes with the assignment. Section 3.1 does not state this as an assumption; the introduction says geos must be large enough "to be reasonably isolated" (p.2). `X_i` is random and "eps_i may be correlated with X_i, which makes the above model different from traditional linear models" (p.4).

*Estimator (Section 3.2, p.5).* Trimmed Match (Chen and Au, 2019): `theta_hat` solves

    (1 / (n - 2*ceil(n*lambda))) * sum_{i = ceil(n*lambda)+1}^{n - ceil(n*lambda)} eps_(i)(theta_hat) = 0

with `eps_(1)(theta) <= ... <= eps_(n)(theta)` the ordered residuals `Y_i - theta * X_i` and trim rate `lambda` in `[0, lambda_bar)` "chosen in a data-driven manner", where the constant `lambda_bar` "must be less than 0.5". Recommendation for `lambda_bar`: "0.25 for post analysis so that no more than half of the data points are trimmed" and "a smaller value (e.g. 0.10)" in the design procedure "in order to be conservative" (p.5). With `I` the untrimmed pairs, `theta_hat = sum_{i in I} Y_i / sum_{i in I} X_i`. Reviewer note: the stated range for `lambda` does not by itself leave any pair untrimmed. The trimmed mean needs `2 * ceil(n * lambda) < n` (with `n = 10` and `lambda = 0.45`, five pairs are trimmed from each tail and none remain), and the ratio needs a non-zero retained spend difference `sum_{i in I} X_i`. The paper states neither condition; for the estimator's details (the data-driven choice of the trim rate, and whatever makes the solution of the trimmed-mean equation unique when spend differences can take either sign) it refers to Chen and Au (2019), which is not part of this review. (That paper's Section 5.1, arXiv:1908.02922v1, read for this point only, defines the estimate as the solution of the trimmed-mean equation and, "When multiple roots exist", takes the one that minimizes a measure of asymmetry of the ordered residuals, its equations 5.2-5.4; an implementation needs that rule, since different roots give different estimates. The same section requires `n - 2m >= 1`, with `m = ceil(n * lambda)`, "so that trimming does not remove all n data points".)

*Pairing criterion (pp.5-6).* "there is no simple closed form for var(theta_hat), since lambda is unknown and needs to be chosen in a data-driven manner", and a budget constraint adds dependence across geos (p.5). The paper therefore fixes the total budget, `sum_{i=1..n} X_i = B` identically, and proceeds "by assuming the final pairs are in good quality such that the conservative choice lambda = 0 is a good approximation", in which case the estimator is the untrimmed ratio and

    var(theta_hat) = (1/B^2) * sum_{i=1..n} E[(Y_i - theta * X_i)^2] = (1/B^2) * sum_{i=1..n} |Z_i1 - Z_i2|^2

"This gives a conservative variance estimate" (p.6). Reviewer note: the additive form has no cross-pair covariance terms, as under independent within-pair sign draws; restricting the joint assignment (rerandomization) can change that. It is an approximation used to motivate the pairing loss, not the variance of the data-adaptively trimmed estimator. To be less sensitive to outlier pairs the paper proposes an `L1` loss instead:

    L = sum_{i=1..n} | Z_i1 - Z_i2 |                                 (Equation 3.4)

"for a fixed n, it is desirable to choose the set of n matched geo pairs which minimizes L" (p.6).

*Distance between geos (Section 4.1, pp.6-7).* Since `theta * |S_i1 - S_i2|` "is usually expected to be much smaller than" `|R_i1 - R_i2|`, "we may approximate |Z_i1 - Z_i2|^2 by |R_i1 - R_i2|^2 based on pretest data" (p.7): squared differences in pretest responses stand in for squared differences in test-period uninfluenced responses. With `T_p` the pairing period and a sub-period `T` within it that "may represent the test period, say with similar seasonality and similar level of responses but with possibly different duration",

    D_gg' proportional to sqrt( sum_{T subset of T_p} | sum_{t in T} R_g(t) - sum_{t in T} R_g'(t) |^2 )      (Equation 4.1)

The outer sum runs over the chosen sub-periods `T` of the pairing period (one may take several "in order to get a more reliable estimate", p.7). For a candidate pairing the loss is the sum over pairs of these distances, `L proportional to sum_{i=1..n} sqrt( sum_{T subset of T_p} | sum_{t in T} R_i1(t) - sum_{t in T} R_i2(t) |^2 )` (p.7); the square root is taken within each pair.

Rule of thumb (p.7): take `T` as a week when the test lasts a few weeks, a month when it lasts a few months. "One may optimize the choice based on a power analysis" (p.7).

*Optimal subset pairing (Section 4.1, p.6).* For `n = floor(N/2)`, minimum-distance non-bipartite matching, polynomial time (Edmonds, 1965). For `n < floor(N/2)`, add `N - 2n` pseudo geos with infinite distance to each other and zero distance to every real geo, and match on the expanded `(2N - 2n) x (2N - 2n)` matrix; the `n` pairs without a pseudo geo are the optimal subset (attributed to Lu et al., 2001). Geos not in a pair are left out of the experiment.

Remark 1 (p.7) warns that "The L criteria described above may be sub-optimal if the pairing quality measured by L using data in T_p differs from the test period dramatically", and offers an alternative: rank geos (for example by size), pair by rank, and select pairs by some measure of within-pair difference.

*Power analysis on a separate evaluation period (Section 4.2, pp.7-9).* "The key idea is to use an evaluation period T_e which does not overlap with the pairing period T_p. Suppose that T_e is similar to the test period in terms of the distribution of the pair-level "uninfluenced response" difference." (p.7). Both conditions are needed to read the evaluation-period RMSE as an estimate of test-period performance; the paper restates the second as similarity of "the distribution of the underlying responses and spends" between the two periods (p.9).

Algorithm 1 (p.8), "RMSE evaluation for a hold-back experiment with a fixed budget":
- Input: budget `B`, iROAS `theta` (0 under `H0`, `theta_0` under `H1`), and for the `n` matched pairs the evaluation-period baseline responses and spend proxies `((R'_i1, S'_i1), (R'_i2, S'_i2))`.
- a) Draw assignments `A_i` i.i.d. with probability 1/2.
- b) Generate experiment data: if `A_i = -1`, `X_i = r * S'_i1` and `Y_i = (R'_i1 + theta * r * S'_i1) - R'_i2`; otherwise `X_i = r * S'_i2` and `Y_i = (R'_i2 + theta * r * S'_i2) - R'_i1`; with `r` chosen so that `sum_{i=1..n} X_i = B`, i.e. `r = B / sum_{i=1..n} ( S'_i1 * 1{A_i = -1} + S'_i2 * 1{A_i = 1} )`: the budget divided by the summed spend proxies of the treated geos. Control geos have zero spend (a "hold-back" experiment, p.8). Reviewer note: `r` is undefined when the treated geos' spend proxies sum to zero, and a draw whose Trimmed Match estimate is undefined (see the note under the estimator) cannot enter the `K`-draw RMSE; the paper gives no rule for either case.
- c) Compute the Trimmed Match estimate from `{(Y_i, X_i)}`.
- d) Repeat a)-c) `K` times ("say 1000") and compute `RMSE(theta) = sqrt( (1/K) * sum_{k=1..K} (theta_hat^(k) - theta)^2 )`.

(Step b as printed treats the FIRST geo when `A_i = -1`. Section 3.1 (p.5) and the prose below Algorithm 1 (p.8) both use the opposite coding, `A_i = -1` meaning the second geo is treated; see Gaps.)

Footnote 2 (p.8): "It is desirable to exclude some of the assignments due to random imbalance, to be consistent with the assignment generation procedure described later, otherwise the estimate of RMSE may be conservative."

*From RMSE to a detectable effect (Equation 4.2, p.9).* Using `RMSE(theta_0)` approximately equal to `RMSE(0)` and the normal approximation `theta_hat ~ N(theta, RMSE^2)`,

    theta_0 = RMSE * ( q_{1-alpha} + q_beta )

with `q_x` the standard normal quantile, `alpha` the Type-I error and `beta` the power: "for a desirable design, the RMSE must be no more than theta_0 / (q_{1-alpha} + q_beta) in order to have power no less than beta."

Direct alternative (Remark 2, p.9): estimate power as `(1/K) * sum_{k=1..K} 1{ theta_hat^(k) > q }` "under H1 at theta = theta_0", where `q` is the `100 x (1 - alpha)%` quantile of `theta_hat` under `H0`, which "can be approximated by the empirical quantile of theta_hat using simulations as described in Algorithm 1" (run at `theta = 0`). With `lambda = 0` and a fixed budget "the procedure is equivalent to the standard permutation test."

*Rerandomization (Section 4.3, pp.9-10).* After pairing, the assignment within each pair is randomized; where random imbalance calls for it ("if needed", p.10), it is redrawn until balance checks pass. Two checks named: "whether the proportion of positive pairs (based on the sign of R_i1 - R_i2) is close to 1/2, and whether the estimated iROAS with simulated incremental ad spend is close to 0" (p.9). "it is still not clear how to perform balance checks in an optimal way - too many balance checks may lead to very few or even no options for geo assignment" (p.10).

*The procedure (Section 4.4, p.10):*
1. "Split the pretest time period into two non-overlapping periods: a time period for geo pairing (T_p) and an evaluation time period for power analysis (T_e)."
2. "For each n <= N/2, use data during T_p to generate the optimal subset of n geo pairs from N geos ... As a rule of thumb, we recommend n >= 10 as too few pairs may make the inference unreliable."
3. "For each n and the optimal subset of n pairs obtained in the previous step, use data during T_e to perform power analysis".
4. "For each candidate set of geo pairs, randomize geo assignment within each pair, and if needed adjust random imbalance through rerandomization".
5. "Finally, choose the design which has small enough RMSE according to the configuration for testing (1.1). Here additional marketing constraints may apply, e.g. avoiding diminishing returns if too much ad spend, limited ad inventory, etc."

Which period is which (p.11, simulation): "the most recent 4 weeks of data is used for pairing (T_p), and the first 2 weeks for evaluation (T_e). The purpose of using the latest data for pairing is to best maintain the pairing quality from the pairing period to the test period." In the case study the evaluation period is the same calendar weeks as the planned test, one year earlier, and "The data after that is used for geo pairing" (p.15).

*Interference caused by the fixed budget (p.9):* because `r` depends on the whole assignment, "the simulation is incompatible with the usual stable unit treatment value assumption"; the paper argues the model (3.1) still holds through (3.3), since assignment changes spend but not the uninfluenced response.

*Edge cases the paper addresses:*
- No spend history (new advertisers): use "some proxy (e.g. population size) to inform the relative level of spend across geos" (p.3).
- Few pairs: `n >= 10` recommended (p.10).
- Evaluation and test periods that differ systematically: "one may need to apply some adjustment to RMSE" (p.9), with no method given; listed as an open problem (p.17).

**Reference implementation(s):** "The implementation code in Python is partially available at GitHub [Chen et al., 2020]" (p.3); the reference is Chen, Longfils and Best, "The Python library for Trimmed Match and Trimmed Match Design" (p.18). The actual case-study design used rank-based pairing "as optimal pairing was not implemented at that time yet" (p.16).

### What the paper does NOT provide

- **No treatment of the second selection step.** Step 5 chooses among the candidate designs (one per `n`) using RMSE values computed on `T_e`. The paper does not discuss whether choosing on `T_e` makes the chosen design's `T_e` RMSE optimistic. Its unbiasedness check (Figure 5.4) is for a given `n`, averaged over replications, not for the selected `n`. The same applies to choosing the sub-period `T` of the distance "based on a power analysis" (p.7): if `T_e` informs that choice it is tuning data as well, and an untouched window would be needed to assess the design finally selected.
- **No size (false-positive rate) evaluation of the test.** The simulations report the RMSE of `theta_hat`; the worked example starts from a specified detectable iROAS and uses the normal approximation to derive the acceptable RMSE (p.12). Remark 2 describes simulating a null critical value (p.9), and "Type-I" error appears as the input `alpha` (pp.8-9) and is set to 10% in the worked example (p.12), but no empirical rejection rate under `H0` is reported.
- **No choice of which pair member is treated by optimization.** Assignment within a pair is random, restricted by balance checks.
- **No non-randomized design.** Au (2018) is described as a design that "only applies to non-randomized experiments and does not meet the gold standard for causal inference" (p.4).
- **No difference-in-differences estimator and no use of pre-period outcomes in the analysis.** The estimator uses test-period pair differences only; pretest data enter through pairing, the power analysis and any rerandomization balance checks. "difference-in-differences" and "parallel" have no match in the text layer.
- **No covariates in the distance** beyond the response; "Extra geo features may be incorporated into the L criteria" (Remark 1, p.7) with no formula.
- **No quantitative spillover constraint.** The paper requires that "each geo must be large enough so as to be reasonably isolated from other geos" (p.2); it gives no buffer, adjacency or other spillover rule for the matching.
- **No target population after trimming geos.** The estimand is defined over all geos `G` (p.2) and the model assumes `theta` equal across geos (p.4); the paper does not discuss what is estimated when geos are dropped and `theta` differs across geos. The word "representative" is used only for a time period representing the test period (p.7).
- **No general rule for the lengths or order of `T_p` and `T_e`.** Both examples use the latest data for pairing and an earlier evaluation window as long as the planned experiment; the case study states the reason ("so as to have the same duration as the experiment", and the same weeks of the year; p.15).

---

**The registry-candidate text ends here. Everything below is audit material.**

## Results

### Simulation (Section 5, pp.10-14)

Data (p.10): `N = 100` geos; sizes `g_i = 10^5 * F^-1(i / (N+1))` with `F^-1` the log-normal quantile function (mean 1, standard deviation 1 on the log scale); daily response `R_i(t) = g_i * (1 + 0.25 * w(t) * (1 + 0.5 * eps_i(t)))` with `w(t) = sin(2*pi*t/7)` and `eps_i(t) = 0.5 * eps_i(t-1) + N_it(0,1)`; spend proxy `S_i(t) = 0.01 * R_i(t) * (1 + 0.5 * U_it(-1, 1))` (Equation 5.1), where `N_it(0,1)` and `U_it(-1,1)` "denote mutually independent random numbers" indexed by geo and time; 6 weeks of pretest data. The paper calls the response "a stationary time series" (p.10). The simulated experiment is a hold-back experiment (p.11). The initial value of the AR(1) recursion and the `theta` passed to Algorithm 1 for the plotted RMSE are not stated. Budget `B = 10^6`, a planned 2-week test, the latest 4 weeks for pairing and the first 2 weeks for evaluation, `n` from 10 to 50, weekly `T` in the distance (p.11). "The entire simulation is replicated for M = 5000 times, that is, we use the above data simulation procedure to generate 5000 pretest data sets and rerun the same design process for each of them. Unless specified, any reported result below is an average value across the 5000 replicates." (p.12)

Findings, all reported as figures (no table):
- **Fewer pairs, lower RMSE (Figure 5.3, p.12):** "By reducing the number of pairs from 50 to 45, RMSE is reduced by nearly 50%." Worked example: with `theta_0 = 2.56`, Type-I error 10% and power 90%, "it is desirable to have RMSE <= 1, for which Figure 5.3 suggests n < 30"; with the budget capped at 2% of "the overall base response for the treatment group during T_e", `n >= 24`, "and thus the optimal design corresponds to n = 24 as it minimizes the RMSE and thus maximizes the power". (Reviewer arithmetic: `q_0.90 + q_0.90` is approximately 2.5631, so `theta_0 / (q_{1-alpha} + q_beta)` is approximately `2.56 / 2.5631 = 0.9988`, about 1.)
- **Effect of the split (Figure 5.4, pp.12-13):** RMSE evaluated on the most recent 2 weeks, which lie inside the pairing period, "is significantly lower than the corresponding RMSE value with cross validation given the same number of geo pairs, which confirms systematic overfitting." Extending the data to 8 weeks and treating weeks 7-8 as the test period, "the green circles almost overlap with the solid blue curve, which suggests that the RMSE evaluation with cross validation is indeed unbiased." The expectation is stated beforehand: "Due to stationarity in the time series data generation, we expect the RMSE evaluation to be unbiased" (p.12); the figure is offered as support for it. No numbers are printed for the size of the gap.
- **Against a permutation-test design (`lambda = 0`; Figure 5.5, p.13):** "the performance is quite comparable except for large n. In fact, when n = 50, Trimmed Match reduces RMSE by a factor larger than 2".
- **Granularity of `T` (discussion p.13, Figure 5.6 p.14):** weekly `T` gives lower RMSE than daily for large `n`, and the reverse for small `n`.
- **Spend proxy (Figure 5.7, p.14):** with the linear response term of the spend proxy replaced by the squared response and the same multiplicative noise factor kept, `S_i(t) proportional to R_i(t)^2 * (1 + 0.5 * U_it(-1, 1))` (Equation 5.2), RMSE is higher near `n = 50` and "quite comparable" for `n` below about 40.

### Case study (Section 6, pp.14-16)

Nielsen's 210 DMAs with a few excluded; more than one year of daily revenue; a 6-week test plus 2 weeks of cooldown "to capture any potential lag effects"; an 8-week evaluation period (the same duration, cooldown included) at the same weeks of the previous year; pairing on the data after it (pp.14-15). "Some historical ad spend data were also collected to inform the split of the client pre-specified budget." (p.15) "the geo sizes measured by revenues approximately follow a heavy-tailed Pareto distribution, with about 80% of the revenue from 20% of the geos" (p.15). Optimal pairing (with `T` a week) against the rank method, which "ranks and pairs the geos based on their revenue volumes during the pairing period": "the reduction of RMSE by using optimal pairing compared to the rank-based pairing can be as large as a factor of 3" (p.15; Figures 6.3-6.4 are on p.16). The design actually run used rank-based pairing and "around 60 pairs by applying the budget and other marketing constraints" (p.16). "After the experiment was completed, the point estimate and confidence interval were obtained by the Trimmed Match estimator (see Chen and Au [2019] for the detailed description)." (p.16) After the experiment: "the half width of the 80% two-sided confidence interval is quite comparable to 1.28 x RMSE, where RMSE was reported from the design phase, despite the irregular temporal dynamics of the revenues" (p.16). No numbers are given for either quantity.

### Related work as the paper describes it (pp.3-4)

- Vaver and Koehler (2011), geo-based regression: "sensitive to the choice of the regression weights and thus is not robust against geo heterogeneity".
- Kerman et al. (2017), time-based regression: "belongs to the class of synthetic control models and requires some untestable assumptions, despite randomness in geo assignment".
- Au (2018), matched markets on the time-based regression model: "a market matching method for designing geo experiments by maximizing a linear relationship between two time series responses, one for treatment and the other for control, which requires very few geos (essentially a single pair of grouped geos) and can be less costly if the TBR model fits and predicts well. However, the design method only applies to non-randomized experiments".

### Open problems stated (p.17)

"how to improve the L criterion by taking into account the uncertainty in the data-driven choice of trim rate, how to address random imbalance in a more rigorous manner, and how to adjust the RMSE due to the potential difference between the evaluation period and the test period."

---

## Implementation Notes (audit notes - NOT registry-candidate)

### Data Structure Requirements
- Pretest panel of responses `R_g(t)` and spends or a spend proxy `S_g(t)` per geo, split in time into a pairing period and an evaluation period.
- A total budget `B`, the experiment type (hold-back in the paper's algorithm), `alpha`, target power, and a detectable `theta_0`. A heavy-up experiment "can be simulated similarly, but the detail is omitted for conciseness" (p.8).

### Computational Considerations
- One non-bipartite matching per candidate `n` (polynomial); the expanded matrix has side `2N - 2n`.
- Power analysis: `K` random assignments per candidate, each a Trimmed Match fit on `n` pairs (the paper gives 1000 as an illustrative `K`); no runtime is reported.
- Evaluation needs only the pairs' data in `T_e`; no model is carried from `T_p` to `T_e` other than the pairing itself.

### Tuning Parameters

| Parameter | Type | Setting reported in the paper | Selection method |
|---|---|---|---|
| `n` (number of pairs) | positive integer, `n <= floor(N/2)`; `n >= 10` recommended as a rule of thumb | simulation: `n` from 10 to 50, with 24 chosen in the worked example; case study: around 60 pairs | a candidate whose RMSE on `T_e` is "small enough" for the configured test, subject to marketing constraints (p.10); the simulation example takes the smallest RMSE among candidates meeting its budget-to-baseline cap (p.12) |
| `T_p`, `T_e` | non-overlapping time periods | examples: latest data for pairing, an earlier window as long as the planned experiment for evaluation | analyst; no general rule |
| `T` (sub-period in the distance) | a sub-period of `T_p` | a week in the simulation and the case study; daily and weekly compared in the simulation | rule of thumb (a week for a test of a few weeks, a month for a test of a few months); may be chosen by the power analysis (p.7) |
| trim rate `lambda` | `0 <= lambda < lambda_bar`, `lambda_bar < 0.5` | data-driven; `lambda_bar = 0.25` recommended for post analysis, a smaller bound for design (0.10 given as an example) | data-driven within the bound; the rule itself is not given in this paper (Chen and Au, 2019, is cited) |
| `K` | integer | 1000 as an example | - |
| balance checks | rules | two named | not settled (p.10) |

### Relation to Existing diff-diff Estimators
- The paper's estimator is a ratio estimator on within-pair differences for a randomized paired design; the library has no such estimator, and the paper has no difference-in-differences content.
- What transfers is the protocol: build the design on one block of pre-treatment periods and compute its power on a disjoint block, with the stated reason that reuse "can lead to overestimating the power of the design due to overfitting" (p.3).
- `diagnose_design` (`diff_diff/design.py`) reports placebo rejection rates on the supplied untreated panel for randomly drawn treated units (after optional unit resampling), and optionally searches for a minimum detectable effect when a target power is supplied and the estimator passes its size check. It has no notion of a design built from one block of periods and scored on another.

---

## Gaps and Uncertainties

1. **Assignment convention.** Section 3.1 (p.5) defines `A_i = 1` as "the first geo is assigned to treatment", and the prose below Algorithm 1 agrees ("assume that A_i = -1, i.e. the first geo is assigned to control, and the second geo to treatment", p.8). Algorithm 1 step b (p.8) and its budget denominator use the opposite coding: for `A_i = -1`, `X_i = r * S'_i1` and `Y_i = (R'_i1 + theta * r * S'_i1) - R'_i2`, which treats the FIRST geo. Reviewer inference: with fair, independent sign draws the relabelling does not change the Monte Carlo distribution; with rerandomization the balance rule must be relabelled consistently too. An implementation must pick one convention.

   Two further printed ambiguities: (a) the rerandomization check uses "the sign of R_i1 - R_i2" (p.9), which does not vary across assignment draws if 1 and 2 are fixed labels; an assignment-oriented reading (treated minus control) would be a reconstruction. (b) The stated rationale for 0.25, "no more than half of the data points are trimmed" (p.5), does not hold exactly under the printed rule, which trims `ceil(n * lambda)` from each tail (for `n = 11` and `lambda = 0.24`, three from each tail is six of eleven; reviewer arithmetic).
2. **Selection among candidates on the evaluation period.** See "What the paper does NOT provide". With one candidate per `n` (about 40 in the simulation) and a smooth RMSE curve the effect may be small; the paper gives no evidence either way.
3. **Stationarity.** Figure 5.4 shows close agreement between the evaluation-window and test-window average RMSE in one simulation (the curves "almost overlap", which "suggests" unbiasedness, p.13); the paper attributes the expected agreement to the stationarity of the simulated series (p.12). Reviewer note: the simulated daily response has a day-of-week component, and the windows compared are whole weeks. The case study's revenue "is quite non-stationary" (p.15), and the only evidence there is the qualitative statement that the realized interval half-width was "quite comparable" to the design-phase figure (p.16).
4. **Normal approximation in (4.2).** The detectable effect uses a normal approximation for a trimmed ratio estimator with as few as 10 pairs; Remark 2 offers the simulation-based alternative. No comparison of the two is reported.
5. **`RMSE(theta_0)` approximately `RMSE(0)`** is asserted from the magnitude argument on p.8, not shown.
6. **Evaluation period earlier than pairing period.** In both examples `T_e` precedes `T_p` in calendar time, so the "held-out" block is older than the block the design was built on, and the test period follows `T_p`. In the simulation the stated reason is that using the latest data for pairing best maintains the pairing quality into the test period (p.11); in the case study the evaluation window is chosen for seasonal alignment with the planned test and the data after it is used for pairing (p.15).
7. **Geo trimming and the estimand.** Dropping hard-to-match geos lowers RMSE sharply in the simulation; the paper does not discuss what `theta_hat` estimates if `theta` varies with geo size.
8. **Version.** arXiv v2 reviewed; no journal version is known to the reviewer.
