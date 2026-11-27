# Paper Review: Synthetic Design: An Optimization Approach to Experimental Design with Synthetic Controls

**Authors:** Nick Doudchenko, Khashayar Khosravi, Jean Pouget-Abadie, Sebastien Lahaie, Miles Lubin, Vahab Mirrokni (Google Research); Jann Spiess, Guido Imbens (Stanford GSB). Doudchenko and Khosravi are marked as equal contributors.
**Citation:** Doudchenko, N., Khosravi, K., Pouget-Abadie, J., Lahaie, S., Lubin, M., Mirrokni, V., Spiess, J., & Imbens, G. (2021). Synthetic Design: An Optimization Approach to Experimental Design with Synthetic Controls. *Advances in Neural Information Processing Systems*, 34. arXiv:2112.00278.
**Source reviewed:** arXiv:2112.00278v1 ([PDF](https://arxiv.org/pdf/2112.00278v1); paper dated December 2, 2021), a 36-page PDF: main text pp.1-19, references pp.19-21, supplementary materials (Appendix A.1-A.5) pp.22-36. The PDF was reviewed externally and is not committed to the repository. The NeurIPS proceedings version was not compared with it. All page citations are the page numbers printed in the PDF. Mathematical symbols are transliterated to ASCII (`tau`, `sigma`, `lambda`, `a_bar_I`); inside quoted passages the paper's em dashes are written as spaced hyphens and its typographic apostrophes as straight ones, and footnote markers are omitted. Where no restriction is shown, `sum_i` and `sum_j` run over units `1, ..., N` and `sum_t` over pre-treatment periods `1, ..., T`.
**Review date:** 2026-11-27

> **Personal verification stamp.** The reviewer read the full main text and the full supplement and checked directly against them: the two conditional-MSE expressions (pp.6-7); the per-unit, two-way global and one-way global objectives and constraints as stated in Section 2 (pp.6-8) and Section 3 (p.9); the three closed forms of Theorem 1 (p.11); every cell of Table 1 (p.16); the simulation design (p.14); the exact mixed-integer formulations (pp.24-27); the inference procedure and Proposition 1 (pp.32-33); and the hardness reduction (pp.34, 36). Terms reported as absent were searched in the PDF text layer.

---

## Methodology Registry Entry

**Status: proposed text for a possible future REGISTRY section; no such section exists in `docs/methodology/REGISTRY.md` today; this file is a non-authoritative source audit.** The paper is a candidate source for a planned unit-selection capability. Nothing below is a methodology contract until a REGISTRY section is written and reviewed alongside an implementation. The registry-candidate text ends at the marker before `## Results`.

## Synthetic design by mixed-integer programming (working title)

**Primary source:** Doudchenko, N., Khosravi, K., Pouget-Abadie, J., Lahaie, S., Lubin, M., Mirrokni, V., Spiess, J., & Imbens, G. (2021). Synthetic Design: An Optimization Approach to Experimental Design with Synthetic Controls. *Advances in Neural Information Processing Systems*, 34. https://arxiv.org/abs/2112.00278

**Key implementation requirements:**

*Scope of the source:*
- "We consider a panel-data setting in which the researcher observes the outcome metric of interest for a number of units in a number of time periods and has to decide: (i) which units to experiment on and (ii) how to estimate the treatment effects after collecting the outcome data in the experimental time periods." (p.3)
- The estimator class: "The average treatment effect is estimated as the difference between the weighted average outcomes of the treated and control units. A number of commonly used approaches fit this formulation, including the difference-in-means estimator and a variety of synthetic-control techniques." (abstract, p.1)
- The design uses pre-treatment outcomes only, with no covariates: "we focus exclusively on the outcome variable and utilize all of the pre-treatment outcome data" (p.5), in contrast to Abadie and Zhao (2021), who "use both pre-treatment unit-level covariates as well as the outcomes from some - but not necessarily all - of the pre-treatment periods" (pp.4-5).
- The estimand is the effect on the units chosen: "the objective function we use focuses on the average treatment effect only for the units we choose for the treatment, making the target estimand stochastic, but potentially easier to estimate and requiring different assumptions" (p.5).

*Setting (Section 2, pp.5-6):* `N` units, `T` pre-treatment periods, an `N x T` matrix `Y_it`. "At time t = T the researcher decides - based on the data observed up until that point - which units should be treated and assigns a binary treatment described by variables D_i" in `{0, 1}`. Outcomes are then observed for `S - T` more periods `t = T+1, ..., S`. Model:

    Y_it(0) = mu_it + eps_it,     Y_it = Y_it(0) + D_i * tau_i

"with homoscedastic noise eps_it that has mean zero and variance sigma^2 and additive treatment effects tau_i" (p.6). One adoption date; the effect is constant over the treated periods for a unit.

*Estimators after the experiment (pp.5-7).* The design step uses the pre-treatment panel; estimation needs the outcomes observed in the treated periods. Written for the first treated period `T + 1`:
- Per unit: `tau_hat_i = Y_{i,T+1} - Y_hat_{i,T+1}(0)` with `Y_hat_{i,T+1}(0) = sum_{j: D_j=0} w^i_j Y_{j,T+1}` (p.6).
- Weighted average effect on the treated, `tau = sum_{i: D_i=1} w_i tau_i`: `tau_hat = sum_{i: D_i=1} w_i Y_{i,T+1} - sum_{i: D_i=0} w_i Y_{i,T+1}` (p.7). With treated weights chosen by the program (two-way) the estimand's weights are the fitted ones; with equal treated weights (one-way) it is the simple average over treated units.

*Conditional mean squared error that motivates the objectives (pp.6-7; derivation in A.1, pp.22-23, which assumes the `eps` are the only source of randomness and independent across units):*

Per treated unit `i` with weights `w^i_j` on controls:

    E[(tau_hat_i - tau_i)^2 | {D_j, w^i_j}_{j=1..N}] = ( mu_{i,T+1} - sum_{j: D_j=0} w^i_j mu_{j,T+1} )^2 + sigma^2 * ( 1 + sum_{j: D_j=0} (w^i_j)^2 )

For a difference in weighted means targeting `tau = sum_{i: D_i=1} w_i tau_i`:

    E[(tau_hat - tau)^2 | {D_i, w_i}_{i=1..N}] = ( sum_{i: D_i=1} w_i mu_{i,T+1} - sum_{i: D_i=0} w_i mu_{i,T+1} )^2 + sigma^2 * sum_{i=1..N} w_i^2

Unless the estimand is that weighted average, "another term, (sum_{i: D_i=1} w_i tau_i - tau)^2, remains as part of the MSE. In the current paper we make no claims about this term apart from the case when the treatment effects are homogeneous and the term vanishes." (p.23) Reviewer derivation from the error decomposition on p.22: with `B` the imbalance in untreated means and `Delta = sum_{i: D_i=1} w_i tau_i - tau`, the squared bias is `(B + Delta)^2`, which contains the cross term `2 B Delta` as well as `Delta^2`.

Reviewer note on both formulas: A.1 derives them with the assignments and weights held fixed and the `eps` as the only randomness, independent across units. When the design is chosen from the pre-treatment panel, as it is here, the formulas need the treated-period noise to have, conditional on the chosen design, mean zero, variance `sigma^2` and no correlation across units. Noise that is independent over time is one sufficient condition; serially dependent noise does not satisfy it in general, because the selected design then carries information about the treated-period noise.

*Three design-and-estimation problems (Section 2, pp.6-8).* Each is the empirical analog of the bias term of the conditional MSE: the squared discrepancy is computed in each pre-treatment period `t = 1, ..., T` from the observed outcomes and the squared discrepancies are then averaged over `t` (and, in the per-unit problem, over the treated units). The weight-variance term is kept as a penalty; the constant `sigma^2 * 1` of the per-unit MSE does not depend on the decision variables and is dropped (it "will not show up in the calculations", p.31).

1. **Per-unit** (one synthetic control per treated unit; `K` treated units):

       min over D, {w^i_j}   (1/(K T)) * sum_i sum_t D_i * ( Y_it - sum_j w^i_j (1 - D_j) Y_jt )^2  +  (sigma^2 / K) * sum_i sum_j D_i (w^i_j)^2

   Constraints proposed for it in Section 2 (p.8): `w^i_j >= 0` for all `i, j`, and `sum_{j: D_j=0} w^i_j = 1` for every treated `i`.

2. **Two-way global** (weights on both sides; estimates a weighted average effect on the treated):

       min over D, w   (1/T) * sum_t ( sum_{i: D_i=1} w_i Y_it - sum_{i: D_i=0} w_i Y_it )^2  +  sigma^2 * sum_i w_i^2

   with `sum_{i: D_i=1} w_i = sum_{i: D_i=0} w_i = 1`. Without the normalization and with `sigma^2 > 0`, all-zero weights are the unique optimum (p.8). Non-negativity is "not strictly required" but is "another reasonable set of constraints" (p.8). Separately, the paper says that letting the experimenter choose both the treated units and the weighting scheme "is especially appropriate when the treatment effects are homogeneous", since then any weighted average of unit effects equals the common effect "and the weights can be chosen in a way that minimizes the mean squared error" (pp.7-8).

3. **One-way global:** the two-way problem with equal weights on the treated, `w_i = w_j` for all treated `i, j` (p.8). "This constraint is particularly justified when the treatment effects are heterogeneous and different weighting schemes lead to different estimands." (p.8)

*Special cases (pp.8-9):* difference in means under randomization is the one-way global problem "with the treatment indicators, D_i, set randomly and the weights on the control units restricted to be equal"; the synthetic-control estimator under a given design is the per-unit problem with the indicators set by that design.

*Mixed-integer formulation as used in the empirical section (Section 3, pp.9-10):* `sigma^2` is replaced by `lambda`, "a nonnegative penalty factor" (p.10). Per-unit:

    min   (1/(K T)) sum_i sum_t D_i ( Y_it - sum_j w^i_j (1 - D_j) Y_jt )^2 + (lambda / K) sum_i sum_j D_i (w^i_j)^2
    s.t.  w^i_j >= 0,  D_i in {0,1},  sum_i D_i = K,  sum_j w^i_j (1 - D_j) = 1 for every i with D_i = 1.

(The last constraint is a reviewer reconstruction of the printed normalization. Pages 9 and 24 print it as `sum_{i=1}^{N} w^i_j (1 - D_j) = 1` "for i = 1, ..., N such that D_i = 1", summing over `i`. Section 2 (p.8) writes the condition as `sum_{j: D_j=0} w^i_j = 1` for treated `i`; the `T = 1` version on p.10 sums over `j` and prints it for every `i`; the linearised form is `sum_j q_ij = D_i` (p.25). See Gaps.)

Two-way global:

    min   (1/T) sum_t ( sum_i w_i D_i Y_it - sum_i w_i (1 - D_i) Y_it )^2 + lambda sum_i w_i^2
    s.t.  w_i >= 0,  D_i in {0,1},  sum_i D_i = K,  sum_i w_i D_i = 1,  sum_i w_i (1 - D_i) = 1.

One-way global: the two-way problem plus `w_i = w_j` for all treated `i, j`.

*Linearisation (A.2, pp.24-27):* products of a weight and a binary are replaced by auxiliary variables with linear inequalities, giving "a quadratic objective with a positive semi-definite Hessian and linear constraints" (p.25).
- Per-unit: `q_ij = w^i_j (1 - D_j)` enforced by `q_ij >= 0`, `q_ij <= 1 - D_j`, `q_ij <= w^i_j`, `q_ij >= w^i_j - D_j`; also `w^i_j <= D_i`, `sum_j q_ij = D_i`, and residuals `z_it = Y_it D_i - sum_j q_ij Y_jt` (pp.24-25).
- Two-way: `q_i = w_i D_i` enforced by `q_i >= 0`, `q_i <= D_i`, `q_i <= w_i`, `q_i >= w_i - (1 - D_i)` (with `w_i >= 0`); `sum_i q_i = 1`, `sum_i w_i = 2`; the residual is printed as `z_t = sum_i (2 q_i - 1) Y_it` (p.26). **As printed this does not reproduce the objective**: the residual in the objective (pp.9, 25) is `sum_i w_i D_i Y_it - sum_i w_i (1 - D_i) Y_it`, which with `q_i = w_i D_i` is `sum_i (2 q_i - w_i) Y_it`. Reviewer reconstruction: `z_t = sum_i (2 q_i - w_i) Y_it`, for both global formulations.
- One-way with `K` fixed: add `q_i = D_i / K` (p.26). Without a fixed `K` the constraint is `q_i = D_i / sum_j D_j`, made linear by multiplying through and introducing `r_ij = q_i D_j`, enforced in the same way as `q_i = w_i D_i`; this "substantially increases the number of required variables from O(N) to O(N^2)" (pp.26-27).
- Objectives after linearisation: per-unit, `(1/(K T)) sum_i sum_t z_it^2 + (lambda/K) sum_i sum_j (w^i_j)^2` (p.25; the paper prints the penalty as a double sum with two `i` indices, and the inner sum over `j`, as on pp.9 and 24, is the consistent reading); global, `(1/T) sum_t z_t^2 + lambda sum_i w_i^2` (p.26).
- For the variable-`K` one-way case the linking constraint, obtained by multiplying `q_i = D_i / sum_j D_j` through by the denominator, is `sum_j r_ij = D_i` for every `i` (reviewer derivation; the paper describes the step in words, p.27).

*Number of treated units:* "the global versions of the problem can be solved without the constraint on the number of treated units, sum_i D_i = K, but the per-unit problem requires it - without the constraint the per-unit problem will tend to select fewer treatment units unless the objective is divided by sum_i D_i which introduces a nonlinearity" (p.10). A.2 gives a variant that removes the need to fix `K` (p.25): with `f` the per-unit objective before division by `K`, introduce an auxiliary variable `y`, minimize `y`, and add two constraints, `y * sum_i D_i >= f` (quadratic) and `sum_i D_i >= 1`.

*Additional constraints (p.4):* "The MIP formulation allows for an easy inclusion of additional constraints as long as those are linear. For instance, it is easy to restrict the overall number of treated units, exclude specific units from treatment, or enforce a budget constraint if there is a varying cost to treat different units." The formulations printed on p.9 impose the count constraint `sum_i D_i = K`, and Section 5 uses `K = 3` and `K = 7` (p.14). Unit exclusions and a budget constraint are mentioned as possibilities only; neither is written out or used.

*Solver (footnote 1, p.9):* "We use SCIP (Gamrath et al., 2020) when generating the empirical results in Sections 4 and 5."

*Penalty (Section 6, pp.17-18):* `lambda` "can be chosen using cross-validation. Specifically, the pre-treatment time periods can be split into the consecutive training and validation time periods and the value of lambda can be chosen by minimizing the RMSE over the validation period in a simulated experiment that is similar to the one we conduct in Section 5." The alternative, used in Sections 4 and 5, "computes the sample variances for every unit i across pre-treatment time periods t = 1, ..., T and then uses the average of those quantities across all units as the penalty factor". The only explicit formula is the `T = 2` example of Section 4, `lambda = sum_i (Y_i1 - Y_i2)^2 / (4N)`, "the average two-period sample variance across all units" (p.12), which divides each unit's sum of squared deviations by `T`; the divisor for longer panels is not printed.

*Which units are selected (Section 4, pp.10-12).* With `T = 1`, `a_i = Y_i1`, weights not restricted to be non-negative, treated set `I` of size `K`, control set `I_bar`, group means `a_bar_I`, `a_bar_Ibar`, and `V_I^2 = sum_{i in I} (a_i - a_bar_I)^2`, Theorem 1 (p.11) gives the objectives after the weights are optimized away, as functions of the treated set. Reviewer note on their domain: they need `1 <= K < N` and non-zero denominators. With `sigma^2 = 0` and no spread in the relevant group the denominators vanish although the constrained problem still has a finite optimum (for example `N = 2`, `K = 1`, outcomes `(0, 1)`: the normalizations force both weights to one and the objective is 1); such cases have to be solved from the original problem.

    J_per-unit(I) = sigma^2 * ( 1/(N-K) + ( (a_bar_I - a_bar_Ibar)^2 + K^-1 V_I^2 ) / ( sigma^2 + V_Ibar^2 ) )
    J_2-way(I)    = sigma^2 * ( 1/K + 1/(N-K) + (a_bar_I - a_bar_Ibar)^2 / ( sigma^2 + V_I^2 + V_Ibar^2 ) )
    J_1-way(I)    = sigma^2 * ( 1/K + 1/(N-K) + (a_bar_I - a_bar_Ibar)^2 / ( sigma^2 + V_Ibar^2 ) )

Reading (pp.11-12): all three make the group averages similar; "The per-unit problem maximizes the sample variance of the control units while minimizing that of the treated units"; "The two-way problem attempts to maximize both (taking into account that they are, of course, interdependent)"; the one-way problem "maximizes the sample variance of the outcomes of the control units only". Reading the formulas for a fixed gap in group means: greater spread among the controls lowers all three objectives; greater spread among the treated raises the per-unit objective, lowers the two-way objective, and does not enter the one-way objective. (These comparisons are weak: when the two group means coincide, the global objectives do not depend on the spreads.)

The weights that go with a chosen set `I` in this `T = 1`, unrestricted-sign case are given in A.3: two-way (p.27), `w*_l = 1/K + (a_bar_I - a_bar_Ibar)(a_bar_I - a_l) / (sigma^2 + V_I^2 + V_Ibar^2)` for treated `l` and `w*_l = 1/(N-K) - (a_bar_I - a_bar_Ibar)(a_bar_Ibar - a_l) / (sigma^2 + V_I^2 + V_Ibar^2)` for control `l`; one-way (p.30), `w*_l = 1/K` for treated `l` and `w*_l = 1/(N-K) - (a_bar_I - a_bar_Ibar)(a_bar_Ibar - a_l) / (sigma^2 + V_Ibar^2)` for control `l`; per-unit (p.31), for treated `i` in `I` and control `j` in `I_bar`, `w^i_j = 1/(N-K) - (a_i - a_bar_Ibar)(a_bar_Ibar - a_j) / (sigma^2 + V_Ibar^2)`. The paper describes `V_I^2` as "a quantity proportional to the sample variance" (p.11).

*Inference (A.4, pp.31-34).* Unit permutation is not available: "our method chooses the treated units themselves which makes this type of permutation infeasible. Instead we focus on inference methods that permute the treatment periods closely following the methodology suggested by Chernozhukov et al. (2021)" (p.32).
- Permutations of the `S` time periods, pre-treatment and treatment periods together (one reordering of periods applied to the whole panel, so that each period's vector of unit outcomes stays together): (i) i.i.d. permutations or (ii) moving-block permutations (cyclic shifts; at most `S` distinct samples) (p.32).
- For each permuted sample the ATET is re-estimated "assuming that the same number of periods at the end of the bootstrap sample are treated as the number of treatment periods in the original sample" (p.32).
- Statistic: "the absolute value of the estimate divided by the square root of the number of treatment periods" (p.32). Reject when the original statistic exceeds the `1 - alpha` quantile of the permuted values (in the power simulation: "If the test statistic computed on the originally sampled data ... exceeds that quantile, we reject", p.34). The estimate in the statistic is a scalar ATET; how per-period estimates are combined into it when there are several treated periods is not spelled out, nor is the finite-sample quantile convention or whether the unpermuted sample is counted among the permuted ones.
- Proposition 1 (pp.32-33): if across `t = 1, ..., S` the vector `Y_t(0)` "is drawn independently and identically, and ... the test statistic U(Y) permits a bounded density function", the test of the sharp null `Y_it(1) = Y_it(0)` for all `i` in all treatment periods "rejects a true null hypothesis with probability alpha". (The text on p.32 calls `1 - alpha` the "significance level" while using it as the quantile threshold; the proposition and the simulation, nominal size 10% with the 90% quantile, fix the intended meaning.) Reviewer notes: (a) the paper does not state the quantile convention, and the size depends on it. One explicit rule is the exhaustive rank p-value `p = #{U_perm >= U_obs} / M` over all `M` reorderings in the reference set, the original ordering included; under exchangeability with distinct statistics it rejects with probability `floor(alpha * M) / M`, which is at most `alpha` (with 12 cyclic shifts and `alpha = 0.05`, never). Another convention gives another size: comparing the original statistic strictly with a linearly interpolated 95th percentile of 12 distinct values rejects exactly when it is the largest, with probability 1/12. So "with probability alpha" can hold exactly only for particular pairs of rule and level. Ties need a stated convention (a bounded density for the statistic does not by itself rule out equal values across reorderings), and with ties the rejection probability lies on the `1/M` grid only conditionally on the set of reorderings. When only a sample of reorderings is used, the reference set, the denominator and the tie rule have to be stated separately. (b) The exchangeability argument applies to a statistic computed in the same way on every reordered panel, and that computation includes the data-driven choice of treated units; holding fixed a treated set that was chosen from the original pre-treatment periods is not covered by it.
- Conditions under weaker dependence are discussed only by reference, and what follows is this paper's own paraphrase of them, not a statement of the conditions in Chernozhukov, Wuthrich and Zhu (for those, see `chernozhukov-wuthrich-zhu-2021-review.md`): Chernozhukov et al. "state sufficient conditions for valid inference in a similar setting where treatment units are fixed": an unbiased estimator of the untreated outcomes, fixed (non-random) additive effects `Y_it(1) = Y_it(0) + tau_it`, which may vary by unit and period (an assumption that "would not be required to test the null hypothesis Y_it(1) = Y_it(0), but allows for constructing of confidence sets and testing other sharp hypotheses"), and residuals, defined as the differences between the estimators of `Y_it(0)` and the values themselves, that are mean zero and either i.i.d. across units and time (i.i.d. permutations) or i.i.d. across units and stationary weakly dependent over time (moving blocks) (p.33). For the paper's own setting, where the treated units are chosen from the data, only Proposition 1 is proved; it rests on exchangeability across time and "does not require that units themselves are exchangeable or that treatment is assigned randomly across units" (p.33).
- The authors' own assessment: "the assumption that the distribution of the full vector of potential control outcomes is independently drawn across time periods, including the treated periods, is unrealistic in many time series settings ... a rigorous proof of a version of Proposition 1 under such weaker assumptions is beyond the scope of the current paper" (p.33). The main text calls the guarantees valid "albeit under rather strong assumptions" (pp.4, 18).

*Hardness (A.5, pp.34, 36):* reduction from the equal-size partitioning problem (Cieliebak et al., 2008) to the `T = 1` forms `J_2-way(I)` and `J_1-way(I)` with half the units treated: with `a_i = b_i + T` (here `T` is the paper's symbol for the total of the `b_i`), an equal-sum half exists exactly when the optimum equals the printed cutoff `1/K + 1/(N-K)`. "Therefore, finding such an optimal solution is NP-hard." (p.36). The reduction is written for the two global problems only. Reviewer note: both formulas of Theorem 1 carry an outer factor `sigma^2`, so the printed cutoff is for the objective divided by a positive `sigma^2` (or for `sigma^2 = 1`); on the formulas as given above it is `sigma^2 * (1/K + 1/(N-K))`.

*Problem size (Section 6, p.18):* "In our simulations we were able to solve problems for N = 50 units ... on a single machine within hours." The per-unit problem "is generally harder to solve and is only tractable for a smaller number of experimental units, N, compared to the global problems" (p.17).

*Theoretical status of the estimators (p.23):* "While we do not provide any theoretical guarantees for the estimators obtained as solutions to the corresponding optimization problems, there are two types of assumptions that are likely necessary for a formal result": that weights fitted on pre-treatment data approximate treated-period counterfactuals (a latent factor model, or treatment periods chosen at random), and that the time average approximates the conditional expectation as `T` grows.

**Reference implementation(s):** none stated. The solver is SCIP (p.9); the data file is taken from the `synthdid` repository (footnote 4, p.14). The reviewed PDF names no code release (search: "code", "github", "package", "replication"; the only "github" match is the data URL).

### What the paper does NOT provide

- **No minimum detectable effect and no power calculation for a chosen design at design time.** Power appears as an evaluation result: Figure 2 (p.35) plots rejection probability against the true ATET for each method in a simulation (search: "power", "detectable"; "detectable" has no match).
- **No held-out pre-treatment periods for the design.** All `T` pre-treatment periods enter the objective (p.5). A training/validation split of pre-treatment periods is mentioned only as one way to choose `lambda` (p.17). "overfit" has no match.
- **No difference-in-differences estimator.** The estimators are weighted differences of post-period means. "difference-in-differences" appears once, in the title of Doudchenko and Imbens (2016) in the references (p.20); footnote 3 (p.14) cites that work by author and year. "parallel" has no match.
- **No covariates in the design** (p.5 and footnote 3, p.14).
- **No target population.** The estimand is defined on the selected units (p.5); the term "representative" has no match.
- **No spillover constraint.** Interference is cited as a reason to treat aggregate units (p.3) and arises as an artefact of one simulation design (footnote 7, p.34).
- **No staggered adoption** (search: "staggered"; no match).

---

**The registry-candidate text ends here. Everything below is audit material.**

## Results

### Simulation design (Section 5, p.14)

US Bureau of Labor Statistics unemployment rates for 50 states over 40 consecutive months. "We run 500 simulations such that each simulation utilizes a 10-by-10 matrix sampled from the original 50-by-40 dataset. Specifically, we randomly select 10 units and the first time period. The remaining 9 time periods are the consecutive months that follow." `K` treated units (3 or 7) are "chosen based on the data in the first 7 periods - or chosen randomly in cases (iv) and (v) - and the treatment is applied in the last 3 of the 10 periods." Effects: homogeneous, additive 0.05 for every treated unit; or heterogeneous, increasing "linearly from 0 to 0.1 from the first unit in the (randomly selected) 10-by-10 matrix to the last one", so that the ATET depends on which units are selected while "the (overall) ATE remains 0.05".

Procedures compared (pp.12, 14): (i) per-unit problem; (ii) two-way global; (iii) one-way global; (iv) per-unit problem with randomly chosen treated units, "equivalent to using the synthetic-control method for each randomly chosen treatment unit separately and then either averaging the unit-level treatment effect estimates or using the individual estimates directly"; (v) randomized assignment with difference in means.

How the RMSEs are formed (pp.14-15): "Only the per-unit problem, (i), and the synthetic control, (iv), allow for nontrivial estimation of heterogeneous treatment effects while the remaining approaches estimate all unit-level effects as being equal to the estimate of the average treatment effect on the treated." Squared errors "are computed for each treatment period (and each treatment unit in case of the unit-level effects) and averaged. The square roots of these quantities are the RMSE's in question. Table 1 reports the RMSE's averaged across all simulations."

### Table 1 (p.16): RMSE of average and unit-level treatment-effect estimates

"The reported RMSE's are multiplied by 10^3 for readability."

Homogeneous treatment:

| | K = 3 ATET | K = 3 unit-level | K = 7 ATET | K = 7 unit-level |
|---|---|---|---|---|
| (i) Per-unit | 8.5 | 13.9 | 8.3 | 16.0 |
| (ii) Two-way global | 8.4 | 8.4 | 8.4 | 8.4 |
| (iii) One-way global | 8.5 | 8.5 | 8.5 | 8.5 |
| (iv) Synthetic control (random treat.) | 9.7 | 15.9 | 10.3 | 19.0 |
| (v) Diff-in-means (random treat.) | 12.1 | 12.1 | 11.5 | 11.5 |

Heterogeneous treatment:

| | K = 3 ATET | K = 3 unit-level | K = 7 ATET | K = 7 unit-level |
|---|---|---|---|---|
| (i) Per-unit | 8.5 | 13.9 | 8.3 | 16.0 |
| (ii) Two-way global | 8.6 | 27.6 | 8.9 | 32.5 |
| (iii) One-way global | 8.5 | 27.6 | 8.5 | 32.5 |
| (iv) Synthetic control (random treat.) | 9.7 | 15.9 | 10.3 | 19.0 |
| (v) Diff-in-means (random treat.) | 12.1 | 29.7 | 11.5 | 33.6 |

Authors' summary (p.15): the global objectives "provide improvements over the baselines that vary from 12% to 31% in the homogeneous treatment case and from 11% to 30% in the heterogeneous treatment case when estimating the average treatment effect on the treated"; the per-unit approach gives "an improvement of over 13% relative to the synthetic control approach, (iv), when the number of treated units is small and 16% when the number of treated units is large" for unit-level effects with heterogeneous treatment. Reviewer arithmetic from the displayed (rounded) cells: (15.9 - 13.9) / 15.9 = 12.6% for `K = 3` and (19.0 - 16.0) / 19.0 = 15.8% for `K = 7`; the rounded table cannot confirm "over 13%".

### Power curves (A.4, pp.33-35)

100 simulations; 10 randomly chosen units and all 40 periods; treatment in the last 5 periods on 3 treated units, chosen "either randomly or using a mixed-integer program"; effects 0, `tau/2`, `tau` assigned to the three chosen treated units "in the order that corresponds to their order in the sampled data" (simple average effect on the treated `tau/2` whichever units are chosen, used as the nominal ATET for all methods; the paper does not say how this relates to the two-way method's estimand, which under Section 2 is weighted by the fitted treated weights, and Section 5 likewise does not say which benchmark its two-way RMSE uses); 40 permuted samples per simulation; nominal size 10%. Footnote 7 (p.34): this effect assignment makes potential outcomes "depend on the total number of treated units as well as the order of the treated units introducing interference", and is used so that the true ATET is the same across simulations and methods. Footnote 8: 40 samples are used for both schemes because moving blocks allow at most `S = 40` distinct samples. "Within each simulation, we repeat this procedure for each of the 40 permuted samples" (p.34). Reported in Figure 2 (p.35) as curves only, no table. Text conclusion: "the methods proposed in the paper - the per unit problem as well as the two- and one-way global problems - have test sizes at most as large and the power generally exceeding those of the randomized methods" (p.34). Reading the figure (no numbers are printed): at zero effect the curves lie roughly between 0.03 and 0.11 for i.i.d. permutations and between 0.04 and 0.12 for moving blocks; the dotted randomized difference-in-means curve is at or near the bottom of that range, and the two solid curves (per-unit and one-way global by the legend) plot above it in both panels; the "at most as large" statement about size is not visibly true against that baseline. (Approximate reading of a small figure; the line styles of the other curves cannot be told apart reliably near zero.) The x-axis of Figure 2 runs from 0 to 0.05; footnote 9 notes "the effect size of 0.05 corresponds to roughly the average outcome value observed in the data."

### How the paper positions itself (pp.4-5)

- Doudchenko et al. (2019): "consider the case when the underlying effects are homogeneous and only a single unit can be treated; they suggest searching for the unit that delivers the highest statistical power when testing the hypothesis of zero treatment effect with an artificially applied treatment."
- Abadie and Zhao (2021): differences listed are the use of covariates and a subset of pre-treatment periods (so that "the inference procedure proposed by Abadie and Zhao (2021) does not apply to our approach"), the formal hardness results, and the estimand (fixed a priori there "as either the population average treatment effect or a weighted version thereof"; the selected units here).

---

## Implementation Notes (audit notes - NOT registry-candidate)

### Data Structure Requirements
- A balanced `N x T` matrix of pre-treatment outcomes for the design; no covariates. The outcomes of the `S - T` treated periods for estimation and inference.
- The number of treated units `K` (required for the per-unit problem as formulated; optional for the global problems).
- A penalty `lambda`.

### Computational Considerations
- Mixed-integer quadratic programs with a convex objective and linear constraints after linearisation (the variable-`K` per-unit variant adds a quadratic constraint). Per-unit: `O(N^2)` continuous variables plus `N T` residual variables; two-way global and fixed-`K` one-way global: `O(N)` plus `T`; variable-`K` one-way: `O(N^2)` plus `T` (reviewer counts from A.2; the paper states the `O(N)` to `O(N^2)` increase).
- NP-hard: proved for the two global formulations (A.5). Problems with `N = 50` were solved "on a single machine within hours" (p.18); SCIP is named as the solver for the results of Sections 4 and 5 (p.9). No optimality gaps, time limits or solver settings are reported.
- Inference re-estimates on each permuted sample (p.32). The simulation description suggests the whole procedure, selection included, is repeated per permuted sample (p.34; see Gaps), which for the optimized designs would mean one mixed-integer program per permuted sample (under that reading method (iv) refits one synthetic control per randomly chosen treated unit and method (v) recomputes a difference in means; the paper leaves the refitting sequence implicit).

### Tuning Parameters

| Parameter | Type | Setting reported in the paper | Selection method |
|---|---|---|---|
| `K` | integer | 3 and 7 in simulations | analyst's choice |
| `lambda` | non-negative real | in Sections 4 and 5: average over units of the per-unit sample variance across pre-treatment periods (pp.17-18); no general default | that rule, or cross-validation on a training/validation split of pre-treatment periods (p.17) |
| formulation | per-unit / two-way / one-way | none | per-unit when distinct unit-level effects are the target (only (i) and (iv) estimate them non-trivially, p.14); for unit-level effects under homogeneous treatment the paper favors pooling (p.17); for the ATET under homogeneous treatment Table 1 has two-way global lowest at `K = 3` and per-unit lowest at `K = 7` |
| permutation scheme | i.i.d. / moving block | both reported | the conditions for moving blocks "are less strict" but there are at most `S` distinct samples (p.32); those conditions are stated for fixed treated units (p.33) |

### Relation to Existing diff-diff Estimators
- The paper's estimators are weighted differences of post-period means with weights fitted on pre-treatment outcomes. The library's `SyntheticControl` fits control weights for a given treated unit; it does not choose treated units.
- The per-unit objective with the treated set held fixed is a ridge-penalized synthetic-control fit per treated unit (the paper notes the penalty and cites Doudchenko and Imbens, 2016, and Abadie and L'Hour, 2021; p.7).
- The paper gives no result for selecting units and then estimating by difference-in-differences. Theorem 1 does show which pre-period features each objective selects on (levels of group means, within-group spread), which is information a difference-in-differences design would also have to reason about.

---

## Gaps and Uncertainties

1. **Index in the per-unit normalization constraint.** Section 3 (p.9) and A.2 (p.24) print the constraint as `sum_{i=1}^{N} w^i_j (1 - D_j) = 1`, summing over `i`; Section 2 (p.8) and the `T = 1` version (p.10) sum over the control index `j`. The `j`-sum is the reading consistent with the text ("for i = 1, ..., N such that D_i = 1") and with the linearised form `sum_j q_ij = D_i` (p.25). Treated here as a typographical slip; confirm against the proceedings version.
2. **Whether permutation inference repeats the selection.** The general description says "For each bootstrap sample we re-estimate the average treatment effect on the treated" (p.32). The simulation description first says the treated units are chosen "either randomly or using a mixed-integer program" and then "Within each simulation, we repeat this procedure for each of the 40 permuted samples" (p.34), which suggests the selection is repeated there. The paper does not list what is refitted per permutation.
3. **Proposition 1's assumption** (outcome vectors i.i.d. across time) is called "unrealistic in many time series settings" by the authors (p.33). Weaker residual-based conditions are cited from Chernozhukov et al. for fixed treated units; no result under them is proved for the paper's own data-driven selection. The size and power evidence is Figure 2 (100 simulations, 40 permutations each).
4. **Section 4's theorem is for `T = 1` and unrestricted-sign weights** (p.10). Footnote 2 (p.10): "In some cases the nonnegativity constraint will not be binding, while in other cases the optimal weights without the constraint may actually turn out to be negative. This implies that only a subset of all units will have strictly positive weights in the optimal constrained solution." No closed form is given for the binding case.
5. **Table 1 magnitudes.** Unscaled RMSEs range from 0.0083 to 0.0336; the homogeneous effect is 0.05, and in the heterogeneous case 0.05 is the overall ATE while the ATET depends on the units selected; the homogeneous and heterogeneous panels report identical values for rows (i) and (iv). Reviewer inference: if the same sampled panels, assignments and pre-treatment fits are used in both scenarios, an additive effect shifts the treated outcome and its target by the same amount, so per-unit estimation errors are unchanged. The paper does not say that the draws were paired across scenarios and does not comment on the equality.
6. **`sigma^2` versus `lambda`.** Section 2 motivates the penalty as the noise variance; Section 3 replaces it by a tuning parameter. The rule used in Sections 4 and 5 (average per-unit variance over time) is a variance of the outcome, which includes signal as well as noise; for `T = 2` in Section 4 it is written as `sum_i (Y_i1 - Y_i2)^2 / (4N)` (p.12), i.e. with divisor `T`; using divisor `T` for longer panels would be an extrapolation from that example.
7. **Other printed slips.** The control-group sums in the intermediate expansions of A.3 (pp.29-30) contain `(a_bar_I - a_j) a_j` where the weight formulas use `a_bar_Ibar - a_j`; the final expressions agree with Theorem 1.
8. **Proceedings version not compared.** Page numbers, the supplement and any corrections may differ in the NeurIPS version.
