# Paper Review: Synthetic Controls for Experimental Design

**Authors:** Alberto Abadie (MIT), Jinglong Zhao (Boston University)
**Citation:** Abadie, A., & Zhao, J. (2026). Synthetic Controls for Experimental Design. Working paper, dated February 2026. Earlier versions: arXiv:2108.02196 (v1 August 2021, v5 April 2025).
**Source reviewed:** the February 2026 draft hosted by the MIT Department of Economics ([PDF](https://economics.mit.edu/sites/default/files/2026-02/Synthetic%20Controls%20for%20Experimental%20Design%20Feb%202026.pdf)), a 41-page PDF: title page and abstract (PDF page 1, which carries no printed number), text (pp.2-30), references (pp.30-33), Tables 1-3 (pp.34-36), Figures 1-5 (pp.37-41). The PDF was reviewed externally and is not committed to the repository. The reviewed PDF does NOT contain the appendix: the text refers to an "appendix" with "proofs and supplemental materials" (p.5) and to an "online appendix" (pp.14, 21, 24, 25, 26, 28) that is a separate document, a 67-page supplementary appendix hosted beside the draft ([PDF](https://economics.mit.edu/sites/default/files/2026-02/Synthetic%20Controls%20for%20Experimental%20Design%20Feb%202026%20supplementary%20appendix.pdf)). The appendix was NOT reviewed, with two exceptions: the proof of Theorem 2 (Section OA.7.2, appendix pp.OA.15-OA.17) and the statement of Theorem OA.1 (Section OA.3, appendix pp.OA.4-OA.5) were read for the reviewer notes on the test, and the appendix's section titles are listed in Gaps item 1 as pointers. All page citations are the page numbers printed in the PDF (citations to p.1 refer to the unnumbered first PDF page). Greek letters and mathematical symbols are transliterated to ASCII (`tau`, `lambda`, `xi`, `||x||`, `sum_j`, `<=`); overlined and underlined symbols are written with `_bar`, `_hi` and `_lo` suffixes; inside quoted passages the paper's em dashes are written as spaced hyphens. Unless a range is shown, `sum_i` and `sum_j` run over units `1, ..., J`.
**Review date:** 2026-11-27

> **Personal verification stamp.** The reviewer read the full 41-page text and checked directly against it: design programs (7), (9) and (10) with every constraint (pp.9-12); the estimator (8) and the aggregation (11) (pp.11, 13); Assumptions 1-4 (pp.14-17); the bias bounds (13) and (14) (pp.17-18); the null hypothesis (15), the statistic (16), the p-value (17) and Theorem 2 (pp.19-21); the interval (18)-(19) and Theorem 3 (pp.22-23); every cell of Tables 1, 2 and 3 (pp.34-36); and the Walmart design parameters (p.23). Terms reported as absent were searched in the PDF text layer (search terms are listed with each such statement).

---

## Methodology Registry Entry

**Status: proposed text for a possible future REGISTRY section; no such section exists in `docs/methodology/REGISTRY.md` today; this file is a non-authoritative source audit.** The paper is a candidate source for a planned unit-selection capability (choosing which units to treat in a planned experiment from pre-treatment data). Nothing below is a methodology contract until a REGISTRY section is written and reviewed alongside an implementation. The registry-candidate text ends at the marker before `## Results`.

## Synthetic control design (working title)

*The paper's own name is "synthetic control designs" (p.3).*

**Primary source:** Abadie, A., & Zhao, J. (2026). Synthetic Controls for Experimental Design. Working paper, February 2026 (arXiv:2108.02196 for earlier versions).

**Key implementation requirements:**

*Scope of the source:*
- Setting: "experimental design in settings where the experimental units are large aggregate entities (e.g., markets), and only one or a small number of units can be exposed to the treatment" (abstract, p.1).
- The designs are NON-RANDOMIZED: "experimental non-randomized synthetic control designs ... that select the units to be treated, as well as the untreated units to be used as a control group" (abstract, p.1).
- Two stated goals for the treated units: "First, the treated units should be representative of an aggregate of interest, such as a national market, so that the estimated effect reflects the aggregate impact of the treatment. Second, the treated units should not be idiosyncratic in the sense that the untreated units cannot closely approximate their features." (p.3)
- Timing: `T` periods, `J` units, `T0` pre-experimental periods with `1 <= T0 < T`; "At the end of period T0, a researcher designs an experiment to conduct during periods T0 + 1, T0 + 2, ..., T." (p.5). All selected units start treatment at `T0 + 1` (p.6); the paper has one adoption date.

*Estimand (Section 2.1, p.6; Section 2.2, pp.8-9):*

    tau_t = sum_{j=1..J} f_j * (Y^I_jt - Y^N_jt)            (Equation 1)

for `t = T0+1, ..., T`, where `f_1, ..., f_J` "are known positive weights that define the average of interest" and are taken to sum to one; equal weighting is `f_j = 1/J` (p.6). `Y^I_jt` is the potential outcome when unit `j` is treated from `T0 + 1`; `Y^N_jt` is the potential outcome under no treatment in all periods (p.6).

The alternative target is the `w`-weighted average treatment effect on the treated,

    tau^T_t = sum_{j=1..J} w_j * (Y^I_jt - Y^N_jt)          (Equation 6)

*Decision variables and feasibility (Equation 2, p.7):* treated weights `w = (w_1, ..., w_J)` and control weights `v = (v_1, ..., v_J)` with

    sum_j w_j = 1,   sum_j v_j = 1,   w_j >= 0,   v_j >= 0,   w_j * v_j = 0 for all j.

"Units with w_j > 0 are units that will be assigned to the intervention of interest from T0 + 1 to T" (p.7). Units with `w_j = 0` form the donor pool; those among them with `v_j > 0` estimate the no-intervention average (p.7). The weights `w = f` are not allowed "because it would leave no units in the donor pool" (p.7).

*The two goals as equations (pp.7-8):*
- Equation (3): `sum_j w_j Y^I_jt = sum_j f_j Y^I_jt` (the treated reproduce the population average under treatment).
- Equation (4): `sum_j v_j Y^N_jt = sum_j f_j Y^N_jt`, "or, alternatively," Equation (5): `sum_j v_j Y^N_jt = sum_j w_j Y^N_jt`.
- If (3) and (4) hold, `tau_t(w, v)` equals `tau_t`; if (5) holds instead it equals `tau^T_t` (pp.8-9). Neither can be imposed directly because the potential outcomes are unobserved at design time; the designs match observed predictors instead (pp.7-8).

*Estimator (Equation 8, p.11):*

    tau_hat_t = sum_j w*_j Y_jt - sum_j v*_j Y_jt,    t = T0+1, ..., T.

This is a difference of two weighted averages of observed outcomes. It is not a difference-in-differences estimator: no pre-period difference is subtracted.

*Predictors (Section 3, p.15):* `X_j = (Y^E_j, Z_j)`, where `Z_j` is an `R x 1` vector of observed covariates and `Y^E_j` is the `T_E x 1` vector of pre-experimental outcomes for the fitting periods `E`, a subset of `{1, ..., T0}` with `T_E = |E|`. `X_bar = sum_j f_j X_j` is the population value (p.9). "In practice, the values in X_j are often scaled to make them independent of units of measurement or to reflect the relative importance of each of the predictors" (p.15).

*Design programs (Section 2.3, pp.9-13).* Programs (7) and (9) use the constraints of Equation (2); program (10) uses its own unit-level constraints, listed with it below. All three bound the number of treated units, `m_lo <= ||w||_0 <= m_hi` with `1 <= m_lo <= m_hi <= J - 1` (the paper writes the two bounds as underlined and overlined `m`; p.9; `m_lo` and `m_hi` are this review's transliteration). `||.||` is the Euclidean norm, and the objectives use its square (p.9).

1. **Base design (Equation 7, pp.9-10):**

       min over w, v   || X_bar - sum_j w_j X_j ||^2  +  || X_bar - sum_j v_j X_j ||^2

   Both the synthetic treated unit and the synthetic control unit are fitted to the population predictor vector. "We say that the design is Unconstrained if m_lo = 1 and m_hi = J - 1; otherwise, we say the design is Constrained." (p.10). Symmetry: "for every solution to (7) with m_lo <= ||v||_0 <= m_hi, there exists another solution that swaps the roles of the treated and the untreated in the experiment without altering the value of the objective function" (p.11).

2. **Weakly targeted design (Equation 9, p.11):**

       min over w, v   || X_bar - sum_j w_j X_j ||^2  +  beta * || sum_j w_j X_j - sum_j v_j X_j ||^2

   with `beta > 0`. "A small value of beta favors designs with treated units that closely match X_bar, and prioritizes estimation of tau_t ... A large value of beta, on the other hand, favors designs with aggregate treated and aggregate control units that closely match each other, and prioritizes estimation of tau^T_t" (p.11). "While it is possible to use data-driven selectors of beta, the rule of thumb beta = 1 provides a natural choice that equally weights the two terms" (pp.11-12). Assignment and estimation "follow the same steps as those used in the formulation in (7)" (p.12): treat the units with `w*_j > 0` and estimate by Equation (8).

3. **Unit-level design (Equation 10, p.12):** one synthetic control per treated unit, with weights `v_ij` on control `i` for treated unit `j` and `J_w = {j : w_j > 0}`:

       min over w, {v_ij}   || X_bar - sum_j w_j X_j ||^2  +  xi * sum_j w_j * || X_j - sum_i v_ij X_i ||^2

   subject to `sum_j w_j = 1`, `w_j >= 0`, `sum_i v_ij = 1` for `j` in `J_w`, `v_ij = 0` for `i` in `J_w` (all `j`), `v_ij >= 0` for `j` in `J_w`, `v_ij = 0` for `j` not in `J_w`, and the cardinality constraint. `xi > 0`: "A small value of xi favors experimental designs with treated units that closely match X_bar. A large value of xi, on the other hand, favors designs where the values of the predictors for the treated units are closely matched by those of their synthetic controls." (p.13). Equation (11) (p.13) prints the aggregate control weight as `v*_j = sum_{i=1..J} w*_i v*_ij`, and the estimator is then expanded as `tau_hat_t = sum_j w*_j ( Y_jt - sum_i v*_ij Y_it )`. **The subscripts of (11) as printed are transposed relative to (10)**, where `v_ij` is the weight on control `i` in the synthetic control of treated unit `j` and is zero whenever `i` is treated or `j` is untreated; read literally, (11) would give every unit zero control weight. Reviewer reconstruction from (10) and from the expanded estimator: the aggregate weight on control unit `i` is `v*_i = sum_j w*_j v*_ij`. The paper states this design "estimates the average treatment effect on the treated, as defined in equation (6), but typically relies on a small set of treated units even in the absence of a sparsity constraint on w" (p.12).

4. **Clustered designs (pp.13-14, Figure 1 p.37):** where units divide naturally into clusters with similar predictors, "it is natural to treat each cluster ... as a distinct experimental design to ameliorate interpolation biases" (p.13). No program is printed for this case in the reviewed PDF; the appendix's Section OA.1 gives a clustered version of the design (not reviewed).

5. **Penalized design:** "the online appendix introduces a penalized version of the synthetic control design, which further limits the number of units in both the synthetic treated and control groups, even in the absence of a hard cardinality constraint" (p.14). Not in the reviewed PDF.

*Optimality is not required (p.10):* "In practice, we do not require optimality of (w*, v*), as long as (w*, v*) is feasible and satisfies X_bar - sum_j w*_j X_j ~ 0 and X_bar - sum_j v*_j X_j ~ 0". For the unit-level design the paper likewise does "not strictly require optimality" provided the solution is feasible, the synthetic treated unit approximately reproduces `X_bar`, and each treated unit is approximately reproduced by its own synthetic control "for all j such that w*_j > 0" (p.13). The second condition is printed as `X_j - sum_{j=1..J} v*_ij X_j ~ 0`, which reuses `j` as the summation index and leaves `i` unbound; reviewer reconstruction from (10): `X_j - sum_i v*_ij X_i ~ 0` for each `j` with `w*_j > 0`.

*Cost constraint (p.10):* "The cardinality constraint in (7) is not the only conceivable restriction to the size or cost of the experiment. An explicit upper bound on the cost of an experiment would be given by c'd <= B, where the j-th coordinate of c is equal to the cost of assigning unit j to treatment, d is a J-dimensional vector with ones at coordinates where w_j > 0, and zeros otherwise, and B is the experimenter's budget." This is the only mention of a budget in the reviewed PDF (search: "budget"); it is not used in the empirical or simulation sections.

*Model (Assumption 1, pp.14-15):* a linear factor model for both potential outcomes,

    Y^N_jt = delta_t   + theta_t' Z_j + lambda_t' mu_j + eps_jt        (Equation 12a)
    Y^I_jt = upsilon_t + gamma_t' Z_j + eta_t'    mu_j + xi_jt         (Equation 12b)

with `mu_j` an `F x 1` vector of unobserved covariates, `theta_t` and `gamma_t` `R x 1` and `lambda_t` and `eta_t` `F x 1` vectors of unknown parameters, and `eps_jt` and `xi_jt` "unobserved idiosyncratic noise terms whose distributions do not depend on the other components of the linear factor model" (p.15). Conditioning convention (p.16): "Unless otherwise noted, all probability statements are over the joint distribution of eps_jt and xi_jt and conditional on the values of the other components on the right-hand sides of equations (12a) and (12b)." (12b) extends the usual model to treated potential outcomes because "synthetic control designs require the choice of a treatment group in addition to the choice of a comparison group" (p.15).

*Regularity (Assumption 2, pp.15-16):* (i) `F <= T_E`, and the ratio `zeta_E / T_E` (the paper's unsubscripted zeta, which appears as the divisor in the bounds below) is strictly positive, where `zeta_E` is the smallest eigenvalue of `lambda_E' lambda_E` and `lambda_E` is the `T_E x F` matrix whose rows are the `lambda_t` for `t` in `E`; (ii) for each `j`, `eps_j1, ..., eps_jT` are i.i.d. sub-Gaussian, mean zero, variance proxy `sigma_bar^2` (the paper prints an overlined sigma; the simulation's noise variance is a separate, unbarred `sigma^2`), and `xi_j,T0+1, ..., xi_jT` likewise and independent of the `eps`. The eigenvalue ratio in (i) is printed as an underlined zeta and is written `zeta_lo` below. "Notably, Assumption 2(ii) allows for dependence of eps_jt and xi_jt across units." (p.16)

*Fit conditions:* Assumption 3 (perfect fit, pp.16-17): with probability one `sum_j w*_j Z_j = sum_j v*_j Z_j = sum_j f_j Z_j` and the same for `Y^E_j`. Assumption 4 (approximate fit, p.17): there is `d > 0` such that, with probability one, the squared Euclidean distances from the population targets are at most `R d^2` for the covariates and `T_E d^2` for the fitting-period outcomes, for both `w*` and `v*`. It is a sufficient condition for the bound (14), which grows with `d`; it is not an acceptance rule for a design.

*Bias bound (Theorem 1, pp.17-18).* `lambda_bar` is the maximum of `|lambda_tf|` over `t = 1, ..., T` and `f = 1, ..., F`; `theta_bar` the maximum of `|theta_tr|` over `t = 1, ..., T` and `r = 1, ..., R`; `eta_bar` and `gamma_bar` the corresponding maxima of `|eta_tf|` and `|gamma_tr|` over the experimental periods `t = T0+1, ..., T` (p.17):

    Under Assumptions 1-3:    |E[tau_hat_t - tau_t]| <= (lambda_bar (eta_bar + lambda_bar) F / zeta_lo) * sqrt(2 log(2J)) * sigma_bar / sqrt(T_E)      (Equation 13)

    Under Assumptions 1, 2, 4:
    |E[tau_hat_t - tau_t]| <= ( (gamma_bar + theta_bar) R + (lambda_bar (eta_bar + lambda_bar) F / zeta_lo) (1 + theta_bar R) ) d
                              + (lambda_bar (eta_bar + lambda_bar) F / zeta_lo) * sqrt(2 log(2J)) * sigma_bar / sqrt(T_E)                    (Equation 14)

Expectations are over the distribution of the noise terms (p.18). See Gaps item 6: the proof requires the expected weighted experimental-period noise to vanish, which the stated conditions do not ensure.

"The results in Theorem 1 do not depend on the specific formulation of the synthetic control design (e.g., Constrained vs. Unconstrained)." (p.18)

*Overfitting, as the paper describes it (p.18):* "Overfitting happens when pre-experimental outcomes are instead fitted out of the variability in the individual transitory shocks, eps_jt. A small number of fitting periods T_E combined with large variability in eps_jt increases the risk of overfitting and, as a result, increases the bias bound. Similarly, for any fixed value of T_E, the bias bound increases with J, reflecting the increased risk of over-fitting caused by increased variability in eps_jt over larger donor pools." The paper adds that the number of unobserved factors `F` enters (13) linearly, "which highlights the importance of including the observed predictors Z_j - other than pre-experimental outcomes - in the vector of fitting variables X_j" (p.18). Footnote 6 (p.19) qualifies this: moving predictors from `Z_j` to `mu_j` cannot increase the eigenvalue ratio `zeta_lo` and "cannot decrease the values of lambda_bar and eta_bar", so the bound (13) "cannot decrease and will typically increase"; "This is not necessarily true for the bound in (14), because a shift of components from Z_j to mu_j decreases the value of R."

*Blank periods (p.19):* the paper uses a set of blank periods `B`, a subset of `{1, ..., T0} \ E`, "which comprise pre-experimental periods whose outcomes Y_jt have not been used to calculate w* or v*." "Because pre-experimental periods that are not in E or B are not used in our procedure, without loss of generality, we consider B = {1, ..., T0} \ E" (p.19), so `T_B = T0 - T_E`: pre-experimental periods outside `E` and `B` may simply be left out; on the periods that are kept, `B` is the complement of `E`. In the blank periods the placebo effects are `u_hat_t = sum_j w*_j Y_jt - sum_j v*_j Y_jt` for `t` in `B`; in the experimental periods `u_hat_t = tau_hat_t` (pp.19-20).

*Test (pp.19-21):*
- Null hypothesis (Equation 15): for `t = T0+1, ..., T` and all `j`, `Y^I_jt = delta_t + theta_t' Z_j + lambda_t' mu_j + xi_jt` "where xi_jt has the same distribution as eps_jt". Under it "the distribution of Y^I_jt is the same as the distribution of Y^N_jt ... But the realized values of Y^I_jt and Y^N_jt may differ." (p.19). This is a distributional null, not a sharp null.
- Statistic (Equation 16): `S(u_hat) = (1 / (T - T0)) * sum_{t=T0+1..T} |u_hat_t|`. It averages the absolute experimental-period gaps. Alternatives named: an `L_p` norm, and one-sided versions "with the positive or the negative parts of u_hat_t replacing |u_hat_t|" (p.20).
- Reference set: `Pi`, all subsets `pi` of size `T - T0` of the union of `B` and `{T0+1, ..., T}` (the experimental periods themselves are one such subset); `|Pi| = (T - T_E)! / ((T - T0)! (T0 - T_E)!)` (p.20). For each `pi`, `S(u_hat_pi) = (1 / (T - T0)) * sum_{t in pi} |u_hat_t|` (p.20). The weights are not refitted. With no blank periods `Pi` contains only the experimental periods and the p-value is 1.
- p-value (Equation 17): `p_hat = (1/|Pi|) * sum_{pi in Pi} 1{ S(u_hat_pi) >= S(u_hat) }`.
- Theorem 2 (pp.20-21): under Assumptions 1, 2(ii), 3(i), continuous distributions for the noises `eps_jt` over `B` and the experimental periods and `xi_jt` over the experimental periods, and `{lambda_t}` exchangeable over `B` and the experimental periods, under the null, for any `alpha` in `[0, 1]`: `alpha - 1/|Pi| <= Pr(p_hat <= alpha) <= alpha`, the probability being over the noises and `{lambda_t}` (p.21). See Gaps item 6: joint exchangeability of the absolute full gaps across blank and experimental periods gives the upper bound `Pr(p_hat <= alpha) <= alpha`, the lower bound needs in addition that the permuted statistics be distinct with probability one, and the stated conditions deliver neither.
- Relaxations stated in prose (p.21): with exchangeable loadings retained, the i.i.d. condition of Assumption 2(ii) can be weakened to exchangeability of each unit's noise sequences (`eps_jt` over blank and experimental periods, `xi_jt` over experimental periods); if Assumption 3(i) fails the result holds if `{(theta_t, lambda_t)}` is exchangeable and independent of the noise; and "Theorem OA.1 in the online appendix relaxes this restriction by showing that for fixed lambda_t ... the p-value in (17) is still approximately valid for large T_E" (its statement, appendix Section OA.3, was read for this point; its proof was not). Theorem OA.1 assumes Assumptions 1-3 and a bounded density for the continuously distributed `eps_jt`, under the null (15). Example (a) in Gaps item 6 satisfies those conditions and has identical `mu_j` for the two units, so it applies whatever the loadings are, and that result is subject to the same caveat.
- Large `|Pi|`: "random samples from Pi can be used to approximate the p-value" (p.21).
- Unit-reassignment tests are ruled out: "Inferential methods that reassign treatment across units (e.g., Abadie, Diamond and Hainmueller, 2010) are not appropriate for the designs of Section 2, which explicitly select treated and control units to satisfy an optimality criterion." (pp.21-22)

*Confidence interval (Equations 18-19, p.22):* for `alpha` in `(0, 1)` and a non-empty blank set (`T_B > 0`), `q_hat_{1-alpha} = inf{ z in R : (1/(T0 - T_E)) * sum_{t in B} 1{ |u_hat_t| <= z } >= 1 - alpha }` (Equation 18), i.e. the `ceil((1 - alpha) * T_B)`-th smallest absolute blank-period gap (reviewer restatement); the interval for `tau_t` is `tau_hat_t +/- q_hat_{1-alpha}` (Equation 19). Theorem 3 (p.23): under Assumptions 1-3, a finite constant `kappa` bounding the densities of the continuously distributed `eps_jt` for all `j` and all `t = 1, ..., T`, and `xi_jt` distributed as `eps_jt` for every `j` and every experimental `t`, coverage differs from `1 - alpha` by `O( (log(T0 - T_E)/(T0 - T_E))^(1/2) + (log T_E / T_E)^(1/2) )` as `T0 - T_E` and `T_E` go to infinity. Coverage is pointwise in `t`. See Gaps item 6 for a case that satisfies the stated conditions and has zero coverage.

*Edge cases the paper addresses:*
- One treated unit not giving a good fit: in the Walmart illustration "using only one store for the synthetic treated fails to produce a good fit between the resulting synthetic treated and synthetic control units during the fitting period" (p.24), so `m_hi = 2` is used.
- Swapping treated and control roles: for program (7) a swap leaves the objective unchanged and is feasible when the control support also satisfies the treated-count bounds (p.11). In the Section 5 simulations, "When it is possible to swap synthetic treated and synthetic control weights, we choose the treated units so that the number of units with positive weights in w* is smaller than the number of units with positive weights in v*. When ||w*||_0 = ||v*||_0, we determine whether to swap using a specific rule described in the online appendix." (pp.25-26). The rule is not in the reviewed PDF.
- Stratified randomization as an alternative: with one treated unit "we would be limited to a single stratum" (p.5).

**Reference implementation(s):** "The replication code can be found at GitHub." (title-page footnote, PDF page 1). The word "GitHub" is a link whose target in the PDF is `https://github.com/jinglongzhao2/SCDesign`; the repository was not examined for this review. The text names no solver or software (search: "package", "solver").

### What the paper does NOT provide

Each item states what was searched in the text layer of the reviewed PDF.

- **No prospective power calculation, power curve or minimum detectable effect for a design.** "power" appears only (a) in the description of Doudchenko et al. (p.4), (b) once as "out-of-sample predictive power" (p.24), and (c) in "demonstrate the power of synthetic controls" and "a powerful tool" (pp.29-30). "detectable" has no match. The closest quantity is the column of empirical rejection rates at the 0.05 level in Table 2, one per design variant, obtained under one simulated non-null data-generating process in which the effect path differs across replications (the model primitives are redrawn in each of the 1000 simulations, p.26).
- **No size (rejection rate under a true null) from simulation in the reviewed PDF.** The online appendix has a test-size simulation (Section OA.9.2.3, Table OA.3; not reviewed). The simulation of Section 5 has a non-zero effect ("the treatment effect is not equal to zero in the simulation of Table 2", p.27). For the Walmart placebo the PDF reports one p-value (0.933, for the `m_hi = 2` design, p.24); Table 1 compares further configurations by normalized RMSE only.
- **No difference-in-differences estimator and no parallel-trends condition.** "parallel" has no match. "Difference-in-differences" matches only two titles in the references: Arkhangelsky et al. (2021) (p.30) and Doudchenko and Imbens (2016) (p.31). The comparison estimators are difference in means, stratified difference in means, regression adjustment and nearest-neighbour matching (pp.27-28).
- **No algorithm or solver for programs (7), (9), (10)**, and no hardness statement, in the reviewed PDF (the appendix has a section on implementations, Section OA.6, not reviewed; search of the reviewed PDF: "solver", "NP", "enumerat" have no match; "algorithm" matches only the book title "Algorithmic Learning in a Random World" in the references, p.33; "integer" matches only "positive integers" in the definition of the cardinality bounds, p.9). The phrase "computationally expensive" refers to the p-value when `|Pi|` is large (p.21); the conclusion mentions "computational feasibility" (p.29) without detail.
- **No rule for choosing `E` or its length.** Given `E`, the paper takes `B` to be the remaining pre-experimental periods that are used (p.19), so choosing `E` and which periods to keep fixes the split. The illustrations use `T_E = 100`, `T_B = 28` of `T0 = 128` (p.23) and `T_E = 20`, `T_B = 5` of `T0 = 25` (p.25). The Walmart text says the fit is judged "during the fitting period" (p.24).
- **No spillover or interference constraint in the design programs.** Interference appears as motivation for market-level treatment (pp.2-3, footnote 1); "spillover" has no match; no exclusion or buffer constraint is printed.
- **No staggered adoption** (search: "staggered"; no match).
- **No random component in the assignment.** The proposed designs are non-randomized; randomized assignment appears as the comparator, in the Walmart illustration (p.24) and in the simulations (pp.27-28). A design is not always unique: the roles of treated and control can be swapped at the same objective value (p.11), and reproducing the simulations' assignments needs their role-selection rule, part of which is in the online appendix (pp.25-26).

---

**The registry-candidate text ends here. Everything below is audit material.**

## Results

### Table 1 (p.34): Walmart data, out-of-sample normalized RMSE

"Root mean square error divided by the average outcome in the experimental periods. m_hi stands for the maximum number of treated units." For the four randomization-based columns "the reported RMSE is the average over 1000 randomized treatment assignments" (p.24).

| | SC | RND | STR | 1-NN | 5-NN |
|---|---|---|---|---|---|
| m_hi = 1 | 0.052 | 0.452 | 0.452 | 0.096 | 0.082 |
| m_hi = 2 | 0.018 | 0.312 | 0.299 | 0.070 | 0.063 |
| m_hi = 3 | 0.019 | 0.254 | 0.173 | 0.059 | 0.053 |
| m_hi = 4 | 0.027 | 0.223 | 0.181 | 0.052 | 0.048 |
| m_hi = 5 | 0.012 | 0.202 | 0.164 | 0.047 | 0.043 |

Rows are indexed by `m_hi`, the maximum number of treated units (the paper's overlined `m`). SC: Constrained synthetic control design. RND: randomized assignment, difference in means. STR: stratified randomization on pre-intervention outcomes, difference in means within stratum. 1-NN / 5-NN: randomized assignment followed by nearest-neighbour matching on all pre-experimental outcomes. The estimators are defined in Section 5.2.2 (see Table 3 below). Setup (pp.23-24): store-level weekly sales from the Walmart data set of Prakash (2023), `J = 45` stores, `T = 143` weeks from the week of February 5, 2010 to the week of October 26, 2012, placebo experiment on July 20, 2012 (week 129), `T0 = 128`, the first `T_E = 100` weeks for fitting and the last 28 pre-experimental weeks blank, 15 experimental weeks, `f_j = 1/J`, each of the 100 fitting outcomes normalized to unit variance (p.24). Placebo p-value from Equation (17): 0.933; "Confidence intervals based on equation (19) cover zero for all t in the experimental period." (p.24); Figure 3's note identifies them as 95% intervals, one per experimental period (p.39).

### Table 2 (p.35): simulation, averages over 1000 simulations

True `tau_t`, averaged over the 1000 simulations, for `t = 26, ..., 30`: -13.58, -10.99, -8.35, -5.00, -2.50.

| Design | tau_hat 26 | 27 | 28 | 29 | 30 | MAE | RMSE | p_hat | p_hat < 0.05 | \|\|w\|\|_0 |
|---|---|---|---|---|---|---|---|---|---|---|
| Unconstrained | -13.57 | -10.97 | -8.38 | -5.07 | -2.53 | 0.83 | 0.97 | 0.014 | 0.944 | 6.76 |
| Constrained m_hi = 1 | -13.61 | -10.97 | -8.39 | -4.86 | -2.41 | 2.93 | 3.45 | 0.057 | 0.668 | 1 |
| m_hi = 2 | -13.58 | -10.90 | -8.43 | -5.01 | -2.40 | 1.69 | 2.00 | 0.028 | 0.854 | 2 |
| m_hi = 3 | -13.56 | -11.00 | -8.38 | -5.05 | -2.52 | 1.26 | 1.49 | 0.019 | 0.916 | 3 |
| m_hi = 4 | -13.59 | -11.06 | -8.40 | -4.99 | -2.50 | 1.06 | 1.25 | 0.016 | 0.935 | 4 |
| m_hi = 5 | -13.57 | -11.01 | -8.37 | -5.02 | -2.48 | 0.93 | 1.09 | 0.015 | 0.933 | 5.00 |
| m_hi = 6 | -13.51 | -10.95 | -8.29 | -5.01 | -2.47 | 0.87 | 1.02 | 0.015 | 0.942 | 5.97 |
| m_hi = 7 | -13.57 | -10.96 | -8.37 | -5.06 | -2.52 | 0.83 | 0.97 | 0.014 | 0.946 | 6.76 |

Rows below "Constrained" are indexed by `m_hi`. Design (p.25): `J = 15`, `R = 7` observed and `F = 11` unobserved covariates, `T = 30`, `T0 = 25`, `T_E = 20`, blank periods 21-25, `f_j = 1/J`; `delta_t` for `t = 1, ..., T` is the small-to-large rearrangement of `T` i.i.d. Uniform(0, 20) draws, and `upsilon_t` for `t = T0+1, ..., T` that of `T - T0` such draws; `Z_j`, `mu_j` are vectors of i.i.d. Uniform(0, 1); for `t = 1, ..., T`, `theta_t`, `gamma_t`, `lambda_t`, `eta_t` are vectors of i.i.d. Uniform(0, 10); for every `j` and every `t = 1, ..., T`, `eps_jt` and `xi_jt` are i.i.d. Normal(0, `sigma^2`) with `sigma^2 = 1`. The model primitives are redrawn independently in each of the 1000 simulations (p.26). MAE and RMSE are computed within a simulation over the five experimental periods, `MAE = (1/(T - T0)) sum_{t=T0+1..T} |tau_hat_t - tau_t|` and `RMSE = sqrt( (1/(T - T0)) sum_{t=T0+1..T} (tau_hat_t - tau_t)^2 )`, and then averaged over simulations (pp.26-27). The text says "the Unconstrained design delivers the best performance across all measures" (p.27); in the table the rejection rate at `m_hi = 7` (0.946) is slightly above the Unconstrained one (0.944). With `T - T0 = 5` and `T_B = 5`, `|Pi| = 10!/(5! 5!) = 252` (reviewer arithmetic from the formula on p.20), so the p-value is bounded below by 1/252, about 0.004 (attained when the observed statistic is strictly the largest), which matches the p-value of the single simulated sample reported on p.26 (base design (7) with `m_lo = 1` and `m_hi = 14`, i.e. no constraint on the number of treated units).

### Table 3 (p.36): simulation RMSE by design and estimator, averages over 1000 simulations

| | SC | RND | STR | REG | 1-NN | 5-NN |
|---|---|---|---|---|---|---|
| m_hi = 1 | 3.45 | 6.35 | 6.35 | 8.14 | 5.28 | 4.40 |
| m_hi = 2 | 2.00 | 4.70 | 3.53 | 6.00 | 3.69 | 3.20 |
| m_hi = 3 | 1.49 | 3.91 | 2.75 | 5.11 | 3.02 | 2.66 |
| m_hi = 4 | 1.25 | 3.49 | 2.44 | 4.49 | 2.67 | 2.40 |
| m_hi = 5 | 1.09 | 3.22 | 2.07 | 4.12 | 2.38 | 2.28 |
| m_hi = 6 | 1.02 | 3.04 | 1.95 | 3.87 | 2.24 | 2.23 |
| m_hi = 7 | 0.97 | 3.01 | 1.85 | 3.90 | 2.18 | 2.32 |

Rows are indexed by `m_hi`. Estimators (Section 5.2.2, pp.27-28): RND assigns `m_hi` units at random and takes the difference in means. STR forms `m_hi` blocks "with at least two units per block", assigns one unit per block to treatment at random, chooses the blocks "to minimize the maximal within-block discrepancy in the covariates, Z_j, and pre-experimental outcomes (all normalized to have unit variance)", and combines the within-block differences (treated unit minus the mean of the block's other units) with weights `J_k / J`, `J_k` the block size. (The printed STR formula gives the inner control sum an upper limit of `k`; reviewer reconstruction from the prose and the block-membership indicator: the sum runs over units `j = 1, ..., J`.) REG assigns `m_hi` units at random and adjusts by regression on `Z_j`. 1-NN and 5-NN assign `m_hi` units at random and match on all pre-experimental outcomes and covariates, "rescaled to have unit variance". Table note: "SC uses outcomes in the fitting periods and covariates as predictors. STR, 1-NN, and 5-NN use all pre-intervention outcomes and covariates. REG adjusts for the covariates only." "Ordinary least-squares adjustment on all pre-treatment outcomes is unfeasible because the number of pre-treatment outcomes exceeds the number of units in the sample" (p.28).

### Authors' summary of the design variants (p.14)

"the baseline formulation in (7) generally has the most accurate estimates of the average treatment effect ... the Weakly targeted formulation in (9) achieves high relative accuracy for the average treatment effect on the treated when beta is large. With large xi, the Unit-level formulation (10) also yields accurate estimates, but remains less accurate than the Weakly targeted design. Its main advantage is parsimony". The paper places the evidence in "Section 5 and the online appendix" (p.14). Section 5 and Tables 2 and 3 of the reviewed PDF cover only the Unconstrained and Constrained forms of (7), so the comparisons involving (9) and (10) are presumably in the online appendix (reviewer inference).

### How the paper positions itself against related work (pp.3-4)

- Doudchenko, Gilinson, Taylor and Wernerfelt (n.d.): "The focus ... is on statistical power, which they calculate by simulating the estimated effects of placebo interventions using historical (pre-experimental) data ... As a result, estimates based on the procedure ... target the effect of the treatment for the unit or units that are most closely tracked in the placebo distribution."
- Doudchenko et al. (2021): "the target parameter ... is the treatment effect for a weighted average of treated units that can be closely matched in their pre-treatment outcomes by a weighted average of untreated units"; its permutation procedure "requires that potential outcomes without the treatment are independent and identically distributed (i.i.d.) in time", whereas "the inferential procedure proposed in the present article allows for time series dependence and non-stationarity in outcomes"; and both Doudchenko papers "make use of pre-treatment outcomes only to select treated and control units", while this paper "allows the use of other observed features of the units".
- The paper says it knows of no other publicly available studies: "Aside from the present article, to our knowledge, Doudchenko et al. (n.d.) and Doudchenko et al. (2021) are the only other publicly available studies on the topic of experimental design with synthetic controls." (pp.3-4)

---

## Implementation Notes (audit notes - NOT registry-candidate)

### Data Structure Requirements
- A balanced panel of `J` units over `T0` pre-experimental periods, split by the analyst into fitting periods `E` and blank periods `B`. The design uses pre-experimental observations only; estimation and inference also need the experimental-period outcomes of the units with positive treated or control weight.
- Optional time-invariant unit covariates `Z_j`.
- Known population weights `f_j`.
- The number of treated units (bounds `m_lo`, `m_hi`), and optionally unit costs and a budget.

### Computational Considerations
- The programs are non-convex: the constraint `w_j * v_j = 0` and the cardinality constraint `||w||_0` are combinatorial. The reviewed PDF gives no solution method.
- The p-value needs `|Pi|` evaluations of a mean of absolute residuals; the weights are not refitted. `|Pi|` is 252 for five blank and five experimental periods and grows quickly (for the Walmart split, 43 choose 15); the paper allows random sampling from `Pi` (p.21).
- The smallest attainable p-value is `1/|Pi|`, so the number of blank periods bounds the significance level that can be reached.

### Tuning Parameters

| Parameter | Type | Default in the paper | Selection method |
|---|---|---|---|
| `m_lo`, `m_hi` | integers | none; `m_hi = 2` in the Walmart illustration | by fit: `m_hi = 1` did not fit, `m_hi = 3` gave "only marginal improvements in fit" (p.24) |
| `E` (and `B`, its complement in the pre-experimental periods that are used, p.19) | set of pre-periods | none | no rule for `E` |
| `beta` (weakly targeted) | positive real | `beta = 1` "rule of thumb" (p.12) | "possible to use data-driven selectors" (p.12), none given |
| `xi` (unit-level) | positive real | none | no numerical default or selector; small `xi` favors population fit of the synthetic treated unit, large `xi` favors fit of each treated unit by its own synthetic control (p.13) |
| predictor scaling | per-predictor | unit variance in the Walmart illustration (p.24) | analyst's choice (p.15) |
| `f_j` | positive weights summing to one | `1/J` in both illustrations | analyst's choice of target population |

### Relation to Existing diff-diff Estimators
- The library has a `SyntheticControl` estimator (`diff_diff/synthetic_control.py`) for a GIVEN treated unit. The paper's designs choose the treated units and fit a synthetic treated unit as well as a synthetic control; neither is something that class does.
- `diagnose_design` (`diff_diff/design.py`) assigns placebo treatment to units drawn at random (`n_treated` or `treatment_fraction`) at declared `cohort_periods` and scores estimators on those draws. The paper fixes one deterministic treated set; scoring a fixed treated set is not what `diagnose_design` does today.
- For each experimental period `t`, the paper's estimator (Equation 8) is the treated-weighted mean of outcomes across units minus the control-weighted mean across units; no pre-period difference is subtracted. The library's design-stage work scores difference-in-differences estimators; the paper supplies no result for a design followed by a difference-in-differences estimate.
- The blank-period idea (periods held out from the design, used for the reference distribution) is the paper's device for inference after a data-driven choice of units.

---

## Gaps and Uncertainties

1. **Appendix reviewed only in part.** Read for this review: the proof of Theorem 2 (Section OA.7.2) and the statement of Theorem OA.1 (Section OA.3). Everything else in it is unreviewed: the other proofs (including that of Theorem OA.1), the penalized design, the swap rule, the nonlinear simulation, further Walmart results for `m_hi = 1` and `m_hi = 3` (the counterparts of Figures 2-3; Table 1 of the reviewed draft does report normalized RMSE for both) are in the separate online appendix; the comparisons involving designs (9) and (10) appear to be there too (reviewer inference, see Results). Its section titles, as pointers only: OA.1 Other Versions of the Synthetic Control Design; OA.2 Designs Based on Penalized and Bias-corrected Synthetic Control Methods; OA.3 Approximate Validity when lambda_t are not Exchangeable; OA.4 Estimating the Average Effect of Treatment on the Treated Units; OA.5 Swapping Treated and Control Weights; OA.6 Implementations of Different Optimization Formulations; OA.7 Proofs; OA.8 Additional Results for the Walmart Data; OA.9 Additional Simulation Results. An implementation of these designs would need OA.5 and OA.6 reviewed first. arXiv v5 (April 2025) is an 80-page PDF that includes an appendix, but it is an earlier version and was not reviewed here; anything taken from it must be checked against the February 2026 text.
2. **Publication status.** The reviewed PDF is a working paper. A DOI at The Review of Economics and Statistics (10.1162/REST.a.1837) resolves to an entry with this title; the publisher's page could not be read during this review, so volume, issue and whether the published text matches the February 2026 draft are unverified.
3. **Solver.** How programs (7), (9) and (10) are solved, and how large a problem is feasible, is not stated in the reviewed PDF.
4. **How the fit is judged.** The design discussion calls for predictor discrepancies that are approximately zero (pp.10, 13) and gives no numerical acceptance threshold; Assumption 4 is a sufficient condition for the bias bound (14), which grows with `d`. The Walmart text compares `m_hi = 1, 2, 3` by fitting-period RMSE or MAE (p.24).
5. **Use of the blank periods before the experiment.** The paper uses `B` for the reference distribution of the test and for the interval, and defines blank periods as ones whose outcomes "have not been used to calculate w* or v*" (p.19). Reviewer implication: choosing among candidate designs by their blank-period gaps would use those outcomes in determining the final weights and so breach the hold-out; the paper does not discuss this case. The Walmart illustration describes the blank-period gap after the design is chosen ("The gap between the two synthetic units remains small after the fitting period", p.24).
6. **Exchangeability.** Theorem 2's exact result needs `{lambda_t}` exchangeable across blank and experimental periods, which the paper calls "a strong restriction" (p.21); the relaxation to fixed loadings, with approximate validity for large `T_E`, is Theorem OA.1 of the appendix (statement read, proof not; see item 1 for the scope of the appendix review). The introduction says the procedure "allows for time series dependence and non-stationarity in outcomes" (p.4). In Section 3, p.21 attributes the allowed non-stationarity to the term `delta_t + theta_t' Z_j`, and separately says the exact result still holds when the noise sequences are exchangeable instead of i.i.d. over time, given exchangeable loadings.

   **Reviewer notes: three results need more than the main text states once noise is dependent across units.** Assumption 2(ii) constrains each unit's own noise sequences (i.i.d. over time, `xi_j` independent of `eps_j`) and "allows for dependence of eps_jt and xi_jt across units" (p.16); the null (15) says that `xi_jt` "has the same distribution as" `eps_jt`. Nothing stated restricts how one unit's noise in one period relates to another unit's noise in another period, or how treated units' `xi_jt` relate to control units' `eps_jt`. Item 1 lists what was read in the appendix (the proof of Theorem 2 and the statement of Theorem OA.1, whose proof was not read); apart from the quoted proof step, the notes below follow from the definitions in the main text.

   - *Bias bound (Theorem 1).* The weights are functions of the fitting-period outcomes of all units. The bias proof needs the expected weighted experimental-period noise to vanish. Noise with mean zero conditional on the fitting data is sufficient for that, and independence of the noise across periods (jointly over units) delivers it; the stated per-unit conditions do not exclude a unit's fitting-period noise predicting another unit's experimental-period noise.
   - *Test (Theorem 2).* The appendix proof (Section OA.7.2) rests on the step that the null and Assumption 2(ii) "imply that" the noise gaps `u_t` (`sum_j w*_j eps_jt - sum_j v*_j eps_jt` in blank periods, `sum_j w*_j xi_jt - sum_j v*_j eps_jt` in experimental periods) form "a sequence of exchangeable random variables". The proof then adds the factor part `lambda_t' sum_j (w*_j - v*_j) mu_j` and concludes that the full gaps `u_hat_t` are exchangeable. A sufficient condition for the upper bound `Pr(p_hat <= alpha) <= alpha` is that the absolute values of the full gaps (factor part plus noise part), which are what the statistic (16) uses, be jointly exchangeable across blank and experimental periods; exchangeability of the gaps conditional on the information used to fit the weights is a stronger condition that delivers it. Exchangeability of the loadings by themselves does not supply that when the fitting data are informative about how the loadings are ordered across those periods (the fitted weights, and so any remaining imbalance in `mu_j`, can then depend on that ordering), and the noise part can fail on its own. Two examples for the noise part that satisfy the stated conditions and break it. (a) Two units with identical `Z_j` and `mu_j`, `f = (1/2, 1/2)`, unit 1 treated and unit 2 its control (a perfect fit); `eps_1t = eps_2t` in every period, a common standard normal draw, i.i.d. over time; `xi_1t` an independent standard normal draw. Every blank-period gap is exactly zero and the experimental-period gap `xi_1t - eps_2t` is non-zero with probability one, so the p-value (17) equals `1/|Pi|` in every sample and the test rejects a true null whenever `alpha >= 1/|Pi|`. (b) Two blank periods in which the two units' noises are `(A, B)` and then `(B, A)`, with `A` and `B` independent standard normal, and one experimental period with independent draws: each unit's sequence is i.i.d., the two absolute blank gaps coincide, and the test at level 1/3 rejects with probability 1/2. The lower bound `alpha - 1/|Pi|` additionally relies on the permuted statistics being distinct with probability one, which the proof asserts from continuity of the noise and which fails in (a) and (b).
   - *Interval (Theorem 3).* The interval's half-width is a quantile of the blank-period gaps, whose noise part has weights `w* - v*`. What it must cover is `tau_hat_t - tau_t`, whose noise part is `sum_j (w*_j - f_j) xi_jt - sum_j (v*_j - f_j) eps_jt`, because `tau_t` is the realized population effect. Reviewer derivation: with noise independent across units and variance `sigma^2`, the first has variance `sigma^2 (sum_j w*_j^2 + sum_j v*_j^2)` and the second `sigma^2 (sum_j (w*_j - f_j)^2 + sum_j (v*_j - f_j)^2)`; for `f_j = 1/J` the second is smaller by `2 sigma^2 / J`. Illustration: two units with identical `Z_j` and `mu_j` and `f = (1/2, 1/2)`, one treated and one control; a common normal shock for both units in the fitting periods (so the fit is exact, as Assumption 3 requires) and, in the blank and experimental periods, normal shocks with the same variance that are independent across units. The blank gaps then have variance `2 sigma^2` and the estimation error `sigma^2`, so as the number of blank periods grows a nominal 95% interval covers about 99.4% of the time (the probability that a standard normal is within `1.96 * sqrt(2)`). In example (a) above the interval has zero width while `tau_hat_t - tau_t = (xi_1t - xi_2t)/2`, so with independent `xi_1t` and `xi_2t` its coverage is zero. The coverage statement of Theorem 3 should therefore be carried as the paper's claim, with the calibration of the population-effect error by the blank gaps still to be justified for a given setting.

7. **Rejection rates in Table 2** come from one simulated non-null data-generating process (effect paths vary across replications; the first panel reports their means) and are not sizes; with `|Pi| = 252` the test at 0.05 is feasible, but no size simulation is reported in the reviewed PDF (the appendix has one, Section OA.9.2.3, not reviewed).
8. **Table 1, `m_hi = 4`.** SC's normalized RMSE is not monotone in `m_hi` (0.019 at 3, 0.027 at 4, 0.012 at 5); the paper does not comment.
9. **Printed slips in the source.** Equation (11) and the unit-level near-fit condition on p.13 have transposed or reused indices (see the unit-level design above); the STR formula on p.28 has an upper summation limit of `k` where a sum over units is meant; and the p.27 statement that the Unconstrained design is best "across all measures" does not hold for the rejection-rate column of Table 2.
