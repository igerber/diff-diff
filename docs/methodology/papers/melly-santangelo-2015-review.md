# Paper Review: The Changes-in-Changes Model with Covariates

**Authors:** Blaise Melly and Giulia Santangelo
**Citation:** Melly, B., & Santangelo, G. (2015). The Changes-in-Changes Model with Covariates. *Working paper*, Bern University and European Commission - JRC.
**PDF reviewed:** **October 2015 working-paper version** (title page: "First version: January 2015. This version: October 2015. Preliminary!"; 32 PDF pages, printed page = PDF page; main text pp. 1-25, Figures pp. 26-30, References pp. 31-32). Distributed via Blaise Melly's website (https://sites.google.com/site/blaisemelly/home/computer-programs/cic_stata). Per the project's PDFs-never-committed convention the local PDF is kept outside the repository (gitignored `papers/melly-santangelo-cic-covariates.pdf`). This paper remains an unpublished working paper as of the review date (2026-07-12); it is nonetheless the standard reference for CiC with covariates. All equation/assumption/theorem numbers below are pinned to the October 2015 draft - later drafts, if any, may renumber.
**Review date:** 2026-07-12

---

## Methodology Registry Entry

*Drafted in docs/methodology/REGISTRY.md format for FUTURE use. Do not copy this section into the registry now - see the status line below.*

## CiC with Covariates (Melly-Santangelo)

**Status: the qte-style SIMPLIFIED form of this pipeline SHIPPED in the covariates PR (2026-07-13)** - per-cell linear quantile regressions on qte's fixed 99-tau grid, raw (un-rearranged) `predict.rqs` conditional CDF/quantile step functions, per-observation imputation integrated over the treated PRE-period covariates, and qte's bootstrap. The FULL estimator this paper develops (monotonized integrated-indicator CDFs, `F_{X|11}` treated-post integration, exchangeable bootstrap with variance-weighted KS bands, tail trimming, specification test) remains unimplemented; this review is its scoped reference.

**Primary source:** Melly, B., & Santangelo, G. (2015). The Changes-in-Changes Model with Covariates. *Working paper*, Bern University and European Commission - JRC. Distributed via https://sites.google.com/site/blaisemelly/home/computer-programs/cic_stata.
- Layout: Sections 1 (Introduction, pp. 2-3), 2 (Model and identification, pp. 4-9), 3 (Estimation, pp. 10-12), 4 (Asymptotic results, pp. 12-21; 4.4 "Inference" pp. 19-21 includes the time-invariance specification test), 5 (Application, pp. 21-24), 6 (Conclusion, pp. 24-25); Figures 1-5 on pp. 26-30; References pp. 31-32.
- **Numbering note (as extracted):** results are numbered sequentially across result types - Proposition 1 (p. 6), Lemmas 2-3 (p. 14), Theorem 4 (p. 15), Corollaries 5-6 (p. 16), Theorem 7 (pp. 16-17), Corollaries 8-9 (p. 18), Theorem 10 (p. 20). There is no "Lemma 1" or "Theorem 1". Equations share one sequence (1)-(15). The draft is flagged "Preliminary!" on its title page and contains printed typos - see the suspected-typos block in Gaps and Uncertainties.

**Motivation (Introduction, pp. 2-3; DiD-bias illustration pp. 22-23, Figures 1-2):**

Athey-Imbens (2006) CiC does not depend on the scale of the dependent variable and recovers the whole counterfactual distribution; its main identifying restriction is time invariance of the distribution of unobservables within each group (p. 2). For covariates, AI suggest either (a) a fully nonparametric strategy, which "naturally suffers from the curse of dimensionality and is not practicable in many applications," or (b) a parametric strategy based on additive separability (OLS regression, then unconditional CiC on the residuals), which "imposes restrictions that limit the appeal of the method to analyze heterogeneous effects" - the paper argues it would be contradictory to use CiC (motivated by non-additive effects) while imposing additive separability for covariates (p. 2). Per Lechner's (2011) survey, the lack of a tractable estimator with covariates is one of the main reasons for the low number of empirical CiC applications (p. 2).

Why covariates matter (p. 2): time-constant covariates help when trends differ by covariates; time-varying covariates capture time-varying differences between groups and allow linear trends as regressors, relaxing the time invariance assumption; even when time invariance holds unconditionally, covariates can bring efficiency gains. Conditioning on X *weakens* the unconditional AI time-invariance requirement (Assumption 3 below is imposed conditionally on X): unconditional CiC assumptions can fail when the conditional ones hold - e.g., when group covariate compositions differ and trends vary by covariates.

The paper also quantifies how badly the common empirical practice of DiD-on-indicators fails for distributional targets (pp. 22-23; Figures 1-2, pp. 26-27):
- Setup: `Y_i ~ N(0.5*T + 0.5*G, 1)` with a uniformly zero treatment effect (no TxG interaction).
- DiD applied to `1(Y <= y)` is inconsistent except in three knife-edge cases (p. 22): "(i) There is no time effect. (ii) There is no group effect. (iii) The dependent variable is uniformly distributed." Root cause: "the time effect and the group effect cannot be additive if Y is not uniformly distributed."
- Analytic probability limit of the DiD estimator for `1(Y <= -0.5)` (p. 23): `Phi(-1.5) - Phi(-1) - Phi(-1) + Phi(-0.5) ~= 0.058`, "large relative to the observe[d] probability of 1(Y <= -0.5) which is 0.067."
- Figure 1 (p. 26): asymptotic DiD bias for the distribution treatment effect across all cutoffs; positive at the lower tail, negative at the upper tail (roughly +/-0.057 over cutoffs -3 to 4).
- Figure 2 (p. 27): asymptotic bias of the QTE implied by DiD on the distribution function; strongly negative at low quantiles (approaching about -1.5 as tau -> 0), rising through zero mid-distribution, positive (~+0.3) near tau = 1 - "very misleading given that the true effects are zero uniformly over (0,1). Therefore, the results reported by Almond, Hoynes, and Schanzenbach (2011) for the effects on the distribution should be taken with caution" (p. 23).
- Footnote 10 (p. 22): additional issues from linear regression of a binary dependent variable on a non-saturated covariate set, especially for conditional probabilities.
- No Monte Carlo simulations appear anywhere in the paper - this analytic illustration is the only numerical-bias evidence.

**Model (Section 2.1, pp. 4-6):**

Two periods and two groups for Sections 2-4 (the paper states it uses "as much as possible the same notation as AI"; the multi-group/multi-period setup appears only in the application, Section 5). Observed data: `(Y, G, T, X)`.

| Symbol | Meaning |
|---|---|
| `G in {0,1}` | Group indicator; group 1 is the treatment group |
| `T in {0,1}` | Time period; only group 1 in period 1 is treated |
| `X` | Covariate vector with support `X` (script) |
| `Y^N`, `Y^I` | Potential outcomes without / with treatment |
| `I = G*T` | Treatment indicator; realized outcome `Y = Y^N*(1-I) + Y^I*I` (p. 4, unnumbered) |
| `U` | Unobservable component of Y (Assumption 1) |
| `Y^N_gtx ~d Y^N | G=g, T=t, X=x` (etc.) | Distributional shorthands (p. 5); conditional CDFs `F_{Y^N|gtx}`, `F_{Y^I|gtx}`, `F_{Y|gtx}`, `F_{U|gx}` with supports `Y^N_gtx`, `Y^I_gtx`, `Y_gtx`, `U_gx` |
| `F_{X|11}` | Covariate distribution in group 1, period 1 (integration measure for the unconditional estimands) |
| `alpha_gt = Pr(T_i=t, G_i=g) > 0` | Cell probabilities (Assumption 5(ii)) |
| `n_11` | Number of group-1 period-1 observations |
| `u_1 < ... < u_S`, mesh width `delta` | Quantile grid for the QR process, `delta*sqrt(n) -> 0` (p. 11) |

The distribution of `Y^I_11x` is identified directly by the observed `Y_11x`; the identification problem is `Y^N_11x` (p. 5). Footnote 2 (p. 5): the paper focuses on group 1 / period 1 because the treated outcome is only observed there; effects for group 0 or period 0 would require much stronger assumptions.

Discrete outcomes (p. 6): the model in principle allows them, but strict monotonicity is very restrictive there (the discreteness must come from the unobserved term; ordered logit or rounded outcomes are excluded), so the authors "do not advise using this model for discrete outcomes." Footnote 4 (p. 6): AI's weakly-increasing-`h` case gives only partial identification; deferred to future work. The Conclusion (p. 25) adds that QR-based estimators "are not well suited for discrete outcomes" and flags distribution regression as the natural route for a discrete/mixed-outcome extension.

**Assumptions (exact statements; conditional restatement of Athey-Imbens Assumptions 3.1-3.4 plus estimation conditions):**

*Assumption 1 (potential outcome) - p. 4:*

> The outcome of an individual in the absence of intervention satisfies the relationship `Y^N = h(X, T, U)`.

Without the other assumptions this does not restrict the DGP - any `Y^N` can be represented as `h(X, T, U)` for unrestricted `h` and `U` (p. 4). Note the control outcome is a *nonseparable* function of covariates, time, and the unobservable.

*Assumption 2 (strict monotonicity) - p. 5:*

> The production function `h(t, x, u)` is strictly increasing in `u` for `t in {0, 1}` and for all `x in X`.

*Assumption 3 (time invariance) - p. 5:*

> We have `U _||_ T | G, X`.

The key restriction: it allows extrapolation of ranks from one period to the other and is "the CIC counterpart of the common trend assumption in the DID model" (p. 5). Only equality in distribution over time is required (rank similarity; footnote 3 cites Chernozhukov and Hansen 2005), not the stronger rank invariance/preservation where `U` is constant per individual. This is AI time invariance imposed *conditionally on X* (as well as G).

*Assumption 4 (support) - p. 5:*

> `U_1x subset U_0x` for all `x in X`.

Guarantees group-0 observations with similar `U` as group-1 observations exist. Testable implication: `Y_10x subset Y_00x` for all `x in X`; if violated, only the part of the counterfactual distribution on the overlap of `U_0x` with `U_1x` is identified (p. 6). Assumptions 1-3 are not testable with only 2 periods and 2 groups; testing time invariance with more periods is in Section 4.4 (p. 6).

Consequence of Assumptions 1-2 (p. 5, unnumbered): `F_{Y^N}^{-1}(tau | X=x, T=t) = h(x, t, F_U^{-1}(tau | X=x, T=t))`. `h` is not identified without a normalization (e.g., `U | X=x, T=t ~ U(0,1)`, under which `h` is the quantile function of `Y^N`); the normalization is not needed because `h` is not the object of interest (p. 5; cites Matzkin 2003).

*Assumption 5 (Data generating process) - p. 12:*

> (i) Conditional on `T_i = t` and `G_i = g`, `(Y_i, X_i)` is a random draw from the subpopulation with `G_i = g` during period `t` that has probability law `P_gt`. (ii) For all `t, g in {0,1}`, `alpha_gt := Pr(T_i = t, G_i = g) > 0`.

Repeated cross-sectional sampling (matches the application). For panel data the results must be modified: as in Section 5.3 of AI, additional terms would account for correlation of estimated conditional distributions over time within groups (p. 12).

*Assumption 6 (Quantile regression regularity conditions) - p. 13:*

> (i) The conditional quantile function takes the form `F_{Y|G,T,X}^{-1}(u | g, t, x) = x' beta_gt(u)` for all `u in (0,1)` and `x in X`. (ii) The conditional density function `f_{Y|G,T,X}(y|x)` exists, is uniformly bounded, and is uniformly continuous in `(y, x)` in the support `Y_gt x X`, which is a compact subset of `R^{K+1}`. (iii) The minimal eigenvalue of `J_gt(u) := E[f_{Y|gtX}(X'beta(u)) X X']` is bounded away from zero uniformly over `u`. (iv) `E||X||^{2+eps} < infinity` for some `eps > 0`.

Described as standard QR regularity conditions. Part (i) is the semiparametric restriction: linear-in-parameters conditional quantile functions in each of the four `(g,t)` cells. The paper notes QR can approximate the true conditional quantile function arbitrarily well with rich enough transformations of the regressors when Y has a smooth conditional density, and that distribution regression "could be used as well" as the first stage (p. 10).

**Target parameters (pp. 8-10):**

Conditional QTE process for group 1 in period 1 at covariate value `x` (p. 8, unnumbered; uses `F_{Y|11x}^{-1}(tau) = F_{Y^I|11x}^{-1}(tau)` since group 1 is treated in period 1):

```
Delta^{QE}(.|x) = F_{Y^I|11x}^{-1}(.) - F_{Y^N|11x}^{-1}(.)
                = F_{Y|11x}^{-1}(.) - F_{Y|01x}^{-1}( F_{Y|00x}( F_{Y|10x}^{-1}(.) ) )
```

Conditional average and distribution effects (p. 9, unnumbered):

```
Delta^{AE}(x)  = E[Y^I_11x] - E[Y^N_11x]
Delta^{DE}(.|x) = F_{Y^I|11x}(.) - F_{Y^N|11x}(.)
```

These are identified for all `x` in the support (heterogeneity analysis in observables), but high-dimensional functions are hard to communicate, so the primary interest is unconditional effects (p. 9). Unconditional distributions (p. 9, unnumbered), both integrating over `F_{X|11}` - the covariate distribution of the *treated group in period 1* - so the estimands are effects on the treated:

```
F_{Y^I|11}(y) = F_{Y|11}(y) = INT_X F_{Y|11x}(y) dF_{X|11}(x)

F_{Y^N|11}(y) = INT_X F_{Y^N|11x}(y) dF_{X|11}(x)
              = INT_X F_{Y|10x}( F_{Y|00x}^{-1}( F_{Y|01x}(y) ) ) dF_{X|11}(x)
```

"All the elements are observable in the last expression." Headline estimand, the unconditional quantile effect process (p. 9, unnumbered):

```
Delta^{QE}(.) = F_{Y^I|11}^{-1}(.) - F_{Y^N|11}^{-1}(.)
```

All functionals of these marginal distributions are identified: distribution functions, quantile functions, quantile effects, distribution effects, average effects, Lorenz curves, Gini coefficients (p. 3).

Distribution of individual treatment effects (p. 9, eq. (6)): with panel data and strengthening time invariance (rank similarity) to rank invariance:

```
(6)   F_{Delta|11}(delta) := F_{Y^I - Y^N|11}(delta)
        = INT_X F_{Y^I - Y^N|11x}(delta) dF_{X|11}(x)
        = INT_X INT_0^1 1( Delta^{QE}(.|x) <= delta ) dF_{X|11}(x)
```

(inner integral over the quantile index; identifies e.g. the proportion who benefit). Not pursued further - the application has only repeated cross-sections; the limiting distribution of the plug-in estimator based on (6) can be derived by combining the paper's results with Appendix C of Chernozhukov, Fernandez-Val, and Melly (2009) (p. 10).

**Identification (Proposition 1, p. 6; proof pp. 7-8, eqs. (1)-(5)):**

> **Proposition 1 (identification of the conditional distribution).** Suppose that Assumptions 1-4 hold and let `0 < tau < 1`. Then the `tau` quantile of `Y^N_11x` is identified for all `x in X` with

```
F_{Y^N|11x}^{-1}(tau) = F_{Y|01x}^{-1}( F_{Y|00x}( F_{Y|10x}^{-1}(tau) ) ).
```

This "trivially extends Theorem 3.1 in AI to the case with covariates" (p. 6) - the Athey-Imbens CiC quantile map applied *within each covariate cell x*. Footnote 5 credits Fortin and Lemieux (1998) and Altonji and Blank (1999) with similar expressions in a different context.

Proof chain (pp. 7-8). Central equation, valid for all four `(g,t)` combinations, using invertibility of `h(t,x,u)` in `u` with inverse `h^{-1}(t,x,y)`:

```
(1)   F_{Y^N|gtx}(y) = Pr(h(t,x,U) <= y | G=g, T=t, X=x)
                     = Pr(U <= h^{-1}(t,x,y) | G=g, T=t, X=x)
                     = Pr(U <= h^{-1}(t,x,y) | G=g, X=x)      [time invariance]
                     = F_{U|gx}(h^{-1}(t,x,y))
```

Applying (1) with `(g,t) = (0,0)`, substituting `y = h(0,x,u)`, and applying `F_{Y|00x}^{-1}`, for all `u in U_0x`:

```
(2)   h(0, x, u) = F_{Y|00x}^{-1}( F_{U|0x}(u) )
```

Applying (1) with `(g,t) = (0,1)`, using `h^{-1}(1,x,y) in U_0x` for all `y in Y_01x`, and applying `F_{U|0x}^{-1}` to both sides of `F_{Y|01x}(y) = F_{U|0x}(h^{-1}(1,x,y))`:

```
(3)   F_{U|0x}^{-1}( F_{Y|01x}(y) ) = h^{-1}(1, x, y)
```

Combining (2) and (3), for all `y in Y_01x`:

```
(4)   h(0, x, h^{-1}(1, x, y)) = F_{Y|00x}^{-1}( F_{Y|01x}(y) )
```

Interpretation (p. 8): `h(0, x, h^{-1}(1, x, y))` is the period-0 outcome for an individual with characteristics `x` and the `u`-realization corresponding to outcome `y` in group 0, period 1; (4) shows it is determined by observable distributions. Applying (1) with `(g,t) = (1,0)` and `y = h(0,x,u)`:

```
(5)   F_{Y|10x}(y) = F_{U|1x}( h^{-1}(0, x, h(0, x, u)) ) = F_{U|1x}(u)
```

Combining (4) and (5) and substituting into (1) with `(g,t) = (1,1)`, for all `y in Y_01` (p. 8, unnumbered):

```
F_{Y^N|11x}(y) = F_{U|1x}( h^{-1}(1,x,y) ) = F_{Y|10x}( h(0,x,h^{-1}(1,x,y)) )
               = F_{Y|10x}( F_{Y|00x}^{-1}( F_{Y|01x}(y) ) )
```

By Assumption 4, `Y^N_11 subset Y_01`, so the directly observable `F_{Y|10x}`, `F_{Y|00x}`, `F_{Y|01x}` identify `F_{Y^N|11x}`; inverting gives Proposition 1. Intuition (p. 8): an individual at a given quantile of one group's period-0 conditional outcome distribution maps to a relative rank in the other group's same-period distribution; by time invariance this relative rank is unchanged over time. (The printed intuition sentence swaps group labels relative to the formula - see suspected typos; transcribed here consistently with the displays.)

Identification of the *unconditional* counterfactual follows constructively by integrating `F_{Y^N|11x}` over `F_{X|11}` (p. 9 displays); there is no separately numbered theorem for the unconditional identification step - it is the display sequence on p. 9.

**Estimator (Section 3, pp. 10-12) - the four-step QR pipeline:**

Plug-in principle: identification is constructive; replace all observables by consistent estimators. At least three conditional distribution/quantile functions must be estimated.

*Step 1 - four QR coefficient processes (p. 10).* Linear quantile regression (Koenker and Bassett 1978) separately in the four `(g,t)` samples:

```
beta_hat_gt(u) = argmin_{b in R^{K+1}} SUM_{i: G_i=g, T_i=t} (u - 1(Y_i <= X_i'b)) * (Y_i - X_i'b)
```

giving `F_hat_{Y|gtx}^{-1}(u) = x' beta_hat_gt(u)`. Practical discretization: the QR process is estimated on a fine mesh `u_1 < ... < u_S` with mesh width `delta` such that `delta*sqrt(n) -> 0`. Footnote 7 (p. 11): estimating *all* distinct quantile regressions is possible - the estimates change at only finitely many points, `O(n log n)` distinct solutions (Portnoy 1991). Computation is reduced ~10x with the algorithms of Melly (2014). Tail trimming "seems unavoidable in practice" - the paper abstracts from it for notational simplicity; formulas can be adapted as in Chernozhukov, Fernandez-Val, and Melly (2013). (The draft's tail-trimming sentence is printed incomplete - see suspected typos.)

*Step 2 - monotonization via the integrated-indicator CDF representation (pp. 10-11).* Estimated conditional quantile functions may be non-monotonic in `u` and cannot be directly inverted. Instead use the sample analog of `F_{Y|gtx}(y) = INT_0^1 1( F_{Y|gtx}^{-1}(u) <= y ) du`:

```
F_hat_{Y|gtx}(y) = INT_0^1 1( F_hat_{Y|gtx}^{-1}(u) <= y ) du
                 = delta * SUM_{s=1}^S 1( x' beta_hat_gt(u_s) <= y )      [p. 11 discretization]
```

Statistical properties of this monotonized (rearrangement-type) CDF estimator are studied in Chernozhukov, Fernandez-Val, and Galichon (2010). This is the key extra numerical ingredient relative to covariate-free CiC.

*Step 3 - conditional CiC transformation at each x (pp. 11-12).* The inner probability-probability step `F_hat_{Y|00x}(x'beta_hat_10(tau))` is computed by the monotonized-CDF integral, and its value is the quantile index at which the group-0 period-1 QR process is evaluated:

```
F_hat_{Y^N|11x}^{-1}(tau) = x' beta_hat_01( INT_0^1 1( x' beta_hat_00(u) <= x' beta_hat_10(tau) ) du )
F_hat_{Y^I|11x}^{-1}(tau) = x' beta_hat_11(tau)
```

Conditional QTE estimator (p. 12, eq. (7)):

```
(7)   Delta_hat^{QE}(.|x) = x' ( beta_hat_11(tau) - beta_hat_01( INT_0^1 1( x'beta_hat_00(u) <= x'beta_hat_10(tau) ) du ) )
```

*Step 4 - integration over the treated-group covariate distribution (p. 12).* Integrate the conditional counterfactual CDF over the *empirical* distribution of X among group-1 period-1 observations:

```
F_hat_{Y^N|11}(y) = INT_X F_hat_{Y^N|11x}(y) dF_{X|11}(x)
   = (1/n_11) SUM_{i: G_i=1, T_i=1} SUM_{j=1}^S 1( x_i' beta_hat_01( INT_0^1 1( x_i'beta_hat_00(u_s) <= x_i'beta_hat_10(tau) ) du ) <= y )
```

(transcribed as printed - the draft's indexing is inconsistent here and the mesh-width factor is missing; see suspected typos. The intended estimator, consistent with the p. 11 discretization, evaluates the outer indicator on the mesh `{u_j}` - i.e., a counterfactual tau-grid - carries the `delta` weight, and averages, mirroring the treated-side formula.) `F_{Y^I|11}(y)` can be estimated by the empirical distribution of `Y_i` in cell (1,1), or - the option used in the application, keeping treated and counterfactual sides methodologically symmetric - by integrating the QR-estimated conditional distribution:

```
F_hat_{Y^I|11}(y) = (1/n_11) SUM_{i: G_i=1, T_i=1} SUM_{j=1}^S 1( x_i' beta_hat_11(u_s) <= y )
```

(same printed index inconsistency and implicit mesh weight). Unconditional QTE estimator (p. 12, eq. (8)), taking (generalized) inverses of the integrated CDF estimators:

```
(8)   Delta_hat^{QE}(.) = F_hat_{Y^I|11}^{-1}(y) - F_hat_{Y^N|11}^{-1}(y)
```

(printed with argument `(y)` though the process is indexed by the quantile level - see suspected typos).

**Asymptotic theory (Section 4, pp. 12-18):**

All limit results are at the parametric `sqrt(n)` rate (a consequence of the linear QR restriction, Assumption 6(i)), as processes, enabling uniform (functional) inference. Section 4.2's "four main ingredients" (pp. 13-15) combine via the functional delta method:

*Ingredient 1 - QR process FCLTs (p. 13).* Under Assumptions 5 and 6, Corollary 5.2 of Chernozhukov, Fernandez-Val, and Melly (2013) implies, for all `t, g in {0,1}`, as `n -> infinity` in `l^inf(U_g X)`:

```
(9)   sqrt(n) ( F_hat_{Y|gtx}^{-1}(u) - F_{Y|gtx}^{-1}(u) ) ⇝ Z^Q_gt(u, x)
```

as stochastic processes indexed by `(u,x)`, with independent tight zero-mean Gaussian limits and covariance function

```
V^Q_gt(u, x, u~, x~) = alpha_gt^{-1} * x' J_gt(u)^{-1} (min(u,u~) - u*u~) E[XX' | G=g, T=t] J_gt(u~)^{-1} x~
```

(the evaluation point of the second Jacobian is hard to resolve at print resolution; transcribed as `J_gt(u~)^{-1}`, the standard QR-process covariance form). The same corollary gives the conditional distribution process, as `n -> infinity` in `l^inf(Y_gt X)`:

```
(10)  sqrt(n*alpha_gt) ( F_hat_{Y|gtx}(y) - F_{Y|gtx}(y) ) ⇝ Z^F_gt(u, x) := -f_{Y|gtx}(y) Z_gt( F_{Y|gtx}(y), x )
```

as processes indexed by `(y,x)` (the index is printed `(u,x)` and the scaling differs from (9) by `sqrt(alpha_gt)`, as printed - see suspected typos), with covariance

```
V^F_gt(y, x, y~, x~) = f_{Y|gtx}(y) f_{Y|gtx~}(y~) V^Q_gt( F_{Y|gtx}(y), x, F_{Y|gtx~}(y~), x~ )
```

*Ingredient 2 - Lemma 2 (quantile-quantile transformation, p. 14).* For CDFs F, G with compact support, continuously differentiable with strictly positive densities f, g, the map `phi^{QQ}(F, G) = F o G^{-1}` is Hadamard-differentiable at (F, G) tangentially to functions `h_1, h_2`, with derivative map

```
phi'_{F,G}(h_1, h_2) = h_1 o G^{-1} - ( f o G^{-1} / g o G^{-1} ) * h_2 o G^{-1}
```

(proof via Hadamard differentiability of the inverse map, Lemma 3.9.23(ii), and the chain rule, Lemma 3.9.27, in Van der Vaart and Wellner 1996; footnote 8 also cites problem 4, p. 398 of VdV-W).

*Ingredient 3 - Lemma 3 (probability-probability transformation, p. 14).* For F with compact support, continuously differentiable, strictly positive density f, the map `phi^{PP}(F, G) = F^{-1} o G` is Hadamard-differentiable with derivative map

```
phi'_{F,G}(h_1, h_2) = -( h_1 / f ) o F^{-1} o G - ( h_2 / (f o F^{-1}) ) o G
```

(as printed - the sign of the `h_2` term is suspect; see the suspected-typos block in Gaps and Uncertainties)

*Ingredient 4 - counterfactual operator (p. 15).* Lemma D.1 of Chernozhukov, Fernandez-Val, and Melly (2013): `phi^C(F, G) = INT F(y, x) dG(x)` is Hadamard-differentiable with derivative map

```
phi^C_{F_{Y|X}, F_X}(gamma, pi) = INT gamma(y, x) dF_X(x) + pi( F_{Y|X}(y|x) )
```

This is the operator integrating the conditional distribution over the covariate distribution.

> **Theorem 4 (limiting distribution of the conditional CiC estimator) - p. 15.** Suppose that Assumptions 1 to 6 hold. Then, (i) `sqrt(n)(F_hat_{Y^I|11x}^{-1}(tau) - F_{Y^I|11x}^{-1}(tau)) ⇝ Z^Q_11(tau, x)` and `sqrt(n)(F_hat_{Y^N|11x}^{-1}(tau) - F_{Y^N|11x}^{-1}(tau)) ⇝ Z^Q_N(tau, x)` as stochastic processes indexed by `(tau, x) in (0,1) x X` and where `Z^Q_11(tau, x)` and `Z^Q_N(tau, x)` are independent tight zero-mean Gaussian process defined in (9) and (11).

Proof intermediates (pp. 15-16): the treated side is (9) directly. For the counterfactual side, by Lemma 2 (QQ step) and the functional delta method:

```
sqrt(n)( F_hat_{Y|00x}(F_hat_{Y|10x}^{-1}(tau)) - F_{Y|00x}(F_{Y|10x}^{-1}(tau)) ) ⇝ Z^R(tau, x)
  := Z^F_00( F_{Y|10x}^{-1}(tau), x ) + [ f_{Y|00x}(F_{Y|10x}^{-1}(tau)) / f_{Y|10x}(F_{Y|10x}^{-1}(tau)) ] Z^Q_10(tau, x)
```

then by Lemma 3 (PP step) and the delta method (p. 16):

```
(11)  Z^Q_N(tau, x) := Z^Q_01( F_{Y|00x}(F_{Y|10x}^{-1}(tau)), x )
                     + Z^R(tau, x) / f_{Y|01x}( F_{Y|01x}^{-1}( F_{Y|00x}(F_{Y|10x}^{-1}(tau)) ) )
```

(the density in the denominator is the density of `Y|0,1,x` evaluated at the counterfactual quantile).

> **Corollary 5 (conditional QTE process) - p. 16.** Under Assumptions 1-6, `sqrt(n)(Delta_hat^{QE}(tau|x) - Delta^{QE}(tau|x)) ⇝ Z^{QTE}(tau, x) = Z^Q_11(tau,x) - Z^Q_N(tau,x)`, a tight zero-mean Gaussian process indexed by `(tau, x) in (0,1) x X`. Proof: "Trivial by the functional delta method."

> **Corollary 6 (Hadamard-differentiable functionals, conditional) - p. 16.** For `phi(F_{Y^I|11x}^{-1}, F_{Y^N|11x}^{-1}, w)` Hadamard differentiable with derivatives `phi'_I` and `phi'_N`: `sqrt(n)(phi(F_hat^{-1}_{Y^I|11x}, F_hat^{-1}_{Y^N|11x}, w) - phi(F^{-1}_{Y^I|11x}, F^{-1}_{Y^N|11x}, w)) ⇝ phi'_I(Z^Q_11(., x), w) + phi'_N(Z^Q_N(., x), w)` as a stochastic process indexed by `w`.

> **Theorem 7 (limiting distribution of the unconditional quantile processes) - pp. 16-17.** Under Assumptions 1-6, `sqrt(n)(F_hat_{Y^I|11}^{-1}(tau) - F_{Y^I|11}^{-1}(tau)) ⇝ Z^Q_11(tau)` and `sqrt(n)(F_hat_{Y^N|11}^{-1}(tau) - F_{Y^N|11}^{-1}(tau)) ⇝ Z^Q_N(tau)` **jointly** as stochastic processes indexed by `tau in (0,1)`, with tight zero-mean Gaussian limits defined in (13) and (12).

Proof intermediates (p. 17): applying Corollary 6 with `phi(.) = F_{Y^N|11x}(y)` gives the conditional distribution process `sqrt(n)(F_hat_{Y^N|11x}(y) - F_{Y^N|11x}(y)) ⇝ Z^F_N(y,x) = f_{Y^N|11x}(y) Z^Q_N(F_{Y^N|11x}(y), x)`. By the Donsker theorem, the empirical covariate distribution in cell (1,1) satisfies

```
(1/sqrt(n)) SUM_{i: G_i=T_i=1} ( f(Y_i, X_i) - INT f(Y_i, X_i) dP_11 ) ⇝ Z^X_11( f(y,x) )
```

indexed by `f in F`, a universal Donsker class; the limits are tight `P_11`-Brownian bridges with covariance `V^X_11(y,x,y~,x~) = alpha_11^{-1} * ( INT f(y,x) f(y~,x~) dP_11 - INT f(y,x) dP_11 INT f(y,x) dP_11 )`. By the functional delta method and the counterfactual-operator differentiability:

```
sqrt(n)( F_hat_{Y^N|11}(y) - F_{Y^N|11}(y) ) ⇝ INT_X Z^F_N(y,x) dF_{X|11}(x) + Z^X_11( F_{Y^N|11x}(y) ) := Z^F_N(y)
```

and finally (inverse map / delta method):

```
(12)  sqrt(n)( F_hat_{Y^N|11}^{-1}(tau) - F_{Y^N|11}^{-1}(tau) ) ⇝ -Z^F_N( F_{Y^N|11}^{-1}(tau) ) / f_{Y^N|11}( F_{Y^N|11}^{-1}(tau) ) := Z^Q_N(tau)
```

For the treated outcome, directly from Theorem 4.1(ii) of Chernozhukov, Fernandez-Val, and Melly (2013):

```
sqrt(n)( F_hat_{Y^I|11}(y) - F_{Y^I|11}(y) ) ⇝ INT_X f_{Y|11x}(y) Z^Q_11( F_{Y|11x}(y), x ) dF_{X|11}(x) + Z^X_11( F_{Y|11x}(y) ) := Z^F_11(y)

(13)  sqrt(n)( F_hat_{Y^I|11}^{-1}(tau) - F_{Y^I|11}^{-1}(tau) ) ⇝ -Z^F_11( F_{Y|11}^{-1}(tau) ) / f_{Y|11}( F_{Y|11}^{-1}(tau) ) := Z^Q_11(tau)
```

> **Corollary 8 (unconditional QTE process) - p. 18.** `sqrt(n)(Delta_hat^{QE}(tau) - Delta^{QE}(tau)) ⇝ Z^{QTE}(tau) = Z^Q_11(tau) - Z^Q_N(tau)`, tight zero-mean Gaussian, indexed by `tau in (0,1)`.

> **Corollary 9 (Hadamard-differentiable functionals, unconditional) - p. 18.** Same structure as Corollary 6 for functionals of the unconditional processes.

Remarks on analytical SEs (p. 18): the functional results imply pointwise estimators (e.g., QTE at a single quantile) are asymptotically normal, and "these formulas can be used to develop analytical estimators of the standard errors. However, all asymptotic variances contain terms that are difficult to estimate such as the conditional density of the dependent variable given the covariates. Therefore, we suggest using resampling methods to estimate the standard errors of the estimates. This method also allows performing inference on the whole processes." The corollaries cover all Hadamard-differentiable functionals: ATE as a simple special case; Lorenz curve and Gini coefficient as a more involved one. Without covariates (X = constant, pointwise), the results simplify to AI; even there the paper contributes the limiting distribution of the whole quantile and distribution processes, with a simpler empirical-process-theory proof.

**Bootstrap inference (Section 4.4 "Inference", pp. 19-21):**

The paper proves validity of the **exchangeable bootstrap**, which "incorporates many popular forms of resampling as special cases, namely the empirical bootstrap, weighted bootstrap, m out of n bootstrap, and subsampling" (p. 19). Motivation: "in small samples with categorical covariates, we might want to use the weighted bootstrap to gain accuracy and robustness to 'small cells,' whereas in large samples, where computational tractability can be an important consideration, we might prefer subsampling" (p. 19).

*Condition BW (bootstrap weights) - p. 19.* For each `t, g in {0,1}`, let `(w_gt1, ..., w_gtn_gt)` be an exchangeable (footnote 9: permutation-invariant), nonnegative random vector, independent of the data, such that for some `eps > 0`:

```
sup_{n_k} E[ w_gt1^{2+eps} ] < infinity
n_gt^{-1} SUM_{i=1}^{n_gt} ( w_gti - w_bar_gt )^2  ->_p  1
w_bar_gt = n_gt^{-1} SUM_{i=1}^{n_gt} w_gti  ->_p  1
```

with the weight vectors independent across `(g,t)`. Multinomial weights with probabilities `(1/n_k, ..., 1/n_k)` give the empirical bootstrap.

*Bootstrap versions of the estimators (pp. 19-20).* Weighted QR:

```
beta_hat*_gt(u) = argmin_{b in R^{K+1}} SUM_{i: G_i=g, T_i=t} w_gti * ( u - 1(Y_i <= X_i'b) ) * ( Y_i - X_i'b )
```

bootstrap conditional quantile function `F_hat*^{-1}_{Y|gtx}(u) = x' beta_hat*_gt(u)`, and bootstrap unconditional counterfactual distribution (p. 20):

```
F_hat*_{Y^N|11}(y) = (1/n_11) SUM_{i: G_i=1, T_i=1} SUM_{j=1}^S w_11i * 1( x_i' beta_hat*_01( INT_0^1 1( x_i' beta_hat*_00(u_s) <= x_i' beta_hat*_10(tau) ) ) <= y )
```

(the p. 20 display is typographically dense; the structure is the weighted empirical analog of the QQ/PP composition over the S-point u-grid, transcribed as printed modulo layout - see suspected typos).

*Uniform confidence bands (p. 20).* An asymptotic simultaneous (1-alpha) band for the unconditional QTE process is defined by end-point functions

```
(14)  Delta_hat^{QE}(tau)+/- = Delta_hat^{QE}(tau) +/- t_hat_{1-alpha} * Sigma_hat^{QE}(tau)^{1/2} / sqrt(n)

(15)  lim_{n->inf} Pr{ Delta^{QE}(tau) in [Delta_hat^{QE}(tau)^-, Delta_hat^{QE}(tau)^+] for all tau in (0,1) } = 1 - alpha
```

`Sigma_hat^{QE}(tau)` is a uniformly consistent estimator of the asymptotic variance function of `sqrt(n)(Delta_hat^{QE}(tau) - Delta^{QE}(tau))`. The critical value `t_hat_{1-alpha}` consistently estimates the (1-alpha)-quantile of the Kolmogorov-Smirnov maximal t-statistic `t = sup_{tau in (0,1)} sqrt(n) Sigma_hat^{QE}(tau)^{-1/2} |Delta_hat^{QE}(tau) - Delta^{QE}(tau)|`. Implementation: draw B bootstrap samples; set `t_hat_{1-alpha}` to the (1-alpha) sample quantile of `{t_hat_b : 1 <= b <= B}` where `t_hat_b = sup_{tau in (0,1)} Sigma_hat^{QE}(tau)^{-1/2} |sqrt(n)(Delta_hat^{QE}(tau)* - Delta_hat^{QE}(tau))|`.

> **Theorem 10 (bootstrap validity, "third main result of this paper") - p. 20.** Assumptions 1 to 6 hold and the bootstrap weights follow the condition BW. Then, the exchangeable bootstrap consistently estimates the law of the limit stochastic processes in the Theorems and Corollaries 4 to 9. The confidence bands have a correct coverage probability.

Proof: Corollary 5.2(ii) of Chernozhukov, Fernandez-Val, and Melly (2013) gives bootstrap validity for the QR-based conditional distribution/quantile estimators; the result follows by the functional delta method for the bootstrap (chapter 3.9, Van der Vaart and Wellner 1996) plus Hadamard differentiability of all functionals involved.

*Functional hypothesis tests and the time-invariance specification test (p. 21).* The bands can test no-effect, positive-effect, or stochastic-dominance hypotheses (check whether the entire null falls within the band). They also give a **specification test of time invariance** when a second pre-treatment period exists: if `(Y, X)` is observed in period -1 with no one treated, Assumptions 1-4 imply

```
F_{Y,0-1x}( F^{-1}_{Y,1-1x}(tau) ) = F_{Y,00x}( F^{-1}_{Y,10x}(tau) )   for all tau in (0,1) and x in X
```

All quantiles and covariate values must be considered to detect deviations; Kolmogorov-Smirnov or Cramer-von-Mises type tests are "the natural way", with Theorem 10 justifying exchangeable-bootstrap critical values.

**Empirical application (Section 5, pp. 21-24; Figures 3-5, pp. 28-30) - food stamps and the birthweight distribution:**

- *Data and design:* same data as Almond, Hoynes, and Schanzenbach (2011) (AHS), complementing their DiD analysis with CiC. The Food Stamp Program rolled out county-by-county during the 1960s and early 1970s across roughly 3,100 U.S. counties; AHS analyzed 1968 (when ~40% of counties were already treated) to 1977. Melly-Santangelo must (i) exclude counties already treated at the start (their counterfactual is not identified) and (ii) stop in June 1974 (effects not identified once everyone is treated) (p. 21).
- *Sources (pp. 21-22):* administrative USDA annual reports on county caseloads (month/year of each county's FSP implementation = policy variable); Vital Statistics Natality Data from 1968 onward, ~2 million observations/year (100% or 50% sample of births by state-year) with birthweight, gender, race; covariates from the 1960 Census of Population, Census of Agriculture, and BEA REIS data.
- *Specification (p. 22):* results only for white mothers, using AHS's first specification: 1960 county variables (log population, % land in farming, % population black, urban, age below 5, age above 65, income < $3,000), each interacted with a linear time trend, plus per capita county transfer income (public assistance, medical care, retirement and disability benefits) and county real per capita income.
- *Practical complications (p. 23):* (1) 26 quarters -> 26 treatment-timing groups; any pair of periods identifies the effect for counties untreated in the first and treated in the second; presenting all pairs would be "almost impossible... and highly imprecise", so they present average results over all possible pairs, weighted to be representative of the treated counties in the data. (2) Time-trend-interacted covariates are not identified in period-by-period QR, so they estimate a single pooled quantile regression process for all non-treated observations, including indicator variables for the time periods and the groups. (3) Scale: CiC needs the whole micro-level distribution (no collapsing to county-quarter cell means), keeping above 5,000,000 observations; "This computational task becomes insurmountable if we need to bootstrap the results. Our solution was to use the newly developed quantile regression algorithms developed in Melly (2014) and subsampling instead of the bootstrap" (pp. 23-24). Footnote 11: each draw samples 500 of the 3,000+ counties, allowing arbitrary within-county correlation (cluster-level subsampling; the number of draws is not stated).
- *Findings (p. 24):*
  - Figure 3 (p. 28): QTE of food stamps on birthweight with subsampling-based uniform band. Positive effect, as in AHS; "The absolute value of our estimated effect is larger but the standard errors are also larger." Effects stronger at the lower tail ("certainly a good outcome for the policy because this is where it matters the most"); contrary to the original article, also a larger effect at the upper end (visually: ~100-150 g near tau ~ 0, a plateau of roughly ~5-15 g through the middle, rising again to ~100-150 g near tau ~ 1; bands exclude zero) - a U-shaped QTE.
  - Figure 4 (p. 29): quantile functions of control vs treated potential outcomes (roughly 1,500-4,600 g): "While the effects we find are strongly significant, this figure makes clear that they are not large in absolute value. The two quantile functions are almost impossible to distinguish."
  - Figure 5 (p. 30): QTE with vs without covariates: "in this application the results depend on the presence of covariates. Without covariates the effects are very close to zero for most of the distribution. On the other hand, the U-shaped pattern is the same with and without covariates."
- *Unexplained tail behavior:* both QTE figures (3 and 5) show large spikes at both extremes (tau near 0 and 1). The paper offers no boundary-artifact discussion of these spikes; the only related statements are substantive (stronger lower-tail and upper-tail effects). No trimming of the quantile grid or tail-truncation rule is reported for the application. Densities appear in denominators of every limit process ((11), (12), (13), Lemmas 2-3 derivative maps), so vanishing tail densities would inflate the asymptotic variance - consistent with, but not explicitly linked by the paper to, the wide bands at the extremes of Figure 3.

**Edge cases and boundary conditions:**

- Quantile crossing: population conditional quantile functions are strictly increasing under Assumption 6; in estimation, non-monotonic QR processes are handled by the integrated-indicator CDF construction (pp. 10-11, CFG 2010) rather than direct inversion. No additional rearrangement step is mentioned.
- Support/compactness: all Hadamard-differentiability lemmas require compact support and strictly positive continuously differentiable densities; Assumption 6(ii) requires `Y_gt x X` compact in `R^{K+1}`.
- Uniform (0,1) indexing: FCLTs are stated over `tau in (0,1)` and the KS statistic takes `sup_{tau in (0,1)}`; no explicit trimming to a compact sub-interval is stated in the theorems (the compact-support and bounded-density assumptions do the work). Separately, tail trimming of the QR grid is called practically unavoidable (p. 11) but left unspecified in this draft.
- Small cells with categorical covariates: acknowledged (p. 19) as a finite-sample fragility; the weighted bootstrap is recommended for "accuracy and robustness to 'small cells'".
- Staggered adoption identification boundaries: already-treated-at-start units must be dropped and the sample must end before universal treatment (p. 21).
- Support violation (Assumption 4): testable via `Y_10x subset Y_00x`; if violated, only the overlap part of the counterfactual distribution is identified (p. 6).
- Discrete outcomes: advised against (p. 6); "the effects are only partially identified with discrete outcomes" (citing AI); distribution regression flagged for a future discrete/mixed-outcome extension (p. 25).

**Relation to other methods:**

- *Athey-Imbens (2006) unconditional CiC:* the model is AI's nonseparable-outcome model `Y^N = h(X, T, U)` with all four AI assumptions imposed conditionally on X; Proposition 1 trivially extends AI's Theorem 3.1 (p. 6). The contribution is not identification but a practicable estimator: AI's own covariate strategies are either fully nonparametric (curse of dimensionality) or additively separable (contradicting CiC's non-additive motivation) (p. 2). Conditional time invariance also weakens the unconditional AI requirement. For panel data, AI Section 5.3-style correlation adjustments would be required (p. 12). Without covariates the paper still adds process-level limit theory that AI does not have (p. 18).
- *DiD / common trends:* Assumption 3 is "the CIC counterpart of the common trend assumption in the DID model" (p. 5). The pp. 22-23 analysis shows DiD applied to outcome indicators is inconsistent for distributional/quantile effects except in knife-edge cases (Figures 1-2).
- *Chernozhukov-Fernandez-Val-Melly (2013) counterfactual framework:* the estimation and inference machinery is built directly on CFM 2013 - the QR-process FCLT and conditional distribution process (Corollary 5.2, eqs. (9)-(10)); the counterfactual operator (Lemma D.1, ingredient 4); the treated-side unconditional FCLT (Theorem 4.1(ii), Theorem 7 proof); tail-trimming adaptations; and exchangeable-bootstrap validity for the QR coefficient process (Corollary 5.2(ii), Theorem 10 proof). The paper's addition on top of CFM is the pair of CiC-specific Hadamard-differentiable maps - the QQ transformation (Lemma 2) and PP transformation (Lemma 3) - composed with the counterfactual operator via the functional delta method.
- *Other connections:* monotonized QR-based CDFs - Chernozhukov, Fernandez-Val, and Galichon (2010); individual-treatment-effect-distribution asymptotics - Appendix C of Chernozhukov, Fernandez-Val, and Melly (2009) (p. 10); nonseparable-model identification - Matzkin (2003); rank similarity vs rank invariance - Chernozhukov and Hansen (2005); similar expressions in decomposition contexts - Fortin and Lemieux (1998), Altonji and Blank (1999); computation - Melly (2014), Portnoy (1991); alternative first stage - distribution regression (p. 10). Conclusion (pp. 24-25) situates the paper next to the instrumental CiC of de Chaisemartin and D'Haultfoeuille (2014) under partial compliance; the "translated quantile treatment effects" of Strittmatter (2014) and Bitler, Domina, and Hoynes (2014) (corresponds to the counterfactual non-treated distribution in the CiC model); Wuthrich (2014)'s closed-form IVQR estimator (Chernozhukov and Hansen 2006) with a binary instrument ("very similar to the expressions for which we have developed estimators"); and Melly and Santangelo (2014)'s panel sample-selection correction imputing conditional ranks across years via time invariance.

**Reference implementation(s):**

- Stata: Blaise Melly's website (https://sites.google.com/site/blaisemelly/home/computer-programs/cic_stata) distributes a Stata implementation accompanying this paper. The paper's Conclusion says only "we provide codes that implement the estimation and inference procedures developed in this paper" (p. 24) and names no command on the reviewed pages - the command-name attribution comes from the website, not the paper.
- **IMPORTANT disambiguation:** this is DISTINCT from Kranker's SSC `cic` Stata module, which implements Athey-Imbens 2006 (unconditional CiC with analytical SEs and discrete bounds). Verify the provenance of any Stata artifact before using it as a parity reference in a future covariates PR.
- ~~No R implementation is known to the initiative for covariate CiC.~~ **Correction (2026-07-13):** this was wrong - qte 1.3.1's `CiC()`/`QDiD()` support covariates via `xformla`, implementing a simplified form of this paper's QR pipeline (fixed 99-tau grid, per-observation imputation, treated-PRE integration, no monotonization). That branch is now diff-diff's R parity target for covariates (see REGISTRY). The FULL Melly-Santangelo estimator still has no R implementation and would need simulation-based validation.

**Requirements checklist (for a future FULL-Melly-Santangelo PR; the qte simplified branch shipped 2026-07-13 covers none of the rows below except as noted in the validation row):**
- [ ] Four per-cell QR coefficient processes on a fine mesh `u_1..u_S` with `delta*sqrt(n) -> 0`; Koenker-Bassett check-function objective per (g,t) cell
- [ ] Monotonized conditional CDFs via the integrated-indicator representation (pp. 10-11, CFG 2010); never invert raw QR processes directly
- [ ] Conditional CiC composition per eq. (7); covariate integration over the empirical `F_{X|11}` per the p. 12 displays (resolving the printed index/mesh-weight typos); treated side either empirical CDF or symmetric QR-based integration (application uses the latter)
- [ ] Unconditional QTE per eq. (8) via generalized inverses of the integrated CDFs
- [ ] Exchangeable bootstrap satisfying Condition BW (weighted QR per draw); weighted bootstrap for small cells, subsampling for very large n; cluster-level subsampling option (footnote 11 pattern)
- [ ] KS maximal-t uniform bands per (14)-(15) with a uniformly consistent variance function estimate from the bootstrap draws
- [ ] Pre-period time-invariance specification test (p. 21) when 2+ pre-treatment periods exist (KS/CvM over all tau and x, bootstrap critical values via Theorem 10)
- [ ] Tail-trimming policy: the paper calls it unavoidable but the draft sentence is incomplete; adapt Chernozhukov, Fernandez-Val, and Melly (2013)
- [ ] Support diagnostic: test `Y_10x subset Y_00x` (Assumption 4's observable implication); warn and restrict to the overlap if violated
- [ ] Staggered adoption: pairwise-period estimates averaged with weights representative of treated units; pooled QR with period and group dummies when covariates are interacted with trends (Section 5 strategy - application-level, no general theorem)
- [ ] Error on discrete outcomes (authors advise against; only partial identification)
- [ ] Validation strategy for the FULL estimator: simulation-based (no R reference exists for the full pipeline; qte's `xformla` branch anchors only the simplified form and is already parity-tested); verify any Stata parity artifact is Melly's covariate-CiC code, NOT Kranker's SSC `cic` (Athey-Imbens unconditional)

---

## Implementation Notes

**Relevance to diff-diff CiC/QDiD (2026-07-12 v1; updated 2026-07-13 covariates PR):**
- (i) This paper's Figures 1-2 bias analysis of DiD (and the analogous concern for unconditional CiC when group compositions differ) is the documented motivation for covariate support - unconditional CiC assumptions can fail when conditional ones hold. v1 shipped covariate-free; the covariates PR shipped the qte-`xformla` simplified form of this paper's pipeline.
- (ii) The four-step QR pipeline (estimate conditional CDFs by quantile regression, apply CiC transformations conditionally, integrate over the treated covariate distribution, invert) IS the implemented blueprint, in qte's simplified form (fixed 99-tau grid, per-observation imputation, treated PRE-period integration, raw un-rearranged step functions).
- (iii) The exchangeable bootstrap + KS-band machinery is shared with Callaway-Li-Oka 2018 and reinforces the bootstrap-first inference choice; the implemented covariate bootstrap is qte's (unit-block / pooled-row), not this paper's exchangeable weights.
- (iv) The monotonization/rearrangement step for QR-estimated conditional CDFs is the key numerical ingredient of the FULL estimator that the implemented qte form deliberately omits (qte uses the raw QR process; documented REGISTRY Note).

### Data Structure Requirements
- Repeated cross sections with four (g,t) cells, each with positive probability `alpha_gt` (Assumption 5). Panel data requires modified theory (AI Section 5.3-style within-group over-time correlation terms) that the paper does not develop.
- Continuous outcomes only; the authors advise against discrete outcomes (p. 6); distribution regression is the flagged route for a discrete/mixed extension (p. 25).
- Covariates enter through linear-in-parameters QR per cell (Assumption 6(i)); flexible transformations of regressors can approximate the truth arbitrarily well when the conditional density is smooth (p. 10).
- Compact support `Y_gt x X` with uniformly bounded, uniformly continuous conditional densities (Assumption 6(ii)); densities appear in the denominators of all limit processes.
- Micro-level data required end to end - CiC needs the whole conditional distribution; collapsing to cell aggregates is not possible (p. 23).
- Multi-group/multi-period designs are handled by the application's strategy (pairwise 2x2 comparisons averaged with treated-representative weights; pooled QR with period/group dummies), not by general theorems.

### Computational Considerations
- The dominant cost is estimating four full QR coefficient processes. The number of distinct QR solutions is `O(n log n)` (Portnoy 1991, footnote 7); the algorithms of Melly (2014) cut computation ~10x; the application runs a pooled QR process on 5,000,000+ observations.
- Bootstrapping the full QR process at that scale is "insurmountable" (pp. 23-24); the paper substitutes cluster-level subsampling (500 of 3,000+ counties per draw). The exchangeable-bootstrap framework (Condition BW) covers empirical, weighted, m-of-n, and subsampling variants under one validity theorem (Theorem 10).
- Densities appear only in the asymptotic-variance expressions, never in the estimator or the bootstrap - the paper explicitly recommends resampling over analytical SEs because the variances contain conditional densities of Y given X that are hard to estimate (p. 18). This matches the bootstrap-first inference design of the v1 CiC/QDiD estimators.
- The monotonized-CDF step is an S-term indicator sum per evaluation point (searchsorted-style over the per-cell QR coefficient mesh after projecting onto x); the conditional CiC composition is two such evaluations plus one QR-process lookup per (tau, x).
- KS maximal-t bands need a uniformly consistent variance function `Sigma_hat^{QE}(tau)`, obtainable from the same bootstrap/subsampling draws.

### Tuning Parameters

| Parameter | Type | Paper guidance | Notes |
|-----------|------|----------------|-------|
| QR grid size S / mesh width `delta` | int / float | `delta*sqrt(n) -> 0` (p. 11); estimating all distinct QRs is feasible (footnote 7) | mesh weight `delta` (or 1/S) must multiply the integrated-indicator sums; the p. 12/p. 20 displays omit it (printed typo) |
| Resampling scheme | {empirical, weighted, m-of-n, subsampling} | weighted bootstrap for small samples with categorical covariates; subsampling for very large samples (p. 19) | all covered by Condition BW / Theorem 10 |
| Number of draws B | int | not reported for the application | library convention required |
| Subsample size (subsampling) | int | application: 500 of 3,000+ counties per draw, cluster-level (footnote 11) | allows arbitrary within-cluster correlation |
| Tail trimming | float pair | "seems unavoidable in practice" but unspecified (incomplete sentence, p. 11); adapt CFM (2013) | a future PR must pick and document a rule |
| tau grid for bands | floats in (0,1) | theorems stated over `tau in (0,1)`; no explicit compact trimming in the statements | compact-support + bounded-density assumptions do the work; tails are where the application's bands blow up |

### Relation to Existing diff-diff Estimators
- `ChangesInChanges` (v1): the exact special case X = constant. The paper's remark 3 on p. 18 notes that even without covariates it contributes the limiting distribution of the whole quantile and distribution processes with a simpler proof than AI - a useful citation when documenting process-level (uniform-band) inference for the v1 estimator.
- `QDiD` (v1): the pp. 22-23 DiD-on-indicators analysis is the paper's cautionary tale about additive-in-distribution shortcuts for distributional targets; relevant background for the choosing-an-estimator docs alongside the AI (2006) QDiD discussion, though the object analyzed (DiD applied to `1(Y <= y)`) is not the QDiD transform itself.
- Callaway-Li-Oka (2018) review (sibling, same initiative): shared inference stack - exchangeable bootstrap weights, functional delta method over Hadamard-differentiable ECDF/quantile compositions, KS-based uniform bands. Both ultimately lean on Van der Vaart-Wellner chapter 3.9; this paper routes through CFM (2013) Corollary 5.2 for the QR first stage.
- A future covariates implementation should reuse the `safe_inference()` joint-NaN contract from `diff_diff.utils`, the results-dataclass conventions, and the bootstrap utilities, and would be a standalone estimator (mixing in `BaseEstimator` from `diff_diff/_base.py` for `get_params`/`set_params` per the estimator-inheritance map in CLAUDE.md).

---

## Gaps and Uncertainties

**Suspected typos in the October 2015 preliminary draft (flagged by extraction; transcribed as printed, not silently corrected):**
- p. 11: the tail-trimming sentence is printed incomplete - "Tail trimming seems unavoidable in practice because ." - so the draft never states the reason or a rule; the surrounding text points to Chernozhukov, Fernandez-Val, and Melly (2013) for adapted formulas.
- p. 11: the inline restatement of Proposition 1's counterfactual quantile writes the left-hand argument as `(y)` where the p. 6 display has `(tau)`.
- p. 12 integration displays (both `F_hat_{Y^N|11}` and `F_hat_{Y^I|11}`): the outer sum runs over `j = 1..S` while the summand prints `u_s` (and, on the counterfactual side, `tau`), and the mesh-width factor `delta` (equivalently 1/S) from the p. 11 discretization is not printed. The intended estimator evaluates on the mesh `{u_j}` with the `delta` weight, mirroring the p. 11 formula.
- eq. (8) (p. 12): both right-hand quantile functions are printed with argument `(y)` although the process is indexed by the quantile level; both sides are (generalized) inverses of the integrated CDF estimators.
- p. 8: the intuition sentence below the Proposition 1 proof swaps group labels relative to the displayed identification formula; this review transcribes consistently with the displays.
- eq. (10) (p. 13): the limit process is printed with index `(u, x)` where the display is a process in `(y, x)`, and the scaling prints `sqrt(n*alpha_gt)` where (9) uses `sqrt(n)` - transcribed as printed.
- Lemma 3 (p. 14): the derivative map for `phi^{PP}(F, G) = F^{-1} o G` prints a NEGATIVE sign on the `h_2` term, `- (h_2/(f o F^{-1})) o G`. Standard inverse-map calculus (Van der Vaart-Wellner Lemma 3.9.23(ii) plus the chain rule) gives a POSITIVE `h_2` term: the G-perturbation enters through `(F^{-1})' = 1/(f o F^{-1})` with no sign flip; only the `h_1` term (through inverting F) is negative. The printed proof also cites Hadamard differentiability of "the inverse map G^{-1}" although the PP map inverts F - both oddities are consistent with a copy-paste slip from Lemma 2 (the QQ map, whose proof genuinely inverts G). Re-derive this derivative before any analytical covariance implementation that composes these maps; the bootstrap-based inference path does not depend on it.
- `V^Q_gt` covariance (p. 13): the evaluation point of the second Jacobian is hard to resolve at print resolution; transcribed as `J_gt(u~)^{-1}`, the standard QR-process covariance form.
- p. 20 bootstrap display: typographically dense, with the same missing mesh-weight issue as the p. 12 displays; the structure is the Condition-BW-weighted empirical analog of the QQ/PP composition over the S-point u-grid.
- Sign-convention asymmetry (as printed, identically in both extraction passes): eq. (10) defines `Z^F_gt` with a leading minus (`-f * Z(F(y), x)`) while the Theorem 7 proof defines `Z^F_N` with a plus (`+f * Z^Q_N(F(y), x)`). Immaterial for the limit laws (zero-mean Gaussian processes are sign-symmetric), but confusing if the formulas are transplanted into covariance code - re-derive signs if analytical variances are ever implemented.

**Extraction cross-check (overlap pp. 13-17):** the two extraction passes double-covered pp. 13-17 (Assumption 6, eqs. (9)-(11), Lemmas 2-3, Theorem 4, Corollaries 5-6, Theorem 7 and its proof intermediates). All displays were transcribed identically by both passes; no contradictions between extraction files were found. Section labeling is consistent: the p. 6 forward reference to "Section 4.4" for time-invariance testing resolves to Section 4.4 "Inference" (pp. 19-21), which contains the specification test.

**Version and coverage gaps:**
- **Preliminary working paper.** The reviewed October 2015 draft is flagged "Preliminary!" and contains the printed typos above. All equation/assumption/theorem numbers are pinned to this draft; later drafts, if any, may renumber or repair the typos. The paper remains unpublished as of 2026-07-12.
- **No Stata command name in the paper.** The Conclusion (p. 24) says "we provide codes" without naming a command anywhere on the reviewed pages; the command-name attribution rests on Melly's website. Verify provenance before parity use, and do not confuse it with Kranker's SSC `cic` module (Athey-Imbens 2006 unconditional CiC).
- **No R implementation of the FULL estimator, and no Monte Carlo.** The paper reports no simulations (only the analytic DiD-bias illustration, pp. 22-23). qte 1.3.1's `xformla` branch is an R implementation of the SIMPLIFIED pipeline and shipped as diff-diff's parity-tested covariate route (2026-07-13); the full estimator developed here (monotonized CDFs, treated-post integration, exchangeable bootstrap, uniform bands, specification test) still has no R reference - implementing it would need simulation-based validation and possibly a purpose-built oracle.
- **Number of resampling draws unreported.** No B is stated for the application's subsampling (500 of 3,000+ counties per draw, footnote 11; draw count absent).
- **Tail spikes unexplained.** Figures 3 and 5 show large QTE spikes at both extremes; the paper offers no boundary-artifact discussion, no quantile-grid trimming, and no tail-truncation rule for the application. The density-in-denominator structure of (11)-(13) is a plausible mechanical explanation but is not drawn by the paper.
- **Staggered-adoption aggregation is informal.** The pairwise-averaging weights ("representative of the treated counties") and the pooled-QR-with-dummies device are described in prose (p. 23) without formulas or dedicated asymptotic theory; the formal results cover only the 2x2 case.
- **Panel data not covered.** Assumption 5 is repeated cross sections; the paper notes AI Section 5.3-style modifications would be needed for panels (p. 12) but does not derive them. The individual-effect-distribution estimand (eq. (6)) needs panel data plus rank invariance and is left to a combination with Chernozhukov, Fernandez-Val, and Melly (2009, Appendix C).
- **Weak monotonicity / discrete outcomes.** Only strict monotonicity is handled; AI's weakly-increasing-`h` partial-identification case is deferred to future work (footnote 4, p. 6), and distribution regression for discrete/mixed outcomes is only sketched in the Conclusion (p. 25).
- **Efficiency claims unquantified.** The introduction asserts covariates can bring efficiency gains even when unconditional time invariance holds (p. 2); no formal efficiency comparison appears in the paper.
