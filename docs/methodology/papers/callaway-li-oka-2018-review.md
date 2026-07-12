# Paper Review: Quantile Treatment Effects in Difference in Differences Models under Dependence Restrictions and with only Two Time Periods

**Authors:** Brantly Callaway, Tong Li, and Tatsushi Oka
**Citation:** Callaway, B., Li, T., & Oka, T. (2018). Quantile Treatment Effects in Difference in Differences Models under Dependence Restrictions and with only Two Time Periods. *Journal of Econometrics*, 206(2), 395-413. https://doi.org/10.1016/j.jeconom.2018.06.008
**PDF reviewed:** **arXiv:1702.03618v1** (https://arxiv.org/abs/1702.03618v1, submitted 13 Feb 2017; title page reads 'This version: February 14, 2017'; 34 PDF pages, PDF page N = printed page N-1). Per the project's PDFs-never-committed convention the local PDF is kept outside the repository (gitignored `papers/callaway-li-oka-1702.03618v1.pdf`). The published Journal of Econometrics 206(2) version (https://doi.org/10.1016/j.jeconom.2018.06.008) is the version of record and may renumber equations/assumptions - all numbers below are pinned to arXiv v1; reconcile against the published version before implementing anything sensitive.
**Review date:** 2026-07-12

---

## Methodology Registry Entry

*Drafted in docs/methodology/REGISTRY.md format for FUTURE use. Do not copy this section into the registry now - see the status line below.*

## PanelQTT (Callaway-Li-Oka two-period QTT)

**Status: NOT shipping as a diff-diff estimator in the CiC/QDiD v1 (scope decision 2026-07-12).** This paper is reviewed as (a) the methodological grounding for panel-mode distributional DiD and the dependence-restriction (Copula Invariance) approach, and (b) part of the foundations of the R `qte` package, the initiative's chosen parity target for CiC/QDiD. Any future implementation of this estimator would be a separate scoped PR.

**Primary source:** Callaway, B., Li, T., & Oka, T. (2018). Quantile Treatment Effects in Difference in Differences Models under Dependence Restrictions and with only Two Time Periods. *Journal of Econometrics*, 206(2), 395-413. https://doi.org/10.1016/j.jeconom.2018.06.008
- All proofs are in the Appendix (printed pp. 22-28); Tables 1-2 (Monte Carlo) are on printed pp. 29-30 and Tables 3-4 / Figure 1 (application) on printed pp. 31-33, after the References (printed pp. 18-21).
- **Numbering quirk (verified by full-text search during extraction):** the main text has Assumptions A1, A2, A3, A4, A6, and B. There is NO Assumption A5 anywhere in the paper - the label "A5" appears only as appendix equation (A5) and appendix Lemma A5. Lemma 1, Proposition 1, Theorem 2, Lemma 2, and Theorem 3 all cite "Assumption A1-A6", which in practice means A1-A4 plus A6.

**Model (Section 2.1, printed pp. 3-4):**

Panel data with exactly two periods. All individuals are untreated in period `t-1`; a fraction becomes treated in period `t`. Data: `{(Y_{i,t-1}, Y_it, X_i, D_it)}_{i=1}^n`. Observed outcomes (eq. (1), printed p. 4):

```
(1)    Y_{i,t-1} := Y_{i,t-1}(0)   and   Y_it := (1 - D_it) Y_it(0) + D_it Y_it(1)
```

The two panel observables that drive everything are the outcome change `DeltaY_it := Y_it - Y_{i,t-1}` and the initial outcome level `Y_{i,t-1}`. `DeltaY_it(0) := Y_it(0) - Y_{i,t-1}(0)` is the time-difference of untreated potential outcomes (printed p. 5); for untreated units `DeltaY_it = DeltaY_it(0)`.

Why two periods suffice (Introduction, printed pp. 1-2): the Copula Invariance assumption replaces the treated group's unknown copula (between the change and the initial level of untreated potential outcomes) with the untreated group's *observed* copula in the same two periods. This contrasts with Callaway and Li (2015), which needs at least three periods of panel data because it replaces an unknown unconditional copula in the last two periods with an observed copula from the first two periods.

Notation (printed pp. 3-4, 8-9):

| Symbol | Meaning |
|---|---|
| `D_it` | Treatment indicator: 1 if individual i treated in period t; nobody treated in t-1 |
| `Y_is(0)`, `Y_is(1)` | Potential outcomes in period `s ∈ {t-1, t}` |
| `X_i`, `X` (calligraphic) | Covariate vector (may include time-varying and time-invariant variables); `X` = common support of `X_i` for treated and untreated groups |
| `F_{Y_s(j)|X,D_t}`, `F_{Y_s|X,D_t}` | Conditional distributions of potential / observed outcomes |
| `F^{-1}_{Y_t(j)|X,D_t}(tau) := inf{y ∈ R : F_{Y_t(j)|X,D_t}(y) >= tau}` | Conditional quantile function, j = 0, 1 (printed p. 4; inf-based, weak inequality) |
| `C_{DeltaY_t(0),Y_{t-1}(0)|X,D_t}` | Conditional copula of `(DeltaY_it(0), Y_{i,t-1}(0))` given `X_i` and `D_it` |
| `tau ∈ T ⊂ (0,1)` | Quantile index; T a compact subset strictly inside the unit interval (printed p. 9) |
| `delta^{(d)}_{i,x} := 1{X_i = x, D_it = d}`, `n_x^{(d)} = SUM_i delta^{(d)}_{i,x}` | Cell indicator and cell size for discrete covariates (printed p. 8) |
| `r_x^{(j)} := lim_{n->inf} (n / n_x^{(j)})^{1/2} ∈ (0, inf)` | Relative cell-size limits, j = 0, 1 (Assumption A6(b)) |
| `m^(s)` | Sample size of the period-s repeated cross section (Corollary 1) |
| `W_i^{(d)}` | Exchangeable bootstrap weights for group d (Assumption B) |
| `∘` | Function composition |

Sampling scheme: i.i.d. observations *within* treatment and control group (conditional random sampling), which "allows for the possibility that the marginal or joint distributions of potential outcomes can be different between treatment and control groups" (printed p. 4).

Covariates are discrete for the estimation theory (Section 3, printed p. 8): "we consider the case where all covariates are discrete, which allows for nonparametric estimation that does not suffer from the curse of dimensionality" (motivated on printed p. 3, citing Chernozhukov et al. 2013b and Graham et al. 2015).

**Target parameters (printed pp. 4, 7):**

CQTT (Conditional Quantile Treatment effect on the Treated) - the QTE within the subpopulation with `D_it = 1` and common history `X_i = x`. For `x ∈ X`, `tau ∈ T ⊂ (0,1)`:

```
Delta_x^QTT(tau) := F^{-1}_{Y_t(1)|X=x,D_t=1}(tau) - F^{-1}_{Y_t(0)|X=x,D_t=1}(tau)
```

`F_{Y_t(1)|X=x,D_t=1}` is identified directly by the observed `F_{Y_t|X=x,D_t=1}`; the entire problem is the counterfactual `F_{Y_t(0)|X=x,D_t=1}`. The unconditional QTT is identified by averaging the conditional counterfactual distribution of Theorem 1 over covariates and inverting (discussion after Theorem 1, printed p. 7), but "for the rest of the paper we focus only on the CQTT" - the paper gives no formal aggregation theory (see Gaps).

**Assumptions (exact statements):**

*Assumption A1 (Random sampling) - printed p. 4:*

> The two-periods panel data consists of observations `{(Y_{i,t-1}, Y_it, X_i, D_it)}_{i=1}^n` from the structure in (1). The potential outcomes `(Y_{i,t-1}(0), Y_{i,t-1}(1))` and `(Y_it(0), Y_it(1))` are cross-sectionally i.i.d. conditional on treatment status `D_it`.

*Assumption A2 (Distributional Difference in Differences) - printed p. 5:*

```
Pr{ DeltaY_it(0) <= Deltay | X_i, D_it = 1 } = Pr{ DeltaY_it(0) <= Deltay | X_i, D_it = 0 },
for all Deltay ∈ supp(DeltaY_it(0) | X_i).
```

- The distributional extension of mean parallel trends `E[DeltaY_it(0)|X_i, D_it = 1] = E[DeltaY_it(0)|X_i, D_it = 0]` (necessary for the ATT; Heckman et al. 1998, Abadie 2005). "The distributional restriction in Assumption A2 replaces this standard mean restriction."
- A2 restricts only the *change* `DeltaY_it(0)`; it does NOT require the marginal distribution of the initial level `Y_{i,t-1}(0)` to be the same across groups: "the initial distribution of outcomes can be different for the two groups" (printed p. 6).
- "If multiple pre-treatment periods in sample are available, then this assumption can be tested under a strict stationary assumption" (printed p. 5).

*Assumption A3 (Copula Invariance) - printed p. 5 (the "dependence restriction"):*

> For each `x ∈ X` and for all `(u, v) ∈ [0,1]^2`,

```
C_{DeltaY_t(0),Y_{t-1}(0)|X=x,D_t=1}(u, v) = C_{DeltaY_t(0),Y_{t-1}(0)|X=x,D_t=0}(u, v).
```

Interpretation (printed p. 6):
- The copula captures the "rank dependency" between `DeltaY_it(0)` and `Y_{i,t-1}(0)`; A3 requires this rank dependency to be the same for treated and control groups. Intuition: "if ... observations in the control group at the top of the distribution of initial outcomes tend to experience the largest increases in outcomes over time, the Copula Invariance assumption implies that this would also occur for the treated group."
- A3 does NOT rule out the joint distribution of `(DeltaY_it(0), Y_{i,t-1}(0))` differing between groups (only the copula is restricted), and it "allows for the marginal distribution of untreated outcomes in the period before treatment to differ for the treated and control groups" (printed p. 1).
- A3 does not imply A2 and A2 does not imply A3 - they are separate restrictions ("A3 restricts only the copula, not the marginal distribution of the change in untreated potential outcomes over time").
- Testability: A2 and A3 "are not directly testable. However, in the spirit of placebo testing in DID models, they both can be tested using periods before the treated group becomes treated." Simpler route: implement the procedure in earlier periods and test `Delta_x^QTT(tau) = 0` for all `tau ∈ T`.

*Assumption A4 (Continuous distributions) - printed p. 6:*

> Each random variable of `DeltaY_it(0)` and `Y_{i,t-1}(0)` has a continuous distribution conditional on `X_i` and `D_it` and a random variable `Y_it(1)` also has a continuous distribution conditional on `X_i` and `D_it = 1`. Each distribution has a compact support with densities uniformly bounded away from 0 and infinity over the support.

- Continuity guarantees the conditional copulas in A3 are unique; identification uses the Rosenblatt transform (Rosenblatt, 1952).
- The compact-support part is "as in Athey and Imbens (2006) in order to avoid technical difficulties in the rest of analysis, while this condition is not used for our identification analysis and can be replaced by other conditions for the rest of the results."

*Assumption A6 - printed p. 9 (estimation; note there is no A5):*

> (a) A pair of random variables `(DeltaY_it, Y_{i,t-1})` is continuously distributed conditional on `X_i` and `D_it = 0` over a compact support with a distribution `F_{DeltaY_t,Y_{t-1}|X,D_t=0}` and a density `f_{DeltaY_t,Y_{t-1}|X,D_t=0}`. A random variable `DeltaY_it` is continuously distributed conditional on `Y_{i,t-1}`, `X_i`, and `D_it = 0` with a uniformly continuous density `f_{DeltaY_t|Y_{t-1},X,D_t=0}` over a compact support. (b) The sample sizes `n_x^{(0)}` and `n_x^{(1)}` go to infinity as `n -> infinity`, while `r_x^{(j)} := lim_{n->inf} (n/n_x^{(j)})^{1/2} ∈ (0, inf)` for `j = 0, 1`.

For Theorem 2 additionally: `F_{Y_t(0)|X,D_t=1}` admits a positive continuous density on an interval `[a,b]` containing an epsilon-enlargement of `{F^{-1}_{Y_t(0)|X,D_t=1}(tau) : tau ∈ T}` (standard for quantile inversion / Hadamard differentiability; printed pp. 10-11).

*Assumption B (bootstrap weights) - printed p. 11:*

> For each `d ∈ {0,1}`, let `(W_1^{(d)}, ..., W_n^{(d)})` be an n-dimensional vector of exchangeable, nonnegative random variables. The vectors `(W_1^{(0)}, ..., W_n^{(0)})` and `(W_1^{(1)}, ..., W_n^{(1)})` are independent of the original sample as well as each other. The vectors of random weights, depending on the size of each group, satisfy:

```
max_{1<=i<=n} E|W_i^{(d)}|^{2+eps} < inf,
W_bar_{n,x}^{(d)} := (1/n_x^{(d)}) SUM_i W_i^{(d)} delta^{(d)}_{i,x} ->_p 1,
(1/n_x^{(d)}) SUM_i (W_i^{(d)} - W_bar_{n,x}^{(d)})^2 delta^{(d)}_{i,x} ->_p 1,
for each d ∈ {0,1}.
```

**Identification (Theorem 1, printed p. 7):**

Under A2 alone, for the treated group one identifies (i) `F_{Y_{t-1}(0)|X,D_t=1}` from observed outcomes and (ii) `F_{DeltaY_t(0)|X,D_t=1}` via A2; hence the ATT is identified. But the CQTT is NOT point-identified: "many possible distributions of untreated potential outcomes in period t are observationally equivalent" - `F_{Y_t(0)|X,D_t=1}` is highly unequal if change and initial level are strongly positively dependent, less unequal if independent or negatively dependent. The CQTT "can be partially identified along the line of Fan and Yu (2012)." By Sklar's theorem (printed p. 5) the missing object is exactly the conditional copula, which A3 supplies from the untreated group.

> **Theorem 1.** Suppose that Assumption A1-A4 hold. Then,

```
F_{Y_t(0)|X=x,D_t=1}(y)
  = Pr{ DeltaY_it + F^{-1}_{Y_{t-1}|X=x,D_t=1} ∘ F_{Y_{t-1}|X=x,D_t=0}(Y_{i,t-1}) <= y | X_i = x, D_it = 0 },
```

> for all `x ∈ X` and `y ∈ supp(Y_it(0) | X_i = x, D_it = 1)`.

Reading: the counterfactual distribution is identified entirely from observed outcomes of *untreated* individuals (`D_it = 0`). Each untreated unit's observed change `DeltaY_it` is added to a quantile-quantile transplanted initial level: the untreated unit's period t-1 outcome is mapped to its rank in the untreated t-1 distribution (`F_{Y_{t-1}|X=x,D_t=0}`), then to the treated group's t-1 outcome at that same rank (`F^{-1}_{Y_{t-1}|X=x,D_t=1}`). "This implies that treated and untreated groups must be similar in the distributional sense of not only marginal distribution but also some dependency over periods, and thus Assumption A2 and A3 play a crucial role."

Identification mechanics (Proof of Theorem 1, printed p. 22): define the ranks

```
(A1)   U_i^d := F_{DeltaY_t(0)|X=x,D_t=d}(DeltaY_it(0))   and   V_i^d := F_{Y_{t-1}(0)|X=x,D_t=d}(Y_{i,t-1}(0)),
```

for `d ∈ {0,1}`; under Assumption A4,

```
(A2)   DeltaY_it(0) = F^{-1}_{DeltaY_t(0)|X=x,D_t=d}(U_i^d)   and   Y_{i,t-1}(0) = F^{-1}_{Y_{t-1}(0)|X=x,D_t=d}(V_i^d)
```

almost surely (Rosenblatt, 1952). The joint distribution of `(U_i^d, V_i^d)` given `(X_i, D_it) = (x, d)` is the conditional copula `C_{DeltaY_t(0),Y_{t-1}(0)|X=x,D_t=d}`, invariant with respect to `D_it` under A3; combining with A2 (`F^{-1}_{DeltaY_t(0)|X=x,D_t=1}(.) = F^{-1}_{DeltaY_t(0)|X=x,D_t=0}(.)`) delivers the counterfactual `F_{Y_t(0)|X=x,D_t=1}`. Note the copula is handled *implicitly* by pairing each untreated observation's own (change, initial level) - it is never estimated as an explicit copula object.

**Repeated cross sections (Corollary 1, printed p. 7):**

> **Corollary 1.** Consider the repeated cross sections `{(Y_is, X_i, D_is)}_{i=1}^{m^(s)}` in period `s ∈ {t-1, t}` with `m^(s)` being the sample size. Suppose that the data generating process for the repeated cross sections satisfy Assumption A1-A4 hold. If the conditional copula of `(Y_{i,t-1}(0), Y_{i,t}(0))` given `X_i = x` and `D_it = 1` satisfies the rank invariance: for every `(u, v) ∈ [0,1]^2`,

```
C_{Y_{t-1}(0),Y_t(0)|X=x,D_t=1}(u, v) = min{u, v},
```

> then, for `y ∈ supp(Y_it(0) | X_i = x, D_it = 1)`,

```
F_{Y_t(0)|X=x,D_t=1}(y)
  = Pr{ DeltaY_tilde_it + F^{-1}_{Y_{t-1}|X=x,D_t=1} ∘ F_{Y_{t-1}|X=x,D_t=0}(Y_{i,t-1}) <= y | X_i = x, D_it = 0 },

where  DeltaY_tilde_it := F^{-1}_{Y_t|X,D_t=0} ∘ F_{Y_{t-1}|X,D_t=0}(Y_{i,t-1}) - Y_{i,t-1}.
```

Discussion (printed pp. 7-8): conditional rank invariance means "for observations with the same observed covariates, individuals maintain their rank in the distribution of outcomes over time." It is "weaker than unconditional rank invariance as some individuals can change their rank"; it "does not imply nor is implied by the Copula Invariance assumption, nor does it imply conditional rank invariance between `DeltaY_it(0)` and `Y_{it-1}(0)`." The extension requires restricting to time-invariant covariates. Proof mechanics (printed p. 22): the individual change `DeltaY_it(0)` is not observed in repeated cross sections, so under rank invariance `F_{Y_t(0)|X=x,D_t=0}(Y_it(0)) = F_{Y_{t-1}(0)|X=x,D_t=0}(Y_{i,t-1}(0))` and the change for untreated individuals is imputed by the `DeltaY_tilde_it` display above.

**IMPORTANT - unresolved conditioning flag:** the Corollary 1 statement conditions the rank-invariance copula on `D_it = 1` (treated group), but the identification formula recovers `DeltaY_tilde_it` from *untreated* observables and the appendix proof applies rank invariance to untreated units (`D_t = 0` in both displays of the proof, printed p. 22). See Gaps and Uncertainties - resolve against the published JoE version before any implementation.

Contrast with Athey-Imbens CiC: CiC's repeated cross-section identification (Athey and Imbens 2006, Section 3.4) requires only equal within-group marginals of the unobservable over time - CiC does not need rank invariance over time, whereas this paper's panel-to-RC extension does. Panel data is what lets this paper avoid rank invariance for the main result.

**Estimator (Section 3.1, printed p. 8; discrete covariates, cell-by-cell):**

Step 1 - conditional empirical distributions (eq. (2)), for `s ∈ {t-1, t}` and `d ∈ {0, 1}`:

```
(2)    F_hat_{Y_s|X=x,D_t=d}(y) := (1 / n_x^{(d)}) SUM_{i=1}^n 1{Y_is <= y} delta^{(d)}_{i,x}
```

The treated-outcome distribution estimator is `F_hat_{Y_t(1)|X=x,D_t=1}(y) = F_hat_{Y_t|X=x,D_t=1}(y)` (since `Y_it = Y_it(1)` if `D_it = 1`).

Step 2 - plug-in counterfactual distribution (eq. (3)): obtain estimated quantiles `F_hat^{-1}_{Y_{t-1}|X=x,D_t=1}` from the empirical distribution, then set

```
(3)    F_hat_{Y_t(0)|X=x,D_t=1}(y)
         := (1 / n_x^{(0)}) SUM_{i=1}^n 1{ DeltaY_it + F_hat^{-1}_{Y_{t-1}|X=x,D_t=1} ∘ F_hat_{Y_{t-1}|X=x,D_t=0}(Y_{i,t-1}) <= y } delta^{(0)}_{i,x}
```

for `y ∈ R` - the empirical analogue of Theorem 1, summing over untreated observations in cell x, each transformed by the empirical quantile-quantile map.

Step 3 - CQTT by quantile inversion (unnumbered display, printed p. 8):

```
Delta_hat_x^QTT(tau) := F_hat^{-1}_{Y_t(1)|X=x,D_t=1}(tau) - F_hat^{-1}_{Y_t(0)|X=x,D_t=1}(tau),   (tau, x) ∈ T x X.
```

Quantile functions use the inf-based convention `F^{-1}(tau) := inf{y : F(y) >= tau}` (printed p. 4). The estimator is a pure empirical-distribution plug-in: no bandwidths, kernels, or smoothing parameters anywhere; covariates are handled as discrete cells, not by smoothing.

Appendix plug-in map (display (A4), printed p. 24): the counterfactual estimator "can be considered as an empirical distribution indexed by estimated distribution functions" (Proof of Proposition 1, printed p. 27), i.e. `F_hat_{Y_t(0)|X=x,D_t=1} = phi_n(F_hat_{Y_{t-1}|X=x,D_t=0}, F_hat_{Y_{t-1}|X=x,D_t=1})` for a generic sample map: for `F = (G, H)` a pair of distribution functions and `w` in a compact set `W`,

```
(A4)   phi_n(F)(w) := n^{-1} SUM_{i=1}^n 1{ V_{1i} + G^{-1} ∘ H(V_{2i}) <= w },
```

with population counterpart `phi(F)(w) := Pr{ V_1 + G^{-1} ∘ H(V_2) <= w }` (Lemma A2). In the paper's objects, `V_1` is the change and `V_2` the initial level for the untreated group. (The two extraction passes disagree on which of G, H maps to the treated vs untreated initial-level CDF in this generic notation - see Gaps and Uncertainties; the main-text eq. (3) composition above is unambiguous.)

**Asymptotic theory (Section 3.2, printed pp. 9-11; proofs printed pp. 26-27):**

Infeasible-estimator machinery (printed p. 9): `Y_tilde_it := DeltaY_it + F^{-1}_{Y_{t-1}|X=x,D_t=1} ∘ F_{Y_{t-1}|X=x,D_t=0}(Y_{i,t-1})` (true, not estimated, transforms) with empirical process

```
(4)    G_tilde_{t,x}^{(0)}(y) := sqrt(n) ( F_tilde_{Y_t(0)|X=x,D_t=1}(y) - F_{Y_t(0)|X=x,D_t=1}(y) ),   y ∈ Y_{t|x,1}(0).
```

(The v1 text describes `F_tilde` as "based on observations {Y_tilde_it} with X_i = x and D_it = 1 as in (2)"; note the feasible counterpart (3) and its bootstrap analogue both weight by `delta^{(0)}_{i,x}`, i.e. untreated observations - the `D_t = 1` in `F_tilde_{Y_t(0)|X=x,D_t=1}` names the distribution being estimated, not the observations used.)

| Result | Page | Establishes |
|---|---|---|
| **Lemma 1** | p. 9 | Functional CLT: `(G_tilde_{t,x}^{(0)}, G_hat_{t-1,x}^{(0)}, G_hat_{t,x}^{(1)}, G_hat_{t-1,x}^{(1)}) ⇝ (V_x^{(0)}, W_x^{(0)}, V_x^{(1)}, W_x^{(1)})` in `S_x := l^inf(Y_{t|x,1}(0)) x l^inf(Y_{t-1|x,0}) x l^inf(Y_{t|x,1}) x l^inf(Y_{t-1|x,1})`, a tight zero-mean Gaussian process with covariance kernel `diag{Sigma_x^{(0)}, Sigma_x^{(1)}}` (under A1-A6, i.e. A1-A4 + A6). Proof (p. 26): functional CLT for empirical distribution functions, van der Vaart and Wellner (1996), Ch. 2 |
| **Proposition 1** | p. 10 | Joint limit of the estimated potential-outcome distributions: `(Z_hat_x^{(0)}, Z_hat_x^{(1)}) ⇝ (Z_x^{(0)}, Z_x^{(1)})` where `Z_hat_x^{(j)}(y) := sqrt(n)(F_hat_{Y_t(j)|X=x,D_t=1}(y) - F_{Y_t(j)|X=x,D_t=1}(y))`; `Z_x^{(0)} := r_x^{(0)} V_x^{(0)} + kappa_x(W_x^{(0)}, W_x^{(1)})` and `Z_x^{(1)} = r_x^{(1)} V_x^{(1)}` - the extra `kappa_x` term reflects first-step estimation error, so the limit is NOT nuisance-parameter free |
| **Theorem 2** | pp. 10-11 | Functional CLT for the CQTT process: `sqrt(n)(Delta_hat_x^QTT(tau) - Delta_x^QTT(tau)) ⇝ Z_bar_x^{(1)}(tau) - Z_bar_x^{(0)}(tau)` in `(l^inf(T))^2`, with `Z_bar_x^{(j)}(tau) := Z_x^{(j)}(F^{-1}_{Y_t(j)|X=x,D_t=1}(tau)) / f_{Y_t(j)|X=x,D_t=1}(F^{-1}_{Y_t(j)|X=x,D_t=1}(tau))` - parametric sqrt(n) rate, via the functional delta method |
| **Corollary 2** | p. 11 | Under `H_0: Delta_x^QTT(tau) = 0` for all `tau ∈ T`: `KS_x ->_d sup_{tau∈T} |Z_bar_x^{(1)}(tau) - Z_bar_x^{(0)}(tau)|`; basis for uniform confidence bands. Proof (p. 27): continuous mapping theorem (Kosorok 2007, Sec. 2.1) |
| **Lemma 2** | p. 13 | Bootstrap empirical processes consistently estimate the Lemma 1 limits (conditional weak convergence in probability, `⇝_p`), under A1-A6 + B. Proof (p. 28): Theorem 3.6.13 of van der Vaart and Wellner (1996) |
| **Theorem 3** | p. 13 | First-order validity of the exchangeable bootstrap: `(Z_hat_x^{(0)*}, Z_hat_x^{(1)*}) ⇝_p (Z_x^{(0)}, Z_x^{(1)})` and `sqrt(n)(Delta_hat_x^{QTT*}(tau) - Delta_hat_x^QTT(tau)) ⇝_p Z_bar_x^{(1)}(tau) - Z_bar_x^{(0)}(tau)`, `tau ∈ T` |

Key auxiliary objects in Proposition 1 (printed p. 10):

```
kappa_x(W_0, W_1) := INTEGRAL { r_x^{(0)} W_0(v) - r_x^{(1)} W_1 ∘ F^{-1}_{Y_{t-1}|X=x,D_t=1} ∘ F_{Y_{t-1}|X=x,D_t=0}(v) } omega_x(y, v) dv,

omega_x(y, v) := f_{DeltaY_t,Y_{t-1}|X=x,D_t=0}( y - F^{-1}_{Y_{t-1}|X=x,D_t=1} ∘ F_{Y_{t-1}|X=x,D_t=0}(v), v )
                 / f_{Y_{t-1}|X=x,D_t=1} ∘ F^{-1}_{Y_{t-1}|X=x,D_t=1} ∘ F_{Y_{t-1}|X=x,D_t=0}(v),

for (y, v) ∈ Y_{t|x,1}(0) x Y_{t-1|x,0}.
```

Proof structure (Appendix, printed pp. 23-27), useful for any influence-function or analytical-SE work later:
- **Lemma A1** (p. 23): the quantile-quantile map `psi(F) = G^{-1} ∘ H` for `F = (G,H)` (E = distribution functions with strictly positive, bounded density) is Hadamard differentiable with derivative `psi'_{F_0}(gamma) = (gamma_2 - gamma_1 ∘ G_0^{-1} ∘ H_0) / (g_0 ∘ G_0^{-1} ∘ H_0)`. Implementation relevance: the transform is only differentiable where the base density is strictly positive and bounded - degenerate/atomic outcome distributions violate the framework.
- **Lemma A2** (p. 23): the plug-in map `phi(F)(w) := Pr{V_1 + G^{-1} ∘ H(V_2) <= w}` is Hadamard differentiable with derivative `phi'_{F_0}(gamma)(w) = INTEGRAL (gamma_2(v_2) - gamma_1 ∘ G_0^{-1} ∘ H_0(v_2)) * f_{V1|V2}(w - G_0^{-1} ∘ H_0(v_2) | v_2) / (g_0 ∘ G_0^{-1} ∘ H_0(v_2)) dv_2`; requires `f_{V1|V2}` uniformly continuous and uniformly bounded.
- **Lemma A3** (p. 25): stochastic equicontinuity `sup_{w∈W} |nu_n(F_n) - nu_n(F_0)|(w) = o_p(1)` for `nu_n(F) := sqrt(n)(phi_n(F) - phi(F))`, along the line of Theorem 2.3 of van der Vaart and Wellner (2007) (empirical processes indexed by estimated functions).
- **Lemma A4** (p. 26): asymptotically linear expansion `sqrt(n)(phi_n(F_n) - phi(F_0)) = nu_n(F_0) + phi'_{F_0}(sqrt(n)(F_n - F_0)) + o_p(1)` uniformly in W.
- **Proof of Proposition 1** (p. 27): `sqrt(n)(F_hat_{Y_t(0)|X=x,D_t=1} - F_{Y_t(0)|X=x,D_t=1}) = r_x^{(0)} G_tilde_{t,x}^{(0)} + kappa_x(G_hat_{t-1,x}^{(0)}, G_hat_{t-1,x}^{(1)}) + o_p(1)` uniformly, where `kappa_x` is the Hadamard derivative of `phi_n` from Lemma A2; concluded via the extended continuous mapping theorem with Lemma 1.
- **Proof of Theorem 2** (p. 27): when `F_hat_{Y_t(j)|X=x,D_t=1}(y)` is weakly increasing in y, the quantile map is Hadamard differentiable; the functional delta method gives `sqrt(n)(F_hat^{-1}_{Y_t(j)|X=x,D_t=1}(tau) - F^{-1}_{Y_t(j)|X=x,D_t=1}(tau)) ⇝ (Z_x^{(j)} / f_{Y_t(j)|X=x,D_t=1}) ∘ F^{-1}_{Y_t(j)|X=x,D_t=1}(tau)`. Quantile inference requires positive density at the relevant quantiles.
- Regularity used throughout (pp. 22, 24): compact supports; densities bounded away from 0 and infinity; `V_tilde := V_1 + G_0^{-1} ∘ H_0(V_2)` has compact support with continuous density bounded away from 0 and infinity.

Extensions noted (printed p. 11): via Proposition 1 + delta method one could obtain limit processes for other Hadamard-differentiable functionals (Lorenz curve, Gini coefficient).

**Bootstrap inference (Section 3.3, printed pp. 11-13; proofs printed pp. 27-28):**

Bootstrap analogues (unnumbered displays, printed p. 12), with exchangeable weights `W_i^{(d)}` satisfying Assumption B:

```
F*_{Y_s|X=x,D_t=d}(y) := (1 / n_x^{(d)}) SUM_i W_i^{(d)} 1{Y_is <= y} delta^{(d)}_{i,x},

F*_{Y_t(0)|X=x,D_t=1}(y)
  := (1 / n_x^{(0)}) SUM_i W_i^{(0)} 1{ DeltaY_it + F*^{-1}_{Y_{t-1}|X=x,D_t=1} ∘ F*_{Y_{t-1}|X=x,D_t=0}(Y_{i,t-1}) <= y } delta^{(0)}_{i,x},

Delta_hat_x^{QTT*}(tau) := F*^{-1}_{Y_t(1)|X=x,D_t=1}(tau) - F*^{-1}_{Y_t(0)|X=x,D_t=1}(tau),   tau ∈ T.
```

- The empirical bootstrap is the special case where `(W_1^{(d)}, ..., W_n^{(d)})` is multinomial with probabilities `delta^{(d)}_{i,x} * (1/n_x^{(d)}, ..., 1/n_x^{(d)})` (printed p. 12). The framework (Praestgaard and Wellner 1993) also covers weighted bootstrap, wild bootstrap, and subsampling. Paper's recommendations: empirical bootstrap for moderate samples, weighted bootstrap if estimation is time-consuming, subsampling if the sample is extremely large.
- Validity (Theorem 3, proof printed p. 28): linearized bootstrap process `Z_tilde_x^{(0)*} := r_x^{(0)} G_tilde_{t,x}^{(0)*} + kappa_x(G_hat*_{t-1,x}, G_hat*_{t,x})`; triangle-inequality split over the bounded-Lipschitz class `BL_1` (displays (A12)-(A13)), with (A12) -> 0 in probability via (A14)-(A16), Lemma 2, Lemma A5 (the bootstrap analogue of Lemma A4, displays (A10)-(A11)), and Markov; (A13) -> 0 via Lemma 2 + continuous mapping. Final step: "Theorem 3.9.11 of van der Vaart and Wellner (1996) shows that the functional delta method can apply for Hadamard differentiable maps under resampling. Since the map from distribution to quantile is Hadamard differentiable, the desired result follows."
- **KS test and uniform bands** (printed p. 11): `KS_x := sqrt(n) sup_{tau∈T} |Delta_hat_x^QTT(tau)|`; simultaneous (1-alpha)% confidence bands `(Delta_hat_x^QTT - c_{1-alpha} n^{-1/2}, Delta_hat_x^QTT + c_{1-alpha} n^{-1/2})` with `c_{1-alpha}` the KS critical value, obtained in practice by bootstrap. "The confidence bands are obtained by inverting the Kolmogorov-Smirnov test" (printed p. 16).
- **Iterations used everywhere: 1000** (Monte Carlo rejection frequencies, application KS critical values, pointwise SEs, and Figure 1 bands all use 1000 bootstrap iterations; printed pp. 14, 16).

**Monte Carlo evidence (Section 4, printed pp. 14-15; Tables 1-2, printed pp. 29-30):**

DGP: `Y_it(d) = mu(d) + theta_t + v_i + eps_it`, treatment effect constant across all quantiles `= mu(1) - mu(0)` set to 0 or 1 ("TE"); `theta_t = 1` (common time fixed effect); `v_i` time-invariant heterogeneity that may differ across treated/untreated; `eps_it` time-varying unobservables. 1000 Monte Carlo simulations, 1000 bootstrap iterations each, nominal size 5%.

*DGP 1 (both this paper's model and Athey-Imbens CiC hold):* `v_i | D=d ~ N(d, 1)`; `eps_it` standard normal. CiC is the benchmark, with CiC standard errors from the "empirical block bootstrap" on the same sample (1000 iterations; no software attribution). Table 1 (bias and rejection probabilities at quantiles 0.1/0.5/0.9; DDID = this paper's estimator):

| | N | DDID 0.1 | DDID 0.5 | DDID 0.9 | CIC 0.1 | CIC 0.5 | CIC 0.9 |
|---|---|---|---|---|---|---|---|
| TE=0 Bias | 100 | 0.044 | 0.045 | 0.081 | 0.012 | -0.097 | -0.295 |
| | 200 | 0.016 | 0.021 | 0.066 | -0.009 | -0.048 | -0.141 |
| | 500 | 0.016 | 0.008 | 0.023 | -0.005 | -0.042 | -0.074 |
| TE=0 Rej. Prob. | 100 | 0.042 | 0.037 | 0.023 | 0.042 | 0.041 | 0.100 |
| | 200 | 0.049 | 0.050 | 0.044 | 0.051 | 0.056 | 0.076 |
| | 500 | 0.043 | 0.047 | 0.034 | 0.035 | 0.043 | 0.069 |
| TE=1 Bias | 100 | 0.059 | 0.064 | 0.109 | 0.251 | -0.051 | -0.293 |
| | 200 | 0.031 | 0.027 | 0.049 | 0.128 | -0.052 | -0.200 |
| | 500 | 0.014 | 0.019 | 0.025 | 0.053 | -0.019 | -0.090 |
| TE=1 Rej. Prob. | 100 | 0.397 | 0.675 | 0.359 | 0.409 | 0.617 | 0.548 |
| | 200 | 0.742 | 0.949 | 0.703 | 0.713 | 0.859 | 0.614 |
| | 500 | 0.994 | 1.000 | 0.992 | 0.983 | 0.992 | 0.756 |

Paper's reading (printed p. 14): DDID is less biased than CiC in finite samples, "especially at the 0.9th quantile"; with only 100 observations DDID inference is "somewhat undersized, but it exhibits good size properties with 200 or 500 observations"; power increases rapidly from 100 to 200 to 500 and is higher at the median than at the 0.1/0.9 quantiles. Note CiC's TE=0 rejection at the 0.9 quantile is oversized (0.100 at N=100, still 0.069 at N=500).

*DGP 2 (Copula Invariance violated while Distributional DiD holds):*

```
(v_i, eps_i2, eps_i1) | D=d ~ N(0, V_d),   V_d = [ 1        rho_dv2   rho_dv1
                                                   rho_dv2  1         rho_d12
                                                   rho_dv1  rho_d12   1       ]
```

`(Y_i1(0), DeltaY_i2(0) | D=d)` is bivariate normal with correlation parameter `rho_d2 - rho_d1 + rho_d12 - 1` (consistent with the DGP algebra: Cov(v_i + eps_i1, eps_i2 - eps_i1), writing rho_d1 / rho_d2 for the matrix entries rho_dv1 / rho_dv2); the copula is Gaussian. They set `rho_d1 = 0` and `rho_d12 = 1/2` for both d; then `rho_d2 = d * rho_bar` and vary `rho_bar`. `rho_bar = 0` means Copula Invariance holds; `rho_bar != 0` violates it. N = 200 throughout. Table 2 (bias and RMSE):

| rho_bar | TE=0 0.1 | TE=0 0.5 | TE=0 0.9 | TE=1 0.1 | TE=1 0.5 | TE=1 0.9 |
|---|---|---|---|---|---|---|
| Bias 0.00 | 0.020 | 0.034 | 0.037 | 0.023 | 0.023 | 0.050 |
| Bias 0.05 | 0.073 | 0.023 | 0.012 | 0.088 | 0.029 | 0.008 |
| Bias 0.10 | 0.121 | 0.028 | -0.033 | 0.112 | 0.019 | -0.032 |
| Bias 0.50 | 0.425 | 0.013 | -0.374 | 0.435 | 0.027 | -0.353 |
| RMSE 0.00 | 0.348 | 0.261 | 0.340 | 0.342 | 0.248 | 0.359 |
| RMSE 0.05 | 0.348 | 0.256 | 0.324 | 0.358 | 0.258 | 0.342 |
| RMSE 0.10 | 0.374 | 0.260 | 0.352 | 0.374 | 0.259 | 0.346 |
| RMSE 0.50 | 0.565 | 0.272 | 0.529 | 0.566 | 0.264 | 0.508 |

Paper's reading (printed p. 15): small violations (`rho_bar` = 0.05 or 0.10) cause small bias increases - at the 0.1 quantile bias goes 0.020 -> 0.073 -> 0.121; a large violation (`rho_bar` = 0.50) causes much larger tail bias (0.425 at the 0.1 quantile; symmetric negative bias -0.374 at the 0.9 quantile). **The median (0.5 quantile) is almost completely insensitive** to Copula Invariance violations: bias 0.013 even at `rho_bar` = 0.5, RMSE barely changes. Implementation takeaway: tail quantiles are where copula-invariance failure bites; the median is robust.

**Empirical application (Section 5, printed pp. 15-17; Tables 3-4 / Figure 1, printed pp. 31-33) - context and potential replication target:**

Effect of Q1 2007 state minimum-wage increases on the earnings distribution. Treated: 5 states raising their minimum wage with close geographic proximity to a federal-minimum state (Arizona, Colorado, Minnesota, Missouri, North Carolina); control: 14 federal-minimum states (Georgia, Idaho, Iowa, Kansas, Kentucky, Nebraska, New Mexico, North Dakota, South Carolina, South Dakota, Tennessee, Utah, Virginia, Wyoming). Data: CPS (Flood et al. 2015), earnings asked in the 4th and 8th interview months - exactly one year apart; IPUMS longitudinal links (Drew et al. 2014); earnings > $10/week; 8,256 linked individuals (2 observations each). Covariates: 8 discrete cells by gender x race (white/non-white) x education (college/non-college); Table 3 motivates conditioning (treated states have higher college share, p = 0.00, and 6.5 log points higher 2006 earnings, p = 0.00 - cross-sectional comparisons would be upward-biased). Per subgroup: KS test of CQTT = 0 at all quantiles, grid tau = 0.05 to 0.95 by 0.01, critical values from 1000 bootstrap iterations, 5% size. Rejections in 3 of 8 subgroups (white female college; non-white male non-college; non-white female non-college) - lower-earning groups, consistent with the minimum wage binding only at the bottom. Table 4 (printed p. 32), CQTT with pointwise bootstrap SEs:

| Race | Gender | Education | N | Reject H0 | 0.1 | 0.5 | 0.9 |
|---|---|---|---|---|---|---|---|
| White | Male | College | 1617 | | 0.004 (0.048) | 0.007 (0.031) | 0.000 (0.037) |
| White | Male | Non-College | 2306 | | -0.021 (0.068) | -0.019 (0.028) | 0.075 (0.068) |
| White | Female | College | 1629 | Yes | -0.029 (0.056) | 0.027 (0.033) | -0.046 (0.052) |
| White | Female | Non-College | 1980 | | -0.038 (0.054) | -0.043 (0.035) | -0.086 (0.068) |
| Non-White | Male | College | 156 | | -0.492 (0.265) | -0.07 (0.161) | 0.292 (0.268) |
| Non-White | Male | Non-College | 282 | Yes | -0.087 (0.186) | -0.044 (0.095) | 0.036 (0.136) |
| Non-White | Female | College | 209 | | 0.097 (0.172) | 0.033 (0.088) | 0.007 (0.121) |
| Non-White | Female | Non-College | 340 | Yes | -0.25 (0.175) | -0.026 (0.086) | -0.043 (0.162) |

Significant effects are concentrated in the lower part of the distribution and negative in sign - explained by earnings mixing wages and hours (hours may fall even if wages rise); footnote 1 (printed p. 17) robustness check: the 10th percentile of earnings rose 3.0 log points for treated vs 10.5 for untreated between 2006 and 2007.

**Relation to other methods:**

- **Athey-Imbens CiC** (printed p. 2): a distinct identification strategy for the same distributional target. CiC posits a structural monotone production function in a scalar unobservable; this paper keeps the DiD flavor (Distributional DiD on the *change* in untreated potential outcomes) and adds Copula Invariance. The compact-support condition of A4 is borrowed "as in Athey and Imbens (2006)". Melly and Santangelo (2015) extend CiC to covariate-conditional assumptions. In Monte Carlo DGP 1, where both models hold, DDID is less biased in finite samples, especially at the 0.9 quantile.
- **Mean DiD / parallel trends** (printed p. 5): A2 is the exact distributional strengthening of parallel trends. Under A2 alone the ATT is identified but the CQTT is only partially identified.
- **Fan and Yu (2012)** (printed pp. 2, 5): partial identification of the QTT via Frechet-Hoeffding copula bounds under Distributional DiD. Copula Invariance is exactly what upgrades that partial identification to point identification with two periods.
- **Callaway and Li (2015)** (printed p. 2): point-identifies the QTT with at least three periods of panel data by replacing an unknown unconditional copula in the last two periods with an observed copula from the first two periods. This paper needs only two periods, works conditionally on covariates, and recovers the missing dependence cross-sectionally (treated vs untreated group) rather than across time.
- **Broader QTE literature** (printed p. 2): conditional QTE via quantile regression (Koenker-Bassett 1978; Koenker 2005); IV-based conditional QTE (Abadie et al. 2002; Chernozhukov-Hansen 2005, 2006, 2013); unconditional QTE via propensity-score weighting (Firpo 2007); IV after conditioning (Abadie 2003; Frolich-Melly 2013); counterfactual-distribution methods (Firpo et al. 2009; Rothe 2012; Chernozhukov et al. 2013a); nonseparable panel models under time homogeneity (Chernozhukov et al. 2013b); repeated cross sections (D'Haultfoeuille et al. 2015). Note the paper does NOT use the quantile-DiD transform (quantile-by-quantile differencing of observed distributions) anywhere; its counterfactual comes from Theorem 1's copula argument.
- **Positioning** (Conclusion, printed p. 17): "the key innovation is to recover the unknown dependence between the change and initial level of untreated potential outcomes for the treated group from the observed dependence from the untreated group." The authors flag replacing an unknown copula with one observed for another group as a reusable strategy (finance, auction models, duration models).

**Reference implementation(s):**
- R: `qte` package by Brantly Callaway (first author of this paper) - the initiative's chosen parity target. Per the qte documentation the two-period panel QTT estimator of this paper is `qte::ddid2()`; **verify this function-name mapping against the qte docs/source during implementation** - it is asserted from package-documentation recall, not from the paper.
- The paper itself mentions no software, package, or replication code anywhere (verified across printed pp. 1-33; the References list contains only academic works, and the Monte Carlo's CiC benchmark carries no software attribution).

**Requirements checklist (for the future scoped PR, not v1):**
- [ ] Verify `qte::ddid2()` is the function implementing this paper's two-period panel QTT (qte docs/source), then build golden parity fixtures against it
- [ ] Reconcile all equation/assumption numbers against the published JoE 206(2) version of record
- [ ] Resolve the Corollary 1 conditioning flag and the (A4) mapping-direction discrepancy (see Gaps) before writing any repeated-cross-section code
- [ ] Empirical CDFs per eq. (2); plug-in counterfactual per eq. (3); inf-based quantile inversion (no interpolating quantile types)
- [ ] Exchangeable bootstrap with weights satisfying Assumption B; empirical bootstrap as default; 1000 iterations as the paper's convention
- [ ] KS statistic `KS_x` and uniform bands by inverting the KS test; tau grid restricted to a compact `T` strictly inside (0,1)
- [ ] Discrete-covariate cells: common support of X across groups; warn on tiny cells (application shows cells as small as N = 156; A6(b) requires proportional growth)
- [ ] Pre-treatment placebo hook: with extra pre-periods, run the estimator on pre-periods and test `Delta_x^QTT(tau) = 0` for all tau (the paper's suggested A2/A3 diagnostic)
- [ ] Continuity/ties check (A4): warn on heavy ties or atoms; copulas non-unique and Hadamard machinery fails on degenerate distributions
- [ ] Document tail-quantile sensitivity to Copula Invariance violations (DGP 2: 0.1-quantile bias 0.020 -> 0.425 as rho_bar goes 0 -> 0.5) and median robustness
- [ ] Panel mode operates on `(DeltaY_it, Y_{i,t-1})` pairs; repeated cross-section mode requires the extra rank-invariance assumption and time-invariant covariates, with a distinct warning

---

## Implementation Notes

**Relevance to diff-diff CiC/QDiD v1 (2026-07-12):**
- (i) The exchangeable/empirical bootstrap machinery (Assumption B, functional delta method, Theorem 3 via van der Vaart-Wellner Theorems 3.6.13 and 3.9.11) is the closest published validity argument for the bootstrap inference diff-diff v1 ships with CiC/QDiD-style plug-in distributional estimators - Athey-Imbens (2006) itself contains no bootstrap theory at all.
- (ii) The Copula Invariance assumption clarifies exactly what panel data adds beyond repeated cross-sections for QTT: the observed within-unit pairing of (change, initial level) among untreated units. CiC itself does not need it - Athey-Imbens Section 3.4 requires only equal within-group marginals of the unobservable over time for panel data.
- (iii) The Monte Carlo DDID-vs-CiC comparison (Table 1) informs our docs on when to recommend which estimator: when both models hold, DDID is less biased in the tails (especially the 0.9 quantile) and CiC's upper-tail test over-rejects; DGP 2 shows where DDID itself breaks (tail bias under copula-invariance violations, median robust).
- (iv) The paper's own comparison of its assumptions vs Athey-Imbens CiC (distributional parallel trends + copula invariance on observables' building blocks, vs a monotone structural function in a scalar unobservable with time-invariant within-group distribution) belongs in the choosing-an-estimator docs.

### Data Structure Requirements
- Two-period panel: one pair `(Y_{i,t-1}, Y_it)` per unit plus covariates and a period-t treatment indicator; nobody treated in the first period. The estimator consumes `(DeltaY_it, Y_{i,t-1})` for untreated units and `Y_{i,t-1}`, `Y_it` marginals for treated units, all within covariate cells.
- Repeated cross sections: supported only via Corollary 1, at the cost of conditional rank invariance over time and time-invariant covariates.
- Covariates must be discrete (cells); the theory is fully nonparametric within cells with no curse of dimensionality (printed pp. 3, 8). Both groups must be represented in every cell used (X is the common support), and both cell sizes must grow proportionally with n (A6(b)).
- Continuous outcomes (A4): ties/atoms break copula uniqueness and the Hadamard-differentiability framework.

### Computational Considerations
- Pure ECDF plug-in: sorting the per-cell arrays dominates, O(n log n); the quantile-quantile transform `F_hat^{-1}_{Y_{t-1}|X=x,D_t=1} ∘ F_hat_{Y_{t-1}|X=x,D_t=0}` evaluates via two searchsorted passes over sorted arrays. No kernels, no bandwidths, no density estimation for point estimates or bootstrap inference.
- The copula is never estimated explicitly - Theorem 1's formula handles it implicitly through the observed pairing of each untreated unit's own change and initial level. Any implementation that separately estimates a copula object is over-engineering the estimator.
- Bootstrap multiplies cost by B (paper convention B = 1000); replicates are embarrassingly parallel. The exchangeable-weights formulation means the empirical bootstrap can be implemented as within-cell multinomial weights rather than physical resampling.
- Densities (`f_{Y_t(j)|X=x,D_t=1}`, the omega_x kernel) appear ONLY in the asymptotic limit expressions, not in the estimator or the bootstrap - inference never requires estimating them. This is a practical advantage over the Athey-Imbens analytical-SE path, which needs a boundary-consistent density estimator.
- The Proposition 1 limit is not nuisance-parameter free (the kappa_x first-step term), which is exactly why the paper routes inference through the bootstrap rather than plug-in analytical SEs.

### Tuning Parameters

| Parameter | Type | Default | Selection Method |
|-----------|------|---------|-----------------|
| `n_bootstrap` | int | 1000 (paper's convention in both simulation and application) | SE/band stability; align with qte for parity tests |
| bootstrap weight scheme | {empirical, weighted, wild, subsampling} | empirical (paper recommendation for moderate samples) | weighted if estimation is time-consuming; subsampling if the sample is extremely large (printed p. 12) |
| `quantiles` (tau grid) | floats in compact `T ⊂ (0,1)` | 0.05 to 0.95 by 0.01 (paper's application grid, printed p. 16) | T must be compact and strictly inside (0,1) (printed p. 9); no extreme quantiles |
| covariate cells | discrete values of X | all common-support cells | both groups present per cell; warn on tiny cells (A6(b)) |

No smoothing parameters exist anywhere in the method.

### Relation to Existing diff-diff Estimators
- `ChangesInChanges` / `QDiD` (v1): same 2x2-flavored distributional target (counterfactual `F_{Y_t(0)}` for the treated), different identification. CiC assumes a monotone structural function and within-group time-invariance of a scalar unobservable; this paper assumes Distributional DiD (A2) + Copula Invariance (A3) directly on the building blocks of the observed outcomes. Neither nests the other (A2/A3 are non-nested with the CiC model; DGP 1 is a case where both hold). The choosing-an-estimator docs should carry the Table 1 and Table 2 evidence.
- `DifferenceInDifferences`: mean DiD's parallel trends is the mean implication of A2; under A2 alone the ATT is identified (so the ATT from this method should match mean DiD asymptotically when A2 holds) while the CQTT needs A3 on top.
- `CallawaySantAnna`: same first author; the CS (2021) group-time ATT framework is the mean-DiD multi-period sibling. Callaway and Li (2015) is the three-period panel QTT sibling of this paper (unconditional copula recovered across time instead of across groups).
- Bootstrap utilities: the exchangeable-bootstrap validity argument here (functional delta method + Hadamard differentiability of the ECDF-composition and quantile maps) is the citation to use when documenting bootstrap inference for the v1 distributional estimators; reuse `safe_inference()` joint-NaN conventions from `diff_diff.utils` for any future implementation.

---

## Gaps and Uncertainties

**Contradictions between extraction passes (both versions preserved; do NOT resolve by guessing - check the arXiv PDF and the published JoE version):**

1. **Corollary 1 rank-invariance conditioning (printed pp. 7 and 22).** The Corollary 1 *statement* (printed p. 7) conditions the rank-invariance copula on the treated group: `C_{Y_{t-1}(0),Y_t(0)|X=x,D_t=1}(u, v) = min{u, v}`. But the identification formula recovers the imputed change `DeltaY_tilde_it` from *untreated* observables, and the appendix *proof* (printed p. 22) applies rank invariance to untreated units: `F_{Y_t(0)|X=x,D_t=0}(Y_it(0)) = F_{Y_{t-1}(0)|X=x,D_t=0}(Y_{i,t-1}(0))`, i.e. it is used to "identify `DeltaY_it(0)` for individuals with `X_i = x` and `D_it = 0`". Either the statement's `D_t = 1` is a typo for `D_t = 0`, or the proof invokes an unstated symmetric condition. Resolve against the published JoE version before implementing the repeated-cross-section path.

2. **Direction of the quantile-quantile transform in the generic appendix map (printed pp. 7-8 vs p. 24).** The main-text extraction (Theorem 1, printed p. 7; eq. (3), printed p. 8) has the transform `F^{-1}_{Y_{t-1}|X=x,D_t=1} ∘ F_{Y_{t-1}|X=x,D_t=0}` applied to *untreated* units' initial levels (transporting untreated initial levels onto the treated scale) - internally consistent with eq. (3) weighting by `delta^{(0)}_{i,x}`. The appendix extraction's prose mapping of display (A4) (printed p. 24) instead describes `G^{-1} ∘ H` as `F^{-1}_{Y_{t-1}|X=x,D_t=0} ∘ F_{Y_{t-1}|X=x,D_t=1}`, "transporting treated-group initial levels onto the untreated scale". The generic display (A4) itself is not in question; the conflict is only in which paper object G and which H denotes. This review's main body follows the main-text version throughout (it is the self-consistent one), but verify the appendix mapping against printed p. 24 before reusing the Lemma A1/A2 derivative formulas with concrete distributions substituted in.

**Verified quirks (not contradictions, but easy to trip over):**

3. **No Assumption A5 exists.** The main text has A1, A2, A3, A4, A6, B; "A5" appears only as appendix equation (A5) and appendix Lemma A5. Citations reading "Assumption A1-A6" (Lemma 1, Proposition 1, Theorem 2, Lemma 2, Theorem 3) mean A1-A4 plus A6. Verified by full-text search during extraction.
4. **`F_tilde` notation on printed p. 9.** The v1 text describes the infeasible `F_tilde_{Y_t(0)|X=x,D_t=1}` as "based on observations {Y_tilde_it} with X_i = x and D_it = 1 as in (2)", but the feasible eq. (3) and its bootstrap analogue weight by `delta^{(0)}_{i,x}` (untreated observations). The `D_t = 1` subscript names the distribution being *estimated*, not the observations used.
5. **DGP 2 "correlation parameter"** `rho_d2 - rho_d1 + rho_d12 - 1` was transcribed identically by both extraction passes; it is the *covariance* Cov(v_i + eps_i1, eps_i2 - eps_i1) of `(Y_i1(0), DeltaY_i2(0))` under the stated V_d, and the text's rho_d1/rho_d2 correspond to the matrix entries rho_dv1/rho_dv2. Consistent, but the naming is loose - transcribed as printed.

**Version and coverage gaps:**

6. **arXiv v1 vs published version of record.** All equation, theorem, and assumption numbers above are pinned to arXiv:1702.03618v1. The published JoE 206(2) version may renumber or reword (in particular the Corollary 1 conditioning in item 1); reconcile before implementing anything sensitive.
7. **Unconditional QTT aggregation is informal.** The paper identifies the unconditional QTT by averaging the conditional counterfactual over covariates and inverting (printed p. 7) but then "focus[es] only on the CQTT" - there is no formal aggregation estimator or inference theory in the paper. Any aggregated-QTT implementation needs its own delta-method argument (or the qte package's approach, whichever it uses - check during implementation).
8. **Tables sit after the References** (printed pp. 29-33, References pp. 18-21). The in-text quantitative claims (printed pp. 14-15) were cross-checked against the table transcriptions during synthesis and are consistent.
9. **No reference implementation named by the paper.** The `qte::ddid2()` mapping comes from qte package documentation recall, not from the paper; verify against qte docs/source. The Monte Carlo's CiC benchmark ("empirical block bootstrap") carries no software attribution either.
10. **Proof details relied on but not independently checked:** Lemma A3's displays (A5)-(A9) (equicontinuity envelopes), Lemma A5's (A10)-(A11), and Theorem 3's (A12)-(A16) were extracted as structure only; the epsilon-delta detail was not verified line-by-line. Lemma 2's proof is explicitly abbreviated in the paper ("follows from Theorem 3.6.13 of van der Vaart and Wellner (1996)").
