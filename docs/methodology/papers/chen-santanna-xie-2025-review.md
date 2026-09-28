# Paper Review: Efficient Difference-in-Differences and Event Study Estimators

**Authors:** Xiaohong Chen (Yale), Pedro H. C. Sant'Anna (Emory), Haitian Xie (Peking University)
**Citation:** Chen, X., Sant'Anna, P. H. C., & Xie, H. (2025). *Efficient Difference-in-Differences and Event Study Estimators.* arXiv:2506.17729v1 [econ.EM]. Also circulated as Cowles Foundation Discussion Paper No. 2470.
**PDF reviewed:** **arXiv:2506.17729v1** (https://arxiv.org/abs/2506.17729v1), submitted 21 Jun 2025, 72 pages. Per the project's PDFs-never-committed convention the local PDF is kept outside the repository; the arXiv v1 page is the authoritative source. **All equation, theorem, corollary, lemma, and remark numbers below are pinned to v1** (it is the only arXiv version). The Cowles Foundation Discussion Paper No. 2470 (PDF: https://cowles.yale.edu/sites/default/files/2025-10/d2470.pdf) is an identical-content mirror (73 pp vs the arXiv 72 pp — same content, differing front-matter/formatting pagination).
**Review date:** 2026-05-31

---

## Methodology Registry Entry

*This file is the canonical **in-repo scholarly paper review** for arXiv:2506.17729v1 and the primary **review artifact** behind the library's `EfficientDiD` class (the **primary source** is the paper itself; `docs/methodology/REGISTRY.md` `## EfficientDiD` is the implementation contract). The authoritative methodology contract remains the existing `## EfficientDiD` section in `docs/methodology/REGISTRY.md`, which already cites the paper's numbered results densely; this review documents the paper foundations behind that contract and is the **Step-1 fidelity baseline** for the source-validation pass (PR-B). PR-B will reconcile any paper-vs-REGISTRY/code discrepancies, add `tests/test_methodology_efficient_did.py` (Theorem 3.1/3.2 / Eq 3.5 / Eq 4.3 numbered Verified Components), build the HRS Table 6 cross-language anchor, document the deviations, and flip the `METHODOLOGY_REVIEW.md` row. This review records paper facts only and makes **no** code-deviation verdicts.*

## EfficientDiD

**Primary source:** [Chen, X., Sant'Anna, P. H. C., & Xie, H. (2025). *Efficient Difference-in-Differences and Event Study Estimators*, arXiv:2506.17729v1.](https://arxiv.org/abs/2506.17729v1)

**Central thesis (Abstract; §1):** Modern heterogeneity-robust DiD/event-study estimators in short panels typically weight all pre-treatment periods and comparison groups equally (or discard all but one), which is generally inefficient. The paper (a) gives an observationally equivalent characterization of the DiD potential-outcome model as **sequential conditional moment restrictions on observables** (in the sense of Ai & Chen 2012), showing DiD models are typically **nonparametrically overidentified** (Chen & Santos 2018); (b) derives the **semiparametric efficient influence function (EIF)** in closed form for `ATT(g,t)` and event-study parameters under commonly imposed parallel-trends assumptions; and (c) proposes simple-to-compute **closed-form efficient estimators** attaining the bound. The central practical message: semiparametric efficiency requires **non-uniform** weighting of pre-treatment periods and untreated cohorts, with weights proportional to the inverse of (conditional) covariances of outcome changes.

**Scope of what the library implements:** the DiD and event-study efficient estimators of §3–§4 (both the no-covariate closed form and the covariate doubly-robust path), plus the Hausman PT-All-vs-PT-Post pretest of Appendix A. The instrumented-DiD extension (Appendix B) is **not** in scope for `EfficientDiD`.

**Key implementation requirements:**

*Setup and notation (§2, pp. 4–5):*
- `T` periods, `t ∈ {1,…,T}`. Binary, **absorbing** treatment (`D_{i,t}=1 ∀ t ≥ G_i`). Cohort/group `G_i = min{t : D_{i,t}=1}`; never-treated set `G_i = ∞`. `G` = support of `G`; `G_trt = G\{∞}` = eventually-treated support. WLOG a never-treated group exists (if all are eventually treated, drop data from when the last cohort is treated, so the last-treated cohort becomes "never-treated").
- Large-`n`, fixed-`T` short panel. Pre-treatment covariates `X_i ∈ X ⊆ R^d`. Neyman–Rubin potential outcomes indexed by the treatment path: `Y_{i,t}(g) = Y_{i,t}(0_{g-1}, 1_{T-g+1})`, never-treated `Y_{i,t}(∞)`. Observed `Y_{i,t} = Σ_g 1{G_i=g} Y_{i,t}(g)`. `G_g := 1{G=g}`.

*Identifying assumptions (§2, pp. 5–6) — quoted in substance:*
- **Assumption S (Random Sampling):** `{(Y_{i,1},…,Y_{i,T}, X_i, G_i)}_{i=1}^n` is a random sample.
- **Assumption O (Overlap):** for each `g ∈ G`, `E[G_g | X] ∈ (0,1)` a.s.
- **Assumption NA (No-anticipation):** for every `g ∈ G_trt` and every pre-treatment `t < g`, `E[Y_{i,t}(g) | G=g, X] = E[Y_{i,t}(∞) | G=g, X]` a.s.
- **Assumption PT-Post (Parallel Trends in post-treatment periods):** for each `t ∈ {2,…,T}` and `g ∈ G_trt` with `t ≥ g`, `E[Y_t(∞) − Y_{t-1}(∞) | G=g, X] = E[Y_t(∞) − Y_{t-1}(∞) | G=∞, X]` a.s. Comparison group = **never-treated only**; reliable baseline = `g−1` only.
- **Assumption PT-All (Parallel Trends for all groups and periods):** for each `t ∈ {2,…,T}` and `(g,g') ∈ G_trt × G`, `E[Y_t(∞) − Y_{t-1}(∞) | G=g, X] = E[Y_t(∞) − Y_{t-1}(∞) | G=g', X]` a.s. Permits **any** untreated group as comparison and **all** pre-periods as baselines. PT-Post is the special case of PT-All restricted to post-treatment periods with the never-treated comparison.
- `X = 1` a.s. recovers the unconditional case (covariates not needed for identification).

*Causal parameters (§2.1, Eqs 2.1–2.3):*
```
ATT(g,t) := E[Y_t(g) − Y_t(∞) | G=g],   for t ≥ g                                   (2.1)
CATT(g,t,X) := E[Y_t(g) − Y_t(∞) | G=g, X]
ES(e) := E[ATT(G, G+e) | G+e ∈ [2,T]] = Σ_{g∈G_trt} P(G=g | G+e∈[2,T]) ATT(g, g+e)   (2.2)
ES_avg := (1/N_E) Σ_{e∈E} ES(e)                                                       (2.3)
```
`ES_avg` (Eq 2.3) is the scalar summary; under PT-All and a single date with no covariates it equals the static-TWFE coefficient `β` in `Y_{i,t} = α_t + η_i + β D_{i,t} + ε_{i,t}` (Eq 2.6). Under PT-Post the `ATT(g,t)` are identified with baseline `t_pre = g−1` only (Eq 2.5).

*Nonparametric overidentification (Lemma 2.1, pp. 7–8; Remark 2.1, p. 9):*
- **Lemma 2.1** establishes that under S, O, NA, PT-All, any covariate-specific weights `w(X)` summing to one across the admissible index set `H^{g,t} = {(g', t_pre, t'_pre) : g > t'_pre, g' > max(t_pre, t'_pre)}` identify `ATT(g,t)` (Eqs 2.7–2.8). It is the identification-with-any-weights result; it does **not** by itself give the *optimal* weights (those come from Theorems 3.1/3.2). One can use any of group `g`'s pre-treatment periods as a baseline (`g > t'_pre`) and form the effective comparison from the never-treated units together with **any** eventually-treated cohort `g'` whose treatment starts after the baseline periods used (`g' > max(t_pre, t'_pre)`, i.e. those periods are pre-treatment for `g'`). The binding restriction on the comparison cohort is `t_pre < g'`, **not** `g' > g`: the same cohort (`g'=g`, contributing overidentifying moments) and the never-treated (`g'=∞`) are both admissible — the staggered generated outcome (Eq 3.9) makes the `g'=g` reduction explicit.
- **Remark 2.1:** because the model is overidentified and the identification (2.7) is *homogeneous* across baselines and comparison groups, **non-convex (negative) weights are not a concern** here (contrast the TWFE negative-weights literature, Goodman-Bacon 2021 / de Chaisemartin–D'Haultfœuille 2020 / Sun–Abraham 2021 / Borusyak et al. 2024): the weakly-causal concern from negative weights (Blandhol et al. 2022) is not binding.

*Single-treatment-date semiparametric efficiency (§3.1, Theorem 3.1, Corollary 3.1):*
Two groups `G ∈ {g, ∞}`, `g ≥ 2`; `p_g(X) = E[G_g|X]` propensity, `p_∞(X) = 1 − p_g(X)`; `π_g := E[G_g]`. With `m_{g,t,t_pre}(X) := E[Y_t − Y_{t_pre} | G=g, X]` (Eq 3.1):
```
Transformed outcome (Eq 3.2):
  Ỹ_{g,t,t_pre} = (1/π_g) ( G_g − (p_g(X)/p_∞(X)) G_∞ ) ( Y_t − Y_{t_pre} − m_{∞,t,t_pre}(X) )

Per-baseline influence function (Eq 3.3):
  IF^{att(g,t)}_{t_pre} = Ỹ_{g,t,t_pre} − (G_g/π_g) ATT(g,t)            (mean zero)

Stack over baselines t_pre ∈ {1,…,g−1}:  IF^{att(g,t)} = (IF_1, …, IF_{g−1})'
Conditional covariance V_gt(X) = Cov(IF^{att(g,t)} | X); and V*_{gt}(X) has (j,k)-th element (Eq 3.4):
  (1/p_g(X)) Cov(Y_t−Y_j, Y_t−Y_k | G=g, X) + (1/(1−p_g(X))) Cov(Y_t−Y_j, Y_t−Y_k | G=∞, X)
```
- **Theorem 3.1** (pp. 11–12): under S, O, NA, PT-All the EIF of `ATT(g,t)`, `t ≥ g`, is the **inverse-covariance optimally-weighted** average of the per-baseline influence functions:
  ```
  EIF = ( 1'V_gt(X)^{-1} / 1'V_gt(X)^{-1} 1 ) IF^{att(g,t)} = ( 1'V*_{gt}(X)^{-1} / 1'V*_{gt}(X)^{-1} 1 ) IF^{att(g,t)}
  ```
  and the semiparametric efficiency bound is
  ```
  V_eff = (1/π_g²) ( E[ G_g (CATT(g,t,X) − ATT(g,t))² ]
                     + E[ p_g(X)² |V*_{gt}(X)| / Σ_{j,j'=1}^{g−1} (−1)^{j+j'} |V*_{gt,jj'}(X)| ] )
  ```
  where `V*_{gt,jj'}` is the minor of `V*_{gt}(X)` deleting row `j`, column `j'` (determinant of an empty matrix ≡ 1).
- **Efficient estimand (Eq 3.5)** and ES_avg (Eq 3.6), the basis for the library's no-covariate estimator:
  ```
  ATT(g,t) = E[ ( 1'V*_{gt}(X)^{-1} / 1'V*_{gt}(X)^{-1} 1 ) Ỹ_{g,t} ],   Ỹ_{g,t} = (Ỹ_{g,t,1},…,Ỹ_{g,t,g−1})'   (3.5)
  ES_avg   = (1/(T−g+1)) Σ_{t=g}^{T} E[ ( 1'V_gt(X)^{-1} / 1'V_gt(X)^{-1} 1 ) Ỹ_{g,t} ]                          (3.6)
  ```
  "More informative" baselines (lower inverse-`V*` weight denominators) receive more weight.
- **Corollary 3.1** (p. 12): under **PT-Post**, the EIF reduces to the single `g−1` baseline term `IF^{att(g,t)}_{g−1}`, and the bound to `V_{eff,j-id} = E[(Ỹ_{g,t,g−1} − (G_g/π_g) ATT(g,t))²]` ("just-identified"). This is the multi-period generalization of the Sant'Anna & Zhao (2020) 2-group-2-period efficiency bound; pre-treatment data beyond the last pre-period **cannot** improve efficiency under PT-Post.
- **Lemma 3.1** (p. 10) gives the equivalent sequential conditional moment restrictions (the overidentification structure) that the efficiency results are built on.

*Staggered semiparametric efficiency (§3.2, Theorem 3.2, Corollary 3.2):*
Generalized propensity `p_g(X) = E[G_g|X]`, `p_∞(X) = 1 − Σ_{g∈G_trt} p_g(X)`.
```
ES(e) = Σ_{g∈G_{trt,e}} ( π_g / Σ_{g∈G_{trt,e}} π_g ) ATT(g, g+e),   G_{trt,e} = {g∈G_trt : g+e ≤ T}     (3.8)

Generated outcome (Eq 3.9), leveraging the never-treated AND a not-yet-treated cohort g' as comparison:
  Ỹ^{att(g,t)}_{g',t_pre} = (G_g/π_g)(Y_t − Y_1 − m_{∞,t,t_pre}(X) − m_{g',t_pre,1}(X))
                           − (p_g(X)/p_∞(X))(G_∞/π_g)(Y_t − Y_{t_pre} − m_{∞,t,t_pre}(X))
                           − (p_g(X)/p_{g'}(X))(G_{g'}/π_g)(Y_{t_pre} − Y_1 − m_{g',t_pre,1}(X))
  Influence function (Eq 3.10): IF^{att(g,t)}_{g',t_pre} = Ỹ^{att(g,t)}_{g',t_pre} − (G_g/π_g) ATT(g,t)
  (when g'=g, (3.9)/(3.10) reduce to the single-date (3.2)/(3.3))

Stack the noncollinear IFs (Eq 3.11): IF^{att(g,t)}_stg ; conditional covariance Ω_gt(X) = Cov(IF^{att(g,t)}_stg | X);
  Ω*_{gt}(X) has the (j,k)-th element given in Eq 3.12 (group-specific covariances of outcome changes
  scaled by 1/p_g(X), 1/p_∞(X), and cross-cohort indicator terms).
```
- **Lemma 3.2** (p. 15) gives the staggered moment restrictions (Eq 3.7).
- **Theorem 3.2** (p. 16): under S, O, NA, PT-All the EIF of `π_g` and `ATT(g,t)`, `t ≥ g`, are
  ```
  EIF^{π_g} = G_g − π_g
  EIF^{att(g,t)}_stg = ( 1'Ω_gt(X)^{-1} / 1'Ω_gt(X)^{-1} 1 ) IF^{att(g,t)}_stg
                     = ( 1'Ω*_{gt}(X)^{-1} / 1'Ω*_{gt}(X)^{-1} 1 ) IF^{att(g,t)}_stg
  ```
  and the EIF of `ES(e)` adds the **weight-estimation (`G_g − π_g`) correction terms** that account for estimating the cohort-size aggregation weights (the influence of the `π_g`/`q_{g,e}` plug-ins, `q_{g,e} = P(G=g | G+e∈[2,T])`). Bounds are the second moments of these EIFs.
- **Eq 3.13 / 3.14** — the staggered efficient estimands:
  ```
  ATT(g,t) = ATT_stg(g,t) := E[ ( 1'Ω*_{gt}(X)^{-1} / 1'Ω*_{gt}(X)^{-1} 1 ) Ỹ^{att(g,t)} ]                 (3.13)
  ES(e)    = ES_stg(e) := Σ_{g∈G_{trt,e}} ( π_g / Σ_{g∈G_{trt,e}} π_g ) ATT_stg(g, g+e)                     (3.14)
  ```
  with `Ỹ^{att(g,t)}` the column vector of all noncollinear generated outcomes over `(g', t_pre)` pairs and efficiency weights `1'Ω*_{gt}(X)^{-1}/(1'Ω*_{gt}(X)^{-1}1)`.
- **Corollary 3.2** (p. 17): under **PT-Post** the staggered `ATT(g,t)` is nonparametrically **just-identified** and the EIF equals the single `g−1`-baseline term `IF^{att(g,t)}_{g,g−1}` (never-treated comparison only).
- **Remark 3.4** (p. 18): PT-Post and PT-All are the two ends of a spectrum; intermediate PT assumptions (e.g. Callaway & Sant'Anna 2021 not-yet-treated, restricting pre-trends only once the first cohort is treated at `g_min`, dropping `t = 1,…,g_min − 2`) are covered by restricting the stacked `IF^{att(g,t)}_stg` in (3.11) to the moment equations the chosen PT justifies.

*Semiparametric efficient estimation and inference (§4, Eqs 4.1–4.5; Theorem 4.1; Remarks 4.1–4.2):*
A two-step plug-in for the EIF-based estimands (3.13)/(3.14): estimate the nuisances `m(X)` (outcome-change regressions), `p_ratio(X) = (p_g(X)/p_{g'}(X))`, and the conditional covariance `Ω*_{gt}(X)`, then plug in.
- **Double robustness (p. 18–19):** with parametric working models the estimator is consistent for `ATT(g,t)` as long as **either** `p_ratio(X)` **or** `m(X)` is correctly specified, regardless of the weighting scheme (cf. Sant'Anna & Zhao 2020 Thm 1; Callaway & Sant'Anna 2021 Thm 2). Footnote 10 refines this: because the estimator averages across working models with efficiency-oriented weights, it stays consistent for *weighted averages* of functionals of the nuisances.
- **Nuisances are nonparametric in general:** `m(X)` is "nothing more than a nonparametric regression problem" — the paper lists **sieve (Chen 2007), kernel (Cattaneo–Jansson–Ma 2018), and ML methods (random forests, lasso, ridge, deep nets, boosted trees, ensembles; Chernozhukov et al. 2018)** as admissible. The paper does **not** mandate any single working model.
- **Propensity *ratio* directly (Eq 4.1):** since propensities enter (3.13) only in ratio form, the paper recommends estimating `r_{g,g'}(X) = p_g(X)/p_{g'}(X)` directly (not bounded to [0,1], more stable near 0) as the minimizer of the **convex loss** `E[r²G_{g'} − 2rG_g]`:
  ```
  p_g(X)/p_{g'}(X) = argmin_r E[(r(X) − p_g(X)/p_{g'}(X))² p_{g'}(X)] = argmin_r E[r(X)² G_{g'} − 2 r(X) G_g]   (4.1)
  ```
  (Footnote 11 notes one *may* instead impose `p_g(X) ∈ (0,1)` via multinomial series/local logit-probit.)
- **Sieve estimator (Eq 4.2)** with information-criterion order selection:
  ```
  r̂_{g,g'}(X) = ψ^K(X)'β̂_K,   ψ^K = K-dim flexible transforms (e.g. tensor products of cubic B-splines)
  β̂_K = argmin_{β_K} E_n[ G_{g'} (ψ^K(X)'β_K)² − 2 G_g (ψ^K(X)'β_K) ]                                       (4.2)
  K̂  = argmin_K  2 E_n[ G_{g'}(ψ^K'β̂_K)² − 2 G_g(ψ^K'β̂_K) ] + C_n K / n      (C_n=2 → AIC, C_n=log n → BIC)
  ```
  consistency of the `K̂`-based estimator follows Chen & Liao (2014). The inverse propensity `s_{g'}(X) = 1/p_{g'}(X)` is estimated analogously as `argmin_a E[a²G_{g'} − 2a]`.
- **Conditional covariance via Nadaraya–Watson kernel:** `Ω̂*_{gt}` plugs in (3.12) using a kernel `Ker`, bandwidth `h`, and `K_h(·) = Ker(·/h)/h`, with each covariance term estimated by a kernel-weighted within-group residual cross-product.
- **Estimators (Eqs 4.3–4.5):**
  ```
  ÂTT_stg(g,t) = E_n[ ( 1'Ω̂*_{gt}(X)^{-1} / 1'Ω̂*_{gt}(X)^{-1} 1 ) Ŷ^{att(g,t)}_stg ]                          (4.3)
  Ŷ^{att(g,t)}_{g',t_pre} = (G_g/π̂_g)(Y_t − Y_1 − m̂_{∞,t,t_pre}(X) − m̂_{g',t_pre,1}(X))
                            − r̂_{g,∞}(X)(G_∞/π̂_g)(Y_t − Y_{t_pre} − m̂_{∞,t,t_pre}(X))
                            − r̂_{g,g'}(X)(G_{g'}/π̂_g)(Y_{t_pre} − Y_1 − m̂_{g',t_pre,1}(X))                   (4.4)
  ÊS(e) = Σ_{g∈G_{trt,e}} ( π̂_g / Σ_{g'∈G_{trt,e}} π̂_g ) ÂTT_stg(g, g+e)                                     (4.5)
  ```
- **Theorem 4.1** (pp. 20–21): under O, NA, PT-All and the regularity conditions of Assumption C.1, `ÂTT_stg(g,t)` is consistent, asymptotically normal, and **attains the semiparametric efficiency bound** of Theorem 3.2; `ÊS(e)` likewise. **Standard errors** (p. 21): *"one can leverage a multiplier bootstrap procedure or take the square root of the average of the estimated EIF squared divided by the sample size"* — i.e. `SE = sqrt( mean(EIF²) / n )` or a multiplier bootstrap.
- **Remark 4.1:** the proof also gives consistency + asymptotic normality when the efficient weights are replaced by **any** consistent estimator of any weights `w(x)` summing to one (Lemma 2.1), with **no rate conditions** on the weights.
- **Remark 4.2 (Neyman orthogonality):** because the estimators are built from EIFs they satisfy Neyman orthogonality **by construction**, so modern ML nuisance estimators can be used without losing asymptotic semiparametric efficiency in large samples (footnote 14 notes ML + sample-splitting can lose finite-sample precision; Chen, Chen & Tamer 2023).

*No-covariate closed form (§4.1, pp. 21–22):*
Under NA and PT-All **unconditionally**, drop `X`: (3.13) reduces to `ATT(g,t) = (1'(Ω*_{gt})^{-1} / 1'(Ω*_{gt})^{-1}1) Ỹ^{att(g,t)}` with `Ỹ^{att(g,t)}_{g',t_pre} = E[Y_t − Y_1 | G=g] − (E[Y_t − Y_{t_pre} | G=∞] + E[Y_{t_pre} − Y_1 | G=g'])` and `Ω*_{gt}` the unconditional analog of (3.12). **Estimation replaces expectations/covariances with within-group sample means/sample covariances — no tuning parameters, no conditional-expectation estimation.** The optimal GMM estimator also attains the bound here but requires high-dimensional optimization; the closed-form weighted estimator matches that efficiency far more cheaply.

*Hausman PT-All vs PT-Post pretest (Appendix A, Theorem A.1, pp. 39–41):*
Because the model is overidentified under PT-All, its validity is directly testable. The test contrasts two event-study estimators, both aggregated to the post-treatment vector `ES = (ES(e))_{e∈E}`: the **efficient** estimator `ÊS` (PT-All, Eq 4.5 — consistent *and* semiparametrically efficient under PT-All) and the **restricted** estimator `ẼS` (PT-Post, Eq A.1 — uses the `g−1` baseline and never-treated comparison only; consistent under PT-Post but *less efficient* than `ÊS`).
- **Theorem A.1:** with consistent covariance estimators, the Hausman statistic `Ĥ = n (ÊS − ẼS)' [aCov(ẼS) − aCov(ÊS)]⁻¹ (ÊS − ẼS)` (Eq A.2; footnote 21) converges to `χ²(|E|)`, where `|E|` is the number of post-treatment event-time horizons, and has nontrivial power against all local alternatives. The covariance is the **restricted-minus-efficient** difference `aCov(ẼS) − aCov(ÊS)`, which is PSD under H0 because the efficient `ÊS` has the smaller variance. Reject PT-All for all periods/groups if `Ĥ` exceeds the `χ²(|E|)` critical value. (Footnote 22: the test compares whether the event-study aggregation of parallel trends is the same under PT-All and PT-Post.)
- **Incremental Sargan / Holm–Bonferroni (pp. 40–41):** a sequential step-down procedure that starts from the PT-Post baseline restriction set `M` (Eq 2.7 with `g'=g`, `t_pre=g−1`), adds overidentifying restrictions one at a time, and Hausman-tests each expanded model **against that PT-Post baseline `ẼS`/`M`** — selecting, after the Holm–Bonferroni family-wise-error adjustment, the largest restriction set not statistically distinguishable from the baseline.
- **Remark A.1:** using `Ĥ` as a hard pretest to choose between the restricted/unrestricted estimator can inflate MSE relative to an oracle; the adaptive bias-variance-weighted procedure of Armstrong et al. (2024) is an alternative.
- **Remark A.2 (placebo pre-trends):** the §4 estimator can be run with `t < g` to construct **pre-treatment "placebo" effects** (same spirit as Borusyak et al. 2024 pre-trend event studies).

**Reference implementation(s):** No canonical R/Stata package ships the efficient estimator. The paper benchmarks against `did` (Callaway–Sant'Anna), `DIDmultiplegt` (de Chaisemartin–D'Haultfœuille), `did2s`/`didimputation` (Gardner / Borusyak–Jaravel–Spiess), and `etwfe` (Wooldridge), and against `synthdid` (Arkhangelsky et al. 2021) in simulations. The library's `EfficientDiD` is an independent implementation; the natural cross-language anchor is the paper's own reported numbers (HRS Table 6 / Tables 7; simulations §5), not a reference package.

**Requirements checklist** (PR-B Verified-Components targets):
- [ ] Single-date EIF (Theorem 3.1) and the inverse-`V*` optimal weights `1'V*^{-1}/(1'V*^{-1}1)`; efficient estimand Eq 3.5.
- [ ] Staggered EIF (Theorem 3.2) with the generalized propensity, generated outcome Eq 3.9, conditional covariance Ω*_{gt}(X) Eq 3.12, efficient estimand Eq 3.13.
- [ ] PT-Post reductions (Corollary 3.1 single-date; Corollary 3.2 staggered) match the standard single-baseline/Callaway–Sant'Anna estimator.
- [ ] Event-study aggregation Eqs 3.8/3.14/(2.2)/(2.3) with cohort-size weights; ES(e) EIF carries the `(G_g − π_g)` weight-estimation correction (Theorem 3.2).
- [ ] No-covariate closed form (§4.1): within-group sample means/covariances, no tuning.
- [ ] Covariate DR path: ratio sieve Eqs 4.1–4.2 with AIC/BIC `K̂`, inverse-propensity sieve, Nadaraya–Watson kernel `Ω̂*`, DR generated outcomes Eq 4.4, efficient ATT Eq 4.3, ES Eq 4.5.
- [ ] Asymptotic normality + efficiency (Theorem 4.1); SE = `sqrt(mean(EIF²)/n)` or multiplier bootstrap (p. 21).
- [ ] Hausman PT-All vs PT-Post pretest (Theorem A.1): quadratic form on the ES vector with covariance difference; reference `χ²(|E|)`.
- [ ] Empirical anchor: HRS Table 6 / Table 7 ARE (see Implementation Notes).

---

## Implementation Notes

### Map to the diff-diff `EfficientDiD` classes
- **`efficient_did_weights.py`** — no-covariate path (§4.1, pp. 21–22): closed-form efficient weights from the inverse of the **sample unconditional** analog of `Ω*` (the §4.1 reduction of Theorem 3.2 / Eqs 3.12–3.13 with `X` dropped and within-group sample covariances replacing the conditional ones), generated outcomes, per-baseline influence functions, enumeration of valid `(g', t_pre)` triples (the `H^{g,t}` index of Lemma 2.1).
- **`efficient_did_covariates.py`** — covariate DR path (§4): sieve propensity-ratio estimation (Eqs 4.1–4.2), inverse-propensity sieve, Nadaraya–Watson kernel `Ω*(X)` (Eq 3.12), DR generated outcomes (Eq 4.4), EIF for analytical SEs.
- **`efficient_did.py`** — `EfficientDiD` estimator: PT-All (overidentified) vs PT-Post (`Corollary 3.2`, reduces to standard single-baseline / Callaway–Sant'Anna) selection, the Hausman pretest (Theorem A.1), survey/cluster inference on the EIF.
- **`efficient_did_bootstrap.py`** — multiplier bootstrap perturbing the stored EIF values (the bootstrap SE option of Theorem 4.1, p. 21).
- **`efficient_did_results.py`** — `EfficientDiDResults`, `HausmanPretestResult`.

### Empirical anchor for PR-B (HRS, §6; Table 6, p. 35; Table 7 ARE, p. 36)
Revisits Dobkin, Finkelstein, Kluender & Notowidigdo (2018)'s analysis of the effect of hospitalization on out-of-pocket medical spending, using the HRS data **compiled by Sun & Abraham (2021)** (record both: Dobkin et al. = data/study; Sun–Abraham = compilation + sample-selection steps). Original balanced panel: **652 individuals** observed in waves 7–11 (5 waves). The **Table 6 estimation sample drops wave 11** (by wave 11 every cohort — including the wave-11 cohort — is treated, so wave 11 carries no clean comparison), leaving **waves 7–10**; treated cohorts `g ∈ {8, 9, 10}` (defined by the wave first hospitalized), with the **wave-11 cohort serving as the pseudo-never-treated comparison** for the `t ≤ 10` window. Table 6 reports six `ATT(g,t)`, `ES(0)`, `ES(1)`, `ES(2)`, and `ES_avg`, with **standard errors clustered at the individual level**, for four estimators: **EDiD** (this paper), **CS-SA**, **CS-dCDH**, **BJS-G-W**. The efficient (EDiD) point estimates and SEs are:

| Target | ATT(8,8) | ATT(8,9) | ATT(8,10) | ATT(9,9) | ATT(9,10) | ATT(10,10) | ES(0) | ES(1) | ES(2) | ES_avg |
|---|---|---|---|---|---|---|---|---|---|---|
| EDiD | 3072 (806) | 1112 (637) | 1038 (817) | 3063 (690) | 90 (641) | 2908 (894) | 3024 (486) | 692 (471) | 1038 (816) | 1585 (521) |

Table 7 reports asymptotic relative efficiency (ARE, EDiD = 1.00): the alternatives need 28–104% larger samples to match EDiD's precision (e.g. ATT(8,9): CS-SA 2.04, CS-dCDH 1.83, BJS-G-W 1.45). Figure 5 (p. 36) shows the per-`(g', t_pre)` efficient weights are **non-uniform and can be negative** (consistent with Remark 2.1) and sum to one per `ATT(g,t)`.

### Monte Carlo design (§5, for context)
Two DGPs: single-treatment-date built on Arkhangelsky et al. (2021) calibrated to CPS; staggered built on Baker et al. (2022) calibrated to Compustat (covariates play no role in these setups). For `ES_avg`, the paper compares the efficient plug-in (EDiD, Eq 3.6) against OLS-TWFE (`β`, Eq 2.6), the average of post-treatment event-study TWFE coefficients (DTWFE, Eq 2.4), and Synthetic DiD (Arkhangelsky et al. 2021). Headline: the efficient estimators deliver markedly lower RMSE and narrower CIs, often exceeding 40% precision gains, with no loss in bias.

### Where the paper leaves a working-model choice open (neutral pointers for PR-B; no verdict here)
- **Outcome regression `m̂(X)`** — the paper admits sieve/kernel/ML and does **not** mandate a parametric form (the DR property covers misspecification of one nuisance). **Resolved in PR-B:** the library implements `m̂(X)` as a polynomial sieve (AIC/BIC order selection), the same basis family as the propensity-ratio sieve, so all nuisances are nonparametric and the covariate doubly-robust path attains the bound under the paper's regularity conditions (degree 1 reproduces a linear working model).
- **Sieve basis `ψ^K`** — the paper gives *(tensor products of) cubic B-splines* as an **example**; it does not mandate a specific basis family.
- **Kernel / bandwidth for `Ω̂*`** — the paper specifies a generic kernel `Ker` and bandwidth `h` (Nadaraya–Watson) without mandating Gaussian/Silverman.
- **Overall scalar summary** — the paper's scalar is `ES_avg` (Eq 2.3), a uniform average over post-treatment horizons; any cohort-size-weighted overall ATT is a different aggregation (PR-B to reconcile).
- **Aggregation-weight influence** — the ES(e) EIF (Theorem 3.2) includes the `(G_g − π_g)` weight-estimation correction; the analytical-vs-bootstrap treatment of this term is a PR-B verification point.

---

## Gaps and Uncertainties
1. **Regularity conditions (Assumption C.1):** Theorem 4.1's rates/smoothness conditions live in the appendix (Appendix C, proofs). PR-B should confirm the sieve/kernel tuning in the library is consistent with C.1 (e.g. `K̂` growth, bandwidth conditions) rather than only with the main-text descriptions.
2. **Finite-sample `Ω*` / `V` conditioning:** the theory inverts `V*_{gt}(X)` / `Ω*_{gt}(X)` and, for the Hausman test, the covariance *difference* `aCov(ẼS) − aCov(ÊS)` (restricted minus efficient), which is PSD only asymptotically. Finite-sample non-PSD handling (pseudoinverse, effective rank as DOF, conditioning guards) is an implementation matter not pinned by the paper; PR-B documents the library's choices against the `χ²(|E|)` reference.
3. **Hausman as a hard pretest (Remark A.1):** the paper itself flags MSE costs of using `Ĥ` to switch estimators; the library's use of the pretest (warning vs auto-switch) is a design choice to document, not a paper deviation.
4. **Survey/clustered inference:** the paper derives i.i.d. (Assumption S) random-sampling asymptotics and SEs via `sqrt(mean(EIF²)/n)` or multiplier bootstrap; clustered/survey-weighted variants of the EIF (Liang–Zeger; TSL) are library extensions beyond the paper's stated scope. PR-B should mark these as documented extensions, not paper requirements.
5. **Incremental Sargan selection:** Appendix A's Holm–Bonferroni sequential restriction-selection is a distinct procedure from the single Hausman pretest; whether the library implements it (vs only Theorem A.1) is a coverage question for PR-B.
6. **Out of scope — Appendix B (Instrumented DiD / DiD-IV):** the LATT/DiD-IV extension (Assumption DiD-IV; Lemmas B.1–B.2) is a separate estimand and is **not** part of `EfficientDiD`; no fidelity obligation here.
