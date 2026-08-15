# Methodology Registry

This document provides the academic foundations and key implementation requirements for each estimator in diff-diff. It serves as a reference for contributors and users who want to understand the theoretical basis of the methods.

**Result-class field naming.** Headline scalar inference fields appear under one of four native naming patterns: flat `att` / `se` / `conf_int` / `p_value` / `t_stat` (`DiDResults`, `SyntheticDiDResults`, `TROPResults`, `TripleDifferenceResults`, `HeterogeneousAdoptionDiDResults`, `ChangesInChangesResults`); `overall_*` (`CallawaySantAnnaResults` and the rest of the staggered family); `overall_att_*` (`ContinuousDiDResults`, where `att` and `acrt` are parallel response curves); and `avg_*` (`MultiPeriodDiDResults`). Result classes in the prefixed `overall_*` / `overall_att_*` / `avg_*` families additionally expose the flat `att` / `se` / `conf_int` / `p_value` / `t_stat` names as read-only `@property` aliases over their canonical fields, for adapter / external-consumer compatibility (see PR for v3.3.3, motivated by `balance.interop.diff_diff`). The flat-native classes (`DiDResults`, `SyntheticDiDResults`, `TROPResults`, `TripleDifferenceResults`, `HeterogeneousAdoptionDiDResults`, `ChangesInChangesResults`) already carry these names as native dataclass fields and are unchanged by this contract. `ContinuousDiDResults` further exposes `overall_*` aliases pointing at the ATT side (so `result.overall_se` reads `result.overall_att_se`, etc.). The native field is canonical for documentation, semantics, and computation — aliases are pure read-throughs and inherit the `safe_inference()` joint-NaN consistency contract automatically. Because aliases are `@property` descriptors (not dataclass fields), they do NOT appear in `dataclasses.fields()` or `dataclasses.asdict()` output, and assignment to an alias raises `AttributeError`; serializers and field-walkers continue to see only the canonical field set.

## Table of Contents

1. [Core DiD Estimators](#core-did-estimators)
   - [DifferenceInDifferences](#differenceindifferences)
   - [MultiPeriodDiD](#multiperioddid)
   - [TwoWayFixedEffects](#twowayfixedeffects)
2. [Modern Staggered Estimators](#modern-staggered-estimators)
   - [CallawaySantAnna](#callawaysantanna)
   - [ChaisemartinDHaultfoeuille](#chaisemartindhaultfoeuille)
   - [ContinuousDiD](#continuousdid)
   - [SunAbraham](#sunabraham)
   - [ImputationDiD](#imputationdid)
   - [TwoStageDiD](#twostagedid)
   - [StackedDiD](#stackeddid)
   - [WooldridgeDiD (ETWFE)](#wooldridgedid-etwfe)
   - [LPDiD](#lpdid)
   - [LWDiD](#lwdid)
3. [Advanced Estimators](#advanced-estimators)
   - [SyntheticDiD](#syntheticdid)
   - [SyntheticControl](#syntheticcontrol)
   - [TripleDifference](#tripledifference)
   - [StaggeredTripleDifference](#staggeredtripledifference)
   - [TROP](#trop)
   - [HeterogeneousAdoptionDiD](#heterogeneousadoptiondid)
   - [ChangesInChanges (CiC) + QDiD](#changesinchanges-cic)
   - [SpilloverDiD](#spilloverdid)
4. [Regression Discontinuity](#regression-discontinuity)
   - [RegressionDiscontinuity](#regressiondiscontinuity)
   - [RDPlot](#rdplot)
5. [Diagnostics and Sensitivity](#diagnostics-and-sensitivity)
   - [PlaceboTests](#placebotests)
   - [BaconDecomposition](#bacondecomposition)
   - [HonestDiD](#honestdid)
   - [PreTrendsPower](#pretrendspower)
   - [PowerAnalysis](#poweranalysis)

---

# Core DiD Estimators

## DifferenceInDifferences

**Primary source:** Canonical econometrics textbooks
- Wooldridge, J.M. (2010). *Econometric Analysis of Cross Section and Panel Data*, 2nd ed. MIT Press.
- Angrist, J.D., & Pischke, J.-S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.

**Key implementation requirements:**

*Assumption checks / warnings:*
- Treatment and post indicators must be binary (0/1) with variation in both
- Warns if no treated units in pre-period or no control units in post-period
- Parallel trends assumption is untestable but can be assessed with pre-treatment data

*Estimator equation (as implemented):*
```
ATT = (Ȳ_{treated,post} - Ȳ_{treated,pre}) - (Ȳ_{control,post} - Ȳ_{control,pre})
    = E[Y(1) - Y(0) | D=1]
```

Regression form:
```
Y_it = α + β₁(Treated_i) + β₂(Post_t) + τ(Treated_i × Post_t) + X'γ + ε_it
```
where τ is the ATT.

*Standard errors:*
- Default: HC1 heteroskedasticity-robust
- Optional: Cluster-robust (specify `cluster` parameter)
- Optional: Wild cluster bootstrap for small number of clusters
  (`inference="wild_bootstrap"` with `cluster=` — required since 3.9, see the WCR Note)
- With `absorb=`, the absorbed-FE degrees-of-freedom adjustment uses the
  component-aware rank (`diff_diff.utils.absorbed_fe_rank`) — see the
  TwoWayFixedEffects section's absorbed-FE degrees-of-freedom note for the
  shared convention and its fixest `K.exact` anchor. Consumed by the
  classical/HC1 variance rescale and the reported residual df.

*Edge cases:*
- Empty cells (e.g., no treated-pre observations) cause rank deficiency, handled per `rank_deficient_action` setting
  - With "warn" (default): emits warning, sets NaN for affected coefficients
  - With "error": raises ValueError
  - With "silent": continues silently with NaN coefficients
- Singleton clusters (one observation): included in variance estimation; contribute to meat matrix via u_i² X_i X_i' (same formula as larger clusters with n_g=1)
- Rank-deficient design matrix (collinearity): warns and sets NA for dropped coefficients (R-style, matches `lm()`)
  - Tolerance: `1e-07` (matches R's `qr()` default), relative to largest diagonal element of R in QR decomposition
  - Controllable via `rank_deficient_action` parameter: "warn" (default), "error", or "silent"
  - **Note (scale invariance):** the shared `diff_diff/linalg.py` rank check re-detects on column-equilibrated (unit 2-norm) columns when the raw pass reports a deficiency (adopting the higher equilibrated rank and its scale-corrected pivot selection only when a large-scale column inflated the threshold; genuine collinearity keeps the raw selection), and the least-squares solve equilibrates columns then unscales the coefficients — so rank detection and the fit are invariant to per-column scaling; for a well-scaled collinear design the dropped column is unchanged, while a scale-induced under-count adopts the scale-corrected equilibrated selection (which may differ from the raw choice but is guaranteed to retain an identified subset). This covers every covariate outcome-regression fit routed through `solve_ols` — DiD, TwoWayFixedEffects, MultiPeriodDiD, ImputationDiD, TwoStageDiD, TripleDifference, and (as of the OR scale-equilibration change) CallawaySantAnna's `_compute_all_att_gt_covariate_reg` / `_doubly_robust` and StaggeredTripleDifference's `_compute_or` — see the CallawaySantAnna rank-deficiency Note's scope statement. **Certification fast path (2026-07):** the common full-rank case is short-circuited by a Gram/eigenvalue certification on the equilibrated Gram at the documented `_rank_guarded_inv` 1e-10 Gram threshold — two orders stricter than the QR full-rank boundary, so it never reports full rank where the pivoted-QR path would report a deficiency; deficient or uncertifiable designs (including all n < k and non-finite inputs) fall through to the pivoted-QR path verbatim, leaving every drop decision, pivot selection, and rank count unchanged. Certification never SELECTS columns (it only confirms none need dropping), so the `_rank_guarded_inv` note about equilibrated-Gram column selection being reserved for the IF inverse still holds.
  - **Note (run-to-run determinism of the clustered vcov, 2026-07):** the Rust backend's clustered `compute_robust_vcov` / `solve_ols(return_vcov=True)` previously aggregated cluster scores in `HashMap` iteration order — SipHash-randomized per call — so identical inputs produced ~1e-14-level distinct vcov values across calls (the Python backend was always bit-stable). Fixed in #653: rows accumulate in first-appearance order (ascending for the factorized 0..G-1 ids the dispatcher passes, matching NumPy's groupby order); repeated identical calls are bit-identical (a 50-call probe during the fix; the shipped regression asserts 20 repeats per id-layout), NumPy parity unchanged (~1e-14 cross-backend GEMM tolerance). Regression: `tests/test_rust_backend.py::TestClusterVcovDeterminism` (contiguous + non-contiguous unsorted ids, plus a raw-kernel path).
- **Note (covariate-name collision guard):** `DifferenceInDifferences`, `MultiPeriodDiD`, and `TwoWayFixedEffects` build their `coefficients` dict by zipping a variable-name list (structural terms + user covariate column names, appended verbatim) with the coefficient vector. A covariate whose name equals a reserved structural name — `const`, the treatment/time column names, the `{treatment}:{time}` interaction (DiD), the `period_{p}` dummies / `{treatment}:period_{p}` interactions (MultiPeriodDiD), `ATT` (TwoWayFixedEffects), any fixed-effect / unit / time dummy name, or an internal `_`-prefixed working column (`_treat_time`, `_did_treatment`, `_treatment_post`) — would silently overwrite that structural coefficient (dict last-write-wins) with no error. `fit()` now calls `utils.validate_covariate_names()` before building the design, raising `ValueError` on such a collision (case-sensitive, so `Const` is allowed) and on duplicate covariate names. Reserved fixed-effect/unit/time dummy names are taken from the same `pd.get_dummies(..., drop_first=True)` call used to build them (exact match, including for `Categorical` columns). The TwoWayFixedEffects guard fires on both variance paths — the within-transform path exposes only `{"ATT": att}`, but a `_treatment_post`-named covariate would still clobber the internal interaction column. The influence-function estimators (CallawaySantAnna, etc.) key results by `(g, t)` and are unaffected.

**Reference implementation(s):**
- R: `fixest::feols()` with interaction term
- Stata: `reghdfe` or manual regression with interaction

**Requirements checklist:**
- [x] Treatment and time indicators are binary 0/1 with variation
- [x] ATT equals coefficient on interaction term
- [x] Wild bootstrap supports Rademacher, Mammen, Webb weight distributions
- [x] Formula interface parses `y ~ treated * post` correctly

### Wild cluster bootstrap (WCR)

`inference="wild_bootstrap"` (with `cluster=`) runs the **Wild Cluster Restricted (WCR)**
bootstrap of Cameron, Gelbach & Miller (2008), matching the defaults of R's
`fwildclusterboot::boottest` (Roodman, MacKinnon, Nielsen & Webb 2019). Implemented in
`diff_diff.utils.wild_bootstrap_se`; used by `DifferenceInDifferences` and (inherited)
`TwoWayFixedEffects`. (`MultiPeriodDiD` does **not** support it — it falls back to analytical
inference and the inherited `p_val_type` is inert there.)

- **Note:** Since 3.9 the selector fails closed (rows M-081/M-096): `inference=`
  accepts exactly `{"analytical", "wild_bootstrap"}` (string-typed) at construction
  and `set_params`; `DifferenceInDifferences` with `wild_bootstrap` and no
  `cluster=` raises `ValueError` at fit — the code now matches this section's own
  "(with `cluster=`)" definition, where it previously fell back to analytical
  silently — and DiD/TWFE with `n_bootstrap < 2` under `wild_bootstrap` raise
  (0 or 1 replications deterministically produce the all-NaN degenerate tuple).
  `TwoWayFixedEffects`' unit auto-cluster satisfies the cluster prerequisite;
  the survey and Conley `NotImplementedError` rejections keep precedence.
  `MultiPeriodDiD` keeps its warn-and-analytical-fallback until its 4.0 removal.

*Algorithm (test of H₀: τ = r, default r = 0):*
1. **Impose the null** by dropping the interaction column and re-fitting the reduced model;
   the restricted residuals are `ũ(r) = M₋ⱼ y − r·M₋ⱼ xⱼ` (linear in `r`, where `M₋ⱼ` is the
   annihilator of the design without column `j`). Rank-deficient nuisance columns are dropped
   via `solve_ols` so the identified ATT (and the bootstrap) stay finite.
2. **Cluster sign-vectors** `w_g`: for Rademacher weights with few clusters the full set of
   `2**n_clusters` sign-vectors is **enumerated** (deterministic) when `2**n_clusters ≤
   n_bootstrap` **and `n_clusters ≤ 20`** (a guard against pathological memory use) — the same
   full-enumeration trigger `boottest` uses (it switches to enumeration once `n_bootstrap`
   reaches the number of possible draws `2**n_clusters`; verified empirically — for `G=10` it
   samples at `B=1023` and enumerates at `B=1024`). Only `2**(n_clusters-1)` of those draws have
   distinct `|t*|` (each draw and its all-signs-flipped mirror share `|t*|`), but the full set is
   materialized and `n_bootstrap` is reported as `2**n_clusters`. Otherwise signs are sampled
   (`seed`-reproducible). Webb/Mammen always sample (the sign-flip symmetry is Rademacher-specific).
3. **Bootstrap statistic** `t*(r) = (β*ⱼ − r) / se*`, where each draw refits the full design on
   `y* = (y − ũ(r)) + ũ(r)·w` and `se*` is the CR1 cluster-robust SE of `β*ⱼ`. The observed
   statistic is `t₀ = (τ̂ − r)/se_a` with the analytical CR1 SE `se_a`.
4. **p-value:** two-tailed `mean(|t*| > |t₀|)` or equal-tailed `2·min(mean(t*<t₀), mean(t*>t₀))`
   — strict `>`, matching boottest (the all-(+1)/all-(−1) enumerated draws reproduce ±t₀ and are
   excluded as boundary ties). Floored at `1/(n_valid+1)` (see Deviation below).
5. **Confidence interval by test inversion:** the set of nulls `r` not rejected at `alpha`,
   located by outward bracketing + bisection on the (monotone, step) rejection frequency. The CI
   is therefore exactly consistent with the p-value (`0 ∈ CI ⟺ p ≥ alpha`) and may be asymmetric.
6. The reported `se` is `se_a` (analytical CR1); `p_val_type ∈ {"two-tailed" (default),
   "equal-tailed"}`. CR1 uses the standard `(G/(G−1))((N−1)/(N−K_reference))` correction —
   under absorbed/nested FE the `k` carries the same signed `cluster_k_adjustment` as the
   analytical lane (3.9 K_reference convergence) — which cancels in `|t*|` vs `|t₀|` so it
   affects only the reported SE (and `t_stat_original` / any requested
   `bootstrap_distribution`, which share `se_a`'s scale), not the p-value; CI endpoints move
   only within the bisection tolerance through `se_a`'s bracketing role
   (`tests/test_variance_conventions.py::TestKReferenceConvergence`).

*Verification — R parity:* validated against `fwildclusterboot::boottest()` defaults on a fixed
few-cluster golden (`benchmarks/R/generate_wild_cluster_boot_golden.R` →
`benchmarks/data/wild_cluster_boot_golden.json`), enumerated and deterministic on both sides. The
bootstrap t-distribution matches R to ~6e-14; `se`, `t₀`, and the (interior) p-value match exactly;
the inverted CI matches to ~1e-4 (bisection vs boottest's grid search). Also pinned against an
independent full-refit enumeration in `tests/test_wild_bootstrap.py::test_wcr_matches_independent_bruteforce`.

- **Note:** The reported quantities mix inference families *by design*: `t_stat_original` is the
  analytical Wald statistic `τ̂/se_a`, while `p_value` and `conf_int` come from the bootstrap test
  inversion. This is intentional (it is exactly the boottest convention) and is not a deviation.
- **Deviation from R:** the p-value is floored at `1/(n_valid+1)` to avoid reporting an exact
  `0` (which boottest can return under full enumeration of a strong effect) — but the floor is
  applied **only when `1/(n_valid+1) < alpha`**. With very few valid draws the floor can exceed
  `alpha`; applying it there would lift a bootstrap-significant p (0 outside the inverted CI) to
  "non-significant", contradicting the CI, so in that regime the raw p (which boottest also
  reports, possibly 0) is returned. The significance verdict therefore always matches the inverted
  CI (`0 ∈ CI ⟺ p ≥ alpha`).

---

## MultiPeriodDiD

- **Note:** Deprecated in 3.9, removed in 4.0 (ledger row M-010): merged into
  the `TwoWayFixedEffects` event-study mode. `TWFE().fit(...,
  event_study=True, spec="pooled", time=<calendar col>)` reproduces this
  estimator's design EXACTLY (same shared estimation core; bit-exact under
  matched cluster settings and unconditionally without a unit id); the
  default `spec="within"` adds unit fixed effects (standard errors shift in
  general, and point estimates too on unbalanced/covariate designs - the
  documented estimate change). Everything below remains the 3.x contract
  for the deprecated class through its removal.

**Primary source:** Event study methodology
- Freyaldenhoven, S., Hansen, C., Pérez, J.P., & Shapiro, J.M. (2021). Visualization,
  identification, and estimation in the linear panel event-study design. NBER Working Paper 29170.
- Wooldridge, J.M. (2010). *Econometric Analysis of Cross Section and Panel Data*, 2nd ed.
  MIT Press, Ch. 10, 13.
- Angrist, J.D., & Pischke, J.-S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.

**Scope:** Simultaneous adoption event study. All treated units receive treatment at the
same time. For staggered adoption (different units treated at different times), use
CallawaySantAnna or SunAbraham instead.

**Key implementation requirements:**

*Assumption checks / warnings:*
- Treatment indicator must be binary (0/1) with variation in both groups
- Requires at least 1 pre-treatment and 1 post-treatment period
- Warns when only 1 pre-period available (≥2 needed to test parallel trends;
  ATT is still valid but pre-trends assessment is not possible)
- Reference period defaults to last pre-treatment period (e=-1 convention)
- Treatment indicator should be time-invariant ever-treated (D_i);
  warns when time-varying D_it detected (requires `unit` parameter)
- Warns if treatment timing varies across units when `unit` is provided
  (suggests CallawaySantAnna or SunAbraham instead)
- Treatment must be an absorbing state (once treated, always treated)

*Estimator equation (target specification):*

With unit and time fixed effects absorbed:

```
Y_it = α_i + γ_t + Σ_{e≠-1} δ_e × D_i × 1(t = E + e) + X'β + ε_it
```

where:
- α_i = unit fixed effects (absorbed)
- γ_t = time fixed effects (absorbed)
- E = common treatment time (same for all treated units)
- D_i = treatment group indicator (1=treated, 0=control)
- e = t - E = event time (relative periods to treatment)
- δ_e = treatment effect at event time e
- δ_{-1} = 0 (reference period, omitted for identification)

For simultaneous treatment, this is equivalent to interacting treatment with
calendar-time indicators:

```
Y_it = α_i + γ_t + Σ_{t≠t_ref} δ_t × (D_i × Period_t) + X'β + ε_it
```

where interactions are included for ALL periods (pre and post), not just post-treatment.

Pre-treatment coefficients (e < -1) test the parallel trends assumption:
under H0 of parallel trends, δ_e = 0 for all e < 0.

- **Note:** The 3.x DEFAULT implementation estimates the pooled form of this
  target - treatment-group main effect + period dummies + interactions, with
  NO unit fixed effects unless `absorb`/`fixed_effects` is passed
  (`estimators.py` design build). The unit-FE form above (the
  `fixest::feols(... | unit + time)` parity target) and the pooled default
  produce identical point estimates only under balanced panels with no
  covariates and simultaneous adoption; with unbalanced panels or covariates
  the two projections differ. The 4.0 merge changes the event-study default
  to the unit-FE spec with the pooled model reachable via `spec="pooled"` —
  plan and gates: `docs/v4-design.md` section 4.1, ledger row [M-010].

Post-treatment coefficients (e ≥ 0) estimate dynamic treatment effects.

Average ATT over post-treatment periods:

```
ATT_avg = (1/|post|) × Σ_{e≥0} δ_e
```

with SE computed from the sub-VCV matrix:

```
Var(ATT_avg) = 1'V1 / |post|²
```

where V is the VCV sub-matrix for post-treatment δ_e coefficients.

*Standard errors:*
- Default: HC1 heteroskedasticity-robust (same as DifferenceInDifferences base class)
- Alternative: Cluster-robust at unit level via `cluster` parameter (recommended for panel data)
- `vcov_type="hc2_bm"` (one-way) computes HC2 + Imbens-Kolesar (2016) Satterthwaite DOF
  per coefficient and a contrast-aware DOF for the post-period-average ATT.
- `cluster` + `vcov_type="hc2_bm"` is now supported (PR for Gate 6 lift). Both per-period
  effects and the post-period-average ATT use a cluster-aware Bell-McCaffrey
  Satterthwaite DOF: the per-coefficient case continues via `_compute_cr2_bm`, and
  the compound `avg_att` contrast DOF goes through the new
  `_compute_cr2_bm_contrast_dof` helper in `diff_diff/linalg.py` (Pustejovsky-Tipton
  2018 §4 algebra generalized to arbitrary `(k, m)` contrast matrices). R parity
  verified at atol=1e-10 vs clubSandwich's
  `Wald_test(constraints=matrix(c, 1), test="HTZ")$df_denom`. Weighted CR2-BM
  under `survey_design=` is not computed: the design-based survey variance (TSL or
  replicate-weight) takes precedence over the analytical HC2-BM sandwich at fit, so
  no CR2 / Bell-McCaffrey DOF is reported on survey paths. (On the low-level
  `solve_ols(weights=..., weight_type="pweight")` surface, plain pweight weighting
  routes through the clubSandwich WLS-CR2 port; `MultiPeriodDiD.fit()` itself does not
  expose a bare `weights=` — survey weights enter via `survey_design=`.)
- **Note:** the cluster-aware MPD `hc2_bm` path computes the CR2 hat matrix,
  per-cluster adjustment matrices, the sandwich vcov, AND the per-coefficient +
  post-period-average contrast DOF together from a single shared precompute build
  (`_compute_cr2_bm_vcov_and_dof` in `diff_diff/linalg.py`); the fit bypasses
  `solve_ols`'s separate vcov dispatch on this path so the CR2 setup (per-cluster
  adjustment blocks) is built once per fit rather than twice. `_compute_cr2_bm` (per-coefficient vcov + DOF) and
  `_compute_cr2_bm_contrast_dof` (DOF-only for arbitrary contrasts) are thin wrappers
  over that shared core, so every CR2 caller routes through one implementation. The
  consolidation is bit-identical to the prior two-call path (proven at atol=0).
- **Note (unweighted per-coef DOF guard):** the unweighted clustered CR2-BM per-coefficient DOF (`_cr2_bm_dof_inner`, the simple `(tr B)²/tr(B²)` form) carries the same two-part reliability guard as the weighted P-array path. (1) **Noise floor:** for a high-leverage FE-dummy / collinear nuisance column `trace_B2 = Σ B_{g,h}²` collapses to float64 accumulation noise while `trace_B` stays O(1), inflating the ratio to a non-physical DOF (observed ~1e61 on the absorbed-FE golden); a contrast whose `max|B_{g,h}|` sits below the batch-relative (`1e-10×max`, computed on the scale-normalized `max|B|/‖c‖²` since `B ∝ ‖c‖²` while the DOF is scale-invariant) or absolute (`(EPS·n·k·bread_scale)²`) floor is NaN'd. (2) **Cluster-count bound:** the Bell-McCaffrey Satterthwaite DOF is `(tr B)²/tr(B²)` with `B` PSD and cluster-structured, so it is bounded by `rank(B) ≤ G` (number of clusters); the simple unweighted form is numerically less faithful than clubSandwich's P-array form on high-leverage columns and can return a finite-but-inflated DOF above `G` (observed ~32.7 and ~16.3 vs R's 6 and 3, `G=8`), which is NaN'd as non-physical. The well-conditioned contrasts estimators consume — the treatment effect, event-study coefficients, and the compound post-period-average — are unaffected and match R clubSandwich; only the non-user-facing high-leverage nuisance DOFs are suppressed (exact P-array reproduction of those is deferred). A `UserWarning` fires per fit. Regression: `tests/test_estimators_vcov_type.py::TestDiDAbsorbedFERParity::test_unweighted_cr2_bm_per_coef_dof_no_nonphysical`. **Evaluation (2026-07):** the pairwise `B` matrix is computed via the algebraic identity `B = Ω'MΩ = diag(‖ω_g‖²) − P' M_U P` with `P = X'Ω` (the Satterthwaite DOF itself is Pustejovsky-Tipton 2018 §3.1 Eq. 11; equivalently the `q=1` reduction of the AHT Eq. 13) — instead of contracting against an explicit dense `n×n` residual-maker (`O(n k + G² k)` per contrast; peak memory = two `O(n k)` input-scale score precomputes plus working buffers capped by a byte cap subject to a one-contrast lower bound (a single contrast intrinsically needs `O(n)` + `O(G k)` buffers) — q vectors, per-cluster omegas, and product buffers contrast-chunked with every width-scaled buffer counted in the chunk denominator, and the `(G, G)` pairwise matrix row-chunked (Frobenius sum and max are row-separable), so none of the `n×n` residual-maker, `O(n m)` score arrays, or `O(G²)` pairwise entries is ever held at once; chunk-count invariant to ~1 ULP — BLAS width-dependent column accumulation; algebraically identical, ~1e-12 float agreement locked by a frozen-oracle parity test). **One-way scores evaluation (2026-07):** the non-clustered unweighted BM DOF denominator `a'(M∘M)a` (`_compute_bm_dof_from_contrasts`) is likewise evaluated without the dense `n×n` residual-maker, via the Schur-product expansion `Σ a_i²(1−2h_ii) + tr((B S_a)²)` with `S_a = X'diag(a)X` — exact algebra, `O(n k² + k³)` per contrast (was `O(n²k)`), frozen-dense-oracle parity ~1e-12, with a noise-floor cancellation guard NaN-ing extreme-leverage contrasts whose expanded denominator collapses below float precision (`TestOneWayBMScoresDOF`). **Low-rank A_g factorization (2026-07):** the unweighted per-cluster adjustment operator `A_g = (I − H_gg)^{−1/2}` is evaluated from the k×k eigenproblem of `U_g'U_g` (`U_g = X_g M_U^{1/2}`; `A_g = I + (U_g Q) diag(γ) (U_g Q)'`, `γ = ((1−λ)^{−1/2}−1)/λ` via expm1/log1p, Moore-Penrose zeroing at `1−λ ≤ 1e-10` matching `_cr2_adjustment_matrix`'s convention) and only ever applied to skinny matrices — the dense `(n_g, n_g)` `A_g` (an `O(n_g³)` eigh per cluster) is never materialized for `n_g > k`; clusters with `n_g ≤ k` keep the (tiny) dense construction since it is the smaller Gram side. Algebraically identical; vcov/DOF match a frozen dense-eigh oracle at rtol 1e-12/1e-10 incl. the leverage-1 absorbed-cluster-FE and singleton-cluster regimes (`TestCR2BMLowRankAdjustment`). The weighted clubSandwich path keeps the dense construction (its `G_g` carries the `S_W` bias term).
- **Note:** `LinearRegression.get_se()` / `get_inference()` clamp the vcov diagonal at 0
  before `sqrt`. A high-leverage / degenerate coefficient (an absorbed-FE dummy
  near-collinear with the treatment, whose Satterthwaite DOF already hits the noise-floor
  guard) has a CR2/HC variance of ~0 (≈1e-32) that can land just-below-zero under
  BLAS-dependent rounding; the clamp keeps the SE finite (0 for a genuinely-zero variance)
  and deterministic across BLAS implementations, never `NaN`. No effect on any positive
  variance. Regression: `tests/test_methodology_wls_cr2.py::TestLinearRegressionFENanGuardEndToEnd`.
- Wild cluster bootstrap: **not supported** — `inference="wild_bootstrap"` warns and
  falls back to analytical inference (multi-coefficient testing would require the
  joint bootstrap distribution; the 4.0 removal replaces the fallback with a raise)
- Degrees of freedom adjusted for absorbed fixed effects: component-aware rank
  via `diff_diff.utils.absorbed_fe_rank` (own `_absorbed_fe_vcov_scale` gate at
  its fit site — MultiPeriodDiD is a second implementation, not an alias of
  DifferenceInDifferences) — see the TwoWayFixedEffects section's absorbed-FE
  degrees-of-freedom note for the shared convention
- **Note (clustered CR1 k — built-in period dummies, 3.9 K_reference):** MPD's
  built-in `period_*` dummies ARE its time-FE block — a time column supplied via
  `fixed_effects=` is skipped as redundant with them, and `absorb=[..., time]` is
  the absorbed spelling of the same block — so they follow the nested convention:
  under a time-like cluster their rank is subtracted from the clustered CR1
  factor's `k` (negative `cluster_k_adjustment`), and under `absorb=` the
  non-nested increment applies, preserving MPD's absorb/fixed_effects clustered-SE
  equivalence (the D1 rule; audit rows `mpd_absorb_hc1_cluster_unit` /
  `mpd_fixed_effects_hc1_cluster_unit` / `mpd_plain_hc1_cluster_time` in
  `tests/test_variance_conventions.py`). The drop is an equivalence/reference
  rule, not a blanket nested-dummies rule: a built-in dummy block subtracts
  exactly when it is the estimator's FE block with an absorbed spelling or a
  reference absorbing it.

*Edge cases:*
- Reference period: omitted from design matrix; coefficient is zero by construction.
  Default is last pre-treatment period (e=-1). User can override via `reference_period`.
- Post-period reference: raises ValueError. Post-period references would exclude a
  post-treatment period from estimation, biasing avg_att and breaking downstream inference.
- Reference period default change: FutureWarning emitted when `reference_period` is not
  explicitly specified and ≥2 pre-periods exist, noting the default changed from first
  to last pre-period (e=-1 convention, matching fixest/did).
- Never-treated units: all event-time indicators are zero; they identify the time
  fixed effects and serve as comparison group.
- Endpoint binning: distant event times (e.g., e < -K or e > K) should be binned
  into endpoint indicators to avoid sparse cells. This prevents imprecise estimates
  at extreme leads/lags.
- Unbalanced panels: only uses observations where event-time is defined. Units
  not observed at all event times contribute to the periods they are present for.
- Rank-deficient design matrix (collinearity): warns and sets NA for dropped
  coefficients (R-style, matches `lm()`)
  - **Note (scale invariance):** shared `diff_diff/linalg.py` behavior — rank detection re-checks on column-equilibrated columns and the solve equilibrates/unscales, so detection and fit are invariant to per-column scaling. For a well-scaled collinear design the dropped column is unchanged; a scale-induced under-count adopts the scale-corrected equilibrated selection (which may differ from the raw choice but retains an identified subset). The common full-rank case is short-circuited by a strictly-stricter Gram certification (2026-07) that leaves all drop decisions unchanged. See the CallawaySantAnna rank-deficiency Note.
  - **Note (covariate-name collision guard):** a covariate named like a reserved structural term (`const`, the treatment column, a `period_{p}` dummy, a `{treatment}:period_{p}` interaction, a fixed-effect dummy, or an internal `_did_*` column) — or a duplicate covariate name — raises `ValueError` (would otherwise silently overwrite a structural coefficient in the coef dict). See the DifferenceInDifferences "covariate-name collision guard" Note.
- Average ATT (`avg_att`) is NA if any post-period effect is unidentified
  (R-style NA propagation)
- NaN inference for undefined statistics:
  - t_stat: Uses NaN (not 0.0) when SE is non-finite or zero
  - p_value and CI: Also NaN when t_stat is NaN
  - avg_se: Checked for finiteness before computing avg_t_stat
  - **Note**: Defensive enhancement matching CallawaySantAnna NaN convention
- Treatment reversal: warns if any unit transitions from treated to untreated
  (non-absorbing treatment violates the simultaneous adoption assumption)
- Time-varying treatment (D_it): warns when `unit` parameter is provided and
  within-unit treatment variation is detected. Advises creating an ever-treated
  indicator. Without ever-treated D_i, pre-period interaction coefficients are
  unidentified.
- **Note (staggered detection requires D_it input):** the staggered-adoption
  and treatment-reversal checks derive adoption timing from within-unit 0→1
  transitions, so they can only fire on time-varying `D_it` input (itself
  off-contract per the bullet above). With the contract-valid time-invariant
  `D_i`, adoption timing is not observable in the inputs and the
  simultaneous-adoption scope is a user-asserted design assumption - the
  estimator cannot verify it. See the TwoWayFixedEffects event-study-mode
  Note (staggered-adoption detection limit); the cohort-timing validation
  input that would close this is tracked in TODO.md.
- Pre-test of parallel trends: joint F-test on pre-treatment δ_e coefficients.
  Low power in pre-test does not validate parallel trends (Roth 2022).

**Reference implementation(s):**
- R: `fixest::feols(y ~ i(time, treatment, ref=ref_period) | unit + time, data, cluster=~unit)`
  or equivalently `feols(y ~ i(event_time, ref=-1) | unit + time, data, cluster=~unit)`
- Stata: `reghdfe y ib(-1).event_time#1.treatment, absorb(unit time) cluster(unit)`

**Requirements checklist:**

- [x] Event-time indicators for ALL periods (pre and post), not just post-treatment
- [x] Reference period coefficient is zero (normalized by omission from design matrix)
- [x] Pre-period coefficients available for parallel trends assessment
- [ ] Default cluster-robust SE at unit level (currently HC1; cluster-robust via `cluster` param)
- [ ] Supports unit and time FE via absorption
- [ ] Endpoint binning for distant event times
- [x] Average ATT correctly accounts for covariance between period effects
- [x] Returns PeriodEffect objects with confidence intervals
- [x] Supports both balanced and unbalanced panels

---

## TwoWayFixedEffects

**Primary source:** Panel data econometrics
- Wooldridge, J.M. (2010). *Econometric Analysis of Cross Section and Panel Data*, 2nd ed. MIT Press, Chapter 10.

### Event-study mode (3.9, ledger row M-010)

`fit(..., event_study=True, time=<calendar col>, spec="within"|"pooled",
reference_period=None, post_periods=<required>)` estimates per-period
treatment effects and returns the unified `EventStudyResults` surface (spec section 5
of `docs/v4-design.md`; `source="TwoWayFixedEffects"`, `estimation_spec` and
the authoritative `post_periods` partition recorded as provenance). The
static 0/1-dummy contract below is unchanged; in event-study mode `time=` is
the CALENDAR column (keyword), and the static dummy parameter is `post=`
(row M-082; `time=` remains its deprecated alias through 3.9).

*Designs:*
```
spec="within":  Y_it = α_i + γ_t + Σ_{e≠ref} δ_e (D_i × 1[t=e]) + X'β + ε_it
spec="pooled":  Y_it = α  + βD_i + Σ_{t≠ref} γ_t Period_t
                      + Σ_{e≠ref} δ_e (D_i × 1[t=e]) + X'β + ε_it
```
`spec="pooled"` is the MultiPeriodDiD design verbatim (shared estimation
core - bit-exact reproduction under matched cluster settings) and the only
spec valid for repeated cross-sections; `spec="within"` absorbs the unit
fixed effects (the treatment main effect is omitted - it is spanned by the
unit FE). Point estimates coincide across specs only in the restricted
equivalence case (balanced panel, no covariates, simultaneous adoption);
otherwise the unit-FE projection changes point estimates too. The reference
period defaults to the last pre-treatment period (e=-1 convention) with no
transition warning (the legacy-default FutureWarning is MultiPeriodDiD-only,
row M-007).

- **Note (staggered-adoption detection limit):** both specs assume
  simultaneous adoption (the MultiPeriodDiD scope, inherited verbatim). The
  staggered-adoption advisory infers adoption timing from within-unit 0→1
  transitions of the treatment column - i.e. it can only fire on a
  time-varying `D_it` input, which the companion time-varying-treatment
  warning already flags as off-contract. Under the documented time-invariant
  ever-treated `D_i` contract, adoption timing is not observable in
  `(y, D_i, unit, time)` at all, so simultaneous adoption cannot be verified
  by the estimator and is asserted by the user (like parallel trends). An
  optional cohort-timing validation input (`first_treat=`/`cohort=`) that
  would make the assumption checkable under `D_i` is tracked in TODO.md.
- **Note (explicit calendar partition):** for the same reason, `post_periods=`
  is REQUIRED (non-empty) in event-study mode: the boundary cannot be
  inferred from the data, so MultiPeriodDiD's midpoint default (last half of
  the calendar) is a silent guess and is deliberately not carried into the
  merged mode - omission raises with a message naming the rejected default.
  MultiPeriodDiD itself keeps the documented midpoint default through 3.9.

*Inference:*
- **Note:** the event-study mode auto-clusters at the unit level from 3.9
  (decision 2026-08-07; new API adopts the section-7 end-state policy
  immediately), mirroring the static carve-outs lane-for-lane: the
  auto-cluster is silently dropped on the Conley path (only explicit
  `cluster=` combines), never injected as a survey PSU (the
  implicit-per-observation-PSU rule), and dropped for explicit one-way
  analytical families (`classical`/`hc2`). Explicit `cluster=` passes
  through and behaves exactly like MultiPeriodDiD's own explicit cluster on
  every lane (survey PSU injection included).
- **Note:** `inference="wild_bootstrap"` in event-study mode raises
  `ValueError` from the mode's 3.9 birth - the WCR implementation covers
  the static ATT only, and MultiPeriodDiD's silent analytical fallback is
  deliberately not carried into the merged mode (no-silent-failures). This
  mode-level error fires BEFORE the survey/Conley front doors (contrast the
  static precedence contract in the Conley section).

**Key implementation requirements:**

*Assumption checks / warnings:*
- **Staggered treatment warning**: If treatment timing varies across units, warns about potential bias from negative weights (Goodman-Bacon 2021, de Chaisemartin & D'Haultfœuille 2020)
- Requires sufficient within-unit and within-time variation
- Warns if any fixed effect is perfectly collinear with treatment

*Estimator equation (as implemented):*
```
Y_it = α_i + γ_t + τ(D_it) + X'β + ε_it
```
Estimated via within-transformation (demeaning):
```
Ỹ_it = τD̃_it + X̃'β + ε̃_it
```
where tildes denote demeaned variables.

**Note:** The interaction term `D_i × Post_t` is within-transformed (demeaned) alongside the
outcome and covariates before regression. This is required by the Frisch-Waugh-Lovell theorem:
all regressors must be projected out of the same fixed effects space as the dependent variable.
This matches the behavior of R's `fixest::feols()` with absorbed FE.

*Standard errors:*
- Default: Cluster-robust at unit level (accounts for serial correlation)
- Degrees of freedom adjusted for absorbed fixed effects: component-aware rank via
  `diff_diff.utils.absorbed_fe_rank` — equal to the historical
  `n_units + n_times - 2` on a connected panel, and `sum(levels) - C - 1` in
  general, where `C` is the number of connected components of the unit x time
  incidence graph (Abowd-Creecy-Kramarz).
- **Note:** the pre-2026-07 count `sum_d (levels_d - 1)` over-stated absorbed rank
  in two supported cases: disconnected panels (the FE are not jointly identified
  across components, true rank `sum(levels) - C`) and nested/hierarchical
  dimensions (e.g. `absorb=["state", "state_year"]`, where each parent level is
  its own component: measured 6 states x 5 years gives true rank 29 vs the old
  count 34). On those inputs the reported residual df was too small and the
  non-clustered classical/hc1 SEs too large; connected panels are bit-identical.
  Levels and connectivity are evaluated over POSITIVE-weight rows only, per the
  zero-weight-padding inference-invariance guarantee (see "Weight Type Effects on
  Inference"); the old all-rows `nunique()` count violated that guarantee on
  weighted fits with inert rows. For 3+ absorbed dimensions the historical
  `sum(levels - 1)` form is retained (exact for independent connected dims;
  the general N-way rank is tracked in TODO.md). Full measured inventory:
  `docs/methodology/variance-conventions.md`.
- **Note (absorbed-FE variance scale = fixest full-K):** for the *non-clustered* `classical`
  and `hc1` (hetero) variance families, the finite-sample scale (`sse/(n-k)` / `n/(n-k)`) now
  counts the absorbed FE in `k` -- i.e. `K_full = k_visible + df_adjustment` -- matching
  `fixest::feols(vcov="iid"/"hetero")` and the reported t-`df` (`linalg._absorbed_fe_vcov_scale`,
  a single scalar rescale of the `k_visible` vcov, fail-closed when `n - K_full <= 0`).
  **Note (deviation from R default — irregular FE designs):** with the 2026-07
  component-aware `absorbed_fe_rank`, `df_adjustment` on disconnected or
  nested/hierarchical two-way designs is the EXACT dummy-space rank
  (`sum(levels) - C`), which matches `fixest::feols(..., ssc = ssc(K.exact = TRUE))`
  at machine precision but deviates from the fixest DEFAULT
  (`ssc(K.exact = FALSE)`), whose approximate `sum(levels - 1)` count reproduces the
  library's own pre-2026-07 over-count on exactly those designs. Measured on the
  committed hierarchical golden (`benchmarks/data/fixest_kexact_golden.json`,
  6 states x 5 nested state-years, C = 6): fixest default `df.K = 36` / iid SE
  0.115064780324081 vs `K.exact = TRUE` `df.K = 31` / iid SE 0.111785906348027;
  the library matches the exact side to <= 1e-12
  (`tests/test_variance_conventions.py::TestFixestKExactParity`). Exact rank is
  chosen deliberately: `K.exact` exists in fixest precisely because its default is
  an approximation, and the approximate count misstates the residual df the
  reported t-distribution uses. Connected, independent designs are identical under
  both conventions.
  Previously the within-transform SE used `k_visible`, sitting ~6.5% below fixest even though
  the t-`df` already used `K_full` (an internal inconsistency). Applies to
  `TwoWayFixedEffects(vcov_type="classical")`, `DifferenceInDifferences(absorb=..., vcov_type in {classical,hc1})`,
  and `MultiPeriodDiD(absorb=..., vcov_type in {classical,hc1})`. **Clustered** CR1 SEs converge
  on the same accounting (3.9, K_reference): the finite-sample factor's `k` is
  `K_reference = k_visible + (1 for the absorbed constant when X carries no intercept column)
  + rank(absorbed FE not nested in the cluster | nested ones)`, computed by
  `absorbed_fe_cr1_k_increment` and threaded as the signed `cluster_k_adjustment` seam through
  `solve_ols`/`compute_robust_vcov`/`LinearRegression.fit`/`wild_bootstrap_se`. This reproduces
  Stata reghdfe 3.2.9 (via `jwdid`) at machine precision on the committed mpdta arms and the
  G≈20..500 subsample ladder (`benchmarks/data/etwfe_cs_stata_golden.json`), and fixest's
  nested-FE convention (`ssc(K.fixef = "nested")`) on connected designs
  (`tests/test_fixest_did_twfe_parity.py`, tightened to ~1e-9). Nested-in-cluster FE are
  dropped exactly as both references do. **Note (deviation from R, exact non-nested rank):**
  the non-nested term is the exact RANK given the nested set for one- and two-dimensional
  absorbed sets, extending the D3/K.exact principle above; for THREE or more absorbed dims
  both ranks inherit the documented D3 `sum(levels) − N + 1` approximation (exact for
  independent connected dims, an over-count for duplicated/nested triples — same limitation
  note as D3, tracked in `TODO.md`; pinned by
  `tests/test_variance_conventions.py::TestClusterKIncrement::test_three_dim_increment_inherits_the_documented_d3_approximation`).
  On connected designs the exact two-way rank equals the references' per-dim count; on
  DISCONNECTED designs it does not, and NO external reference implements the exact
  composition — fixest's `ssc(K.exact=TRUE)` composes incoherently with its nested drop, and
  reghdfe's pairwise correction skips pairs containing a nested-dropped dim, so both land one
  df higher per extra component. The deviation is pinned as an exact one-df SE ratio against
  BOTH references (`benchmarks/data/fixest_cr1_nonnested_golden.json`,
  `benchmarks/data/reghdfe_kref_golden.json`;
  `tests/test_variance_conventions.py::TestFixestCr1NonNestedParity` /
  `TestReghdfeKReferenceParity`), while the crossed-cluster (nothing-nested) arm matches both
  references at machine precision. This is distinct from the TAIL-df convention, which
  converged separately in 3.9 (M-127: the three-value `df_convention` knob below — the
  SE-scale `k` here and the inference df are independent accountings). `hc2`/`hc2_bm`
  use leverage / Satterthwaite DOF and are unaffected.
- **Note (deviation from R — clustered CR1 inference df):** under unit clustering, Python's
  CR1 t-statistics/p-values/CIs use the **residual df** from the fitted design
  (`n − K_full`, e.g. 148 on the live-R benchmark panel: n=200, 50 unit FE + time + treatment),
  while `fixest::feols(cluster=...)` uses the **cluster df** `G − 1` (49 on the same panel;
  Stata's `min(G−1)` convention). Both sides use t-distributions. At moderate |t| the CIs and
  p-values are close; at large |t| the tails diverge by orders of magnitude (measured on the
  live-R benchmark: |t|≈22 → Python `p≈1e-48` at t(148) vs fixest `p≈4e-27` at t(49) — both
  numerically zero, which is what `tests/test_methodology_twfe.py::test_pvalue_matches_r_twfe`
  pins; the convention itself is locked at a distinguishable moderate |t|~1.8 by
  `tests/test_methodology_twfe.py::test_moderate_t_pins_residual_df_convention`, where the
  t(148) and t(49) tails differ by ~5%). The SE itself matches fixest under the K_reference
  convergence above (the historical ~0.25% non-nested-FE band is closed). **Three-value knob
  (2026-07, widened 3.9 / M-127):** `df_convention ∈ {"residual", "cluster", "normal"}` on the
  library-wide knob roster — `DifferenceInDifferences` / `TwoWayFixedEffects` /
  `MultiPeriodDiD` / `LinearRegression` plus (since 3.9) `SunAbraham`, `WooldridgeDiD` (OLS
  arms), `StackedDiD`, `ImputationDiD` (pretrends lead regression), and `LPDiD` (default
  `"cluster"`, its Stata-anchored convention). `"cluster"` switches clustered analytical
  t/p/CI to the Stata/fixest `G − 1` convention and is inert on unclustered and conley fits;
  `"normal"` is deliberate normal-theory z at the fallback level on EVERY fit (clustered,
  unclustered, and conley alike — on lanes that were already t it is a new deliberate option,
  not a compatibility path). Fallback-level only under every value: survey/replicate df and
  per-coefficient Bell-McCaffrey DOF keep precedence, and point estimates/SEs/t-statistics
  never move. The default remains `"residual"` on every surface except LPDiD and flips at v4
  (lifecycle: `docs/v4-deprecations.yaml` M-004..M-006 + M-128..M-131; work row in
  `DEFERRED.md` → Version-gated (v4); flipping moves every clustered p-value/CI, so it is a
  major-version change). **Out of the knob's scope** (own inference stacks): the IF-based
  family — `CallawaySantAnna`, `TwoStageDiD`, `ChaisemartinDHaultfoeuille`, `EfficientDiD`,
  `ContinuousDiD`, `ChangesInChanges`, `SyntheticDiD`, `HeterogeneousAdoptionDiD` — and the
  OLS-based `TripleDifference` / `TROP` (own `safe_inference` df conventions, outside the
  variance-conventions audit matrix's shared-CR1 scope), plus `WooldridgeDiD`'s
  logit/poisson arms (survey df or normal theory, knob-independent). Locked by
  `tests/test_estimators_vcov_type.py::TestDfConvention` (G−1 tail match, precedence, no-op default). The full-dummy (`fixed_effects=`) idiom carries
  `df_adjustment == 0` and is unchanged — its residual t-df already matched fixest's
  full-K count because every FE column sits in `k_visible`. **That claim is scoped to
  the residual df and the non-clustered variance families; the clustered CR1
  finite-sample factor instead applies the negative nested-rank adjustment** — see
  the deviation note below.
- **Deviation from R (clustered CR1, full-dummy `fixed_effects=` path):** with
  `fixed_effects=[unit, time]` and `cluster=unit`, the CR1 factor subtracts the
  rank of the cluster-NESTED FE dummies from the design rank (audit panel:
  `k = 66 → K_reference = 7`), so the two documented-equivalent idioms return
  IDENTICAL clustered SEs
  (`tests/test_variance_conventions.py::test_d1_convergence_is_pinned`; the
  historical 10.35% split, `sqrt((360−2)/(360−66))`, is closed). This is a
  deliberate deviation from a LITERAL R comparison in the other direction:
  fixest's `ssc(K.fixef = "nested")` drops nested ABSORBED FE only, not nested
  explicit dummy REGRESSORS — `feols(y ~ ... + i(unit) + i(time), cluster = ~unit)`
  and `lm()` + clubSandwich CR1S both count all 66 columns where K_reference drops
  the nested 59. No external reference implements K_reference on explicit dummies,
  so this side is anchored by (a) the `absorb=` side's external parity (Stata
  jwdid/reghdfe + fixest arms) and (b) the internal equality gate — two
  documented-equivalent spellings of the same model must return the same SE (the
  D1 equivalence, which also holds under collinearity drops and on disconnected
  designs because the kernel `k` is the design RANK). Full map:
  `docs/methodology/variance-conventions.md` (D1/D2, fixed 3.9).

*Edge cases:*
- Singleton units/periods are RETAINED, not dropped (both the static path and
  the event-study mode, which shares the class inference stack). A singleton
  unit's within-demeaned row is identically zero, so point estimates are
  unchanged by its presence; it does count toward `N`, the cluster count `G`,
  and residual df, so CR1/finite-sample SEs shift slightly (measured
  `0.41019 -> 0.40962` on a 20-unit static fixture with one added
  single-observation control).
  - **Deviation from R:** Stata's `reghdfe` iteratively DROPS singleton
    groups by default; R's `fixest::feols` retains them (a console notice
    only, unless `fixef.rm` is escalated). diff-diff matches the fixest
    default. An opt-in reghdfe-style pruning knob is tracked in TODO.md -
    changing the default would move published SEs and needs its own
    reviewed change.
- Treatment perfectly collinear with FE raises error with informative message listing dropped columns
- Covariate collinearity emits warning but estimation continues (ATT still identified)
- Rank-deficient design matrix: warns and sets NA for dropped coefficients (R-style, matches `lm()`)
  - **Note (scale invariance):** shared `diff_diff/linalg.py` behavior — rank detection re-checks on column-equilibrated columns and the solve equilibrates/unscales, so detection and fit are invariant to per-column scaling. For a well-scaled collinear design the dropped column is unchanged; a scale-induced under-count adopts the scale-corrected equilibrated selection (which may differ from the raw choice but retains an identified subset). The common full-rank case is short-circuited by a strictly-stricter Gram certification (2026-07) that leaves all drop decisions unchanged. See the CallawaySantAnna rank-deficiency Note.
  - **Note (covariate-name collision guard):** a covariate named `const`, `ATT`, `_treatment_post`, or a unit/time fixed-effect dummy name — or a duplicate covariate name — raises `ValueError` on both variance paths (would otherwise silently overwrite a structural coefficient on the full-dummy HC2/HC2-BM path). See the DifferenceInDifferences "covariate-name collision guard" Note.
- Unbalanced panels handled via proper demeaning
  - **Note (iterative demeaning):** the two-way within transformation
    (`diff_diff.utils.within_transform`) uses the method of alternating projections
    (iteratively demean by unit, then time, until convergence) for BOTH the weighted
    and unweighted paths. This is exact on unbalanced panels. The unweighted path
    previously used the closed-form additive demean `y - ȳ_i - ȳ_t + ȳ`, which is
    exact only for balanced fully-crossed panels; on unbalanced panels it was a
    biased approximation. Balanced-panel results are unchanged to machine precision.
- Multi-period `time` parameter: only binary (0/1) post indicator is recommended; multi-period values
  produce `treated × period_number` rather than `treated × post_indicator`. A `UserWarning` is
  emitted when `time` has >2 unique values, advising users to create a binary post column.
  Non-{0,1} binary time (e.g., {2020, 2021}) also emits a warning, though the ATT is mathematically
  correct — the within-transformation absorbs the scaling.
- Staggered warning limitation: requires `time` to have actual period values (not binary 0/1)
  so that different cohort first-treatment times can be distinguished. With binary `time="post"`,
  all treated units appear to start at `time=1`, making staggering undetectable. Users with
  staggered designs should use `decompose()` or `CallawaySantAnna` directly.

**Reference implementation(s):**
- R: `fixest::feols(y ~ treat:post | unit + post, data, cluster = ~unit)`
- Stata: `reghdfe y treat, absorb(unit time) cluster(unit)`

**Requirements checklist:**
- [ ] Staggered adoption detection warning (only fires when `time` has >2 unique values; with binary `time`, staggering is undetectable)
- [x] Multi-period time warning (fires when `time` has >2 unique values)
- [x] Auto-clusters standard errors at unit level
- [x] `decompose()` method returns BaconDecompositionResults
- [x] Within-transformation correctly handles unbalanced panels
- [x] Non-{0,1} binary time warning (fires when time has 2 unique values not in {0,1})
- [x] ATT invariance to time encoding (verified by test)

---

# Modern Staggered Estimators

## IF-based variance estimators vs analytical-sandwich estimators

diff-diff houses two structural families for variance computation, and the
distinction governs which `vcov_type` values an estimator can accept:

**Analytical-sandwich estimators** fit a single (or per-cohort) linear
regression and derive variance via `solve_ols(..., vcov_type=...)`, returning
a sandwich `(X'X)^{-1} M (X'X)^{-1}` whose meat `M` is parameterized by
`vcov_type ∈ {classical, hc1, hc2, hc2_bm}` (plus `conley` for spatial-HAC).
Examples: `DifferenceInDifferences`, `MultiPeriodDiD`, `TwoWayFixedEffects`,
`SunAbraham`, `StackedDiD`, `WooldridgeDiD`, `LinearRegression`. The full
`vcov_type` contract is methodologically applicable because every family has
a defined interpretation on the hat-matrix-bearing design (HC2 leverage
`1/(1-h_ii)`, Bell-McCaffrey Satterthwaite DOF, etc.).

**IF-based estimators** derive variance from an asymptotic influence function
`Var(θ̂) = (1/n) Σ_i ψ_i²` per estimator-specific derivations (Callaway &
Sant'Anna 2021 for `CallawaySantAnna`; Borusyak-Jaravel-Spiess 2024 for
`ImputationDiD`; Sant'Anna & Zhao 2020 for `EfficientDiD`; Ortiz-Villavicencio
& Sant'Anna 2025 for `TripleDifference`, where the variance is built on the
3-pairwise-DiD decomposition `inf = w3·IF_3 + w2·IF_2 - w1·IF_1`). For these:

- `hc1` with `cluster=None` ≡ per-unit IF variance — the default
  (Williams 2000 form).
- `hc1` with `cluster=X` ≡ CR1 Liang-Zeger on the IF:
  `Var = (G/(G-1)) Σ_c (Σ_{i∈c} ψ_i)² / n²`. The activation path is
  estimator-specific: `CallawaySantAnna` synthesizes `SurveyDesign(psu=X)`
  internally and routes through the shared PSU-meat machinery
  (`_compute_stratified_psu_meat`); `TripleDifference` computes the
  algebraically equivalent CR1 directly from cluster-summed IFs inline at
  `triple_diff.py` (no SurveyDesign synthesis — the IF is already in scope
  at the SE call site). Both produce the same numerical result.
- `classical`, `hc2`, `hc2_bm` are **N/A** for IF-based estimators —
  hat-matrix leverage and Bell-McCaffrey Satterthwaite DOF are defined on a
  single regression's design matrix, and IF-based estimators have no
  equivalent global hat matrix (they compose per-(g,t) or per-cohort fits
  with custom IF derivations). Rejected at `__init__` with
  methodology-rooted messages.
- `conley` (spatial-HAC) — could conceptually apply to the IF (spatial
  aggregation of per-unit IFs) but requires separate methodology work;
  deferred.

This split is a structural property of the estimator's variance derivation,
not a missing feature. The `vcov_type` input contract for IF-based estimators
is **permanently narrow** at `{"hc1"}`. Enforced today on
`CallawaySantAnna`, `TripleDifference`, `ImputationDiD`, `EfficientDiD`, and
`TwoStageDiD`. `TwoStageDiD` is sandwich-class rather than pure-IF — its
variance is the Gardner (2022) two-stage GMM sandwich — but reaches the same
narrow `{"hc1"}` contract for a meat-specific reason: the GMM-corrected score
`S_g = gamma_hat' c_g - X'_{2g} eps_{2g}` folds first-stage FE uncertainty into
the meat, so no single hat matrix spans both stages and `{classical, hc2,
hc2_bm}` have no derivation (see the TwoStageDiD section).

**Note:** This routing is a documented synthesis. The clustered-`hc1`
activation path is estimator-specific: `CallawaySantAnna` synthesizes
`SurveyDesign(psu=X)` internally and routes through the existing
PSU-meat machinery (`_compute_stratified_psu_meat`); `TripleDifference`
computes the algebraically equivalent CR1 directly from cluster-summed
IFs inline; `ImputationDiD` computes the Theorem 3 conservative variance
(`sigma_sq = (cluster_psi_sums**2).sum()`) directly from per-cluster
influence-function sums; `EfficientDiD` aggregates per-unit EIF within
clusters, centers, and applies the standard `G/(G-1)` correction
(`_cluster_aggregate` + `_compute_se_from_eif` in
`diff_diff/efficient_did_aggregation.py`). The CR1 Liang-Zeger algebra on the
IF is Williams (2000) / Hansen (2007) in all four cases — no new
methodology is introduced.

---

## CallawaySantAnna

**Primary source:** [Callaway, B., & Sant'Anna, P.H.C. (2021). Difference-in-Differences with multiple time periods. *Journal of Econometrics*, 225(2), 200-230.](https://doi.org/10.1016/j.jeconom.2020.12.001)

**Key implementation requirements:**

*Assumption checks / warnings:*
- Requires never-treated units as comparison group (identified by `first_treat=0` or `never_treated=True`)
- Warns if no never-treated units exist (suggests alternative comparison strategies)
- Limited pre-treatment periods reduce ability to test parallel trends
- **Note (rank-guarded IF standard errors):** The analytical SE paths invert the propensity-score Hessian (`H_psi`) and outcome-regression bread (`X'WX`) across every `(g, t)` cell via `_safe_inv()`. `np.linalg.solve`/`inv` raise `LinAlgError` only on *exactly* singular matrices; a **near**-singular Gram (a constant or collinear covariate) does **not** raise, so the prior code returned a garbage inverse (entries ~1e13) that flowed straight into the SE (`overall_se` ~1e13 was reproduced for `estimation_method="dr"`). `_safe_inv()` now delegates to the shared `_rank_guarded_inv()` (`diff_diff/linalg.py`): it symmetrically equilibrates `A → D^{-1/2} A D^{-1/2}` (`D = diag(A)`), then, when rank-deficient, inverts a **column-dropped** principal submatrix — keeping the most-independent columns via pivoted QR on the **equilibrated** Gram. This is a column-drop generalized inverse in the **same family** as the point estimate / R's `lm()` (drop redundant columns, *not* a minimum-norm pseudo-inverse), but the column *selection* is computed on the equilibrated Gram and is therefore **scale-invariant** by construction — so it need not pick the *same member* of a collinear set as the point estimate's raw-pivot `_detect_rank_deficiency` under mixed-scale *exact* collinearity (see the selection caveat below); the resulting SE is well-defined and order-invariant regardless of which redundant member is dropped. The `rcond = 1e-10` relative-eigenvalue threshold on the equilibrated Gram sets the rank (chosen because a Gram squares the condition number of `X`; matches EfficientDiD's `tol/max_eigval`). The result is a **finite SE on the identified covariate subset**; an all-NaN inverse (and NaN SE) is returned **only** on true rank-0. **Column-drop equals the full-rank limit:** using a column-drop generalized inverse (rather than a minimum-norm pseudo-inverse) makes the analytical SE equal the **well-conditioned near-collinear limit** — replacing an exactly-collinear covariate with a near-collinear (full-rank) one yields the same SE to working precision (verified `se_ratio ≈ 1` across `reg`/`ipw`/`dr`), for every per-cell bread (the PS Hessian — including unweighted ipw's since v3.7 — and the control- *and* treated-side OR breads, including reg's centered estimation-effect Gram `G` since v3.7). A minimum-norm pseudo-inverse would instead diverge sharply when the IF multiplier leaves `range(A)` — e.g. a control (or treated-sub-cell) bread multiplied by a mean from a cell where the covariate is *not* collinear — so it is rejected. With column-drop there is no such divergence; a covariate that is rank-deficient only within one cell still legitimately enters the other cells' full-rank fits, so the ATT and SE reflect that (poor) covariate specification, surfaced by the aggregate warning. **Selection caveat (equilibrated vs raw pivot):** because `_rank_guarded_inv` pivots the *equilibrated* Gram, the *member* it drops from a collinear set can differ from the point estimate's raw-pivot `_detect_rank_deficiency` choice **only** under mixed-scale *exact* collinearity (e.g. a covariate and its `1e8×` duplicate). This selection difference does **not** change the SE: dropping either member of an exactly-collinear pair leaves the identified subspace — hence the variance — unchanged, and the SE is order-invariant (verified for both column orders and under survey weighting). It is a documented, intentional deviation in the *generalized-inverse column choice* (equilibrated, scale-invariant) from the point-estimate's raw pivot; it is not a parity claim with the raw selection. **Caveat (point-estimate side, NOT the rank-guard):** CS/SDDD's reg/DR *point-estimate* OR fit now routes through the scale-equilibrated `solve_ols` (SVD/gelsd), so a covariate correlated with another regressor at a very large scale no longer perturbs the point-estimate ATT — or the IF SE that consistently follows it — under `reg`/`dr`; scale-invariance is pinned by tests (covariate + large constant offset → ATT(g,t) unchanged to ~1e-11). This is independent of the rank-guarded IF inverse, whose SE is invariant to which collinear member is dropped (see the selection caveat above). The well-conditioned fast path returns `np.linalg.solve(A, I)` unchanged (R-parity preserved). `fit()` emits ONE aggregate `UserWarning` (count + max condition number) reporting the dropped redundant direction(s); suppressed under `rank_deficient_action="silent"`. **`rank_deficient_action` enforcement:** `"error"` is enforced *upstream* at the point-estimate solve (`solve_ols` / `solve_logit` / the covariate-regression fit), which raises `ValueError` when the covariate **design** is rank-deficient at its `1e-7` threshold (`_detect_rank_deficiency`) **before** any influence-function SE is computed — so for a genuinely (design-)rank-deficient covariate the rank-guarded IF inverse is reached only under `"warn"`/`"silent"`. The IF guard truncates on a *stricter* relative threshold, though: `1e-10` on the **equilibrated Gram** vs `1e-7` on the design, and a Gram squares `X`'s condition number, so the IF guard drops a direction once `X`'s singular-value ratio falls below ~`1e-5` (= √`1e-10`), well above the design's `1e-7`. A cell that is *near-singular yet still full-rank by the upstream design check* (singular-value ratio between ~`1e-7` and ~`1e-5`) therefore passes the `"error"` gate without an exception and is **still** column-dropped by the IF guard — the guard does not re-raise, and (under `"error"`/`"warn"`, not `"silent"`) the aggregate warning still fires. So `"error"` blocks design-rank-deficient *covariates*; it does not promise that every near-singular IF bread/Hessian raises. Sibling of axis-A finding #17 in the Phase 2 silent-failures audit.

*Variance families (`vcov_type`, IF-based):*
- `hc1` (default, only accepted value) — per-unit IF variance per Callaway & Sant'Anna (2021) when `cluster=None`; cluster-robust CR1 Liang-Zeger on the IF when `cluster=X` is set (synthesizes `SurveyDesign(psu=X)` internally, threading through the same PSU machinery as explicit survey designs). When `survey_design=SurveyDesign(psu=Y)` is provided, the explicit PSU takes precedence; if `cluster=X` is also set with a different partition, emits a `UserWarning` (PSU wins).
- `classical`, `hc2`, `hc2_bm`, `conley` — REJECTED at `__init__`. The rejection is **library-architectural, not paper-prescribed**: analytical-sandwich variance families (`classical`, `hc2`, `hc2_bm`) are defined on a single regression's hat matrix, and CS's per-(g,t) doubly-robust / IPW / outcome-regression structure has no equivalent single design matrix to compute hat-matrix leverage or Bell-McCaffrey Satterthwaite DOF on. Spatial-HAC (`conley`) likewise has no defined composition with per-unit IF aggregation today. See ["IF-based variance estimators vs analytical-sandwich estimators"](#if-based-variance-estimators-vs-analytical-sandwich-estimators) above for the structural taxonomy.

*Cluster wiring:*
Prior to the bare-`cluster=` wiring fix, `CallawaySantAnna(cluster="X")` was a silent no-op — the parameter was stored at `__init__` but never consumed in the fit / aggregator / bootstrap pipeline (users got per-unit IF variance silently, even when they explicitly set `cluster="state"`). The fix synthesizes a minimal `SurveyDesign(psu=X, weight_type="pweight")` when bare `cluster=` is set without an explicit survey design, threading the synthesized PSU through the existing `_compute_stratified_psu_meat` aggregator (`staggered_aggregation.py:735-749`) and PSU-level multiplier bootstrap (`staggered_bootstrap.py:334-437`). Three-branch wiring at `staggered.py:~1500`:
  1. Bare `cluster=X` + no `survey_design` → synthesize `SurveyDesign(psu=X)`; refit `_resolve_survey_for_fit` on synthetic; `effective_survey_design = synthetic` so `_validate_unit_constant_survey` runs on it (preventing first-value-wins collapse for movers on panel data).
  2. `survey_design` without PSU + `cluster=X` → call `_inject_cluster_as_psu(resolved_survey, cluster_ids)`.
  3. `survey_design` with PSU + `cluster=X` → PSU wins; `_resolve_effective_cluster` emits `UserWarning` if partitions differ.

The `cluster_name` and `n_clusters` fields on `CallawaySantAnnaResults` report the effective clustering level: `survey_design.psu` (canonical column) when explicit PSU is provided, `self.cluster` when bare cluster synthesizes or injects.

- **Note (API decision — `cluster=` retained, NOT deprecated):** the `cluster=` → `SurveyDesign(psu=cluster)` synthesis above is an internal implementation detail, not a user-facing redundancy to be consolidated away. `cluster=` is the **canonical ergonomic single-level clustering kwarg** and is intentionally retained on `CallawaySantAnna` (and the sibling IF-based estimators `EfficientDiD` / `ImputationDiD` / `TwoStageDiD`): it matches the field's universal convention (R `fixest::feols(..., cluster = ~unit)`, Stata `vce(cluster id)`, statsmodels `cov_type="cluster"`), so users reach for `cluster=` first. `survey_design=SurveyDesign(psu=X, ...)` is the **advanced** entry point (adds strata / FPC / replicate weights / explicit weights); a bare `cluster=` is the shorthand for the common "just cluster at X" case and would be strictly less ergonomic if forced through `survey_design=`. This mirrors the HAD survey-API consolidation, which deprecated only the *redundant* `survey=` / `weights=` entry points in favor of `survey_design=` while deliberately keeping `cluster=`. Decision recorded 2026-07-04: do not deprecate `cluster=`; the former "decide whether to deprecate `CallawaySantAnna.cluster=X`" `TODO.md` row is closed as resolved (keep).

*Estimator equation (as implemented):*

Group-time average treatment effect:
```
ATT(g,t) = E[Y_t - Y_{g-1} | G_g=1] - E[Y_t - Y_{g-1} | C=1]
```
where G_g=1 indicates units first treated in period g, and C=1 indicates never-treated.

*Note:* This equation uses g-1 as the base period, which applies to post-treatment effects (t ≥ g) and `base_period="universal"`. With `base_period="varying"` (default), pre-treatment effects use the immediately-preceding observed period as base. Base periods are selected *positionally* (nearest observed period), so on gapped grids the base is the nearest observed period rather than literal g-1 / t-1 (see Base period selection in Edge cases).

With covariates (doubly robust):
```
ATT(g,t) = E[((G_g - p̂_g(X))/(1-p̂_g(X))) × (Y_t - Y_{g-1} - m̂_{0,g,t}(X) + m̂_{0,g,g-1}(X))] / E[G_g]
```

Aggregations:
- Simple: `ATT = Σ_{g,t} w_{g,t} × ATT(g,t)` weighted by group size
- Event-study: `ATT(e) = Σ_g w_g × ATT(g, g+e)` for event-time e
- Group: `ATT(g) = Σ_t ATT(g,t) / T_g` average over post-periods

*Standard errors:*
- Default: Analytical (influence function-based)
- **Note (v3.7 DRDID IF parity, panel reg/ipw):** Every panel reg/ipw per-cell SE is
  derived from the SAME influence function that feeds aggregation, the bootstrap, the
  cluster override, and `event_study_vcov`: `se = sqrt(sum(phi^2))` (DRDID convention;
  dr already worked this way). Two defects were fixed to get there. (1) The reg+cov IF
  omitted `DRDID::reg_did_panel`'s OLS estimation-effect term (`asy.lin.rep.ols %*% M1`)
  — mean-zero by the OLS normal equations, so ATTs matched R at ~1e-11 while per-cell
  SEs sat 4-13% and aggregated SEs 3-20% from the R golden fixtures (anti-conservative
  on some: 0.958x on the two-period golden). (2) The unweighted ipw+cov branch computed
  its per-cell SE as `sqrt(var_t/n_t + weighted_var_c)` where `weighted_var_c` was a
  weighted POPULATION variance never scaled by an effective sample size (~7x inflated
  per-cell SEs), and its IF lacked `std_ipw_did_panel`'s PS estimation-effect correction
  (`asy.lin.rep.ps %*% M2`; aggregated SEs ~2.4% off). The survey ipw branch already
  carried both (Phase 7a); the fix mirrors it. No-covariate reg/ipw per-cell SEs also
  switched from ddof=1 plug-ins (ipw's `var_c*(1-p)/(n_c*p)` matched R only at p=0.5)
  to the IF-based form, which is R's analytical SE exactly. Post-fix parity vs R:
  per-cell and aggregated ~5e-12 on the reg goldens, ~1e-10 vs fresh did 2.5.1 ipw
  aggregates (`tests/test_csdid_ported.py`,
  `tests/test_methodology_callaway.py::TestDRDIDPanelIFParity`). Point estimates are
  unchanged. Residual deviations kept: propensity scores are CLIPPED at
  `pscore_trim` (R drops at `trim.level=0.995`; differs only at extreme propensities);
  no-covariate ipw is treated as unconditional (R fits an intercept-only logit whose
  estimation effect is identically zero in the IF, so this is presentation-only).
  **Note:** decided document-only (2026-07-07) — the intercept-only logit is deliberately
  NOT mirrored structurally: its estimation-effect correction is identically zero at the
  MLE, so mirroring would add a per-cell IRLS solve (and its non-convergence failure
  surface) for zero numerical change. No-covariate ipw/reg/dr all reduce to the same
  difference-in-means IF, bit-identical per cell (locked by
  `tests/test_methodology_callaway.py::TestDRNoCovariateSEUniformity::test_ipw_no_cov_per_cell_identical_to_reg`).
  DR's no-covariate per-cell SE now also uses the IF-based `sqrt(sum(phi^2))` form (it
  had lagged on the ddof=1 plug-in `sqrt(var_t/n_t + var_c/n_c)`, O(1/n) from R): without
  covariates DR reduces to difference in means, so its per-cell SE is now bit-identical to
  the no-covariate reg path and matches R's analytical SE — point estimates and aggregated
  SEs are unchanged, since the same IF already fed aggregation
  (`tests/test_methodology_callaway.py::TestDRNoCovariateSEUniformity`).
  Side effect: reg/ipw fits with collinear covariates now route their IF breads
  through the rank-guarded inverse and fire the same aggregate warning as dr.
  Rank-0 semantics of the reg CENTERED Gram: because the intercept direction is
  handled analytically by the `1/sum(W_c)` leading term, a rank-0 centered Gram
  (zero within-control covariate variation, e.g. a constant as the only
  covariate) is the benign identified-subset case — the estimation-effect
  correction is exactly zero and the fit collapses to the no-covariate reg fit
  with finite SEs (`_centered_or_bread` maps `_safe_inv`'s all-NaN rank-0
  sentinel to a zero correction; the aggregate rank-guard warning still fires).
  This differs from the `[1, X]` breads (ipw PS Hessian, dr), where all-NaN is
  a true pathology that NaN-propagates.
- All aggregation SEs (simple, event study) include the weight influence function (WIF)
  adjustment, matching R's `did::aggte()`. The WIF accounts for uncertainty in estimating
  group-size aggregation weights. Group aggregation uses equal time weights (deterministic),
  so WIF is zero.
  - **Unbalanced panels — default within-cell differencing vs `allow_unbalanced_panel=True`:**
    on an unbalanced panel (some units unobserved in some periods) the **default** path estimates
    each ATT(g,t) by within-cell panel differencing on the units observed at BOTH the base period
    and t, and weights a multi-cell event-study horizon by per-cell valid `n_treated`. This is a
    valid but **different estimand** than R `did::att_gt(allow_unbalanced_panel=TRUE)`, which sets
    `panel=FALSE` and runs the repeated-cross-section levels estimator (`DRDID::reg_did_rc`) on the
    pooled observations, weighting by the fixed cohort probability `pg = n_g / N` over UNITS. Both
    the cell estimator AND the weighting differ from R on unbalanced data (the estimator choice
    dominates); on **balanced** panels they coincide exactly (each cell's valid count equals the
    cohort mass), so the default path matches R. The default path emits a `UserWarning` on
    unbalanced input (no-silent-failures) pointing to the flag.
  - **`allow_unbalanced_panel=True` (RC-on-panel parity with R):** routes an unbalanced panel's
    pooled observations through diff-diff's RC estimator (bit-exact vs `reg_did_rc`) and clusters
    the per-observation influence function by the original unit. ATT matches R **bit-for-bit** —
    cells AND dynamic aggregation, including the fixed unit-cohort-mass `pg` reweighting and the
    per-unit WIF (the per-observation WIF is divided by each unit's observation count so the
    unit-clustered sum is not over-counted). Inert on balanced panels (byte-identical to the
    default) — matching R, whose `pre_process_did` likewise recomputes balance and keeps
    `panel=TRUE` (differencing) on balanced input, so the flag engages RC only when the panel is
    genuinely unbalanced. Panel structure is validated before routing (no duplicate `(unit,
    period)` rows, time-invariant treatment cohort and cluster per unit), matching R's
    preprocessing — fail-closed since the RC precompute reads cohort/cluster per observation.
    `survey_design=` with the flag raises `NotImplementedError` (per-obs vs per-unit weight
    resolution deferred). Verified vs R `did` 2.5.1 in
    `tests/test_csdid_ported.py::TestAllowUnbalancedPanel` (golden
    `benchmarks/data/cs_unbalanced_golden.json`).
    - **Deviation from R:** the analytical SE equals R's up to the CR1 finite-sample factor
      `sqrt(G/(G-1))` (G = number of units): diff-diff's cluster-robust variance applies the
      `G/(G-1)` Bessel correction that R's `att_gt` `getSE` (`sqrt(mean(inf^2)/n)`) omits. Exact
      factor, ~0.25% at G=200, vanishing as G → ∞ — the same convention class as the fixest
      cluster-SE band (G2).
- Bootstrap: Multiplier bootstrap with Rademacher, Mammen, or Webb weights. Bootstrap
  perturbs the combined influence function (standard IF + WIF) directly, not just fixed-weight
  re-aggregation. This correctly propagates weight estimation uncertainty.
- **Note:** The combined-IF assembly (`_compute_combined_influence_function`) routes
  in-package callers through an O(n_units) fast path (v3.7): per-fit cohort tables
  (unit counts or survey-weighted masses via `np.unique` + `np.bincount`, cached on the
  precomputed structures with array-identity validation) and a closed-form WIF
  `wif_i = w_i * (E(c_i)/S - K(c_i) * d / S**2)` — algebraically identical to the prior
  dense `(n_units x n_gt)` `wif_matrix @ effects` (E(c)/K(c) = per-cohort effect
  sums/keeper counts, S = sum of keeper pg, d = pg @ effects, w_i = survey weight or 1).
  Point estimates are unaffected (bit-identical); aggregated SEs agree with the dense
  form only to floating-point reassociation (measured ≤ 5e-16 relative; drift-bound
  frozen-copy tests pin ≤ 1e-9, `tests/test_staggered_aggregation.py`), not
  bit-for-bit — same accumulation-order posture as the shared demeaning engine
  (see "Absorbed Fixed Effects with Survey Weights"). Units whose cohort is not
  among the keeper (g,t) groups get an exact 0 WIF contribution (the dense form
  realized the same value through cancelling terms). The pre-rewrite general path
  remains the fallback for direct callers with foreign index maps, non-numeric
  cohort dtypes, or absent precomputed structures; its per-cell IF scatter uses
  fancy `+=` (bit-identical to the prior `np.add.at` — index arrays are
  duplicate-free by construction at every producer), same mathematical contract.
- Block structure preserves within-unit correlation
- Simultaneous confidence bands (`cband=True`, default): Uses sup-t bootstrap to compute
  a uniform critical value across event times, controlling family-wise error rate. Matches
  R's `did::aggte(..., cband=TRUE)` default. Requires `n_bootstrap > 0`.

*Bootstrap weight distributions:*

The multiplier bootstrap uses random weights w_i with E[w]=0 and Var(w)=1:

| Weight Type | Values | Probabilities | Properties |
|-------------|--------|---------------|------------|
| Rademacher | ±1 | 1/2 each | Simplest; E[w³]=0 |
| Mammen | -(√5-1)/2, (√5+1)/2 | (√5+1)/(2√5), (√5-1)/(2√5) | E[w³]=1; better for skewed data |
| Webb | ±√(3/2), ±1, ±√(1/2) | 1/6 each | 6-point; recommended for few clusters |

**Webb distribution details:**
- Values: {-√(3/2), -1, -√(1/2), √(1/2), 1, √(3/2)} ≈ {-1.225, -1, -0.707, 0.707, 1, 1.225}
- Equal probabilities (1/6 each) giving E[w]=0, Var(w)=1
- Matches R's `did` package implementation
- **Verification**: Implementation matches `fwildclusterboot` R package
  ([C++ source](https://github.com/s3alfisc/fwildclusterboot/blob/master/src/wildboottest.cpp))
  which uses identical `sqrt(1.5)`, `1`, `sqrt(0.5)` values with equal 1/6 probabilities.
  Some documentation shows simplified values (±1.5, ±1, ±0.5) but actual implementations
  use square root values to achieve unit variance.
- Reference: Webb, M.D. (2023). Reworking Wild Bootstrap Based Inference for Clustered Errors.
  Queen's Economics Department Working Paper No. 1315. (Updated from Webb 2014)

*Edge cases:*
- Groups with single observation: included but may have high variance
- Non-estimable group-time cells: materialized as NaN entries in `group_time_effects` with a consolidated warning listing skip reasons and counts
  - **Note:** Non-estimable cells (missing base/post period, zero treated/control, zero survey-weight mass, non-finite regression solve) are stored as NaN entries — `effect`/`se`/`t_stat`/`p_value`/`conf_int` all NaN — carrying a machine-readable `skip_reason` code (`"missing_period"`, `"zero_treated_control"`, `"zero_weight_mass"`, `"non_finite_regression"`; estimable cells carry `None`). This is uniform across ALL estimation paths (no-covariate regression, covariate regression, IPW/DR, repeated cross-section, survey-weighted). A consolidated `UserWarning` is still emitted from `fit()`. The NaN cells are **excluded from every aggregation** (simple/overall, group, event-study), from `balance_e`, and from the bootstrap (they carry no influence-function entry, and all consumers finite-mask on `np.isfinite(effect)` or filter to IF members), so all aggregate point estimates and SEs — and `n_groups`/`n_periods` metadata — are **unchanged** from the prior omit behavior and match R `did`'s `aggte()` exactly. A fit where no cell is estimable (no finite effect) still raises a `ValueError`.
  - **Deviation from R:** R's `did::att_gt` omits non-estimable cells from its result table entirely; diff-diff materializes them as NaN rows (with `skip_reason`) so the `(g,t)` grid is inspectable via `group_time_effects` / `to_dataframe("group_time")`. This is a per-cell *surface* difference only — R's `aggte()` aggregation behavior is matched exactly (non-estimable cells contribute nothing to any aggregate).
  - **Note:** When `balance_e` is specified, cohorts with NaN effects at the anchor horizon are excluded from the balanced panel
- Anticipation: `anticipation` parameter shifts reference period
  - Group aggregation includes periods t >= g - anticipation (not just t >= g)
  - Both analytical SE and bootstrap SE aggregation respect anticipation
  - Not-yet-treated + anticipation: control mask uses `G > max(t, base_period) + anticipation`
    to exclude cohorts treated at either the evaluation period or the base period.
    This prevents control contamination when `base_period="universal"` and the base
    period is later than the evaluation period (e.g., pre-treatment ATT with universal base)
- Rank-deficient design matrix (covariate collinearity):
  - Detection: Pivoted QR decomposition with tolerance `1e-07` (R's `qr()` default), with a **column-equilibration re-check** (unit 2-norm) that makes the rank *count* invariant to per-column scaling; the dropped-column selection is unchanged for well-scaled collinear designs (a scale-induced under-count instead adopts the scale-corrected equilibrated selection)
  - Handling: Warns and drops linearly dependent columns, sets NA for dropped coefficients (R-style, matches `lm()`)
  - Parameter: `rank_deficient_action` controls behavior: "warn" (default), "error", or "silent"
  - **Note:** Rank detection and the least-squares solve are invariant to per-column scaling in BOTH the Python and Rust backends. The rank threshold is anchored to the largest pivot/singular value, so a column on a large raw scale (e.g. an unstandardized covariate in raw population or currency units) previously inflated the threshold and false-dropped the intercept/treatment/interaction to NaN on an otherwise full-rank design — or truncated the small-scale direction in the solve, returning finite-but-wrong coefficients. Detection now runs a raw pivoted QR first and only re-checks on column-equilibrated (unit 2-norm) columns when the raw pass reports a deficiency. If the equilibrated rank is higher (a scale-induced false-drop), the equilibrated rank AND its pivot selection are adopted — its first `rank` columns are independent under the scale-corrected criterion, so the retained design is guaranteed identified; otherwise (genuine collinearity, no scale disparity) the raw rank and pivot selection are kept unchanged. The solve equilibrates columns and unscales the coefficients. The Python (LAPACK `gelsd`) and Rust (faer thin-SVD) backends implement this same contract and agree at the parity-suite tolerances (coefficients ~1e-8, vcov ~1e-5), never bit-identically. This repairs the scale bug while leaving everything else unchanged: it is a no-op for full-rank well-conditioned designs (R-parity unaffected) and does **not** change which column is dropped in a *well-scaled* collinear design (the established raw pivot selection is preserved). A scale-induced under-count instead adopts the scale-corrected equilibrated selection — which may differ from the raw choice in a mixed scale+collinearity design but is guaranteed to retain an identified (full-rank) subset. This shared `diff_diff/linalg.py` behavior covers every covariate outcome-regression fit routed through `solve_ols` — DiD, TwoWayFixedEffects, MultiPeriodDiD, ImputationDiD, TwoStageDiD, and TripleDifference. **Scope (now scale-equilibrated):** CallawaySantAnna's covariate outcome-regression (`_compute_all_att_gt_covariate_reg`) and doubly-robust (`_doubly_robust`) nuisance solves — and StaggeredTripleDifference's per-cohort OR solve (`_compute_or`) — now route through `solve_ols` (column-equilibrated SVD/gelsd), matching TripleDifference's already-`solve_ols`-routed OR fit (`triple_diff.py:1438/1444`) and R's `lm()`/QR. The prior estimator-local `cho_solve(X'X)` / `scipy.linalg.lstsq(cond=1e-7)` fast paths were not scale-equilibrated: a covariate **correlated with another regressor at a very large scale** (e.g. a large constant offset, near-collinear with the intercept) could perturb the point-estimate ATT — and the IF SE that follows it — because the normal-equations Cholesky squares the condition number (pure orthogonal ill-scaling was already safe). The equilibrated SVD is offset-invariant to ~1e-11 where the prior solve drifted; scale-invariance is now pinned by tests (covariate + 1e6 offset → ATT(g,t) unchanged). The change is not bit-identical (cho/normal-equations → SVD) but well-scaled designs move only ~1e-12. The separate DR/OR influence-function SE rank-guard — which previously returned garbage SEs (~1e13) when these local Gram matrices were near-singular — is also **implemented** for CS / TripleDifference / StaggeredTripleDifference via `_rank_guarded_inv` (see the "rank-guarded IF standard errors" Note above).
  - **Note (opt-in solve_ols Cholesky fast path, 2026-07):** setting `DIFF_DIFF_SOLVE_OLS_FASTPATH=1` (positive integer; read per call) routes full-rank `solve_ols` fits through an equilibrated normal-equations Cholesky solve, certification-gated at reciprocal condition 1e-6 on the equilibrated Gram (Python: LAPACK `dpocon`, the same guard shape as the IRLS inner-solver Note under Propensity score estimation; Rust: the exact 1-norm rcond, whose inverse byproduct is reused as the sandwich bread). Any certification decline falls back VERBATIM to the SVD/gelsd solve, so the knob-off default path - and every rank-deficiency decision - stays byte-identical to the legacy behavior on both backends. The fast path never runs on a design the QR detector would call deficient: the 1e-6 solve guard is strictly inside the 1e-10 stage-0 rank certification, which is itself two orders stricter than the 1e-7 QR boundary. Within the opt-in path, parity vs the default solve is tol-bounded, not bit-level (fitted values ~1e-8 abs, SEs ~1e-6 rel; certified forward-error budget ~eps*cond(G_eq) <= 2e-10 in the equilibrated basis). The default path keeps the SVD's independent singular-value truncation as defense-in-depth; the knob is opt-in per the 2026-07 correctness-first decision.
- Non-finite inference values:
  - Analytic SE: Returns NaN to signal invalid inference (not biased via zeroing)
  - Bootstrap: Drops non-finite samples, warns, and adjusts p-value floor accordingly. SE, CI, and p-value are all NaN if the original point estimate is non-finite, SE is non-finite or zero (e.g., n_valid=1 with ddof=1, or identical samples)
  - Threshold: Returns NaN if <50% of bootstrap samples are valid
  - Per-effect t_stat: Uses NaN (not 0.0) when SE is non-finite or zero (consistent with overall_t_stat)
  - **Note**: This is a defensive enhancement over reference implementations (R's `did::att_gt`, Stata's `csdid`) which may error or produce unhandled inf/nan in edge cases without informative warnings
- No post-treatment effects (all treatment occurs after data ends):
  - Overall ATT set to NaN (no post-treatment periods to aggregate)
  - All overall inference fields (SE, t-stat, p-value, CI) also set to NaN
  - Warning emitted: "No post-treatment effects for aggregation"
  - Individual pre-treatment ATT(g,t) are computed (for parallel trends assessment)
  - Bootstrap runs for per-effect SEs even without post-treatment; only overall statistics are NaN
  - **Principle**: NaN propagates consistently through overall inference fields; pre-treatment effects get full bootstrap inference
- Aggregated t_stat (event-study, group-level):
  - Uses NaN when SE is non-finite or zero (matches per-effect and overall t_stat behavior)
  - Previous behavior (0.0 default) was inconsistent and misleading
- Base period selection (`base_period` parameter):
  - **Positional (sorted-index) selection** (`_select_base_period`): base periods are
    selected by *position* in the sorted list of observed periods, not by literal
    calendar arithmetic. On consecutive grids this reduces to `t-1` / `g-1-anticipation`;
    on **gapped (non-consecutive) grids** (e.g. biennial surveys, skipped years) the base
    is the nearest observed period, so R-estimable cells that literal `t-1` / `g-1` would
    NaN are estimated. The pre/post split is on the current period vs the cohort
    (`t < g` -> pre), **independent of anticipation**; only the post/universal base uses
    anticipation. This matches R `did::att_gt()` (verified against a deparse of `did` 2.5.1
    `compute.att_gt`) and resolves the prior internal inconsistency with the library's own
    dCDH estimator, which already used positional neighbors.
  - "varying" (default): pre-treatment (`t < g`) uses the immediately-preceding observed
    period as base (consecutive comparisons); post-treatment uses the last observed
    pre-treatment period (largest observed `p` with `p + anticipation < g`) as a long
    difference. Pre-treatment tests check nearest-observed-period comparisons.
  - "universal": ALL effects (pre and post) are long differences from the last observed
    pre-treatment period. Pre-treatment coefficients test cumulative divergence from it.
  - Both produce identical post-treatment ATT(g,t); differ only pre-treatment
  - `anticipation` shifts the post/universal base to the last observed `p` with
    `p + anticipation < g`, moving it further from treatment and strengthening the
    parallel trends assumption (it does not change the pre/post split).
  - Matches R `did::att_gt()` base_period parameter, including on gapped panels (base
    selection, estimable ATT/SE cells, zero reference cells, and all aggregations).
  - **Event study output**: With "universal", each cohort's positional base contributes a
    zero reference row at `e = base - g` (effect=0, se=NaN, conf_int=(NaN, NaN)); on a
    consecutive grid these coincide at `e = -1-anticipation`, on a gapped grid they can fall
    at `e = -2, -3, …`. Inference fields are NaN since this is a normalization constraint, not
    an estimated effect (see the full-R-parity note below).
  - **Universal-mode zero reference cells (full R parity):** with `base_period="universal"`,
    R `did::att_gt()` materializes each cohort's base reference period as a zero cell
    (`att = 0`, `se = NA`) in its `att_gt` table and includes it in `aggte(type="dynamic")`.
    diff-diff now does the same: `fit()` materializes each cohort's positional base as a zero
    reference cell (`att = 0`, `se = NaN`, zero influence function, flagged `is_reference`) in
    `group_time_effects` / `to_dataframe("group_time")` **at its positional base event time**
    (`e = base - g`, which on gapped grids can be `-2`, `-3`, … not just `-1`). Because the
    reference is a real cell, every consumer weights it uniformly — the **event-study dynamic
    aggregation**, the **multiplier bootstrap**, and `balance_e` — matching R exactly, including
    the overlapping-reference case where a cohort's zero base shares an event time with another
    cohort's estimated pre-trend cell (the reference correctly dilutes that horizon; verified vs
    `did` 2.5.1 on gapped **balanced** universal panels to ~1e-5 for the analytical AND bootstrap
    paths). The reference weight is the fixed cohort mass (R's `pg` numerator), so on unbalanced
    panels the general per-cell multi-cell-horizon weighting deviation documented above applies to
    the reference exactly as it does to any real cell (the reference is not special).
    Reference-only horizons report `att = 0, se = NaN`. The zero reference carries no influence
    function, so it adds nothing to any variance; the `group` / `simple` aggregations use
    post-treatment cells only (`t >= g - anticipation`) and exclude it. Regression-guarded by
    `tests/test_csdid_ported.py::TestCSDIDPositionalBasePeriod`.
- Base period interaction with Sun-Abraham comparison:
  - CS with `base_period="varying"` produces different pre-treatment estimates than SA
  - This is expected: CS uses consecutive comparisons, SA uses fixed reference (e=-1-anticipation)
  - Use `base_period="universal"` for methodologically comparable pre-treatment effects
  - Post-treatment effects match regardless of base_period setting
- Propensity score estimation:
  - Algorithm: IRLS (Fisher scoring), matching R's `glm(family=binomial)` default
  - **Note:** Uses IRLS (Fisher scoring) for propensity score estimation, consistent
    with R's `did::att_gt()` which uses `glm(family=binomial)` internally
  - **Note (IRLS inner solver, 2026-07):** each IRLS iteration's weighted
    least-squares step is solved via equilibrated normal equations + Cholesky with
    an explicit reciprocal-condition guard (LAPACK `dpocon`; guard 1e-6), falling
    back to the exact legacy tall-matrix `lstsq` solve for any iteration whose
    normal matrix cannot be certified well-conditioned (near-separation fits, where
    working weights crush a column's effective scale). The IRLS ALGORITHM — working
    weights `mu*(1-mu)`, working response, `tol=1e-8` on the raw-basis coefficient
    change, `max_iter=25` — is unchanged; both inner solvers converge to the same
    MLE, and measured coefficients move at the ~1e-15 level (tol-bounded worst case
    ~1e-8 if an iteration count ever shifts by one; the R-golden ipw SE pin at
    1e-6 abs is unaffected). The per-fit fallback count is exposed via
    `diagnostics_out["irls_chol_fallback_iters"]` (0 on well-conditioned fits).
  - Near-separation detection: Warns when predicted probabilities are within 1e-5
    of 0 or 1, or when IRLS fails to converge
  - Trimming: Propensity scores clipped to `[pscore_trim, 1-pscore_trim]` (default
    0.01) before weight computation. Warning emitted when scores are trimmed.
  - **Events Per Variable (EPV) diagnostics:** Per-cohort EPV =
    min(n_treated, n_control) / n_covariates checked before IRLS.
    Default threshold: 10 (Peduzzi et al. 1996). Warns when EPV < threshold;
    errors when `rank_deficient_action="error"`. Pre-estimation check via
    `diagnose_propensity()`. Results stored in `results.epv_diagnostics`.
  - Fallback: Controlled by `pscore_fallback` parameter (default `"error"`).
    If IRLS fails entirely (LinAlgError/ValueError) and `pscore_fallback="error"`,
    the error is raised. If `pscore_fallback="unconditional"`, falls back to
    unconditional propensity score with warning. For IPW, this effectively
    drops all covariates. For DR, the propensity model is unconditional but
    the outcome-regression component still uses covariates.
  - **Note:** `pscore_fallback` default changed from unconditional to error.
    Set `pscore_fallback="unconditional"` for legacy behavior.
  - **Note:** When `pscore_fallback="unconditional"` triggers, the propensity-
    score influence function correction is skipped (constant pscore has zero
    estimation uncertainty). SEs reflect outcome-model uncertainty only.
- Control group with `control_group="not_yet_treated"`:
  - Always excludes cohort g from controls when computing ATT(g,t)
  - This applies to both pre-treatment (t < g) and post-treatment (t >= g) periods
  - For pre-treatment periods: even though cohort g hasn't been treated yet at time t, they are the treated group for this ATT(g,t) and cannot serve as their own controls
  - Control mask: `never_treated OR (first_treat > max(t, base_period) + anticipation AND first_treat != g)`
  - The `max(t, base_period)` ensures controls are untreated at both the evaluation period
    and the base period, preventing contamination when `base_period="universal"` uses
    a base period later than `t` (matching R's `did::att_gt()`)
  - Does not require never-treated units: when all units are eventually treated,
    not-yet-treated cohorts serve as controls for each other (requires ≥2 cohorts)
- **Note:** CallawaySantAnna survey support: weights, strata, PSU, and FPC are all supported for all estimation methods (reg, ipw, dr) with or without covariates. Analytical (`n_bootstrap=0`): aggregated SEs use design-based variance via `compute_survey_if_variance()`. Bootstrap (`n_bootstrap>0`): PSU-level multiplier weights replace analytical SEs for aggregated quantities. ALL covariate methods carry the DRDID panel nuisance IF corrections, method-uniformly across survey and unweighted paths (Phase 7a added the PS IF correction via survey-weighted Hessian/score and the DR OR IF correction via WLS bread and gradient, Sant'Anna & Zhao 2020, Theorem 3.1; v3.7 extended the PS correction to unweighted ipw and the OR estimation-effect correction to reg on both paths). Survey weights compose with IPW weights multiplicatively. WIF in aggregation matches R's did::wif() formula. Per-unit survey weights are extracted via `groupby(unit).first()` from the panel-normalized pweight array; on unbalanced panels the pweight normalization (`w * n_obs / sum(w)`) preserves relative unit weights since all IF/WIF formulas use weight ratios (`sw_i / sum(sw)`) where the normalization constant cancels. Scale-invariance tests pass on both balanced and unbalanced panels.
- **Note (deviation from R):** Panel DR control augmentation is normalized by treated mass (`sw_t_sum` or `n_t`) rather than control IPW mass (`sum(w_cont)`). R's `DRDID::drdid_panel` uses `mean(w.cont)` as the control normalizer. Both are consistent asymptotically (under correct model specification, `E[w_cont] = E[D]` so the normalizers converge), but they differ in finite samples when IPW reweighting doesn't perfectly balance. The treated-mass normalization is simpler and matches the `did::att_gt` convention where ATT is defined per treated unit. Aligning to `DRDID::drdid_panel`'s exact `w.cont` normalization is deferred.
- **Note:** PS nuisance IF corrections follow DRDID's M-estimation convention: `asy_lin_rep_psi` is computed on O(1) psi scale (matching R's `asy.lin.rep.ps = score %*% Hessian.ps`), then the correction `asy_lin_rep_psi @ M2` is converted to the library's O(1/n) phi convention via a single `/n` division. OR corrections use the same phi-scale pattern via `solve(X'WX)` (unnormalized Hessian).
- **Note:** CallawaySantAnna panel reg+covariates per-cell IF (survey AND unweighted) carries `DRDID::reg_did_panel`'s full OLS estimation-effect term (v3.7). The treated IF is unchanged: `inf_treated_i = (sw_i/sum(sw_treated)) * (resid_i - ATT)` (unweighted collapse `(resid-ATT)/n_t`). The control IF is `inf_control_i = -(sw_i * wls_resid_i) * proj_i` where `proj_i = 1/sum(sw_control) + (x_i - x̄_c)' G^{-1} (x̄_t - x̄_c)` with `G` the centered (weighted) control Gram and `x̄_t`/`x̄_c` the (weighted) treated/control covariate means — algebraically identical to R's `asy.lin.rep.ols %*% M1 / mean(w.treat)` but evaluated in the centered basis, which is offset-invariant (the raw Gram squares the design's conditioning; a large constant covariate offset would otherwise push the equilibrated rank check below threshold and silently truncate a genuine direction). The intercept-only collapse is exactly `-resid_c/n_c`. HISTORY: the pre-v3.7 IF used the raw-residual plug-in (`-sw_c_norm * wls_resid_i`), documented at the time as "asymptotically valid but may be conservative" — that rationale was WRONG: the omitted term is mean-zero (WLS orthogonality), which is why point estimates always matched R at ~1e-11, but its variance contribution is first-order, leaving per-cell SEs 4-13% and aggregated SEs 3-20% from R, and ANTI-conservative on some fixtures (0.958x on the two-period golden). Per-cell parity vs R goldens is now ~5e-12 (`tests/test_csdid_ported.py`, `tests/test_methodology_callaway.py::TestDRDIDPanelIFParity`). SEs pass weight-scale-invariance tests (only weight ratios enter every term).
- **Note (deviation from R):** Per-cell ATT(g,t) SEs under survey weights use influence-function-based variance (matching R's `did::att_gt` analytical SE path) rather than full Taylor-series linearization. When strata/PSU/FPC are present, analytical aggregated SEs (`n_bootstrap=0`) use `compute_survey_if_variance()` on the combined IF/WIF; bootstrap aggregated SEs (`n_bootstrap>0`) use PSU-level multiplier weights.

- **Note:** Repeated cross-sections (`panel=False`, Phase 7b): supports surveys like BRFSS, ACS annual, and CPS monthly where units are not followed over time. Uses cross-sectional DRDID (Sant'Anna & Zhao 2020, Section 4): `reg` matches `DRDID::reg_did_rc` (Eq 2.2), `dr` matches `DRDID::drdid_rc` (locally efficient, Eq 3.3+3.4 with 4 OLS fits), `ipw` matches `DRDID::std_ipw_did_rc`. Per-observation influence functions instead of per-unit. All three estimation methods support covariates and survey weights.
- **Note:** Panel and RCS influence functions use the library-wide `phi_i = psi_i / n` convention (SE = `sqrt(sum(phi^2))`, algebraically equivalent to R's `sd(psi)*sqrt(n-1)/n`). Leading IF terms are computed on psi scale and divided by n; PS nuisance corrections are computed on psi scale (`score @ solve(Hessian)`) with a single `/n` conversion to phi.
- **Note:** Non-survey DR path also includes nuisance IF corrections (PS + OR), matching the survey path structure (Phase 7a). Previously used plug-in IF only. As of v3.7 the non-survey reg and ipw paths carry their corrections too (OR estimation-effect / PS score), so the nuisance-IF treatment is method-uniform.
- **Note (post-fit aggregate() - rows M-020/M-117):** `fit(aggregate=)` is deprecated (3.9; removed 4.0) in favor of post-fit `results.aggregate(type, balance_e=)` on the fit-retained aggregation kit ('simple' relays the stored overall inference; 'event_study'/'group' recompute from the kit and fail closed on bootstrapped fits per the M-027 per-level rule). `DiagnosticReport` now derives the event-study container internally on plain fits (raw `event_study_effects` absent), so its `parallel_trends`, `pretrends_power`, and `sensitivity` checks run without the deprecated fit-time kwarg — CS containers are M-093-admitted into `compute_pretrends_power` / `HonestDiD.sensitivity_analysis` with pinned raw-route parity, and the raw field, when present (the requested-but-empty `{}` included), always takes precedence. `heterogeneity` is unaffected (it reads `group_time_effects` on plain fits). Bootstrapped fits surface the fail-closed `NotImplementedError` as an explicit per-check skip reason.

**Reference implementation(s):**
- R: `did::att_gt()` (Callaway & Sant'Anna's official package)
- Stata: `csdid`

**Requirements checklist:**
- [ ] Requires never-treated units when `control_group="never_treated"` (default); not required for `"not_yet_treated"`
- [ ] Bootstrap weights support Rademacher, Mammen, Webb distributions
- [ ] Aggregations: simple, event_study, group all implemented
- [ ] Doubly robust estimation when covariates provided
- [ ] Multiplier bootstrap preserves panel structure
- [x] Repeated cross-sections (`panel=False`) for non-panel surveys (Phase 7b)

---

## ChaisemartinDHaultfoeuille

**Primary sources:**
- [de Chaisemartin, C. & D'Haultfœuille, X. (2020). Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects. *American Economic Review*, 110(9), 2964-2996.](https://doi.org/10.1257/aer.20181169)
- [de Chaisemartin, C. & D'Haultfœuille, X. (2022, revised July 2023). Difference-in-Differences Estimators of Intertemporal Treatment Effects. NBER Working Paper 29873.](https://www.nber.org/papers/w29873) — Web Appendix Section 3.7.3 contains the cohort-recentered plug-in variance formula implemented here.

**Phase 1-2 scope:** Ships the contemporaneous-switch estimator `DID_M` (= `DID_1` at horizon `l = 1`) from the AER 2020 paper **plus** the full multi-horizon event study `DID_l` for `l = 1..L_max` from the dynamic companion paper. Phase 2 adds: per-group `DID_{g,l}` building block (Equation 3), dynamic placebos `DID^{pl}_l`, normalized estimator `DID^n_l`, cost-benefit aggregate `delta`, sup-t simultaneous confidence bands, and `plot_event_study()` integration. Phase 3 adds covariate adjustment (`DID^X`), group-specific linear trends (`DID^{fd}`), state-set-specific trends, and HonestDiD integration. Survey design supports pweight with strata/PSU/FPC via Taylor Series Linearization (analytical) or replicate-weight variance (BRR/Fay/JK1/JKn/SDR) across all IF sites, plus opt-in PSU-level Hall-Mammen wild bootstrap via `n_bootstrap > 0` (see the full checklist + Notes below for the contract). **This is the most general library estimator for non-absorbing (reversible) treatments** - treatment can switch on AND off over time, switcher vs non-switcher is its primitive object, and it allows dynamic (carryover) effects with explicit joiner/leaver (`DID_+` / `DID_-`) decomposition - making it the natural fit for marketing campaigns, seasonal promotions, on/off policy cycles. (`LPDiD` with `non_absorbing="first_entry"` / `"effect_stabilization"` and `TROP` with `non_absorbing=True` under a no-dynamic-effects assumption also accept non-absorbing treatment under stronger assumptions.)

**Key implementation requirements:**

*Assumption checks / warnings:*
- **Note:** Treatment supports both binary `{0, 1}` and non-binary (ordinal or continuous) values. Non-binary treatment requires `L_max >= 1` because the per-period DID path uses binary joiner/leaver categorization; the multi-horizon per-group path (`DID_{g,l}`) handles non-binary correctly. The paper's setup (Section 2 of the dynamic companion) defines treatment as a general variable `D_{g,t}` - the binary case is a special case. Under non-binary treatment: baselines are `D_{g,1}` (float), control pools match on exact baseline value, cohorts are defined by `(D_{g,1}, F_g, S_g)` where `S_g = sign(D_{g,F_g} - D_{g,1})`, and groups with different dose magnitudes but same baseline/timing are pooled within a cohort for variance recentering.
- NaN values in `treatment` or `outcome` columns raise `ValueError` early in `fit()` (no silent drops).
- Treatment must be constant within each `(g, t)` cell. Within-cell-varying treatment (cell min != cell max) raises `ValueError`. Pre-aggregate your data to constant cell-level treatment before fitting. Fuzzy DiD is deferred to a separate dCDH 2018 paper.
- **Note:** Multi-switch groups (those with more than one treatment-change period) are dropped before estimation when `drop_larger_lower=True` (the default, matching R `DIDmultiplegtDYN`). For binary treatment, >1 change means a reversal (e.g., 0->1->0). For non-binary, >1 change includes both reversals (0->2->1) and monotone multi-step paths (0->1->2); both are dropped because the per-group `DID_{g,l}` building block attributes the full outcome change from `F_g-1` to `F_g-1+l` to the first treatment change, and a second change would confound that attribution. A single jump of any magnitude (0->3->3->3) has 1 change period and is kept. Each drop emits a warning with the count and example group IDs.
- Singleton-baseline groups — groups whose `D_{g,1}` value is unique in the post-drop dataset — are excluded from the **variance computation only** (per footnote 15 of the dynamic paper, they have no cohort peer). They are **retained** in the point-estimate sample as period-based stable controls. Each emits a warning. See the singleton-baseline Note below.
- Never-switching groups (`S_g = 0`) participate in the variance computation when they serve as stable controls under the full influence function. The `n_groups_dropped_never_switching` results field is reported for backwards compatibility but the count no longer represents an actual exclusion.
- **Note:** Provenance exposure (4.0 program row M-092, no numeric change):
  `ChaisemartinDHaultfoeuilleResults.event_study_df` records the ONE df every
  stored event-study AND placebo row's `safe_inference` received. A single
  scalar is faithful because both surfaces are computed from the same design
  df: None without a survey design (normal-theory inference), the design df
  under Taylor linearization, and the final effective df under replicate
  weights (where the late refresh re-runs every stored row - including, from
  this change, the Phase 1 `L_max=None` event-study row, which is a value
  copy of the overall inference - to that single value). Cleared under
  bootstrap, whose percentile p/CIs never used a df; that clear is required
  rather than defensive, since Taylor-linearized survey designs DO compose
  with bootstrap (only replicate weights + bootstrap is rejected).
- **Balanced-baseline panel required (deviation from R `DIDmultiplegtDYN`).** Every group must have an observation at the **first global period** (the panel's earliest time value); groups missing this baseline raise `ValueError` with the offending group IDs. Groups with **interior period gaps** (missing observations between their first and last observed period) are dropped with a `UserWarning`. **Terminal missingness** (groups observed at the baseline but missing one or more *later* periods) is **retained**: the group contributes from its observed periods only, masked out of the missing transitions by the per-period `present = (N_mat[:, t] > 0) & (N_mat[:, t-1] > 0)` guard. See the ragged-panel deviation Note below.
- **Period-index semantics.** The estimator operates on **sorted period indices**, not calendar dates. Per-period DIDs use `Y_{g,t} - Y_{g,t-1}` where `t-1` is the *previous observed period in the sorted panel*, not the previous calendar unit. A panel with periods `[2000, 2001, 2003]` (missing year 2002 for ALL groups) is treated as a valid 3-period panel where 2003 is the immediate successor of 2001. The estimator does NOT validate that periods are evenly spaced or that calendar gaps have been imputed. This matches the AER 2020 paper's Theorem 3, which defines transition sets by adjacent sorted periods without assuming calendar regularity, and is consistent with R `DIDmultiplegtDYN`'s behavior. If your data has calendar gaps that should be treated as missing periods rather than adjacent transitions, insert placeholder rows for the missing periods with the group's lagged treatment value and a reasonable imputed outcome (e.g., the group's last observed outcome), so the cell-aggregation step treats the gap as a stable-treatment period rather than a missing one. The validator rejects NaN in outcome and treatment columns, so placeholders must have finite values.
- Per-period Assumption 11 violations (joiners exist but no stable-untreated controls in some period, or leavers exist but no stable-treated controls) trigger zero-retention behavior with a consolidated warning. See the A11 Note below.

*Estimator equations (Theorem 3 of AER 2020 / Section 3.7.2 of the dynamic paper):*

Per-period DiDs at each switching period `t >= 2`:

```
DID_{+,t} = (1/N_{1,0,t}) * sum_{g in joiners(t)} (Y_{g,t} - Y_{g,t-1})
          - (1/N_{0,0,t}) * sum_{g in stable_0(t)} (Y_{g,t} - Y_{g,t-1})

DID_{-,t} = (1/N_{1,1,t}) * sum_{g in stable_1(t)} (Y_{g,t} - Y_{g,t-1})
          - (1/N_{0,1,t}) * sum_{g in leavers(t)} (Y_{g,t} - Y_{g,t-1})
```

where `joiners(t)` are groups switching from `D_{g,t-1}=0` to `D_{g,t}=1`, `leavers(t)` are groups switching `1->0`, `stable_0(t)` are groups with `D_{g,t-1}=D_{g,t}=0`, and `stable_1(t)` are groups with `D_{g,t-1}=D_{g,t}=1`. **In the library's implementation, `N_{a,b,t}` is the COUNT of `(g, t)` cells in each transition state**, not the sum of within-cell observation counts. Each `(g, t)` cell contributes once to its transition's count regardless of how many original observations fed into the cell mean. The cell mean `Y_{g,t}` is computed at the cell-aggregation step via `groupby([group, time]).agg(y_gt=mean)`; the per-period DIDs use these cell means directly without further sample-size weighting. This is the **library's documented choice**; the AER 2020 paper's Equation 3 explicitly defines `N_{d,d',t} = sum_{g} N_{g,t}` (observation sums, cell-size weighting). See the new paper review at `docs/methodology/papers/dechaisemartin-dhaultfoeuille-2020-review.md` (L76-L88 + L278-L280) for the verbatim paper transcription. **Note (deviation from R `DIDmultiplegtDYN` and from the paper's main-text formulas):** On individual-level inputs with uneven `(group, time)` cell sizes, Python gives each cell **equal weight** (cell-count weighting after up-front cell aggregation). R `DIDmultiplegtDYN`, absent an explicit weight variable, weights estimation by the number of observations in each cell (cell-size weighting), matching the paper's main-text observation-sum formulas. The two agree exactly on cell-aggregated input where every cell has the same number of observations. The Python parity tests in `tests/test_chaisemartin_dhaultfoeuille_parity.py` use the `generate_reversible_did_data()` generator, which produces exactly one observation per cell, so parity holds. The regression test `test_cell_count_weighting_unbalanced_input` in `tests/test_chaisemartin_dhaultfoeuille.py` explicitly pins the equal-cell contract.

Aggregate `DID_M`:

```
N_S   = sum_{t>=2} (N_{1,0,t} + N_{0,1,t})
DID_M = (1/N_S) * sum_{t>=2} (N_{1,0,t} * DID_{+,t} + N_{0,1,t} * DID_{-,t})
```

Joiners-only and leavers-only views (each weighted by its own switcher count):

```
DID_+ = sum_{t>=2} (N_{1,0,t} / sum_{t} N_{1,0,t}) * DID_{+,t}
DID_- = sum_{t>=2} (N_{0,1,t} / sum_{t} N_{0,1,t}) * DID_{-,t}
```

Single-lag placebo (AER 2020 placebo specification, same section as Theorem 3) — applies the same Theorem 3 logic to the pre-period first difference on cells with 3-period histories. Writing `d_a(cells) = mean over `cells` of (Y_{g,t-1} - Y_{g,t-2})` for the pre-period forward first difference:

```
DID_M^pl = (1/N_S^pl) * sum_{t>=3} (
              N_{1,0,t} * [ d_a(stable_0(t)) - d_a(joiners(t)) ]      # joiners side, S=+1
            + N_{0,1,t} * [ d_a(leavers(t))  - d_a(stable_1(t)) ]     # leavers side, S=-1
          )
```

The per-side terms are the code's `placebo_plus_t = stable0_avg - joiner_avg` and `placebo_minus_t = leaver_avg - stable1_avg` — the backward-difference × switch-direction convention of the sign Note below (equivalently, `S_g · (Y_{g,t-2} - Y_{g,t-1})_switcher` minus the same for its stable controls, matching `_compute_multi_horizon_placebos` and R).

**Note (sign convention):** the reported `placebo_effect` uses the **backward-difference × switch-direction** convention of the multi-horizon placebo path (`_compute_multi_horizon_placebos`: `switcher_change - ctrl_avg` with `Y_bwd - Y_ref`, times `S_g = +1` joiners / `-1` leavers) and R `did_multiplegt_dyn`. In the phase-1 (`L_max=None`) code this is `placebo_plus_t = stable0_avg - joiner_avg` (joiners) and `placebo_minus_t = leaver_avg - stable1_avg` (leavers). Prior to this the phase-1 path used the opposite (forward-difference) order, so `placebo_effect` was **sign-flipped vs R** on pure-direction panels (magnitude bit-identical); the multi-horizon path was always correct. Pinned by `tests/test_chaisemartin_dhaultfoeuille_parity.py::TestDCDHDynRParity::test_parity_{joiners,leavers}_only`. On **mixed-direction** panels the placebo magnitude additionally carries the documented period-vs-cohort stable-control-set / equal-cell-weighting deviation (see the `**Note (deviation from R DIDmultiplegtDYN):**` above), so it is not gate-asserted against R there.

*Phase 2: Multi-horizon event study (Equation 3 and 5 of the dynamic companion paper):*

When `L_max >= 1`, the estimator computes the per-group building block `DID_{g,l}` and the aggregate `DID_l` for each horizon. When `L_max=1`, `overall_att` holds `DID_1` (the per-group estimand, not the per-period `DID_M`). When `L_max >= 2`, `overall_att` holds the cost-benefit delta. When `L_max=None`, the per-period `DID_M` path is used:

```
DID_{g,l} = Y_{g, F_g-1+l} - Y_{g, F_g-1}
            - (1/N^g_{F_g-1+l}) * sum_{g': same baseline, F_{g'}>F_g-1+l}
                (Y_{g', F_g-1+l} - Y_{g', F_g-1})

DID_l     = (1/N_l) * sum_{g: F_g-1+l <= T_g} S_g * DID_{g,l}
```

Normalized estimator `DID^n_l = DID_l / delta^D_l` where `delta^D_l = (1/N_l) * sum |delta^D_{g,l}|` and `delta^D_{g,l} = sum_{k=0}^{l-1} (D_{g,F_g+k} - D_{g,1})`. For binary treatment: `DID^n_l = DID_l / l`.

Cost-benefit aggregate `delta = sum_l w_l * DID_l` (Lemma 4) where `w_l` are non-negative weights reflecting the cumulative dose at each horizon. When `L_max > 1`, `overall_att` holds this delta.

Dynamic placebos `DID^{pl}_l` look backward from each group's reference period, with a dual eligibility condition: `F_g - 1 - l >= 1` AND `F_g - 1 + l <= T_g`.

- **Note (Phase 2 `DID_1` vs Phase 1 `DID_M`):** When `L_max >= 2`, `event_study_effects[1]` uses the per-group `DID_{g,1}` building block (Equation 3 of the dynamic paper) with cohort-based controls, which may differ slightly from the Phase 1 `DID_M` value (Theorem 3 of AER 2020 with period-based stable-control sets). The Phase 1 `DID_M` value remains accessible via `fit(..., L_max=None).overall_att`. The difference arises because the per-group path conditions on baseline treatment `D_{g,1}` when selecting controls, while the per-period path does not. On pure-direction panels (all joiners or all leavers) the two agree; on mixed-direction panels they can differ by O(1%). This is the same period-vs-cohort control-set deviation documented in the Phase 1 Note above, extended to the `l=1` event-study entry.

- **Note (Phase 2 equal-cell weighting, deviation from R `DIDmultiplegtDYN`):** The Phase 1 equal-cell weighting contract carries forward to all Phase 2 estimands (`DID_l`, `DID^{pl}_l`, `DID^n_l`, `delta`). Each `(g, t)` cell contributes equally regardless of within-cell observation count. On individual-level inputs with uneven cell sizes, this produces a different estimand than R `DIDmultiplegtDYN` which weights by cell size. The parity tests use one-observation-per-cell generators so parity holds. See the Phase 1 weighting Note above for the full rationale.

- **Note (Phase 2 `<50%` switcher warning):** When fewer than 50% of the l=1 switchers contribute at a far horizon l, `fit()` emits a `UserWarning`. The paper recommends not reporting such horizons (Favara-Imbs application, footnote 14).

- **Note (Phase 2 Assumption 7 and cost-benefit delta):** Assumption 7 (`D_{g,t} >= D_{g,1}`) is required for the single-sign cost-benefit interpretation. When leavers are present (binary: 1->0 groups violate Assumption 7), the estimator emits a `UserWarning` and provides `delta_joiners` / `delta_leavers` separately on `results.cost_benefit_delta`.

- **Note (Phase 2 cost-benefit delta SE):** When `L_max >= 2`, `overall_att` holds the cost-benefit `delta`. Its SE is computed via the delta method from per-horizon SEs: `SE(delta) = sqrt(sum w_l^2 * SE(DID_l)^2)`, treating horizons as independent (conservative under Assumption 8). When bootstrap is enabled, per-horizon bootstrap SEs flow through the delta-method formula, so `overall_se` reflects bootstrap-derived per-horizon uncertainty but the delta aggregation itself uses normal-theory (not bootstrap percentile). This is an intentional exception to the general bootstrap-inference-surface contract: `overall_p_value` and `overall_conf_int` for `delta` use `safe_inference(delta, delta_se)`, not percentile bootstrap, because the delta is a derived aggregate rather than a directly bootstrapped estimand.

- **Note (post-fit `aggregate()` is a view - row M-026):** `results.aggregate('event_study')` and `results.aggregate('simple')` are pure VIEWS over stored fields - nothing is recomputed. `'event_study'` returns the unified `EventStudyResults` container (Phase-1 `L_max=None` fits: the 2-row l=1 view; `L_max >= 1`: the multi-horizon `l1_first_switch` surface); `'simple'` a one-row `AggregationResult` relaying `overall_att/se/t/p/CI` bit-exactly with the estimand-aware `target` label (DID_M / DID_1 / delta, or the trends-linear first-difference label whose overall row is all-NaN by design). Because it is a view, BOOTSTRAP FITS ARE PERMITTED - the library-wide per-level relay rule (since M-027 the recompute adopters' 'simple' relays are permitted on bootstrap fits too, while their kit-based recompute levels stay closed - a recompute could silently substitute analytical inference); here each row relays exactly the inference the fit stored, including the delta's analytical-with-df numbers under bootstrap per the `Note (Phase 2 cost-benefit delta SE)` above (the view's `df` column resolves from the actual inference path, not the bootstrap-cleared `event_study_df` channel). The dCDH event-study CONTAINER is deliberately rejected by `compute_honest_did`/`compute_pretrends_power` - its l1 placebo semantics need HonestDiD's native dCDH branch (mandatory reinterpretation warning + horizon trimming); pass the results object itself.

- **Note (dynamic placebo SE - library extension):** Dynamic placebos `DID^{pl}_l` (negative horizons in `placebo_event_study`) now have analytical SE and bootstrap SE when `L_max >= 1`. The placebo IF uses the same cohort-recentered structure as positive horizons, applied to backward outcome differences `Y_{g, F_g-1-l} - Y_{g, F_g-1}` with the dual-eligibility control pool (forward + backward observation required). The paper's Theorem 1 variance result is stated for `DID_l`, not `DID^{pl}_l` - this extension applies the same IF/variance structure to the placebo estimand as a library enhancement. The single-period placebo `DID_M^pl` (`L_max=None`) retains NaN SE because the per-period aggregation path has no IF derivation.

*Standard errors (Web Appendix Section 3.7.3 of the dynamic companion paper):*

Default: cohort-recentered analytical plug-in variance, evaluated at horizon `l = 1`. Cohorts are defined by the triple `(D_{g,1}, F_g, S_g)` (baseline treatment, first-switch period, switch direction). Each group's per-period role weights (joiner, stable_0, leaver, stable_1) sum to a per-group `U^G_g` value via the full `Lambda^G_{g,l=1}` weight vector from Section 3.7.2 of the dynamic paper:

```
N_S * DID_M = sum_t [
      sum_{g in joiners(t)}  (Y_{g,t} - Y_{g,t-1})
    - (N_{1,0,t} / N_{0,0,t}) * sum_{g in stable_0(t)} (Y_{g,t} - Y_{g,t-1})
    + (N_{0,1,t} / N_{1,1,t}) * sum_{g in stable_1(t)} (Y_{g,t} - Y_{g,t-1})
    - sum_{g in leavers(t)}  (Y_{g,t} - Y_{g,t-1})
]
```

Reading off the coefficient on each `(Y_{g,t} - Y_{g,t-1})` gives the per-cell role weight, which sums across periods to:

```
U^G_g     = sum_t lambda^G_{g,t} * (Y_{g,t} - Y_{g,t-1})    # full IF
U_bar_k   = (1/|C_k|) * sum_{g in C_k} U^G_g                # cohort-conditional mean
sigma_hat^2 = sum_g (U^G_g - U_bar_{cohort(g)})^2 / N_l
SE         = sqrt(sigma_hat^2 / N_l)
```

Each switching group typically contributes from MULTIPLE periods: its own switch period plus every period where it serves as a stable control for another cohort's switch. Never-switching groups can also have non-zero `U^G_g` when they serve as stable controls. Singleton-baseline groups (footnote 15 of dynamic paper) are excluded from this sum because they have no cohort peer.

The cohort recentering is critical: subtracting cohort-conditional means is **not** the same as subtracting a single grand mean. The implementation has a dedicated regression test (`test_cohort_recentering_not_grand_mean`) that computes both formulas on a designed DGP and asserts they differ materially.

Alternative: Multiplier bootstrap clustered at group via the `n_bootstrap` parameter. Available weight distributions: `"rademacher"` (default), `"mammen"`, `"webb"`. The bootstrap is a library extension beyond the original papers and is provided for consistency with `CallawaySantAnna` / `ImputationDiD` / `TwoStageDiD`.

*Edge cases:*
- **No switchers in data** (after filtering): raises `ValueError` with a clear message indicating which filters dropped which groups.
- **No joiners** (only leavers in data): `joiners_available = False`, all `joiners_*` fields are `NaN`. Symmetric for `leavers_available = False`.
- **`T < 3`**: placebo cannot be computed; `placebo_available = False` with a `UserWarning`.
- **NaN inference**: `safe_inference()` produces NaN-consistent inference fields (t-stat, p-value, conf int) when SE is non-finite or zero. `assert_nan_inference()` is used in tests to enforce consistency.
- **TWFE diagnostic with zero denominator**: when `sum(d_gt - d_bar)^2 == 0` (e.g., all cells have identical treatment), the diagnostic returns NaN for `beta_fe` and `sigma_fe` with a `UserWarning`. The diagnostic is non-fatal — it does not block the main estimation.
- **`placebo=False`** (gating): the results object still exposes `placebo_*` fields, but with `NaN` values and `placebo_available = False`. This keeps the API surface stable.

- **Note:** The analytical CI is **conservative** under Assumption 8 (independent groups) of the dynamic companion paper, and exact only under iid sampling. This is documented as a deliberate deviation from "default nominal coverage". The bootstrap CI uses the same conservative weighting and is provided for users who want a non-asymptotic alternative.

- **Note (deviation from R DIDmultiplegtDYN - SE normalization):** The analytical SE is ~4% smaller than R `did_multiplegt_dyn` on identical data. This is a normalization difference, not a bug. Python implements the paper's Section 3.7.3 plug-in formula verbatim: `SE = sigma-hat / sqrt(N_l)` where `sigma-hat^2 = (1/N_l) * sum_g U^{G,2}_{g,l} - sum_k (#C_k^G / N_l) * U-bar_k^2` and `N_l` is the number of eligible switcher groups at horizon `l`. R normalizes the influence function by `G` (total number of groups including never-switchers and stable controls) and computes `SE = sqrt(sum(U_R^2)) / G`. Both converge to the same asymptotic variance as `G -> infinity`. In finite samples R's formula produces slightly larger (more conservative) SEs because the `G`-normalization interacts with cohort recentering differently than the paper's `N_l`-normalization. Since the paper's formula is already an upper bound on the true variance (Eq 54, Jensen's inequality under Assumption 8), Python's tighter SE remains conservative. The observed gap is consistent across horizons and scenarios (~3.5-5.1%), deterministic on identical data, and does not involve any randomization.

- **Note:** Placebo SE is `NaN` for the single-period `DID_M^pl` (`L_max=None`). Multi-horizon placebos (`L_max >= 1`) have valid analytical SE and bootstrap SE via the placebo IF (see the dynamic placebo SE Note above).

- **Note:** When every variance-eligible group forms its own `(D_{g,1}, F_g, S_g)` cohort (a degenerate small-panel case where the cohort framework has zero degrees of freedom), the cohort-recentered plug-in formula is unidentified: cohort recentering subtracts the cohort mean from each group's `U^G_g`, and for singleton cohorts the centered value is exactly zero, so the centered influence function vector collapses to all zeros. The estimator returns `overall_se = NaN` with a `UserWarning` rather than silently collapsing to `0.0` (which would falsely imply infinite precision). The `DID_M` point estimate remains well-defined. The bootstrap path inherits the same degeneracy on these panels — the multiplier weights act on an all-zero vector, so the bootstrap distribution is also degenerate. **Deviation from R `DIDmultiplegtDYN`:** R returns a non-zero SE on the canonical 4-group worked example via small-sample sandwich machinery that Python does not implement. Both responses are valid for a degenerate case; Python's `NaN`+warning is the safer default. To get a non-degenerate SE, include more groups so cohorts have peers (real-world panels typically have `G >> K`).

- **Note (cluster contract):** `ChaisemartinDHaultfoeuille` clusters at the group level by default. The analytical SE plug-in operates on per-group influence-function values (one `U^G_g` per group) and, under the cell-period allocator, on their per-cell decomposition `U[g, t]` which telescopes back to `U^G_g` at the PSU-level sum. The multiplier bootstrap generates one weight per group. The user-facing `cluster=` kwarg is not supported: the constructor accepts `cluster=None` (the default and only supported value); passing any non-`None` value raises `NotImplementedError` at construction time (and the same gate fires from `set_params`) — custom user-specified clustering is reserved for a future phase. **Automatic PSU-level clustering under `survey_design`:** the analytical TSL path supports PSU labels that vary across cells of a group (within-cell constancy required); the multiplier bootstrap supports the same regime via the cell-level wild PSU bootstrap documented in the survey + bootstrap contract Note below. Under PSU-within-group-constant regimes (including the default auto-inject `psu=group` and strictly-coarser PSU with within-group constancy), the bootstrap dispatcher routes through the legacy group-level path so the SE is bit-identical to pre-cell-level releases via the identity-map fast path. The matching test for the `cluster=` gate is `test_cluster_parameter_raises_not_implemented` in `tests/test_chaisemartin_dhaultfoeuille.py::TestForwardCompatGates`.

- **Note (bootstrap inference surface):** When `n_bootstrap > 0`, the top-level `results.overall_p_value` / `results.overall_conf_int` (and joiners/leavers analogues) hold **percentile-based bootstrap inference** computed by the multiplier bootstrap, NOT normal-theory recomputations from the bootstrap SE. The t-stat (`overall_t_stat`, etc.) is computed from the SE via `safe_inference()[0]` to satisfy the project's anti-pattern rule (never compute `t = effect / se` inline) — bootstrap does not define an alternative t-stat semantic for percentile bootstrap, so the SE-based t-stat is the natural choice. `event_study_effects[1]`, `summary()`, `to_dataframe()`, `is_significant`, and `significance_stars` all read from these top-level fields and therefore reflect the bootstrap inference automatically. The library precedent for this propagation is `imputation.py:790-805`, `two_stage.py:778-787`, and `efficient_did.py:1009-1013`. The single-period placebo (`L_max=None`) still has NaN bootstrap fields; multi-horizon placebos (`L_max >= 1`) have valid bootstrap SE/CI/p via `placebo_horizon_ses/cis/p_values` on the bootstrap results object. The matching test is `test_bootstrap_p_value_and_ci_propagated_to_top_level` in `tests/test_chaisemartin_dhaultfoeuille.py::TestBootstrap`.

- **Note:** Placebo Assumption 11 violations (placebo joiners exist but no 3-period stable_0 controls, or symmetric for leavers/stable_1) trigger zero-retention in the placebo numerator AND emit a consolidated `Placebo (DID_M^pl) Assumption 11 violations` warning from `fit()`, mirroring the main DID path's contract documented above. The zeroed placebo periods retain their switcher counts in the placebo `N_S^pl` denominator, biasing `DID_M^pl` toward zero in the offending direction (matching the placebo paper convention).

- **Note:** The TWFE diagnostic (`twfe_diagnostic=True` in `fit()` and the standalone `twowayfeweights()`) requires binary `{0, 1}` treatment. On non-binary data, `fit()` emits a `UserWarning` and skips the diagnostic (all `twfe_*` fields are `None`), while `twowayfeweights()` raises `ValueError`. The diagnostic uses `d_gt == 1` as the treated-cell mask per Theorem 1 of AER 2020, which is undefined for non-binary treatment.

- **Note (TWFE diagnostic sample contract):** The fitted `results.twfe_weights` / `results.twfe_fraction_negative` / `results.twfe_sigma_fe` / `results.twfe_beta_fe` are computed on the **FULL pre-filter cell sample** — the data the user passed in, after `_validate_and_aggregate_to_cells()` runs but **before** the ragged-panel validation (Step 5b) and the multi-switch filter (`drop_larger_lower`, Step 6). They do NOT describe the post-filter estimation sample used by `overall_att`, `results.units`, and the inference fields. `fit()` has three sample-shaping filters in total: (1) interior-gap drops in Step 5b, (2) multi-switch drops in Step 6, and (3) the singleton-baseline filter in Step 7. Filters (1) and (2) actually shrink the point-estimate sample, so when either fires, the fitted TWFE diagnostic and `overall_att` describe **different samples** and the estimator emits a `UserWarning` explaining the divergence with explicit counts. Filter (3) is **variance-only** — singleton-baseline groups remain in the point-estimate sample as period-based stable controls (see the singleton-baseline Note above) — so it does NOT create a fitted-vs-`overall_att` mismatch and does NOT trigger the divergence warning. Rationale for the pre-filter design: the TWFE diagnostic answers "what would the plain TWFE estimator say on the data you passed in?" — not "what would TWFE say on the data dCDH actually used after filtering?" — so users comparing TWFE vs dCDH on a fixed input can do so without an interaction effect from the dCDH-specific filters. The standalone `twowayfeweights()` function uses the same pre-filter sample and accepts the same `survey_design` parameter as `fit()`, so the fitted and standalone APIs always produce identical numbers on the same input — including survey-weighted cell aggregation (`twowayfeweights(data, ..., survey_design=sd)` matches `fit(data, ..., survey_design=sd).twfe_*`). To reproduce the dCDH estimation sample for an external TWFE comparison, pre-process your data to drop the multi-switch and interior-gap groups before fitting (the warning lists offending IDs). The matching tests are `test_twfe_pre_filter_contract_with_interior_gap_drop` and `test_twfe_pre_filter_contract_with_multi_switch_drop` in `tests/test_chaisemartin_dhaultfoeuille.py`.

- **Note:** By default (`drop_larger_lower=True`), the estimator drops groups whose treatment switches more than once before estimation. This matches R `DIDmultiplegtDYN`'s default and is required for the analytical variance formula (Web Appendix Section 3.7.3 of the dynamic paper, which assumes Assumption 5 / no-crossing) to be consistent with the AER 2020 Theorem 3 point estimate. Both formulas operate on the same post-drop dataset. Setting `drop_larger_lower=False` is supported for diagnostic comparison but produces an inconsistent estimator-variance pairing for any multi-switch groups present, and emits an explicit warning.

- **Note:** When Assumption 11 (existence of stable controls) is violated for some period `t` — i.e., joiners exist but no stable-untreated controls, or leavers exist but no stable-treated controls — `DID_{+,t}` (or `DID_{-,t}`) is set to zero by paper convention, and the period's switcher count is **retained** in the `N_S` denominator. This means the affected period contributes a zero to the numerator with a non-zero weight in the denominator, biasing `DID_M` toward zero in the offending direction. Users can detect this by inspecting `results.per_period_effects[t]['did_plus_t_a11_zeroed']` (or `did_minus_t_a11_zeroed`) or the consolidated `fit()` warning. This matches the AER 2020 Theorem 3 paper convention and the worked example arithmetic.

- **Note:** Groups whose baseline treatment value `D_{g,1}` is unique in the post-drop panel (not shared by any other group) are excluded from the **variance computation only** per footnote 15 of the dynamic companion paper. They have no cohort peer for the cohort-recentered plug-in formula. They are **retained in the point-estimate sample** as period-based stable controls (Python's documented period-vs-cohort interpretation). The dropped count is stored on `results.n_groups_dropped_singleton_baseline`, a warning lists example group IDs, and the warning text explicitly states "VARIANCE computation only" so users know the filter does not change `DID_M`.

- **Note (deviation from R DIDmultiplegtDYN):** Python uses **period-based** stable-control sets — `stable_0(t)` is any cell with `D_{g,t-1} = D_{g,t} = 0` regardless of baseline `D_{g,1}`, and similarly for `stable_1(t)`. R `DIDmultiplegtDYN` uses **cohort-based** stable-control sets that additionally require `D_{g,1}` to match the side. Python's definition matches the AER 2020 Theorem 3 transition-state notation `N_{0,0,t}` and `N_{1,1,t}` literally (treating those as the paper-defined sets of groups in each transition state, independent of the observation-sum-vs-cell-count weighting axis covered in the equal-cell Note above); R's definition matches the dynamic companion paper's cohort `(D_{g,1}, F_g, S_g)` framework. The two definitions agree exactly on (a) panels containing only joiners, (b) panels containing only leavers, (c) the hand-calculable 4-group worked example, or (d) any panel where no joiner's post-switch state overlaps a period when leavers are switching. They disagree by O(1%) on the **point estimate** when both joiners and leavers exist AND some joiners' post-switch cells could serve as leavers' controls (or vice versa). After the Round 2 fix that implemented the full `Lambda^G_{g,l=1}` influence function, the **standard error** parity gap on pure-direction scenarios narrowed from ~18% to ~3%. The R parity tests in `tests/test_chaisemartin_dhaultfoeuille_parity.py` use a tight `1e-4` tolerance for pure-direction point estimates, 10% rtol for multi-horizon SEs (15% for L_max=5 long panels where the cell-count weighting deviation compounds), 5% rtol for single-horizon SEs, and a 2.5% tolerance for mixed-direction point estimates (with the SE check skipped on mixed scenarios because the period-vs-cohort point-estimate deviation cascades into the variance).

- **Note (deviation from R DIDmultiplegtDYN):** Phase 1 requires panels with a **balanced baseline** (every group observed at the first global period) and **no interior period gaps**. The Step 5b validation in `fit()` enforces this contract: groups missing the baseline raise `ValueError`; groups with interior gaps are dropped with a `UserWarning`; groups with **terminal missingness** (early exit / right-censoring — observed at the baseline but missing one or more later periods) are retained and contribute from their observed periods only. R `DIDmultiplegtDYN` accepts unbalanced panels with documented missing-treatment-before-first-switch handling. Python's restriction is a Phase 1 limitation: the cohort enumeration uses `D_{g,1}` as the canonical baseline (so the baseline observation must exist) and the first-switch detection walks adjacent observed periods (so interior gaps create ambiguous transition counts). Terminal missingness is supported at the POINT-ESTIMATE level because the per-period `present = (N_mat[:, t] > 0) & (N_mat[:, t-1] > 0)` guard appears at three sites in the variance computation (`_compute_per_period_dids`, `_compute_full_per_group_contributions`, `_compute_cohort_recentered_inputs`) and cleanly masks out missing transitions without propagating NaN into the arithmetic. **Scope limitation (terminal missingness under any cell-period-allocator path):** under any survey variance path that uses the cell-period allocator, a targeted `ValueError` is raised when cohort-recentering leaks non-zero centered IF mass onto cells with no positive-weight observations. Affected paths:
- **Binder TSL with within-group-varying PSU** (`n_bootstrap=0`, explicit `psu=<col>` that varies within group).
- **Rao-Wu replicate-weight ATT** (`compute_replicate_if_variance` always reads the cell-allocator `psi_obs` per the Class A contract shipped in PR #323, regardless of PSU structure).
- **Cell-level wild PSU bootstrap** (`n_bootstrap > 0` with within-group-varying PSU).

The guard is fired by `_survey_se_from_group_if` (analytical and replicate) and by `_unroll_target_to_cells` (bootstrap). **Unaffected paths**: Binder TSL under PSU-within-group-constant regimes (including PSU=group auto-inject) falls back to the legacy group-level allocator where the row-sum identity `sum_{c in g} U_centered_per_period[g, t] == U_centered[g]` makes the two statistically equivalent, and the bootstrap dispatcher routes the same regimes through the legacy group-level path. **Workaround:** pre-process the panel to remove terminal missingness (drop late-exit groups or trim to a balanced sub-panel). For Binder TSL, using an explicit `psu=<group_col>` routes through the legacy group allocator. For replicate ATT and within-group-varying-PSU bootstrap, there is no allocator fallback — the panel itself must be pre-processed. The broader unbalanced-panel workaround (back-fill the baseline or drop late-entry groups before fitting, or use R `DIDmultiplegtDYN`) also applies. The Step 5b `ValueError` and `UserWarning` messages name the offending group IDs so you can locate them quickly.

- **Note (Phase 3 DID^X covariate adjustment):** When `covariates` is set, `per_period_effects` (the Phase 1 per-period DID_M decomposition) remains **unadjusted** (computed on raw outcomes). The covariate residualization applies only to the per-group `DID_{g,l}` path (`L_max >= 1`), which produces `event_study_effects` and `overall_att`. This means `per_period_effects` and `event_study_effects[1]` may diverge when covariates are active - by design (the per-period path uses binary joiner/leaver categorization and is not part of the DID^X contract). Implements the residualization-style covariate adjustment from Web Appendix Section 1.2 (Assumption 11). For each baseline treatment value `d`, estimates `theta_hat_d` via OLS of first-differenced outcomes on first-differenced covariates with time FEs, restricted to not-yet-treated observations. Residualizes at levels: `Y_tilde[g,t] = Y[g,t] - X[g,t] @ theta_hat_d`. All downstream DID computations use residualized outcomes. This is NOT doubly-robust, NOT IPW, NOT Callaway-Sant'Anna-style. Plug-in IF (treating `theta_hat` as fixed) is valid by FWL theorem. **Deviation from R `DIDmultiplegtDYN`:** The first-stage OLS uses equal cell weights (one observation per `(g,t)` cell), consistent with the library's cell-count weighting convention documented in Phase 1. R weights by `N_gt` (observation count per cell). On panels with 1 observation per cell (the common case), results are identical. When baseline-specific first stages fail (`n_obs = 0` or `n_obs < n_params`), the affected strata are excluded from the estimation (outcomes set to NaN) rather than retained unadjusted - matching R's "drop failed strata" behavior. Requires `L_max >= 1`. Activated via `covariates=["col1", "col2"]` in `fit()`.

- **Note (Phase 3 DID^{fd} linear trends):** Implements group-specific linear trends from Web Appendix Section 1.3 (Assumption 12, Lemma 6). Uses the Z_mat transformation: `Z[g,t] = Y[g,t] - Y[g,t-1]` (first-differenced outcomes). Since `DID_{g,l}(Z) = DID^{fd}_{g,l}` algebraically, the existing multi-horizon DID code produces trend-adjusted estimates when fed Z_mat. Requires F_g >= 3 (at least 2 pre-switch periods); groups with F_g < 3 are excluded with a `UserWarning`. Cumulated level effects `delta^{fd}_l = sum_{l'=1}^l DID^{fd}_{l'}` stored in `results.linear_trends_effects`. Cumulated SE uses conservative upper bound (sum of per-horizon SEs); cross-horizon covariance from IF vectors is a library extension (paper proves Theorem 1 per-horizon, not cross-horizon). When combined with DID^X, residualization is applied first, then first-differencing (per paper assumption ordering). **Suppressed surfaces under `trends_linear`:** `normalized_effects` (`DID^n_l`) and `cost_benefit_delta` are suppressed because they would operate on second-differences rather than level effects. Users should access cumulated level effects via `linear_trends_effects`. Activated via `trends_linear=True` in `fit()`.

- **Note (Phase 3 state-set trends):** Implements state-set-specific trends from Web Appendix Section 1.4 (Assumptions 13-14). Restricts the control pool for each switcher to groups in the same set (e.g., same state in county-level data). The restriction applies in all four DID/IF paths: `_compute_multi_horizon_dids()`, `_compute_per_group_if_multi_horizon()`, `_compute_multi_horizon_placebos()`, and `_compute_per_group_if_placebo_horizon()`. Cohort structure stays as `(D_{g,1}, F_g, S_g)` triples (does not incorporate set membership). Set membership must be time-invariant per group. **Note on Assumption 14 (common support):** The paper requires a common last-untreated period across sets (`T_u^s` equal for all `s`). This implementation does NOT enforce Assumption 14 up front. Instead, when within-set controls are exhausted at a given horizon (because a set has shorter untreated support than others), the affected switcher/horizon pairs are silently excluded via the existing empty-control-pool mechanism. This means `N_l` may be smaller under `trends_nonparam` than without it, and the effective estimand is trimmed to the within-set support at each horizon. The existing multi-horizon A11 warning fires when exclusions occur. Activated via `trends_nonparam="state_column"` in `fit()`.

- **Note (Phase 3 heterogeneity testing - partial implementation):** Partial implementation of the heterogeneity test from Web Appendix Section 1.5 (Assumption 15, Lemma 7). Computes post-treatment saturated OLS regressions of `S_g * (Y_{g, F_g-1+l} - Y_{g, F_g-1})` on a time-invariant covariate `X_g` plus cohort indicator dummies. Standard OLS inference is valid (paper shows no DID error correction needed). **Deviation from R `predict_het`:** Python now matches R on per-horizon placebo regressions when the user sets `placebo=True` together with `heterogeneity=` (post-2026-05-15; see "Placebo predict_het" sub-note below for the full contract). The remaining gap is the joint null F-test that R aggregates across all `predict_het` rows — Python emits per-horizon `t_stat` / `p_value` / `conf_int` only and does NOT compute a joint Wald test across forward + placebo coefficients (tracked at REGISTRY note's "Per-horizon regressions only (no joint F-test)" rendering line). R also disallows combination with `controls`, which Python continues to enforce as an explicit `ValueError`. **Rejected combinations:** `covariates` (matching R's `controls`), `trends_linear` (heterogeneity test uses raw level changes, incompatible with second-differenced outcomes), and `trends_nonparam` (heterogeneity test does not thread state-set control-pool restrictions). Results stored in `results.heterogeneity_effects`. Activated via `heterogeneity="covariate_column"` in `fit()`. **Note (survey support):** Under `survey_design`, heterogeneity uses WLS with per-group weights `W_g = sum of obs-level survey weights in group g`, and the group-level WLS coefficient influence function is `ψ_g[X] = inv(X'WX)[1,:] @ x_g * W_g * r_g`. The group-level IF is then attributed to observation level via **one of two allocators, chosen by variance helper** so each path preserves byte-identity for its aggregation rule: (1) **Binder TSL** (`compute_survey_if_variance`) uses the **cell-period single-cell allocator** — at each horizon `l_h`, `ψ_g` is assigned in full to the post-period cell `(g, out_idx)` with `out_idx = first_switch_idx[g] - 1 + l_h` and expanded as `ψ_i = ψ_g * (w_i / W_{g, out_idx})` for obs in that cell, zero elsewhere (matches the DID_l post-period convention in the Survey IF expansion Note below). Under PSU=group per-observation distribution differs from the legacy `ψ_i = ψ_g * (w_i / W_g)`, but PSU-level aggregates telescope to the same `ψ_g` — so Binder TSL variance is byte-identical to the pre-cell-period release under PSU=group. Under within-group-varying PSU mass lands in the post-period PSU of the transition, which is what Binder TSL needs. An **empty post-period cell under zero-weight obs** (all obs at `(g, out_idx)` have `w_i = 0` despite `N > 0`) drops the group's contribution, matching the ATT cell allocator's convention; the pre-cell-period path diverged here by redistributing mass to other cells of the group. (2) **Rao-Wu replicate** (`compute_replicate_if_variance`) uses the **legacy group-level allocator** `ψ_i = ψ_g * (w_i / W_g)`. Replicate variance computes `θ_r = sum_i ratio_ir * ψ_i` at the observation level, so moving `ψ_g` mass onto the post-period cell only would silently change the replicate SE whenever a replicate column's ratios vary within group (the library accepts arbitrary per-row replicate matrices, not just PSU-aligned ones). Keeping the legacy allocator on this branch preserves byte-identity of replicate SE across every previously-supported fit; replicate + within-group-varying PSU is unreachable by construction (`SurveyDesign` rejects `replicate_weights` combined with explicit `strata/psu/fpc`). Inference uses the t-distribution with `df_survey` when provided. Under rank deficiency (any regression coefficient dropped by `solve_ols`'s R-style drop), all inference fields return NaN (conservative, matches the NaN-consistent contract). **Library extension (replicate weights):** Under a replicate-weight design (BRR/Fay/JK1/JKn/SDR), the heterogeneity regression dispatches to `compute_replicate_if_variance` (Rao-Wu weight-ratio rescaling) instead of the Binder TSL formula. The effective df is the shared `min(resolved_survey.df_survey, min(n_valid_across_sites) - 1)` used by the rest of the dCDH surfaces; if the base `df_survey` is undefined (QR-rank ≤ 1), heterogeneity inference is NaN regardless of the local `n_valid_het` (matching the dCDH top-level contract — per-site `n_valid` cannot rescue a rank-deficient design). **Library extension:** R `DIDmultiplegtDYN::predict_het` does not natively support survey weights. **Scope note (bootstrap):** Heterogeneity inference is analytical (no bootstrap path). When `n_bootstrap > 0` is combined with `heterogeneity=`, the main ATT surfaces receive bootstrap SE/CI (via the cell-level wild PSU bootstrap described in the survey + bootstrap contract Note below) while `heterogeneity_effects` continues to use the Binder TSL / Rao-Wu analytical SE described above. No gate; the two inference paths are independent.

- **Note (HonestDiD integration):** HonestDiD sensitivity analysis (Rambachan & Roth 2023) is available on the placebo + event study surface via `honest_did=True` in `fit()` or `compute_honest_did(results)` post-hoc. **Library extension:** dCDH HonestDiD uses `DID^{pl}_l` placebo estimates as pre-period coefficients rather than standard event-study pre-treatment coefficients. The Rambachan-Roth restrictions bound violations of the parallel trends assumption underlying the dCDH placebo estimand; interpretation differs from canonical event-study HonestDiD. A `UserWarning` is emitted at runtime. Uses diagonal variance (no full VCV available for dCDH). Relative magnitudes (DeltaRM) with Mbar=1.0 is the default when called from `fit()`, targeting the equal-weight average over all post-treatment horizons (`l_vec=None`). R's HonestDiD defaults to the first post/on-impact effect; use `compute_honest_did(results, ...)` with a custom `l_vec` to match that behavior. When `trends_linear=True`, bounds apply to the second-differenced estimand (parallel trends in first differences). Requires `L_max >= 1` for multi-horizon placebos. Gaps in the horizon grid from `trends_nonparam` support-trimming are handled by filtering to the largest consecutive block and warning.

- **Note (Phase 3 Design-2 switch-in/switch-out):** Convenience wrapper for Web Appendix Section 1.6 (Assumption 16). Identifies groups with exactly 2 treatment changes (join then leave), reports switch-in and switch-out mean effects. This is a descriptive summary, not a full re-estimation with specialized control pools as described in the paper. **Always uses raw (unadjusted) outcomes** regardless of active `covariates`, `trends_linear`, or `trends_nonparam` options - those adjustments apply to the main estimator surface but not to the Design-2 descriptive block. For full adjusted Design-2 estimation with proper control pools, the paper recommends "running the command on a restricted subsample and using `trends_nonparam` for the entry-timing grouping." Activated via `design2=True` in `fit()`, requires `drop_larger_lower=False` to retain 2-switch groups.

- **Note (Phase 3 `by_path` per-path event-study disaggregation):** Per-path disaggregation of the multi-horizon event study, mirroring R `did_multiplegt_dyn(..., by_path=k)`. Activated via `ChaisemartinDHaultfoeuille(by_path=k, drop_larger_lower=False)` where `k` is a positive integer (top-k most common observed paths by switcher-group frequency). **Window convention:** the path tuple for a switcher group `g` is `(D_{g, F_g-1}, D_{g, F_g}, ..., D_{g, F_g-1+L_max})` — length `L_max + 1`, matching R's window `[F_{g-1}, F_{g-1+l}]`. **Ranking:** paths are ranked by descending frequency; ties are broken lexicographically on the path tuple for deterministic ordering, so every selected path has a unique `frequency_rank`. If `by_path` exceeds the number of observed paths, all observed paths are returned with a `UserWarning`. **Per-path SE convention (joiners/leavers precedent):** the per-path influence function follows the joiners-only / leavers-only IF construction at `chaisemartin_dhaultfoeuille.py:5495-5504`: the switcher-side contribution `+S_g * (Y_{g,out} - Y_{g,ref})` is zeroed for groups whose observed trajectory is NOT the selected path; control contributions and the full cohort structure `(D_{g,1}, F_g, S_g)` are unchanged. After applying the singleton-baseline eligible mask and cohort-recentering with the original cohort IDs, the plug-in SE uses the path-specific divisor `N_l_path` (count of path switchers eligible at horizon `l`) — same pattern as `joiners_se` using `joiner_total`. This gives the **within-path mean** estimand `DID_{path,l}` as the within-path average of `DID_{g,l}`. **Degenerate-cohort behavior per path:** when a path's centered IF at some horizon is identically zero (every variance-eligible path switcher forms its own `(D_{g,1}, F_g, S_g)` cohort, or the path has a single contributing group), SE / t_stat / p_value / conf_int are NaN-consistent and a `UserWarning` is emitted scoped to `(path, horizon)`. This mirrors the overall-path degenerate-cohort surface and is common for rare paths with few contributing groups. **Empty-state contract:** `results.path_effects` distinguishes "not requested" (`None`) from "requested but empty" (`{}` — all switchers have windows outside the panel or unobserved cells). The empty-dict case emits a `UserWarning` at fit-time and renders as an explicit "no observed paths" notice in `summary()`; `to_dataframe(level="by_path")` returns an empty DataFrame with the canonical column set (mirrors the `linear_trends` pattern when `trends_linear=True` but no horizons survive). **Requirements:** `drop_larger_lower=False` (multi-switch groups are the object of interest; default `True` filters them out) and `L_max >= 1` (path window depends on the horizon). **Scope:** combinations with `design2` and `honest_did` remain gated behind explicit `NotImplementedError` (deferred to follow-up wave PRs); `heterogeneity` is supported per-path — see the **Per-path heterogeneity testing** paragraph below. `n_bootstrap > 0` is now supported — see the **Bootstrap SE** paragraph below. `survey_design` is supported under analytical Binder TSL and replicate-weight bootstrap — see the **Per-path survey-design SE** paragraph below; multiplier bootstrap (`n_bootstrap > 0`) under `survey_design + by_path/paths_of_interest` remains gated. `placebo=True` is now supported per-path — see the **Per-path placebos** paragraph below. **TWFE diagnostic** remains a sample-level summary (not computed per path) in this release. Results are exposed on `results.path_effects` as `Dict[Tuple[int, ...], Dict[str, Any]]` with nested `horizons` dicts per horizon `l`, and on `results.to_dataframe(level="by_path")` as a long-format table with columns `[path, frequency_rank, n_groups, horizon, effect, se, t_stat, p_value, conf_int_lower, conf_int_upper, n_obs, cband_lower, cband_upper, cumulated_effect, cumulated_se, het_beta, het_se, het_t_stat, het_p_value, het_conf_int_lower, het_conf_int_upper]` (the `cband_*` columns are added by the joint sup-t Note below, populated for positive-horizon rows of paths with a finite sup-t crit and NaN otherwise; the `cumulated_*` columns are added by the per-path linear-trends Note below, populated for positive-horizon rows when `trends_linear=True` is set and NaN otherwise). Gated tests live in `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathGates` / `::TestByPathBehavior` / `::TestByPathEdgeCases`. **R-parity** against `DIDmultiplegtDYN 2.3.3` is confirmed at `tests/test_chaisemartin_dhaultfoeuille_parity.py::TestDCDHDynRParityByPath` via two scenarios: `mixed_single_switch_by_path` (2 paths, `by_path=2`) and `multi_path_reversible_by_path` (4 paths, `by_path=3`; path-assignment deterministic on `F_g` so each `(D_{g,1}, F_g, S_g)` cohort contains switchers from a single path). Per-path point estimates and per-path switcher counts match R exactly; per-path SE matches within the Phase 2 multi-horizon SE envelope (observed rtol ≤ 10.2% on the 2-path mixed scenario, ≤ 4.2% on the 4-path cohort-clean scenario). **Deviation from R (cross-path cohort-sharing SE):** our analytical SE is the marginal variance of the path-contribution estimator cohort-centered on the *full-panel* cohort structure (joiners/leavers precedent — non-path switchers contribute to cohort means via their zeroed switcher row). R's `did_multiplegt_dyn(..., by_path=k)` re-runs the estimator per path, so cohort means are computed over the path's own switchers only. When a cohort `(D_{g,1}, F_g, S_g)` spans multiple observed paths, Python and R SE diverge materially (our empirical probes with random post-window toggling saw rtol > 100%); when every cohort is single-path (scenario 13 by design, scenario 14 by construction), the two approaches coincide up to the documented Phase 2 envelope. Practitioners with cohort structures that mix paths should interpret the per-path SE as a within-full-panel marginal variance, not a per-path conditional variance. **Bootstrap SE:** when `n_bootstrap > 0` is set, the top-k paths are enumerated once on the observed data (R-faithful: matches `did_multiplegt_dyn(..., by_path=k, bootstrap=B)`'s path-stability convention — verified empirically against DIDmultiplegtDYN 2.3.3) and the multiplier bootstrap (`bootstrap_weights ∈ {"rademacher", "mammen", "webb"}`) runs per `(path, horizon)` target via the shared `_bootstrap_one_target` / `compute_effect_bootstrap_stats` helpers. Point estimates are unchanged from the analytical path. Bootstrap SE replaces the analytical SE in `path_effects[path]["horizons"][l]["se"]`, and `p_value` / `conf_int` are taken as the **bootstrap percentile** statistics, matching the Round-10 library convention for overall / joiners / leavers / multi-horizon bootstrap (see the `Note (bootstrap inference surface)` elsewhere in this file and the pinned regression `test_bootstrap_p_value_and_ci_propagated_to_top_level`). `t_stat` is SE-derived via `safe_inference` per the anti-pattern rule. Interpretation: inference is *conditional on the observed path set*. **SE inherits the analytical cross-path cohort-sharing deviation:** the bootstrap input is the exact same full-panel cohort-centered path IF that the analytical path computes (`_collect_path_bootstrap_inputs` reuses the same enumeration / cohort IDs / IF construction), so the bootstrap SE is a Monte Carlo analog of the analytical SE — it inherits the same cross-path cohort-sharing deviation from R's per-path re-run convention documented above. On single-path-cohort panels (scenarios 13 and 14 of the R-parity fixture, and any DGP where `(D_{g,1}, F_g, S_g)` cohorts never span multiple observed paths), bootstrap SE tracks analytical SE up to Monte Carlo noise and both coincide with R up to the Phase 2 envelope. On cross-path cohort panels, bootstrap SE inherits the >100% rtol divergence from R that analytical already has. **Deviation from R (CI method):** R's per-path CI is normal-theory around the bootstrap SE (half-width ≈ `1.96·se`); ours is the bootstrap percentile CI, intentionally diverging from R to keep the dCDH inference surface internally consistent across all bootstrap targets. Practitioners who want *unconditional* inference capturing path-selection uncertainty need a pairs-bootstrap (deferred — no R precedent). Positive regressions live in `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathBootstrap` (gated `@pytest.mark.slow`): point-estimate invariance, finite positive SE on non-degenerate panels, SE-within-30%-rtol of analytical on cohort-clean fixtures, degenerate-cohort NaN propagation, Rademacher/Mammen/Webb parity, seed reproducibility, and percentile-vs-normal-theory CI pinning. **Per-path placebos:** when `placebo=True` (and `L_max >= 1`) is combined with `by_path=k`, per-path backward-horizon placebos `DID^{pl}_{path, l}` for `l = 1..L_max` are computed using the same joiners/leavers IF precedent applied to `_compute_per_group_if_placebo_horizon` (with the new `switcher_subset_mask` parameter): switcher contributions are zeroed for groups not in the path; the control pool and the variance-eligible cohort structure `(D_{g,1}, F_g, S_g)` are unchanged. Plug-in SE uses the path-specific divisor `N^{pl}_{l, path}` (count of path switchers eligible at backward lag `l`). Surfaced on `results.path_placebo_event_study[path][-l]` with the same `{effect, se, t_stat, p_value, conf_int, n_obs}` shape as `placebo_event_study` (negative-int inner keys parallel the existing per-path event-study positive-int keys, so a unified forward+backward view is well-formed). **Inherits the cross-path cohort-sharing SE deviation from R** documented above for `path_effects` (same convention applied backward); tracks R within numerical tolerance on single-path-cohort panels and diverges on cohort-mixed panels. Multiplier bootstrap (when `n_bootstrap > 0`) runs per `(path, lag)` target via the same `_bootstrap_one_target` dispatch used for the per-path event-study, with the canonical NaN-on-invalid contract. The bootstrap SE is a Monte Carlo analog of the analytical placebo SE — same per-path centered IF input — and inherits the same deviation. Surfaced through `summary()` (negative-keyed rows rendered alongside positive-keyed event-study rows under each path block) and `to_dataframe(level="by_path")` (`horizon` column takes negative ints for placebo rows). **Empty-state contract:** `results.path_placebo_event_study` mirrors `path_effects` — `None` when `by_path + placebo` was not requested, `{}` when requested but no observed path has a complete window within the panel (same regime that returns `{}` for `path_effects`, with the same fit-time `UserWarning`). R-parity is confirmed at `tests/test_chaisemartin_dhaultfoeuille_parity.py::TestDCDHDynRParityByPathPlacebo` on the `multi_path_reversible_by_path_placebo` scenario; positive analytical + bootstrap invariants live in `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathPlacebo` (with the gated `::TestByPathPlacebo::TestBootstrap` subclass). **Per-path covariate residualization (DID^X):** when `covariates=[...]` is set with `by_path=k`, the per-baseline OLS residualization (Web Appendix Section 1.2) runs once on the first-differenced outcome BEFORE path enumeration. All four downstream surfaces — analytical per-path SE, bootstrap SE, per-path placebos, and per-path joint sup-t bands — consume the residualized `Y_mat` automatically (Frisch-Waugh-Lovell). Per-period effects remain unadjusted, consistent with the existing `covariates` + per-period DID contract (per-period DID does not support residualization). Failed-stratum baselines (rank-deficient X) zero out `N_mat` for affected groups, which the path enumeration treats as ineligible per its existing convention. **Deviation from R on multi-baseline switcher panels (point estimates):** R `did_multiplegt_dyn(..., by_path, controls)` re-runs the per-baseline residualization on each path's restricted subsample (`R/R/did_multiplegt_dyn.R` lines 401-405: rows of the path's switchers OR rows where `yet_to_switch=1 AND baseline matches the path's baseline`). The first-stage residualization sample R uses for path B equals: pre-switch rows of all switchers with matching baseline + all rows of never-switchers with matching baseline — bit-identical to our global first-stage sample under single-baseline switcher panels (every switcher shares the same `D_{g,1}`, regardless of how `F_g` or path identity varies across switchers). Per-path point estimates therefore coincide with R on those panels up to the existing **DID^X first-stage cell-weighting deviation** documented above in `Note (Phase 3 DID^X covariate adjustment)` (Python's first-stage OLS uses equal cell weights — one observation per `(g, t)` cell, consistent with the library's cell-aggregated input convention; R weights by `N_gt`). On panels with one observation per `(g, t)` cell (the common case after the cell-aggregation step in `fit()`), Python matches R bit-exactly: the `multi_path_reversible_by_path_controls` parity fixture has 4 paths with switcher `F_g` values spanning [0..6] under `D_{g,1}=0` and Python matches R to rtol ~1e-11. On multi-baseline switcher panels (some switchers have `D_{g,1}=0`, others have `D_{g,1}=1`) R's per-path subset drops switchers whose baseline differs from the path's baseline, so the per-baseline regression coefficients diverge per path under R and point estimates can diverge between Python and R — a `UserWarning` is emitted at fit-time when this configuration is detected so practitioners do not silently consume estimates that disagree with R. The warning filters to switcher groups only; never-switchers (never-treated + always-treated controls) at multiple baseline values do NOT trigger the warning because they don't affect R's per-path subset construction. **Inherits the cross-path cohort-sharing SE deviation from R** documented above for `path_effects` — bootstrap SE, placebo SE, and sup-t crit are Monte Carlo / joint-distribution analogs of the same residualized analytical IF and carry the same deviation. R-parity is confirmed against `did_multiplegt_dyn(..., by_path=3, controls="X1")` at `tests/test_chaisemartin_dhaultfoeuille_parity.py::TestDCDHDynRParityByPathControls` on the `multi_path_reversible_by_path_controls` scenario (single-baseline DGP, exact point-estimate match measured rtol ~1e-11); cross-surface inheritance and the multi-baseline warning are regression-tested at `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathControls` (analytical + bootstrap + placebo + sup-t + `to_dataframe(level="by_path")` cband columns + multi-baseline `UserWarning`). **Per-path linear-trends DID^{fd}:** when `trends_linear=True` is set with `by_path=k`, the first-differencing transform at `chaisemartin_dhaultfoeuille.py:1599-1630` runs once globally BEFORE path enumeration (replaces `Y_mat` with `Z_mat = Y_t - Y_{t-1}` and shrinks the time axis by one), so per-path raw second-differences `DID^{fd}_{path, l}` surface on `path_effects[path]["horizons"][l]` automatically. Per-path cumulated level effects `delta_{path, l} = sum_{l'=1..l} DID^{fd}_{path, l'}` (the quantity R returns under `did_multiplegt_dyn(..., by_path, trends_lin)` per the existing parity test pivot at `tests/test_chaisemartin_dhaultfoeuille_parity.py:403-409`) surface on the new `results.path_cumulated_event_study[path][l]` field — a per-group running sum of `DID^{fd}_{g, l'}` averaged over the path's switchers eligible at horizon `l`, mirroring the global `linear_trends_effects` cumulation logic at `chaisemartin_dhaultfoeuille.py:3340-3398`. SE on the cumulated layer is the conservative upper bound (sum of per-horizon component SEs from `path_effects[path]["horizons"][l]["se"]`, NaN-consistent: any non-finite component yields a NaN cumulated SE). **Post-bootstrap recomputation:** the cumulated layer is built AFTER the bootstrap propagation block at `chaisemartin_dhaultfoeuille.py:3034-3081` so it reads the FINAL post-bootstrap per-horizon SEs (mirrors the global `linear_trends_effects` placement). When `n_bootstrap > 0`, cumulated SE / t / p / CI are derived from bootstrap per-horizon SEs; when bootstrap produces non-finite SE (e.g., `n_bootstrap=1` degenerate distribution), the cumulated layer's full inference tuple is NaN per the library-wide NaN-on-invalid bootstrap contract. `to_dataframe(level="by_path")` exposes `cumulated_effect` and `cumulated_se` columns (always present, NaN-when-None — mirrors the `cband_*` always-present convention from PR #374). `summary()` renders a `Cumulated Level Effects (DID^{fd}, trends_linear)` sub-section under each per-path block. **Path enumeration uses the post-first-differenced `N_mat_fd`**: switchers with `F_g==2` fail the window-eligibility check and are dropped from path enumeration entirely (the existing global `F_g >= 3` warning at line 1620 surfaces the issue), so a path whose switchers all have `F_g < 3` is silently absent from `path_effects` rather than present-with-NaN. **F_g=3 boundary-case divergence (`by_path + trends_linear`):** `F_g=3` switchers have exactly 2 pre-switch periods, which after first-differencing and the `time==1` filter leaves only 1 valid pre-window Z value. R's per-path full-pipeline call handles this single-pre-period regime differently from Python's global-then-disaggregate architecture, producing 30%+ relative divergence on point estimates for paths whose switchers include `F_g=3` (empirically observed on the parity fixture's earlier `F_g=3` variant). A separate `UserWarning` fires at fit-time when the panel includes any `F_g=3` switcher AND `by_path + trends_linear` is set, mirroring the `F_g < 3` exclusion warning. The shipped parity fixture (`single_baseline_multi_path_by_path_trends_lin`) restricts to `F_g >= 4` exclusively to avoid this regime; per-path R parity is asserted only there. **Placebo under `trends_linear` returns RAW per-horizon values** (no per-path placebo cumulation surface) — verified empirically against the existing `joiners_only_trends_lin` parity fixture: R's per-path Placebo_l matches Python's `path_placebo_event_study[path][-l]` (raw) bit-exactly under non-`by_path` trends_lin. **Deviation from R on multi-baseline switcher panels (point estimates):** R `did_multiplegt_dyn(..., by_path, trends_lin)` re-runs the full pipeline (including first-differencing) on each path's restricted subsample, so it operates on different switcher samples per path when switchers have different baseline values `D_{g,1}`. Python first-differences once globally before path enumeration. On single-baseline switcher panels the two architectures coincide; on multi-baseline switcher panels per-path point estimates can diverge — a `UserWarning` is emitted at fit-time when this configuration is detected so practitioners do not silently consume estimates that disagree with R (mirroring the analogous `by_path + covariates` warning). Per-path R parity is confirmed against `did_multiplegt_dyn(..., by_path=3, trends_lin=TRUE, placebo=1)` at `tests/test_chaisemartin_dhaultfoeuille_parity.py::TestDCDHDynRParityByPathTrendsLinear` on the `single_baseline_multi_path_by_path_trends_lin` scenario (single-baseline + cohort-single-path + `F_g >= 4` DGP designed to eliminate the multi-baseline divergence, the cross-path cohort-sharing deviation, and the F_g=3 boundary case under R's per-path full-pipeline call). Per-path cumulated point estimates match R bit-exactly (rtol ~1e-9) on event horizons under those conditions; cumulated SE_RTOL is widened to `0.20` (vs `0.12` used for non-cumulated by_path parity) because the conservative upper-bound SE compounds the cross-path cohort-sharing deviation under summation. **Placebo parity is intentionally skipped for `trends_linear`**: R's per-path placebo computation re-runs on the path-restricted subsample with different control eligibility than Python's global-then-disaggregate architecture surfaces, producing a sign-and-magnitude divergence on paths whose switchers have minimal pre-window depth (e.g., `F_g=4` switchers). Placebo under `by_path + trends_linear` is exercised via internal regression in `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathTrendsLinear` (finite values, bootstrap inheritance) but not pinned to R bit-by-bit. Cross-surface invariants (analytical + bootstrap + placebo + sup-t + `path_cumulated_event_study` + `to_dataframe` columns + `summary()` rendering) are regression-tested at `TestByPathTrendsLinear`. **Per-path state-set trends:** when `trends_nonparam="state_col"` is set with `by_path=k`, the set membership column is validated and stored once globally as `set_ids_arr` (time-invariance, NaN rejection, partition-coarseness checks unchanged from the non-by_path path). The `set_ids` parameter is threaded through the four per-path IF helpers (`_compute_path_effects`, `_compute_path_placebos`, `_collect_path_bootstrap_inputs`, `_collect_path_placebo_bootstrap_inputs`) so per-path analytical SE, bootstrap, placebos, and sup-t bands all consume the set-restricted control pool automatically. R does NOT first-difference and does NOT cumulate under `trends_nonparam` (unlike `trends_lin`); per-horizon `Effect_l` is a normal DID with set-restricted controls. Per-path R parity is confirmed against `did_multiplegt_dyn(..., by_path=3, trends_nonparam="state", placebo=1)` at `tests/test_chaisemartin_dhaultfoeuille_parity.py::TestDCDHDynRParityByPathTrendsNonparam` on the `multi_path_reversible_by_path_trends_nonparam` scenario; per-path point estimates AND placebos match R bit-exactly (rtol ~1e-9), per-path SE matches within the Phase 2 envelope (~13% rtol observed). Cross-surface invariants are regression-tested at `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathTrendsNonparam`. **Per-path non-binary treatment:** integer-coded discrete treatment (D in Z, e.g. ordinal {0, 1, 2}) is supported under `by_path=k` and `paths_of_interest`. Path tuples become integer-state tuples (`(0, 2, 2, 2)`) keyed bit-for-bit against R's comma-separated path strings (`"0,2,2,2"`) for D in {0..9}. Continuous D (e.g. `1.5`) raises `ValueError` at fit-time per the no-silent-failures contract — the existing `int(round(float(v)))` cast in `_enumerate_treatment_paths` is now defensive (no-op for integer-coded D). **Deviation from R for multi-character baseline states (D >= 10 or negative D):** R's `did_multiplegt_by_path` derives the per-path baseline via `path_index$baseline_XX <- substr(path_index$path, 1, 1)` (extracted 2026-05-03 via `Rscript -e 'cat(paste(deparse(DIDmultiplegtDYN:::did_multiplegt_by_path), collapse="\n"))'`), capturing only the first character of the comma-separated path string. For multi-character baselines this drops the rest of the value: for `path = "12,12,..."` it captures `"1"` instead of `"12"`; for `path = "-1,-1,..."` it captures `"-"` instead of `"-1"`. R's per-path control-pool subset is mis-allocated in both regimes. Python's tuple-key matching is correct in both — the per-path point estimates we compute are correct, R's per-path subset for the same path is buggy. The shipped R-parity scenarios stay in `D in {0, 1, 2}` to avoid the R bug; R-parity is asserted on that set at `tests/test_chaisemartin_dhaultfoeuille_parity.py::TestDCDHDynRParityByPathNonBinary` via the `multi_path_reversible_by_path_non_binary` scenario (78 switchers, 3 paths, single-baseline custom DGP, F_g >= 4). The string-encoding compatibility extends to all single-digit nonnegative D (`{0..9}`) since each value renders as a single character, but no R-parity scenario currently exercises D outside `{0, 1, 2}` — per-path point estimates match R bit-exactly (rtol ~1e-9 events; rtol+atol envelope for placebo near-zero values), SE inherits the documented cross-path cohort-sharing deviation (~5% rtol observed; SE_RTOL=0.15 envelope). Negative-integer treatment-state support is regression-tested in Python only (no R parity — R is the buggy side on multi-character baselines) at two sites: `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathNonBinary::test_negative_integer_D_supported` covers paths with negative values in non-baseline positions (e.g. `(0, -1, -1, -1)`), and `::test_negative_baseline_path_supported` covers paths starting with a negative baseline `D_{g,1} = -1` (e.g. `(-1, 0, 0, 0)`, `(-1, 1, 1, 1)`) — the exact regime that triggers R's `substr` bug. Cross-surface invariants regression-tested at `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathNonBinary`. **Per-path survey-design SE** (analytical Binder TSL + replicate-weight bootstrap): under `by_path` / `paths_of_interest` + `survey_design`, the per-path per-horizon SE routes through `_survey_se_from_group_if` using the cell-period allocator. The per-path influence function `U_pp_l_path` is the per-period IF with non-path switcher-side contributions skipped — control contributions remain unchanged, matching the joiners/leavers IF convention from the **Per-path SE convention** paragraph above (the `switcher_subset_mask` zeroes the switcher row of the per-group IF, which trivially zeroes the corresponding row of the per-cell IF, preserving the row-sum identity `U_pp.sum(axis=1) == U`). The IF is cohort-recentered via `_cohort_recenter_per_period` and expanded to observations as `psi_i = U_pp[g_i, t_i] · (w_i / W_{g_i, t_i})`. Replicate-weight designs unconditionally route through the cell allocator (Class A contract, PR #323). Multiplier bootstrap (`n_bootstrap > 0`) under `survey_design + by_path/paths_of_interest` raises `NotImplementedError` at fit-time — the survey-aware perturbation pivot for path-restricted IFs is methodologically underived and deferred to a future wave; the global non-by_path TSL multiplier bootstrap is unaffected and continues to ship. **Path-enumeration ranking is unweighted** under `survey_design`: top-k selection uses group cardinality (`path_to_count[p]` = number of groups), not population-weight mass — survey weights do not affect which paths are selected as "top-k". A weighted-ranking variant (sum of survey weights per path) is deferred until concrete demand. **`df_survey` propagation:** under replicate weights, every per-path per-horizon fit contributes an `n_valid` count to the shared `_replicate_n_valid_list` accumulator and the final `_effective_df_survey = min(...) - 1` reflects all per-path replicate fits. A post-call `_refresh_path_inference` helper re-runs `safe_inference` on every populated entry so `multi_horizon_inference`, `placebo_horizon_inference`, `path_effects`, and `path_placebos` all use the same final df after per-path appends complete. **Lonely-PSU policy is sample-wide, not per-path** — the `lonely_psu` policy (`remove`/`certainty`/`adjust`) operates on the full design-level PSU/strata structure, not on path-restricted subsamples. **Telescope invariant:** on a single-path panel where every switcher follows the same trajectory and `eligible_groups` matches between by_path and non-by_path, per-path SE equals the global non-by_path survey SE bit-exactly — pinned at `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathSurveyDesignTelescope::test_telescope_analytical_TSL`. **Deviation from R:** none — R `did_multiplegt_dyn` does not support survey weighting, so this is a Python-only methodology extension (no R parity available; no R parity test class). Regression test anchor: `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathSurveyDesignAnalytical` covering analytical SE, replicate-weight SE, the `n_bootstrap` gate, the global anti-regression, per-path placebos, `trends_linear` composition, and unobserved-path warnings under survey. **Per-path heterogeneity testing** (analytical OLS / WLS + survey-aware Binder TSL + replicate-weight): under `by_path` / `paths_of_interest` + `heterogeneity="<col>"`, the per-path per-horizon coefficient `beta_X^path_l` is computed by re-running `_compute_heterogeneity_test` on the path-restricted switcher subsample. The path filter (`path_groups: Optional[Set[int]]`) restricts eligibility to switchers ON path `p` inside the inner regression; the variance machinery (HC1-robust OLS vcov for non-survey via `solve_ols(..., return_vcov=True)` (`vcov_type="hc1"` default), WLS-on-pweights with cell-period IF allocator for analytical Binder TSL, group-level allocator for Rao-Wu replicate) is unchanged from the global heterogeneity path. **Cohort dummies absorb baseline by construction** — the cohort key `(D_{g,1}, F_g, S_g)` includes baseline, so multi-baseline switcher panels do not produce R-divergence (unlike `covariates` / `trends_linear`); no parallel `UserWarning` is emitted. **R parity:** matches `did_multiplegt_dyn(..., by_path, predict_het)` per-by_level on the `multi_path_reversible_by_path_predict_het` scenario for `beta`, `se`, `t_stat`, and `n_obs` (`BETA_RTOL = 1e-6` on `beta`, `SE_RTOL = 1e-5` on `se` / `t_stat`; the SE tolerance is one decade looser than `BETA_RTOL` to absorb the small OLS denominator-and-cohort-recentering numerical drift observed on this fixture; `n_obs` matches exactly). Inherits the same tolerances as the new global `multi_path_reversible_predict_het` scenario (`TestDCDHDynRParityHeterogeneity`) since the per-path R call is `did_multiplegt_main(..., predict_het=...)` per path-restricted subsample with no additional numerical loss. **R parity (heterogeneity inference, post-2026-05-15 df threading):** Python now passes `df = n_obs - rank(design)` to `safe_inference` on the non-survey OLS path at `chaisemartin_dhaultfoeuille.py`'s `_compute_heterogeneity_test`, matching R's t-distribution with df from the OLS regression (`DIDmultiplegtDYN:::did_multiplegt_main` `t_stat <- qt(0.975, df.residual(model))` site). The numerical rank is computed via `_detect_rank_deficiency` (the same helper `solve_ols` calls internally); the small-sample short-circuit also uses `n_obs <= rank` rather than the pre-PR pre-drop `n_obs <= n_params`, so boundary cases where alias dropping leaves `n_obs > rank > 0` fit correctly instead of NaN-filling. Parity tolerance is `INFERENCE_RTOL = 1e-4` on `p_value` and `conf_int`; `beta` / `se` / `t_stat` continue to use `BETA_RTOL = 1e-6` / `SE_RTOL = 1e-5`. The `t_stat = beta / se` field is distribution-invariant. **Rank-deficient designs:** ``df = n_obs - rank(design)`` uses the post-drop numerical rank via the same ``_detect_rank_deficiency`` helper that ``solve_ols`` calls internally. For full-rank designs (``rank == n_params``) behavior is bit-identical to the pre-PR ``n_obs - n_params`` path; for near-rank-deficient designs that ``solve_ols`` retains rather than NaN-out (e.g., cohort-collinearity at high horizons), the post-drop rank is strictly lower and the post-PR ``df`` is strictly larger, matching R's ``lm()`` convention. Fully rank-deficient designs continue to NaN-fill via the rank-deficient short-circuit at ``_compute_heterogeneity_test``. R's `dont_drop_larger_lower=TRUE` is set in both fixture scenarios to match the Python `drop_larger_lower=False` requirement. **Survey composition:** inherits from the **Per-path survey-design SE** paragraph above — analytical Binder TSL routes through `_survey_se_from_group_if`'s cell-period allocator on the post-period of the transition; replicate-weights route through the group-level allocator. Multiplier bootstrap (`n_bootstrap > 0`) under `by_path + heterogeneity + survey_design` inherits the existing per-path multiplier-bootstrap-survey gate. **`df_survey` propagation:** every per-(path, horizon) replicate-weight fit appends `n_valid` to the shared `_replicate_n_valid_list` accumulator; per-path heterogeneity inference is refreshed with the FINAL `_effective_df_survey(...)` in the R2 P1b refresh block (separate dedicated loop because the schema shape is `{path: {l: {...}}}` rather than `{path: {"horizons": {l: {...}}}}`). **Result schema:** `results.path_heterogeneity_effects: Dict[Tuple[int, ...], Dict[int, Dict[str, Any]]]` keyed `{path: {l: {beta, se, t_stat, p_value, conf_int, n_obs}}}`. Empty-state contract mirrors `path_effects`: `None` when not requested, `{}` when requested but no path has eligible switchers. **DataFrame integration:** `to_dataframe(level="by_path")` adds always-present `het_*` columns (`het_beta`, `het_se`, `het_t_stat`, `het_p_value`, `het_conf_int_lower`, `het_conf_int_upper`), populated for positive-horizon rows when `heterogeneity` is set and NaN otherwise (mirrors the `cband_*` and `cumulated_*` always-present convention). **Per-path placebo heterogeneity (`placebo + predict_het + by_path`, post-2026-05-15):** R-verified — `did_multiplegt_dyn(by_path, predict_het, placebo)` emits per-path heterogeneity OLS results on backward (placebo) horizons via R's per-by_level dispatcher (`DIDmultiplegtDYN:::did_multiplegt_main` placebo block at the `effect = matrix(-i, ...)` rbind site). R's predict_het syntax: passing `predict_het = list("X", c(-1))` with `placebo > 0` triggers "compute heterogeneity for ALL forward (1..effects) AND ALL placebo (1..placebo) positions"; forward rows have positive `effect` values, placebo rows negative. Python mirrors via `_compute_heterogeneity_test(..., placebo=L_max)` (set when `self.placebo` is truthy) — the function iterates forward (1..L_max) and backward (-1..-L_max) horizons in a single loop with an explicit `out_idx < 0` eligibility guard for backward horizons whose `F_g` is too small (would otherwise silently misread `N_mat` via numpy negative indexing). Placebo rows in `to_dataframe(level="by_path")` have non-NaN `het_*` columns when `placebo=True` and `heterogeneity=` are both set; `path_heterogeneity_effects` uses negative-int keys for backward horizons, mirroring the existing `path_placebo_event_study` convention. **Survey gate (warn + skip):** `survey_design + placebo + heterogeneity` emits a `UserWarning` at fit-time and falls back to forward-horizon-only heterogeneity (codex R1 P1 #1: the eager raise broke the previously-supported forward-horizon survey + predict_het path under the default `placebo=True`) — the Binder TSL cell-period allocator's justification (Survey IF expansion Note above) is tied to **post-period** attribution (`out_idx = first_switch_idx[g] - 1 + l_h` with `l_h > 0`); backward-horizon attribution puts ψ_g mass on a pre-period cell, which is a separate library-extension claim that needs its own derivation. Forward-horizon `predict_het + survey_design` continues to work unchanged on both global and per-path surfaces. The function-level `_compute_heterogeneity_test` keeps a per-iteration backstop that raises `NotImplementedError` if a direct caller bypasses fit() and passes `survey + placebo > 0` (regression-tested at `test_compute_heterogeneity_test_direct_call_raises_on_backward_survey`). Pre-period allocator derivation is deferred to a follow-up methodology PR. R parity confirmed at `tests/test_chaisemartin_dhaultfoeuille_parity.py::TestDCDHDynRParityByPathHeterogeneityWithPlacebo` on the `multi_path_reversible_predict_het_with_placebo` fixture (scenario 22, `placebo=2, effects=3, by_path=3, predict_het=list("het_x", c(-1))`) AND `::TestDCDHDynRParityHeterogeneityWithPlacebo` on the global anchor (`multi_path_reversible_predict_het_with_placebo_global`, scenario 23, same DGP without by_path) — both surfaces emit forward + backward heterogeneity rows in matching parity. Pinned at `BETA_RTOL=1e-6` / `SE_RTOL=1e-5` for `beta` / `se` / `t_stat` / `n_obs`; `INFERENCE_RTOL=1e-4` for `p_value` / `conf_int`. Cross-surface invariants regression-tested at `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathPredictHetPlacebo`. Regression test anchors: `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathHeterogeneity` (gate dispatch, behavior, telescope-to-global on single-path panel, zero-signal anti-regression, multi-baseline UserWarning anti-regression, DataFrame integration, edge cases) + `tests/test_chaisemartin_dhaultfoeuille_parity.py::TestDCDHDynRParityHeterogeneity` (global anchor, FIRST `predict_het` parity baseline) + `::TestDCDHDynRParityByPathHeterogeneity` (per-path).

**Per-path user-specified path selection (`paths_of_interest`):** Python-only API extension — R's `did_multiplegt_dyn(..., by_path=k)` only accepts a positive int (top-k automatic ranking) or `-1` (all observed paths) and provides no list-based selection. Activated via `ChaisemartinDHaultfoeuille(paths_of_interest=[(0, 1, 1, 1), (0, 1, 0, 0)], drop_larger_lower=False)` as an alternative to `by_path=k`; the two are **mutually exclusive** (setting both raises `ValueError` at `__init__` and `set_params` time). Each path tuple must have length `L_max + 1`; the type / element / non-empty / length-uniformity checks fire at `__init__`, the length-vs-L_max check fires at fit-time. `bool` and `np.bool_` are explicitly rejected; `np.integer` is accepted and canonicalized to Python `int` for tuple-key consistency. Duplicates emit a `UserWarning` and are deduplicated; paths not observed in the panel emit a `UserWarning` and are omitted from `path_effects`. Paths appear in `results.path_effects` in the user-specified order, modulo deduplication and unobserved-path filtering. Composes with non-binary D and all downstream `by_path` surfaces (bootstrap, per-path placebos, per-path joint sup-t bands, `covariates`, `trends_linear`, `trends_nonparam`) — mechanical filter on observed paths, no methodology change. Behavior + cross-feature regressions live at `tests/test_chaisemartin_dhaultfoeuille.py::TestPathsOfInterest`.

- **Note (Phase 3 `by_path` per-path joint sup-t bands):** When `n_bootstrap > 0` is set with `by_path=k`, per-path joint sup-t simultaneous confidence bands are computed across horizons `1..L_max` within each path. **Methodology:** a single `(n_bootstrap, n_eligible)` multiplier weight matrix (using the estimator's configured `bootstrap_weights` — Rademacher / Mammen / Webb) is drawn per path and broadcast across all horizons of that path, producing correlated bootstrap distributions across horizons within the path. The path-specific critical value `c_p = quantile(max_l |t_l|, 1 - α)` is then used to construct symmetric joint bands `effect_l ± c_p · se_l` per horizon, surfaced in `path_effects[path]["horizons"][l]["cband_conf_int"]` and at top-level `results.path_sup_t_bands[path] = {"crit_value", "alpha", "n_bootstrap", "method", "n_valid_horizons"}`. **Gates:** a path must have `>= 2` valid horizons (finite bootstrap SE > 0) AND a strict majority (more than 50%) of finite sup-t draws to receive a band; otherwise the path is absent from `path_sup_t_bands`. Both gates mirror the OVERALL `event_study_sup_t_bands` semantics at `chaisemartin_dhaultfoeuille_bootstrap.py:605,612`: `len(valid_horizons) >= 2` AND `finite_mask.sum() > 0.5 * n_bootstrap`. Exactly half-finite draws are NOT enough — the gate is strictly greater than half. **Empty-state contract:** `path_sup_t_bands is None` when not requested (no bootstrap, or both `by_path` and `paths_of_interest` are `None`); `{}` when requested but no path passes both gates. **`to_dataframe(level="by_path")` integration:** the table now includes `cband_lower` / `cband_upper` columns for parity with OVERALL `level="event_study"`; populated for positive-horizon rows of paths with a finite sup-t crit, NaN for placebo rows / unbanded paths / the requested-but-empty fallback DataFrame. **Methodology asymmetry vs OVERALL:** OVERALL sup-t reuses the same multi-horizon shared-draw distribution for both the SE in the t-stat denominator and the bootstrap distribution in the numerator. The per-path sup-t draws a fresh shared weight matrix per path AFTER the per-path SE bootstrap block has already populated `results.path_ses` via independent per-(path, horizon) draws — numerator: fresh shared draws, denominator: bootstrap SEs from the earlier independent draws. Asymptotically equivalent to OVERALL's self-consistent reuse, but NOT bit-identical. The fresh draw is intentional: it preserves RNG-state isolation and keeps every existing per-path SE seed-reproducibility test bit-stable post-implementation. **Inherited deviation from R:** the bootstrap SE used as the t-stat denominator carries the cross-path cohort-sharing SE deviation from R documented for `path_effects` above; the per-path sup-t crit therefore inherits the same deviation. **Interpretation:** the band covers joint inference *within a single path across horizons*; it does NOT provide simultaneous coverage *across paths* (a different inference target requiring a `path × horizon` re-derivation, deferred to a future wave). **Deviation from R:** `did_multiplegt_dyn` provides no joint / sup-t / simultaneous bands at any surface — this is a Python-only methodology extension, consistent with the existing OVERALL `event_study_sup_t_bands` (also Python-only). Regression test anchor: `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathSupTBands`.

**Reference implementation(s):**
- R: [`DIDmultiplegtDYN`](https://cran.r-project.org/package=DIDmultiplegtDYN) (CRAN, maintained by the paper authors). The Python implementation matches `did_multiplegt_dyn(..., effects=1)` at horizon `l = 1`. Parity tests live in `tests/test_chaisemartin_dhaultfoeuille_parity.py`.
- Stata: `did_multiplegt_dyn` (SSC, also maintained by the paper authors).

**Requirements checklist:**
- [x] Single class `ChaisemartinDHaultfoeuille` (alias `DCDH`); not a family
- [x] Forward-compat `fit()` signature with `NotImplementedError` gate for `aggregate`; survey_design now supported (pweight + strata/PSU/FPC via TSL); Phase 3 gates lifted for `covariates` (formerly `controls`), `trends_linear`, `trends_nonparam`, `honest_did`
- [x] `DID_M` point estimate with cohort-recentered analytical SE
- [x] Joiners-only `DID_+` and leavers-only `DID_-` decompositions with their own inference
- [x] Single-lag placebo `DID_M^pl` (point estimate; SE deferred to Phase 2)
- [x] TWFE decomposition diagnostic (Theorem 1 of AER 2020): per-cell weights, fraction negative, `sigma_fe`, `beta_fe`
- [x] Standalone `twowayfeweights()` helper for users who only want the TWFE diagnostic
- [x] Multiplier bootstrap with Rademacher / Mammen / Webb weights, clustered at group by default; automatically upgraded to PSU-level Hall-Mammen wild clustering under `survey_design` with strictly-coarser PSUs
- [x] `drop_larger_lower=True` default (matches R `DIDmultiplegtDYN`); `False` opt-in with explicit inconsistency warning
- [x] Singleton-baseline filter (footnote 15 of dynamic paper, variance computation only) with explicit warning
- [x] Never-switching groups participate in the variance via stable-control roles after the Round 2 full-IF fix; `n_groups_dropped_never_switching` field retained as backwards-compatibility metadata only
- [x] Balanced-baseline panel requirement: missing-baseline groups raise `ValueError`; interior-gap groups dropped with `UserWarning`; terminal missingness retained (deviation from R `DIDmultiplegtDYN` documented as a Note)
- [x] A11 zero-retention convention with per-period boolean flags (`did_plus_t_a11_zeroed` / `did_minus_t_a11_zeroed`) and consolidated warning
- [x] No silent failures: every drop / round / fallback emits a `warnings.warn()` or `ValueError`
- [x] Hand-calculable 4-group worked example: `DID_M = 2.5`, `DID_+ = 2.0`, `DID_- = 3.0` exactly
- [x] R `DIDmultiplegtDYN` parity tests at `l = 1` (fixture skips cleanly when R or `DIDmultiplegtDYN` is unavailable)
- [x] DID^X covariate residualization via per-baseline OLS (Web Appendix Section 1.2)
- [x] DID^{fd} group-specific linear trends via Z_mat first-differencing (Web Appendix Section 1.3)
- [x] State-set-specific trends via control-pool restriction (Web Appendix Section 1.4)
- [x] Heterogeneity testing via saturated OLS (Web Appendix Section 1.5, Lemma 7)
- [x] Design-2 switch-in/switch-out descriptive wrapper (Web Appendix Section 1.6)
- [x] `by_path` per-path event-study disaggregation (binary or integer-coded discrete treatment, joiners/leavers IF precedent; mirrors R `did_multiplegt_dyn(..., by_path=k)`); plus `paths_of_interest=[(...), ...]` for user-specified path subsets (Python-only API; mutex with `by_path`); composes with `survey_design` for analytical Binder TSL and replicate-weight bootstrap SE (multiplier-bootstrap path under survey gated, deferred)
- [x] HonestDiD (Rambachan-Roth 2023) integration on placebo + event study surface
- [x] Survey design support: pweight with strata/PSU/FPC via Taylor Series Linearization (analytical) **or replicate-weight variance (BRR/Fay/JK1/JKn/SDR)**, covering the main ATT surface, covariate adjustment (DID^X), heterogeneity testing, the TWFE diagnostic (fit and standalone `twowayfeweights()` helper), and HonestDiD bounds. Opt-in **PSU-level Hall-Mammen wild bootstrap** is also supported via `n_bootstrap > 0`.
- **Note (Survey IF expansion — library convention):** Survey IF expansion is a library extension not in the dCDH papers (the paper's plug-in variance assumes iid sampling). The library convention builds observation-level `psi_i` by proportionally distributing per-group IF mass within weight share: either at the group level (`psi_i = U_centered[g] * w_i / W_g`, the previous convention) or at the per-`(g, t)` cell level via the cell-period allocator shipped in this release. Cell-level expansion: decompose `U[g]` into per-period attributions `U[g, t]`, cohort-center each column independently, then expand to observation level as `psi_i = U_centered_per_period[g_i, t_i] * (w_i / W_{g_i, t_i})`. Binder (1983) stratified-PSU variance aggregates the resulting `psi` at PSU level. **Post-period attribution convention:** each transition term in the IF sum (of the form `role_weight * (Y_{g, t} - Y_{g, t-1})` for DID_M or `S_g * (Y_{g, out} - Y_{g, ref})` for DID_l) is attributed as a single *difference* to the POST-period cell, not split into a `+Y_post` / `-Y_pre` pair across two cells. This is a library *convention*, not a theorem — adopted because it preserves the group-sum, PSU-sum, and cohort-sum identities of the previous group-level expansion (so Binder variance coincides with the group-level variance under the auto-injected `psu=group`) and because Monte Carlo coverage at nominal 95% is empirically close to nominal on a DGP where PSUs vary across the cells of each group (see `tests/test_dcdh_cell_period_coverage.py`). A covariance-aware two-cell allocator is a plausible alternative and may be worth exploring if future designs motivate an explicit observation-level IF derivation; the method currently in the library is **not derived from the observation-level survey linearization of the contrast** and makes no stronger claim than "coverage is approximately nominal under the tested DGPs and the group-sum identity holds exactly." Under within-group-constant PSU (the pre-allocator accepted input), per-cell sums telescope to `U_centered[g]` and Binder variance is byte-identical (up to single-ULP floating-point noise) to the previous group-level expansion. **Strata and PSU must be constant within each `(g, t)` cell** (trivially satisfied in one-obs-per-cell panels — the canonical dCDH structure); variation **across cells of a group** is supported by the allocator. Within-group-varying **weights** are supported as before. When `survey_design.psu` is not specified, `fit()` auto-injects `psu=<group column>` so the TSL variance, `df_survey`, and t-based inference match the per-group PSU structure. **Strata that vary across cells of a group require either an explicit `psu=<col>` or the original `SurveyDesign(..., nest=True)` flag** — under `nest=True` the resolver combines `(stratum, psu)` into globally-unique labels, so the auto-injected `psu=<group>` is re-labeled per stratum and the cell allocator proceeds. Only the `nest=False` + varying-strata + omitted-psu combination is rejected up front with a targeted `ValueError` at `fit()` time (the synthesized PSU column would reuse group labels across strata and trip the cross-stratum PSU uniqueness check in `SurveyDesign.resolve()`). Under replicate-weight designs, the same cell-level `psi_i` is aggregated via Rao-Wu weight-ratio rescaling (`compute_replicate_if_variance` at `diff_diff/survey.py:1681`) rather than the Binder TSL formula. All five methods (BRR/Fay/JK1/JKn/SDR) are supported method-agnostically through the unified helper; the effective `df_survey` is reduced to `min(n_valid) - 1` across IF sites when some replicate solves fail (matching `efficient_did.py:1133-1135` and `triple_diff.py:676-686` precedents). Under DID^X, the first-stage residualization coefficient `theta_hat` is computed once on full-sample weights and treated as fixed (FWL plug-in IF convention) — per-replicate refits of `theta_hat` are not performed. **Post-period attribution extends to heterogeneity (Binder TSL branch only):** the heterogeneity WLS coefficient IF `ψ_g = inv(X'WX)[1,:] @ x_g * W_g * r_g` is attributed in full to the single post-period cell `(g, out_idx)` at each horizon (same single-cell convention as DID_l), then expanded as `ψ_i = ψ_g * (w_i / W_{g, out_idx})`, and fed through `compute_survey_if_variance`. Under PSU=group the PSU-level aggregate telescopes to `ψ_g`, so Binder variance is byte-identical relative to the pre-cell-period release; under within-group-varying PSU mass lands in the post-period PSU. **Replicate-weight branch keeps the legacy group-level allocator** `ψ_i = ψ_g * (w_i / W_g)` because `compute_replicate_if_variance` computes `θ_r = sum_i ratio_ir * ψ_i` at observation level and is therefore not PSU-telescoping: redistributing mass onto the post-period cell would silently change the replicate SE whenever a replicate column's ratios vary within a group (the library accepts arbitrary per-row replicate matrices, not just PSU-aligned ones). The legacy allocator preserves byte-identity of the replicate SE for every previously-supported fit. Replicate + within-group-varying PSU is unreachable by construction (`SurveyDesign` rejects `replicate_weights` combined with explicit `strata/psu/fpc`).
- **Note (survey + bootstrap contract):** When `survey_design` and `n_bootstrap > 0` are both active, the bootstrap uses Hall-Mammen wild multiplier weights (Rademacher/Mammen/Webb) **at the PSU level**. Under the default auto-injected `psu=group`, the PSU coincides with the group so the wild bootstrap is a clean group-level clustered bootstrap (identity-map fast path, bit-identical to the non-survey multiplier bootstrap). When the user passes an explicit strictly-coarser PSU (e.g., `psu=state` with groups at county level), the IF contributions of all groups within a PSU receive the same bootstrap multiplier — the standard Hall-Mammen wild PSU bootstrap. Strata do not participate in the bootstrap randomization (they contribute only through the analytical TSL variance); this is conservative when strata differ substantially in variance. A `UserWarning` fires only when PSU is strictly coarser than group. **Cell-level wild PSU bootstrap under within-group-varying PSU:** when the PSU varies across the cells of a group, the bootstrap switches to a cell-level allocator: each `(g, t)` cell draws its multiplier from `w[psu(cell)]` via the per-cell PSU map `psu_codes_per_cell` (shape `(n_eligible_groups, n_periods)`, -1 sentinel for zero-weight cells). The bootstrap statistic becomes `theta_r = sum_c w[psu(c)] * u_centered_pp[c] / divisor` using the cohort-recentered per-cell IF `U_centered_per_period`. Under PSU-within-group-constant regimes (including PSU=group and strictly-coarser PSU with within-group constancy), the per-cell sum telescopes to the group-level form via the row-sum identity `sum_{c in g} U_centered_per_period[g, t] == U_centered[g]` (enforced by `_cohort_recenter_per_period`). A dispatcher in `_compute_dcdh_bootstrap` detects within-group-constancy and routes those regimes through the legacy group-level bootstrap path so their SE is **bit-identical** to the pre-cell-level release (guarded primarily by `test_bootstrap_se_matches_pre_pr4_baseline` and by the existing `test_auto_inject_bit_identical_to_group_level`). Under within-group-varying PSU, a group contributing cells to PSUs `p1, p2, ...` receives independent multiplier draws per PSU — the correct Hall-Mammen wild PSU clustering at cell granularity. **Multi-horizon bootstraps** draw a single shared `(n_bootstrap, n_psu)` PSU-level weight matrix per block and broadcast per-horizon via each horizon's cell-to-PSU map, so the sup-t simultaneous confidence band remains a valid joint distribution across horizons. **Library extension** — R `DIDmultiplegtDYN` does not support survey designs, so "deviation from R" does not apply. **Scope note (terminal missingness + any cell-period-allocator path):** see the balanced-baseline Note above for the full carve-out. In brief: when a terminally-missing group is in a cohort whose other groups still contribute at the missing period, `_cohort_recenter_per_period` leaks non-zero centered IF mass onto cells with no positive-weight observations. The targeted `ValueError` fires from every survey variance path that uses the cell-period allocator: Binder TSL with within-group-varying PSU, Rao-Wu replicate ATT (which always uses the cell allocator), and the cell-level wild PSU bootstrap. Pre-process the panel to remove terminal missingness, or (for Binder TSL only) use an explicit `psu=<group_col>` so the analytical path routes through the legacy group-level allocator. **Replicate-weight designs and `n_bootstrap > 0` are mutually exclusive** (replicate variance is closed-form; bootstrap would double-count variance) — the combination raises `NotImplementedError`, matching `efficient_did.py:989`, `staggered.py:1869`, `two_stage.py:251-253`. For HonestDiD bounds under replicate weights, the replicate-effective `df_survey = min(resolved_survey.df_survey, min(n_valid_across_sites) - 1)` propagates to t-critical values — capped by the design's QR-rank-based df so a rank-deficient replicate matrix never produces a larger effective df than the design supports. When `resolved_survey.df_survey` is undefined (QR-rank ≤ 1), the effective df stays `None` and all inference fields (including HonestDiD bounds) are NaN — per-site `n_valid` cannot rescue a rank-deficient design.

### Deviations from the paper / from R / library extensions

*Notes #1, #2, #3, #4, #5, and #7 codify deviations from R `DIDmultiplegtDYN` (and from the paper's Equation 3 in the case of #1). Note #6 codifies a library extension with no R correspondence. The original scattered `**Note:**` and `**Note (deviation from R...):**` entries throughout the section above remain in place — this Deviations block is the canonical AI-review surface per CLAUDE.md "Documenting Deviations (AI Review Compatibility)" labels. Cross-references back to the existing Notes use semantic anchors (Phase / section names) rather than line numbers because the DCDH section is liable to shift as new contracts land; test-file references retain line numbers / class names because test files are more stable.*

1. **Deviation from R / Deviation from the paper (Equation 3):** Equal-cell weighting — each `(g,t)` cell contributes equally regardless of within-cell observation count. AER 2020 Equation 3 prescribes `N_{d,d',t} = sum_g N_{g,t}` (observation sums); R `DIDmultiplegtDYN` weights by cell size. Phase 2 estimands (`DID_l`, `DID^{pl}_l`, `DID^n_l`, delta cost-benefit) inherit the same contract. Locked in `tests/test_chaisemartin_dhaultfoeuille.py::TestDropLargerLower::test_cell_count_weighting_unbalanced_input`. Cross-references the Phase 1 Theorem 3 equation block above (where `N_{a,b,t}` is documented as the count of `(g, t)` cells in each transition state) and `METHODOLOGY_REVIEW.md` § DCDH Deviations #1.
2. **Deviation from R:** Period-based stable-control sets (`stable_0(t)` = any cell with `D_{g,t-1} = D_{g,t} = 0` regardless of baseline `D_{g,1}`) — R uses cohort-based control sets that additionally require baseline `D_{g,1}` to match the side. Pure-direction panels agree exactly; ~1% point-estimate divergence on mixed-direction panels where joiners' post-switch cells could serve as leavers' controls. SE parity gap on pure-direction scenarios narrowed from ~18% to ~3% after the Round 2 full-IF fix. Cross-references the existing `**Note (deviation from R DIDmultiplegtDYN):**` in the period-vs-cohort discussion above and `METHODOLOGY_REVIEW.md` § DCDH Deviations #2.
3. **Deviation from R:** Balanced-baseline panel required + interior-gap drops + terminal-missingness retention + cell-period-allocator targeted `ValueError` — one composite deviation with four enforcement paths. Step 5b validation in `fit()` enforces the contract via `ValueError` (missing baseline) / `UserWarning` (interior gaps) / silent retention (terminal missingness). R accepts unbalanced panels. The cell-period allocator paths (Binder TSL with within-group-varying PSU, Rao-Wu replicate ATT, cell-level wild PSU bootstrap) have a targeted `ValueError` when cohort recentering would leak nonzero centered IF mass onto cells with no positive-weight observations. The four enforcement paths share a single underlying contract — "the panel must be balanced at baseline; terminal missingness is the only allowed unbalance; downstream variance machinery refuses to silently leak IF mass past the cell-period boundary". Cross-references the existing `**Note (deviation from R DIDmultiplegtDYN):**` in the ragged-panel discussion above (which itself details the three affected cell-period-allocator sub-paths) and `METHODOLOGY_REVIEW.md` § DCDH Deviations #3.
4. **Deviation from R:** SE normalization — Python uses paper Section 3.7.3 verbatim `SE = sigma-hat / sqrt(N_l)`; R normalizes by `G` (total groups). Analytical SE is ~4% smaller than R on identical data (deterministic; ~3.5-5.1% across horizons and scenarios). Both converge to the same asymptotic variance as `G → ∞`. Cross-references the `**Note (deviation from R DIDmultiplegtDYN - SE normalization):**` in the SE / variance discussion above and `METHODOLOGY_REVIEW.md` § DCDH Deviations #4.
5. **Deviation from R:** Singleton-cohort degeneracy → `NaN` with `UserWarning`. R returns a non-zero SE via small-sample sandwich machinery that Python does not implement. Bootstrap inherits the same degeneracy. Cross-references the singleton-cohort `**Note:**` in the SE / variance discussion above and `METHODOLOGY_REVIEW.md` § DCDH Deviations #5.
6. **Library extension (no R correspondence):** `<50%` switcher warning at far horizons. Library convention is to warn but compute; the dynamic paper (NBER WP 29873) recommends not reporting such horizons (Favara-Imbs application, footnote 14). Cross-references the `**Note (Phase 2 \`<50%\` switcher warning):**` in the Phase 2 discussion above and `METHODOLOGY_REVIEW.md` § DCDH Deviations #6.
7. **Deviation from R:** Phase 3 `DID^X` covariate adjustment uses equal cell weights in the first-stage OLS (consistent with the Phase 1 cell-count convention, deviation #1). R weights by `N_{gt}`. On one-observation-per-cell panels results are identical. When baseline-specific first stages fail (`n_obs = 0` or `n_obs < n_params`), both Python and R drop the affected strata. Cross-references the `**Note (Phase 3 DID^X covariate adjustment):**` in the Phase 3 discussion above and `METHODOLOGY_REVIEW.md` § DCDH Deviations #7.

---

## ContinuousDiD

**Primary Source:** Callaway, Goodman-Bacon & Sant'Anna (2024), "Difference-in-Differences with a Continuous Treatment," NBER Working Paper 32117.

**R Reference:** `contdid` v0.1.0 (CRAN).

- **Note (rank-guarded ACRT-variance bread):** The ACRT influence-function bread `(Psi'WPsi / mass)^{-1}` (`_compute_dose_response_gt`, `continuous_did.py`) is inverted by the shared `_rank_guarded_inv` (`diff_diff/linalg.py`). `np.linalg.inv` raises only on an *exactly* singular B-spline design Gram; a **near**-singular Gram (clustered doses / near-duplicate knots) previously returned a garbage inverse (~1e13), and the exact-singular fallback was a *silent* minimum-norm `pinv`. The rank-guard truncates redundant directions on the equilibrated Gram → a finite SE on the identified subspace (the well-conditioned near-collinear limit, **not** minimum-norm; NaN only at rank 0), and `fit()` warns when a direction is dropped. See the CallawaySantAnna "rank-guarded IF standard errors" Note for the generalized-inverse semantics.

### Identification

Two levels of parallel trends (following CGBS 2024, Assumptions 1-2):

**Parallel Trends (PT):** for all doses d in D_+,
`E[Y_t(0) - Y_{t-1}(0) | D = d] = E[Y_t(0) - Y_{t-1}(0) | D = 0]`.
Untreated potential outcome paths are the same across all dose groups and the
untreated group. Stronger than binary PT because it conditions on specific dose values.
Identifies: `ATT(d|d)`, `ATT^{loc}`. Does NOT identify `ATT(d)`, `ACRT`, or cross-dose comparisons.

**Strong Parallel Trends (SPT):** additionally, for all d in D,
`E[Y_t(d) - Y_{t-1}(0) | D > 0] = E[Y_t(d) - Y_{t-1}(0) | D = d]`.
No selection into dose groups on the basis of treatment effects.
Implies `ATT(d|d) = ATT(d)` for all d.
Additionally identifies: `ATT(d)`, `ACRT(d)`, `ACRT^{glob}`, and cross-dose comparisons.

**Conditional Parallel Trends (with covariates).** When `covariates=` is passed the PT/SPT
assumptions are conditional on covariates `X`:
`E[Y_t(0) - Y_{t-1}(0) | D = d, X] = E[Y_t(0) - Y_{t-1}(0) | D = 0, X]`. The per-`(g,t)` cell's
control counterfactual becomes a covariate-adjusted prediction instead of the unconditional control
mean (see Key Equations and the covariate Note below).

See `docs/methodology/continuous-did.md` Section 4 for full details.

### Key Equations

**Target parameters:**
- `ATT(d|d) = E[Y_t(d) - Y_t(0) | D = d]` — effect of dose d on units who received dose d (PT)
- `ATT(d) = E[Y_t(d) - Y_t(0) | D > 0]` — dose-response curve (SPT required)
- `ACRT(d) = dATT(d)/dd` — average causal response / marginal effect (SPT required)
- `ATT^{loc} = E[ATT(D|D) | D > 0] = E[Delta Y | D > 0] - E[Delta Y | D = 0]` — binarized ATT (PT); equals `ATT^{glob}` under SPT
- `ATT^{glob} = E[ATT(D) | D > 0]` — global average dose-response level (SPT required)
- `ACRT^{glob} = E[ACRT(D_i) | D > 0]` — plug-in average marginal effect (SPT required)
- **Lowest-dose-as-control (Remark 3.1, `control_group="lowest_dose"`):** when `P(D=0) = 0` (no untreated group) the lowest-dose group `d_L` is the comparison and the targets are rebased to it — `ATT(d) - ATT(d_L)` (with `ATT(d_L) = 0` by construction, the omitted reference), `ACRT(d)` backward-differenced to `d_L` (`ACRT(d_1) = ATT(d_1)/(d_1 - d_L)`), and `ATT^{glob} = E[ΔY | D > d_L] - E[ΔY | D = d_L]`. Requires a genuine lowest-dose group (`P(D = d_L) > 0`). See Note #7.

**Estimation via B-spline OLS:**
1. Compute `Delta_tilde_Y = (Y_t - Y_{t-1})_treated - mean((Y_t - Y_{t-1})_control)`
2. Build B-spline basis `Psi(D_i)` from treated doses
3. OLS: `beta = (Psi'Psi)^{-1} Psi' Delta_tilde_Y`
4. `ATT(d) = Psi(d)' beta`, `ACRT(d) = dPsi(d)/dd' beta`

**Covariate-adjusted `Delta_tilde_Y` (conditional PT).** With `covariates=`, step 1's scalar control
mean is replaced by a per-treated-unit covariate-adjusted counterfactual (`X_i` includes an intercept):
- `reg` (outcome regression): fit `gamma_hat = (X_C'X_C)^{-1} X_C' Delta_Y_C` on controls;
  `Delta_tilde_Y_i = Delta_Y_i - X_i' gamma_hat`.
- `dr` (doubly-robust, DRDID `drdid_panel`): same OLS `gamma_hat`, plus a propensity model and a scalar
  augmentation `eta_cont = odds_weighted_mean_C(Delta_Y - X' gamma_hat)`;
  `Delta_tilde_Y_i = Delta_Y_i - X_i' gamma_hat - eta_cont`.
Steps 2-4 are unchanged. Because the augmentation is a constant, `reg` and `dr` share the same
`ACRT(d)` for the continuous B-spline path (a constant only shifts the B-spline intercept, which
`dPsi` annihilates); they differ only in the `ATT(d)` / `ATT^{glob}` level (by `-eta_cont`) and in the
doubly-robust SE. (On the discrete saturated basis reg/dr share `ACRT(d_j)` for `j >= 2` but differ at
`ACRT(d_1)` — see below.)

**Discrete treatment: saturated regression (`treatment_type="discrete"`, CGBS 2024 Eq. 4.1).** For a
dose taking distinct levels `d_1 < ... < d_J`, steps 2-4 swap the B-spline basis for a saturated
(indicator) basis:
2'. `Psi(D_i)` = indicator columns `1{D_i = d_j}` (no intercept; a partition of unity among treated).
3'. `beta = (Psi'Psi)^{-1} Psi' Delta_tilde_Y`, so `beta_j = mean_{D=d_j}(Delta_tilde_Y) = ATT(d_j)`
    (a per-level 2×2 DiD).
4'. `ATT(d_j) = beta_j`; `ACRT(d_j) = (L beta)_j` where `L` is the paper's **backward**-difference
    operator on the grid `{d_0 = 0, d_1, ..., d_J}` (`ATT(0) = 0`): `[ATT(d_j) - ATT(d_{j-1})]/(d_j -
    d_{j-1})` for `j >= 2`, and at the lowest positive level `ACRT(d_1) = [ATT(d_1) - 0]/(d_1 - 0) =
    ATT(d_1)/d_1` (so binary `D in {0,1}` gives `ACRT = ATT`). `ACRT^{glob} = mean_i ACRT(D_i)`.
The B-spline and saturated paths share the same linear influence-function / bootstrap / covariate /
survey machinery (`bread @ psi_bar = ones(J)` makes the control-side IF reduce to the per-level 2×2
control variance; it cancels in the `j >= 2` adjacent differences whose `L`-rows sum to 0). `reg`/`dr`
share `ACRT(d_j)` point AND SE for `j >= 2` (the constant `eta_cont` cancels in those differences), but
**differ at `ACRT(d_1)` by `eta_cont/d_1`** — the lowest-dose row references the fixed baseline
`ATT(0) = 0`, which the augmentation does not shift, so the dr IF carries the augmentation variance
there (analytical `ACRT(d_1)` SE matches the bootstrap). Otherwise the
`ATT(d_j)` levels differ uniformly by `-eta_cont`. See Notes #5-#6.

### Edge Cases

- **No untreated group**: `control_group="lowest_dose"` implements Remark 3.1 — the lowest-dose group `d_L` becomes the comparison (estimand `ATT(d) - ATT(d_L)`); requires a genuine lowest-dose group (`P(D=d_L) > 0`, `>= 2` units at `d_L`) and no never-treated units. Single-cohort only (multi-cohort and `covariates=` raise; see Note #7). The default `never_treated`/`not_yet_treated` paths still require `P(D=0) > 0`.
- **Discrete treatment**: `treatment_type="discrete"` fits the saturated per-dose-level regression (CGBS 2024 Eq. 4.1; see Note #6). On the default `treatment_type="continuous"` path an integer-valued dose is detected and warns, pointing to `treatment_type="discrete"`.
- **All-same dose**: B-spline basis collapses; ACRT(d) = 0 everywhere.
- **Rank deficiency**: When n_treated <= n_basis, cell is skipped.
- **Balanced panel required**: Matches R `contdid` v0.1.0.
- **Anticipation + not-yet-treated**: Control mask uses `G > t + anticipation`
  (not just `G > t`) to exclude cohorts in the anticipation window from
  not-yet-treated controls. When `anticipation=0` (default), behavior is
  unchanged.
- **Boundary knots**: Knots are built once from all treated doses (global, not per-cell) to ensure a common basis across (g,t) cells for aggregation. Evaluation grid is clamped to training-dose boundary knots (`range(dose)`). R's `contdid` v0.1.0 has an inconsistency where `splines2::bSpline(dvals)` uses `range(dvals)` instead of `range(dose)`, which can produce extrapolation artifacts at dose grid extremes. Our approach avoids extrapolation and is methodologically sound.
- **Note:** `bspline_derivative_design_matrix` previously swallowed `ValueError` from `scipy.interpolate.BSpline` in the per-basis derivative loop, leaving affected columns of the derivative design matrix as zero with no user-facing signal. It now aggregates the failed basis indices and emits ONE `UserWarning` naming them. Both ACRT point estimates and analytical/bootstrap inference read the same `dPsi` matrix (see `_compute_dose_response_gt` - the `acrt_d = dPsi_eval @ beta_pred` point estimate at `continuous_did.py:1806` and the covariate-path IFs at `:1485-1526` - and the bootstrap ACRT path in `_bootstrap_gt_cell` at `continuous_did.py:2274-2340`), so both are biased on a partial derivative-construction failure — the warning wording makes that explicit. The all-identical-knot degenerate case (single dose value) remains silently handled — derivatives there are mathematically zero. Axis-C finding #12 in the Phase 2 silent-failures audit.

### Deviations from the paper / from R / library extensions

*Note #1 codifies a deviation from R `contdid` v0.1.0's boundary-knot
choice (library extension toward methodological soundness — avoids
extrapolation that `contdid` exhibits). Notes #2-#4 codify library
extensions with NO R correspondence — Phase 2 silent-failures audit
fixes that surface previously silent behavior as `UserWarning` or
`ValueError`; `contdid` v0.1.0 absorbs the same conditions without a
signal. The original Edge Cases bullet (under § Edge Cases above) and
the two `**Note:**` entries (under § Implementation Checklist below)
remain in place — this Deviations block is the canonical AI-review
surface per CLAUDE.md "Documenting Deviations (AI Review Compatibility)"
labels.*

1. **Deviation from R:** `range(dose)` vs `range(dvals)` boundary knots — the library uses `range(dose)` (training-dose range) for B-spline boundary knots; R's `contdid` v0.1.0 uses `range(dvals)` via `splines2::bSpline(dvals)`, which can produce extrapolation artifacts at dose-grid extremes. **Scope caveat:** R cross-language coverage therefore runs at **relative** tolerance bands across two surfaces: (a) **scalar parity with raw R `cont_did` / `pte_default`** at 1% relative on overall ATT for all 6 benchmarks and on overall ACRT for benchmarks 4-5; (b) **harmonized boundary-knot-normalized curve parity** with R-side ATT(d) / ACRT(d) reconstructed under `Boundary.knots = range(treated_doses)` (matching the library) on benchmarks 1-3 via the benchmark harness — `_run_r_contdid` does the R-side rebuild at `tests/test_methodology_continuous_did.py:333-367`, and `_compare_with_r` orchestrates the Python-vs-R comparison at `:395-459` — max ATT(d) at 1% and max ACRT(d) at 2%. Benchmark 6 is event-study, scalar `overall_att` only. NOT bit-exact (`atol=1e-8`) like HAD. Library extension toward methodological soundness (avoids extrapolation). Cross-references the § Edge Cases "Boundary knots" bullet above and `METHODOLOGY_REVIEW.md` § ContinuousDiD Deviations #1.
2. **Note:** `bspline_derivative_design_matrix` derivative-failure `UserWarning` — Phase 2 axis-C #12 silent-failures audit fix. No R correspondence; `contdid` v0.1.0 does not implement an equivalent warning. Cross-references the § Edge Cases `**Note:**` bullet above (`bspline_derivative_design_matrix` entry) and `METHODOLOGY_REVIEW.md` § ContinuousDiD Deviations #2. Locked in `tests/test_continuous_did.py::TestBSplineDerivativeDegenerateBasis` (3 tests); source-level aggregate-warning block at `diff_diff/continuous_did_bspline.py:150-187`.
3. **Note:** `+inf` → `0` never-treated recoding emits `UserWarning` reporting the affected row count; negative `first_treat` (including `-inf`) raises `ValueError`. Axis-E silent-coercion fix per Phase 2 audit. No R correspondence; `contdid` v0.1.0 silently absorbs `+inf` without a signal. Cross-references the § Implementation Checklist `**Note:**` below and `METHODOLOGY_REVIEW.md` § ContinuousDiD Deviations #3.
4. **Note:** Zero-`first_treat` rows with nonzero `dose` are force-zeroed with `UserWarning` reporting the affected row count (axis-E silent-coercion). No R correspondence; `contdid` v0.1.0 has the same `first_treat = 0` → `D = 0` invariant but silently coerces without a warning. Cross-references the § Implementation Checklist `**Note:**` below and `METHODOLOGY_REVIEW.md` § ContinuousDiD Deviations #4.
5. **Note (covariate support — library extension beyond `contdid` v0.1.0):** `covariates=` with `estimation_method ∈ {reg, dr}` adds conditional-parallel-trends adjustment. This is a **library extension**: `contdid` v0.1.0 hard-stops on any covariate (`stop("covariates not currently supported…")`), so there is **no external R anchor for the covariate-adjusted dose *curve***. Validation instead: (a) the **scalar `overall_att` + SE** map *exactly* onto `DRDID::reg_did_panel` (reg) / `DRDID::drdid_panel` (dr) — a tight (~1e-8) component anchor, skip-guarded since DRDID is not in CI (`tests/test_methodology_continuous_did.py::TestCovariateReg`); (b) an **R-free NumPy reconstruction** of the reg/dr `att`+SE runs *in CI* at p≥2 (`test_dr_reg_numpy_crosscheck_p2`) — the guard the p=1 reduction cannot provide (at p=1 the intercept-only propensity is constant, so `eta_cont ≡ 0` and dr collapses to reg); (c) DGP recovery + MC coverage (reg 96%, dr 95%). **`ipw` restricted:** `estimation_method="ipw"` with covariates raises `NotImplementedError` — pure IPW's covariate adjustment is a single scalar (a propensity-reweighted control mean) that shifts only the ATT(d) level and leaves `ACRT(d)` identical to the unconditional fit, so it cannot adjust the dose-response *shape*. **Deviations from DRDID:** unit weights are 1 (unweighted; `covariates=` + `survey_design=` raises `NotImplementedError`, deferred); propensity trimming uses clip semantics (`pscore_trim`) rather than DRDID's drop-trimming — identical on moderate-overlap data (the anchor regime), diverging only at extreme propensities. **Fail-closed policies (no-silent-failures):** (i) missing/non-finite covariate values raise `ValueError` up front — a per-cell fallback to unconditional estimation would silently mix conditional-PT and unconditional-PT cells in the aggregate; (ii) `dr` propensity-estimation failure raises by default (`pscore_fallback="error"`) so a `dr` fit never silently degrades to a non-DR estimate — `pscore_fallback="unconditional"` opts into the graceful (warned, reg-like) fallback. **`treatment_type="discrete"` composability:** on the intercept-free saturated basis the constant augmentation `eta_cont` shifts every `beta_j` equally (indicator partition of unity), so it cancels in the adjacent differences and reg/dr share `ACRT(d_j)` point AND SE for `j >= 2` — but the lowest-dose ACRT references the fixed baseline `ATT(0) = 0` (backward-to-zero), so reg/dr **differ at `ACRT(d_1)` by `eta_cont/d_1`** (the dr IF carries the augmentation variance there). See Note #6. Cross-references `docs/methodology/continuous-did.md` § Covariates.
6. **Note (discrete-treatment saturated regression — library extension beyond `contdid` v0.1.0):** `treatment_type="discrete"` estimates the dose-response by a **saturated regression** (CGBS 2024 Eq. 4.1) — one indicator per distinct dose level, so `beta_j = mean_{D=d_j}(ΔY − control) = ATT(d_j)` (a per-level 2×2 DiD) — instead of the B-spline sieve. `ACRT(d_j)` is the paper's **backward difference** on the grid `{d_0 = 0, d_1, …, d_J}` (Eq. 4.1 makes `d_0 = 0` the omitted category with `ATT(0) = 0`): `ACRT(d_j) = [ATT(d_j) − ATT(d_{j-1})]/(d_j − d_{j-1})` for `j ≥ 2`, and at the lowest positive level it references the zero-dose baseline, `ACRT(d_1) = [ATT(d_1) − 0]/(d_1 − 0) = ATT(d_1)/d_1`. So a single positive dose (`J = 1`, e.g. binary `D ∈ {0,1}`) yields `ACRT(d_1) = ATT(d_1)/d_1`, and for `d_1 = 1` the documented binary identity `ACRT = ATT` holds exactly. This is a **library extension**: `contdid` v0.1.0 accepts `treatment_type` in its signature but **does not implement the discrete path** (documented "Discrete treatment not yet implemented"), so there is **no external R anchor**. It is instead an *exact* basis swap of the B-spline design/evaluation/derivative trio for an indicator/identity/finite-difference trio; every downstream quantity is linear in `beta`, so the analytical-SE / multiplier-bootstrap / covariate (reg,dr) / survey machinery is reused unchanged and reduces *analytically* to the per-level 2×2 DiD (`bread @ psi_bar = ones(J)`; the common control mean cancels in the `j ≥ 2` adjacent differences whose `L`-rows sum to 0). **reg vs dr:** the constant DR augmentation `η̄_cont` cancels in the `j ≥ 2` differences, so `ACRT(d_j)` point AND SE are identical for `reg`/`dr` there; but `ACRT(d_1) = ATT(d_1)/d_1` references the fixed baseline `ATT(0) = 0` (not shifted by `η̄_cont`), so `reg` and `dr` genuinely **differ at `ACRT(d_1)` by `η̄_cont/d_1`** (and correspondingly in `ACRT^glob` via the `d_1` mass) — the dr influence function carries the augmentation variance at `d_1` (validated: analytical `ACRT(d_1)` SE matches the multiplier bootstrap). Validation (R-free, in CI): exact hand-calc of `ATT(d_j)`/`ACRT`/`overall_att` and the analytical SE against a direct per-level 2×2 reconstruction (`~1e-12`/`~1e-10`), DGP recovery, and MC coverage for analytical + bootstrap (`tests/test_methodology_continuous_did.py::TestDiscreteSaturated`, `tests/test_continuous_did.py::TestDiscreteSaturatedAPI`). **Fail-closed policies (no-silent-failures):** (i) multi-cohort fits with **heterogeneous dose support** across cohorts raise `NotImplementedError` — an absent global level yields a dropped zero column (`att_d[level]=0`) that the plain-sum dose aggregation would bias toward zero (support-aware aggregation is deferred; single-cohort, 2-period, and shared-support multi-cohort are supported); (ii) a requested `dvals` value that is not an observed dose level raises `ValueError` (a saturated model cannot be evaluated off-support); (iii) an over-parameterized fit (`< 2` treated units per level, or `J > n_treated/2`) warns (degenerate per-level SE); (iv) with `survey_design=`, any dose level with **zero effective treated mass in a `(g,t)` cell** raises `ValueError` — a per-cell check (not just the global positive-weight check), so a level that survey/subpopulation weights zero out for one cohort while another cohort keeps it cannot silently drop to a zero-coefficient saturated column. Cross-references `docs/methodology/continuous-did.md` § 5.1.
7. **Note (lowest-dose-as-control, Remark 3.1 — library extension beyond `contdid` v0.1.0):** `control_group="lowest_dose"` implements CGBS 2024 Remark 3.1 for settings with no untreated group (`P(D=0) = 0`): the lowest-dose group `d_L` becomes the comparison and the estimand is `ATT(d) − ATT(d_L)` (SPT), with `ATT(d_L) = 0` the omitted reference. Mechanically it is a **control-group swap** — the D=0 control pool is replaced by the `d_L` group; the entire linear influence-function / bootstrap / event-study / survey machinery is control-group-generic and reused unchanged (`ee_control` already carries the reference-group variance, so **no new SE plumbing**). On the discrete saturated basis the backward-difference operator's reference shifts from `0` to `d_L` (`ACRT(d_1) = ATT(d_1)/(d_1 − d_L)`); on the continuous B-spline path the reference shifts only `μ_0` (the level), leaving `ACRT = spline'` unchanged. `contdid` v0.1.0 does **not** implement Remark 3.1, so there is **no external R anchor**; validation (R-free, in CI): an **exact `d_L → 0` equivalence** anchor (relabelling a `never_treated` panel's D=0 group as a tiny common dose `d_L = ε` reproduces the `never_treated` ATT and SE exactly, for any ε), a discrete hand-calc of `ATT(d)−ATT(d_L)`/`ACRT`/`overall_att`/`overall_acrt` and the per-level 2×2 SE (`~1e-10`), continuous mass-point DGP recovery, analytical-vs-bootstrap SE agreement, a pre-period placebo, and MC coverage (`tests/test_methodology_continuous_did.py::TestLowestDose`, `tests/test_continuous_did.py::TestLowestDoseAPI`). The continuous path requires a genuine **mass point** at the minimum dose (`>= 2` units at `d_L`, i.e. `P(D=d_L) > 0`) — the Remark 3.1 identification condition; a singleton minimum fails closed. **Fail-closed policies (no-silent-failures):** (i) never-treated units present with `lowest_dose` → `ValueError` (they would be silently dropped); (ii) singleton `d_L` (no mass point) → `ValueError`; (iii) no treated dose above `d_L` → `ValueError`; (iv) user `dvals ≤ d_L` → `ValueError` (`d_L` is the omitted reference); (v) survey/subpopulation weighting that leaves the `d_L` group with `< 2` positive-weight units → `ValueError` (a single positive-weight reference unit gives `ee_control = 0`, i.e. zero control-side variance — the effective-`>= 2` analogue of the raw mass-point guard, applied after weighting); (vi) a boundary gap `d_1 − d_L` that is a tiny fraction of the dose range warns (huge boundary ACRT/SE). **Deferred (fail-closed `NotImplementedError` + TODO):** multi-cohort `lowest_dose` (needs a within-cohort reference + support-aware cross-cohort aggregation) and `covariates=` × `lowest_dose` (conditional-PT-relative-to-`d_L` estimand). Cross-references `docs/methodology/continuous-did.md` § 5.6.

8. **Note (post-fit aggregate() - rows M-025/M-122):** `fit(aggregate=)` is deprecated in 3.9 (removed in 4.0; the no-underscore `"eventstudy"` spelling dies with it) in favor of post-fit `results.aggregate(type)` on the unified vocabulary + `'dose'` as this estimator's documented extra level; the PRE-EXISTING fit-time value validation is retained (unknown strings still raise `ValueError` after the deprecation warning - unlike the EfficientDiD/Imputation shims, which never validated). (a) **MIXED view/recompute architecture** (unique among the aggregate-postfit adopters): the dose-response curves and the overall binarized ATT (ATT^{loc} under PT; equals ATT^{glob} under SPT) plus ACRT^{glob} are ALWAYS computed by `fit()` (`aggregate="dose"` was a fit-time no-op), so `aggregate('simple')` (2 rows, targets att/acrt - the dual-estimand case the `AggregationResult.target` column exists for; `n` = the DISJOINT treated+control unit total, `n_kind='units'`) and `aggregate('dose')` (2N target-discriminated rows; labels = the dose grid twice; no count/mass per row) are pure VIEWS over stored public fields, PERMITTED on bootstrap fits (the library-wide per-level relay rule, since M-027 converged CS/EDiD/Imputation/TwoStage onto it) - they relay stored inference verbatim, including the FINITE `safe_inference` t-stat fit stores beside the percentile p/CI on bootstrapped overall rows and the `DoseResponseCurve.to_dataframe`-exact NaN-t derivation on dose rows; only the df column is uniformly NaN under bootstrap. (b) **`aggregate('event_study')` recomputes** the binarized event study from a pruned per-cell IF payload retained on the fit-built kit: per-(g,t) treated/control positional indices, `delta_y_treated`, `ee_control`, masses and the covariate-path `if_att_glob` (O(n_treated+n_control) per cell), unit-level arrays, and - on survey fits - the PANEL-LEVEL `ResolvedSurveyDesign` ref (the recompute performs the unit collapse itself, keeping the moved body verbatim; on replicate designs the (n_obs x R) replicate matrix rides along - the documented memory cost). The K-dimensional spline machinery (bread, `ee_treated`, `Psi_eval`, `dPsi_*`) is NOT retained; no panel data columns and no raw unit identifiers are retained. Replicate-weight designs ARE supported post-fit (IF-based `compute_replicate_if_variance` - no refit replay). Bootstrap fits carry a SCALARS-ONLY kit and the event-study route fails closed (`NotImplementedError`; the deprecated fit-time `aggregate='eventstudy'` computes the bootstrap surface until 4.0, or re-fit with `n_bootstrap=0`) - a seeded post-fit bootstrap replay is the TODO.md row. (c) **Fit-faithful quirk:** when no post-treatment (g,t) cells exist, event-study rows keep NaN inference on BOTH routes (the fit-time surface never fills them). (d) **df provenance:** the stored `dose_response_att.df_survey` channel (the value every fit-time `safe_inference` received) drives the views' df column (finite-and-positive else NaN - the replicate-undefined 0 sentinel reports NaN in the column but feeds the t/p derivation raw); the post-fit event-study container exposes the scalar `df_survey` channel only (all-NaN per-row df - the M-092 completion hole, tracked in TODO.md). (e) **Rendering:** this is the FIRST heterogeneous-`target` `AggregationResult`; `summary()`/`to_dataframe()` gained the target column / first-appearance target-block ordering amendment (normative rule in `docs/v4-design.md` section 6; uniform-target producers byte-stable). (f) **Consumer admission:** `compute_honest_did`/`compute_pretrends_power` reject ContinuousDiD containers BY DESIGN - no joint event-study covariance exists (per-bin IF SEs only) and the binarized bins carry no reference-period normalization at all (see M-093). Warning stacklevels in the moved bodies remain tuned for the fit-time frame depth, so post-fit-route warnings attribute to a library frame (the shipped EfficientDiD convention). (g) **Report-layer consumption:** `DiagnosticReport` now derives this container internally on plain fits (raw `event_study_effects` absent) for its `heterogeneity` check — the only ES-gated check in ContinuousDiD's applicability; captured recompute warnings are re-published on the section and derivation failures (bootstrap fits) surface as explicit skip context.

### Implementation Checklist

- [x] B-spline basis construction matching R's `splines2::bSpline` (global knots from all treated doses; boundary knots use training-dose range; see deviation note above)
- [x] Multi-period (g,t) cell iteration with base period selection
- [x] Dose-response and event-study aggregation with group-proportional weights (n_treated/n_total per group, divided among post-treatment cells; R `ptetools` convention)
- [x] Multiplier bootstrap for inference
- [x] Analytical SEs via influence functions
- [x] Equation verification tests (linear, quadratic, multi-period)
- [x] Covariate support (reg / dr) — conditional parallel trends. **ipw restricted** (see Note below); survey × covariate deferred; covariates × lowest_dose deferred.
- [x] Discrete treatment saturated regression (`treatment_type="discrete"`) — CGBS 2024 Eq. 4.1; per-level ATT(d_j) + backward-difference ACRT on `{0, d_1, …, d_J}` (`ACRT(d_1) = ATT(d_1)/d_1`; binary → ACRT = ATT). Multi-cohort heterogeneous dose support deferred (see Note #6).
- [x] Lowest-dose-as-control (Remark 3.1, `control_group="lowest_dose"`) — estimand `ATT(d) - ATT(d_L)` for `P(D=0)=0`; discrete + continuous (mass-point) paths; single-cohort. Multi-cohort and covariates × lowest_dose deferred (see Note #7).
- [x] Survey design support (Phase 3): weighted B-spline OLS, TSL on influence functions; bootstrap+survey supported (Phase 6)
- **Note:** ContinuousDiD bootstrap with survey weights supported (Phase 6) via PSU-level multiplier weights
- **Note:** The R-style convention of coding never-treated units as `first_treat=inf` is still accepted and normalized to `first_treat=0` internally, but the estimator now emits a `UserWarning` reporting the row count so the silent recategorization is surfaced (axis-E silent coercion under the Phase 2 audit). Only `+inf` is recoded (matching the R convention). Any **negative** `first_treat` value (including `-inf`) raises `ValueError` with the row count, since such units would otherwise silently fall out of both the treated (`g > 0`) and never-treated (`g == 0`) masks. Pass `0` directly for never-treated units to avoid the warning.
- **Note:** Rows where `first_treat=0` (never-treated) carry a nonzero `dose` are silently zeroed for internal consistency (never-treated cells must have `D=0` in the dose response). The estimator now emits a `UserWarning` with the affected row count before the zeroing, so unintended nonzero doses on never-treated rows are no longer absorbed without a signal (axis-E silent coercion).

---

## EfficientDiD

**Primary source:** [Chen, X., Sant'Anna, P. H. C., & Xie, H. (2025). Efficient Difference-in-Differences and Event Study Estimators. arXiv:2506.17729v1.](https://arxiv.org/abs/2506.17729v1) (Cowles Foundation Discussion Paper No. 2470). Paper review on file: `docs/methodology/papers/chen-santanna-xie-2025-review.md` (theorem/equation numbering pinned to arXiv v1, currently the only version).

**Key implementation requirements:**

*Assumption checks / warnings:*
- **Random Sampling (Assumption S)**: Data is a random sample of `(Y_{1}, ..., Y_{T}, X', G)'`
- **Overlap (Assumption O)**: For each group g, generalized propensity score `E[G_g | X]` must be in `(0, 1)` a.s. Near-zero propensity scores cause ratio `p_g(X)/p_{g'}(X)` to explode; warn on finite-sample instability
- **No-anticipation (Assumption NA)**: For all treated groups g and pre-treatment periods t < g: `E[Y_t(g) | G=g, X] = E[Y_t(infinity) | G=g, X]` a.s.
- **Parallel Trends -- two variants**:
  - **PT-Post** (weaker): PT holds only in post-treatment periods, comparison group = never-treated only, baseline = period g-1 only. Estimator is just-identified and reduces to standard single-baseline DiD (Corollary 3.2)
  - **PT-All** (stronger): PT holds for all groups and all periods. Enables using any not-yet-treated cohort and any pre-treatment period as baseline. Model is overidentified (Lemma 2.1); paper derives optimal combination weights
- **Absorbing treatment**: Binary treatment must be irreversible (once treated, stays treated)
- **Balanced panel**: Short balanced panel required ("large-n, fixed-T" regime). Does not handle unbalanced panels or repeated cross-sections
- Warn if treatment varies within units (non-absorbing treatment)
- Warn if propensity score estimates are near boundary values
- **Note:** Polynomial-sieve propensity fits now reject any K whose normal-equations matrix has condition number above `1/sqrt(eps)` (≈ 6.7e7) — previously a near-singular `np.linalg.solve` could return numerically meaningless coefficients without raising. If at least one K succeeds but others were skipped via this precondition, a `UserWarning` lists the skipped K values. If every K is skipped, the existing "estimation failed for all K values" fallback warning still fires. Axis-A finding #18 in the Phase 2 silent-failures audit.

*Estimator equation -- single treatment date (Equations 3.2, 3.5):*

Transformed outcome (Equation 3.2):
```
Y_tilde_{g,t,t_pre} = (1/pi_g) * (G_g - p_g(X)/p_inf(X) * G_inf) * (Y_t - Y_{t_pre} - m_{inf,t,t_pre}(X))
```

Efficient ATT estimand (Equation 3.5):
```
ATT(g, t) = E[ (1' V*_{gt}(X)^{-1} / (1' V*_{gt}(X)^{-1} 1)) * Y_tilde_{g,t} ]
```

where:
- `G_g = 1{G = g}` = indicator for belonging to treatment cohort g
- `G_inf = 1{G = infinity}` = indicator for never-treated
- `pi_g = P(G = g)` = population share of cohort g
- `p_g(X) = E[G_g | X]` = generalized propensity score
- `m_{inf,t,t_pre}(X) = E[Y_t - Y_{t_pre} | G = infinity, X]` = conditional mean outcome change for never-treated
- `V*_{gt}(X)` = `(g-1) x (g-1)` conditional covariance matrix with `(j,k)`-th element (Equation 3.4):
  ```
  (1/p_g(X)) Cov(Y_t - Y_j, Y_t - Y_k | G=g, X) + (1/(1-p_g(X))) Cov(Y_t - Y_j, Y_t - Y_k | G=inf, X)
  ```

*Estimator equation -- staggered adoption (Equations 3.9, 3.13, 4.3, 4.4):*

Generated outcome for each `(g', t_pre)` pair (Equation 3.9 / sample analog 4.4):
```
Y_hat^{att(g,t)}_{g',t_pre} = (G_g / pi_hat_g) * (Y_t - Y_1 - m_hat_{inf,t,t_pre}(X) - m_hat_{g',t_pre,1}(X))
    - r_hat_{g,inf}(X) * (G_inf / pi_hat_g) * (Y_t - Y_{t_pre} - m_hat_{inf,t,t_pre}(X))
    - r_hat_{g,g'}(X) * (G_{g'} / pi_hat_g) * (Y_{t_pre} - Y_1 - m_hat_{g',t_pre,1}(X))
```

where:
- `r_hat_{g,g'}(X) = p_g(X)/p_{g'}(X)` = estimated propensity score ratio
- `m_hat_{g',t,t_pre}(X) = E[Y_t - Y_{t_pre} | G = g', X]` = estimated conditional mean outcome change

Efficient ATT for staggered adoption (Equation 4.3):
```
ATT_hat_stg(g,t) = E_n[ (1' Omega_hat*_{gt}(X)^{-1}) / (1' Omega_hat*_{gt}(X)^{-1} 1) * Y_hat^{att(g,t)}_stg ]
```

where `Omega*_{gt}(X)` is the conditional covariance matrix with `(j,k)`-th element (Equation 3.12):
```
(1/p_g(X)) Cov(Y_t - Y_1, Y_t - Y_1 | G=g, X)
+ (1/p_inf(X)) Cov(Y_t - Y_{t'_j}, Y_t - Y_{t'_k} | G=inf, X)
- 1{g=g'_j}/p_g(X) * Cov(Y_t - Y_1, Y_{t'_j} - Y_1 | G=g, X)
- 1{g=g'_k}/p_g(X) * Cov(Y_t - Y_1, Y_{t'_k} - Y_1 | G=g, X)
+ 1{g_j=g'_k}/p_{g'_j}(X) * Cov(Y_{t'_j} - Y_1, Y_{t'_k} - Y_1 | G=g'_j, X)
```

*Event study aggregation (Equations 3.8, 3.14, 4.5):*

```
ES_hat(e) = sum_{g in G_{trt,e}}  (pi_hat_g / sum_{g' in G_{trt,e}} pi_hat_{g'})  * ATT_hat_stg(g, g+e)
```

where `G_{trt,e} = {g in G_trt : g + e <= T}` and weights are cohort relative size weights.

Overall average event-study parameter (Equation 2.3):
```
ES_avg = (1/N_E) * sum_{e in E} ES(e)
```

*With covariates / doubly robust:*

The estimator is doubly robust by construction. Consistency requires correct specification of either:
- Outcome regression: `m_{g',t,t_pre}(X) = E[Y_t - Y_{t_pre} | G = g', X]`, OR
- Propensity score ratio: `r_{g,g'}(X) = p_g(X)/p_{g'}(X)`

The Neyman orthogonality property (Remark 4.2) permits modern ML estimators (random forests, lasso, ridge, neural nets, boosted trees) for nuisance parameters without loss of efficiency.

*Without covariates (Section 4.1):*

Estimator simplifies to closed-form expressions using only within-group sample means and sample covariances. **No tuning parameters** are needed. The covariance matrix `Omega*_gt` uses unconditional within-group covariances with `pi_g` replacing `p_g(X)`.

*Standard errors (Theorem 4.1, Section 4):*
- Default: Analytical SE computed as the square root of the sample variance of estimated EIF values divided by n:
  ```
  SE_analytical = sqrt( (1/n^2) * sum_{i=1}^{n} EIF_hat_i^2 )
  ```
- Alternative: Cluster-robust SE at cross-sectional unit level (used in empirical application, page 34-35)
- Bootstrap: Nonparametric clustered bootstrap (resampling clusters with replacement); 300 replications recommended (page 23, footnote 16)
- **Small sample recommendation** (Section 5.1): Use cluster bootstrap SEs rather than analytical SEs when n is small (n <= 50). Analytical SEs are anticonservative with n=50 (coverage ~0.80) but perform well with n >= 200 (coverage ~0.94)
- Simultaneous confidence bands: Multiplier bootstrap procedure for multiple `(g,t)` pairs (footnote 13, referencing Callaway and Sant'Anna 2021, Theorems 2-3, Algorithm 1)
- **Implementation note**: Phase 1 uses multiplier bootstrap on EIF values (Rademacher/Mammen/Webb weights) rather than nonparametric clustered bootstrap. This is asymptotically equivalent and computationally cheaper, consistent with the CallawaySantAnna implementation pattern. Clustered resampling bootstrap may be added in a future version

*Efficient influence function for ATT(g,t) (Theorem 3.2):*
```
EIF^{att(g,t)}_stg = (1' Omega*_{gt}(X)^{-1}) / (1' Omega*_{gt}(X)^{-1} 1) * IF^{att(g,t)}_stg
```

*Efficient influence function for ES(e) (following Theorem 3.2, page 17):*
```
EIF^{es(e)}_stg = sum_{g in G_{trt,e}} ( q_{g,e} * EIF^{att(g,g+e)}_stg
    + ATT(g,g+e) / (sum_{g' in G_{trt,e}} pi_{g'}) * (G_g - pi_g)
    - q_{g,e} * sum_{s in G_{trt,e}} (G_s - pi_s) )
```
where `q_{g,e} = pi_g / sum_{g' in G_{trt,e}} pi_{g'}`.

*Edge cases:*
- **Single pre-treatment period (g=2)**: `V*_{gt}(X)` is 1x1, efficient weights are trivially 1, estimator collapses to standard DiD with single baseline
- **Rank deficiency in `V*_{gt}(X)` or `Omega*_{gt}(X)`**: Inverse does not exist if outcome changes are linearly dependent conditional on covariates. Under the default `omega_ridge > 0` the ridge-regularized solve (see the Omega* ridge Note below) handles this without a routing cliff; at `omega_ridge=0` the legacy behavior applies (detect via matrix condition number; fall back to pseudoinverse with a per-cell warning)
- **Near-zero propensity scores**: Ratio `p_g(X)/p_{g'}(X)` explodes. Overlap assumption (O) rules this out in population; implement trimming or warn on finite-sample instability
- **Note:** When no sieve degree K succeeds for ratio estimation (basis dimension exceeds comparison group size, or all linear systems are singular), the estimator falls back to a constant ratio of 1 for all units with a UserWarning. The outcome regression adjustment remains active, so the generated outcomes (Eq 4.4) still incorporate covariate information via the m_hat terms. The DR property ensures consistency as long as the outcome regression is correctly specified.
- **Note:** When no sieve degree K succeeds for inverse propensity estimation (algorithm step 4), the estimator falls back to unconditional n/n_group scaling with a UserWarning, which reduces to the unconditional Omega* approximation for the affected group.
- **All units eventually treated**: Last cohort serves as "never-treated" by dropping time periods from `last_g - anticipation` onward (where `last_g` is the latest treatment cohort). This excludes anticipation-contaminated periods from the pseudo-control's pre-treatment window. With `anticipation=0` (default), this equals `last_g`. Use `control_group="last_cohort"` to enable; default `"never_treated"` raises ValueError if no never-treated units exist
- **Negative weights**: Explicitly stated as harmless for bias and beneficial for precision; arise from efficiency optimization under overidentification (Section 5.2)
- **PT-Post regime (just-identified)**: Under PT-Post, EDiD automatically reduces to standard single-baseline estimator (Corollary 3.2). No downside to using EDiD -- it subsumes standard estimators
- **Duplicate rows**: Duplicate `(unit, time)` entries are rejected with `ValueError`. The estimator requires exactly one observation per unit-period
- **Note:** PT-All index set includes g'=∞ (never-treated) as a candidate comparison group and excludes period_1 for all g'. When g'=∞, the second and third Eq 3.9 terms telescope so all (∞, t_pre) moments produce the same 2x2 DiD value; these redundant moments are damped by the default Omega* ridge (see the Omega* ridge Note below; at `omega_ridge=0`, by the legacy pseudoinverse). When t_pre = period_1, the third term degenerates to E[Y_1 - Y_1 | G=g'] = 0 for any g', adding no information. Valid pairs require only t_pre < g' (pre-treatment for comparison group), not t_pre < g. Same-group pairs (g'=g) are valid and contribute overidentifying moments (Equation 3.9) — except the degenerate PRE-treatment self-pair (g'=g, t_pre=t), which the default ridge path excludes (see the ridge Note).
- **Note:** Omega* ridge regularization (`omega_ridge`, default 1e-6; v3.7) — a documented refinement of the Omega* inversion in Eq 3.5/3.13/4.3, in the space the paper leaves open (it assumes Omega* is invertible and does not prescribe finite-sample handling of singularity). Under PT-All the overidentified moment set makes the SAMPLE Omega* numerically singular by construction (the telescoping (∞, t_pre) moments above plus near-duplicate cross-cohort moments): measured cond 1e17–1e22 for 100% of units on realistic panels, with a spectrum of one exact-null direction (the degenerate self-pair, below) plus a cluster of statistically-null directions at relative eigenvalue 1e-5–1e-8, far below the ~1e-2 sampling noise of the covariance entries — and no clean spectral gap. The prior pseudoinverse fallback therefore sat on the rcond-cutoff cliff: ANY floating-point-level change (BLAS reordering, platform change, a 1-ulp data perturbation) redrew per-cell weights and moved per-cell ATT(g,t) at up to ~1e-2 relative (measured 1.2e-4 per-cell rel for a 1-ulp outcome perturbation on the pre-v3.7 code; overall ATT stable at ~1e-9 because the redraw averages out across units and cells). The ridge solves `(Omega* + omega_ridge * max(trace/H, 0) * I) x = 1` (trace-scaled: scale-equivariant, O(H), no SVD) — a smooth regularization with sensitivity bounded by 1/omega_ridge, not a cutoff. **Why the deviation is safe:** every moment individually identifies ATT(g,t), so any fixed weights summing to 1 keep the estimator consistent; the ridge trades a numerically ill-defined efficiency optimum for a stable one, and the plug-in EIF treatment of estimated weights (Remark 4.2) is unaffected. **Calibration evidence (2026-07):** default 1e-6 chosen by 1-ulp stability (per-cell rel <= 3e-9 vs 1.2e-4 legacy; candidates 1e-4/1e-6/1e-8 all pass the <=1e-6 target) with the HRS Table 6 anchors as a hard gate (all anchors within the published-value tolerance at every candidate; worst deviation 0.0257 SE, unchanged from legacy; shift <= 0.0001 SE); Monte Carlo on covariate-confounded DGPs shows bias/RMSE/SE-calibration/coverage statistically identical to legacy (ridge marginally better point metrics). **One-time value shift:** per-cell and event-study values on the covariate path move within the pre-existing indeterminacy band when upgrading (worst observed post-treatment cell shift ~0.6 of its own SE at n=500, shrinking with n: overall-ATT shift 1.6e-2 → 4e-3 → 1.1e-3 rel at n=500/1k/2k); the no-covariates path is essentially unchanged (~1e-7). `omega_ridge=0` restores the ENTIRE legacy code path bit-for-bit (both the omega constructions and the inv/pinv weights, including per-cell condition-number warnings and the legacy O(n^2 H^2) runtime). **Degenerate self-pair:** for PRE-treatment cells (t < g), the pair (g'=g, t_pre=t) telescopes to the identically-zero moment 0=0 (the exact-null Omega* direction). The legacy pseudoinverse truncated it, spreading weight over noisy moments (spurious pre-treatment placebos of ~5% of the effect size, pure noise amplification); a naive ridge would instead load all weight on the zero-variance moment, making placebos deterministically zero and silently disabling the pre-trend diagnostic. The default ridge path therefore drops this zero-information pair (fit-level filter; post-treatment cells never contain it), restoring honest data-driven placebos; `omega_ridge=0` keeps the legacy pair set. **Warnings/diagnostics:** the no-covariates path still computes per-cell condition numbers (cheap at (H,H)) for `results.omega_condition_numbers` and consolidates cells with cond > 1e12 into ONE fit-level warning (count + max cond) instead of the legacy per-cell pseudoinverse warnings; the covariate path intentionally computes NO per-unit condition numbers — they would cost exactly the per-unit SVDs the v3.7 rewrite removes, near-singularity there is structural (~always true under PT-All, so a warning would be always-on noise rather than signal), and the ridge handles it by design (the legacy covariate path likewise had no per-unit diagnostics). The scalability warning threshold moved from n > 5000 (legacy per-cell warning) to n > 50,000 (one fit-level warning; the kernel stage is still intrinsically O(n^2) but with a ~100x lower constant and tile-bounded memory). **Cross-cell table hoisting (2026-07 follow-up):** the tiled implementation now builds the kernel-covariance tables once per comparison group per unit-tile instead of per (g, t) cell — every Omega* term is `s_group * KCov(Y_u1 - Y_v1, Y_u2 - Y_v2 | group)`, keyed only by wide-outcome columns, so the per-cell H(H+1)/2-pair tables dedup to distinct product columns per group (~26x fewer kernel GEMM columns on a PT-All fit), and the kernel weight matrices are built and freed one group at a time (the tile memory budget is governed by the largest single group rather than the sum, giving proportionally fatter tiles at large n). Each cell's Omega* is then gathered from the group tables in the same per-entry operation order as the per-cell construction (value-exact, locked by test); this is a pure implementation change — measured results move only at floating-point reassociation level (post-treatment cells ~1e-12 relative, overall ATT ~1e-13; the no-covariates path is byte-identical), and `omega_ridge=0` still routes the entire legacy path. **Rust-backend batched-Cholesky ridge solve (2026-07 follow-up):** on the Rust backend the batched ridge solve `(Omega* + lam * max(trace/H, 0) * I) x = 1` dispatches to a batched Cholesky kernel (the ridged Omega* is SPD by construction; a non-SPD row — measured zero on realistic panels — falls back to LU in-kernel, and any non-finite row is recomputed through the exact legacy numpy chain including its pseudoinverse backstop, so edge-case semantics are unchanged), parallelized over units. This too is a pure implementation change: Cholesky and LU solutions differ only at the condition-bounded floating-point level, and measured results move at reassociation level (post-treatment cells ~2e-12 relative, overall ATT ~1e-13); the pure-Python backend is byte-identical, and `omega_ridge=0` never reaches the kernel.
- **Note:** Bootstrap aggregation uses fixed cohort-size weights for overall/event-study reaggregation, matching the CallawaySantAnna bootstrap pattern (staggered_bootstrap.py:281 computes `bootstrap_overall = bootstrap_atts_gt[:, post_indices] @ weights`; L297 uses the same fixed-weight pattern for event study). The analytical path includes a WIF correction; fixed-weight bootstrap captures the same sampling variability through per-cell EIF perturbation without re-estimating aggregation weights, consistent with both the library's CS implementation and the R `did` package.
- **Overall ATT convention**: The library's `overall_att` uses cohort-size-weighted averaging of post-treatment (g,t) cells, matching the CallawaySantAnna simple aggregation. This differs from the paper's ES_avg (Eq 2.3), which uniformly averages over event-time horizons. ES_avg can be computed from event study output as `mean(event_study_effects[e]["effect"] for e >= 0)`

*Algorithm (two-step semiparametric estimation, Section 4):*

**Step 1: Estimate nuisance parameters**
1. Estimate outcome regressions `m_hat_{g',t,t_pre}(X)` using sieve regression, kernel smoothing, or ML methods (for each valid `(g', t_pre)` pair)
2. Estimate propensity score ratios `r_hat_{g,g'}(X) = p_g(X)/p_{g'}(X)` via convex minimization (Equation 4.1):
   ```
   r_{g,g'}(X) = arg min_{r} E[ r(X)^2 * G_{g'} - 2*r(X)*G_g ]
   ```
   Sieve estimator (Equation 4.2): `beta_hat_K = arg min_{beta_K} E_n[ G_{g'} * (psi^K(X)' beta_K)^2 - 2*G_g * (psi^K(X)' beta_K) ]`
3. Select sieve index K via information criterion: `K_hat = arg min_K { 2*loss(K) + C_n * K / n }` where `C_n = 2` (AIC) or `C_n = log(n)` (BIC)
4. Estimate `s_hat_{g'}(X) = 1/p_{g'}(X)` via analogous convex minimization
5. Estimate conditional covariance `Omega_hat*_{gt}(X)` using kernel smoothing with bandwidth h

**Step 2: Construct efficient estimator**
6. Compute generated outcomes `Y_hat^{att(g,t)}_{g',t_pre}` for each valid `(g', t_pre)` pair using Equation 4.4
7. Compute efficient weights `w(X) = 1' Omega_hat*_{gt}(X)^{-1} / (1' Omega_hat*_{gt}(X)^{-1} 1)`
8. Compute `ATT_hat_stg(g,t) = E_n[ w(X_i) * Y_hat^{att(g,t)}_stg ]` (Equation 4.3)
9. Aggregate to event-study: `ES_hat(e) = sum_g (pi_hat_g / sum pi_hat) * ATT_hat_stg(g, g+e)` (Equation 4.5)
10. Compute SE from sample variance of estimated EIF values

**Without covariates**: Steps 1-5 simplify to within-group sample means and sample covariances. No nuisance estimation or tuning needed.

**Reference implementation(s):**
- No specific software package named in the paper for the EDiD estimator
- Estimators compared against: Callaway-Sant'Anna (`did` R package), de Chaisemartin-D'Haultfoeuille (`DIDmultiplegt` R package / `did_multiplegt` Stata), Borusyak-Jaravel-Spiess / Gardner / Wooldridge imputation estimators
- Empirical replication: HRS data from Dobkin et al. (2018) following Sun and Abraham (2021) sample selection

**Requirements checklist:**
- [x] Implements two-step semiparametric estimator (Equation 4.3)
- [x] Supports both PT-Post (just-identified) and PT-All (overidentified) regimes
- [x] Computes efficient weights from conditional covariance matrix inverse
- [x] Doubly robust: consistent if either outcome regression or propensity score ratio is correct
- [x] No-covariates case uses closed-form sample means/covariances (no tuning)
- [x] With covariates: sieve-based propensity ratio estimation with AIC/BIC selection
- [x] Kernel-smoothed conditional covariance estimation
- [x] Analytical SE from EIF sample variance
- [x] Cluster-robust SE option (analytical from EIF + cluster-level multiplier bootstrap)
- [x] Event-study aggregation ES(e) with cohort-size weights
- [x] Hausman-type pre-test for PT-All vs PT-Post (Theorem A.1)
- [x] Each ATT(g,t) can be estimated independently (parallelizable)
- [x] Absorbing treatment validation
- [x] Overlap diagnostics for propensity score ratios
- [x] Survey design support (Phase 3): survey-weighted means/covariances in Omega*, TSL on EIF scores; bootstrap+survey supported (Phase 6)
- **Note:** Sieve ratio estimation uses polynomial basis functions (total degree up to K) with AIC/BIC model selection. The paper describes sieve estimators generally without specifying a particular basis family; polynomial sieves are a standard choice (Section 4, Eq 4.2). Negative sieve ratio predictions are clipped to a small positive value since the population ratio p_g(X)/p_{g'}(X) is non-negative. The outcome regression m_hat(X) uses the same polynomial sieve basis family (see the outcome-regression Note below).
- **Note:** Kernel-smoothed conditional covariance Omega*(X) uses Gaussian kernel with Silverman's rule-of-thumb bandwidth by default. The paper specifies kernel smoothing (step 5, Section 4) without mandating a particular kernel or bandwidth selection method.
- **Note:** Conditional covariance Omega*(X) scales each term by per-unit sieve-estimated inverse propensities s_hat_{g'}(X) = 1/p_{g'}(X) (algorithm step 4), matching Eq 3.12. The inverse propensity estimation uses the same polynomial sieve convex minimization as the ratio estimator. Estimated s_hat values are clipped to [1, n] with a UserWarning when clipping binds, mirroring the ratio path's overlap diagnostics.
- **Note:** Outcome regressions m_hat_{g',t,tpre}(X) use a polynomial sieve (total degree up to K) with AIC/BIC order selection — the same basis family as the propensity-ratio sieve — matching the paper's flexible nonparametric nuisance specification (Section 4). The sieve order is selected by an OLS information criterion `IC = n*ln(RSS/n) + c_n*p_K`, where `p_K = comb(K+d, d)` is the **sieve basis dimension** (the number of fitted coefficients; `d` = covariate count) and `c_n = 2` (AIC) / `ln(n)` (BIC), on the within-group (survey-weighted) residual sum of squares; the within-group **positive-weight support** count is used for both `n` and the penalty (the raw row count when unweighted) so the selected order — and hence m_hat — is invariant both to the survey-weight scale and to zero-weight (survey-subpopulation / padded) rows. Zero-weight observations are inert in the weighted RSS, weighted Gram, and weighted loss totals; keying order selection (auto-k_max, the `n_basis` admissibility cap, and the IC sample-size terms) off the positive-weight support keeps them inert for selection too — otherwise padding the panel with zero-weight rows could push `floor(n^{1/5})` to a higher candidate degree and silently change the selected K (hence the DR estimate). This applies identically to the two sieve propensity nuisances. Degree 1 reproduces a linear working model up to floating point, so AIC/BIC degrades to linear when the conditional mean is linear and grows the order only when the data support it. There is **no fixed order ceiling**: the candidate maximum degree grows as `floor(n_pos^{1/5})` with the (positive-weight) group support `n_pos`, giving a sieve dimension `p_K = comb(K+d, d)` bounded by `n_basis = p_K < n_pos` — a *growing* sieve. Assumption C.1's regularity conditions are stated in terms of the sieve **dimension** (not the polynomial degree, which differ once `d > 1`): uniform consistency (C.1(5), `||m_hat - m||_inf = o_p(1)`) and the `o_p(n^{-1/2})` product rate (C.1(6)) require `p_K -> ∞` with `p_K = o(n)`. The growing sieve satisfies this for the low-dimensional covariate settings typical of DiD — with the `floor(n^{1/5})` degree rule, `p_K = comb(floor(n^{1/5})+d, d) = o(n)` for small `d`; for high-dimensional `X` a polynomial sieve faces the curse of dimensionality (`p_K` can outpace `o(n)`), where the paper's ML-nuisance option (Remark 4.2) is preferable. Under C.1 the doubly robust covariate path attains the semiparametric efficiency bound asymptotically (Theorem 4.1); a frozen finite-order sieve (fixed `p_K`) would violate C.1(5)/(6). The DR property still ensures consistency, regardless of `d`, if either the outcome regression or the propensity ratio is correctly specified.
- **Note:** If every sieve degree is rank-skipped for an outcome regression (a comparison group too small for even the linear basis, or a degenerate/constant covariate), the estimator falls back to the intercept-only within-group mean of `Y_t - Y_{tpre}` (the unconditional outcome regression) with a UserWarning — distinct from the propensity-ratio sieve's constant-ratio-1 fallback.
- **Note:** EfficientDiD bootstrap with survey weights supported (Phase 6) via PSU-level multiplier weights
- **Note:** EfficientDiD covariates (DR path) with survey weights supported — WLS outcome regression, weighted sieve normal equations for propensity ratios/inverse propensities, survey-weighted Nadaraya-Watson kernel for conditional Omega*(X), and survey-weighted ATT averaging. The auto Silverman bandwidth for the conditional Omega*(X) kernel is evaluated on the **positive-weight support** (rows with `w > 0`) with a **survey-weighted dispersion**: `median_std` is the median across covariate dimensions of the weighted standard deviation `sqrt(sum_i w_i (x_i - xbar_w)^2 / sum_i w_i)` (weighted mean `xbar_w = sum_i w_i x_i / sum_i w_i`), so the bandwidth reflects the population covariate distribution the kernel targets rather than the unweighted sample. The rate term `n` stays the positive-weight support count (the dispersion is weighted; the sample-size term is not — a deliberately scoped refinement, not Kish `n_eff`). Because zero-weight rows drop from the support and contribute nothing to the weighted moments or the count, and because the weighted mean/std and the count are invariant to weight rescaling `w -> c*w`, the bandwidth — and hence Omega*(X) and the per-unit efficient weights it feeds in overidentified (H>1) cells — is invariant to both zero-weight (survey-subpopulation / padded) rows and weight scale; under uniform positive weights the weighted std reduces to the unweighted population std, matching the pre-refinement bandwidth up to floating point. Together with the positive-weight-support sieve order selection, this keeps the DR point estimate exactly invariant to zero-weight padding (a zero-weight row with an extreme covariate would otherwise move the dispersion and the bandwidth).
- **Note:** Cluster-robust SEs use the standard Liang-Zeger clustered sandwich estimator applied to EIF values: aggregate EIF within clusters, center, and compute variance with G/(G-1) small-sample correction. Cluster bootstrap generates multiplier weights at the cluster level (all units in a cluster share the same weight). Analytical clustered SEs are the default when `cluster` is set; cluster bootstrap is opt-in via `n_bootstrap > 0`.
- **Note:** Hausman pretest operates on the post-treatment event-study vector ES(e) per Theorem A.1. Both PT-All and PT-Post fits are aggregated to ES(e) using cohort-size weights before computing the test statistic H = delta' V^{-1} delta where delta = ES_post - ES_all and V = Cov(ES_post) - Cov(ES_all). Covariance is computed from aggregated ES(e)-level EIF values. The variance-difference matrix V is inverted via Moore-Penrose pseudoinverse to handle finite-sample non-positive-definiteness. Effective rank of V (number of positive eigenvalues) is used as degrees of freedom.
- **Note:** Last-cohort-as-control (`control_group="last_cohort"`) reclassifies the latest treatment cohort as pseudo-never-treated and drops time periods at `t >= last_g - anticipation`, excluding anticipation-contaminated periods from the pseudo-control's pre-treatment window. This is distinct from CallawaySantAnna's `not_yet_treated` option which dynamically selects not-yet-treated units per (g,t) pair.
- **Note:** `vcov_type` is permanently narrow to `{"hc1"}` per the Chen-Sant'Anna-Xie (2025) EIF-based variance achieving the semiparametric efficiency bound. Analytical-sandwich families `{classical, hc2, hc2_bm}` are rejected at `__init__` — the per-unit EIF aggregation has no equivalent single design matrix on which hat-matrix leverage or Bell-McCaffrey Satterthwaite DOF can be defined. `cluster=` invokes Liang-Zeger CR1 on cluster-aggregated EIF (`_compute_se_from_eif` with `cluster_indices`); `survey_design=` invokes TSL on the combined IF (`_compute_survey_eif_se`); both live in `diff_diff/efficient_did_aggregation.py` since the M-023 post-fit aggregate() extraction. `vcov_type='conley'` deferred to the EfficientDiD Conley follow-up row in DEFERRED.md.
- **Note:** Default `cluster=None` (no survey design) renders summary label "HC1 heteroskedasticity-robust" because the per-unit EIF SE `sqrt(mean(EIF²)/n)` is methodologically HC1-style (no Liang-Zeger G/(G-1) finite-sample correction). `EfficientDiDResults.cluster_name` and `n_clusters` stay None under unclustered fits. This diverges from `ImputationDiD` which auto-clusters at unit per Borusyak-Jaravel-Spiess (2024) Theorem 3 — there the default summary renders the CR1 unit-clustered label.
- **Note:** `set_params(vcov_type=bad)` raises immediately on EVERY estimator: since the shared `BaseEstimator` mixin (`diff_diff/_base.py`, v4 2(c)-i), `set_params` validates transactionally by constructor probe re-init, so it enforces exactly `__init__`'s validation, eagerly, library-wide. The former split — EfficientDiD eager vs `ImputationDiD`/`TripleDifference`/`CallawaySantAnna` (and six more: SunAbraham, StackedDiD, StaggeredTripleDifference, SpilloverDiD, TROP, PreTrendsPower) accepting constructor-rejected values until `fit()` — is retired; the fit-time re-validation layers remain as a second check against DIRECT attribute mutation (`est.vcov_type = ...`), which no setter can see.
- **Note (post-fit aggregate() - rows M-023/M-120):** `fit(aggregate=, balance_e=)` is deprecated (3.9; removed 4.0; joint FutureWarning, warn-and-still-work) in favor of post-fit `EfficientDiDResults.aggregate(type, balance_e=)` - a LAZY RECOMPUTING KIT (the CallawaySantAnna class, not a StackedDiD/dCDH view relay): `fit()` computes nothing extra, the results object retains an `AggregationKit`, and `aggregate('event_study'/'group', balance_e=)` re-runs the extracted `_EfficientAggregationMixin` aggregators on a throwaway host while `aggregate('simple')` relays the stored overall row bit-exact. (a) RETAINED BUFFERS (memory contract; phrased as maxima - optional design fields stay None when unsupplied): the per-(g,t) EIF dict, O(n_units x n_gt), the dominant payload - retained on EVERY fit regardless of `store_eif`, which since 3.9 governs only the public `influence_functions` field; `unit_cohorts` (cohort labels), `unit_level_weights`, factorized cluster codes - O(n_units) each; on ordinary (TSL) survey fits the unit-level `ResolvedSurveyDesign` adds `weights` plus, where supplied, `strata`/`psu`/`fpc` (factorized int codes / float values, never raw labels) - up to four O(n_units) arrays; on replicate designs it adds the O(n_units x n_replicates) replicate matrix plus, where supplied, `replicate_strata`/`replicate_rscales` (O(n_replicates)); per-row dict SNAPSHOTS of `group_time_effects` plus copies of the `groups`/`time_periods` lists and the scalar `pt_assumption`/`n_treated+n_control` provenance (aggregate() recomputes exclusively from these private snapshots, never from the mutable public result fields - a user edit of the public rows cannot mix altered point estimates with the retained EIF variance); scalars `n_units`, `cohort_fractions`, and the POST-OVERALL `df_survey` snapshot (captured after the overall inference and before the ES/group gates: the group pass can degenerate the working df to None on replicate designs with `n_valid <= 1`, and every fit-time aggregation seeds from the post-overall value, so recompute replays the exact seed). The data-minimization guarantee is scoped to unit identifiers - no unit-label container is retained. (b) `balance_e` uses the ANCHOR-HORIZON rule (keep cohorts with a finite effect at `e == balance_e`, then retain all their horizons) - the SAME rule CallawaySantAnna uses, divergent only from ImputationDiD/TwoStageDiD's balanced-window rule; an anchor no cohort reaches warns and yields a legal zero-row container. (c) BOOTSTRAP fits: 'simple' RELAYS the stored overall row verbatim (percentile se/p/CI beside the finite safe_inference t) with a NaN df column, while the RECOMPUTE levels (ES/group) fail closed - per-horizon draws are not retained (exact-replay wiring is a TODO row). The prior uniform-conservatism BY-DECISION rule was superseded 2026-08-05 with the M-027 per-level convergence; its rationale - no level publishes analytical-provenance fields beside percentile inference - is honored by the relay's NaN df column. The fit-time bootstrap override still clears the group rows' `df_used` key. (d) CONTAINER ADMISSION into `compute_honest_did`/`compute_pretrends_power` is REJECTED BY DESIGN (both terminal TypeErrors state it): the primary ground is the absent joint event-study covariance (container `vcov=None`, all-NaN per-row df - the scalar `df_survey` channel is the container's only df provenance; the per-row hole is the tracked M-092-completion TODO row). Reference semantics are regime-dependent: under `pt_assumption="all"` there is NO reference row (universal first-period baseline; e=-1 is a genuine estimate); under `"post"` the per-cohort baseline cell is materialized as a mechanical zero anchor at `e = -1 - anticipation` whenever it is not the panel's first period, and the MEMBERSHIP-GATED `reference_period` property (the SunAbraham rule - never synthesized when the anchor cell was not estimated) marks it `is_reference` in the container and corrects `plot_event_study`'s inferred reference (previously the `-1` fallback) on PT-Post `anticipation>0` fits. (e) 'simple' relay conventions: `target="att"`, `n = n_treated_units + n_control_units` with `n_kind="units"` (DISJOINT by construction - `last_cohort` trimming reassigns before the counts, so a true total exists, unlike StackedDiD's overlapping sets), `df` = the post-overall snapshot (provenance-exact where `survey_metadata.df_survey` can diverge in the degenerate replicate state); 'group' relay: `n_kind="cells"`, `weight=None` (equal within-cohort weights, no cross-cohort mass), per-row `df_used` array captured at each row's `safe_inference` call (exact by construction; a stated divergence from CS's conservative-min scalar broadcast); 'event_study' rides the shared `_from_relative_dict` builder via a carrier whose `survey_metadata` copy carries the snapshot `df_survey`.

---

## SunAbraham

**Primary source:** [Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects in event studies with heterogeneous treatment effects. *Journal of Econometrics*, 225(2), 175-199.](https://doi.org/10.1016/j.jeconom.2020.09.006)

**Key implementation requirements:**

*Assumption checks / warnings:*
- Requires never-treated units as control group
- Warns if treatment effects may be heterogeneous across cohorts (which the method handles)
- Reference period: e=-1-anticipation (defaults to e=-1 when anticipation=0)

*Estimator equation (as implemented):*

Saturated regression with cohort-specific effects:
```
Y_it = α_i + γ_t + Σ_{g∈G} Σ_{e≠-1} δ_{g,e} × 1(G_i=g) × D^e_{it} + ε_it
```
where G_i is unit i's cohort (first treatment period), D^e_{it} = 1(t - G_i = e).

Interaction-weighted estimator:
```
δ̂_e = Σ_g ŵ_{g,e} × δ̂_{g,e}
```
where weights ŵ_{g,e} = n_{g,e} / Σ_g n_{g,e} (sample share of cohort g at event-time e).

*Standard errors:*
- Default: Cluster-robust HC1 at unit level (`vcov_type="hc1"`)
- `vcov_type ∈ {"classical","hc1","hc2","hc2_bm"}` supported as of Phase 1b
  PR 1/8 (mirrors the DiD/MPD/TWFE chain established in Phase 1a):
  - `"hc1"` (default): Eicker-Huber-White HC1 with cluster-at-unit default.
    Auto-clusters at unit unless an explicit `cluster=` is passed.
  - `"classical"`: homoskedastic OLS standard errors. Auto-cluster is
    dropped (one-way only). Routes through the full-dummy saturated
    design (see Implementation note below) for R-parity.
  - `"hc2"`: HC2 leverage correction. Auto-cluster is dropped (one-way
    only); the linalg validator rejects `hc2 + cluster_ids`. Routes
    through full-dummy.
  - `"hc2_bm"`: HC2 + Bell-McCaffrey CR2 Satterthwaite DOF for
    cluster-robust inference. Auto-cluster fires at unit (or explicit
    `cluster=`); routes through full-dummy. R-parity matches
    `clubSandwich::vcovCR(..., type="CR2")` + `coef_test()$df_Satt` at
    atol=1e-10.
  - `"conley"` (spatial-HAC, Conley 1999): threaded through the
    within-transform saturated regression via `solve_ols` / `conley.py` —
    within-period-spatial-only (`conley_lag_cutoff=0`) and panel block-decomposed
    (`conley_lag_cutoff>0`: within-period spatial + within-unit Bartlett
    serial). Reuses the already-`conleyreg`-validated machinery (no new
    variance code). The unit auto-cluster is dropped on the conley path (an
    explicit `cluster=` enables the spatial+cluster product kernel);
    `survey_design=` / `weights` / `n_bootstrap>0` are rejected. **Note:**
    the FWL-demeaned conley sandwich equals the full-dummy conley SE (pinned
    in `tests/test_conley_vcov.py::TestConleySunAbraham::test_fwl_composability_vs_full_dummy`).
- **Note (Phase 1b auto-route):** When `vcov_type ∈ {"classical","hc2",
  "hc2_bm"}`, `_fit_saturated_regression` bypasses the within-transform
  path and builds the full-dummy saturated design `[intercept +
  cohort × event-time interactions + covariates + unit_dummies +
  time_dummies]` directly. The FWL theorem preserves cohort coefficients
  and residuals but does NOT preserve the hat matrix, so HC2 leverage
  and Bell-McCaffrey Satterthwaite DOF must be computed on the full FE
  projection (matches `lm() + sandwich::vcovHC` / `clubSandwich::vcovCR`).
  Classical SE also routes through full-dummy so the `(n-k)` finite-
  sample correction matches R's `lm()` interpretation at atol=1e-10.
  `hc1` stays on the within-transform path (cluster-robust HC1 doesn't
  depend on the hat matrix); matches `fixest::sunab()` event-study
  aggregates closely (see deviation note below).
- **Note (Phase 1b aggregated BM contrast DOF):** Under
  `vcov_type="hc2_bm"`, the user-facing aggregated inference
  (`event_study_effects[e]['p_value']`/`['conf_int']`,
  `overall_p_value`/`overall_conf_int`) uses CR2 Bell-McCaffrey
  Satterthwaite DOF per contrast — not the normal distribution.
  Per-event-time contrast `c_e[full_idx(g,e)] = w_{g,e}` (IW weight)
  and overall ATT contrast `c_overall[full_idx(g,e)] = w_e × w_{g,e}`
  are passed to `_compute_cr2_bm_contrast_dof` (the helper PR #465
  added for MultiPeriodDiD's post-period-average DOF). The resulting
  per-contrast DOF threads into `safe_inference(..., df=<contrast_dof>)`.
  Matches `clubSandwich::Wald_test(constraints=matrix(c, 1),
  test="HTZ")$df_denom` at atol=1e-10 (pinned in
  `tests/test_methodology_sun_abraham.py`). Cohort-level coefficients
  separately get per-coefficient BM DOF via
  `LinearRegression.get_inference()` inside `_fit_saturated_regression`.
  If the linalg helper fails (rank-deficient design, singular bread),
  the aggregated inference falls back to the `df_convention`-resolved
  analytical df (t(residual) under the default) with an explicit
  `UserWarning`.
- **Note:** Provenance exposure (4.0 program row M-092; df values updated by
  the 3.9 M-127 tail-df fix): `SunAbrahamResults.event_study_df` records,
  per relative time, the df each stored event-study row's `safe_inference`
  actually received - the per-event BM contrast DOF above under `hc2_bm`,
  the survey design df (post-drop under a replicate refit) on survey fits,
  the `df_convention`-resolved fallback on plain analytic fits (FINITE
  residual df under the 3.9 default; `G − 1` under `"cluster"`; NaN under
  `"normal"`), and NaN on rows whose BM DOF was non-finite, where
  `safe_inference`'s own non-finite-df guard already yields all-NaN
  inference. Note the two failure modes differ and are both faithfully
  recorded: a helper-level failure warns and falls back to the resolved
  analytical df (documented above), while a per-row non-finite DOF fails
  closed. The channel clears under bootstrap (percentile p/CIs used no df) -
  deliberately narrower than the `event_study_vcov` clear, which also fires
  under replicate refits whose rows DID use a genuine df. The scalar
  `inference_df` (added 3.9) mirrors the same rule for the overall ATT.
- **HC1 finite-sample correction (K_reference convergence, 3.9):** SA's
  clustered CR1 factor counts the absorbed unit + time FE via the
  `cluster_k_adjustment` increment (`K_reference = k_dm + 1 + rank(non-nested
  FE | nested)`), matching `fixest::sunab(cluster=~unit)` — measured gap
  ~5e-15 on the R-parity fixture (`tests/test_estimators_vcov_type.py`, the
  sunab gate). Historically `solve_ols` counted only the within-transformed
  design columns (`n / (n - k_dm)`), sitting ~1-2% below fixest on typical
  panel sizes; that follow-up (DEFERRED row 53) is closed. The IW
  aggregation step is unchanged.
- The within-transform residual df subtracts the absorbed unit + time FE via
  the component-aware rank (`diff_diff.utils.absorbed_fe_rank`, no-intercept
  form — the saturated design carries no intercept column) — see the
  TwoWayFixedEffects section's absorbed-FE degrees-of-freedom note for the
  shared convention. On disconnected panels the absorbed count falls by C-1,
  raising the per-cell residual df used for cohort-period t-inference.
- Survey designs (`survey_design=`) + `vcov_type ∈ {"classical","hc2",
  "hc2_bm"}` are rejected at fit-time: the survey-design Taylor Series
  Linearization (or replicate-weight refit) variance overrides the
  analytical sandwich family, so the requested HC2/HC2-BM/classical
  family would be silently discarded. Additionally, the auto-cluster
  guard for one-way families (classical/hc2) would drop the unit
  auto-cluster before survey-PSU injection, downgrading the panel
  structure from unit-level to per-observation PSUs. Mirrors the
  TWFE Gate 1 + replicate-weight gate from PR #469 and the
  `linalg.py::_validate_vcov_args` `hc2_bm + weights` gate. Use
  `vcov_type="hc1"` (default) for survey designs; the survey TSL
  machinery computes the design-aware SE on the within-transform path.
- Delta method for aggregated coefficients
- Optional: Pairs bootstrap for robustness

*Edge cases:*
- Single cohort: reduces to standard event study
- Cohorts with no observations at some event-times: weighted appropriately
- Extrapolation beyond observed event-times: not estimated
- Event-time range: no artificial cap (estimates all available relative times, matching R's `fixest::sunab()`)
- No post-treatment effects: returns `(NaN, NaN)` for overall ATT/SE; all inference fields (t_stat, p_value, conf_int) propagate NaN via `np.isfinite()` guards
- `min_pre_periods`/`min_post_periods` parameters: removed (previously deprecated with `FutureWarning`; callers passing these will now get `TypeError`)
- Variance fallback: when full weight vector cannot be constructed for overall ATT SE, uses simplified variance (ignores covariances between periods) with `UserWarning`
- Rank-deficient design matrix (covariate collinearity):
  - Detection: Pivoted QR decomposition with tolerance `1e-07` (R's `qr()` default), with a **column-equilibration re-check** (unit 2-norm) that makes the rank *count* invariant to per-column scaling; the dropped-column selection is unchanged for well-scaled collinear designs (a scale-induced under-count instead adopts the scale-corrected equilibrated selection)
  - Handling: Warns and drops linearly dependent columns, sets NA for dropped coefficients (R-style, matches `lm()`)
  - Parameter: `rank_deficient_action` controls behavior: "warn" (default), "error", or "silent"
- NaN inference for undefined statistics:
  - t_stat: Uses NaN (not 0.0) when SE is non-finite or zero
  - Analytical inference: p_value and CI also NaN when t_stat is NaN (NaN propagates through `compute_p_value` and `compute_confidence_interval`)
  - Bootstrap inference: p_value and CI computed from bootstrap distribution. SE, CI, and p-value are all NaN if the original point estimate is non-finite, SE is non-finite or zero, or if <50% of bootstrap samples are valid
  - Applies to overall ATT, per-effect event study, and aggregated event study
  - **Note**: Defensive enhancement matching CallawaySantAnna behavior; R's `fixest::sunab()` may produce Inf/NaN without warning
- Inference distribution (ONE df source per fit since 3.9 / M-127 — the D4
  fix; previously cells used t(residual) while aggregates dropped to normal
  theory inside the same fit):
  - Cohort-level p-values: t-distribution via
    `LinearRegression.get_inference()`, following the `df_convention` knob
    (residual df by default; `G − 1` under `"cluster"`; z under `"normal"`).
  - Aggregated event study and overall ATT p-values:
    - Under `vcov_type="hc2_bm"`: t-distribution with CR2 Bell-McCaffrey
      contrast DOF per aggregated effect (see "Phase 1b aggregated BM
      contrast DOF" Note above). Matches `clubSandwich::Wald_test(
      test="HTZ")$df_denom`.
    - Under `vcov_type ∈ {"classical","hc1","hc2","conley"}` (no survey):
      the SAME `df_convention`-resolved df as the cells — t(residual df of
      the saturated regression) by default, t(G − 1) under `"cluster"`
      (inert under conley), z under `"normal"`.
    - Under replicate-weight survey: t-distribution with replicate-derived
      DOF (`survey_metadata.df_survey`).
  - **Deviation from R**: R's fixest uses t-distribution at all levels with
    the CLUSTER df `G − 1` on clustered fits; the 3.9 default uses the
    residual df instead (the library-wide `df_convention` deviation
    documented in the TwoWayFixedEffects section — `df_convention=
    "cluster"` opts into the fixest convention today, and the default flips
    at v4). The `hc2_bm` aggregated path matches clubSandwich exactly.

**Reference implementation(s):**
- R: `fixest::sunab()` (Laurent Bergé's implementation)
- Stata: `eventstudyinteract`

**Requirements checklist:**
- [x] Never-treated units required as controls
- [x] Interaction weights sum to 1 within each relative time period
- [x] Reference period defaults to e=-1, coefficient normalized to zero
- [x] Cohort-specific effects recoverable from results
- [x] Cluster-robust SEs with delta method for aggregates
- [x] R comparison: ATT matches within machine precision (<1e-11)
- [x] R comparison: SE matches within 0.3% (well within 1% threshold)
- [x] R comparison: Event study effects match perfectly (correlation 1.0)
- [x] Survey design support (Phase 3): weighted within-transform, survey weights in LinearRegression with TSL vcov; bootstrap+survey supported (Phase 6) via Rao-Wu rescaled bootstrap. Replicate weights supported via estimator-level refit (see Replicate Weight Variance section); replicate+bootstrap rejected.

---

## ImputationDiD

**Primary source:** [Borusyak, K., Jaravel, X., & Spiess, J. (2024). Revisiting Event-Study Designs: Robust and Efficient Estimation. *Review of Economic Studies*, 91(6), 3253-3285.](https://doi.org/10.1093/restud/rdae007)

**Key implementation requirements:**

*Assumption checks / warnings:*
- **Parallel trends (Assumption 1):** `E[Y_it(0)] = alpha_i + beta_t` for all observations. General form allows `E[Y_it(0)] = alpha_i + beta_t + X'_it * delta` with time-varying covariates.
- **No-anticipation effects (Assumption 2):** `Y_it = Y_it(0)` for all untreated observations. Adjustable via `anticipation` parameter.
- Treatment must be absorbing: `D_it` switches from 0 to 1 and stays at 1.
- Covariate space of treated observations must be spanned by untreated observations (rank condition). For unit/period FE case: every treated unit must have ≥1 untreated period; every post-treatment period must have ≥1 untreated unit.
- Without never-treated units, long-run effects at horizon `K_it >= H_bar` (where `H_bar = max(first_treat) - min(first_treat)`) are not identified (Proposition 5). Set to NaN with warning.

*Estimator equation (Theorem 2, as implemented):*

```
Step 1. Estimate counterfactual model on untreated observations only (it in Omega_0):
    Y_it = alpha_i + beta_t [+ X'_it * delta] + epsilon_it

Step 2. For each treated observation (it in Omega_1), impute:
    Y_hat_it(0) = alpha_hat_i + beta_hat_t [+ X'_it * delta_hat]
    tau_hat_it  = Y_it - Y_hat_it(0)

Step 3. Aggregate:
    tau_hat_w = sum_{it in Omega_1} w_it * tau_hat_it
```

where:
- `Omega_0 = {it : D_it = 0}` — all untreated observations (never-treated + not-yet-treated)
- `Omega_1 = {it : D_it = 1}` — all treated observations
- `w_it` = pre-specified weights (overall ATT: `w_it = 1/N_1`)

*Common estimation targets (weighting schemes):*
- Overall ATT: `w_it = 1/N_1` for all `it in Omega_1`
- Horizon-specific: `w_it = 1[K_it = h] / |Omega_{1,h}|` for `K_it = t - E_i`
- Group-specific: `w_it = 1[G_i = g] / |Omega_{1,g}|`

*Standard errors (Theorem 3, Equation 7):*

Conservative clustered variance estimator:
```
sigma_hat^2_w = sum_i ( sum_{t: it in Omega} v_it * epsilon_tilde_it )^2
```

Observation weights `v_it`:
- For treated `(i,t) in Omega_1`: `v_it = w_it` (the aggregation weight)
- For untreated `(i,t) in Omega_0` (FE-only **and** covariate cases): the exact imputation projection `v_untreated = -A_0 (A_0' A_0)^{-1} A_1' w_treated` (survey-weighted, with the left WLS weight factor `W_0`: `-W_0 A_0 (A_0' W_0 A_0)^{-1} A_1' w_treated`), where `A_0`, `A_1` are the two-way-FE (all unit dummies + time dummies dropping the first; plus any covariates) design matrices for untreated/treated observations.

**Note (pretrends lead FE-span guard, 2026-07):** the Test-1 lead model (`_compute_lead_coefficients`) routes its lead indicators + covariates through `snap_absorbed_regressors` after the within-transform (same two-stage snap + LSMR confirmation as the `absorb=` estimators; see "Absorbed Fixed Effects"). Lead indicators are the most plausible FE-spanned regressors: when a lead's calendar period contains only that cohort's rows on Omega_0 (e.g. never-treated units unobserved there), the lead collapses to a calendar-time dummy in the span of the absorbed time FE — it now snaps to exact zero and reports a deterministic NaN coefficient with a cause-specific warning (label `lead[h]`), instead of relying on the raw rank check alone (which the truncated-MAP-iterate exposure can defeat in slow-convergence regimes). Identified leads are unchanged (snap is a no-op; suite bit-stable).

**Note on v_it derivation:** The paper's Supplementary Proposition A3 gives the explicit `v_it^*` formula; it is not in the reviewed main-article PDF, so the projection is validated *empirically* against R `didimputation` (`tests/test_methodology_imputation.py::TestImputationDiDParityR`, SEs match to ~1e-10; the covariate branch — first stage `y ~ x | unit + time` on the untreated sample — is anchored separately by `TestImputationDiDCovariateParityR` on a time-varying-X panel, SEs ~2e-10). **Deviation note (superseded closed form):** the FE-only path previously used a closed form `-(w_i./n_{0,i} + w_.t/n_{0,t} - w../N_0)`, which is exact only for a *balanced* untreated set; because `Omega_0` is generically unbalanced in staggered designs (treated observations are removed), that form biased the SE (~27% on the parity panel) and was replaced by the exact projection above during the ImputationDiD methodology validation. A genuinely rank-deficient `A_0' A_0` (e.g. an unidentified period FE) routes to a sparse LSMR least-squares fallback with a `UserWarning` (no dense materialization; see the sparse-variance-solver Note below).

Auxiliary model residuals (Equation 8):
- Partition `Omega_1` into groups `G_g` (default: cohort × horizon)
- Compute `tau_tilde_g` for each group (weighted average within group)
- `epsilon_tilde_it = Y_it - alpha_hat_i - beta_hat_t [- X'delta_hat] - tau_tilde_g` (treated)
- `epsilon_tilde_it = Y_it - alpha_hat_i - beta_hat_t [- X'delta_hat]` (untreated, i.e., Step 1 residuals)

The `aux_partition` parameter controls the partition: `"cohort_horizon"` (default, tightest SEs), `"cohort"` (coarser, more conservative), `"horizon"` (groups by relative time only).

*Pre-trend test (Test 1, Equation 9):*
```
Y_it = alpha_i + beta_t [+ X'_it * delta] + W'_it * gamma + epsilon_it
```
- Estimate on untreated observations only
- Test `gamma = 0` via cluster-robust Wald F-test
- Independent of treatment effect estimation (Proposition 9)

*Pre-period event study coefficients (`pretrends=True`, Test 1 / Equation 9):*

Pre-period coefficients reuse the existing pre-trend test machinery (BJS Equation 9):
```
Y_it = alpha_i + beta_t [+ X'_it * delta] + sum_h gamma_h * W_it(h) + epsilon_it
```
where `W_it(h) = 1[K_it = h]` are lead indicators, estimated on `Omega_0` only.
- `gamma_h` are the pre-period event study coefficients (cluster-robust SEs by default; design-based survey VCV when analytical `survey_design` is present)
- Under parallel trends (Assumption 1), `gamma_h = 0` for all `h < -anticipation`
- Reference period `h = -1 - anticipation` is the omitted category (normalized to zero)
- SEs from cluster-robust Wald variance by default; design-based when survey present (consistent with `pretrend_test()`)
- Bootstrap does not update pre-period SEs (they are from the lead regression)
- When `balance_e` is set, lead indicators are restricted to balanced cohorts; the full Omega_0 sample (including never-treated) is kept for within-transformation
- Only affects event study aggregation; overall ATT and group aggregation unchanged
- **Note:** `pretrends=True` with analytical `survey_design` (strata/PSU/FPC) is supported. The lead regression uses survey-weighted demeaning, WLS point estimates, and `compute_survey_vcov()` for design-based VCV. The full survey design is preserved (subpopulation approach): Omega_0 scores are zero-padded back to full-panel length so PSU/strata structure is maintained for variance estimation. The F-test in `pretrend_test()` uses the full-design `df_survey` as denominator df. Replicate-weight survey designs raise `NotImplementedError` with `pretrends=True` because per-replicate lead regression refits are not yet implemented.
- **Note (pretrends lead-regression tail df, 3.9 / M-127):** the per-lead
  `gamma_h` t/p/CI previously used silent normal-theory z on plain clustered
  fits (the tail-df defect family); they now follow the `df_convention`
  knob — t(residual df `n − k_kept − rank(absorbed [time, unit])` on the
  Omega_0 lead design) under the default `"residual"`, t(G − 1) under
  `"cluster"`, z under `"normal"` — with the full-design survey df keeping
  precedence. This is the ONE ImputationDiD surface on the shared clustered
  CR1 sandwich; the BJS Theorem-3 overall/post-treatment inference is
  knob-independent (L3), and the JOINT pretrend Wald F deliberately keeps
  its cluster-robust `F(q, max(G − 1, 1))` denominator (the standard
  joint-test convention; it consumes gamma/V_gamma, not the per-lead t's,
  coheres with `df_convention="cluster"`, and converges with the t's at the
  v4 flip) — the per-lead-t vs joint-F pairing is a stated convention, not
  a silent split. Since the M-021 post-fit migration the
  inert-config warning predicate is REACHABILITY-based, no longer
  aggregate-keyed: an explicitly non-default value warns iff the per-lead
  inference is unreachable for the fit config — reachable iff
  `pretrends=True` AND not a replicate-weight design AND (the deprecated
  fit-time `aggregate='event_study'/'all'` was supplied OR
  `n_bootstrap <= 0`, since post-fit `results.aggregate('event_study')`
  reaches the leads on any analytical pretrends fit but fails closed on
  bootstrapped ones). Reachability-BASED, not exact: a fit whose bootstrap
  later FAILS (`bootstrap_results=None`) can still aggregate post-fit, so
  that corner warns spuriously — the warning fires before the bootstrap
  runs and cannot know.

*Edge cases:*
- **Unbalanced panels:** FE estimated via iterative alternating projection (Gauss-Seidel), equivalent to OLS with unit+time dummies. Converges in O(max_iter) passes; typically 5-20 iterations for unbalanced panels, 1-2 for balanced. One-pass demeaning is only exact for balanced panels.
- **No never-treated units (Proposition 5):** Long-run effects at horizons `h >= H_bar` are not identified. Set to NaN with warning listing affected horizons.
- **Rank condition failure:** Every treated unit must have ≥1 untreated period; every post-treatment period must have ≥1 untreated unit. Behavior controlled by `rank_deficient_action`: "warn" (default), "error", or "silent". Missing FE produce NaN treatment effects for affected observations.
- **Always-treated units:** Units with `first_treat` at or before the earliest time period have no untreated observations. Warning emitted; these units are excluded from Step 1 OLS but their treated observations contribute to aggregation if imputation is possible.
- **NaN propagation:** If all `tau_hat` values for a given horizon or group are NaN, the aggregated effect and all inference fields (SE, t-stat, p-value, CI) are set to NaN. NaN in v*eps product (from missing FE) is zeroed for variance computation (matching R's did_imputation which drops unimputable obs).
- **NaN inference for undefined statistics:** t_stat uses NaN when SE is non-finite or zero; p_value and CI also NaN. Matches CallawaySantAnna NaN convention.
- **Pre-trend test:** Uses iterative demeaning (same as Step 1 FE) for exact within-transformation on unbalanced panels. One-pass demeaning is only exact for balanced panels.
- **Overall ATT variance:** Weights zero out non-finite tau_hat and renormalize, matching the ATT estimand (which averages only finite tau_hat). `_compute_conservative_variance` returns 0.0 for all-zeros weights, so the n_valid==0 guard is necessary to return NaN SE.
- **`balance_e` cohort filtering:** When `balance_e` is set, cohort balance is checked against the *full panel* (pre + post treatment) via `_build_cohort_rel_times()`, requiring observations at every relative time in `[-balance_e, max_h]`. Both analytical aggregation and bootstrap inference use the same `_compute_balanced_cohort_mask` with pre-computed cohort horizons.
- **Bootstrap clustering:** Multiplier bootstrap generates weights at `cluster_var` granularity (defaults to `unit` if `cluster` not specified). Invalid cluster column raises ValueError.
- **Non-constant `first_treat` within a unit:** Emits `UserWarning` identifying the count and example unit. The estimator proceeds using the first observed value per unit (via `.first()` aggregation), but results may be unreliable.
- **treatment_effects DataFrame weights:** `weight` column uses `1/n_valid` for finite tau_hat and 0 for NaN tau_hat, consistent with the ATT estimand (unweighted), or normalized survey weights `sw_i/sum(sw)` when `survey_design` is active.
- **Rank-deficient covariates in variance:** Covariates with NaN coefficients (dropped for rank deficiency in Step 1) are excluded from the variance design matrices `A_0`/`A_1`. Only covariates with finite coefficients participate in the `v_it` projection.
- **Sparse variance solver:** the untreated projection `v_untreated = -A_0 (A_0'[W]A_0)^{-1} A_1'w` factorizes the normal-equations matrix `(A_0'[W]A_0)` once per `fit()` via `scipy.sparse.linalg.factorized` and reuses the factorization across every estimand target (overall ATT, each event-study horizon, each group, and the bootstrap precompute), solving only the target-specific RHS `A_1'w` per target -- factorize-once / solve-many (the design is target-invariant; only `weights` vary). This is **bit-identical** to the prior per-target `scipy.sparse.linalg.spsolve` for a single dense RHS (both use the SuperLU simple driver with the same defaults), built once instead of `O(targets)` times. Mirrors the TwoStageDiD GMM-sandwich `factorized` pattern. An exactly singular `(A_0'[W]A_0)` makes `factorized` raise `RuntimeError`; the build emits a `UserWarning` once per fit (silent-failure audit axis C) and the solve routes to a **sparse LSMR least-squares fallback** (`scipy.sparse.linalg.lsmr`, `atol=btol=1e-14`) — the previous dense `lstsq(toarray())` fallback materialized the `O((U+T+K)^2)` normal matrix, an OOM risk on large panels (2026-07). Solver choice cannot change the estimator output: least-squares solutions of the singular system differ only by `null(sqrt(W) A_0)` components, which the downstream projection `v = -[W_0] A_0 z` annihilates (dense-oracle parity test). Convergence is validated fail-closed: `istop` in `{0, 1, 2, 4, 5}` counts as certified (4/5 are SciPy's machine-precision analogues of 1/2); an uncertified stop gets one retry with an uncapped condition limit, then raises internally and the variance boundary reports a **full NaN inference tuple** — raising rather than returning NaN matters because the missing-FE `nan_to_num` in the psi product would otherwise launder a NaN vector into zeros and a finite, wrong variance. A defensive per-target non-finite solve likewise routes to the LSMR fallback with a per-target `UserWarning`, so callers always know variance estimates came from the degraded path. The design is built/cached in `_build_untreated_projection` and solved per target in `_solve_untreated_v`.
- **Note:** Survey weights enter ImputationDiD via weighted iterative FE (Step 1), survey-weighted ATT aggregation (Step 3), and design-based variance via `compute_survey_if_variance()`. PSU clustering, stratification, and FPC are fully supported in the Theorem 3 variance path. When `resolved_survey` is present, the observation-level influence function (`v_it * epsilon_tilde_it`) is passed to `compute_survey_if_variance()` which applies the stratified PSU-level sandwich with FPC correction. Strata also enters survey df (n_PSU - n_strata) for t-distribution inference. Bootstrap + survey supported (Phase 6) via PSU-level multiplier weights.
- **Bootstrap inference:** Uses multiplier bootstrap on the Theorem 3 influence function: `psi_i = sum_t v_it * epsilon_tilde_it`. Cluster-level psi sums are pre-computed for each aggregation target (overall, per-horizon, per-group), then perturbed with multiplier weights (Rademacher by default; configurable via `bootstrap_weights` parameter to use Mammen or Webb weights, matching CallawaySantAnna). This is a library extension (not in the paper) consistent with CallawaySantAnna/SunAbraham bootstrap patterns.
- **Auxiliary residuals (Equation 8):** Implements the paper's *unit-clustered* Equation 8 aggregator, `tau_tilde_g = sum_i (sum_{t in G_g,i} v_it)(sum_{t in G_g,i} v_it * tau_hat_it) / sum_i (sum_{t in G_g,i} v_it)^2` (Borusyak-Jaravel-Spiess 2024, eq. 8, p. 3272; minimal-excess-variance derivation in Supplementary Appendix A.8): for each unit form the within-unit weight sum `a_{i,g}` and weighted-effect sum `b_{i,g}` over the unit's observations in group `g`, then combine across units. Groups partition `Omega_1` via `aux_partition` (default `"cohort_horizon"` = cohort × event-time; also `"cohort"` / `"horizon"`). Unimputable (NaN `tau_hat`) and off-target observations carry `v_it = 0` and are excluded from the aggregation — exact for finite `tau_hat` (a zero-weight row adds 0 to both `a` and `b`) and NaN-safe; a group with no contributing observations falls back to the unweighted group mean (a variance no-op, since `psi_g = sum_t v_it * eps_tilde_it = 0` there).
- **Note (deviation from R):** R `didimputation::did_imputation` computes the auxiliary aggregator as `sum(v_it^2 * tau_hat_it) / sum(v_it^2)` grouped by cohort × event-time only (no partition control is exposed). At that partition each unit contributes at most one observation per group, so the paper's unit-clustered Equation 8 reduces exactly to `sum(v^2 * tau)/sum(v^2)` — i.e. diff-diff matches R at the default `aux_partition="cohort_horizon"` (pinned in `tests/test_methodology_imputation.py::TestImputationDiDParityR`). diff-diff additionally offers the coarser `aux_partition="cohort"` / `"horizon"` groupings (where a unit may contribute several observations to a group), which have no R analogue and are validated by hand-calculation. Both implement the same paper Equation 8; only the available partition granularity differs. The earlier observation-level mean `sum(v*tau)/sum(v)` (pre-3.5.x) coincided with this only under uniform within-group weights; it was corrected to the exact unit-clustered form during the ImputationDiD methodology validation.
- **Note (leave-one-out variance refinement, Supp. App. A.9):** the opt-in `leave_one_out=True` applies the Borusyak-Jaravel-Spiess (2024) Supplementary Appendix A.9 finite-sample refinement. The non-LOO `tau_tilde_g` (eq. 8) is built from the fitted `tau_hat_it`, which contain the noise `epsilon_it`, so it partially overfits and the auxiliary residuals `epsilon_tilde_it` are too small, biasing the variance **downward**. LOO recomputes each unit's group aggregate excluding that unit, `tau_tilde_it^LO = sum_{j!=i} v_jg^2 T_jg / sum_{j!=i} v_jg^2`, implemented efficiently (A.9) by rescaling each treated residual `epsilon_tilde_it^LO = epsilon_tilde_it / (1 - v_ig^2 / sum_j v_jg^2)` — where `v_ig = sum_{t in G_g,i} v_it` and `sum_j v_jg^2` are already materialized in `_compute_auxiliary_residuals_treated` as `a_{i,g}` and `per_group['den']`. That rescale reproduces the direct-LOO per-unit cluster sum `psi_i = sum_t v_it * epsilon_tilde_it` **exactly** (`psi_i^rescale = a_{i,g}(T_ig - tau_tilde_g) D/(D - a^2) = psi_i^direct-LOO`; a machine-precision-verified identity — **paper-fidelity** to the source `tau_tilde_it^LO`, not merely internal consistency). At the **default unit clustering** LOO gives a larger, less-downward-biased SE (Prop. A8: unbiased for an upper bound on the true variance; an equal-weight K-unit group inflates residuals by exactly `K/(K-1)`). **Default `leave_one_out=False`** preserves R `didimputation` parity (which omits LOO). **Edge (App. A.9 fn. 51):** a group with a single positive-weight unit has an undefined LOO (rescale factor `1/0`); such groups keep the non-LOO residual and the fit emits one consolidated `UserWarning` (a coarser `aux_partition` reduces singletons); a genuinely unit-dominated `>=2`-unit group keeps its large finite factor (intended inflation). **Composition scope:** the rescale operates on `epsilon_tilde`, so it flows through the coarser-`cluster=` CR sum, the analytical survey PSU-TSL variance, and the multiplier bootstrap unchanged — but Prop. A8's upper-bound guarantee and the `LOO >= non-LOO` direction hold at the default unit clustering only (under a coarser `cluster=`, per-unit `psi` inflation can partially cancel), so those compositions are a documented library extension, not paper-derived. **Replicate-weight survey designs** (BRR / Fay / JK1 / JKn / SDR) raise `NotImplementedError` with `leave_one_out=True`: replicate variance is computed by per-replicate point-estimate refits, bypassing the conservative-IF residual path where the LOO rescale lives, so LOO would silently no-op (fail-closed, no-silent-failures). **Effective-singleton guard:** a group's singleton test counts units with *positive* squared weight (not raw rows), and the leave-one-out denominator is the exact sum of the *other* units' squared weights (not `D - v_ig^2`, which can cancel to `<= 0` in float64 for an extremely dominated `>=2`-unit group and would silently revert it to non-LOO). **Reference / validation:** the authors' Stata `did_imputation` ships the same option (`leaveout`), and the library LOO SE is now **measured against it** — the LOO SE matches to ~1e-9 (the point to ~2e-8; cross-implementation: `did_imputation` goes through `reghdfe`, the library through its own sparse IF solver; the parity test gates at `abs=1e-7`) across the overall ATT and all event-study horizons on the committed panel (`benchmarks/stata/generate_imputation_loo_golden.do` → `benchmarks/data/didimputation_loo_stata_golden.json`; `tests/test_imputation_loo_stata_parity.py`, which also three-way-confirms the non-LOO SE against R). The partition is pinned explicitly `avgeffectsby(Ei t)` (== `aux_partition="cohort_horizon"`), which is also `did_imputation`'s current default — pinned to keep the validation estimand self-describing and robust to a future default change. Stata is node-locked (not in CI, golden committed, exactly like the R arm), so the CI-runnable guards remain the exact psi-identity + hand-calc + MC coverage (`tests/test_methodology_imputation.py::TestB2024AppendixA9LeaveOneOut`). Source: arXiv:2108.12419v5 App. A.9 (the REStud Supplementary Material is canonical); the main-article review's A.9 GAP is now filled (see the `borusyak-jaravel-spiess-2024-review.md` provenance note). The stronger Prop. A8 variant that *also* leaves out for the `delta_hat` covariate estimation (exact-unbiased upper bound) is noted but not implemented — the Stata option is the `tau_tilde` residual rescale only.
- **Note:** The Step-1 iterative FE solver (`_iterative_fe`) routes through the shared bincount Gauss-Seidel helper `diff_diff.utils._iterative_fe_solve`, and the covariate/pre-trend within-transformation routes through the shared MAP engine `diff_diff.utils.demean_by_groups` (factorize-once + `np.bincount`, optional Rust kernel; group order `[time, unit]` preserving the historical time-then-unit sweep) — the same convergence contract, accumulation-order numerics (~1e-10 vs the pre-3.7 pandas loops, not bit-for-bit), and `max_iter=10_000` budget documented under "Absorbed Fixed Effects with Survey Weights". Both surfaces emit `UserWarning` via `diff_diff.utils.warn_if_not_converged` when `max_iter` exhausts without reaching `tol` (the demean warning now carries the shared-engine label naming the affected variables rather than the estimator name). Silent return of the current iterate was classified as a silent failure under the Phase 2 audit and replaced with an explicit signal to match the logistic/Poisson IRLS pattern in `linalg.py`.
- **Note:** Zero-total-weight groups (e.g. whole PSUs zeroed by JK1/BRR replicate weights, which reach Step 1 unmasked — `keep_mask` only drops always-treated units): a unit/period whose observations ALL carry zero weight has no identifying contribution and surfaces as `NaN` FE (key retained for the rank-condition membership check; matches the SpilloverDiD `_iterative_fe_subset` REGISTRY contract — never a silent finite `0.0`), and the shared demean engine's inert-row guard leaves those rows un-demeaned instead of NaN-poisoning the column. Before 3.7 the pandas loops divided 0/0 there: the covariate replicate path NaN-poisoned `y_dm`/`X_dm`, failed EVERY replicate refit inside `solve_ols(check_finite=True)`, and returned NaN SEs after a non-convergence warning storm; a main fit with zero-weight rows + covariates raised the same opaque `ValueError`. Both now produce finite results. `TwoStageDiD._mask_nan_ytilde`'s "non-finite imputed outcomes" `UserWarning` is suppressed (via `warn_nan=False`) ONLY inside the replicate-refit closures, where NaN FE for zeroed PSUs is expected mechanics — the main-fit warning is unchanged.
- **Note:** `vcov_type` is permanently narrow to `{"hc1"}` per the Theorem 3 IF-based variance decomposition. Analytical-sandwich families `{classical, hc2, hc2_bm}` are rejected at `__init__` — the per-unit influence function aggregation has no equivalent single design matrix on which hat-matrix leverage or Bell-McCaffrey Satterthwaite DOF can be defined. `cluster=<col>` invokes per-cluster IF summation (Theorem 3 equation 7 conservative variance, `sigma_sq = (cluster_psi_sums**2).sum()` — plain CR1 with no Stata-style `(n-1)/(n-p)` finite-sample factor because the IF has no design-matrix `p` in the OLS sense); `cluster=None` (the default) routes the SAME Theorem 3 cluster-summed IF variance with `cluster_var = unit` (the unit column passed to `fit()`), so the summary renders `"CR1 cluster-robust at <unit>, G=<n_units>"` rather than the generic `"HC1"` label; `survey_design=` invokes TSL on the combined IF. Under bootstrap (`n_bootstrap > 0`) the analytical variance-family label is suppressed in `summary()` because `fit()` overwrites the reported SE/CI/p-value with bootstrap_results (mirrors the canonical `DiDResults` gate at `results.py:213-226`). `vcov_type='conley'` is deferred to the ImputationDiD Conley follow-up row in DEFERRED.md.
- **Note:** `cluster=<col>` combined with a replicate-weight `SurveyDesign` raises `NotImplementedError` at `fit()`. Replicate-weight variance ignores PSU/cluster structure entirely (replicates encode the design implicitly), so honoring `cluster=` would silently no-op while populating `cluster_name`/`n_clusters` on Results dishonestly. Either omit `cluster=` (the replicate weights encode the design structure implicitly) or use a non-replicate survey design (with explicit strata/psu/fpc). Mirrors the `CallawaySantAnna` and `TripleDifference` fail-closed guards.
- **Note:** Bootstrap path returns NaN SE when fewer than 2 independent clusters/PSUs are available (`n_clusters < 2` for the analytical-cluster bootstrap path, `n_psu < 2` for the survey-PSU bootstrap path). Without this guard the multiplier bootstrap SE collapses to ≈0 from BLAS roundoff (NOT NaN), and downstream zero-SE guards check exact 0 and miss the degenerate-design case. NaN propagates to all inference fields (SE/CI/p-value) plus per-horizon and per-group bootstrap dicts.

**Reference implementation(s):**
- Stata: `did_imputation` (Borusyak, Jaravel, Spiess; available from SSC)
- R: `didimputation` package (Kyle Butts)

**Requirements checklist:**
- [x] Step 1: OLS on untreated observations only (never-treated + not-yet-treated)
- [x] Step 2: Impute counterfactual `Y_hat_it(0)` for treated observations
- [x] Step 3: Aggregate with researcher-chosen weights `w_it`
- [x] Conservative clustered variance estimator (Theorem 3, Equation 7)
- [x] Auxiliary model for treated residuals (unit-clustered Equation 8) with configurable partition (`aux_partition`)
- [x] Leave-one-out finite-sample variance refinement (Supp. App. A.9, opt-in `leave_one_out`, default off)
- [x] Supports unit FE, period FE, and time-varying covariates
- [x] Refuses to estimate unidentified estimands (Proposition 5) — sets NaN with warning
- [x] Pre-trend test uses only untreated observations (Test 1, Equation 9)
- [x] Supports balanced and unbalanced panels (iterative Gauss-Seidel demeaning for exact FE)
- [x] Event study and group aggregation

- **Note (post-fit aggregate() - rows M-021/M-118):** `fit(aggregate=, balance_e=)` is deprecated (3.9; removed 4.0; joint FutureWarning, warn-and-still-work; the `imputation_did` wrapper forwards the shared sentinel so plain wrapper calls never fire the aggregate warning; since 3.9 the wrapper itself warns per M-070) in favor of post-fit `ImputationDiDResults.aggregate(type, balance_e=)` - a PANEL-BACKED lazy recompute kit (not an EIF-payload kit: ES/group aggregation is a target-specific Theorem-3 recompute - each `balance_e` re-masks which treated observations enter every horizon and re-solves the untreated projection - so no compact influence payload can replace the frame). (a) RETAINED BUFFERS (memory contract): the kit's bookkeeping holds REFERENCES to the SAME per-fit objects `_fit_data` already retains for `pretrend_test()` - the working panel copy (all user columns plus `_tau_hat`/`_rel_time`/`_never_treated`), the Omega masks, `unit_fe`/`time_fe`/`grand_mean`/`delta_hat`/`kept_cov_mask`, the resolved survey design, and `survey_weights` - ZERO marginal memory, and pickles are unchanged via memoization (`_estimator_ref` already ships these objects); plus value SNAPSHOTS for isolation (a `treatment_groups` copy, config scalars, a `dataclasses.replace` copy of `survey_metadata`, `overall_att`, `n_treated_obs`) and TWO df-provenance scalars (`survey_df_seed`, what the analytical aggregators received; `survey_df_final`, what the stored overall inference received). Each `aggregate()` call runs on a fresh throwaway host with a call-local projection cache (the fit-local factorizations are unpicklable and never retained). (b) `balance_e` uses the BALANCED-WINDOW rule: a cohort is retained iff its observed relative-time set - checked against the FULL panel via `_build_cohort_rel_times()` - covers the contiguous window `[-balance_e, max_h]`; the SAME rule TwoStageDiD uses, divergent from CS/EfficientDiD's anchor-horizon rule. A window no cohort satisfies warns and yields the reference-marker-only dict (a legal near-empty container). (c) BOOTSTRAP fits: 'simple' RELAYS the stored overall quintet verbatim (finite safe_inference t included) with a NaN df column, while the RECOMPUTE levels fail closed (the per-target psi machinery makes exact replay tractable - a TODO row); the prior uniform fail-closed rule was superseded 2026-08-05 with the M-027 per-level convergence. (d) CONTAINER ADMISSION into `compute_honest_did`/`compute_pretrends_power` is REJECTED BY DESIGN (both terminal TypeErrors state it): the surface carries no joint event-study covariance - per-horizon conservative SEs only (container `vcov=None`; the scalar `df_survey` channel is its only df provenance, the per-row hole being the tracked M-092-completion TODO row). (e) RELAY CONVENTIONS: 'simple' relays the stored overall quintet bit-exact with `n = n_treated_obs`, `n_kind="obs"` (the treated/control UNIT sets overlap - a treated unit with pre-periods counts in both - so the CS/EDiD disjoint-units convention cannot apply; |Omega_1| is the population the ATT averages over, of which only finite-tau-hat observations enter the average - `n` reports the raw count, so on partially unidentified fits `n` exceeds the averaged support) and `df = survey_df_final`; 'group' rows carry per-row `df_used` captured at each row's `safe_inference` (the replicate override rewrites it, the bootstrap override clears it, the all-NaN cohort branch writes no key - consumers read via `.get`); 'event_study' rides the shared `_from_relative_dict` builder via a carrier whose metadata is a copy-on-use of the KIT's fit-final metadata copy. REPLICATE-WEIGHT fits replay the extracted `_replicate_override_aggregates` with a LEVEL-MATCHED stack: `compute_replicate_refit_variance` validates replicates JOINTLY (all-finite rows), so `aggregate(L)` reproduces `fit(aggregate=L)` exactly, a `fit(aggregate='all')` surface is NOT the equivalence target when a replicate NaNs on exactly one family's targets, and - the documented migration delta - moving a replicate fit from `fit(aggregate=)` to plain fit + post-fit `aggregate()` can change the public OVERALL row's se/CI/df on such degenerate designs (each surface self-consistent; pinned in the contract tests). `pretrends=True` + replicate: post-fit `aggregate('event_study')` raises the same NotImplementedError the fit-time gate raises (per-replicate lead refits unimplemented); 'group'/'simple' still work. Recompute re-emits the fit-time warnings (LSMR, Prop-5, empty-window) with fit-tuned stacklevels - post-fit attribution lands on a library frame, an accepted verbatim-move trade-off. `DiagnosticReport` now derives this container internally on plain fits (raw `event_study_effects` absent), so its `parallel_trends` (`pretrends=True` fits) and `heterogeneity` checks run without the deprecated fit-time kwarg; the re-emitted recompute warnings are captured and re-published on the consuming report section (record-and-republish), and derivation failures (bootstrap, missing kit, the replicate gate above) surface as explicit per-check skip reasons.

---

## TwoStageDiD

**Primary source:** [Gardner, J. (2022). Two-stage differences in differences. arXiv:2207.05943.](https://arxiv.org/abs/2207.05943)

- **Note (rank-guarded TSL-variance bread):** The Stage-2 variance bread `(X_2'WX_2)^{-1}` is inverted by the shared `_rank_guarded_inv` (`diff_diff/linalg.py`) on BOTH the analytical (`two_stage.py`) and multiplier-bootstrap (`two_stage_bootstrap.py`) surfaces. `np.linalg.solve` raised only on an exactly-singular bread (prior fallback: dense `lstsq`); a **near**-singular `X_2'WX_2` would otherwise flow a garbage inverse (~1e13) into the SE. The rank-guard truncates redundant directions → finite SE on the identified subspace (NaN at rank 0) and warns. A dropped (unidentified) Stage-2 coefficient (event-time / group effect) is reported with **NaN** SE on both the analytical and bootstrap surfaces — not the zero-filled `0`. `X_2` is the Stage-2 indicator design (treatment/event-time/group dummies), not user covariates. See the CallawaySantAnna "rank-guarded IF standard errors" Note. Sibling of axis-A finding #17.

**Key implementation requirements:**

*Assumption checks / warnings:*
- **Parallel trends (same as ImputationDiD):** `E[Y_it(0)] = alpha_i + beta_t` for all observations.
- **No-anticipation effects:** `Y_it = Y_it(0)` for all untreated observations.
- Treatment must be absorbing: `D_it` switches from 0 to 1 and stays at 1.
- Always-treated units (treated in all periods) are excluded with a warning, since they have no untreated observations for Stage 1 FE estimation.

*Estimator equation (two-stage procedure, as implemented):*

```
Stage 1. Estimate unit + time fixed effects on untreated observations only (it in Omega_0):
    Y_it = alpha_i + beta_t + epsilon_it
    Compute residuals: y_tilde_it = Y_it - alpha_hat_i - beta_hat_t  (for ALL observations)

Stage 2. Regress residualized outcomes on treatment indicators (on treated observations):
    y_tilde_it = tau * D_it + eta_it
    (or event-study specification with horizon indicators)
```

Point estimates are identical to ImputationDiD (Borusyak et al. 2024). The two-stage procedure is algebraically equivalent to the imputation approach: both estimate unit+time FE on untreated observations and recover treatment effects from the difference between observed and counterfactual outcomes.

*Variance: GMM sandwich (Newey & McFadden 1994 Theorem 6.1):*

The variance accounts for first-stage estimation error propagating into Stage 2, following the GMM framework:

```
V(tau_hat) = (D'D)^{-1} * Meat * (D'D)^{-1}     [(D'D)^{-1} = GLOBAL GMM bread (Jacobian inverse)]

Meat = sum_c ( sum_{i in c} psi_i )( sum_{i in c} psi_i )'   [score outer-product, clustered at unit]
```

where `psi_i` is the stacked influence function for unit i across all its observations, combining the Stage 2 score and the Stage 1 correction term.

**Variance is faithful to the paper (global Jacobian inverse).** Gardner (2022) §3.3 derives the variance by reading the two stages as a joint GMM estimator (Hansen 1982) and applying Newey & McFadden (1994) Theorem 6.1: `v` is the last element of `E[∂f/∂(λ,γ,β)]^{-1} E[ff'] E[∂f/∂(λ,γ,β)]^{-1'}` — the **global** Jacobian inverse (the `(D'D)^{-1}` bread above), with the score outer-product `E[ff']` clustered at the unit per the reference Stata GMM `vce(cluster id)` (Appendix B). Our global `(D'D)^{-1}` bread + unit-clustered meat **matches** this and the R `did2s` implementation; there is **no** per-cluster inverse. (Equation (6) in the paper is the *event-study regression specification*, not a variance formula — an earlier "Equation 6 per-cluster inverse `(D_c'D_c)^{-1}`" note was a misattribution, corrected per `docs/methodology/papers/gardner-2022-review.md`.)

**No finite-sample adjustments:** The variance estimator uses the raw asymptotic sandwich without degrees-of-freedom corrections (no HC1-style `n/(n-k)` adjustment). This matches the R `did2s` implementation.

*Bootstrap:*

Our implementation uses multiplier bootstrap on the GMM influence function: cluster-level `psi` sums are pre-computed, then perturbed with multiplier weights (Rademacher by default; configurable via `bootstrap_weights` parameter to use Mammen or Webb weights, matching CallawaySantAnna). The R `did2s` package **defaults to analytical corrected clustered SEs** (`bootstrap = FALSE`, the same GMM sandwich); its block bootstrap is *optional* (`bootstrap = TRUE`, resampling clusters with replacement). All approaches are asymptotically valid; the multiplier bootstrap is computationally cheaper and consistent with the CallawaySantAnna/ImputationDiD bootstrap patterns in this library.

*Edge cases:*
- **Always-treated units:** Units treated in all observed periods have no untreated observations for Stage 1 FE estimation. These are excluded with a warning listing the affected unit IDs. Their treated observations do NOT contribute to Stage 2.
- **Rank condition violations:** If the Stage 1 design matrix (unit+time dummies on untreated obs) is rank-deficient, or if certain unit/time FE are unidentified (e.g., a unit with no untreated periods after excluding always-treated), the affected FE produce NaN. Behavior controlled by `rank_deficient_action`: "warn" (default), "error", or "silent".
- **NaN y_tilde handling:** When Stage 1 FE are unidentified for some observations, the residualized outcome `y_tilde` is NaN. These observations are zeroed out (excluded) from the Stage 2 regression and variance computation, matching the treatment of unimputable observations in ImputationDiD.
- **NaN inference for undefined statistics:** t_stat uses NaN when SE is non-finite or zero; p_value and CI also NaN. Matches CallawaySantAnna/ImputationDiD NaN convention.
- **Event study aggregation:** Horizon-specific effects use the same two-stage procedure with horizon indicator dummies in Stage 2. Unidentified horizons (e.g., long-run effects without never-treated units, per Proposition 5 of Borusyak et al. 2024) produce NaN.
- **Pre-period event study coefficients (`pretrends=True`):** When enabled, the Stage 2 design matrix `X_2` includes pre-period relative-time dummies. Pre-period observations have `y_tilde = Step 1 residual` by construction. The GMM sandwich variance accounts for Stage 1 estimation error (Gardner 2022 §3.3; Newey-McFadden 1994, Theorem 6.1 — the paper has no numbered theorems). Only affects event study aggregation; overall ATT unchanged.
- **balance_e with no qualifying cohorts:** If no cohorts have sufficient pre/post coverage for the requested `balance_e`, a warning is emitted and event study results contain only the reference period.
- **All-Proposition-5 event studies (3.9 fix, 2(b) PR-3b):** when every non-reference horizon is Proposition-5-unidentified (`h >= h_bar` with treated observations), the previously reference-only early return now retains the Prop-5 rows — all-NaN inference with `n_obs > 0` — and emits the consolidated Proposition-5 warning, identically on the fit-time and post-fit routes (previously the rows were dropped from the dict in this corner; the accompanying unidentified-FE warnings still fired, but real treated horizons read as absent rather than unidentified).
- **No never-treated units (Proposition 5):** When there are no never-treated units and multiple treatment cohorts, horizons h >= h_bar (where h_bar = max(groups) - min(groups)) are unidentified per Proposition 5 of Borusyak et al. (2024). These produce NaN inference with n_obs > 0 (treated observations exist but counterfactual is unidentified) and a warning listing affected horizons. Matches ImputationDiD behavior. Proposition 5 applies to event study horizons only, not cohort aggregation — a cohort whose treated obs all fall at Prop 5 horizons naturally gets n_obs=0 in group effects because all its y_tilde values are NaN.
- **Zero-observation horizons after filtering:** When `balance_e` or NaN `y_tilde` filtering results in zero observations for some non-Prop-5 event study horizons, those horizons produce NaN for all inference fields (effect, SE, t-stat, p-value, CI) with n_obs=0.
- **Zero-observation cohorts in group effects:** If all treated observations for a cohort have NaN `y_tilde` (excluded from estimation), that cohort's group effect is NaN with n_obs=0.
- **Note:** Survey weights in TwoStageDiD GMM sandwich via weighted cross-products: bread uses (X'_2 W X_2)^{-1}, gamma_hat uses (X'_{10} W X_{10})^{-1}(X'_1 W X_2), per-cluster scores multiply by survey weights. PSU clustering, stratification, and FPC are fully supported in the meat matrix via `_compute_stratified_meat_from_psu_scores()`. When strata or FPC are present, the meat computation replaces `S' S` with the stratified formula `sum_h (1 - f_h) * (n_h/(n_h-1)) * centered_h' centered_h`. Strata also enters survey df (n_PSU - n_strata) for t-distribution inference. Bootstrap + survey supported (Phase 6) via PSU-level multiplier weights.
- **Note (documented synthesis — Wave E.3 parity, full-domain survey design under always-treated drop):** when the always-treated handler drops units that lack untreated observations, TwoStageDiD preserves the FULL-DOMAIN resolved survey design (`n_psu`, `n_strata`, `df_survey`, `strata`, `fpc`, `psu`) for variance estimation. Per-cluster stage-1 / stage-2 score aggregates are computed at the post-drop fit-sample length and then zero-padded onto the full-domain unique-PSU list via `score_pad_mask` + `cluster_ids_full` kwargs on `_compute_gmm_variance`; PSUs that contain only always-treated rows get zero score rows but still count toward `G_full` for `n_psu` / `df_survey` accounting. Stage-1 / stage-2 OLS solve continues to operate on the post-drop sample (`survey_weights` subsetted for OLS arithmetic; bread `(X'_2 W X_2)^{-1}` unchanged because dropped rows contribute zero score under zero-padded weights). Mirrors SpilloverDiD Wave E.3 (PR #482, merge 24de9062) and adopts the canonical "zero-pad scores to full panel + retain full-design resolved survey" convention from R `survey::svyrecvar(subset())` (Lumley 2010 §2.5 "Domains and subpopulations") and the in-library precedents at `imputation_aggregation.py` (`_compute_lead_coefficients`, PreTrendsImputation) and `prep.py:1401-1432` (DCDH cell variance). Cluster-injection (`_inject_cluster_as_psu`) operates on the FULL-DOMAIN cluster column (sourced from `data` pre-drop, not the post-drop `df`) so `resolved_survey.strata` and the injected `psu` array stay length-aligned. Pre-PR, the always-treated drop physically subsetted `resolved_survey.weights / strata / psu / fpc / replicate_weights` via `replace(resolved_survey, ...)` and recomputed `n_psu` / `n_strata` on the post-drop sample, producing artificially-deflated `df_survey` when a PSU contained only always-treated rows; tests at `tests/test_two_stage.py::TestTwoStageDiDWaveE3ParityAlwaysTreated` lock the parity contract.
- **Note:** The Stage-1 iterative FE solver (`_iterative_fe`) routes through the shared bincount Gauss-Seidel helper `diff_diff.utils._iterative_fe_solve`, and the covariate within-transformation routes through the shared MAP engine `diff_diff.utils.demean_by_groups` (factorize-once + `np.bincount`, optional Rust kernel; group order `[time, unit]` preserving the historical time-then-unit sweep) — the same convergence contract, accumulation-order numerics (~1e-10 vs the pre-3.7 pandas loops, not bit-for-bit), and `max_iter=10_000` budget documented under "Absorbed Fixed Effects with Survey Weights". Both surfaces emit `UserWarning` via `diff_diff.utils.warn_if_not_converged` when `max_iter` exhausts without reaching `tol` (the demean warning now carries the shared-engine label naming the affected variables rather than the estimator name). Silent return of the current iterate was classified as a silent failure under the Phase 2 audit and replaced with an explicit signal to match the logistic/Poisson IRLS pattern in `linalg.py`.
- **Note:** Zero-total-weight groups (e.g. whole PSUs zeroed by JK1/BRR replicate weights, which reach Stage 1 unmasked — `keep_mask` only drops always-treated units): a unit/period whose observations ALL carry zero weight surfaces as `NaN` FE (key retained for the rank-condition membership check; matches the SpilloverDiD `_iterative_fe_subset` REGISTRY contract — never a silent finite `0.0`), and the shared demean engine's inert-row guard leaves those rows un-demeaned instead of NaN-poisoning the column. Before 3.7 the pandas loops divided 0/0 there: the covariate replicate path NaN-poisoned `y_dm`/`X_dm`, failed EVERY replicate refit inside `solve_ols(check_finite=True)`, and returned NaN SEs after a non-convergence warning storm. It now produces finite replicate SEs. `_mask_nan_ytilde`'s "non-finite imputed outcomes" `UserWarning` is suppressed (via `warn_nan=False`) ONLY inside the replicate-refit closures, where NaN FE for zeroed PSUs is expected mechanics — the main-fit warning is unchanged.
- **Note:** When the Stage-2 bread `X'_2 W X_2` is singular, both the analytical TSL variance (`two_stage.py`) and the multiplier-bootstrap bread (`two_stage_bootstrap.py`) now emit a `UserWarning` before falling back to `np.linalg.lstsq`. Previously this fallback was silent. Sibling of axis-A finding #17 in the Phase 2 silent-failures audit; surfaced by the repo-wide lstsq-fallback pattern grep that accompanied the StaggeredTripleDifference fix.
- **Note:** The GMM sandwich and bootstrap paths both use `scipy.sparse.linalg.factorized` for the Stage 1 normal-equations solve `(X'_{10} W X_{10}) gamma = X'_1 W X_2` and fall back to a **certified sparse LSMR** solve when the sparse factorization raises `RuntimeError` on a near-singular matrix (2026-07; previously dense `lstsq(toarray())`, an `O((U+T+K)²)` materialization and OOM risk on large panels — the pattern the ImputationDiD LSMR fix closed). Solver choice cannot change the output: least-squares solutions differ only by `null(X'X) = null(X_10)` components (weighted: `null(W^{1/2}X_10)`; zero-weight rows are inert in every weighted consumer), and every `gamma_hat` consumer is an `X_10`-range functional — `Psi = X_10 gamma`, the GMM score correction `c_g' gamma` with `c_g = X_{10,g}' eps_{10,g}` in `rowspace(X_10)` — so its null component annihilates. The bootstrap's `X_1_sparse @ theta_exact` consumer evaluates theta on treated rows where annihilation does not apply; parity there holds because both dense `lstsq` (SVD) and LSMR return the MIN-NORM least-squares solution, so the solvers agree on the whole vector to iterative tolerance (helper-level dense-lstsq-oracle parity test on a singular Gram + fit-level singular-design V/SE parity test). Convergence is validated fail-closed: `istop` in `{0,1,2,4,5}` certified, one uncapped-conlim retry, then `_LSMRUnconvergedError` — the analytical variance boundary reports NaN vcov/SE (raising matters: the GMM score `nan_to_num` would otherwise launder NaN scores into zeros and a finite, wrong variance) and the bootstrap boundary degenerates to the established `None` contract. All fallback sites (analytical unweighted + weighted, SpilloverDiD Wave-D meat, bootstrap) emit a `UserWarning` (silent-failure audit axis C).
- **Note:** The GMM sandwich re-solves the Stage-1 unit+time fixed effects **exactly** (sparse OLS reusing the `scipy.sparse.linalg.factorized` factorization of `(X'_{10} W X_{10})` already computed for `gamma_hat`), rather than reusing the iterative alternating-projection FE (`_iterative_fe`) that produces the point estimate. The iterative solver converges only to ~1e-7 on unbalanced *untreated* panels — negligible for the ATT, but enough to perturb the variance by ~1% relative to the analytical GMM sandwich. The exact re-solve makes the analytical GMM SE match R `did2s` to ~1e-7 (`tests/test_methodology_two_stage.py::TestTwoStageDiDParityR`), mirroring ImputationDiD's exact-sparse variance path (`imputation.py` `_build_A_sparse`). It also required adding an **intercept column** to `_build_fe_design` so the first-stage column space spans the constant (the grand mean): the prior intercept-free `[unit_1.., time_1..]` layout (drop first unit + first time, no intercept) silently omitted the grand mean, which the exact residual is first-order sensitive to (the iterative point-estimate solver absorbs the grand mean into its mean-based FE, so the point estimate was unaffected). Obs whose unit or time FE are unidentified (NaN; rank-deficient / Proposition-5) fall back to the iterative residual, so those edge cases are unchanged. The reported `overall_att` still uses the iterative FE (preserving point-estimate equivalence with ImputationDiD at 1e-10); only the variance uses the exact residuals.
- **Note:** `vcov_type` is permanently narrow to `{"hc1"}` (Phase 1b threading). TwoStageDiD's variance is the Gardner (2022) two-stage GMM cluster-sandwich `V = (X'_2 W X_2)^{-1} (S' S) (X'_2 W X_2)^{-1}` with the per-cluster GMM-corrected score `S_g = gamma_hat' c_g - X'_{2g} eps_{2g}`. Analytical-sandwich families `{classical, hc2, hc2_bm}` are rejected at `__init__`/`fit()`: the GMM-corrected meat folds first-stage FE estimation uncertainty into the score via the `gamma_hat' c_g` term, so there is no single hat matrix spanning both stages on which HC2 leverage or Bell-McCaffrey Satterthwaite DOF can be defined, and the Gardner first-stage correction has not been derived for the leverage-corrected or homoskedastic meat structures (no reference implementation — `clubSandwich` covers single-equation WLS/OLS CR2, not two-stage GMM; mirrors the SpilloverDiD `vcov_type="classical"` rejection). `cluster=<col>` selects the cluster level; `cluster=None` (the default) still clusters at the `unit` column (`cluster_var = unit`), so the summary renders `"CR1 cluster-robust at <unit>, G=<n_units>"` rather than the generic `"HC1"` label. **Note (deviation from R):** the did2s GMM sandwich uses NO finite-sample multiplier (meat `= S' S`), so the rendered `CR1` family label carries no Stata-style `(n-1)/(n-p)` or `G/(G-1)` factor (matches R `did2s`; same FSA-free convention as ImputationDiD's Theorem 3 variance). Under bootstrap (`n_bootstrap > 0`) the analytical variance-family label is suppressed in `summary()` because `fit()` overwrites the reported SE/CI/p-value with `bootstrap_results` (mirrors `DiDResults` at `results.py:213-226`). `cluster=<col>` combined with a replicate-weight survey design raises `NotImplementedError` (replicate-refit variance ignores `cluster=`). `vcov_type='conley'` is deferred to the TwoStageDiD Conley follow-up row in DEFERRED.md.
- **Note:** Provenance exposure (4.0 program row M-092, no numeric change): the full Gardner-GMM event-study covariance `V` is now persisted as `TwoStageDiDResults.event_study_vcov` (+ `event_study_vcov_index` = the estimated horizons, in Stage-2 column order). The reference period and Proposition-5 horizons are never Stage-2 columns and are absent from the index; an estimated horizon whose observations are all filtered keeps its column with the rank-guard's NaN row/col (consistent with its NaN marginal SE). Persisted only on the analytical paths - `None` under bootstrap (percentile inference has no covariance) and replicate-weight survey designs (the replicate VCV's [overall, ES, groups] layout is not an ES matrix; CS precedent). `event_study_df` records the scalar survey df every estimated ES row's `safe_inference` received (None on non-survey fits: normal-theory GMM inference).

**Reference implementation(s):**
- R: `did2s::did2s()` (Kyle Butts & John Gardner)

**Requirements checklist:**
- [x] Stage 1: OLS on untreated observations only for unit+time FE
- [x] Stage 2: Regress residualized outcomes on treatment indicators
- [x] Point estimates match ImputationDiD
- [x] GMM sandwich variance (Newey & McFadden 1994 Theorem 6.1)
- [x] Global `(D'D)^{-1}` in variance (faithful to Gardner §3.3 / Newey-McFadden GMM sandwich; matches R `did2s`)
- [x] No finite-sample adjustment (raw asymptotic sandwich)
- [x] Analytical GMM SE matches R `did2s` to ~1e-7 (exact Stage-1 re-solve; `TestTwoStageDiDParityR`)
- [x] Always-treated units excluded with warning
- [x] Multiplier bootstrap on GMM influence function
- [x] Event study and overall ATT aggregation

- **Note (post-fit aggregate() - rows M-022/M-119):** `fit(aggregate=, balance_e=)` is deprecated (3.9; removed 4.0; joint FutureWarning, warn-and-still-work; the `two_stage_did` wrapper forwards the shared sentinel so plain wrapper calls never fire the aggregate warning; since 3.9 the wrapper itself warns per M-071) in favor of post-fit `TwoStageDiDResults.aggregate(type, balance_e=)` - a PANEL-BACKED lazy recompute kit: each level is a fresh Stage-2 OLS + joint Gardner-GMM sandwich on a level-specific design, so no compact influence payload exists. (a) RETAINED BUFFERS (memory contract - the FIRST panel retention on TwoStageDiD results, a deliberate break from the CS/EDiD identifier-minimization guarantee, with a `store_kit` opt-out tracked in DEFERRED.md): a COLUMN-SUBSET COPY of the working frame - `unit`/`time`/`outcome`/`first_treat` + covariates + the cluster column (deduplicated: `cluster=` may legally name a core column) + `_never_treated`/`_rel_time`/`_y_tilde` - O(n_obs) on every results object and pickle; the Stage-1 FE model (`unit_fe`/`time_fe`/`grand_mean`/`delta_hat`/`kept_cov_mask`), the Omega masks, the full-domain `keep_mask`, the Wave-E.3-GATED `score_pad_mask`/`cluster_ids_full` values fit actually passed (None unless the always-treated pad was active), `survey_weights`, and the resolved survey design - on replicate designs that adds the O(n_obs x R) replicate matrix; plus value snapshots (`treatment_groups` copy, `ref_period`, `overall_att`, `n_treated_obs`, a `dataclasses.replace` copy of `survey_metadata`) and TWO df scalars (`survey_df_stage2`, the recompute seed; `survey_df_final`, what the stored overall inference received). (b) `balance_e` uses the BALANCED-WINDOW rule (`[-balance_e, max_h]` coverage against the full panel - the ImputationDiD rule, divergent from CS/EfficientDiD's anchor-horizon rule); zero qualifying cohorts warns and yields the reference-row-only dict with `vcov=None`. (c) BOOTSTRAP fits: 'simple' RELAYS the stored overall quintet verbatim (finite safe_inference t included) with a NaN df column, while the RECOMPUTE levels fail closed (per-level GMM scores are function-locals; replay is a TODO row) - the prior uniform rule superseded 2026-08-05 with the M-027 per-level convergence; a fit whose bootstrap FAILED (`bootstrap_results=None`, analytical inference retained) aggregates normally. (d) CONTAINER ADMISSION into `compute_honest_did`/`compute_pretrends_power` is DEFERRED, not by-design (both terminal TypeErrors state it): analytical surfaces DO carry the real joint Gardner-GMM covariance (M-092), but the pre-period coefficients are stage-1 residual MEANS - the reference horizon is dropped from the no-intercept Stage-2 design and the zero anchor row is appended mechanically - not contrasts against the advertised reference, while HonestDiD's Delta^RM/Delta^SD arithmetic hard-codes the `delta_0 = 0` normalization into its boundary/bridge constraints; admission awaits a normalization derivation (either re-estimating Stage 2 with the reference horizon in the design or deriving the residual-to-reference mapping) - the DEFERRED.md paper-gated row. (e) RELAY CONVENTIONS: 'simple' relays the stored overall quintet bit-exact with `n = n_treated_obs`, `n_kind="obs"` (overlapping unit sets - the StackedDiD carve-out class; the pre-filter |Omega_1| count, while the ATT's Stage-2 support excludes rows whose `y_tilde` is non-finite - on such degenerate fits `n` exceeds the averaged support) and `df = survey_df_final` (on replicate fits that value came from the `[overall]`-only joint stack - snapshotted, never re-derived); 'group' relays a SCALAR df broadcast (deliberate divergence from ImputationDiD's per-row `df_used`: `_stage2_group` passes one immutable `survey_df` to every row's `safe_inference`, so the scalar is provenance-exact by construction and the moved method stays verbatim); 'event_study' reproduces the M-092 container contract exactly - analytical fits thread the recomputed joint vcov + `vcov_index` + the finite-and->0 df scalar through the carrier, replicate fits thread `vcov=None`/`index=None` with the REPLAYED level-matched df, and the carrier's metadata is a copy-on-use of the KIT's fit-final metadata copy. REPLICATE-WEIGHT fits replay the extracted `_replay_replicate_inference` with a LEVEL-MATCHED stack (the ImputationDiD semantics: `aggregate(L)` reproduces `fit(aggregate=L)`; `fit(aggregate='all')` is not the equivalence target on degenerate designs; the OVERALL-row migration delta on such designs is documented and pinned). Recompute re-emits fit-time warnings with fit-tuned stacklevels - an accepted verbatim-move trade-off. `DiagnosticReport` now derives this container internally on plain fits (raw `event_study_effects` absent), so its `parallel_trends` (`pretrends=True` fits; the recomputed joint vcov drives the joint-Wald path) and `heterogeneity` checks run without the deprecated fit-time kwarg; re-emitted recompute warnings are captured and re-published on the consuming report section, and derivation failures surface as explicit per-check skip reasons.

---

## StackedDiD

**Primary source:** Wing, C., Freedman, S. M., & Hollingsworth, A. (2024). Stacked Difference-in-Differences. NBER Working Paper 32054. http://www.nber.org/papers/w32054

**Key implementation requirements:**

*Assumption checks / warnings:*
- Assumption 1 (No Anticipation): ATT(a, a+e) = 0 for all e < 0
- Assumption 2 (Common Trends): E[Y_{s,a+e}(0) - Y_{s,a-1}(0) | A_s = a] = E[Y_{s,a+e}(0) - Y_{s,a-1}(0) | A_s > a + e]
- Clean controls must exist for each sub-experiment (IC2)
- Event window must fit within observed data range (IC1)

*Target parameter (Equation 2):*

    theta_kappa^e = sum_{a in Omega_kappa} ATT(a, a+e) * (N_a^D / N_Omega_kappa^D)

where:
- `theta_kappa^e` = trimmed aggregate ATT at event time e
- `Omega_kappa` = trimmed set of adoption events satisfying IC1 and IC2
- `N_a^D` = number of treated units in sub-experiment a
- `N_Omega_kappa^D` = total treated units across all sub-experiments in trimmed set

*Estimator equation (Equation 3 — weighted saturated event study, recommended):*

    Y_sae = alpha_0 + alpha_1 * D_sa + sum_{h != -1} [lambda_h * 1(e=h) + delta_h * D_sa * 1(e=h)] + U_sae

Estimated via WLS with Q-weights. The delta_h coefficients identify theta_kappa^e.

*Q-weights (Section 5.3, Table 1):*

    Q_sa = 1                                           if D_sa = 1 (treated)
    Q_sa = (N_a^D / N^D) / (N_a^C / N^C)             if D_sa = 0 (control, aggregate weighting)
    Q_sa = (Pop_a^D / Pop^D) / (N_a^C / N^C)         if D_sa = 0 (control, population weighting)
    Q_sa = ((N_a + N_a^C)/(N^D+N^C)) / (N_a^C/N^C)  if D_sa = 0 (control, sample share weighting)

*Standard errors (Section 5.4):*
- Default: Cluster-robust standard errors at the group (unit) level
- Alternative: Cluster at group x sub-experiment level
- Both approaches yield approximately correct coverage when clusters > 100 (Table 2)
- No special bootstrap procedure specified; standard cluster-robust SEs recommended
- For post-period average: delta method or `lincom`/`marginaleffects`

*Variance families (`vcov_type`):*
- `hc1` (default) — CR1 Liang-Zeger cluster-robust via `solve_ols(weights=composed_weights, vcov_type="hc1")` (Stata-style `G/(G-1) * (n-1)/(n-p)` finite-sample correction). Matches `clubSandwich::vcovCR(lm(weights=Q,...), cluster=~unit, type="CR1S")` at atol=1e-10 on the new `benchmarks/data/stacked_did_golden.json` fixture (R-side target is `CR1S` not plain `CR1` — the latter omits the `(n-1)/(n-p)` term and diverges by ~1.4% on a 325-row, 10-column stacked design). Bit-equal to the prior bake-Q-into-X output up to float64 multiplication ordering (atol=1e-13; WLS-CR1 score is algebraically invariant between the two forms).
- `hc2_bm` — CR2 Bell-McCaffrey via `solve_ols(weights=composed_weights, vcov_type="hc2_bm")` routed through the clubSandwich WLS-CR2 port (PR #475). The diff-diff implementation matches clubSandwich's specific algebra (W not √W in the hat matrix, W² in the bias-correction term, unweighted residuals in the score construction) — see Phase 1a `hc2_bm + weights` registry row for the full derivation. Matches `clubSandwich::vcovCR(lm(weights=Q,...), cluster=~unit, type="CR2") + coef_test()$df_Satt` at atol=1e-10 (pinned in `tests/test_methodology_stacked_did.py`). **Bell-McCaffrey Satterthwaite DOF is threaded into the user-facing aggregated inference** (`event_study_effects[h]['p_value']`/`['conf_int']` use the per-event-time contrast DOF from `_compute_cr2_bm_contrast_dof`; `overall_p_value`/`overall_conf_int` use the post-period-average contrast DOF, matching R `Wald_test(constraints=row, vcov=CR2, test="HTZ")$df_denom` at atol=1e-10). Mirrors the SunAbraham aggregated-inference pattern from PR #472.
- `classical`, `hc2` — REJECTED at `__init__` with `ValueError`. StackedDiD clusters intrinsically at `'unit'` or `'unit_subexp'` (no `cluster=None` opt-out); the linalg validator rejects one-way families paired with `cluster_ids`. Use `vcov_type='hc1'` or `'hc2_bm'`.
- `conley` — REJECTED at `__init__` with `ValueError`. Deferred for a **methodology** reason, NOT plumbing (unlike the SunAbraham / WooldridgeDiD-OLS conley threading): the stacked design replicates each control unit across every sub-experiment it qualifies for (`_build_sub_experiment`), so one geographic unit occupies many stacked rows. Conley's pairwise distance matrix would see those same-unit copies at distance 0 (`K(0)=1`), conflating the stacking-replication device with real spatial correlation; a correct treatment needs a per-stack spatial identifier and has no `conleyreg` analogue to anchor parity. Paper-gated; tracked in `DEFERRED.md`.
- `survey_design=` + `vcov_type ∈ {classical, hc2, hc2_bm}` — REJECTED at `fit()` with `NotImplementedError`; the survey TSL or replicate-weight refit variance overrides the analytical sandwich. Use `vcov_type='hc1'` (default) for survey designs. Reject order: the existing fweight/aweight check at `stacked_did.py:309` fires first (Q-weight ratio semantics), then the survey + non-hc1 check at `stacked_did.py:~325` — locked by `test_aweight_plus_hc2_bm_rejected_by_stacked_did_level_guard`.

**Note:** This routing inherits the WLS-CR2 methodology choice from the Phase 1a clubSandwich port (PR #475 / REGISTRY Phase 1a `hc2_bm + weights` row); see that row for the full PT2018-§3.3-vs-clubSandwich algebra deviation derivation. No new methodology choice is introduced in this PR. The change is purely surface: switching from the prior bake-Q-into-X pattern (`X_t = X * sqrt(Q)`, `solve_ols(X_t, Y_t, cluster_ids=)`) to `solve_ols(weights=composed_weights, vcov_type=...)` opens the hc2_bm path without modifying the small-sample WLS-CR2 algebra. The HC1 path is preserved bit-equal (up to float64 multiplication ordering at machine precision).

**Note (tail df, 3.9 / M-127):** the `hc1` analytical t/p/CI previously used
silent normal-theory z on the non-survey lane (the tail-df defect family);
they now follow the library-wide `df_convention` knob — t(pooled residual df
`n_eff − k_kept`, positive-weight rows) under the default `"residual"`,
t(G − 1) under `"cluster"`, z under `"normal"` — with hc2_bm BM contrast DOF
and survey/replicate df keeping precedence. Two StackedDiD-specific caveats:
(a) the `"cluster"` G counts POSITIVE-WEIGHT clusters
(`effective_cluster_count`), while the reported `results.n_clusters` keeps
the raw unique count — the two diverge only when a cluster's total composed
weight is zero; (b) the stacked design replicates control units across
sub-experiments, so the pooled residual df is typically large and the
default's z → t movement is a convention alignment with near-zero numeric
effect — the materially moving step on this surface is the v4 flip to
`G − 1` (M-130). No public per-row df channel changes: `event_study_df`
records the resolved df per event time as before.

*Edge cases:*
- All events trimmed: `len(Omega_kappa) == 0` -> ValueError suggesting reduced kappa
- No clean controls for event a: IC2 check fails -> Trim event, warn user
- Single cohort in trimmed set: Valid — Q-weights simplify
- Duplicate observations: Same (unit, time) appears in multiple sub-experiments -> handled by clustering at unit level
- Constant treatment share across sub-exps: Unweighted FE recovers correct estimand (special case, Section 5.5)
- Anticipation > 0: Reference period shifts to e = -1 - anticipation. Post-treatment includes anticipation periods (e >= -anticipation). Window expands by anticipation pre-periods.
- Group aggregation: Not supported — pooled stacked regression cannot produce cohort-specific effects. Use CallawaySantAnna or ImputationDiD.

*Algorithm (Section 5):*
1. Choose kappa_pre, kappa_post event window
2. Apply IC1 (window fits in data) and IC2 (clean controls exist) to get Omega_kappa
3. For each a in Omega_kappa: build sub-experiment with treated (A_s = a), clean controls (A_s > a + kappa_post), time window [a - kappa_pre, a + kappa_post] (with anticipation: [a - kappa_pre - anticipation, a + kappa_post])
4. Stack all sub-experiments vertically
5. Compute Q-weights: aggregate weighting uses observation counts per (event_time, sub_exp), matching R reference. Population/sample_share use unit counts per sub_exp (paper notation).
6. Run WLS regression of Equation 3 with Q-weights
7. Extract delta_h coefficients as event-study ATTs
8. Compute cluster-robust SEs at unit level

*IC1 (Adoption Event Window, Section 3):*

    IC1_a = 1[a - kappa_pre >= T_min  AND  a + kappa_post <= T_max]

Note: Matches R reference implementation (`focalAdoptionTime - kappa_pre >= minTime`).
The reference period a-1 is included in the window [a-kappa_pre, a+kappa_post] when kappa_pre >= 1.
The paper text states a stricter bound (T_min + 1) but the R code by the co-author uses T_min.

*IC2 (Clean Controls Exist, Section 3):*

    IC2_a = 1[exists s with A_s > a + kappa_post]    (not_yet_treated)
    IC2_a = 1[exists s with A_s > a + kappa_post + kappa_pre]  (strict)
    IC2_a = 1[exists s with A_s = infinity]           (never_treated)

**Reference implementation(s):**
- R: https://github.com/hollina/stacked-did-weights (`create_sub_exp()`, `compute_weights()`)
- No Stata or Python package; Stata estimation via standard `reghdfe` with Q-weight column

**Requirements checklist:**
- [x] Sub-experiment construction with treated + clean controls + time window
- [x] IC1 and IC2 trimming with warnings
- [x] Q-weight computation for all three weighting schemes (Table 1)
- [x] WLS via sqrt(w) transformation
- [x] Event study regression (Equation 3) with reference period e=-1
- [x] Cluster-robust SEs at unit or unit x sub-exp level
- [x] Overall ATT as average of post-treatment delta_h with delta-method SE
- [x] Anticipation parameter support
- [x] Never-treated encoding (0 and inf)
- [x] Survey design support (Phase 3): Q-weights compose multiplicatively with survey weights; TSL vcov on composed weights; survey design columns propagated through sub-experiments. Replicate weights supported via estimator-level refit with Q-weight composition (see Replicate Weight Variance section).
- **Note (intercept SEs excluded from R parity):** for the base StackedDiD `clubSandwich` parity fixture (not the CBWSDID `cbwsdid` fixture below), stacked event-study *interaction* SEs match R at ~2e-13, but the regression *intercept* SE differs from R by a measured ~0.3% — a nuisance-parameter reference-cell/parameterization gap, not machine-precision lockable. Surfacing `se_cr1_intercept`/`se_cr2_intercept` would add an unasserted, R-divergent public field, so intercept SEs are deliberately not exposed and are excluded from the parity lock (`tests/test_methodology_stacked_did.py`; SE-audit C1/(c) decision, 2026-07-09).
- **Note:** Survey weights compose multiplicatively with Q-weights for StackedDiD; only `weight_type="pweight"` (default) is supported — `fweight` and `aweight` are rejected because Q-weight composition changes weight semantics (non-integer for fweight, non-inverse-variance for aweight)

### Covariate balancing (CBWSDID)

**Primary source:** Ustyuzhanin, V. (2026). Covariate-Balanced Weighted Stacked Difference-in-Differences. arXiv:2604.02293v1. https://arxiv.org/abs/2604.02293 (in-repo paper review: `docs/methodology/papers/ustyuzhanin-2026-review.md`). Within-sub-experiment refinement via **entropy balancing** (Hainmueller 2012).

Optional `balance="entropy"` (constructor) + `fit(..., covariates=[...])` adds a within-sub-experiment design stage that reweights the clean controls toward the treated cohort under **conditional** parallel trends, before the Wing et al. (2024) corrective aggregation.

*Design weights (Ustyuzhanin 2026 §3.1):*
For each sub-experiment `a`, entropy balancing produces nonnegative control weights `b_{sa}` matching the treated covariate means (covariates read at the last pre-treatment period `t = a-1-anticipation`; treated keep `b=1`). The final stacked weights compose `b_{sa}` with the corrective factor via the **effective control mass** `Ñ^C_a = Σ_{s∈C_a} b_{sa}`:

    W_{sa} = b_{sa} · (N^D_a / N^D_Ω) / (Ñ^C_a / Ñ^C_Ω)    for s ∈ C_a
    W_{sa} = 1                                            for s ∈ D_a

The pooled estimator is `DID^{CBWSDID}_e = Σ_a (N^D_a/N^D_Ω)(Δ̄^D_{a,e} − Δ̄^{C,b}_{a,e})`, recovered by the existing Q-weighted WLS when `W_{sa}` is injected. **Estimand preservation:** because only controls are reweighted (treated cohorts and their shares `N^D_a/N^D_Ω` are unchanged), the target remains the trimmed aggregate ATT `θ^e_κ` — the refinement changes only how untreated trends are estimated. At `b_{sa}=1` this reduces to the paper's **unit-count** weighted stacked DID, which equals `StackedDiD(weighting="aggregate")` on **balanced** event windows (where unit and observation counts coincide). Validated end-to-end by `tests/test_methodology_stacked_did.py::TestCBWSDIDCovariateBalance` (closed-form `DID^{CBWSDID}_e` anchor at 1e-8), `::TestCBWSDIDEffectiveMass` (effective-mass corrector is load-bearing vs a naive `b·Q` multiply), and `::TestCBWSDIDRParity` (cross-language parity vs the R `cbwsdid` package, `refinement.method="weightit"` / `method="ebal"`, at 1e-5 — golden in `benchmarks/data/cbwsdid_golden.json`, regenerate via `benchmarks/R/generate_cbwsdid_golden.R`).

- **Note:** The effective-mass `W_{sa}` is computed directly from cohort unit-counts + `Ñ^C_a` (a naive `b_{sa}·Q_aggregate` multiply is NOT equivalent — it aggregates control means with weights ∝ `(N^D_a/N^D_Ω)(Ñ^C_a/N^C_a)` instead of the required `∝ (N^D_a/N^D_Ω)`, biased unless `b` is uniform).
- **Note:** Provenance exposure (4.0 program row M-092, no numeric change): the event-time sub-block of the pooled-regression coefficient covariance is now persisted as `StackedDiDResults.event_study_vcov` (+ `event_study_vcov_index` = the estimated event times; the synthesized reference period is not a regression column and is absent). Persisted in every inference mode - analytical hc1/hc2_bm sandwich, survey replicate refit, and survey TSL all reassign the coefficient covariance the reported ES SEs are read from, so the marginal SEs are always this matrix's diagonal. `event_study_df` records, per event time, the df actually passed to `safe_inference` (the per-event Bell-McCaffrey Satterthwaite contrast df under `hc2_bm`, the scalar survey df under survey designs, NaN on the normal-theory hc1 path and on hc2_bm fail-closed rows). DiagnosticReport's parallel-trends check consumes the persisted VCV via its joint-Wald path for hc1/survey fits; `hc2_bm` fits are routed to Bonferroni over the BM-adjusted per-row p-values instead (the chi-square Wald reference would discard the CR2/BM small-sample correction; multi-constraint AHT/HTZ CR2 testing is tracked in DEFERRED.md - see REPORTING.md "hc2_bm parallel-trends policy").
- **Note:** Inference is conditional-on-the-estimated-weights cluster-robust (the existing `hc1`/`hc2_bm` path with `W_{sa}` as the WLS weights) — the paper's default. The paper's weight-re-estimating bootstrap is NOT implemented in v1 (deliberate scope; entropy balancing is smooth so the Abadie–Imbens (2008) nonsmooth-matching bootstrap caveat does not apply). `cluster` is orthogonal to `b_{sa}` (weights conditioned-on); default `unit` matches the paper.
- **Note:** v1 scope — only `balance="entropy"` with `weighting="aggregate"`. `balance` + `population`/`sample_share` and `balance` + `survey_design=` raise `NotImplementedError`; matching-based balancing and the repeated `0→1/1→0` episode extension are out of scope.
- **Note (post-fit aggregate() - row M-024):** `fit(aggregate=)` is deprecated (3.9; removed 4.0) and the event-study surface is ALWAYS materialized at fit — the pooled regression always included the event-time interactions, so only the extraction and, under `hc2_bm`, the per-event Bell-McCaffrey contrast DOFs were fit-time-gated; both gates are gone and the param is behaviorally inert (its `group`/`all`/unknown ValueErrors persist after the shim warning). `aggregate('event_study')` returns the unified `EventStudyResults` container (a pure view; no recomputation, so every stored inference mode — survey TSL, replicate refit, hc2_bm fail-close — relays faithfully); `aggregate('simple')` relays `overall_att/se/t/p/CI` bit-exactly with `target="att"` (the scalar is the equally-weighted post-period average, NOT the per-event-time trimmed aggregate ATT `θ^e_κ` — `describe_target_parameter` remains the estimand's prose source of truth), `n = n_treated_units` with `n_kind="units"` (TREATED-only scope: under dynamic clean-control definitions a later-treated unit is treated in its own sub-experiment and a clean control in earlier ones, so the treated and control unit sets OVERLAP and no disjoint total exists — cross-container `n` comparisons vs CallawaySantAnna's treated+control convention are out of contract), and `df = inference_df` (the stored overall inference's exact provenance). Consequences of the always-computed surface: plain-fit `summary()` renders the event-study table, `to_dataframe('event_study')` is total on 3.9+ fits, `plot_event_study` renders plain fits, DiagnosticReport's `heterogeneity` (always) and `parallel_trends` (`kappa_pre >= 2`) checks run on plain fits, and the `hc2_bm` overall-ATT contrast now shares the m>1 batch of `_compute_cr2_bm_contrast_dof`, whose degeneracy guard is batch-relative — the noise-floor carve-out on the `Wald_test(test="HTZ")$df_denom` parity claim in the variance-families bullets above widens quantitatively (unchanged behavior when tripped: fail-closed NaN inference plus linalg's loud noise-floor UserWarning; the absolute floor already made this reachable at m=1).
- **Note (container admission - rows M-024/M-093):** `compute_honest_did` and `compute_pretrends_power` admit Stacked-sourced `aggregate('event_study')` containers. Provenance: `base_period` is the class-level constant `"universal"` (every sub-experiment normalizes against the single omitted reference `e = -1 - anticipation`) and `reference_event_times` is the singleton `(-1 - anticipation,)`. Requirements: `kappa_pre >= 2` (the default `kappa_pre=1` grid minus the reference has ZERO estimated pre-periods — both consumers reject), and for HonestDiD a non-singular FULL retained event-study covariance — the validated sub-block spans the retained pre AND post rows (`allow_singular=False` per Rambachan-Roth's eigenvalues-bounded-away-from-zero assumption on the full Σ — keep `kappa_pre` small relative to the cluster count; PreTrendsPower keeps its documented singular support). The container's scalar `df_survey` is None on analytical fits — the generic resolver's fallback reads `df_inference`, a name StackedDiD does not carry (`inference_df` is the overall POST-AVERAGE contrast df, not a design-level cluster df, and misusing it for FLCI critical values would be worse than the asymptotic default) — so honest FLCI critical values are NORMAL-THEORY on analytical Stacked fits, diverging from bare-`cluster=` CS fits (which thread `G-1`); survey fits thread the survey df and replicate-undefined designs the fail-closed `0.0` sentinel (NaN FLCI CIs; the identified-set `lb`/`ub` bounds are df-independent and stay finite). Withheld-inference rows (finite `se`, non-finite `p_value` — the hc2_bm BM-DOF fail-close and replicate-undefined designs both emit this shape) are ADMITTED with a source-scoped warning in BOTH consumers: bounds/power consume only the point estimates and covariance, which remain valid; the warning is shape-descriptive because containers are publicly constructible. A rank-dropped interior event-time column (NaN-se row) fails closed via honest's consecutive-grid ValueError with a Stacked-specific remedy (no `balance_e` — Stacked's aggregate() has no such level). REFERENCE-SUPPORT GUARD: `fit()` raises when either omitted-reference cell (treated / control) carries no POSITIVE COMPOSED WLS WEIGHT — Equation 3 is Q-weighted, so raw row presence is not support: a gapped panel can empty the cell outright (IC1 checks only min/max periods), aggregate Q-weights zero a sub-experiment's controls at any event time where that sub-experiment has no treated rows, and survey pweights can zero a cell — without the guard the design rank-drops and QR pivoting silently re-normalizes against an arbitrary horizon while the surface certifies `delta_0 = 0` at the reference. Partial support (some cohorts' own ref period absent while the POOLED omitted category is populated) still fits: the coefficient-vector normalization is real; the per-cohort baseline raggedness is estimand composition, the documented ragged-window behavior.

*Covariate-balancing edge cases:*
- Infeasible cohort (treated covariate mean outside the clean-control hull → entropy balancing cannot match the moments): **fail-closed** `ValueError` naming the cohort and worst covariate — NOT silently dropped (dropping a cohort would shift the estimand to an overlap-trimmed ATT, Ustyuzhanin 2026 §3.1).
- Degenerate design weights (`Ñ^C_a → 0` / highly concentrated weights): low effective sample size → `UserWarning` with the per-cohort diagnostic.
- Missing pre-treatment row, or covariate absent / `balance`↔`covariates` mismatch: `ValueError` at `fit()`.
- **Ragged / unbalanced event windows** (a unit not observed at every event time in a sub-experiment): **fail-closed `ValueError`** — `balance="entropy"` requires balanced windows. The paper assumes balanced event windows; off them the unit-count corrector and the observation-count `aggregate` Q diverge (the count-convention is unresolved, deferred). `balance="none"` continues to support unbalanced panels via observation-count Q.

---

## WooldridgeDiD (ETWFE)

**Primary source:** Wooldridge, J. M. (2025). Two-way fixed effects, the two-way Mundlak regression, and difference-in-differences estimators. *Empirical Economics*, 69(5), 2545–2587. (Published version of the 2021 SSRN working paper NBER WP 29154.)

**Secondary source:** Wooldridge, J. M. (2023). Simple approaches to nonlinear difference-in-differences with panel data. *The Econometrics Journal*, 26(3), C31–C66. https://doi.org/10.1093/ectj/utad016

**Application reference:** Nagengast, A. J., Rios-Avila, F., & Yotov, Y. V. (2026). The European single market and intra-EU trade: an assessment with heterogeneity-robust difference-in-differences methods. *Economica*, 93(369), 298–331.

**Reference implementation:** Stata: `jwdid` package (Rios-Avila, 2021). R: `etwfe` package (McDermott, 2023).

**Key implementation requirements:**

*Core estimand:*

    ATT(g, t) = E[Y_it(g) - Y_it(0) | G_i = g, T = t]    for t >= g

where `g` is cohort (first treatment period), `t` is calendar time.

*OLS design matrix (Wooldridge 2025, Section 5):*

The saturated ETWFE regression includes:
1. Unit fixed effects (absorbed via within-transformation or as dummies)
2. Time fixed effects (absorbed or as dummies)
3. Cohort×time treatment interactions: `I(G_i = g) * I(T = t)` for each post-treatment (g, t) cell
4. Additional covariates X_it interacted with cohort×time indicators (optional)

The interaction coefficient `δ_{g,t}` identifies `ATT(g, t)` under parallel trends.
- **Note (reference-period normalization, W2025 Eq. 6.1/6.4 — `never_treated` OLS only):** this applies to the **lead-and-lag** specification, i.e. `control_group="never_treated"` on the OLS path, which is the only configuration that emits pre-treatment cells (`include_pre`). The default `not_yet_treated` (and both nonlinear paths) emit `t >= g − anticipation` cells only — the lag-only specification — so no cell is omitted from those designs and no placebo cells are produced; the reference period is still computed there, but solely to detect cohorts that have none (see the unidentified-cohort Note). For the lead-and-lag specification, for each treated cohort `g` the design OMITS one cell — the reference period — because the cohort's full cell block spans the cohort indicator `1{G_i = g}`, which the unit fixed effects absorb. The paper is explicit: Eq. 6.1 excludes `dg_i · f(g-1)_t` "so that `s = g − 1` is the reference period", and Eq. 6.4 describes the saturated regression as a collection of 2×2 DiDs using `g − 1` as reference. The library omits `ref(g) = max{t : t < g − anticipation AND cohort g is OBSERVED at t}` — `g − 1` on a balanced panel, and the cohort's own latest available pre-period otherwise. The support must be **per-cohort**: a panel-wide rule omits an identically-zero column whenever that cohort is unobserved at the panel's latest pre-period, leaving the collinearity intact. Anchored to Stata `jwdid ... never`, which omits the same cell for every cohort (`tests/test_etwfe_cs_stata_parity.py::TestNeverTreatedVsStataJwdid`). **Before v3.9 the reference was emitted and generic QR rank detection dropped an arbitrary column instead** — on `mpdta` that silently removed two genuine post-treatment effects, `(2004, 2004)` and `(2006, 2007)` (issue #724).
- **Note (reference-period sensitivity, W2025 Section 6.1):** the choice of reference is a NORMALIZATION, not an identifying assumption. The paper notes any set of pre-treatment periods may serve, and the pre-trend `t`-test is numerically identical whichever is used. `g − 1` is the paper's canonical choice and the one this library implements. **The point estimates are NOT invariant to it** — only the two-sided pre-trend test is (Section 6.1). Different references give different finite-sample 2×2 contrasts, all consistent for ATT(g, t) under parallel trends: on `mpdta`, ATT(2007, 2007) computed by hand against references 2006/2005/2004/2003 gives −0.0261 / −0.0571 / −0.0599 / −0.0294. This is exactly why issue #724 mattered — QR dropped `g2007_t2005`, silently making 2005 the reference, and the estimator returned −0.0571: a correct 2×2 DiD against the wrong baseline. It also means the unbalanced-panel fallback (the cohort's latest AVAILABLE pre-period) yields a different contrast from `g − 1`, which is legitimate but should be read as such.
- **Note (unidentified cohorts — library identification limit):** a cohort with NO observed period before `g − anticipation` has no reference cell, so none of its ATTs are identified. The library warns naming the cohort and **excludes its observations** from the estimation sample. *(Conditional since comparison-support filtering landed: when the filter removes that cohort's rows first, the cohort never reaches this path and is instead named by the zero-cell / fully-dropped warning above. The observations are excluded either way; only which warning names them differs.)* Excluding only its columns would be worse than the original bug: `_filter_sample` retains every treated row, so the cohort would join the omitted baseline beside the controls and load its treatment effect onto the time fixed effects — measured on `mpdta` at `anticipation=1`, that moved `ATT(2006, 2006)` by 0.0077 and `ATT(2006, 2007)` by 0.0055, past the 5e-3 this library treats as material. Exclusion can cascade (the removed rows may have been another cohort's not-yet-treated comparison), so the surviving sample is re-checked and `fit()` raises when no comparison observations remain rather than returning an all-NaN fit. This is a **library limit, not the paper's last-cohort result** — W2025's "the last cohort `T` plays the role of the never-treated group" is conditioned on there being no never-treated group at all, a different case.
- **Note (per-period comparison support, W2025 Section 5.4 — the cell half):** a period at which **no unit is untreated** carries no identified `ATT(g, t)`, because there is no untreated outcome to difference against. Such periods are removed from the estimation sample **before the solve**. The eligible set is keyed on the regression baseline, not on the method: on the lead-and-lag branch (`never_treated` + OLS) only never-treated rows qualify, because every `(g, t)` except each cohort's reference is emitted, so a later cohort's rows sit in their own indicator; elsewhere not-yet-treated rows qualify too. **The omitted reference cells do NOT count as support** even though they sit in the baseline: cohort `h`'s indicator is absorbed by the unit FE (`1{h,t} = D_h − Σ_{t'≠t} h_{t'}`), so the period dummy stays reproducible from the emitted cells and the design remains collinear — measured on a panel with never-treated units through `t=4` and cohorts 3, 6 through `t=6`, retaining `t=5` on the strength of `ref(6)=5` raises on the lost `(6,1)`, while dropping it fits at `overall_att = 1.0171`. On an all-eventually-treated panel this yields exactly Eq. 5.15's cell set: cohort `G_max` is the reference and receives nothing. Anchored to Stata `jwdid` (`tests/test_etwfe_cs_stata_parity.py::TestAllEventuallyTreatedVsStataJwdid`): identical cell set, identical `N` (764 of 955), ATTs agreeing to ~1e-15.
- **Note (the reduction is always reported — deviation from `jwdid`):** Stata `jwdid` performs the same reduction **silently**, reporting only a smaller `N`. This library warns on every fit that drops rows, naming the periods, the observation count, and the branch-correct cause, and separately naming any cohort left with no estimated cells or stripped of every row. Neither warning is gated on `rank_deficient_action` — that setting governs how rank warnings surface, not whether the estimation sample changed. Being explicit is a deliberate improvement, not a numerical deviation: the estimates are identical.
- **Note (reference movement under filtering):** on the `not_yet_treated` branch a reference period can never be filtered — reference eligibility (`t < g − anticipation`) IS the support predicate's own second disjunct evaluated at that cohort's rows — and the estimator asserts this at runtime, raising if it is ever violated (silent renormalization is the issue #724 defect class). On the `never_treated` + OLS branch it CAN move, because eligibility there does not imply never-treated presence: measured, dropping `t=5` moves `ref(6)` from 5 to 4. That is legitimate — the ATTs stay correctly labelled and validly identified, just normalized against a different baseline period — so it **warns** naming the cohort and both periods rather than raising. A reference that becomes `None` is not a move: that cohort is unidentified and the exclusion path already warns by name.
- **Note (zero-cell cohorts are warned, not raised):** a cohort retained only as a control — cohort `G_max` under the Section 5.4 normalization — or one that loses every row to filtering produces no `ATT(g, t)`. `fit()` reports it by name and continues. `results.groups` and `_n_g_per_cohort` are derived from the **emitted cell set** so such a cohort is not advertised as estimated (`set(results.groups) == {g for (g, t) in group_time_effects}`). `cohort_trend_coefs` is keyed on PRESENT cohorts instead, because the trend baseline is exactly the zero-cell cohort; the two therefore differ by design, and the invariant on that dict is `set(cohort_trend_coefs) ⊆ present cohorts`.
- **Note (`aggregate(weights="cohort_share")` is REFUSED when filtering removed units):** `_n_g_per_cohort` — `N_g` in W2025 Eqs. 7.4/7.6 — is read off the FINAL sample. Comparison-support filtering is the first thing in this estimator that can remove SOME units of a RETAINED cohort: `_filter_sample` keeps every row of every treated unit, and unidentified-cohort exclusion removes whole cohorts, so before this the count was always the full supplied cohort. On an **unbalanced** panel a unit observed only at unsupported periods vanishes with them, and `N_g` then describes a strictly smaller cohort than the user supplied — measured on a cohort supplied with 100 units of which 90 appear only at a dropped period: `aggregate(weights="cohort_share")` moves from 1.8078 to 3.8157. W2025 Section 7 defines `N_g` on a balanced panel and does not say whether it counts the supplied cohort or the surviving units; the two disagree materially, so `aggregate` **fails closed** naming the cohorts and unit counts rather than choosing silently (mirroring the survey + `cohort_share` refusal). `weights="cell"` never reads `N_g` and is unaffected, and the refusal cannot fire on a **balanced** panel — filtering there removes whole periods and no units — so the all-eventually-treated capability is untouched. Defining the unbalanced estimand is tracked in `TODO.md`.
- **Note (comparison-support filtering is REFUSED under `survey_design=`):** a second survey boundary alongside the unidentified-cohort refusal below, and for the same reason — deleting rows removes their PSUs and strata from the Taylor-linearized meat and from `df_survey = n_PSU − n_strata`. Raised **before any row is removed**, and strictly conditional on the filter actually dropping rows (survey fits that lose nothing are unaffected). The message points at **explicit frame restriction**, not `SurveyDesign.subpopulation()`: subpopulation zero-pads the excluded rows, which `_reject_zero_weight_groups` then refuses on the OLS path (measured: `Survey weights sum to zero for time period(s) [8, 9]`). **That workaround is exact only when every PSU and stratum survives the restriction**, which holds on a BALANCED panel — measured there: no PSU and no stratum is removed, `df_survey` is unchanged, and ATTs/SEs agree to ≤5e-15. It does NOT hold in general: on an unbalanced panel a PSU or stratum observed only at unsupported periods disappears with them (measured: restricting a 6-period frame to its 5 supported periods dropped one PSU and one stratum entirely), which changes the Taylor-linearized meat and can change `df_survey`. Before relying on the restriction, confirm the restricted frame retains every PSU and stratum of the design you specified and is the survey universe you intend; if it does not, there is no supported path until domain estimation lands.
- **Note (control-pool asymmetry on `never_treated`):** `control_group="never_treated"` restricts the comparison pool to never-treated units **on the OLS path only**, where `include_pre` emits every `(g, t)` except each cohort's reference and treated units' pre-treatment rows therefore sit in their own indicators. On `method="logit"` / `"poisson"` only post-treatment cells are emitted — including all cells would make each cohort dummy collinear with the sum of its own indicators — so treated units' pre-treatment rows ARE the identifying comparison there, exactly as under `not_yet_treated`. Measured at HEAD on `never_treated` + Poisson with never-treated units observed only at `t=1..3` and cohorts 4, 7 at `t=1..5`: the fit succeeds and perturbing only cohort-7 rows at `t∈{4,5}` moves `ATT(4,4)` and `ATT(4,5)` materially, while the OLS counterpart on a panel where that fit is defined is invariant to ~1e-15. This asymmetry is **pre-existing and structural**, not introduced by the support filter. Related: `n_control_units` counts never-treated UNITS on this setting regardless of method, so on the nonlinear paths it under-reports the rows actually doing the comparison — widening it is tracked in `TODO.md`.
- **Note:** unobserved `(g, t)` pairs are skipped rather than emitted as identically-zero columns, and the skipped pairs are named in a `UserWarning` (gated on `rank_deficient_action`, so `"silent"` remains silent).
- **Note (within-cohort support connectivity — OLS/unit-FE path only):** per-period support is **necessary but not sufficient** for identification. Identification runs through the unit fixed effects, so each emitted `(g, t)` cell must be connected to an **unemitted** period in the bipartite graph whose nodes are that cohort's units and periods (edge = a supported observation). Within any connected component in which *every* period is a treatment cell, those columns sum to the component's unit indicators — absorbed by the unit FE — so the block is rank-deficient and QR drops one, potentially a genuine post-treatment effect, leaving the overall ATT an average over an incomplete cell set. This is issue #724's failure mode reached through **unit** support rather than through the reference period, and per-period observation counts cannot detect it. Measured on a panel where cohort 4's units split into `{2, 4}` and `{1, 5}` groups: `g4_t5` was silently dropped and `overall_att` computed from the single surviving post cell. `fit()` now **fails closed** naming the unidentified cells (not gated on `rank_deficient_action` — that setting governs how rank warnings surface, not whether an unidentified estimand may be returned as a number). On `not_yet_treated` and the nonlinear paths pre-treatment periods are not emitted, so a component holding any pre-period is safe — which is why the condition is "contains an unemitted period", not "contains the reference". The condition applies **only to the OLS path**, whose within-transformation absorbs the unit fixed effects: `logit`/`poisson` use explicit cohort + time dummies, nothing absorbs a component's cell block, and such designs are full rank — so the check is gated on `method="ols"` and those paths estimate the same panel normally. Component-aware estimation (rather than refusal) is tracked in `TODO.md`.
- **Note (unidentified-cohort exclusion is REFUSED under `survey_design=`):** the row exclusion above is naive subsetting, which under a complex survey design is not domain estimation — it removes the excluded rows' PSUs and strata from the Taylor-linearized meat and from `df_survey = n_PSU − n_strata`, so surviving ATTs would be reported with variance computed on a design the user never specified (measured on a two-stratum panel: `df_survey` 22 → 14, finite SE, no diagnostic). *Subpopulation Analysis (Phase 6)* below and Lumley (2004) §3.4 require the opposite — zero the excluded rows' weights and RETAIN the strata/PSU layout — and `SurveyDesign.subpopulation()` implements that contract, but composing it here additionally requires the weighted within-transformation to tolerate zero-weight units (shared machinery). Until that lands, `fit()` raises `NotImplementedError` when `survey_design=` is combined with an unidentified cohort, on all three methods, rather than returning a confident number from the wrong design (tracked in `TODO.md`). Relatedly, the **full** `SurveyDesign` is resolved against the **pre-exclusion** sample for validation only (weights, `weight_type`, strata, PSU, FPC, nesting): every check that ran after exclusion was blind to whatever the deleted rows contained, so invalid metadata confined to an excluded cohort used to vanish silently.
- **Note (finite-overall-ATT invariant, ledger `M-124`):** a successful `fit()` reports a **finite** `overall_att` or raises explaining why it cannot, so a result object never presents a NaN point estimate as a completed fit. The invariant covers the **point estimate only**. `overall_se` and the inference fields may legitimately be NaN on a successful fit where the *variance* is unidentifiable while the ATT is not — Bell-McCaffrey DOF unavailable on the `hc2_bm` path, and `aggregate(weights="cohort_share")`, both of which fail closed to NaN through `safe_inference()` by design (see those entries). `_require_estimable_overall_att` accordingly checks `overall_att` and nothing else. Enforced at two levels because neither subsumes the other: at BUILD time (is any post-treatment cell estimable, `t >= g − anticipation`) and again POST-SOLVE in `_require_estimable_overall_att`, before every results construction. The build-time notion of a treatment cell is deliberately **wider** than the aggregation's — cells in the anticipation window (`g − anticipation <= t < g`) are estimated and reported per-cell, but excluded from the overall average per W2025 — so a cohort never observed once actually treated produces cells yet no overall estimand. Three causes are distinguished by message: no cell survived the solve (collinear design); cells survived but none has `t >= g`; post-treatment cells survived with NaN coefficients (rank-dropped). **Cohort count is not a proxy for comparison support** — a design with two surviving treated cohorts can still be fully degenerate when neither post cell has a same-period comparison unit, so the check is on surviving finite coefficients, not on how many cohorts remain.
- **Note:** OLS path uses iterative alternating-projection within-transformation (uniform weights) for exact FE absorption on both balanced and unbalanced panels. One-pass demeaning (`y - ȳ_i - ȳ_t + ȳ`) is only exact for balanced panels.
- **Note:** The weighted within-transformation (`utils.within_transform` with `weights`) is invoked on every WooldridgeDiD fit (survey weights when provided, `np.ones` otherwise) and emits a `UserWarning` on non-convergence per the shared convention documented under *Absorbed Fixed Effects with Survey Weights*.
- **Note:** NaN values in the `cohort` column are filled with 0 (treated as never-treated), both in `_filter_sample` and in `fit()`. This recategorization now emits a `UserWarning` reporting the affected row count so it is no longer silent (axis-E silent coercion under the Phase 2 audit). Pass `0` directly for never-treated units to avoid the warning.

*Nonlinear extensions (Wooldridge 2023):*

For binary outcomes (logit) and count outcomes (Poisson), Wooldridge (2023) provides an
Average Structural Function (ASF) approach. For each treated cell (g, t):

    ATT(g, t) = mean_i[g(η_i + δ_{g,t}) - g(η_i)]   over units i in cell (g, t)

where `g(·)` is the link inverse (logistic or exp), `η_i` is the individual linear predictor
(fixed effects + controls), and `δ_{g,t}` is the interaction coefficient from the nonlinear model.

- **Note (outcome-fit hint):** When `method="ols"` is used on a **binary** (`{0,1}`) or a **non-negative integer count** outcome, `WooldridgeDiD.fit()` emits a `UserWarning` noting that a matching nonlinear model (`method="logit"` / `method="poisson"`) is often the **more appropriate specification** for such outcomes. The framing is the paper's, not an efficiency heuristic: the nonlinear paths impose parallel trends on the **link/index scale** (logit index / Poisson log-mean) rather than in **levels**, and Wooldridge (2023) states the **linear-PT assumption is only valid for continuous/unbounded outcomes** (Eq. 2.5 vs the Index-PT Eq. 2.6–2.7). In the paper's **Section 5 simulations** the linear model is both **biased** (POLS −0.15 to −0.29 for binary; OLS >30%, >50% in places, for counts) **and less precise** (nonlinear SEs ~30–70% smaller) where the nonlinear mean holds, and the paper notes pre-trends tests often **fail to detect** the linear model's misspecification. This is a **different identifying assumption** than linear OLS — which one fits depends on which parallel-trends restriction holds — so the warning frames the nonlinear model as a **recommended comparison, not an automatic switch or a free upgrade**. Per **Table 1**, Gaussian/OLS remains a valid QMLE for *any* response, so the hint never asserts OLS is mechanically wrong. **Framing provenance (two-sided):** the earlier "canonical link requirement tied to Prop 3.1" reading was **incorrect** (corrected in **PR #453 R1**); the warning is equally careful *not* to over-correct into framing the nonlinear path as a pure efficiency add-on, since the paper's case is primarily about **bias / appropriateness** (codex R1, reconciled against `docs/methodology/papers/wooldridge-2023-review.md` §5). Detection is deliberately high-signal — binary requires support exactly `{0,1}`; the count branch suggests Poisson for non-negative integers with `>2` distinct values (Poisson is the natural *unbounded* nonnegative/count model per Table 1; bounded-support / **binomial** outcomes with a known upper bound are not separately detected — that routing would need user-supplied metadata). Fractional / bounded-continuous outcomes are **not flagged by this heuristic** (fractional responses are themselves a logistic/Bernoulli case in Table 1; the heuristic simply does not attempt to detect them — a coverage limit, not a claim that OLS is preferred for such outcomes). **Detection sample:** the hint reads the *full* outcome column passed to `fit()` — outcome *type* is a property of the variable, so the named column is the right signal — and it is read *before* any row filtering. `WooldridgeDiD._filter_sample` still expresses the control-group choice through the design matrix (`_build_interaction_matrix`), **not** by dropping rows (invariant pinned by `tests/test_wooldridge.py::TestOutcomeFitHint::test_filter_sample_preserves_outcome_support`), but as of 3.9 the full column and the fitted sample **no longer always share the same outcome support**: a cohort with no ETWFE reference period has its observations excluded downstream (see the unidentified-cohorts Note above and ledger row `M-123`). Classifying on the variable remains the correct choice precisely because of that — a count is a count regardless of which rows survive identification, so the hint must not flip on a sample reduction. Always-on (suppress via `warnings.filterwarnings`), implemented as the pure helper `_suggest_nonlinear_method` + the OLS-only fit-time gate `0g`; it never alters the fit or raises.

*Standard errors:*
- OLS: Cluster-robust sandwich estimator at the unit level (default)
- Logit/Poisson: QMLE sandwich `(X'WX)^{-1} meat (X'WX)^{-1}` via `compute_robust_vcov(..., weights=w, weight_type="aweight")` where `w = p_i(1-p_i)` for logit or `w = μ_i` for Poisson
- Delta-method SEs for ATT(g,t) from nonlinear models: `Var(ATT) = ∇θ' Σ_β ∇θ`
- Joint delta method for overall ATT: `agg_grad = Σ_k (w_k/w_total) * ∇θ_k`
- **Deviation from R:** R's `etwfe` package uses `fixest` for nonlinear paths; this implementation uses direct QMLE via `compute_robust_vcov` to avoid a statsmodels/fixest dependency.
- **Note:** QMLE sandwich uses `weight_type="aweight"` which applies `(G/(G-1)) * ((n-1)/(n-k))` small-sample adjustment. Stata `jwdid` uses `G/(G-1)` only. The `(n-1)/(n-k)` term is conservative (inflates SEs slightly). For typical ETWFE panels where n >> k, the difference is negligible.

*Variance families (`vcov_type`, OLS path only):*
- `hc1` (default) — CR1 Liang-Zeger cluster-robust on the within-transformed design. Point estimates, residuals, and the unscaled cluster-score meat are unchanged by the within-transform (FWL preserves the score); the CR1 finite-sample scalar uses `K_reference` (3.9). The natural R anchor is `fixest::feols(y ~ <interactions> | unit + time, cluster=~unit)` or Stata `jwdid` (both within-transform). **Parity with Stata `jwdid` (K_reference convergence, 3.9; `tests/test_etwfe_cs_stata_parity.py`):** the ATT(g,t) POINT estimates match `jwdid` exactly (~3e-8 on the `mpdta` panel, i.e. Stata's log-output rounding), and the `hc1` SEs now match `jwdid`/reghdfe at machine precision (ratio 1.0, spreads ~1e-15..1e-14) on all three committed arms AND at every rung of the committed G≈20..500 subsample ladder (`etwfe_cs_stata_golden.json` `ladder` block). Historically the SEs were uniformly SMALLER by a cluster-count-dependent factor (1.0280 at G=20 down to 1.0010 at G=500) — defect D2 (`docs/methodology/variance-conventions.md`): the clustered CR1 factor's `k` counted only the visible treatment-cell columns, omitting the absorbed FE not nested in the cluster. The applied correction is the closed form `K_reference = explicit columns + (1 if no intercept column) + rank(non-nested FE | nested)` (here `cells + T`), threaded as `cluster_k_adjustment`; the ladder doubles as the K-accounting probe (reghdfe `df_a == absorbed_fe_cr1_k_increment − 1` per rung). This is distinct from the `lm + clubSandwich` deviation below, whose factor is `k`-based. `CallawaySantAnna` shows no such gap - its SEs match Stata `csdid` outright - which localizes this to the ETWFE path rather than a library-wide convention. **Deviation from R `lm + clubSandwich::vcovCR(type="CR1S")`:** the full-dummy `lm` SE differs by a factor of `sqrt((n - K_reference) / (n - k_total))` because clubSandwich's `(n-1)/(n-p)` finite-sample correction counts ALL columns (intercept + treatment + unit dummies + time dummies = `k_total`) while WooldridgeDiD's within-transform CR1 counts `K_reference` (cells + constant + the non-nested time-FE rank; the cluster-NESTED unit dummies are the remaining difference). On the 240-obs / 51-column R-parity fixture (`k_total = 51`, `K_reference = 12`) this is `sqrt(228/189)` ≈ 10% (was ~11% under the historical cells-only count); on typical larger panels (n >> k_total) the gap shrinks to <2%. No public WooldridgeDiD code path exposes the `lm + CR1S` (CR1 cluster-robust on the full-dummy design) finite-sample correction — `vcov_type="hc2_bm"` routes to the CR2 Bell-McCaffrey sandwich on the full-dummy design (different variance estimator entirely), not CR1S. Users who need exact `lm + clubSandwich::vcovCR(type="CR1S")` parity must call `solve_ols` directly on a full-dummy design or fit via R. Same deviation pattern as SunAbraham PR #472 (`fixest::sunab` vs `lm + clubSandwich`).
- `hc2_bm` — CR2 Bell-McCaffrey via auto-route to full-dummy design (`[intercept, X_design, unit_dummies, time_dummies]`), then `solve_ols(..., vcov_type="hc2_bm")` through the clubSandwich port (PR #475). FWL does NOT preserve the hat matrix; HC2 leverage + BM DOF require the full-projection design. Per-coefficient SE matches `clubSandwich::vcovCR(lm(...), cluster=~unit, type="CR2")` at atol=1e-10. Per-cell `(g, t)` inference fields use `coef_test()$df_Satt` Bell-McCaffrey DOF (pinned at atol=1e-6 from CI half-width inversion). Aggregated inference (overall ATT + `.aggregate("group" | "calendar" | "event_study")`) uses contrast-specific BM DOFs from `_compute_cr2_bm_contrast_dof` (matches R `Wald_test(constraints=matrix(w, 1), vcov=vcov_CR2, test="HTZ")$df_denom`); the overall ATT contrast DOF is computed at fit time, the other three aggregations lazily on each `.aggregate(...)` call from BM artifacts (the REDUCED kept-column `X` / `cluster_ids` / bread matrix + the reduced-space coef-index map) stored on the Results object — using the reduced design after rank-deficient drops keeps the bread non-singular and matches the subspace `solve_ols` actually estimated in. Fail-closed across all surfaces: when BM DOF is unavailable (helper raises or returns non-finite), the affected inference fields are NaN — not normal-theory fallback (per `feedback_bm_contrast_dof_fail_closed`).
- `classical`, `hc2` — supported via auto-route to full-dummy AND auto-drop of the unit auto-cluster (one-way families don't compose with `cluster_ids` per the linalg validator). Set `self.cluster=None` (default) for these; explicit `cluster="state"` + one-way family raises at the linalg validator. SE matches `summary(lm(...))$coefficients` (classical) and `sandwich::vcovHC(type="HC2")` respectively. Per-cell + aggregate p-values/CIs use the residual DOF `n - rank(X)` (matches R `lm()` / `coef_test()` t-distribution under both classical OLS SE and `sandwich::vcovHC` defaults) — not normal-theory, so inference is correct under small samples.
- `conley` (spatial-HAC, Conley 1999) — supported on the **OLS path** via the within-transform design (or the full-dummy design when `cohort_trends=True`, like the other full-dummy families — see the cohort-trends row below), threading the `conley_*` params through `solve_ols` / `conley.py` (`conley_lag_cutoff=0` = within-period spatial only; `>0` adds within-unit Bartlett serial — the panel-aware path, since `conley_time`/`conley_unit` are always supplied, not pooled cross-sectional). Reuses the already-`conleyreg`-validated machinery (no new variance code). The unit auto-cluster is dropped on the conley path (an explicit `cluster=` enables the spatial+cluster product kernel); `survey_design=` / `weights` / `n_bootstrap>0` are rejected, and `method ∈ {logit, poisson}` + conley remains rejected (the `method != "ols"` guard — a QMLE-on-pseudo-residuals Conley sandwich is a separate derivation). FWL-composability (the within-transform conley SE equals the full-dummy conley SE) is pinned in `tests/test_conley_vcov.py::TestConleyWooldridge::test_fwl_composability_vs_full_dummy`.
- `method ∈ {"logit","poisson"}` + `vcov_type != "hc1"` — REJECTED at `__init__`. GLM QMLE sandwich with HC2 leverage on canonical-link pseudo-residuals (`w = p(1-p)` for logit, `w = μ_i` for Poisson) needs CR2-BM-on-GLM derivation + R parity against `clubSandwich::vcovCR(glm(...))`. Tracked in DEFERRED.md (WooldridgeDiD logit/poisson follow-up row).
- `survey_design=` + `vcov_type != "hc1"` — REJECTED at `fit()` with `NotImplementedError`. Survey TSL/replicate-refit overrides analytical sandwich. Use `vcov_type="hc1"` (default) for survey designs.
- `n_bootstrap > 0` + `vcov_type ∈ {"hc2","classical"}` — REJECTED at `fit()` regardless of `self.cluster` setting. The multiplier bootstrap is intrinsically clustered, but one-way vcov_type does not compose with `cluster_ids`: with `cluster=None` the auto-cluster is dropped (bootstrap has no cluster to draw weights at); with `cluster=X` the linalg validator rejects one-way + cluster_ids downstream with a less-informative error. User must drop bootstrap (`n_bootstrap=0`) or pick a cluster-compatible `vcov_type` (`hc1` or `hc2_bm`).
- **Note:** This routing is a documented synthesis of two existing methodology ingredients: the full-dummy auto-route from the Phase 1b PR 1/8 SunAbraham pattern (PR #472, which itself reused the Phase 1a Gate 1 TWFE lift from PR #469), and the clubSandwich WLS-CR2 algebra from the Phase 1a port (PR #475). The BM contrast DOF threading reuses `_compute_cr2_bm_contrast_dof` from PR #465 (MPD). No new methodology choice is introduced — the change is purely surface: extending the existing pattern from SA-OLS to WooldridgeDiD-OLS.
- **Note:** Bootstrap is supported only with `vcov_type ∈ {"hc1","hc2_bm"}` (one-way `classical`/`hc2` + bootstrap is rejected at `fit()` per the previous bullet). On the supported paths, the bootstrap clusters at `self.cluster if self.cluster else unit` — i.e., it matches the user's explicit cluster column if set, falling back to unit otherwise (the panel's natural unit of variation). The bootstrap SE overrides the analytical SE for `overall_*` on `n_bootstrap > 0` paths; per-cell `(g, t)` SEs still come from the analytical vcov.
- **Note (tail df, 3.9 / M-127):** every non-BM OLS arm's per-cell and
  aggregated t/p/CI now consume ONE `df_convention`-resolved fallback df —
  the default-`hc1` arms (within-transform AND `hc1`+`cohort_trends`
  full-dummy) previously used silent normal-theory z (the tail-df defect
  family), and now use t(residual df): `n − k_kept − rank(absorbed
  [unit, time])` on the within arm, the historical `n − rank(X)` on the
  full-dummy arms (`classical`/`hc2` values bit-identical). `"cluster"`
  gives t(G − 1) on hc1-clustered fits (inert on one-way/conley families);
  `"normal"` gives z. Survey design df and hc2_bm BM DOF keep precedence;
  the logit/poisson arms are knob-INDEPENDENT (survey df when present,
  normal theory otherwise — an explicitly non-default `df_convention`
  warns at fit time). Post-fit `aggregate()` reproduces fit-time inference
  on every OLS arm via the stored resolved fallback
  (`_df_analytic_fallback`, live on bootstrap fits whose per-cell inference
  stays analytical). No public per-row df channel ships in 3.9 — the
  unified event-study surface's Wooldridge df column stays NaN (a per-key
  dict that could also carry the hc2_bm BM contrast dofs is the TODO.md
  df-provenance follow-up).

*Aggregations (matching `jwdid_estat`):*
- `simple`: Weighted average across all post-treatment (g, t) cells. Default
  `weights="cell"` uses cell-count `n_{g,t}`:

      ATT_overall = Σ_{(g,t): t≥g} n_{g,t} · ATT(g,t) / Σ_{(g,t): t≥g} n_{g,t}

  Cell weight `n_{g,t}` = count of obs in cohort g at time t in estimation sample.
  - **Note:** `aggregate(type="simple", weights="cell")` (default) matches Stata `jwdid_estat` behavior. The opt-in `weights="cohort_share"` exposes the paper W2025 Eq. 7.4 cohort-share form `ω̂_g ∝ N_g`; the two coincide on balanced panels with uniform within-cohort cell counts (paper Section 7.5 footnote). The cohort-share path raises `ValueError` for `type="group"` and `type="calendar"` (no paper formula). See `docs/methodology/papers/wooldridge-2025-review.md` § Section 7 for derivation.
  - **Note:** `weights="cohort_share"` inference contract: the analytical SE returned under cohort-share aggregation is **conditional on the observed cohort shares ω̂_g / ω̂_{ge}**, treating them as fixed. Per paper W2025 Section 7.5, unconditional inference should also account for sampling uncertainty in the cohort shares themselves (the `N_g` are random in a stochastic sample). The library **fail-closes** the t-stat / p-value / conf-int fields to NaN under `weights="cohort_share"` and emits a `UserWarning` documenting the limitation; the point estimate and the conditional-on-shares SE are still computed and returned for reference. Proper APE/GMM-style aggregate inference (Wooldridge 2023 Section 4 framework) is tracked as a deferred follow-up in DEFERRED.md.
  - **Note:** `weights="cohort_share"` is **NOT supported on survey-weighted fits** (raises `ValueError` when `survey_design` is supplied). The library populates `_n_g_per_cohort` from `unit.nunique()` (raw counts); composing these with the design-weighted ATTs would target a mixed estimand inconsistent with paper W2025 Section 7's design-population cohort-share form. Design-consistent cohort totals (survey-weighted unit totals per cohort) are tracked as a deferred follow-up in DEFERRED.md.

- `group`: Weighted average across t for each cohort g (cell-count weights only — paper W2025 has no closed-form cohort-share weights for this aggregation type)
- `calendar`: Weighted average across g for each calendar time t (cell-count weights only)
- `event_study`: Weighted average across (g, t) cells by relative period k = t - g. Default `weights="cell"`; opt-in `weights="cohort_share"` exposes paper Eq. 7.6 cohort-share-by-exposure form `ω̂_{ge} ∝ N_g` with per-event-time normalization across cohorts present at event-time `e`. The cohort-share event path is **restricted to `k >= 0`** (post-treatment exposure times only); paper Eq. 7.6 is defined for post-treatment exposure, and pre-treatment leads use a separate Eq. 7.7 `nw_it`-based construction not yet exposed in the library. Under the default `weights="cell"`, negative-`k` placebo cells (e.g., from OLS + `control_group="never_treated"` or `anticipation > 0`) remain in the event aggregation for the placebo-test use case.

*Covariates:*
- `exovar`: Time-invariant covariates, added without demeaning (corresponds to W2025 Eq. 5.2 `x_i`)
- `xtvar`: Time-varying covariates, demeaned within cohort×period cells when `demean_covariates=True` (corresponds to W2025 Eq. 10.2 `x_hat_itgs = x_it - x_bar_gs`)
- `xgvar`: Covariates interacted with each cohort indicator
- **Note:** Covariate-adjusted ETWFE includes the full W2025 Eq. 5.3 basis: raw X, cohort × X (D_g × X for treated cohorts, auto-generated for `exovar`/`xtvar`), time × X (f_t × X, drop first period), and cell × demeaned X (D_{g,t} × X̃). Variables in `xgvar` already contribute D_g × X via `_prepare_covariates`; `exovar`/`xtvar` get automatic D_g × X generation.
- **Note:** `xtvar` demeaning operates at the cohort×period level (W2025 Eq. 10.2), not the cohort level (W2025 Eq. 5.2). These are identical for time-constant covariates but differ for time-varying covariates.

*Control groups:*
- `not_yet_treated` (default): Control pool includes units not yet treated at time t (same as Callaway-Sant'Anna)
- `never_treated`: Control pool restricted to never-treated units **on the OLS path only**. See the control-pool asymmetry Note below — on the nonlinear paths this setting does NOT restrict the comparison to never-treated units.

*Edge cases:*
- Single cohort (no staggered adoption): Reduces to standard 2×2 DiD
- Missing cohorts: Only cohorts observed in the data are included in interactions
- Anticipation: When `anticipation > 0`, interactions include periods `t >= g - anticipation`
- **Note:** Aggregation (simple/group/calendar) uses `t >= g` as the post-treatment threshold regardless of `anticipation`. Anticipation-window cells (g - anticipation <= t < g) are estimated but treated as pre-treatment placebos in aggregation, not included in overall ATT. This matches the standard post-treatment ATT definition; users who want anticipation cells in the aggregate should compute custom weighted averages from `group_time_effects`.
- Never-treated control only: Pre-treatment periods still estimable as placebo ATTs
- **Note:** Poisson QMLE with cohort+time dummies (not unit dummies) is consistent even in short panels (Wooldridge 1999, JBES). The exponential mean function is unique in that incidental parameters from group dummies do not cause inconsistency.
- **Note:** Logit path uses cohort×time additive dummies (not unit dummies) to avoid incidental parameters bias — a standard limitation of logit FE in short panels. This matches Stata `jwdid method(logit)` which uses `i.gvar i.tvar`.
- **Note:** Nonlinear methods (logit, Poisson) with `control_group="never_treated"` restrict the interaction matrix to post-treatment cells only. Pre-treatment placebo cells are OLS-only (where within-transformation absorbs FE). Including all (g,t) cells in the nonlinear design creates exact collinearity between cohort dummies and cell indicator sums. The OLS path resolves this by omitting each cohort's `ref(g)` reference cell explicitly (W2025 Eq. 6.1/6.4); the nonlinear paths avoid it by restricting to post-treatment cells instead, since their explicit cohort + time dummies (no within-transformation) make the reference omission insufficient on its own.

*Algorithm:*
1. Identify cohorts G and time periods T from data
2. Build within-transformed design matrix (absorb unit + time FE)
3. Append cohort×time interaction columns for all post-treatment cells
4. Fit OLS/logit/Poisson
5. For nonlinear: compute ASF-based ATT(g,t) and delta-method SEs per cell
6. For OLS: extract δ_{g,t} coefficients directly as ATT(g,t)
7. Compute overall ATT as weighted average; store full vcov for aggregate SEs
8. Optionally run multiplier bootstrap for overall SE

**Requirements checklist:**
- [x] Saturated cohort×time interaction design matrix
- [x] Unit + time FE absorption (within-transformation)
- [x] OLS, logit (IRLS), and Poisson (IRLS) fitting methods
- [x] Cluster-robust SEs at unit level for all methods
- [x] ASF-based ATT for nonlinear methods with delta-method SEs
- [x] Joint delta-method SE for aggregate ATT in nonlinear models
- [x] Four aggregation types: simple, group, calendar, event_study
- [x] Both control groups: not_yet_treated, never_treated
- [x] Anticipation parameter support
- [x] Multiplier bootstrap (Rademacher/Webb/Mammen) for OLS overall SE
- [x] Survey design support (strata/PSU/FPC with TSL variance)

**Survey design notes:**
- **OLS path:** Survey-weighted within-transformation + WLS via `solve_ols(weights=...)` + TSL vcov via `compute_survey_vcov()`.
- **Logit/Poisson paths:** Survey-weighted IRLS via `solve_logit(weights=...)`/`solve_poisson(weights=...)` + X_tilde linearization trick for TSL vcov: `X_tilde = X * sqrt(V)`, `r_tilde = (y - mu) / sqrt(V)`, then `compute_survey_vcov(X_tilde, r_tilde, resolved)` gives correct QMLE sandwich. ASF means and gradients use survey-weighted averaging.
- **Note:** Only `pweight` (probability weights) are supported; `fweight`/`aweight` raise `ValueError` because the composed survey/QMLE weighting changes their semantics.
- **Note:** Replicate-weight variance is not yet supported (`NotImplementedError`). Use TSL (strata/PSU/FPC) instead.
- **Note:** Bootstrap inference (`n_bootstrap > 0`) cannot be combined with `survey_design` — no survey-aware bootstrap variant is implemented.

**Heterogeneous cohort trends (paper W2025 Section 8 / Eq. 8.1):**
- `WooldridgeDiD(cohort_trends=True)` adds linear `dg_i · t` interactions to the design matrix for each treated cohort. Under the heterogeneous-trends DGP `y = c_i + α_t + δ_g · t + τ · w_{it} + u_{it}`, the parameter recovers `τ` even when parallel trends fails (paper Eq. 8.3 commentary on the Walmart application: "the estimated effects are much smaller than either the lags only or leads and lags estimates").
- **Identification (paper Section 8 / Eq. 8.1):** each treated cohort must have at least 2 pre-treatment periods (`t < g - anticipation`) for `dg_i · t` to be separately identified from cohort + time FE. `fit()` raises `ValueError` when the contract is violated.
- **OLS-path only:** `cohort_trends=True` is rejected at `__init__` for `method ∈ {"logit", "poisson"}` per paper Section 8's OLS scope. `NotImplementedError` cites the paper section explicitly.
- **Auto-routes to full-dummy mode** regardless of `vcov_type` (matching the absorb→fixed_effects auto-route pattern). Composing `dg_i · t` with the within-transformation yields `(dg_i − mean(dg_i)) · (t − mean(t))`, which is algebraically correct but non-trivial to verify on every panel shape; routing to the existing full-dummy auto-route used by `vcov_type ∈ {classical, hc2, hc2_bm}` keeps math closure verified against PR #483's R-parity goldens. UX implication: `cohort_trends=True` is silently more expensive than `cohort_trends=False` (carries N unit dummies); for very high-cardinality panels, the design-size warning at `wooldridge.py` fires.
- **`vcov_type="hc1"` finite-sample correction under `cohort_trends=True`:** the full-dummy auto-route changes the HC1 finite-sample factor from `(n-1)/(n-K_reference)` (within-transform default; K_reference counts the non-nested absorbed FE, see § Variance families) to `(n-1)/(n-k_total)` (full-dummy: counts intercept + treatment + unit + time + cohort-trend columns, nested included — the L1 clubSandwich-CR1S-style opt-in convention, which the K_reference adjustment deliberately does not touch). On typical panels where `n >> k_total` the gap is small (<2%); on small panels it can reach ~10%. This is a documented opt-in deviation specific to `cohort_trends=True` — users who need the within-transform HC1 finite-sample factor with cohort trends should use `vcov_type="hc1"` + `cohort_trends=False` and supply the cohort-trend interactions through a custom design (out-of-scope for the standard library surface).
- **Result attribute:** `WooldridgeDiDResults.cohort_trend_coefs: Dict[g → δ_g]` populated under `cohort_trends=True`; empty dict otherwise.
- **Note:** Polynomial-trend extensions (`"quadratic"`, `"cubic"` per paper p. 2572 footnote) are NOT yet exposed — `cohort_trends` is a binary `True/False` flag for linear `dg_i · t` only.
- **Note:** `cohort_trends=True` + `survey_design` is **NOT yet supported** (raises `NotImplementedError` at `fit()`). The full-dummy auto-route composed with the survey TSL variance has not been validated against R-parity goldens. Tracked in DEFERRED.md follow-up.
- **Note:** `cohort_trends=True` + `control_group="never_treated"` is **NOT yet supported** (raises `NotImplementedError` at `fit()`). The OLS + never_treated branch emits the `(g, t)` placebo cells (paper W2025 Section 4.4 placebo coverage) MINUS each cohort's `ref(g)` reference cell (W2025 Eq. 6.1/6.4; see the reference-period Note below). The appended `dg_i · t` trend columns are still spanned — jointly by the emitted cells and the unit fixed effects, which absorb `1{cohort=g}` and thereby recover the omitted reference — so the Section 8 trend specification remains unidentified on this branch. Use `control_group="not_yet_treated"` (the default) for `cohort_trends=True`. Tracked in DEFERRED.md follow-up.
- **Note:** Identification + baseline normalization for `cohort_trend_coefs` on all-eventually-treated panels: when a never-treated cohort (`g = 0`) is present, **all** `G` treated cohorts get a `dg_i · t` interaction column and `cohort_trend_coefs[g]` reports each cohort's linear slope relative to the never-treated baseline (absorbed by time FE). When **no** never-treated cohort exists, the **last cohort's** trend column is dropped deterministically per paper W2025 Section 5.4 ("all variables in regression (5.3) involving `dT_i` get dropped"); that cohort serves as the trend baseline, and `cohort_trend_coefs` surfaces `G - 1` entries (the last cohort is absent — its slope is the baseline in deviation form). **Note (deviation from W2025 Section 5.4 — narrowed to the covariate block):** Section 5.4's "all variables in regression (5.3) involving `dT_i` get dropped" governs four regressor families: the cells `dT·f_t`, the cell x covariate block `dT·f_t·ẋ`, the cohort trend `dT·t`, and the cohort x covariate block `D_T × X`. The library implements the first three — comparison-support filtering (below) removes the periods at which no unit is untreated, so cohort `G_max` receives no cells and no trend column, and the trend-drop path described here is reachable end-to-end. **The `D_{G_max} × X` normalization is NOT applied:** when covariates are supplied, `D_g × X` is still built for every cohort including `G_max`, so `Σ_g (D_g · x) = x` and the design is rank-deficient. Coefficients are unaffected (QR's arbitrary pick spans the same column space, `max |Δ| = 1.35e-14`), but `rank_deficient_action="error"` raises and `"warn"` drops an arbitrarily-chosen cohort's column rather than deterministically the last. Tracked in `TODO.md`; note for whoever takes it that `xtvar` under the default `demean_covariates=True` is FULL RANK (the raw block carries demeaned values while `D_g × X` carries raw ones, so the dependency does not arise) and forcing the drop there moves `overall_att` 1.11903 → 1.46269.

### Deviations from the paper / from R / library extensions

Consolidated list of substantive deviations from the W2025 paper and from R `etwfe`. Each is documented in the relevant section above with a labeled `**Note:**` or `**Deviation from R:**` line. AI PR reviewer recognizes these as documented (P3 informational) per the project's documented-deviation convention.

1. **Cell-count default for aggregation** (vs paper Eq. 7.4 / 7.6 cohort-share). `aggregate(weights="cell")` (default) matches Stata `jwdid_estat`. The opt-in `weights="cohort_share"` exposes the paper-Eq. 7.4 / 7.6 forms. Cohort-share is supported only for `type="simple"` and `type="event_study"`. See § Aggregations Note.
2. **HC1 finite-sample correction `(n-1)/(n-K_reference)`** (vs R `lm + clubSandwich::vcovCR(type="CR1S")` which uses `(n-1)/(n-k_total)`, nested dummies included). K_reference counts the non-nested absorbed FE + constant (matches Stata jwdid/reghdfe at machine precision); the remaining CR1S gap is the cluster-nested unit-dummy count. See § Variance families.
3. **QMLE sandwich `(G/(G-1)) · ((n-1)/(n-k))`** (vs Stata `jwdid` `G/(G-1)` only). Conservative; for typical panels n >> k the difference is negligible. Tracked in DEFERRED.md (WooldridgeDiD follow-up cluster row). See § Method Note.
4. **Nonlinear methods via direct QMLE** (vs R `etwfe` fixest backend). Avoids statsmodels/fixest dependency. See § Method Deviation from R.
5. **Logit cohort+time additive dummies** (not unit FE) to avoid incidental-parameters bias in short panels. Matches Stata `jwdid method(logit)`. See § Edge cases Note.
6. **Anticipation + aggregation**: `aggregate(type="simple", weights="cell")` uses `t >= g` as the post-treatment threshold regardless of `anticipation`. Anticipation-window leads are estimated as placebos but excluded from `overall_att`. See § Edge cases Note.
7. **Response-scale ATT vs R `etwfe` log-link coefficients** (Poisson + logit): diff-diff's `WooldridgeDiD(method="poisson" | "logit")` returns ATT on the response scale (counterfactual mean difference per paper W2023 ASF / APE framework); R `etwfe(family="poisson" | "logit")` returns the cell-level log-link / log-odds coefficient. Numerical cell-level R-parity for nonlinear paths requires either `emfx()`-based APE extraction on the R side or link-function inversion with baseline-mean adjustment; deferred (DEFERRED.md row, added in PR-B). See `tests/test_methodology_wooldridge.py::TestWooldridgeParityRPoisson` / `TestWooldridgeParityRLogit` for the current surface-test scope.

---

## LPDiD

**Primary source:** [Dube, A., Girardi, D., Jordà, Ò., & Taylor, A. M. (2025). "A Local Projections Approach to Difference-in-Differences." *Journal of Applied Econometrics*, 40(7), 741-758.](https://doi.org/10.1002/jae.70000) (Open Access; NBER Working Paper 31184; FRBSF Working Paper 2023-12.) Paper review on file: `docs/methodology/papers/dube-2025-review.md` (main article + official online appendix; equation/section numbering pinned to the JAE 2025 version).

**Reference implementations:** Stata `lpdid` (SSC `s459273`, the authors' reference); R `alexCardazzi/lpdid` (third-party; absorbing + non-absorbing); authors' example scripts `danielegirardi/lpdid` (R + Stata).

### Identification

Model-based DiD: untreated potential outcomes follow the two-way fixed-effects DGP `E[y_it(0)|i,t] = alpha_i + delta_t` (paper Eq. 1), under two assumptions:

- **No anticipation (Assumption 1):** `E[y_it(p) - y_it(0)] = 0` for all `t < p`.
- **Parallel trends (Assumption 2):** `E[y_it(0) - y_{i1}(0) | p_i = p] = E[y_it(0) - y_{i1}(0)]` for all `t in {2..T}`, `p in {1..T, inf}` - untreated potential-outcome trends are common across cohorts, stated relative to the first period (`t = 1`). The base period used for LP-DiD's long difference (first-lag `t-1` vs premean) is a separate efficiency/robustness choice, NOT part of this identification assumption (see the PMD edge case).

Treatment is binary; the main path assumes **absorbing** treatment (`D_{is} <= D_{it}` for `s < t`). Target parameter: the cohort-specific dynamic ATT `tau_h^g = E[y_{i,p_g+h}(p_g) - y_{i,p_g+h}(0) | p_i = p_g]`, h periods after group g enters at p_g. Treatment effects may be dynamic and heterogeneous across cohorts.

The key device is the **clean-control restriction**: each horizon-h regression keeps only newly-treated obs (`Delta_D_it = 1`) and not-yet-treated "clean" controls (`D_{i,t+h} = 0`, absorbing case). Excluding already-treated units from the control group is what eliminates the negative-weighting bias of naive TWFE/LP (paper Eqs. 6-7). Only the entry-period rows (`t = p_g`) identify each `beta_h` (online Appendix A.2).

### Key Equations

**LP-DiD regression (paper Eq. 4 restricted by Eq. 8), run separately per horizon `h in {-Q..H}`, `h != -1`:**

    y_{i,t+h} - y_{i,t-1} = beta_h^{LP-DiD} * Delta_D_it + delta_t^h + e_it^h
    sample:  Delta_D_it = 1 (newly treated)   OR   D_{i,t+h} = 0 (clean control)

`delta_t^h` = calendar-time fixed effects; **no unit FE** (differenced out). `h = -1` is the reference (coefficient fixed at 0); negative h give pre-trend placebos.

**Estimand = variance-weighted ATT (paper Eqs. 9-10; online Appendix B):**

    E(beta_h^{LP-DiD}) = sum_{g != 0} omega_{g,h} * tau_h^g
    omega_{g,h} = N_CCS_{g,h} * n_{g,h} * (1 - n_{g,h}) / sum_{g != 0}[...],   n_{g,h} = N_g / N_CCS_{g,h}

Weights are **always non-negative** (the central result). Via Frisch-Waugh-Lovell, the residualized treatment dummy is the per-group constant `Delta_D~_g = 1 - N_g/N_CCS_{g,h}` (online Appendix Eqs. B.4-B.6) - the hook for the reweighting implementation.

**Equally-weighted ATT (paper Section 3.3) - two equivalent routes:**
- `reweight=True`: weight each observation in the clean-control sample `CCS_{g,h}` (the newly-treated obs and their clean controls) by `(omega_{g,h}/N_g)^{-1}`. Numerically equivalent to **Callaway-Sant'Anna (2021)**.
- Regression adjustment (RA): fit the long difference on time FE using clean controls only, predict each treated obs's counterfactual, average residuals: `beta_h^{RA} = N_TR^{-1} sum_{TR}[(y_{i,t+h}-y_{i,t-1}) - Ehat((y_{i,t+h}-y_{i,t-1}) | D_{i,t+h}=0)]`. An imputation estimator in the **BJS (2024)** sense.

**Covariates (paper Section 4.1):** recommended RA path `beta_{h,x}^{RA} = N_TR^{-1} sum_{TR}[(y_{i,t+h}-y_{i,t-1}) - gamma~^h x_i - delta~_t^h]`, with `gamma~^h, delta~_t^h` from a clean-control-only regression. **PMD base period (Section 3.4):** replace `y_{i,t-1}` with the mean of the last `k` pretreatment periods (`k=t-1` = all); single-cohort `k=t-1` == BJS. **Pooled estimand (Section 3.5):** posttreatment-mean long difference `(1/(H+1)) sum_{h=0}^H y_{i,t+h} - y_{i,t-1}` as the dependent variable.

**Non-absorbing treatment (paper Section 4.2; `non_absorbing=`).** Two distinct entry-effect estimands, both fitted by the same per-horizon LP-DiD regression with mode-specific clean-sample restrictions (`Delta_D_it = D_it - D_{i,t-1}`):

    Eq. 12 (first-time entry, "first_entry"):
      treated:        D_{i,t+j} = 1 for 0 <= j <= h  AND  untreated before t   (stays treated through t+h)
      clean control:  D_{i,t+j} = 0 for j <= h                                 (== absorbing Eq. 8 control)

    Eq. 13 (effect stabilization, "effect_stabilization", window L; Assumption 9):
      treated:        fresh entry at t, D = 0 on [t-L, t-1], no other change in (t, t+h]
      clean control:  no treatment change in [t-L, t+h]                        (admits stabilized already-treated units)

Eq. 12 reuses the absorbing clean control and only restricts the treated set (a unit exiting within the horizon drops out); on an absorbing panel it is numerically identical to the absorbing path. Eq. 13 lets units whose treatment has been stable for at least `L` periods serve as clean controls (Assumption 9: dynamic effects stabilize after `L` periods), making estimation feasible when there are few or no never-treated units (e.g. minimum wage). Non-negative weights (online Appendix C, `omega''`) carry over. Placebo horizons (`h < 0`) use the clean window `[t - max(L, -h), t-1]` so the long difference's reach-back to `t+h` cannot be contaminated.

### Standard Errors

**The paper specifies no SE formula** - Section 1 defers to "standard, well-understood techniques." The reference Stata uses **cluster-robust SEs at the unit level** (`vce(cluster unit)`, footnote 9); pooled / joint tests stack the per-horizon regressions (`suest`). No bootstrap is discussed. Any analytical SE the library ships - and in particular an influence-function cluster variance for the RA path - is therefore an **implementation choice validated against the reference package, not against the paper**, and must be documented under Deviations once implemented (PR-B).

### Edge Cases

- **Composition effects (Section 3.6):** the treated/clean-control set can change across horizons. `no_composition` tightens the clean-control condition to `D_{i,t+H}=0` at all horizons (and excludes cohorts with `p_g > T-H` to fix the treated set). Costs statistical power. **Implementation note:** this fixes the *post-treatment* composition (every post horizon shares the same realized sample, even on unbalanced panels); pre-treatment placebo horizons use whatever pre-period data is available. Reweighting denominators and the regression-adjustment counterfactual are computed from the realized post-drop sample (not the pre-drop panel) so they stay consistent with the regression's risk set on unbalanced panels.
- **Bias-variance (Sections 3.3, 5.3):** variance weighting (default) -> lower variance, some bias; equal weighting (`reweight`) -> unbiased, higher variance. Variance won at short horizons, equal at long horizons in the paper's simulation.
- **PMD vs first-lag (Section 3.4):** PMD gains efficiency under low autocorrelation but can amplify bias if PT holds only in some pretreatment periods; first-lag relies on weaker PT (Marcus & Sant'Anna 2021). Choose the base period ex-ante.
- **Covariate-weight positivity (online Appendix B.2):** direct covariate inclusion keeps non-negative weights ONLY under linear + homogeneous covariate effects (B.2.1; main-text Assumption 6); in the general case (B.2.2) weights are not guaranteed positive -> prefer the RA covariate path (the direct path should carry a homogeneity-assumption warning).
- **Non-absorbing (Section 4.2, online Appendix C):** implemented via `non_absorbing="first_entry"` (Eq. 12) and `non_absorbing="effect_stabilization"` (Eq. 13, requires `stabilization_window=L`); the default `non_absorbing=None` keeps the absorbing path and still rejects non-absorbing input. Both modes are **entry-effect** estimands; the Appendix-C exit-event dynamics (`eta_h^{g,n}`, separate switch-off event-studies) are a deferred follow-up. **Boundary convention:** periods before a unit's first observed period are treated as untreated with no change (extends Deviation 5), so window conditions clamp pre-`min_t` offsets to 0 - a unit genuinely treated before the panel starts could be misread as a fresh entry under `effect_stabilization` (PR-C2 documented this as a known divergence from `alexCardazzi/lpdid`, which NA-excludes such first-rows - see Deviation #4). **Interior gaps** make the `[t-L, t+h]` window conditions unverifiable, so non-absorbing modes require gap-free panels within each unit's observed span and raise otherwise (the absorbing path's interior-gap reindex is a deferred follow-up for non-absorbing).

### Deviations from the paper / from R / library extensions

The paper specifies no standard-error formula (Section 1 defers to "standard, well-understood techniques"); the reference Stata `lpdid` uses `vce(cluster unit)`. The entries below document diff-diff's inference and scope choices.

1. **Note:** Standard errors are **cluster-robust at the unit level by default** - `cluster=None` auto-clusters at the unit identifier and the results record `cluster_name`/`n_clusters` - with a `t(G-1)` reference distribution (G = realized clusters in each horizon's clean-control sample). Matches Stata `lpdid` `vce(cluster unit)`; the paper prescribes no SE. **Since 3.9 (M-127) the reference is the `df_convention` knob's `"cluster"` value — LPDiD's DEFAULT, so nothing moved**: `"residual"` opts into t(per-horizon residual df `n_eff − k_kept`; on the RA path `n_total − k0_kept − 1`, the pooled M-estimator's parameter count — nuisance coefficients plus the ATT — a library convention with no external anchor since Stata `teffects ra` reports z), and `"normal"` into z. The G here stays the RAW unique cluster count (deliberately not `effective_cluster_count`: LPDiD's reweights are strictly positive and there is no user weight column, so the two cannot diverge). **Carve-out to the flat `t(G-1)` claim:** the degenerate lanes — the unclustered-refit fallback when the clustered solve fails, the RA `G<=1` lane, and the saturated-design early return (`n_obs <= k`) — keep the literal `df=None` (normal theory, NaN provenance) under ALL knob values, exactly as before 3.9.
2. **Note:** The regression-adjustment (RA) covariate path (`reweight=True` with covariates/absorb) reports an **influence-function cluster variance** `sum_c (sum_{i in c} psi_i)^2 / n^2`, in the same family as `ImputationDiD`'s Theorem-3 / BJS variance (see "IF-based variance estimators vs analytical-sandwich estimators" above). Its single Gram inversion is routed through `linalg._rank_guarded_inv` (finite SE on the identified subspace under near-collinearity; NaN at rank 0). Unlike the default/weighted `solve_ols` `hc1`-cluster path - which applies the `(G/(G-1))*((n-1)/(n-K_reference))` finite-sample factor (see the nested-dummy Note below) - the RA IF variance carries **no finite-sample factor**, while both paths share the `t(G-1)` reference. **PR-B2 validated this asymmetry as faithful to the authors' own tooling**, not a defect: the no-factor RA convention matches the canonical Stata `teffects ra ... atet vce(cluster)`, while the default path matches `feols`/`reghdfe`. **Originally inferred from the authors' `lpdid_regression_adjustment.do` degrees-of-freedom comments, this is now MEASURED against a runnable `teffects`:** `benchmarks/stata/generate_lpdid_ra_golden.do` runs `teffects ra (Dy x i.time) (tdiff), atet vce(cluster unit)` on an independently reconstructed clean sample (a Stata port of the `generate_lpdid_golden.R` `prep`/`clean_h` recipe), and the library RA IF SE matches it to **~1e-16 at all 7 event-study horizons** - the SE ratio is a flat 1.0 (no `G/(G-1)`, no `(n-1)/(n-k)`), directly confirming the no-finite-sample-factor convention (golden `benchmarks/data/lpdid_ra_stata_golden.json`; test `tests/test_lpdid_ra_stata_parity.py`, which also gates the Stata clean sample's realized size and cluster count `(e(N), e(N_clust))` against the library's `(n_obs, n_clusters)`). The RA *point* estimate is triple-anchored: Python == R full-interaction (`i.dtreat##(i.time c.x)`) == Stata `teffects`, to ~1e-10..1e-13 (`tests/test_methodology_lpdid.py::test_ra_covariate_point`). Because Stata is node-locked and cannot run in CI (goldens are committed, exactly like the R arm), the CI-runnable guards remain the regression pin (`test_ra_covariate_se_regression_pin` / `RA_SE_PIN`) and the ungated Monte-Carlo coverage study `benchmarks/python/coverage_lpdid_ra.py` (~0.95 empirical coverage of the true effect at cluster counts G in {30, 100, 300}).
3. **Note:** Direct covariate inclusion (`reweight=False` with covariates/absorb) emits a `UserWarning`: per online Appendix B.2.2 it preserves the non-negative LP-DiD weighting result only under linear and homogeneous covariate effects, so the regression-adjustment path (`reweight=True`) is preferred.
4. **Deviation from R:** Scope - non-absorbing treatment (Section 4.2) implements the **entry-effect** estimands (`non_absorbing="first_entry"` / `"effect_stabilization"`, PR-C1). **PR-C2 R-parity-validated both modes against an INDEPENDENT `fixest::feols` reconstruction of the paper's Eq. 12 / Eq. 13 clean-sample restrictions** (point and SE match to ~1e-13/~1e-15 for the variance-weighted variants; the `effect_stabilization` reweighted point matches and its SE is pinned as a regression guard - a small weighted-cluster convention difference vs feols; `tests/test_methodology_lpdid.py::TestLPDiDNonAbsorbingParityR`). The recipe's independence was demonstrated when an earlier draft's Eq. 12 control off-by-one diverged from the already-correct library and was corrected against the paper, plus a hand-computed Python micro-check. **`alexCardazzi/lpdid`'s `nonabsorbing_lag` is NOT a faithful Eq. 13** (it clamps `treat_diff[<0]<-0`, so its clean-control window blocks only treatment turn-*ons*; it reuses a forward placebo window; and it NA-excludes pre-panel-treated rows where the library clamps pre-`min_t` to untreated): it diverges ~0.01-0.05 from Eq. 13 even on a monotone no-off-switch panel, so it is **recorded in the golden `meta` as a divergent third-party reference, not a parity gate** (the alexCardazzi-pooled precedent). The library's "no treatment change" (both directions) and backward placebo window are the more paper-faithful choices. `first_entry` (Eq. 12) has no R-package analogue (anchored on the independent feols recipe only). Appendix-C exit-event dynamics and the Stata canonical SE remain deferred follow-ups.
5. **Note:** LP-DiD's per-unit quantities (outcome lags `ylags`, first-difference lags `dylags`, integer-`pmd` premean baselines, treatment-entry detection) are **calendar** quantities (`t-1`, `t-k`), so the estimator requires integer-valued, globally consecutive `time` labels. A unit with an **interior time gap** is handled by reindexing that unit to its complete interior calendar grid `[min_t, max_t]`, computing the features on the grid, then **restricting back to the observed rows** - so a lag/first-difference spanning a gap is NaN and the observation fails closed (never the previous-*observed* row), and no synthetic gap row enters a regression. A gap-free panel skips this entirely and is bit-identical. **Entry = first OBSERVED treated period** (`min(t | D_it=1)`): an unobserved pre-onset gap cannot move a cohort earlier, the only well-defined convention when the true switch falls in an unobserved period.
6. **Note (pooled estimand):** The pooled pre/post ATT (the headline `results.att` is the pooled-post row) is the **unit-equal-weighted average of each unit-event-time's mean long difference** over the window - `mean_h(y_{i,t+h}) - baseline_{i,t}`, one observation per (unit, event-time), regressed on the treatment-switch indicator with event-time fixed effects on the **fixed-composition** sample (only units observing *every* pooled target, with clean controls required through `max(h)`). This equals the mean of the per-horizon event-study coefficients on a balanced panel. **PR-B2 validated it against the authors' runnable R reference**: the pooled estimand matches the authors' own R pooled recipe (`danielegirardi/lpdid`: a `slider` window-mean minus `y_{t-1}` on the clean-through-window-end sample) to ~1e-13 (`tests/test_methodology_lpdid.py::test_pooled`). A prior version of this note speculated the authors used a horizon-**stacked** pooled regression; the authors' R reference in fact uses this same fixed-composition mean-long-difference, so that speculation was incorrect. Unlike the event-study variants (where `alexCardazzi` is a cross-check gate), pooled is anchored to the authors' recipe **only**: `alexCardazzi`'s pooled uses a **laxer** clean-control window, so it differs and is recorded in the golden `meta` for transparency, not as a parity target.
7. **Deviation from R:** `no_composition` is intentionally more faithful to the paper's fixed-composition intent (Section 3.6) than the R packages: it fixes the realized sample across *all* post horizons (every post coefficient shares one sample, even on unbalanced panels) and excludes cohorts with `p_g > T-H`, whereas `alexCardazzi/lpdid` uses a looser per-horizon sample and a stricter `treat_date < T-H` cutoff. It therefore has **no exact R-package anchor** and is validated by the pure-Python tests in `tests/test_lpdid.py` (the R-parity golden omits it; `alexCardazzi`'s looser-semantics value is recorded in the golden `meta`).
8. **Note (survey design):** Complex-survey support (`survey_design=SurveyDesign(...)`, PR-D1) covers the **variance-weighted default path** (`reweight=False`, with or without direct-inclusion covariates): each horizon's long-difference regression is fit by WLS on the survey probability weights, and the SE is the stratified-PSU **Taylor-linearization (Binder 1983 TSL)** sandwich `meat = sum_h (1-f_h)*(n_h/(n_h-1))*sum_j (S_hj - S_h_bar)(S_hj - S_h_bar)'` with `df = n_PSU - n_strata`, reusing the shared `diff_diff/survey.py` helpers (`compute_survey_vcov` / `_compute_stratified_psu_meat`). The design is re-resolved on each realized (post-clean-control) sample so weights/strata/PSU align with the regression rows; with no explicit PSU the unit (LP-DiD's default cluster) is injected as the PSU. Supports pweight + strata + PSU + FPC + lonely-PSU handling. It **rejects** `survey_design` combined with `reweight=True` (the equally-weighted / regression-adjustment IF path has no validated survey reference - the same gap as the RA SE in Deviation #2), replicate-weight designs, and non-pweight (fweight/aweight) types, each a deferred follow-up. The non-survey path is byte-for-byte unchanged (gated on `survey_design is None`). **PR-D2 validated all three survey paths end-to-end against `survey::svyglm`** - per-horizon point/SE/df + pooled for the variance-weighted full design (strata+PSU+FPC), the weights-only unit-injected-PSU design, and the direct-covariate variant (`tests/test_methodology_lpdid.py::TestLPDiDSurveyParityR`; point ~1e-6, SE ~1e-5, df exact via the per-design `n_PSU - n_strata` / `n_PSU - 1` formula). `svyglm` is itself the reference implementation of the Binder TSL sandwich, so it anchors the variance directly (no third-party survey-package gate is needed); the clean-sample construction is independently cross-checked in the generator (the unweighted variance-weighted event study matches `alexCardazzi/lpdid` to <1e-8, and selection is weight-independent). A dedicated survey panel (`benchmarks/data/lpdid_survey_panel.csv`, own seed) keeps the absorbing / non-absorbing goldens byte-identical.
9. **Note (clustered CR1 k, nested dummy blocks — 3.9 K_reference):** LPDiD's per-horizon design carries two built-in dummy blocks — the `_event_time` (time-FE) dummies added under `include_time_fe=True` and the inline `drop_first` `absorb=` dummies. The authors' reference recipe ABSORBS the time FE (`feols(... | time, vcov = ~unit)`, `benchmarks/R/generate_lpdid_golden.R`), so both blocks follow the nested convention: when a block is nested in the raw `cluster` (e.g. a time-like cluster for `_event_time`, or an absorb dim coarser than the cluster), its rank is SUBTRACTED from the CR1 factor's `k` via a negative `cluster_k_adjustment`, converging on the same `K_reference` accounting as `feols`/`reghdfe`. Non-nested configs are no-ops (unit clustering leaves `_event_time` non-nested → adjustment 0, preserving bit-identical Rust-lane output). The solve's broad `except (ValueError, ZeroDivisionError)` re-solve re-raises `InvalidClusterKAdjustment` so a contract violation can never degrade to a silent `se=NaN` (`tests/test_variance_conventions.py::TestKReferenceConvergence::test_lpdid_never_swallows_the_contract_raise`). The `t(G-1)` tail reference (Note 1) is unchanged and is, since 3.9, the `df_convention="cluster"` DEFAULT (see Note 1's knob addendum).

### Implementation Checklist

- [x] Per-horizon long-difference OLS with time FE, no unit FE; `h=-1` reference fixed at 0 (PR-B1)
- [x] Clean-control sample restriction (absorbing: `D_{i,t+h}=0`) (PR-B1)
- [x] Variance-weighted (default) + reweighted (equal-weight) estimands (PR-B1)
- [x] Regression-adjustment covariate path (recommended) + direct-inclusion path with homogeneity warning (PR-B1)
- [x] PMD base period; pooled pre/post estimands (PR-B1)
- [x] `no_composition` option (PR-B1)
- [x] Cluster-robust SE at unit level by default; NaN-consistent inference via `safe_inference` (PR-B1)
- [x] `LPDiDResults` with `summary()` / `to_dict()` / cluster metadata (PR-B1)
- [x] doc-deps.yaml mapping for `diff_diff/lpdid.py` + `lpdid_results.py`; llms.txt / llms-full.txt catalog entries (PR-B1, test-enforced)
- [x] B1 pure-Python tests: analytical DGPs + cross-estimator equivalence (CS / BJS / DiD; Cengiz-stacked dropped, documented) + unbalanced / interior-gap / RA-overlap / pmd-missing edge cases (PR-B1)
- [x] B2: self-generated R-parity (authors' `danielegirardi/lpdid` recipes + `alexCardazzi/lpdid` cross-check; VW / reweight / pmd / direct / pooled / RA-point to ~1e-12; RA SE pinned + MC-coverage-validated; `no_composition` more paper-faithful than R, B1-tested) (PR-B2)
- [x] Non-absorbing extension (Section 4.2): entry-effect estimands - first-entry (Eq. 12) + effect-stabilization (Eq. 13, window `L`) via `non_absorbing`; mode-aware clean-sample masks, `C=0`-below-`min_t` boundary convention, gap-free requirement; pure-Python tests (absorbing reduction, re-entry mechanism, placebo, no-negative-weighting, stabilized-control, DGP recovery) (PR-C1)
- [x] Non-absorbing R-parity: both modes vs an independent `fixest::feols` Eq. 12/13 reconstruction (point+SE ~1e-13/~1e-15 vw; reweighted point + pinned SE); `alexCardazzi nonabsorbing_lag` recorded as a divergent reference (not a gate); absorbing B2 goldens byte-identical (PR-C2)
- [ ] Non-absorbing exit-event dynamics (Appendix C `eta_h`) + the Stata canonical RA/SE - deferred
- [x] Survey-design support (PR-D1): pweight + stratified-PSU Taylor-linearization (Binder TSL) variance on the variance-weighted default path; per-sample design re-resolution + unit-as-PSU injection; reweight (incl. RA), replicate-weight, and non-pweight designs rejected; pure-Python invariants (reduction/unit-clustering, FPC-shrinks-SE, stratification, lonely-PSU, NaN-consistency, metadata) (PR-D1)
- [x] Survey-design R-parity (PR-D2): all three survey paths (VW strata+PSU+FPC, weights-only inject, direct-covariate) validated end-to-end vs `survey::svyglm` - per-horizon point/SE/df + pooled (point ~1e-6, SE ~1e-5, df exact); clean sample independently `alexCardazzi`-cross-checked (<1e-8); dedicated survey panel keeps the absorbing / non-absorbing goldens byte-identical (PR-D2)

---

## LWDiD

**Primary sources:**
- [Lee, S.J. & Wooldridge, J.M. (2025). A Simple Transformation Approach to Difference-in-Differences Estimation for Panel Data. SSRN Working Paper No. 4516518 (61-page revision, cover page June 8, 2026; SSRN "last revised" 8 Jun 2026).](https://ssrn.com/abstract=4516518) — estimation core (rolling transformations, RA/IPW/IPWRA, multiplier bootstrap). DOI: 10.2139/ssrn.4516518.
- [Lee, S.J. & Wooldridge, J.M. (2026). Simple Approaches to Inference with Difference-in-Differences Estimators with Small Cross-Sectional Sample Sizes. SSRN Working Paper No. 5325686 (36 pages, cover page February 3, 2026; SSRN "last revised" 13 Jun 2026).](https://ssrn.com/abstract=5325686) — exact small-sample inference layer. DOI: 10.2139/ssrn.5325686.

Exact reviewed artifacts (PDF SHA-256) and live-verified SSRN metadata are pinned in the paper-review headers below; stale SSRN caches/mirrors may still show the superseded December 2025 / January 2026 revisions.

Full maintainer paper reviews (equation-level detail, replication targets): `docs/methodology/papers/lee-wooldridge-2025-review.md`, `docs/methodology/papers/lee-wooldridge-2026-review.md`.

- **Note:** Registry entry authored with the paper reviews ahead of the implementation (PR #588, third-party contribution under maintainer revision). Checklist boxes are unchecked until the implementation lands; implementation-specific edge-case notes will be finalized in that PR.
- **Note:** Maintainer validation suite: `tests/test_methodology_lwdid.py` (import-skip-gated until `diff_diff.lwdid` exists; xfail markers encode PR #588's outstanding acceptance criteria; replication goldens in `benchmarks/data/lwdid_walmart_eventstudy_golden.json` and `benchmarks/data/real/castle_lw_subset.csv`).

**Key implementation requirements:**

*Assumption checks / warnings:*
- Treatment is absorbing (no reversibility), common timing (`1 < S <= T`) or staggered (cohorts `g in {S,...,T,infinity}`, mutually exclusive and exhaustive); at least one pre-treatment period.
- Common timing: **NAC** (no anticipation, eq. (2.7)), **CPTC** (conditional parallel trends, eq. (2.10)), **OVLC** (overlap, eqs. (2.14)-(2.15)); Theorem 2.1 identifies `tau_r`, `r = S,...,T`.
- Staggered: **CNAS** (eq. (4.4); X-conditioning droppable with NT-only controls), **CPTS** (eq. (4.6)), **OVLS** (eq. (4.10), control pool `A_{r+1} = D_{r+1} + ... + D_T + D_infinity`); Theorem 4.1.
- Heterogeneous linear trends: **CHT** (eq. (5.3)) — demeaning inconsistent under CHT; unit-specific detrending (Procedure 5.1) restores consistency under CNAS + CHT + OVLS.
- Minimum pre-periods: >= 1 per cohort for demeaning, >= 2 for detrending (rank condition, Appendix B); failing cells not estimable.
- Small-sample (exact) inference layer: classical linear model assumptions — conditional normality AND homoskedasticity of the collapsed cross-sectional error (2026 paper eqs. (2.7)-(2.9); with controls (2.18)-(2.19)). Sample-size guards: `N0 >= 1`, `N1 >= 1`, `N >= 3`; `N > K + 2` with K controls; interacted controls need `N0 > K + 1` AND `N1 > K + 1`; NT-only controls in the staggered case need `N_infinity >= 2`.
- IPWRA is doubly robust: consistent if either the propensity-score model or the outcome model is correct.

*Transformations (LW 2025 eqs. (3.2)/(4.11), (5.6); as specified for implementation):*

    Demeaning:   Y_dot_irg  = Y_ir - (1/(g-1)) * sum_{s=1}^{g-1} Y_is
    Detrending:  Yddot_irg  = Y_ir - Yhat_irg,  Yhat from unit OLS of Y_it on (1, t), t < g  (out-of-sample residuals)

Event-study/placebo transformations over ALL periods (Appendix D): demeaning (D.1) excludes `t = g-1`; detrending (D.2) excludes `t in {g-2, g-1}` (anchor periods, event times `r = -1` / `r = -2, -1`).

*Estimation (per (g, t) cell; LW 2025 eq. (E.1)):*
- Cross-sectional RA regression on the cell sample `A_{g,t}`: `Y_dot on 1, D_g, X, D_g (X - Xbar_g)`; ATT(g,t) = coefficient on `D_g`. No-covariate case reduces to plain DiD (eq. (3.4)); Theorem 3.1: common-timing per-period regressions are numerically equivalent to pooled OLS (3.6) (r = g reproduces the ETWFE estimand; `r > g` does not).
- IPWRA (workhorse): logit propensity score per cell + WLS with weights `w = D + (1-D) p/(1-p)`; IPW = special case without the outcome-regression component.
- Control pool at (g, r): `A_{r+1} = 1` (never-treated + not-yet-treated) by default; NT-only optional. Pre-treatment placebo cells use the Appendix D.3 rule `A_{g,t} = {G = g} ∪ {G = 0} ∪ {G > max(g,t)}`.

*Aggregation:*
- Event-study: `WATT(r) = sum_{g in G_r} omega_{g,r} ATT(g, g+r)` with `omega_{g,r}` = (treated units of cohort g contributing at event time r) / (total treated units contributing at event time r) - the operative definition per LW 2025 Appendix E.1, required under unbalanced panels where a cohort's contributing count at r can differ from `N_g`. In balanced panels this simplifies to `N_g / N_{G_r}` (Sec. 6.2/D.3). Aggregated influence function `IF_{i,r} = sum_g omega_{g,r} IF_{i,g,g+r}`.
- Overall: composite-outcome single regression (LW 2026 eqs. (7.18)-(7.19)) — `tau_omega` with cohort-share weights `omega_g = N_g / N_treat`; automatically accounts for correlation among per-cohort effects and supports exact small-N inference.

*Standard errors:*
- Large-N default: influence-function **multiplier bootstrap** (LW 2025 Algorithm 1): IFs per (g,t) from E.2 (RA, finite-sample exact), E.3 (IPWRA, stacked M-estimator with logit-score correction), E.4 (IPW, `psi - Gamma' IF_gamma` correction); centered IFs; **unit-level Rademacher multipliers** (one draw per unit across all cells — unit clustering by construction); sup-t simultaneous bands over the event-study path; B = 999 in the paper's application; anchor periods excluded.
- Small-N exact (LW 2026): usual OLS SE on the collapsed cross-sectional regression with exact `T_{N-2}` / `T_{N-K-2}` reference distribution; valid down to `N = 3` and a single treated unit (`N1 = 1` — the t statistic is the studentized residual; same for `N_g = 1` per cohort in (7.8)/(7.10)).
- Alternatives: HC3 when there are "at least a handful" of treated units; randomization inference for the sharp null (two-sided p = c / #permutations; Stata `lwdid` `ri` option); higher-level clustering and Conley SHAC SEs for larger cross sections (LW 2026 Sec. 8.2, citing Abadie-Athey-Imbens-Wooldridge 2023).

*Edge cases:*
- Anchor periods: event-study omits `r = -1` (demeaning) / `r = -2, -1` (detrending); bootstrap excludes them.
- All units eventually treated (LW 2025 Sec. 4.3): drop `D_infinity`; effects defined relative to the last cohort; no effect estimable for the last cohort.
- Unbalanced panels (Sec. 4.4): transform observed data; per-cell observability requirements (1 pre-period demeaning / 2 detrending + outcome observed at r); selection may correlate with unit heterogeneity (and trends, under detrending) but not with shocks.
- Anticipation: drop periods just prior to the intervention from the pre-average/trend (LW 2025 Sec. 4.4; LW 2026 eq. (2.22) anchor `Ybar_{i,S0}`, `S0 < S-1`).
- Periods with no newly-treated units: no effects estimated there.
- Seasonality: seasonal dummies in the transformation step (LW 2026, p12).
- Complex nonlinear pre-trends: linear detrending may be inadequate — authors' caveat that the method "will not always work better than existing alternatives"; higher-order trend terms possible at a power cost.
- Per-period effect combinations: SEs for linear combinations of per-period estimates are NOT easily obtained (serial dependence); the composite regression (7.19) is the paper's device for valid aggregate SEs.
- CS normalization: rolling event studies are deviations from own pre-treatment averages, NOT normalized to `r = -1` as in Callaway-Sant'Anna — event-study points are not numerically comparable across the two.
- No incidental-parameters problem from unit-specific trend regressions with small T.

*Procedures (verbatim transcriptions in the paper reviews):*
- LW 2026 Procedure 2.1 (unit demeaning, common timing) / Procedure 3.1 (unit detrending) — collapse to `{(Ybar_dot_i, D_i)}` and run the exact-t cross-sectional regression.
- LW 2025 Procedure 3.1 (rolling, common timing) / Procedure 4.1 (rolling, staggered; control pool `A_{r+1}`) / Procedure 5.1 (staggered detrending).
- LW 2025 Algorithm 1 (multiplier bootstrap, 7 steps): compute per-cell IFs -> aggregate WATT(r) -> center IFs -> B unit-level Rademacher draws -> bootstrap SEs -> sup statistics -> sup-t bands.

**Reference implementation(s):**
- Stata: user-written `lwdid` (Hur, Lee and Wooldridge 2026; SSC) — implements the full procedure, multiplier-bootstrap inference, and randomization inference (`ri` option); ancillary datasets `lw_smoking.dta` / `lw_walmart.dta` (MIT) are the sources for `load_prop99()` / `load_walmart()`.
- R: none.

**Replication targets (from the papers; datasets available via `diff_diff.datasets`):**
- Prop 99 (LW 2026 Table 3, 38-state donor pool): demeaning ATT = -0.422 (SE 0.121); detrending ATT = -0.227 (SE 0.094), exact p = 0.021 vs randomization-inference p = 0.020.
- Castle laws (LW 2026 Sec. 7): `tau_omega` = 0.092 (demeaning; OLS SE 0.057), 0.067 (detrending).
- Walmart entry (LW 2025 Tables A4/A5, 1,277 counties): per-relative-period WATT(r) with SEs for r = 0..13.

**Requirements checklist:**
- [ ] Rolling demeaning (3.2)/(4.11) using ALL pre-g periods; detrending (5.6)/(D.2) via unit OLS with out-of-sample residuals
- [ ] Minimum pre-period enforcement (>= 1 demeaning / >= 2 detrending); failing cells dropped with warning
- [ ] Control pools: NT + NYT (`A_{r+1} = 1`) default, NT-only option (`N_infinity >= 2` guard); placebo cells per D.3 rule `G > max(g,t)`
- [ ] RA (E.1) with treated-cohort-centered interactions; IPWRA (logit + WLS); IPW special case
- [ ] Influence functions per E.2/E.3/E.4 including first-stage logit-score corrections; IFs centered
- [ ] WATT(r) event-study aggregation with contributing-treated-unit weights (E.1 definition; cohort-size weights `N_g / N_{G_r}` only as the balanced-panel simplification); anchor periods excluded (r = -1 / r = -2,-1)
- [ ] Algorithm 1 multiplier bootstrap: unit-level Rademacher, sup-t simultaneous bands
- [ ] Composite-outcome overall aggregation (7.18)/(7.19) with cohort-share weights
- [ ] Exact-t inference: `T_{N-2}` / `T_{N-K-2}`, valid to `N = 3`, `N1 = 1`, `N_g = 1`; sample-size guards enforced
- [ ] HC3 alternative; randomization inference (two-sided p = c / #permutations); higher-level clustering / SHAC for larger N
- [ ] Anticipation-robustness period dropping; seasonal dummies in the transformation step
- [ ] All-eventually-treated (Sec. 4.3) and unbalanced-panel (Sec. 4.4) support
- [ ] Common-timing no-covariate case reproduces plain DiD (3.4); Theorem 3.1 pooled-OLS equivalence (cross-estimator test vs `DifferenceInDifferences` / ETWFE at r = g)
- [ ] Prop 99 / castle-laws / Walmart replication targets pinned as tests

---

# Advanced Estimators

## SyntheticDiD

**Primary source:** [Arkhangelsky, D., Athey, S., Hirshberg, D.A., Imbens, G.W., & Wager, S. (2021). Synthetic Difference-in-Differences. *American Economic Review*, 111(12), 4088-4118.](https://doi.org/10.1257/aer.20190159)

**Key implementation requirements:**

*Assumption checks / warnings:*
- Requires balanced panel (same units observed in all periods)
- Warns if pre-treatment fit is poor (high RMSE)
- Treatment must be "block" structure: all treated units treated at same time

*Estimator equation (as implemented):*

```
τ̂^sdid = Σ_t λ_t (Ȳ_{tr,t} - Σ_j ω_j Y_{j,t})
```

where Ȳ_{tr,t} is the mean treated outcome at time t, ω_j are unit weights, and λ_t are time weights.

*Unit weights ω (Frank-Wolfe on collapsed form):*

Build collapsed-form matrix Y_unit of shape (T_pre, N_co + 1), where the last column is the mean treated pre-period outcomes. Solve via Frank-Wolfe on the simplex:

```
min_{ω on simplex}  ζ_ω² ||ω||₂² + (1/T_pre) ||A_centered ω - b_centered||₂²
```

where A = Y_unit[:, :N_co], b = Y_unit[:, N_co], and centering is column-wise (intercept=True).

**Two-pass sparsification procedure** (matches R's `synthdid::sc.weight.fw` + `sparsify_function`):
1. First pass: Run Frank-Wolfe for 100 iterations (max_iter_pre_sparsify) from uniform initialization
2. Sparsify: `v[v <= max(v)/4] = 0; v = v / sum(v)` (zero out small weights, renormalize)
3. Second pass: Run Frank-Wolfe for 10000 iterations (max_iter) starting from sparsified weights

The sparsification step concentrates weights on the most important control units, improving interpretability and stability.

*Time weights λ (Frank-Wolfe on collapsed form):*

Build collapsed-form matrix Y_time of shape (N_co, T_pre + 1), where the last column is the per-control post-period mean (averaged across post-periods for each control unit). Solve:

```
min_{λ on simplex}  ζ_λ² ||λ||₂² + (1/N_co) ||A_centered λ - b_centered||₂²
```

where A = Y_time[:, :T_pre], b = Y_time[:, T_pre], and centering is column-wise.

*Auto-regularization (matching R's synthdid):*
```
noise_level = sd(first_differences of control outcomes)   # pooled across units
zeta_omega  = (N_treated × T_post)^(1/4) × noise_level
zeta_lambda = 1e-6 × noise_level
```

The noise level is computed as the standard deviation (ddof=1) of `np.diff(Y_pre_control, axis=0)`,
which computes first-differences across time for each control unit and pools all values.
This matches R's `sd(apply(Y[1:N0, 1:T0], 1, diff))`.

*Frank-Wolfe step (matches R's `fw.step()`):*
```
half_grad = A' (A x - b) + η x        # η = N × ζ²
i = argmin(half_grad)                   # vertex selection (simplex corner)
d_x = e_i - x                          # direction toward vertex i
step = -(half_grad · d_x) / (||A d_x||² + η ||d_x||²)
step = clip(step, 0, 1)
x_new = x + step × d_x
```

Convergence criterion: stop when objective decrease < min_decrease² (default min_decrease = 1e-5 × noise_level, max_iter = 10000, max_iter_pre_sparsify = 100).

*Standard errors:*

- Default: Placebo variance estimator (Algorithm 4 in paper)
  1. Randomly permute control unit indices
  2. Split into pseudo-controls (first N_co - N_tr) and pseudo-treated (last N_tr)
  3. Re-estimate unit weights (Frank-Wolfe) on pseudo-control/pseudo-treated data
  4. Re-estimate time weights (Frank-Wolfe) on pseudo-control data
  5. Compute SDID estimate with re-estimated weights
  6. Repeat `replications` times (default 200)
  7. `SE = sqrt((r-1)/r) × sd(placebo_estimates)` where r = number of successful replications

  This matches R's `synthdid::vcov(method="placebo")` which passes `update.omega=TRUE, update.lambda=TRUE` via `opts`.

- Alternative: Bootstrap at unit level — paper-faithful refit (`variance_method="bootstrap"`)
  Arkhangelsky et al. (2021) Algorithm 2 step 2 verbatim, and also the behavior of R's default `synthdid::vcov(method="bootstrap")`. On each pairs-bootstrap draw:
  1. Resample ALL units (control + treated) with replacement.
  2. Identify which resampled units are control vs treated.
  3. Re-estimate `ω̂_b = compute_sdid_unit_weights(Y_boot_pre_c, Y_boot_pre_t_mean, zeta_omega=ζ_ω / Y_scale, init_weights=_sum_normalize(ω̂[boot_control_idx]))` — two-pass sparsified Frank-Wolfe, **warm-started from the fit-time ω renormalized over the resampled controls** (matching R's `sum_normalize(weights$omega[sort(ind[ind <= N0])])` shape).
  4. Re-estimate `λ̂_b = compute_time_weights(Y_boot_pre_c, Y_boot_post_c, zeta_lambda=ζ_λ / Y_scale, init_weights=λ̂)` — **warm-started from the fit-time λ unchanged** (matching R's `weights.boot$lambda = weights$lambda` shape).
  5. Compute SDID estimate with refit `ω̂_b` and `λ̂_b`.
  6. `SE = sqrt((r-1)/r) × sd(bootstrap_estimates, ddof=1)` where `r = n_successful` (equivalent to the paper's `σ̂² = (1/r) Σ (τ_b − τ̄)²`).

  R-parity rationale: `synthdid_estimate()` (synthdid.R) stores `update.omega = TRUE` in `attr(estimate, "opts")`, and `vcov.R::bootstrap_sample` rebinds those `opts` inside its `do.call` back into `synthdid_estimate`, so the renormalized ω passed via `weights$omega` is used as Frank-Wolfe initialization (the `sum_normalize` helper in R's source explicitly says so). The Python path threads the same warm-start via `compute_sdid_unit_weights(..., init_weights=...)` and `compute_time_weights(..., init_weights=...)`. The FW objective is strictly convex on the simplex (quadratic loss + ζ² ridge on simplex), so warm- and cold-start converge to the same global minimum given enough iterations; warm-start matters in practice because the 100-iter first pass then sparsification is path-dependent on draws where the pre-sparsify budget is tight. Cross-language SE parity at bit tolerance is not claimed — different BLAS / RNG paths — but the procedure matches R's default bootstrap shape at the algorithm level, and Python-only bit-identity on non-survey data is asserted via `TestScaleEquivariance::test_baseline_parity_small_scale[bootstrap]` at `rel=1e-14`.

  Expected wall-clock ~5–30× slower per fit than placebo (panel-size dependent; Frank-Wolfe second-pass can hit its 10K-iter cap on larger panels; warm-start plumbing closes the gap vs cold-start, which would be closer to 10–30× on these DGPs). Per-draw Frank-Wolfe non-convergence UserWarnings are suppressed inside the loop and aggregated into a single summary warning emitted after the loop when the share of valid bootstrap draws with any non-convergence event (counted once per draw — each draw runs Frank-Wolfe once for ω and once for λ, and any of those calls firing a non-convergence warning trips the draw) exceeds 5% of `n_successful`. Composed with survey designs (pweight-only OR strata/PSU/FPC) this path uses the weighted Frank-Wolfe kernel and the per-draw dispatch described in the "Note (survey + bootstrap composition)" below.

- Alternative: Jackknife variance (matching R's `synthdid::vcov(method="jackknife")`)
  Implements Algorithm 3 from Arkhangelsky et al. (2021):
  1. For each control unit j=1,...,N_co:
     - Remove unit j, renormalize omega: `ω_jk = _sum_normalize(ω[remaining])`
     - Keep λ unchanged, keep treated means unchanged
     - Compute SDID estimate τ_{(-j)}
  2. For each treated unit k=1,...,N_tr:
     - Keep ω and λ unchanged
     - Recompute treated mean from remaining N_tr-1 treated units
     - Compute SDID estimate τ_{(-k)}
  3. `SE = sqrt( ((n-1)/n) × Σ (τ_{(-i)} - τ̄)² )` where n = N_co + N_tr

  Fixed weights: No Frank-Wolfe re-estimation (`update.omega=FALSE, update.lambda=FALSE`).
  Returns NaN SE for single treated unit or single nonzero-weight control.
  Deterministic: exactly N_co + N_tr iterations, no replications parameter.
  P-value: analytical (normal distribution), not empirical.

*Edge cases:*
- **Frank-Wolfe non-convergence**: Returns current weights after max_iter iterations when the convergence check `vals[t-1] - vals[t] < min_decrease²` never triggers early exit. On `variance_method="bootstrap"` specifically, `_bootstrap_se` uses the `sc_weight_fw_with_convergence` Rust entry point (or `_sc_weight_fw_numpy(return_convergence=True)` on the pure-Python backend) to thread a convergence bool out of every per-draw FW pass; draws with any non-convergence are tallied and, if the rate exceeds 5% of valid draws, aggregated into a single `UserWarning` after the loop (avoids 200+ warnings per fit). Standalone calls to `_sc_weight_fw` / `compute_sdid_unit_weights` / `compute_time_weights` that do not pass `return_convergence=True` follow the legacy behavior: the numpy path emits a per-call `UserWarning` via `diff_diff.utils.warn_if_not_converged` and the Rust path silently returns the final iterate.
- **`_sparsify` all-zero input**: If `max(v) <= 0`, returns uniform weights `ones(len(v)) / len(v)`.
- **Single control unit**: `compute_sdid_unit_weights` returns `[1.0]` immediately (short-circuit before Frank-Wolfe).
- **Zero control units**: `compute_sdid_unit_weights` returns empty array `[]`.
- **Single pre-period**: `compute_time_weights` returns `[1.0]` when `n_pre <= 1` (Frank-Wolfe on a 1-element simplex is trivial).
- **Bootstrap with 0 control or 0 treated in resample (or non-finite `τ_b`)**: retry the draw. Matches R's `synthdid::bootstrap_sample` (`while (count < replications) { ...; if (!is.na(est)) count = count + 1 }`) and paper Algorithm 2 (B bootstrap replicates). A bounded attempt guard of `20 × n_bootstrap` prevents pathological-input hangs; normal fits finish far inside this budget because degenerate-draw probability scales as `(N_co / N)^N + (N_tr / N)^N`, which is small for any non-trivial split. If the budget is exhausted with 0 successful draws, raises `ValueError`. With 1 successful draw, warns and returns `SE = 0.0`. With fewer than `n_bootstrap` valid draws, warns that the attempt budget was exhausted and SE may be unreliable.
- **Placebo with n_control <= n_treated**: Warns that not enough control units for placebo variance estimation, returns SE=0.0 and empty placebo effects array. The check is `n_control - n_treated < 1`.
- **Note:** Power analysis functions (`simulate_power`, `simulate_mde`, `simulate_sample_size`) raise `ValueError` for placebo variance when `n_control <= n_treated`. The registry path checks pre-generation using `n_units * treatment_fraction`; the custom-DGP path checks post-generation on the realized data (first iteration only, since treatment allocation is deterministic per `n_units`/`treatment_fraction`).
- **Negative weights attempted**: Frank-Wolfe operates on the simplex (non-negative, sum-to-1), so weights are always feasible by construction. The step size is clipped to [0, 1] and the move is toward a simplex vertex.
- **Perfect pre-treatment fit**: Regularization (ζ² ||ω||²) prevents overfitting by penalizing weight concentration.
- **Single treated unit**: Valid; placebo variance uses jackknife-style permutations of controls.
- **Noise level with < 2 pre-periods**: Returns 0.0, which makes both zeta_omega and zeta_lambda equal to 0.0 (no regularization). **Note (deviation from R):** `min_decrease` uses a `1e-5` floor when `noise_level == 0` to enable Frank-Wolfe early stopping. R would use `0.0`, causing FW to run all `max_iter` iterations; the result is equivalent since zero-noise data has no variation to optimize.
- **NaN inference for undefined statistics**: t_stat uses NaN when SE is zero or non-finite; p_value and CI also NaN. Matches CallawaySantAnna NaN convention.
- **Placebo p-value floor**: `p_value = max(empirical_p, 1/(n_replications + 1))` to avoid reporting exactly zero.
- **Varying treatment within unit**: Raises `ValueError`. SDID requires block treatment (constant within each unit). Suggests CallawaySantAnna or ImputationDiD for staggered adoption.
- **Unbalanced panel**: Raises `ValueError`. SDID requires all units observed in all periods. Suggests `balance_panel()`.
- **Poor pre-treatment fit**: Warns (`UserWarning`) when `pre_fit_rmse > std(treated_pre_outcomes, ddof=1)`. Diagnostic only; estimation proceeds.
- **Jackknife with single treated unit**: Returns NaN SE. Cannot leave-one-out with N_tr=1; R returns NA for the same condition.
- **Jackknife with single nonzero-weight control**: Returns NaN SE. Leaving out the only effective control is not meaningful.
- **Jackknife with non-finite LOO estimate**: Returns NaN SE. Unlike bootstrap/placebo, jackknife is deterministic and cannot skip failed iterations; NaN propagates through `var()` (matches R behavior).
- **Jackknife with survey weights**: Guards on effective positive support (omega * w_control > 0 and w_treated > 0) after composition, not raw FW counts. Returns NaN SE if fewer than 2 effective controls or 2 positive-weight treated units. Per-iteration zero-sum guards return NaN for individual LOO iterations when remaining composed weights sum to zero.
- **Note (survey support matrix):**

  | variance_method | pweight-only | strata/PSU/FPC |
  |-----------------|:------------:|:--------------:|
  | `bootstrap`     | ✓ weighted FW | ✓ weighted FW + Rao-Wu rescaling (PR #355) |
  | `placebo`       | ✓             | ✓ stratified permutation + weighted FW |
  | `jackknife`     | ✓             | ✓ PSU-level LOO with stratum aggregation |

  **Pweight-only path** (placebo / jackknife / bootstrap): treated-side means are survey-weighted (Frank-Wolfe target and ATT formula); control-side synthetic weights are composed with survey weights post-optimization (ω_eff = ω * w_co, renormalized). Fit-time Frank-Wolfe is unweighted — survey importance enters after trajectory-matching. Covariate residualization uses WLS with survey weights.

  **Bootstrap survey path** (PR #355): for pweight-only the per-draw FW uses constant `rw = w_control`; for full design (strata/PSU/FPC) the per-draw `rw = generate_rao_wu_weights(resolved_survey, rng)` rescaling is composed with the same weighted-FW kernel. See "Note (survey + bootstrap composition)" below for the full objective and the argmin-set caveat.

  **Placebo survey path**: for pweight-only the existing Algorithm 4 flow applies with survey-weighted pseudo-treated means + post-hoc ω_eff composition. For designs with explicit `strata` and/or `psu` the allocator switches to **stratified permutation** (Pesarin 2001): pseudo-treated indices are drawn within each stratum containing actual treated units; weighted-FW re-estimates ω and λ per draw with per-control survey weights threaded into both loss and regularization. See "Note (survey + placebo composition)" below. **FPC is a documented no-op for placebo** — permutation tests are conditional on the observed sample (Pesarin 2001 §1.5), so the sampling fraction does not enter Algorithm 4 or its survey extension; an `fpc=` column on a placebo fit emits a `UserWarning` and is preserved in the design metadata but never enters the variance computation. Routing is gated on `strata` / `psu` only — FPC alone does not flip dispatch from the non-survey to the survey placebo path.

  **Jackknife survey path**: for pweight-only the existing Algorithm 3 flow applies (unit-level LOO with subset + rw-composed-renormalized ω; λ fixed). For full design the allocator switches to **PSU-level LOO with stratum aggregation** (Rust & Rao 1996): leave out one PSU at a time within each stratum, aggregate as `SE² = Σ_h (1-f_h)·(n_h-1)/n_h·Σ_{j∈h}(τ̂_{(h,j)} - τ̄_h)²`. See "Note (survey + jackknife composition)" below.

  **Allocator asymmetry** (placebo ignores PSU axis; jackknife respects it): intentional. Placebo is a null-distribution test — within-stratum unit-level permutation is the classical stratified permutation test (Pesarin 2001 Ch. 3-4); PSU-level permutation on few PSUs (2-8 typical for survey designs) produces near-degenerate permutation support and poor power. Jackknife is a design-based variance approximation — PSU-level LOO within strata is the canonical survey jackknife (Rust & Rao 1996); unit-level LOO under clustering would underestimate SE. Both allocators respect strata (the primary survey-design axis). Neither is "right" in all dimensions; each is the defensible analog for its hypothesis-testing vs variance-approximation role.
- **Note (default variance_method deviation from R):** R's `synthdid::vcov()` defaults to `method="bootstrap"`; our `SyntheticDiD.__init__` defaults to `variance_method="placebo"`. Library deviation rationale: placebo avoids the ~5–30× per-draw Frank-Wolfe refit slowdown. All three variance methods (placebo, bootstrap, jackknife) now support both pweight-only and full strata/PSU/FPC survey designs (see the survey support matrix above); users can opt into R's default with `variance_method="bootstrap"`, which is also the recommended choice on surveys with few PSUs per stratum (jackknife is anti-conservative in that regime per the "Note (survey + jackknife composition)" above). Placebo (Algorithm 4) and bootstrap (Algorithm 2 step 2) both track nominal calibration in the committed coverage MC; see the calibration table below.
- **Note (survey + bootstrap composition — PR #355):** Restored capability. The bootstrap survey path solves the **weighted Frank-Wolfe** variant of `_sc_weight_fw` accepting per-unit weights in loss and regularization. For unit weights:
  ```
  min_{ω simplex}  Σ_t (Σ_i rw_i · ω_i · Y_i,pre[t] - treated_pre[t])²  +  ζ²·Σ_i rw_i · ω_i²
  ```
  Implementation: column-scales A by `rw_control` (so the loss term reads `||A·diag(rw)·ω - b||²`) and passes `reg_weights=rw_control` to the weighted Rust kernel for the diag(rw) penalty. The FW returns ω on the standard simplex; `_bootstrap_se` composes `ω_eff = rw·ω / Σ(rw·ω)` for the downstream `compute_sdid_estimator` call (which expects a normalized weight vector). For time weights, the loss becomes per-row weighted (`||diag(√rw)·(A·λ - b)||²`) and regularization on λ stays uniform — λ is per-period, rw is per-control, no alignment for per-λ reg weighting.

  **Argmin-set caveat (deliberate):** this objective is NOT the same as directly minimizing the standard SDID loss on `ω_eff` (the scaling factor `(Σ rw·ω)²` enters the loss-on-θ reparameterization non-constantly). The chosen form mirrors the spirit of the pre-PR-#351 Rao-Wu fixed-weight composition (rescale + renormalize) but with ω re-estimated per draw under the weighted objective, so weight-estimation uncertainty propagates correctly. No external R/Julia parity anchor exists because neither package defines survey-weighted SDID FW; validation rests on the coverage MC calibration row below (`stratified_survey × bootstrap`, target rejection ∈ [0.02, 0.10] at α=0.05).

  **Per-draw dispatch:**
  - pweight-only → `rw = w_control[boot_idx_control]` (constant per draw, no Rao-Wu).
  - full design → `rw = generate_rao_wu_weights(resolved_survey_unit, rng)` per draw, sliced over the resampled units. Rao-Wu rescales weights by `(n_h/m_h)·r_hi` within each stratum; degenerate-retry on zero-mass control or treated draws.
  - **single-PSU short-circuit**: unstratified single-PSU designs return NaN SE (resampling one PSU yields the same subset every draw — bootstrap distribution is unidentified).
- **Note (survey + placebo composition):** Stratified-permutation allocator composed with the same weighted Frank-Wolfe kernel from the bootstrap survey path. Each placebo draw:
  1. For each stratum `h` containing actual treated units, draws `n_treated_h` pseudo-treated indices uniformly without replacement from `controls_in_h`. Non-treated strata contribute their controls unconditionally to the pseudo-control set.
  2. Pseudo-treated means are survey-weighted: `Y_pseudo_t = np.average(Y[:, pseudo_treated_idx], weights=w_control[pseudo_treated_idx])`.
  3. Weighted Frank-Wolfe re-estimates ω and λ on the pseudo-panel using `compute_sdid_unit_weights_survey(rw_control=w_control[pseudo_control_idx], ...)` and `compute_time_weights_survey(...)`. Post-optimization composition `ω_eff = rw·ω/Σ(rw·ω)` with zero-mass retry.
  4. SDID estimator on the pseudo-panel; Algorithm 4 SE `sqrt((r-1)/r)·std(placebo_estimates, ddof=1)`.

  **Fit-time feasibility guards** (per `feedback_front_door_over_retry_swallow.md`): four distinct failure cases are rejected *before* entering the retry loop, each with a targeted `ValueError`:
  * **Case B** (`n_controls_h == 0` for some treated-containing stratum): the stratum has treated units but no controls — no pseudo-treated set can be drawn.
  * **Case C** (`0 < n_controls_h < n_treated_h`): the stratum has fewer controls than treated units, so exact-count without-replacement sampling is impossible.
  * **Case E** (row-count guards passed but `n_positive_weight_controls_h < n_treated_h`): the stratum has enough raw controls but too few have positive survey weight. Since the pseudo-treated mean uses `np.average(Y, weights=w_control[idx])`, draws can pick all-zero-weight subsets (ZeroDivisionError on np.average) and the retry loop would swallow them as a generic ``n_successful=0`` warning + ``SE=0.0``.
  * **Case D** (effective single-support — *every* treated stratum collapses to one positive-mass mean): two shapes trigger this. **(D-classical)** `n_controls_h == n_treated_h` so the without-replacement permutation has only one subset. **(D-effective)** `n_c_h > n_t_h` (raw count allows multiple subsets) but `n_positive_weight_controls_h < 2` — every successful pseudo-treated mean reduces to the unique positive-weight control's outcome (zero-weight cohabitants contribute 0 to numerator and denominator). Both shapes give a degenerate null (SE = FP noise ~1e-16). Non-degeneracy requires **both** `n_c_h > n_t_h` AND `n_positive_weight_controls_h >= 2` for at least one treated stratum.

  Partial-permutation fallback is rejected for all four cases — it would silently change the null distribution and produce an incoherent test.

  **Scope note — what is NOT randomized:** the stratum marginal is preserved exactly by construction (each draw pulls the same count per treated stratum). The PSU axis is not randomized (permutation is unit-level within strata). This is conservative under clustering (ignores within-stratum PSU correlation in the null) but aligns with the classical stratified permutation test literature. See Pesarin (2001) *Multivariate Permutation Tests*, Ch. 3-4; Pesarin & Salmaso (2010) *Permutation Tests for Complex Data*.

  **Validation:** no external R/Julia parity anchor (neither package defines survey-weighted SDID placebo). Correctness rests on: (a) stratum-membership contract enforced by construction + per-draw `rng.choice` interception regression that captures every actual sampled `pseudo_treated_idx` and asserts each sampled control's stratum membership ⊆ treated-strata set, (b) Case B / C / D / E front-door guards with targeted-message regression tests, (c) SE-differs-from-pweight-only cross-surface sanity, (d) deterministic-dispatch regression.

- **Note (survey + jackknife composition):** PSU-level leave-one-out with stratum aggregation (Rust & Rao 1996). For a design with strata `h = 1..H` and PSUs `j = 1..n_h` within each stratum:

  ```
  SE² = Σ_h (1 - f_h) · (n_h - 1)/n_h · Σ_{j∈h} (τ̂_{(h,j)} - τ̄_h)²
  ```

  where `τ̂_{(h,j)}` is the SDID estimator computed after leaving out all units in PSU `j` of stratum `h`; `τ̄_h` is the stratum-level mean of successful LOO estimates; `f_h = n_h_sampled / fpc[h]` is the per-stratum sampling fraction. FPC is stored as a per-unit **population-count** array by `SurveyDesign.resolve` (see `survey.py:338-356`, where `fpc_h < n_psu_h` is the validation constraint), so `f_h` is recovered by `f_h = n_h / fpc[strata == h][0]`. No FPC → `f_h = 0`.

  **Fixed weights per LOO:** matches Algorithm 3 of Arkhangelsky et al. (2021). ω is subsetted over kept controls, composed with kept `w_control`, renormalized (`ω_eff_kept = rw·ω / Σ(rw·ω)`); λ is held at the fit-time value. Rationale: jackknife is a design-based variance approximation, not a refit-variance bootstrap. Re-estimating λ or ω per LOO would conflate weight-estimation uncertainty (bootstrap's domain) with sampling uncertainty (jackknife's domain).

  **Undefined-replicate handling** (return NaN, do NOT silently skip): the Rust & Rao formula requires `τ̂_{(h,j)}` be defined for every PSU `j` in every contributing stratum. If any single LOO in a contributing stratum (`n_h ≥ 2`) is not computable — (a) deletion removes all treated units (e.g., all treated in one PSU), (b) `ω_eff_kept.sum() ≤ 0` after composition, (c) `w_treated_kept.sum() ≤ 0`, (d) the SDID estimator raises or returns non-finite τ̂ — the overall SE is **undefined** and the method returns `SE=NaN` with a targeted `UserWarning` naming the stratum / PSU / reason. Silently skipping the missing LOO while still applying the `(n_h-1)/n_h` factor would systematically under-scale variance (silently wrong SE). Users needing a variance estimator that accommodates PSU-deletion infeasibility should use `variance_method="bootstrap"`, whose pairs-bootstrap has no per-LOO feasibility constraint.

  **Zero-variance vs undefined distinction:** when every stratum contributes but `total_variance == 0.0` by legitimate design — full-census FPC (`f_h = 1` → `(1 - f_h) = 0` zeros the contribution even when within-stratum dispersion is non-zero) or exact-zero within-stratum dispersion — the jackknife SE is **zero**, not undefined. `_jackknife_se_survey` returns `SE = 0.0` in that case. `SE = NaN` is reserved for the truly-undefined cases documented above (all strata skipped; any undefined delete-one replicate).

  **`lonely_psu` contract:** `SurveyDesign(lonely_psu="remove")` (default) and `"certainty"` are both accepted, but with **different semantics** when *every* stratum is singleton:
  * `"remove"` silently skips singleton strata (matches R `survey::svyjkn` — they're dropped from the variance computation). If every stratum is skipped, returns ``SE = NaN`` with the "every stratum was skipped" warning (no contributing stratum, undefined).
  * `"certainty"` treats singleton strata as **explicit zero-variance contributors** (sampled with certainty, no sampling variance). Singleton strata still contribute 0 to total variance, but the stratum counts as "contributing" to the overall design — so an all-singleton design returns ``SE = 0.0`` (legitimate zero variance), not NaN. Mirrors `compute_survey_vcov`'s ``test_all_certainty_psu_zero_vcov`` contract for other estimators.

  `lonely_psu="adjust"` (R's overall-mean fallback) is **not yet supported** on the SDID jackknife path and raises `NotImplementedError` at fit-time; users needing that semantic should pick `variance_method="bootstrap"` (which supports all three modes via the weighted-FW + Rao-Wu path) or switch the design to `"remove"` / `"certainty"`.

  **Stratum-skip handling** (silent, documented): strata with `n_h < 2` are silently skipped (stratum-level variance unidentified — the `lonely-PSU` case in R `survey::svyjkn`). If every stratum is skipped, returns `SE=NaN` with a separate `UserWarning`. PSU-None designs: each unit is treated as its own PSU within its stratum (matches the implicit-PSU convention established in PR #355 R8 P1). Unstratified single-PSU short-circuits to `SE=NaN`.

  **Scope note — what is NOT randomized:** stratum membership and PSU composition are fixed by design. The formula only captures within-stratum variation; between-stratum variance is absorbed into the analytical-TSL / design assumption. This is canonical survey-jackknife behavior (Rust & Rao 1996) and matches R's `survey::svyjkn` under stratified designs.

  **Known limitation — anti-conservatism with few PSUs per stratum:** with `n_h = 2` per stratum (the minimum for variance identifiability), within-stratum jackknife has only 1 effective DoF per stratum — a well-documented limitation of the stratified jackknife formula. On the coverage MC `stratified_survey` DGP (2 PSUs × 2 strata), `se_over_truesd ≈ 0.46` at α=0.05. **Users needing tight SE calibration with few PSUs should prefer `variance_method="bootstrap"`**, which validates at near-nominal calibration on the same DGP.

  **Validation:** (a) hand-computed 2-stratum FPC magnitude regression (`test_jackknife_full_design_fpc_reduces_se_magnitude` — asserts `SE_fpc == SE_nofpc · sqrt(1 - f)` at `rtol=1e-10`), (b) self-consistency between the returned SE and the stratum-aggregation formula applied to the returned LOO estimates, (c) single-PSU-stratum skip, (d) all-strata-skipped UserWarning + NaN, (e) unstratified single-PSU short-circuit, (f) deterministic-dispatch regression.

- **Note:** P-value computation is variance-method dependent. Placebo (Algorithm 4) uses the empirical null formula `max(mean(|variance_effects| ≥ |att|), 1/(r+1))` because permuting control indices generates draws from the null distribution (centered on 0). Bootstrap (Algorithm 2) and jackknife (Algorithm 3) use the analytical p-value from `safe_inference(att, se)` (normal-theory): bootstrap draws are centered on `τ̂` (sampling distribution of the estimator) and jackknife pseudo-values are not null draws, so the empirical null formula is invalid for them. This matches R's `synthdid::vcov()` convention, where variance is returned and inference is normal-theory from the SE.
- **Note (benchmark SE gate is Monte Carlo-bounded, 2026-07):** The public
  benchmark refresh (`benchmarks/refresh_2026_07/`) compares placebo SEs
  against R `synthdid` across INDEPENDENT placebo draws: R's
  `vcov(method="placebo")` does not seed its permutation sequence, so the two
  implementations agree draw-by-draw only when the permutation sequence is
  threaded explicitly (verified at < 1e-8 SE tolerance in
  `tests/test_methodology_sdid.py::TestJackknifeSERParity::test_placebo_se_matches_r`);
  across independent draws the SEs agree in distribution, not point-by-point.
  The benchmark harness therefore hard-gates SyntheticDiD SEs at a 35%
  relative bound (vs the standard 10% used for analytical-SE estimators),
  records R's rep-to-rep SE values in the committed results artifact so the
  observed Monte Carlo dispersion is auditable, and gates the deterministic
  Frank-Wolfe ATT at 1e-8. The generated benchmark summary table carries the
  same footnote. **Weight-vector gates:** unit (ω) and time (λ) weight
  vectors both reproduce R at machine precision (observed ≤ 5e-13 on the
  benchmark small fixture) and are hard-gated at 1e-8 max abs diff, aligned
  by unit/period id. The id alignment is load-bearing:
  `SyntheticDiDResults.get_unit_weights_df()` returns weights sorted by
  DESCENDING WEIGHT while R's `synthdid` emits panel control-row order, so a
  naive positional comparison of the two vectors fabricates apparent ~1e-2
  divergence out of pure ordering. The benchmark scripts emit id-keyed
  vectors on both sides and the comparator aligns on ids (fail-closed on key
  mismatch/duplicates). Per-run max-abs-diff metrics for both vectors are
  committed with the results.

- **Note (coverage Monte Carlo calibration):** `benchmarks/data/sdid_coverage.json` carries empirical rejection rates across the three variance methods on 4 representative null-panel DGPs (500 seeds × B=200, regenerable via `benchmarks/python/coverage_sdid.py`). The fourth DGP (`stratified_survey`, added in PR #355) validates the survey-bootstrap calibration; jackknife is also reported with a documented anti-conservatism caveat; placebo is N/A on this DGP because its cohort packs into a single stratum with 0 never-treated units (stratified-permutation allocator is structurally infeasible — see `test_placebo_full_design_raises_on_zero_control_stratum` / `_undersupplied_stratum` for the enforced behavior). Under H0 the nominal rejection rate at each α equals α; rates substantially above α indicate anti-conservatism, rates below indicate over-coverage.

    | DGP                                                       | method     | α=0.01 | α=0.05 | α=0.10 | mean SE / true SD |
    |-----------------------------------------------------------|------------|--------|--------|--------|-------------------|
    | balanced (N_co=20, N_tr=3, T_pre=8, T_post=4)             | placebo    | 0.016  | 0.060  | 0.086  | 1.05 |
    | balanced                                                  | bootstrap  | 0.028  | 0.078  | 0.116  | 1.05 |
    | balanced                                                  | jackknife  | 0.066  | 0.112  | 0.154  | 1.08 |
    | unbalanced (N_co=30, N_tr=8, heterogeneous unit-FE)       | placebo    | 0.006  | 0.032  | 0.070  | 1.08 |
    | unbalanced                                                | bootstrap  | 0.008  | 0.038  | 0.080  | 1.11 |
    | unbalanced                                                | jackknife  | 0.024  | 0.076  | 0.120  | 0.99 |
    | AER §6.3 (N=100, N_tr=20, T=120, T_pre=115, rank=2, σ=2)  | placebo    | 0.018  | 0.058  | 0.086  | 0.99 |
    | AER §6.3                                                  | bootstrap  | 0.010  | 0.040  | 0.078  | 1.05 |
    | AER §6.3                                                  | jackknife  | 0.030  | 0.080  | 0.150  | 0.90 |
    | stratified_survey (N=40, strata=2, PSU=2/stratum, ICC≈0.84) | bootstrap  | 0.024  | 0.058  | 0.094  | 1.13 |
    | stratified_survey                                         | jackknife  | 0.358  | 0.450  | 0.512  | 0.46 |

    Reading: **`bootstrap` (paper-faithful refit)** and **`placebo`** both track nominal calibration across all three non-survey DGPs (rates within Monte Carlo noise at 500 seeds; 2σ MC band ≈ 0.02–0.05 at p ≈ 0.05–0.10). **`jackknife`** is slightly anti-conservative on the smaller panels (balanced, AER §6.3) at α=0.05 (rejection 0.112 and 0.080 vs the 0.05 target). Arkhangelsky et al. (2021) §6.3 reports mixed jackknife evidence (98% coverage — slightly conservative — under iid, and 93% coverage — slightly anti-conservative — under AR(1) ρ=0.7), so the direction of our observation is consistent with the AR(1) branch of the paper's evidence rather than the iid branch. The `mean SE / true SD` column compares mean estimated SE to the empirical sampling SD of τ̂ across seeds.

    **`stratified_survey × bootstrap` (PR #355)**: validates the weighted-FW + Rao-Wu composition added in that PR. Rejection at α=0.05 is 0.058 (inside the calibration gate [0.02, 0.10] widened from a 2σ band to accommodate the high ICC ≈ 0.84 induced by `psu_re_sd=1.5` with only 4 PSUs total). `mean SE / true SD = 1.13` indicates the bootstrap is slightly conservative (overestimates the empirical sampling SD by ~13%) — the safer direction; expected under Rao-Wu rescaling with few PSUs because the per-draw weights inflate variance from the resampling structure on top of the fit-time uncertainty.

    **`stratified_survey × jackknife`**: reported with an anti-conservative caveat. Rejection at α=0.05 is 0.450 (far outside any reasonable calibration gate) and `se_over_truesd ≈ 0.46`. This is the documented limitation of the stratified PSU-level jackknife formula with `n_h = 2` PSUs per stratum: within-stratum variance has only 1 effective DoF per stratum, and between-stratum variation is absorbed into the design assumption rather than the SE. The bootstrap row on the same DGP demonstrates that the fix is to pick `variance_method="bootstrap"` when the design has few PSUs per stratum. This row is committed for transparency; the methodology Note above (§"Note (survey + jackknife composition)") explicitly flags this regime and recommends bootstrap.

    **`stratified_survey × placebo`**: N/A on this DGP by construction (its cohort packs all treated units into stratum 1, which has 0 never-treated units, so the stratified-permutation allocator raises **Case B** — treated-containing stratum with zero controls — at fit-time; see Case B / C definitions in "Note (survey + placebo composition)" above). The placebo survey path is exercised under feasible structures in `tests/test_survey_phase5.py::TestSDIDSurveyPlaceboFullDesign`; calibration on a placebo-feasible DGP is a future MC extension.

    The schema smoke test is `TestCoverageMCArtifact::test_coverage_artifacts_present`; regenerate the JSON via `python benchmarks/python/coverage_sdid.py --n-seeds 500 --n-bootstrap 200 --output benchmarks/data/sdid_coverage.json` (~15–40 min on M-series Mac, Rust backend — warm-start convergence makes newer runs faster than the original cold-start one).

    **Artifact cadence (documentation-grade, not pinned-regression-grade):** this file is a documentation substrate — the table above transcribes from it, and the schema test just checks structure. It is **not** numerically pinned by any regression test, because MC noise at 500 seeds × B=200 (2σ ≈ 0.02–0.05 per cell) makes tight bounds fragile and loose bounds uninformative. The runtime characterization test that guards the one regression class worth catching (dispatch breakage → rejection rate ≈0 or ≈0.5, or SE-collapse) is `TestPValueSemantics::test_bootstrap_p_value_null_dispersion` (slow; calibration-agnostic dispersion + loose rejection-rate band). Regenerate the JSON when a methodology change materially shifts per-draw numerics — SE formula, new variance method, FW solver swap, paper-procedure correction. **Do not** regenerate on refactors that preserve the FW global optimum (warm-start vs cold-start, backend migration, pure renames, docstring fixes) — those stay within MC noise of the committed numbers. Per-seed bit-identity on the captured fixture is the cheaper, stricter check: see `TestScaleEquivariance::test_baseline_parity_small_scale[bootstrap]` at `rel=1e-14`.
- **Note:** Internal Y normalization. Before weight optimization, the estimator, and variance procedures, `fit()` centers Y by `mean(Y_pre_control)` and scales by `std(Y_pre_control)`; `Y_scale` falls back to `1.0` when std is non-finite or below `1e-12 * max(|mean|, 1)`. Auto-regularization and `noise_level` are computed on normalized Y; user-supplied `zeta_omega` / `zeta_lambda` are divided by `Y_scale` internally for Frank-Wolfe. τ, SE, CI, the placebo/bootstrap/jackknife effect vectors, `results_.noise_level`, and `results_.zeta_omega` / `results_.zeta_lambda` are all reported on the user's original outcome scale (user-supplied zetas are echoed back exactly to avoid float roundoff). Mathematically a no-op — τ is location-invariant and scale-equivariant, and FW weights are invariant under `(Y, ζ) → (Y/s, ζ/s)` — but prevents catastrophic cancellation in the SDID double-difference when outcomes span millions-to-billions (see synth-inference/synthdid#71 for the R-package version of this issue). Normalization constants are derived from controls' pre-period only so the reference is unaffected by treatment. `in_time_placebo()` and `sensitivity_to_zeta_omega()` reuse the exact same `Y_shift` / `Y_scale` captured on the fit snapshot: they normalize the re-sliced arrays before re-running Frank-Wolfe, pass `zeta / Y_scale` to the weight solvers, and rescale the returned `att` and `pre_fit_rmse` by `Y_scale` before reporting; unit-weight diagnostics (`max_unit_weight`, `effective_n`) are scale-invariant and reported directly.

*Validation diagnostics (post-fit methods on `SyntheticDiDResults`):*

- **Trajectories** (`synthetic_pre_trajectory`, `synthetic_post_trajectory`, `treated_pre_trajectory`, `treated_post_trajectory`): retained on results to support plotting and custom fit metrics. `synthetic_pre_trajectory = Y_pre_control @ ω_eff`; `treated_pre_trajectory` is the survey-weighted treated mean (matches the Frank-Wolfe target). `pre_treatment_fit` is recoverable as `RMSE(treated_pre_trajectory, synthetic_pre_trajectory)`.
- **`get_loo_effects_df()`**: user-facing join of the jackknife leave-one-out pseudo-values (stored in `variance_effects`) to the underlying unit identities. **Unit-level LOO only** — available on the non-survey and pweight-only jackknife paths (classical Algorithm 3: one LOO per unit, first `n_control` positions map to `control_unit_ids`, next `n_treated` to `treated_unit_ids`; `att_loo` is NaN when the zero-sum composed-weight guard fired for that unit; `delta_from_full = att_loo - att`). Under the full-design survey jackknife path (PSU-level LOO with stratum aggregation, Rust & Rao 1996), the underlying replicates are PSU-level rather than unit-level — the accessor raises `NotImplementedError` pointing to `result.variance_effects` for the raw PSU-level replicate array. Dispatch is gated by an explicit `_loo_granularity` flag set at fit-time (`"unit"` vs `"psu"`). Requires `variance_method='jackknife'`; raises `ValueError` otherwise.
- **`get_weight_concentration(top_k=5)`**: returns `effective_n = 1/Σω²` (inverse Herfindahl), `herfindahl`, `top_k_share`, `top_k`. Operates on `self.unit_weights` which stores the composed `ω_eff`; for survey-weighted fits the metrics reflect the population-weighted concentration, not the raw Frank-Wolfe solution.
- **`in_time_placebo(fake_treatment_periods=None, zeta_omega_override=None, zeta_lambda_override=None)`**: re-slices the pre-window at each fake treatment period and re-fits both ω and λ via Frank-Wolfe. Default sweeps every feasible pre-period (position index `i ≥ 2` so ≥2 pre-fake periods remain for weight estimation, `i ≤ n_pre - 1` so ≥1 post-fake period exists). Credible designs produce near-zero placebo ATTs; departures indicate pre-treatment dynamics the estimator is picking up.
  - **Note:** Regularization reuses `self.zeta_omega` / `self.zeta_lambda` from the original fit (matches R `synthdid` convention of treating regularization as a property of the fit). `*_override` re-fits with new values.
  - **Note:** Infeasibility-only NaN — the method emits NaN for dimensional infeasibility (e.g., survey composition producing zero weight sum on the fake window); Frank-Wolfe non-convergence is not detectable mid-solver, so `pre_fit_rmse` is the user-facing signal for poor refit quality. Passing a `fake_treatment_period` in `post_periods` raises `ValueError` (not a placebo).
- **`sensitivity_to_zeta_omega(zeta_grid=None, multipliers=(0.25, 0.5, 1.0, 2.0, 4.0))`**: re-fits ω at each zeta value on the original pre-window. Default grid is `multipliers * self.zeta_omega` — a 5-point grid spanning 16x from smallest to largest multiplier, symmetric in log space around 1.0. Returns `att`, `pre_fit_rmse`, `max_unit_weight`, `effective_n` per row.
  - **Note:** Time weights are held fixed at the original Frank-Wolfe output (`self.time_weights_array`), not re-fit. This isolates sensitivity to `zeta_omega` specifically; sensitivity to `zeta_lambda` is not currently exposed.
  - **Note:** At `multiplier=1.0` (or `zeta_grid` containing `self.zeta_omega`), the ATT reproduces `self.att` to machine precision with the same seeded draw.

**Reference implementation(s):**
- R: `synthdid::synthdid_estimate()` (Arkhangelsky et al.'s official package)
- Key R functions matched: `sc.weight.fw()` (Frank-Wolfe), `sparsify_function` (sparsification), `vcov.synthdid_estimate()` (variance)

**Requirements checklist:**
- [x] Unit weights: Frank-Wolfe on collapsed form (T_pre, N_co+1), two-pass sparsification (100 iters -> sparsify -> 10000 iters)
- [x] Time weights: Frank-Wolfe on collapsed form (N_co, T_pre+1), last column = per-control post mean
- [x] Unit and time weights: sum to 1, non-negative (simplex constraint)
- [x] Auto-regularization: noise_level = sd(first_diffs), zeta_omega = (N1*T1)^0.25 * noise_level, zeta_lambda = 1e-6 * noise_level
- [x] Sparsification: v[v <= max(v)/4] = 0; v = v/sum(v)
- [x] Placebo SE formula: sqrt((r-1)/r) * sd(placebo_estimates)
- [x] Placebo SE: re-estimates omega and lambda per replication (matching R's update.omega=TRUE, update.lambda=TRUE), with R-default warm-start (`init_omega = sum_normalize(unit_weights[pseudo_control_idx])` per draw; `init_lambda = time_weights`) matching `synthdid:::placebo_se`'s `weights.boot$omega = sum_normalize(weights$omega[ind[1:N0_placebo]])` pattern. R-parity verified at `< 1e-8` SE tolerance via `tests/test_methodology_sdid.py::TestJackknifeSERParity::test_placebo_se_matches_r` using R's exact permutation sequence threaded through the `_placebo_indices` test seam.
- [x] Bootstrap: paper-faithful Algorithm 2 step 2 — re-estimates ω̂_b and λ̂_b per draw via two-pass sparsified Frank-Wolfe on the resampled panel using the fit-time normalized-scale zeta. Matches R's default `synthdid::vcov(method="bootstrap")` (which rebinds `attr(estimate, "opts")` so the renormalized ω serves only as Frank-Wolfe initialization). Survey designs (pweight-only AND strata/PSU/FPC) are supported via the weighted-FW + hybrid pairs-bootstrap + Rao-Wu rescaling composition described in the "Note (survey + bootstrap composition)" above (PR #355).
- [x] Placebo: Survey-weighted pseudo-treated means + weighted-FW re-estimation on pseudo-panel for both pweight-only and full-design paths. Full-design path (strata/PSU/FPC) uses stratified-permutation allocator — see "Note (survey + placebo composition)" above.
- [x] Jackknife SE: fixed weights, LOO all units, formula `sqrt((n-1)/n * sum((u-ubar)^2))`. Full-design path (strata/PSU/FPC) uses PSU-level LOO with stratum aggregation — see "Note (survey + jackknife composition)" above.
- [x] Jackknife: NaN SE for single treated or single nonzero-weight control
- [x] Jackknife: analytical p-value (not empirical)
- [x] Returns both unit and time weights for interpretation
- [x] Column centering (intercept=True) in Frank-Wolfe optimization

---

## SyntheticControl

**Primary source:** [Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control Methods for Comparative Case Studies: Estimating the Effect of California's Tobacco Control Program." *JASA*, 105(490), 493–505.](https://doi.org/10.1198/jasa.2009.ap08746) Method originates in Abadie & Gardeazabal (2003). Paper reviews on file: `docs/methodology/papers/abadie-diamond-hainmueller-2010-review.md` (primary), `...-2015-review.md`, `abadie-2021-review.md`, `chernozhukov-wuthrich-zhu-2021-review.md`, `firpo-possebom-2018-review.md` (inference: sensitivity analysis + confidence sets by test inversion).

Classic synthetic control (donor/unit weights only) for a single treated unit, distinct from `SyntheticDiD` (Arkhangelsky et al. 2021), which adds time weights and ridge regularization. Equation (1) of ADH 2010 shows classic SCM **generalizes the TWFE/DiD model** (recovered when the factor loadings `λ_t` are constant in time).

**Assumption checks / warnings:**
- **One treated unit, block (absorbing) assignment** after `T0`; remaining units are the never-exposed donor pool (Section 2.2). Rejects >1 ever-treated unit (pass `treated_unit=` + curate `donor_pool=`); rejects an ever-treated unit in the donor pool (contamination).
- **No anticipation / absorbing treatment.** `post_periods` must be a contiguous suffix of the time axis, cross-checked against the treated unit's `D` column (`D==1` in any pre period → `ValueError`), on both the inferred and explicit branches.
- **Good pre-treatment fit is required, not assumed** (journal p. 495). Emits a `UserWarning` when pre-period RMSPE exceeds the SD of the treated unit's pre-period outcomes.
- **No interference / SUTVA across units; donor-pool curation** (exclude units with their own intervention/large shocks) — analyst-supplied via `donor_pool=`.

**Estimator (Section 2.3):**
- Predictor matrices `X1` (k×1 treated) / `X0` (k×J donors) = covariates `Z` + linear combinations of pre-period outcomes (`predictors` averaged over `predictor_window`, `special_predictors`, and/or per-period outcome lags `pre_period_outcomes`). Canonical row order: predictor averages → special predictors → outcome lags (the row *order* matches R `Synth::dataprep`; aggregation semantics differ — see the `na.rm` deviation below).
- **Inner solve:** `W*(V) = argmin_W (X1 − X0 W)' diag(V) (X1 − X0 W)` s.t. `w_j ≥ 0, Σ w_j = 1`. Implemented by folding `V^½` into the predictors (`packed = [V^½·X0 | V^½·X1]`) and calling the Frank-Wolfe simplex solver `utils._sc_weight_fw(intercept=False, zeta=0)`.
- **`V` selection (`v_method`):** `"nested"` chooses diagonal PSD `V` minimizing the pre-period **outcome** MSPE `mean((Z1 − Z0·W*(V))²)` over all pre periods; `"cv"` chooses `V` by **out-of-sample cross-validation** (ADH 2015 §; Abadie 2021 Eq. 9 — see the per-window re-aggregation Note); `"inverse_variance"` uses the closed-form `v_h = 1/Var(X_{h·})` (Abadie 2021 §3.2(a); no search); `"custom"` skips the search and uses a user-supplied `custom_v` (trace-normalized). `mspe_v` reports the selected `V`'s objective value — the pre-period MSPE under `"nested"`, the held-out validation-window MSPE under `"cv"`, and `None` for the search-free `"custom"`/`"inverse_variance"` paths (not comparable across methods).
- **Effect:** gap path `α̂_1t = Y_1t − Σ_j w_j·Y_jt`; `att` = mean post-period gap; `pre_rmspe` = pre-period fit diagnostic.

**Inference:** **No analytical standard error** (Section 2.4) — `se`/`t_stat`/`p_value`/`conf_int` are always NaN. Significance comes from **in-space placebo permutation inference** via `SyntheticControlResults.in_space_placebo()`: reassign treatment to each donor, refit a synthetic control for it, and rank the treated unit's post/pre RMSPE ratio (`rmspe_ratio` = `RMSPE_post / RMSPE_pre` = `sqrt(MSPE_post / MSPE_pre)`) among all units; `placebo_p_value = rank / (n_placebos + 1)`, where `rank = 1 + #{placebos with ratio ≥ treated ratio}` — an **upper-tail rank test on the (unsigned) RMSPE-ratio statistic**, ties counted conservatively via `≥`. Because the ratio squares the gaps it is direction-agnostic: a large ratio signals an effect of *either* sign, so this is NOT a signed/one-directional ("one-sided") hypothesis test on the treatment-effect direction. ADH 2010 §3.4 reports the *MSPE* ratio (the square of `rmspe_ratio`); the two are monotone-equivalent, so the rank and p-value are identical — only the reported statistic's scale differs. `rmspe_ratio` (the treated statistic) is computed at fit time; `placebo_p_value` / `n_placebos` / `n_failed` / `n_infeasible` are populated by the opt-in `in_space_placebo()` call (`n_failed` = solver non-convergences, `n_infeasible` = structural `cv` exclusions — see the split Note below). `get_placebo_df()` returns the per-unit ratio table used for the rank (the per-period placebo gap paths for a "spaghetti" plot are retained internally, not in this summary).

**ADH-2015 §4 robustness diagnostics (opt-in):** Four further diagnostics from Abadie-Diamond-Hainmueller (2015, §4), each a thin re-run of the validated solver (the regression-weight diagnostic is pure linear algebra, no re-fit) — populated only when called and surfaced under `estimator_native_diagnostics` (the analytical inference contract is unchanged; `se`/`t_stat`/`p_value`/`conf_int`/`is_significant` stay bound to the NaN analytical `p_value`):
- **Leave-one-out donor robustness** (`leave_one_out()`): drops each **reportably-weighted** donor and re-fits the treated unit against the reduced pool, returning a per-drop ATT / `delta_att` table (a `status="baseline"` row first, then one row per dropped donor sorted by `|delta_att|`). A large `delta_att` flags single-donor dependence (a single *dominant* donor is still dropped — the others absorb its mass — and its large `delta_att` is the intended signal, not a failure). The reporting stack's headline donor-sensitivity number is `max_abs_delta_att` = `max |delta_att|` over the drops (baseline-relative, so a uniform shift of every drop away from the full-fit ATT is not masked the way a raw ATT range would be). `get_leave_one_out_gaps()` returns the per-drop trajectories for the overlay plot. Fails closed on a non-converged treated fit or `< 2` donors.
- **In-time (backdating) placebo** (`in_time_placebo()`): reassigns the intervention to an earlier pre-date `t_f`, re-fits using ONLY pre-`t_f` information (TRUNCATE convention — see Note), and reports the placebo "effect" over the held-out window `[t_f, T0)` — ~0 if there is no real pre-period effect (ADH 2015 Fig. 4, German reunification backdated to 1975). Sweeps every feasible interior pre-date by default (≥2 pre-fake + ≥1 post-fake); an explicit post-period / non-pre date raises, a valid-but-dimensionally-infeasible date yields a `status="infeasible"` row (no raise).
- **Regression-weight extrapolation diagnostic** (`regression_weights()`, journal pp. 498-499): computes the implied donor weights `W^reg = X0a'(X0a X0a')^{-1} X1a` of the regression counterfactual `B̂'X_1` on the baseline fit's predictor matrices with an **intercept row prepended** (so `ι'W^reg = 1` under full row rank — matching ADH's "if a constant is included"). Regression is then also a weighting estimator summing to one, but with UNRESTRICTED weights: `w_reg < 0` or `> 1` flags donors it extrapolates on, outside the donors' convex hull the simplex-constrained SC never leaves (in ADH's application regression assigned negative weights to Greece/Italy/Portugal/Spain). Returns a per-donor `w_reg` / `w_sc` / `extrapolates` / `abs_extrapolation` table sorted by extrapolation magnitude; the reporting headline is `n_extrapolating`. Pure linear algebra (no re-fit), computed as the min-norm least-squares solution. **At full row rank** the system is exactly solvable, so `W^reg` is invariant to per-predictor row scaling — identical across the standardized (nested/custom) and raw (inverse_variance) predictor spaces (differing only for `cv`, which matches on validation-window predictors). Fails closed (for consistency with the other diagnostics) on a non-converged treated fit or `< 2` donors. **Note:** the sum-to-one property AND the row-scaling invariance both hold only at full ROW rank; when the intercept-augmented predictor matrix is rank-deficient (`k+1 > J` — e.g. the default per-period outcome lags when `T0 > J` — or collinear predictors) the reported `W^reg` is the MIN-NORM least-squares solution (a `UserWarning` fires; `_regw_rank_deficient` is set), `Σ W^reg` need not equal 1, and — because the inexact fit's residual metric is reweighted by row scaling — the min-norm `W^reg` is computed in the captured fit space and can itself differ across predictor spaces (still an informative extrapolation witness; within a fixed space, uniqueness of the least-squares solution depends on the predictor COLUMN rank — an overdetermined full-column-rank case has a unique solution that merely need not sum to 1).
- **Sparse-SC subset search** (`sparse_synthetic_control(sizes=None, max_subsets=50000)`, journal pp. 506-507): for each target size `l < J`, EXHAUSTIVELY searches all `C(J, l)` donor subsets **holding V FIXED at the baseline fit's V** (ADH hold V fixed to make the combinatorial search tractable, footnote 20 — this reuses the captured fit-time predictor matrices + V, unlike `leave_one_out()`/`in_time_placebo()` which re-search V), refits the inner simplex solve per subset, and reports the best size-`l` synthetic (lowest pre-period outcome MSPE). Shows how the fit degrades and the ATT moves as the synthetic is forced sparse (`l = 4, 3, 2` degrade "moderately", `l = 1` much worse — a single-match design close to DiD, footnote 23). A `status="baseline"` row (the full fit) first, then one row per size with the winning subset's ids / weights / `pre_rmspe` / `att` / `delta_att`; `get_sparse_synthetic_control_gaps()` returns the per-size winner paths for the overlay. `sizes=None` sweeps `[1, 2, 3]` (clipped to `l < J`); a DEFAULTED size whose `C(J, l)` exceeds `max_subsets` is **skipped with a warning** (a defaulted call never raises), while an **explicitly** requested `l` with `C(J, l) > max_subsets` raises `ValueError` (the user-facing "exhaustive or raise" contract — no silent approximation). Non-converged subset solves are excluded and counted (`n_failed`); a size with no converged subset gets `status="all_subsets_failed"`. Fails closed (warn + baseline-only) on a non-converged treated fit or `< 2` donors. **Note:** pre-fit typically degrades as `l` shrinks but strict monotonicity is NOT guaranteed — subsets are ranked by the uniform-outcome pre-period MSPE while each subset's weights are V-optimal on the predictor objective.

**Confidence sets by test inversion (Firpo & Possebom 2018, §4, opt-in):** A confidence set for the treatment-effect path, built ON TOP of the in-space placebo and surfaced under `estimator_native_diagnostics` (the analytical inference contract is unchanged — see the non-analytical Note below). Under a common-effect sharp null `H_0^f: α_1t = f(t)` (Eq 11) the donor synthetic controls and the pre-period MSPE denominators do not change — only the post-period gaps shift by `f(t)` — so the test is a pure **re-ranking** of the gap paths `in_space_placebo()` already computed (no synthetic-control refits):
- **`test_sharp_null(effect, gamma=0.1)`** forms the modified RMSPE ratio `RMSPE^f_j = sqrt(mean_post((α̂_jt − f(t))²) / pre_denom_j)` for every unit (Eq 12) and the permutation p-value `p^f = (1 + #{converged placebos j : RMSPE^f_j ≥ RMSPE^f_1}) / (n_ref + 1)` (Eq 13 at `φ=0`, `v=(1,…,1)` — the equal-weights benchmark). `effect` is a scalar (a constant-in-time effect) or a length-`n_post` array (an arbitrary post-period path `f(t)` — e.g. an intervention cost path or a theory-predicted shape). At `f≡0` this is identically the in-space `placebo_p_value` (Eq 5 = Eq 13), held bit-for-bit by reusing each unit's floored pre-period denominator persisted from the placebo run (the pre window is `f`-free, so the denominator is grid-invariant; the floor scale is **per-unit** `max|Z1_j|`).
- **`confidence_set(family="constant"|"linear", gamma=0.1, bounds=None, n_grid=200)`** inverts that test over a one-parameter family: `"constant"` → `f(t)=c` (Eq 15), a confidence **interval** for a constant effect (Eq 16); `"linear"` → `f(t)=c̃·(t−T0)` with the 1-based post-period index (Eq 17), a confidence **set** over the slope `c̃` (Eq 18). Membership is the paper's **strict** `p^c > γ` (Eq 14 — see the boundary Note). With `bounds=None` the set is recovered **EXACTLY**: `p^c` is a piecewise-constant step function (each placebo's indicator flips only at the real roots of `A_j(c)·D_1 = A_1(c)·D_j`, a quadratic in `c`), so the placebo breakpoints partition the line and `p` is evaluated once per induced open interval AND at each breakpoint — where a tie under `≥` can lift `p` above γ, so an isolated accepted singleton (a tangent / co-located root) is captured. The accepted set is the union of accepted intervals/points, with **no centering or monotonicity assumption** (a poor-pre-fit treated unit can have its accepted region in the tails, not around the point estimate). `bounds=(lo,hi)` instead scans a fixed grid (grid-limited membership). The summary is on `effect_confidence_set` (`status ∈ {"ran","empty","unbounded"}`) and the full grid on `get_confidence_set_df()`. **Fail-closed:** `γ < 1/(J+1)` ⇒ `p^c > γ` for every `c` ⇒ `"unbounded"` (`±inf` endpoints + warning — the discrete-granularity point, fn 8); a treated unit lacking the best pre-fit can give a one-sided unbounded edge; if no interval or breakpoint is accepted the set is `"empty"` (NaN endpoints); a non-contiguous accepted region (disjoint components / an isolated singleton) reports the `[lower, upper]` hull with `contiguous=False` + a warning; and `< 2` donors / a non-converged treated fit / an unpickled result (no placebo reference set) raise `ValueError`. **Scope:** sensitivity weights (`φ≠0`, Eqs 7–9), the general test-statistic menu `θ¹–θ⁵` (Eq 19), one-sided (§7's signed-`t` statistic), and the multiple-outcome / multiple-treated extensions (§6) are **deferred** (flagged in the paper review checklist).

**Conformal inference (Chernozhukov-Wüthrich-Zhu 2021, opt-in):** Valid p-values for a hypothesized post-period effect trajectory and pointwise confidence intervals — what the placebo / test-inversion paths cannot give. Surfaced under `estimator_native_diagnostics`; the analytical `se`/`t_stat`/`p_value`/`conf_int` stay NaN (separate permutation object — see the non-analytical Note). Under a sharp null `H0: θ = θ0` the counterfactual treated outcomes are imputed (`Y^N_{1t} = Y_{1t} − θ0_t`, `t>T0`; pre unchanged), a **time-permutation-invariant proxy** is fit UNDER THE NULL on **all** periods, and the statistic `S_q(û) = ((1/√T*)·Σ_{t>T0}|û_t|^q)^{1/q}` (CWZ §2.2) is referred to its permutation distribution (eq 2) by reshuffling residuals over time:
- **Proxy (CWZ §2.3, eqs 3–4):** the canonical constrained-LS synthetic control (outcomes by default; optional RAW covariate-matching rows stack into the same objective when `covariates=` is supplied — see the scope note) — simplex weights `argmin_w Σ_{t=1}^T (Y^N_{1t} − Σ_j w_j Y^N_{jt})²` s.t. `w≥0, Σw=1`, **NO V-matrix, no intercept** (residuals/statistic are outcome-only regardless of covariate rows), fit over ALL periods (footnote 9: "we estimate w under the null based on all the data"). Reuses the Frank-Wolfe simplex solver `utils._sc_weight_fw` (packed `(T,J+1)`, `zeta=0, intercept=False`).
- **`conformal_test(effect, q=1, alternative="two-sided", covariates=None, scheme="moving_block", n_iid=10000, seed=None)`** — joint sharp-null test (eqs 1–2). `effect` is a scalar (constant) or length-`n_post` path; `q ∈ {1,2,∞}` (`q=∞` is `max|û_t|`). The proxy is fit ONCE under the null and only residuals are permuted (footnote 7 — permuting residuals ≡ permuting data for a time-permutation-invariant proxy). Returns `p_value`, `S_observed`, `n_perms`, `proxy_converged`.
- **`conformal_confidence_intervals(alpha=0.1, alternative="two-sided", covariates=None, scheme="moving_block", n_iid=10000, bounds=None, n_grid=100, seed=None)`** — pointwise per-period CIs by test inversion (Algorithm 1). Per the paper (§2.2), each per-period CI for period `t` uses `Z = (Z_1,…,Z_{T0}, Z_t)` — the pre-periods PLUS only period `t`, the **other post-periods dropped** — a clean `T*=1` conformal test (`q` is therefore inert: `S_q = |û_t|`). Returns one row per post period (`lower`/`upper`/`point_estimate`/`status`/`contiguous`); the full grid is on `get_conformal_grid_df()`.
- **`conformal_average_effect(alpha=0.1, alternative="two-sided", covariates=None, scheme="moving_block", n_iid=10000, bounds=None, n_grid=200, seed=None)`** — CI for the average effect `T*^{-1}Σ_{t>T0}θ_t` by test inversion (Appendix A.1): **collapse** the panel into non-overlapping `T*`-blocks (per-unit block averages), fit the proxy on the collapsed `Z̄`, permute the `T/T*` **block** residuals. The earliest `T0 mod T*` pre-periods are dropped to make the block count integral. Requires `T0 ≥ T*`.
- **Permutation schemes:** `"moving_block"` (`Π_→`, `m` cyclic shifts — valid under serially-dependent / stationary weakly-dependent errors, Assumption 2.2; the robust default) and `"iid"` (`Π_all`, sampled — valid under i.i.d. errors, Assumption 2.1, finer p-values). Both include the identity, so the permutation p-value is `≥ 1/|Π|` by construction.
- **Fail-closed:** `<1` donor or an unpickled result (no fit snapshot) or non-finite panel cells / `<2` periods → `ValueError`; a single donor warns (degenerate proxy `w=[1]`); `T*≥T0` warns (the validity caveat — large `T0` drives exactness); a non-converged grid point is **indeterminate**, not rejected — only a converged `p≤α` rejects, so the set is the complement of the rejections (excluding non-converged points would understate the CI width); an accepted set touching a grid edge is `status="grid_limited"`; an empty accepted set is `status="empty"`; `α < 1/|Π|` ⇒ every value accepted, short-circuited to `status="unbounded"` with `(−inf, +inf)` endpoints (warned). **Scope:** one-sided alternatives and proxy covariates are **implemented (2026-07)** — `alternative={"two-sided","greater","less"}` on all three conformal methods uses the SIGNED average-effect statistic `S(û)=T_*^{-1/2}·Σ_{t>T0}û_t` for one-sided tests (grounded in **Remark 1**'s statistic freedom + eq 2; **the paper has no §7** — the prior "(§7 signed-t)" wording here was a cross-contamination from the Firpo-Possebom scope sentence above), with `q=1` enforced (the signed statistic has no norm order) and CI inversion reporting half-line HULLS (`[lower,+inf)` / `(-inf,upper]`; the infinite side is attached because `p → 1` in the accepted direction, and the finite endpoint is the accepted-set hull — Algorithm 1 has no monotonicity guarantee after per-candidate proxy refits, so `contiguous` is recomputed against the final reported ray (a rejected scanned point anywhere inside it — including above the finite hull edge — flips `contiguous=False`); an all-rejected grid reports the infinite side + an uncertified NaN finite endpoint as `grid_limited`); `covariates=[...]` stacks pivoted-variable matching rows into the constrained-LS proxy objective per the paper's note after eq 6 ("straightforward to incorporate (transformations of) covariates X_jt into the estimation problems (4) and (6)") — rows stack RAW (the paper's "(transformations of)" delegates scaling to the caller; documented in the docstring), residuals/statistic stay outcome-only, and exchangeability is preserved (covariate rows are fixed features of every permuted dataset; the stacked objective is invariant to time permutations of Z). **Exogeneity assumption (UserWarning on every covariate run):** `Z(θ0)` null-imputes only the OUTCOME — covariates enter as their observed values, which the framework treats as no-intervention paths (`X^N`), so covariates must be unaffected by the policy; a policy-affected covariate silently conditions the proxy and p-value on treated post-period intervention information. On the block-collapsed average-effect surface, covariate rows collapse with the same T*-block structure. Locked by `TestConformalOneSided` (hand-rolled signed-statistic oracle, directional rejection, half-line endpoints) + `TestConformalCovariates` (weight-mechanism, outcome-only residuals, permutation floor). The AR/innovation-permutation path (Lemmas 5–7, for time-series proxies) remains **deferred** (tracked in TODO.md).

**Notes / deviations:**
- **Note:** The standardization divisor `divisor = sqrt(apply(cbind(X0,X1), 1, var))` (per-predictor SD over donors+treated, ddof=1) and the inner/outer optimizer are **not specified in ADH 2010** (which defers these numerics to Abadie & Gardeazabal 2003 App. B / the `Synth` software). The divisor is pinned from the R `Synth::synth` source; `solution.v` lives in this scaled predictor space, so the deterministic R-parity test feeds `custom_v` in the same scaled space.
- **Note:** The outer objective minimizes the pre-period outcome MSPE over **all** pre periods, whereas R `Synth` uses a `time.optimize.ssr` window (1960–1969 in the Basque example). The nested `V` therefore differs from R by an efficiency-only choice (the paper notes inferential validity holds for *any* `V`), so end-to-end nested parity is a tolerance band, not equality.
- **Note:** `V` is parametrized on the unit simplex via a softmax of an unconstrained vector (trace-normalization is identification-fixing, not a constraint loss); the multistart Nelder-Mead + derivative-free Powell polish approximates R's best-of-`optimx` behavior over the non-smooth outer objective.
- **Note (out-of-sample CV `V`-selection — per-window re-aggregation + fully-spanning precondition):** `v_method="cv"` implements ADH 2015's four-step out-of-sample procedure (Abadie 2021 Eq. 9): (1) split the pre-period positionally at `t0` (`v_cv_t0`; default `len(pre)//2`, Abadie 2021's `t0 = T0/2`) into a training window `pre[:t0]` and a validation window `pre[t0:]`; (2) for each candidate `V`, fit the training weights `W̃(V)` on the **training-window** predictors; (3) pick `V*` minimizing the **validation-window** outcome MSPE of `W̃(V)`; (4) re-estimate the final `W* = W(V*)` on the **validation-window** predictors. Each predictor spec is **re-aggregated** over each window — its op (mean/sum/identity) is recomputed over only the periods that fall in that window — exactly as ADH 2015's CV re-runs `dataprep` separately on each window (`_truncate_specs_to_window` + a per-window `_build_predictor_matrix` + a per-window `_standardize`, since `V*` is predictor-importance applied in each window's own scaled space). The predictor dimension `k` is preserved: re-aggregation changes each row's **values** per window, not the row count, so `V` stays `k`-dimensional throughout. Because each predictor is re-measured on the **training window only** for the V-search, the selection is genuinely out-of-sample for **all** predictor types — a spanning predictor's validation-window observations never enter the training fit (the property a masked full-pre aggregate could not provide, since the full-pre value already bakes in the validation-window observations). The same `V*` drives both fits with no zeroed coordinate, so `v_weights` (= `V*`) reproduce `donor_weights` on the validation-window predictors; `predictor_balance` is reported on that same validation-window basis (each row's values are the spec re-aggregated over `pre[v_cv_t0:]`; the row label remains the full spec identity so it stays aligned with `v_weights`). `mspe_v` reports the step-3 validation MSPE at `V*`. **Deviation from R:** R `Synth` has no built-in CV function — ADH 2015's CV is a *manual* two-`dataprep` re-run — so this re-aggregation reproduces that manual procedure for our absolute-period spec aggregates.
  - **Fully-spanning precondition (fail-closed):** steps 2-3 re-aggregate each predictor on the training window and step 4 re-aggregates it on the validation window, so every predictor must be observed in **both** windows (have ≥1 period in each) to be re-measurable on each. `cv` therefore **requires every predictor to span both windows**; a violation raises `ValueError`. This holds for ADH 2015's shared covariate / multi-period special predictors (which span the windows) but NOT for the **default per-period outcome lags** (each is single-period and lives in one window only) — `cv` with the bare default predictors is rejected with guidance to pass spanning `predictors` / multi-period `special_predictors`. **Window-information gate (fail-closed):** spanning is necessary but not sufficient — `W` is identified by the DONORS being distinguishable (`X0·W` is a convex combination of the donor columns), so if a window's re-aggregated predictors are constant **across donors** (all donor columns identical) then `X0·W` is the same for every simplex `W`, the inner solve's objective is flat, and `_inner_solve_W` returns arbitrary weights while reporting convergence — even when the treated unit differs (the treated unit is the matching target, not part of `W`'s identification), and `_standardize` only *warns* on the zero-variance rows. `cv` therefore also **requires each window to have cross-DONOR variation in at least one predictor**: the headline fit raises `ValueError`; an in-space placebo refit whose (pseudo-treated) donor pool is indistinguishable in a window is dropped from the permutation reference set; an in-time-truncated date whose window goes donor-flat is marked `status="infeasible"`. Validated by deterministic equivalence to the R-anchored `custom_v` path (the step-3 validation MSPE of the training-window fit and the step-4 validation-window weights each match a `custom_v=V*` fit on the correspondingly re-aggregated predictors) + cv self-consistency (`in_time_placebo` under cv == a fresh cv fit on the backdated panel) + a rejection test for non-spanning predictors. An explicitly pinned `v_cv_t0` that **no longer fits the truncated pre-fake window** is nulled to the `//2` default for the placebo refit (a pinned value that still fits the truncated window is kept); in-time truncation that breaks the fully-spanning precondition for an otherwise-valid headline fit (a kept spec stops spanning both windows at the truncated split) marks that date `status="infeasible"`.
- **Note (CV non-uniqueness — deterministic tie-break):** Abadie 2021 footnote 7 warns the CV `V*` "need not be unique" (the validation MSPE can be flat in `V`) and that "an implementation must pick a deterministic tie-break." Among candidate `V` whose validation MSPE ties to tolerance, this implementation selects the one **closest to uniform** (the densest `V`, mirroring footnote 7's ridge-toward-dense remedy in spirit) — a principled choice that makes the selected `V*` among equally-good optima independent of the **multistart evaluation order**. The cv fit is reproducible for a **fixed `seed`** (like `nested`); it is NOT seed-independent — the multistart fills any slots beyond the distinct heuristic starts (uniform, inverse-variance, univariate-fit — at most three, fewer if any coincide) with seed-dependent random Dirichlet draws, so different seeds can explore different optima. The tie-break removes only the start-order dependence among ties, not the seed dependence. A user-supplied `inner` ridge penalty `γ·Σv_h²` is a deferred extension (out of scope this release).
- **Note (inverse-variance `V`):** `v_method="inverse_variance"` uses the closed form `v_h = 1/Var(X_{h·})` (Abadie 2021 §3.2(a)), variance taken over donors+treated on the **unstandardized** predictors, and applies that `V` to the **raw** predictors so the effective objective is the unit-variance-rescaled `Σ_h (X_{1h} − X_{0h}W)²/Var_h`. Abadie 2021 describes inverse-variance `V` precisely as "rescal[ing] each predictor row to unit variance" — which is exactly what `standardize="std"` already does — so the `standardize` pre-scaling is intentionally **bypassed** on this branch (it is equivalent to uniform `V` on the standardized predictors). Applying `1/Var` on top of the standardized rows would double-rescale to `Σ_h diff_h²/Var_h²` (inverse-variance *squared*), which is NOT the paper's selector. Deterministic (no search; `mspe_v` is `None`). A zero-variance predictor row (no cross-unit information) gets 0 `V` weight; if **every** row is zero-variance the result falls back to uniform `V` with a `UserWarning`. `custom_v` is rejected for this method (and for `"cv"`) — fail-closed, never silently ignored.
- **Note (single-donor degeneracy — uniform `V` for all methods):** with a single donor (`J=1`) the synthetic control is forced to `w=[1]`, so the predictor-importance `V` is **unidentified** — every `V` yields the same synthetic. `fit()` short-circuits to `w=[1]` with a **uniform** `v_weights` and `mspe_v=None` for ALL `v_method`s, INCLUDING `"cv"` and `"inverse_variance"` (their selected / closed-form `V` would be inert, so it is not computed or reported), emitting a `UserWarning`. The donor weights / gap path / ATT do not depend on `V` when `J=1`, so they are unaffected; for `"cv"` the `v_cv_t0` resolution + spanning/variation preconditions still run (and can still raise) before this short-circuit. This is a deliberate, consistent degenerate contract — not a per-method `V` mismatch.
- **Note:** The 1×SD poor-fit threshold is a defensive implementation choice in the spirit of the `SyntheticDiD` convention; ADH 2010 gives only the qualitative guidance "do not use SCM when the fit is poor" (no numeric cutoff). The warning fires whenever pre-period RMSPE exceeds the SD of the treated unit's pre-period outcomes — **including a flat treated pre-path** (`SD = 0`) with non-trivial RMSPE (a scale-aware roundoff floor suppresses the warning on a near-perfect flat fit). This is slightly broader than `SyntheticDiD`'s `SD > 0`-gated form, matching the literal RMSPE-exceeds-SD contract above.
- **Deviation from R:** `standardize="none"` disables predictor standardization entirely; R `Synth` always scales by the predictor SD. Provided for diagnostics; changes the geometry of the `V` objective.
- **Note:** predictor rows support only **equal-weight** linear combinations of pre-period values — `mean` (`k_s = 1/T0`), `sum` (`k_s = 1`), and per-period outcome lags (identity, a single `k_s = 1`). ADH (2010) §2.3 defines the general form `Ȳ_i^{K_m} = Σ_s k_s Y_is` with *arbitrary* weights `k_s`; this release does NOT accept user-supplied non-uniform `K_m` weight vectors (and `median` and other non-linear aggregations are intentionally excluded). The supported set still spans the standard `Synth::dataprep` `predictors.op` + `special.predictors` usage; arbitrary-weight `K_m` is a deferred extension.
- **Deviation from R:** predictor/outcome **aggregation fails closed on any non-finite (NaN/inf) cell**, whereas R `Synth::dataprep` hardwires `na.rm=TRUE` (aggregating over the observed cells of a partially-missing window). The fail-closed contract is deliberate: na-dropping silently aggregates different period subsets across units, yielding incomparable predictors with no warning. The analyst must restrict `predictor_window` / `special_predictors` / `pre_period_outcomes` periods (and the outcome panel) to where each variable is observed; both partially- and fully-missing windows raise `ValueError`. Only the row *ordering* matches `dataprep`, not the missing-data handling.
- **Note (in-space placebo donor pool):** the real treated unit is **excluded from every placebo's donor pool** — when donor `j` is pseudo-treated it is fit against the other `J−1` donors, never the real treated unit (whose post-period is treatment-contaminated). The ranking set is still the `J+1` units {treated} ∪ {J placebos}. ADH 2010 §2.4 does not spell out placebo donor-pool composition; this matches the standard `SCtools::generate.placebos` construction (rotate the pseudo-treated identity through the donor pool; the original treated unit is never re-added as a donor).
- **Note (placebo failure handling):** a placebo is **excluded from both the numerator and the denominator** of the rank (never penalized into it) and tallied in `n_failed` when its fit is not a valid optimum — EITHER its **inner Frank-Wolfe weight solve** did not converge (a truncated `W` is unusable) OR its **outer `V` search** did not converge (an under-optimized `V` fits the pre-period worse, shrinking the RMSPE ratio and biasing the p-value anti-conservatively, so it must not silently enter the rank). The reported p-value uses the **effective** count `rank / (n_placebos + 1)`, where `n_placebos` is the number of placebos that entered the reference set. Failed donors still appear in `get_placebo_df()` (`status="failed"`, NaN metrics), so once a reference set is produced the table is the full treated + every-donor unit set (`n_donors + 1` rows). In the fail-closed cases the placebo loop does not run and only the treated row is returned: `J < 2` → `placebo_p_value` is NaN with a warning (no placebo distribution; `J == 2` warns the distribution is coarse), and a treated fit whose own **inner OR outer** search did not converge also fails closed (ranking a truncated / under-optimized treated statistic would not be a valid permutation). **Caveat:** each placebo refit inherits the original fit's `optimizer_options` / `n_starts`, so valid inference requires settings adequate for the outer `V` search to converge to a comparable-quality synthetic (production defaults do; cheap settings under-optimize placebo `V` and those placebos are dropped as failed — raise `n_starts` on `in_space_placebo()` or re-fit with a larger `optimizer_options['maxiter']`).
- **Note (in-space / leave-one-out infeasible vs failed split):** an excluded `in_space_placebo()` / `leave_one_out()` refit is attributed to one of two causes, mirroring the split `in_time_placebo` already reports. A **solver non-convergence** (a truncated inner `W` or outer `V` search) is tallied in `n_failed` (`_loo_n_failed`) with a `status="failed"` row; a **structural cv infeasibility** — under `v_method="cv"`, the pseudo-treated donor pool (in-space) or reduced pool (leave-one-out) is indistinguishable in a re-aggregated CV window, so the weights are unidentified (the window-information gate above; `_outer_solve_V_cv` returns a structural sentinel that `_placebo_fit_unit` threads out) — is tallied in `n_infeasible` (`_loo_n_infeasible`) with a `status="infeasible"` row. Both are excluded from the rank / ATT range identically, so `placebo_p_value` / `n_placebos` are **unchanged** by the attribution — only the diagnostic accounting is refined. `_placebo_status` / `_loo_status` gain `all_placebos_infeasible` / `all_placebos_unusable` (resp. `all_refits_infeasible` / `all_refits_unusable`) codes for a no-usable-refit run that is purely structural or a mix, and `DiagnosticReport` surfaces the machine-readable `reason_code` alongside `n_failed` / `n_infeasible`. `n_infeasible` is 0 for the non-`cv` `v_method`s (no structural-identification gate). The remedy differs by cause: a structural infeasibility needs different predictors / `v_cv_t0` / donor pool, NOT a larger `n_starts` / `inner_max_iter`.
- **Note (RMSPE-ratio floor):** the reported `rmspe_ratio = sqrt(MSPE_post / MSPE_pre)` floors the pre-period MSPE denominator at a scale-aware `1e-8 · max(|pre-outcomes|, 1)²` (before the square root) so a (near-)perfect pre-fit (`pre-MSPE → 0`) yields a large-but-FINITE ratio rather than `inf`/`nan` (which would corrupt the rank). Ties (`ratio_j ≥ treated_ratio`) are counted, making the p-value conservative. Mirrors the `_fit_tol` poor-fit guard.
- **Note (placebo p-value is non-analytical):** `placebo_p_value` is deliberately a SEPARATE field from `p_value` (which stays NaN) — it is a permutation p-value with no SE / t-stat, so it does not flow through `safe_inference`. `is_significant` likewise stays bound to the (NaN) `p_value`, NOT `placebo_p_value`; a tool gating on `is_significant` will see `False` even when `placebo_p_value` is small. The reporting stack surfaces the placebo p-value through `estimator_native_diagnostics`, never the analytical headline.
- **Note (confidence set is non-analytical — `conf_int` stays NaN):** the Firpo & Possebom test-inversion `effect_confidence_set` is a permutation set at level `1−γ`, kept DELIBERATELY SEPARATE from the analytical `conf_int` (which stays `(NaN, NaN)` — classic SCM has no Wald interval). It is parametrized by `γ` (not the estimator's `alpha`, and granular in `1/(J+1)`), can be a *set* rather than an interval (the linear family) and can be unbounded or non-contiguous, so it cannot be coerced into the `(lo, hi)` `conf_int` tuple without breaking the `safe_inference` NaN-consistency contract. `se`/`t_stat`/`p_value`/`conf_int`/`is_significant` all stay at their NaN state; the set is surfaced only via `confidence_set()` / `get_confidence_set_df()` / `effect_confidence_set` (and `estimator_native_diagnostics`) — mirroring how `placebo_p_value` is kept off `p_value`.
- **Note (test-inversion boundary convention — strict `p^f > γ`):** Firpo & Possebom's inequalities are non-uniform at `p = γ` — the RMSPE-based tests reject at `p < γ` (Eqs 5/9/13), the general-statistic test rejects at `p ≤ γ` (Eq 19), and the confidence set is the **strict** `p^f > γ` (Eq 14), so Eq 14's set is NOT the exact complement of the Eq 13 rejection region (they differ at `p^f = γ`). Because the permutation p-value is discrete (a multiple of `1/(J+1)`), `p = γ` is reachable, so this implementation pins Eq 14's strict `p^f > γ` for set membership (a `p = γ` value is **excluded**) and documents it (matches the `firpo-possebom-2018-review.md` §4.2 boundary note).
- **Note (test-inversion set construction is an implementation choice):** Firpo & Possebom §4.2 gives the set definitions (Eqs 14/16/18) but does NOT prescribe how to enumerate the set. The default is **exact piecewise-constant breakpoint inversion** — `p^c` is constant between the real roots of the placebo-vs-treated comparison quadratics, so evaluating `p` per induced interval and at each breakpoint recovers the set exactly (no grid resolution / shape assumption); the optional fixed `bounds=` grid is the grid-limited alternative. Either is OUR choice (the paper leaves it unspecified) — a documented deviation. (`n_grid` controls only the returned inspection table, not membership, when `bounds=None`.)
- **Note (test-inversion validation — no R anchor):** R `Synth` has no test-inversion confidence-set function, and the authors' Code Ocean replication capsule was not consulted (paper-sourced only). The confidence sets are validated by (a) **self-consistency** — `test_sharp_null(0)` equals the in-space `placebo_p_value` exactly (transitively R-anchored via the Basque placebo parity), (b) a **numpy oracle** re-implementing Eqs 12–14 on hand-built gap paths (incl. the strict `p = γ` boundary and the per-unit floor), and (c) a **coverage simulation** (a constant-effect DGP; the `1−γ` set covers the truth at ≈ `1−γ`).
- **Note (in-time placebo windowing — TRUNCATE):** ADH 2015 §4 says to re-estimate the in-time placebo "with the same predictors lagged accordingly." Because `diff_diff`'s predictor specs reference **absolute** periods, the in-time placebo re-cuts them by TRUNCATION: pre-period-outcome predictors become the pre-`t_f` outcomes, and covariate / special-predictor windows are intersected with the pre-`t_f` window; a window lying ENTIRELY in the held-out region `[t_f, T0)` is **dropped** (surfaced in the `n_dropped_specs` column + an aggregated warning), and `custom_v` is subset in lockstep with the surviving specs. For an outcome-predictor fit (the R-anchorable case) TRUNCATE is identical to ADH's "lag" — both equal a manual `Synth::synth` re-run with `time.optimize.ssr` cut at `t_f`. The held-out window never enters the fit (the placebo's `all_periods` is the pre-fake + post-fake span; the true post-treatment periods are excluded entirely), so there is no "peeking." This concrete convention is NOT spelled out in ADH 2015 (which gives only the qualitative "lag accordingly").
- **Note (in-time placebo requires ≥2 pre-fake periods):** the in-time placebo treats a date with fewer than 2 pre-fake periods as `status="infeasible"` (the default sweep starts at the 3rd pre-period). This is DELIBERATELY stricter than the base estimator's `T0 ≥ 1` allowance (which permits a single-pre-period fit but warns that nested-`V` selection is unreliable): an auto-swept placebo date with a single pre-fake period is a trivially-matchable, non-credible pre-fit, so it is dropped rather than surfaced as a `ran` placebo (mirrors `SyntheticDiD.in_time_placebo`'s `i ≥ 2` rule). A date whose surviving `custom_v` has zero mass after truncation is likewise infeasible (not a convergence failure).
- **Note (leave-one-out weight floor):** ADH 2015 §4 leave-one-out omits "each donor that received positive weight." This implementation drops each donor with **reportable** weight — above the `1e-6` interpretability floor (`synthetic_control._MIN_REPORT_WEIGHT`), i.e. exactly the donors in `donor_weights` — rather than every strictly-positive weight. A donor with `0 < w ≤ 1e-6` is numerical dust whose removal moves the ATT by ~its weight (its `delta_att` would be ~0, an uninformative row), and the floor keeps the LOO table aligned with the reported donor support. The drop-set is **frozen at fit time** on the fit snapshot (`weighted_donor_ids`), so `leave_one_out()` is immune to post-fit mutation of the presentation-level `donor_weights` dict.
- **Note (ADH-2015 diagnostics validation):** R `Synth` has **no** in-time-placebo or leave-one-out function (verified against its full CRAN function index; `SCtools` adds only the *in-space* placebo battery, `scpi` only prediction-interval uncertainty), so there is no canonical R *output* to match for these diagnostics — in R they are hand-rolled by re-running `dataprep()`+`synth()`. They are validated instead by (a) the solver's existing Basque R parity (above), and (b) deterministic **self-consistency** tests proving each diagnostic equals a from-scratch `SyntheticControl` fit on the equivalent sub-problem — `leave_one_out()` drop-`d` == a fit on the donor pool minus `d`; `in_time_placebo([t_f])` == a fit on the backdated/truncated panel — both via a fixed `custom_v` (match to 1e-7). The two §4-tail diagnostics are likewise R-anchor-free (R `Synth` has neither): `regression_weights()` is validated by a **numpy oracle** re-implementing `W^reg = X0a'(X0a X0a')^{-1}X1a` on hand-built matrices (incl. the full-rank sum-to-one property, the rank-deficient min-norm branch, and row-scaling invariance across `v_method` spaces); `sparse_synthetic_control()` by **self-consistency** — its exhaustive size-`l` winner (and the winner's weights) match an independent brute-force enumeration using the SAME fixed baseline `V`, which also confirms `V` is held fixed rather than re-searched.
- **Note (conformal proxy ≠ ADH V-matrix — deliberate):** the CWZ conformal layer fits its OWN counterfactual proxy — the canonical constrained-LS synthetic control on **raw outcomes over all periods** (eqs 3–4, no V-matrix) — NOT the headline ADH V-matrix weights (which match on pre-period predictors only). This is required, not incidental: CWZ's exactness theory (Lemma 1; Appendix D exchangeability under the null) holds for a **time-permutation-invariant** estimator, which the ADH pre-period V-fit is not (it treats pre and post asymmetrically). So the conformal counterfactual / `point_estimate` can differ from the headline `att`, and is reported as a separate object.
- **Note (conformal permutation floor — `1/|Π|`, distinct from Firpo's `1/(J+1)`):** the conformal p-value is `(1/|Π|)·#{π : S(û_π) ≥ S(û)}` (eq 2). The permutation set `Π` **includes the identity** (`S(û_id)=S(û)`, counted under `≥`), so `p̂ ≥ 1/|Π|` automatically — there is **no extra `+1`** (unlike the cross-unit placebo / Firpo `(1+n)/(n+1)`, where the treated unit is not a member of the placebo reference set). `|Π| = T` (moving-block, joint), `T0+1` (pointwise sub-series), or `T/T*` (average-effect blocks), capped at `n_iid` for the i.i.d. scheme.
- **Note (conformal per-period CI drops the other post-periods — paper-sourced):** for a pointwise CI of period `t`, CWZ §2.2 defines the data under the null as `Z = (Z_1,…,Z_{T0}, Z_t)` — the pre-periods plus ONLY period `t`. The other post-periods are dropped (not plugged in, not zeroed), making each per-period CI a clean `T*=1` conformal test on the `(T0+1)`-length sub-series. (Confirmed against arXiv:1712.09089v10 §2.2 + Algorithm 1.) The grid is centred on a **pre-only** proxy fit (predict the post slot from the pre slots) — the unconditional all-slot fit would soak the effect into the weights and bias the naïve residual toward 0, so it is not used as the centre.
- **Note (conformal is non-analytical — `conf_int` stays NaN):** the conformal p-value / CI is a permutation object kept DELIBERATELY SEPARATE from the analytical `se`/`t_stat`/`p_value`/`conf_int`/`is_significant` (all NaN — classic SCM has no Wald interval). It can be a *set* (unbounded / `grid_limited` / non-contiguous) parametrised by `α` (granular in `1/|Π|`), so it cannot be coerced into the `(lo,hi)` `conf_int` tuple. Surfaced only via `conformal_test()` / `conformal_confidence_intervals()` / `conformal_average_effect()` / `conformal_inference` (and `estimator_native_diagnostics`) — mirroring how `placebo_p_value` and `effect_confidence_set` are kept off the analytical headline.
- **Note (conformal validity caveats — Remark 2; documented, not auto-corrected):** CWZ requires a **large `T0`** (drives exactness; `T*` may be short) and that `{u_t}` be stationary and **conditionally** heteroscedastic at most — **unconditional** heteroscedasticity / non-stationarity of the shocks invalidates the basic procedure (Remark 2), as does a change in the shock distribution under the intervention (Assumption 1 invariance). No standardizing filter is applied by default; the analyst must pre-filter if these are suspected. A `T*≥T0` fit emits a validity-caveat `UserWarning`.
- **Note (conformal validation — no R anchor):** the authors state "all computations in R" with no named package, and the replication capsule was not consulted (paper-sourced only). Validated by (a) a **numpy oracle** re-implementing `S_q` (`q=1,2,∞`), the moving-block cyclic-shift set, i.i.d. identity-inclusion, and the eq-2 floor on hand-built residuals; (b) a **brute-force permutation-equivalence** test (independently materialising every shift and recomputing `S_q`) matching the production p-value bit-for-bit; (c) `_cwz_proxy_fit` vs `scipy` SLSQP on the simplex-LS; and (d) a **coverage simulation** (the `1−α` average-effect CI covers a known constant effect at ≈ `1−α`).

**Reference implementation:** authors' `Synth` package for R/MATLAB/Stata (`Synth::synth`); in-space placebo construction follows `SCtools::generate.placebos`. **R-parity anchor:** the Basque Country study (Abadie-Gardeazabal 2003, `data("basque")`) — published synthetic = region 10 (Cataluña) 0.851 + region 14 (Madrid) 0.149, `loss.v` 0.0089. Two-tier test (`tests/test_methodology_synthetic_control.py`): Tier-1 feeds R's `solution.v` via `custom_v` → donor weights match to atol 1e-3 (deterministic); Tier-2 checks the nested fit in a band.

**Requirements checklist:**
- [x] Donor weights on the unit simplex; exactly one treated unit, block assignment.
- [x] Predictors = covariates + linear combinations of pre-period outcomes (incl. "all pre-period outcomes" default).
- [x] Inner simplex-constrained weighted LS via `_sc_weight_fw` with diagonal PSD `V`.
- [x] Outer nested `V` (pre-period outcome MSPE) + user-supplied `custom_v`.
- [x] Out-of-sample cross-validation `V`-selection (`v_method="cv"`, ADH 2015 §; Abadie 2021 Eq. 9 `t0=T0/2`): per-window re-aggregation train/validation split (fully-spanning precondition) + deterministic flat-MSPE tie-break (fn. 7); threaded through the placebo refits.
- [x] Inverse-variance `V` (`v_method="inverse_variance"`, Abadie 2021 §3.2(a)): closed-form `1/Var(X)`, zero-variance-row + all-zero-uniform-fallback handling.
- [x] Gap path + pre-period RMSPE + predictor-balance table.
- [x] No analytical SE (NaN inference); in-space placebo permutation inference (`in_space_placebo()`, `rank/(n_placebos+1)`) with the real treated unit excluded from every placebo pool, effective-count denominator, and a scale-aware RMSPE-ratio floor.
- [x] Leave-one-out donor robustness (`leave_one_out()`, ADH 2015 §4): per-drop ATT / `delta_att` table + overlay gaps; fail-closed.
- [x] In-time (backdating) placebo (`in_time_placebo()`, ADH 2015 §4): TRUNCATE windowing (drop held-out-window predictors + lockstep `custom_v` subset), feasible-date sweep, fail-closed.
- [x] Confidence sets by test inversion (`test_sharp_null()` + `confidence_set()`, Firpo & Possebom 2018 §4): sharp-null `RMSPE^f` re-ranking of the in-space placebo gaps (Eqs 12–13) + constant/linear one-parameter sets (Eqs 14/16/18) with the strict `p^f > γ` boundary, EXACT piecewise-constant breakpoint inversion (no shape assumption; isolated/disjoint/unbounded sets handled), and fail-closed unbounded/empty/non-contiguous handling. *Deferred:* sensitivity weights (φ≠0), the general-θ menu (Eq 19), one-sided (§7), multiple-outcome/treated (§6).
- [x] Conformal inference (`conformal_test()` + `conformal_confidence_intervals()` + `conformal_average_effect()`, Chernozhukov-Wüthrich-Zhu 2021): own constrained-LS proxy under the null on all periods (eqs 3–4, no V-matrix) + `S_q` statistic (`q=1,2,∞`) + permutation p-value (eq 2, `1/|Π|` floor) over moving-block (`Π_→`) / i.i.d. (`Π_all`) schemes; joint sharp-null test (eqs 1–2), pointwise per-period CIs (Algorithm 1, `Z=(pre,t)`), and the average-effect block-collapse CI (Appendix A.1); fail-closed grid_limited/empty/indeterminate handling; analytical `conf_int` stays NaN. *Implemented (2026-07):* one-sided alternatives (Remark 1 signed statistic — the paper has no §7) + proxy covariates (eq 4/6 note). *Deferred:* AR/innovation-permutation path (Lemmas 5–7) — see `TODO.md`.
- [x] Regression-weight `W^reg` extrapolation diagnostic (`regression_weights()`, ADH 2015 §4): intercept-augmented `W^reg = X0a'(X0a X0a')^{-1}X1a`, flag donors outside `[0,1]`; min-norm + rank-deficient handling; pure linear algebra, analytical inference unchanged.
- [x] Sparse-SC subset search (`sparse_synthetic_control()`, ADH 2015 §4): exhaustive `C(J,l)` subset search holding `V` fixed at the baseline; default-skip vs explicit-raise `max_subsets` guard; per-size winner table + overlay gaps.
- [x] Predictor-leakage, absorbing-suffix/no-anticipation, empty-window, duplicate-label, and inner-non-convergence validation gates.

---

## TripleDifference

**Primary source:** [Ortiz-Villavicencio, M., & Sant'Anna, P.H.C. (2025). Better Understanding Triple Differences Estimators. arXiv:2505.09942v3.](https://arxiv.org/abs/2505.09942v3). Paper review on file: `docs/methodology/papers/ortiz-villavicencio-santanna-2025-review.md`.

**Key implementation requirements:**

*Assumption checks / warnings:*
- Requires all 8 cells of the 2×2×2 design: Group(0/1) × Period(0/1) × Treatment(0/1)
- Warns if any cell has fewer than threshold observations
- Propensity score overlap required for IPW/DR methods

*Estimator equation (as implemented):*

Three-DiD decomposition (matching R's `triplediff::ddd()`):
```
Subgroups: 4=G1P1, 3=G1P0, 2=G0P1, 1=G0P0
DDD = DiD_3 + DiD_2 - DiD_1
```
where `DiD_j` is a pairwise DiD comparing subgroup j vs subgroup 4 (reference).

Each pairwise DiD uses the selected estimation method (DR, IPW, or RA) with
repeated cross-section implementation (`panel=FALSE` in R).

Regression adjustment (RA): Separate OLS per subgroup-time cell within each
pairwise comparison, imputed counterfactual means.

IPW: Propensity score P(subgroup=4|X) within {j, 4} subset, Hajek normalization.

Doubly robust (DR): Combines outcome regression and IPW with efficiency correction
(OR bias correction term).

*Standard errors (all methods):*

Individual-level (default):
```
SE = std(w₃·IF₃ + w₂·IF₂ - w₁·IF₁, ddof=1) / sqrt(n)
```
where `w_j = n / n_j`, `n_j = |{subgroup=j}| + |{subgroup=4}|`, and `IF_j` is the
per-observation influence function for pairwise DiD j (padded to full n with zeros).

Cluster-robust (when `cluster` parameter is provided):
```
SE = sqrt( (G/(G-1)) * (1/n²) * Σ_c ψ_c² )
```
where `G` is the number of clusters, `ψ_c = Σ_{i∈c} IF_i` is the sum of the combined
influence function within cluster `c`, and the `G/(G-1)` factor is the Liang-Zeger
finite-sample adjustment.

Note: IF-based SEs are inherently heteroskedasticity-robust; the `robust` parameter
has no additional effect.

**Note (panel-shaped input, `generate_ddd_panel_data`):** `TripleDifference` is the
repeated-cross-section `panel=FALSE` implementation: the individual-level default SE
treats each row as an independent observation (df = n_obs - 8). When fitting against
panel-shaped data with repeated unit rows and within-unit serial correlation
(e.g., `generate_ddd_panel_data` output), unclustered SE understates sampling
variability and overstates power. Pass `cluster="unit"` to invoke the Liang-Zeger
CR1 path so the influence functions are aggregated within unit before variance
computation. The point estimate `att` is invariant to clustering, only the inference
contract changes.

*Edge cases:*
- Propensity scores near 0/1: trimmed at `pscore_trim` (default 0.01)
- Empty cells: raises ValueError with diagnostic message
- Low cell counts: warns when any cell has fewer than 10 observations
- Cluster-robust SE: requires at least 2 clusters (raises `ValueError`)
- Cluster IDs: must not contain NaN (raises `ValueError`)
- Overlap warning: emitted when >5% of observations are trimmed at pscore bounds (IPW/DR only)
- Propensity score estimation failure: controlled by `pscore_fallback` parameter
  (default `"error"`). If `pscore_fallback="error"`, the error is raised. If
  `pscore_fallback="unconditional"`, falls back to unconditional probability
  P(subgroup=4), sets hessian=None (skipping PS correction in influence
  function), emits UserWarning. When `rank_deficient_action="error"`, errors
  are always re-raised regardless of `pscore_fallback`.
- **Events Per Variable (EPV) diagnostics:** Per-logit EPV =
  min(n_subgroup_j, n_subgroup_4) / n_covariates checked before IRLS.
  Default threshold: 10 (Peduzzi et al. 1996). Warns when EPV < threshold;
  errors when `rank_deficient_action="error"`.
- **Note:** `pscore_fallback` default changed from unconditional to error.
  Set `pscore_fallback="unconditional"` for legacy behavior.
- Collinear covariates: detected via pivoted QR in `solve_ols()`, action controlled by
  `rank_deficient_action` ("warn", "error", "silent")
- Non-finite influence function values (e.g., from extreme propensity scores in IPW/DR
  or near-singular design): warns and sets SE to NaN, propagated to t_stat/p_value/CI
  via safe_inference()
- **Note (rank-guarded IF standard errors):** The per-comparison outcome-regression bread
  and propensity-score Hessian in the influence-function SE (`_compute_did_rc_reg/_dr`,
  `_estimate_ddd_decomposition`) are inverted by the shared `_rank_guarded_inv`. A
  near-singular Gram from a constant/collinear covariate previously gave a garbage
  `np.linalg.inv` (reproduced: `se` ~1e17 for `reg`, ~43 for `dr`); redundant directions are
  now truncated, giving a finite SE on the identified covariate subset (see the
  CallawaySantAnna "rank-guarded IF standard errors" Note for the generalized-inverse
  equivalence argument and the `rcond=1e-10` threshold). `fit()` emits ONE aggregate
  `UserWarning` across the three DiD comparisons, suppressed under
  `rank_deficient_action="silent"`.
- NaN inference for undefined statistics:
  - t_stat: Uses NaN (not 0.0) when SE is non-finite or zero
  - p_value and CI: Also NaN when t_stat is NaN
  - **Note**: Defensive enhancement; reference implementation behavior not yet documented

**Reference implementation(s):**
- R `triplediff::ddd()` (v0.2.1, CRAN) — official companion by paper authors

**Requirements checklist:**
- [x] All 8 cells (G×P×T) must have observations
- [x] Propensity scores clipped at `pscore_trim` bounds
- [x] Doubly robust consistent if either propensity or outcome model correct
- [x] Returns cell means for diagnostic inspection
- [x] Supports RA, IPW, and DR estimation methods
- [x] Three-DiD decomposition: DDD = DiD_3 + DiD_2 - DiD_1 (matching R)
- [x] Influence function SE: std(w3·IF_3 + w2·IF_2 - w1·IF_1) / sqrt(n)
- [x] Cluster-robust SE via Liang-Zeger variance on influence function
- [x] ATT and SE match R within <0.001% for all methods and DGP types
- [x] Survey design support: all methods (reg, IPW, DR) with weighted OLS/logit + TSL on combined influence functions. Weighted solve_logit() for propensity scores in IPW/DR paths.
- **Note:** TripleDifference survey SE: for IPW/DR, pairwise IFs incorporate survey weights via weighted Riesz representers (`riesz *= weights`), so the combined IF is divided by per-observation survey weights (`inf / sw`) before passing to `compute_survey_vcov()` to prevent double-weighting. For regression (RA), pairwise IFs are already on the unweighted residual scale (WLS fits use weights internally but the IF is not Riesz-multiplied), so the combined IF passes directly to TSL without de-weighting. The OLS nuisance IF corrections in DR mode use weighted cross-products normalized by subgroup row count `n` (not `sum(weights)`).
- **Note (vcov_type contract):** `vcov_type` is permanently narrow to `{"hc1"}` per the IF-based variance decomposition. Analytical-sandwich families `{classical, hc2, hc2_bm}` are rejected at `__init__` with a methodology-rooted message citing Ortiz-Villavicencio & Sant'Anna (2025) — the 3-pairwise-DiD decomposition has no single design matrix on which hat-matrix leverage or Bell-McCaffrey Satterthwaite DOF can be defined. `cluster=` continues to invoke Liang-Zeger CR1 on the combined influence function (`(G/(G-1)) · Σ_c (Σ_{i∈c} ψ_i)² / n²`, plain CR1 — no Stata-style `(n-1)/(n-p)` finite-sample factor because the IF has no design-matrix `p` in the OLS sense); `survey_design=` continues to invoke TSL on the combined IF. `vcov_type='conley'` is deferred to the TripleDifference Conley follow-up row in `DEFERRED.md`. See ["IF-based variance estimators vs analytical-sandwich estimators"](#if-based-variance-estimators-vs-analytical-sandwich-estimators) above for the structural taxonomy.
- **Note (`cluster=` + replicate-weight survey rejection):** `TripleDifference(cluster=X)` + `SurveyDesign(replicate_weights=[...], replicate_method=...)` is rejected at `fit()` with `NotImplementedError`. Replicate-weight variance is computed by replicate reweighting (BRR / Fay / JK1 / JKn / SDR) and ignores PSU/cluster entirely (the survey-side gate at `survey.py:104-109` enforces `replicate_weights` are mutually exclusive with `strata`/`psu`/`fpc`); honoring `cluster=` here would silently have no effect on the variance estimate while populating `cluster_name`/`n_clusters` on Results dishonestly. Mirrors the `CallawaySantAnna` guard at `staggered.py:1705-1719`. Either omit `cluster=` (the replicate weights encode the design structure implicitly) or use a non-replicate survey design with explicit `strata`/`psu`/`fpc`.

---


### Staggered mode (3.9, ledger row M-013)

Since 3.9 `TripleDifference` serves BOTH DDD designs from one signature, mirroring
the reference implementation (`triplediff::ddd()` does the same). The estimand,
estimator and variance of each design are unchanged - the merge is API-only, and
the staggered estimation core is the relocated `StaggeredTripleDifference` engine,
shared verbatim.

- **Dispatch.** `first_treat=` selects the staggered engine; the 2x2x2 engine
  serves `(group, partition, post)`. Mixing the two parameter sets is an ERROR in
  both directions, constructor params included, never a silent guess. The
  staggered-only fit params (`unit`, `first_treat`, `aggregate`, `balance_e`) are
  keyword-only, so a positionally-written staggered call cannot bind to the 2x2x2
  slots; positional slots 1-8 are unchanged.
- **Note (`time=` semantics):** the dual role is resolved by dispatch ORDER, not
  value inspection. In staggered mode `time=` is the calendar column and emits NO
  rename warning; in 2x2x2 mode it stays the deprecated alias for `post=` (row
  M-031) and warns. At 4.0 it means the calendar column only (row M-085).
- **Note (vocabulary):** the third dimension is `partition` in both modes, and
  `control_group` takes the underscored values `"not_yet_treated"` /
  `"never_treated"`. R's compact spellings are accepted only by the deprecated
  `StaggeredTripleDifference` and die with it.
- **Note (`cluster=` raises in staggered mode):** cluster-robust ANALYTICAL SEs
  are not implemented for the staggered engine, so on this surface `cluster=`
  raises a `ValueError` steering to `n_bootstrap > 0` (unit-level clustering via
  the multiplier bootstrap). The deprecated class keeps its 3.x
  accepted-then-warned-then-ignored `UserWarning`. **Deviation from the dying
  class, deliberate:** the raise is live from the mode's 3.9 birth, because a new
  surface needs no deprecation window and should not accept a parameter it does
  not honor. In 2x2x2 mode `cluster=` continues to give Liang-Zeger CR1.
- **Note (unreachable-by-construction params):** `bootstrap_weights`, `seed` and
  `cband` pass silently in 2x2x2 mode. They take effect only through
  `n_bootstrap > 0`, which 2x2x2 mode rejects outright, so they are unreachable
  rather than silently ignored. `control_group`, `anticipation`, `base_period` and
  `n_bootstrap` themselves all raise in 2x2x2 mode. **The power surface applies
  the SAME boundary:** `simulate_power` / `simulate_mde` / `simulate_sample_size`
  reject only those four and accept the three inert ones, so an estimator that is
  legal to `fit()` is legal to simulate. That equivalence extends to the FIT
  kwargs, and the test differs per param because `fit()`'s does: `first_treat`
  and `unit` are sentinel-defaulted, so SUPPLYING them at all selects staggered
  mode (an explicit `first_treat=None` still does, then fails on a missing
  column), while `aggregate`/`balance_e` default to `None` and only a NON-`None`
  value is staggered-only. Power therefore keys on presence for the first pair
  and on value for the second — keying on presence throughout would reject
  `estimator_kwargs={"aggregate": None}`, a configuration `fit()` accepts. An earlier 3(b) revision rejected all
  seven at the power front door, which made `TripleDifference(seed=7)` fittable
  but not simulatable - `seed` is set habitually for reproducibility and the
  `simulate_*` helpers take their own separate `seed=`.
- **Note (`pscore_trim` validation tightened, ledger row M-142):** the merged
  constructor adopts the staggered engine's `0 < pscore_trim < 0.5` rule;
  `TripleDifference`'s `pscore_trim` was previously unvalidated. The value feeds
  `np.clip(pscore, trim, 1 - trim)` in both engines, so `trim=0` disabled the
  overlap guard that keeps the `1/(1-p)` IPW/DR weights finite and `trim >= 0.5`
  inverted the clip bounds. `TripleDifference(pscore_trim=0)` therefore changes
  from accepted to a loud `ValueError`. **Sibling divergence, recorded rather than
  silently tolerated:** `ContinuousDiD` still validates `0.0 <= pscore_trim < 0.5`
  and so still admits `0`; aligning it is out of scope for a DDD merge and is
  tracked as a `TODO.md` row.
- **Note (the `triple_difference()` wrapper stays 2x2x2-only):** the deprecated
  functional wrapper is deliberately NOT extended to staggered mode. It reaches
  only the 2x2x2 design and forwards its own `time=` as `post=`. Three reasons,
  all program-level rather than local: the wrapper is itself deprecated in 3.9
  and removed at 4.0 (row M-075), so widening it would mint new removal
  obligations for params introduced solely to be deleted; phase 3(a) set the
  precedent by extending no wrapper with its new event-study mode; and the
  wrapper's own deprecation message already steers to the class, which is where
  staggered mode lives. A caller who needs staggered DDD migrates to
  `TripleDifference(...).fit(..., first_treat=...)` - the migration the row
  exists to produce. Pinned by
  `tests/test_v4_merge_ddd.py::TestGateHConsumers::test_deprecated_wrapper_stays_2x2x2_only`.
- **Note (bootstrap `seed` is estimator state, not results provenance):** the
  staggered container does NOT carry the `seed` that generated its bootstrap
  SEs, p-values and sup-t bands; it is readable from the estimator
  (`est.seed` / `get_params()`), not from the returned result. `seed` IS fully
  propagated in the operative sense - it reaches the multiplier draws, so the
  same seed reproduces an SE bit-exactly and a different seed moves it - and it
  appears in `get_params()`/`set_params()`. What is absent is a results-object
  FIELD. This is deliberate and pre-dates the 3.9 merge: `seed` was already a
  `StaggeredTripleDifference` constructor param that its container never
  recorded, and `CallawaySantAnnaResults` has exactly the same gap through the
  shared `CallawaySantAnnaBootstrapMixin`. Adding the field to one container
  alone would leave the two siblings inconsistent, and
  `StaggeredTripleDiffResults` is already slated for the 4.0 container port
  (row M-014), so the fix is sequenced there and tracked as a `TODO.md` row
  covering BOTH containers. Estimators that do expose it
  (`ContinuousDiDResults`, `EfficientDiDResults`, `SyntheticDiDResults`) are
  the precedent for the eventual shape.
- **Note (`anticipation` domain validated from birth):** `anticipation` is one of
  the seven staggered-only constructor params introduced on `TripleDifference` in
  3.9, and it is validated as a non-negative integer (`bool` rejected) from that
  birth - so this is a new param's input contract, not a tightening of an existing
  one, and it needs no lifecycle row. The guard is load-bearing rather than
  cosmetic: the value feeds BOTH the base-period rule and the comparison-cohort
  threshold, so `anticipation=-1` would make the universal base period `g` - an
  ALREADY-TREATED period - and relax the not-yet-treated threshold to
  `max(t, base) - 1`, admitting cohorts treated at the evaluation period as clean
  controls. Neither condition is observable in the output, so an unvalidated
  window is a silent estimand change (see the no-silent-failures policy).
  `bool` is rejected because `True` would otherwise coerce to a silent
  one-period window. The check is the shared `utils.validate_anticipation`, and
  it is ALSO called from the staggered engine, so the deprecated
  `StaggeredTripleDifference` fails closed too — its frozen 3.x API shape was
  never a licence to emit silently-biased numbers from the same engine. The
  split is deliberate: on the dying class CONSTRUCTION still succeeds (its
  signature contract is untouched) and FIT raises, because this is an
  identification guard rather than an API change. The engine-level call also
  covers direct attribute mutation, which bypasses `__init__` and `set_params`
  alike. **Sibling divergence, recorded rather than silently tolerated:** among
  the remaining estimators taking `anticipation`, only `spillover.py` and
  `wooldridge.py` validate it (and neither rejects `bool`); aligning the family
  behind the shared helper is tracked as a `TODO.md` row.
- **Note (degenerate enabling cohorts are reported, not hidden):** a positive
  `first_treat` cohort whose units are ALL `partition == 0` cannot identify
  `ATT(g,t)` — the DDD contrast needs the eligible-treated cell — so it
  contributes to no aggregate. That was previously invisible: the cohort was
  still listed in `results.groups`/`n_groups`, and the only warnings naming it
  described its role as a COMPARISON cohort for other `g`, never as a treated
  cohort that dropped out. The estimate quietly covered fewer cohorts than the
  metadata advertised. Now a `UserWarning` names the cohort, its unit count and
  the consequence, and `groups` is derived from the cohorts that actually
  produced a `(g, t)` cell — so it is honest for every drop-out reason (no
  eligible treated units, no valid comparison cohort, all base periods outside
  the panel), not just this one. **The estimate is deliberately unchanged:** it
  remains valid for the cohorts that do identify, and this engine already
  warns-and-skips rather than raising for unidentified CELLS, so raising for a
  cohort would have been the odd one out. Pre-existing behavior on
  `StaggeredTripleDifference`; the fix is in the shared engine, so both
  surfaces report identically.
- **Note (`first_treat` cohort encoding is fail-closed on negatives):** the
  contract is `0` for never-treated (or `+inf`, recoded to `0` with a
  `UserWarning`) and positive period labels for treated cohorts. Negative values
  — including the common `-1` never-treated convention and `-inf` — raise a
  `ValueError` naming the offending values. They are NOT recoded: unlike `+inf`,
  a negative value carries no unambiguous intent, and guessing would be the
  silent sample change the guard exists to stop. Why it is a guard and not a
  documentation matter: `_precompute_structures` builds the treated set from
  `g > 0` and the never-enabled set from `g == 0`, so a negatively-encoded unit
  belonged to NEITHER — still counted toward `n_obs` while contributing to no
  ATT comparison — and the fit returned a plausible finite estimate for a
  different population (measured on the 3(b) fixture: `overall_att`
  3.29438582 → 2.99470938 with `n_never_enabled` 24 → 0, no error and no
  warning). The guard lives in the shared engine, so BOTH surfaces fail closed;
  the behavior it replaces was pre-existing on `StaggeredTripleDifference`.
- **Note (results containers):** 2x2x2 mode returns `TripleDifferenceResults` and
  staggered mode returns `StaggeredTripleDiffResults` through 3.9; `fit`'s return
  type is a `Union`. The one-shape unification lands at 4.0 with row M-014, so no
  downstream consumer changes in 3.9.

## StaggeredTripleDifference

- **Note:** Deprecated in 3.9, removed in 4.0 (ledger row M-013). The successor is
  `TripleDifference().fit(..., unit=, time=, first_treat=, partition=)`, which runs
  this EXACT engine - the estimation core moved verbatim to
  `diff_diff/_staggered_triple_diff_engine.py` and both classes mix it in, so the
  numbers are identical by construction (pinned bit-exactly in
  `tests/test_v4_merge_ddd.py`). Two vocabulary differences on the merged surface:
  `eligibility=` is named `partition=`, and `control_group` takes the underscored
  values `"not_yet_treated"`/`"never_treated"` (R's compact spellings stay on this
  class and die with it). One behavior difference: `cluster=` RAISES on the merged
  surface instead of being accepted-then-warned-then-ignored, because a new surface
  should not ship a parameter it does not honor. Everything below remains the 3.x
  contract for the deprecated class through its removal.

**Primary source:** [Ortiz-Villavicencio, M., & Sant'Anna, P.H.C. (2025). Better Understanding Triple Differences Estimators. arXiv:2505.09942v3.](https://arxiv.org/abs/2505.09942v3). Paper review on file: `docs/methodology/papers/ortiz-villavicencio-santanna-2025-review.md`.

**Key implementation requirements:**

*Assumption checks / warnings:*
- Requires balanced panel with enabling-group `S_i`, binary eligibility `Q_i` (time-invariant), and outcome `Y`
- Eligibility must be binary (0/1) — raises `ValueError` if not
- Eligibility must be time-invariant within each unit — raises `ValueError` if varying
- Requires both eligible (Q=1) and ineligible (Q=0) units
- Warns if any (S, Q) cell in a three-DiD comparison has < 5 units
- Warns if no valid comparison groups exist for a (g, t) pair (skips that pair)
- Propensity score overlap enforced by clipping at `pscore_trim` (default 0.01)
- **Events Per Variable (EPV) diagnostics:** Per-DiD EPV =
  min(n_subgroup_j, n_subgroup_4) / n_covariates checked before IRLS.
  Default threshold: 10 (Peduzzi et al. 1996). Warns when EPV < threshold;
  errors when `rank_deficient_action="error"`.
- **Note:** When multiple comparison cohorts `g_c` contribute to the same
  ATT(g,t) cell, `results.epv_diagnostics[(g,t)]` retains the worst-case
  (minimum EPV) across all contributing propensity fits, rather than per-fit
  diagnostics. This is a conservative cell-level summary.
- Propensity score estimation failure: controlled by `pscore_fallback` parameter
  (default `"error"`). If `pscore_fallback="error"`, the error is raised. If
  `pscore_fallback="unconditional"`, falls back to unconditional propensity with
  warning. When `rank_deficient_action="error"`, errors are always re-raised.
- **Note:** `pscore_fallback` default changed from unconditional to error.
  Set `pscore_fallback="unconditional"` for legacy behavior.
- Warns on singular GMM covariance matrix (falls back to pseudoinverse)
- **Note:** Rank-deficient X'WX in the per-pair outcome-regression influence-function step (`_compute_did_panel`) is inverted by the shared rank-guarded generalized inverse `_rank_guarded_inv` (see the CallawaySantAnna "rank-guarded IF standard errors" Note): a near-singular X'WX no longer returns a garbage inverse (the prior `np.linalg.solve` only raised on exact singularity) — redundant directions are truncated, giving a finite SE on the identified covariate subset. `fit()` emits ONE aggregate `UserWarning` (counting affected (g, g_c, t) cells + max condition number), suppressed under `rank_deficient_action="silent"`. Axis-A finding #17 in the Phase 2 silent-failures audit.
- **Note (rank-guard column-drop):** The rank-guarded IF inverse uses an **equilibrated** column-drop generalized inverse (pivoted QR on the equilibrated Gram) — the same column-drop *family* as the point estimate, with a scale-invariant column selection that may differ from the point estimate's raw pivot only under mixed-scale *exact* collinearity (a documented deviation that leaves the SE unchanged). The analytical SE equals the well-conditioned near-collinear limit (`se_ratio ≈ 1`) for StaggeredTripleDifference's per-pair OR bread (`_compute_did_panel`) and PS Hessian (`_compute_pscore`), with no minimum-norm divergence — see the CallawaySantAnna "rank-guarded IF standard errors" Note (and its selection caveat) for the full statement.
- **Note:** The per-pair propensity-score Hessian inversion in `_compute_pscore` (used under `estimation_method` in `{ipw, dr}`) uses the same rank-guarded generalized inverse `_rank_guarded_inv`: the prior `np.linalg.inv(X'WX)` returned a garbage inverse on a near-singular (but not exactly singular) PS design, inflating IPW/DR influence-function corrections. Redundant directions are now truncated and `fit()` emits a sibling aggregate `UserWarning` (cell count + max condition number), suppressed under `rank_deficient_action="silent"`. Sibling of axis-A finding #17, surfaced during PR #334 CI review.

*Data structure:*

Balanced panel. Key variables:
- `S_i` (`first_treat`): enabling group — 0 or inf for never-enabled
- `Q_i` (`eligibility`): binary, time-invariant eligibility indicator
- Treatment: `D_{i,t} = 1{t >= S_i AND Q_i = 1}` (absorbing)
- Covariates `X_i`: time-invariant (first observation per unit used)
- **Note (scale-equilibrated OR solve):** StaggeredTripleDifference's per-cohort outcome-regression fit (`_compute_or`) now routes through the shared scale-equilibrated `solve_ols` (column-equilibrated SVD/gelsd; matches TripleDifference's `_fit_predict_mu` and R's `lm()`/QR), replacing the prior estimator-local `cho_solve(X'X)` cache fast path. A covariate on a very large raw scale (e.g. correlated with the intercept via a large constant offset) no longer perturbs the nuisance fit; the equilibrated SVD is offset-invariant to ~1e-11 where the prior normal-equations solve drifted (a `tests/test_methodology_staggered_triple_diff.py` scale-invariance test pins this — covariate + 1e6 offset leaves ATT(g,t) unchanged). The change is not bit-identical (cho/normal-equations → SVD) but well-scaled designs move only ~1e-12. (The separate DR/OR influence-function SE rank-guard is also **implemented** via `_rank_guarded_inv` — see the CallawaySantAnna "rank-guarded IF standard errors" Note.)
- **Note:** `first_treat=inf` (R-style never-enabled marker) is accepted and normalized to `0` internally. The recoding now emits a `UserWarning` reporting the affected row count so the reclassification is not silent (axis-E silent coercion under the Phase 2 audit, mirroring the StaggeredDiD behavior). Pass `first_treat=0` directly to avoid the warning.

*Estimator equation (Equation 4.1 in paper, as implemented):*

Three-DiD decomposition for each (g, g_c, t) triple:

```
DDD(g, g_c, t) = DiD_A + DiD_B - DiD_C
```

where each pairwise DiD operates on panel outcome changes `delta_Y = Y_t - Y_b`:
- DiD_A: treated (S=g, Q=1) vs (S=g, Q=0)     [+1, paper Term 1]
- DiD_B: treated (S=g, Q=1) vs (S=g_c, Q=1)   [+1, paper Term 2]
- DiD_C: treated (S=g, Q=1) vs (S=g_c, Q=0)   [-1, paper Term 3]

This sign convention matches both the paper's Equation 4.1 and the existing
`TripleDifference` decomposition (DDD = DiD_3 + DiD_2 - DiD_1 with subgroups
4=G1P1, 3=G1P0, 2=G0P1, 1=G0P0).

Valid comparison groups: for `control_group="nevertreated"`, only the never-enabled
cohort (S=0). For `control_group="notyettreated"`, `G_c = {g_c : g_c > max(t, base_period)
+ anticipation}`, plus never-enabled.

- **Deviation from paper:** The paper's Section 4 defines admissible comparison cohorts
  as `g_c > max(g, t)`. The implementation follows the companion R package `triplediff`
  which uses `g_c > max(t, base_period) + anticipation`. These rules differ for
  pre-treatment cells (`t < g`) when a later cohort lies in `(t, g)`: the paper would
  exclude it, while the R package (and this implementation) may include it depending
  on the base period. The R-matching rule correctly accounts for the anticipation
  parameter and base-period selection in the comparison-group filter.

*With covariates / doubly robust (DR, recommended):*

Each pairwise DiD uses the CallawaySantAnna DR estimator on outcome changes:
1. Fit outcome regression `E[delta_Y | X]` on control units (OLS)
2. Estimate propensity score `P(treated | X)` within each 2-cell subset (logistic)
3. Combine: `ATT = mean(treated_change - m_hat) + sum(w_ipw * (m_hat - control_change)) / n_t`

*GMM-optimal combination across comparison groups (Equations 4.11-4.12):*

```
ATT_gmm(g,t) = w_gmm' @ [ATT_1, ..., ATT_k]
w_gmm = Omega^{-1} @ 1 / (1' @ Omega^{-1} @ 1)
```

where `Omega[j,l] = (1/n) * sum_i IF_j[i] * IF_l[i]` is estimated from influence
functions across comparison groups. Minimizes asymptotic variance subject to `sum(w) = 1`.

*Aggregation:*

Event study (Equation 4.13): cohort-share-weighted average across cohorts for each
relative time `e = t - g`. Reuses `CallawaySantAnnaAggregationMixin._aggregate_event_study()`.

Overall ATT: two summaries are available. (1) The default `overall_att` is the
cohort-size-weighted average across post-treatment (g,t) pairs (reuses
`CallawaySantAnnaAggregationMixin._aggregate_simple()`) — the library-wide
Callaway-Sant'Anna "simple" convention, matching R `agg_ddd(type="simple")`.
(2) The opt-in `overall_att_es` is the paper's Equation 4.14 overall — the unweighted
mean of the post-treatment event-study effects ES(e) — populated when
`aggregate="event_study"` or `"all"` (see the labeled Note below).

Group effects: average across post-treatment time periods for each cohort.
Reuses `CallawaySantAnnaAggregationMixin._aggregate_by_group()`.

All aggregation SEs include the WIF (Weight Influence Function) adjustment for
uncertainty in cohort-share weights, inherited from the CallawaySantAnna mixin.

- **Deviation from R:** Aggregation weights (and the WIF) use the eligible-treated
  population `P(S=g, Q=1)` — matching the paper's Eq 4.13, where `G_i` is defined
  only for `Q=1` units (`G_i = g` iff `S_i = g` and `Q_i = 1`), so the paper's
  `P(G=g)` *is* `P(S=g, Q=1)`. R's `agg_ddd()` instead weights by `P(S=g)` (all units
  in the enabling group, including ineligible). Implemented by setting `unit_cohorts=0`
  for ineligible units before calling the aggregation mixin. Group-time `ATT(g,t)`
  values are identical to R; only the weighted average across (g,t) pairs differs —
  this is the source of the larger tolerance in the R cross-validation tests.
- **Note:** Per-cohort group-effect SEs include WIF via the inherited mixin.
  R's `agg_ddd(type="group")` uses `wif=NULL` for per-cohort aggregation since
  within-cohort weights are fixed. This makes our per-cohort group-effect SEs
  slightly conservative relative to R.
- **Note:** The default overall ATT (`overall_att`) is the Callaway-Sant'Anna simple
  post-treatment (g,t) average — the library-wide convention across staggered
  estimators — and is NOT the paper's Equation 4.14 overall (which averages the
  event-study effects). The paper's Eq 4.14 form is exposed opt-in as `overall_att_es`
  (populated only when `aggregate` is `"event_study"`/`"all"`), computed as the
  unweighted mean of the post-treatment ES(e) over `e >= -anticipation`. Its analytical
  SE is the influence function of that mean (the average of the per-event-time combined
  IFs, routed through the same survey-aware variance estimator as the per-e effects);
  a multiplier-bootstrap SE replaces it when `n_bootstrap > 0`. The two summaries answer
  different questions and generally differ; `overall_att_es` is cross-validated against
  R `agg_ddd(type="eventstudy")$overall.att`/`overall.se`.

*Standard errors:*

Individual (g,t) level:
```
SE(g,t) = std(IF_gmm, ddof=1) / sqrt(n)
```
where `IF_gmm = w_gmm' @ IF_matrix` is the GMM-combined unit-level influence function
(length n_units, zero-padded for non-participating units). Inherently
heteroskedasticity-robust via the influence function approach.

Aggregation SEs: via WIF-adjusted combined influence functions from the
CallawaySantAnna aggregation mixin.

Bootstrap: multiplier bootstrap (Algorithm 1 of Callaway & Sant'Anna 2021) via
`CallawaySantAnnaBootstrapMixin._run_multiplier_bootstrap()`. Supports
Rademacher, Mammen, and Webb weight distributions. Provides simultaneous
confidence bands (sup-t) for event study.

- **Note:** Matches R `triplediff` package `compute_did()` formulation:
  Hajek-normalized Riesz representers, separate M1/M3 OR corrections on
  treated/control IF components, PS correction via logistic Hessian and score
  function, hessian = (X'WX)^{-1} * n_pair. Three-DiD IF combination weights
  use `w_j = n_cell / n_pair_j` (matching R's att_dr). GMM Omega estimated via
  sample covariance (ddof=1). Per-(g,t) SE uses R's GMM formula
  `sqrt(1 / (n * sum(Omega_inv)))` for multiple comparison groups, or
  `sqrt(sum(IF^2) / n^2)` for single comparison group.
- **Deviation from R:** Propensity scores are clipped to `[pscore_trim, 1-pscore_trim]`
  (default 0.01). R's `triplediff` uses hard exclusion (`keep_ps`) for control units
  with `pscore >= 0.995` but does not apply a lower bound. The soft-clipping approach
  retains all observations with bounded weights, which is more conservative under
  moderate overlap violations.
- **Note:** The `cluster` parameter is accepted but not currently wired to the
  analytical SE computation. The multiplier bootstrap provides unit-level
  clustering. Full cluster-robust analytical SEs are deferred.
- **Note:** Full survey design support (pweight only). Survey weights enter
  propensity score estimation (weighted IRLS), outcome regression (WLS), and
  Riesz representer computation. IF combination weights (w1/w2/w3) use
  survey-weighted cell sizes. Aggregated SEs use `compute_survey_if_variance()`
  (TSL) or `compute_replicate_if_variance()` (replicate weights). Bootstrap
  uses PSU-level multiplier weights. The R `triplediff` package does not
  support survey weights.

*Edge cases:*
- Single comparison group: GMM reduces to w=[1], no matrix inversion
- Zero valid comparison groups for a (g,t): skipped with warning
- Singular GMM covariance: falls back to pseudoinverse with warning
- Small cells (< 5 units): warns but proceeds
- Non-finite ATT from a comparison group: excluded from GMM combination
- Never-enabled encoded as inf: normalized to 0 internally
- No valid (g,t) pairs at all: raises `ValueError`

**Reference implementation(s):**
- R `triplediff` (companion package by paper authors) — cross-validated in
  `tests/test_methodology_staggered_triple_diff.py` (group-time ATT/SE for both control
  groups, plus the Eq. 4.14 overall `overall_att_es` against `agg_ddd(type="eventstudy")`).
  CSV fixtures are gitignored and regenerated on-the-fly from
  `benchmarks/R/benchmark_staggered_triplediff.R`; the JSON golden is committed.

**Requirements checklist:**
- [x] Panel data with (unit, time, enabling-group S, eligibility Q, outcome Y)
- [x] Three comparison sub-groups per (g, g_c): (S=g, Q=0), (S=g_c, Q=1), (S=g_c, Q=0)
- [x] Individual comparison cohorts, never pooled — combined via GMM weights
- [x] Comparison groups satisfy g_c > max(t, base_period) + anticipation (notyettreated)
  or g_c = never-enabled only (nevertreated)
- [x] Doubly robust: consistent if either propensity or outcome model correct (per component)
- [x] GMM-optimal weighting via closed-form inverse-variance formula
- [x] Event-study aggregation with cohort-share weights (via CS mixin)
- [x] Pre-treatment event-study coefficients constructable
- [x] Influence-function-based SEs
- [x] Multiplier bootstrap for simultaneous confidence bands (via CS mixin)
- [ ] Cluster-robust analytical SEs (accepted but not wired — deferred)
- [x] Survey design support (pweight, strata/PSU/FPC, replicate weights)
- [x] Validation against R `triplediff` package: group-time ATT and SE match within
  0.001% across 10 scenarios (3 seeds, 3 methods, both control group modes).
  Aggregation (event study, overall ATT) uses CS mixin cohort-size weights which
  differ from R's `agg_ddd()` group-probability weights (within 25%); this is a
  documented weighting choice, not a specification violation.

---

## TROP

**Primary source:** [Athey, S., Imbens, G.W., Qu, Z., & Viviano, D. (2025). Triply Robust Panel Estimators. arXiv:2508.21536.](https://arxiv.org/abs/2508.21536)

**Note (version pinning):** the methodology promotion (`METHODOLOGY_REVIEW.md` `#### TROP` → **Complete** as of 2026-05-24) is anchored on **arXiv:2508.21536v2**; the current arXiv version is **v3**. The **v3 PDF was consulted for the treatment-assignment-pattern sections** as part of the non-absorbing support work (§2.1 general assignment / "units moving into and out of treatment"; §2.2 Eq. 2 masking; §6.1 Eq. 12 / Algorithm 2; Assumption 1(i); Theorem 5.1) and confirms the general-assignment scope used here. A full v2→v3 source delta-check across all promoted sections (Eqs. 2-3, Algorithms 1-3, Section 2.2, Section 5.2-5.3, Section 6.1-6.2, Theorem 5.1, Corollary 1, Appendix Theorem 8.1) is still **deferred**. See `docs/methodology/papers/athey-2025-review.md` "Version-pinning note" for the deferred action item.

**Key implementation requirements:**

*Assumption checks / warnings:*
- Requires sufficient pre-treatment periods for factor estimation (at least 2)
- Unit weights can become degenerate if λ_unit too large
- Returns Q(λ) = ∞ if ANY LOOCV fit fails (Equation 5 compliance)

*Treatment indicator (D matrix) semantics:*

By default (`non_absorbing=False`) D must be an **ABSORBING STATE** indicator, not a
treatment timing indicator:
- D[t, i] = 0 for all t < g_i (pre-treatment periods for unit i)
- D[t, i] = 1 for all t >= g_i (during and after treatment for unit i)

where g_i is the treatment start time for unit i.

For **staggered adoption** (different units treated at different times, but still
absorbing) the D matrix naturally handles this - distances use periods where BOTH units
have D=0, matching the paper's (1 - W_iu)(1 - W_ju) formula in Equation 3.

**True non-absorbing assignment** (treatment switches on *and* off) is a distinct case
from staggered adoption. The paper (§2.1: "units moving into and out of treatment")
supports it via the same Eq. 12 / Algorithm 2 masking, and the library exposes it through
the opt-in `TROP(non_absorbing=True)` (`method='local'` only). See the requirements
checklist below and the `**Note:**` entries on the no-dynamic-effects requirement and the
block-only inference theory.

**Wrong D specification**: With the default `non_absorbing=False`, an event-style D (only
the first treatment period has D=1, then back to 0) is a non-monotonic indicator and is
**rejected** with a `ValueError` (see "D matrix validation" below). This guards against the
common mistake of encoding absorbing treatment as an event spike, which would silently bias
the ATT. A user with genuinely non-absorbing treatment passes `non_absorbing=True`.

*ATT definition (Equation 1, Section 6.1):*
```
τ̂ = (1 / Σ_i Σ_t W_{it}) Σ_{i=1}^N Σ_{t=1}^T W_{it} τ̂_{it}(λ̂)
```
- ATT averages over all cells where D_it=1 (treatment indicator) that are
  **estimable**. On balanced / support-complete absorbing panels every treated
  cell is estimable, so this is all D=1 cells. A cell is non-estimable (NaN,
  excluded) when `alpha_i + beta_t` is unidentified — its target unit and period
  are not in the same connected component of the observed-control graph; this is
  reachable under `non_absorbing=True` (always-treated unit, fully-treated period,
  disconnected support) and on unbalanced absorbing panels (entirely-missing
  unit/period controls). See the non-estimable-cell `**Note:**` below — matching
  the library-wide non-estimable→NaN convention (cf. CallawaySantAnna group-time
  cells).
- No separate "post_periods" concept - D matrix is the sole input for treatment timing
- Supports general assignment patterns including staggered adoption and (with
  `non_absorbing=True`) on/off switching

*Estimator equation (as implemented, Section 2.2):*

Working model (separating unit/time FE from regularized factor component):
```
Y_it(0) = α_i + β_t + L_it + ε_it,   E[ε_it | L] = 0
```
where α_i are unit fixed effects, β_t are time fixed effects, and L = UΣV' is a low-rank
factor structure. The FE are estimated separately from L because L is regularized but
the fixed effects are not.

Optimization (Equation 2):
```
(α̂, β̂, L̂) = argmin_{α,β,L} Σ_j Σ_s θ_s^{i,t} ω_j^{i,t} (1-W_js)(Y_js - α_j - β_s - L_js)² + λ_nn ||L||_*
```
Solved via alternating minimization. For α, β: weighted least squares (closed form).
The global solver adds an intercept μ and solves for (μ, α, β, L) on control data only,
extracting τ_it post-hoc as residuals (see Global section below).
For L: proximal gradient with step size η = 1/(2·max(W)):
```
Gradient step: G = L + (W/max(W)) ⊙ (R - L)
Proximal step: L = U × soft_threshold(Σ, η·λ_nn) × V'  (SVD of G = UΣV')
```
where R is the residual after removing fixed effects.
Both the local and global solvers use FISTA/Nesterov acceleration for the
inner L update (O(1/k²) convergence rate, up to 20 inner iterations per
outer alternating step).

Per-observation weights (Equation 3):
```
θ_s^{i,t}(λ) = exp(-λ_time × |t - s|)

ω_j^{i,t}(λ) = exp(-λ_unit × dist^unit_{-t}(j, i))

dist^unit_{-t}(j, i) = (Σ_u 1{u≠t}(1-W_iu)(1-W_ju)(Y_iu - Y_ju)² / Σ_u 1{u≠t}(1-W_iu)(1-W_ju))^{1/2}
```
Note: weights are per-(i,t) observation-specific. The distance formula excludes the
target period t and uses only periods where both units are untreated (W=0).

*Special cases (Section 2.2):*
- λ_nn=∞, ω_j=θ_s=1 (uniform weights) → recovers DID/TWFE
- ω_j=θ_s=1, λ_nn<∞ → recovers Matrix Completion (Athey et al. 2021)
- λ_nn=∞ with specific ω_j, θ_s → recovers SC/SDID

*LOOCV tuning parameter selection (Equation 5, Footnote 2):*
```
Q(λ) = Σ_{j,s: D_js=0} [τ̂_js^loocv(λ)]²
```
- Score is **SUM** of squared pseudo-treatment effects on control observations
- **Two-stage procedure** (per paper's footnote 2):
  - Stage 1: Univariate grid searches with extreme fixed values
    - λ_time search: fix λ_unit=0, λ_nn=∞ (disabled)
    - λ_nn search: fix λ_time=0 (uniform time weights), λ_unit=0
    - λ_unit search: fix λ_nn=∞, λ_time=0
  - Stage 2: Cycling (coordinate descent) until convergence
- **"Disabled" parameter semantics** (per paper Section 4.3, Table 5, Footnote 2):
  - `λ_time=0`: Uniform time weights (disabled), because exp(-0 × dist) = 1
  - `λ_unit=0`: Uniform unit weights (disabled), because exp(-0 × dist) = 1
  - `λ_nn=∞`: Factor model disabled (L=0), because infinite penalty; converted to `1e10` internally
  - **Note**: `λ_nn=0` means NO regularization (full-rank L), which is the OPPOSITE of "disabled"
  - **Validation**: `lambda_time_grid` and `lambda_unit_grid` must not contain inf. A `ValueError` is raised if they do, guiding users to use 0.0 for uniform weights per Eq. 3.
- **LOOCV failure handling** (Equation 5 compliance):
  - If ANY LOOCV fit fails for a parameter combination, Q(λ) = ∞
  - A warning is emitted on the first failure with the observation (t, i) and λ values
  - Subsequent failures for the same λ are not individually warned (early return)
  - This ensures λ selection only considers fully estimable combinations

*Standard errors:*
- Block bootstrap preserving panel structure (Algorithm 3)

*Edge cases:*
- Rank selection: implicit via nuclear-norm soft-thresholding (paper Section 5.3 + Appendix); `TROPResults.effective_rank` reports the diagnostic. No discrete `rank_selection` constructor parameter (cv / ic / elbow) is exposed — earlier prose claiming "automatic via cross-validation, information criterion, or elbow" was an overclaim, corrected in the 2026-05-24 methodology-promotion PR. See the Requirements checklist Rank-selection bullet below.
- Zero singular values: handled by soft-thresholding
- Extreme distances: weights regularized to prevent degeneracy
- LOOCV fit failures: returns Q(λ) = ∞ on first failure (per Equation 5 requirement that Q sums over ALL control observations where D==0); if all parameter combinations fail, falls back to defaults (1.0, 1.0, 0.1)
- **λ_nn=∞ implementation**: Only λ_nn uses infinity (converted to 1e10 for computation):
  - λ_nn=∞ → 1e10 (large penalty → L≈0, factor model disabled)
  - Conversion applied to grid values during LOOCV (including Rust backend)
  - Conversion applied to selected values for point estimation
  - Conversion applied to selected values for variance estimation (ensures SE matches ATT)
  - **Results storage**: `TROPResults` stores *original* λ_nn value (inf), while computations use 1e10. λ_time and λ_unit store their selected values directly (0.0 = uniform).
- **Empty control observations**: If no valid control observations exist, returns Q(λ) = ∞ with warning. A score of 0.0 would incorrectly "win" over legitimate parameters.
- **Infinite LOOCV score handling**: If best LOOCV score is infinite, `best_lambda` is set to None, triggering defaults fallback
- Validation: by default requires at least 2 periods before first treatment; with `non_absorbing=True` this becomes "at least 2 periods contain untreated cells" (the leading all-control block is ill-defined when treatment toggles)
- **D matrix validation** (default `non_absorbing=False`): Treatment indicator must be an absorbing state (monotonic non-decreasing per unit)
  - Detection: `np.diff(D, axis=0) < 0` for any column indicates violation
  - Handling: Raises `ValueError` with list of violating unit IDs and remediation guidance
  - Error message includes: "convert to absorbing state: D[t, i] = 1 for all t >= first treatment period" AND the opt-in pointer ("if treatment genuinely turns on and off, pass `non_absorbing=True`")
  - **Rationale**: Event-style D (0→1→0) silently biases ATT when the user *meant* absorbing treatment; runtime validation prevents that misuse while the opt-in serves genuine on/off designs
  - **`non_absorbing=True`**: the monotonicity check is skipped entirely, so on/off (and event-style) D matrices are accepted. Identification falls back to untreated *cells* (the per-(i,t) estimator masks treated cells via (1-W) and fits the rest), so even a fully toggling panel with no never-treated unit is admitted; only "no D=0 cells at all" is rejected. See the requirements checklist + Notes for the no-dynamic-effects requirement and the block-only inference caveat.
  - **Unbalanced panels**: Missing unit-period observations are allowed. Monotonicity validation (default mode) checks each unit's *observed* D sequence for monotonicity, which correctly catches 1→0 violations that span missing period gaps (e.g., D[2]=1, missing [3,4], D[5]=0 is detected as a violation even though the gap hides the transition in adjacent-period checks).
  - **n_post_periods metadata**: Counts periods where D=1 is actually observed (at least one unit has D=1), not calendar periods from first treatment. In unbalanced panels where treated units are missing in some post-treatment periods, only periods with observed D=1 values are counted.
- Wrong D specification: with the default `non_absorbing=False`, an event-style D (only first treatment period) is rejected with a `ValueError` carrying both the convert-to-absorbing guidance and the `non_absorbing=True` opt-in pointer
- **Bootstrap minimum**: `n_bootstrap` must be >= 2 (enforced via `ValueError`). TROP uses bootstrap for all variance estimation — there is no analytical SE formula.
- **Note:** TROP bootstrap loops (`_bootstrap_variance`, `_bootstrap_rao_wu`, and their global counterparts, including both Rust happy paths — local and global) emit a proportional `UserWarning` via `diff_diff.bootstrap_utils.warn_bootstrap_failure_rate` when the replicate failure rate exceeds 5%. The previous hard-coded `< 10 successes` threshold let high-failure runs (e.g. 11 of 200) pass silently; this was classified as a silent failure under the Phase 2 audit (axis D — degenerate-replicate handling). The 5% threshold matches the existing SyntheticDiD bootstrap and placebo guards. When zero replicates succeed, SE is set to `NaN` (unchanged). The local Rust path previously also used `len >= 10` as a Python-fallback trigger; it now accepts any non-zero Rust result and emits the proportional warning instead of path-switching silently.
- **LOOCV failure metadata**: When LOOCV fits fail in the Rust backend, the first failed observation coordinates (t, i) are returned to Python for informative warning messages
- **Inference CI distribution**: After `safe_inference()` migration, CI uses t-distribution (df = max(1, n_treated_obs - 1)), consistent with p_value. Previously CI used normal-distribution while p_value used t-distribution (inconsistent). This is a minor behavioral change; CIs may be slightly wider for small n_treated_obs.
- **Note:** Both the `local` alternating-minimization solver (`_estimate_model`) and the `global` alternating-minimization solver (`_solve_global_with_lowrank`, including its hard-coded inner FISTA loop of 20 iterations) emit `UserWarning` via `diff_diff.utils.warn_if_not_converged` when the outer loop exhausts `max_iter` without reaching `tol`. The global-method warning surfaces the inner-FISTA non-convergence count as diagnostic context. Silent return of the current iterate was classified as a silent failure under the Phase 2 audit and replaced with an explicit signal to match the convention used across other iterative solvers in the library.

**Reference implementation(s):**
- Authors' replication code (forthcoming)

**Requirements checklist:**
- [x] Factor matrix estimated via soft-threshold SVD
- [x] Unit weights: `exp(-λ_unit × distance)` (unnormalized, matching Eq. 2)
- [x] LOOCV implemented for tuning parameter selection
- [x] LOOCV uses SUM of squared errors per Equation 5
- [x] Rank selection implicit via nuclear-norm soft-thresholding (paper Section 5.3 + Appendix); `TROPResults.effective_rank` reports the diagnostic. No discrete `rank_selection` constructor parameter is exposed — earlier mention of "cv / ic / elbow" methods in this checklist was an overclaim, corrected in the methodology-promotion PR. Locked by `tests/test_methodology_trop.py::TestTROPDeviations::test_rank_selection_is_implicit_via_nuclear_norm`.
- [x] Returns the fitted factor matrix and an effective-rank diagnostic (`TROPResults.factor_matrix` and `TROPResults.effective_rank`). The library does NOT expose separate factor-loading / factor-score outputs — earlier prose claiming "factor loadings and scores" was an overclaim corrected in the 2026-05-24 methodology-promotion PR (TROP's nuclear-norm soft-thresholded L is delivered as a single (n_periods × n_units) matrix, not decomposed into loading / score components on Results).
- [x] ATT averages over all **estimable** D==1 cells (staggered adoption by default; on/off switching with `non_absorbing=True`). All D==1 cells are estimable on balanced / support-complete panels; cells whose `alpha_i + beta_t` is unidentified (target unit and period in different connected components of the observed-control graph) are NaN and excluded (see the non-estimable-cell `**Note:**`).
- [x] No post_periods parameter (D matrix determines treatment timing)
- [x] D matrix semantics documented (absorbing state, not event indicator)
- [x] Unbalanced panels supported — missing control / pre-treatment cells don't trigger false absorbing-state violations. Locked by `tests/test_methodology_trop.py::TestTROPDeviations::test_unbalanced_panels_supported` (10% random drops on control + pre-treatment subset). Three additional unbalanced-panel regressions live in `tests/test_trop.py::TestPR110FeedbackRound8` (`test_unbalanced_panel_d_matrix_validation`, `test_unbalanced_panel_real_violation_still_caught`, `test_unbalanced_panel_multiple_missing_periods`). Absorbing-state monotonicity validation (which fires on unbalanced cases too) is covered by `tests/test_trop.py::TestDMatrixValidation`.
- [x] Per-observation treatment-effect estimation (Eq. 13 / Algorithm 2) — `treatment_effects` dict contains one `τ_hat_it` entry per treated cell (finite for estimable cells; NaN for a missing outcome or, under `non_absorbing`, a cell with no weighted control support — see the no-support `**Note:**`), and the aggregate ATT equals the unweighted mean of the finite per-cell effects (Eq. 1). **The methodology test exercises block adoption with a constant treatment effect**; **absorbing-state staggered adoption** and **heterogeneous per-cell effects** (paper Remark 6.1) are SUPPORTED by the code path (the implementation does not gate on cohort or effect-magnitude pattern), but are not directly verified in the methodology test surface for those specific patterns. Cross-coverage of the staggered-cohort fit path is `tests/test_methodology_trop.py::TestTROPAlgorithm1LOOCV::test_control_set_includes_pretreat_of_eventually_treated` (two-cohort early-/late-treated panel under LOOCV-tuned `λ_unit`); absorbing-state structural validation is `tests/test_trop.py::TestDMatrixValidation`.
- [x] **Section 6.1 non-absorbing / on-off / switching assignment patterns are SUPPORTED via the opt-in `TROP(non_absorbing=True)` (`method='local'` only)** — matching the paper's general-assignment scope (§2.1 "units moving into and out of treatment"; Eq. 12 / Algorithm 2 mask treated cells per (i,t) with no monotonicity requirement). The default (`non_absorbing=False`) still rejects non-monotonic D as a defensive guard (see the `**Note:**` entries below). Removing this opt-in restriction *narrows* a prior implementation over-restriction (the shipped estimator was stricter than the paper); it is **not** a new methodology deviation. Recovery on a no-dynamic-effects toggling DGP, the per-cell effect count, and the caveat warning are locked by `tests/test_methodology_trop.py::TestTROPDeviations::test_non_absorbing_general_assignment_supported`; the default-mode rejection contract by `TestTROPDeviations::test_event_style_d_rejected_with_value_error`; opt-in acceptance, the local-only guard, params round-trip, and Rust/Python parity by `tests/test_trop.py::TestDMatrixValidation`.
- [x] Special-case reductions (paper Section 2.2): **DiD benchmark sanity check** (NOT a direct algebraic-equivalence proof) — TROP with `λ_nn=∞` + uniform weights produces an ATT within 0.5 of `DifferenceInDifferences` fitted as a basic 2×2 design on a TWFE-clean multi-period panel. This is empirical numerical agreement on a friendly DGP. A direct Section 2.2 reduction lock (true 2-period block-assignment panel where basic DiD is the algebraic target, or a comparison against `TwoWayFixedEffects` with explicit unit FE) is deferred. **Matrix Completion code path exercised** — TROP with uniform weights + finite `λ_nn` engages the nuclear-norm prox solver (effective_rank > 0) and beats the DiD-style baseline on a factor-confounded DGP; not an equivalence check against an independent MC reference. SC and SDID reductions are paper-claimed under "specific (omega, theta) weight choices" not provided in the paper text; cross-language anchor deferred until paper-author reference implementation clarifies the weight map. See `tests/test_methodology_trop.py::TestTROPSpecialCases`.
- **Note:** The balancing representation / decomposition (paper Eq. 10, Section 5.2) is a paper-side identity. Direct numerical reconstruction of the four-term sum requires the internal `θ_s^{i,t}` / `ω_j^{i,t}` weight vectors, which are not exposed on the public TROP API; numerical Eq. 10 verification is therefore out of scope. The test `tests/test_methodology_trop.py::TestTROPNuclearNormProx::test_factor_matrix_consistent_with_treatment_effects` is a structural pointer only — it checks `factor_matrix` shape + finiteness + that `treatment_effects` is populated with finite entries, but does NOT lock the magnitude of `L_hat`. (The test DGP uses additive unit + time effects only; on a no-interactive-FE panel, the paper's framework absorbs the additive surfaces into `α_i` / `β_t`, so a near-zero `L_hat` is methodologically correct. An `effective_rank > 0` assertion would lock a solver artifact, not the intended low-rank behavior.) This is NOT a full Eq. 10 lock. The Eq. 2 ingredients (soft-threshold SVD, **plain prox-gradient monotonicity** — NOT the shipped accelerated FISTA outer loop, which uses Nesterov momentum and does not guarantee per-step monotonicity, see `TestTROPNuclearNormProx` class docstring — weighted-prox) that the Eq. 10 derivation relies on are independently verified in the same class.
- **Note (library-side choice):** Weight normalization (Gap #5 in `docs/methodology/papers/athey-2025-review.md`): paper Section 5 (p. 20) states weights sum to one (`1ᵀω = 1ᵀθ = 1`), but Eq. 3 (p. 7) writes unnormalized exponential weights. **The paper-side ambiguity remains open**; the library resolves it as a documented deviation — the shipped implementation matches Eq. 2 (unnormalized). Verified by `tests/test_methodology_trop.py::TestTROPDeviations::test_unnormalized_weights_match_eq2`. Will be revisited once paper-author reference implementation lands.
- **Note (deferral):** Equation 14 covariate extension (`Y_it = α_i + β_t + X_it·β_coef + R_it` with R low-rank, paper Section 6.2) is **not implemented**. `TROP.fit()` does not accept a `covariates` keyword argument. The corresponding Theorem 8.1 covariate-triple-robustness result is correspondingly out of scope. The non-support is locked by `tests/test_methodology_trop.py::TestTROPDeviations::test_covariates_not_supported`, which uses `inspect.signature` to guard against future `**kwargs` silently breaking the contract. Deferred until use cases motivate the X threading through `trop_local.py` / `trop_global.py` / LOOCV / bootstrap.
- **Note:** Survey support: weights, strata, PSU, and FPC are all supported via Rao-Wu rescaled bootstrap with cross-classified pseudo-strata (Phase 6). Rust backend remains pweight-only; full-design surveys fall back to the Python bootstrap path. Survey weights enter ATT aggregation only — population-weighted average of per-observation treatment effects. Model fitting (kernel weights, LOOCV, nuclear norm regularization) stays unchanged. Rust and Python bootstrap paths both support survey-weighted ATT in each iteration.
- **Note (defensive default):** `non_absorbing` defaults to `False`, retaining the absorbing-state monotonicity gate. This is an implementation choice, not a paper requirement: the gate's primary value is catching the common mistake of encoding *absorbing* treatment as an event-style spike (a single D=1 period), which silently biases the ATT. Genuine on/off designs opt in with `non_absorbing=True`. The default-mode rejection message carries both the convert-to-absorbing guidance and the opt-in pointer.
- **Note (scope — local only):** `non_absorbing=True` is supported only for `method='local'`. The `global` method's post-hoc weighting and bootstrap bake in a contiguous, simultaneous treated block (it already rejects staggered adoption), so `TROP(method='global', non_absorbing=True)` raises a `ValueError`. The Rust local LOOCV/bootstrap paths are already mask-driven (`D==0`/`D==1`) and required no change; Rust/Python ATT parity on a non-absorbing panel is locked by `tests/test_trop.py::TestDMatrixValidation::test_non_absorbing_rust_python_parity`. For a fully toggling panel (no never-treated unit), the local Rust bootstrap is bypassed in favour of the Python loop (the Rust stratified resampler can return a degenerate ~0 SE on an empty control stratum).
- **Note (inference caveat for non-absorbing):** The paper's *point estimator* (Eq. 12 / Algorithm 2) supports general assignment, but the formal **triple-robustness guarantee (Theorem 5.1) is proven only under Assumption 1(i) block assignment** `W_it = 1{i>N0}·1{t>T0}`; the paper does not extend that guarantee to general/non-absorbing patterns (cf. `docs/methodology/papers/athey-2025-review.md`). The non-parametric bootstrap (Algorithm 3) is offered generally but "its validity requires a growing number of treated units." Non-absorbing validity additionally relies on the paper's **no-spillover / no-dynamic-effects (no carryover)** assumption (paper §2.1). `TROP.fit()` emits a one-time `UserWarning` carrying these caveats whenever `non_absorbing=True`; the warning is locked by `tests/test_methodology_trop.py::TestTROPDeviations::test_non_absorbing_general_assignment_supported` and its absence in default mode by `test_non_absorbing_no_caveat_in_default_mode`.
- **Note (non-absorbing non-estimable-cell trimming → estimable-cell ATT):** The working model fits unregularized unit/time fixed effects `alpha_j` / `beta_s` on the weighted observed control cells, then sets `tau_it = Y_it - alpha_i - beta_t - L_it`. A treated cell (i,t) is **estimable** only if the sum `alpha_i + beta_t` is identified by that two-way-FE fit. In a two-way FE model the effects are pinned only **within each connected component** of the bipartite graph whose nodes are units and periods and whose edges are the positively-weighted **observed** control cells (`usable = (D==0) & ~missing & isfinite(Y) & ω>0`); across components there is a free per-component offset. So estimability requires the **target unit node and target period node to lie in the same connected component** of that graph (predicate `diff_diff.trop_local._treated_cell_is_estimable`, a bipartite BFS run per treated cell with a cheap empty-row/empty-column fast-path). A marginal "the target unit has some usable control AND the target period has some usable control" test is **necessary but not sufficient** — e.g. usable cells at `(unitA,t0)` and `(unitB,t1)` with target `(unitA,t1)` pass it yet span two disconnected components, leaving `alpha_A + beta_1` unidentified. The connected-component check subsumes the simpler degeneracies: under `non_absorbing=True` (1) an **always-treated unit** has an empty control column (isolated unit node) — true even with `lambda_unit=0`; and (2) a **fully-treated period** has an empty control row (isolated period node). In all these cases tau would silently leak the fixed effect. Non-estimable cells are materialized as `NaN` in `treatment_effects` and **excluded from the ATT**, which is therefore the mean over **estimable** treated cells — NOT all D=1 cells. This matches the library-wide non-estimable→NaN convention (the per-named-cell analogue of CallawaySantAnna materializing non-estimable (g,t) as NaN); it is a **defensive choice for a degeneracy the paper does not cover** (the paper assumes enough overlap), not a deviation from Eq. 1 on the cells it covers. There is no `λ` that restores identification for these cells (the missing control row/column is structural), so the warning does not suggest one. **The predicate is applied to every local fit (absorbing and non-absorbing) as a general correctness guard** — it NaNs exactly the cells whose FE is genuinely unidentified. It is a **no-op whenever every treated cell's target unit and period have an OBSERVED control cell**: always true on a balanced panel, and in **absorbing** mode also true on unbalanced panels (a never-treated unit is a control at every observed period and each treated unit's pre-treatment controls are observed) — *unless* an unbalanced absorbing panel happens to leave a treated unit's pre-period controls or a period's controls entirely missing, in which case NaN-ing those cells is the correct fix to the identical latent FE leak (the prior behavior silently reported a contaminated tau). So estimable-cell trimming is the contract for **all** local TROP fits on unbalanced panels, not only non-absorbing ones. The point fit and the bootstrap refit apply the identical predicate; a draw with no estimable cell returns NaN and counts as a failed bootstrap replicate. **Rust/bootstrap parity:** the Rust per-cell bootstrap lacks the estimability guard, so whenever the point fit trims any cell (`force_python=True`, set from `n_no_support>0`) — or under `non_absorbing` generally — the bootstrap is routed to the guarded Python `_fit_with_fixed_lambda`, keeping the SE and the point ATT on the same estimable-cell set. (Rust remains the happy path for clean fits with no trimming.) **LOOCV is support-agnostic** by design: a degenerate pseudo-control cell yields a large raw-outcome pseudo-effect that inflates `Q(λ)`, so support-destroying `λ_unit` values are *naturally disfavored* (a soft penalty) rather than hard-rejected — hard-rejecting (`Q=∞`) would over-restrict. `TROP.fit()` emits a `UserWarning` naming the count of non-estimable cells. Locked by `tests/test_methodology_trop.py::TestTROPDeviations`: `test_non_absorbing_always_treated_unit_not_raw_outcome` (always-treated unit, `lambda_unit>0` and `lambda_unit=0`), `test_non_absorbing_fully_treated_period_not_estimable` (fully-treated period), `test_non_absorbing_disconnected_support_not_estimable` (disconnected bipartite control graph), and `test_unbalanced_absorbing_unidentified_unit_not_estimable` (the guard + `force_python` bootstrap parity in default absorbing mode).

### TROP Global Estimation Method

**Method**: `method="global"` in TROP estimator

**Approach**: Computationally efficient adaptation using the (1-W) masking
principle from Eq. 2. Fits a single global model on control data, then
extracts treatment effects as post-hoc residuals. For the paper's full
per-treated-cell estimator (Algorithm 2), use `method='local'`.

**Objective function** (Equation G1):
```
min_{μ, α, β, L}  Σ_{i,t} (1-W_{it}) × δ_{it} × (Y_{it} - μ - α_i - β_t - L_{it})² + λ_nn×||L||_*
```

where:
- (1-W_{it}) masks out treated observations — model is fit on control data only
- δ_{it} = δ_time(t) × δ_unit(i) are observation weights (product of time and unit weights)
- μ is the intercept
- α_i are unit fixed effects
- β_t are time fixed effects
- L_{it} is the low-rank factor component

**Post-hoc treatment effect extraction**:
```
τ̂_{it} = Y_{it} - μ̂ - α̂_i - β̂_t - L̂_{it}    for all (i,t) where W_{it} = 1
ATT = mean(τ̂_{it})  over all treated observations
```

Treatment effects are **heterogeneous** per-observation values. ATT is their mean.

**Weight computation** (differs from local):
- Time weights: δ_time(t) = exp(-λ_time × |t - center|) where center = T - treated_periods/2
- Unit weights: δ_unit(i) = exp(-λ_unit × RMSE(i, treated_avg))
  where RMSE is computed over pre-treatment periods comparing to average treated trajectory
- (1-W) masking applied after outer product: δ_{it} = 0 for all treated cells

**Implementation approach** (without CVXPY):

1. **Without low-rank (λ_nn = ∞)**: Standard weighted least squares
   - Build design matrix with unit/time dummies (no treatment indicator)
   - Solve via np.linalg.lstsq for (μ, α, β) using (1-W)-masked weights
   - Both the Python fallback and the Rust acceleration path use SVD-based
     minimum-norm least squares with numpy-compatible rcond = eps × max(n, k),
     so they return the canonical minimum-norm solution on rank-deficient Y
     (e.g., two near-parallel control units)

2. **With low-rank (finite λ_nn)**: Alternating minimization
   - Alternate between:
     - Fix L, solve weighted LS for (μ, α, β)
     - Fix (μ, α, β), proximal gradient for L:
       - Lipschitz constant of ∇f is L_f = 2·max(δ)
       - Step size η = 1/L_f = 1/(2·max(δ))
       - Proximal operator: soft_threshold(gradient_step, η·λ_nn)
       - Inner solver uses FISTA/Nesterov acceleration (O(1/k²))
   - Continue until max(|L_new - L_old|) < tol

3. **Post-hoc**: Extract τ̂_{it} = Y_{it} - μ̂ - α̂_i - β̂_t - L̂_{it} for treated cells

**LOOCV parameter selection** (unified with local, Equation 5):
Following paper's Equation 5 and footnote 2:
```
Q(λ) = Σ_{j,s: D_js=0} [τ̂_js^loocv(λ)]²
```
where τ̂_js^loocv is the pseudo-treatment effect at control observation (j,s)
with that observation excluded from fitting.

For global method, LOOCV works as follows:
1. For each control observation (t, i):
   - Zero out weight δ_{ti} = 0 (exclude from weighted objective)
   - Fit global model on remaining data → obtain (μ̂, α̂, β̂, L̂)
   - Compute pseudo-treatment: τ̂_{ti} = Y_{ti} - μ̂ - α̂_i - β̂_t - L̂_{ti}
2. Score = Σ τ̂_{ti}² (sum of squared pseudo-treatment effects)
3. Select λ combination that minimizes Q(λ)

**Rust acceleration**: The LOOCV grid search is parallelized in Rust for 5-10x speedup.
- `loocv_grid_search_global()` - Parallel LOOCV across all λ combinations
- `bootstrap_trop_variance_global()` - Parallel bootstrap variance estimation

**Key differences from local method**:
- Global weights (distance to treated block center) vs. per-observation weights
- Single model fit per λ combination vs. N_treated fits
- Treatment effects are post-hoc residuals from a single global model (global)
  vs. post-hoc residuals from per-observation models (local)
- Both use (1-W) masking (control-only fitting)
- Faster computation for large panels

**Assumptions**:
- **Simultaneous adoption (enforced)**: The global method requires all treated units
  to receive treatment at the same time. A `ValueError` is raised if staggered
  adoption is detected (units first treated at different periods). Treatment timing is
  inferred once and held constant for bootstrap variance estimation.
  For staggered adoption designs, use `method="local"`.

**Reference**: Adapted from reference implementation. See also Athey et al. (2025).

**Edge Cases (treated NaN outcomes):**
- **Partial NaN**: When some treated outcomes Y_{it} are NaN/missing:
  - `_extract_posthoc_tau()` (global) skips these cells; only finite τ̂ values are averaged
  - Local loop skips NaN outcomes entirely (no model fit, no tau appended)
  - `n_treated_obs` in results reflects valid (finite) count, not total D==1 count
  - `df_trop = max(1, n_valid_treated - 1)` uses valid count
  - Warning issued when n_valid_treated < total treated count
- **All NaN**: When all treated outcomes are NaN:
  - ATT = NaN, warning issued
  - `n_treated_obs = 0`
- **Bootstrap SE with <2 draws**: Returns `se=NaN` (not 0.0) when zero bootstrap
  iterations succeed. `safe_inference()` propagates NaN downstream.

**Requirements checklist:**
- [x] Same LOOCV framework as local (Equation 5)
- [x] Global weight computation using treated block center
- [x] (1-W) masking for control-only fitting (per paper Eq. 2)
- [x] Alternating minimization for nuclear norm penalty
- [x] Returns ATT = mean of per-observation post-hoc τ̂_{it}
- [x] Rust acceleration for LOOCV and bootstrap

---

## HeterogeneousAdoptionDiD

**Implementation status (2026-04-18):** Methodology plan approved; implementation queued across 7 phased PRs (Phase 1a kernels + local-linear + HC2/Bell-McCaffrey; Phase 1b MSE-optimal bandwidth; Phase 1c bias-corrected CI + `nprobust` parity; Phase 2 `HeterogeneousAdoptionDiD` class + multi-period event study; Phase 3 QUG/Stute/Yatchew-HR diagnostics; Phase 4 Pierce-Schott replication harness; Phase 5 docs + tutorial + `practitioner_next_steps` integration). Full plan at `~/.claude/plans/vectorized-beaming-feather.md`; full paper review at `docs/methodology/papers/dechaisemartin-2026-review.md`. The requirements checklist at the end of this section tracks phase completion.

**Primary source:** de Chaisemartin, C., Ciccia, D., D'Haultfœuille, X., & Knau, F. (2026). Difference-in-Differences Estimators When No Unit Remains Untreated. arXiv:2405.04465v6.

**Scope:** Heterogeneous Adoption Design (HAD): a single-date, two-period DiD setting in which no unit is treated at period one and at period two all units receive strictly positive, heterogeneous treatment doses `D_{g,2} >= 0`. The estimator targets a Weighted Average Slope (WAS) when no genuinely untreated group exists. Extensions cover multiple periods without variation in treatment timing (Appendix B.2) and covariate-adjusted identification (Appendix B.1, future work).

**Key implementation requirements:**

*Assumption checks / warnings:*
- Data must be panel (or repeated cross-section) with `D_{g,1} = 0` for all `g` (nobody treated in period one).
- Treatment dose `D_{g,2} >= 0`. For Design 1' (the QUG case) the support infimum `d̲ := inf Supp(D_{g,2})` must equal 0; for Design 1 (no QUG) `d̲ > 0` and Assumption 5 or 6 must be invoked.
- Assumption 1 (i.i.d. sample): `(Y_{g,1}, Y_{g,2}, D_{g,1}, D_{g,2})_{g=1,...,G}` i.i.d.
- Assumption 2 (parallel trends for the least-treated): `lim_{d ↓ d̲} E[ΔY(0) | D_2 ≤ d] = E[ΔY(0)]`. Testable with pre-trends when a pre-treatment period `t=0` exists. Reduces to standard parallel trends when treatment is binary.
- Assumption 3 (uniform continuity of `d → Y_2(d)` at zero): excludes extensive-margin effects; holds if `d → Y_2(d)` is Lipschitz. Not testable.
- Assumption 4 (regularity for nonparametric estimation): positive density at boundary (`lim_{d ↓ 0} f_{D_2}(d) > 0`), twice-differentiable `m(d) := E[ΔY | D_2 = d]` near 0, continuous `σ²(d) := V(ΔY | D_2 = d)` with `lim_{d ↓ 0} σ²(d) > 0`, bounded kernel, bandwidth `h_G → 0` with `G h_G → ∞`.
- Assumption 5 (for Design 1 sign identification): `lim_{d ↓ d̲} E(TE_2 | D_2 ≤ d) / WAS < E(D_2) / d̲`. Not testable via pre-trends. Sufficient version Equation 9: `0 ≤ E(TE_2 | D_2 = d) / E(TE_2 | D_2 = d') < E(D_2) / d̲` for all `(d, d')` in `Supp(D_2)²`.
- Assumption 6 (for Design 1 WAS_{d̲} identification): `lim_{d ↓ d̲} E[Y_2(d̲) - Y_2(0) | D_2 ≤ d] = E[Y_2(d̲) - Y_2(0)]`. Not testable.
- Warn (do NOT fit silently) when staggered treatment timing is detected: the paper's Appendix B.2 excludes designs with variation in treatment timing and no untreated group (only the last treatment cohort's effects are identified in a staggered setting).
- Warn when Assumption 5/6 is invoked that these are not testable via pre-trends.
- With Design 1 (no QUG) WAS is NOT point-identified under Assumptions 1-3 alone (Proposition 1); only sign identification (Theorem 2) or the alternative target WAS_{d̲} (Theorem 3) is available.

*Target parameter - Weighted Average Slope (WAS, Equation 2):*

    WAS := E[(D_2 / E[D_2]) · TE_2]
         = E[Y_2(D_2) - Y_2(0)] / E[D_2]

where `TE_2 := (Y_2(D_2) - Y_2(0)) / D_2` is the per-unit slope relative to "no treatment". Authors prefer WAS over the unweighted Average Slope `AS := E[TE_2]` because AS suffers a small-denominator problem near `D_2 = 0` that prevents `√G`-rate estimation.

Alternative target (Design 1 under Assumption 6):

    WAS_{d̲} := E[(D_2 - d̲) / E[D_2 - d̲] · TE_{2,d̲}]

where `TE_{2,d̲} := (Y_2(D_2) - Y_2(d̲)) / (D_2 - d̲)`. Compares to a counterfactual where every unit gets the lowest dose, not zero; authors describe it as "less policy-relevant" than WAS.

*Estimator equations:*

Design 1' identification (Theorem 1, Equation 3):

    WAS = (E[ΔY] - lim_{d ↓ 0} E[ΔY | D_2 ≤ d]) / E[D_2]

Nonparametric local-linear estimator (Equation 7):

    β̂_{h*_G}^{np} := ((1/G) Σ_{g=1}^G ΔY_g  -  μ̂_{h*_G}) / ((1/G) Σ_{g=1}^G D_{g,2})

where `μ̂_h` is the intercept from a local-linear regression of `ΔY_g` on `D_{g,2}` using weights `k(D_{g,2}/h)/h`. This estimates the conditional mean `m(0) = lim_{d ↓ 0} E[ΔY | D_2 ≤ d]`.

Design 1 mass-point case (Section 3.2.4, discrete bunching at `d̲`):

    target = (E[ΔY] - E[ΔY | D_2 = d̲]) / E[D_2 - d̲]
           = (E[ΔY | D_2 > d̲] - E[ΔY | D_2 = d̲]) / (E[D_2 | D_2 > d̲] - E[D_2 | D_2 = d̲])

Compute via sample averages or a 2SLS of `ΔY` on `D_2` with instrument `1{D_2 > d̲}`. Convergence rate is `√G`.

Design 1 continuous-near-`d̲` case: use the same kernel construction as Equation 7 with 0 replaced by `d̲` and `D_2` replaced by `D_2 - d̲`. `d̲` is estimated by `min_g D_{g,2}`, which converges at rate `G` (asymptotically negligible versus the `G^{2/5}` nonparametric rate of `β̂_{h*_G}^{np}`).

Sign identification for Design 1 (Theorem 2, Equation 10):

    WAS ≥ 0  ⟺  (E[ΔY] - lim_{d ↓ d̲} E[ΔY | D_2 ≤ d]) / E[D_2 - d̲] ≥ 0

WAS_{d̲} identification (Theorem 3, Equation 11):

    WAS_{d̲} = (E[ΔY] - lim_{d ↓ d̲} E[ΔY | D_2 ≤ d]) / E[D_2 - d̲]

*With covariates / conditional identification (Equation 19, Appendix B.1):*

Assumption 9 (conditional parallel trends): almost surely, `lim_{d ↓ 0} E[ΔY(0) | D_2 ≤ d, X] = E[ΔY(0) | X]`.

Theorem 6 (Design 1' + Assumptions 3 and 9):

    WAS = (E[ΔY] - E[ lim_{d ↓ 0} E[ΔY | D_2 ≤ d, X] ]) / E[D_2]

Implementing Equation 19 requires MULTIVARIATE nonparametric regression `E[ΔY | D_2, X]`; Calonico et al. (2018) covers only the univariate case, so the authors leave this extension to future work. The Phase-2 estimator will raise `NotImplementedError` when `covariates=` is passed, pointing to this section.

TWFE-with-covariates (Appendix B.1, Equations 20-21): under linearity Assumption 10 (`E[ΔY(0) | D_2, X] = X' γ_0`) and homogeneity `E[TE_2 | D_2, X] = X' δ_0`,

    E[ΔY | D_2, X] = X' γ_0 + D_2 X' δ_0    (21)

so `δ_0` is recovered by OLS of `ΔY` on `X` and `D_2 * X`; Average Slope is `((1/n) Σ X_i)' δ̂^X`.

*Standard errors (Section 3.1.3-3.1.4, 4):*

- Nonparametric estimator (Design 1' and Design 1 continuous-near-`d̲`): bias-corrected Calonico-Cattaneo-Farrell (2018, 2019) 95% CI (Equation 8):

      [ β̂_{ĥ*_G}^{np} + M̂_{ĥ*_G} / ((1/G) Σ D_{g,2})  ±  q_{1-α/2} sqrt(V̂_{ĥ*_G} / (G ĥ*_G)) / ((1/G) Σ D_{g,2}) ]

  The procedure ports the Calonico et al. `nprobust` machinery in-house (Phase 1a/1b/1c of the implementation plan): estimate optimal bandwidth `ĥ*_G`, compute `μ̂_{ĥ*_G}`, the first-order bias estimator `M̂_{ĥ*_G}`, and the variance estimator `V̂_{ĥ*_G}`.

*Weighted extension (Phase 4.5 survey support):* `HeterogeneousAdoptionDiD.fit()` accepts `survey_design=SurveyDesign(...)` (design-based inference) on the two continuous-dose paths (`continuous_at_zero`, `continuous_near_d_lower`). Weights thread through `_nprobust_port.lprobust` via pointwise multiplication into the kernel weights: `W_combined = k((D − d̲)/h) · w_g`. Design matrices, Q.q bias-correction matrix, and variance matrices inherit the combined weights automatically. The β-scale rescaling uses weighted population moments: β̂_weighted = (Σ w_g ΔY_g / Σ w_g − μ̂_weighted(h)) / (Σ w_g (D_g − d̲) / Σ w_g).

Under `survey_design=SurveyDesign(weights, strata, psu, fpc)`, the variance composes via Binder (1983) Taylor-series linearization — the per-unit influence function of μ̂ is aggregated by PSU within strata with FPC correction, using the shared `compute_survey_if_variance` helper (`diff_diff/survey.py:1802`) consumed by dCDH / EfficientDiD / CallawaySantAnna. Survey design columns (strata / PSU / FPC) are required to be constant within unit (sampling-unit assignment convention); within-unit variance raises `ValueError` front-door per `feedback_no_silent_failures`.

- **Note (parity gap):** no public weighted-CCF bias-corrected local-linear reference exists in any language. `nprobust::lprobust` has no weight argument; Calonico-Cattaneo-Farrell's companion `rdrobust` is RD-shaped (not HAD-shaped); `np::npreg`'s local-linear algorithm does not reduce to a straightforward weighted-OLS at the intercept. The Phase 1c `atol=1e-12` R bit-parity bar is therefore NOT reachable on the weighted bias-corrected CI. Methodology confidence under informative weights comes from the stack documented below.
- **Note:** Uniform-weights bit-parity — `lprobust(..., weights=np.ones(N))` ≡ unweighted at `atol=1e-14, rtol=1e-14` across the full output struct (`tau_cl`, `tau_bc`, `se_cl`, `se_rb`, `V_Y_cl`, `V_Y_bc`). Regression tests in `tests/test_nprobust_port.py::TestWeightedLprobust` and `tests/test_bias_corrected_lprobust.py::TestWeightedBiasCorrectedLocalLinear`.
- **Note:** Cross-language weighted-OLS parity — `benchmarks/R/generate_np_npreg_weighted_golden.R` produces a manually-implemented-R weighted-OLS reference against which Python recovers the intercept + slope at `atol=1e-12` on 4 DGPs (`tests/test_np_npreg_weighted_parity.py`). This is a regression lock on the kernel + weighted-OLS formula, not third-party validation of the CCF bias correction.
- **Note:** Monte Carlo oracle consistency — `tests/test_had_mc.py` validates that the weighted estimator recovers the oracle τ under informative sampling, with coverage near nominal and visible bias reduction vs unweighted. Slow-gated; 4 tests.
- **Note:** Auto-bandwidth selection (Phase 1b MSE-DPI via `lpbwselect_mse_dpi`) remains UNWEIGHTED in this phase; users who want a weight-aware bandwidth should pass `h`/`b` explicitly. The auto path with uniform weights reduces to the existing unweighted bandwidth selector, so the uniform-weights bit-parity chain is preserved.
- **Note:** Replicate-weight SurveyDesigns (BRR / Fay / JK1 / JKn / SDR) on the HAD continuous path raise `NotImplementedError` in this PR; Rao-Wu-style rescaled bootstrap is deferred to Phase 4.5 C (survey-under-pretests).
- **Note:** `HeterogeneousAdoptionDiD.fit()` dispatch matrix after Phase 4.5 B + 4.5 C — survey-weighted inference (`survey_design=`) is supported on ALL design × aggregate combinations (continuous × {overall, event-study}, mass-point × {overall, event-study}). The HAD pretests (`qug_test`, `stute_test`, `yatchew_hr_test`, joint Stute variants, `did_had_pretest_workflow`) ship survey support in Phase 4.5 C (PR #370) and the Phase 4.5 C strata extension (this PR) — `qug_test` permanently rejects (Phase 4.5 C0 deferral; see "QUG Null Test" §); the linearity family supports pweight + PSU + FPC + strata via PSU-level Mammen multipliers with within-stratum demean + Bessel rescale (Stute, see "Note (Stute stratified survey-bootstrap calibration)" below) + closed-form weighted variance components (Yatchew). Replicate-weight designs raise `NotImplementedError` (parallel follow-up); `lonely_psu='adjust'` + singleton-strata raises `NotImplementedError` on the Stute family (same pseudo-stratum centering gap as the HAD sup-t deviation, parallel follow-up). The canonical kwarg on all 8 HAD surfaces is `survey_design=` (see "Note (HAD survey-design API consolidation)" below); the deprecated `survey=` / `weights=` aliases were removed in 3.7.0 on `HeterogeneousAdoptionDiD.fit` and in 3.7.x on the 7 pretest helpers (passing them raises `TypeError`).
- **Note (HAD survey-design API consolidation):** All 8 HAD surfaces — `HeterogeneousAdoptionDiD.fit`, `did_had_pretest_workflow`, `qug_test`, `stute_test`, `yatchew_hr_test`, `stute_joint_pretest`, `joint_pretrends_test`, `joint_homogeneity_test` — accept the canonical kwarg `survey_design=` (matching `ContinuousDiD`, `EfficientDiD`, `ChaisemartinDHaultfoeuille`). On `HeterogeneousAdoptionDiD.fit`, the deprecated `survey=` / `weights=` kwargs were **removed in 3.7.0** — `survey_design=` is the sole weighting entry (the pweight/CCT-2014 dispatch branches are gone, so `variance_formula` is now one of `None` / `"survey_binder_tsl"` / `"survey_binder_tsl_2sls"`), and `cband` is now keyword-only. On the 7 pretest helpers (`did_had_pretest_workflow`, `qug_test`, `stute_test`, `yatchew_hr_test`, `stute_joint_pretest`, `joint_pretrends_test`, `joint_homogeneity_test`) the deprecated `survey=` / `weights=` aliases were removed in 3.7.x — as on `fit`, passing either raises `TypeError`, and `survey_design=` is the sole weighting entry (the 3-way alias mutex and its per-surface-group error messages are gone with the aliases). Internal back-end behavior for the surviving `survey_design=` / unweighted paths is UNCHANGED (byte-identical output). Migration by surface group: data-in surfaces (workflow + joint data-in wrappers) take `survey_design=SurveyDesign(weights='col_name', ...)` — the former row-level `weights=` array shortcut is gone, add the weights as a column instead (per-unit aggregation + mean-1 normalization make the two forms numerically identical); the three array-in linearity helpers (`stute_test` / `yatchew_hr_test` / `stute_joint_pretest`) take `survey_design=make_pweight_design(arr)` (for pweight-only) or `survey_design=<pre-resolved ResolvedSurveyDesign>` (for full PSU/strata/FPC). The 8th surface — `qug_test` — permanently rejects `survey_design=` (Phase 4.5 C0 deferral, see "QUG Null Test" §). Array-in helpers reject `survey_design=SurveyDesign(...)` with `TypeError` since they have no `data` to resolve column names against. The `make_pweight_design(weights: np.ndarray) -> ResolvedSurveyDesign` factory is exported from the `diff_diff` top level (formerly `survey._make_trivial_resolved`, kept as a permanent private alias for back-compat); `weights` must be 1-D (scalar / 0-D / column-vector inputs raise `ValueError` at the front door).

*Weighted 2SLS (Phase 4.5 B):* `_fit_mass_point_2sls(..., weights=, return_influence=)` extends the Wald-IV / 2SLS sandwich with pweight semantics:
- **Weighted bread**: `Z'WX = Z'·diag(w)·X` (`w¹`, matches `estimatr::iv_robust(..., weights=)` weighted-bread convention).
- **HC1 pweight meat**: `Ω_HC1 = (n/(n-k)) · Z'·diag(w² u²)·Z` (`w²` squared, Wooldridge 2010 Eq. 12.37; matches `linalg.py:1141` pweight convention). Bit-exact with `estimatr::iv_robust(..., weights=, se_type="HC1")` at `atol=1e-10`.
- **CR1 pweight-cluster meat**: for each cluster c, `s_c = Z'_c·(w·u)_c`; `Ω_CR1 = (G/(G-1))·((n-1)/(n-k))·Σ_c s_c s_c'` (`w¹` inside cluster score). Bit-exact with `estimatr::iv_robust(..., weights=, clusters=, se_type="stata")` at `atol=1e-10`.
- **Classical**: sandwich form `Ω_cl = σ²·Z'·diag(w²)·Z` with `σ² = Σw²u²/(Σw-k)`. Deviates from `estimatr` classical (projection-form + `n-k` DOF) by `O(1/n)` at non-uniform weights; unweighted path is bit-exact by equivalence. Skipped in cross-language parity tests.
- **Per-unit IF on β̂-scale** (for Binder-TSL survey composition): `psi_g = [(Z'WX)^{-1} · z_g · w_g · u_g][1] · sqrt((n-1)/(n-k))`. The scaling factor absorbs DOF / small-sample differences so `compute_survey_if_variance(psi, trivial_resolved) ≈ V_HC1[1,1]` at `atol=1e-10` (mirrors PR #359 convention; asserted by `TestIFScaleInvariant` and bit-exact against estimatr HC1 on 4 DGPs). Fixture: `benchmarks/R/generate_estimatr_iv_robust_golden.R` → `benchmarks/data/estimatr_iv_robust_golden.json`.
- **Note (deviation from R — classical intercept SE):** `return_intercept_se=True`: the 2×2 sandwich `V` already carries the intercept variance `V[0,0]`; the opt-in hook surfaces `sqrt(V[0,0])` (default off — the production 3-tuple return is byte-unchanged, so no public-API change). HC1 and CR1 intercept SEs are bit-exact with `estimatr` `se_intercept` at `atol=1e-10` (`tests/test_estimatr_iv_robust_parity.py`); the **classical** intercept carries the same `O(1/n)` projection/DOF deviation as the slope (bullet above) and is likewise excluded from the parity lock.

*Event-study survey composition (Phase 4.5 B):* The per-horizon loop in `_fit_event_study` threads `weights_unit_full` + `resolved_survey_unit_full` through to both `_fit_continuous` and `_fit_mass_point_2sls` (the latter with `return_influence=True` under weighted fits). The returned IF matrix `Psi ∈ R^{G × H}` has a shared construction contract across paths — each column on the β̂-scale, such that `compute_survey_if_variance(Psi[:, e], resolved) ≈ V_β[e]`. Per-horizon analytical variance uses Binder-TSL via `compute_survey_if_variance` (the `survey_design=` path). `survey_metadata`, `variance_formula` (`"survey_binder_tsl"` / `"survey_binder_tsl_2sls"`), and `effective_dose_mean` populate identically to the static path. Pre-PR numerical output is preserved bit-exactly on the unweighted path when `cband=False` (stability invariant; Phase 2b convention unchanged for unweighted fits).

*Sup-t multiplier bootstrap (Phase 4.5 B):* Simultaneous confidence band on the weighted/survey OR clustered event-study path via `_sup_t_multiplier_bootstrap` (the clustered branch is documented in "Note (HAD clustered event-study sup-t band)"):

1. **Multiplier draws**: reuse `diff_diff.bootstrap_utils.generate_survey_multiplier_weights_batch` (survey_design= path: PSU-level draws with stratum centering, FPC scaling, lonely-PSU handling) or `generate_bootstrap_weights_batch` (clustered band: cluster-level Rademacher). Default `n_bootstrap=999` (CS parity); `seed` exposed on `HeterogeneousAdoptionDiD.__init__` for reproducibility.
2. **Perturbations**: `delta = xi @ Psi` — shape `(B, H)` matrix-matrix product, NO `(1/n)` prefactor (matches `staggered_bootstrap.py:373` idiom; `Psi` is already on the β̂-scale).
3. **t-statistics**: `t[b, e] = delta[b, e] / se[e]` where `se[e]` is the per-horizon analytical Binder-TSL / HC1 SE from the loop above.
4. **Sup-t distribution**: `sup_t[b] = max_e |t[b, e]|` with finite-mask filtering of degenerate horizons.
5. **Critical value**: `q = quantile(sup_t[finite], 1 - alpha)`. Simultaneous band: `cband_low[e] = att[e] - q · se[e]`.

**Reduction invariant**: at `H=1`, the sup collapses to the marginal and `q → Φ⁻¹(1 - alpha/2) ≈ 1.96` at `alpha=0.05` up to MC noise. Locked by `TestSupTReducesToNormalAtH1` (G=500, B=5000, seed=42, `atol=0.15` on the quantile), its clustered variant `test_clustered_sup_t_h1_reduces_to_normal` (both the continuous scale-1.0 and mass-point `√(G/(G-1))` scalars), and `TestEventStudySurveyCband::test_trivial_survey_h1_sup_t_matches_analytical` / `test_stratified_h1_sup_t_matches_analytical` for the trivial-survey and stratified cases respectively.

**Scope**: sup-t bootstrap runs in the event-study mode (multi-period panels; the deprecated `aggregate="event_study"` override maps to it, row M-027) when `cband=True` (default) AND either (a) `survey_design=` is supplied (the survey band) OR (b) `cluster=` is supplied (the clustered band — see "Note (HAD clustered event-study sup-t band)"). An unweighted, unclustered event-study skips the bootstrap entirely — pre-Phase 4.5 B numerical output bit-exactly preserved. Setting `cband=False` disables the bootstrap on any path.

- **Deviation from shared survey-bootstrap contract:** `_sup_t_multiplier_bootstrap` raises `NotImplementedError` on `SurveyDesign(lonely_psu="adjust")` with singleton strata. The shared `generate_survey_multiplier_weights_batch` helper pools singleton PSUs into a pseudo-stratum with NONZERO multipliers, but `compute_survey_if_variance` centers singleton PSU scores at the GLOBAL mean of PSU scores (rather than the pseudo-stratum mean). Matching the two would require a pooled-singleton pseudo-stratum centering transform in the HAD sup-t path that has not been derived. The HAD-specific limitation is scoped to: weighted event-study + `cband=True` + `lonely_psu="adjust"` + at least one singleton stratum. Practitioners can use `lonely_psu="remove"` or `"certainty"` (matches the analytical target bit-exactly on the HAD sup-t path), or pass `cband=False` to skip the simultaneous band. All other survey-bootstrap consumers (CallawaySantAnna, dCDH, SDID) retain full `lonely_psu="adjust"` support through the shared helper.

- **Deviation: weighted mass-point `vcov_type="classical"` on survey/sup-t paths:** `vcov_type="classical"` raises `NotImplementedError` whenever the mass-point IF matrix is consumed downstream — specifically on `design="mass_point"` + `survey_design=` (static and event-study, regardless of `cband`, since the Binder-TSL analytical SE consumes the HC1-scaled IF either way) — **only when `cluster=` is NOT set** (with `cluster=` the mass-point path computes the CR1 sandwich regardless of `vcov_type`, so no classical-vs-HC1 mismatch exists and the classical-rejection is guarded by `cluster_arg is None`). The per-unit 2SLS IF returned by `_fit_mass_point_2sls` is scaled (`sqrt((n-1)/(n-k))`) to match V_HC1 via `compute_survey_if_variance`; mixing it with a classical analytical SE would silently return a V_HC1-targeted variance under a classical label. A classical-aligned IF derivation is queued for a follow-up PR. The allowed weighted-mass-point combinations are: `vcov_type="hc1"` on every `survey_design=` path; and any `cluster=` composition (resolves to CR1).

- **Deviation: `cluster=` + `survey_design=` rejected (both designs):** the `survey_design=` path composes Binder-TSL variance via `compute_survey_if_variance`, which would silently overwrite the cluster-robust sandwich while result metadata still reports `vcov_type="cr1"`. Rejected up-front on `cluster=` + `survey_design=` for BOTH designs (`continuous_*` and `mass_point`), static and event-study — for weighted clustering route through `survey_design=SurveyDesign(weights='<weight_col>', psu='<cluster_col>')` instead. All other `cluster=` compositions WORK end-to-end (pointwise CIs AND the simultaneous band): a bare unweighted `cluster=` with `cband` either `False` (pointwise cluster-robust only) or `True` (adds the clustered sup-t band — see "Note (HAD clustered event-study sup-t band)" below).

- **Note (HAD clustered event-study sup-t band):** when a bare `cluster=` is set, `_fit_event_study` produces cluster-robust per-horizon pointwise CIs AND a cluster-robust simultaneous band on BOTH designs (continuous CCT and mass-point 2SLS) on an unweighted fit (a `cluster=` + `survey_design=` composition is rejected — see the deviation above; weighted clustering routes through `survey_design=SurveyDesign(weights=..., psu=...)`, which takes the survey branch, not this clustered branch). Pointwise: `cluster_arr` threads into `bias_corrected_local_linear` (continuous, static-path parity) or `_fit_mass_point_2sls` (mass-point CR1). Band: `_sup_t_multiplier_bootstrap` takes a dedicated **clustered branch** — it aggregates the per-unit β̂-scale influence function to cluster level (`s_c = Σ_{i∈c} ψ_i`) and draws cluster-level iid Rademacher multipliers, so the perturbation variance is the RAW cluster sandwich `Σ_c s_c²` (no stratum-centering / FPC / Bessel — distinct from the survey branch). This matches each path's analytical cluster-robust SE via a path scalar: **1.0** for continuous (the `lprobust_vce` cluster meat carries no `g/(g-1)` correction, so `Σ_c s_c² == se_rb²` exactly) and **`√(G/(G-1))`** for mass-point (the returned IF carries `√((n-1)/(n-k))` but not the CR1 `G/(G-1)` factor; `G` is the full-array cluster count, identical to the bootstrap branch's own `n_clusters`, so a wholly-zero-weight cluster contributes `s_c=0` to both the analytical Ω and the bootstrap yet is counted in `G` by both). The variance-family reconciliation is validated bootstrap-free: `sqrt(Σ_c (scale·s_c)²) == se` to `atol=1e-10` on the real IF for both paths (`TestEventStudyClusterBand::test_{continuous,masspoint}_if_reconciliation_deterministic`), plus the `H=1 → 1.96` reduction on the clustered branch (`TestSupTReducesToNormalAtH1::test_clustered_sup_t_h1_reduces_to_normal`). Single cluster (`G<2`) → NaN band + `RuntimeWarning` (CR undefined). `cband_method="cluster_multiplier_bootstrap"` on this path. No R anchor (no reference package computes an HAD clustered sup-t band); the reconciliation identity is the validation.
- 2SLS (Design 1 mass-point case): standard 2SLS inference (details not elaborated in the paper).
- TWFE with small `G`: HC2 standard errors with Bell-McCaffrey (2002) degrees-of-freedom correction, following Imbens and Kolesar (2016). Used in the Pierce and Schott (2016) application with `G=103`. Added library-wide to `diff_diff/linalg.py` as a new `vcov_type` dispatch (Phase 1a), exposed on `DifferenceInDifferences` and `TwoWayFixedEffects`.
- Bootstrap: wild bootstrap with Mammen (1993) two-point weights is used for the Stute test (see Diagnostics below), NOT for the main WAS estimator. Reuses the existing `diff_diff.bootstrap_utils.generate_bootstrap_weights(..., weight_type="mammen")` helper.
- Clustering: no explicit clustering formulas in the paper's core equations.

*Convergence rates:*
- Design 1' nonparametric estimator: `G^{2/5}` (univariate nonparametric rate; Equations 5-6).
- Design 1 discrete-mass-point case: `√G` (parametric rate).
- Estimate of `d̲` via `min_g D_{g,2}`: rate `G` (asymptotically negligible).

*Asymptotic distributions (Equations 5-6):*
- Equation 5: `√(G h_G) (β̂_{h_G}^{np} - WAS - h_G² · C m''(0) / (2 E[D_2])) →^d N(0, σ²(0) ∫_0^∞ k*(u)² du / (E[D_2]² f_{D_2}(0)))`
- Equation 6 (optimal rate, `G^{1/5} h_G → c > 0`): `G^{2/5} (β̂_{h_G}^{np} - WAS) →^d N(c² C m''(0) / (2 E[D_2]), σ²(0) ∫_0^∞ k*(u)² du / (c E[D_2]² f_{D_2}(0)))`
- Kernel constants: `κ_k := ∫_0^∞ t^k k(t) dt`, `k*(t) := (κ_2 - κ_1 t) / (κ_0 κ_2 - κ_1²) · k(t)`, `C := (κ_2² - κ_1 κ_3) / (κ_0 κ_2 - κ_1²)`.

*Edge cases:*
- **No genuinely untreated units, D_2 continuous with `d̲ = 0` (Design 1')**: use `β̂_{h*_G}^{np}` (Equation 7) with bias-corrected CI (Equation 8).
- **No untreated units, `d̲ > 0`, `D_2` has mass point at `d̲`**: use 2SLS of `ΔY` on `D_2` with instrument `1{D_2 > d̲}`, or equivalent sample-average formula. Identifies WAS_{d̲} under Assumption 6 (Theorem 3) or the sign of WAS under Assumption 5 (Theorem 2).
- **No untreated units, `d̲ > 0`, `D_2` continuous near `d̲`**: replace 0 by `d̲` and `D_2` by `D_2 - d̲` in Equation 7; estimate `d̲` by `min_g D_{g,2}`.
- **Genuinely untreated units present but a small share**: Authors do NOT require untreated units to be dropped. In the Garrett et al. (2020) bonus-depreciation application with 12 untreated counties out of 2,954, they keep the untreated subsample. Simulations (DGP 2, DGP 3) suggest CIs retain close-to-nominal coverage even when `f_{D_2}(0) = 0`.
- **WAS is not point-identified without a QUG (Proposition 1, proof C.1)**: the proof explicitly constructs `tilde-Y_2(d) := Y_2(d) + (c / d̲) · E[D_2] · (d - d̲)` for any `c ∈ R`, compatible with the data under Assumptions 2 and 3 but with `tilde-WAS = WAS + c`. Practical consequence: do NOT report a point estimate of WAS under Design 1 without Assumption 5 or 6; fall back to Theorem 2 (sign) or Theorem 3 (WAS_{d̲}).
- **Extensive-margin effects**: ruled out by Assumption 3. If a jump `Y_2(0) ≠ Y_2(0+)` is suspected, the target parameter and estimator are not appropriate.
- **Partial identification of WAS_{d̲}**: only identified up to a positive constant offset `≤ ε` by the bound in Equation 22 (Jensen inequality argument in Appendix C.3).
- **Density at boundary**: Assumption 4 requires `f_{D_2}(0) > 0`. This is a non-trivial assumption since 0 is on the boundary of `Supp(D_2)`.
- **Variation in treatment timing**: Appendix B.2 - "in designs with variation in treatment timing, there must be an untreated group, at least till the period where the last cohort gets treated." In the event-study mode (Phase 2b) the implementation auto-filters to the last-treatment cohort plus never-treated units with a `UserWarning` when `first_treat` is supplied (see Phase 2b last-cohort filter note below); when `first_treat` is omitted the estimator detects multiple first-positive-dose cohorts from the dose path and raises a front-door `ValueError` directing users to pass `first_treat` or use `ChaisemartinDHaultfoeuille`.
- **Mechanical zero at reference period under linear trends (Footnote 13, main text p. 31)**: with industry/unit-specific linear trends, the pre-trends estimator is mechanically zero in the second-to-last pre-period (the slope anchor year). Practical consequence: that year is not an informative placebo check.

*Algorithm (Design 1' nonparametric - summarized from Section 3.1.3-3.1.4 and Equations 7-8):*
1. Compute bandwidth `ĥ*_G` via Calonico et al. (2018) plug-in MSE-optimal bandwidth selector on the local-linear regression of `ΔY_g` on `D_{g,2}` with kernel weights `k(D_{g,2}/h)/h`.
2. Fit the local-linear regression at bandwidth `ĥ*_G`; read off the intercept `μ̂_{ĥ*_G}`.
3. Compute `β̂_{ĥ*_G}^{np} = ((1/G) Σ ΔY_g - μ̂_{ĥ*_G}) / ((1/G) Σ D_{g,2})` (Equation 7).
4. Compute the first-order bias estimator `M̂_{ĥ*_G}` and the variance estimator `V̂_{ĥ*_G}` (Calonico et al. 2018, 2019).
5. Form the bias-corrected 95% CI by Equation 8.

*Algorithm variant - Design 1 mass-point 2SLS (Section 3.2.4):*
1. Detect a mass point at `d̲`: either user-supplied `d̲` or detected automatically via the `design="auto"` rule (fraction of observations at `min_g D_{g,2}` exceeds 2%).
2. Either compute `(Ȳ_{D_2 > d̲} - Ȳ_{D_2 = d̲}) / (D̄_{D_2 > d̲} - D̄_{D_2 = d̲})` (sample averages), or run 2SLS of `ΔY_g` on `D_{g,2}` with instrument `1{D_{g,2} > d̲}`.
3. Report the estimate as WAS_{d̲} under Assumption 6 or as the sign-identifying quantity under Assumption 5.

*Algorithm variant - QUG null test (Theorem 4, Section 3.3):*
Tuning-parameter-free test of `H_0: d̲ = 0` versus `H_1: d̲ > 0`. Shipped in `diff_diff/had_pretests.py` as `qug_test()`.
1. Sort `D_{2,g}` ascending to obtain order statistics `D_{2,(1)} ≤ D_{2,(2)} ≤ ... ≤ D_{2,(G)}`.
2. Compute test statistic `T := D_{2,(1)} / (D_{2,(2)} - D_{2,(1)})`.
3. Reject `H_0` if `T > 1/α - 1`.
4. Theorem 4 establishes: asymptotic size `α`; uniform consistency against fixed alternatives; local power at rate `G` on the class `F^{d̲,d̄}_{m,K}` of differentiable cdfs with positive density and Lipschitz derivative.
5. Li et al. (2024, Theorem 2.4) implies the QUG test is asymptotically independent of the WAS / TWFE estimator, so conditional inference on WAS given non-rejection does not distort inference (asymptotically; the paper's Footnote 8 notes the extension to triangular arrays is conjectured but not proven).
- **Note:** Implementation is `O(G)` via `np.partition`; no sort required.
- **Note (Phase 4.5 C0):** `qug_test(..., survey_design=...)` raises `NotImplementedError` **permanently** (Phase 4.5 C0 decision gate, 2026-04 -- direct-helper gate is permanent; the removed 3.7.x `survey=`/`weights=` aliases now raise `TypeError` at the signature). The Phase 4.5 C0 release also gated `did_had_pretest_workflow(..., survey=...)` / `weights=` with `NotImplementedError`, but that workflow gate was **temporary**: Phase 4.5 C (PR #370, 2026-04) replaces it with functional dispatch that skips the QUG step with `UserWarning` and runs the linearity family with the survey-aware mechanism (see Note (Phase 4.5 C) below for the full algorithm). Direct callers of `qug_test` still get the permanent rejection. Three reasons QUG-under-survey is genuinely hard, not "we just haven't done the lit review":
  1. **Extreme order statistics are not smooth functionals of the empirical CDF.** Standard survey machinery (Binder-TSL linearization via `compute_survey_if_variance`, Rao-Wu rescaled bootstrap via `bootstrap_utils.generate_rao_wu_weights`, Krieger-Pfeffermann (1997) EDF tests for complex surveys) all rely on Hadamard differentiability of the test statistic in the empirical CDF. The first two order statistics are NOT differentiable functionals — small perturbations to F near zero produce O(1) shifts in `D_{(1)}`. None of the standard survey-bootstrap or linearization tools give a calibrated test for QUG.
  2. **The `Exp(1)/Exp(1)` limit law assumes iid sampling with smooth density at zero.** Under cluster sampling, `D_{(1)}` and `D_{(2)}` may both come from the same PSU, breaking the independence required for the Poisson-process limit of rescaled spacings near the boundary. Under stratification, the smallest dose may come from a small stratum that's systematically over- or under-sampled, biasing the test.
  3. **The literature on EVT under unequal-probability sampling is sparse.** Quintos et al. (2001) and Beirlant et al. cover tail-INDEX estimation under unequal sample sizes. There is no off-the-shelf method for "test the support endpoint under complex sampling" in the standard survey-statistics toolkit. Adapting Hill / Pickands / DEdH estimators to the boundary problem would be novel research, not engineering. The de Chaisemartin et al. (2026) paper itself does not discuss survey extensions of QUG.
  The survey-compatible alternative for HAD pretesting is **joint Stute** (a CvM cusum of regression residuals) — a smooth functional of the empirical CDF for which Krieger-Pfeffermann (1997) + a survey-aware multiplier bootstrap give a calibrated test. Phase 4.5 C (PR #370) ships survey support for the linearity family — the **PSU-level Mammen multiplier bootstrap** for `stute_test` and the joint variants (NOT Rao-Wu rescaling — multiplier bootstrap is a different mechanism), and **closed-form weighted OLS + pweight-sandwich variance components** for `yatchew_hr_test`. See the dedicated Note (Phase 4.5 C) below for the full algorithm.
  **Research direction (out of scope for diff-diff):** the bridge IS sketchable by combining (a) endpoint-estimation EVT under iid (Hall 1982, Aarssen-de Haan 1994, Hall-Wang 1999, Beirlant-de Wet-Goegebeur 2006); (b) survey-aware functional CLT for the empirical process (Boistard-Lopuhaä-Ruiz-Gazen 2017, Bertail-Chautru-Clémençon 2017); and (c) tail-empirical-process theory (Drees 2003) to define a "design-effective boundary intensity" `λ_eff = Σ_h W_h · f_h(0+)`. Under a "no boundary clumping" assumption (`P(D_{(1)}, D_{(2)}` in same PSU `| both ≤ δ) → 0`), the `Exp(1)/Exp(1)` limit law's pivotality is preserved and only the calibration needs a survey-aware bootstrap (subsampling within strata per Politis-Romano-Wolf, or Bertail et al.'s design-aware bootstrap). This is publishable methodology research — one paper, ~6-12 months for a methods PhD student. If the bridge gets built and published externally, this gate can be revisited.
- **Note (Phase 4.5 C):** `stute_test`, `yatchew_hr_test`, `stute_joint_pretest`, `joint_pretrends_test`, `joint_homogeneity_test`, and `did_had_pretest_workflow` accept `survey_design=` (the sole weighting entry — the deprecated `survey=`/`weights=` aliases were removed in 3.7.x; see "Note (HAD survey-design API consolidation)" below). On data-in surfaces (`did_had_pretest_workflow`, `joint_pretrends_test`, `joint_homogeneity_test`), `survey_design=` accepts a `SurveyDesign` (resolved against `data` at fit time). On array-in surfaces (`stute_test`, `yatchew_hr_test`, `stute_joint_pretest`), `survey_design=` accepts a pre-resolved `ResolvedSurveyDesign`; for the pweight-only convenience, construct via `survey_design=make_pweight_design(arr)` (`make_pweight_design` exported from the `diff_diff` top level). Mechanism varies by test:
  - **Stute family** (`stute_test`, `stute_joint_pretest`, joint wrappers) uses **PSU-level Mammen multiplier bootstrap** via `bootstrap_utils.generate_survey_multiplier_weights_batch` (the same kernel as PR #363's HAD event-study sup-t bootstrap). Each replicate draws an `(n_bootstrap, n_psu)` Mammen multiplier matrix; multipliers broadcast to per-obs perturbation `eta_obs[g] = eta_psu[psu(g)]`. The bootstrap residual perturbation is `dy_b = fitted + eps * eta_obs` (paper Appendix D wild-bootstrap form — multipliers attach to UNWEIGHTED residuals; the weighting flows through the OLS refit + the weighted CvM, NOT through the perturbation step). Followed by weighted OLS refit (`_fit_weighted_ols_intercept_slope`) and weighted CvM recompute via `_cvm_statistic_weighted`. Joint Stute SHARES the multiplier matrix across horizons within each replicate, preserving both the vector-valued empirical-process unit-level dependence (Delgado 1993; Escanciano 2006) AND PSU clustering (Krieger-Pfeffermann 1997). PSU-shared multipliers are conservative under no-within-PSU outcome correlation (over-clustering gives conservative size in finite samples), asymptotically correct under the standard survey assumption that PSU is the ultimate sampling unit AND outcomes correlate within PSU. The pweight-only entry (`survey_design=make_pweight_design(arr)`) routes through a synthetic trivial `ResolvedSurveyDesign` (constructed via `make_pweight_design`, the public alias for the formerly private `survey._make_trivial_resolved`) so the kernel is shared across both entry paths. NOT "Rao-Wu rescaled bootstrap" — different mechanism (the Rao-Wu kernel rescales per-unit weights via stratified PSU resampling, while this kernel applies multipliers without resampling).
  - **Yatchew** (`yatchew_hr_test`) uses **closed-form weighted OLS + pweight-sandwich variance components** (no bootstrap). All three components reduce bit-exactly to the unweighted formulas at `w=ones(G)` (locked at `atol=1e-14` in `TestYatchewHRTestSurvey::test_weighted_reduces_to_unweighted_at_uniform_weights`):
    - `sigma2_lin = sum(w * eps^2) / sum(w)` (weighted OLS residual variance).
    - `sigma2_diff = sum(w_avg * (dy_g - dy_{g-1})^2) / (2 * sum(w))` with arithmetic-mean pair weights `w_avg_g = (w_g + w_{g-1})/2`. Divisor uses `sum(w)` (=G at `w=1`), NOT `sum(w_avg)`, to match the existing `(1/(2G))` unweighted formula in `yatchew_hr_test`.
    - `sigma4_W = sum(w_avg * eps_g^2 * eps_{g-1}^2) / sum(w_avg)` reduces to `(1/(G-1)) * sum(prod)` at `w=1`.
    - `T_hr = sqrt(sum(w)) * (sigma2_lin - sigma2_diff) / sigma2_W` (effective-sample-size convention; reduces to `sqrt(G)` at `w=1`).
    Strictly positive weights required (the adjacent-difference variance is undefined under contiguous-zero blocks). PSU clustering is NOT propagated through the variance-ratio statistic (would require a survey-aware variance-of-variance estimator, out of scope). Pair-weight convention follows Krieger-Pfeffermann (1997, §3) for design-consistent inference on smooth functionals.
  - **Workflow** (`did_had_pretest_workflow`) under `survey_design=`: skips the QUG step with a `UserWarning` (per Phase 4.5 C0 deferral), sets `qug=None` on the report, and dispatches the linearity family with the survey-aware mechanism. Verdict carries a `"linearity-conditional verdict; QUG-under-survey deferred per Phase 4.5 C0"` suffix. `all_pass` drops the QUG-conclusiveness condition; the linearity-conditional rule splits by aggregate:
    - the overall battery (two-period panels): `True` iff at least one of `stute`/`yatchew` is conclusive AND no conclusive test rejects (paper Section 4 step-3 "Stute OR Yatchew" wording carries through).
    - the event-study battery (multi-period panels): `True` iff `pretrends_joint` is non-None and conclusive, `homogeneity_joint` is conclusive, AND neither rejects. Both joint variants must be conclusive on the event-study path (same step-2 + step-3 closure as the unweighted aggregate, just without the QUG step).
  - **Replicate-weight survey designs (BRR/Fay/JK1/JKn/SDR) deferred** to a parallel follow-up. Each helper raises `NotImplementedError` on `survey.replicate_weights is not None` (defense in depth: workflow + every direct-helper entry rejects, mirroring the reciprocal-guard discipline from PR #346). The per-replicate weight-ratio rescaling for the OLS-on-residuals refit step is not covered by the multiplier-bootstrap composition above.
  - **`lonely_psu='adjust'` with singleton strata is rejected** with `NotImplementedError` on the Stute family (mirrors HAD sup-t bootstrap at `had.py:2194-2231`). The bootstrap multiplier helper pools singleton strata into a pseudo-stratum with nonzero multipliers, but the analytical variance target requires a pseudo-stratum centering transform that has not been derived for the Stute CvM. Use `lonely_psu='remove'` (drops singleton contributions) or `'certainty'` (zero-variance singletons); both produce all-zero singleton multipliers that match a well-defined analytical target. Variance-unidentified designs (`df_survey <= 0` after the adjust+singleton case is handled) return `NaN` with a `UserWarning` (single-PSU unstratified or one-PSU-per-stratum under remove/certainty).
  - **Stratified designs (`SurveyDesign(strata=...)`) are supported** via the standard stratified clustered wild bootstrap correction on the PSU multipliers (Cameron-Gelbach-Miller 2008; Davidson-Flachaire 2008; Djogbenou-MacKinnon-Nielsen 2019; Kreiss-Lahiri 2012; Wu 1986; Liu 1988). See the dedicated **"Note: Stute stratified survey-bootstrap calibration"** below for the algorithm. Remaining deferrals: `lonely_psu='adjust'` + singleton strata (same pseudo-stratum centering gap as the HAD sup-t deviation documented above; requires a separate analytical-target derivation) and replicate-weight designs (BRR/Fay/JK1/JKn/SDR; separate Rao-Wu/JKn bootstrap composition).

- **Note (Stute stratified survey-bootstrap calibration):** The Stute survey-bootstrap is a wild residual bootstrap (Hlávka-Hušková 2020) with cluster-level multipliers (Cameron, Gelbach & Miller 2008). The per-replicate loop is `eta_obs = psu_mults[b, psu_col_idx]; dy_b = fitted + eps * eta_obs; refit weighted OLS; recompute weighted CvM` (the per-replicate loops in `stute_test` and `stute_joint_pretest`). Under stratified PSU sampling, the multipliers `psu_mults[b, :]` exit `generate_survey_multiplier_weights_batch` as within-stratum-independent draws with the `(1 - f_h)` FPC factor already baked in (`bootstrap_utils.py:579-651`). To make the bootstrap CvM variance match the analytical Binder-TSL stratified target `V_S = sum_h (1 - f_h) * (n_h / (n_h - 1)) * sum_{j in h} (psi_hj - psi_h_bar)²`, two additional corrections are applied to the multipliers BEFORE the per-obs broadcast. **Citations below are ingredients, not direct papers on this exact composition** — the specific recipe (within-stratum demean + Bessel rescale on PSU multipliers applied before broadcast in a wild-residual refit-in-loop bootstrap for the Stute CvM functional) is a library synthesis; no single paper covers all of it.

    1. **Within-stratum demean**: for each stratum `h`, `psu_mults[b, cols_h] -= psu_mults[b, cols_h].mean()`. The within-cluster mean-zero requirement is the canonical wild-bootstrap centering for heteroskedastic regression (Davidson & Flachaire 2008); applying it within each stratum (rather than across all clusters) under stratified PSU sampling is the library synthesis. The Kreiss-Lahiri (2012) block-bootstrap family supplies the methodological analogy for within-block centering under hierarchical sampling but does not cover stratified-survey designs directly.

    2. **Bessel rescale**: multiply by `sqrt(n_h / (n_h - 1))`. Standard small-sample correction (Wu 1986; Liu 1988) — makes the bootstrap variance match the unbiased within-stratum variance estimator and bakes the remaining `(n_h / (n_h - 1))` factor of `V_S`.

    The combined correction is *algebraically identical* to the HAD sup-t event-study bootstrap's stratum centering (`had.py:2188-2204`), applied to PSU multipliers instead of the PSU-aggregated influence tensor. Both call sites consume the shared helper `bootstrap_utils.apply_stratum_centering(psu_mults, resolved_survey, psu_ids)` to lock the algebraic identity architecturally rather than relying on parallel code blocks staying in sync.

    The pipeline-position difference between Stute and HAD sup-t is forced by the bootstrap structure, NOT a difference in the correction algebra: HAD sup-t is a multiplier bootstrap on a *precomputed* PSU influence tensor (the sup-t statistic is a linear functional of that tensor — `perturbations = psu_weights @ Psi_psu_scaled`), so the demean + Bessel rescale can be applied to the tensor after PSU aggregation. Stute is a wild residual bootstrap that *refits* OLS and recomputes the *nonlinear* CvM functional inside the per-draw loop, so there is no precomputed PSU influence tensor to scale; the correction has to be applied at the multiplier-generation step, before the per-obs broadcast.

    Consistency of the resulting bootstrap CvM distribution under stratified PSU sampling follows from Djogbenou, MacKinnon & Nielsen (2019) Theorem 2 (empirical-process consistency of the cluster wild bootstrap), with the Krieger-Pfeffermann (1997) survey-weighted multiplier-bootstrap extension routed through the multiplier draws rather than the influence tensor. For the multi-horizon joint Stute (`stute_joint_pretest`), the same `psu_mults[b, :]` row is shared across horizons within each replicate, preserving cross-horizon empirical-process dependence (Hlávka-Hušková 2020 §3 condition) and PSU clustering. The combined correction is the standard non-parametric requirement and does not depend on the CvM functional shape — it works for any nonlinear smooth-functional bootstrap consumer of `eta_obs = psu_mults[b, psu_col_idx]`.

    **Non-strata calibration improvement.** When `strata=None`, the correction is applied uniformly with a single implicit stratum (`n_h = n_psu`): demean across all PSUs, multiply by `sqrt(n_psu / (n_psu - 1))`. This mirrors the HAD sup-t convention at `had.py:2312-2317` and brings Stute non-strata into line with the sibling event-study path. The pre-PR Phase 4.5 C non-strata path applied no centering or rescaling — multipliers were raw iid draws. The bootstrap CvM p-values on non-strata designs (pweight-only, PSU-only, FPC-only) **shift by approximately `sqrt(n_psu / (n_psu - 1)) - 1`** relative to the pre-PR path (≈ 1.7% for `n_psu = 60`, decreasing to ≈ 0.5% for `n_psu = 100`). This is a calibration improvement, NOT a regression: the pre-PR path was under-corrected by exactly this factor. Two complementary regressions cover any revert of the helper or its wiring: (1) the helper bit-parity regression at `tests/test_bootstrap_utils.py::TestApplyStratumCentering::test_bit_parity_vs_pre_refactor_inline_block` (locked at `atol=1e-14`) catches any change to the helper's axis-0 algebra; (2) a wired-in regression at `tests/test_had_pretests.py::TestStuteStratifiedSurveyBootstrap::test_stute_call_sites_invoke_apply_stratum_centering` monkey-patches the helper and asserts both Stute call sites (`stute_test` and `stute_joint_pretest`) invoke it with `psu_axis=1`, which catches the disconnection case the helper bit-parity test does not. End-to-end Stute non-strata fit is exercised as a finite + range smoke (`tests/test_had_pretests.py::TestStuteStratifiedSurveyBootstrap::test_calibration_shift_non_strata_end_to_end_smoke`); a heavier worktree-based pre/post baseline comparison was considered and intentionally skipped as redundant with the helper-level bit-parity lock and the call-site wiring regression.

    **Validated via:** MC oracle consistency under a stratified null DGP (200 draws, 4 strata × 6 PSUs/stratum, weights+strata+PSU design — no FPC at the panel level; the helper's FPC bake-in is covered separately by `tests/test_bootstrap_utils.py::TestApplyStratumCentering::test_fpc_baked_in_helper_is_fpc_agnostic`); empirical Type I error at α=0.05 in `[0.0, 0.10]` (3σ band, seed-set). MC power under a stratified known-alternative DGP (same shape, quadratic `E[ΔY|D]`); rejection rate > 0.50 at α=0.05.

    **Known parity gap.** No R reference implements stratified Stute under survey weights — `chaisemartin::did_had` does not run pretests at all, and `nprobust` has no weight argument. Methodology confidence comes from the algebraic-identity reduction to the existing HAD sup-t centering (locked at `atol=1e-14` by the shared-helper unit test + HAD sup-t bit-parity regression) + the MC oracle + power simulations above. Same parity-ceiling acknowledgment as Phase 4.5 A0 (no public weighted-CCF reference for the bias-corrected local-linear).
  - **Constant-within-unit invariant**: the per-row `survey_design=SurveyDesign(weights='col', ...)` inputs are aggregated to per-unit `(G,)` arrays via the existing HAD helpers `_aggregate_unit_weights` / `_aggregate_unit_resolved_survey` (had.py:1604, :1671); these enforce constant-within-unit invariant on weights and on every survey design column (strata, psu, fpc) and raise on violation. Direct callers passing already-resolved `ResolvedSurveyDesign` (or per-unit `weights` array) bypass this aggregation; the invariant is the caller's responsibility on that path.
  - **Distributional parity, NOT bit-exact**: at unit weights (`make_pweight_design(ones(G))`) the survey path produces a different bootstrap p-value than the unweighted path because RNG consumption differs (batched `generate_survey_multiplier_weights_batch` vs the unweighted `_generate_mammen_weights` stream — itself batched into memory-bounded chunks since the 2026-07 loop-invariant hoist, which is IDENTICAL to per-iteration draws because `rng.choice` fills C-order; see the Phase-3 hoist Note). The two paths agree DISTRIBUTIONALLY at large B (`|p_avg_diff| < 0.03` over 100 reps at `B=5000`); they DO NOT agree numerically at `atol=1e-10`. The unweighted code path is preserved bit-exactly (stability invariant; the survey-aware `survey_design=` branch is a separate `if` arm).

*Algorithm variant - TWFE linearity test via Stute (1997) Cramér-von Mises with wild bootstrap (Section 4.3, Appendix D):*
Shipped in `diff_diff/had_pretests.py` as `stute_test()`. Tests whether `E(ΔY | D_2)` is linear, the testable implication of TWFE's homogeneity assumption (Assumption 8) in HADs.
1. Fit linear regression of `ΔY_g` on constant and `D_{g,2}`; collect residuals `ε̂_{lin,g}`.
2. Form cusum process `c_G(d) := G^{-1/2} Σ_{g=1}^G 1{D_{g,2} ≤ d} · ε̂_{lin,g}`.
3. Compute Cramér-von Mises statistic `S := (1/G) Σ_{g=1}^G c_G²(D_{g,2})`. Equivalently, after sorting by `D_{g,2}`: `S = Σ_{g=1}^G (g/G)² · ((1/g) Σ_{h=1}^g ε̂_{lin,(h)})²`.
4. Wild bootstrap for p-value (Stute, Manteiga, Quindimil 1998; Algorithm in main text p. 25 and vectorized form in Appendix D):
   - Draw `(η_g)_{g=1,...,G}` i.i.d. from the Mammen two-point distribution: `η_g = (1+√5)/2` with probability `(√5-1)/(2√5)`, else `η_g = (1-√5)/2`. Reuses `diff_diff.bootstrap_utils.generate_bootstrap_weights(..., "mammen")`.
   - Set `ε̂*_{lin,g} := ε̂_{lin,g} · η_g`.
   - Compute `ΔY*_g = β̂_0 + D_{g,2} · β̂_{fe} + ε̂*_{lin,g}` (paper writes `ΔD_g` here, which equals `D_{g,2}` since `D_{g,1} = 0`; the two forms are equivalent in this design).
   - Re-fit OLS on the bootstrap sample to get `ε̂*_{lin,g}`, compute `S*`.
   - Repeat B times; the p-value is the fraction of `S*` exceeding `S`.
5. Properties (page 26): asymptotic size, consistency under any fixed alternative, non-trivial local power at rate `G^{-1/2}`.
6. Vectorized implementation (Appendix D): with `L` a `G × G` lower-triangular matrix of ones, `S = (1/G²) · 1ᵀ (L · E)^{∘2}`. Bootstrap uses a `G × G` realization matrix `H` of Mammen weights; memory-bounded at `G ≈ 100,000`.
- **Note:** Default `n_bootstrap = 999` is a diff-diff choice; the paper does not prescribe. A front-door `ValueError` is raised for `n_bootstrap < 99` (below which the discretised bootstrap p-value grid `1/(B+1)` is too coarse to be meaningful).

*Algorithm variant - Yatchew (1997) heteroskedasticity-robust specification test (Appendix E, Theorem 7):*
Shipped in `diff_diff/had_pretests.py` as `yatchew_hr_test()`. Alternative to Stute when `G` is large or heteroskedasticity is suspected. Two nulls supported via the keyword-only `null=` argument: `"linearity"` (default; paper Theorem 7) and `"mean_independence"` (R-parity extension, see Note below).
1. Sort `(D_{g,2}, ΔY_g)` by `D_{g,2}`.
2. Compute difference-based variance estimator: `σ̂²_{diff} := (1/(2G)) Σ_{g=2}^G [(Y_{2,(g)} - Y_{1,(g)}) - (Y_{2,(g-1)} - Y_{1,(g-1)})]²`.
3. Fit OLS; compute residual variance `σ̂²_{lin}`. Under `null="linearity"` (default): residuals from `dy ~ 1 + d` (paper Theorem 7). Under `null="mean_independence"`: residuals from intercept-only `dy ~ 1`, i.e. `eps = dy - mean(dy)` (R-parity extension).
4. Heteroskedasticity-robust variance: `σ̂⁴_W := (1/(G-1)) Σ_{g=2}^G ε̂²_{lin,(g)} ε̂²_{lin,(g-1)}`.
5. Robust test statistic: `T_{hr} := √G · (σ̂²_{lin} - σ̂²_{diff}) / σ̂²_W`. Under `null="linearity"`, reject linearity if `T_{hr} ≥ q_{1-α}` (Equation 29 and downstream in Theorem 7). Under `null="mean_independence"`, the same statistic and critical value reject the mean-independence null `H_0: E[dY|D] = E[dY]` — only the residual definition (and therefore `σ̂²_lin`) differs between modes; the `σ̂²_diff`, `σ̂⁴_W`, and sort-by-`d` machinery are shared.
6. Theorem 7: under `H_0`, `lim E[φ_α] = α`; under fixed alternative, `lim E[φ_α] = 1`; local power against alternatives at rate `G^{-1/4}` (slower than Stute's `G^{-1/2}` rate, but scales to `G ≥ 10⁵`).
7. Inference on `β̂_{fe}` conditional on accepting the linearity test is asymptotically valid (Theorem 7, Point 1; citing de Chaisemartin and D'Haultfœuille 2024 arXiv:2407.03725).
- **Note (mean-independence null is the R-parity extension, not paper-derived):** The paper (Appendix E, Theorem 7) defines `yatchew_hr_test` only under the linearity null. The `null="mean_independence"` mode mirrors R `YatchewTest::yatchew_test(order=0)` and is used by R `DIDHAD::did_had(yatchew=TRUE)` on placebo rows ("non-parametric pre-trends test" per the package README). Exposed for parity coverage of the placebo-Yatchew rows in `tests/test_did_had_parity.py::TestYatchewParity` (PR #397). The default `"linearity"` mode is paper-derived; users invoking `"mean_independence"` are running an R-parity extension, not a paper-prescribed test.

*Four-step pre-testing workflow (Section 4.2-4.3):*
Shipped as `did_had_pretest_workflow()` in Phase 3 (two-period `aggregate="overall"`) and extended in the Phase 3 follow-up with `aggregate="event_study"` dispatch that closes the step-2 pre-trends gap on multi-period panels. The paper's decision rule for TWFE reliability in HADs:
1. Test the null of a QUG (`H_0: d̲ = 0`) using `qug_test()`.
2. Run a pre-trends test of Assumption 7 (requires at least one earlier pre-period).
3. Test that `E(ΔY | D_2)` is linear (`stute_test` or `yatchew_hr_test`; or the joint Stute variants below in event-study dispatch).
4. If NONE of the three is rejected, `β̂_{fe}` from TWFE may be used to estimate the treatment effect.

**Phase 3 delivery (`aggregate="overall"`, two-period):** `did_had_pretest_workflow()` runs steps 1 + 3 (QUG + Stute + Yatchew). Step 2 is NOT run on this path because a two-period panel has no pre-period placebo horizon to test against; the verdict explicitly flags the Assumption 7 gap via the "paper step 2 deferred" caveat.

**Phase 3 follow-up delivery (`aggregate="event_study"`, multi-period):** `did_had_pretest_workflow(..., aggregate="event_study")` dispatches on a balanced ≥3-period panel. Runs QUG at `F` + joint pre-trends Stute across earlier pre-periods (step 2, mean-independence null) + joint homogeneity-linearity Stute across post-periods (step 3 joint extension, linearity null). Step 2 closure requires at least TWO pre-periods (the base pre-period plus one earlier placebo); on panels with only a single pre-period (the base `F-1`) the workflow emits `pretrends_joint=None` and the verdict flags the skip ("joint pre-trends skipped (no earlier pre-period)"). `all_pass` is False in this degenerate case. The verdict on the event-study path does NOT emit the "paper step 2 deferred" caveat when step 2 runs.

*Algorithm variant - Joint Stute tests (Section 4.2-4.3 joint; Phase 3 follow-up 2026-04):*
Shipped in `diff_diff/had_pretests.py` as `stute_joint_pretest()` (residuals-in core) plus two thin data-in wrappers `joint_pretrends_test()` (mean-independence null) and `joint_homogeneity_test()` (linearity null). Generalizes the single-horizon Stute CvM (above) to K horizons with joint inference.
1. Per-horizon statistic: for each horizon `k`, compute `S_k` via the tie-safe CvM on residuals `ε̂_{g,k}` sorted by dose `D_g`.
2. Joint aggregation: `S_joint = Σ_k S_k` (sum-of-CvMs).
3. Wild bootstrap with **shared** Mammen multiplier `η_g` across horizons per unit — preserves the vector-valued empirical process's unit-level dependence (Delgado-Manteiga 2001; Hlávka-Hušková 2020 for related vector wild-bootstrap theory). Per-horizon OLS refit with shared design matrix precomputes `(X'X)^{-1} X'` once; the bootstrap loop cost per draw is `O(G·p·K)` for K horizons.
4. Per-horizon exact-linear short-circuit: scale- and translation-invariant `Σ eps_k² / centered_TSS_k < _EXACT_LINEAR_RELATIVE_TOL` test applied per horizon. Joint short-circuit fires only when EVERY horizon is machine-exact linear; a single degenerate horizon does not collapse the test when others have nontrivial residuals.
5. Two data-in wrappers:
   - `joint_pretrends_test(pre_periods, base_period)`: `null_form="mean_independence"`, design matrix `[1]`; residuals from `OLS(Y_t - Y_base ~ 1)` per pre-period (paper Section 4.2 footnote 6 + Section 4.3 paragraph 1: "regress Y_1 − Y_0 on a constant [only], then apply CvM to residuals vs D_2").
   - `joint_homogeneity_test(post_periods, base_period)`: `null_form="linearity"`, design matrix `[1, D]`; residuals from `OLS(Y_t - Y_base ~ 1 + D)` per post-period (paper Section 4.3 page 32 joint across post-periods, Pierce-Schott reports p=0.40).
- **Note:** Sum-of-CvMs aggregation is a standard joint specification-test construction (Delgado 1993; Escanciano 2006); the paper does not prescribe an aggregation rule. Sum-of-CvMs balances power across diffuse vs concentrated alternatives and bootstraps cleanly with shared-η.
- **Note:** Event-study dispatch adjudicates step 3 via joint Stute only; there is no joint Yatchew variant because the paper does not derive one. The overall two-period path still uses the Phase 3 "Stute OR Yatchew" adjudication. Users who need Yatchew-style adjacent-difference variance-ratio robustness under multi-period data can run `yatchew_hr_test` on each (base, post) pair manually.
- **Note (Phase 4 — Eq 17 / Eq 18 linear-trend detrending shipped):** `trends_lin: bool = False` (keyword-only) on `HeterogeneousAdoptionDiD.fit(aggregate="event_study")`, `joint_pretrends_test`, and `joint_homogeneity_test` (PR #389, 2026-04). Mirrors R `DIDHAD::did_had(..., trends_lin=TRUE)` (Credible-Answers/did_had v2.0.0, SHA `edc09197`). Per-group linear-trend slope estimated as `Y[g, F-1] - Y[g, F-2]` and applied as `(t - base) × slope` adjustment to per-event-time outcome evolutions (HAD.fit) or to `Y[g, t] - Y[g, base]` directly (joint pretests). The "consumed" placebo at our event-time `e=-2` is auto-dropped (R reduces max placebo lag by 1 with the same effect). Requires F ≥ 3 / `base_period - 1` in panel — front-door `ValueError` if not. Mutually exclusive with survey weighting (raises `NotImplementedError` per `feedback_per_method_survey_element_contract`; weighted slope estimator not derived from paper). Pierce-Schott published-number replication (paper p=0.51 / p=0.40 anchors) deferred indefinitely — primary analysis panel is LBD-restricted (Census FSRDC); the public-deposit proxy panel has filtering ambiguity that prevents exact published-number parity. Replaced by end-to-end R-package parity below, which is a strictly stronger correctness signal.
- **Note (R-package end-to-end parity, PR #389):** Validated against `DIDHAD` v2.0.0 (Credible-Answers/did_had, SHA `edc09197`) on the **`design="continuous_at_zero"` (Design 1') surface**, on 3 paper-derived synthetic DGPs (Uniform, Beta(2,2), Beta(0.5,1)) × 5 method combinations (overall, event-study, placebo, yatchew, trends_lin). Generator: `benchmarks/R/generate_did_had_golden.R`; fixture: `benchmarks/data/did_had_golden.json`; test: `tests/test_did_had_parity.py`. **Scope qualifier (PR #392 R8 P3):** the harness explicitly forces `HeterogeneousAdoptionDiD(design="continuous_at_zero")` because R `did_had` always evaluates the local-linear at `d=0` regardless of dose distribution. Our default `design="auto"` may legitimately resolve to `continuous_near_d_lower` (`d_lower=d.min()`, Design 1) or `mass_point` (Design 2) on dose distributions with boundary density bounded away from zero (e.g., Beta(2,2) at G=200), in which case the WAS estimand evaluates at a different point and diverges from R's `did_had` numerically. That divergence is methodologically defensible — our auto-detect uses more information when boundary mass is sparse — but is out of scope for this parity contract. Tolerances: point estimate / SE / CI bounds at `atol=1e-8`; closed-form Yatchew T-stat at `atol=1e-10` after a documented `× G/(G-1)` finite-sample convention shift. Two intentional convention deviations from R: **(a)** we report the **bias-corrected** point estimate `att = (mean(ΔY) - tau.bc) / mean(D)` (modern CCF 2018 convention); R's `Estimate` column reports the **conventional** estimate `(mean(ΔY) - tau.us) / mean(D)` with the bias-corrected CI separately — our `att` matches R's CI midpoint, our `se` / `conf_int_low` / `conf_int_high` match R's `se` / `ci_lo` / `ci_hi` directly. **(b)** Our `yatchew_hr_test` follows paper Appendix E's literal `(1/G)` and `(1/(2G))` variance-denominator convention; R's `YatchewTest::yatchew_test` uses base-R `var()`'s `(1/(N-1))` sample-variance convention. Ratio is exactly `N/(N-1)`; both converge to the same asymptotic null distribution. Yatchew on placebos with R's mean-independence null (`order=0`, fits `Y ~ 1`) is exposed via `yatchew_hr_test(null="mean_independence")` (added post-PR #392). The parity test routes effect rows through `null="linearity"` (R `order=1`) and placebo rows through `null="mean_independence"` (R `order=0`); both modes share the same `(1/G)` vs `(1/(N-1))` finite-sample convention shift and parity holds at `atol=1e-10` after the documented `× G/(G-1)` transform.
- **Note:** Horizon labels in `StuteJointResult.horizon_labels` are `str(t)` verbatim and carry STRING IDENTITY ONLY — NOT a chronological ordering key. Callers who need chronological order must preserve the original period values alongside (e.g. from the `pre_periods` / `post_periods` argument).
- **Note:** NaN propagation is explicit: when any horizon has NaN in residuals, `cvm_stat_joint=NaN`, `p_value=NaN`, `reject=False`, AND `per_horizon_stats={label: np.nan for every horizon}` (full dict preserved with NaN values — not empty, not partial).

**Phase 3 follow-up delivery:** `stute_joint_pretest()`, `joint_pretrends_test()`, `joint_homogeneity_test()`, `StuteJointResult`, and `did_had_pretest_workflow(aggregate="event_study")` shipped together in PR #353 (2026-04). The `practitioner_next_steps()` HAD handlers landed in Phase 5 wave 1 (PR #402); the T21 HAD pretest workflow tutorial landed in PR #409 (Phase 5 wave 2 first slice). The T22 survey-weighted HAD tutorial (`docs/tutorials/22_had_survey_design.ipynb`) shipped as the follow-up to PR #432 (2026-05).

**Reference implementation(s):**
- R: `did_had` (de Chaisemartin, Ciccia, D'Haultfœuille, Knau 2024a); `stute_test` (2024c); `yatchew_test` (Online Appendix, Table 3).
- Stata: `did_had` (2024b); `stute_test` (2024d); `yatchew_test`. Also `twowayfeweights` (de Chaisemartin, D'Haultfœuille, Deeb 2019) for negative-weight diagnostics.
- Underlying bias-correction machinery: Calonico, Cattaneo, Farrell (2018, 2019) `nprobust`; ported in-house for diff-diff (decision recorded in the plan).

**Deviations and library extensions:**

*Notes #1-#2 lock implementation choices (paper-permitted choices the library codified); Notes #3-#4 document validation-harness work waived in this PR with documented rationale; #5 is a Library extension where the library departs from the paper's prescription toward stricter safety.*

- **Note:** Equal-weighting on the continuous path. Paper does not prescribe a unit-weighting scheme on the continuous local-linear paths. Library uses per-unit equal weighting (`w_g = 1` default, matching `diff_diff/_nprobust_port.lprobust`'s default), NOT dose-cell-size weights. Practical consequence: WAS is the population-mean slope from Eq. 3 — `[E(ΔY) − lim_{d↓d̲} E(ΔY | D ≤ d)] / E(D)` (computed as `att = (mean(ΔY) − τ_bc) / mean(D)`), not a cell-size-weighted average; with cell-size weighting, units in less-densely-populated regions of the dose distribution would contribute disproportionately to the boundary slope. User-supplied weights (via `survey_design=SurveyDesign(weights=...)`, pweight) override the equal-weight default and thread through as `W_combined = k((D − d̲)/h) · w_g`. Lock in `tests/test_methodology_had.py::TestHADDeviations::test_equal_weighting_is_per_row_not_per_dose_cell`.
- **Note:** Sup-t bootstrap gating. Simultaneous-band sup-t multiplier bootstrap runs in the event-study mode when `cband=True` (default) AND either `survey_design=` is supplied (survey band) OR `cluster=` (cluster-robust band — fires even on an unweighted fit, Phase 2b). The unweighted, unclustered event-study path bit-exactly preserves pre-Phase 4.5 B numerical output (stability invariant). Setting `cband=False` disables the bootstrap on any path. See the algorithmic contract above at `_sup_t_multiplier_bootstrap`.
- **Note (post-fit aggregate() + mode inference - rows M-027/M-139):** `fit(aggregate=)` and `did_had_pretest_workflow(aggregate=)` are deprecated in 3.9 (removed in 4.0). UNLIKE every prior aggregate-postfit adopter, the param was a MODE SELECTOR over mutually exclusive panel shapes ('overall' = exactly two periods -> `HeterogeneousAdoptionDiDResults`; 'event_study' = more than two -> `HeterogeneousAdoptionDiDEventStudyResults`), so the successor is panel-shape INFERENCE: both surfaces resolve the mode from the shared `_infer_aggregate_mode` helper (distinct time values <= 2 -> overall, else event_study), running after the deprecated column-alias reconciliation (the M-035..M-039 renamed-kwarg shims). BEHAVIOR DELTA: a plain multi-period `fit()` (or workflow call) previously raised the two-period shape error and now runs the event-study mode - error -> works; no numbers change on any previously-working call, and a supplied legacy value warns then runs the legacy routing unchanged (invalid values still raise; the one sentinel-reachable shape error is the staggered last-cohort auto-filter dropping a T>2 panel below three periods, its message reworded to panel-shape vocabulary). POST-FIT `aggregate()` is a PURE VIEW with per-class supported subsets - the overall class supports only `'simple'` (one-row bit-exact relay: `target = target_parameter`, the estimand label "WAS"/"WAS_d_lower"; `n = n_obs` contributing units with `n_kind='units'` - the treated/control split is disjoint on continuous designs but the mass-point masks can overlap in a ~1-ULP tolerance band at the `d_lower` boundary, so `n_obs` is the single source; `df = resolve_inference_df`, provenance-exact: the finite survey df on TSL fits, NaN otherwise - replicate designs are rejected at fit so the 0-sentinel branch is unreachable) and the event-study class supports only `'event_study'` (a `build_event_study_surface` passthrough via the `_from_had` adapter, cband fields included; the adapter's `n_kind` was corrected "obs" -> "units" with this row - `n_obs_per_horizon` counts units). Cross-mode levels fail closed (the two modes share no computable estimand; joint cross-horizon covariance is the DEFERRED.md row). NO kit is retained (views work on pickles from any release) and NO bootstrap gate exists (`n_bootstrap` feeds only the sup-t cband critical value; pointwise inference is always analytical). `AggregationResult.summary()` gained the single-non-'att'-target rendering with this row (target column + neutral 'estimate' heading, column sized to the longest label - previously a WAS or dCDH estimand row rendered under a hard-coded 'ATT' heading; uniform-'att' and ContinuousDiD att/acrt output byte-stable). CONSUMER ADMISSION into `compute_honest_did`/`compute_pretrends_power` is DEFERRED, not by-design (see M-093): HAD's event-study coefficients ARE reference-normalized (each horizon differences against the F-1 anchor) but the anchor row is omitted from the container (its coefficient is identically zero AND the WAS is not identified there) and no joint covariance exists; both terminal TypeErrors state the deferral in exactly that wording. The `HADPretestReport.aggregate` FIELD survives as output metadata recording which battery ran (only the params die).
- **Note:** Pierce-Schott (2016) Figure 2 replication harness deferred. The paper's empirical application self-acknowledges (Section 5.2; mirrored in `dechaisemartin-2026-review.md:321`) that "NP estimators are too noisy to be informative" on the LBD-restricted PNTR panel. R parity at `atol=1e-8` on 3 DGPs × 5 method combos via `tests/test_did_had_parity.py` (bit-exact, `rtol=0`) is a stronger correctness anchor than reproducing pointwise CIs on LBD-restricted data. **Scope caveat:** R parity locks point estimate, SE, and CI bounds bit-exactly to R's bounds — it does NOT independently verify the asymptotic-coverage properties of the bias-corrected CI in small samples. Paper Table 1 documents under-coverage at small G (89% at G=100 on DGP 1, 93% at G=500, 95% at G=2500); this is inherited from the CCF asymptotic theory itself, and Python is exact-parity with R at the limit-law machinery.
- **Note:** Table 1 coverage-rate reproduction deferred. Paper Section 3.1.5 reports 2,000-iter Monte Carlo coverage rates at `G ∈ {100, 500, 2500}` on DGPs 1/2/3. The existing `tests/test_did_had_parity.py` R parity at `atol=1e-8` on the same 3 DGPs reproduces the exact point estimate and SE algorithm to bit-exact tolerance; coverage-rate MC would re-verify the CCF asymptotic coverage already pinned by R parity (Python ≡ R ≡ paper) at the sample-mean level. **Scope caveat (mirrors above):** R parity does NOT re-prove asymptotic-coverage at small G; paper Table 1's 89% / 93% / 95% under-coverage band is valid for both R and Python.
- **Library extension:** Staggered-timing fail-closed. Paper Appendix B.2 prescribes "Warn" when staggered treatment timing is detected; library raises `ValueError` at `diff_diff/had.py:1624` when multiple first-treat cohorts are detected without `first_treat`. Library extension toward stricter safety: `UserWarning` would let the silent-misuse bug class through (HAD's Appendix B.2 only identifies the LAST cohort under staggered timing); fail-closed forces the user to either supply `first_treat` (which activates auto-filter to last-cohort + never-treated per Appendix B.2) or redirect to `ChaisemartinDHaultfoeuille` (`did_multiplegt_dyn`). Lock in `tests/test_methodology_had.py::TestHADDeviations`.
- **Note:** Extensive-margin / positive-untreated-mass fit-time warning (library convention). The paper (de Chaisemartin et al. 2026, Section 2 / Assumption 3) defines HAD for the case where no genuine untreated group exists and recommends (Section 4 practitioner checklist) that a user with a positive mass of untreated units consider a standard DiD instead — but it prescribes only "warn" with NO numeric cutoff, and explicitly RETAINS small untreated shares (the Garrett et al. bonus-depreciation application keeps 12 untreated counties out of 2,954 ≈ 0.4%, with simulations showing close-to-nominal coverage even at `f_{D_2}(0) = 0`). The library therefore emits a `UserWarning` at `HeterogeneousAdoptionDiD.fit()` time only when the fraction of units with EXACTLY-zero post-period dose is `>= 0.10` (`_HAD_EXTENSIVE_MARGIN_ZERO_DOSE_FRAC` in `diff_diff/had.py`) — a 10% library-convention cutoff chosen to sit ~25× above the paper's kept 0.4% example, so valid small-share fits are not nagged while a substantial untreated mass is flagged. **Overall path only:** the warning is emitted after the `aggregate="event_study"` dispatch returns, because the event-study path REQUIRES never-treated (zero-dose) units per Appendix B.2 (the last-cohort filter retains them), so an untreated mass is expected there, not a misuse signal. Surfaces the recommendation at fit time rather than only via `qug_test()`'s zero-dose `UserWarning` (which fires only when the user runs the pretests). Lock in `tests/test_methodology_had.py::TestHADDeviations::test_extensive_margin_warning_is_10pct_library_convention`.
- **Note:** `covariates=` is reserved but NOT implemented. `HeterogeneousAdoptionDiD.fit(covariates=...)` raises `NotImplementedError` — an explicit keyword-only param, so the message points to the deferred extension instead of letting an unknown kwarg surface as a bare `TypeError`. Covariate-adjusted HAD identification is the paper's Appendix B.1 / Theorem 6 multivariate-covariate extension (a multivariate nonparametric regression of ΔY on (D, X) at the dose boundary), which is not derived in the library. Workaround: pre-residualize the outcome on the covariates before calling `fit()`, or omit `covariates=` for the unconditional WAS estimand. Lock in `tests/test_methodology_had.py::TestHADDeviations::test_covariates_not_implemented_is_documented`.

**Requirements checklist (tracks implementation phase completion):**
- [x] Phase 1a: Epanechnikov / triangular / uniform kernels with closed-form `κ_k` constants (`diff_diff/local_linear.py`).
- [x] Phase 1a: Univariate local-linear regression at a boundary (`local_linear_fit` in `diff_diff/local_linear.py`).
- [x] Phase 1a: HC2 + Bell-McCaffrey DOF correction in `diff_diff/linalg.py` via `vcov_type="hc2_bm"` enum (both one-way and CR2 cluster-robust with Imbens-Kolesar / Pustejovsky-Tipton Satterthwaite DOF). **Weighted Bell-McCaffrey (`hc2_bm + weights`, both one-way and cluster) is now supported on the analytical linalg surface (`compute_robust_vcov` / `solve_ols` / `LinearRegression`) via the clubSandwich WLS-CR2 port.** The port matches clubSandwich's `pweight` (sampling-weight) convention only — `weight_type="aweight"` and `weight_type="fweight"` raise `NotImplementedError` on `hc2_bm + weights` (separate methodology task; CR1 / `vcov_type="hc1"` continues to support all three weight types). Estimator-level `survey_design=` paths continue to use the Taylor-series linearization (TSL) survey variance, which takes precedence over the analytical sandwich (no change in this PR). **Known precision limit on high-leverage FE-dummy coefficients**: the Satterthwaite DOF formula `(tr P)² / sum(P²)` is at the float64 noise floor for contrasts whose projection onto the design is essentially zero (typically FE-dummy coefficients where the dummy fires in a single cluster). In this regime, BLAS reduction-order differences between NumPy and R's BLAS produce 15-30% DOF discrepancies despite vcov matching at machine precision. `_cr2_bm_dof_inner_weighted` detects this regime via two criteria applied union-wise — **(a) batch-relative**: per-contrast max|P| below `1e-10 ×` the largest contrast's max|P| (catches noise-floor coefficients in a per-coefficient sweep where a non-noise reference is available); **(b) absolute single-contrast safe**: per-contrast max|P| below `(EPS × n × k × max(bread_inv_scale, 1))²` (catches single-contrast calls like MPD avg_att where no batch reference exists) — and returns NaN with a `UserWarning` rather than ship arbitrarily-different DOF on the user-facing surface. The coefficient SEs remain valid; only the DOF (and any t-test / CI that depends on it) is suppressed. The diff-diff implementation matches clubSandwich's specific algebra (R source: `CR-adjustments.R::CR2`, `clubSandwich.R::vcov_CR`, `coef_test.R::Satterthwaite_df`), **not** a textbook Pustejovsky-Tipton (2018) §3.3 transform-once derivation. clubSandwich uses W (not √W) in the hat matrix `H_gg = X_g M_U X_g' W_g`, W² in the bias-correction term `S_W = Σ_g X_g' W_g² X_g`, and unweighted residuals in the score construction `s_g = X_g' W_g A_g e_g`. The Satterthwaite DOF uses the full H1/H2/H3 array construction (`get_arrays.R::get_GH`), not the simplified `(tr B)² / tr(B²)` form (which works for the unweighted case but diverges from clubSandwich by 0.5-30% on weighted designs — see `feedback_wls_cr2_clubsandwich_parity`). Pinned at atol=1e-10 against `clubSandwich::vcovCR(..., type="CR2") + coef_test(test="Satterthwaite")$df_Satt` and `Wald_test(test="HTZ")$df_denom` on six weighted scenarios in `benchmarks/data/clubsandwich_cr2_golden.json` (vcov + non-noise-floor per-coefficient DOF + compound-contrast DOF; high-leverage FE-dummy coefficients are suppressed to NaN per the precision-limit note below) (`tests/test_methodology_wls_cr2.py`). Unweighted CR2-BM behavior is algebraically identical to prior (parity within floating-point tolerance after the 2026-07 scores-based evaluation change; regression-safe via `TestUnweightedRegressionStillBitEqual` + `TestDOFFormulaDualPathEquivalence`). The one-way weighted HC2-BM path uses clubSandwich's singleton-cluster CR2 reduction (each obs is its own cluster), routed via `_compute_bm_dof_from_contrasts` when `weights is not None`. clubSandwich pin: `≥ 0.7.0`.
    - **Note (scope limitation on absorbed FE):** HC2 and HC2 + Bell-McCaffrey on within-transformed designs still depend on the FULL FE hat matrix because FWL preserves coefficients and residuals but NOT the hat matrix: `h_ii = x_i' (X'X)^{-1} x_i` on the reduced design is not the diagonal of the full FE projection, and CR2's block adjustment `A_g = (I - H_gg)^{-1/2}` likewise depends on the full cluster-block hat matrix. The status across the three estimators that previously rejected this combination:
        - **`DifferenceInDifferences(absorb=..., vcov_type in {"hc2","hc2_bm"})` — SUPPORTED (auto-route).** When the user pairs `absorb=` with HC2 / HC2-BM, `DiD.fit()` internally promotes the absorb columns to `fixed_effects=` so the existing full-dummy code path computes the algebraically correct vcov from the full FE projection. Verified at ~1e-10 vs `lm() + sandwich::vcovHC(type="HC2")` and `lm() + clubSandwich::vcovCR(cluster=..., type="CR2")` (singleton-cluster CR2 trick for one-way HC2-BM Satterthwaite DOF; PT2018 §3.3 unweighted CR2 algebra). **User-visible surface change**: under the auto-route, the entire `DiDResults` (coefficients, vcov, residuals, fitted_values, r_squared) reflect the full-dummy fit rather than the within-transformed fit — the FE-dummy entries are included in `result.coefficients` / `result.vcov`, `r_squared` is computed on the un-demeaned outcome, and `residuals` / `fitted_values` are on the original scale. `result.att` is unaffected (FWL-equivalent). HC1/CR1 paths on `absorb=` are unchanged (no leverage term). **Survey-design scope**: when `survey_design=` is supplied, the existing survey variance path (Taylor-series linearization / replicate weights) takes precedence over the analytical HC2/HC2-BM sandwich; the auto-route only changes the FE handling (removing the prior reject) and does not redirect to the analytical small-sample sandwich on survey fits.
        - **`TwoWayFixedEffects(vcov_type in {"hc2","hc2_bm"})` — SUPPORTED (inline full-dummy build).** TWFE has no `absorb=` / `fixed_effects=` parameter (the unit + time FE are baked into the estimator's identity), so the same parameter-swap auto-route used for DiD-absorb / MPD-absorb is not directly applicable. Instead, `TwoWayFixedEffects.fit()` bypasses the within-transform when `vcov_type in {"hc2","hc2_bm"}` and builds the full-dummy design `[intercept, treated×post, covariates, factor(unit), factor(time)]` explicitly, then runs OLS through the standard `solve_ols` path so the leverage correction and BM DOF compute on the full FE projection. Verified at atol=1e-10 vs `lm(y ~ treat_post + factor(unit) + factor(post)) + sandwich::vcovHC(type="HC2")` for HC2 and vs `clubSandwich::vcovCR(cluster=seq_len(n), type="CR2")` for the singleton-cluster one-way HC2-BM Satterthwaite DOF; vs `vcovCR(cluster=unit, type="CR2")` for the auto-cluster CR2-BM path. **Auto-cluster default (non-survey analytical path):** TWFE's unit auto-cluster is preserved on `hc2_bm` (routes to CR2-BM at unit) and on `hc2 + wild_bootstrap` (the bootstrap consumes the cluster structure for resampling regardless of the analytical sandwich choice); dropped on explicit `hc2 + analytical` to match the one-way contract (the linalg validator rejects `hc2 + cluster_ids`). `hc2_bm + analytical` with no explicit cluster yields the auto-cluster CR2-BM path. **Survey-design exception:** under `survey_design=` with no explicit `cluster=`, TWFE intentionally keeps the documented implicit-PSU path (the auto-cluster is NOT injected into the survey PSU structure); users who want unit-level PSU injection under a survey design must pass explicit `cluster="unit"` or set `survey_design.psu` directly. **User-visible surface change** (matches the DiD-absorb / MPD-absorb disclosure above): under HC2 / HC2-BM, `result.coefficients`, `result.vcov`, `result.residuals`, `result.fitted_values`, and `result.r_squared` reflect the full-dummy fit rather than the within-transformed reduced fit (FE-dummy entries are included, `r_squared` is computed on the un-demeaned outcome, residuals/fitted are on the original scale). `result.att`, its SE, and analytical inference are unchanged (FWL-equivalent). HC1 / CR1 / Conley / classical paths remain on the within-transform (no leverage term in those vcov families). **Survey-design scope** (mirrors the DiD-absorb auto-route contract above): when `survey_design=` is supplied, the existing survey variance path (Taylor-series linearization for analytical-weight designs, or replicate-weight variance for BRR/Fay/JK1/JKn/SDR) takes precedence over the analytical HC2/HC2-BM sandwich; the full-dummy build only changes the FE handling (removing the prior reject) and does not redirect to the analytical small-sample sandwich on survey fits. **Replicate-weight survey designs** are blocked at the estimator level: `vcov_type in {"hc2","hc2_bm"}` + replicate weights raises `NotImplementedError` because the replicate refit path re-demeans per replicate, which doesn't compose with the full-dummy build (would require per-replicate full-dummy refit); workaround: use `vcov_type="hc1"` for replicate-weight CR1. `hc2_bm + weights` (Gates 4-5) is now lifted via the clubSandwich WLS-CR2 port; the survey-design rejection here is a separate estimator-level gate (analytical sandwich vs survey TSL precedence), independent of the linalg validator.
        - **`MultiPeriodDiD(absorb=..., vcov_type in {"hc2","hc2_bm"})` — SUPPORTED (auto-route).** Same auto-route pattern as `DifferenceInDifferences`: `MultiPeriodDiD.fit()` internally promotes the absorb columns to `fixed_effects=` for HC2 / HC2-BM callers, so the existing full-dummy code path computes the algebraically correct vcov from the full FE projection on the event-study design (`treated + period_X dummies + treated:period_X interactions + factor(unit)`). Verified at ~1e-10 vs `lm() + sandwich::vcovHC(type="HC2")` and `lm() + clubSandwich::vcovCR(cluster=1:n, type="CR2")` on a 5-cohort × 5-period event-study fixture; the parity target is a per-period interaction `treated:period_X` because MPD requires the `treated` column to be a time-invariant ever-treated indicator, which lies in the span of the intercept and the post-auto-route unit FE dummies (under `pd.get_dummies(..., drop_first=True)` the dropped reference unit is implicit in the intercept, so the exact alias relation depends on the omitted FE category — it is NOT simply "the sum of treated-cohort unit dummies"). `solve_ols` drops one column from the collinear set under R-style rank-deficiency handling; in the shipped parity fixture (4 ever-treated cohorts of 5 units + 1 never-treated cohort of 5 units) it drops a unit dummy from the never-treated cohort (`unit_25`) and the `treated` main effect remains finite, but the specific column that gets NaN'd is pivot-order and dummy-coding dependent. Either way, the slope coefficients (`treated:period_X`) and the post-period-average `avg_att` are identified and invariant to which column was dropped. Same `MultiPeriodDiDResults` surface change as DiD: `vcov`, `residuals`, `fitted_values`, `r_squared`, and `coefficients` reflect the full-dummy fit, with `period_effects[t].effect` / `.se` / `.p_value` / `.conf_int` invariant by FWL. HC1/CR1 paths on `absorb=` are unchanged (no leverage term). Same survey-design scope as DiD: replicate-weight variance routes through the standard `compute_replicate_vcov` path on the fixed full-dummy design rather than the per-replicate refit branch (which targets the demeaning path); since the auto-routed design does not depend on replicate weights, no refit is needed. **Redundant time-FE skip:** when the routed (or directly-supplied) `fixed_effects` list contains the `time` column, MPD silently skips emitting `<time>_<X>` dummies for that entry because the design already absorbs time via the non-reference period dummies. Without the skip, those blocks collide on dummy names and `MultiPeriodDiDResults.coefficients` (built as `{name: coef for name, coef in zip(var_names, coefficients)}`) would silently drop duplicates, breaking the coefficients-vs-vcov alignment that downstream consumers (HonestDiD sub-VCV extraction, BusinessReport, etc.) rely on. The skip applies to BOTH the new `absorb=` auto-route AND the pre-existing `fixed_effects=[<time_col>]` invocation (pre-PR, `fixed_effects=["unit", time]` produced a dict with `len < vcov.shape[0]` and NaN values overwriting the real event-study period coefficients).
        - All three previously-rejecting absorbed-FE paths are now SUPPORTED. **Weighted-CR2 variants (Gates 4-5: `vcov_type="hc2_bm" + weights`; weighted one-way HC2-BM) are now LIFTED via the clubSandwich WLS-CR2 port** documented above on the Phase 1a row. The lift applies to the **analytical** surface: `compute_robust_vcov`, `solve_ols`, and `LinearRegression` direct callers. The helper `_compute_cr2_bm_contrast_dof(weights=...)` is also helper-ready for the contrast-DOF case (e.g., MPD `avg_att`), but no public estimator API currently passes non-None weights to that helper — `MultiPeriodDiD.fit()` has no non-survey weighted entry point, and `survey_design=` paths route through TSL (which takes precedence over the analytical CR2 sandwich). Estimator-level weighted CR2-BM via the public estimator surface would require a separate non-survey weights entry point and is out of scope here.
- [x] Phase 1a: `vcov_type` enum threaded through `DifferenceInDifferences` (`MultiPeriodDiD`, `TwoWayFixedEffects` inherit); `robust=True` <=> `vcov_type="hc1"`, `robust=False` <=> `vcov_type="classical"`. Conflict detection at `__init__`. Results summary prints the variance-family label.
    - **Note (`MultiPeriodDiD(cluster=..., vcov_type="hc2_bm")` — SUPPORTED via cluster-aware contrast DOF):** the scalar-coefficient `DifferenceInDifferences` path uses `_compute_cr2_bm`'s per-coefficient Satterthwaite DOF directly for the single-ATT contrast, but `MultiPeriodDiD` also reports a post-period-average ATT constructed as a *contrast* of the event-study coefficients (`avg_att = (1/n_post) Σ_{t ≥ t_treat} β_t`). Pre-PR the combination raised `NotImplementedError` because the cluster-aware CR2 Bell-McCaffrey Satterthwaite DOF for an arbitrary linear combination was not implemented — only the per-coefficient case existed. The new `_compute_cr2_bm_contrast_dof` helper in `diff_diff/linalg.py` generalizes the per-coefficient loop to arbitrary `(k, m)` contrast matrices using the identical Pustejovsky-Tipton 2018 Section 4 algebra (`q = X bread_inv c`, `omega_g = A_g X_g bread_inv c`, `DOF = trace(B)² / trace(B²)`), and `_compute_cr2_bm` is refactored to call it with `contrasts=eye(k)` so the per-coefficient case is recovered at machine precision (atol=1e-10, see refactor regression in `tests/test_linalg_hc2_bm.py::TestCR2BMContrastDOF`). `MultiPeriodDiD.fit()` extends its existing avg_att DOF block to branch on cluster presence: one-way `_compute_bm_dof_from_contrasts` for `effective_cluster_ids is None`, cluster-aware `_compute_cr2_bm_contrast_dof` otherwise. R parity verified against clubSandwich's `Wald_test(constraints=matrix(c, 1), test="HTZ")$df_denom` at atol=1e-10 on the `mpd_clustered_avg_att_dof` fixture in `benchmarks/data/clubsandwich_cr2_golden.json` (Wald_test's HTZ on a 1-row constraint matrix yields the Satterthwaite t-test DOF). Cluster IDs are per-observation length `n` and are NOT subscripted by the rank-deficient column-drop mask `_kept` — the helper accepts the full `effective_cluster_ids` array. Weighted CR2-BM (`survey_design=` paths) remains a separate gate.
- [x] Phase 1a: `clubSandwich::vcovCR(..., type="CR2")` parity harness committed: R script at `benchmarks/R/generate_clubsandwich_golden.R` plus the authoritative R-generated JSON at `benchmarks/data/clubsandwich_cr2_golden.json` (`"source": "clubSandwich"`, with `clubSandwich_version`, `R_version`, and `generated_at` captured in `meta` for forensic traceability). The parity test at `tests/test_linalg_hc2_bm.py::TestCR2BMCluster::test_cr2_parity_with_golden` runs at 1e-6 tolerance and passes at ≤ 7.1e-15 across all three datasets — Python's `_compute_cr2_bm` matches clubSandwich at machine precision.
- [x] Phase 1b: Calonico-Cattaneo-Farrell (2018) MSE-optimal bandwidth selector. In-house port of `nprobust::lpbwselect(bwselect="mse-dpi")` (nprobust 0.5.0, SHA `36e4e53`) as `diff_diff.mse_optimal_bandwidth` and `BandwidthResult`, backed by the private `diff_diff._nprobust_port` module (`kernel_W`, `lprobust_bw`, `lpbwselect_mse_dpi`). Three-stage DPI with four `lprobust.bw` calls at orders `q+1`, `q+2`, `q`, `p`. Python matches R to `0.0000%` relative error (i.e., bit-parity within float64 precision, ~8-13 digits agreement) on all five stage bandwidths (`c_bw`, `bw_mp2`, `bw_mp3`, `b_mse`, `h_mse`) across three deterministic DGPs (uniform, Beta(2,2), half-normal) via `benchmarks/R/generate_nprobust_golden.R` → `benchmarks/data/nprobust_mse_dpi_golden.json`. **Note:** `weights=` is currently unsupported (raises `NotImplementedError`); nprobust's `lpbwselect` has no weight argument so there is no parity anchor. Weighted-data support deferred to Phase 2 (survey-design adaptation). **Note (public API scope restriction):** the exported wrapper `mse_optimal_bandwidth` hard-codes the HAD Phase 1b configuration (`p=1`, `deriv=0`, `interior=False`, `vce="nn"`, `nnmatch=3`). The underlying port supports a broader surface (`hc0`/`hc1`/`hc2`/`hc3` variance, interior evaluation, higher `p`), but those paths are not parity-tested against `nprobust` and are deferred. Callers needing the broader surface should use `diff_diff._nprobust_port.lpbwselect_mse_dpi` directly and accept that parity has not been verified on non-HAD configurations. **Note (input contract):** the wrapper enforces HAD's support restriction `D_{g,2} >= 0` (front-door `ValueError` on negative doses and empty inputs). `boundary` must equal `0` (Design 1') or `float(d.min())` (Design 1 continuous-near-d_lower) within float tolerance; off-support values raise `ValueError`. When `boundary ~ 0`, the wrapper additionally requires `d.min() <= 0.05 * median(|d|)` as a Design 1' support plausibility heuristic, chosen to pass the paper's thin-boundary-density DGPs (Beta(2,2), d.min/median ~ 3%) while rejecting substantially off-support samples (U(0.5, 1.0), d.min/median ~ 1.0). Detected mass-point designs (`d.min() > 0` with modal fraction at `d.min() > 2%`) raise `NotImplementedError` pointing to the Phase 2 2SLS path per paper Section 3.2.4.
- [x] Phase 1c: First-order bias estimator `M̂_{ĥ*_G}` and robust variance `V̂_{ĥ*_G}`. Implemented via Calonico-Cattaneo-Titiunik (2014) bias-combined design matrix `Q.q` in the in-house port `diff_diff._nprobust_port.lprobust` (single-eval-point path of `nprobust::lprobust`, npfunctions.R:177-246).
- [x] Phase 1c: Bias-corrected CI (Equation 8) with `nprobust` parity. Public wrapper `diff_diff.bias_corrected_local_linear` returns `BiasCorrectedFit` with μ̂-scale point estimate, robust SE, and bias-corrected 95% CI `[tau.bc ± z_{1-α/2} * se.rb]`. The β-scale rescaling from Equation 8, `(1/G) Σ D_{g,2}`, is applied by Phase 2's `HeterogeneousAdoptionDiD.fit()`. Parity against `nprobust::lprobust(..., bwselect="mse-dpi")` is asserted at `atol=1e-12` on `tau_cl`/`tau_bc`/`se_cl`/`se_rb`/`ci_low`/`ci_high` across the three unclustered golden DGPs (DGP 1 and DGP 3 typically land closer to `1e-13`). The Python wrapper computes its own `z_{1-α/2}` via `scipy.stats.norm.ppf` inside `safe_inference()`; R's `qnorm` value is stored in the golden JSON for audit, and the parity harness compares Python's CI bounds to R's pre-computed CI bounds so any residual drift is purely the floating-point arithmetic in `tau.bc ± z * se.rb`, not a critical-value disagreement. The clustered DGP achieves bit-parity (`atol=1e-14`) when cluster IDs are in first-appearance order; otherwise BLAS reduction ordering can drift to `atol=1e-10`. Generator: `benchmarks/R/generate_nprobust_lprobust_golden.R`. **Note:** The wrapper matches nprobust's `rho=1` default (`b = h` in auto mode), so Phase 1b's separately-computed `b_mse` is surfaced via `bandwidth_diagnostics.b_mse` but not applied. **Note (public-API surface restriction):** Phase 1c restricts the public wrapper's `vce` parameter to `"nn"`; hc0/hc1/hc2/hc3 raise `NotImplementedError` and are queued for Phase 2+ pending dedicated R parity goldens. The port-level `diff_diff._nprobust_port.lprobust` still accepts all five vce modes (matching R's `nprobust::lprobust` signature) for callers who need the broader surface and accept that the hc-mode variance path — which reuses p-fit hat-matrix leverage for the q-fit residual in R (lprobust.R:229-241) — has not been separately parity-tested. **Note (Phase 1c internal bug workaround):** The clustered golden DGP 4 uses manual `h=b=0.3` to sidestep an nprobust-internal singleton-cluster shape bug in `lprobust.vce` fired by the mse-dpi pilot fits; the Python port has no equivalent bug.
- [x] Phase 2a: `HeterogeneousAdoptionDiD` class with separate code paths for Design 1' (`continuous_at_zero`), Design 1 continuous-near-`d̲` (`continuous_near_d_lower`), and Design 1 mass-point. Continuous paths compose Phase 1c's `bias_corrected_local_linear` and form the beta-scale WAS estimate `β̂ = (mean(ΔY) - τ̂_bc) / den` where `τ̂_bc` is the bias-corrected local-linear estimate of the boundary limit `lim_{d↓d̲} E[ΔY | D_2 ≤ d]` and `den = E[D_2]` for Design 1' (paper Theorem 1 / Equation 3 identification; Equation 7 sample estimator) or `den = E[D_2 - d̲]` for Design 1 (paper Theorem 3 / Equation 11, `WAS_{d̲}` under Assumption 6). Mass-point path uses a sample-average 2SLS estimator with instrument `1{D_{g,2} > d̲}` (paper Section 3.2.4).
- [x] Phase 2a: `design="auto"` detection rule (`min_g D_{g,2} < 0.01 · median_g D_{g,2}` → continuous_at_zero; modal-min fraction > 2% → mass_point; else continuous_near_lower). Implemented as strict first-match in `diff_diff.had._detect_design`; when `d.min() == 0` exactly, resolves `continuous_at_zero` unconditionally (modal-min check runs only when `d.min() > 0`). Edge case covered: 3% at `D=0` + 97% `Uniform(0.5, 1)` resolves to `continuous_at_zero`, matching the paper-endorsed Design 1' handling of small-share-of-treated samples.
- [x] Phase 2a: Panel validator (`diff_diff.had._validate_had_panel`) verifies `D_{g,1} = 0` for all units, rejects negative post-period doses (`D_{g,2} < 0`) front-door on the original (unshifted) scale, rejects `>2` time periods on the `aggregate="overall"` path (multi-period panels must use `aggregate="event_study"`, Phase 2b), and rejects unbalanced panels and NaN in outcome/dose/unit columns. Both Design 1 paths (`continuous_near_d_lower` and `mass_point`) additionally require `d_lower == float(d.min())` within float tolerance; mismatched overrides raise with a pointer to the unsupported (LATE-like / off-support) estimand.
- [x] Phase 2a: NaN-propagation tests covering constant-y, degenerate-mass-point, and single-cluster-CR1 inputs. The guaranteed NaN coupling is on the DOWNSTREAM triple (`t_stat`, `p_value`, `conf_int`) via the `safe_inference()` gate, which returns NaN on all three whenever `se` is non-finite, zero, or negative. `att` and `se` themselves are raw estimator outputs: on constant-y / no-dose-variation / divide-by-zero the fit paths return `(att=nan, se=nan)` so all five fields move to NaN together; on the degenerate single-cluster CR1 configuration on the mass-point path, `_fit_mass_point_2sls` returns `(att=beta_hat, se=nan)` - `att` is finite (Wald-IV is well defined) while `se` is NaN, so the downstream triple is NaN while `att` remains the raw 2SLS coefficient. The `assert_nan_inference` fixture in `tests/conftest.py` checks the downstream triple against this contract without requiring `att` to be NaN.
    - **Note (mass-point SE):** Standard errors on the mass-point path use the structural-residual 2SLS sandwich `[Z'X]^{-1} · Ω · [Z'X]^{-T}` with `Ω` built from the structural residuals `u = ΔY - α̂ - β̂·D` (not the reduced-form residuals from an OLS-on-indicator shortcut). Supported: `classical`, `hc1`, and CR1 (cluster-robust) when `cluster=` is supplied. `hc2` and `hc2_bm` raise `NotImplementedError` pending a 2SLS-specific leverage derivation (the OLS leverage `x_i' (X'X)^{-1} x_i` is wrong for 2SLS; the correct finite-sample correction depends on `(Z'X)^{-1}` rather than `(X'X)^{-1}`) plus a dedicated R parity anchor. Queued for the follow-up PR.
    - **Note (continuous cluster-robust SE, Phase 2a):** On the continuous designs (`continuous_at_zero` / `continuous_near_d_lower`), `cluster=` threads the per-unit cluster IDs into `bias_corrected_local_linear`, whose CCT-2014 robust variance then becomes the cluster-robust nonparametric SE; the β̂-scale SE is `se_robust / |den|`. Because the β-scale rescale is a deterministic linear transform, the estimator-level clustered SE equals the direct `bias_corrected_local_linear(cluster=...).se_robust / |den|` to machine precision — this is the in-library validation anchor, and the clustered CCT SE is itself golden-tested against `nprobust` on DGP 4 (see the Phase 1c note above: `atol=1e-14` in first-appearance cluster order). Cluster IDs must be unit-constant (validated up front; a nonexistent column, NaN, or within-unit-varying cluster now raises rather than being silently ignored, mirroring the mass-point path). Cluster-robust inference is **unidentified with fewer than two clusters in the active kernel window** (`eC = cluster[ind]`, the in-bandwidth subset the CCT variance is actually computed on — a stricter condition than the global cluster count, since clusters can be separated from the boundary by the bandwidth): the guard lives in `_nprobust_port.lprobust` and NaNs `se_rb`/`se_cl`, so `bias_corrected_local_linear.se_robust` is NaN and HAD's `safe_inference` NaNs the t-stat / p-value / CI while the point estimate stays finite — the same single-cluster NaN contract as the mass-point CR1 path (`_fit_mass_point_2sls`), applied at the variance-computation site so it also covers the direct `bias_corrected_local_linear` API. A window with ≥2 distinct clusters is bit-identical (DGP-4 golden parity preserved). A bare `cluster=` gives unweighted cluster-robust inference; the `cluster=` + `survey_design=` composition raises `NotImplementedError`: the Binder (1983) TSL survey variance is composed from the per-unit influence function via `compute_survey_if_variance` and would silently override the cluster-robust SE — for weighted clustering route through `survey_design=SurveyDesign(weights='<weight_col>', psu='<cluster_col>')` instead. Result metadata reports `vcov_type="cr1"` + `cluster_name=<col>` with `inference_method="analytical_nonparametric"` (distinguishing the clustered CCT variance from the mass-point 2SLS CR1 sandwich). Estimator-level cluster threading also extends to the **Phase 2b event-study** path (cluster-robust per-horizon pointwise CIs on both designs plus a cluster-robust simultaneous sup-t band; the per-horizon variance-family reconciliation is documented in "Note (HAD clustered event-study sup-t band)" above) — the former "cluster ignored" `UserWarning` is removed.
    - **Note (Design 1 identification):** `continuous_near_d_lower` and `mass_point` fits emit a `UserWarning` surfacing that `WAS_{d̲}` identification requires Assumption 6 (or Assumption 5 for sign identification only) beyond parallel trends, and that neither is testable via pre-trends. `continuous_at_zero` (Design 1', Assumption 3 only) does not emit this warning.
    - **Note (CI endpoints):** Because the continuous-path `att` is `(mean(ΔY) - τ̂_bc) / den`, the beta-scale CI endpoints reverse relative to the Phase 1c boundary-limit CI: `CI_lower(β̂) = (mean(ΔY) - CI_upper(τ̂_bc)) / den` and `CI_upper(β̂) = (mean(ΔY) - CI_lower(τ̂_bc)) / den`. The `HeterogeneousAdoptionDiD.fit()` implementation computes `att ± z · se` directly via `safe_inference`, which handles the reversal naturally from the transformed point estimate.
    - **Note (Phase 2a/2b scope, superseded by Phase 4.5):** Phase 2a ships the single-period `aggregate="overall"` path; Phase 2b lifts `aggregate="event_study"` (Appendix B.2 multi-period extension) which returns a `HeterogeneousAdoptionDiDEventStudyResults` with per-event-time WAS estimates and pointwise CIs. The original Phase 2a/2b release raised `NotImplementedError` on all weighting; Phase 4.5 (A/B/C0) lifted that with the per-design vcov contract documented above (see L2340-L2379, including the mass-point `vcov_type="classical"` deviation and `cband=True` sup-t restriction); the `survey_design=` path composes Binder (1983) TSL via `compute_survey_if_variance` on both continuous and mass-point IFs.
    - **Note (panel-only):** The paper (Section 2) defines HAD on *panel or repeated cross-section* data, but both the overall and event-study paths ship a panel-only implementation: `HeterogeneousAdoptionDiD.fit()` requires a balanced panel with a unit identifier so that unit-level first differences `ΔY_{g,t} = Y_{g,t} - Y_{g,t_anchor}` can be formed. Repeated-cross-section inputs (disjoint unit IDs between periods) are rejected by the balanced-panel validator. RCS support is queued for a follow-up PR (tracked in `DEFERRED.md`); it will need a separate identification path based on pre/post cell means rather than unit-level differences.
- [x] Phase 2b: Multi-period event-study extension (Appendix B.2). `aggregate="event_study"` produces per-event-time WAS estimates using a uniform `F-1` baseline (`ΔY_{g,t} = Y_{g,t} - Y_{g,F-1}` for every horizon), reusing the three Phase 2a design paths on per-horizon first differences. Pre-period placebos included for `e <= -2` (the anchor `e = -1` is skipped since `ΔY = 0` trivially). Post-period estimates for `e >= 0`. The joint Stute test (Equation 18) across pre-periods is a SEPARATE diagnostic deferred to a **Phase 3 follow-up patch** (initially deferred; shipped in Phase 3 via `stute_joint_pretest()` — see the Phase 3 entries below).
- [x] Phase 2b: event-study `cluster=` threading — cluster-robust per-horizon pointwise CIs (both designs) AND a cluster-robust simultaneous band via the clustered branch of `_sup_t_multiplier_bootstrap` (continuous scale 1.0; mass-point `√(G/(G-1))`). Closes the former "cluster ignored on the nonparametric path" deferral (`TODO.md`). See "Note (HAD clustered event-study sup-t band)". `cluster=` + `survey_design=` remains rejected (for weighted clustering route through `survey_design=SurveyDesign(weights=..., psu=...)`).
    - **Note (Phase 2b last-cohort filter):** When `first_treat` indicates more than one nonzero cohort, the panel is auto-filtered to the last-treatment cohort (`F_last = max(cohorts)`) **plus never-treated units** (`first_treat = 0`), with a `UserWarning` naming kept/dropped unit counts and dropped cohort labels. Paper Appendix B.2 is explicit that HAD "may be used only for the LAST treatment cohort in a staggered design"; the auto-filter implements this prescription, retaining never-treated units per the paper's "there must be an untreated group, at least till the period where the last cohort gets treated" requirement. Only earlier-cohort units (with `first_treat > 0` and `< F_last`) are dropped — never-treated units satisfy the dose invariant at every period (`D = 0` throughout) and preserve Design 1' identifiability (boundary at `0`) when last-cohort doses are uniformly positive. When `first_treat` is omitted on a >2-period panel, the validator infers each unit's first-positive-dose period from the dose path; if multiple distinct first-positive-dose cohorts are detected, the estimator raises a front-door `ValueError` directing users to pass `first_treat` (which activates the auto-filter) or use `ChaisemartinDHaultfoeuille` for full staggered support — there is no silent acceptance of staggered panels without cohort metadata. Common-adoption panels (single first-positive-dose cohort, or only never-treated + one cohort) pass through unchanged with `F` inferred from the dose invariant, and require dose contiguity (pre-periods < post-periods in natural ordering). Non-contiguous dose sequences (e.g., reverse treatment) raise with a pointer to `ChaisemartinDHaultfoeuille`.
    - **Note (Phase 2b constant-dose requirement):** The event-study aggregation uses `D_{g, F}` (first-treatment-period dose) as the single regressor for every event-time horizon, per paper Appendix B.2's "once treated, stay treated with the same dose" convention. The validator REJECTS panels where a unit has time-varying dose across post-treatment periods (`D_{g, t} != D_{g, F}` for any `t >= F` within-unit, beyond float tolerance) with a front-door `ValueError`, directing users with genuinely time-varying post-treatment doses to `ChaisemartinDHaultfoeuille` (`did_multiplegt_dyn`). Silent acceptance would misattribute later-horizon treatment-effect heterogeneity to the period-F dose. A follow-up PR could implement a time-varying-dose estimator; tracked in `DEFERRED.md`.
    - **Note (Phase 2b per-horizon SE):** Each event-time horizon uses an INDEPENDENT sandwich computed on that horizon's first differences: continuous paths use the CCT-2014 robust SE from Phase 1c divided by `|den|`; mass-point path uses the structural-residual 2SLS sandwich from Phase 2a. This produces pointwise CIs per horizon, matching the paper's Pierce-Schott application (Section 5.2, Figure 2: "nonparametric pointwise CIs"). Joint cross-horizon covariance (IF-based stacking or block bootstrap) is NOT computed — the paper does not derive it and all reported CIs are pointwise. Follow-up PRs may add joint covariance for cross-horizon hypothesis tests; current tracking in `DEFERRED.md`.
    - **Note (Phase 2b baseline convention):** All event-time horizons use a uniform `F-1` anchor: `ΔY_{g,t} = Y_{g,t} - Y_{g,F-1}` for every `t`. This is consistent with the paper's Garrett-et-al. application (Section 5.1: "outcome `Y_{g,t} - Y_{g,2001}`" where `F = 2002`), simplifies event-time indexing (`e = t - F` so `e = -1` is the anchor, skipped), and keeps the implementation symmetric for pre- and post-period horizons. The paper review text's asymmetric "`Y_{g,t} - Y_{g,1}` for pre" / "`Y_{g,t} - Y_{g,F-1}` for post" phrasing is covered by the uniform convention since both give the same placebo interpretation under parallel trends (the paper's own applications use the uniform anchor).
    - **Note (Phase 2b result class):** `aggregate="event_study"` returns a new `HeterogeneousAdoptionDiDEventStudyResults` dataclass (distinct from the single-period `HeterogeneousAdoptionDiDResults`) with per-horizon arrays (`event_times`, `att`, `se`, `t_stat`, `p_value`, `conf_int_low`, `conf_int_high`, `n_obs_per_horizon`) and shared metadata. `to_dataframe()` returns a tidy per-horizon DataFrame; `to_dict()` returns a dict with list-of-per-horizon fields. The static return-type annotation on `fit()` is `Union[HeterogeneousAdoptionDiDResults, HeterogeneousAdoptionDiDEventStudyResults]`, matching the runtime polymorphism on `aggregate`; callers should narrow via `isinstance` (or via the aggregate they passed) when reading aggregate-specific fields.
    - **Note (Phase 3 pretest workflow):** Phase 3 ships the Section 4 pre-test diagnostics in a new module `diff_diff/had_pretests.py` (separate from the 2,800-line `had.py` estimator module). The three tests — `qug_test`, `stute_test`, `yatchew_hr_test` — accept raw `(d, dy)` arrays and return their own result dataclasses (`QUGTestResults`, `StuteTestResults`, `YatchewTestResults`); the composite `did_had_pretest_workflow` dispatches on `aggregate` (`"overall"` requires a balanced two-period panel; `"event_study"` accepts a multi-period panel with >= 3 periods) and returns `HADPretestReport` with a priority-ordered verdict string. The `data`-only workflow signature (no `result=`-consuming variant) avoids coupling pretests to the `BiasCorrectedFit` internal state, which does not expose residuals. Below-minimum sample sizes emit `UserWarning` and return all-NaN inference (no `ValueError` raise) for library consistency with the `safe_inference` convention; input-shape violations (non-1D, NaN-containing, mismatched length) still raise. No multiple-testing adjustment (Bonferroni/Holm) is applied — the paper does not prescribe one. **Partial-workflow semantic (path-dependent):** `aggregate="overall"` (default; two-period panel) runs paper steps 1 (QUG) + 3 (linearity via Stute + Yatchew-HR) ONLY; step 2 (Assumption 7 pre-trends test via Equation 18) is NOT covered on this path and a fail-to-reject verdict appends an explicit Assumption 7 gap flag (`"QUG and linearity diagnostics fail-to-reject; Assumption 7 pre-trends test NOT run (paper step 2 deferred)"`). `aggregate="event_study"` (multi-period panel, requires >=3 periods + earlier pre-period for joint pretrends) closes step 2 via joint Stute pre-trends (Equation 18) over pre-period horizons AND joint Stute homogeneity over post-period horizons; the verdict on a fail-to-reject across QUG + joint pretrends + joint homogeneity reads "TWFE admissible under Section 4 assumptions" without the Assumption 7 caveat. The `HADPretestReport` docstring and structural fields (`pretrends_joint`, `homogeneity_joint`) reflect the aggregate-dependent coverage.
    - **Note (Phase 3 Stute bootstrap):** The Stute CvM statistic is implemented via the simplified-form `S = (1/G²) · Σ cumsum²`, which is algebraically equivalent to the paper's stated `Σ(g/G)² · ((1/g) Σ eps_{(h)})²` form. Bootstrap follows the paper's Appendix D Algorithm literally: each iteration refits OLS on `dy_b = a_hat + b_hat · d + eps · eta` (null DGP multiplied by Mammen weights). The ~2× performance follow-up landed 2026-07-07 as **loop-invariant hoisting**, not the once-sketched `M = I - X(X'X)⁻¹X'` materialization (O(G²) memory — would not scale): Mammen draws are batched through memory-bounded `rng.choice` calls whose variate stream is identical to per-iteration draws, and the d-moments + CvM tie-block indices are precomputed once per test — each replicate still runs the paper-literal O(G) refit on the same values, so `bootstrap_S`/`cvm_stat`/`p_value` are bit-identical to the per-iteration form (frozen byte-copy parity test at `tests/test_had_pretests.py::TestStuteBootstrapHoistedInvariants`). Bootstrap reproducibility uses `np.random.default_rng(seed)` (library convention, matching `wild_bootstrap_se`); `seed=42` in tests for bitwise stability.
    - **Note (Phase 3 Yatchew normalizer):** `σ̂²_diff` in `yatchew_hr_test` divides by `2G` (paper-literal from Theorem 7), NOT by `2(G-1)`. A unit test hand-computes `σ̂²_diff` at `G=4` on deterministic inputs and asserts the `2G`-normalizer form at `atol=1e-12` so any later regression to the (asymptotically equivalent but finite-sample different) `2(G-1)` form would fail the test. `σ̂⁴_W` uses `np.mean(eps_{(g)}² · eps_{(g-1)}²)` which divides by `G-1` via `np.mean` length, matching the paper's Theorem 7 / Equation 29 normalization.
    - **Note (Phase 3 Yatchew tie policy):** `yatchew_hr_test` REJECTS duplicate dose values with a `UserWarning` + all-NaN result at the front door. The difference-based variance estimator `σ̂²_diff` and the heteroskedasticity-robust scale `σ̂⁴_W` both use adjacent differences (of sorted dy and of adjacent squared residuals, respectively); under tied doses the within-tie row ordering is arbitrary (stable sort falls back to input order), so the statistic becomes non-methodological and order-dependent. Callers with tied doses (mass-point designs, discretised dose registers) should use `stute_test` instead — its tie-safe Cramer-von Mises statistic collapses tie blocks to the post-tie cumulative sum and is provably order-invariant under within-tie permutations. A regression test on `stute_test` asserts bit-identical `cvm_stat` across tied-dose permutations at `atol=1e-14`; a matching regression on `yatchew_hr_test` asserts the NaN+warning behavior on duplicated and constant doses. This tie policy is a Phase 3 diff-diff choice (the paper describes Yatchew only as "sort by D" and does not specify a tie rule); an alternative implementation could document a justified tie-resolution convention and back it with permutation tests, but at the cost of a methodology surface the paper does not cover. **Workflow fallback:** `did_had_pretest_workflow` follows the paper's "Stute OR Yatchew" step-3 wording — a conclusive Stute (which is tie-safe) is sufficient to adjudicate linearity even when Yatchew returns NaN on tied-dose panels. The composite verdict then reads "QUG and linearity diagnostics fail-to-reject (Yatchew NaN - skipped); ..." rather than forcing the whole report to "inconclusive", and `all_pass` is `True` when QUG + at least one conclusive linearity test fail-to-reject.
    - **Note (Phase 3 exact-linear short-circuit):** Both `stute_test` and `yatchew_hr_test` detect numerically-exact linear fits (OLS residuals below machine precision relative to the signal) and short-circuit to `p=1.0, reject=False` without running the bootstrap / computing `T_hr`. The detection compares `sum(eps^2)` against the CENTERED total sum of squares `sum((dy - dybar)^2)` (ratio `<= 1e-24`), which is equivalent to `1 - R^2` and is TRANSLATION-INVARIANT under additive shifts in dy. Comparing against uncentered `sum(dy^2)` would NOT be translation-invariant: adding a large constant to dy inflates `sum(dy^2)` and can spuriously trip the short-circuit on genuinely noisy data. Regression tests pin (a) the exact-linear fail-to-reject behavior for `dy = a + b*d`, (b) translation-invariance under `dy -> dy + 1e12` (the short-circuit must not fire on noisy data regardless of the constant offset).
- [x] Phase 3: `qug_test()` (`T = D_{2,(1)} / (D_{2,(2)} - D_{2,(1)})`, rejection `{T > 1/α - 1}`). One-sided asymptotic p-value `1 / (1 + T)` under the Exp(1)/Exp(1) limit law (Theorem 4). Zero-dose observations filtered upfront (with `UserWarning`); `D_{(1)} == D_{(2)}` tie returns all-NaN inference (conservative). Closed-form tight parity tested at `atol=1e-12`.
- [x] Phase 3: `stute_test()` Cramér-von Mises with Mammen wild bootstrap. Statistic `S = (1/G^2) Σ (cumsum_g)^2` (algebraically equivalent to paper's `Σ(g/G)^2 · ((1/g) Σ eps_{(h)})^2`). Bootstrap follows paper Appendix D Algorithm literal (per-iteration OLS refit). `n_bootstrap=999` default, `n_bootstrap >= 99` validated. `G < 10` returns NaN; `G > 100_000` emits a `UserWarning` pointing to `yatchew_hr_test`. The ~2× performance follow-up landed 2026-07-07 (loop-invariant hoisting, bit-identical — see the Phase 3 Stute bootstrap Note above).
- [x] Phase 3: `yatchew_hr_test()` heteroskedasticity-robust specification test. Test statistic `T_hr = sqrt(G) · (σ̂²_lin - σ̂²_diff) / σ̂²_W` from paper Equation 29. Normalizer `σ̂²_diff` divides by `2G` (paper-literal Theorem 7), NOT `2(G-1)`; hand-computed tight parity asserted at `atol=1e-12`. One-sided standard-normal critical value. `G < 3` returns NaN. Phase 3 shipped only the linearity null (paper Theorem 7); the `null="mean_independence"` R-parity extension shipped post-PR #392 (see the algorithm-variant block above for the contract).
- [x] Phase 3: `did_had_pretest_workflow()` composite helper. `data`-only entry point with `aggregate` dispatch: `aggregate="overall"` (default) requires a balanced two-period panel — multi-period panels are rejected at the front door by `_validate_had_panel` with a pointer to `aggregate="event_study"` — and runs steps 1 (QUG) + 3 (Stute + Yatchew-HR) only; `aggregate="event_study"` takes a multi-period panel (>=3 periods) and additionally runs step 2 (joint Stute pre-trends over pre-period horizons) + joint Stute homogeneity over post-period horizons, populating `pretrends_joint` / `homogeneity_joint`. `seed` forwards to all bootstrap-based tests (QUG and Yatchew are deterministic). Returns `HADPretestReport` with priority-ordered verdict string. On `aggregate="overall"` a fail-to-reject verdict explicitly flags the Assumption 7 gap rather than claiming unconditional TWFE safety: `"QUG and linearity diagnostics fail-to-reject; Assumption 7 pre-trends test NOT run (paper step 2 deferred)"`; on `aggregate="event_study"` a fail-to-reject across all three covered diagnostics reads `"TWFE admissible under Section 4 assumptions"` without the Assumption 7 caveat. Verdict priority follows the paper's one-way rule (TWFE admissible only if NO test rejects): **conclusive rejections are the primary verdict and are NEVER hidden by inconclusive status** — any unresolved-step note is appended via `"; additional steps unresolved: ..."` rather than replacing the rejection. The pure `"inconclusive - QUG NaN"` / `"inconclusive - both Stute and Yatchew linearity tests NaN"` forms only fire when NO conclusive test rejects AND a required step is unresolved. The partial-workflow fail-to-reject verdict may carry a `"(Yatchew NaN - skipped)"` (or Stute) suffix when one linearity test is NaN but the other is conclusive (step 3 resolved via the paper's "Stute OR Yatchew" wording). Bundled rejection-reason strings name each failed assumption in the conclusive-rejection case. `all_pass` is `True` iff QUG is conclusive AND at least one of Stute/Yatchew is conclusive AND no conclusive test rejects. **Non-negative-dose contract**: all three raw linearity helpers (`qug_test`, `stute_test`, `yatchew_hr_test`) raise a front-door `ValueError` on any `d < 0`, mirroring the `_validate_had_panel` guard (paper Section 2 HAD support restriction). On the `aggregate="overall"` path, the panel must already be exactly two periods (`_validate_had_panel` raises with a pointer to `aggregate="event_study"` otherwise); the first-difference helper computes `(t_post, t_pre)` per unit and feeds each raw helper directly. On the `aggregate="event_study"` path, joint Stute is dispatched across pre-period and post-period horizons directly (the joint Equation-18 form, no per-horizon pre-slicing).
- [x] Phase 4: Pierce-Schott (2016) replication harness reproduces Figure 2 values. **Waived 2026-05-20:** see Deviations block above; the paper itself self-acknowledges that NP estimators are too noisy to be informative on the LBD-restricted PNTR panel (Section 5.2), and R parity at `atol=1e-8` via `tests/test_did_had_parity.py` is a strictly stronger correctness anchor than Figure-2 reproduction on a proxy panel. Recorded as waived in the `DEFERRED.md` decision record.
- [x] Phase 4: Full DGP 1/2/3 coverage-rate reproduction from Table 1. **Waived 2026-05-20:** see Deviations block above; R parity at `atol=1e-8` on the same 3 DGPs reproduces the exact point estimate and SE algorithm (Python ≡ R ≡ paper) at sample-mean level — stronger than coverage-rate MC, which re-verifies asymptotic-coverage already pinned by R parity. Recorded as waived in the `DEFERRED.md` decision record.
- [x] Phase 5 (wave 1, PR #402): `practitioner_next_steps()` integration for HAD results - `_handle_had` and `_handle_had_event_study` route both result classes through HAD-specific Baker et al. (2025) step guidance with bidirectional HAD ↔ ContinuousDiD Step-4 routing closure. The `_check_nan_att` helper extends to ndarray `att` (HAD event-study) via `np.all(np.isnan(arr))` semantics; scalar path bit-exact preserved. The `llms-full.txt` HAD section's documented constructor and `fit()` parameter lists are regression-locked against `inspect.signature(HeterogeneousAdoptionDiD.__init__)` and `HeterogeneousAdoptionDiD.fit` for parameter-name presence (parameter defaults and the non-return parameter type annotations remain unpinned by the current `inspect.signature` test). The `fit()` return annotation is widened to `Union[HeterogeneousAdoptionDiDResults, HeterogeneousAdoptionDiDEventStudyResults]` at the source-code level to match the runtime polymorphism, AND that union is now pinned at the test level by `tests/test_had.py::TestFitReturnAnnotation::test_fit_return_annotation_is_union_of_result_classes` via `typing.get_type_hints` so the contract cannot drift silently.
- [x] Phase 5 (wave 1, PR #402): `llms-full.txt` HeterogeneousAdoptionDiD section + result-class blocks + `## HAD Pretests` index + Choosing-an-Estimator row landed; constructor / fit() parameter names are regression-locked against `inspect.signature(HeterogeneousAdoptionDiD.__init__)` and `HeterogeneousAdoptionDiD.fit` for parameter-name presence (parameter defaults and the non-return parameter type annotations remain unpinned; the `fit()` return-type union is locked BOTH at the source-code level AND at the test level by `TestFitReturnAnnotation`); result-class field tables enumerate every public dataclass field (regression-tested via `dataclasses.fields()`); `llms-practitioner.txt` Step 4 decision tree distinguishes ContinuousDiD (per-dose ATT(d), needs never-treated) from HeterogeneousAdoptionDiD (WAS, universal-rollout-compatible).
- [x] Phase 5 (partial): README catalog one-liner, bundled `llms.txt` `## Estimators` entry, `docs/api/had.rst` (autoclass for the three classes), and `docs/references.rst` citation landed in PR #372 docs refresh.
- [x] Phase 5 (wave 2 first slice, PR #409): T21 HAD pretest workflow tutorial (`docs/tutorials/21_had_pretest_workflow.ipynb`) — composite pre-test walkthrough for `did_had_pretest_workflow`. Uses a `Uniform[$0.01K, $50K]` dose-distribution variant of T20's brand-campaign panel (true support strictly positive but near-zero, chosen so QUG fails-to-reject `H0: d_lower = 0` in finite sample). Walks through `aggregate="overall"` (Steps 1 + 3 only, verdict explicitly flags Step 2 deferral) and upgrades to `aggregate="event_study"` (joint pre-trends Stute + joint homogeneity Stute close the gap). Side panel exercises both `yatchew_hr_test` null modes (`linearity` vs `mean_independence`). Companion drift-test file `tests/test_t21_had_pretest_workflow_drift.py` (17 tests pinning panel composition, both verdict pivots, structural anchors, deterministic stats, bootstrap p-value tolerance bands per backend, and `HAD(design="auto")` resolution to `continuous_at_zero` on this panel).
- [x] Phase 5 (wave 2 second slice): T22 weighted/survey HAD tutorial (`docs/tutorials/22_had_survey_design.ipynb`) - shipped as the follow-up to PR #432. End-to-end walkthrough of `HeterogeneousAdoptionDiD` + `did_had_pretest_workflow` under `SurveyDesign(weights, strata, psu, fpc)` on a BRFSS-shape state-rollout panel (5 strata x 6 PSUs/stratum x 2 states/PSU = 60 states; post-stratification raking weights with CV ~ 0.30; FPC = 30 PSUs/stratum). Companion drift-test file `tests/test_t22_had_survey_design_drift.py` (32 tests pinning panel composition, naive-vs-survey SE inflation direction, design auto-detection, event-study cband-vs-pointwise width ordering, `_QUG_DEFERRED_SUFFIX` substring on `report.verdict` for both overall and event-study paths, the distinct `report.summary()` QUG-skip note on the event-study path, deterministic Yatchew sigma2_*, bootstrap p-value anchored windows of total width 0.30 (± 0.15 around seeded centers) per `feedback_strata_bootstrap_path_divergence`, workflow-surface separation between overall and event-study paths, and the weighted point-estimation contract via the `_fit_continuous` algebraic identity).
- [x] Documentation of non-testability of Assumptions 5 and 6. **Closed 2026-05-20:** `HeterogeneousAdoptionDiD` class docstring carries a "Non-testable assumptions (paper Section 3.1.2)" Notes block; `qug_test` / `stute_test` / `yatchew_hr_test` / `did_had_pretest_workflow` Notes sections carry "Scope (what this test does NOT cover)" clauses explicitly stating they verify ADJACENT identifying conditions (QUG: support-infimum null `d_lower = 0`; Stute / Yatchew: Assumption 8 linearity; `joint_pretrends_test`: Assumption 7 mean-independence) and CANNOT test Assumptions 5 or 6. The composite workflow verdict string does NOT mention Assumptions 5 or 6 — it only flags the Assumption 7 step-2 gap on the two-period `aggregate="overall"` path. The Assumption 5/6 non-testability caveat is surfaced separately by (a) `HAD.fit()`'s fit-time `UserWarning` in `diff_diff/had.py` (search for "---- Assumption 5/6 warning on Design 1 paths ----") which fires whenever the resolved design is Design 1 family (`continuous_near_d_lower` or `mass_point`), and (b) T21 (HAD pretest workflow tutorial) tutorial prose.
- [x] Warnings for staggered treatment timing (redirect to `ChaisemartinDHaultfoeuille`). **Closed 2026-05-20:** fail-closed `ValueError` at `diff_diff/had.py:1624` (see Deviations § "Library extension: Staggered-timing fail-closed" for the rationale on raising vs warning).
- [x] `NotImplementedError` phase pointer when `covariates=` is passed (Theorem 6 future work). **Closed 2026-06-01:** `HAD.fit()` now takes an explicit keyword-only `covariates=None` param and raises `NotImplementedError` (with the Appendix B.1 / Theorem 6 multivariate-covariate-extension pointer + a pre-residualization workaround) when it is not None, replacing the prior bare `TypeError` from the absent kwarg. See the `- **Note:**` ("`covariates=` is reserved but NOT implemented") above and `diff_diff/had.py::HeterogeneousAdoptionDiD.fit`; locked by `tests/test_methodology_had.py::TestHADDeviations::test_covariates_not_implemented_is_documented`.

---

# Regression Discontinuity

## RegressionDiscontinuity

**Primary source:** Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014).
Robust Nonparametric Confidence Intervals for Regression-Discontinuity
Designs. *Econometrica*, 82(6), 2295-2326. https://doi.org/10.3982/ECTA11757.
Software/parity reference: Calonico, Cattaneo, Farrell & Titiunik (2017),
*Stata Journal* 17(2), 372-404 (rdrobust); CER-optimal bandwidth theory:
Calonico, Cattaneo & Farrell (2018), *JASA* 113(522), 767-779; covariate
adjustment: Calonico, Cattaneo, Farrell & Titiunik (2019), *REStat*
101(3), 442-451. https://doi.org/10.1162/rest_a_00760.

**Estimand and estimator (sharp RD):** with running variable `X`, known
cutoff `c`, and treatment `T = 1(X >= c)` (units exactly at the cutoff are
TREATED, matching rdrobust), the estimand is
`tau_SRD = E[Y(1) - Y(0) | X = c] = mu_+ - mu_-` (Hahn-Todd-van der Klaauw
2001 identification). Point estimation: order-`p` kernel-weighted local
polynomials on each side at bandwidth `h` (`tau_cl` = difference in
intercepts); bias correction via order-`q > p` fits at pilot bandwidth `b`
(`tau_bc = tau_cl - h^{p+1} B_hat`, computed through the `Q_q` score-matrix
formulation of rdrobust); robust bias-corrected variance per CCT 2014
Theorem 1 / Theorem A.1(V) (the `Q_q` sandwich with same-side
nearest-neighbor residuals shared with the conventional sandwich).
Defaults reproduce `rdrobust(y, x)`: `p=1`, `q=2`, triangular kernel,
`bwselect="mserd"`, `vcov_type="nn"` (J=3), `masspoints="adjust"`,
normal-quantile CIs.

**Estimand and estimator (fuzzy RD; `fit(..., takeup=...)`):**
crossing the cutoff shifts OBSERVED take-up `T` instead of determining
it; the estimand is the local Wald ratio
`tau_FRD = tau_Y,SRD / tau_T,SRD` (CCT 2014 Section 3.2; identification
requires `tau_T,SRD != 0`, Theorem 3). For binary take-up under
monotonicity this is the LATE for compliers at the cutoff; for
non-binary (dose) take-up - accepted, matching R's `fuzzy=` - it is the
ratio-of-jumps estimand without the complier-LATE reading. The
`estimand` results field is DATA-DEPENDENT accordingly: binary take-up
(values in {0, 1}) reports "fuzzy (LATE for compliers at the cutoff)";
anything else reports "fuzzy (local Wald ratio at the cutoff;
non-binary take-up)" so the label never overclaims. Implementation stacks `T` as a second response column
through the SAME local-polynomial fits (rdrobust.R:581-591); the bias
correction is the LINEARIZATION of the ratio via the delta vector
`s_Y = [1/tau_T_cl, -tau_Y_cl/tau_T_cl^2]`:
`tau_bc = tau_cl - s_Y . B_F` with
`B_F = [tau_Y_cl - tau_Y_bc, tau_T_cl - tau_T_bc]` (rdrobust.R:636-657)
- NOT a per-component bias-corrected ratio, per CCT 2014's "bias-correct
the first-order linear approximation of the ratio". Variances reuse the
same sandwiches with the (n, 2) residual matrix collapsed by `s_Y`
(ratio) or `sV_T = [0, 1]` (first stage); the Y-T covariance is the
cross term of `res @ s` (functions.R:379-385). The per-side biases are
the linearized `s_Y . B_F_side` (rdrobust.R:649-652). The first stage is
exposed as a full three-row mirror (`first_stage*` fields, robust-row
canonical binding like `att`) and `summary()` prints a first-stage block
above the treatment effects, as R does. A take-up column deterministic
in `X` reproduces the sharp fit (first stage == 1, ULP-level agreement -
the ratio divides by a float-solved 1).

**Covariate adjustment (`fit(..., covariates=[...])`; CCFT 2019, R's
`covs=`):** covariates enter ADDITIVELY with a COMMON coefficient pooled
across sides - CCFT 2019 Equation 2, the paper's recommended (and only
implemented) specification. Lemma 1 derives the probability limits of the
five candidate specifications: the treatment-interacted variant
(equivalent to separate per-side fits) needs the strictly stronger
condition `mu'_Z+ gamma_Y+ = mu'_Z- gamma_Y-` and the three
demeaning-based variants have slower rates and extra misspecification
bias - all four are deliberately NOT implemented. The ESTIMAND IS
UNCHANGED (`tau_SRD`/`tau_FRD` as above): adjustment buys precision, and
the `estimand` label deliberately does not change (unlike the DiD
estimators, where `covariates` switches identification to conditional
parallel trends). The operative consistency condition is covariate
BALANCE at the cutoff (`tau_Z = 0`, the testable sufficient condition of
Lemma 1: the plim shifts by `[mu_Z+ - mu_Z-]' gamma_Y` under imbalance,
and adjusting "for" imbalance cannot restore identification without
functional-form assumptions). The balance placebo is the estimator
itself - fit each covariate as `outcome` and inspect its RD p-value
- documented in the module docstring; a packaged `covariate_balance`
helper is a named follow-up (diagnostics wave). Implementation stacks
`Z` after `(Y, T)` as extra response columns through the SAME
local-polynomial fits (rdrobust.R:593-598); the common gamma solves the
PARTIALLED normal equations with per-side blocks summed
(`ZWZ = sum_side [Z'W D - U' invG U]`, rdrobust.R:659-671), and the
adjusted estimates are the delta-vector combinations
`s_Y = [1, -gamma[,1]]` (sharp; rdrobust.R:672-686 - the R branch omits
`factorial(deriv)` present in the no-covariate branch, identical at the
fixed `deriv=0` surface and replicated verbatim) or the fuzzy+covariates
system of rdrobust.R:688-723 (adjusted Y and T jumps via
`s_Y`/`s_T = [1, -gamma[,2]]`, their ratio, linearized bias correction,
and the EXTENDED length-`(2+dZ)` variance vector
`[1/tau_T, -tau_Y/tau_T^2, -(1/tau_T)gamma[,1] + (tau_Y/tau_T^2)gamma[,2]]`;
first-stage selector `sV_T = [0, 1, -gamma[,2]]`). Variances reuse the
same sandwiches with the `(n, 1+dT+dZ)` NN residual matrix collapsed by
the corresponding vector (functions.R:146-204, 374-385). The partial-out
identity `tau_adj = tau_unadj - gamma' tau_Z` (each covariate fit as an
outcome) holds EXACTLY at common manual `(h, b)` by Frisch-Waugh and is
locked for both the conventional and bias-corrected rows
(`test_partial_out_identity_exact`). Fitted gammas are exposed
name-keyed (`covariate_coefficients`, and
`first_stage_covariate_coefficients` on fuzzy fits) as NUISANCE
coefficients, not causal effects.

**Bandwidth selection:** all 10 rdrobust data-driven selectors (`mserd`
default; `msetwo`/`msesum`/`msecomb1`/`msecomb2`; CER-optimal `cer*`
variants = the matching MSE selector's `h` shrunk by
`N^(-p/((3+p)(3+2p)))` with `b` inherited unchanged), via the faithful
4.0.0 port in `diff_diff/_rdrobust_port.py` (three-stage d->b->h pilot
chain, per-kernel pilot constants, IK-style regularization,
bwrestrict/bwcheck clamps, masspoints unique-count adjustment). Manual
`h`/`b`/`rho` semantics mirror rdrobust exactly: `h` alone -> `b = h`;
`h`+`rho` -> `b = h/rho` (overriding a supplied `b`, with a warning);
`rho` without `h` applies to the SELECTED bandwidths. Fuzzy bandwidth
logic (CCFT 2017 Section 6, R-exact): by default bandwidths are selected
for the FUZZY RATIO objective - `T` is stacked into every pilot fit and
the pilot ratio + delta vector feed the V/B constants (approach 2);
`sharpbw=True` forces the sharp reduced-form-on-`Y` objective
(approach 1); and ONE-SIDED PERFECT COMPLIANCE (zero take-up variance on
either side) auto-switches to approach 1 regardless of the flag
(`perf_comp`, rdrobust.R:164-185 / rdbwselect.R:334-346) - selection
only; estimation always remains fuzzy. Covariate-adjusted bandwidths are
COVARIATE-AWARE (CCFT 2019 Theorem 1: the bias/variance constants differ
from the no-covariate case, so selecting `h` unadjusted and then adding
covariates is not MSE-optimal): `Z` is stacked into every pilot fit with
a PER-PILOT partialled gamma and the extended `s` vector
(functions.R:241-274, 349). `perf_comp`/`sharpbw` null ONLY `T` - `Z`
always stays in selection (rdbwselect.R:343-345), so `sharpbw` with
covariates selects on the covariate-adjusted sharp objective. In the
port, `T` and `Z` thread through the shared `_bw` closure so all three
selector chains (mserd, msetwo, msesum - 14 pilot call sites) receive
them; the `msetwo`/`cercomb2` fuzzy and covariate golden configs pin the
per-side and comb chains.

- **Note (canonical inference binding; deviation from R's printed
  output):** the result's canonical `att`/`se`/`t_stat`/`p_value`/
  `conf_int` are ONE coherent row - the ROBUST row: `att = tau_bc`
  (bias-corrected), `se = se_robust`, with `t_stat == att/se` and
  `conf_int` centered on `att`, preserving the library-wide field
  identities. rdrobust instead PRINTS the conventional `tau_cl` as its
  headline coefficient while taking inference from the robust row (whose
  CI is centered at `tau_bc`, not `tau_cl`). `tau_cl` is first-class as
  `att_conventional` with a full inference row, the middle
  (bias-corrected, conventional-SE) row is exposed as
  `*_bias_corrected` fields, and `summary()` prints the three-row
  rdrobust table verbatim. Golden tests pin every row against R.
- **Note (parity pin):** the port targets the installed CRAN release
  `rdrobust` 4.0.0, pinned by source-tarball sha256
  (`78f0d6b4...d613a36f` in `diff_diff/_rdrobust_port.py`). The GitHub
  development tree (4.1.0-dev) is NOT the parity source: it adds an
  `nn_tol` nearest-neighbor tie tolerance, flips the `stdvars` default
  to TRUE, and drops 4.0.0's `+1e-8` bwcheck floor. The port implements
  the 4.0.0 behaviors: EXACT nearest-neighbor distance comparison,
  standardization machinery present but fixed to the 4.0.0 default
  (off), and the `+1e-8` floor (rdbwselect.R:374-375).
- **Deviation from R:** missing/non-numeric rows are dropped WITH a
  `UserWarning` naming the count (`fit()`), where R's `complete.cases`
  filter drops silently (no-silent-failures policy). The private port
  layer rejects non-finite inputs outright.
- **Deviation from R:** `b=` without `h=` is ignored WITH a warning
  (R ignores it silently). `h`+`b`+`rho` warns that rho takes precedence
  (R applies `b = h/rho` silently).
- **Note (N < 20, R-exact):** fewer than 20 complete-case observations
  triggers rdrobust.R:303-307's behavior exactly - warn and estimate on
  the entire sample (`h = b = max` distance to the cutoff), OVERRIDING
  even a user-supplied `h` (the R block runs after manual-bandwidth
  resolution); `bwselect` reports "Manual".
- **Deviation from R:** degenerate designs fail closed with targeted
  `ValueError`s where R propagates NaN, errors opaquely, or silently
  returns a pseudo-inverse fit: a non-finite pilot bandwidth from a
  degenerate MSE objective (e.g. constant outcome under data-driven
  selection), non-finite/non-positive bandwidths reaching estimation,
  one-sided data, a cutoff outside the observed support, zero-variance
  running variable, and - per side, per window - a main (`h`) or bias
  (`b`) window whose positive-weight support has fewer than `p+1`
  (resp. `q+1`) DISTINCT running-variable values. The distinct-support
  count is the exact rank condition for a weighted Vandermonde design,
  so this guard catches empty windows (e.g. manual `b` far smaller than
  `h`, or a huge `rho` collapsing `b = h/rho`) and mass-point-only
  support, where R's `ginv` fallback would silently collapse the
  bias-correction score `Q_q` to the conventional one or report a
  zeroed fit. A constant outcome under MANUAL bandwidths yields
  `se = 0` and the standard joint-NaN inference gate via
  `safe_inference()` (att = 0 is well defined; all three rows' triples
  gate independently).
- **Note:** `qrXXinv`'s Cholesky-failure fallback maps `MASS::ginv(G)`
  to `numpy.linalg.pinv(G, rcond=sqrt(eps))` - the same Moore-Penrose
  pseudo-inverse with the same default cutoff. The fallback IS the 4.0.0
  behavior (verified against both the pinned tarball source,
  functions.R:128-132, and `deparse` of the installed namespace);
  rank/support protection for identification is provided by the
  per-window distinct-support guards above (in the estimation port,
  ahead of any Gram inversion), not by changing `qrXXinv`'s parity
  behavior.
- **Deviation from R (weak first stage):** rdrobust is SILENT when the
  take-up jump is statistically indistinguishable from zero (verified
  live on installed 4.0.0: a near-zero first stage returns a wildly
  inflated ratio with no message). The estimator emits a `UserWarning`
  when the first-stage ROBUST confidence interval at the fit's own
  `alpha` is FINITE and contains 0 - a parameter-free gate (no imported
  F-statistic threshold) backed by CCT 2014's Theorem 3 requirement
  `tau_T,SRD != 0` and its weak-identification "guard and warn"
  discussion citing Feir-Lemieux-Marmer. The finite-CI gate means
  NaN-gated first stages (see next Note) never fire the warning.
  Weak-IV-ROBUST fuzzy inference (Feir-Lemieux-Marmer 2016) is a
  documented seam.
- **Note (first-stage `se = 0`, deviation from R's `Inf`):** perfect
  compliance (take-up deterministic in the running variable) gives
  exactly-zero first-stage NN residuals, so `se_T = 0` and the
  first-stage inference triples NaN-gate via `safe_inference()` where R
  prints `z_T = Inf, pv_T = 0`. The first-stage point estimate itself
  (== 1) is reported. A DEGENERATE-ZERO first stage under manual
  bandwidths (no take-up variation inside the window despite variation
  in the sample) follows R's Inf/NaN arithmetic flow-on: the ratio rows
  come out non-finite and joint-NaN-gate - loud, not silent. Under
  DATA-DRIVEN bandwidths the same degeneracy hits the pilot ratio
  (`tau_T == 0` in a pilot window) and fails closed with the targeted
  "non-finite pilot bandwidth" error (documented deviation; R flows
  Inf/NaN through selection into opaque downstream errors).
- **Deviation from R:** `sharpbw=True` on a SHARP fit warns and is
  ignored (R ignores it silently); on manual-bandwidth fuzzy fits it is
  a silent no-op in both implementations (selection is skipped).
- **Note (fuzzy identification guard, R-exact):** a take-up variable
  with zero variance on BOTH sides and no mean jump at the cutoff raises
  R's exact error ("Fuzzy RD: first-stage variable has no variation and
  no jump at the cutoff...") in both entry points, and the check runs
  BEFORE mass-point detection, matching R's rdrobust ordering
  (rdrobust.R:175 precedes :365-380; live-verified that R raises with no
  mass-point warning on degenerate fuzzy + tied data). One-sided zero
  variance is a legitimate design and routes to the `perf_comp`
  bandwidth switch instead. The R `var(T_side) == 0` test is implemented
  as exact constancy (R's two-pass `mean()` makes its variance of a
  constant vector exactly zero; numpy's single-pass mean does not).
- **Note (covariate redundancy pipeline, R-exact under `covs_drop=True`):**
  R first sorts covariate columns by NAME LENGTH (`order(nchar)`, stable
  - rdrobust.R:131; the sort decides WHICH of a collinear set survives),
  then drops redundant columns via a rank-revealing pivoted QR
  (`qr(z, tol=1e-7)`, LINPACK dqrdc2's per-column relative rule: a
  column is negligible when its reduced norm falls below tol times its
  OWN original norm, so small-but-independent covariates are never
  dropped; `covs_drop_fun`, functions.R:683-688). Both are replicated
  exactly - the port implements the dqrdc2 loop directly (LAPACK
  pivoting differs on near-ties) and the estimator applies the name sort
  before building the matrix; the sort never leaks because every
  user-facing covariate surface is name-keyed (order-invariance tested).
  The drop runs hoisted in `fit()` on x-sorted rows - the same rows R's
  QR sees - immediately after the NaN drop and before the fuzzy
  identification stop, matching rdrobust.R:121-140's ordering.
  **Deviation from R:** the drop warning NAMES the dropped columns
  (`covariates_dropped` echoes them) where R's message is generic, and a
  rank-0 covariate matrix fails closed with a targeted `ValueError`
  where R would error opaquely downstream. `covs_drop=False` (strict
  mode) rejects ANY covariate degeneracy - mutual collinearity or
  collinearity with the local polynomial design - with a deterministic
  `ValueError`; R's `covs_drop=FALSE` relies on `chol()` erroring, which
  on an exactly-singular float matrix is roundoff-dependent.
- **Deviation from R (degenerate covariate adjustment is guarded, not
  reproduced):** R solves the partialled system with
  `MASS::ginv(ZWZ, tol=1e-20)`; on an EXACTLY-degenerate system -
  covariates collinear with the local polynomial design after
  partialling, e.g. a constant covariate or a full one-hot dummy set
  (both pass the intercept-free QR check above) - that tolerance INVERTS
  a float-noise singular value, making R's gamma platform-noise
  (observed 28% cross-implementation spread) and silently shifting tau
  (~0.5% in the smoke). The port instead (a) EXCLUDES per-column
  degeneracies - explained-ratio `diag(ZWZ)_j/(z_j'Wz_j) < 1e-14` -
  zeroing their gamma rows, so a constant covariate reproduces the fit
  without it to floating-point roundoff (tested at rel 1e-12; not
  bit-for-bit - the response matrix still carries the excluded column
  and BLAS matmul kernels differ with matrix shape across platforms);
  (b) cuts SET-level noise directions with an
  equilibrated (scale-invariant) pseudo-inverse (`sv_min < 1e-12 *
  sv_max` -> `rcond=1e-12` cut), so a full dummy set reproduces the
  drop-one-category fit (span invariance, tested at rel 1e-9); and (c)
  warns once per fit naming the affected columns. Well-posed systems
  take `np.linalg.pinv(rcond=1e-20)` - the same semantics as R's ginv -
  and match R at machine precision (a 1e-9-scaled independent covariate
  stays on this path untouched; scale-invariance tested). Bandwidth
  pilots apply the same guard silently (per-window transients would
  otherwise warn dozens of times per fit).
- **Note (v1 scope seams):** only `vcov_type="nn"` (rdrobust's default)
  ships; `hc0`-`hc3` and cluster modes raise `NotImplementedError`.
  Weights, kink estimands (`deriv`), `scalepar`, `stdvars`, per-side
  manual bandwidths, weak-IV-robust fuzzy inference
  (Feir-Lemieux-Marmer), the density/manipulation-test diagnostic, and a
  packaged covariate-balance helper are documented follow-ups (the
  rdplot diagnostic ships as `RDPlot` - see its section below);
  cluster/weights are not parameters at all. The port's `deriv` machinery is
  golden-covered for `deriv in {0, 1}` only.
- **Note (p/q surface, R-exact):** public `p`/`q` validation mirrors
  rdrobust.R:47-57 exactly - integers in 0:20 with `q > p`; `p=0` is R's
  local-constant fit and is accepted. R resolves a NULL `q` to `p + 1`
  BEFORE its `q` validation and never re-checks the default
  (rdrobust.R:53-57), so `p=20` with `q=None` yields `q=21` in both R
  and diff-diff while an EXPLICIT `q=21` is rejected - the asymmetry is
  mirrored deliberately (locked by
  `test_p20_default_q_resolution_matches_r_quirk`). Golden-anchored at
  `p in {0, 1, 2}` (`p0q1`, default `p1q2`, `p2q3` configs); the
  Remark-7 identity additionally anchors the `p=0 -> p=1` and
  `p=1 -> p=2` steps R-free. All 10 public `bwselect` routings are
  locked end-to-end against the bandwidth goldens
  (`test_rdd_parity.py::TestPublicSelectorRouting`).
- **Note (cross-implementation float tolerance):** R's sample SD
  accumulates sequentially while numpy sums pairwise, seeding a ~2-ULP
  difference in the pilot constant `BWp` that the chained d->b->h
  bandwidth stages amplify to an observed worst-case ~5e-11 relative
  disagreement (p=2/q=3 config, darwin-arm64). Golden parity asserts
  rtol=1e-9 on bandwidths and estimates; pre-linalg intermediates
  (type-2 quantile IQR, unique counts) agree at float64 identity.

**Edge cases:**
- Mass points (CCT 2014 Remark 1): `masspoints="adjust"` (default)
  substitutes unique-value counts into the pilot bandwidth and floors
  the window at 10 unique support points when >= 20% mass is detected;
  "check" warns and suggests adjust; "off" reproduces the
  pre-masspoints package (and the published 2017 Stata Journal Senate
  numbers exactly). Adjust == off on tie-free data (tested). Detection
  runs in `fit()` BEFORE the manual-vs-data-driven bandwidth branch,
  exactly where R's rdrobust() runs it (rdrobust.R:365-380), so the
  warning fires on manual-`h` fits too; the port's rdbwselect-level
  copy of the warning is suppressed on the estimator path
  (`warn_masspoints=False`) to mirror R's single warning per fit
  (R's estimation-path selection is inline, so its rdbwselect warning
  never stacks; verified against installed 4.0.0 on both paths).
- CCT 2014 Remark 7 (internal consistency, tested): with `b = h` and
  the same kernel, `tau_bc(p, q=p+1)` is numerically identical to the
  order-`(p+1)` conventional estimate, and the robust variance equals
  the order-`(p+1)` conventional variance (verified at rel 1e-10 across
  all three kernels).

**Reference implementation(s):**
- R: `rdrobust::rdrobust()`, `rdrobust::rdbwselect()` 4.0.0 (CRAN).
- Stata: `rdrobust` (Calonico-Cattaneo-Farrell-Titiunik 2017).

**Validation:** bandwidth goldens `benchmarks/data/rdrobust_golden.json`
(17 configs x 10 selectors, `tests/test_rdrobust_port.py`); estimation
goldens `benchmarks/data/rdrobust_estimates_golden.json` (32 configs
incl. 7 fuzzy - default/sharpbw/manual-h/epa/msetwo/one-sided-perf_comp/
ties - with full first-stage three-row blocks, and 9 covariate configs -
default/manual-h/msetwo/cercomb2/epa/collinear-drop/ties/fuzzy/
fuzzy-sharpbw - with `coef_covs` gamma pins and UNSORTED
differing-length names so every config also pins the nchar column sort;
the per-side LINEARIZED fuzzy biases and the covariate gamma matrices
are pinned at port level in
`tests/test_rdrobust_port.py::TestFuzzyPortGoldenParity` /
`::TestCovsPortGoldenParity`, with `covs_drop_fun` unit-pinned against
live-R `qr()` rank/pivot results in `::TestCovsDropFun`;
`tests/test_rdd_parity.py` pins everything the public results expose);
vendored Senate data (`benchmarks/data/rdrobust_senate.csv`,
Cattaneo-Frandsen-Titiunik 2015) anchoring the published 2017 Stata
Journal numbers under `masspoints="off"`; R-free methodology anchors in
`tests/test_rdd_methodology.py` (Remark 7 equivalence, invariances,
perfect-compliance == sharp, perf_comp/sharpbw bandwidth switches,
weak-first-stage warning gate, NaN/degenerate contracts, and the
covariate partial-out identity / span-invariance / order-invariance /
CI-shrinkage anchors).

**Paper reviews on file:**
`docs/methodology/papers/calonico-cattaneo-titiunik-2014-review.md` (CCT 2014,
Econometrica - robust bias-corrected RD inference),
`calonico-cattaneo-titiunik-2015-review.md` (JASA - optimal data-driven RD
plots; bin-count selector formulas, implemented as `RDPlot`),
`calonico-cattaneo-farrell-titiunik-2017-review.md` (Stata Journal - the
rdrobust software reference this port parity-targets),
`calonico-cattaneo-farrell-2018-review.md` (JASA - CER-optimal bandwidths),
`calonico-cattaneo-farrell-titiunik-2019-review.md` (REStat - covariate
adjustment, implemented),
`cattaneo-jansson-ma-2020-review.md` (JASA - boundary-adaptive local
polynomial density estimators + the density-discontinuity manipulation test;
the rddensity methodology. Review only - the packaged density test remains a
documented v1 seam pending implementation).

## RDPlot

**Primary source:** Calonico, S., Cattaneo, M. D., & Titiunik, R. (2015).
Optimal Data-Driven Regression Discontinuity Plots. *JASA*, 110(512),
1753-1769. https://doi.org/10.1080/01621459.2015.1017578 (+ Supplemental
Appendix). Full formula extraction in
`docs/methodology/papers/calonico-cattaneo-titiunik-2015-review.md`.

**What it is:** the exploratory/graphical companion to
`RegressionDiscontinuity` (`diff_diff/rdplot.py`) - NOT a treatment-effect
estimator. Two ingredients (CCT 2015 Section 2.1): per-side global order-`p`
polynomial fits (default `p=4`, uniform kernel over each side's full range -
R's rdplot defaults, deliberately different from estimation), and binned
local sample means of the outcome over a per-side partition with a
data-driven number of bins. Parity target: `rdplot()` in CRAN rdrobust 4.0.0
(same tarball as the estimator port), golden-tested end-to-end
(`benchmarks/data/rdplot_golden.json`, 24 configs incl. the vendored Senate
data; the supplement's Figures SA-1/SA-2 selector outputs on that dataset -
esmv J=(15,35), es (8,9), qs (21,16), qsmv (28,49) - are asserted as
JSON-independent paper anchors in `tests/test_rdplot.py`).

**Bin selectors (rdplot.R:276-447):** all 8 `binselect` values =
{ES, QS partition} x {IMSE-optimal, mimicking-variance target} x
{spacings (`hat`), polynomial-regression (`chk`, the `*pr` names) variance
estimator}. The bias estimator is SHARED between the spacings and `pr`
families (main-text Eq 8/10 raw-observation form - the source read resolved
the paper review's Gap #3); only the variance estimator differs. Selector
fits use `k=4` raw-power Vandermonde regressions of `y` and `y^2`
(unweighted, full samples, NOT centered at the cutoff) with a `k=4 -> 3 -> 2`
fallback ladder that fires ONLY when `qrXXinv` itself raises (non-finite
Gram from `x^8` overflow / SVD failure); finite singular Grams are absorbed
inside `qrXXinv`'s pseudo-inverse fallback at unchanged `k`, mirroring R's
internal `ginv` fallback exactly. `J = ceil(((2B/V) n)^(1/3))` (IMSE) or
`ceil((var(y_side)/V) n / log(n)^2)` (mimicking variance); the result
reports the selected pair alongside `J_IMSE`, `J_MV`, the implied scale
`rscale = J/J_IMSE`, and the Supplement-S.1 WIMSE weights
`(1/(1+rscale^3), rscale^3/(1+rscale^3))`, matching R's `summary.rdplot`.

**Edge cases (R-verified):**
- **`n < 20` hard error** (rdplot.R:190-193) - unlike the estimator's
  small-sample bandwidth fallback, rdplot refuses outright; replicated.
- **Mass points** (rdplot.R:116-137): >= 20% duplicate running values on
  either side warns; `masspoints="adjust"` remaps a spacings `binselect` to
  its `*pr` sibling (spacings degenerate under ties); `"check"` warns
  without remapping; explicit `*pr` picks are untouched.
- **Zero outcome variance on a side**: R propagates `Inf`/`NaN` through the
  selectors (`J_IMSE = Inf`) and rescues `J = 1` with a warning; replicated
  including the `Inf` echo (numpy division semantics, no exception).
- **Negative variance-function values** in the `chk` estimators are floored
  to the side's sample variance (rdplot.R:356-371; an R-source fact absent
  from the paper).
- **Empty bins** are dropped from `vars_bins` (R's `tapply`), while
  `bin_avg`/`bin_med` summarize ALL partition cells; per-bin CI columns are
  ALWAYS computed (level 95 when `ci` unset - the flag only turns on plotted
  error bars, matching R), and single-observation bins get `se = 0` via R's
  `NA -> 0` sd fixup.
- **Note (left-edge slot reflection, R quirk replicated):** R indexes the
  left side's `rdplot_min_bin`/`rdplot_max_bin` by `rev(-bins)` - the
  ascending-sorted ABSOLUTE bin ids, i.e. the occupied slots reflected
  around the side's center - while `rdplot_mean_bin` uses the true slots
  (rdplot.R:503-508 vs 551-552). With any empty left-side bin the reported
  left edges therefore disagree with `rdplot_mean_bin` (live-verified: 10 of
  20 left rows on the Senate `esmvpr` config have
  `mean_bin != (min+max)/2` in R itself). Replicated exactly for parity;
  `rdplot_mean_bin` is the reliable per-row bin identity.
- **Note (defensive guard):** `n >= 20` can still leave a side with < 2
  observations, where R degenerates opaquely (`NA` variance poisoning an
  `if`); a clear `ValueError` naming the side is raised instead.
- **Note (defensive guard, zero-weight global fit):** a manual `h` smaller
  than the nearest observation's distance zeroes every kernel weight; R
  silently fits the zero-weight design (`ginv` of a zero Gram) and returns
  an all-zero "curve" that used no data. A clear `ValueError` naming the
  side is raised instead.
- **Note (defensive guard, single-support-point side):** a side whose
  running variable has only ONE distinct value cannot be partitioned -
  every spacing is zero, the variance components collapse, `J` becomes
  `Inf`, and R crashes opaquely inside `seq()` on the zero jump. A clear
  `ValueError` naming the side is raised instead (>= 2 distinct running
  values per side required).
- **Note (missing rows warned; deviation from R):** rows with missing or
  non-numeric outcome/running/covariate values are dropped WITH a
  `UserWarning` reporting the count (R's `complete.cases` drops silently) -
  the same contract as `RegressionDiscontinuity.fit()`.
- **Note (defensive warning, noncontinuous outcomes):** the
  spacings/concomitant selectors assume a CONTINUOUSLY distributed outcome
  (CCT 2015 Theorems 3-4; Remarks 1-2 route noncontinuous outcomes to the
  series estimators), but R runs them on any `y` without comment. When a
  spacings `binselect` is active and either side's outcome has >= 20%
  duplicates (the masspoints criterion applied to `y` - e.g. a binary
  first-stage plot), a `UserWarning` recommends the `*pr` sibling. No
  auto-remap (R does not remap; numbers stay golden-comparable).
- **Note (QS cutpoints, R deviates from the paper):** CCT 2015 defines the
  QS partition through the empirical generalized inverse
  `Fhat^{-1}(y) = inf{x : Fhat(x) >= y}`, but R's rdplot builds the QS
  jumps with `quantile()` at its default TYPE-7 (linearly interpolated)
  quantiles (rdplot.R:477-478). The port follows R (numpy's default
  `np.quantile` is the same type-7 estimator), so QS bin edges are
  interpolated values rather than order statistics; golden-locked.
- **Deviation from R (fractional / pair scale):** R 4.0.0 crashes on a
  fractional `scale * J` product ("arguments imply differing number of
  rows", a vector-indexing accident) and rejects a length-2 `scale` outright
  on R >= 4.2 (vectorized-`if` error at rdplot.R:178). CCT 2015 Equation 2
  places the ceiling over the rescaled product, so `RDPlot` applies that
  paper-faithful ceiling and supports per-side scale pairs; integer-product
  configs remain golden-identical to R.
- **Deviation from R (covs message):** R prints an unconditional `message()`
  whenever `covs=` is used with the global fit; that caveat lives in the
  `fit()` docstring instead of an unavoidable runtime warning.
- **`nbins` label quirk (replicated):** a manual `nbins` sets the
  description string to "manually evenly spaced" even when `binselect`
  chose quantile spacing - the bins themselves stay quantile-spaced
  (rdplot.R:452-456; locked by test).

**Covariate adjustment (`fit(..., covariates=...)`; rdplot.R:139-159,
223-270, 492-499):** the global fit partials covariates out with the SAME
pooled-across-sides partialled-normal-equations gamma as the estimator
(nchar-sorted `covs_drop` pipeline, degenerate-system guard, and
`covs_drop=False` strict-error contract all REUSED from the #691 machinery -
including its documented Deviation from R's noise-singular-value `ginv`);
the plotted curve adds the scalar `colMeans(Z) @ gamma` (R's
`covs_eval="mean"`, the only mode in v1), and bin means use per-side fitted
values of `y ~ 1 + Z + bin dummies` (R's `lm(y ~ z + factor(bin))`;
computed via least-squares projection, which reproduces R's pivoted-QR
fitted values exactly even under rank deficiency). Gammas are exposed
name-keyed as `covariate_coefficients` - nuisance coefficients, not causal
effects.

**Scope seams (documented):** sampling weights and R's `subset=` are not
parameters (filter the DataFrame; consistent with the estimator's weights
seam); `covs_eval` variants beyond `"mean"`; rendering is an optional lazy
matplotlib `.plot(ax=...)` on the result (numbers are the parity surface -
no ggplot cosmetic parity is claimed). Manual `nbins`/`scale` accept any
positive value (R permissiveness); very large bin counts allocate
proportional jump grids and - with covariates - a dense per-side bin-dummy
matrix (performance seam, no cap imposed). `covariate_coefficients` key
order follows R's processing order: nchar-sorted under `covs_drop=True`,
user-given under `covs_drop=False` (R sorts only inside the drop pipeline;
identical per-name values on full-rank covariates - locked by test).

**Regression tests:** `tests/test_rdplot.py` (24-config golden parity at
rtol=1e-9, paper anchors, quirk locks incl. the reflection quirk via the
esmvpr golden, masspoints remap/check, zero-variance rescue, fallback-ladder
overflow config with integer-only assertions plus a finite-singular
no-downgrade companion, deviation locks for fractional/pair scale, API
contracts, and the matplotlib ImportError path).



---

## ChangesInChanges (CiC)

- **Note:** The class dispatches over two 2x2 distributional estimators via the keyword-only `method=` parameter (`"cic"` default | `"qdid"`, lowercase only - `"CiC"`/`"QDiD"` casing raises). The default encodes Athey-Imbens' recommendation of CiC over QDiD (p. 447). Mode-specific behaviour, easy to misread from the shared signature: the eq. 17 interior-range guard and `q_lower`/`q_upper` are CiC-ONLY (NaN for every `method="qdid"` fit, which has no eq. 17 analogue), and the covariate branch fits quantile regressions in TWO control cells for CiC but THREE for QDiD (both control cells plus treated-pre, with qte's verbatim-ported type-7 / type-1 asymmetry).
- **Note:** The results field carrying that tag is `ChangesInChangesResults.method` (ledger row M-143). It was named `estimator` before 3.9; the old name still reads through a deprecation property until 4.0, and `to_dict()` emits BOTH keys during the 3.9 window so serialized consumers keep working. The rename follows the rule that a parameter and the results field echoing it share a name, and resolves a collision with `AggregationResult.estimator`, which holds a CLASS NAME rather than a method tag.
- **Note:** The deprecation window for that rename covers the READ path, not the CONSTRUCT path: `results.estimator` keeps working (with a `FutureWarning`) and old pickles migrate on load, but `ChangesInChangesResults(..., estimator=...)` raises `TypeError` from 3.9 rather than warning until 4.0. This is deliberate and matches both prior results-field renames - `RegressionDiscontinuityResults.treatment_col` -> `takeup` (M-094) and `ChaisemartinDHaultfoeuilleResults.groups` -> `units` (M-114) - neither of which preserved a deprecated constructor keyword. Results containers are constructed by the library (one call site) and consumed by users, so the read path is what a deprecation window protects; adding a constructor shim would introduce a deprecated surface with no ledger row and diverge from two shipped precedents. Recorded because an automated reviewer reads the immediate `TypeError` as an early removal of a 4.0-scheduled surface.

**Primary source:** Athey, S., & Imbens, G. W. (2006). Identification and Inference in Nonlinear Difference-in-Differences Models. *Econometrica*, 74(2), 431-497. https://doi.org/10.1111/j.1468-0262.2006.00668.x

Full equation-level review (all equation/theorem/page pins below refer to the published version): `docs/methodology/papers/athey-imbens-2006-review.md`. Companion reviews: `melly-santangelo-2015-review.md` (covariates - the qte-style QR branch below is its simplified form; the full MS estimator remains deferred), `callaway-li-oka-2018-review.md` (panel QTT / bootstrap validity machinery, deferred), `ciaccio-2024-review.md` (staggered, deferred).

**Model (Sections 2-3):** two groups x two periods; untreated outcomes generated by `Y^N = h(U, T)` with `h` strictly increasing in a scalar unobservable U (Assumptions 3.1-3.2), `U ⊥ T | G` (Assumption 3.3, time-invariance within groups), and support inclusion `U_1 ⊆ U_0` (Assumption 3.4, relaxable per Corollary 3.1). The counterfactual distribution of the treated group's untreated post-period outcome is `F_{Y^N,11}(y) = F_10(F_00^{-1}(F_01(y)))` (Theorem 3.1, eq. 9). With continuous outcomes the model imposes no testable restrictions and is invariant to monotone transformations of the outcome (p. 437-439).

**Key implementation requirements:**

*Assumption checks / warnings:*
- All four (group, period) cells non-empty (Assumption 5.1(ii)) -> `ValueError`.
- Support check (unconditional fits): treated pre-period outcomes outside the control pre-period range (Assumption 3.4 / Corollary 3.1) -> `UserWarning`; quantile effects remain point-identified only on the eq. (17) interior range. Covariate fits replace this with the conditional-envelope diagnostic (see the covariates block below).
- Heavy ties (> 10% duplicate values within a cell; library heuristic - the paper's Assumption 5.1(iii) is continuity, with no finite-sample ties rule, and this threshold is the library's operationalization of "discrete-looking" for the mixed continuous/discrete edge case) -> `UserWarning` citing the Section 4 bounds deferral: the continuous machinery applied to discrete data silently delivers one endpoint of the Athey-Imbens bounds, not a point estimate.
- Docstring warning (not detectable in a 2x2 design): additive random group-time shocks BIAS CiC - unlike linear DiD where they only complicate inference; testable only with more than two groups/periods via Theorem 6.4 (p. 476; deferred).

*Estimator equations (eqs. 34-36 and 17-18, as implemented):*

    cf_i  = Q1( y01, ECDF_y00(y10_i) )                      counterfactual draws (eq. 36 inner term)
    ATT   = mean(y11) - mean(cf)                            (eq. 36)
    QTE(tau) = Q1(y11, tau) - Q1(cf, tau)                   (eq. 18 plug-in)

where `Q1` is the R type-1 quantile - the paper's eq. (35)/(A.1) inf-based ceiling-order-statistic inverse (`inf{y : F_hat(y) >= q}` = `Y_(ceil(Nq))`, `Q1(., 0)` = sample minimum) - and `ECDF` is the eq. (34) `<=`-semantics empirical CDF. The implementation ports R `quantile.default` type-1 arithmetic exactly, INCLUDING its `4 * .Machine$double.eps` fuzz on the index computation (`diff_diff/changes_in_changes.py::_quantile_type1`); this is where cross-language implementations diverge, and the type-1 micro-fixtures in `benchmarks/data/qte_golden.json` pin it bit-exactly against R.
- **Note:** the estimator is deliberately smoothing-free (p. 451): empirical CDFs and order statistics only, no interpolating quantile definitions anywhere in the CiC pipeline. This yields exact monotone-transform equivariance (locked by `tests/test_methodology_changes_in_changes.py`).

*Interior-range guard (eq. 17 / Theorem 5.3; unconditional fits only):* `q_lower = ECDF_y10(min y00)`, `q_upper = ECDF_y10(max y00)`; requested quantiles outside the open interval `(q_lower, q_upper)` keep their point estimates (qte parity - `qte::CiC()` has no guard) but report NaN inference with a `UserWarning` (Theorem 5.3's asymptotics are interior-only). With covariates the eq. (17) bounds are unconditional-distribution objects and do not apply: `q_lower`/`q_upper` are NaN, no inference overwrite runs (qte applies no guard on its covariate path either), and the conditional-envelope diagnostic below is the support check.

*Covariates (qte `xformla` parity - the Melly-Santangelo 2015 QR pipeline in qte's simplified form):* `fit(covariates=[...])` or trailing formula terms (`"y ~ treated * post + x1"`). Linear quantile regressions `rq(y ~ X)` are fit in the control pre- and post-period cells on qte's fixed internal tau grid `seq(0.01, 0.99, 0.01)` (99 points, hardcoded in qte's `compute.CiC`, not user-configurable); each treated pre-period observation receives its conditional rank `Fhat_{00|X_i}(Y_i)` and imputed counterfactual `y0t_i = Qhat_{01|X_i}(rank_i)` via quantreg `predict.rqs`'s `Fhat`/`Qhat` step-function conventions (ported verbatim, including their `taus[0]` floor and one-step lag); the counterfactual distribution is the empirical distribution of the imputations, and ATT/QTE follow the unconditional arithmetic with type-1 quantiles. Bootstrap replicates refit every per-cell quantile regression (qte's `bootiter` re-runs the whole estimator), so a covariate fit at `n_bootstrap=200` solves ~40k small LPs - tens of seconds at moderate cell sizes, the same cost profile as `qte::CiC`.
- **Note (integration over treated-PRE covariates):** qte imputes per treated pre-period observation and averages over the treated PRE-period covariate distribution; Melly-Santangelo integrate over the treated POST-period covariates (`F_{X|11}`). For panel data with time-invariant covariates they coincide; for repeated cross-sections with composition change they differ. qte's convention is canonical for diff-diff (locked parity direction).
- **Note (linear QR, numeric covariates, no rearrangement):** covariates enter linearly (dummy-encode categoricals - non-numeric columns raise, unlike R's silent `model.matrix` factor expansion); the `Fhat`/`Qhat` step functions use the raw (un-rearranged) QR process, so under quantile crossing the imputation map can be non-monotone - Melly-Santangelo would monotonize, qte does not. Monotone-transform equivariance does NOT extend to the covariate branch (linear-in-covariates QR is not equivariant to nonlinear monotone transforms).
- **Note (conditional-envelope support diagnostic; CiC only):** warns when more than 10% (`_ENVELOPE_SHARE_WARN`, analogous to the ties heuristic) of treated pre-period outcomes fall outside their per-observation conditional quantile envelope - the min/max SPAN of the 99 predicted control pre-period grid quantiles at the observation's covariates, which is exactly the region where the Fhat rank is an extrapolated floor/ceiling plateau (under quantile crossing the span can exceed the tau=0.01/0.99 endpoint columns; the span is the predicate because it matches the step-function semantics). This is the Melly-Santangelo Assumption-4 support/overlap analogue of Athey-Imbens Assumption 3.4. ~2% outside is expected under correct specification (the envelope spans taus 0.01-0.99). Scope is deliberately the rank cell only (c00); the post-period `Qhat` lookup is not separately checked.
- **Note (QDiD monotonicity check moot under covariates):** the footnote-21 warning diagnoses the unconditional additive quantile model; the covariate-path `q0` is a type-1 quantile of the imputed sample and monotone by construction, so the check is skipped on covariate fits.
- **Note (NA handling deviates from qte):** qte silently `na.omit`s inside `rq`; diff-diff extends its fit-level dropna-with-warning to covariate columns and rejects non-finite covariate values, so no row is silently dropped inside the quantile regressions.
- **Deviation from R (solver + exact-tie selection):** the quantile regressions solve the Koenker-Bassett LP with scipy's HiGHS instead of quantreg's Barrodale-Roberts simplex. Both return exact vertex solutions and agree to ~1e-14 when the optimum is unique, but two exact-tie artifacts make bit-level end-to-end agreement unattainable: (i) the check loss can have a non-trivial optimal face (structural with binary covariates), where the two solvers legitimately return different exact minimizers; (ii) R's predicted `Fhat` knots are often EXACTLY tied across adjacent taus (shared interpolating basis) and `predict.rqs` orders ties by tau index, while Python's ~1e-15-different predictions break those ties by value, so a share of conditional ranks lands one grid step away. Both are selections among equally valid solutions. Parity is therefore layered: the step-function conventions and both compositions are pinned BIT-EXACTLY conditioned on R's stored coefficients/predictions (`qr_cases` fixtures), the LP solver is proven exactly optimal at every tau (equal coefficients OR equal check loss to 1e-10), and end-to-end covariate results are gated at empirically calibrated tie-selection bounds (ATT 0.04, per-quantile QTE 0.25). The committed fixtures measure worst 9.3e-3 / 8.1e-2 on the generation platform; a 30-dataset randomized decomposition audit (5 DGP families incl. binary-covariate and ties-heavy cells, 2026-07-13) observed same-mechanism deviations up to 3.5e-2 / 0.40 on adversarial data and verified the full attribution story (composition-on-R-coefficients exact 30/30; 8,910 LP solves all coefficient-matching or two-sided loss-equal; all 1,586 rank flips attributed; no directional bias) - the gates carry cross-platform margin because a different BLAS can flip different ties on the same fixture data, while a real composition bug errs at O(0.5-1). Sharpness floors additionally require most CiC quantiles (>= 10 of 19) and at least one QDiD quantile to match R within 1e-8.
- **Note (QDiD covariate branch, asymmetric quantile types):** qte's covariate QDiD fits QR in THREE cells (both control cells plus treated-pre), takes conditional ranks from the treated pre-period cell's OWN conditional distribution, imputes additively `y0t_i = Y_i + Qhat_{01|X_i}(rank_i) - Qhat_{00|X_i}(rank_i)`, and then mixes quantile types: `q1 = Q7(y11, tau)` (R's default type-7) but `q0 = Q1(y0t, tau)` (type-1 via `quantile.ecdf`). This asymmetry is a qte 1.3.1 implementation fact, ported verbatim. Own-cell ranks make QDiD's covariate path structurally knot-exact (the c10 QR interpolates its own evaluation points), which is why its tie-selection deviations run larger than CiC's.

*Standard errors:*
- Bootstrap-only in v1 (`n_bootstrap=200` default, `seed` param; `n_bootstrap=0` -> joint-NaN inference via `safe_inference`/`safe_inference_batch`).
- **Note (bootstrap is a qte-parity convention, not a paper prescription):** Athey-Imbens 2006 contains no bootstrap - its inference is analytical (influence functions, Theorems 5.1-5.7; deferred). The bootstrap convention matches `qte` 1.3.1 exactly: panel mode resamples units with both periods traveling together; repeated cross-section mode draws a single pooled row resample of the stacked two-period data (unstratified - replicate cell sizes vary, and replicates that empty a cell contribute NaN rows through a >= 50%-finite gate). SEs are replicate SDs (ddof=1); CIs are symmetric normal-approximation `est +/- z_{1-alpha/2} * se`. The closest published validity argument for this style of bootstrap on distributional plug-ins is the exchangeable-bootstrap machinery in Callaway-Li-Oka 2018 (see its review doc).
- **Note (uniform bands at fixed 95%):** `results_.sup_t_crit` ports qte's sup-t critical value (IQR-scaled `sigmahalf` with an SD-floored-at-1e-9 fallback; the 0.95 type-1 quantile of the sup statistics is HARD-CODED in qte independent of `alpha`, and `uniform_bands()` inherits that fixed level - pointwise CIs follow `alpha`, the uniform bands do not). The critical value is computed over ALL grid columns before the interior-range NaN overwrite.
- Point estimates are invariant to `panel`: the estimator uses only the four marginal cell distributions; `panel=True` changes the resampling scheme only (locked by tests).

*Edge cases:*
- Empty cell -> `ValueError` (Assumption 5.1(ii)).
- NA rows in used columns -> dropped with a counting `UserWarning`.
- Panel mode: duplicate (unit, period) rows -> `ValueError`; group indicator varying within unit -> `ValueError`; units missing a period -> dropped with `UserWarning` (matches qte's `makeBalancedPanel`).
- `unit=` supplied with `panel=False` -> documented-ignore (qte `idname` precedent).
- Requested quantiles at/outside the interior range -> NaN inference, finite points, one warning.
- Bootstrap with < 50% finite replicate rows -> all SEs and `sup_t_crit` NaN + failure-rate warning (`warn_bootstrap_failure_rate`).

**Deferred (reviewed, documented - see the paper-review docs for full scoping):**
- The FULL Melly-Santangelo (2015) covariate estimator: monotonized integrated-indicator conditional CDFs (CFG 2010 rearrangement), integration over the treated post-period covariates (`F_{X|11}`), exchangeable bootstrap with variance-weighted KS uniform bands, tail trimming, and the pre-period time-invariance specification test. The implemented covariate branch above is qte's simplified form of the same pipeline. The Athey-Imbens Section 5.1 parametric residual route (OLS residualization) is also not implemented.
- Discrete-outcome bounds and DCIC point identification (Sections 4, 5.2; Kranker's Stata `cic` implements them).
- Analytical SEs (Theorems 5.1-5.3 influence functions; panel Theorems 5.5-5.7; Appendix B covariances).
- Multiple groups/periods (Section 6) - `ecic` (R) is the staggered event-study CiC lineage; Ciaccio's copula-based staggered distributional DiD is a distinct method (reviewed separately, do not conflate).
- Treatment on the controls (Theorem 3.2, group-label exchange).

**Reference implementation(s):**
- R: `qte::CiC()` (v1.3.1, pinned; quantreg 6.1 pinned too - the covariate fixtures embed quantreg's `rq`/`predict.rqs` behavior). Golden fixtures: `benchmarks/R/generate_qte_golden.R` -> `benchmarks/data/qte_golden.json`; parity tests: `tests/test_changes_in_changes_parity.py` (unconditional point ATT/QTE at atol=1e-10 across 4 scenarios x panel/RCS, type-1 micro-fixtures at atol=0; covariate conventions at atol=0 via the `qr_cases` block conditioned on R's stored coefficients/predictions, LP optimality per tau, end-to-end covariate results at the tie-selection bounds documented above; seeded-R SE blocks compared statistically - generated with `cores=1`, since qte's default `cores=2` forks the bootstrap and is NOT reproducible even under `set.seed`).
- Stata: `cic` (Kranker) - analytical SEs + discrete bounds (deferred scope).

### QDiD (quantile DiD comparison estimator)

- **Note:** As of 3.9 this estimator is selected with `ChangesInChanges(method="qdid")`; the standalone `QDiD` class is deprecated (ledger row M-015) and removed in 4.0. Only the CLASS SPELLING is deprecated - the estimator is fully supported and selecting it emits no advisory warning. The existing footnote-21 non-monotonicity `UserWarning` and `practitioner.py`'s "Prefer ChangesInChanges over QDiD" step already carry Athey-Imbens' p. 447 recommendation at the moments a user can act on it; a third warning on every `method="qdid"` fit would blur which of the class/method is dying. Both engines and every number are unchanged by the merge.

**Primary source:** Athey & Imbens (2006), Section 3.3 (pp. 446-447) - the paper formalizes QDiD as the natural comparison estimator and recommends CiC over it: QDiD's justifying model (eq. 22) is not invariant to monotone rescaling of the outcome, forces `U ⊥ (G, T)` (identical unobservable distributions in all four cells), and places testable restrictions on the data (footnote 21 - the implementation warns when the implied counterfactual quantile function is non-monotone on the requested grid; unconditional fits only, see the covariates block above).

*Estimator equations (as implemented, matching `qte::QDiD()` v1.3.1 exactly):*

    QTE(tau) = Q7(y11, tau) - [ Q7(y10, tau) + Q7(y01, tau) - Q7(y00, tau) ]
    ATT      = mean(y11) - ( mean(y10) + mean(Q7(y01, r)) - mean(Q7(y00, r)) ),   r = ECDF_y10(y10)

where `Q7` is the R default type-7 (linear-interpolation) quantile (numpy `method="linear"`).
- **Note (finite-sample deviation from the paper's k^QDID transformation):** the paper's p. 447 display defines QDiD through the transformation `k^QDID(y) = y + F_01^{-1}(F_10(y)) - F_00^{-1}(F_10(y))`; qte 1.3.1 instead computes the additive quantile-DiD above with type-7 quantiles, and its ATT evaluates the control-group quantile functions at the treated pre-period's own-sample ranks. The two constructions are population-equivalent (and QDiD's mean effect equals standard DiD's ATT under continuity, p. 447 - locked as a large-N tolerance test) but are different finite-sample estimators. diff-diff implements the qte form as canonical per the locked parity direction (2026-07-12); the paper-exact transformation form is not implemented.
- No asymptotic theory for QDiD exists in the paper; bootstrap inference (identical machinery to CiC above) is a library/qte convention. No interior-range guard applies (eq. 17 has no QDiD analogue): `q_lower`/`q_upper` are NaN on QDiD results.
- Covariates: see the CiC covariates block above (three-cell QR, own-cell conditional ranks, additive imputation, and the verbatim-ported asymmetric Q7/Q1 quantile-type pair).

**Reference implementation(s):**
- R: `qte::QDiD()` (v1.3.1, pinned) - same fixture/test infrastructure as CiC.

---

# Diagnostics and Sensitivity

## PlaceboTests

**Primary source:** [Bertrand, M., Duflo, E., & Mullainathan, S. (2004). How Much Should We Trust Differences-in-Differences Estimates? *The Quarterly Journal of Economics*, 119(1), 249-275.](https://doi.org/10.1162/003355304772839588). Paper review on file: `docs/methodology/papers/bertrand-duflo-mullainathan-2004-review.md`.

**Module:** `diff_diff/diagnostics.py`

**Scope:** A battery of placebo / randomization-inference diagnostics for the parallel-trends assumption, built on the base 2×2 `DifferenceInDifferences`. BDM (2004) introduce the **placebo-law experiment** — randomly assign a fake treatment date and/or fake treated group, estimate the DD, and check whether a (necessarily spurious) "effect" is significant more than ~5% of the time. These are **diagnostics**, not estimators, and are distinct from the per-estimator placebo/LOO surfaces documented elsewhere (SyntheticDiD donor leave-one-out per ADH 2015 §4, HAD pre-tests, DCDH placebo). Public API: `run_placebo_test` (dispatcher) → `placebo_timing_test` / `placebo_group_test` / `permutation_test` / `leave_one_out_test`, plus `run_all_placebo_tests` and `PlaceboTestResults`.

**Key implementation requirements:**

*The four diagnostics:*
- **`placebo_timing_test` (fake timing):** restrict to pre-treatment periods, assign a fake post indicator (`time >= fake_treatment_period`), and run the DD on the real treated vs control units. A significant effect flags differential pre-trends. Requires ≥2 pre-periods; a `fake_treatment_period` inside `post_periods` raises `ValueError`.
- **`placebo_group_test` (fake group):** designate control units as fake-treated. With the optional `treatment=` column, units that are *ever* real-treated are dropped first (placebo on never-treated units only, uncontaminated by the real effect); without it, the caller must pass control-only data. Degenerate designs (all fake-treated dropped, or no controls remain) raise `ValueError`; a fake-treated unit that is itself real-treated emits a `UserWarning`. Via the `run_placebo_test` dispatcher (which always has the `treatment` column) the `fake_group` path filters ever-treated units by default.
- **`permutation_test` (randomization inference):** randomly reassign the treated-group label across units (BDM placebo-law over a fixed outcome set; the randomization-inference link is BDM fn 11, citing Rosenbaum 1996) and form the null distribution of the DD estimate.
- **`leave_one_out_test`:** drop each treated unit, refit, and report the per-drop ATTs (single-unit sensitivity).

*Randomization-inference p-value (Phipson & Smyth 2010):*

```
p = (1 + #{ |ATT*_b| >= |ATT_obs| }) / (B + 1)
```

where `ATT*_b` are the `B` valid permutation estimates and the `+1` includes the observed statistic in both numerator and denominator (intrinsic floor `1/(B+1)`). Assignments are drawn independently each iteration (Monte-Carlo sampling **with replacement** from the assignment space), so this is the Phipson & Smyth (2010) **valid but slightly conservative** RI p-value — *not* an exact finite-sample value. The term **exact** is reserved for the full enumeration of all `C(N, n_treated)` assignments (observed included), `#{|ATT*| >= |ATT_obs|} / total`, to which the sampled value converges as `B → ∞` (this enumeration is the R-parity reference).

*Standard errors / inference (stated honestly):*
- `placebo_timing_test` / `placebo_group_test` surface the underlying `DifferenceInDifferences` estimator's own jointly-computed inference (HC1 default via `safe_inference`).
- `permutation_test` does **not** use `safe_inference`: the p-value is the count-based RI value above, the confidence interval is the **percentile interval of the null (permutation) distribution — not an effect CI**, and `t_stat = original_att / se` is computed individually (`se` = std of the null distribution). `placebo_effect` reports the **mean of the null distribution** (≈0), with `original_effect` holding the observed ATT.
- `leave_one_out_test` uses `safe_inference` (t-distribution, `df = n_valid − 1`), but its `se` is the **dispersion (sample std) of the leave-one-out ATTs — a sensitivity spread, not a design-based jackknife SE**; the per-unit `leave_one_out_effects` dict is the primary output and `t_stat`/`p_value`/CI are heuristic.

*Edge cases:*
- **Permutation NaN-decoupling (deliberate):** the RI p-value is count-based and stays finite even when the permutation `se` is degenerate (`se == 0` for identical permutations, or `se` NaN at `n_valid == 1`), in which case `t_stat` is NaN. This intentionally departs from the bootstrap-NaN contract (non-finite SE → full NaN tuple), because the RI p-value does not depend on `se` (BDM fn 12: the placebo reference distribution is not standard normal, so the count-based RI p-value — not an `se`-based statistic — is the valid one). **Note:** intentional contract; see `tests/test_methodology_placebo.py::TestPlaceboInferenceContracts`.
- NaN inference for undefined statistics:
  - `permutation_test`: t_stat is NaN when the permutation SE is zero/degenerate (the p-value remains valid — see decoupling above).
  - `leave_one_out_test`: t_stat, p_value, CI are NaN when the LOO SE is zero (all LOO effects identical) or `< 2` valid effects.
  - **Note**: Defensive enhancement matching CallawaySantAnna NaN convention.
- Fail-closed: `permutation_test` and `leave_one_out_test` raise `RuntimeError` if **all** refits fail; `permutation_test` emits a `UserWarning` when `> 10%` of permutations fail.

**Reference implementation(s):**
- No single canonical R/Stata package implements the DiD placebo *battery* as one command (BDM's own code is custom). R parity is anchored by **exact enumeration** in base R (`combn`) of the full randomization distribution, with an optional `ri2`/`coin` convention cross-check (guarded by `requireNamespace`, not a committed dependency). Generator: `benchmarks/R/generate_placebo_golden.R`; golden: `benchmarks/data/placebo_golden.json` (+ `placebo_test_panel.csv`). The deterministic `leave_one_out_test` / `placebo_group_test` / observed ATT match R exactly (`atol≈1e-10`); the sampled `permutation_test` matches the exact value within Monte-Carlo tolerance.

**Deviations:**
- **Note (permutation inference is not `safe_inference`):** the permutation path uses the RI p-value + a null-distribution percentile interval (not an effect CI) and a null-mean `placebo_effect`. This is the standard randomization-inference convention and deliberately differs from the project-wide `safe_inference` t-based inference (which assumes a sampling-distribution SE the permutation test does not have).
- **Note (leave-one-out spread):** `leave_one_out_test`'s reported `se`/`t_stat`/`p_value`/CI summarize the dispersion of the per-drop ATTs (a robustness signal), not a design-based jackknife standard error; the per-unit effects are the primary output.
- **Note (BDM scope):** BDM (2004) is primarily about serial-correlation-robust *standard errors* (parametric AR, block bootstrap, cluster/arbitrary VCV, time aggregation). Those inference corrections are out of scope for this diagnostic surface — diff-diff's cluster-robust SE and bootstrap paths cover them under the relevant estimators. This entry covers only the placebo-law / randomization-inference diagnostics that BDM motivate.

**Requirements checklist:**
- [x] RI p-value `(1+count)/(B+1)` (sampled) converges to the exact enumeration `count/total`: `tests/test_methodology_placebo.py::TestPlaceboRandomizationInference`
- [x] R parity: exact enumeration + observed ATT at `atol=1e-12`; deterministic LOO / fake-group at `atol=1e-10`; sampled permutation within MC tolerance: `TestPlaceboParityR` (skip-guarded)
- [x] Fake-timing detects differential pre-trends; null under parallel trends; pre-data only: `TestPlaceboFakeTiming`
- [x] Fake-group never-treated `treatment` filter + degenerate `ValueError` + misuse `UserWarning`; backward-compatible without `treatment`: `TestPlaceboFakeGroup`
- [x] Permutation NaN-decoupling + fail-closed `RuntimeError`: `TestPlaceboInferenceContracts`
- [x] Functional coverage (dispatch routing, zero-SE / `<2`-LOO NaN-inference): `tests/test_diagnostics.py`

---

## BaconDecomposition

**Primary source:** [Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing. *Journal of Econometrics*, 225(2), 254-277.](https://doi.org/10.1016/j.jeconom.2021.03.014). Paper review on file: `docs/methodology/papers/goodman-bacon-2021-review.md`.

**Scope:** Decomposes the two-way fixed-effects DD (TWFEDD) estimator in Equation (2) when treatment timing varies across units. Theorem 1 expresses `β̂^DD` as a **positively-weighted average of all possible 2x2 DD estimators** in the data, with weights summing to 1. The decomposition is a **diagnostic tool**, not a treatment-effect estimator: it explains *which comparisons drive* the TWFEDD coefficient and *why* the estimator can fail to identify an interpretable causal parameter when treatment effects vary over time.

**Key implementation requirements:**

*Assumption checks / warnings:*
- Requires variation in treatment timing (staggered adoption)
- Always-treated units (`first_treat <= min(time)`, excluding the never-treated sentinels `0` and `np.inf`; per paper footnote 11 with a library-convention extension on the first-period boundary case, see `**Deviation (first-period boundary extension)**` below) are automatically remapped to the `U` (untreated) bucket with a `UserWarning`; see the `**Note (always-treated remap)**` below for the full ordered-time / sentinel contract
- Unbalanced panels are accepted with a `UserWarning`; the paper's Appendix A proof assumes balanced panels
- Falls back to timing-only comparisons when no never-treated units are present (no untreated group → `s_{kU}` terms drop, weights rescale to sum to 1; **VWCT and ΔATT can still bias the result** — see paper Eqs. 14-15)

*Estimator equation (Theorem 1, Equation 10a):*

```
β̂^DD = Σ_{k ≠ U} s_{kU} · β̂_{kU}^{2x2}
      + Σ_{k ≠ U} Σ_{ℓ > k} [ s_{kℓ}^k · β̂_{kℓ}^{2x2,k} + s_{kℓ}^ℓ · β̂_{kℓ}^{2x2,ℓ} ]
```

The three 2x2 estimators (Eqs. 10b-d):
```
β̂_{kU}^{2x2}   = (ȳ_k^POST(k)   - ȳ_k^PRE(k))    - (ȳ_U^POST(k)   - ȳ_U^PRE(k))         (Eq. 10b)
β̂_{kℓ}^{2x2,k} = (ȳ_k^MID(k,ℓ) - ȳ_k^PRE(k))    - (ȳ_ℓ^MID(k,ℓ) - ȳ_ℓ^PRE(k))         (Eq. 10c)
β̂_{kℓ}^{2x2,ℓ} = (ȳ_ℓ^POST(ℓ)  - ȳ_ℓ^MID(k,ℓ)) - (ȳ_k^POST(ℓ)  - ȳ_k^MID(k,ℓ))        (Eq. 10d)
```

Comparison-type labels in `BaconDecompositionResults.comparisons`:
- `"treated_vs_never"` ↔ Eq. 10b
- `"earlier_vs_later"` ↔ Eq. 10c (k = early = treated; ℓ = late = control during MID)
- `"later_vs_earlier"` ↔ Eq. 10d (ℓ = late = treated; k = early = already-treated control)

*Weight construction (Eqs. 7-9 for variances, 10e-g for weights):*

Fixed-effects-adjusted treatment-dummy variances:
```
V̂_{kU}^D     = n_{kU}(1 - n_{kU}) · D̄_k(1 - D̄_k)                              (Eq. 7)
V̂_{kℓ}^{D,k} = n_{kℓ}(1 - n_{kℓ}) · (D̄_k - D̄_ℓ)/(1 - D̄_ℓ) · (1 - D̄_k)/(1 - D̄_ℓ)   (Eq. 8)
V̂_{kℓ}^{D,ℓ} = n_{kℓ}(1 - n_{kℓ}) · D̄_ℓ/D̄_k · (D̄_k - D̄_ℓ)/D̄_k                  (Eq. 9)
```

Decomposition weights:
```
s_{kU}   = ((n_k + n_U)^2 · V̂_{kU}^D)        / V̂^D                                  (Eq. 10e)
s_{kℓ}^k = ((n_k + n_ℓ)(1 - D̄_ℓ))^2 · V̂_{kℓ}^{D,k} / V̂^D                            (Eq. 10f)
s_{kℓ}^ℓ = ((n_k + n_ℓ) · D̄_k)^2 · V̂_{kℓ}^{D,ℓ}    / V̂^D                            (Eq. 10g)
```

Where `n_k` is the sample share of timing group `k`, `n_{kℓ} = n_k / (n_k + n_ℓ)`, and `D̄_k = (T - k + 1)/T` is the share of periods group `k` spends treated. Weights are **strictly positive** and sum to 1 (Theorem 1).

*Standard errors:*
- The decomposition is a deterministic algebraic identity; standard errors are not the paper's focus. The point estimates and weights are exact given the data, and on **balanced** panels `β̂^DD` from the decomposition matches the OLS coefficient from TWFEDD to machine precision under `weights="exact"`. The machine-precision claim does **not** extend to unbalanced panels (see edge cases below).
- Inference for the TWFEDD coefficient itself is typically cluster-robust at the unit level.

*Edge cases:*
- Continuous treatment: not supported, requires binary
- Single treatment time: K=1 with U is valid (only `treated_vs_never` terms); K=1 without U has no decomposition (zero variation in `D`).
- `D̄_k → 0` or `D̄_k → 1`: subsample treatment variance goes to zero, that timing group contributes zero weight
- Always-treated units: see `**Note (always-treated remap)**` below

**Reference implementation(s):**
- R: `bacondecomp::bacon()` (CRAN). Parity script at `benchmarks/R/generate_bacon_golden.R`; goldens committed at `benchmarks/data/r_bacondecomp_golden.json` (generated against `bacondecomp` 0.1.1 + R 4.5.2). Parity validated at `atol=1e-6` via `tests/test_methodology_bacon.py::TestBaconParityR` (4 tests: TWFE coefficient + weights-sum match across 3 fixtures; per-component estimate + weight parity locked on the 2 non-remap fixtures and on the 6 timing-vs-timing rows of `always_treated_remapped`; the U-bucket convention divergence on `always_treated_remapped` is pinned by a dedicated fold-back test).
- Stata: `bacondecomp` (SSC). Authors: Goodman-Bacon, Goldring, Nichols (2019).

**Requirements checklist:**
- [x] Three comparison types: treated_vs_never, earlier_vs_later, later_vs_earlier
- [x] Weights sum to 1 (machine precision under `weights="exact"` on **balanced** panels; see PR-B audit)
- [x] TWFE coefficient = weighted sum of 2×2 estimates (machine precision under `weights="exact"` on **balanced** panels)
- [x] Visualization shows weight vs. estimate by comparison type
- [x] Always-treated remap to U per Goodman-Bacon (2021) footnote 11 (PR-B audit)
- [x] Hand-calculable Theorem 1 verification: `tests/test_methodology_bacon.py::TestBaconHandCalculation` (7 tests, atol=1e-10)
- [x] R `bacondecomp::bacon()` parity at atol=1e-6 (3 fixtures; TWFE coefficient + weights-sum match across all 3; per-component parity locked on the 2 non-remap fixtures and on the 6 timing-vs-timing rows of `always_treated_remapped`; the U-bucket fold-back is asserted by a dedicated `test_always_treated_remapped_fold_back_matches_r` — see `**Note (R parity convention divergence)**` below)
- [x] Survey design support (Phase 3): weighted cell means, weighted within-transform, weighted group shares
- **Note (weight modes):** `weights="exact"` (default, paper-faithful Eqs. 7-9 + 10e-g) vs `weights="approximate"` (simplified variance, opt-in for speed-sensitive diagnostic loops). The PR-A paper review (#451) and PR-B audit established `"exact"` as the default to match R `bacondecomp::bacon()` and the paper's Theorem 1 contract; R parity is validated at `atol=1e-6` (see `**Note (R parity convention divergence)**` below for the one structural convention difference). Hand-calculation + TWFE-vs-weighted-sum identity hold at `atol=1e-10`. The approximate path is retained for backward compatibility; numerical output may differ from R.
- **Note (always-treated remap):** Units whose `first_treat` is at or before the first observable period (`first_treat <= min(time)`, excluding the never-treated sentinels `0` and `np.inf`) are automatically remapped to the `U` bucket via an internal column (`__bacon_first_treat_internal__`) with a `UserWarning` — per paper footnote 11 (with a library boundary extension on `first_treat == min(time)`; see `**Deviation (first-period boundary extension)**` below). Detection uses ordered-time logic on the **time axis**, so panels whose `time` column has negative or zero-crossing labels (e.g. event-time `time ∈ [-2,..,3]`) are handled correctly: a cohort at `first_treat=-1` on such a panel is a valid timing group; a cohort at `first_treat=-3` is remapped to U. The user's original `first_treat` column on the input `data` frame is preserved unchanged. The count of remapped units is surfaced via `BaconDecompositionResults.n_always_treated_remapped`. **Sentinel restriction:** `first_treat ∈ {0, np.inf}` is reserved as the never-treated marker and is not configurable today; a real treatment cohort with `first_treat == 0` would be folded into `U` and should be re-labeled to a non-sentinel value before fitting. The `0` reservation applies to `first_treat` only, not to `time`.
- **Note (Bacon survey diagnostic):** Bacon decomposition with survey weights is diagnostic; exact-sum guarantee holds at machine precision under `weights="exact"` **on balanced panels**. `weights="exact"` requires within-unit-constant survey columns (approximate path accepts time-varying weights).
- **Note (R parity convention divergence on always-treated):** R `bacondecomp::bacon()` keeps `first_treat=1` (the always-treated cohort) as a separate timing cohort and emits an additional comparison type `Later vs Always Treated` (cohort k vs the always-treated cell) alongside the standard `Treated vs Untreated` row. Python's footnote-11 convention remaps these units to the `U` bucket and folds those R-side rows into a single `treated_vs_never` cell per treated cohort. The aggregate (TWFE coefficient + sum of weights) is invariant to this re-bucketing — Theorem 1's identity holds identically because the U bucket's total weight gets re-allocated across nested 2x2 cells but the total weight on `{cohort_k vs U}` is the same. The per-component breakdown, however, differs structurally between the two conventions. The R parity test (`tests/test_methodology_bacon.py::TestBaconParityR::test_component_estimates_match_r`) asserts per-component parity at `atol=1e-6` on the 2 fixtures without always-treated (`uniform_3groups_with_never_treated`, `two_groups_no_never_treated`) AND on the 6 timing-vs-timing rows of `always_treated_remapped` — the carve-out is narrowed to U-bucket rows only (R's `Later vs Always Treated` rows canonicalize to `treated_vs_never` and are dropped alongside the matching Python rows). The R→Python U-bucket fold-back is pinned separately by `test_always_treated_remapped_fold_back_matches_r`, which aggregates R's split `Later vs Always Treated` + `Treated vs Untreated` rows per treated cohort and asserts the combined weight + weight-averaged estimate match Python's single `treated_vs_never` cell at `atol=1e-6`. Aggregate parity (`test_twfe_coef_matches_r`, `test_weights_sum_matches_r`) is locked across all 3 fixtures.
- **Deviation (first-period boundary extension on always-treated remap):** Paper footnote 11 (Goodman-Bacon 2021) uses the strict inequality `t_i < 1` (units treated *before* the first observable period) for the always-treated bucket. The library applies the **inclusive** `first_treat <= min(time)` rule, which additionally folds units treated *at* the first observable period (`first_treat == min(time)`) into `U`. This is a library boundary convention, not a paper-faithful rule: such units have no untreated cell in the observed panel and so cannot contribute to any 2x2 DD as a treated cohort, so folding them into the U bucket mirrors the always-treated handling rather than dropping them silently. R `bacondecomp::bacon()` does not apply this boundary fold-back — it keeps `first_treat == min(time)` cohorts in their own bucket and emits `Later vs Always Treated` comparisons (see the **Note (R parity convention divergence on always-treated)** above for how the parity tests handle the resulting structural breakdown difference; aggregate Theorem 1 identity remains invariant). When `min(time)` is strictly greater than 1 (no first-period-treated cohorts), the library rule reduces to the paper's strict rule and the two conventions coincide.
- **Deviation (unbalanced-panel library extension):** Unbalanced panels are accepted with a `UserWarning` ("Unbalanced panel detected. Bacon decomposition assumes balanced panels. Results may be inaccurate."). Goodman-Bacon (2021) Appendix A's proof assumes a balanced panel; under unbalance, the Theorem 1 identity holds only approximately. The decomposition still returns finite, well-defined outputs but `weights="exact"` does NOT achieve the machine-precision algebraic identity that the balanced-panel claims above describe.

---

## HonestDiD

**Primary source:** [Rambachan, A., & Roth, J. (2023). A More Credible Approach to Parallel Trends. *Review of Economic Studies*, 90(5), 2555-2591.](https://doi.org/10.1093/restud/rdad018)

**Key implementation requirements:**

*Assumption checks / warnings:*
- Requires event-study estimates with pre-treatment coefficients
- Warns if pre-treatment coefficients suggest parallel trends violation
- M=0 for Delta^SD: enforces linear trend extrapolation (not exact parallel trends)
- **Note (positional restriction geometry, 3.9):** the Delta^SD/Delta^RM
  restriction matrices are built POSITIONALLY over the concatenated
  declared pre/post coefficient lists, assuming one chronological pre/post
  boundary. The native `MultiPeriodDiDResults` route does NOT validate a
  declared partition against that assumption - a non-suffix
  `post_periods` (e.g. `[2, 5]` on periods 0-5) or a non-last-pre
  `reference_period` is accepted and produces bounds whose restriction
  system does not match the equations above (a pre-existing limitation;
  the transform-or-reject fix is the DEFERRED.md row). The 3.9 TWFE
  event-study CALENDAR container route fails closed instead: it rejects
  non-chronological declared partitions outright (row M-010).
  - **Note (string calendar labels):** chronology on the calendar route is
    `sorted()` order - verifiable for numeric/Period/Timestamp labels, but
    for STRING labels lexical order is only ASSUMED chronology (unpadded
    numeric suffixes sort `'c10'` before `'c2'`), and a mismatch would
    silently shift the positional `l_vec` target. The calendar route emits
    a `UserWarning` on string labels recommending orderable label types.
    First-party surfaces are internally CONSISTENT either way - the
    estimator sorted its calendar the same way at fit time, so the fit's
    own ordering (reference selection, partition suffix semantics, plot
    order) already reflects the same assumption; the warning marks the
    input-type ambiguity, not a route divergence.

*Restriction classes (Equations 8, Section 2.3):*

Delta^SD(M) — Smoothness (second differences, all periods):
```
Δ^SD(M) = {δ : |(δ_{t+1} − δ_t) − (δ_t − δ_{t-1})| ≤ M, for all t}
```
with δ_0 = 0 at the pre-post boundary. M=0 enforces linear trends.

Delta^RM(M̄) — Relative magnitudes (post-treatment first differences):
```
Δ^RM(M̄) = {δ : |δ_{t+1} − δ_t| ≤ M̄ × max_{s<0} |δ_{s+1} − δ_s|, for all t ≥ 0}
```
Post-treatment consecutive first differences bounded by M̄ × max pre-treatment first difference. Union of polyhedra (one per max location).

*Identified set (Equations 5-6):*
```
θ^lb = l'β_post − max{l'δ_post : δ ∈ Δ, δ_pre = β_pre}
θ^ub = l'β_post − min{l'δ_post : δ ∈ Δ, δ_pre = β_pre}
```
CRITICAL: δ_pre = β_pre pins pre-treatment violations to observed coefficients. Solved via LP (scipy.optimize.linprog).

*Inference (Sections 3.2, 4.1):*
- Delta^SD: Optimal FLCI — a nested convex program (inner: minimize worst-case bias at a fixed estimator SD `h`; outer: minimize the half-length over `h`) that matches R `HonestDiD::findOptimalFLCI`'s optimal affine estimator, using the folded-normal quantile cv_α(bias/se) (Equation 18). See the "Δ^SD FLCI optimizer" note below for the algorithm + R parity. When `df_survey` is present, uses folded non-central t (`scipy.stats.nct`) instead of folded normal; `df_survey=0` → NaN inference. For M=0, uses `_get_critical_value(alpha, df)` (standard t/z).
- Delta^RM: Paper recommends ARP conditional/hybrid confidence sets (Equations 14-15, κ = α/10). Currently uses **naive FLCI** unconditionally (conservative — wider CIs, valid coverage). ARP infrastructure exists but is disabled.
- **Note (deviation from R):** Delta^RM CIs use naive FLCI (`lb - z*se, ub + z*se`) instead of the paper's ARP hybrid. R's `HonestDiD` package implements full ARP conditional/hybrid. Our naive FLCI is conservative (wider, valid coverage) but does not adapt to the length of the identified set. ARP implementation deferred (see DEFERRED.md).
- **Note:** `method="combined"` (Delta^SDRM) uses naive FLCI on the intersection of Delta^SD and Delta^RM bounds. The paper proves FLCI is NOT consistent for Delta^SDRM (Proposition 4.2). The paper recommends ARP hybrid for non-SD restriction classes. This is a known conservative approximation; a runtime UserWarning is emitted.

*Standard errors:*
- Inherits Σ̂ from underlying event-study estimation
- Sensitivity analysis reports identified set bounds and confidence sets

*Edge cases:*
- M=0 for Δ^SD: linear extrapolation, point identification, FLCI near-optimal
- M̄=0 for Δ^RM: post-treatment first differences = 0, point identification
- Breakdown point: smallest M where CI includes zero
- Negative M: not valid (constraints become infeasible)
- **Note:** Δ^SD **empty estimated identified set** (the observed pre-trend's curvature exceeds M, so the `δ_pre = β_pre`-pinned identified-set LP is infeasible → `lb`/`ub` = NaN) does **not** suppress the FLCI. The optimal FLCI is an affine estimator whose worst-case bias is taken over `δ ∈ Δ^SD(M)` treating β as random, so it is well-defined given (Σ, M) regardless of whether the realized `β_pre` lies in Δ; R's `HonestDiD::createSensitivityResults` returns it in exactly this case. `_compute_smoothness_bounds` therefore returns the finite FLCI with `lb`/`ub` = NaN (empty id-set), matching R (prior behavior NaN-propagated the whole result, silently yielding no inference on high-curvature pre-trends). The naive fallback FLCI (diagonal Σ only) still requires finite id-set bounds and returns NaN CI when infeasible. Guarded by `test_infeasible_smoothness_fit_returns_flci_with_empty_idset`.
- **Note (Δ^SD FLCI optimizer — R-faithful nested algorithm; B2b resolved):** The optimal
  FLCI is a faithful port of R `HonestDiD::findOptimalFLCI`'s **nested convex program**. For a
  fixed estimator standard deviation `h`, an inner problem minimizes the worst-case bias over
  `Δ^SD(M)` subject to `sd(estimator) ≤ h` — a smooth convex QCQP over the slope weights `w`
  (variables `[U; w]`, `U ≥ |cumsum(w)|`, one second-order-cone `var(w) ≤ h²`, linear-trend
  neutrality `Σw = Σ_s s·l_s`), solved with `scipy.optimize` SLSQP (no cvxpy). An outer 1-D
  search over `h ∈ [hMin, h0]` (a warm-started grid zoom) minimizes the half-length
  `cv_α(M·bias(h)/h)·h`. The worst-case bias uses R's exact **closed form**
  `M·(constant + Σ_i |cumsum(w)_i|)` (a direct port of R's `.createObjectiveObjectForBias`), so
  the inner objective is linear in the abs-linearized weights; `hMin`/`h0` are the min-variance
  and min-bias SDs (R's `.findLowestH` / `.findHForMinimumBias`). For **nonnegative (averaging)**
  `l_vec` — the standard case, including the default first-post-period basis vector and equal
  weights — this closed form equals the general LP oracle `_compute_worst_case_bias` (max |v'δ|
  over Δ^SD) to machine precision. For a **signed / contrast** `l_vec` (e.g. `[1, -1]`) R's closed
  form is *conservative* relative to the exact LP-oracle bias (a shared R property, not a
  diff-diff bug); the FLCI still matches R (verified to ~1e-6 for `l_vec=[1,-1]`,
  `test_signed_contrast_lvec_matches_r`), because we port R's objective exactly. This matches R's
  optimal FLCI **center + half-length + optimalVec** to ~1e-3 (median ~1e-5) across a stress grid
  (`num_pre × num_post × AR(1) ρ × M`); `TestHonestFLCIParityR`, golden
  `benchmarks/data/honest_flci_golden.json`. The prior flat Nelder-Mead optimizer drifted from
  R's center by up to ~9% at intermediate M (widths matched, coverage unaffected); the nested
  structure removes that drift (SE-audit B2b). A derivative-based outer solve is not viable: the
  min-bias inner solutions are degenerate (most `cumsum(w)` components sit at the abs-value
  kink), so the SD-constraint multiplier is not recoverable from `sign(Lw)`, and the width
  surface is near-flat at the optimum.
- **Deviation from R:** diff-diff uses an **analytical** folded-normal critical value
  (`_cv_alpha`, `scipy.stats`) in place of R's Monte-Carlo `.qfoldednormal` (10⁶ draws, seed 0)
  — strictly more accurate (noise-free). Against **override-R** (R with the MC quantile replaced
  by the analytical one, so both solve the same deterministic outer problem) the center matches
  to ~1e-3; against **stock** MC-R the center matches to ~1.4e-2, the residual being R's own
  simulation noise on the near-flat width surface. The committed golden stores both tiers.
- **Note:** Phase 7d: survey variance support. When input results carry `survey_metadata` with `df_survey`, Delta^SD smoothness uses folded non-central t critical values (`scipy.stats.nct`); Delta^RM and naive FLCI paths use `_get_critical_value(alpha, df)` (standard t-distribution). `df_survey=0` → NaN inference. CallawaySantAnnaResults stores `event_study_vcov` (full cross-event-time VCV from IF vectors), which HonestDiD uses instead of the diagonal fallback. For replicate-weight designs, the event-study VCV falls back to diagonal (multivariate replicate VCV deferred).
- **Note (deviation from R):** When HonestDiD receives bootstrap-fitted CallawaySantAnna results (`n_bootstrap > 0`), the full event-study covariance is unavailable (cleared to prevent mixing analytical VCV with bootstrap SEs). HonestDiD falls back to `diag(se^2)` from the bootstrap SEs with a UserWarning. R's `honest_did.AGGTEobj` computes a full covariance from the influence function matrix; implementing bootstrap event-study covariance is deferred. For full covariance structure in HonestDiD, use analytical SEs (`n_bootstrap=0`).
- **Note (deviation from R):** When CallawaySantAnna results are passed to HonestDiD, `base_period != "universal"` emits a warning but does not error. R's `honest_did::honest_did.AGGTEobj` requires universal base period. Our implementation warns because the varying-base pre-treatment coefficients use consecutive comparisons (not a common reference), which changes the parallel-trends restriction interpretation.
- **Note (common-reference guard — gapped universal grids):** CallawaySantAnna `base_period="universal"` fits record `reference_event_times` — the distinct per-cohort positional-base event times (also carried on the `EventStudyResults` container). On a gapped time grid the positional bases land at different event times, and a cohort's base can OVERLAP another cohort's estimated horizon, where the aggregated row is a real estimate and NO reference-only row marks the anchor — `is_reference` cannot see it. When the field carries more than one entry, HonestDiD raises on BOTH input routes (native results and container) rather than returning bounds over coefficients normalized against different bases: Rambachan-Roth's `δ₀ = 0` normalization is defined around one common reference. PreTrendsPower applies the same guard (its hypothesized violation `δ` is likewise defined relative to one reference). Varying-base fits carry `reference_event_times=None` (no constant per-cohort reference exists) and are covered by the varying-base warnings instead. Universal-base inputs WITHOUT the field are never failed open: the native route re-derives the bases from the materialized reference cells (covers pre-3.9 pickles), and a provenance-less universal container (hand-built; producer-built containers always record the field) warns fail-safe that the common reference cannot be verified. Container admission covers CS- and Stacked-sourced containers (row M-024): StackedDiD threads `base_period="universal"` and a singleton `reference_event_times` by construction (one omitted reference per fit), needs `kappa_pre >= 2` for estimated pre-periods, and HonestDiD additionally requires the full retained event-study covariance (pre+post sub-block) non-singular — see the StackedDiD section's container-admission Note for the df_survey and withheld-inference semantics.

**Reference implementation(s):**
- R: `HonestDiD` package (Rambachan & Roth's official package)

**Requirements checklist:**
- [x] Δ^SD constrains second differences with δ_0 = 0 boundary handling
- [x] Δ^RM constrains first differences (not levels), union of polyhedra
- [x] Identified set LP pins δ_pre = β_pre (Equations 5-6)
- [x] Optimal FLCI for Δ^SD (convex optimization, folded normal quantile)
- [x] ARP hybrid framework for Δ^RM (vertex enumeration, truncated normal)
- [x] Sensitivity analysis over M/M̄ grid with breakdown value
- [x] M parameter must be ≥ 0
- [ ] ARP hybrid produces valid (non-degenerate) CIs for all test cases

---

## PreTrendsPower

**Primary source:** [Roth, J. (2022). Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends. *American Economic Review: Insights*, 4(3), 305-322.](https://doi.org/10.1257/aeri.20210236). Paper review on file: `docs/methodology/papers/roth-2022-review.md`.

**Key implementation requirements:**

*Assumption checks / warnings:*
- Requires specification of variance-covariance matrix Σ_22 of pre-period coefficients
- Pre-trend zero-anticipation: τ_pre = 0 (so β̂_pre estimates δ_pre directly) — same convention as Rambachan-Roth (2023) HonestDiD
- Warns if pre-trends test has low power (uninformative) relative to typical effect sizes
- Different violation types and pretest forms have different power properties

*Estimator equation (primary form — NIS box probability; Roth 2022 Section II.A-B):*

The paper-analyzed pretest is the **no-individually-significant (NIS)** test: reject parallel trends if any pre-period coefficient lies outside its own (1 - α) CI. The acceptance region is

```
B_NIS(Σ) = { b ∈ R^K : |b_t| ≤ z_{1-α/2} · σ_t,  for all t ∈ pre-periods }
```

Under H1 with violation `δ_pre = M · weights` and `β̂_pre ~ N(δ_pre, Σ_22)`, the rejection probability is computed via the centered change-of-variable `Y = β̂_pre - δ_pre ~ N(0, Σ_22)`:

```
Power(δ_pre) = 1 - P( Y_t ∈ [-z·σ_t - δ_t, z·σ_t - δ_t]  for all t )
             = 1 - F_MVN(upper, lower; mean=0, cov=Σ_22)
```

where `F_MVN` is the multivariate normal CDF over the rectangular box. Computed via `scipy.stats.multivariate_normal.cdf(upper, lower_limit=lower, mean=zeros, cov=Σ_22, allow_singular=True)` (Genz method; supports K up to ~20). Falls back to MC simulation (N=20000 draws) when the analytical CDF returns NaN on degenerate Σ.

MDV: solve `Power(γ · weights) = target_power` for γ via doubling expansion + `optimize.brentq` bisection. Non-convergence cap at γ_high = 1000 returns `np.inf`.

*Estimator equation (paper-supported alternative — Wald pretest form):*

```
W = δ̂_pre' Σ_22^{-1} δ̂_pre ~ χ²(K)
Power(δ_pre) = 1 - F_ncχ²(c_α; K, λ),  where λ = δ_pre' Σ_22^{-1} δ_pre
                                        (noncentrality parameter)
```

The Wald acceptance region is a convex ellipsoid, so Propositions 1+3+4 of Roth (2022) all apply. Retained for backwards compatibility with the pre-PR-B shipped numerical output (Wald was the implicit default before PR-B 2026-05-17). Activated via `pretest_form='wald'`.

Violation types:
- **Linear**: `δ_t = γ · t` (Roth's slope convention). When `relative_times` is threaded through `fit()`, weights = `|t|` directly with no L2 normalization, so the reported MDV is in Roth's γ units.
- **Constant**: `δ_t = c` (level shift)
- **Last period**: `δ_{-1} = c`, others zero
- **Custom**: user-specified `violation_weights` pattern

- **Note (paper-supported alternative — Wald pretest form):** the library retains the Wald noncentral-χ² form as `pretest_form='wald'`. NIS is the paper's primary analysis convention (used for all 12 surveyed papers' empirical exercises in Section I), but the Wald form is also a paper-supported alternative: Roth's Propositions 1, 3, and 4 apply to any (measurable) acceptance region for the conditional moments (Props 1+3) and to any convex acceptance region for the variance-reduction guarantee (Prop 4). The Wald ellipsoid is convex, so all four propositions apply. Wald is faster (no MVN CDF call) and matches the pre-PR-B shipped numerical baseline. Use Wald for backwards-compat / speed; use NIS for canonical paper alignment and R `pretrends` parity.

- **Note (convention — `linear` violation pattern, γ-unit MDV):** `_get_violation_weights('linear')` consumes actual pre-period relative-time labels threaded through `fit()` (PR-B 2026-05-17 resolution of the PR-A linear-pattern deviation). When `relative_times` is provided, weights = `|t|` directly with NO L2 normalization, so `δ_pre = M · |t|` reflects Roth's `δ_t = γ · t` convention and the reported MDV equals γ. **The relative times are REFERENCE-ANCHORED on common-reference routes (2026-08 correction):** Roth's violation is normalized like the coefficients — it must vanish at the omitted period (his labeling puts the reference at `t = 0`) — so the threaded values are `t - t_ref`, not raw treatment-relative event labels: MPD derives them from `reference_period` (`_coerce_relative_times_from_reference`), the CS universal route anchors at the singleton `reference_event_times`, SunAbraham at its omitted `e = -1 - anticipation`, and the `EventStudyResults` container at its single marked reference row (else the singleton provenance). Raw labels are retained ONLY where no common reference exists (CS `base_period="varying"` — covered by the varying-base warning and the TODO.md transformation row). Before the correction the CS/SA/container routes passed raw treatment-relative labels, overstating the violation at each pre-period by the reference offset (e.g. weights `[3, 2]` instead of `[2, 1]` for pre `[-3, -2]` around `t_ref = -1`) and understating γ-unit MDV correspondingly. Callers that bypass `fit()` and supply only `n_pre` retain the previous count-based, L2-normalized `[n_pre-1, ..., 0]` direction (preserves shipped Wald numerical baselines for unit tests). **MPD period-label coverage:** for `MultiPeriodDiDResults`, the relative-time derivation in `_extract_pre_period_params` supports numeric labels (`int` / `float` / `np.int64`) and `pandas.Period` / `pandas.Timestamp` / `np.datetime64` (via Period or Timedelta arithmetic with units of frequency / days respectively). For genuinely non-numeric or unordered labels (string period IDs, unranked categoricals), the helper emits an explicit `UserWarning` and falls back to the legacy count-based normalized direction — the reported MDV is then NOT in Roth's γ units. Users on string period IDs who need γ-unit MDV should re-fit with numeric labels.
- **Note (CS varying-base input — warned, not rejected):** the `linear` construction `δ_pre = M · |t|` assumes pre-period coefficients are LEVELS against one common reference period. CallawaySantAnna `base_period="varying"` (the CS default) pre-treatment effects are consecutive-period comparisons — under a linear underlying trend they are constant increments, not values proportional to `|t|` — so linear power/MDV computed on them target a different violation shape. Both CS-sourced input routes (the native results object and the `EventStudyResults` container, which also warns fail-safe on `base_period=None` provenance) emit a `UserWarning` recommending `base_period='universal'`, mirroring HonestDiD's universal-base warning. (Admitted Stacked-sourced containers — row M-024 — are universal-base by construction and never trip this warning from a producer-built container; the guard messages derive the producer from `surface.source`.) Universal-base fits on GAPPED grids do not merely warn — they fail closed via the common-reference guard (`reference_event_times` provenance; see the HonestDiD note), because cohort-specific positional bases make the violation space ill-defined regardless of the pattern. Transforming the varying-base violation vector through each coefficient's actual base mapping (or requiring universal base) is tracked in TODO.md.

*Standard errors:*
- Power calculations are exact (no sampling variability — power is computed against a hypothesized population trend, not estimated)
- Uncertainty comes from the user-supplied Σ_22
- Footnote 7 equivariance: the distribution of `β̂_post` conditional on `β̂_pre` passing the pretest is equivariant w.r.t. `τ_post` (Roth 2022 Section I.C); MDV/power do not depend on the value of `τ_post`

*Edge cases:*
- Perfect collinearity in pre-periods: test not well-defined; `multivariate_normal.cdf(allow_singular=True)` may return NaN — MC simulation fallback kicks in.
- Single pre-period (K=1): NIS power reduces to a univariate normal-tail probability; closed-form match with Roth Section II.B Proposition 2 proof: `E[β̂_pre | β̂_pre ∈ B_NIS] - β_pre ∝ φ(-z - β_pre/σ) - φ(z - β_pre/σ)`.
- Very high power: MDV approaches zero.
- Symmetric two-sided pretests under parallel trends: `β̂_post` remains unbiased for `τ_post` (Roth Section II.B paragraph after Prop 1 — `E[β̂_pre | β̂_pre ∈ B] = 0` if B is symmetric and `β_pre = 0`).

- **Note (deviation from paper — diagonal pre-period VCV fallback, bootstrap-only after PR-B):** Roth (2022)'s power and bias objects operate on the full pre-period covariance block Σ_22. After PR-B 2026-05-17, the shipped `compute_pretrends_power` adapter consumes full Σ_22 on the non-bootstrap paths for ALL three natively supported result types (and, since row M-024, on admitted CS-/StackedDiD-sourced `aggregate('event_study')` containers carrying `vcov` — StackedDiD persists its ES VCV in every inference mode, so Stacked containers always take the full-Σ_22 tier):
  - `MultiPeriodDiDResults`: full pre-period sub-block from `results.vcov` when `interaction_indices` is populated; diag fallback only when `interaction_indices` is None.
  - `CallawaySantAnnaResults`: full `event_study_vcov` sub-block on non-bootstrap fits (the matrix is persisted at `staggered_results.py:126-128`). Bootstrap CS fits clear `event_study_vcov` at `staggered.py:2032-2036` to prevent mixing analytical VCV with bootstrap SEs, so they fall through to `diag(ses^2)`.
  - `SunAbrahamResults`: full `event_study_vcov` sub-block on non-bootstrap fits, constructed in `sun_abraham.py` via `W @ vcov_cohort @ W.T` where W is the cohort-aggregation matrix (PR-B Step 3 SA extension). Bootstrap SA fits and replicate-weight survey fits clear `event_study_vcov` for the same reason as CS.

  The diag-fallback path is therefore reserved for cases where the analytical VCV is genuinely unavailable (bootstrap fits, replicate-weight survey fits, MPD without `interaction_indices`). In those cases dropping off-diagonals is documented as a non-paper approximation — not provably conservative, since the direction of the discrepancy with the full-Σ_22 calc depends on the sign and magnitude of the dropped correlations. See `docs/methodology/papers/roth-2022-review.md` for the full derivation.

- **Backwards-compat addendum (`power_at()` for `violation_type='custom'`):** `PreTrendsPowerResults` now persists `violation_weights` on fresh fits (PR-B Step 5), so `power_at(M)` works for all four violation types including custom. Old serialized results from before PR-B's field addition have `violation_weights=None`; for those legacy results, `power_at(M)` falls back to weight reconstruction from `violation_type + n_pre_periods`, but for `violation_type='custom'` the custom weights cannot be reconstructed and `power_at(M)` raises `NotImplementedError` with a "refit with current library version" message. Fresh fits do not hit this guard.

**Reference implementation(s):**
- R: [`pretrends`](https://github.com/jonathandroth/pretrends) (Roth's official package). NIS-based (`pretrends()`, `slope_for_power()`, `*_NIS` helpers). R-parity goldens at `benchmarks/data/r_pretrends_golden.json` generated by `benchmarks/R/generate_pretrends_golden.R` against commit `122731d082` (package version 0.1.0; PR-C, 2026-05-19). Parity at `atol=1e-4` on both NIS power and γ_p MDV — R hardcodes `thresholdTstat=1.96` while Python uses `scipy.stats.norm.ppf(0.975) = 1.959963984540054`, and R `slope_for_power` uses `uniroot(tol = .Machine$double.eps^0.25 ≈ 1.22e-4)` versus Python `brentq(xtol=2e-12)`; the inverse-solver tolerance gap dominates the γ_p residual.
- R dependency: [`tmvtnorm`](https://cran.r-project.org/package=tmvtnorm) (Manjunath & Wilhelm 2012) — used by R `pretrends` for truncated multivariate normal moments. The Python library uses `scipy.stats.multivariate_normal.cdf` directly for the box probability (does not require a `tmvtnorm` port).

**Requirements checklist:**
- [x] NIS box probability implemented via scipy MVN CDF (PR-B)
- [x] Wald form retained as paper-supported alternative under `pretest_form='wald'` (PR-B)
- [x] Non-bootstrap CS/SA route through full `event_study_vcov` sub-block (PR-B Step 3)
- [x] Linear-violation weights honor actual relative-time labels → γ-unit MDV (PR-B Step 4)
- [x] Custom-violation weights persisted on `PreTrendsPowerResults`; `power_at(custom)` works on fresh fits (PR-B Step 5)
- [x] Helper API (`compute_pretrends_power` / `compute_mdv`) supports `violation_weights` + `pretest_form` (PR-B Step 6)
- [x] Methodology test file with paper-equation-numbered Verified Components walk-through (PR-B Step 7 — `tests/test_methodology_pretrends.py`)
- [x] R `pretrends` parity at commit `122731d082` (PR-C, 2026-05-19; 4 fixtures × NIS power + γ_p MDV at `atol=1e-4`)
- [x] Power curve plotting over violation magnitudes (preserved from pre-PR-B)
- [x] Integrates with HonestDiD for combined sensitivity analysis (preserved from pre-PR-B)

---

## PowerAnalysis

**Primary source:**
- Bloom, H.S. (1995). Minimum Detectable Effects: A Simple Way to Report the Statistical Power of Experimental Designs. *Evaluation Review*, 19(5), 547-556. https://doi.org/10.1177/0193841X9501900504
- Burlig, F., Preonas, L., & Woerman, M. (2020). Panel Data and Experimental Design. *Journal of Development Economics*, 144, 102458.

**Key implementation requirements:**

*Assumption checks / warnings:*
- Requires the outcome SD `sigma` (validated finite and ≥ 0); within-unit (serial) equicorrelation via `rho` (default 0).
- The analytical methods validate `n_pre ≥ 1`, `n_post ≥ 1`, `rho ∈ [-1/(T-1), 1)`, positive group counts (`n_treated`, `n_control` > 0), and `treat_frac ∈ (0, 1)` — each raises `ValueError` on violation. The analytical path does **not** emit low-power / insufficient-sample warnings (those live only on the simulation path).
- Survey / cluster design effects enter via the multiplicative `deff` parameter (default 1.0; `deff < 1` emits a `UserWarning`); there is no separate cluster-randomization parameter set.

*Estimator equation (as implemented):*

Critical values use the **normal (z)** distribution (Bloom 1995), not t.

Minimum detectable effect (MDE) — Bloom (1995), p.549:
```
MDE = (z_{1-α/2} + z_{1-κ}) × SE(τ̂)        [two-sided]
MDE = (z_{1-α}   + z_{1-κ}) × SE(τ̂)        [one-sided]
```
where κ is target power (typically 0.8) and z_q is the standard-normal quantile.

- **Deviation from R:** the MDE multiplier uses the **normal** distribution
  following Bloom (1995), whereas Burlig et al. (2020) Eq. 1 and `pwr::pwr.t.test`
  use the **t** distribution. This is a deliberate large-sample approximation
  (exact as the number of units grows; mildly anti-conservative — a few percent —
  for very small unit counts). The parity reference for the analytical path is
  therefore normal-based (`pwr::pwr.norm.test` / a hand-derived `qnorm` golden),
  **not** `pwr.t.test`.

Standard error — DiD with m = n_pre pre-periods and r = n_post post-periods: the
within-unit equicorrelated special case of Burlig, Preonas & Woerman (2020),
Eq. 2 (ψ^B = ψ^A = ψ^X = ρσ²):
```
SE(τ̂) = σ × √( (1/n_T + 1/n_C) × (1/m + 1/r) × (1 - ρ) )
```
where ρ is the within-unit (serial) equicorrelation. Cross-period correlation
**lowers** the DiD variance (differencing cancels the shared within-unit
component), so the MDE *decreases* as ρ increases — the opposite of a Moulton
mean-inflation factor. Valid range: ρ ∈ [-1/(T-1), 1), T = m + r; `n_pre ≥ 1` and
`n_post ≥ 1` are required and validated before computation (for *all* designs,
not only T > 2).

The basic 2×2 design (n_pre = n_post = 1) is the m = r = 1 special case (Burlig
footnote 11): SE = σ × √( 2 × (1/n_T + 1/n_C) × (1 - ρ) ), which reduces to the
DiD analog of Bloom (1995) Eq. 1 — σ × √( 2 × (1/n_T + 1/n_C) ) — at ρ = 0.

- **Note:** only the **equicorrelated** special case (a single ρ) is implemented;
  the fully general serial-correlation-robust form with independent ψ^B, ψ^A, ψ^X
  (Burlig Eq. 2) is not. The equicorrelated case has lineage Frison & Pocock
  (1992) / McKenzie (2012), which Burlig generalizes.
- **Note:** the analytical path does not model Bloom's covariate-adjustment factor
  (1 - R²); R² = 0 is assumed. Survey design effects enter separately via the
  multiplicative `deff` parameter (Kish 1965).

Power function — exact two-tailed normal:
```
Power = 1 - Φ(z_{1-α/2} - |τ|/SE) + Φ(-z_{1-α/2} - |τ|/SE)
```
(the lower-tail term is negligible but retained for exactness).

Sample size for target power (treatment fraction f = n_T/n; allocation factor
f(1-f), maximized at f = 1/2; M = z_{1-α/2 or 1-α} + z_{1-κ}):
```
n = 2 M² σ² (1 - ρ) / ( δ² f(1-f) )                  [basic 2×2, m=r=1 case]
n = M² σ² (1/m + 1/r)(1 - ρ) / ( δ² f(1-f) )         [panel]
```
where δ is the target effect.

*Standard errors:*
- Analytical formulas (no estimation uncertainty in power calculation)
- Simulation-based power accounts for finite-sample and model-specific factors

*Edge cases:*
- Very small effects: may require infeasibly large samples
- Within-unit equicorrelation ρ: higher ρ *lowers* the DiD variance (and hence the MDE and required N) because the difference-in-differences cancels the shared within-unit component (Burlig Eq. 2 equicorrelated case) — the opposite of survey/cluster design effects (`deff`), which inflate variance (see the `deff` Note below)
- Unequal allocation: optimal is often 50-50 but depends on costs
- **Note:** `data_generator_kwargs` keys that overlap with registry-managed simulation inputs (`treatment_effect`, `noise_sd`, `n_units`, `n_periods`, `treatment_fraction`, `treatment_period`, `n_pre`, `n_post`) are rejected with `ValueError` to prevent silent desync between the DGP and result metadata. `n_pre` and `n_post` are derived from `treatment_period` and `n_periods` in factor-model DGPs (SyntheticDiD, TROP); the 3-way intersection check naturally scopes the rejection to those estimators only. Use the corresponding `simulate_power()` parameters directly, or pass a custom `data_generator` to override the DGP entirely.
- **Note:** For the **cross-sectional** `TripleDifference` path (`n_periods ≤ 2`), `simulate_sample_size()` rejects `n_per_cell` in `data_generator_kwargs` because `n_per_cell` is derived from `n_units` (the search variable). A fixed override would freeze the effective sample size across bisection iterations, making the search degenerate. Use `simulate_power()` with a fixed `n_per_cell` override instead, or pass a custom `data_generator`. (On the panel path — `n_periods > 2` — `n_per_cell` is rejected upstream in `simulate_power` because the panel DGP does not accept it; see the panel routing Note below.)
- **Note:** The simulation-based power registry (`simulate_power`, `simulate_mde`, `simulate_sample_size`) uses a single-cohort staggered DGP by default. Estimators configured with `control_group="not_yet_treated"`, StackedDiD `control_group="strict"`, or `anticipation>0` will receive a `UserWarning` because the default DGP does not match their identification strategy. Users must supply `data_generator_kwargs` (e.g., `cohort_periods=[2, 4]`, `never_treated_frac=0.0`) or a custom `data_generator` to match the estimator design.
- **Note:** The `TripleDifference` registry adapter routes by `n_periods`. For `n_periods ≤ 2` it uses `generate_ddd_data`, a fixed 2×2×2 factorial DGP (group × partition × time): the `n_periods`, `treatment_period`, and `treatment_fraction` parameters are ignored — it always simulates 2 periods with balanced groups. `n_units` is mapped to `n_per_cell = max(2, n_units // 8)` (effective total N = `n_per_cell × 8`), so non-multiples of 8 are rounded down and values below 16 are clamped to 16. A `UserWarning` is emitted when simulation inputs differ from the effective DDD design. When rounding occurs, all result objects (`SimulationPowerResults`, `SimulationMDEResults`, `SimulationSampleSizeResults`) set `effective_n_units` to the actual sample size used; it is `None` when no rounding occurred. `simulate_sample_size()` snaps bisection candidates to multiples of 8 so that `required_n` is always a realizable DDD sample size. Passing `n_per_cell` in `data_generator_kwargs` suppresses the effective-N rounding warning but not warnings for ignored parameters (`treatment_period`, `treatment_fraction`).
- **Note:** For `n_periods > 2`, `TripleDifference` power routes to `generate_ddd_panel_data` (a balanced panel with time-invariant `group`/`partition` and a derived `post` indicator), which **honors** `n_periods` and `treatment_period`. `n_units` maps directly to panel units (no `// 8` rounding; `effective_n_units` is `None`), and `simulate_sample_size()` uses a continuous (step-1) search. `treatment_fraction` is still ignored (the design is a balanced 2×2×2 with `group_frac = partition_frac = 0.5`); pass `group_frac`/`partition_frac` via `data_generator_kwargs` to vary the split, and a `UserWarning` fires if `treatment_fraction ≠ 0.5`. `n_per_cell` (a cross-sectional-only key) raises `ValueError`. **Clustering caveat:** the panel DGP has within-unit serial correlation, so unclustered standard errors are anti-conservative and overstate power — construct the estimator as `TripleDifference(cluster="unit")` (Liang-Zeger CR1; the panel DGP's unit column is named `"unit"`). A `UserWarning` is emitted when the estimator lacks `cluster="unit"`. The point estimate is invariant to clustering; the inference contract is not.
- **Note:** The analytical power methods (`PowerAnalysis.power/mde/sample_size` and the `compute_power/compute_mde/compute_sample_size` convenience functions) accept a `deff` parameter (survey design effect, default 1.0). This inflates variance multiplicatively: `Var(ATT) *= deff`, and inflates required sample size: `n_total *= deff`. The `deff` parameter is **not redundant** with `rho` (intra-cluster correlation): `rho` models within-unit (serial) equicorrelation in panel data via the Burlig (2020) Eq. 2 equicorrelated factor `(1/m + 1/r)(1 - rho)`, while `deff` models the survey design effect from stratified multi-stage sampling (clustering + unequal weighting). A survey panel study may need both. Values `deff > 0` are accepted; `deff < 1.0` (net variance reduction, e.g., from stratification gain) emits a warning.
- **Note:** `simulate_power()` catches a narrow set of exception types — `ValueError`, `numpy.linalg.LinAlgError`, `KeyError`, `RuntimeError`, `ZeroDivisionError` — raised inside the per-simulation fit and result-extraction block, increments a per-effect failure counter, and skips the replicate. Programming errors (`TypeError`, `AttributeError`, `NameError`, `IndexError`, etc.) are allowed to propagate so that bugs in the estimator or custom result extractor surface loudly instead of being absorbed as simulation failures. The primary-effect failure count is surfaced on the result object as `SimulationPowerResults.n_simulation_failures`; a `UserWarning` still fires when the failure rate exceeds 10% for any effect size, and all-failed runs raise `RuntimeError`. This replaces the prior bare `except Exception` that swallowed root causes and kept the counter internal to the function (axis C — silent fallback — under the Phase 2 audit).
- **Note:** `SurveyPowerConfig._build_survey_design()` no longer caches its return value in `self._cached_survey_design`. Reassigning `config.survey_design` (either replacing a user-supplied `SurveyDesign` with another, or toggling between `None` and a user-supplied design) after the first call used to silently return the stale cached design; the method now returns the live `self.survey_design` (or the default construction when `None`) every call. Other config fields (`n_strata`, `icc`, `weight_variation`, etc.) never influenced the returned design, so the staleness surface was specifically `survey_design` reassignment. Construction is microseconds — the cache never earned its complexity. Axis-J finding #28 in the Phase 2 silent-failures audit.
- **Note:** The simulation-based power functions (`simulate_power/simulate_mde/simulate_sample_size`) accept a `survey_config` parameter (`SurveyPowerConfig` dataclass). When set, the simulation loop uses `generate_survey_did_data` instead of the default registry DGP, and automatically injects `SurveyDesign(weights="weight", strata="stratum", psu="psu", fpc="fpc")` into the estimator's `fit()` call. Supported estimators: DifferenceInDifferences, TwoWayFixedEffects, MultiPeriodDiD, CallawaySantAnna, SunAbraham, ImputationDiD, TwoStageDiD, StackedDiD, EfficientDiD. Unsupported (raises `ValueError`): TROP, SyntheticDiD, TripleDifference (generate_survey_did_data produces staggered cohort data incompatible with factor-model/DDD DGPs). `survey_config` and `data_generator` are mutually exclusive. `data_generator_kwargs` may not contain keys managed by `SurveyPowerConfig` (n_strata, psu_per_stratum, etc.) but may contain passthrough DGP params (unit_fe_sd, add_covariates, strata_sizes). Repeated cross-section survey power (`panel=False`) is only supported for `CallawaySantAnna(panel=False)` with a matching `data_generator_kwargs={"panel": False}`; both mismatch directions are rejected. `estimator_kwargs` may not contain `survey_design` when `survey_config` is set (use `SurveyPowerConfig(survey_design=...)` instead). Estimator settings that require a multi-cohort DGP (`control_group="not_yet_treated"`, `control_group="last_cohort"`, StackedDiD `control_group="strict"`) are rejected because the survey DGP uses a single cohort; use the custom `data_generator` path for these configurations. `simulate_sample_size` raises the bisection floor to `n_strata * psu_per_stratum * 2` to ensure viable survey structure and rejects `strata_sizes` in `data_generator_kwargs` (it depends on `n_units` which varies during bisection).

**Reference implementation(s):**
- R: `pwr::pwr.norm.test` (analytical path — normal-based, matching D1; **not** `pwr.t.test`, which is noncentral-t), `DeclareDesign` (simulation-based)
- Stata: `pcpanel` (Burlig et al. 2020 panel variance), `power` command (cross-sectional)

**Requirements checklist:**
- [x] MDE calculation given sample size and variance parameters
- [x] Power calculation given effect size and sample size
- [x] Sample size calculation given MDE and target power
- [x] Simulation-based power for complex designs
- [x] Cluster adjustment for clustered designs (within-unit `rho` + survey `deff`)

---

# Visualization

## Event Study Plotting (`plot_event_study`)

**Reference Period Normalization**

Normalization only occurs when `reference_period` is **explicitly specified** by the user:

- **Explicit `reference_period=X`**: Normalizes effects (subtracts ref effect), sets ref SE to NaN
  - Point estimates: `effect_normalized = effect - effect_ref`
  - Reference period SE → NaN (it's now a constraint, not an estimate)
  - Other periods' SEs unchanged (uncertainty relative to the constraint)
  - CIs recomputed from normalized effects and original SEs

- **Auto-inferred reference** (from CallawaySantAnna results): Hollow marker styling only, no normalization
  - Original effects are plotted unchanged
  - Reference period shown with hollow marker for visual indication
  - All periods retain their original SEs and error bars

This design prevents unintended normalization when the reference period isn't a true
identifying constraint (e.g., CallawaySantAnna with `base_period="varying"` where different
cohorts use different comparison periods).

The explicit-only normalization follows the `fixest` (R) convention where the omitted/reference
category is an identifying constraint with no associated uncertainty. Auto-inferred references
follow the `did` (R) package convention which does not normalize and reports full inference.

**Rationale**: When normalizing to a reference period, we're treating that period as an
identifying constraint (effect ≡ 0 by definition). The variance of a constant is zero,
but since it's a constraint rather than an estimated quantity, we report NaN rather than 0.
Auto-inferred references may not represent true identifying constraints, so normalization
should be a deliberate user choice.

**Edge Cases:**
- If `reference_period` not in data: No normalization applied
- If reference effect is NaN: No normalization applied
- Reference period CI becomes (NaN, NaN) after normalization (explicit only)
- Reference period is plotted with hollow marker (both explicit and auto-inferred)
- Reference period error bars: removed for explicit, retained for auto-inferred
- Multiple reference rows (an `EventStudyResults` container from CallawaySantAnna
  `base_period="universal"` on a gapped grid): every marked row plots hollow at 0 with no
  error bar via row-aligned marking - never dropped, never shown as a filled estimate; the
  scalar reference (and its vertical line) stays unset. Explicit `reference_period=`
  normalization is accepted only when the chosen period is one of the marked rows (a no-op
  shift); any other period raises `ValueError`, because each anchor is a constraint under
  its own cohort base and no single shift represents them faithfully

**Reference implementation(s):**
- R: `fixest::coefplot()` with reference category shown at 0 with no CI
- R: `did::ggdid()` does not normalize; shows full inference for all periods

---

# Cross-Reference: Standard Errors Summary

| Estimator | Default SE | Alternatives |
|-----------|-----------|--------------|
| DifferenceInDifferences | HC1 robust | Cluster-robust, wild bootstrap (with `cluster=`) |
| MultiPeriodDiD (deprecated 3.9 → TWFE event-study mode) | HC1 robust | Cluster-robust (via `cluster` param); no wild path — `inference="wild_bootstrap"` warns and falls back to analytical |
| TwoWayFixedEffects | Cluster at unit (static AND event-study mode; ES carve-outs: dropped on Conley / never a survey PSU / explicit one-way) | Wild bootstrap (static only; event-study mode raises) |
| CallawaySantAnna | Analytical (influence fn) | Multiplier bootstrap |
| SunAbraham | Cluster-robust + delta method | Pairs bootstrap |
| ImputationDiD | Conservative clustered (Thm 3) | Multiplier bootstrap (library extension; percentile CIs and empirical p-values, consistent with CS/SA) |
| TwoStageDiD | GMM sandwich (Newey & McFadden 1994) | Multiplier bootstrap on GMM influence function |
| SyntheticDiD | Placebo variance (Alg 4) | Unit-level pairs bootstrap (paper-faithful refit, Alg 2 step 2); jackknife (Alg 3) |
| TripleDifference | Influence function (all methods) | SE = std(IF) / sqrt(n) |
| StackedDiD | Cluster-robust (unit) | Cluster at unit × sub-experiment |
| TROP | Block bootstrap | — |
| ChangesInChanges | Bootstrap (replicate-SD SEs, normal-approx CIs) | Sup-t uniform bands (fixed 95%, qte parity) |
| QDiD | Bootstrap (replicate-SD SEs, normal-approx CIs) | Sup-t uniform bands (fixed 95%, qte parity) |
| BaconDecomposition | N/A (exact decomposition) | Individual 2×2 SEs |
| HonestDiD | Inherited from event study | FLCI, C-LF |
| PreTrendsPower | Exact (analytical) | - |
| PowerAnalysis | Exact (analytical) | Simulation-based |

---

# Cross-Reference: R Package Equivalents

| diff-diff Estimator | R Package | Function |
|---------------------|-----------|----------|
| DifferenceInDifferences | fixest | `feols(y ~ treat:post, ...)` |
| MultiPeriodDiD | fixest | `feols(y ~ i(time, treat, ref=ref) \| unit + time)` |
| TwoWayFixedEffects | fixest | `feols(y ~ treat \| unit + time, ...)` |
| CallawaySantAnna | did | `att_gt()` |
| SunAbraham | fixest | `sunab()` |
| ImputationDiD | didimputation | `did_imputation()` |
| TwoStageDiD | did2s | `did2s()` |
| ContinuousDiD | contdid | `cont_did()` |
| SyntheticDiD | synthdid | `synthdid_estimate()` |
| TripleDifference | triplediff | `ddd()` |
| StackedDiD | stacked-did-weights | `create_sub_exp()` + `compute_weights()` |
| TROP | - | (forthcoming) |
| ChangesInChanges | qte | `CiC()` |
| QDiD | qte | `QDiD()` |
| BaconDecomposition | bacondecomp | `bacon()` |
| HonestDiD | HonestDiD | `createSensitivityResults()` |
| RegressionDiscontinuity | rdrobust | `rdrobust()` + `rdbwselect()` (4.0.0; sharp + fuzzy + covariate-adjusted, nn path; `takeup` = R's `fuzzy=`, `covariates` = R's `covs=`) |
| PreTrendsPower | pretrends | `pretrends()` |
| PowerAnalysis | pwr / DeclareDesign / pcpanel | `pwr::pwr.norm.test` (analytical, normal-based — D1) + `pcpanel` (Burlig 2020 panel, equicorrelated case) + simulation. The analytical multiplier is normal (z), so `pwr.t.test` is **not** the faithful parity target. |

---

## SpilloverDiD

**Primary source:** Butts, K. (2023). Difference-in-Differences with Spatial Spillovers. *arXiv:2105.03737v3* (originally posted 2021). https://arxiv.org/abs/2105.03737

**Secondary sources:**
- Gardner, J. (2022). Two-stage differences in differences. *arXiv:2207.05943*. The stage-1+2 residualization pattern this estimator reuses.
- Conley, T. G. (1999). GMM Estimation with Cross-Sectional Dependence. *Journal of Econometrics* 92(1), 1-45. The spatial-HAC variance estimator recommended by Butts page 13 for inference with cutoff = `d_bar`.

**Scope:** Spillover-aware Difference-in-Differences. Augments two-stage Gardner DiD with ring-indicator covariates that identify the direct effect on treated units (`tau_total`) alongside per-ring spillover effects on near-control units (`delta_j`). Handles both panel non-staggered timing and Section 5 staggered timing in a single estimator; non-staggered is the special case where all treated units share an onset time.

**Note:** This estimator is a **documented synthesis** of ingredients — no single published software package implements the exact recipe. `did2s` (R/Stata, Butts & Gardner 2022) implements the Gardner two-stage residualization but does NOT support ring covariates. The Butts (2021/2023) paper proposes the ring estimator in Equations 5/6/8 (non-staggered) and Section 5 / Table 2 (staggered) but does not ship reference software. The diff-diff implementation combines: (a) Butts (2021) Section 5 / Table 2 identification, (b) Gardner (2022) two-stage residualize-then-fit, (c) Wave A's Conley spatial-HAC vcov.

- **Note (rank-guarded Wave D bread):** The Wave D variance bread `A_22 = (X_2'WX_2)^{-1}` is inverted by the shared `_rank_guarded_inv` (`diff_diff/linalg.py`) on the already-column-dropped kept submatrix. `np.linalg.solve` raised only on an exactly-singular bread (prior fallback: dense `lstsq`); a **near**-singular kept Gram would otherwise flow a garbage inverse (~1e13) into the SE. The rank-guard truncates redundant directions → finite SE on the identified subspace (NaN at rank 0), re-inflated to (k, k) with NaN at the dropped positions, and warns. See the CallawaySantAnna "rank-guarded IF standard errors" Note. Sibling of axis-A finding #17.

**Identification spec (committed):**

The stage-2 regressor for ring `j` is the **time-varying** form

```
(1 - D_it) * Ring_{it,j}
```

where `D_it` is the treatment indicator (1 if unit `i` is treated by time `t`) and `Ring_{it,j}` is the indicator that unit `i`'s nearest currently-treated unit (treated by time `t`) is in distance bin `j`. For non-staggered timing, `Ring_{it,j}` is unit-static post-treatment and zero pre-treatment (no unit treated yet); for staggered timing, `Ring_{it,j}` is unit-time-varying.

**Note:** Reading the literal unit-static `(1 - D_it) * S_i` from paper Equation 5 yields a **rank-deficient design** under TWFE: `(1 - D_it) * S_i = S_i - D_it * S_i = S_i - D_it` (since `D_it = 1 ⇒ S_i = 1`), and `S_i` is absorbed by the unit FE `mu_i`, leaving `-D_it`. The two regressors are perfectly anti-collinear after FE absorption. The paper's identification argument (Proposition 2.3, Section 3.1 subsample language) only resolves when `S_it = S_i * 1{t >= first_treat}` is read as the **time-varying** form — which the paper defines on page 12 (the line directly above Equation 5) and which Section 5 Table 2 makes explicit (`S^k_{it}`, `Ring^k_{it,j}`). The implementation matches the paper's intent once the `S_it` notation is interpreted as time-varying.

**Note:** Stage-1 fits unit + time FE on Butts' STRICTER subsample `Omega_0 = {D_it = 0 AND S_it = 0}` (untreated AND unexposed) — the clean far-away control group. This differs from `TwoStageDiD`'s `Omega_0 = {D_it = 0}` (untreated; includes near-controls in post-treatment periods). The stricter Butts subsample prevents spillover-contaminated near-controls from biasing the time FE; near-controls post-treatment carry `delta_j` variation that the ring covariates pick up at stage 2.

**Note (shared FE engine, 2026-07):** The stage-1 solver `_iterative_fe_subset` is a thin Butts-subsample wrapper over the shared bincount Gauss-Seidel helper `diff_diff.utils._iterative_fe_solve` (the same engine ImputationDiD/TwoStageDiD route through) — the wrapper owns the SpilloverDiD front door (empty-Omega_0 and empty positive-weight-Omega_0 `ValueError` gates, subsample extraction) and delegates the iteration, the zero-weight/positive-weight NaN-FE convention, and the `warn_if_not_converged` non-convergence `UserWarning` (labelled "SpilloverDiD stage-1 FE (Butts Omega_0 subsample)") to the shared engine. Per sweep the shared engine computes the identical group means and convergence metric as the historical local loop, so converged fits are bit-identical; `max_iter` is aligned from the historical local cap of 100 to the shared 10,000 convention (the R fixest/pyfixest budget already used by ImputationDiD/TwoStageDiD) — fits that previously exhausted 100 iterations and warned may now converge instead (strictly more accurate FE; `tol=1e-10` unchanged).

**Note (Omega_0 row-level identification — period strict, unit warn-and-drop, plus connectivity):** Every period must have at least one Omega_0 row (else time FE is structurally unidentified for that period, and dropping it would lose all units' cross-time identification) — hard `ValueError`. Units lacking Omega_0 rows (e.g. baseline-treated units with `D_it = 1` at every observed `t`) are warned-and-dropped: their unit FE is NaN, residualization writes NaN on their rows, and the downstream finite-mask path at stage 2 excludes them from estimation. This mirrors `TwoStageDiD`'s always-treated unit handling (`two_stage.py:294-336`) and Gardner's framework, which identifies effects from supported observations rather than requiring every unit estimable. **Connectivity:** the supported-units bipartite graph (supported units linked by shared Omega_0 periods) must form a single connected component. If the graph splits into K > 1 components, the iterative FE solver identifies (`mu_i`, `lambda_t`) only up to component-specific constants, and residualization combines `mu_i` from one component with `lambda_t` from another — silently corrupting `y_tilde` and downstream `tau_total` / `delta_j`. Balanced panel + per-unit/per-period Omega_0 coverage is NECESSARY but NOT SUFFICIENT; connectivity is the load-bearing identification condition. Under the current absorbing-treatment regime the disconnected case is plausibly unreachable in practice (we were unable to construct an example surviving the upstream validators), but `_check_omega_0_connectivity` is in place as defense-in-depth and future-proofs Wave B follow-ups (event-study, survey-design integration, reversible-treatment relaxation if ever added).

**Note (Gardner identity, non-staggered):** Under non-staggered timing, the two-stage Gardner residualize-then-fit with the Omega_0-restricted stage 1 is **empirically bit-identical** to the single-stage TWFE ring regression on the full sample using the time-varying form `y_it ~ mu_i + lambda_t + tau * D_it + sum_j delta_j * (1 - D_it) * Ring_{it,j}`. This is the non-staggered ring estimator from Butts Equations 4-6. The empirical equivalence is verified by a 20-seed deterministic regression test (`TestSpilloverDiDNonStaggeredFEEquivalence`) at `atol=1e-10`. The Omega_0 restriction is therefore innocent for the non-staggered point estimate — it only changes the variance composition (which is why the stage-1 GMM correction enters at stage 2 in the staggered case). Reported `tau_total` for non-staggered timing IS the Butts Eqs. 4-6 estimator.

**Note:** Treated units have `d_it = 0` (their own location) and fall in Ring_1 geometrically, but the `(1 - D_it)` factor zeros their ring contribution. `n_units_ever_in_ring` counts include treated units in Ring_1 by construction. Under staggered timing this field counts each unit in EVERY ring it visits across periods (not a clean partition); under non-staggered timing it IS a partition.

**Note (anticipation shift):** The `anticipation: int` constructor kwarg (default 0) shifts BOTH the treatment and ring-membership clocks by `-anticipation` so the stage-1 Omega_0 subsample correctly excludes the `anticipation` pre-treatment periods (which would otherwise contaminate the FE estimation with anticipation effects). Concretely, the effective treatment indicator becomes `D'_it = 1{t >= first_treat - anticipation}` and ring membership uses the corresponding shifted "currently-treated" set when computing `d_it` for staggered timing. Stage 2 then estimates `tau_total` and `delta_j` net of the anticipation window. Mirrors `TwoStageDiD`'s `anticipation` parameter semantics — recommended use is a small integer (1-3 periods) when domain knowledge suggests treatment effects begin before formal onset (e.g., policy announcements). Setting `anticipation > 0` AND providing a `first_treat` column where `first_treat - anticipation < min(time)` for any unit will mark that unit as treated from the panel's first observation, with the same baseline-treated handling as if it were always-treated (warn-and-drop via the Omega_0 unit-level check above).

**Note:** No published R/Stata software exists for the two-period ring estimator (Butts Eqs. 5-6). Correctness anchored on (a) synthetic-DGP Monte Carlo identification tests on the **non-staggered** DGP (50-seed default + 200-seed `@pytest.mark.slow` variant — both recover `tau_total` AND `delta_1` within 0.02 absolute tolerance on Butts-Assumption-satisfying DGPs) plus a **staggered**-DGP MC test (30-seed, no slow variant) that anchors both `tau_total` within 0.04 and `delta_1` within 0.03 absolute tolerance (looser bounds because each staggered DGP is larger and noisier); per-event-time `delta_jk` recovery on staggered DGPs is shipped in Wave C (`TestSpilloverDiDEventStudyIdentification`); (b) reduce-to-TWFE limit (non-staggered, 20-seed deterministic bit-identity at `atol=1e-10` via `TestSpilloverDiDNonStaggeredFEEquivalence`); (c) Wave A's Conley sparse-vs-dense parity inherited via `solve_ols`. A reduce-to-`TwoStageDiD` limit was scoped during planning but not shipped in Wave B (the Omega_0 stricter subsample requires additional fixture work to align with `TwoStageDiD`'s `{D_it = 0}` subsample); queued as a follow-up alongside the Gardner GMM correction.

### Event-study mode (Wave C, `event_study=True`)

Replaces the aggregate spec with the per-event-time × ring decomposition from Butts Section 5 / Table 2. Direct effects `tau_k` and per-(ring, event-time) spillover effects `delta_jk` are emitted in `att_dynamic` and a MultiIndex `spillover_effects`. A TwoStageDiD-compatible `event_study_effects: Dict[int, Dict]` alias (mirroring `two_stage_aggregation.py::_stage2_event_study` row schema with `conf_int = (low, high)` tuple) is also emitted for `plot_event_study` consumption — `_extract_plot_data` prefers the new `reference_period` attribute over the legacy `n_obs==0` heuristic. `DiagnosticReport` routing is now wired (see the DiagnosticReport applicability Note below).

- **Note (DiagnosticReport applicability):** `SpilloverDiDResults` is registered in `DiagnosticReport`'s `_APPLICABILITY` / `_PT_METHOD` tables with applicable checks {`parallel_trends` (method `event_study`, on the per-event-time DIRECT-effect dynamics — populated when `event_study=True`; Bonferroni fallback across pre-period coefs since there is no `event_study_vcov`), `design_effect` (instance-gated on `survey_metadata`), `heterogeneity` (reads `event_study_effects`)}. This mirrors the `TwoStageDiD` set **minus `bacon`**. `bacon` is **deliberately excluded**: SpilloverDiD identifies the direct effect off FAR-AWAY control observations (`d_it > d_bar`, Assumption 5), not off the TWFE 2×2 comparisons a Goodman-Bacon decomposition enumerates; running the Bacon decomposition on the raw binary treatment would ignore the ring/distance structure and pool spillover-contaminated in-ring units into the control group — the exact SUTVA violation this estimator handles (same rationale that excludes `bacon` for SyntheticControl / TROP / ContinuousDiD). `pretrends_power` / `sensitivity` / `epv` / `estimator_native` are not applicable (no `compute_pretrends_power` adapter, no HonestDiD adapter, no `epv_diagnostics`, not an SDiD/TROP/SCM native path).

**Note (two-clock K_it):** Butts Section 5 uses one symbol `K_it`, but operationally there are TWO event-time clocks. The **direct-effect clock** is `K_direct_{it} = t - effective_first_treat(i)` for ever-treated unit rows (NaN for never-treated). The **spillover-exposure clock** is `K_spill_{it} = t - min{ effective_first_treat(j) : d(i, j) ≤ d_bar AND effective_first_treat(j) ≤ t }` — time since `i` first became exposed to ANY treated neighbor (running min across activated cohorts). `K_spill` is structurally non-negative (pre-trigger rows are NaN and contribute to stage 1 only); spillover placebos at `k < 0` are therefore NOT meaningful and rectangular emission of those cells uses NaN coef + `n_obs = 0`. The trigger cohort for unit `i` is the earliest activated cohort whose treated members fall within `d_bar` of `i` — NOT necessarily the geographically nearest cohort.

**Note (endpoint binning, divergence from TwoStageDiD):** `horizon_max` bins event-times outside `[-H, +H]` into endpoint pools (`k ≤ -H` aggregated into one pre-bin dummy; `k ≥ +H` into one post-bin dummy). No observations are dropped. This **diverges intentionally** from `TwoStageDiD`'s `horizon_max` semantic, which **filters** rows with `|K_it| > H` out of the stage-2 sample. SpilloverDiD's bin-into-endpoint behavior honors the no-silent-data-drop policy; the divergence is documented on both estimators' docstrings to prevent future unification. With `horizon_max=None`, the helper auto-detects the event-time bin set from the observed K values (no binning).

**Note (reference period `-1 - anticipation`):** The reference period (the event-time dummy dropped to anchor the level interpretation) is `ref_period = -1 - anticipation`. With `anticipation=0`, `ref_period = -1` (standard event-study convention; coefficients are relative to one period before treatment). With `anticipation > 0`, the reference period shifts to `-1 - anticipation` so the "pre-treatment" anchor sits BEFORE the anticipation window. Mirrors `TwoStageDiD`'s convention at `two_stage_aggregation.py` (`ref_period = -1 - anticipation`). The reference row appears in `att_dynamic` and `event_study_effects` with `coef = 0.0`, `se = 0.0`, `n_obs = 0`, `conf_int = (0.0, 0.0)` (TwoStageDiD parity, `two_stage_aggregation.py::_stage2_event_study`). When `horizon_max` is set and `ref_period < -horizon_max` (i.e., `anticipation > horizon_max - 1`), the fit raises `ValueError` — silently floor-shifting the reference to `-horizon_max` would change identification (rejected per `feedback_no_silent_failures`).

**Note (post-finite_mask sample):** `att_dynamic["n_obs"]`, `event_study_effects[k]["n_obs"]`, AND the scalar `att` share weights all reflect the POST-`finite_mask` stage-2 estimation sample — not the pre-mask design built by `_build_event_study_design`. On warn-and-drop fits (baseline-treated units without Omega_0 rows are excluded via `finite_mask`), counts and weights are recomputed from `X_2_fit` so the reported metadata matches the actual stage-2 sample that `solve_ols` sees.

**Note (fail-closed scalar `att`):** When `event_study=True`, if any post-treatment direct-effect coefficient is NaN (rank-deficient drop by `solve_ols`), the scalar `att` is set to NaN with an explicit warning rather than silently zeroing the dropped column's contribution via `np.nansum` on a fixed weight vector. The library convention (`feedback_no_silent_failures`) is to surface unidentified aggregates as NaN; users should inspect `att_dynamic` for the per-event-time coefficients and re-aggregate manually with a documented reweighting rule if appropriate.

**Note (scalar `att` aggregation):** The top-level `att` field, when `event_study=True`, is a **share-weighted average** of post-treatment direct effects:
```
att = sum_{k >= 0} (share_k / sum_{k >= 0} share_k) * tau_k
```
The standard error comes from **linear-combination inference** on the post-treatment block of the stage-2 vcov: `Var(att) = w' V_subset w` where `w` is the share-weight row vector and `V_subset` is the sub-vcov of post-treatment `tau_k` rows. This properly accounts for cross-event-time covariance and does NOT require a separate fit. CallawaySantAnna's `aggregate_method="simple"` and TwoStageDiD's analogous aggregate-from-event-study path use the same share-weighting convention.

**Note (Wave E.1 — survey-weighted share definition):** when `survey_design=` is supplied, the per-horizon shares `share_k` are **survey-weight totals** rather than raw observation counts:
```
share_k = sum_{i: K_direct_i = k AND treated_i = 1} w_i
```
where `w_i` are the Hájek-normalized survey weights. Using raw `n_treated_at_k` shares on weighted WLS horizon coefficients would mix unweighted aggregation with weighted horizons and target the wrong estimand. The same survey-weight totals enter both `att` and `Var(att) = w' V_subset w`, so the lincom variance stays consistent with the point estimate. On the no-survey path, `share_k = n_treated_at_k` (sample-share rule) — backward-compatible with Wave C.

**Note (rectangular schema):** `att_dynamic` and the MultiIndex `spillover_effects` are emitted **rectangularly** across the full event-time × ring grid implied by `horizon_max`. Empty cells (no rows contribute to the corresponding stage-2 design column — pre-filtered to keep `solve_ols` rank warnings quiet) emit `coef = NaN`, `se = NaN`, `n_obs = 0`. This includes negative-k spillover cells (structurally empty: `K_spill ≥ 0`) and outer-ring cells when the panel has no units in that ring band. The `(ring_label, ref_period)` cells are also emitted in `spillover_effects` (one per ring) with `coef = 0.0, se = 0.0, n_obs = 0` to mirror the direct-effect reference row anchor — without this, consumers iterating `[-H, ..., +H]` would hit a missing `(ring, ref_period)` slice. Downstream consumers can iterate the rectangular grid without `KeyError` on missing cells.

**Reduce-to-aggregate equivalence:** Under a **constant-tau DGP** with `horizon_max=None`, the sample-share-weighted scalar `att` reproduces Wave B's aggregate `tau_total` (bit-identical at machine precision in the deterministic limit; small MC noise on realized panels). This is the canonical equivalence path. Note: `horizon_max=0` is **not supported** under `event_study=True` (rejected at validation with a clear remediation message): the single bin `k=0` leaves no event-time pair to anchor the reference period `ref_period = -1 - anticipation` against. Users wanting a single aggregate direct effect should use `event_study=False` (the Wave B static spec); event-study mode requires `horizon_max>=1` or `horizon_max=None`.

**Variance:** Per-event-time SEs apply the Wave D Gardner GMM first-stage uncertainty correction (see "Variance (Wave D)" subsection below). Per-`tau_k` and per-`delta_jk` SEs are shifted upward by a few percent relative to Wave C uncorrected SEs.

**Assumptions (Butts 2021):**

- **Assumption 1 (Random Sampling)** — i.i.d. panel.
- **Assumption 3 (Parallel Counterfactual Trends)** — counterfactual trends in absence of all treatment AND zero exposure do not depend on own treatment status.
- **Assumption 5 (Spillovers Are Local):** (i) `min_{j: D_j=1} d(i,j) > d_bar ⇒ h_i(D-vector) = 0` (spillovers vanish past `d_bar`); (ii) at least one treated unit AND one control unit exist with `min d > d_bar` (clean far-away control group). The validator enforces (ii) via `_validate_far_away_exists` with a strict `> d_bar` check; failure raises `ValueError` citing the assumption.
- **Assumption 6 (Total Effect Parallel Trends)** — counterfactual trends do not depend on `(D_i, S_i)`. Stronger than Assumption 3.
- **Assumption 7 (Spillover Effect Parallel Trends)** — counterfactual trends do not depend on `(D_i, S_i)` for `S_i ∈ {0, 1}`. Required to identify `gamma_0` / `delta_j`.
- **Assumption 8 (Parallel Counterfactual Trends, Staggered)** — additive unit + time FE structure on untreated/unexposed potential outcomes. Stronger than Assumption 3.

**Variance (Wave D — Gardner GMM first-stage correction across HC1 / Conley / cluster):**

Stage-2 variance applies the Gardner (2022) GMM sandwich influence-function correction for stage-1 FE estimation uncertainty across all three `vcov_type` paths. The unified IF outer-product formula:

```
psi_i  = gamma_hat' * X_{10,i} * eps_{10,i} - X_{2,i} * eps_{2,i}     # shape (p_2,)
Psi    = [psi_1; ...; psi_n]                                          # (n, p_2)
gamma_hat = (X_10' X_10)^{-1} (X_1' X_2)                              # (p_1, p_2)
meat   = Psi' @ K @ Psi                                               # (p_2, p_2)
vcov   = (X_2' X_2)^{-1} @ meat @ (X_2' X_2)^{-1}
```

where the kernel `K` is path-dependent:

- **HC1**: `K = I_n` → `meat = Psi' Psi` with `n / (n - p_2)` finite-sample multiplier.
- **Cluster CR1**: `K_ij = 1{cluster_i = cluster_j}` → per-cluster sum + outer product, with `G / (G-1) * (n-1) / (n - p_2)` finite-sample multiplier.
- **Conley**: `K_ij = kernel(d_ij / cutoff) * 1{cluster_i = cluster_j}` (cross-sectional) or panel-block decomposed (`conley_time` / `conley_unit` / `conley_lag_cutoff` set). No finite-sample multiplier — matches `conleyreg` convention.

The correction applies unconditionally (no opt-out kwarg). Point estimates (`tau_total`, `delta_j`, event-study `tau_k` / `delta_jk`) are byte-identical to the pre-Wave-D path; SE values shift upward by 1-few percent.

- **Note (documented synthesis):** no R / Stata software combines all three ingredients (Butts (2021) §3.1 IF construction for spillover-aware DiD + Gardner (2022) §4 two-stage GMM sandwich + Conley (1999) spatial kernel). `did2s` (Gardner) implements GMM with HC1/cluster but no Conley. `conleyreg` / `acreg` implement Conley but no two-stage correction. Wave D is the documented synthesis.
- **Note (no finite-sample multiplier on Conley path):** preserves the `conleyreg` / Wave B convention. HC1 and cluster paths apply the standard `n/(n-p)` and `G/(G-1) * (n-1)/(n-p)` multipliers respectively.
- **Note (Conley meat may be non-PSD):** the radial 1-D Bartlett and uniform kernels are practitioner specializations of Conley 1999 and are not formally PSD-guaranteed; a `UserWarning` fires when the smallest meat eigenvalue is < -1e-12. Applies on both standard-sandwich and GMM-corrected sandwich paths.

**Implementation:** new module-level helper `_compute_gmm_corrected_meat` at `diff_diff/two_stage.py` (NOT a modification of the existing `_compute_gmm_variance` method — TwoStageDiD's path is unchanged); new helper `_build_butts_fe_design_csr` at `diff_diff/spillover.py`; `_compute_conley_meat` factored out of `_compute_conley_vcov` at `diff_diff/conley.py` so the same kernel-application code path handles both standard sandwich (`X * residuals`) and Wave D IF outer product (`Psi`).

### Variance (Wave E.1 — survey-design integration via Binder TSL)

`survey_design=` is now supported on `vcov_type ∈ {"hc1"}` and CR1 (`cluster=<col>`) paths. The vcov is the design-consistent sandwich `V = A_22^{-1} @ meat @ A_22^{-1}` where:

- `A_22 = X_2' W X_2` (bread with weighted-OLS structure when `survey_weights` provided)
- `meat = sum_h (1-f_h) * n_h/(n_h-1) * sum_j (S_psu_hj - S_psu_h_bar) (S_psu_hj - S_psu_h_bar)'`
- `S_psu[g] = sum_{i in PSU g} Psi[i]` with `Psi[i] = w_i * gamma_hat' * X_{10,i} * eps_{10,i} - w_i * X_{2,i} * eps_{2,i}`

The `_compute_stratified_meat_from_psu_scores` helper at `diff_diff/survey.py` implements the Binder TSL formula and handles `lonely_psu ∈ {"remove", "certainty", "adjust"}` and FPC consistently across the package.

Degrees of freedom for the t-distribution lookup use `ResolvedSurveyDesign.df_survey` (the standard survey 4-way branch: PSU+strata → `n_PSU - n_strata`; PSU only → `n_PSU - 1`; strata only → `n_obs - n_strata`; neither → `n_obs - 1`). Threaded through all four `safe_inference` call sites: aggregate `tau_total`, per-ring `delta_j`, event-study per-event-time `tau_k` / `delta_jk`, and the scalar `att` lincom in event-study mode.

- **Note (documented synthesis):** Wave E.1 composes Gerber (2026, arXiv:2605.04124) Proposition 1 — Binder Taylor Series Linearization for IF representations of smooth functionals; explicitly derived for TwoStageDiD in the paper's Appendix — with the Wave D Gardner GMM first-stage uncertainty correction (Butts 2021 §3.1 + Gardner 2022 §4) applied to SpilloverDiD's ring-indicator stage-2 design. The composition is mechanical: SpilloverDiD's Wave D Psi is aggregated to PSU level and passed to the audited Binder TSL meat helper. Survey weights enter via Hájek normalization at the gamma_hat solve, eps construction, and bread inversion. No reference software combines all ingredients; Wave E.2 extends with the Conley × survey product-kernel composition (cross-sectional `conley_lag_cutoff = 0`) — see "Variance (Wave E.2)" subsection below; Wave E.2 follow-up further extends with the within-PSU serial Bartlett HAC for `conley_lag_cutoff > 0` — see "Variance (Wave E.2 follow-up)" subsection below.
- **Note (warn-and-use-PSU for cluster + survey):** when both `cluster=<col>` and `survey_design.psu` are supplied with **different groupings**, the cluster argument emits a `UserWarning` and is overridden by PSU (mirrors `TwoStageDiD._resolve_effective_cluster`). PSU is the design-relevant cluster on survey panels; `cluster=<col>` on SpilloverDiD is more often a spatial / unit-level label, so the design constraint wins. When both knobs are supplied with the **same** groupings, no warning fires and PSU still takes precedence (the inference is unchanged either way).
- **Note (`SurveyDesign.subpopulation()` + warn-and-drop full-design retention, Wave E.3):** when `survey_design` is built via `SurveyDesign.subpopulation()` (or otherwise carries zero-weight padding rows) AND those zero-weight rows lose stage-1 FE support (warn-and-drop unit path), Wave E.3 preserves the full-domain resolved survey design: `n_psu`, `df_survey`, and the Binder TSL centering reflect the full domain. Both drop mechanisms (subpopulation weight=0 and warn-and-drop) are treated as zero-score padding rows — the meat-helper boundary at `_compute_gmm_corrected_meat` builds Psi on the SURVEY-FINITE-MASK subset of fit-sample inputs (`finite_mask & (survey_weights > 0)`, which EXCLUDES zero-weight subpop rows so the FE drop-first basis is invariant to which units the subpop mask removed) and zero-pads it to full panel length via a new `score_pad_mask=survey_finite_mask` kwarg AFTER construction but BEFORE kernel dispatch, while the kernel-dispatch arrays (`cluster_ids`, `conley_*`, `resolved_survey`) are passed at full length. See "Variance (Wave E.3)" subsection below.
- **Note (saturated `df_survey = 0` NaN-fail):** when `lonely_psu="remove"` removes all strata (single PSU per stratum), `_compute_stratified_meat_from_psu_scores` returns `(_, var_computed=False, legit_zero=0)`. SpilloverDiD's Wave E.1 path returns NaN meat with a `UserWarning` matching `"df_survey"` so callers can pin via `pytest.warns(UserWarning, match="df_survey")`. This is a **departure from TwoStageDiD** (`two_stage.py:2003-2005`) which currently NaN-fails SILENTLY; Wave E.1 surfaces the diagnostic per `feedback_no_silent_failures`.
- **Note (`weights is None` bit-identical fallback):** the `_iterative_fe_subset` weighted-bincount path falls through to the existing unweighted `np.bincount` code when `weights is None`. The Wave B/C/D no-survey contract is unchanged; the bit-identity test `test_a_uniform_weight_degenerate_matches_wave_d` is the load-bearing check.
- **Note (replicate-weight follow-up):** BRR / Fay / JK1 / JKn / SDR are rejected upfront (`NotImplementedError` with `"follow-up"` matcher). Per Gerber (2026) Appendix A, the per-replicate IF-rescaling shortcut does NOT apply to TwoStageDiD-class estimators because `gamma_hat` is weight-sensitive; correct support requires per-replicate full re-fit of stage 1 and stage 2.

**Implementation:** `_compute_gmm_corrected_meat` extended with `survey_weights` + `resolved_survey` kwargs at `diff_diff/two_stage.py:56`; new module-level helper `_compute_binder_tsl_meat` at `diff_diff/two_stage.py` wraps `_compute_stratified_meat_from_psu_scores` with the Wave E.1 NaN-fail + warning. `_iterative_fe_subset` weighted path at `diff_diff/spillover.py:1382` (in-place extension, bit-identical fallback). `SpilloverDiDResults` extended with `survey_metadata`, `n_psu`, `n_strata` fields at `diff_diff/results.py`. Tests: `TestSpilloverDiDWaveE1SurveyDesignHc1` + `TestSpilloverDiDWaveE1SurveyDesignEventStudy` at `tests/test_spillover.py`.

### Variance (Wave E.2 — Conley × survey via stratified-Conley sandwich on PSU totals)

`vcov_type="conley" + survey_design=` is now supported via a per-stratum Conley sandwich applied to PSU-aggregated Wave D Gardner GMM influence functions. SHIPPED in Wave E.2.

- **Note (documented synthesis):** Wave E.2 composes Conley (1999) spatial-HAC with Gerber (2026, arXiv:2605.04124) Proposition 1 Binder TSL (the Wave E.1 foundation) and the Wave D Gardner GMM first-stage uncertainty correction (Butts 2021 §3.1 + Gardner 2022 §4) applied to SpilloverDiD's ring-indicator stage-2 design. The composition is **panel-aware** — it preserves the library's existing `conley_lag_cutoff = 0` semantic ("within-period spatial only — exclude cross-period pairs") at `diff_diff.conley._compute_conley_meat`. Per-PSU centroids are computed as `centroid_g = mean over i in PSU g of conley_coords[i]` (panel-constant — PSU is a sampling unit with fixed location). For each period `t`, SpilloverDiD's per-obs Wave D IF `psi_i` is aggregated to per-period PSU totals `S_psu_t[g] = sum_{i in PSU g, time t} psi_i`; the within-stratum sandwich is `M_h_t = (1 - f_h) * n_h/(n_h-1) * sum_{j,k in PSUs_h} K(d(centroid_j, centroid_k) / cutoff) * (S_psu_t[j] - S_bar_h_t)(S_psu_t[k] - S_bar_h_t)'`, where K is the Bartlett kernel (SpilloverDiD currently exposes Bartlett only and hardcodes it at the fit-call site; the survey helper's `kernel` parameter can also take `"uniform"`, but exposing that on the SpilloverDiD constructor is a separate follow-up) and `d` is haversine / euclidean / callable per `ConleyMetric`. Cross-stratum kernel weights are exactly zero by sampling design (strata are independence partitions). Total meat is `sum_t sum_h M_h_t`. Cross-period spatial pairs are excluded by construction — the per-period loop aggregates only within-period observations into each `S_psu_t`, matching the Wave D `conley_lag_cutoff = 0` block decomposition. No reference software combines all three ingredients (Conley spatial-HAC + Binder TSL + Gardner GMM correction) on a two-stage influence function.

- **Reduction semantics (load-bearing for tests):**
  - Per-period sum invariant: the orchestrator's panel-aware meat equals `sum_t` of per-period within-stratum stratified-Conley sandwiches on per-period PSU totals. Pinned at `tests/test_spillover.py::TestSpilloverDiDWaveE2ConleySurveyDesign::test_b_panel_aware_per_period_sum_invariant` (pure unit test on the orchestrator + helper composition).
  - Single stratum (H = 1, FPC = inf): reduces to `sum_t` plain Conley sandwich on per-period PSU totals via `_compute_conley_meat(S_psu_t_centered, centroids, ...)`. Note this is NOT plain Conley on time-collapsed PSU totals — the per-period loop preserves the library's `lag_cutoff = 0` semantic.
  - All PSUs singleton in their stratum + `lonely_psu="remove"`: `df_survey = 0` and the stratified-Conley meat NaN-fails (matches Wave E.1 saturation behaviour, with `UserWarning` template "Wave E.2 stratified-Conley sandwich: df_survey = 0...").
  - Cross-stratum kernel weight is exactly zero (sampling-design assumption — no kernel pair crosses a stratum boundary).

- **Note (singleton-stratum `lonely_psu="adjust"` FPC skip parity):** when a stratum has `n_h = 1` and `lonely_psu="adjust"`, the new `_compute_stratified_conley_meat_from_psu_scores` helper mirrors the Binder helper's `continue`-skip-FPC pattern exactly (the FPC scale `(1 - f_h) * n_h / (n_h - 1)` would divide by zero with `n_h = 1`). The degenerate one-PSU kernel `K = [[K(0)]] = [[1.0]]` reduces to `centered.T @ centered`, matching Binder's singleton-adjust contribution bit-identically.

- **Cluster + Conley + survey routing:** `cluster=<col> + survey_design.psu + vcov_type="conley"` coerces `cluster=<col>` to PSU per Wave E.1's `_resolve_effective_cluster` warn-and-use-PSU pattern. The dispatch wrapper `_compute_stratified_conley_meat` intentionally does NOT thread `cluster_ids` into the inner Conley kernel call — after PSU aggregation every PSU is its own cluster, so a cluster product kernel `1{cluster_j == cluster_k}` would be zero for all `j != k` and the cross-PSU kernel weights would be silently dropped. The Wave E.2 architectural choice: PSU-aggregation handles within-PSU clustering exactly; cross-PSU spatial dependence enters via the kernel; cross-stratum independence is exact.

- **Restrictions / out-of-scope (Wave E.2):**
  - Replicate-weight variance (BRR / Fay / JK1 / JKn / SDR) raises `NotImplementedError` (inherits Wave E.1 gate; per-replicate full refit is separate follow-up scope).
  - LinearRegression-side `vcov_type="conley" + survey_design=` gate in `LinearRegression.fit()` (Conley + survey rejection block) remains. **Note (open methodological question):** weighted spatial-HAC under probability sampling is an open methodological question; no canonical extension of Conley (1999) exists for the combination — separate roadmap, not Wave E.
  - DiagnosticReport routing for `SpilloverDiDResults(vcov_type="conley", survey_design=)` is queued for a follow-up Wave F PR. The base `_APPLICABILITY` / `_PT_METHOD` registration prerequisite has now **landed** (parallel_trends/design_effect/heterogeneity; per `feedback_audit_diagnostic_report_wiring_before_claim`); the Conley + survey-design *combination* still needs its own downstream-consumability validation before being claimed.

**Implementation:** new `_compute_stratified_conley_meat_from_psu_scores` helper in `diff_diff/survey.py` (parallel to existing Binder helper; 3-tuple `(meat, variance_computed, legitimate_zero_count)` return contract; per-stratum loop replaces the inner `centered.T @ centered` with `_compute_conley_meat(centered, coords_h, cutoff, metric, kernel)` cross-sectional mode). New dispatch wrapper `_compute_stratified_conley_meat` in `diff_diff/two_stage.py` (parallel to `_compute_binder_tsl_meat`; per-obs Psi → PSU aggregation via `np.add.at` + PSU centroid derivation via vectorized `np.add.at` sums / `np.bincount` counts + dispatch to survey helper; intentionally no `cluster_ids` parameter). `_compute_gmm_corrected_meat` conley branch extended at `diff_diff/two_stage.py` with `if resolved_survey is not None` routing to the new wrapper; the `resolved_survey is None` branch is bit-identical to Wave D no-survey Conley. Saturation NaN-fail mirrors Wave E.1 (`UserWarning` template "Wave E.2 stratified-Conley sandwich: df_survey = 0..."). Wave E.1 stage-1 weighted FE solver, `finite_mask` survey-array subsetting, `df_survey` threading to `safe_inference` call sites, bread weighting, and `SpilloverDiDResults` survey metadata are all inherited UNCHANGED — Psi construction is bit-identical regardless of `vcov_type`. Tests: `TestSpilloverDiDWaveE2ConleySurveyDesign` + `TestSpilloverDiDWaveE2ConleySurveyDesignEventStudy` at `tests/test_spillover.py`.

### Variance (Wave E.2 follow-up — `conley_lag_cutoff > 0` panel-block composition via spatial + serial Bartlett HAC)

`vcov_type="conley" + conley_lag_cutoff > 0 + survey_design=` is now supported via panel-block stratified-Conley sandwich. SHIPPED in Wave E.2 follow-up.

- **Note (documented synthesis):** Wave E.2 follow-up composes Wave E.2's panel-aware stratified-Conley spatial sandwich (Conley 1999 spatial-HAC × Binder/Gerber 2026 stratified TSL × Wave D Gardner GMM correction) with within-PSU serial Bartlett HAC over time (Newey-West 1987 form, kernel weights `1 - |t-s|/(L+1)` for `|t-s| ≤ L, t ≠ s`). The composition is `meat = meat_spatial + meat_serial` with disjoint index sets, exactly matching the no-survey panel-block decomposition at `diff_diff.conley._compute_conley_meat` (Conley 1999 + Newey-West 1987 separable form, NOT Driscoll-Kraay 2D-HAC). The serial term aggregates per-period PSU totals `S_psu_t[g] = sum_{i in PSU g, time t} psi_i` along the time axis: for each stratum `h`, `meat_serial_h = FPC_h_panel * sum_{g in stratum h} sum_{|t-s| ≤ L, t ≠ s, both periods present for PSU g} K_serial(|t-s|/(L+1)) * S_centered_t[g] @ S_centered_s[g]'` where `S_centered_t[g] = S_psu_t[g] - S_bar_h(g)_t` is per-period within-stratum centered (Binder TSL form — matches the spatial helper's centering exactly), and `|t-s|` is computed on **panel-wide dense time codes** (matches `conley.py:940` documented R deviation that mirrors R `conleyreg`). Serial Bartlett kernel is hardcoded regardless of `conley_kernel` (mirrors conley.py:951-965; the user-selected `conley_kernel` governs the spatial kernel only). Total meat is `sum_t sum_h M_h_t + sum_h meat_serial_h`. No reference software combines panel-block Conley + Binder TSL + Gardner GMM correction on a two-stage influence function.

- **FPC convention (panel-wide per-stratum) — standalone Newey-West composition on stratified clusters:** the serial sum aggregates within-PSU temporal correlation across all observed periods — it is a PANEL-level construct, not a period-level construct. The cluster set for the panel-level sum is the panel-wide set of PSUs in stratum `h`, so the FPC denominator uses `n_h_panel = |unique PSUs in stratum h across the active sample|` and `N_h = fpc_per_psu[first PSU in h]` (panel-constant by sampling design); serial term in stratum h scales by `(1 - n_h_panel/N_h) * n_h_panel/(n_h_panel - 1)`. The spatial term keeps its existing per-period FPC unchanged (the period-`t` spatial sum IS a within-period stratified-cluster sandwich at one time index). For balanced panels with PSU present in every period, `n_h_panel = n_h_t` for all `t` so the two FPC denominators converge; the difference surfaces under unbalanced panels. **Standalone citation chain** (Binder 1983 for FPC factor form, Gerber 2026 Prop 1 for Binder TSL composition with two-stage IF, Newey-West 1987 for serial Bartlett kernel weights, Conley 1999 for spatial kernel and panel-block decomposition) — deliberately NOT by analogy to the Binder helper's per-stratum first-occurrence FPC convention at `_compute_binder_tsl_meat`, which is a cross-sectional application of the same principle.

- **Centering asymmetry vs no-survey reference:** the no-survey panel-block path at `conley.py:949-965` uses RAW scores for the serial term (no centering) because it assumes `E[scores] = 0` under correct specification — centering is a no-op under that assumption. The survey-weighted Binder TSL form estimates the within-stratum mean and centers explicitly (textbook stratified-cluster sandwich; the per-period stratum mean enters via Binder's finite-population variance derivation). Using raw scores in the survey case would inflate variance by twice the squared per-period stratum mean and would NOT reduce to the cross-sectional Wave E.2 form at `lag = 0`. The centering applied to each leg (`t` and `s`) of the cross-period pair matches the spatial helper's per-period centering exactly. Hand-check at `TestSpilloverDiDWaveE2FollowupConleySurveyLagCutoff::test_c0_serial_centering_hand_check_raw_vs_centered` pins this contract.

- **Reduction semantics (load-bearing for tests):**
  - `conley_lag_cutoff = 0` or `None`: bit-identical to shipped Wave E.2 ATT and scalar SE (orchestrator skips the serial helper invocation; the spatial loop + saturation guard + new PSD/finite guard still run on the spatial-only meat). `assert_array_equal` regression pin at test_a covers ATT + scalar SE; test_a2 mock-spy independently pins that the serial helper is NOT invoked at lag=0.
  - `conley_time is None` or `T = 1`: serial helper short-circuits to zero meat (no cross-period pairs possible) — the degenerate panel-block path, NOT a saturation diagnostic.
  - Single stratum (`H = 1`, `FPC = inf`): spatial reduces to `sum_t` Conley sandwich on per-period **within-stratum-CENTERED** PSU totals (NOT raw — at H=1 the centering still subtracts the per-period mean over all G PSUs); serial reduces to Newey-West Bartlett HAC on per-period within-stratum-CENTERED PSU totals (NOT raw scores; the survey-weighted form retains Binder TSL centering even at H=1). Both reductions carry the panel-wide `G/(G-1)` survey factor in lieu of FPC.
  - Bandwidth → 0, `L > 0`: spatial reduces to `sum_t` per-period within-stratum HC sandwich on PSU totals; serial term unchanged (separable form).
  - All PSUs singleton in stratum + `lonely_psu="remove"`: `df_survey = 0` and the meat NaN-fails (saturation warning template "Wave E.2 stratified-Conley sandwich" covers both cross-sectional and panel-block cases).
  - Cross-stratum kernel weight is exactly zero on the serial term too (the within-PSU loop is necessarily within-stratum).

- **Singleton-stratum `lonely_psu="adjust"` panel-wide mean asymmetry:** for the serial helper, `_global_psu_mean` is the panel-wide mean of per-period PSU totals (averaged over all `(g, t)` with `present[g, t]`), NOT the per-period within-stratum mean used by the spatial helper. The scope difference reflects the serial term's panel-level nature: a singleton stratum at the panel level has no within-stratum cross-PSU variation to demean against, so the only meaningful centering target is the panel-wide PSU mean. The `continue`-skip-FPC pattern matches the spatial helper at `survey.py:2007-2017` exactly to avoid divide-by-zero on `n_h_panel = 1`.

- **Restrictions (Wave E.2 follow-up):**
  - Requires an **effective PSU**: either an explicit `survey_design.psu` OR a `cluster=<col>` argument that gets injected as the effective PSU per Wave E.1's `_inject_cluster_as_psu` routing. No-effective-PSU survey designs (weights-only / strata-only WITHOUT a cluster fallback) raise `NotImplementedError` at `SpilloverDiD.fit` post-resolution because under the pseudo-PSU = obs-index fallback each pseudo-PSU appears in exactly one period — the per-PSU serial cross-period loop would silently contribute zero. Routing the serial loop to `conley_unit` would mix IF allocators with the spatial term's pseudo-PSU aggregation (the user-confirmed methodology pins a single IF allocator); fail-closed per `feedback_no_silent_failures` until a no-effective-PSU derivation is queued (tracked in DEFERRED.md).
  - Replicate-weight variance (BRR / Fay / JK1 / JKn / SDR) raises `NotImplementedError` (inherits Wave E.1 gate).
  - LinearRegression-side `vcov_type="conley" + survey_design=` gate in `LinearRegression.fit()` (Conley + survey rejection block) remains. **Note (open methodological question):** weighted spatial-HAC under probability sampling is an open methodological question; no canonical extension of Conley (1999) exists for the combination — separate roadmap, not Wave E.
  - DiagnosticReport routing for `SpilloverDiDResults(vcov_type="conley", conley_lag_cutoff > 0, survey_design=)` is queued for Wave F follow-up. Base `_APPLICABILITY` / `_PT_METHOD` registration has **landed** (per `feedback_audit_diagnostic_report_wiring_before_claim`); the panel-block Conley + survey-design *combination* still needs its own downstream-consumability validation before being claimed.

**Implementation:** new sibling helper `_compute_stratified_serial_bartlett_meat` in `diff_diff/two_stage.py` (parallel to the Wave E.2 spatial orchestrator; ~200 LoC; three-mode singleton-stratum branching with FPC scaling inside the multi-PSU branch to avoid divide-by-zero; panel-wide dense time codes for the lag math). Orchestrator `_compute_stratified_conley_meat` signature extended with `conley_lag_cutoff: Optional[int] = None`; spatial loop unchanged; serial helper called after spatial loop when `conley_lag_cutoff > 0`; saturation NaN-fail accounting merges both terms' `(variance_computed, legitimate_zero)` flags. Dispatch in `_compute_gmm_corrected_meat` conley branch threads `conley_lag_cutoff` through to the orchestrator. Spillover-side gate at `spillover.py:2210` deleted (Wave E.2 era `NotImplementedError` for lag>0 + survey). Stage-1 weighted FE solver, `finite_mask` survey-array subsetting, `df_survey` threading, bread weighting, and `SpilloverDiDResults` survey metadata are all inherited UNCHANGED — Psi construction is bit-identical regardless of vcov_type or lag. Tests: `TestSpilloverDiDWaveE2FollowupConleySurveyLagCutoff` + `TestSpilloverDiDWaveE2FollowupConleySurveyLagCutoffEventStudy` at `tests/test_spillover.py`.

**Note (shared helpers):** the within-PSU serial Bartlett kernel matrix construction and the post-meat finite/PSD guard are shared with the no-survey reference path via `_serial_bartlett_kernel_matrix` and `_validate_meat_psd` in `conley.py`. The survey orchestrator's per-PSU serial loop (`_compute_stratified_serial_bartlett_meat`) and the orchestrator's combined-meat PSD guard call the same helpers as `_compute_conley_meat`, so survey and no-survey paths cannot drift on kernel weights or PSD threshold (`< -1e-12`).

### Variance (Wave E.3 — `SurveyDesign.subpopulation()` / warn-and-drop full-design retention via zero-pad scores)

`SurveyDesign.subpopulation()`-derived designs AND warn-and-drop fits now preserve the full-domain resolved survey design for inference bookkeeping. `n_psu`, `n_strata`, `df_survey`, and the Binder TSL per-stratum centering reflect the FULL domain rather than the post-`finite_mask` fit sample. SHIPPED in Wave E.3.

- **Note (documented synthesis — library-convention adoption, NOT new methodology):** Wave E.3 adopts the canonical "zero-pad scores to full panel + retain full-design resolved survey" pattern from R's `survey::svyrecvar` applied to a `subset()` design (Lumley 2010 §2.5 "Domains and subpopulations"). The same convention is already established in `diff_diff/imputation.py:2175-2183` (PreTrendsImputation lead regression — Omega_0 scores zero-padded back to full panel length so PSU/strata structure is maintained) and `diff_diff/prep.py:1401-1432` (DCDH cell variance — IF zero-padded outside the cell, preserving full strata/PSU structure). Wave E.3 propagates the same convention to SpilloverDiD's Wave E.1 Binder TSL × Wave D Gardner GMM × Wave E.2/follow-up stratified-Conley + serial Bartlett meat. The "documented synthesis" framing applies because no SpilloverDiD-specific derivation is required — Psi is constructed on the survey-finite-mask subset (`finite_mask & (survey_weights > 0)`) and then explicitly zero-padded back to full panel length at the meat-helper boundary via the new `score_pad_mask=survey_finite_mask` kwarg, so excluded rows contribute exactly zero score by construction. The meat helpers' per-stratum centering + FPC scaling produce the same numeric output as if those rows had been correctly zero-padded inside the helpers.

- **Mechanical realization (one new `_compute_gmm_corrected_meat` kwarg):** the gamma_hat / Psi construction stays on the SURVEY-FINITE-MASK inputs (`X_1_sparse_fit`, `X_10_sparse_fit`, `eps_10_fit` built on `survey_finite_mask = finite_mask & survey_weights > 0`; `X_2_kept_gamma`, `eps_2_fit_gamma`, `survey_weights_fit_gamma` projected from the fit-sample frame down to survey_finite_mask) so the drop-first stage-1 FE column space is bit-identical to the pre-E.3 path — critical because `_build_butts_fe_design_csr` re-factorizes inputs via `pd.factorize` and drops the first unit / time code; if a warn-dropped unit sorts first, the full-length build would produce a DIFFERENT column space (all-zero `X_10` column for the dropped unit → rank-deficient `X_10' W X_10` → LSMR fallback → different `gamma_hat`). The full-domain zero-pad invariant is delivered by:
  - **`score_pad_mask=survey_finite_mask`**: a new optional kwarg on `_compute_gmm_corrected_meat`. When provided, the helper zero-pads the survey-finite-mask `Psi` (built on rows where BOTH `finite_mask == True` AND `survey_weights > 0`) to full panel length AFTER construction but BEFORE kernel dispatch via `Psi_padded = np.zeros((n_full, p_2)); Psi_padded[score_pad_mask] = Psi`. Warn-dropped rows have positive survey weights but are physically removed from the fit sample (NaN `y_tilde` → `finite_mask=False`). Subpopulation rows have weight=0 via `_subpop_weight` and are explicitly excluded by the `(survey_weights > 0)` filter; the prior R6 fix added this filter because otherwise `_build_butts_fe_design_csr`'s `pd.factorize` compaction would include zero-weight rows in the FE drop-first basis ordering and silently shift `gamma_hat`. Both drop mechanisms are zero-padded back into the meat boundary by the same `score_pad_mask` step. Either way, excluded rows contribute exactly zero score at the meat boundary.
  - **Kernel-dispatch arrays at FULL length**: `cluster_ids_full`, `coord_array_full`, `np.asarray(time_vals)`, `np.asarray(unit_vals)`, `resolved_survey` are passed un-subsetted so the meat helpers (Binder TSL / stratified-Conley / serial Bartlett) see the full-domain PSU / strata / centroid / time geometry.
  - **Resolved survey design is NOT subsetted via `finite_mask`** — the prior Wave E.1 `replace(resolved_survey, weights=resolved_survey.weights[finite_mask], ...)` block at `spillover.py:2845-2896` is removed.
  - **Conley validator length**: `_validate_conley_kwargs` at `two_stage.py:222-237` reads `n_for_conley = len(score_pad_mask)` when the kwarg is set, so the Conley shape checks see the full-length geometry (matches the post-pad Psi length the kernel actually consumes).
- **`gamma_hat` solve invariance:** the gamma_hat solve at `two_stage.py:244-269` operates on fit-sample inputs — bit-identical to the pre-E.3 path. Pre-E.3 baseline-parity test at `TestSpilloverDiDWaveE3SubpopulationFullDesign::test_c` pins this via fixed goldens.
- **Bread sandwich invariance:** the bread `A_22^{-1} = (X_2_kept' W X_2_kept)^{-1}` at `spillover.py:3187-3214` still uses the fit-length `X_2_kept` because the OLS solve operates on the active sample. Mathematically `A_22_full = X_2_full' W_full X_2_full` equals `A_22_kept` because the zero-weight rows contribute zero to the cross-product.
- **Conley centroid scope:** under Wave E.3 the survey path passes the FULL-LENGTH `coord_array_full` (no `finite_mask` or `weight > 0` filter) to `_compute_stratified_conley_meat`, which then derives panel-constant per-PSU centroids from all coord rows present in the full panel. Zero-weight subpopulation rows AND warn-and-dropped rows both pin their PSU's geometric centroid — matches R `svyrecvar` (the PSU's geometric footprint is fixed by sampling design, not by the analyst's domain filter or by stage-1 FE identifiability). On the no-survey path the conley helper still receives the fit-length `coord_array_fit` (preserves the pre-E.3 Wave E.2 R4 fix invariant for the no-survey case).
- **Panel-block conley + subpopulation (Wave E.2 follow-up cross-reference):** the panel-block `n_h_panel` FPC convention (panel-wide unique PSU count per stratum, see Wave E.2 follow-up subsection above) is preserved under subpopulation — zero-weighted PSU-period cells still count toward `n_h_panel` because the design retains those PSUs at full length. End-to-end smoke at `test_g1` / `test_g2` (conley + lag>0 + subpopulation × explicit-PSU and cluster-injection branches) covers the path; a dedicated `n_h_panel` hand-computation test on a fixture with mixed zero/positive-weight periods per PSU is deferred to the Wave E.3 parity follow-up.
- **Reduction semantics:**
  - When `finite_mask.all() == True` AND all weights `> 0` (no zero-pad needed): bit-identical to shipped Wave E.2 / E.2-follow-up baseline. `assert_array_equal` regression pin at `test_c`.
  - A2 invariant: warn-and-drop and subpopulation drops are treated identically (both apply the zero-pad mechanism). Test class explicitly covers both drop mechanisms in test (`a`), (`b`), (`i2`). The "both mechanisms compose cleanly" case (subpop-excluded row that is ALSO warn-and-dropped) is locked at test (`i2`).
  - Subpopulation parity vs upstream-subset: `df_survey` matches the full domain (`n_psu_full - n_strata_full`) regardless of how many rows the subpopulation mask excludes. SE may differ from `fit(data[mask], ..., survey_design=plain_design)` by design (subpopulation retains zero-padded PSU geometry; subset drops PSUs entirely).
- **Restrictions / unchanged from Wave E.1/E.2/follow-up:**
  - Replicate-weight variance (BRR / Fay / JK1 / JKn / SDR) raises `NotImplementedError` (Wave E.1 gate at `spillover.py:2400` inherited; per-replicate refit + subpopulation is a separate follow-up).
  - TwoStageDiD's analogous always-treated design-subset pattern was shipped in a follow-up PR (TwoStageDiD Wave E.3 parity) — see "Note (documented synthesis — Wave E.3 parity, full-domain survey design under always-treated drop)" in the TwoStageDiD section above. The two estimators now share the full-design retention contract; the trigger differs (TwoStageDiD: always-treated unit detection; SpilloverDiD: `finite_mask` warn-and-drop / subpopulation zero-weight rows) but the invariant is identical (zero-pad scores at the meat-helper boundary; retain full-domain `resolved_survey.psu / strata / fpc`).
  - `df_for_inference` at `spillover.py:3229-3234` reads `resolved_survey_fit.df_survey` which is the full-domain value under Wave E.3 (since `resolved_survey_fit = resolved_survey`). All four `safe_inference` call sites (aggregate `tau_total`, per-ring `delta_j`, event-study per-event-time `tau_k` / `delta_jk`, scalar `att` lincom) inherit the full-domain df.

**Implementation:** `spillover.py:2845-2896` design-subset block deleted; `survey_weights_fit = survey_weights[finite_mask]` retained for the stage-2 OLS solve (which still operates on the fit sample); `cluster_ids_full[finite_mask]` subset dropped on the survey path (`cluster_ids_fit = cluster_ids_full` under survey). The `_compute_gmm_corrected_meat` call at `spillover.py:3163` is rewired to receive SURVEY-FINITE-MASK gamma_hat-construction inputs (`X_1_sparse_fit`, `X_10_sparse_fit`, `eps_10_fit` built on `survey_finite_mask`; `X_2_kept_gamma`, `eps_2_fit_gamma`, `survey_weights_fit_gamma` projected from the fit-sample frame down to survey_finite_mask) plus FULL-LENGTH kernel-dispatch arrays (`cluster_ids_for_meat`, `conley_coords_for_meat`, `conley_time_for_meat`, `conley_unit_for_meat`, `resolved_survey_fit`) plus the new `score_pad_mask=survey_finite_mask` kwarg; no-survey path passes `score_pad_mask=None` and uses fit-length variables throughout (bit-identical to pre-E.3). `_compute_gmm_corrected_meat` at `two_stage.py` adds one new optional kwarg `score_pad_mask: Optional[np.ndarray] = None` and one post-Psi-construction zero-pad block; the `_validate_conley_kwargs` call uses `n_for_conley = len(score_pad_mask)` when the kwarg is set. Within-unit-constancy validator at `spillover.py:2913` updated to operate on full-length unit array (matches full-length cluster array under Wave E.3). Second `compute_survey_metadata` recompute at `spillover.py:2954-2959` now uses full-length `raw_w` (no `[finite_mask]` subset). Tests: `TestSpilloverDiDWaveE3SubpopulationFullDesign` + `TestSpilloverDiDWaveE3SubpopulationFullDesignEventStudy` at `tests/test_spillover.py`.

**Edge cases (from paper Section 3.2 / Discussion):**

| # | Edge case | Handling |
|---|---|---|
| 1 | No nearby control units (Assumption 5(ii) fails) | Hard error via `_validate_far_away_exists` |
| 2 | `d_bar` too small (some affected units classified as far) | User responsibility; sensitivity analysis across `d_bar` values recommended (vary `rings[-1]`) |
| 3 | `d_bar` too large (`S=0` group fewer characteristics) | Same as #2; bias-variance trade-off (paper page 13-14) |
| 4 | Single-ring `S_i` attenuation (many unaffected units) | Use multiple rings (Equation 6 multi-ring spec); supported by passing more breakpoints |
| 5 | Spillovers extend past largest ring | User inspects outermost `delta_K` significance; if non-zero, extend the outermost ring |
| 6 | Additive spillovers in count of nearby treated | `ring_method="count"` deferred (paper notes count form re-introduces functional-form dependence) |
| 7 | Staggered + negative Goodman-Bacon weights | Two-stage Gardner methodology avoids this (paper page 22) |

**Restrictions / deferred features:**

- `event_study=True` SHIPPED in Wave C — see Event-study mode subsection above. Emits `att_dynamic`, MultiIndex `spillover_effects`, and a TwoStageDiD-compatible `event_study_effects` dict alias.
- `survey_design=` for `vcov_type ∈ {"hc1"}` (plus `cluster=<col>` for CR1) SHIPPED in Wave E.1 — see "Variance (Wave E.1)" subsection below. Threads Hájek-normalized survey weights through stage-1 FE estimation, gamma_hat solve, eps construction, and bread inversion; aggregates the Wave D Psi to PSU totals and routes through the audited `_compute_stratified_meat_from_psu_scores` Binder TSL meat helper. `vcov_type="conley"` combined with `survey_design=` SHIPPED in Wave E.2 for cross-sectional Conley (`conley_lag_cutoff = 0`) — see "Variance (Wave E.2)" subsection below (stratified-Conley sandwich on PSU totals). Wave E.2 follow-up adds the panel-block composition (`conley_lag_cutoff > 0`) via spatial + serial Bartlett HAC — see "Variance (Wave E.2 follow-up)" subsection below. `SurveyDesign.subpopulation()` and warn-and-drop full-design retention via zero-pad scores SHIPPED in Wave E.3 — see "Variance (Wave E.3)" subsection below (matches R `survey::svyrecvar(subset())` + in-library precedent at `imputation.py:2175-2183` and `prep.py:1401-1432`). Replicate-weight variance (BRR / Fay / JK1 / JKn / SDR) raises `NotImplementedError` — Gerber (2026) Appendix A notes the IF-reweighting shortcut does NOT apply to TwoStageDiD-class estimators because `gamma_hat` is weight-sensitive; correct support requires per-replicate full re-fit and is queued as a follow-up.
- `covariates=` raises `NotImplementedError` — Gardner-style stage-1 residualization not yet wired through; planned follow-up.
- `ring_method="count"` not exposed — only the nearest-treated-ring specification.
- `vcov_type` ∈ {`"hc2"`, `"hc2_bm"`, `"classical"`} raises `NotImplementedError` — `hc2`/`hc2_bm` because current stage-2 inference uses generic residual df rather than per-coefficient Bell-McCaffrey / CR2 DOF; `classical` because the Wave D Gardner GMM first-stage correction has not been derived for the classical homoskedastic variance (different meat structure `sigma_hat^2 * (X_10' X_10)` vs the Wave D IF outer product `Psi' Psi`). Use `"hc1"` or `"conley"`, or pair with `cluster=` for CR1 — all three apply the Wave D GMM correction.
- **`rings[0]` must equal 0** — the partition must cover treated locations (`d_it = 0` belongs to Ring 1). Rings starting at a nonzero inner edge would leave units in `0 <= d_it < rings[0]` as exposed-but-unmodeled, silently biasing the estimator. Validator rejects such inputs.
- **Balanced panel required (Wave B MVP)** — every unit must observe every period. An unbalanced (unit, time) Ω₀ bipartite graph can produce disconnected FE components and unidentified stage-1 residuals on treated rows. Exact graph-connectivity-based identification (which would relax this to a strictly weaker condition) is queued as a follow-up extension. Validator rejects unbalanced inputs.
- **One row per `(unit, time)` cell required** — duplicate cells silently re-weight stage-1 FE estimation AND stage-2 OLS. Validator rejects duplicate cells.
- Data-driven `d_bar` selection (Butts 2021b / Butts 2023 JUE Insight) not exposed.
- Gardner GMM first-stage correction at stage 2 SHIPPED in Wave D — see "Variance (Wave D)" subsection above. Applies unconditionally across HC1 / Conley / cluster.

**Implementation:** `diff_diff/spillover.py`. Public class `SpilloverDiD`; result class `SpilloverDiDResults(DiDResults)` at `diff_diff/results.py`. Tests at `tests/test_spillover.py`; DGP factories `tests/_dgp_utils.py::generate_butts_nonstaggered_dgp` / `generate_butts_staggered_dgp` (satisfy Butts Assumptions 1/3/5/7 by construction).

---

## ConleySpatialHAC

**Primary source:** Conley, T. G. (1999). GMM Estimation with Cross-Sectional Dependence. *Journal of Econometrics* 92(1), 1-45. DOI: 10.1016/S0304-4076(98)00084-0

**Secondary sources:**
- Andrews, D. W. K. (1991). Heteroskedasticity and autocorrelation consistent covariance matrix estimation. *Econometrica* 59(3), 817-858.
- Düsterhöft, C. (2021). conleyreg: Estimations using Conley Standard Errors. CRAN R package, https://github.com/cdueben/conleyreg. Our parity benchmark target.
- Colella, F., Lalive, R., Sakalli, S. O., & Thoenig, M. (2019). Inference with Arbitrary Clustering. IZA DP No. 12584. Stata `acreg` reference implementation; cited as the parallel canonical implementation in the Stata ecosystem (not parity-tested here).

**Scope:** Spatial heteroskedasticity-and-autocorrelation-consistent
standard errors for OLS when residuals are spatially (and optionally
temporally) correlated. Extends White (1980) HC0 by allowing pairwise
correlation that decays with geographic distance, plus a within-unit
Newey-West-style Bartlett temporal HAC on panel data.

- **Note (rank-guarded design bread):** The spatial-HAC sandwich bread `(X'WX)^{-1}` (`_compute_conley_vcov`, `conley.py`) is inverted by the shared `_rank_guarded_inv` (`diff_diff/linalg.py`). A near-singular design Gram previously returned a garbage inverse (~1e13) and an *exactly* singular one **raised `ValueError`**; the bread now rank-reduces to a finite SE on the identified subspace (NaN only at rank 0) and warns, matching the other structural bread guards. A dropped (unidentified) regression coefficient is reported with **NaN** SE (its row/col in the returned vcov), not the zero-filled `0`. **Behavior change:** a rank-deficient design (collinear regressors) no longer raises — it rank-reduces with a warning. The well-conditioned path is unchanged (`np.linalg.solve(A, I)`). `_rank_guarded_inv` is imported lazily inside the function because `linalg` imports `conley` (one-way), so a top-level import would be circular. See the CallawaySantAnna "rank-guarded IF standard errors" Note.

**Two operating modes:**
- **Cross-sectional (Phase 1):** Pass `vcov_type="conley"` plus
  `conley_coords` (n × 2 array) and `conley_cutoff_km` on direct
  `compute_robust_vcov` / `LinearRegression`.
- **Panel block-decomposed (Phase 2):** Additionally pass the three
  co-required kwargs `conley_time` (n-length array), `conley_unit` (n-length
  array), and `conley_lag_cutoff=<int>` (non-negative). `MultiPeriodDiD` and
  `TwoWayFixedEffects` auto-derive `conley_time` and `conley_unit` from the
  estimator's `time` / `unit` column-name arguments; only
  `conley_lag_cutoff` is set on the constructor.

**Panel API restrictions (Phase 2 + Wave A):**
- `DifferenceInDifferences(vcov_type="conley")` is supported (Wave A #118):
  pass `unit=<col>` as a fit-time argument to `fit(...)` (NOT on `__init__`;
  unused unless Conley is set). DiD inherits the same panel block-decomposed
  sandwich as MPD/TWFE on the two-period design.
- `SyntheticDiD(vcov_type="conley")` raises `TypeError`. SyntheticDiD uses
  bootstrap/jackknife/placebo variance, not the analytical sandwich.
- TWFE's default auto-cluster on the Conley path is silently dropped (no
  combined kernel from auto-cluster). Explicit `cluster=<col>` + Conley
  enables the **combined spatial + cluster product kernel** (Wave A #119;
  see "Combined spatial + cluster product kernel" subsection below). On
  the panel path the validator enforces that cluster membership is
  constant within each unit across periods.
- `inference="wild_bootstrap"` + Conley raises (wild bootstrap is a separate
  inference path that does not consume the analytical sandwich).

**Note (DiD vs TWFE cluster asymmetry on the Conley path):** TWFE auto-clusters
at the unit level by default, so combining with Conley silently drops the
auto-cluster (otherwise every between-unit pair would be zeroed out, defeating
the spatial pooling). To opt into the combined kernel, the user must pass an
explicit `cluster=<col>` that is constant within each unit (typically an
above-unit grouping like region). DiD has no auto-cluster — combining with
Conley is fully opt-in: absent `cluster=`, pure Conley spatial HAC applies;
with `cluster=`, the combined kernel applies. This asymmetry preserves the
existing TWFE auto-cluster contract while making the cluster intent explicit
on the Conley path.

**Variance estimator — cross-sectional (Phase 1, Conley 1999 Eq 4.2 in pairwise-distance form, OLS specialization):**

    Var̂(β) = (X'X)^{-1} · ( Σ_{i,j} K(d_ij / h) · X_i ε_i ε_j X_j' ) · (X'X)^{-1}

where `d_ij` is the geographic distance, `h` is the user-supplied bandwidth
(`conley_cutoff_km`), and `K(·)` is the kernel. The `i = j` diagonal contributes
the standard White HC0 term `X_i ε_i² X_i'`.

**Variance estimator — panel block-decomposed (Phase 2, R `conleyreg` form
with `lag_cutoff > 0`):**

    XeeX_spatial = Σ_t  Σ_{i,j∈units}    K_space(d_ij/h)             · X_{i,t} ε_{i,t} ε_{j,t} X_{j,t}'
    XeeX_serial  = Σ_u  Σ_{|t-s|≤L,t≠s}  (1 - |t-s|/(L+1))           · X_{u,t} ε_{u,t} ε_{u,s} X_{u,s}'
    Var̂(β)       = (X'X)^{-1} · ( XeeX_spatial + XeeX_serial ) · (X'X)^{-1}

The spatial part sums **within each time period only** (cross-time spatial
pairs are NOT paired). The serial part sums **within each unit only** with
`lag = 0` (same-time) excluded to avoid double-counting the diagonal already
in the spatial component. The temporal kernel is hardcoded Bartlett-style
regardless of `conley_kernel` (matches `conleyreg::time_dist.cpp`). This is
NOT a multiplicative product kernel — verified against R `conleyreg` at
~1e-14 on the panel parity fixtures.

**Note (deviation from R-symmetric API):** R `conleyreg`'s `kernel` argument
controls ONLY the spatial component; the temporal kernel is unconditionally
the Bartlett form `(1 - |lag|/(L+1))` (visible in `conleyreg::time_dist.cpp`
where the formula is written explicitly with no `bartlett` flag passed
through). diff-diff matches this asymmetry exactly for R parity. Independent
temporal kernel choice would be a follow-up API extension if user demand
emerges.

**Note (deviation from R conleyreg literal: time-label normalization):**
R `conleyreg` uses raw `time` values directly in the lag computation
(`time_dist.cpp`'s `t_diff = abs(times - times[i])`). On non-dense time
encodings (e.g., `time = 202012, 202101` for monthly panels), the raw
difference is 89, so a `lag_cutoff=1` request silently drops valid lag-1
serial pairs in R. diff-diff normalizes `time` to dense panel-period codes
`0..T-1` via `np.unique(return_inverse=True)` before the lag computation,
so `conley_lag_cutoff` always counts panel periods regardless of label
encoding (int year, YYYYMM, datetime64, `pd.Period`, strings). On dense
integer labels (the parity-test convention), the two paths produce
bit-identical results. For non-dense encodings, diff-diff is the more
robust default; pass `time` as a dense integer index for bit-exact R parity.

**Kernel functions:**
- `conley_kernel="bartlett"` (default): `K(u) = max(0, 1 - |u|)` evaluated on the pairwise distance `d_ij/h`. The radial 1-D form on pairwise distance, matching R `conleyreg`, Stata `acreg` (Colella et al. 2019), and Hsiang (2010).
- `conley_kernel="uniform"`: `K(u) = 1{|u| ≤ 1}`. Conley 1999 page 11; spectral window negative in regions (footnote 11).

**Note (deviation / source specialization):** Conley 1999's explicitly PSD-guaranteed Bartlett formula (Eq 3.14, page 12) is the **2-D separable product window** `K(j, k) = (1 - |j|/L_M)(1 - |k|/L_N)` indexed on a lattice. The 1-D radial form on pairwise distance that diff-diff implements (matching R `conleyreg`) is a practitioner specialization that is not explicitly written in the paper and is therefore **not formally PSD-guaranteed**. We apply the same indefiniteness check to both kernels: a `UserWarning` is emitted if any meat eigenvalue is materially negative (< `-1e-12`).

**Note (shared helpers):** the within-unit serial Bartlett kernel matrix construction and the finite/PSD guard are shared with the SpilloverDiD Wave E.2 follow-up survey path via `_serial_bartlett_kernel_matrix` and `_validate_meat_psd` in `conley.py`. The no-survey reference path (`_compute_conley_meat`) and the survey panel-block path (`_compute_stratified_serial_bartlett_meat`'s caller in `two_stage.py`) exercise the same kernel weights and PSD threshold (`< -1e-12`); see the parallel note in the SpilloverDiD Wave E.2 follow-up section below.

**Distance metrics:**
- `conley_metric="haversine"` (default): great-circle in km using Earth's mean radius (6371.01 km, matching R `conleyreg`). Validates `lat ∈ [-90, 90]`, `lon ∈ [-180, 180]`.
- `conley_metric="euclidean"`: Euclidean from projected coords. Skips lat/lon range checks (user owns the projection's units).
- `conley_metric=callable(coords1, coords2) -> n×n array`: custom distance for non-geographic networks.

**Note:** No default bandwidth. Conley 1999 does not propose a plug-in selector;
the empirical example (Section 5) uses a sensitivity grid. Implementation
requires `conley_cutoff_km` to be supplied; `None` raises `ValueError` (per the
project's no-silent-failures rule). Practitioners should rerun on a coarse cutoff
grid (e.g., 50, 100, 200, 500 km) and report the SE range, mirroring Conley's
Section 5 robustness check.

**Note (FWL composability under TWFE):** Conley's meat depends only on
scores `X_i·ε_i`, which FWL preserves under within-transformation. The
block-decomposed sandwich applied to FE-residualized scores produces the
same meat as the full-dummy-expansion design (unlike `vcov_type="hc2"` /
`vcov_type="hc2_bm"`, whose leverage corrections depend on the full hat
matrix). `TwoWayFixedEffects(vcov_type="conley", conley_lag_cutoff=...)`
threads the within-transformed scores plus the original `time` / `unit`
vectors into the same helper that `LinearRegression` uses.

**Note (R conleyreg parity):** diff-diff's Conley implementation matches R
`conleyreg` (Düsterhöft 2021, CRAN v0.1.9) to ≤ 1e-6 on six benchmark
fixtures (`benchmarks/data/r_conleyreg_conley_golden.json`): three
cross-sectional (Phase 1) plus three panel fixtures with `lag_cutoff > 0`
(Phase 2). Earth radius constant is 6371.01 km (mean radius), matching
`conleyreg::haversine_dist`. Regeneration:
`cd benchmarks/R && Rscript generate_conley_golden.R`.

### Combined spatial + cluster product kernel (Wave A #119)

When `cluster_ids` is supplied alongside `vcov_type="conley"`, the meat
applies the combined product kernel:

    K_total[i, j] = K_space(d_ij/h) · 1{cluster_i = cluster_j}

On the panel block-decomposed path the cluster indicator multiplies BOTH
the within-period spatial sandwich AND the within-unit serial sandwich:

    XeeX_spatial = Σ_t Σ_{i,j∈units}    K_space(d_ij/h) · 1{c_{i,t}=c_{j,t}} · X_{i,t} ε_{i,t} ε_{j,t} X_{j,t}'
    XeeX_serial  = Σ_u Σ_{|t-s|≤L, t≠s} (1 - |t-s|/(L+1))  · 1{c_{u,t}=c_{u,s}}  · X_{u,t} ε_{u,t} ε_{u,s} X_{u,s}'

**Cluster-time-invariance contract:** on the panel block-decomposed path
the validator REQUIRES that `cluster_ids` be constant within each unit
across periods. The within-unit serial sandwich's cluster mask is then
trivially all-ones, and the math simplifies to the bare serial Bartlett
HAC weighted by the spatial mask only. If a unit's cluster changes
across periods (e.g. a unit migrating between regions), the within-unit
mask would zero out adjacent-time pairs that should contribute,
producing a methodologically-muddled meat — the validator raises
`ValueError` naming the violating unit(s). The cross-sectional path
has no time dimension, so no invariance constraint applies.

**Note:** R `conleyreg` does not support a combined spatial + cluster
product kernel; this is a diff-diff convention validated by two limit
fixtures rather than R parity:
1. **All-unique-clusters reduction:** when every observation is in its
   own cluster, the cluster mask is the identity, and the meat reduces
   to the diagonal HC0 contribution `Σ_i X_i ε_i² X_i'`.
2. **Huge-cutoff reduction:** when `conley_cutoff_km` is large enough
   that `K_space = 1` on every pair, the meat reduces to the pure
   within-cluster sum `Σ_g X_g' ε_g ε_g' X_g` (CR1 without the
   Liang-Zeger small-sample correction). This exact reduction holds
   only for `conley_kernel="uniform"` (`K_uniform(u) = 1` for `|u| ≤ 1`).
   The Bartlett kernel gives `K_bartlett(u) = 1 - |u|`, which is
   strictly less than 1 for `0 < |u| ≤ 1`, so the huge-cutoff limit
   under Bartlett is asymptotic (`K → 1` as `cutoff → ∞` only at finite
   off-diagonal distances), not exact at any finite cutoff. The fixture
   anchor uses uniform for an exact identity check.

The combined-kernel meat is well-defined either way; the two fixture
limits anchor the math, and the panel time-invariance contract
guarantees the serial component is unaffected by the cluster choice.

### Performance / scale (Wave A #120)

A sparse k-d-tree fast path auto-activates for the spatial Bartlett meat
when `n > _CONLEY_SPARSE_N_THRESHOLD` (default 5,000) AND `conley_metric`
is `"haversine"` or `"euclidean"` (NOT a callable) AND `conley_kernel`
is `"bartlett"`. The dense O(n²) distance matrix is replaced with a CSR
sparse kernel matrix built from `scipy.spatial.cKDTree.query_ball_tree`
neighbor queries.

**Why bartlett-only:** Bartlett at `u = 1.0` returns exactly `0.0`, so
pairs at exactly the cutoff distance contribute zero to the meat — the
sparse path can safely drop them. Uniform at `u = 1.0` returns `1.0`,
which would require a closed-interval query semantic that the haversine
chord-projection roundoff cannot reliably preserve. The auto-toggle
falls back to the dense path for uniform regardless of n. Callable
metrics also fall back (kd-tree needs a vectorizable Minkowski distance).

For haversine, the kd-tree operates on a 3-D unit-sphere projection
(`x = cos(lat)cos(lon), y = cos(lat)sin(lon), z = sin(lat)`) with the
chord radius matching the arc-length cutoff; the exact great-circle
distance is recomputed only for in-range neighbors before the kernel
evaluation. The numerical tolerance vs the dense path is typically
~1e-12 in absolute terms; R parity at `atol=1e-6` is preserved on the
existing fixtures under the auto-toggle.

The `_CONLEY_DENSE_OOM_WARN_N = 20_000` constant remains as a separate
warning threshold for the dense fallback (callable metrics, uniform
kernel) where O(n²) memory is at material risk. The two thresholds are
independent — sparse auto-toggle at 5,000 is a compute optimization;
dense OOM warning at 20,000 is a memory caution.

**Density gate (`_CONLEY_SPARSE_DENSITY_THRESHOLD = 0.3`):** the sparse
path's CSR storage carries ~12 bytes per non-zero (data + indices +
indptr) vs 8 bytes per cell for dense float64. The memory crossover
is at ~67% density, but at high density the CSR overhead loses its
advantage well before that. The sparse helper measures actual neighbor
density via `cKDTree.count_neighbors` (shares tree traversal with
`query_ball_tree`, no extra allocation) and falls back to the dense
path with a `UserWarning` when neighbor density exceeds 30%. This
prevents the "sparse" path from silently using MORE memory than dense
when cutoffs are large relative to the data span (e.g. cutoffs above
half-Earth circumference on a global panel, or unit-scale cutoffs on
a clustered dataset). Users see one line explaining the fallback so
they can either reduce `conley_cutoff_km` or accept the dense path.

### Callable conley_metric validation (Wave A #123)

When `conley_metric` is a user-supplied callable, the result is
validated at the boundary via `_validate_callable_metric_result`:

1. Result casts to a float64 array (raises `ValueError` if not).
2. Shape is exactly `(n, n)` (raises if mismatched).
3. All entries are finite (NaN/inf raises).
4. All entries are non-negative (negative distances raise).
5. Symmetric to within `atol=1e-10` (asymmetric matrix raises).
6. Zero diagonal: `|d(i, i)| ≤ 1e-10` for all `i` (nonzero diagonal raises).

Each failure produces a `ValueError` naming the violated invariant.
Sub-tolerance asymmetry (eps-level roundoff) is accepted. The zero-
diagonal invariant is load-bearing for the Conley sandwich: the
`i = j` term contributes `K(d_ii / h) · X_i ε_i² X_i'`, which must
reduce to the HC0 diagonal `X_i ε_i² X_i'` (i.e., `K(0) = 1`). A
callable with positive self-distance would attenuate the HC0 term
by `K(d_ii / h) < 1` and silently misstate Conley SEs. Built-in
metrics (`"haversine"`, `"euclidean"`) satisfy this by construction.

**Edge cases / restrictions:**
- `DifferenceInDifferences(vcov_type="conley")` is supported (Wave A #118): pass `unit=<col>` to `fit(...)` (NOT on `__init__`; unused unless Conley is set; not part of `get_params()` / `set_params()`).
- `MultiPeriodDiD` / `TwoWayFixedEffects` / `DifferenceInDifferences` `+ vcov_type="conley"` without `conley_lag_cutoff` → `ValueError`.
- `MultiPeriodDiD` / `DifferenceInDifferences` `(vcov_type="conley")` without `unit=` at fit-time → `ValueError`.
- `TwoWayFixedEffects(vcov_type="conley", cluster=<col>)` is supported (Wave A #119): combined spatial + cluster product kernel applies. The cluster must be time-invariant within each unit on the panel path (validator-enforced). TWFE's default auto-cluster is silently dropped on the Conley path; explicit cluster is required to opt in.
- `DifferenceInDifferences(vcov_type="conley", cluster=<col>)`: combined kernel applies; same time-invariance contract on the panel path. DiD has no auto-cluster, so the cluster choice is fully explicit.
- `DifferenceInDifferences` / `MultiPeriodDiD` / `TwoWayFixedEffects` `(vcov_type="conley", inference="wild_bootstrap")` → `NotImplementedError`. (MPD's pre-Conley analytical-fallback `UserWarning` is suppressed when `vcov_type="conley"` so the user gets one consistent error message.) **Note (mode qualifier, 3.9):** this is the STATIC contract; in the TWFE event-study mode (`event_study=True`, row M-010) the mode-level wild `ValueError` fires first - a mode-capability error precedes the Conley front door (see the TwoWayFixedEffects event-study section).
- `DifferenceInDifferences` / `MultiPeriodDiD` / `TwoWayFixedEffects` `(vcov_type="conley")` + `survey_design=` → `NotImplementedError` at the estimator level. **Note (open methodological question):** weighted spatial-HAC under probability sampling is an open methodological question; no canonical extension of Conley (1999) exists for the combination.
- `SyntheticDiD(vcov_type="conley")` → `TypeError` (SyntheticDiD uses bootstrap/jackknife/placebo variance, not the analytical sandwich; tracked in DEFERRED.md).
- Generic `LinearRegression(vcov_type="conley", survey_design=...)` → `NotImplementedError`. Generic `LinearRegression / compute_robust_vcov` Conley rejects `weights=` for any `weight_type` (`pweight` / `aweight` / `fweight`) → `NotImplementedError` (weighted Conley is not implemented on the generic linalg surface). `compute_robust_vcov` does not accept `survey_design=`; the survey-design surface is `LinearRegression` only. **Note (open methodological question):** the `pweight` / `survey_design` subset additionally reflects an open methodological question — no canonical extension of Conley (1999) exists for weighted spatial-HAC under probability sampling. (Estimator-specific shipped surfaces — SpilloverDiD via Wave E.1/E.2/E.3 and TwoStageDiD via Wave E.3 parity — are explicitly excepted; see "Note (deferral status, 2026-05-26)" below.)
- Panel path: partial `conley_time` / `conley_unit` / `conley_lag_cutoff` (not all three set) → `ValueError` at validator.
- Panel path with `cluster_ids` that vary across periods within a unit → `ValueError` (time-invariance contract).
- `n > 20_000` with `conley_kernel='uniform'` or callable metric: emits `UserWarning` about O(n²) distance-matrix memory (sparse fast path doesn't apply; consider switching to bartlett or projecting to euclidean for performance).
- `conley_cutoff_km ≤ 0`, `nan`, or `inf`: rejected with `ValueError`. The HC0 reduction at h→0 is documented but not the sanctioned path; users should pass `vcov_type="hc1"`.
- Identical coordinates (`d_ij = 0` for `i ≠ j`): `K(0) = 1`, contributing the full HC0 weight per Conley 1999 page 19. Documented behavior; no warning.
- Callable `conley_metric` returning a non-(n,n)/NaN/inf/negative/asymmetric/non-zero-diagonal matrix raises `ValueError` naming the violated invariant. The zero-diagonal contract (`|d(i, i)| ≤ 1e-10`) is load-bearing for the Conley sandwich's HC0 reduction `K(0) = 1`; see "Callable conley_metric validation" subsection above.

**Note (deferral status, 2026-05-26):** Conley + survey_design / weights surface boundary —
- **Shipped (survey)**: SpilloverDiD + Conley + survey via Wave E.1/E.2/E.3 (PR #468 / #474 / #482, stratified-Conley sandwich on PSU totals with within-PSU serial Bartlett HAC for `lag_cutoff > 0`); TwoStageDiD + Conley + survey via Wave E.3 parity (PR #485 / `fdf2cebc`).
- **Shipped (non-survey OLS-sandwich)**: `DifferenceInDifferences` / `MultiPeriodDiD` / `TwoWayFixedEffects` (Wave A #118-#123), plus `SunAbraham` and `WooldridgeDiD`-OLS — conley threaded through the within-transform design via `solve_ols` / `conley.py` (cross-sectional + panel block-decomposed; no survey/weights). `StackedDiD` conley is deferred for a **methodology** reason (unit replication across sub-experiments breaks the spatial-distance interpretation; no `conleyreg` anchor — see the StackedDiD variance-families row), NOT plumbing.
- **Deferred (generic linalg surface — any weight_type)**: DiD / MPD / TWFE / LinearRegression generic path + Conley + `survey_design=`; `LinearRegression` / `compute_robust_vcov` Conley + `weights=` (rejected for `pweight`, `aweight`, AND `fweight`; weighted Conley is not implemented on the generic linalg surface).
- **Open methodological question (subset)**: the `pweight` / `survey_design` portion of the deferral additionally has no canonical methodological extension — weighted spatial-HAC under probability sampling lacks a canonical extension of Conley (1999).

**Reference implementations:**
- R: `conleyreg::conleyreg(...)` (Düsterhöft 2021, CRAN v0.1.9) — **parity benchmark for diff-diff**
- Stata: `acreg dep indeps, latitude(lat) longitude(lon) spatial dist(km) bartlett` (Colella et al. 2019) — academic spec source for the cluster-flexible variance, not parity-tested here
- MATLAB: `ols_spatial_HAC.m` (Hsiang 2010)

---

## Survey Data Support

Survey-weighted estimation allows correct population-level inference from data
collected via complex survey designs (multi-stage sampling, stratification,
unequal selection probabilities).

### Weighted Estimation

- **Reference**: Lumley (2004) "Analysis of Complex Survey Samples", Journal of
  Statistical Software 9(8). Solon, Haider, & Wooldridge (2015) "What Are We
  Weighting For?" Journal of Human Resources 50(2).
- **WLS formula**: `beta_WLS = (X'WX)^{-1} X'Wy` where `W = diag(w_i)`
- **Implementation**: Equivalent transformation via `sqrt(w)` scaling, then
  standard OLS. Residuals back-transformed to original scale.
- **Weight types**: pweight (inverse selection probability), fweight
  (frequency/expansion), aweight (inverse variance/precision)
- **Note:** Weight normalization uses `sum(w) = n` convention (DRDID/Stata), not
  raw weights (R `survey`). Coefficients are identical; SEs differ by constant
  factor.

### Taylor Series Linearization (TSL) Variance

- **Reference**: Binder (1983) "On the Variances of Asymptotically Normal
  Estimators from Complex Surveys", International Statistical Review 51(3)
  (paper review on file: `docs/methodology/papers/binder-1983-review.md`).
  Lumley (2004).
- **Formula**: `V_TSL = (X'WX)^{-1} [sum_h V_h] (X'WX)^{-1}` with stratified
  PSU-level scores
- **Relationship to sandwich estimator**: TSL is a generalization of the
  Huber-White sandwich estimator that accounts for stratification and finite
  population correction
- **Deviation from R:** R `survey` defaults `lonely_psu` to "fail"; we default
  to "remove" with warning, matching common applied practice
- **Edge case**: Singleton strata (one PSU per stratum) — handled via
  `lonely_psu` parameter ("remove", "certainty", or "adjust")
- **Note:** For unstratified designs with a single PSU, all `lonely_psu` modes
  produce NaN variance. The "adjust" mode cannot center against a global mean
  when there is only one stratum (the single PSU is the entire sample).
- **Note:** Weights-only designs (no explicit PSU or strata) use implicit
  per-observation PSUs for the TSL meat computation, consistent with the
  stratified-no-PSU path. The adjustment factor is `n/(n-1)` (not HC1's
  `n/(n-k)`).
- **Note:** TSL now precondition-checks `X'WX` via `np.linalg.cond` before
  solving the sandwich. If the condition number exceeds `1/sqrt(eps)` (≈
  6.7e7) a `UserWarning` fires stating that the bread is ill-conditioned
  and variance estimates may be numerically unstable. Previously a near-
  singular `X'WX` could silently produce unstable SEs. Axis-A finding #19
  in the Phase 2 silent-failures audit.

### Weight Type Effects on Inference

- **Note:** aweights use unweighted meat in the sandwich estimator (no `w` in
  `u^2` term). This matches Stata convention. Rationale: aweights model known
  heteroskedasticity; after WLS transformation, errors are approximately
  homoskedastic.
- **Note:** fweights affect degrees of freedom (`df = sum(w) - k`, not
  `n - k`). This matches Stata convention for frequency-expanded data.
- **Note:** pweight HC1 meat uses score outer products (Σ s_i s_i' where
  s_i = w_i x_i u_i), giving w² in the meat. fweight HC1 meat uses
  X'diag(w u²)X (one power of w), matching frequency-expanded HC1.
- **Note:** fweights must be non-negative integers; fractional values are
  rejected by `_validate_weights()`. All-zero vectors rejected at solver
  level. This matches Stata's convention.

### Absorbed Fixed Effects with Survey Weights

- **Note:** When `absorb` is used in DiD/MultiPeriodDiD, all regressors
  (treatment, time, interactions, covariates) are within-transformed alongside
  the outcome per the FWL theorem. Regressors collinear with the absorbed FE
  (e.g., treatment after absorbing unit FE) are dropped via rank-deficiency
  handling. Multiple absorbed variables (weighted or unweighted) are
  within-transformed via the **method of alternating projections**
  (`diff_diff.utils.demean_by_groups`): each variable is demeaned by each FE
  dimension in turn until the iterate converges. This is the exact (weighted)
  FWL residualization onto the combined column space of all absorbed dummies for
  both balanced and unbalanced panels, matching R `fixest` / `reghdfe` / `lfe`.
  A single sequential sweep (the prior behavior) is only the one-iteration
  approximation and is exact only when the FE subspaces are orthogonal (balanced
  fully-crossed panels); on unbalanced panels it was wrong. A single absorbed
  variable converges in one pass (delegates to `demean_by_group`).
- **Note:** The shared within-transformation path (`diff_diff.utils.demean_by_groups`,
  also reached via the two-way `within_transform`) emits a `UserWarning` per call
  when any transformed variable exits the alternating-projection loop without
  reaching `tol` within `max_iter`. This now covers the **unweighted** path as well
  (previously the unweighted two-way transform used a closed-form additive demean
  and could not warn). Defaults: `max_iter=10_000` in both `demean_by_groups` and
  `within_transform` (raised from 100 in v3.6.x to match the compiled-library
  convention - R `fixest` `fixef.iter` and `pyfixest` `fixef_maxiter` both default
  to 10,000; correlated FE incidence such as contiguous unit lifetimes in
  order-level data genuinely requires hundreds of iterations, measured ~250-280 on
  the `tail_stress` benchmark scenario where the old cap of 100 warned and
  returned slightly-off residuals). `tol=1e-8` via `within_transform`
  (TwoWayFixedEffects, SunAbraham, BaconDecomposition, WooldridgeDiD) and `tol=1e-10`
  for the DiD/MultiPeriodDiD `absorb=` path. Balanced panels converge in ~2
  iterations. Worst-case trade-off, accepted deliberately: an input that cannot
  converge at all now burns the full 10,000 iterations before warning
  (correctness-over-latency). Silent return of the current iterate was classified
  as a silent failure under the Phase 2 audit and replaced with this explicit signal.
- **Note:** The MAP inner loop factorizes each absorbed dimension once
  (`pd.factorize`) and forms group means via `np.bincount` accumulation. Plain
  summation is not Kahan-compensated the way pandas `groupby().mean()` is, so
  demeaned values agree with the pre-v3.6.x pandas implementation to ~1e-10 order
  (drift compounds across MAP iterations), not bit-for-bit; estimator estimates
  are validated unchanged at the FE-absorption benchmark identity gate
  (`benchmarks/speed_review/bench_fe_absorption.py --check-estimates`).
- **Note:** When the optional Rust backend is available, the MAP sweeps run in
  the compiled `demean_map` kernel (rayon-parallel across the demeaned
  variables). The kernel mirrors the canonical numpy engine
  (`diff_diff.utils._demean_map_numpy`) exactly: same per-variable independent
  convergence loops, same dimension sweep order, the same row-order
  scatter-add accumulation as `np.bincount`, division by the per-group sums,
  the same zero-total-weight inert-row guard, and the same
  `max|x - x_old| < tol` stopping rule with NaN-poisoning semantics. Per the
  python-canonical policy, numpy is the reference implementation; equivalence
  tests assert iteration-count equality plus `assert_allclose` at atol=1e-12
  (never a bit-identity claim). `DIFF_DIFF_BACKEND=python` disables the
  kernel; any kernel-side validation error falls back to the numpy engine.
  The dispatch can partition variables into balanced column blocks
  (`DIFF_DIFF_DEMEAN_CHUNK_COLS`, internal opt-in knob, OFF by default) to
  bound the kernel's transient memory; per-column results and iteration
  counts are unchanged by construction (each column's MAP loop is fully
  independent - no cross-column arithmetic).
- **Edge case:** NaN in an absorbed group column raises a `ValueError` naming the
  column. `pd.factorize` codes NaN keys as -1, which would otherwise silently index
  the last group's mean; the prior pandas behavior was itself silently bad
  (unweighted: NaN-poisoned the affected rows; weighted: passed those rows through
  un-demeaned into the regression), so the explicit error replaces two distinct
  silent failure modes.
- **Edge case (FE-spanned regressors, v3.6.x):** a regressor lying exactly in the
  span of the absorbed/within-transformed FE dummies (e.g. the treated-group
  indicator after absorbing the unit dimension, a period dummy after absorbing
  time, or a unit-constant covariate) demeans to numerical junk (relative norm
  ~1e-13), NOT exact zero. Such a column previously reached the solver, where
  column equilibration re-inflated it to unit norm, it passed the rank check as
  linearly independent, and its arbitrary direction perturbed the identified
  coefficients at the ~1e-5 level (tolerance- and implementation-dependent).
  `diff_diff.utils.snap_absorbed_regressors` now zeroes spanned regressors at
  every demeaning consumer (DiD/MultiPeriodDiD `absorb=`, TwoWayFixedEffects,
  SunAbraham, BaconDecomposition, WooldridgeDiD) — including the replicate-refit
  closures (silently, matching their `rank_deficient_action="silent"` solves) —
  so the rank-deficiency machinery drops them deterministically (coefficient
  NaN). Detection is **two-stage**, because the MAP stopping rule bounds the
  last iteration step, not the distance to the limit, and a spanned column in a
  slow-convergence regime (unbalanced, correlated FE incidence — e.g.
  `x = a_unit + b_time` on a contiguous-lifetimes panel) can stop with a
  structured truncation residual far above any fixed norm threshold (measured
  1.9e-10 at tol=1e-10 and 2e-8 at tol=1e-8; left unsnapped it shifted ATT by
  ~3e-3 and reported a ~1e14-scale garbage coefficient): (1) relative demeaned
  norm <= 1e-10 snaps immediately; (2) candidates in (1e-10, 1e-3] get an exact
  span-membership confirmation via sparse LSMR on the (weighted) FE incidence
  and snap iff the true projection residual is <= 1e-10 relative — genuinely
  identified low-within-variation regressors are left untouched (their LSMR
  residual equals their real within-variation). Norms are `sqrt(w)`-weighted
  under WLS so zero-weight domain rows (left inert by the weighted demean)
  cannot mask spanning on the positive-weight sample. A cause-specific
  `UserWarning` naming the spanned regressors is emitted under
  `rank_deficient_action="warn"`; `"silent"` and `"error"` defer to the rank
  machinery per that parameter's existing contract. Identified coefficients are
  consequently stable in the demeaning tolerance (verified to ~1e-14 across
  tol=1e-8..1e-12, previously ~1e-5 swings).

### Survey Degrees of Freedom

- **Reference**: Korn & Graubard (1990) "Simultaneous Testing of Regression
  Coefficients with Complex Survey Data: Use of Bonferroni t Statistics",
  The American Statistician 44(4), 270-276.
- **Formula**: `df = n_PSU - n_strata` (replaces `n - k` for t-distribution
  inference)
- **Deviation from R:** Some software uses Satterthwaite-type df approximation;
  we use the simpler and more common `n_PSU - n_strata` convention.
- **Note:** When no explicit PSU is specified (weights-only or stratified-no-PSU
  designs), each observation is treated as its own PSU for df purposes. Survey df
  becomes `n_obs - n_strata` (or `n_obs - 1` when unstratified).
- **Note:** When survey_design specifies weights only (no PSU) and cluster=
  is specified, cluster IDs are injected as effective PSUs for Taylor Series
  Linearization variance estimation, matching the R `survey` package
  convention that clusters are the primary sampling units.

### Survey Aggregation (`aggregate_survey`)

Aggregation of individual-level survey microdata to geographic-period cells with
design-based precision estimates, for use as a pre-processing step before panel
DiD estimation on repeated cross-section survey data.

- **Reference**: Lumley (2004) "Analysis of Complex Survey Samples", Journal of
  Statistical Software 9(8), Section 3.4 (domain estimation).
- **Cell mean**: Design-weighted mean `ȳ_g = Σ w_i y_i / Σ w_i` for each cell g
  defined by grouping columns (e.g., state × year).
- **Cell variance**: Each cell is treated as a subpopulation/domain of the full
  survey design (consistent with `SurveyDesign.subpopulation()` and the
  Subpopulation Analysis section below). The influence function
  `ψ_i = w_i (y_i - ȳ_g) / Σ w_j` is zero-padded outside the cell, preserving
  full strata/PSU structure for variance estimation via `compute_survey_if_variance()`
  (TSL) or `compute_replicate_if_variance()` (replicate designs).
- **Second-stage weights** (`second_stage_weights` parameter):
  - `"pweight"` (default): Population weight = mean of per-cell `Σ w_i` within each
    geographic unit (first `by` column), constant across periods. Proportional to
    the Horvitz-Thompson estimated population count, averaged over periods to
    satisfy the unit-constant survey column contract required by panel estimators.
    Compatible with all survey-capable estimators including pweight-only estimators
    (CallawaySantAnna, ImputationDiD, TwoStageDiD, StackedDiD, etc.).
  - `"aweight"`: Precision weight = `1 / V(ȳ_g)` (inverse variance). Produces
    efficiency-weighted estimates via WLS. Compatible only with estimators that
    accept aweight (DifferenceInDifferences, TwoWayFixedEffects, MultiPeriodDiD,
    SunAbraham, ContinuousDiD, EfficientDiD).
  - **Reference**: Solon, Haider & Wooldridge (2015) "What Are We Weighting For?",
    Journal of Human Resources 50(2), 301-316. Population weights estimate the
    population parameter; precision weights are efficient under correct variance
    specification. Both are valid with heteroskedasticity-robust standard errors.
  - **Reference**: Donald & Lang (2007) "Inference with Difference-in-Differences
    and Other Panel Data", Review of Economics and Statistics 89(2), 221-233.
  - **Note:** The pweight default matches the R `did` package convention where
    `weightsname` accepts sampling/population weights, not inverse-variance weights.
- **Note:** SRS fallback when design-based variance is unidentifiable (e.g., all
  strata contribute zero variance) or when the cell has fewer than `min_n` valid
  observations. Formula: `V_SRS = Σ w_i(y_i - ȳ)² / (Σ w_j)² × n/(n-1)`.
  Cells using SRS fallback are flagged via `srs_fallback` column.
- **Edge case**: Zero-variance cells (all observations identical) set precision to
  NaN. Under aweight mode this maps to weight 0.0; under pweight mode the cell
  retains its positive population weight.

### Survey-Aware Bootstrap (Phase 6)

Two strategies for bootstrap variance under complex survey designs:

**Multiplier Bootstrap at PSU Level** (CallawaySantAnna, ImputationDiD, TwoStageDiD,
ContinuousDiD, EfficientDiD):

- **Reference**: Standard Taylor linearization bootstrap (Shao 2003, "Impact of the
  Bootstrap on Sample Surveys", Statistical Science 18(2))
- **Formula**: Generate multiplier weights independently within strata at the PSU level.
  Scale by `sqrt(1 - f_h)` for FPC. Perturbation:
  `ATT_boot[b] = ATT + w_b^T @ psi_psu` where `psi_psu` are PSU-aggregated IF sums.
- **Note:** When no strata/PSU/FPC, degenerates to standard unit-level multiplier bootstrap.

**Rao-Wu Rescaled Bootstrap** (SunAbraham, TROP):

- **Reference**: Rao & Wu (1988) "Resampling Inference with Complex Survey Data",
  JASA 83(401); Rao, Wu & Yue (1992) "Some Recent Work on Resampling Methods for
  Complex Surveys", Survey Methodology 18(2), Section 3.
- **Formula**: Within each stratum *h* with *n_h* PSUs, draw `m_h` PSUs with replacement.
  Without FPC: `m_h = n_h - 1`. With FPC: `m_h = max(1, round((1 - f_h) * (n_h - 1)))`.
  Rescaled weight: `w*_i = w_i * (n_h / m_h) * r_hi` where `r_hi` = count of PSU *i* drawn.
- **Note:** FPC enters through the resample size `m_h`, not as a post-hoc scaling factor.
  When `f_h >= 1` (census stratum), observations keep original weights (zero variance).
- **Note:** SyntheticDiD joins this list via a **hybrid** pairs-bootstrap +
  Rao-Wu rescaling (PR #352). Unlike SunAbraham / TROP, which use standalone
  Rao-Wu (resample PSUs within strata and rescale weights), SDID first
  performs unit-level pairs-bootstrap (`boot_idx = rng.choice(n_total)`) and
  then slices Rao-Wu rescaled weights over the resampled units, passing them
  into a **weighted Frank-Wolfe** re-estimation of ω̂ and λ̂ per draw. The
  full objective and argmin-set caveat live in §SyntheticDiD "Note (survey +
  bootstrap composition)"; the previous fixed-ω-and-rescaled-weights path
  was removed in PR #351 and replaced with this weighted-FW derivation.
- **Note:** Bootstrap paths support all three `lonely_psu` modes: `"remove"`, `"certainty"`,
  and `"adjust"`. For `"adjust"`, singleton PSUs from different strata are pooled into a
  combined pseudo-stratum and weights are generated for the pooled group. This is the
  bootstrap analogue of the TSL "adjust" behavior (centering around the global mean).
  Applies to both multiplier bootstrap (CallawaySantAnna, ImputationDiD, TwoStageDiD,
  ContinuousDiD, EfficientDiD) and Rao-Wu bootstrap (SunAbraham, TROP).
  FPC scaling is skipped for pooled singletons (conservative). When only one singleton
  stratum exists total, pooling is not possible — the singleton contributes zero bootstrap
  variance (same as `remove`), with a `UserWarning` emitted. This is a library-specific
  documented fallback (R's analytical `adjust` uses grand-mean centering, but the bootstrap
  analogue for a single singleton is not defined in the literature). Reference: Rust & Rao (1996).
- **Deviation from R:** For the no-FPC case (`m_h = n_h - 1`), this matches R
  `survey::as.svrepdesign(type="subbootstrap")`. The FPC-adjusted resample size
  `m_h = round((1-f_h)*(n_h-1))` follows Rao, Wu & Yue (1992) Section 3.

**CallawaySantAnna Design-Based Aggregated SEs**:

- **Formula**: `V_design = sum_h (1-f_h) * (n_h/(n_h-1)) * sum_j (psi_hj - psi_h_bar)^2`
  where `psi_hj = sum_{i in PSU j} psi_i` and `psi_i` is the combined IF (standard + WIF).
- **Note:** Per-(g,t) cell SEs use the simpler IF-based formula `sqrt(sum(psi^2))` which
  already incorporates survey weights. Only aggregated SEs (overall, event study, group)
  use the full design-based variance.

**TROP Cross-Classified Strata**:

- **Note (deviation from R):** When survey strata and treatment groups both exist, TROP
  creates pseudo-strata as `(survey_stratum x treatment_group)` for Rao-Wu resampling.
  This preserves both survey variance structure and treatment ratio. Survey df computed
  from pseudo-strata structure.
- **Note:** When `survey_design.strata` is None but PSU/FPC trigger full-design bootstrap,
  TROP uses treatment group (treated vs control) as pseudo-strata for Rao-Wu resampling
  to preserve treatment ratio. FPC is applied within these pseudo-strata. This matches
  TROP's existing treatment-stratified resampling pattern.
- **Note (deviation from block bootstrap):** In Rao-Wu survey bootstrap, per-observation
  treatment effects tau_{it} are deterministic given (Y, D, lambda) because survey weights
  do not enter the kernel-weighted matrix completion. The Rao-Wu path therefore precomputes
  tau values once and only varies the ATT aggregation weights across draws. This is
  mathematically equivalent to refitting per draw and avoids redundant computation.

### Replicate Weight Variance (Phase 6)

Alternative to TSL: re-run WLS for each replicate weight column and compute
variance from the distribution of replicate estimates.

- **Reference**: Wolter (2007) "Introduction to Variance Estimation", 2nd ed.
  Rao & Wu (1988).
- **Supported methods**: BRR, Fay's BRR, JK1, JKn, SDR
- **Formulas**:
  - BRR: `V = (1/R) * sum_r (theta_r - theta)^2`
  - SDR: `V = (4/R) * sum_r (theta_r - theta)^2` (Fay & Train 1995)
  - Fay: `V = 1/(R*(1-rho)^2) * sum_r (theta_r - theta)^2`
  - JK1: `V = (R-1)/R * sum_r (theta_r - theta)^2`
  - JKn: `V = sum_h ((n_h-1)/n_h) * sum_{r in h} (theta_r - theta)^2`
- **Note:** SDR (Successive Difference Replication) uses variance factor 4/R, following Fay & Train (1995). Used by ACS PUMS (80 replicate columns). Treated identically to BRR for scaling purposes — no fay_rho, no replicate_strata, custom scale/rscales ignored.
- **Note (vcov_type has no effect under replicate variance, 2026-07):** with `uses_replicate_variance`, the analytical sandwich is replaced wholesale by the replicate-refit variance — the per-replicate refits return point estimates only, which are identical across vcov families (FWL: full-dummy and within-transformed fits give the same coefficients). An explicit non-`hc1` analytical `vcov_type` (`hc2`, `hc2_bm`, `classical`) on `DifferenceInDifferences` / `MultiPeriodDiD` / `TwoWayFixedEffects` therefore emits a `UserWarning` and the (discarded) base fit remaps to `hc1` — avoiding wasted CR2-BM work, one-way-only validator rejections, and the TWFE full-dummy auto-route (which does not compose with per-replicate re-demeaning). Explicit `hc1` stays silent (it is the remap target). `conley` is excluded from the remap: it carries its own survey-design support contract (TSL stratified-Conley sandwich; dedicated per-design validators), which keeps firing unchanged. This replaces two prior inconsistent behaviors: `TwoWayFixedEffects(vcov_type="hc2"/"hc2_bm")` raised `NotImplementedError`, while `DifferenceInDifferences` silently ignored the kwarg — a per-replicate full-dummy HC2 implementation was considered and rejected as a costly no-op (it cannot change the replicate variance).
- **IF-based replicate variance**: For influence-function estimators (CS
  aggregation, ContinuousDiD, EfficientDiD, TripleDifference), replicate
  contrasts are formed via weight-ratio rescaling:
  `theta_r = sum((w_r/w_full) * psi)` when `combined_weights=True`,
  `theta_r = sum(w_r * psi)` when `combined_weights=False`.
- **Survey df**: QR-rank of the analysis-weight matrix minus 1,
  matching R's `survey::degf()` which uses `qr(..., tol=1e-5)$rank`.
  For `combined_weights=True` (default), analysis weights are the raw
  replicate columns. For `combined_weights=False`, analysis weights are
  `replicate_weights * full_sample_weights`. Returns `None` (undefined)
  when rank <= 1, yielding NaN inference. Replaces `n_PSU - n_strata`.
- **Mutual exclusion**: Replicate weights cannot be combined with
  strata/psu/fpc (the replicates encode design structure implicitly)
- **Design parameters** (matching R `svrepdesign()`):
  - `combined_weights` (default True): replicate columns include full-sample
    weight. If False, replicate columns are perturbation factors multiplied
    by full-sample weight before WLS.
  - `replicate_scale`: overall variance multiplier, applied multiplicatively
    with `replicate_rscales` when both are provided (`scale * rscales`)
  - `replicate_rscales`: per-replicate scaling factors (vector of length R).
    BRR and Fay ignore custom `replicate_scale`/`replicate_rscales` with a
    warning (fixed scaling by design); JK1/JKn allow overrides.
  - `mse` (default False, matching R's `survey::svrepdesign()`): if True,
    center variance on full-sample estimate; if False, center on mean of
    replicate estimates. When `replicate_rscales` contains zero entries
    and `mse=False`, centering excludes zero-scaled replicates, matching
    R's `survey::svrVar()` convention.
- **Note:** Replicate columns are NOT normalized — raw values are preserved
  to maintain correct weight ratios in the IF path.
- **Note:** JKn requires explicit `replicate_strata` (per-replicate stratum
  assignment). Auto-derivation from weight patterns is not supported.
- **Note:** Invalid replicate solves (singular/degenerate) are dropped with
  a warning. Variance is computed from valid replicates only. Fewer than 2
  valid replicates returns NaN variance. The variance scaling factor
  (e.g., `1/R` for BRR, `(R-1)/R` for JK1) uses the original design's `R`,
  not the valid count — matching R's `survey` package convention where the
  design structure is fixed and dropped replicates contribute zero to the
  sum without changing the scale. Survey df uses `n_valid - 1` for
  t-based inference.
- **Note:** Replicate-weight support matrix (13 of 20 public estimators):
  - **Supported**: CallawaySantAnna (reg/ipw/dr with or without covariates,
    no bootstrap; IF-based replicate variance is covariate-agnostic),
    ContinuousDiD (no bootstrap), EfficientDiD (no bootstrap),
    TripleDifference (all methods), StaggeredTripleDifference (IF-based),
    DifferenceInDifferences (no-absorb via LinearRegression dispatch,
    absorb via estimator-level refit), MultiPeriodDiD (no-absorb via
    `compute_replicate_vcov`, absorb via estimator-level refit),
    TwoWayFixedEffects (estimator-level refit with within-transformation),
    SunAbraham (estimator-level refit, replaces `vcov_cohort`),
    StackedDiD (estimator-level refit with Q-weight composition),
    ImputationDiD (two-stage refit), TwoStageDiD (two-stage refit),
    ChaisemartinDHaultfoeuille (closed-form cell-collapse replicate ATT,
    multi-horizon and placebo paths; replicate + `n_bootstrap > 0` rejected
    — see the ChaisemartinDHaultfoeuille Notes for the allocator contract)
  - **Rejected with NotImplementedError**: SyntheticDiD, TROP
    (bootstrap-based variance), WooldridgeDiD, LPDiD, SpilloverDiD,
    HeterogeneousAdoptionDiD (TSL-only survey paths; replicate designs
    rejected at `fit()`), SyntheticControl (rejects `survey_design`
    entirely)
  - **BaconDecomposition** is diagnostic-only — outside the 20-estimator
    count — and likewise rejects replicate designs
  - Estimators with replicate support reject replicate + bootstrap
    (replicate weights provide analytical variance)
- **Note:** When invalid replicates are dropped in `compute_replicate_vcov`
  (OLS path), `n_valid` is returned and used for `df_survey = n_valid - 1`
  in `LinearRegression.fit()`. For IF-based replicate paths, replicates
  essentially never fail (weighted sums cannot be singular), so `n_valid`
  equals `R` in practice and df propagation is not needed.

### DEFF Diagnostics (Phase 6)

Per-coefficient design effect comparing survey variance to SRS variance.

- **Reference**: Kish (1965) "Survey Sampling", Wiley. Chapter 8.
- **Formula**: `DEFF_k = Var_survey(beta_k) / Var_SRS(beta_k)` where
  SRS baseline uses HC1 sandwich ignoring design structure
- **Effective n**: `n_eff_k = n / DEFF_k`
- **Display**: Existing weight-based DEFF labeled "Kish DEFF (weights)";
  per-coefficient DEFF available via `compute_deff_diagnostics()` or
  `LinearRegression.compute_deff()` post-fit
- **Note:** Opt-in computation — not run automatically. Users call standalone
  function or post-fit method when diagnostics are needed.

### Subpopulation Analysis (Phase 6)

Domain estimation preserving full design structure.

- **Reference**: Lumley (2004) Section 3.4. Stata `svy: subpop`.
- **Method**: `SurveyDesign.subpopulation(data, mask)` zeros out weights for
  excluded observations while retaining strata/PSU layout for correct
  variance estimation
- **Note:** Unlike naive subsetting, subpopulation analysis preserves design
  information (PSU structure, strata counts) that would be lost by dropping
  observations. This is the methodologically correct approach for domain
  estimation under complex survey designs.
- **Note:** Weight validation relaxed from "strictly positive" to
  "non-negative" to support zero-weight observations. Negative weights
  still rejected. All-zero weight vectors rejected at solver level.
- **Note:** Survey design df (`n_PSU - n_strata`) uses the full design
  structure (including zero-weight rows), ensuring variance estimation
  accounts for all strata and PSUs. The generic HC1/classical inference
  paths use positive-weight count for df adjustments, ensuring zero-weight
  padding is inference-invariant outside the survey vcov path. DEFF
  effective-n also uses positive-weight count.
- **Note:** The TSL meat itself follows the same full-design convention:
  `_compute_stratified_psu_meat`'s per-stratum finite-sample correction
  `(1 - f_h)·n_{PSU,h}/(n_{PSU,h}-1)` and PSU-mean centering count
  zero-weight PSUs (a genuine-subpopulation PSU with all members outside the
  domain contributes a zero PSU-score `0` that is centered to `-\bar{z}_h`
  and still increments `n_{PSU,h}`). This is the Lumley (2004 §3.4) /
  R `survey::svyrecvar(subset())` domain estimator — so the survey SE is
  deliberately **not** invariant to genuine-subpopulation zeroing: it
  differs from the SE of a naive physical subset, which is the whole point
  of preserving the design (`subpopulation()` is correct *because* it does
  not equal the naive subset). Zero-weight rows that reuse an existing PSU
  label are inert (their weighted score is `0`, so the PSU-score sum is
  unchanged), so padding that preserves PSU membership is bit-invariant;
  only adding *new* synthetic all-zero PSUs would shift the SE, and no
  estimator path does so (domain padding goes through the zero-padded
  full-design cell variance in `prep.py`, which retains the real PSU layout).
  A former TODO proposed counting only positive-weight PSUs to force SE
  invariance; that was waived (DEFERRED.md § "Decision record — won't-fix / waived") because it would
  break this documented R parity.
- **Deviation from R:** `subpopulation()` preserves all strata in df
  computation even when a stratum has no positive-weight observations,
  while R's `subset()` drops empty strata from `survey::degf()`. For
  example, subsetting a 3-stratum design to one stratum gives df=n-3
  in diff-diff vs df=n-1 in R. Both ATT and SE match; only df (and
  therefore t-based CI width) differs. The diff-diff approach is
  conservative (more strata → lower df → wider CI) and preserves the
  full design structure per Lumley (2004) Section 3.4.
- **Note:** For replicate-weight designs, `subpopulation()` zeros out both
  full-sample and replicate weight columns for excluded observations,
  preserving all replicate metadata.
- **Note:** Estimator-level replicate refits (TWFE, SunAbraham, DiD/MultiPeriodDiD
  with `absorb`) drop zero-weight observations before weighted demeaning to
  prevent division-by-zero in within-transformation group means.  This matches
  R's `survey::withReplicates()` convention where zero-weight units are excluded
  from per-replicate estimation.  Replicates that fail despite this (e.g.,
  rank-deficient after unit deletion) are counted as invalid and excluded from
  variance computation.
- **Note:** Defensive enhancement: ContinuousDiD and TripleDifference
  validate the positive-weight effective sample size before WLS cell fits.
  After `subpopulation()` zeroes weights, raw row counts may exceed the
  regression rank requirement while the weighted effective sample does not.
  Underidentified cells are skipped (ContinuousDiD) or fall back to
  weighted means (TripleDifference).

---

# Practitioner Guide

The 8-step workflow in `diff_diff/guides/llms-practitioner.txt` is adapted from Baker et al. (2025)
"Difference-in-Differences Designs: A Practitioner's Guide" (arXiv:2503.13323), not a
1:1 mapping of the paper's forward-engineering framework.

- **Note:** The diff-diff canonical numbering is: 1-Define, 2-Assumptions, 3-Test PT,
  4-Choose estimator, 5-Estimate, 6-Sensitivity, 7-Heterogeneity, 8-Robustness.
  Paper's numbering: 1-Define, 2-Assumptions, 3-Estimation method, 4-Uncertainty,
  5-Estimate, 6-Sensitivity, 7-Heterogeneity, 8-Keep learning.
- **Note:** Parallel trends testing is a separate Step 3 (paper embeds it in Step 2),
  to ensure AI agents execute it as a distinct action.
- **Note:** Sources of uncertainty (paper's Step 4) is folded into Step 5 (Estimate)
  with an explicit cluster-count check directive (>= 50 clusters for asymptotic SEs,
  otherwise wild bootstrap). The 50-cluster threshold is a diff-diff convention.
- **Note:** Step 8 is "Robustness & Reporting" (compare estimators, report with/without
  covariates). Paper's Step 8 is "Keep learning." The mandatory with/without covariate
  comparison is a diff-diff convention.

## Survey DGP (`generate_survey_did_data`)

- **Note:** The `icc` parameter calibrates `psu_re_sd` using the full variance
  decomposition `Var(Y) = sigma²_psu * (1 + psu_period_factor²) + sigma²_unit +
  sigma²_noise + sigma²_cov`. When `add_covariates=True`, covariate variance
  `sigma²_cov = beta1² * Var(x1) + beta2² * Var(x2)` is included, where
  `(beta1, beta2)` defaults to `(0.5, 0.3)` but is configurable via
  `covariate_effects`.
- **Note:** When `informative_sampling=True` and `add_covariates=True`, covariate
  contributions are included in the Y(0) ranking used for weight assignment.
  Covariates are pre-drawn before the ranking step (panel: once before the loop;
  cross-section: each period) and reused in the outcome generation.
- **Note:** When `conditional_pt != 0`, the DGP creates X-dependent time trends
  that violate unconditional parallel trends while preserving conditional PT.
  Two mechanisms activate: (1) treated units' x1 is drawn from N(1, 1) instead
  of N(0, 1), creating differential covariate distributions; (2) the outcome
  includes `conditional_pt * x1_i * (t / n_periods)` for all units. Because
  E[x1 | treated] != E[x1 | control], the average time trend differs by group
  (unconditional PT fails). Conditional on x1, trends are identical (conditional
  PT holds). DR/IPW estimators with x1 as covariate recover the true ATT.
  Requires at least one ever-treated and one never-treated unit (rejected
  otherwise because the x1 mean shift only differentiates ever-treated from
  never-treated units).
- **Note:** When `conditional_pt != 0` is combined with `icc`, the ICC
  calibration is approximate. The x1 mean shift creates a mixture distribution
  with marginal Var(x1) = 1 + p_treated * (1 - p_treated) > 1, slightly
  inflating non-PSU variance and causing realized ICC to undershoot the target.

---

# Reporting

BusinessReport and DiagnosticReport are the practitioner-ready output
layer. Their methodology (phrasing rules, pre-trends verdict
thresholds, power-aware phrasing, unit-translation policy, schema
stability, no-traffic-light-gates decision, estimator-native diagnostic
routing) is recorded in a dedicated file to keep this registry
estimator-focused:

- See [`REPORTING.md`](./REPORTING.md).

---

## MMM Calibration Export (interop)

**Primary sources:** [PyMC-Marketing lift-test calibration](https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_lift_test.html) (`MMM.add_lift_test_measurements` schema); [Google Meridian, "Set custom prior distributions using past experiments"](https://developers.google.com/meridian/docs/advanced-modeling/set-custom-priors-past-experiments) and the closed-form lognormal conversion in `meridian/model/prior_distribution.py` (`lognormal_dist_from_mean_std`, Meridian 1.7.0); the roi_m/mroi_m estimand definitions in [Meridian's ROI parameterization docs](https://developers.google.com/meridian/docs/advanced-modeling/roi-mroi-contribution-parameterizations); the `roi_calibration_period` mask shape/semantics contract in [Meridian's "Set the ROI calibration period"](https://developers.google.com/meridian/docs/user-guide/configure-model) (configure-model guide).

**Module:** `diff_diff/mmm.py`

**Scope:** Interop builders (not estimators), **explicit numbers in, or the pinned aggregation contract in - validated out**. Reconciling an experiment's estimate to a calibration input requires context diff-diff cannot always see - the target MMM's row granularity (per-geo vs national), its time window, and the outcome's scale (additive levels vs log/rate/share). The default route stays fully explicit: the caller supplies the already-scoped incremental outcome and its SE (the numbers read off `summary()`, aggregated to the population and window their MMM row represents). The 3.9 post-fit `results.aggregate()` layer - the route this section previously deferred to - now exists, and both exporters consume its pinned `AggregationResult` container via `aggregation_result=` + `scale=` (the exporters never call `aggregate()` themselves and reject raw results objects and `EventStudyResults`; the module remains purely additive and touches no estimator). diff-diff assembles the target schema, enforces each consumer's guards, converts to the lognormal parameterization, pools, and emits snippets. Pure numpy/pandas; no MMM package imported.

**Key implementation requirements:**

*Container mode (`aggregation_result=` + `scale=`, both exporters):*
- **Note:** Scale contract and row order. Derivation is `effect_i = att_i * scale_i`, `se_out_i = se_i * scale_i` per container row, with rows consumed in `to_dataframe()` order (sorted by label when sortable, producer order otherwise - the order `summary()` prints); every per-row sequence kwarg (`scale`, `spend`, `x`, `delta_x`, `dims`) aligns to that order, and the helper reads all per-row values from one `to_dataframe()` call. `scale` is always required with a container: a number (finite, positive; scalar or per-row) or the literal `"auto"`. Accepted levels are `simple` (one row; a multi-row simple container is out-of-contract and rejected) and `group` (one experiment per cohort); every row's `target` must be `"att"` (ContinuousDiD's `acrt`, HAD's WAS estimands, and dCDH's estimand relays reject the container WHOLE - no silent row filtering), and every row needs finite `att` and finite positive `se`.
- **Note:** `scale="auto"` allowlist and routing. Auto-derivation (`scale_i = n_i`) is honored ONLY for container `estimator` provenance in `{ImputationDiD, TwoStageDiD}`, the audited producers whose `n` is the treated observations (unit-periods) the ATT averages over at both supported levels. Routing keys on the `estimator` field, never on `n_kind` alone - CallawaySantAnna repeated-cross-section fits report `n_kind="obs"` with `n` = treated+control observations - and `n_kind == "obs"` serves only as a schema-drift sanity guard on the allowlist path. CallawaySantAnna (`simple`: treated+control units/obs; `group`: contributing (g, t) cells) and EfficientDiD (units/cells) are refused with hints naming their `n` semantics; StackedDiD is refused because its `n` is a deduplicated distinct-treated-UNIT count (no per-observation mass) and its `weighting="population"`/`"sample_share"` estimands make raw treated exposure the wrong multiplier.
- **Note:** `scale="auto"` is an explicit caller acknowledgement of three assumptions the container cannot verify: the outcome is in additive levels (no outcome-scale provenance exists on the container); the fit is unweighted (survey-weighted fits pair a weighted `att` with a raw-count `n` and carry no survey marker - pass a numeric `scale` there); and every treated observation's effect is identified. On the identification point, ImputationDiD averages finite tau-hat only while `n` stays the raw treated-obs count at BOTH levels, and TwoStageDiD's `simple` ATT support excludes rows whose first-stage residual is non-finite while `n` reports the pre-filter count (its `group` `n` IS post-filter) - both estimators warn at fit time on the degenerate branch, and `"auto"` on such a fit overcounts (behaviorally pinned in `tests/test_mmm.py::TestAggregationWorkflow`).
- **Note:** `group` containers carry correlated cohorts, both exporters: all cohort rows come from ONE fit sharing controls/windows, and the container stores only per-row SEs (joint-covariance off-diagonals are discarded upstream). Meridian pooling treats rows as independent (see the pooling Note below) and PyMC-Marketing scores each lift row independently with its marginal `sigma`; the omitted cross-cohort covariance can misstate the joint/pooled uncertainty - anti-conservatively when the net weighted covariance is positive (shared controls typically induce positive correlation) - and its direction cannot be determined from marginal SEs alone. `se_widening > 1` is a conservative heuristic, not an exact correction (docstring-noted on both exporters).

*Lift-test frame (`to_pymc_marketing_lift_test`):*
- Assembles one row per experiment with the exact schema `[channel, *dims, x, delta_x, delta_y, sigma]` from caller-supplied values. `delta_y`/`sigma` are the scoped incremental outcome and its SE; all four numeric columns must describe the SAME calibration observation (same channel, population, period span, additive-level outcome) - stated in the docstring, not machine-checkable, since spend and scope are user input.
- Sign/zero policy (`on_wrong_sign={raise,drop,keep}`, default raise): PyMC-Marketing rejects `sign(delta_y) != sign(delta_x)` with `NonMonotonicError`; `delta_y == 0` is a distinct class (degenerate for its strictly-positive Gamma lift likelihood, which its own monotonicity check does not catch). Both share one disposition; `drop` raises rather than emit an empty frame; `keep` warns the frame is not valid PyMC-Marketing input.
- Guards: `sigma > 0` and finite; `delta_x` finite and nonzero; `x >= 0` finite; `x + delta_x >= 0` (post-test spend cannot be negative - the saturation curve is evaluated there); `delta_y` finite; go-dark tests (`delta_x < 0` with sign-consistent negative lift) pass. `dims` keys must not collide with reserved schema columns and must share one key set across rows; non-mapping dims raise `TypeError`.

*Meridian ROI prior (`to_meridian_roi_prior`):*
- Per experiment `roi = incremental_outcome / spend`, `roi_sd = incremental_outcome_se / spend * se_widening`. Multiple experiments pooled spend-weighted - the spend-weighted average ROI the [Meridian FAQ](https://developers.google.com/meridian/docs/faqs) suggests (citing sec 3.4 of Google's MMM calibration whitepaper): `roi_mean = sum(w_i * roi_i)`, `w_i = spend_i / sum(spend)`.
- **Estimand (`parameter={roi_m,mroi_m}`, default roi_m):** `roi_m` is the return on the channel's full spend (a zero-spend/full-holdout estimand; `spend` = total channel spend); `mroi_m` is the marginal return of a spend change (`spend` = the spend change). Under saturation these differ, so the caller declares which their `incremental_outcome` measures; the returned prior and `.to_code()` target that parameter with its own Meridian default for non-experiment channels (roi_m `LogNormal(0.2, 0.9)`, mroi_m `LogNormal(0.0, 0.5)`, both verified at 1.7.0). Sign, aggregation, and population are the caller's responsibility - a go-dark contract's numerator is `Y_exposed - Y_zero`, which the caller supplies with the correct sign.
- Lognormal conversion matches Meridian's `lognormal_dist_from_mean_std` exactly: `sigma = sqrt(log1p((s/m)^2))`, `mu = ln(m) - log1p((s/m)^2)/2` (`log1p` keeps a ~1e-8 relative SE from rounding `sigma` to 0).
- A non-positive pooled ROI mean raises (lognormal positivity), pointing to pooling, wider priors, or Meridian's contribution/coefficient parameterizations. `spend`, `incremental_outcome_se`, and `se_widening` must be finite and positive; `incremental_outcome` finite.
- **Note:** Pooled `roi_sd = sqrt(sum((w_i * sd_i)^2))` treats experiments as independent (no covariance term). Ignoring covariance can misstate the pooled sd - it is anti-conservative when the net weighted covariance is positive, which experiments sharing control units or overlapping windows typically induce - and the direction cannot be determined from marginal SEs alone; the docstring instructs users to widen via `se_widening` (a conservative heuristic, not an exact correction).
- **Channel scope in `.to_code()`**: roi_m/mroi_m have batch shape `n_media_channels` and a scalar distribution broadcasts to EVERY channel, so the snippet helper requires explicit scope - `channel=`+`media_channels=` emits a vector prior in model channel order (non-experiment channels keep the parameter's Meridian default), `single_channel=True` emits a scalar snippet marked single-channel-only, and neither raises. It also requires the prior's TIME scope (`roi_calibration_period=<boolean ndarray, typically from meridian_calibration_mask(...)>`, `roi_calibration_period=<mask expr>`, or `full_model_window=True`), because Meridian's default applies an experiment-window prior over all model times; and it sets `media_prior_type="roi"`/`"mroi"` on `ModelSpec` (Meridian ignores a supplied `roi_m`/`mroi_m` unless the matching prior type is selected). Snippets use the TensorFlow substrate; a comment points JAX users at `tensorflow_probability.substrates.jax`.
- **Note (mask builder + ndarray serialization):** `meridian_calibration_mask(media_times=, media_channels=, channel=, window=)` builds the boolean `(n_media_times, n_media_channels)` mask: the experiment channel's column(s) are True exactly on the window, and every OTHER channel's column is ALL-TRUE - Meridian's documented convention ("any media channels not specified ... will utilize all available periods for ROI calibration", configure-model guide; an all-False column zeroes that channel's aggregated calibration spend via `input_data._aggregate_spend`'s einsum). Window convention: a 2-TUPLE is `(start, end)` INCLUSIVE bounds (order-compared by value, `pd.to_datetime`-coerced when `media_times` is datetime-like, timezone mismatches fail closed both ways); any OTHER sequence is explicit labels, each required present (fail-closed membership, no silent drop). `to_code()` accepts the array and serializes it into the snippet as an `np.ones` prelude plus, per group of columns sharing a row pattern, a column-clear + `np.ix_` window assignment (indices `.tolist()`-coerced). Meridian's guide adds that `roi_calibration_period` "is not generally recommended because calibrating the ROI of a specific time period does not necessarily improve estimation of the overall ROI" - the docstrings point users at `full_model_window=True` when the experiment evidence reasonably transfers, reserving the mask for genuinely period-specific evidence.
- **Note (ndarray acceptance rules):** boolean dtype, or numeric containing ONLY 0/1 (cast to bool - Google's own configure-model example builds the mask with float `np.zeros`); anything else rejected, masked arrays rejected (`np.asarray` would silently drop the mask). An ALL-FALSE mask is rejected, and so is ANY entirely-False COLUMN (it would zero that channel's aggregated calibration spend; channels without an experiment must use all periods). Only `roi_m` priors accept a mask: Meridian 1.7.0's `ModelSpec._validate_roi_calibration_period` rejects a non-None `roi_calibration_period` unless the media prior type is `'roi'`, so `to_code()` fails closed for `mroi_m` on BOTH the ndarray and expression-string routes (`full_model_window=True` remains the mroi route) - this also corrects the pre-3.10 expression route, which emitted mroi snippets Meridian rejects at `ModelSpec` construction.
- **Note (what the mask route cannot verify):** the mask's ROW count (to_code receives no time coordinate; the builder guarantees consistency when its `media_times` matches the model's coordinates - hand-built arrays are the caller's responsibility) and the column ORDER/identity (the mask carries no channel labels; the `media_channels` given to the builder and to `to_code` must be the same list in the same order - only the count is machine-checked). Coordinates are taken VERBATIM: Meridian's `n_media_times` can exceed `len(data.time)` under `max_lag > 0`, yet Google's own example builds `len(data.time)` rows; the builder does not resolve that nuance - the caller passes whichever coordinate list their `ModelSpec` expects.

*Outputs:* All scaled and pooled values (`roi`, `roi_sd`, pooled moments, lognormal `mu`/`sigma`) are validated finite-and-positive after arithmetic; overflow/underflow raises rather than emitting non-finite or zero-uncertainty calibration data.

---

# Replication Data Conventions

## Castle Doctrine treatment coding (`load_castle_doctrine`)

**Primary source:** Cheng, C., & Hoekstra, M. (2013). Does Strengthening Self-Defense Law Deter Crime or Escalate Violence? Evidence from Expansions to Castle Doctrine. *Journal of Human Resources*, 48(3), 821-854.

**Module:** `diff_diff/datasets.py`

**Scope:** Treatment-variable coding in `load_castle_doctrine()` only - no estimator, variance, or inference behavior. Loader provenance, checksum pinning, and the synthetic-fallback contract are documented in the loader docstrings and `docs/api/datasets.rst`.

- **Note:** The paper's estimating equation is `Outcome_it = b0*CDL_it + b1*X_it + c_i + u_t + e_it`, "where `CDL_it` is the treatment variable that equals the proportion of year `t` in which state `i` has an effective Castle Doctrine law" (p. 830, Sec. III). For partial-year adopters it states: "we set CDL equal to the proportion of the year in which the law was in effect, though estimates are almost identical when we exclude the year of adoption" (fn. 21 reports the excluded-year estimates). That regressor is preserved verbatim as `treatment_exposure` (the source file's `cdl`). Verified in the file: 0 before adoption and for never-adopters, exactly 1.0 after, and a day-count fraction in the adoption year - `cdl * 365` recovers an integer day count for 17 of the 21 adopting states (59 to 306 days) and 182.5 (half a year) for the other four. `cdl` is stored as float32, so those products are integers only to within storage rounding (max deviation 1.03e-5), not bit-exact.
- **Note:** The binary `treated` column follows the standard staggered-DiD convention (`year >= first_treat`), so that `treated`, `first_treat`, and `cohort` stay mutually consistent - which `_validate_panel_keys` enforces and which the cohort estimators (`CallawaySantAnna`, `SunAbraham`) require. This convention matches **neither** of the paper's treatments of the adoption year: not the fractional `CDL_it` of the main specification, nor the drop-the-adoption-year robustness check. It also differs from the source file's own binary `post` indicator (0 in the adoption year, 1 thereafter) on exactly 21 of 550 rows. **Replicating Cheng-Hoekstra requires regressing on `treatment_exposure`, not `treated`.** Synthetic fallback frames carry `treatment_exposure` as a binary 0/1 copy of `treated` and therefore cannot support the paper's specification at all.
- **Note:** The paper's dependent variable is the log of the outcome per 100,000 population (p. 830). `homicide_rate` is the level (the source file's `homicide`, labelled "homicide count per 100,000 state population"), so a replication must take logs. The file's `l_homicide`, which is exactly `log(homicide)`, is not carried into the public schema.

---

# Version History

- **v1.3** (2026-03-26): Added Replicate Weight Variance, DEFF Diagnostics,
  and Subpopulation Analysis sections (Phase 6 completion)
- **v1.2** (2026-03-24): Added Survey-Aware Bootstrap section (Phase 6)
- **v1.1** (2026-03-20): Added Survey Data Support section
- **v1.0** (2025-01-19): Initial registry with 12 estimators
