Spillover-Aware DiD (Butts 2021)
=================================

Ring-indicator spillover-aware Difference-in-Differences estimator.

This module implements the methodology from Butts, K. (2023, originally 2021),
"Difference-in-Differences with Spatial Spillovers" (arXiv:2105.03737v3).
The estimator augments two-stage Gardner (2022) DiD with ring-indicator
covariates that identify, alongside the direct effect on treated units
(``tau_total``), per-ring spillover effects on near-control units
(``delta_j``).

The standard DiD estimator is biased for ``tau_total`` by
``-tau_spill(0)`` (paper Proposition 2.1 / Equations 1-2); the
ring-augmented regression removes this bias. The "far-away" cutoff
``d_bar`` defines the boundary beyond which spillovers vanish
(Assumption 5).

**When to use SpilloverDiD:**

- Spatial-policy DiD settings where treatment may spill over onto nearby
  control units (border-RD, place-based policy, neighborhood effects,
  geographic policy boundaries)
- When the canonical DiD coefficient is suspected to be attenuated or
  inflated by spillover bias and a far-away control group exists
- As a robustness check on TwoStageDiD / ImputationDiD when the
  treatment has plausibly local geographic externalities

**Reference:** Butts, K. (2023). Difference-in-Differences with Spatial
Spillovers. *arXiv:2105.03737v3*. Gardner, J. (2022). Two-stage
differences in differences. *arXiv:2207.05943*.

.. module:: diff_diff.spillover

SpilloverDiD
------------

Main estimator class.

.. autoclass:: diff_diff.SpilloverDiD
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   .. rubric:: Methods

   .. autosummary::

      ~SpilloverDiD.fit
      ~SpilloverDiD.get_params
      ~SpilloverDiD.set_params

SpilloverDiDResults
-------------------

Results container with per-ring spillover-effect table.

.. autoclass:: diff_diff.SpilloverDiDResults
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::

      ~SpilloverDiDResults.summary
      ~SpilloverDiDResults.to_dict

Example Usage
-------------

Non-staggered 2-period panel with synthetic spillovers::

    from diff_diff import SpilloverDiD

    est = SpilloverDiD(
        rings=[0, 50, 100, 200],          # 3 rings: [0,50), [50,100), [100,200]
        conley_coords=("lat", "lon"),
        vcov_type="conley",
        conley_cutoff_km=200.0,
        conley_lag_cutoff=0,
    )
    result = est.fit(
        data=df,
        outcome="y",
        unit="unit",
        time="time",
        treatment="D",                    # binary; auto-converts to first_treat
    )
    print(result.summary())
    # Total effect on treated (Butts tau_total):
    print(f"tau_total = {result.att:.4f}")
    # Per-ring spillover-on-control:
    print(result.spillover_effects)

Staggered timing (Gardner-convention first_treat column)::

    result = est.fit(
        data=df,
        outcome="y",
        unit="unit",
        time="time",
        first_treat="first_treat",        # 0 / inf = never-treated
    )

Identification spec
-------------------

The stage-2 regressor for ring j is the time-varying form

.. math::

    (1 - D_{it}) \cdot \mathrm{Ring}_{it,j}

where ``Ring_{it,j}`` is the indicator that unit *i*'s nearest currently-
treated unit is in distance bin *j* at time *t*, and ``D_it`` is the
treatment indicator (1 if unit *i* is treated by time *t*). For
non-staggered timing, all treated units share one onset and ``Ring`` is
unit-static post-treatment / zero pre-treatment. For staggered timing,
``Ring_{it,j}`` is unit-time-varying (a unit's nearest treated neighbor
changes as more cohorts enter treatment).

Stage 1 fits unit + time fixed effects on Butts' subsample
``Omega_0 = {D_it = 0 AND S_it = 0}`` (untreated AND unexposed) — the
clean far-away control group. Reading the literal unit-static
``(1 - D_it) * S_i`` from paper Equation 5 yields a rank-deficient
design under TWFE; the diff-diff implementation uses the time-varying
form (paper page 12's ``S_it`` notation, made explicit in Section 5
Table 2).

Estimator Comparison
--------------------

.. list-table:: SpilloverDiD vs. TwoStageDiD vs. TwoWayFixedEffects
   :header-rows: 1
   :widths: 25 25 25 25

   * - Feature
     - SpilloverDiD
     - TwoStageDiD
     - TwoWayFixedEffects
   * - Spillover handling
     - Identifies per-ring delta_j
     - None (assumes SUTVA)
     - None (assumes SUTVA)
   * - Methodology
     - Two-stage Gardner + ring covariates
     - Two-stage Gardner
     - Single-stage TWFE
   * - Staggered timing
     - Yes
     - Yes
     - Biased under heterogeneity
   * - Stage-1 subsample
     - Butts: ``D=0 AND S=0`` (untreated AND unexposed)
     - ``D=0`` (untreated)
     - N/A (single stage)
   * - Conley spatial-HAC SE
     - Yes (Wave D GMM-corrected sandwich)
     - Not yet supported
     - Yes
   * - Cluster-robust SE
     - Yes (HC1 + CR1, Wave D GMM-corrected sandwich)
     - Yes (GMM sandwich + clusters)
     - Yes

Restrictions and follow-ups
---------------------------

The current implementation has the following documented restrictions
and planned follow-up enhancements:

- **Gardner GMM first-stage correction at stage 2** — SHIPPED in Wave D.
  Stage-2 variance now applies the influence-function-based correction
  for stage-1 FE estimation uncertainty across all three ``vcov_type``
  paths (HC1, Conley, cluster) on both ``event_study=False`` AND
  ``event_study=True``. The IF formula is
  ``psi_i = gamma_hat' * X_{10,i} * eps_{10,i} - X_{2,i} * eps_{2,i}``
  with ``gamma_hat = (X_10' X_10)^{-1} (X_1' X_2)``; the meat is
  ``Psi' K Psi`` where ``K`` is the path-dependent kernel matrix
  (identity for HC1, block-indicator for cluster, spatial kernel for
  Conley). Documented synthesis of Butts (2021) Section 3.1 + Gardner
  (2022) Section 4 + Conley (1999); no reference software combines all
  three ingredients. Point estimates unchanged; SE values shift upward
  by 1-few percent depending on first-stage residual variance.
- **Event-study mode** — ``event_study=True`` is SHIPPED in Wave C.
  The per-event-time × ring decomposition (Butts Section 5 / Table 2)
  emits per-event-time direct effects ``tau_k`` and per-(ring,
  event-time) spillover effects ``delta_jk`` as a ``att_dynamic``
  DataFrame plus MultiIndex ``spillover_effects``. The
  ``event_study_effects: Dict[int, Dict]`` alias mirrors
  ``TwoStageDiD``'s schema for ``plot_event_study`` consumption (the
  plotter prefers the new ``reference_period`` attribute over the
  legacy ``n_obs==0`` heuristic). ``DiagnosticReport`` routing for
  ``SpilloverDiDResults`` is wired: parallel-trends (event-study on
  the direct-effect dynamics), design-effect, and heterogeneity apply;
  Goodman-Bacon is excluded (spillover identifies off far-away
  controls, not TWFE 2x2 comparisons). Reference
  period ``-1 - anticipation`` (TwoStageDiD parity). ``horizon_max``
  bins event-times into endpoint pools (no row drop — divergence
  from TwoStageDiD's filtering semantic, intentional per
  ``feedback_no_silent_failures``). ``horizon_max`` must be ``>=1`` or
  ``None`` under ``event_study=True``; ``horizon_max=0`` is rejected
  (the single bin ``k=0`` leaves no event-time pair to anchor the
  reference period — for a single aggregate effect, use
  ``event_study=False`` instead). Scalar ``att`` becomes a
  share-weighted average of post-treatment ``tau_k`` with SE
  from linear-combination inference on the post-treatment vcov block.
  Per-event-time SEs apply the Wave D Gardner GMM first-stage
  uncertainty correction (see the "Gardner GMM first-stage correction"
  entry above). Wave E.1 amendment: when ``survey_design=`` is supplied,
  the per-horizon shares are **survey-weight totals** rather than raw
  observation counts (``share_k = sum_i w_i * 1{K_direct_i = k AND
  treated_i = 1}``); the same vector enters both the ``att`` point
  estimate and ``Var(att) = w' V_subset w``. Without this, the
  lincom would mix unweighted aggregation shares with weighted WLS
  horizon coefficients and target the wrong estimand. See the registry
  "Variance (Wave E.1)" subsection for the full survey-aggregation
  contract.
- **Survey-design integration (Wave E.1 — HC1 / CR1 via Binder TSL).**
  SHIPPED in Wave E.1. ``survey_design=`` is now supported on
  ``vcov_type ∈ {"hc1"}`` and CR1 (``cluster=<col>``) paths.

  .. note::

     Wave E.1 composes Gerber (2026, arXiv:2605.04124) Proposition 1 —
     Binder Taylor Series Linearization for IF representations of smooth
     functionals; explicitly derived for TwoStageDiD in the paper's
     Appendix — with the Wave D Gardner GMM first-stage uncertainty
     correction (Butts 2021 §3.1 + Gardner 2022 §4) applied to
     SpilloverDiD's ring-indicator stage-2 design. The composition is
     mechanical: SpilloverDiD's per-obs IF
     ``psi_i = gamma_hat' * X_{10,i} * eps_{10,i} - X_{2,i} * eps_{2,i}``
     is aggregated to PSU level, then passed to the audited Binder TSL
     meat helper. Survey weights enter via Hájek normalization at the
     gamma_hat solve, eps construction, and bread inversion. Degrees of
     freedom for the t-distribution lookup use
     ``ResolvedSurveyDesign.df_survey`` (4-way branch: PSU+strata →
     ``n_PSU - n_strata``; PSU only → ``n_PSU - 1``; strata only →
     ``n_obs - n_strata``; neither → ``n_obs - 1``). No reference
     software combines all ingredients.

  Restrictions:

  - Replicate-weight variance (BRR / Fay / JK1 / JKn / SDR) raises
    ``NotImplementedError``; per Gerber (2026) Appendix A, the
    IF-reweighting shortcut does not apply because ``gamma_hat`` is
    weight-sensitive — follow-up requires per-replicate full re-fit.
  - Singleton-stratum ``lonely_psu="remove"`` with all strata singletons
    saturates ``df_survey = 0`` and the meat helper NaN-fails; SE /
    t-stat / p-value / CI all NaN-propagate (no silent fallback to HC1).
  - ``cluster=<col> + survey_design.psu`` with **different groupings**
    emits a ``UserWarning`` and uses PSU as the cluster (mirrors
    ``TwoStageDiD._resolve_effective_cluster``). When the two groupings
    are identical, no warning fires; PSU still takes precedence
    (inference is unchanged either way).
  - ``SurveyDesign.subpopulation()`` and warn-and-drop full-design
    retention SHIPPED in Wave E.3 — see "Survey-design integration
    (Wave E.3)" entry below.
- **Survey-design integration (Wave E.2 — Conley × survey via
  stratified-Conley sandwich on PSU totals).** SHIPPED in Wave E.2.
  ``vcov_type="conley" + survey_design=`` is now supported via a
  per-stratum Conley sandwich applied to PSU-aggregated Wave D Gardner
  GMM influence functions.

  .. note::

     Wave E.2 composes Conley (1999) spatial-HAC with Gerber (2026,
     arXiv:2605.04124) Proposition 1 Binder TSL (the Wave E.1 foundation)
     and the Wave D Gardner GMM first-stage uncertainty correction
     (Butts 2021 §3.1 + Gardner 2022 §4) applied to SpilloverDiD's
     ring-indicator stage-2 design. The composition is **panel-aware** —
     it preserves the library's existing ``conley_lag_cutoff = 0``
     semantic ("within-period spatial only, exclude cross-period pairs")
     by looping over periods and aggregating Psi to PSU totals WITHIN
     each period (not over the whole panel). For each period ``t``,
     ``S_psu_t[g] = sum_{i in PSU g, time t} psi_i``; per-PSU centroids
     are panel-constant (mean of per-observation ``conley_coords``);
     for each stratum the within-stratum sandwich is
     ``M_h_t = (1 - f_h) * n_h/(n_h-1) * sum_{j,k in PSUs_h}
     K(d(centroid_j, centroid_k) / cutoff) *
     (S_psu_t[j] - S_bar_h_t)(S_psu_t[k] - S_bar_h_t)'``, where K is the
     Bartlett kernel (SpilloverDiD currently exposes Bartlett only and
     hardcodes it at the fit-call site; the survey helper's ``kernel``
     parameter can also take ``"uniform"``, but exposing that on the
     SpilloverDiD constructor is a separate follow-up). Cross-stratum
     kernel weights are exactly zero by sampling design (strata are
     exact independence partitions). Total meat is ``sum_t sum_h M_h_t``.
     Cross-period spatial pairs are excluded by construction. No
     reference software combines all three ingredients on a two-stage
     influence function.

  Reduction semantics:

  - Per-period sum invariant: ``sum_t`` of per-period within-stratum
    stratified-Conley sandwiches on per-period PSU totals. Pinned at
    ``tests/test_spillover.py::TestSpilloverDiDWaveE2ConleySurveyDesign::test_b_panel_aware_per_period_sum_invariant``
    (pure unit test on the orchestrator + helper composition).
  - Single stratum (H = 1, FPC = inf): reduces to ``sum_t`` plain
    Conley sandwich on per-period PSU totals (NOT on time-collapsed
    PSU totals — the per-period loop preserves ``lag_cutoff = 0``
    semantics).
  - All PSUs singleton + ``lonely_psu="remove"``: ``df_survey = 0`` and
    the stratified-Conley meat NaN-fails (matches Wave E.1 saturation
    behaviour, with ``UserWarning`` template "Wave E.2 stratified-Conley
    sandwich: df_survey = 0...").

  Restrictions:

  - Replicate-weight variance (BRR / Fay / JK1 / JKn / SDR) raises
    ``NotImplementedError`` (inherits Wave E.1 gate; per-replicate refit
    is separate follow-up scope).
  - ``cluster=<col> + survey_design.psu + vcov_type="conley"``:
    ``cluster=<col>`` is coerced to PSU per Wave E.1's warn-and-use-PSU
    pattern; the Conley cluster product kernel becomes a no-op after
    PSU aggregation.
  - The LinearRegression-side ``vcov_type="conley" + survey_design=``
    gate at ``diff_diff/linalg.py`` remains a separate roadmap (not Wave E)
    — weighted spatial-HAC under probability sampling is an open
    methodological question; no canonical extension of Conley (1999) exists
    for the combination.
  - DiagnosticReport routing for ``SpilloverDiDResults(vcov_type="conley",
    survey_design=)`` is queued for a follow-up (the
    ``_APPLICABILITY`` / ``_PT_METHOD`` wiring must register the new
    combination first).
- **Survey-design integration (Wave E.2 follow-up —
  ``conley_lag_cutoff > 0`` panel-block composition).** SHIPPED in
  Wave E.2 follow-up. ``vcov_type="conley" + conley_lag_cutoff > 0 +
  survey_design=`` is now supported by adding a within-PSU serial
  Bartlett HAC term to the Wave E.2 spatial sandwich, **provided the
  survey design has an effective PSU** (either explicit
  ``survey_design.psu`` or a ``cluster=<col>`` argument that gets
  injected as the effective PSU per Wave E.1's
  ``_inject_cluster_as_psu`` routing). No-effective-PSU survey designs
  (weights-only / strata-only WITHOUT a cluster fallback) raise
  ``NotImplementedError`` at ``SpilloverDiD.fit`` post-resolution —
  see the Restrictions list below.

  .. note::

     Wave E.2 follow-up composes Wave E.2's panel-aware stratified-Conley
     spatial sandwich with within-PSU serial Bartlett HAC over time
     (Newey-West 1987 form, kernel weights ``1 - |t-s|/(L+1)`` for
     ``|t-s| <= L, t != s``). The composition is
     ``meat = meat_spatial + meat_serial`` with disjoint index sets,
     exactly matching the no-survey panel-block decomposition at
     ``diff_diff.conley._compute_conley_meat`` (Conley 1999 + Newey-West
     1987 separable form, NOT Driscoll-Kraay 2D-HAC). For each stratum
     ``h``: ``meat_serial_h = FPC_h_panel * sum_{g in stratum h}
     sum_{|t-s| <= L, t != s, both periods present for PSU g}
     K_serial(|t-s|/(L+1)) * S_centered_t[g] @ S_centered_s[g]'`` where
     ``S_centered_t[g] = S_psu_t[g] - S_bar_h(g)_t`` is per-period
     within-stratum centered (Binder TSL form — matches the spatial
     helper's centering exactly), and ``|t-s|`` uses panel-wide dense
     time codes (matches ``conley.py:940`` documented R deviation that
     mirrors R ``conleyreg``). Serial Bartlett kernel is hardcoded
     regardless of ``conley_kernel`` (mirrors ``conley.py:951-965``;
     ``conley_kernel`` governs spatial kernel only). FPC for serial uses
     panel-wide ``n_h_panel = |unique PSUs in stratum h across active
     sample|``, NOT per-period ``n_h_t`` (the serial sum is a PANEL-level
     construct; standalone Newey-West composition on stratified clusters,
     deliberately NOT by analogy to the cross-sectional Wave E.2 spatial
     FPC convention). No reference software combines panel-block Conley +
     Binder TSL + Gardner GMM correction on a two-stage influence
     function.

  Reduction semantics:

  - ``conley_lag_cutoff = 0`` or ``None``: **bit-identical** to shipped
    Wave E.2 ATT and scalar SE (orchestrator skips the serial helper
    invocation when ``L = 0`` so meat_serial does not contribute; the
    test_a2 mock-spy independently asserts the helper isn't invoked).
  - ``conley_time is None`` or ``T = 1``: serial helper short-circuits to
    zero meat (no cross-period pairs possible).
  - Single stratum (H = 1, FPC = inf) with ``L > 0``: serial reduces to
    Newey-West Bartlett HAC on per-period within-stratum-CENTERED per-PSU
    score sequences (NOT raw scores — Binder TSL centering is retained at
    H=1 because the survey path always subtracts the per-period stratum
    mean before the kernel application; the panel-wide ``G/(G-1)``
    survey factor replaces FPC).
  - Bandwidth → 0 with ``L > 0``: spatial reduces to per-period
    within-stratum HC sandwich; serial term unchanged (separable form).
  - All PSUs singleton + ``lonely_psu="remove"`` with ``L > 0``: meat
    NaN-fails on the saturation diagnostic (template "Wave E.2
    stratified-Conley sandwich" covers both spatial-only and panel-block
    cases).

  Centering asymmetry vs no-survey reference:

  - The no-survey panel-block path at ``conley.py:949-965`` uses RAW
    scores for the serial term (no centering) because it assumes
    ``E[scores] = 0`` under correct specification. The survey-weighted
    Binder TSL form estimates the within-stratum mean and centers
    explicitly (textbook stratified-cluster sandwich); raw scores in the
    survey case would inflate variance by twice the squared per-period
    stratum mean and would NOT reduce to the cross-sectional Wave E.2
    form at ``lag = 0``.

  Restrictions:

  - **Requires an effective PSU** — either explicit ``survey_design.psu``
    OR ``cluster=<col>`` injected as the effective PSU per Wave E.1's
    ``_inject_cluster_as_psu``. No-effective-PSU survey designs
    (weights-only / strata-only WITHOUT a cluster fallback) raise
    ``NotImplementedError`` post-resolution at ``SpilloverDiD.fit``
    because the pseudo-PSU = obs-index fallback would silently zero
    the serial sum (each pseudo-PSU appears in exactly one period).
    Tracked as a follow-up in ``DEFERRED.md``.
  - Replicate-weight variance (BRR / Fay / JK1 / JKn / SDR) raises
    ``NotImplementedError`` (inherits Wave E.1 gate).
  - DiagnosticReport routing for the panel-block case is queued for the
    same Wave F follow-up as the Wave E.2 cross-sectional case.
- **Survey-design integration (Wave E.3 — `SurveyDesign.subpopulation()`
  / warn-and-drop full-design retention via zero-pad scores).** SHIPPED
  in Wave E.3. ``SurveyDesign.subpopulation()``-derived designs AND
  warn-and-drop fits now preserve the full-domain resolved survey
  design: ``n_psu`` / ``n_strata`` / ``df_survey`` and the Binder TSL
  per-stratum centering reflect the FULL domain rather than the
  post-``finite_mask`` fit sample.

  .. note::

     Wave E.3 adopts the canonical zero-pad convention from R
     ``survey::svyrecvar`` applied to a ``subset()`` design (Lumley
     2010 §2.5 "Domains and subpopulations") — the same convention
     already established in ``diff_diff/imputation.py:2175-2183``
     (PreTrendsImputation lead regression) and
     ``diff_diff/prep.py:1401-1432`` (DCDH cell variance). The
     ``gamma_hat`` / ``Psi`` construction stays on SURVEY-FINITE-MASK
     inputs (``X_1_sparse_fit``, ``X_10_sparse_fit``, ``eps_10_fit``
     built on ``survey_finite_mask = finite_mask & survey_weights > 0``;
     ``X_2_kept_gamma``, ``eps_2_fit_gamma``,
     ``survey_weights_fit_gamma`` projected from the fit-sample frame
     down to ``survey_finite_mask``) so the drop-first stage-1 FE
     column space excludes zero-weight subpop rows — critical because
     ``_build_butts_fe_design_csr`` re-factorizes inputs via
     ``pd.factorize`` and drops the first unit / time code; including
     zero-weight subpop rows in the factorize input would change which
     unit/time code sorts first and silently shift ``gamma_hat``. The full-domain zero-pad
     invariant is delivered by a new optional
     ``score_pad_mask=survey_finite_mask`` kwarg on
     ``_compute_gmm_corrected_meat``, where
     ``survey_finite_mask = finite_mask & (survey_weights > 0)``
     under the survey path (= ``finite_mask`` on the no-survey
     path). The survey-path subset removes zero-weight subpop rows
     from the FE basis-construction sample so the drop-first
     identification is invariant to which units the subpop mask
     excluded; warn-and-dropped rows (NaN ``y_tilde``) are also
     excluded since ``finite_mask`` is False for them. The helper
     zero-pads the resulting survey-finite-mask ``Psi`` back to
     full panel length AFTER construction but BEFORE kernel dispatch
     via ``Psi_padded[score_pad_mask] = Psi``. Kernel-dispatch arrays
     (``cluster_ids``, ``conley_coords``, ``conley_time``,
     ``conley_unit``, ``resolved_survey``) are passed at FULL
     length so the meat helpers (Binder TSL / stratified-Conley /
     serial Bartlett) see the full-domain PSU / strata / centroid
     / time geometry. The stage-2 OLS solve still runs on
     fit-sample-aligned ``X_2_kept`` / ``y_tilde_fit`` (active
     sample). No reference software combines all four ingredients
     (Conley + Binder TSL + Gardner GMM + R
     ``svyrecvar``-style zero-pad) on a two-stage influence function.

  Reduction semantics:

  - When ``finite_mask.all() == True`` AND all weights ``> 0`` (no
    zero-pad needed): bit-identical to shipped Wave E.2 /
    E.2-follow-up baseline.
  - A2 invariant: warn-and-drop and subpopulation drops are treated
    identically (both apply the zero-pad mechanism).
  - Subpopulation parity vs upstream-subset: ``df_survey`` matches the
    full domain (``n_psu_full - n_strata_full``) regardless of how
    many rows the subpopulation mask excludes. SE may differ from
    ``fit(data[mask], ..., survey_design=plain_design)`` by design
    (subpopulation retains zero-padded PSU geometry; subset drops
    PSUs entirely).

  Restrictions:

  - Replicate-weight variance + subpopulation continues to raise
    ``NotImplementedError`` at the Wave E.1 gate
    (``spillover.py:2400``).
  - TwoStageDiD's analogous ``finite_mask + design-subset`` pattern at
    ``two_stage.py:567-601`` is NOT yet adopted to Wave E.3 — separate
    parity follow-up tracked in ``DEFERRED.md``.
- **Count-of-treated-in-ring** — only the "nearest-treated ring"
  specification is implemented. The "count" form re-introduces
  functional-form dependence (paper Section 3.2 end) and is queued.
- **Data-driven d_bar selection** — Butts (2021b) / Butts (2023) JUE
  Insight propose cross-validation under stronger parallel-trends
  assumptions. Not in this PR.
- **HC2 / HC2_BM (Bell-McCaffrey / CR2) variance** — current stage-2
  inference uses a generic residual-df (n - effective rank) for
  t-distribution lookups. ``vcov_type="hc2"`` / ``"hc2_bm"`` require
  per-coefficient BM / CR2 DOF and raise ``NotImplementedError``.
  Routing stage 2 through ``LinearRegression`` (which supplies the
  per-coefficient DOF metadata) is queued.
- **`vcov_type="classical"` (Wave D restriction)** — raises
  ``NotImplementedError``. The Wave D Gardner GMM first-stage
  uncertainty correction has not been derived for the classical
  homoskedastic variance (different meat structure
  ``sigma_hat^2 * (X_10' X_10)`` vs the Wave D IF outer product
  ``Psi' Psi``). Use ``vcov_type="hc1"`` for heteroskedasticity-robust
  SE with the GMM correction, or combine with ``cluster=<col>`` for
  CR1 with the GMM correction; both apply the Wave D synthesis
  (Butts §3.1 + Gardner §4 + Conley 1999) unconditionally.
- **Balanced panel required** — every unit must observe every period.
  An unbalanced (unit, time) Ω₀ bipartite graph can produce disconnected
  FE components and unidentified stage-1 residuals on treated rows.
  Validator rejects unbalanced inputs.
- **Omega_0 row-level identification (period strict, unit warn-drop, plus connectivity)** —
  Every period must have at least one Omega_0 row (else time FE for
  that period is structurally unidentified; hard ``ValueError``). Units
  lacking Omega_0 rows (e.g. baseline-treated units with ``D_it = 1`` at
  every observed ``t``) are warned-and-dropped: their unit FE is NaN
  and the downstream finite-mask path excludes them from stage 2.
  Additionally, the supported-units bipartite graph (units linked by
  shared Omega_0 periods) must form a single connected component;
  ``K > 1`` components raise ``ValueError`` because the FE solver would
  return only component-specific constants and residualization would
  silently mix them across components. Mirrors ``TwoStageDiD``'s
  always-treated unit handling on the warn-drop axis.
- **One row per (unit, time) cell required** — duplicate cells silently
  re-weight both stage-1 FE estimation and stage-2 OLS. Validator
  rejects duplicate cells; aggregate to unique cells before fitting.
- **rings[0] must equal 0** — the partition must cover treated
  locations (``d_it = 0`` belongs to Ring 1). Validator rejects rings
  that start at a nonzero inner edge to prevent a silent
  exposed-but-unmodeled population in ``0 <= d_it < rings[0]``.
- **conley_coords must be constant within unit** — ring construction
  uses a single (lat, lon) per unit (units are stationary point
  locations). Validator rejects coordinates that vary across rows of
  the same unit; aggregate to one location per unit (e.g.
  ``df.groupby(unit)[["lat","lon"]].first()``) before fitting if your
  raw data has moving coordinates. This requirement applies on every
  fit path, not just ``vcov_type="conley"``, because the rings
  themselves are always coordinate-derived.
