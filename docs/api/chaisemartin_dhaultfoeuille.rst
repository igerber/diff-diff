de Chaisemartin-D'Haultfœuille (dCDH) DiD
============================================

The most general estimator in diff-diff for **non-absorbing (reversible)
treatments** — treatment may switch on AND off over time, with explicit
joiner/leaver decomposition and multi-horizon dynamics. (:class:`~diff_diff.LPDiD`
and :class:`~diff_diff.TROP` also support non-absorbing treatment under stronger
assumptions; see their ``non_absorbing`` parameters.)

This module implements the methodology from de Chaisemartin & D'Haultfœuille
(2020/2022). The estimator ships the contemporaneous-switch path ``DID_M``
(= ``DID_1`` at horizon ``l = 1``); the full multi-horizon event study
``DID_l`` for ``l = 1..L_max`` via the ``L_max`` parameter, with normalized
estimator ``DID^n_l``, cost-benefit aggregate ``delta``, dynamic placebos
``DID^{pl}_l``, and sup-t simultaneous confidence bands; residualization-style
covariate adjustment (``controls``); group-specific linear trends
(``trends_linear``); state-set-specific trends (``trends_nonparam``);
heterogeneity testing; non-binary treatment; HonestDiD sensitivity
integration on placebos; survey support via Taylor-series linearization
(pweight + strata/PSU/FPC); and per-path event-study disaggregation via
``by_path=k`` (mirrors R ``did_multiplegt_dyn(..., by_path=k)``,
including per-path backward placebos and per-path joint sup-t
simultaneous bands when ``n_bootstrap > 0`` — Python-only extension
beyond R, which provides no joint bands at any surface) or via
``paths_of_interest=[(...), ...]`` for an explicit user-specified
path subset (Python-only API; mutex with ``by_path``). ``by_path``
supports binary or integer-coded discrete (D in Z) treatment, and
composes with ``survey_design`` for analytical Binder TSL SE and
replicate-weight bootstrap variance (multiplier bootstrap under
survey + by_path remains gated; no R parity since R
``did_multiplegt_dyn`` does not support survey weighting). ``by_path``
and ``paths_of_interest`` also compose with ``heterogeneity="<col>"``:
per-path heterogeneity coefficient surfaces on
``results.path_heterogeneity_effects`` (mirrors R
``did_multiplegt_dyn(..., by_path, predict_het)`` per-by_level). When
combined with ``placebo=True``, heterogeneity is also computed on
backward (placebo) horizons and surfaced under negative-int keys —
both globally on ``results.heterogeneity_effects[-l]`` and per-path
on ``results.path_heterogeneity_effects[path][-l]``;
``to_dataframe(level="by_path")`` placebo rows have populated ``het_*``
columns. ``survey_design + placebo + heterogeneity`` emits a
``UserWarning`` at fit-time and falls back to forward-horizon-only
heterogeneity until the pre-period cell allocator is derived; forward-
horizon ``predict_het + survey_design`` continues to work unchanged.

The estimator:

1. Aggregates individual-level panel data to ``(group, time)`` cells
2. Drops multi-switch groups by default (matches R ``DIDmultiplegtDYN``)
3. Excludes singleton-baseline groups from the variance computation only (footnote 15 of the dynamic paper)
4. Computes per-period joiner (``DID_{+,t}``) and leaver (``DID_{-,t}``)
   contributions via Theorem 3 of the AER 2020 paper
5. Aggregates them into ``DID_M``, the joiners-only ``DID_+``, and the
   leavers-only ``DID_-``
6. Computes the single-lag placebo ``DID_M^pl``
7. When ``L_max >= 2``: computes per-group ``DID_{g,l}`` building blocks,
   multi-horizon ``DID_l``, dynamic placebos ``DID^{pl}_l``, normalized
   ``DID^n_l``, and cost-benefit aggregate ``delta``
8. Optionally computes the TWFE decomposition diagnostic from Theorem 1
   (per-cell weights, fraction negative, ``sigma_fe``)
9. Inference uses the cohort-recentered analytical plug-in variance from
   Web Appendix Section 3.7.3 of the dynamic paper, optionally
   complemented by a multiplier bootstrap clustered at the group level
   (with sup-t simultaneous confidence bands when ``L_max >= 2``)

**When to use ChaisemartinDHaultfoeuille:**

- Treatment can switch on **and** off over time (e.g., marketing campaigns,
  seasonal promotions, on/off policy cycles)
- You need separate joiners (``DID_+``) and leavers (``DID_-``) views, plus
  the aggregate ``DID_M``
- You want a built-in placebo and a TWFE decomposition diagnostic computed
  on the data you pass in (pre-filter) for direct comparison against
  ``DID_M``. The fitted TWFE diagnostic uses the FULL pre-filter cell
  sample (matching :func:`twowayfeweights`); when ``fit()`` drops groups
  via the ragged-panel or ``drop_larger_lower`` filters, a ``UserWarning``
  is emitted to make the divergence from the post-filter ``DID_M`` sample
  explicit. See REGISTRY.md ``ChaisemartinDHaultfoeuille`` ``Note (TWFE
  diagnostic sample contract)`` for the rationale.
- You want a Python implementation that matches R ``DIDmultiplegtDYN`` at
  ``l = 1`` on cell-aggregated input (see REGISTRY.md for documented
  deviations on individual-level inputs with uneven cell sizes)

The remaining staggered estimators in diff-diff (:class:`~diff_diff.CallawaySantAnna`,
:class:`~diff_diff.SunAbraham`, :class:`~diff_diff.ImputationDiD`,
:class:`~diff_diff.TwoStageDiD`, :class:`~diff_diff.EfficientDiD`,
:class:`~diff_diff.WooldridgeDiD`) assume treatment is **absorbing** —
once treated, stays treated. ``ChaisemartinDHaultfoeuille`` is the most general
option for non-absorbing treatments; :class:`~diff_diff.LPDiD`
(``non_absorbing="first_entry"`` / ``"effect_stabilization"``) and
:class:`~diff_diff.TROP` (``non_absorbing=True``, under a no-dynamic-effects
assumption) also support non-absorbing treatment.

**Panel requirements (deviation from R DIDmultiplegtDYN):**

- Every group must have an observation at the **first global period**
  (the panel's earliest time value). Groups missing this baseline raise
  ``ValueError`` with the offending group IDs.
- Groups with **interior period gaps** (missing observations between
  their first and last observed period) are dropped with a
  ``UserWarning``.
- **Terminal missingness** (groups observed at the baseline but missing
  one or more later periods - early exit / right-censoring) is supported.
  The group contributes from its observed periods only, masked out of
  the missing transitions by the per-period ``present`` guard in the
  variance computation.
- This is a documented deviation from R ``DIDmultiplegtDYN``, which
  supports unbalanced panels with missing-treatment-before-first-switch
  handling. **Workaround:** pre-process your panel to back-fill the
  baseline (or drop late-entry groups before fitting), or use R until
  this restriction is lifted. See the
  ``Note (deviation from R DIDmultiplegtDYN)`` block in
  ``docs/methodology/REGISTRY.md`` for the rationale and the exact
  defensive guards that make terminal missingness safe.

**References:**

- de Chaisemartin, C. & D'Haultfœuille, X. (2020). Two-Way Fixed Effects
  Estimators with Heterogeneous Treatment Effects. *American Economic
  Review*, 110(9), 2964-2996.
- de Chaisemartin, C. & D'Haultfœuille, X. (2022, revised July 2023).
  Difference-in-Differences Estimators of Intertemporal Treatment
  Effects. NBER Working Paper 29873.

ChaisemartinDHaultfoeuille
--------------------------

Main estimator class for de Chaisemartin-D'Haultfœuille (dCDH) DiD estimation.
The alias :class:`~diff_diff.DCDH` is also available.

.. autoclass:: diff_diff.ChaisemartinDHaultfoeuille
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   .. rubric:: Methods

   .. autosummary::

      ~ChaisemartinDHaultfoeuille.fit
      ~ChaisemartinDHaultfoeuille.get_params
      ~ChaisemartinDHaultfoeuille.set_params

ChaisemartinDHaultfoeuilleResults
---------------------------------

Results container for dCDH estimation.

.. autoclass:: diff_diff.ChaisemartinDHaultfoeuilleResults
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::

      ~ChaisemartinDHaultfoeuilleResults.summary
      ~ChaisemartinDHaultfoeuilleResults.print_summary
      ~ChaisemartinDHaultfoeuilleResults.to_dataframe

DCDHBootstrapResults
--------------------

Multiplier-bootstrap inference results, populated when ``n_bootstrap > 0``.

.. autoclass:: diff_diff.DCDHBootstrapResults
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Convenience Function
--------------------

.. autofunction:: diff_diff.chaisemartin_dhaultfoeuille

Standalone TWFE Decomposition Diagnostic
----------------------------------------

The TWFE decomposition diagnostic from Theorem 1 of de Chaisemartin &
D'Haultfœuille (2020) is also available as a standalone function for
users who want the diagnostic without fitting the full estimator. It
returns per-cell weights, the fraction of treated cells with negative
weights, and ``sigma_fe`` — the smallest standard deviation of per-cell
treatment effects that could flip the sign of the plain TWFE coefficient.

.. autofunction:: diff_diff.twowayfeweights

.. autoclass:: diff_diff.TWFEWeightsResult
   :no-index:
   :members:

Example Usage
-------------

Basic usage with reversible treatment::

    from diff_diff import ChaisemartinDHaultfoeuille
    from diff_diff.prep import generate_reversible_did_data

    data = generate_reversible_did_data(
        n_groups=80, n_periods=6, pattern="single_switch", seed=42,
    )

    est = ChaisemartinDHaultfoeuille()
    results = est.fit(
        data,
        outcome="outcome",
        group="group",
        time="period",
        treatment="treatment",
    )
    results.print_summary()

Joiners and leavers views::

    print(f"DID_M (overall):  {results.overall_att:.3f}")
    print(f"DID_+ (joiners):  {results.joiners_att:.3f}")
    print(f"DID_- (leavers):  {results.leavers_att:.3f}")
    print(f"Placebo (DID^pl): {results.placebo_effect:.3f}")

Per-period decomposition::

    for t, cell in results.per_period_effects.items():
        print(
            f"t={t}: DID+={cell['did_plus_t']:.3f} "
            f"({cell['n_10_t']} joiners, {cell['n_00_t']} stable_0 controls)"
        )

Multiplier bootstrap inference::

    est = ChaisemartinDHaultfoeuille(
        n_bootstrap=999, bootstrap_weights="rademacher", seed=42,
    )
    results = est.fit(
        data, outcome="outcome", group="group",
        time="period", treatment="treatment",
    )
    # When n_bootstrap > 0, the top-level overall_*/joiners_*/leavers_*
    # p-value and conf_int fields hold percentile-based bootstrap
    # inference (not normal-theory recomputations from the bootstrap SE).
    # The t-stat is computed from the SE in both cases. See REGISTRY.md
    # `Note (bootstrap inference surface)` for the full contract.
    print(f"Top-level p-value (bootstrap): {results.overall_p_value:.4f}")
    print(f"Top-level CI (bootstrap):     {results.overall_conf_int}")
    print(f"bootstrap_results.overall_se: {results.bootstrap_results.overall_se:.3f}")
    print(f"bootstrap_results.overall_ci: {results.bootstrap_results.overall_ci}")

Standalone TWFE diagnostic (without fitting the full estimator)::

    from diff_diff import twowayfeweights

    diagnostic = twowayfeweights(
        data, outcome="outcome", group="group", time="period", treatment="treatment",
    )
    print(f"Plain TWFE coefficient: {diagnostic.beta_fe:.3f}")
    print(f"Fraction of negative weights: {diagnostic.fraction_negative:.3f}")
    print(f"sigma_fe (sign-flipping threshold): {diagnostic.sigma_fe:.3f}")

The ``DCDH`` alias::

    from diff_diff import DCDH

    est = DCDH()  # equivalent to ChaisemartinDHaultfoeuille()
