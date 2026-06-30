Local Projections Difference-in-Differences
===========================================

Local Projections DiD (LP-DiD) estimator for staggered event studies, from
Dube, Girardi, Jordà & Taylor (2025). Absorbing treatment by default, with
optional non-absorbing (reversible) treatment via ``non_absorbing``.

LP-DiD estimates a separate regression at each event-time horizon ``h`` of a
long difference of the outcome (``y_{i,t+h} - y_{i,t-1}``) on the
treatment-switch indicator, restricted to a flexible "clean control" sample of
newly-treated observations and not-yet-treated controls. Excluding
already-treated units from the control group removes the negative-weighting
bias of naive two-way fixed effects, so the default (variance-weighted)
estimand is a strictly non-negatively-weighted average of cohort effects.

.. note::

   Treatment is binary. By default (``non_absorbing=None``) the estimator
   follows the **absorbing main path**: once switched on, treatment stays on,
   and panels where a unit's treatment turns off are rejected. Non-absorbing
   (switch on/off) treatment is supported via ``non_absorbing="first_entry"``
   (Eq. 12, the effect of entering for the first time and staying treated) or
   ``non_absorbing="effect_stabilization"`` (Eq. 13, which requires
   ``stabilization_window=L`` and lets units whose treatment has been stable for
   at least ``L`` periods serve as clean controls — feasible with few or no
   never-treated units). Non-absorbing modes require a gap-free panel within
   each unit's observed span and cover the entry-effect estimands. The
   non-absorbing entry-effect paths are R-parity-validated against an independent
   ``fixest::feols`` reconstruction of the paper's Eq. 12/13 (see
   ``docs/methodology/REGISTRY.md``); the Appendix-C exit-event dynamics and the
   Stata canonical SE remain planned follow-ups.
   Complex-survey designs (probability weights + stratified-PSU
   Taylor-linearization standard errors with optional finite-population
   correction and lonely-PSU handling) are supported on the variance-weighted
   default path via the ``survey_design`` argument to ``fit()`` (pass a
   :class:`~diff_diff.SurveyDesign`); ``df = n_PSU - n_strata``. The
   reweighted / regression-adjustment path, replicate-weight designs, and
   non-pweight (fweight/aweight) types are not yet supported with a survey
   design.
   Covariates and absorbed fixed
   effects are supported; under ``reweight=False`` they enter by direct
   inclusion, which preserves the non-negative weighting result only under
   homogeneous covariate effects (online Appendix B.2.2) — the
   regression-adjustment path (``reweight=True``) is preferred for
   covariate-adjusted designs (it does not auto-switch; the default remains
   ``reweight=False``, which emits the warning). The ``time`` column must be
   numeric with integer-spaced periods (long differences use ``t-1`` / ``t+h``
   arithmetic on the labels); map irregular or datetime periods to consecutive
   integers first. See ``docs/methodology/REGISTRY.md`` for the full contract.

**When to use LPDiD:**

- Staggered, **absorbing** adoption where you want a fast, transparent,
  regression-based event study free of negative weighting
- You want both a dynamic event-study path and a single pooled pre/post ATT
- You want to flexibly choose the pretreatment base period (first-lag or
  premean-differenced) or hold the post-treatment sample composition fixed across post horizons
- You want an estimator that is numerically equivalent to Callaway-Sant'Anna
  (reweighted) or to a Cengiz et al. (2019)-style stacked regression
  (variance-weighted), but much faster

**Reference:** Dube, A., Girardi, D., Jordà, Ò., & Taylor, A. M. (2025). A Local
Projections Approach to Difference-in-Differences. *Journal of Applied
Econometrics*, 40(5), 741-758.

.. module:: diff_diff.lpdid

LPDiD
-----

Main estimator class for Local Projections Difference-in-Differences.

.. autoclass:: diff_diff.LPDiD
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::

      ~LPDiD.fit
      ~LPDiD.get_params
      ~LPDiD.set_params

LPDiDResults
------------

Results container for LP-DiD estimation (event-study and pooled tables).

.. autoclass:: diff_diff.lpdid_results.LPDiDResults
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::

      ~LPDiDResults.summary
      ~LPDiDResults.print_summary
      ~LPDiDResults.to_dataframe
      ~LPDiDResults.to_dict

Example Usage
-------------

Basic usage (LP-DiD takes a binary, absorbing treatment indicator)::

    from diff_diff import LPDiD, generate_staggered_data

    data = generate_staggered_data(n_units=300, n_periods=12,
                                   cohort_periods=[4, 7, 10], seed=42)
    # Binary absorbing indicator: 1 from a unit's first treated period onward.
    data["treated"] = (data["period"] >= data["first_treat"]).astype(int)

    lp = LPDiD(pre_window=5, post_window=4)
    results = lp.fit(data, outcome="outcome", unit="unit",
                     time="period", treatment="treated")
    results.print_summary()
    print(results.event_study)   # per-horizon coefficients
    print(results.pooled)        # pooled pre (placebo) / post (ATT) rows

Equally-weighted ATT (numerically equivalent to Callaway-Sant'Anna)::

    lp_rw = LPDiD(pre_window=5, post_window=4, reweight=True)
    results_rw = lp_rw.fit(data, outcome="outcome", unit="unit",
                           time="period", treatment="treated")
    print(f"Variance-weighted ATT: {results.att:.4f} (SE={results.se:.4f})")
    print(f"Equally-weighted ATT:  {results_rw.att:.4f} (SE={results_rw.se:.4f})")

Premean-differenced base period and fixed-composition sample::

    lp_pmd = LPDiD(pre_window=5, post_window=4, pmd="max", no_composition=True)
    results_pmd = lp_pmd.fit(data, outcome="outcome", unit="unit",
                             time="period", treatment="treated")

Non-absorbing (reversible) treatment — units may switch on and off. Use
``effect_stabilization`` (Eq. 13) when there are few or no never-treated units,
so that units whose treatment has been stable for ``L`` periods can serve as
clean controls::

    # `treated` here is a non-absorbing 0/1 indicator that can turn on and off.
    lp_na = LPDiD(pre_window=3, post_window=4,
                  non_absorbing="effect_stabilization", stabilization_window=5)
    results_na = lp_na.fit(panel, outcome="y", unit="unit",
                           time="t", treatment="treated")

Comparison with Other Staggered Estimators
------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 27 27 26

   * - Feature
     - LPDiD
     - CallawaySantAnna
     - ImputationDiD
   * - Approach
     - Per-horizon long-difference LP regression on clean controls
     - Separate 2x2 DiD aggregation
     - Impute Y(0) via FE model
   * - Treatment
     - Binary; absorbing or non-absorbing (``non_absorbing=``)
     - Binary, absorbing
     - Binary, absorbing
   * - Default estimand
     - Variance-weighted ATT (non-negative weights)
     - Equally-weighted ATT
     - Equally-weighted ATT
   * - Equivalences
     - Reweighted == CS; variance-weighted == Cengiz (2019)-style stacking (not ``diff_diff.StackedDiD``, which is Wing et al. 2024 Q-weights); PMD single-cohort == BJS
     - Baseline
     - == reweighted PMD LP-DiD (single cohort)
   * - Covariates
     - Supported (regression adjustment preferred; direct inclusion under homogeneity)
     - Supported (OR, IPW, DR)
     - Supported
   * - Inference
     - Cluster-robust at unit (default)
     - Multiplier bootstrap
     - Influence-function cluster variance
   * - Speed
     - Very fast (stack of small OLS fits)
     - Slower (pairwise group-time)
     - Fast
