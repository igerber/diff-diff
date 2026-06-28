Local Projections Difference-in-Differences
===========================================

Local Projections DiD (LP-DiD) estimator for staggered, absorbing-treatment
event studies, from Dube, Girardi, Jordà & Taylor (2025).

LP-DiD estimates a separate regression at each event-time horizon ``h`` of a
long difference of the outcome (``y_{i,t+h} - y_{i,t-1}``) on the
treatment-switch indicator, restricted to a flexible "clean control" sample of
newly-treated observations and not-yet-treated controls. Excluding
already-treated units from the control group removes the negative-weighting
bias of naive two-way fixed effects, so the default (variance-weighted)
estimand is a strictly non-negatively-weighted average of cohort effects.

.. note::

   This release implements the **absorbing-treatment main path**: treatment is
   binary and, once switched on, stays on. The estimator rejects panels where a
   unit's treatment turns off. Non-absorbing (switch on/off) treatment and
   survey-design support are planned follow-ups. Covariates and absorbed fixed
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
     - Binary, absorbing (this release)
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
