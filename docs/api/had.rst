Heterogeneous Adoption Difference-in-Differences
================================================

Estimator for designs where **no unit remains untreated** at the post period.
Every unit `g` is exposed to treatment at the same single date but adoption
intensity (dose) varies across units; there is no genuinely untreated control
group to anchor a standard DiD contrast.

This module implements the methodology from de Chaisemartin, Ciccia,
D'Haultfœuille & Knau (2026), "Difference-in-Differences Estimators When No
Unit Remains Untreated" (arXiv:2405.04465v6), which:

1. **Targets WAS or WAS_{d̲} depending on design path:** Design 1' (the
   QUG / Quasi-Untreated-Group case with ``d̲ = 0``) identifies the
   Weighted Average Slope (WAS, paper Equation 2); Design 1 (no QUG,
   ``d̲ > 0``) identifies ``WAS_{d̲}`` under Assumption 6, or sign
   identification only under Assumption 5 (neither additional assumption
   is testable via pre-trends). The shipped result classes expose
   ``target_parameter == "WAS"`` versus ``"WAS_d_lower"`` so callers can
   key on the resolved estimand.
2. **Estimates the target via local-linear regression at the dose support
   boundary**, with three concrete fit paths: ``continuous_at_zero`` for
   Design 1', and ``continuous_near_d_lower`` or ``mass_point`` for
   Design 1 (auto-detected from the dose distribution).
3. **Provides bias-corrected confidence intervals** ported from the
   ``nprobust`` machinery for the continuous-dose paths, and a
   structural-residual 2SLS sandwich for the mass-point path.
4. **Extends to multi-period event-study settings** (paper Appendix B.2),
   restricting staggered-timing panels to the last-treatment cohort (which
   retains never-treated units as comparisons) with pointwise per-horizon CIs.

.. note::

   **When to use HAD.** Use ``HeterogeneousAdoptionDiD`` when your panel has
   no untreated unit at the post period (e.g. universal-rollout policies,
   industry-wide tariff changes) but treatment intensity varies across
   units. For panels with a never-treated control group and continuous
   treatment, use :class:`~diff_diff.ContinuousDiD` instead. For binary
   reversible treatments, use :class:`~diff_diff.ChaisemartinDHaultfoeuille`.

.. note::

   **Inference contract.** Per-horizon CIs are always pointwise. There are
   two SE regimes selected by call site:

   - **Unweighted** - continuous paths use the CCT-2014 robust SE
     from the in-house ``lprobust`` port; the mass-point path uses a
     structural-residual 2SLS sandwich. No cross-horizon covariance.
   - **``survey_design=SurveyDesign(weights="col", ...)``** (the sole
     weighting entry; accepts strata / PSU / FPC) - both paths compose
     Binder (1983) Taylor-series linearization with ``df_survey`` threaded
     into ``safe_inference``. Yields ``variance_formula="survey_binder_tsl"``
     / ``"survey_binder_tsl_2sls"``. A bare ``cluster=`` (unweighted) gives
     the CR1 2SLS sandwich on the mass-point path; for weighted clustering
     use ``survey_design=SurveyDesign(weights='<weight_col>',
     psu='<cluster_col>')`` (which composes Binder-TSL through the PSU).
     ``hc2`` / ``hc2_bm`` raise ``NotImplementedError`` pending a
     2SLS-specific leverage derivation.

   On ``HeterogeneousAdoptionDiD.fit`` the deprecated ``weights=`` and
   ``survey=`` aliases were removed in 3.7.0; ``survey_design=`` is the sole
   weighting entry (Binder-TSL). On the HAD pretest helpers the aliases
   remain deprecated pending a follow-up removal: array-in helpers
   (``stute_test``, ``yatchew_hr_test``, ``stute_joint_pretest``) take the
   pweight-only shortcut ``survey_design=make_pweight_design(weights)``;
   data-in surfaces use ``survey_design=SurveyDesign(weights="col_name",
   ...)`` against ``data``. ``qug_test`` is the exception: the QUG step has
   no survey-aware migration target (Phase 4.5 C0 decision; see methodology
   REGISTRY) and permanently raises ``NotImplementedError`` on
   ``survey_design=``. The composite workflow ``did_had_pretest_workflow``
   handles this by skipping QUG under survey dispatch and emitting a
   ``UserWarning``.

   A simultaneous confidence band (sup-t) is available on the event-study
   path via ``cband=True`` whenever ``survey_design=`` **or**
   ``cluster=`` is supplied. With a bare ``cluster=`` the band is
   cluster-robust (pointwise CIs are cluster-robust too) on an unweighted
   fit, on both designs; ``cluster=`` + ``survey_design=`` is rejected -
   for weighted clustering pass
   ``survey_design=SurveyDesign(weights='<weight_col>', psu='<cluster_col>')``
   (which takes the survey Binder-TSL branch, not the bare-cluster branch).
   Joint cross-horizon analytical covariance is not computed in this
   release; tracked in ``TODO.md``.

   **Mass-point ``vcov_type="classical"`` deviation.** The mass-point
   ``survey_design=SurveyDesign(...)`` paths (static and event-study)
   reject ``vcov_type="classical"`` with
   ``NotImplementedError`` **when ``cluster=`` is not set** (a ``cluster=``
   fit computes the CR1 sandwich regardless of ``vcov_type``, so no
   classical/HC1 mismatch arises). The per-unit 2SLS influence function returned
   by the mass-point fit is HC1-scaled so that
   ``compute_survey_if_variance`` and the sup-t bootstrap target
   ``V_HC1`` consistently; mixing it with a classical analytical SE
   would silently report a ``V_HC1``-targeted variance under a
   ``classical`` label. Use ``vcov_type="hc1"`` or set ``robust=True``
   explicitly (the constructor default ``robust=False`` maps to
   ``vcov_type="classical"``, which triggers the guard); a
   classical-aligned IF derivation is queued for a follow-up PR.

   **``cluster=`` + ``survey_design=`` deviation.** On both designs,
   ``cluster=`` + ``survey_design=SurveyDesign(...)`` (static and
   event-study) is rejected outright regardless of ``vcov_type``: the
   survey path composes Binder-TSL variance, which would silently
   override the cluster-robust sandwich. Workarounds: ``cluster=`` alone
   (unweighted cluster-robust), or route clustering through
   ``survey_design=SurveyDesign(weights='<weight_col>', psu='<cluster_col>')``. All other
   ``cluster=`` compositions are supported end-to-end, including the
   ``cluster=`` + ``aggregate="event_study"`` + ``cband=True`` mass-point
   path: the clustered sup-t band draws cluster-level multipliers on the
   per-unit influence function and normalizes by the CR1 analytical SE
   (variance families reconciled via the ``√(G/(G-1))`` CR1 scalar).

.. tip::

   For an end-to-end walkthrough of the survey-aware HAD workflow on a
   BRFSS-shape stratified household-survey panel - including the now-
   supported ``SurveyDesign(strata=...)`` path through the Stute pretest
   family (lifted in PR #432, 2026-05) - see
   `Tutorial 22: Survey-Weighted HAD
   <../tutorials/22_had_survey_design.ipynb>`_.

HeterogeneousAdoptionDiD
------------------------

.. autoclass:: diff_diff.HeterogeneousAdoptionDiD
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

HeterogeneousAdoptionDiDResults
-------------------------------

Single-period results container for ``HeterogeneousAdoptionDiD`` estimation.

.. autoclass:: diff_diff.HeterogeneousAdoptionDiDResults
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

HeterogeneousAdoptionDiDEventStudyResults
-----------------------------------------

Multi-period event-study results container for the Appendix B.2 extension.

.. autoclass:: diff_diff.HeterogeneousAdoptionDiDEventStudyResults
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

HAD Pretests
------------

Diagnostic pretests for the HAD identification assumptions from de Chaisemartin
et al. (2026). The composite orchestrator
:func:`~diff_diff.did_had_pretest_workflow` is a diagnostic battery only - it
does NOT pick the HAD design path (``continuous_at_zero`` /
``continuous_near_d_lower`` / ``mass_point``); that is auto-detected inside
:meth:`HeterogeneousAdoptionDiD.fit` from the dose support. The workflow has
two explicit modes selected by the caller via the ``aggregate=`` kwarg:
``aggregate="overall"`` (default, two-period first-differenced sample) runs
single-period tests; ``aggregate="event_study"`` (multi-period panel with
three or more periods) runs joint multi-period tests. Both modes return a
unified :class:`~diff_diff.HADPretestReport`.

.. autofunction:: diff_diff.did_had_pretest_workflow

.. autoclass:: diff_diff.HADPretestReport
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Single-period tests (``aggregate="overall"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: diff_diff.qug_test

.. autofunction:: diff_diff.stute_test

.. autofunction:: diff_diff.yatchew_hr_test

.. autoclass:: diff_diff.QUGTestResults
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: diff_diff.StuteTestResults
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: diff_diff.YatchewTestResults
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Joint multi-period tests (``aggregate="event_study"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: diff_diff.stute_joint_pretest

.. autofunction:: diff_diff.joint_pretrends_test

.. autofunction:: diff_diff.joint_homogeneity_test

.. autoclass:: diff_diff.StuteJointResult
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
