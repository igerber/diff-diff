Two-Stage DiD (Gardner 2022)
============================

Two-stage residualization estimator for staggered Difference-in-Differences.

This module implements the methodology from Gardner (2022), "Two-stage
differences in differences". The method:

1. Estimates unit + time fixed effects on untreated observations only
2. Residualizes ALL outcomes using the estimated fixed effects
3. Regresses residualized outcomes on treatment indicators (Stage 2)

Inference uses the GMM sandwich variance estimator from Butts & Gardner
(2022) that correctly accounts for first-stage estimation uncertainty.
Point estimates are identical to ImputationDiD (Borusyak et al. 2024);
the key difference is the variance estimator (GMM sandwich vs. conservative).

**When to use TwoStageDiD:**

- Staggered adoption settings where you want **efficient point estimates**
  with variance that accounts for first-stage estimation uncertainty
- When you prefer the GMM sandwich variance over the conservative variance
  used by ImputationDiD — the sandwich estimator can yield tighter
  confidence intervals when first-stage uncertainty is small
- As a robustness check alongside CallawaySantAnna and ImputationDiD:
  if all estimators agree, results are robust; if they disagree, investigate
  treatment effect heterogeneity
- When you need an event study that is **free of TWFE contamination bias**

**Reference:** Gardner, J. (2022). Two-stage differences in differences.
*arXiv:2207.05943*. Butts, K. & Gardner, J. (2022). did2s: Two-Stage
Difference-in-Differences. *R Journal*, 14(1), 162-173.

.. module:: diff_diff.two_stage

TwoStageDiD
------------

Main estimator class for two-stage DiD estimation.

.. autoclass:: diff_diff.TwoStageDiD
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   .. rubric:: Methods

   .. autosummary::

      ~TwoStageDiD.fit
      ~TwoStageDiD.get_params
      ~TwoStageDiD.set_params

TwoStageDiDResults
------------------

Results container for two-stage DiD estimation.

.. autoclass:: diff_diff.TwoStageDiDResults
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::

      ~TwoStageDiDResults.summary
      ~TwoStageDiDResults.print_summary
      ~TwoStageDiDResults.to_dataframe

TwoStageBootstrapResults
------------------------

Bootstrap inference results.

.. autoclass:: diff_diff.TwoStageBootstrapResults
   :members:
   :undoc-members:
   :show-inheritance:

Convenience Function
--------------------

.. autofunction:: diff_diff.two_stage_did

Example Usage
-------------

Basic usage::

    from diff_diff import TwoStageDiD, generate_staggered_data

    data = generate_staggered_data(n_units=200, seed=42)
    est = TwoStageDiD()
    results = est.fit(data, outcome='outcome', unit='unit',
                      time='period', first_treat='first_treat')
    results.print_summary()

Event study with visualization::

    from diff_diff import TwoStageDiD, plot_event_study

    est = TwoStageDiD()
    results = est.fit(data, outcome='outcome', unit='unit',
                      time='period', first_treat='first_treat',
                      aggregate='event_study')
    plot_event_study(results)

Comparison with other estimators::

    from diff_diff import TwoStageDiD, CallawaySantAnna, ImputationDiD

    # All three should agree under homogeneous effects
    ts = TwoStageDiD().fit(data, outcome='outcome', unit='unit',
                           time='period', first_treat='first_treat')
    cs = CallawaySantAnna().fit(data, outcome='outcome', unit='unit',
                                time='period', first_treat='first_treat')
    imp = ImputationDiD().fit(data, outcome='outcome', unit='unit',
                              time='period', first_treat='first_treat')

    print(f"Two-Stage ATT: {ts.overall_att:.3f} (SE: {ts.overall_se:.3f})")
    print(f"CS ATT:        {cs.overall_att:.3f} (SE: {cs.overall_se:.3f})")
    print(f"Imputation ATT:{imp.overall_att:.3f} (SE: {imp.overall_se:.3f})")

Estimator Comparison
--------------------

.. list-table:: TwoStageDiD vs. CallawaySantAnna vs. ImputationDiD
   :header-rows: 1
   :widths: 25 25 25 25

   * - Feature
     - TwoStageDiD
     - CallawaySantAnna
     - ImputationDiD
   * - Point estimates
     - Identical to ImputationDiD
     - Group-time ATT(g,t)
     - Identical to TwoStageDiD
   * - Variance estimator
     - GMM sandwich (accounts for first-stage uncertainty)
     - Analytical IF/WIF or multiplier bootstrap
     - Conservative (Theorem 3)
   * - Control group
     - Never-treated + not-yet-treated
     - Never-treated or not-yet-treated
     - Never-treated + not-yet-treated
   * - Efficiency
     - High (uses all untreated obs)
     - Lower (2x2 comparisons)
     - High (uses all untreated obs)
   * - Heterogeneous effects
     - Consistent under homogeneity
     - Robust to heterogeneity
     - Consistent under homogeneity
   * - Covariates
     - Supported
     - Supported (outcome regression or IPW)
     - Supported
   * - Bootstrap
     - Multiplier bootstrap on GMM influence function
     - Multiplier bootstrap (IF/WIF)
     - Multiplier bootstrap
