Imputation DiD (Borusyak et al. 2024)
=======================================

Efficient imputation estimator for staggered Difference-in-Differences.

This module implements the methodology from Borusyak, Jaravel & Spiess (2024),
"Revisiting Event-Study Designs: Robust and Efficient Estimation",
*Review of Economic Studies*.

The estimator:

1. Runs OLS on untreated observations to estimate unit + time fixed effects
2. Imputes counterfactual Y(0) for treated observations
3. Aggregates imputed treatment effects with researcher-chosen weights

Inference uses the conservative clustered variance estimator from Theorem 3.

**When to use ImputationDiD:**

- Staggered adoption settings where treatment effects may be **homogeneous**
  across cohorts and time — produces ~50% shorter CIs than Callaway-Sant'Anna
- When you want to use **all untreated observations** (never-treated +
  not-yet-treated) for maximum efficiency
- As a complement to Callaway-Sant'Anna or Sun-Abraham: if all three agree,
  results are robust; if they disagree, investigate heterogeneity

**Reference:** Borusyak, K., Jaravel, X., & Spiess, J. (2024). Revisiting
Event-Study Designs: Robust and Efficient Estimation. *Review of Economic
Studies*, 91(6), 3253-3285.

.. module:: diff_diff.imputation

ImputationDiD
-------------

Main estimator class for imputation DiD estimation.

.. autoclass:: diff_diff.ImputationDiD
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   .. rubric:: Methods

   .. autosummary::

      ~ImputationDiD.fit
      ~ImputationDiD.get_params
      ~ImputationDiD.set_params

ImputationDiDResults
--------------------

Results container for imputation DiD estimation.

.. autoclass:: diff_diff.ImputationDiDResults
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::

      ~ImputationDiDResults.aggregate
      ~ImputationDiDResults.summary
      ~ImputationDiDResults.print_summary
      ~ImputationDiDResults.to_dataframe
      ~ImputationDiDResults.pretrend_test

ImputationBootstrapResults
--------------------------

Bootstrap inference results.

.. autoclass:: diff_diff.ImputationBootstrapResults
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Convenience Function
--------------------

.. autofunction:: diff_diff.imputation_did

Example Usage
-------------

Basic usage::

    from diff_diff import ImputationDiD, generate_staggered_data

    data = generate_staggered_data(n_units=200, seed=42)
    est = ImputationDiD()
    results = est.fit(data, outcome='outcome', unit='unit',
                      time='period', first_treat='first_treat')
    results.print_summary()

Event study with visualization::

    from diff_diff import ImputationDiD, plot_event_study

    est = ImputationDiD()
    results = est.fit(data, outcome='outcome', unit='unit',
                      time='period', first_treat='first_treat')
    es = results.aggregate('event_study')   # post-fit (M-021)
    plot_event_study(es)

Pre-trend test::

    results = est.fit(data, outcome='outcome', unit='unit',
                      time='period', first_treat='first_treat')
    pt = results.pretrend_test(n_leads=3)
    print(f"F-stat: {pt['f_stat']:.3f}, p-value: {pt['p_value']:.4f}")

Comparison with other estimators::

    from diff_diff import ImputationDiD, CallawaySantAnna, SunAbraham

    # All three should agree under homogeneous effects
    imp = ImputationDiD().fit(data, ...)
    cs = CallawaySantAnna().fit(data, ...)
    sa = SunAbraham().fit(data, ...)

    print(f"Imputation ATT: {imp.overall_att:.3f} (SE: {imp.overall_se:.3f})")
    print(f"CS ATT: {cs.overall_att:.3f} (SE: {cs.overall_se:.3f})")
    print(f"SA ATT: {sa.overall_att:.3f} (SE: {sa.overall_se:.3f})")
