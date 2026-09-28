.. meta::
   :description: Get started with diff-diff for Difference-in-Differences analysis in Python. Step-by-step tutorial covering basic DiD, formulas, covariates, and robust inference.
   :keywords: difference-in-differences tutorial, DiD python getting started, causal inference quickstart

.. _getting-started:

Quickstart
==========

This guide will help you get started with diff-diff for Difference-in-Differences analysis.

Installation
------------

Install diff-diff using pip:

.. code-block:: bash

   pip install diff-diff

Basic 2x2 DiD
-------------

The simplest DiD design has two groups (treated/control) and two periods (pre/post).

.. code-block:: python

   import pandas as pd
   from diff_diff import DifferenceInDifferences, generate_did_data

.. tip::

   Most estimators have short aliases for convenience — e.g.
   ``from diff_diff import DiD, TWFE, CS, DDD``.
   See the :doc:`API reference <api/estimators>` for the full list.

.. code-block:: python

   # Generate synthetic data with a known treatment effect
   data = generate_did_data(
       n_units=100,
       n_periods=10,
       treatment_effect=5.0,
       treatment_period=5,
       treatment_fraction=0.5,
       seed=42,
   )

   # Fit the model
   did = DifferenceInDifferences()
   results = did.fit(
       data,
       outcome='outcome',
       treatment='treated',
       post='post'
   )

   # View results
   print(results.summary())

Output:

.. code-block:: text

   ======================================================================
                Difference-in-Differences Estimation Results
   ======================================================================

   Observations:                   1000
   Treated:                         500
   Control:                         500
   R-squared:                    0.7332
   Variance:                            HC1 heteroskedasticity-robust

   ----------------------------------------------------------------------
   Parameter           Estimate    Std. Err.     t-stat      P>|t|
   ----------------------------------------------------------------------
   ATT                   5.1216       0.2455     20.863     0.0000   ***
   ----------------------------------------------------------------------

   95% Confidence Interval: [4.6399, 5.6034]
   CV (SE/abs(ATT)):             0.0479

   Signif. codes: '***' 0.001, '**' 0.01, '*' 0.05, '.' 0.1
   ======================================================================

Using Formula Interface
-----------------------

You can also use R-style formulas:

.. code-block:: python

   did = DifferenceInDifferences()
   results = did.fit(data, formula='outcome ~ treated * post')

Adding Covariates
-----------------

Control for confounders with the ``covariates`` parameter:

.. code-block:: python

   import numpy as np

   # Add two confounders to the simulated panel
   rng = np.random.default_rng(0)
   data["age"] = rng.integers(20, 70, size=len(data))
   data["income"] = rng.normal(50_000, 15_000, size=len(data)).round(0)

   results = did.fit(
       data,
       outcome='outcome',
       treatment='treated',
       post='post',
       covariates=['age', 'income']
   )

   print(f"Covariate-adjusted ATT: {results.att:.4f}")

Cluster-Robust Standard Errors
------------------------------

For panel data, cluster standard errors at the unit level:

.. code-block:: python

   did = DifferenceInDifferences(cluster='unit')
   results = did.fit(data, outcome='outcome', treatment='treated', post='post')

Two-Way Fixed Effects
---------------------

For panel data with multiple periods:

.. code-block:: python

   from diff_diff import TwoWayFixedEffects

   twfe = TwoWayFixedEffects()
   results = twfe.fit(
       data,
       outcome='outcome',
       treatment='treated',
       unit='unit',
       post='post'
   )

Event Study Design
------------------

Examine treatment effects over time with the TwoWayFixedEffects
event-study mode (``spec="pooled"`` reproduces the deprecated
``MultiPeriodDiD`` design; the default ``spec="within"`` adds unit fixed
effects):

.. code-block:: python

   from diff_diff import TwoWayFixedEffects

   event = TwoWayFixedEffects()
   event_study_results = event.fit(
       data,
       outcome='outcome',
       treatment='treated',
       unit='unit',
       event_study=True,
       time='period',
       post_periods=[5, 6, 7, 8, 9],
       reference_period=4
   )

   # Plot the event study
   from diff_diff.visualization import plot_event_study
   ax = plot_event_study(event_study_results)

Staggered Adoption
------------------

When treatment is adopted at different times across units:

.. code-block:: python

   from diff_diff import CallawaySantAnna, generate_staggered_data

   # Staggered data carries a ``first_treat`` column (0 for never treated)
   staggered = generate_staggered_data(
       n_units=100,
       n_periods=10,
       cohort_periods=[4, 7],
       never_treated_frac=0.3,
       seed=42,
   )

   cs = CallawaySantAnna()
   results = cs.fit(
       staggered,
       outcome='outcome',
       unit='unit',
       time='period',
       first_treat='first_treat'
   )

   # View aggregated treatment effect
   print(f"Overall ATT: {results.overall_att:.3f}")

Parallel Trends Testing
-----------------------

Test the key identifying assumption:

.. code-block:: python

   from diff_diff.utils import check_parallel_trends

   trends_result = check_parallel_trends(
       data,
       outcome='outcome',
       time='period',
       treatment_group='treated',
       pre_periods=[0, 1, 2, 3]
   )

   if trends_result['p_value'] > 0.05:
       print("Parallel trends assumption supported")

Sensitivity Analysis
--------------------

Assess robustness to parallel trends violations with Honest DiD:

.. code-block:: python

   from diff_diff import HonestDiD

   # Compute bounds under relative magnitudes restriction
   honest = HonestDiD(method="relative_magnitude", M=1.0)
   bounds = honest.fit(event_study_results)

   print(f"Robust CI: [{bounds.ci_lb:.3f}, {bounds.ci_ub:.3f}]")

Next Steps
----------

- :doc:`choosing_estimator` - Learn which estimator to use for your design
- :doc:`r_comparison` - See how diff-diff compares to R packages
- :doc:`api/index` - Explore the full API reference
- `Survey-aware DiD tutorial <https://github.com/igerber/diff-diff/blob/main/docs/tutorials/16_survey_did.ipynb>`_ - Using DiD with complex survey designs (strata, PSU, FPC, weights)
