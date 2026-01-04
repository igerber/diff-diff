Getting Started
===============

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

   # Generate synthetic data with a known treatment effect
   data = generate_did_data(
       n_units=100,
       n_periods=10,
       treatment_effect=5.0,
       treatment_start=5,
       treatment_fraction=0.5,
   )

   # Fit the model
   did = DifferenceInDifferences()
   results = did.fit(
       data,
       outcome='outcome',
       treated='treated',
       post='post'
   )

   # View results
   print(results.summary())

Output:

.. code-block:: text

   Difference-in-Differences Results
   ==================================
   ATT:           5.123
   Std. Error:    0.456
   t-statistic:   11.23
   p-value:       0.000
   95% CI:        [4.229, 6.017]

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

   results = did.fit(
       data,
       outcome='outcome',
       treated='treated',
       post='post',
       covariates=['age', 'income']
   )

Cluster-Robust Standard Errors
------------------------------

For panel data, cluster standard errors at the unit level:

.. code-block:: python

   did = DifferenceInDifferences(cluster_col='unit_id')
   results = did.fit(data, outcome='y', treated='treated', post='post')

Two-Way Fixed Effects
---------------------

For panel data with multiple periods:

.. code-block:: python

   from diff_diff import TwoWayFixedEffects

   twfe = TwoWayFixedEffects()
   results = twfe.fit(
       data,
       outcome='outcome',
       treated='treated',
       unit='unit_id',
       time='period'
   )

Event Study Design
------------------

Examine treatment effects over time:

.. code-block:: python

   from diff_diff import MultiPeriodDiD

   event = MultiPeriodDiD(reference_period=-1)
   results = event.fit(
       data,
       outcome='outcome',
       treated='treated',
       time='period',
       unit='unit_id',
       treatment_start=5
   )

   # Plot the event study
   from diff_diff import plot_event_study
   fig = plot_event_study(results)

Staggered Adoption
------------------

When treatment is adopted at different times across units:

.. code-block:: python

   from diff_diff import CallawaySantAnna

   cs = CallawaySantAnna()
   results = cs.fit(
       data,
       outcome='outcome',
       unit='unit_id',
       time='period',
       first_treat='first_treatment_period'
   )

   # View aggregated treatment effect
   print(f"Overall ATT: {results.att:.3f}")

Parallel Trends Testing
-----------------------

Test the key identifying assumption:

.. code-block:: python

   from diff_diff import check_parallel_trends

   trends_result = check_parallel_trends(
       data,
       outcome='outcome',
       unit='unit_id',
       time='period',
       treated='treated',
       pre_periods=4
   )

   if trends_result['p_value'] > 0.05:
       print("Parallel trends assumption supported")

Sensitivity Analysis
--------------------

Assess robustness to parallel trends violations with Honest DiD:

.. code-block:: python

   from diff_diff import HonestDiD, DeltaRM

   # Compute bounds under relative magnitudes restriction
   honest = HonestDiD(delta=DeltaRM(M_bar=1.0))
   bounds = honest.fit(event_study_results)

   print(f"Robust CI: [{bounds.robust_ci[0]:.3f}, {bounds.robust_ci[1]:.3f}]")

Next Steps
----------

- :doc:`choosing_estimator` - Learn which estimator to use for your design
- :doc:`r_comparison` - See how diff-diff compares to R packages
- :doc:`api/index` - Explore the full API reference
