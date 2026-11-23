Design Diagnosis
================

Check a planned DiD design on your own untreated panel before analysing it.

.. module:: diff_diff.design

Overview
--------

:func:`~diff_diff.diagnose_design` answers two questions about a planned study, for
each candidate estimator, using the panel you will actually analyse:

1. **Size**: when nothing happened, how often does the estimator reject?
2. **Power**: how often does it detect an effect of a given size, and how biased
   and how precise are its estimates?

It takes a panel in which no unit is treated yet, repeatedly assigns placebo
treatments that follow your declared design (how many units, which adoption
periods), optionally adds a known effect to the placebo-treated observations, and
fits every estimator on the same placebo draws. No real treatment effect is
estimated at any point: pass pre-treatment data only, or pass ``first_treat`` so
that every period from the earliest real adoption onward is dropped.

Estimators are reported in the order you pass them and are never ranked. An
estimator fails the size check when its no-effect rejection rate is above
``alpha + 1.96 * sqrt(alpha * (1 - alpha) / R)`` - the nominal level plus a Monte
Carlo allowance for the ``R`` replicates scored - or when it loses more than 10%
of its replicates. Its power is then not meaningful and no minimum detectable
effect is reported for it.

The methodology, its sources and the choices that have no paper source are in
the Methodology Registry under ``DesignDiagnosis``.

diagnose_design
---------------

.. autofunction:: diff_diff.diagnose_design

DesignDiagnosisResults
----------------------

.. autoclass:: diff_diff.DesignDiagnosisResults
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Example
-------

.. code-block:: python

   from diff_diff import (
       CallawaySantAnna,
       ImputationDiD,
       TwoWayFixedEffects,
       diagnose_design,
   )

   # `panel` holds pre-treatment data only: one row per unit and year.
   result = diagnose_design(
       panel,
       outcome="sales",
       unit="store",
       time="year",
       estimators={
           "twfe": TwoWayFixedEffects(),
           "cs": CallawaySantAnna(),
           "imputation": ImputationDiD(),
       },
       cohort_periods=[2019],        # the planned adoption year
       n_treated=12,                 # the planned number of treated stores
       effect_sizes=[0.02, 0.05],    # effects to inject, on the outcome's scale
       n_simulations=500,
       power=0.80,                   # also search for the minimum detectable effect
       seed=1,
   )

   result.print_summary()
   table = result.to_dataframe()     # one row per estimator and effect size

   # The no-effect run of one estimator: its rejection_rate is the size.
   null_run = result.results["cs"][0]
   print(null_run.rejection_rate, result.size_controlled["cs"], result.mde["cs"])

A staggered design declares several adoption periods; each placebo-treated unit
draws its period from the list:

.. code-block:: python

   result = diagnose_design(
       panel,
       outcome="sales",
       unit="store",
       time="year",
       estimators=[CallawaySantAnna(), ImputationDiD()],
       cohort_periods=[2018, 2019, 2020],
       treatment_fraction=0.4,
       n_simulations=500,
   )

See Also
--------

- :doc:`power` - Power analysis on synthetic data
- :doc:`diagnostics` - Placebo tests for a fitted estimator
