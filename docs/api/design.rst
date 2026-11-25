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

Candidates are estimator instances, so two configurations of one estimator can
be compared. With few units, ``CallawaySantAnna(vcov_type="jackknife")`` (the
cluster jackknife, which deletes one unit at a time here) is a natural candidate
next to the default ``CallawaySantAnna()``: the size check then shows on your own
panel whether the default standard errors over-reject and whether the jackknife
corrects it.

**What it does not test.** Placebo treatment is assigned at random, so these runs
test inference - size and power - on your panel's own error structure. They do
not test parallel trends: a design that passes the size check can still be
biased if treatment goes to units on a different trend.

**Design lock.** :meth:`~diff_diff.DesignDiagnosisResults.lock` records one
estimator and its configuration as a :class:`~diff_diff.DesignLock`. Comparing
several estimators and then quoting the chosen one's power from the same runs
overstates it, so when more than one estimator was diagnosed, ``lock`` first
reruns the chosen one alone on the same panel with new placebo draws (pass the
panel again as ``data=``) and the lock reports that run. The rerun removes the
part of the overstatement that comes from the draws; the part that comes from
the panel itself remains - about one percentage point of power in the library's
simulations (Methodology Registry, ``Note (lock rerun)``). With one estimator
nothing was chosen and there is no rerun, so a search made of separate
one-estimator calls is not corrected. The lock is plain JSON data: write
``lock.to_json()`` to a file and read it back with
:meth:`~diff_diff.DesignLock.from_json`.

**Fit from the lock.** Once the study has run, :meth:`~diff_diff.DesignLock.fit`
builds the locked estimator, fits it on the analysis data with the locked
columns and ``fit()`` keywords, and checks the result against the lock item by
item - estimator class, every constructor parameter, ``fit()`` keywords, column
names, the fingerprint of the diagnosed periods and the realized design. It
returns a :class:`~diff_diff.DesignReconciliationResults` with the status
``match`` or ``deviates`` and the estimate in ``results``. The configuration is
the locked one by construction; what can still deviate is the realized design
(how many units, how many treated, how long the post window is) and the data in
the diagnosed periods.

**Reconcile.** :meth:`~diff_diff.DesignLock.reconcile` runs the same checks on an
analysis you fitted yourself. It compares what you declare with the lock: fitted
results do not record the parameters or columns that produced them, so a
``fit()`` keyword that was declared but not used cannot be seen. Its result
carries ``declared=True``; :meth:`~diff_diff.DesignLock.to_estimator` builds the
estimator straight from the lock for that path.

Either way, the lock does not record when it was made. ``match`` says the
analysis agrees with the lock on the checks that were run; whether the design
was fixed before the outcomes were seen depends on when you saved or registered
the lock file. A deviation is not graded, and checks whose inputs were not passed
are reported as ``not checked`` next to the status.

**Reports.** Pass the reconciliation to
:class:`~diff_diff.BusinessReport` or :class:`~diff_diff.DiagnosticReport` as
``design_reconciliation=`` and the report states the status, the deviating items
and the checks not run. It must have been made for the results being reported.

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

DesignLock
----------

.. autoclass:: diff_diff.DesignLock
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

DesignReconciliationResults
---------------------------

.. autoclass:: diff_diff.DesignReconciliationResults
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

Three estimators were compared, so locking the choice reruns it alone, with new
placebo draws:

.. code-block:: python

   from pathlib import Path

   from diff_diff import DesignLock

   lock = result.lock("cs", data=panel)     # reruns "cs" only, on the same panel
   lock.print_summary()
   Path("design_lock.json").write_text(lock.to_json())

Later, once the study has run:

.. code-block:: python

   from diff_diff import BusinessReport

   lock = DesignLock.from_json(Path("design_lock.json").read_text())
   check = lock.fit(analysis, first_treat="adopted")   # the locked analysis
   print(check.status)                      # "match" or "deviates"
   check.to_dataframe()                     # one row per checked item
   check.results.print_summary()            # the estimate

   report = BusinessReport(check.results, design_reconciliation=check)
   print(report.summary())

An analysis fitted by hand is reconciled by declaring what was used:

.. code-block:: python

   estimator = lock.to_estimator()          # the estimator exactly as locked
   fit = estimator.fit(
       analysis, outcome="sales", unit="store", time="year", first_treat="adopted"
   )
   check = lock.reconcile(
       estimator,
       outcome="sales",
       unit="store",
       time="year",
       first_treat="adopted",
       data=analysis,
       results=fit,
   )
   check.declared                           # True: the configuration was declared

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
