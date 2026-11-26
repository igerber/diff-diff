Honest DiD
==========

Sensitivity analysis for violations of parallel trends (Rambachan & Roth 2023).

.. module:: diff_diff.honest_did

Overview
--------

The Honest DiD approach provides inference that is robust to violations of the
parallel trends assumption. Instead of assuming parallel trends holds exactly,
it bounds the violation and provides confidence intervals that account for
potential deviations.

Two main restriction types are supported:

1. **Relative Magnitudes (ΔRM)**: Post-treatment violations are bounded by
   M̄ times the maximum pre-treatment violation.

2. **Smoothness (ΔSD)**: Bounds on the second differences of the trend violations,
   restricting how much the violation can change between periods.

**Before the study has run.** :meth:`~diff_diff.HonestDiD.power_by_M` plans the
sensitivity analysis for a design that has no outcome yet. Give it a
:func:`~diff_diff.diagnose_design` result run with ``event_study=True`` and it
runs the analysis on the event study of every no-effect placebo draw, then
reports, per ``M``: how often the robust interval excluded zero although no
effect was present, its mean width, the minimum detectable effect, and the
power at an effect size you name. Nothing is approximated by a formula. Rows on
which the interval excludes zero too often are shown with that rate and without
planned numbers. ``fit``, ``sensitivity_analysis`` and ``breakdown_value``
describe an estimate and refuse a placebo event study. See
:doc:`design` and the ``HonestDiD`` section of the methodology registry.

HonestDiD
---------

Main class for computing honest bounds and confidence intervals.

.. autoclass:: diff_diff.HonestDiD
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   .. rubric:: Methods

   .. autosummary::

      ~HonestDiD.fit
      ~HonestDiD.sensitivity_analysis
      ~HonestDiD.breakdown_value
      ~HonestDiD.power_by_M

Example
~~~~~~~

.. code-block:: python

   from diff_diff import TwoWayFixedEffects, HonestDiD

   # First fit an event study (TwoWayFixedEffects event-study mode;
   # HonestDiD also accepts the surface directly)
   model = TwoWayFixedEffects()
   results = model.fit(data, outcome='y', treatment='treated',
                       unit='unit_id', event_study=True, time='period',
                       post_periods=[5, 6, 7, 8, 9], reference_period=4)

   # Compute bounds under relative magnitudes restriction
   honest = HonestDiD(method='relative_magnitude', M=1.0)
   bounds = honest.fit(results)

   print(f"Original estimate: {bounds.original_estimate:.3f}")
   print(f"Robust CI: [{bounds.ci_lb:.3f}, {bounds.ci_ub:.3f}]")

HonestDiDResults
----------------

Results from HonestDiD estimation.

.. autoclass:: diff_diff.HonestDiDResults
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

SensitivityResults
------------------

Results from sensitivity analysis over a grid of M values.

.. autoclass:: diff_diff.SensitivityResults
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Restriction Classes
-------------------

DeltaSD
~~~~~~~

Smoothness restriction class.

.. autoclass:: diff_diff.DeltaSD
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

DeltaRM
~~~~~~~

Relative magnitudes restriction class.

.. autoclass:: diff_diff.DeltaRM
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

DeltaSDRM
~~~~~~~~~

Combined smoothness and relative magnitudes restriction.

.. autoclass:: diff_diff.DeltaSDRM
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Convenience Functions
---------------------

compute_honest_did
~~~~~~~~~~~~~~~~~~

Quick computation of honest bounds.

.. autofunction:: diff_diff.compute_honest_did

sensitivity_plot
~~~~~~~~~~~~~~~~

Convenience function for sensitivity visualization.

.. autofunction:: diff_diff.sensitivity_plot

Complete Example
----------------

.. code-block:: python

   import numpy as np
   from diff_diff import (
       TwoWayFixedEffects,
       HonestDiD,
       plot_sensitivity,
       plot_honest_event_study,
   )

   # Fit event study (TwoWayFixedEffects event-study mode)
   model = TwoWayFixedEffects()
   results = model.fit(data, outcome='y', treatment='treated',
                       unit='unit_id', event_study=True, time='period',
                       post_periods=[5, 6, 7, 8, 9], reference_period=4)

   # Sensitivity analysis under relative magnitudes
   honest_rm = HonestDiD(method='relative_magnitude', M=1.0)
   sensitivity_rm = honest_rm.sensitivity_analysis(
       results,
       M_grid=np.linspace(0, 2, 21).tolist()
   )

   # Find breakdown value
   breakdown = honest_rm.breakdown_value(results)
   print(f"Breakdown M̄: {breakdown}")

   # Plot sensitivity
   ax1 = plot_sensitivity(sensitivity_rm)
   ax1.figure.savefig('sensitivity_rm.png')

   # Event study with honest CIs
   bounds = honest_rm.fit(results)
   ax2 = plot_honest_event_study(bounds)
   ax2.figure.savefig('honest_event_study.png')

References
----------

- Rambachan, A., & Roth, J. (2023). A More Credible Approach to Parallel Trends.
  *Review of Economic Studies*, 90(5), 2555-2591.

- R package: `HonestDiD <https://github.com/asheshrambachan/HonestDiD>`_
