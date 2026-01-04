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

HonestDiD
---------

Main class for computing honest bounds and confidence intervals.

.. autoclass:: diff_diff.HonestDiD
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::

      ~HonestDiD.fit
      ~HonestDiD.sensitivity_analysis
      ~HonestDiD.breakdown_value

Example
~~~~~~~

.. code-block:: python

   from diff_diff import MultiPeriodDiD, HonestDiD, DeltaRM

   # First fit an event study
   model = MultiPeriodDiD(reference_period=-1)
   results = model.fit(data, outcome='y', treated='treated',
                       time='period', unit='unit_id', treatment_start=5)

   # Compute bounds under relative magnitudes restriction
   honest = HonestDiD(delta=DeltaRM(M_bar=1.0))
   bounds = honest.fit(results)

   print(f"Original CI: [{results.att - 1.96*results.se:.3f}, "
         f"{results.att + 1.96*results.se:.3f}]")
   print(f"Robust CI: [{bounds.robust_ci[0]:.3f}, {bounds.robust_ci[1]:.3f}]")

HonestDiDResults
----------------

Results from HonestDiD estimation.

.. autoclass:: diff_diff.HonestDiDResults
   :members:
   :undoc-members:
   :show-inheritance:

SensitivityResults
------------------

Results from sensitivity analysis over a grid of M values.

.. autoclass:: diff_diff.SensitivityResults
   :members:
   :undoc-members:
   :show-inheritance:

Restriction Classes
-------------------

DeltaSD
~~~~~~~

Smoothness restriction class.

.. autoclass:: diff_diff.DeltaSD
   :members:
   :undoc-members:
   :show-inheritance:

DeltaRM
~~~~~~~

Relative magnitudes restriction class.

.. autoclass:: diff_diff.DeltaRM
   :members:
   :undoc-members:
   :show-inheritance:

DeltaSDRM
~~~~~~~~~

Combined smoothness and relative magnitudes restriction.

.. autoclass:: diff_diff.DeltaSDRM
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
       MultiPeriodDiD,
       HonestDiD,
       DeltaRM,
       DeltaSD,
       plot_sensitivity,
       plot_honest_event_study,
   )

   # Fit event study
   model = MultiPeriodDiD(reference_period=-1)
   results = model.fit(data, outcome='y', treated='treated',
                       time='period', unit='unit_id', treatment_start=5)

   # Sensitivity analysis under relative magnitudes
   honest_rm = HonestDiD(delta=DeltaRM(M_bar=1.0))
   sensitivity_rm = honest_rm.sensitivity_analysis(
       results,
       M_grid=np.linspace(0, 2, 21)
   )

   # Find breakdown value
   breakdown = honest_rm.breakdown_value(results)
   print(f"Breakdown M̄: {breakdown:.3f}")

   # Plot sensitivity
   fig1 = plot_sensitivity(sensitivity_rm)
   fig1.savefig('sensitivity_rm.png')

   # Event study with honest CIs
   bounds = honest_rm.fit(results)
   fig2 = plot_honest_event_study(results, bounds)
   fig2.savefig('honest_event_study.png')

References
----------

- Rambachan, A., & Roth, J. (2023). A More Credible Approach to Parallel Trends.
  *Review of Economic Studies*, 90(5), 2555-2591.

- R package: `HonestDiD <https://github.com/asheshrambachan/HonestDiD>`_
