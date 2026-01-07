Pre-Trends Power Analysis
=========================

Power analysis for pre-trends tests (Roth 2022).

.. module:: diff_diff.pretrends

Overview
--------

Passing a pre-trends test does not guarantee that parallel trends holds. Roth (2022)
shows that pre-trends tests are often underpowered to detect economically meaningful
violations of parallel trends. This module provides tools to assess:

1. **Power**: The probability of rejecting the null (no pre-trends) given a violation
2. **Minimum Detectable Violation (MDV)**: The smallest violation detectable at target power

Key insights from Roth (2022):

- Pre-trends tests are joint tests that pre-period coefficients equal zero
- Standard pre-trends tests often have low power against linear trends
- A "passed" pre-trends test may simply reflect lack of statistical power
- MDV provides the minimum violation the test could have detected

PreTrendsPower
--------------

Main class for pre-trends power analysis.

.. autoclass:: diff_diff.PreTrendsPower
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::

      ~PreTrendsPower.fit
      ~PreTrendsPower.power_curve
      ~PreTrendsPower.sensitivity_to_honest_did

Example
~~~~~~~

.. code-block:: python

   from diff_diff import MultiPeriodDiD, PreTrendsPower

   # First fit an event study
   model = MultiPeriodDiD(reference_period=-1)
   results = model.fit(data, outcome='y', treated='treated',
                       time='period', unit='unit_id', treatment_start=5)

   # Compute pre-trends power for linear violations
   pt = PreTrendsPower(alpha=0.05, power=0.80, violation_type='linear')
   pt_results = pt.fit(results)

   print(f"MDV: {pt_results.mdv:.3f}")
   print(f"Power at M=1: {pt_results.power_at_m:.2%}")

PreTrendsPowerResults
---------------------

Results from pre-trends power analysis.

.. autoclass:: diff_diff.PreTrendsPowerResults
   :members:
   :undoc-members:
   :show-inheritance:

PreTrendsPowerCurve
-------------------

Power curve across violation magnitudes.

.. autoclass:: diff_diff.PreTrendsPowerCurve
   :members:
   :undoc-members:
   :show-inheritance:

Convenience Functions
---------------------

compute_pretrends_power
~~~~~~~~~~~~~~~~~~~~~~~

Quick computation of pre-trends power.

.. autofunction:: diff_diff.compute_pretrends_power

compute_mdv
~~~~~~~~~~~

Compute minimum detectable violation.

.. autofunction:: diff_diff.compute_mdv

Violation Types
---------------

The module supports several types of pre-trends violations:

**linear**
   Linear trend violations where each pre-period differs from the reference by
   an amount proportional to distance. ``delta[t] = M * t`` for pre-periods.

**constant**
   Constant violations where all pre-periods have the same deviation.
   ``delta[t] = M`` for all pre-periods.

**last_period**
   Only the period immediately before treatment is violated.
   ``delta[-1] = M``, all other pre-periods are zero.

**custom**
   User-specified violation pattern via the ``custom_delta`` parameter.

Complete Example
----------------

.. code-block:: python

   import numpy as np
   from diff_diff import (
       MultiPeriodDiD,
       PreTrendsPower,
       compute_mdv,
       plot_pretrends_power,
   )

   # Fit event study
   model = MultiPeriodDiD(reference_period=-1)
   results = model.fit(data, outcome='y', treated='treated',
                       time='period', unit='unit_id', treatment_start=5)

   # Compute MDV
   mdv = compute_mdv(results, alpha=0.05, power=0.80)
   print(f"Minimum Detectable Violation: {mdv:.3f}")

   # Power curve analysis
   pt = PreTrendsPower(alpha=0.05, violation_type='linear')
   curve = pt.power_curve(results, violation_range=(0, 5), n_points=50)

   # Plot power curve
   fig = plot_pretrends_power(curve, show_mdv=True, target_power=0.80)
   fig.savefig('pretrends_power.png')

   # Integration with HonestDiD
   sensitivity = pt.sensitivity_to_honest_did(
       results,
       honest_method='smoothness',
       M_grid=np.linspace(0, mdv, 21)
   )

References
----------

- Roth, J. (2022). Pretest with Caution: Event-Study Estimates after Testing
  for Parallel Trends. *American Economic Review: Insights*, 4(3), 305-322.

- R package: `pretrends <https://github.com/jonathandroth/pretrends>`_

See Also
--------

- :doc:`honest_did` - Sensitivity analysis under parallel trends violations
- :doc:`utils` - Standard parallel trends tests
