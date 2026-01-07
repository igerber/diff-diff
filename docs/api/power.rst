Power Analysis
==============

Power analysis for DiD study design.

.. module:: diff_diff.power

Overview
--------

Power analysis helps researchers design studies with adequate statistical power to detect
meaningful treatment effects. This module provides:

1. **Analytical Power Calculations**: Fast closed-form power for standard DiD designs
2. **Minimum Detectable Effect (MDE)**: Smallest effect detectable at target power
3. **Sample Size Calculations**: Required sample size for target power
4. **Simulation-Based Power**: Monte Carlo power for any DiD estimator

PowerAnalysis
-------------

Main class for analytical power calculations.

.. autoclass:: diff_diff.PowerAnalysis
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::

      ~PowerAnalysis.compute_power
      ~PowerAnalysis.compute_mde
      ~PowerAnalysis.compute_sample_size

Example
~~~~~~~

.. code-block:: python

   from diff_diff import PowerAnalysis

   # Create power analysis object
   pa = PowerAnalysis(
       effect_size=0.5,
       n_treated=100,
       n_control=100,
       n_pre=4,
       n_post=4,
       sigma=1.0,
       rho=0.5,  # Within-unit correlation
       alpha=0.05
   )

   # Compute power
   power = pa.compute_power()
   print(f"Power: {power:.2%}")

   # Compute MDE at 80% power
   mde = pa.compute_mde(power=0.80)
   print(f"MDE: {mde:.3f}")

   # Required sample size
   n = pa.compute_sample_size(power=0.80)
   print(f"Required N per group: {n}")

PowerResults
------------

Results from power analysis.

.. autoclass:: diff_diff.PowerResults
   :members:
   :undoc-members:
   :show-inheritance:

SimulationPowerResults
----------------------

Results from simulation-based power analysis.

.. autoclass:: diff_diff.SimulationPowerResults
   :members:
   :undoc-members:
   :show-inheritance:

Convenience Functions
---------------------

compute_power
~~~~~~~~~~~~~

Quick power computation.

.. autofunction:: diff_diff.compute_power

compute_mde
~~~~~~~~~~~

Compute minimum detectable effect.

.. autofunction:: diff_diff.compute_mde

compute_sample_size
~~~~~~~~~~~~~~~~~~~

Compute required sample size.

.. autofunction:: diff_diff.compute_sample_size

simulate_power
~~~~~~~~~~~~~~

Simulation-based power for any DiD estimator.

.. autofunction:: diff_diff.simulate_power

Complete Example
----------------

.. code-block:: python

   from diff_diff import (
       PowerAnalysis,
       compute_mde,
       simulate_power,
       DifferenceInDifferences,
       plot_power_curve,
   )

   # Quick MDE calculation
   mde = compute_mde(
       n_treated=50,
       n_control=50,
       n_pre=4,
       n_post=4,
       sigma=1.0,
       rho=0.5,
       power=0.80,
       alpha=0.05
   )
   print(f"MDE: {mde:.3f}")

   # Simulation-based power for DiD estimator
   sim_results = simulate_power(
       estimator=DifferenceInDifferences(),
       effect_size=0.5,
       n_treated=100,
       n_control=100,
       n_periods=8,
       treatment_start=4,
       sigma=1.0,
       n_simulations=1000
   )
   print(f"Simulated power: {sim_results.power:.2%}")

   # Power curve
   pa = PowerAnalysis(n_treated=100, n_control=100, n_pre=4, n_post=4, sigma=1.0)
   fig = plot_power_curve(pa, effect_range=(0, 1), n_points=50)
   fig.savefig('power_curve.png')

See Also
--------

- :doc:`pretrends` - Pre-trends power analysis (Roth 2022)
