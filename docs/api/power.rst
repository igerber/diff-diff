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

      ~PowerAnalysis.power
      ~PowerAnalysis.mde
      ~PowerAnalysis.sample_size
      ~PowerAnalysis.power_curve
      ~PowerAnalysis.sample_size_curve

Example
~~~~~~~

.. code-block:: python

   from diff_diff import PowerAnalysis

   pa = PowerAnalysis(alpha=0.05, power=0.80)

   # Compute power
   result = pa.power(effect_size=0.5, n_treated=100, n_control=100, sigma=1.0)
   print(f"Power: {result.power:.2%}")

   # Compute MDE at 80% power
   result = pa.mde(n_treated=100, n_control=100, sigma=1.0)
   print(f"MDE: {result.mde:.3f}")

   # Required sample size
   result = pa.sample_size(effect_size=0.5, sigma=1.0)
   print(f"Required N: {result.required_n}")

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

SimulationMDEResults
--------------------

Results from simulation-based MDE search.

.. autoclass:: diff_diff.SimulationMDEResults
   :members:
   :undoc-members:
   :show-inheritance:

SimulationSampleSizeResults
---------------------------

Results from simulation-based sample size search.

.. autoclass:: diff_diff.SimulationSampleSizeResults
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

simulate_mde
~~~~~~~~~~~~~

Simulation-based MDE for any DiD estimator.

.. autofunction:: diff_diff.simulate_mde

simulate_sample_size
~~~~~~~~~~~~~~~~~~~~

Simulation-based sample size for any DiD estimator.

.. autofunction:: diff_diff.simulate_sample_size

Complete Example
----------------

.. code-block:: python

   from diff_diff import (
       PowerAnalysis,
       compute_mde,
       simulate_power,
       simulate_mde,
       DifferenceInDifferences,
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
       treatment_effect=5.0,
       n_units=100,
       n_periods=4,
       treatment_period=2,
       sigma=1.0,
       n_simulations=20,
   )
   print(f"Simulated power: {sim_results.power:.2%}")

   # Simulation-based MDE
   mde_results = simulate_mde(
       estimator=DifferenceInDifferences(),
       n_units=100,
       n_simulations=10,
       max_steps=5,
   )
   print(f"Simulated MDE: {mde_results.mde:.3f}")

See Also
--------

- :doc:`pretrends` - Pre-trends power analysis (Roth 2022)
