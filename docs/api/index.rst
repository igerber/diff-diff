API Reference
=============

This section provides complete API documentation for all diff-diff modules.

Estimators
----------

Core estimator classes for DiD analysis:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   diff_diff.DifferenceInDifferences
   diff_diff.TwoWayFixedEffects
   diff_diff.MultiPeriodDiD
   diff_diff.SyntheticDiD
   diff_diff.CallawaySantAnna
   diff_diff.SunAbraham
   diff_diff.TripleDifference

Results Classes
---------------

Result containers returned by estimators:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   diff_diff.DiDResults
   diff_diff.MultiPeriodDiDResults
   diff_diff.SyntheticDiDResults
   diff_diff.PeriodEffect
   diff_diff.CallawaySantAnnaResults
   diff_diff.CSBootstrapResults
   diff_diff.GroupTimeEffect
   diff_diff.SunAbrahamResults
   diff_diff.SABootstrapResults
   diff_diff.TripleDifferenceResults

Visualization
-------------

Plotting functions for results:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   diff_diff.plot_event_study
   diff_diff.plot_group_effects
   diff_diff.plot_sensitivity
   diff_diff.plot_honest_event_study
   diff_diff.plot_bacon
   diff_diff.plot_power_curve
   diff_diff.plot_pretrends_power

Diagnostics
-----------

Placebo tests and model diagnostics:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   diff_diff.run_placebo_test
   diff_diff.placebo_timing_test
   diff_diff.placebo_group_test
   diff_diff.permutation_test
   diff_diff.leave_one_out_test
   diff_diff.run_all_placebo_tests
   diff_diff.PlaceboTestResults

Sensitivity Analysis
--------------------

Honest DiD for robust inference:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   diff_diff.HonestDiD
   diff_diff.HonestDiDResults
   diff_diff.SensitivityResults
   diff_diff.DeltaSD
   diff_diff.DeltaRM
   diff_diff.DeltaSDRM
   diff_diff.compute_honest_did
   diff_diff.sensitivity_plot

Parallel Trends Testing
-----------------------

Testing the parallel trends assumption:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   diff_diff.check_parallel_trends
   diff_diff.check_parallel_trends_robust
   diff_diff.equivalence_test_trends

Bootstrap Inference
-------------------

Wild cluster bootstrap for valid inference:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   diff_diff.wild_bootstrap_se
   diff_diff.WildBootstrapResults

Power Analysis
--------------

Power analysis for study design:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   diff_diff.PowerAnalysis
   diff_diff.PowerResults
   diff_diff.SimulationPowerResults
   diff_diff.compute_power
   diff_diff.compute_mde
   diff_diff.compute_sample_size
   diff_diff.simulate_power

Pre-Trends Power Analysis
-------------------------

Power analysis for pre-trends tests (Roth 2022):

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   diff_diff.PreTrendsPower
   diff_diff.PreTrendsPowerResults
   diff_diff.PreTrendsPowerCurve
   diff_diff.compute_pretrends_power
   diff_diff.compute_mdv

Data Preparation
----------------

Utilities for preparing DiD data:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   diff_diff.generate_did_data
   diff_diff.make_treatment_indicator
   diff_diff.make_post_indicator
   diff_diff.wide_to_long
   diff_diff.balance_panel
   diff_diff.validate_did_data
   diff_diff.summarize_did_data
   diff_diff.create_event_time
   diff_diff.aggregate_to_cohorts
   diff_diff.rank_control_units

Module Documentation
--------------------

Detailed documentation by module:

.. toctree::
   :maxdepth: 2

   estimators
   staggered
   triple_diff
   results
   visualization
   diagnostics
   honest_did
   power
   pretrends
   utils
   prep
