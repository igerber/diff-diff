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
   diff_diff.GroupTimeEffect

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
   results
   visualization
   diagnostics
   honest_did
   utils
   prep
