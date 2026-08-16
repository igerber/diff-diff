API Reference
=============

This section provides complete API documentation for all diff-diff modules.

Estimators
----------

Core causal-inference estimator classes - the DiD family plus synthetic control,
regression discontinuity, and the Goodman-Bacon decomposition diagnostic:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   diff_diff.DifferenceInDifferences
   diff_diff.TwoWayFixedEffects
   diff_diff.MultiPeriodDiD
   diff_diff.SyntheticDiD
   diff_diff.CallawaySantAnna
   diff_diff.ChaisemartinDHaultfoeuille
   diff_diff.SunAbraham
   diff_diff.ImputationDiD
   diff_diff.StackedDiD
   diff_diff.TripleDifference
   diff_diff.TROP
   diff_diff.SyntheticControl
   diff_diff.ContinuousDiD
   diff_diff.HeterogeneousAdoptionDiD
   diff_diff.EfficientDiD
   diff_diff.TwoStageDiD
   diff_diff.SpilloverDiD
   diff_diff.WooldridgeDiD
   diff_diff.LPDiD
   diff_diff.ChangesInChanges
   diff_diff.QDiD
   diff_diff.BaconDecomposition
   diff_diff.StaggeredTripleDifference
   diff_diff.RegressionDiscontinuity

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
   diff_diff.ChaisemartinDHaultfoeuilleResults
   diff_diff.DCDHBootstrapResults
   diff_diff.SunAbrahamResults
   diff_diff.SABootstrapResults
   diff_diff.ImputationDiDResults
   diff_diff.ImputationBootstrapResults
   diff_diff.TripleDifferenceResults
   diff_diff.StackedDiDResults
   diff_diff.TROPResults
   diff_diff.SyntheticControlResults
   diff_diff.ContinuousDiDResults
   diff_diff.DoseResponseCurve
   diff_diff.HeterogeneousAdoptionDiDResults
   diff_diff.HeterogeneousAdoptionDiDEventStudyResults
   diff_diff.EfficientDiDResults
   diff_diff.EDiDBootstrapResults
   diff_diff.TwoStageDiDResults
   diff_diff.TwoStageBootstrapResults
   diff_diff.SpilloverDiDResults
   diff_diff.BaconDecompositionResults
   diff_diff.wooldridge_results.WooldridgeDiDResults
   diff_diff.lpdid_results.LPDiDResults
   diff_diff.changes_in_changes_results.ChangesInChangesResults
   diff_diff.Comparison2x2
   diff_diff.StaggeredTripleDiffResults
   diff_diff.TWFEWeightsResult
   diff_diff.RegressionDiscontinuityResults
   diff_diff.RDPlotResult
   diff_diff.RDDensityTestResult
   diff_diff.BaseResults
   diff_diff.Diagnostic
   diff_diff.EventStudyResults
   diff_diff.AggregationResult

Visualization
-------------

Plotting functions and plot builders:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   diff_diff.plot_event_study
   diff_diff.plot_group_effects
   diff_diff.plot_sensitivity
   diff_diff.plot_honest_event_study
   diff_diff.RDPlot
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
   diff_diff.RDDensityTest

Panel Profiling
---------------

Pre-fit description of panel structure for estimator selection. The
:class:`~diff_diff.PanelProfile` return type and its supporting dataclasses
are documented in :doc:`profile`.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   diff_diff.profile_panel
   diff_diff.PanelProfile
   diff_diff.OutcomeShape
   diff_diff.TreatmentDoseShape
   diff_diff.Alert

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

HAD Pretest Workflow
--------------------

Companion pretest battery for ``HeterogeneousAdoptionDiD`` implementing the
Section 4 QUG / Stute / Yatchew tests from de Chaisemartin, Ciccia,
D'Haultfœuille & Knau (2026), plus a unified report wrapper:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   diff_diff.HADPretestReport
   diff_diff.QUGTestResults
   diff_diff.StuteTestResults
   diff_diff.YatchewTestResults
   diff_diff.StuteJointResult

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
   diff_diff.SimulationMDEResults
   diff_diff.SimulationSampleSizeResults
   diff_diff.compute_power
   diff_diff.compute_mde
   diff_diff.compute_sample_size
   diff_diff.simulate_power
   diff_diff.simulate_mde
   diff_diff.simulate_sample_size

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

Reporting
---------

Stakeholder-facing report and diagnostic battery wrappers around fitted
result objects:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   diff_diff.BusinessReport
   diff_diff.BusinessContext
   diff_diff.DiagnosticReport
   diff_diff.DiagnosticReportResults

MMM Calibration Export
----------------------

Convert experiment results into Marketing Mix Model calibration inputs
(PyMC-Marketing lift tests, Google Meridian ROI priors):

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   diff_diff.to_pymc_marketing_lift_test
   diff_diff.to_meridian_roi_prior
   diff_diff.meridian_calibration_mask
   diff_diff.MeridianROIPrior

Boundary Local-Linear Estimators
--------------------------------

Calonico-Cattaneo-Farrell (2018) MSE-optimal bandwidth selector and
Calonico-Cattaneo-Titiunik (2014) robust-bias-corrected local-linear fit
used by ``HeterogeneousAdoptionDiD``'s continuous-dose fit paths
(``continuous_at_zero`` and ``continuous_near_d_lower``):

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   diff_diff.LocalLinearFit
   diff_diff.BandwidthResult
   diff_diff.BiasCorrectedFit

Data Preparation
----------------

Utilities for preparing DiD data:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   diff_diff.generate_did_data
   diff_diff.generate_continuous_did_data
   diff_diff.generate_staggered_data
   diff_diff.generate_event_study_data
   diff_diff.generate_ddd_data
   diff_diff.generate_factor_data
   diff_diff.generate_panel_data
   diff_diff.make_treatment_indicator
   diff_diff.make_post_indicator
   diff_diff.wide_to_long
   diff_diff.balance_panel
   diff_diff.validate_did_data
   diff_diff.summarize_did_data
   diff_diff.create_event_time
   diff_diff.aggregate_to_cohorts
   diff_diff.rank_control_units

Datasets
--------

Built-in datasets for examples and testing:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   diff_diff.load_card_krueger
   diff_diff.load_castle_doctrine
   diff_diff.load_divorce_laws
   diff_diff.load_mpdta
   diff_diff.load_prop99
   diff_diff.load_walmart
   diff_diff.load_dataset
   diff_diff.list_datasets
   diff_diff.clear_cache

Module Documentation
--------------------

Estimators
~~~~~~~~~~

.. toctree::
   :maxdepth: 2

   estimators
   staggered
   chaisemartin_dhaultfoeuille
   imputation
   stacked_did
   triple_diff
   trop
   synthetic_control
   continuous_did
   had
   regression_discontinuity
   efficient_did
   two_stage
   spillover
   wooldridge_etwfe
   lpdid
   changes_in_changes
   bacon

Infrastructure
~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 2

   local_linear

Pre-Fit Profiling
~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 2

   profile

Diagnostics & Inference
~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 2

   diagnostics
   honest_did
   power
   pretrends

Reporting
~~~~~~~~~

.. toctree::
   :maxdepth: 2

   business_report
   diagnostic_report
   mmm

Results & Visualization
~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 2

   results
   visualization

Data & Utilities
~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 2

   utils
   prep
   datasets
