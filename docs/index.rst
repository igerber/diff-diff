.. meta::
   :description: diff-diff — Python library for Difference-in-Differences causal inference. Callaway-Sant'Anna, Synthetic DiD, Honest DiD, event studies, parallel trends. sklearn-like API, validated against R.
   :keywords: difference-in-differences, python, causal inference, DiD, econometrics, treatment effects, staggered adoption, event study

diff-diff: Difference-in-Differences in Python
==============================================

**diff-diff** is a Python library for Difference-in-Differences (DiD) causal inference analysis.
It provides sklearn-like estimators with statsmodels-style output for econometric analysis.

.. code-block:: python

   from diff_diff import DifferenceInDifferences

   # Fit a basic DiD model
   did = DifferenceInDifferences()
   results = did.fit(data, outcome='y', treatment='treated', time='post')
   print(results.summary())

Key Features
------------

- **13+ Estimators**: Basic DiD, TWFE, Event Study, Synthetic DiD, plus modern staggered estimators (Callaway-Sant'Anna, Sun-Abraham, Imputation, Two-Stage, Stacked DiD), advanced methods (TROP, Continuous DiD, Efficient DiD, Triple Difference), and Bacon Decomposition diagnostics
- **Modern Inference**: Robust standard errors, cluster-robust SEs, wild cluster bootstrap, and multiplier bootstrap
- **Assumption Testing**: Parallel trends tests, placebo tests, Bacon decomposition, and comprehensive diagnostics
- **Sensitivity Analysis**: Honest DiD (Rambachan & Roth 2023) for robust inference under parallel trends violations
- **Built-in Datasets**: Real-world datasets from published studies (Card & Krueger, Castle Doctrine, and more)
- **High Performance**: Optional Rust backend for compute-intensive estimators like Synthetic DiD and TROP
- **Publication-Ready Output**: Summary tables, event study plots, and sensitivity analysis figures

Installation
------------

.. code-block:: bash

   pip install diff-diff

For development:

.. code-block:: bash

   pip install diff-diff[dev]

Quick Links
-----------

- :doc:`practitioner_getting_started` - Measuring campaign impact? Start here
- :doc:`practitioner_decision_tree` - Which method fits your business problem?
- :doc:`quickstart` - Installation and your first DiD analysis
- :doc:`choosing_estimator` - Which estimator should I use?
- :func:`~diff_diff.aggregate_survey` - Have BRFSS/ACS/CPS microdata? Bridge it to a geographic panel for DiD
- :doc:`tutorials/01_basic_did` - Hands-on basic tutorial
- :doc:`troubleshooting` - Common issues and solutions
- :doc:`r_comparison` - Coming from R?
- :doc:`api/index` - Full API reference

Explore the Documentation
-------------------------

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Getting Started
      :link: getting_started
      :link-type: doc

      Install, run your first DiD analysis, and pick the right estimator
      for your design.

   .. grid-item-card:: Practitioner Guide
      :link: practitioners
      :link-type: doc

      Measuring campaign impact? A business-first path through DiD, no
      econometrics background required.

   .. grid-item-card:: Tutorials
      :link: tutorials/index
      :link-type: doc

      28 hands-on notebooks, from basic 2x2 DiD to survey-weighted and
      spillover-aware designs.

   .. grid-item-card:: User Guide
      :link: user_guide
      :link-type: doc

      References, R and Python comparisons, benchmarks, and the
      methodology registry.

   .. grid-item-card:: API Reference
      :link: api/index
      :link-type: doc

      Complete reference for all estimators, results classes, diagnostics,
      and utilities.

.. toctree::
   :maxdepth: 2
   :hidden:

   Getting Started <getting_started>
   Practitioner Guide <practitioners>
   Tutorials <tutorials/index>
   User Guide <user_guide>
   API Reference <api/index>

What is Difference-in-Differences?
----------------------------------

Difference-in-Differences (DiD) is a quasi-experimental research design that estimates
causal treatment effects by comparing outcome changes over time between treated and
control groups. It is one of the most widely used methods in applied economics,
public policy evaluation, and social science research.

Why diff-diff?
--------------

- **Complete method coverage**: 13+ estimators from basic 2x2 DiD to cutting-edge methods like Efficient DiD (Chen et al. 2025) and TROP (Athey et al. 2025)
- **Familiar API**: sklearn-like ``fit()`` interface — if you know scikit-learn, you know diff-diff
- **Modern staggered methods**: Callaway-Sant'Anna, Sun-Abraham, Imputation DiD, Two-Stage DiD, and Stacked DiD handle heterogeneous treatment timing correctly
- **Robust inference**: Heteroskedasticity-robust, cluster-robust, wild cluster bootstrap, and multiplier bootstrap
- **Sensitivity analysis**: Honest DiD (Rambachan & Roth 2023) for robust inference under parallel trends violations
- **Validated against R**: Benchmarked against ``did``, ``synthdid``, and ``fixest`` — see :doc:`benchmarks`
- **No heavy dependencies**: Only numpy, pandas, and scipy

Supported Estimators
--------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Estimator
     - Description
   * - :class:`~diff_diff.DifferenceInDifferences`
     - Basic 2x2 DiD with robust/clustered standard errors
   * - :class:`~diff_diff.TwoWayFixedEffects`
     - Panel data with unit and time fixed effects
   * - :class:`~diff_diff.MultiPeriodDiD`
     - Event study with period-specific treatment effects
   * - :class:`~diff_diff.CallawaySantAnna`
     - Callaway & Sant'Anna (2021) for staggered adoption
   * - :class:`~diff_diff.SunAbraham`
     - Sun & Abraham (2021) interaction-weighted estimator
   * - :class:`~diff_diff.ImputationDiD`
     - Borusyak, Jaravel & Spiess (2024) imputation estimator
   * - :class:`~diff_diff.TwoStageDiD`
     - Gardner (2022) two-stage residualized estimator
   * - :class:`~diff_diff.SpilloverDiD`
     - Butts (2021) ring-indicator spillover-aware DiD
   * - :class:`~diff_diff.SyntheticDiD`
     - Synthetic DiD combining DiD and synthetic control
   * - :class:`~diff_diff.StackedDiD`
     - Wing, Freedman & Hollingsworth (2024) stacked DiD
   * - :class:`~diff_diff.EfficientDiD`
     - Chen, Sant'Anna & Xie (2025) efficient DiD
   * - :class:`~diff_diff.TripleDifference`
     - Triple difference (DDD) estimator
   * - :class:`~diff_diff.ContinuousDiD`
     - Continuous treatment DiD
   * - :class:`~diff_diff.TROP`
     - Triply Robust Panel with factor model adjustment (Athey et al. 2025)
   * - :class:`~diff_diff.BaconDecomposition`
     - Goodman-Bacon decomposition diagnostics

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
