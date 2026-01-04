diff-diff: Difference-in-Differences in Python
==============================================

**diff-diff** is a Python library for Difference-in-Differences (DiD) causal inference analysis.
It provides sklearn-like estimators with statsmodels-style output for econometric analysis.

.. code-block:: python

   from diff_diff import DifferenceInDifferences

   # Fit a basic DiD model
   did = DifferenceInDifferences()
   results = did.fit(data, outcome='y', treated='treated', post='post')
   print(results.summary())

Key Features
------------

- **Multiple Estimators**: Basic DiD, Two-Way Fixed Effects, Multi-Period Event Studies, Synthetic DiD, and Callaway-Sant'Anna for staggered adoption
- **Modern Inference**: Robust standard errors, cluster-robust SEs, and wild cluster bootstrap
- **Assumption Testing**: Parallel trends tests, placebo tests, and comprehensive diagnostics
- **Sensitivity Analysis**: Honest DiD (Rambachan & Roth 2023) for robust inference under parallel trends violations
- **Publication-Ready Output**: Summary tables and event study plots

Installation
------------

.. code-block:: bash

   pip install diff-diff

For development:

.. code-block:: bash

   pip install diff-diff[dev]

Quick Links
-----------

- :doc:`quickstart` - Get started with basic examples
- :doc:`choosing_estimator` - Which estimator should I use?
- :doc:`r_comparison` - Comparison with R packages
- :doc:`python_comparison` - Comparison with Python packages
- :doc:`api/index` - Full API reference

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   quickstart
   choosing_estimator
   r_comparison
   python_comparison

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/index
   api/estimators
   api/staggered
   api/results
   api/visualization
   api/diagnostics
   api/honest_did
   api/utils
   api/prep

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
