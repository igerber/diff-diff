Utilities
=========

Statistical utilities for parallel trends testing, robust standard errors,
and bootstrap inference.

.. module:: diff_diff.utils

Parallel Trends Testing
-----------------------

check_parallel_trends
~~~~~~~~~~~~~~~~~~~~~

Test for parallel trends using pre-treatment data.

.. autofunction:: diff_diff.check_parallel_trends

Example
^^^^^^^

.. code-block:: python

   from diff_diff import check_parallel_trends

   result = check_parallel_trends(
       data,
       outcome='y',
       time='period',
       treatment_group='treated',
       pre_periods=[0, 1, 2, 3]
   )

   print(f"t-statistic: {result['t_statistic']:.3f}")
   print(f"p-value: {result['p_value']:.3f}")

   if result['p_value'] > 0.05:
       print("Cannot reject parallel trends")
   else:
       print("Evidence against parallel trends")

check_parallel_trends_robust
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Robust parallel trends test using Wasserstein distance with permutation-based inference.

.. autofunction:: diff_diff.check_parallel_trends_robust

equivalence_test_trends
~~~~~~~~~~~~~~~~~~~~~~~

Equivalence test for parallel trends (TOST procedure).

.. autofunction:: diff_diff.equivalence_test_trends

Example
^^^^^^^

.. code-block:: python

   from diff_diff import equivalence_test_trends

   # Test if pre-trends are equivalent to zero within bounds
   result = equivalence_test_trends(
       data,
       outcome='y',
       time='period',
       treatment_group='treated',
       equivalence_margin=0.5  # Effect size bound
   )

   if result['equivalent']:
       print("Pre-trends are practically equivalent to zero")

Wild Cluster Bootstrap
----------------------

wild_bootstrap_se
~~~~~~~~~~~~~~~~~

Compute wild cluster bootstrap standard errors.

.. autofunction:: diff_diff.wild_bootstrap_se

Example
^^^^^^^

.. code-block:: python

   from diff_diff import DifferenceInDifferences, generate_did_data

   panel = generate_did_data(n_units=200, n_periods=10, treatment_effect=2.0)

   # Use wild bootstrap via the estimator's inference parameter (recommended)
   did = DifferenceInDifferences(inference='wild_bootstrap', n_bootstrap=999,
                                  cluster='unit')
   results = did.fit(panel, outcome='outcome', treatment='treated',
                     time='post')

   print(f"Bootstrap SE: {results.se:.3f}")
   print(f"Bootstrap 95% CI: [{results.conf_int[0]:.3f}, {results.conf_int[1]:.3f}]")

.. note::

   ``wild_bootstrap_se()`` is a low-level function that operates on numpy arrays
   (X, y, residuals, cluster_ids). For most users, the estimator-level
   ``inference='wild_bootstrap'`` parameter shown above is more convenient.

WildBootstrapResults
~~~~~~~~~~~~~~~~~~~~

Container for wild bootstrap results.

.. autoclass:: diff_diff.WildBootstrapResults
   :members:
   :undoc-members:
   :show-inheritance:

Weight Types
^^^^^^^^^^^^

The wild bootstrap supports several weight distributions:

- ``'rademacher'``: ±1 with equal probability (default, good general choice)
- ``'mammen'``: Two-point distribution matching higher moments
- ``'webb'``: Six-point distribution, better for few clusters

.. code-block:: python

   # Using different weight types (low-level array API)
   # wild_bootstrap_se(X, y, residuals, cluster_ids, coefficient_index, ...)
   boot_rad = wild_bootstrap_se(X, y, resid, clusters, 0, weight_type='rademacher')
   boot_webb = wild_bootstrap_se(X, y, resid, clusters, 0, weight_type='webb')
   boot_mammen = wild_bootstrap_se(X, y, resid, clusters, 0, weight_type='mammen')

Recommendation
^^^^^^^^^^^^^^

- Use ``'rademacher'`` (default) for most cases
- Use ``'webb'`` when you have fewer than 10 clusters
- The ``n_bootstrap`` should typically be at least 999 for reliable inference
