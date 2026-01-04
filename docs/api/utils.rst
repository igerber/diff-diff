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
       unit='unit_id',
       time='period',
       treated='treated',
       pre_periods=4
   )

   print(f"F-statistic: {result['f_stat']:.3f}")
   print(f"p-value: {result['p_value']:.3f}")

   if result['p_value'] > 0.05:
       print("Cannot reject parallel trends")
   else:
       print("Evidence against parallel trends")

check_parallel_trends_robust
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Robust parallel trends test with heteroskedasticity-consistent standard errors.

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
       unit='unit_id',
       time='period',
       treated='treated',
       equivalence_bound=0.5  # Effect size bound
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

   from diff_diff import DifferenceInDifferences, wild_bootstrap_se

   # Fit model
   did = DifferenceInDifferences()
   results = did.fit(data, outcome='y', treated='treated', post='post')

   # Bootstrap standard errors
   boot_results = wild_bootstrap_se(
       data,
       outcome='y',
       treated='treated',
       post='post',
       cluster='unit_id',
       n_bootstrap=999,
       weight_type='rademacher'
   )

   print(f"Bootstrap SE: {boot_results.se:.3f}")
   print(f"Bootstrap 95% CI: [{boot_results.ci[0]:.3f}, {boot_results.ci[1]:.3f}]")

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

   # Using different weight types
   boot_rad = wild_bootstrap_se(data, ..., weight_type='rademacher')
   boot_webb = wild_bootstrap_se(data, ..., weight_type='webb')
   boot_mammen = wild_bootstrap_se(data, ..., weight_type='mammen')

Recommendation
^^^^^^^^^^^^^^

- Use ``'rademacher'`` (default) for most cases
- Use ``'webb'`` when you have fewer than 10 clusters
- The ``n_bootstrap`` should typically be at least 999 for reliable inference
