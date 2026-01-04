Troubleshooting
===============

This guide covers common issues and their solutions when using diff-diff.

Data Issues
-----------

"No treated observations found"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** The estimator raises an error that no treated units were found.

**Causes:**

1. Treatment column contains wrong values (e.g., strings instead of 0/1)
2. Treatment column has all zeros
3. Column name is misspelled

**Solutions:**

.. code-block:: python

   # Check your treatment column
   print(data['treated'].value_counts())

   # Ensure binary 0/1 values
   data['treated'] = (data['group'] == 'treatment').astype(int)

   # Or use make_treatment_indicator
   from diff_diff import make_treatment_indicator
   data['treated'] = make_treatment_indicator(data, 'group', treated_value='treatment')

"Panel is unbalanced"
~~~~~~~~~~~~~~~~~~~~~

**Problem:** TwoWayFixedEffects or CallawaySantAnna fails with unbalanced panel.

**Causes:**

1. Some units are missing observations for certain time periods
2. Units have different numbers of observations

**Solutions:**

.. code-block:: python

   from diff_diff import balance_panel

   # Balance the panel (keeps only units with all periods)
   balanced = balance_panel(data, unit='unit_id', time='period')
   print(f"Dropped {len(data) - len(balanced)} observations")

   # Alternative: check balance first
   from diff_diff import validate_did_data
   issues = validate_did_data(data, outcome='y', treated='treated',
                               unit='unit_id', time='period')
   print(issues)

Estimation Errors
-----------------

"Singular matrix" or "Matrix is singular"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** Linear algebra error during estimation.

**Causes:**

1. Perfect collinearity in covariates
2. Too few observations relative to parameters
3. Fixed effects that absorb all variation

**Solutions:**

.. code-block:: python

   # Check for collinearity
   import numpy as np
   X = data[['x1', 'x2', 'x3']].values
   print(f"Matrix rank: {np.linalg.matrix_rank(X)} vs {X.shape[1]} columns")

   # Remove redundant covariates
   # Or use fewer fixed effects

   # For SyntheticDiD, increase regularization
   sdid = SyntheticDiD(lambda_reg=1e-4)  # default is 1e-6

"Bootstrap iterations failed" warning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** SyntheticDiD warns that many bootstrap iterations failed.

**Causes:**

1. Small sample size leads to singular matrices in resamples
2. Insufficient pre-treatment periods for weight computation
3. Near-singular weight matrices

**Solutions:**

.. code-block:: python

   # Increase regularization
   sdid = SyntheticDiD(lambda_reg=1e-4, n_bootstrap=500)

   # Or use placebo-based inference instead
   sdid = SyntheticDiD(n_bootstrap=0)  # Uses placebo inference

   # Ensure sufficient pre-treatment periods (recommend >= 4)

Standard Error Issues
---------------------

"Standard errors seem too small/large"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** SEs don't match expectations or other software.

**Causes:**

1. Wrong clustering level
2. Not accounting for serial correlation
3. Different SE formulas (HC0 vs HC1 vs cluster)

**Solutions:**

.. code-block:: python

   # For panel data, always cluster at unit level
   results = did.fit(data, outcome='y', treated='treated',
                     post='post', cluster_col='unit_id')

   # Compare SE methods
   did_robust = DifferenceInDifferences()
   did_cluster = DifferenceInDifferences()
   did_wild = DifferenceInDifferences(inference='wild_bootstrap')

   r1 = did_robust.fit(data, outcome='y', treated='treated', post='post')
   r2 = did_cluster.fit(data, outcome='y', treated='treated',
                        post='post', cluster_col='unit_id')
   r3 = did_wild.fit(data, outcome='y', treated='treated',
                     post='post', cluster_col='unit_id')

   print(f"Robust SE: {r1.se:.4f}")
   print(f"Cluster SE: {r2.se:.4f}")
   print(f"Wild bootstrap SE: {r3.se:.4f}")

"Wild bootstrap takes too long"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** Bootstrap inference is slow.

**Solutions:**

.. code-block:: python

   # Reduce number of bootstrap iterations (default is 999)
   did = DifferenceInDifferences(inference='wild_bootstrap', n_bootstrap=499)

   # Note: Fewer iterations = less precise p-values
   # 499 is minimum recommended for publication

Staggered Adoption Issues
-------------------------

"No never-treated units found"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** CallawaySantAnna fails when using ``control_group='never_treated'``.

**Causes:**

1. All units are eventually treated
2. ``first_treat`` column has no never-treated indicator (typically 0 or inf)

**Solutions:**

.. code-block:: python

   # Check first_treat distribution
   print(data['first_treat'].value_counts())

   # Option 1: Use not-yet-treated as controls
   cs = CallawaySantAnna(control_group='not_yet_treated')

   # Option 2: Mark never-treated units correctly
   # Never-treated should have first_treat = 0 or np.inf
   data.loc[data['ever_treated'] == 0, 'first_treat'] = 0

"Group-time effects have large standard errors"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** ATT(g,t) estimates are imprecise.

**Causes:**

1. Small cohort sizes
2. Few comparison periods
3. High variance in outcomes

**Solutions:**

.. code-block:: python

   # Check cohort sizes
   print(data.groupby('first_treat')['unit_id'].nunique())

   # Use bootstrap for better inference
   results = cs.fit(data, ...)
   bootstrap_results = results.bootstrap(n_bootstrap=999)

   # Aggregate to get more precise estimates
   event_study = results.aggregate('event_time')
   overall_att = results.att  # Aggregated ATT

Visualization Issues
--------------------

"Event study plot looks wrong"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** Plot has unexpected gaps, wrong reference period, or missing periods.

**Solutions:**

.. code-block:: python

   from diff_diff import plot_event_study

   # Check your results first
   print(results.period_effects)  # or results.event_study_effects

   # Specify reference period explicitly
   plot_event_study(results, reference_period=-1)

   # For CallawaySantAnna, aggregate first
   event_study = results.aggregate('event_time')
   plot_event_study(event_study)

"Plot doesn't show in Jupyter"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** Matplotlib figure doesn't display.

**Solutions:**

.. code-block:: python

   import matplotlib.pyplot as plt

   # Option 1: Use plt.show()
   fig = plot_event_study(results)
   plt.show()

   # Option 2: Use inline magic (Jupyter)
   %matplotlib inline

   # Option 3: Return and display figure
   fig = plot_event_study(results)
   fig  # Display in Jupyter

Performance Issues
------------------

"Estimation is slow"
~~~~~~~~~~~~~~~~~~~~

**Problem:** Fitting takes a long time.

**Causes:**

1. Large dataset with many fixed effects
2. Bootstrap inference with many iterations
3. CallawaySantAnna with many cohorts and time periods

**Solutions:**

.. code-block:: python

   # Use absorb instead of fixed_effects for high-dimensional FE
   twfe = TwoWayFixedEffects()
   results = twfe.fit(data, outcome='y', treated='treated',
                      unit='unit_id', time='period',
                      absorb=['unit_id', 'period'])  # Faster than fixed_effects

   # Reduce bootstrap iterations for initial exploration
   did = DifferenceInDifferences(inference='wild_bootstrap', n_bootstrap=99)

   # For CallawaySantAnna, start without bootstrap
   cs = CallawaySantAnna()
   results = cs.fit(data, ...)
   # Only bootstrap for final results
   bootstrap_results = results.bootstrap(n_bootstrap=999)

Getting Help
------------

If you encounter issues not covered here:

1. **Check the API documentation** for parameter details
2. **Run validation** with ``validate_did_data()`` to catch data issues
3. **Start simple** with basic DiD before adding complexity
4. **Compare with known results** using ``generate_did_data()``

.. code-block:: python

   # Generate test data with known effect
   from diff_diff import generate_did_data, DifferenceInDifferences

   data = generate_did_data(n_units=100, n_periods=10, treatment_effect=2.0)
   did = DifferenceInDifferences()
   results = did.fit(data, outcome='y', treated='treated', post='post')
   print(f"True effect: 2.0, Estimated: {results.att:.3f}")

For bugs or feature requests, please open an issue on
`GitHub <https://github.com/igerber/diff-diff/issues>`_.
