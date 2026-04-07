.. meta::
   :description: Troubleshooting guide for diff-diff. Solutions for common DiD issues including singular matrices, collinear covariates, insufficient variation, and convergence problems.
   :keywords: difference-in-differences troubleshooting, DiD singular matrix, collinear covariates fix, parallel trends test fails

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
   data = make_treatment_indicator(data, 'group', treated_values='treatment')

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
   balanced = balance_panel(data, unit_column='unit_id', time_column='period')
   print(f"Dropped {len(data) - len(balanced)} observations")

   # Alternative: check balance first
   from diff_diff import validate_did_data
   issues = validate_did_data(data, outcome='y', treatment='treated',
                               time='period', unit='unit_id')
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
   sdid = SyntheticDiD(zeta_omega=1e-4)  # increase unit weight regularization

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
   sdid = SyntheticDiD(zeta_omega=1e-4, zeta_lambda=1e-4, n_bootstrap=500)

   # Or use placebo-based inference instead
   sdid = SyntheticDiD(variance_method="placebo")  # Uses placebo inference

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
   did = DifferenceInDifferences(cluster='unit_id')
   results = did.fit(data, outcome='y', treatment='treated', time='post')

   # Compare SE methods
   did_robust = DifferenceInDifferences()
   did_cluster = DifferenceInDifferences(cluster='unit_id')
   did_wild = DifferenceInDifferences(inference='wild_bootstrap', cluster='unit_id')

   r1 = did_robust.fit(data, outcome='y', treatment='treated', time='post')
   r2 = did_cluster.fit(data, outcome='y', treatment='treated', time='post')
   r3 = did_wild.fit(data, outcome='y', treatment='treated', time='post')

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
   cs = CallawaySantAnna(n_bootstrap=999)
   results = cs.fit(data, outcome='y', unit='unit_id',
                    time='period', first_treat='first_treat',
                    aggregate='event_study')

   # Access aggregated results
   print(results.overall_att)  # Overall ATT
   print(results.event_study_effects)  # Event study effects

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

   # For CallawaySantAnna, fit with aggregate='event_study'
   results = cs.fit(data, outcome='y', unit='unit_id',
                    time='period', first_treat='first_treat',
                    aggregate='event_study')
   plot_event_study(results)

"Plot doesn't show in Jupyter"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** Matplotlib figure doesn't display.

**Solutions:**

.. code-block:: python

   import matplotlib.pyplot as plt

   # Option 1: Use plt.show()
   ax = plot_event_study(results)
   plt.show()

   # Option 2: Use inline magic (Jupyter)
   %matplotlib inline

   # Option 3: Return and display figure
   ax = plot_event_study(results)
   ax  # Display in Jupyter

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

   # TWFE already handles unit + time FE via within-transformation
   twfe = TwoWayFixedEffects()
   results = twfe.fit(data, outcome='y', treatment='treated',
                      unit='unit_id', time='period')

   # Reduce bootstrap iterations for initial exploration
   did = DifferenceInDifferences(inference='wild_bootstrap', n_bootstrap=99)

   # For CallawaySantAnna, start without bootstrap
   cs = CallawaySantAnna()
   results = cs.fit(data, outcome='y', unit='unit_id',
                    time='period', first_treat='first_treat')
   # Use n_bootstrap for final results
   cs_boot = CallawaySantAnna(n_bootstrap=999)
   results = cs_boot.fit(data, outcome='y', unit='unit_id',
                         time='period', first_treat='first_treat')

Rust Backend Issues
-------------------

"Rust backend is not available"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** ``ImportError`` when using ``DIFF_DIFF_BACKEND=rust`` or attempting to
use Rust-accelerated operations.

**Causes:**

1. Rust backend was not compiled during installation
2. The ``maturin`` build step was skipped or failed
3. Platform does not have a pre-built wheel available

**Solutions:**

.. code-block:: python

   # Check if Rust backend is available
   from diff_diff import HAS_RUST_BACKEND
   print(f"Rust backend available: {HAS_RUST_BACKEND}")

   # Force pure Python mode (no Rust required)
   import os
   os.environ['DIFF_DIFF_BACKEND'] = 'python'

.. code-block:: bash

   # Rebuild with Rust backend
   pip install -e ".[dev]"
   maturin develop --release

   # On macOS with Apple Accelerate
   maturin develop --release --features accelerate

TROP Issues
-----------

"All tuning parameter combinations failed"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** TROP raises an error that all tuning parameter combinations failed
during leave-one-out cross-validation (LOOCV).

**Causes:**

1. Insufficient pre-treatment periods (minimum 2; recommend 4+ for stability)
2. Near-constant outcomes that leave no variation to fit
3. Data is too sparse for the requested lambda grids

**Solutions:**

.. code-block:: python

   from diff_diff import TROP

   # Widen the lambda grids to give the optimizer more room
   trop = TROP(
       lambda_time_grid=[0.0, 0.5, 1.0, 2.0, 5.0],
       lambda_unit_grid=[0.0, 0.5, 1.0, 2.0, 5.0],
       lambda_nn_grid=[0.0, 0.1, 1.0, 10.0],
   )

   # TROP requires at least 2 pre-treatment periods (4+ recommended)
   pre_periods = data.loc[data['post'] == 0, 'period'].nunique()
   print(f"Pre-treatment periods: {pre_periods}")  # Must be >= 2; stability improves with >= 4

   # If TROP cannot find valid parameters, try CallawaySantAnna as a fallback
   from diff_diff import CallawaySantAnna
   cs = CallawaySantAnna()
   results = cs.fit(data, outcome='y', unit='unit_id',
                    time='period', first_treat='first_treat')

"LOOCV fits failed / numerical instability"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** Partial LOOCV failures during TROP tuning, or warnings about
numerical instability in cross-validation fits.

**Causes:**

1. Poor data quality (missing values, outliers)
2. Regularization parameters too small for the data scale

**Solutions:**

.. code-block:: python

   # Check data quality
   print(data[['y', 'treatment', 'post']].describe())
   print(f"Missing values:\n{data.isnull().sum()}")

   # Increase regularization to improve numerical stability
   trop = TROP(
       lambda_nn_grid=[0.1, 1.0, 10.0, 100.0],  # Larger minimum lambda
   )

"Few bootstrap iterations succeeded"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** TROP warns that only N of M bootstrap iterations completed
successfully, leading to imprecise standard errors.

**Causes:**

1. Small sample sizes cause singular matrices in bootstrap resamples
2. Complex model specification amplifies resampling instability

**Solutions:**

.. code-block:: python

   # Increase total bootstrap iterations to get enough successes
   trop = TROP(n_bootstrap=999)

   # Simplify the model to reduce bootstrap failures
   trop = TROP(method='global', n_bootstrap=999)

Continuous DiD Issues
---------------------

"Dose appears discrete"
~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** ``ContinuousDiD`` warns that the dose variable appears to contain
only integer or discrete values.

**Causes:**

1. Treatment is truly binary (0/1) and should use standard DiD
2. Dose variable is coded as integers but represents a continuous measure

**Solutions:**

.. code-block:: python

   # Check dose distribution
   print(data['dose'].value_counts())

   # If treatment is truly binary, use standard DiD instead
   from diff_diff import DifferenceInDifferences
   did = DifferenceInDifferences()
   results = did.fit(data, outcome='y', treatment='treatment', time='post')

   # If dose is continuous but stored as int, convert
   data['dose'] = data['dose'].astype(float)

"No post-treatment cells available for aggregation"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** No (g, t) cells are available after filtering, so aggregation
cannot produce an ATT estimate.

**Causes:**

1. ``first_treat`` is miscoded (e.g., all zeros or all the same value)
2. No post-treatment periods exist in the data for treated cohorts
3. Filtering removed all valid cells

**Solutions:**

.. code-block:: python

   # Check first_treat coding
   print(data['first_treat'].value_counts())

   # Verify that post-treatment periods exist for treated units
   treated = data[data['first_treat'] > 0]
   for g, group in treated.groupby('first_treat'):
       post_obs = group[group['period'] >= g]
       print(f"Cohort {g}: {len(post_obs)} post-treatment observations")

Imputation / Two-Stage DiD Issues
----------------------------------

"Non-constant first_treat values"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** ``ImputationDiD`` or ``TwoStageDiD`` issues a warning because
``first_treat`` varies within units. The estimator coerces to a single value
per unit (using the first observed value) and proceeds, but results may be
unreliable.

**Causes:**

1. Units switch treatment status back and forth
2. Data merge errors created inconsistent ``first_treat`` values

**Solutions:**

.. code-block:: python

   # Check for non-constant first_treat within units
   varying = data.groupby('unit_id')['first_treat'].nunique()
   bad_units = varying[varying > 1].index
   print(f"Units with varying first_treat: {len(bad_units)}")

   # Fix: ensure first_treat is constant per unit (absorbing state)
   first_treat_map = data.groupby('unit_id')['first_treat'].first()
   data['first_treat'] = data['unit_id'].map(first_treat_map)

"Units treated in all observed periods"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** All observed periods for some units are post-treatment, so no
pre-treatment outcomes exist to construct counterfactuals.

**Causes:**

1. Always-treated units entered the panel already treated
2. Observation window starts after treatment onset for some cohorts

**Solutions:**

.. code-block:: python

   # Identify always-treated units (treated at or before first observed period)
   # Exclude never-treated (first_treat == 0) which are the control group
   unit_ft = data.groupby('unit_id')['first_treat'].first()
   min_period = data['period'].min()
   always_treated = unit_ft[(unit_ft > 0) & (unit_ft <= min_period)]
   print(f"Always-treated units: {len(always_treated)}")

   # Drop always-treated units (keep never-treated controls)
   data = data[~data['unit_id'].isin(always_treated.index)]

"Horizons not identified without never-treated units"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** Certain event study horizons return NaN because they require
never-treated units for identification (Proposition 5 in Borusyak et al.).

**Causes:**

1. No never-treated units in the data
2. Specific long-horizon estimates need a comparison group that spans those periods

**Solutions:**

.. code-block:: python

   # Check for never-treated units
   never_treated = data.groupby('unit_id')['first_treat'].first()
   print(f"Never-treated units: {(never_treated == 0).sum()}")

   # Option 1: Include never-treated units in your sample
   # Option 2: Accept NaN for unidentified horizons
   results = ImputationDiD().fit(data, outcome='y', unit='unit_id',
                                time='period', first_treat='first_treat')
   # NaN horizons are expected when never-treated units are absent

Bacon Decomposition Issues
--------------------------

"Unbalanced panel detected"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** ``BaconDecomposition`` issues a warning because the panel is
unbalanced. Bacon decomposition assumes balanced panels and results may be
inaccurate with missing observations.

**Causes:**

1. Some units are missing observations for certain time periods
2. Units entered or exited the panel at different times

**Solutions:**

.. code-block:: python

   from diff_diff import balance_panel, BaconDecomposition

   # Balance the panel first
   balanced = balance_panel(data, unit_column='unit_id', time_column='period')
   print(f"Dropped {len(data) - len(balanced)} observations to balance panel")

   # Then run decomposition
   bacon = BaconDecomposition()
   results = bacon.fit(balanced, outcome='y', unit='unit_id',
                       time='period', first_treat='first_treat')

Deprecation Warnings
--------------------

"method='twostep' is deprecated"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** TROP emits a ``FutureWarning`` that ``method='twostep'`` is
deprecated.

**Causes:**

1. Code uses the old ``method='twostep'`` parameter name

**Solutions:**

.. code-block:: python

   # Old (deprecated)
   trop = TROP(method='twostep')

   # New (use 'local' instead)
   trop = TROP(method='local')

"method='joint' is deprecated"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** TROP emits a ``FutureWarning`` that ``method='joint'`` is
deprecated.

**Causes:**

1. Code uses the old ``method='joint'`` parameter name

**Solutions:**

.. code-block:: python

   # Old (deprecated)
   trop = TROP(method='joint')

   # New (use 'global' instead)
   trop = TROP(method='global')

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
   results = did.fit(data, outcome='outcome', treatment='treated', time='post')
   print(f"True effect: 2.0, Estimated: {results.att:.3f}")

For bugs or feature requests, please open an issue on
`GitHub <https://github.com/igerber/diff-diff/issues>`_.
