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
   results = did.fit(data, outcome='y', treatment='treated', post='post')

   # Compare SE methods
   did_robust = DifferenceInDifferences()
   did_cluster = DifferenceInDifferences(cluster='unit_id')
   did_wild = DifferenceInDifferences(inference='wild_bootstrap', cluster='unit_id')

   r1 = did_robust.fit(data, outcome='y', treatment='treated', post='post')
   r2 = did_cluster.fit(data, outcome='y', treatment='treated', post='post')
   r3 = did_wild.fit(data, outcome='y', treatment='treated', post='post')

   print(f"Robust SE: {r1.se:.4f}")
   print(f"Cluster SE: {r2.se:.4f}")
   print(f"Wild bootstrap SE: {r3.se:.4f}")

"Wild bootstrap takes too long"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** Bootstrap inference is slow.

**Solutions:**

.. code-block:: python

   # Reduce number of bootstrap iterations (default is 999)
   did = DifferenceInDifferences(inference='wild_bootstrap', cluster='unit_id',
                                  n_bootstrap=499)

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

   # Use bootstrap for better inference. On a BOOTSTRAPPED fit, post-fit
   # results.aggregate('event_study') raises - the percentile draws are not
   # retained - so the deprecated fit-time aggregate= remains the documented
   # route for this case until 4.0. (results.aggregate('simple') - and,
   # where supported, results.aggregate('total') - relays the stored
   # bootstrap inference.)
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

   from diff_diff import CallawaySantAnna, plot_event_study

   # For CallawaySantAnna, aggregate to an event study post-fit, then plot.
   # base_period="universal" gives the event study explicit reference row(s),
   # marked in the container - under the default varying base every
   # pre-treatment point is an estimated effect with no common anchor, so
   # renormalizing a plot around one would silently shift every point.
   cs = CallawaySantAnna(base_period='universal')
   results = cs.fit(data, outcome='y', unit='unit_id',
                    time='period', first_treat='first_treat')
   es = results.aggregate('event_study')

   # Inspect the actual reference row(s) - on gapped period grids the
   # positional base can sit at an event time other than -1
   print(es.to_dataframe().query("is_reference"))

   # Plot the container - it carries its own reference; no manual
   # reference_period override is needed (or safe to hard-code)
   plot_event_study(es)

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
   # (post= is the 0/1 post-treatment dummy; the deprecated time= alias
   # still works through 3.9)
   twfe = TwoWayFixedEffects()
   results = twfe.fit(data, outcome='y', treatment='treated',
                      unit='unit_id', post='post')

   # Reduce bootstrap iterations for initial exploration
   did = DifferenceInDifferences(inference='wild_bootstrap', cluster='unit_id',
                                  n_bootstrap=99)

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
   results = did.fit(data, outcome='y', treatment='treatment', post='post')

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

HeterogeneousAdoptionDiD (HAD) Issues
-------------------------------------

"Resolved estimand is not what I expected (WAS vs WAS_d_lower)"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** ``HeterogeneousAdoptionDiD`` resolves ``target_parameter`` to
``"WAS_d_lower"`` when you expected ``"WAS"`` (or vice versa).

**Cause:** HAD auto-detects the design path from the unit-level
post-treatment dose ``D_{g,F}`` (the dose at the first treated period
``F``, one value per unit), NOT from the full panel ``dose`` column. The
panel column carries structural pre-period zeros (HAD requires
``D_{g,t} = 0`` for ``t < F``), so ``had_data['dose'].min()`` is always
zero on a valid HAD panel and tells you nothing about the resolved
design. ``_detect_design`` then resolves on ``D_{g,F}`` and picks Design
1' (``continuous_at_zero``, targets WAS) when EITHER
``D_{g,F}.min() == 0`` exactly OR ``D_{g,F}.min()`` is a small positive
value below ``0.01 * median(|D_{g,F}|)`` (the small-share-of-treated
escape clause). Otherwise the estimator routes to Design 1, with a
further check for mass-point structure (modal fraction at ``D_{g,F}.min()``
exceeding 2% routes to ``mass_point``; otherwise
``continuous_near_d_lower``); both Design 1 paths target ``WAS_{d_lower}``.

**Solutions:**

.. code-block:: python

   import numpy as np
   import pandas as pd
   from diff_diff import HeterogeneousAdoptionDiD

   # Build a HAD-shape panel: D=0 in pre-periods (t < F), D > 0 only at F+.
   rng = np.random.default_rng(42)
   G, F, T = 200, 4, 5
   doses = rng.beta(0.5, 1.0, size=G)
   rows = []
   for g in range(G):
       for t in range(1, T + 1):
           y = (rng.normal()
                + (doses[g] + doses[g] ** 2) * (t >= F)
                + rng.normal(0, 0.5))
           d = doses[g] if t >= F else 0.0
           rows.append({'unit': g, 'period': t, 'y': y, 'dose': d})
   had_data = pd.DataFrame(rows)

   # Inspect the support the detector actually uses: per-unit dose at the
   # first treated period F. Pre-period zeros on the panel column are
   # structural and ignored by `_detect_design()`.
   d_at_F = had_data.loc[had_data['period'] == F].set_index('unit')['dose']
   print(d_at_F.describe())
   d_min = float(d_at_F.min())
   d_thr = 0.01 * float(np.median(np.abs(d_at_F)))
   print(f"D_{{g,F}}.min() = {d_min:.6g}; "
         f"0.01 * median(|D_{{g,F}}|) = {d_thr:.6g}; "
         f"D_{{g,F}}.min() < threshold => Design 1' (WAS)")

   # Check the resolved estimand after fitting
   est = HeterogeneousAdoptionDiD()
   results = est.fit(had_data, outcome='y', unit='unit',
                     time='period', dose='dose')
   print(f"Resolved: {results.target_parameter}")

   # If you intend Design 1' but `D_{g,F}.min()` exceeds the threshold,
   # verify the dose-variable encoding (e.g. log-transformed doses where
   # 0 was mapped to a small positive value larger than 1% of the median).

"Mass-point design selected"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** HAD reports that the ``mass_point`` design was selected
instead of ``continuous_at_zero`` or ``continuous_near_d_lower``.

**Cause:** ``mass_point`` is a distinct Design 1 estimator path from the
dCDH 2026 paper (Section 3.2.4), not a fallback from the continuous
local-linear fits. ``_detect_design()`` resolves to ``mass_point`` when the
modal fraction at ``d.min()`` exceeds 2%, signalling a heavy point mass at
the dose-support boundary. On this path both the point estimate and the SE
differ from the continuous paths: the estimator uses the Wald-IV
sample-average ratio with binary instrument ``Z_g = 1{D_{g,2} > d_lower}``
- ``(Ybar_{Z=1} - Ybar_{Z=0}) / (Dbar_{Z=1} - Dbar_{Z=0})`` - and inference
uses the structural-residual 2SLS sandwich (the local-linear / CCT-2014
SE path is not used here).

**Solutions:**

.. code-block:: python

   import numpy as np
   import pandas as pd
   from diff_diff import HeterogeneousAdoptionDiD

   # Build a HAD panel with a heavy boundary mass at d_lower so the
   # modal fraction at d.min() exceeds 2% and `_detect_design` resolves
   # to `mass_point`.
   rng = np.random.default_rng(42)
   G, F, T = 200, 4, 5
   d_lower = 0.5
   mass_frac = 0.3
   doses = np.where(
       rng.uniform(size=G) < mass_frac,
       d_lower,
       rng.uniform(d_lower + 0.1, 2.0, size=G),
   )
   rows = []
   for g in range(G):
       for t in range(1, T + 1):
           y = (rng.normal()
                + doses[g] * (t >= F)
                + rng.normal(0, 0.5))
           d = doses[g] if t >= F else 0.0
           rows.append({'unit': g, 'period': t, 'y': y, 'dose': d})
   had_data = pd.DataFrame(rows)

   est = HeterogeneousAdoptionDiD()
   results = est.fit(had_data, outcome='y', unit='unit',
                     time='period', dose='dose')

   # Inspect the resolved design
   print(f"Design: {results.design}")  # 'mass_point' here

   # The mass-point Wald-IV estimator + structural-residual 2SLS
   # sandwich is the canonical Section 3.2.4 path for designs with a
   # heavy boundary point mass; accept the resolution unless you can
   # re-bin the dose variable so the modal fraction at d.min() drops
   # below 2% (then the detector picks continuous_near_d_lower).

"NotImplementedError on survey + mass-point + vcov_type='classical'"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** Calling ``HeterogeneousAdoptionDiD.fit(..., vcov_type="classical")``
under ``survey_design=SurveyDesign(...)`` raises ``NotImplementedError`` on the
mass-point path (both the static and the event-study survey paths; the event-study
rejection fires regardless of ``cband`` because the Binder-TSL analytical SE
consumes the HC1-scaled influence function either way).

**Cause:** The per-unit 2SLS influence function returned by the mass-point fit
is HC1-scaled so that ``compute_survey_if_variance`` and the sup-t bootstrap
target ``V_HC1`` consistently. Mixing it with a classical analytical SE would
silently report a ``V_HC1``-targeted variance under a ``classical`` label.

**Solutions:**

.. code-block:: python

   # The constructor default (`vcov_type='classical'`) triggers the guard
   # on the mass-point survey path - so plain
   # `HeterogeneousAdoptionDiD()` is NOT a workaround. Use:
   est = HeterogeneousAdoptionDiD(vcov_type='hc1')

A classical-aligned IF derivation is queued for a follow-up release; until
then, ``vcov_type='hc1'`` is the
recommended path for survey + mass-point fits. See :doc:`api/had` for the
full SE-regime contract.

"Panel-only event-study restriction"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** ``HeterogeneousAdoptionDiD.fit()`` on a multi-period
(event-study mode) staggered panel raises.

**Cause:** The Appendix B.2 event-study extension requires either a
common-adoption panel (single first-treat period; ``first_treat`` is
then optional and the period is inferred from the dose invariant) or a
staggered panel with ``first_treat`` provided so the estimator can
auto-filter to the last-treatment cohort plus never-treated units (with
a ``UserWarning``). The fit raises only when the panel is staggered
**and** ``first_treat`` is missing.

**Solutions:**

.. code-block:: python

   import numpy as np
   import pandas as pd

   # Build a staggered HAD panel for this example: 120 units, three
   # cohorts (30 never-treated + 30 treated at period 5 + 60 treated at
   # period 8). Dose is zero pre-treatment per unit and a constant
   # positive value post-treatment, so the first_treat / dose-path
   # consistency validator passes. The 60-unit last cohort gives the
   # boundary local-linear estimator enough distinct dose values to fit.
   np.random.seed(42)
   n_units, n_periods = 120, 10
   first_treat_per_unit = np.array([0] * 30 + [5] * 30 + [8] * 60)
   dose_per_unit = np.where(
       first_treat_per_unit > 0, np.random.uniform(0.5, 2.0, n_units), 0.0
   )
   rows = []
   for u in range(n_units):
       ft = first_treat_per_unit[u]
       for t in range(n_periods):
           d_ut = dose_per_unit[u] if (ft > 0 and t >= ft) else 0.0
           y_ut = (d_ut > 0) * dose_per_unit[u] * 0.5 + np.random.normal()
           rows.append((u, t, d_ut, ft, y_ut))
   data = pd.DataFrame(rows, columns=["unit", "period", "dose", "first_treat", "y"])

   # Primary remedy: pass `first_treat` so the estimator auto-filters
   # to the last-treatment cohort + never-treated and emits a UserWarning.
   est = HeterogeneousAdoptionDiD()
   results = est.fit(data, outcome='y', unit='unit',
                     time='period', dose='dose',
                     first_treat='first_treat')

   # Equivalent: subset to the last-treatment cohort + never-treated
   # before fitting (skips the UserWarning).
   last_cohort = data['first_treat'].max()
   subset = data[(data['first_treat'] == last_cohort) |
                 (data['first_treat'] == 0)]
   results = est.fit(subset, outcome='y', unit='unit',
                     time='period', dose='dose')

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
   results = did.fit(data, outcome='outcome', treatment='treated', post='post')
   print(f"True effect: 2.0, Estimated: {results.att:.3f}")

For bugs or feature requests, please open an issue on
`GitHub <https://github.com/igerber/diff-diff/issues>`_.
