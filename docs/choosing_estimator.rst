Choosing an Estimator
=====================

This guide helps you select the right estimator for your research design.

Decision Flowchart
------------------

Start here and follow the questions:

1. **Is treatment staggered?** (Different units treated at different times)

   - **No** → Go to question 2
   - **Yes** → Use :class:`~diff_diff.CallawaySantAnna`

2. **Do you have panel data?** (Multiple observations per unit over time)

   - **No** → Use :class:`~diff_diff.DifferenceInDifferences` (basic 2x2)
   - **Yes** → Go to question 3

3. **Do you need period-specific effects?** (Event study design)

   - **No** → Use :class:`~diff_diff.TwoWayFixedEffects`
   - **Yes** → Use :class:`~diff_diff.MultiPeriodDiD`

4. **Is your treated group small?** (Few treated units, many controls)

   - Consider :class:`~diff_diff.SyntheticDiD` for better pre-treatment fit

Quick Reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 20 30 25 25

   * - Estimator
     - Best For
     - Key Assumption
     - Output
   * - ``DifferenceInDifferences``
     - Simple 2x2 designs, cross-sectional comparisons
     - Parallel trends (2 periods)
     - Single ATT
   * - ``TwoWayFixedEffects``
     - Panel data, simultaneous treatment
     - Parallel trends (all periods)
     - Single ATT with unit/time FE
   * - ``MultiPeriodDiD``
     - Event studies, dynamic effects
     - Parallel trends (pre-periods)
     - Period-specific effects
   * - ``CallawaySantAnna``
     - Staggered adoption, heterogeneous timing
     - Conditional parallel trends
     - Group-time ATT(g,t), aggregations
   * - ``SyntheticDiD``
     - Few treated units, many controls
     - Synthetic parallel trends
     - ATT with unit/time weights

Detailed Guidance
-----------------

Basic 2x2 DiD
~~~~~~~~~~~~~

Use :class:`~diff_diff.DifferenceInDifferences` when:

- You have a simple before/after, treatment/control design
- Treatment occurs simultaneously for all treated units
- You want a single average treatment effect

.. code-block:: python

   from diff_diff import DifferenceInDifferences

   did = DifferenceInDifferences()
   results = did.fit(data, outcome='y', treated='treated', post='post')

Two-Way Fixed Effects
~~~~~~~~~~~~~~~~~~~~~

Use :class:`~diff_diff.TwoWayFixedEffects` when:

- You have panel data with multiple time periods
- Treatment timing is the same for all treated units
- You want to control for unit and time fixed effects
- You don't need to see period-by-period effects

.. warning::

   TWFE can be biased with staggered treatment timing. Already-treated units
   act as controls for newly-treated units, which can cause negative weighting.
   Use :class:`~diff_diff.CallawaySantAnna` for staggered designs.

.. code-block:: python

   from diff_diff import TwoWayFixedEffects

   twfe = TwoWayFixedEffects()
   results = twfe.fit(data, outcome='y', treated='treated',
                      unit='unit_id', time='period')

Multi-Period Event Study
~~~~~~~~~~~~~~~~~~~~~~~~

Use :class:`~diff_diff.MultiPeriodDiD` when:

- You want to visualize treatment effects over time
- You need to test for pre-trends (placebo effects before treatment)
- You want to examine treatment effect dynamics

.. code-block:: python

   from diff_diff import MultiPeriodDiD, plot_event_study

   event = MultiPeriodDiD(reference_period=-1)
   results = event.fit(data, outcome='y', treated='treated',
                       time='period', unit='unit_id', treatment_start=5)

   # Visualize
   plot_event_study(results)

Callaway-Sant'Anna
~~~~~~~~~~~~~~~~~~

Use :class:`~diff_diff.CallawaySantAnna` when:

- Treatment is adopted at different times (staggered rollout)
- You want valid treatment effect estimates with heterogeneous timing
- You need group-time specific effects ATT(g,t)

This is the recommended estimator for most applied work with staggered adoption.

.. code-block:: python

   from diff_diff import CallawaySantAnna

   cs = CallawaySantAnna(
       control_group='never_treated',  # or 'not_yet_treated'
       estimation_method='dr'  # doubly robust (recommended)
   )
   results = cs.fit(data, outcome='y', unit='unit_id',
                    time='period', first_treat='first_treat',
                    covariates=['x1', 'x2'])

   # Get aggregated effects
   print(f"Overall ATT: {results.att:.3f}")

   # Event study aggregation
   event_study = results.aggregate('event_time')

Synthetic DiD
~~~~~~~~~~~~~

Use :class:`~diff_diff.SyntheticDiD` when:

- You have few treated units but many control units
- Pre-treatment fit between treated and control is poor
- You want to construct a weighted synthetic control

.. code-block:: python

   from diff_diff import SyntheticDiD

   sdid = SyntheticDiD()
   results = sdid.fit(data, outcome='y', unit='unit_id',
                      time='period', treated='treated',
                      treatment_start=5)

   # View the unit weights
   print(results.unit_weights)

Common Pitfalls
---------------

1. **Using TWFE with staggered adoption**

   TWFE estimates a weighted average of all 2x2 comparisons, including
   "forbidden" comparisons where already-treated units serve as controls.
   This can lead to severe bias, even negative weights on treatment effects.

   *Solution*: Use CallawaySantAnna for staggered designs.

2. **Ignoring treatment effect heterogeneity**

   If treatment effects vary by cohort (when units are treated) or over time
   (dynamic effects), aggregated estimators may be misleading.

   *Solution*: Use CallawaySantAnna and examine ATT(g,t) and event study plots.

3. **Failing to test parallel trends**

   The parallel trends assumption is untestable in the post-period but can
   be assessed using pre-treatment data.

   *Solution*: Use :func:`~diff_diff.check_parallel_trends` and
   :class:`~diff_diff.HonestDiD` for sensitivity analysis.

4. **Inappropriate clustering**

   Standard errors should typically be clustered at the level of treatment
   assignment (often the unit level).

   *Solution*: Always specify ``cluster_col`` for panel data.

Standard Error Methods
----------------------

Different estimators compute standard errors differently. Understanding these
differences helps interpret results and choose appropriate inference.

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Estimator
     - Default SE Method
     - Details
   * - ``DifferenceInDifferences``
     - HC1 (heteroskedasticity-robust)
     - Uses White's robust SEs by default. Specify ``cluster_col`` for cluster-robust SEs. Use ``inference='wild_bootstrap'`` for few clusters (<30).
   * - ``TwoWayFixedEffects``
     - Cluster-robust (unit level)
     - Always clusters at unit level after within-transformation. Specify ``cluster_col`` to override. Use ``inference='wild_bootstrap'`` for few clusters.
   * - ``MultiPeriodDiD``
     - HC1 (heteroskedasticity-robust)
     - Same as basic DiD. Cluster-robust available via ``cluster_col``. Wild bootstrap not yet supported for multi-coefficient inference.
   * - ``CallawaySantAnna``
     - Analytical (simple difference)
     - Uses simple variance of group-time means. Use ``bootstrap()`` method for multiplier bootstrap inference with proper SEs, CIs, and p-values.
   * - ``SyntheticDiD``
     - Bootstrap or placebo-based
     - Default uses bootstrap resampling. Set ``n_bootstrap=0`` for placebo-based inference using pre-treatment residuals.

**Recommendations by sample size:**

- **Large samples (N > 1000, clusters > 50)**: Default analytical SEs are reliable
- **Medium samples (clusters 30-50)**: Cluster-robust SEs recommended
- **Small samples (clusters < 30)**: Use wild cluster bootstrap (``inference='wild_bootstrap'``)
- **Very few clusters (< 10)**: Use Webb 6-point distribution (``weight_type='webb'``)

**Common pitfall:** Forgetting to cluster when units are observed multiple times.
For panel data, always cluster at the unit level unless you have a strong reason not to.

.. code-block:: python

   # Good: Cluster at unit level for panel data
   did = DifferenceInDifferences()
   results = did.fit(data, outcome='y', treated='treated',
                     post='post', cluster_col='unit_id')

   # Better for few clusters: Wild bootstrap
   did = DifferenceInDifferences(inference='wild_bootstrap')
   results = did.fit(data, outcome='y', treated='treated',
                     post='post', cluster_col='state')

When in Doubt
-------------

If you're unsure which estimator to use:

1. **Start with CallawaySantAnna** - It's valid even for non-staggered designs
   and provides the most flexible output (group-time effects, aggregations)

2. **Check for heterogeneity** - Plot event studies to see if effects vary

3. **Run sensitivity analysis** - Use HonestDiD to assess robustness

4. **Compare estimators** - If results differ substantially across estimators,
   investigate why (often reveals violations of assumptions)
