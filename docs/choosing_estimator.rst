.. meta::
   :description: Guide to choosing the right Difference-in-Differences estimator. Covers basic DiD, TWFE, staggered adoption methods (Callaway-Sant'Anna, Sun-Abraham), Synthetic DiD, and more.
   :keywords: which DiD estimator, staggered DiD estimator, difference-in-differences method selection, TWFE alternatives

Choosing an Estimator
=====================

This guide helps you select the right estimator for your research design.

Decision Flowchart
------------------

Start here and follow the questions:

0. **Is treatment continuous?** (Units receive different doses or intensities)

   - **No** → Go to question 1
   - **Yes** → Use :class:`~diff_diff.ContinuousDiD`

1. **Is treatment staggered?** (Different units treated at different times)

   - **No** → Go to question 2
   - **Yes** → Use :class:`~diff_diff.CallawaySantAnna` (or :class:`~diff_diff.EfficientDiD` for tighter SEs under PT-All)
   - **Yes, and you suspect homogeneous effects** → Use :class:`~diff_diff.ImputationDiD` or :class:`~diff_diff.TwoStageDiD` for tighter CIs
   - **Want to diagnose TWFE bias?** → Use :class:`~diff_diff.BaconDecomposition` first

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
   * - ``EfficientDiD``
     - Staggered adoption with optimal efficiency
     - PT-All (overidentified) or PT-Post
     - Group-time ATT(g,t), aggregations
   * - ``ContinuousDiD``
     - Continuous dose / treatment intensity
     - Strong Parallel Trends (SPT) for dose-response; PT for binarized ATT
     - ATT\ :sup:`loc` (PT); ATT(d), ACRT(d) (SPT)
   * - ``SunAbraham``
     - Staggered adoption, interaction-weighted
     - Conditional parallel trends
     - Cohort-specific ATTs, event study
   * - ``ImputationDiD``
     - Staggered, homogeneous effects
     - Unit + time FE structure
     - Imputed treatment effects, event study
   * - ``TwoStageDiD``
     - Staggered adoption, efficient
     - Unit + time FE structure
     - Single ATT or event study
   * - ``StackedDiD``
     - Staggered, sub-experiment approach
     - Parallel trends per cohort
     - Trimmed aggregate ATT
   * - ``TROP``
     - Factor confounding suspected
     - Factor model + weights
     - ATT with triple robustness
   * - ``BaconDecomposition``
     - TWFE diagnostic
     - (diagnostic tool)
     - 2x2 decomposition weights

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
   results = did.fit(data, outcome='y', treatment='treated', time='post')

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
   results = twfe.fit(data, outcome='y', treatment='treated',
                      unit='unit_id', time='period')

Multi-Period Event Study
~~~~~~~~~~~~~~~~~~~~~~~~

Use :class:`~diff_diff.MultiPeriodDiD` when:

- You want a full event-study with pre and post treatment effects
- You need pre-period coefficients to assess parallel trends
- You want to visualize treatment effect dynamics over time
- All treated units receive treatment at the same time (simultaneous adoption)

.. code-block:: python

   from diff_diff import MultiPeriodDiD, plot_event_study

   event = MultiPeriodDiD()
   results = event.fit(data, outcome='y', treatment='treated',
                       time='period', unit='unit_id', reference_period=2)

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

   # Overall ATT
   print(f"Overall ATT: {results.overall_att:.3f}")

   # Event study aggregation
   es = cs.fit(data, outcome='y', unit='unit_id',
               time='period', first_treat='first_treat',
               covariates=['x1', 'x2'], aggregate='event_study')
   event_study_df = es.to_dataframe('event_study')

Synthetic DiD
~~~~~~~~~~~~~

Use :class:`~diff_diff.SyntheticDiD` when:

- You have few treated units but many control units
- Pre-treatment fit between treated and control is poor
- You want to construct a weighted synthetic control

.. code-block:: python

   from diff_diff import SyntheticDiD, generate_did_data

   # SyntheticDiD requires block treatment (constant within units)
   block_data = generate_did_data(n_units=40, n_periods=10, treatment_effect=2.0)
   sdid = SyntheticDiD()
   results = sdid.fit(block_data, outcome='outcome', unit='unit',
                      time='period', treatment='treated')

   # View the unit weights
   print(results.unit_weights)

Continuous Treatment
~~~~~~~~~~~~~~~~~~~~

Use :class:`~diff_diff.ContinuousDiD` when:

- Treatment varies in **intensity or dose** (e.g., subsidy amount, hours of training)
- You want to estimate how effects change with treatment dose
- You need the full dose-response curve, not just a single average effect
- Staggered adoption where units receive different treatment levels

.. note::

   Dose-response curves ATT(d) and ACRT(d) require **Strong Parallel Trends (SPT)**.
   Under standard PT only the binarized ATT\ :sup:`loc` is identified.
   Data must include an untreated group (D = 0), a balanced panel, and
   time-invariant dose (each unit's dose is fixed across periods).

.. code-block:: python

   from diff_diff import ContinuousDiD, generate_continuous_did_data

   data = generate_continuous_did_data(n_units=200, seed=42)

   est = ContinuousDiD(n_bootstrap=199, seed=42)
   results = est.fit(data, outcome='outcome', unit='unit',
                     time='period', first_treat='first_treat',
                     dose='dose', aggregate='dose')

   # Overall effect and dose-response curve
   print(f"Overall ATT: {results.overall_att:.3f}")
   att_curve = results.dose_response_att.to_dataframe()

Efficient DiD
~~~~~~~~~~~~~

Use :class:`~diff_diff.EfficientDiD` when:

- You have staggered adoption and want **maximum statistical efficiency**
- You believe parallel trends holds across all pre-treatment periods (PT-All)
- You want tighter confidence intervals than Callaway-Sant'Anna
- You need a formal efficiency benchmark for comparing estimators

.. note::

   Phase 1 supports the **no-covariates** path only. If you need covariate
   adjustment, use :class:`~diff_diff.CallawaySantAnna` with ``estimation_method='dr'``
   or :class:`~diff_diff.ImputationDiD`.

.. code-block:: python

   from diff_diff import EfficientDiD

   edid = EfficientDiD(pt_assumption="all")  # or "post" for post-treatment CS match
   results = edid.fit(data, outcome='y', unit='unit_id',
                      time='period', first_treat='first_treat',
                      aggregate='all')
   results.print_summary()

Sun-Abraham
~~~~~~~~~~~

Use :class:`~diff_diff.SunAbraham` when:

- You have staggered adoption and want an interaction-weighted event study
- You want to decompose effects by cohort and relative time
- You need a regression-based complement to Callaway-Sant'Anna

Sun & Abraham (2021) uses a saturated TWFE regression with cohort x relative-time
interactions, then aggregates cohort-specific effects using interaction weights.

.. code-block:: python

   from diff_diff import SunAbraham

   sa = SunAbraham(control_group='never_treated')
   results = sa.fit(data, outcome='y', unit='unit_id',
                    time='period', first_treat='first_treat')
   results.print_summary()

.. note::

   Running both Sun-Abraham and Callaway-Sant'Anna provides a useful robustness
   check. Both are consistent under heterogeneous treatment effects.

Imputation DiD
~~~~~~~~~~~~~~

Use :class:`~diff_diff.ImputationDiD` when:

- You have staggered adoption with homogeneous treatment effects
- You want shorter confidence intervals than Callaway-Sant'Anna (~50% shorter)
- You need imputed counterfactual outcomes for treated observations

Borusyak, Jaravel & Spiess (2024) estimate unit + time FE on untreated observations,
impute counterfactual Y(0) for treated observations, then aggregate.

.. code-block:: python

   from diff_diff import ImputationDiD

   imp = ImputationDiD()
   results = imp.fit(data, outcome='y', unit='unit_id',
                     time='period', first_treat='first_treat',
                     aggregate='event_study')
   results.print_summary()

.. note::

   Under homogeneous effects, ImputationDiD is semiparametrically efficient.
   If you suspect heterogeneous effects across cohorts, prefer Callaway-Sant'Anna.

Two-Stage DiD
~~~~~~~~~~~~~

Use :class:`~diff_diff.TwoStageDiD` when:

- You want the same point estimates as ImputationDiD with a different variance estimator
- You prefer the GMM sandwich variance that accounts for first-stage uncertainty
- You want a single ATT or an event study from a two-stage procedure

Gardner (2022) estimates FE on untreated obs (stage 1), residualizes all outcomes,
then regresses residuals on treatment indicators (stage 2).

.. code-block:: python

   from diff_diff import TwoStageDiD

   ts = TwoStageDiD()
   results = ts.fit(data, outcome='y', unit='unit_id',
                    time='period', first_treat='first_treat',
                    aggregate='event_study')
   results.print_summary()

.. note::

   Point estimates are identical to ImputationDiD; the key difference is the
   variance estimator (GMM sandwich vs. conservative clustered).

Stacked DiD
~~~~~~~~~~~

Use :class:`~diff_diff.StackedDiD` when:

- You have staggered adoption and want a sub-experiment approach
- You want to avoid forbidden comparisons in TWFE by construction
- You need corrective Q-weights for unbiased stacked estimation

Wing, Freedman & Hollingsworth (2024) create one sub-experiment per adoption cohort
with clean controls and apply Q-weights to reweight the stacked regression.

.. code-block:: python

   from diff_diff import StackedDiD

   stk = StackedDiD(kappa_pre=2, kappa_post=3)
   results = stk.fit(data, outcome='y', unit='unit_id',
                     time='period', first_treat='first_treat',
                     aggregate='event_study')
   results.print_summary()

.. note::

   The trimmed aggregate ATT may exclude early or late cohorts whose event
   windows do not fit in the data. Check ``results.trimmed_groups``.

TROP
~~~~

Use :class:`~diff_diff.TROP` when:

- You suspect interactive fixed effects (factor confounding)
- Standard parallel trends may not hold due to unobserved factors
- You want triple robustness: factor model + unit weights + time weights

Athey, Imbens, Qu & Viviano (2025) combine nuclear norm regularization,
exponential unit distance weights, and time decay weights with LOOCV tuning.

.. code-block:: python

   from diff_diff import TROP

   trop = TROP(n_bootstrap=200)
   results = trop.fit(data, outcome='y', treatment='treated',
                      unit='unit_id', time='period')
   results.print_summary()

.. note::

   TROP is computationally intensive. Use ``method='global'`` for faster
   estimation at the cost of some flexibility vs. ``method='local'``.

Bacon Decomposition
~~~~~~~~~~~~~~~~~~~

Use :class:`~diff_diff.BaconDecomposition` when:

- You want to **diagnose** whether TWFE is biased in your staggered setting
- You need to see which 2x2 comparisons drive the TWFE estimate
- You want to check whether later-vs-earlier or already-treated-as-control comparisons carry substantial weight

Goodman-Bacon (2021) decomposes the TWFE estimate into a weighted average of
all 2x2 DiD comparisons and their weights.

.. code-block:: python

   from diff_diff import BaconDecomposition, plot_bacon

   bacon = BaconDecomposition()
   results = bacon.fit(data, outcome='y', unit='unit_id',
                       time='period', first_treat='first_treat')
   results.print_summary()

   # Visualize the decomposition
   plot_bacon(results)

.. note::

   This is a diagnostic tool, not an estimator. If the decomposition reveals
   problematic weights, switch to Callaway-Sant'Anna or another robust estimator.

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

   *Solution*: Always specify ``cluster`` for panel data.

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
     - Uses White's robust SEs by default. Specify ``cluster`` for cluster-robust SEs. Use ``inference='wild_bootstrap'`` for few clusters (<30).
   * - ``TwoWayFixedEffects``
     - Cluster-robust (unit level)
     - Always clusters at unit level after within-transformation. Specify ``cluster`` to override. Use ``inference='wild_bootstrap'`` for few clusters.
   * - ``MultiPeriodDiD``
     - HC1 (heteroskedasticity-robust)
     - Same as basic DiD. Cluster-robust available via ``cluster``. Wild bootstrap not yet supported for multi-coefficient inference.
   * - ``CallawaySantAnna``
     - Analytical (influence function)
     - Uses influence-function SEs with WIF adjustment by default. Set ``n_bootstrap=999`` for multiplier bootstrap inference (weight types: ``rademacher``, ``mammen``, ``webb``).
   * - ``SyntheticDiD``
     - Placebo or bootstrap
     - Default uses placebo-based variance (``variance_method="placebo"``). Set ``variance_method="bootstrap"`` for bootstrap inference. Both methods use ``n_bootstrap`` replications (default 200).
   * - ``ContinuousDiD``
     - Analytical (influence function)
     - Uses influence-function-based SEs by default. Use ``n_bootstrap=199`` (or higher) for multiplier bootstrap inference with proper CIs.
   * - ``SunAbraham``
     - Cluster-robust (unit level)
     - Clusters at unit level by default. Specify ``cluster`` to override. Use ``n_bootstrap`` for pairs bootstrap inference.
   * - ``ImputationDiD``
     - Conservative clustered (Theorem 3)
     - Uses conservative clustered variance from Borusyak et al. Theorem 3, clustered at unit level. Use ``n_bootstrap`` for multiplier bootstrap.
   * - ``TwoStageDiD``
     - GMM sandwich (clustered)
     - Uses GMM sandwich variance accounting for first-stage estimation uncertainty, clustered at unit level. Use ``n_bootstrap`` for multiplier bootstrap.
   * - ``StackedDiD``
     - Cluster-robust (unit level)
     - Clusters at unit level by default. Set ``cluster='unit_subexp'`` for (unit, sub-experiment) clustering.
   * - ``TripleDifference``
     - Influence function (robust)
     - Uses influence-function-based SEs (inherently heteroskedasticity-robust). Specify ``cluster`` for cluster-robust SEs.
   * - ``TROP``
     - Bootstrap (n_bootstrap=200)
     - Uses unit-level block bootstrap for variance estimation. Bootstrap is always required (minimum n_bootstrap=2).
   * - ``EfficientDiD``
     - Analytical (EIF-based)
     - Uses efficient influence function SE = sqrt(mean(EIF^2) / n). Use ``n_bootstrap`` for multiplier bootstrap.
   * - ``BaconDecomposition``
     - N/A (diagnostic)
     - Diagnostic tool only; does not produce standard errors.

**Recommendations by sample size:**

- **Large samples (N > 1000, clusters > 50)**: Default analytical SEs are reliable
- **Medium samples (clusters 30-50)**: Cluster-robust SEs recommended
- **Small samples (clusters < 30)**: Use wild cluster bootstrap (``inference='wild_bootstrap'``)
- **Very few clusters (< 10)**: Use Webb 6-point distribution (``weight_type='webb'``)

**Common pitfall:** Forgetting to cluster when units are observed multiple times.
For panel data, always cluster at the unit level unless you have a strong reason not to.

.. code-block:: python

   from diff_diff import DifferenceInDifferences, generate_did_data

   panel = generate_did_data(n_units=200, n_periods=10, treatment_effect=2.0)

   # Good: Cluster at unit level for panel data
   did = DifferenceInDifferences(cluster='unit')
   results = did.fit(panel, outcome='outcome', treatment='treated',
                     time='post')

   # Better for few clusters: Wild bootstrap
   did = DifferenceInDifferences(inference='wild_bootstrap', cluster='unit')
   results = did.fit(panel, outcome='outcome', treatment='treated',
                     time='post')

When in Doubt
-------------

If you're unsure which estimator to use:

1. **Start with CallawaySantAnna** - It's valid even for non-staggered designs
   and provides the most flexible output (group-time effects, aggregations)

2. **Check for heterogeneity** - Plot event studies to see if effects vary

3. **Run sensitivity analysis** - Use HonestDiD to assess robustness

4. **Compare estimators** - If results differ substantially across estimators,
   investigate why (often reveals violations of assumptions)
