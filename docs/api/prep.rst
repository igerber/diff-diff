Data Preparation
================

Utilities for preparing and validating data for DiD analysis.

.. module:: diff_diff.prep

Data Generation
---------------

generate_did_data
~~~~~~~~~~~~~~~~~

Generate synthetic data with known treatment effects for testing.

.. autofunction:: diff_diff.generate_did_data

Example
^^^^^^^

.. code-block:: python

   from diff_diff import generate_did_data

   # Generate basic 2x2 DiD data
   data = generate_did_data(
       n_units=100,
       n_periods=10,
       treatment_effect=5.0,
       treatment_period=5,
       treatment_fraction=0.5,
       noise_sd=1.0
   )

   print(data.head())
   # Columns: unit_id, period, outcome, treated, post

generate_staggered_data
~~~~~~~~~~~~~~~~~~~~~~~

Generate synthetic staggered adoption data for testing.

.. autofunction:: diff_diff.generate_staggered_data

Example
^^^^^^^

.. code-block:: python

   from diff_diff import generate_staggered_data

   data = generate_staggered_data(
       n_units=200,
       n_periods=10,
       cohort_periods=[4, 6, 8],
       seed=42
   )

generate_event_study_data
~~~~~~~~~~~~~~~~~~~~~~~~~

Generate synthetic event study data for testing.

.. autofunction:: diff_diff.generate_event_study_data

generate_ddd_data
~~~~~~~~~~~~~~~~~

Generate synthetic Triple Difference data.

.. autofunction:: diff_diff.generate_ddd_data

generate_ddd_panel_data
~~~~~~~~~~~~~~~~~~~~~~~

Generate synthetic panel-structured Triple Difference data for power analysis.

.. autofunction:: diff_diff.generate_ddd_panel_data

generate_factor_data
~~~~~~~~~~~~~~~~~~~~

Generate synthetic data with factor structure for TROP testing.

.. autofunction:: diff_diff.generate_factor_data

generate_panel_data
~~~~~~~~~~~~~~~~~~~

Generate generic synthetic panel data.

.. autofunction:: diff_diff.generate_panel_data

generate_continuous_did_data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate synthetic continuous treatment DiD data with known dose-response.

.. autofunction:: diff_diff.generate_continuous_did_data

generate_reversible_did_data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate synthetic **reversible-treatment** panel data — treatment can switch on
and off over time. Use this with :class:`~diff_diff.ChaisemartinDHaultfoeuille`
for testing the dCDH estimator on non-absorbing treatments.

.. autofunction:: diff_diff.generate_reversible_did_data

Example
^^^^^^^

.. code-block:: python

   from diff_diff import generate_reversible_did_data, ChaisemartinDHaultfoeuille

   data = generate_reversible_did_data(
       n_groups=80,
       n_periods=6,
       pattern="single_switch",  # or "joiners_only", "leavers_only", "mixed_single_switch"
       treatment_effect=2.0,
       seed=42,
   )

   est = ChaisemartinDHaultfoeuille()
   results = est.fit(
       data, outcome="outcome", group="group",
       time="period", treatment="treatment",
   )

generate_synthetic_control_data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate a **single-treated-unit** panel for synthetic control demos. A factor model
places the treated unit's *noiseless* trajectory in the span of the donor trajectories
(so a synthetic control reproduces it closely — the observed fit is approximate under
the added transitory/predictor noise) and injects a known ramping or constant treatment
effect. Use this with :class:`~diff_diff.SyntheticControl`.

.. autofunction:: diff_diff.generate_synthetic_control_data

Example
^^^^^^^

.. code-block:: python

   from diff_diff import generate_synthetic_control_data, SyntheticControl

   data = generate_synthetic_control_data(
       n_donors=10,
       n_pre=20,
       n_post=4,
       effect_type="ramp",
       seed=0,
   )

   res = SyntheticControl(n_starts=1, seed=0).fit(
       data, outcome="outcome", treatment="treatment",
       unit="unit", time="period", predictors=["x1", "x2", "x3"],
   )
   print(round(res.att, 2), round(res.pre_rmspe, 2))

Indicator Creation
------------------

make_treatment_indicator
~~~~~~~~~~~~~~~~~~~~~~~~

Create binary treatment indicator from categorical or numeric columns.

.. autofunction:: diff_diff.make_treatment_indicator

Example
^^^^^^^

.. code-block:: python

   from diff_diff import make_treatment_indicator

   # From categorical
   data = make_treatment_indicator(
       data,
       column='group',
       treated_values='treatment'
   )

   # From numeric threshold
   data = make_treatment_indicator(
       data,
       column='exposure',
       threshold=0.5,
       new_column='high_exposure'
   )

make_post_indicator
~~~~~~~~~~~~~~~~~~~

Create post-treatment period indicator.

.. autofunction:: diff_diff.make_post_indicator

Example
^^^^^^^

.. code-block:: python

   from diff_diff import make_post_indicator

   data = make_post_indicator(
       data,
       time_column='period',
       treatment_start=5
   )

Panel Data Utilities
--------------------

wide_to_long
~~~~~~~~~~~~

Reshape wide panel data to long format.

.. autofunction:: diff_diff.wide_to_long

Example
^^^^^^^

.. code-block:: python

   from diff_diff import wide_to_long

   # Wide format: each column is a time period
   # unit_id, y_2019, y_2020, y_2021, y_2022
   long_data = wide_to_long(
       wide_data,
       id_col='unit_id',
       value_name='outcome',
       var_name='year'
   )

balance_panel
~~~~~~~~~~~~~

Balance panel data by filling or dropping incomplete observations.

.. autofunction:: diff_diff.balance_panel

Example
^^^^^^^

.. code-block:: python

   from diff_diff import balance_panel

   # Fill missing periods with NaN
   balanced = balance_panel(
       data,
       unit_column='unit_id',
       time_column='period',
       method='fill'
   )

   # Or keep only units with all periods (default)
   balanced = balance_panel(
       data,
       unit_column='unit_id',
       time_column='period',
       method='inner'
   )

Staggered Adoption Utilities
----------------------------

create_event_time
~~~~~~~~~~~~~~~~~

Create event-time column for staggered adoption designs.

.. autofunction:: diff_diff.create_event_time

Example
^^^^^^^

.. code-block:: python

   from diff_diff import create_event_time

   data = create_event_time(
       data,
       time_column='period',
       treatment_time_column='first_treat'
   )

   # event_time = period - first_treat
   # Negative values: pre-treatment
   # Zero: treatment period
   # Positive values: post-treatment
   # NaN for never-treated

aggregate_to_cohorts
~~~~~~~~~~~~~~~~~~~~

Aggregate unit-level data to cohort means.

.. autofunction:: diff_diff.aggregate_to_cohorts

Example
^^^^^^^

.. code-block:: python

   from diff_diff import aggregate_to_cohorts

   cohort_data = aggregate_to_cohorts(
       data,
       unit_column='unit_id',
       time_column='period',
       treatment_column='first_treat',
       outcome='outcome'
   )

Survey Aggregation
------------------

aggregate_survey
~~~~~~~~~~~~~~~~

Aggregate survey microdata to geographic-period cells with design-based precision.

.. autofunction:: diff_diff.aggregate_survey

Example
^^^^^^^

.. code-block:: python

   from diff_diff import aggregate_survey, SurveyDesign, DifferenceInDifferences

   # Define the survey design for the microdata
   design = SurveyDesign(weights="finalwt", strata="strat", psu="psu")

   # Aggregate to state-year panel with design-based SEs
   panel, stage2 = aggregate_survey(
       microdata,
       by=["state", "year"],
       outcomes="smoking_rate",
       covariates=["age", "income"],
       survey_design=design,
   )

   # panel has: state, year, smoking_rate_mean, smoking_rate_se,
   #   smoking_rate_n, smoking_rate_precision, smoking_rate_weight,
   #   age_mean, income_mean, cell_n, cell_n_eff, cell_sum_w, srs_fallback
   #
   # *_weight is fit-ready: unit-constant population weight (pweight, default)
   #   or cleaned precision with NaN/Inf -> 0.0 (aweight opt-in).
   # cell_sum_w is a per-cell diagnostic (sum of survey weights per cell).
   # Non-estimable cells and zero-weight geos are dropped automatically.

   # stage2 is pre-configured: pweights + state-level clustering
   # Add treatment/time indicators at the panel level, then fit:
   # panel["treated"] = ...  # from policy adoption data
   # panel["post"] = (panel["year"] >= treatment_year).astype(int)
   # result = DifferenceInDifferences().fit(
   #     panel, outcome="smoking_rate_mean",
   #     treatment="treated", time="post", survey_design=stage2,
   # )

Weight calibration with balance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``SurveyDesign`` expects **pre-calibrated** weights: post-stratification,
raking, and calibration are deliberately out of scope for diff-diff and
remain upstream. Meta's `balance <https://import-balance.org/>`_ package
(>= 0.21) is the recommended companion - it rakes survey samples to
population margins and ships a dedicated adapter,
``balance.interop.diff_diff``, that hands the calibrated sample straight
to diff-diff (``to_survey_design`` / ``to_panel_for_did`` / ``fit_did`` /
``as_balance_diagnostic``; installable via ``pip install "balance[did]"``).

The handoff needs no adapter if you prefer the native seam - calibrated
weights are just a column::

   design = SurveyDesign(weights="raked_wt", strata="strat", psu="psu")
   panel, stage2 = aggregate_survey(
       microdata, by=["state", "year"], outcomes="smoking_rate",
       survey_design=design,
   )
   result = CallawaySantAnna().fit(
       panel, outcome="smoking_rate_mean", unit="state", time="year",
       first_treat="g", survey_design=stage2,
   )

**When calibration matters for the causal estimand** (not just
descriptives): non-response drift that is differential by treatment arm
and time does *not* difference out of a DiD. See the
:doc:`composition-drift tutorial <../tutorials/26_composition_drift_calibration>`
for a worked BRFSS-style failure mode - including why raking granularity
must match the comparison units (state-level raking, as BRFSS itself
does, not a pooled national rake) - and the companion
`balance tutorial <https://github.com/facebookresearch/balance/blob/main/tutorials/balance_diff_diff_brfss.ipynb>`_
for the robust case (common drift) and descriptive-estimand repair.

Data Validation
---------------

validate_did_data
~~~~~~~~~~~~~~~~~

Validate data structure for DiD analysis.

.. autofunction:: diff_diff.validate_did_data

Example
^^^^^^^

.. code-block:: python

   from diff_diff import validate_did_data

   result = validate_did_data(
       data,
       outcome='outcome',
       treatment='treated',
       time='period',
       unit='unit_id'
   )

   if not result['valid']:
       for error in result['errors']:
           print(f"Error: {error}")
       for warning in result['warnings']:
           print(f"Warning: {warning}")

summarize_did_data
~~~~~~~~~~~~~~~~~~

Generate summary statistics for DiD data.

.. autofunction:: diff_diff.summarize_did_data

Example
^^^^^^^

.. code-block:: python

   from diff_diff import summarize_did_data

   summary = summarize_did_data(
       data,
       outcome='outcome',
       treatment='treated',
       time='period',
       unit='unit_id'
   )

   print(summary)

Control Unit Selection
----------------------

rank_control_units
~~~~~~~~~~~~~~~~~~

Rank control units by suitability for DiD or synthetic control.

.. autofunction:: diff_diff.rank_control_units

Example
^^^^^^^

.. code-block:: python

   from diff_diff import rank_control_units, generate_did_data

   panel = generate_did_data(n_units=100, n_periods=10, treatment_effect=2.0)
   ranked = rank_control_units(
       panel,
       unit_column='unit',
       time_column='period',
       outcome_column='outcome',
       treatment_column='treated',
       pre_periods=[0, 1, 2, 3, 4]
   )

   # Select top 10 control units
   best_controls = ranked.head(10)['unit'].tolist()
