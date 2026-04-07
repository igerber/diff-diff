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
