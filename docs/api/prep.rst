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
       treatment_start=5,
       treatment_fraction=0.5,
       noise_sd=1.0
   )

   print(data.head())
   # Columns: unit_id, period, outcome, treated, post

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
   data['treated'] = make_treatment_indicator(
       data,
       column='group',
       treated_value='treatment'
   )

   # From numeric threshold
   data['high_exposure'] = make_treatment_indicator(
       data,
       column='exposure',
       threshold=0.5
   )

make_post_indicator
~~~~~~~~~~~~~~~~~~~

Create post-treatment period indicator.

.. autofunction:: diff_diff.make_post_indicator

Example
^^^^^^^

.. code-block:: python

   from diff_diff import make_post_indicator

   data['post'] = make_post_indicator(
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
       unit='unit_id',
       time='period',
       method='fill'
   )

   # Or drop units with missing periods
   balanced = balance_panel(
       data,
       unit='unit_id',
       time='period',
       method='drop'
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

   data['event_time'] = create_event_time(
       data,
       time_col='period',
       first_treat_col='first_treatment'
   )

   # event_time = period - first_treatment
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
       outcome='outcome',
       time='period',
       cohort='first_treatment',
       agg_func='mean'
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

   is_valid, issues = validate_did_data(
       data,
       outcome='outcome',
       treated='treated',
       post='post',
       unit='unit_id',
       time='period'
   )

   if not is_valid:
       for issue in issues:
           print(f"Issue: {issue}")

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
       treated='treated',
       post='post',
       unit='unit_id',
       time='period'
   )

   print(f"N units: {summary['n_units']}")
   print(f"N periods: {summary['n_periods']}")
   print(f"Treatment fraction: {summary['treatment_fraction']:.1%}")

Control Unit Selection
----------------------

rank_control_units
~~~~~~~~~~~~~~~~~~

Rank control units by suitability for DiD or synthetic control.

.. autofunction:: diff_diff.rank_control_units

Example
^^^^^^^

.. code-block:: python

   from diff_diff import rank_control_units

   ranked = rank_control_units(
       data,
       outcome='outcome',
       unit='unit_id',
       time='period',
       treated='treated',
       pre_periods=4,
       method='correlation'  # or 'rmse'
   )

   # Select top 10 control units
   best_controls = ranked.head(10)['unit_id'].tolist()
