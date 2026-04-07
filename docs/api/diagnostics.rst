Diagnostics
===========

Placebo tests and diagnostic tools for validating DiD assumptions.

.. module:: diff_diff.diagnostics

run_placebo_test
----------------

Main dispatcher for running different types of placebo tests.

.. autofunction:: diff_diff.run_placebo_test

placebo_timing_test
-------------------

Test using fake treatment timing.

.. autofunction:: diff_diff.placebo_timing_test

Example
~~~~~~~

.. code-block:: python

   from diff_diff import placebo_timing_test

   # Test if effect exists at a fake treatment time
   result = placebo_timing_test(
       data,
       outcome='y',
       treatment='treated',
       time='period',
       fake_treatment_period=3  # Test earlier period
   )

   print(f"Placebo effect: {result.placebo_effect:.3f}")
   print(f"p-value: {result.p_value:.3f}")

placebo_group_test
------------------

Test using fake treatment groups (DiD on never-treated).

.. autofunction:: diff_diff.placebo_group_test

Example
~~~~~~~

.. code-block:: python

   from diff_diff import placebo_group_test

   # Run DiD among never-treated units
   result = placebo_group_test(
       data,
       outcome='y',
       time='period',
       unit='unit_id',
       fake_treated_units=[10, 11, 12]  # Assign some control units as fake-treated
   )

   # Should find no effect if parallel trends holds
   print(f"Placebo effect: {result.placebo_effect:.3f}")

permutation_test
----------------

Permutation-based inference for treatment effects.

.. autofunction:: diff_diff.permutation_test

Example
~~~~~~~

.. code-block:: python

   from diff_diff import permutation_test, generate_did_data

   panel = generate_did_data(n_units=100, n_periods=10, treatment_effect=2.0)
   result = permutation_test(
       panel,
       outcome='outcome',
       treatment='treated',
       time='post',
       unit='unit',
       n_permutations=1000
   )

   print(f"Permutation p-value: {result.p_value:.3f}")

leave_one_out_test
------------------

Sensitivity analysis removing individual treated units.

.. autofunction:: diff_diff.leave_one_out_test

Example
~~~~~~~

.. code-block:: python

   from diff_diff import leave_one_out_test, generate_did_data

   panel = generate_did_data(n_units=100, n_periods=10, treatment_effect=2.0)
   result = leave_one_out_test(
       panel,
       outcome='outcome',
       treatment='treated',
       time='post',
       unit='unit'
   )

   # Check if results are driven by single units
   loo = result.leave_one_out_effects
   print(f"Effect range: [{min(loo.values()):.3f}, {max(loo.values()):.3f}]")

run_all_placebo_tests
---------------------

Run comprehensive suite of diagnostic tests.

.. autofunction:: diff_diff.run_all_placebo_tests

PlaceboTestResults
------------------

Container for placebo test results.

.. autoclass:: diff_diff.PlaceboTestResults
   :members:
   :undoc-members:
   :show-inheritance:
