Stacked Difference-in-Differences
==================================

Stacked DiD estimator for staggered adoption designs with corrective Q-weights.

This module implements the methodology from Wing, Freedman & Hollingsworth (2024),
which addresses bias in naive stacked DiD regressions by:

1. **Constructing sub-experiments**: One per adoption cohort with clean controls
2. **Applying corrective Q-weights**: Ensures proper weighting of treatment and
   control group trends across sub-experiments
3. **Running weighted event-study regression**: WLS with Q-weights identifies
   the "trimmed aggregate ATT"

**When to use Stacked DiD:**

- Staggered adoption design with multiple treatment cohorts
- Want an intuitive sub-experiment-based approach (vs. aggregation methods)
- Desire compositional balance: treatment group composition fixed across event times
- Need direct access to the stacked dataset for custom analysis

**Reference:** Wing, C., Freedman, S. M., & Hollingsworth, A. (2024). Stacked
Difference-in-Differences. *NBER Working Paper* 32054.
`<http://www.nber.org/papers/w32054>`_

.. module:: diff_diff.stacked_did

StackedDiD
----------

Main estimator class for Stacked Difference-in-Differences.

.. autoclass:: diff_diff.StackedDiD
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   .. rubric:: Methods

   .. autosummary::

      ~StackedDiD.fit
      ~StackedDiD.get_params
      ~StackedDiD.set_params

StackedDiDResults
-----------------

Results container for Stacked DiD estimation.

.. autoclass:: diff_diff.stacked_did.StackedDiDResults
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::

      ~StackedDiDResults.summary
      ~StackedDiDResults.print_summary
      ~StackedDiDResults.to_dataframe

Convenience Function
--------------------

.. autofunction:: diff_diff.stacked_did

Example Usage
-------------

Basic usage::

    from diff_diff import StackedDiD, generate_staggered_data

    data = generate_staggered_data(n_units=200, n_periods=12,
                                    cohort_periods=[4, 6, 8], seed=42)

    est = StackedDiD(kappa_pre=2, kappa_post=2)
    results = est.fit(data, outcome='outcome', unit='unit',
                      time='period', first_treat='first_treat',
                      aggregate='event_study')
    results.print_summary()

Accessing the stacked dataset::

    # The stacked data is available for custom analysis
    stacked = results.stacked_data
    print(stacked[['unit', 'period', '_sub_exp', '_event_time', '_D_sa', '_Q_weight']].head())

Different weighting schemes::

    # Population-weighted ATT (requires population column)
    est = StackedDiD(kappa_pre=2, kappa_post=2, weighting='population')
    results = est.fit(data, outcome='outcome', unit='unit',
                      time='period', first_treat='first_treat',
                      population='pop_size')

    # Sample-share weighted ATT
    est = StackedDiD(kappa_pre=2, kappa_post=2, weighting='sample_share')
    results = est.fit(data, outcome='outcome', unit='unit',
                      time='period', first_treat='first_treat')

Comparison with Other Staggered Estimators
------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Feature
     - Stacked DiD
     - Callaway-Sant'Anna
   * - Approach
     - Pooled WLS on stacked sub-experiments
     - Separate group-time regressions
   * - Compositional balance
     - Enforced by IC1/IC2 trimming
     - Via balanced event study aggregation
   * - Target parameter
     - Trimmed aggregate ATT
     - Weighted average of ATT(g,t)
   * - Custom analysis
     - Full stacked dataset accessible
     - Group-time effects accessible
   * - Covariates
     - Not yet supported
     - Supported (OR, IPW, DR)
