Triple Difference (DDD)
=======================

Triple Difference estimator for designs where treatment requires two criteria.

This module implements the methodology from Ortiz-Villavicencio & Sant'Anna (2025),
which correctly handles covariate adjustment in DDD designs. Unlike naive implementations
that difference two DiDs, this approach provides valid estimates when identification
requires conditioning on covariates.

**When to use DDD instead of DiD:**

DDD allows for violations of parallel trends that are:

- Group-specific (e.g., economic shocks affecting treatment states)
- Partition-specific (e.g., trends affecting women everywhere)

As long as these biases are additive, DDD differences them out. The key assumption
is that the *differential* trend between eligible and ineligible units would be
the same across groups.

**Reference:** Ortiz-Villavicencio, M., & Sant'Anna, P. H. C. (2025). Better Understanding
Triple Differences Estimators. *Working Paper*. `arXiv:2505.09942 <https://arxiv.org/abs/2505.09942>`_

.. module:: diff_diff.triple_diff

TripleDifference
----------------

Main estimator class for Triple Difference designs.

Since 3.9 this class serves BOTH triple-difference designs (ledger row M-013):
the 2x2x2 design, and the staggered-adoption design that
:class:`~diff_diff.StaggeredTripleDifference` used to own. ``first_treat=``
selects the staggered engine; mixing the two designs' parameters raises rather
than guessing. The estimation cores are unchanged - both surfaces share one
engine, so the staggered numbers are identical to the deprecated class's.

.. code-block:: python

   from diff_diff import TripleDifference, generate_ddd_data, generate_staggered_ddd_data

   # 2x2x2 design (unchanged) - columns: outcome, group, partition, time
   df = generate_ddd_data(n_per_cell=100, treatment_effect=2.0, seed=42)
   ddd = TripleDifference(estimation_method="dr")
   res = ddd.fit(df, outcome="outcome", group="group", partition="partition", post="time")

   # staggered adoption - the staggered params are keyword-only
   sdf = generate_staggered_ddd_data(n_units=120, n_periods=8, seed=42)
   sddd = TripleDifference(estimation_method="dr", control_group="not_yet_treated")
   res = sddd.fit(
       sdf,
       outcome="outcome",
       partition="eligibility",
       unit="unit",
       time="period",
       first_treat="first_treat",
       aggregate="event_study",
   )

2x2x2 mode returns :class:`~diff_diff.TripleDifferenceResults`; staggered mode
returns :class:`~diff_diff.StaggeredTripleDiffResults` (the containers unify at
4.0, row M-014). ``cluster=`` gives Liang-Zeger CR1 in 2x2x2 mode and raises in
staggered mode, where unit-level clustering is available through
``n_bootstrap > 0``.

.. autoclass:: diff_diff.TripleDifference
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   .. rubric:: Methods

   .. autosummary::

      ~TripleDifference.fit
      ~TripleDifference.get_params
      ~TripleDifference.set_params

TripleDifferenceResults
-----------------------

Results container for Triple Difference estimation.

.. autoclass:: diff_diff.TripleDifferenceResults
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::

      ~TripleDifferenceResults.summary
      ~TripleDifferenceResults.print_summary
      ~TripleDifferenceResults.to_dict
      ~TripleDifferenceResults.to_dataframe

Convenience Function
--------------------

.. autofunction:: diff_diff.triple_difference

Estimation Methods
------------------

The estimator supports three estimation methods:

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - Method
     - Description
     - When to use
   * - ``"dr"``
     - Doubly robust
     - Recommended. Consistent if either outcome or propensity model is correct
   * - ``"reg"``
     - Regression adjustment
     - Simple outcome regression with full interactions
   * - ``"ipw"``
     - Inverse probability weighting
     - When propensity score model is well-specified

Example Usage
-------------

Basic usage::

    from diff_diff import TripleDifference, generate_ddd_data

    # Synthetic 2x2x2 DDD sample (repeated cross-section - each row is its own
    # unit): group (0/1), partition (0/1), time (0=pre/1=post)
    data = generate_ddd_data(n_per_cell=100, treatment_effect=2.0, seed=42)

    ddd = TripleDifference(estimation_method='dr')
    results = ddd.fit(
        data,
        outcome='outcome',
        group='group',          # 1=state enacted policy, 0=control state
        partition='partition',  # 1=women (affected by policy), 0=men
        post='time'             # 1=post-policy, 0=pre-policy
    )
    results.print_summary()

With covariates::

    from diff_diff import TripleDifference, generate_ddd_data

    data = generate_ddd_data(
        n_per_cell=100, treatment_effect=2.0,
        add_covariates=True, seed=42,
    )

    ddd = TripleDifference(estimation_method='dr')
    results = ddd.fit(
        data,
        outcome='outcome',
        group='group',
        partition='partition',
        post='time',
        covariates=['age', 'education']
    )

Quick one-call estimation (the ``triple_difference()`` wrapper is
deprecated since 3.9 and removed in 4.0)::

    from diff_diff import TripleDifference, generate_ddd_data

    data = generate_ddd_data(n_per_cell=100, treatment_effect=2.0, seed=42)

    results = TripleDifference(estimation_method='dr').fit(
        data,
        outcome='outcome',
        group='group',
        partition='partition',
        post='time',
    )
