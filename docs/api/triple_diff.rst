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

.. autoclass:: diff_diff.TripleDifference
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

    from diff_diff import TripleDifference

    ddd = TripleDifference(estimation_method='dr')
    results = ddd.fit(
        data,
        outcome='wages',
        group='policy_state',       # 1=state enacted policy, 0=control state
        partition='female',         # 1=women (affected by policy), 0=men
        time='post'                 # 1=post-policy, 0=pre-policy
    )
    results.print_summary()

With covariates::

    results = ddd.fit(
        data,
        outcome='wages',
        group='policy_state',
        partition='female',
        time='post',
        covariates=['age', 'education', 'experience']
    )

Using the convenience function::

    from diff_diff import triple_difference

    results = triple_difference(
        data,
        outcome='wages',
        group='policy_state',
        partition='female',
        time='post',
        estimation_method='dr'
    )
