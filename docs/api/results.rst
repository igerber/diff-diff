Results Classes
===============

Dataclass containers for estimation results from various estimators.

.. module:: diff_diff.results

DiDResults
----------

Results from basic DifferenceInDifferences estimation.

.. autoclass:: diff_diff.DiDResults
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Attributes

   .. autosummary::

      ~DiDResults.att
      ~DiDResults.se
      ~DiDResults.t_stat
      ~DiDResults.p_value
      ~DiDResults.conf_int
      ~DiDResults.n_obs
      ~DiDResults.is_significant
      ~DiDResults.significance_stars

   .. rubric:: Methods

   .. autosummary::

      ~DiDResults.summary
      ~DiDResults.to_dict
      ~DiDResults.to_dataframe

MultiPeriodDiDResults
---------------------

Results from MultiPeriodDiD event study estimation.

.. autoclass:: diff_diff.MultiPeriodDiDResults
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Attributes

   .. autosummary::

      ~MultiPeriodDiDResults.period_effects
      ~MultiPeriodDiDResults.att
      ~MultiPeriodDiDResults.pre_periods
      ~MultiPeriodDiDResults.post_periods
      ~MultiPeriodDiDResults.reference_period
      ~MultiPeriodDiDResults.interaction_indices
      ~MultiPeriodDiDResults.pre_period_effects
      ~MultiPeriodDiDResults.post_period_effects

PeriodEffect
------------

Container for a single period's treatment effect in event studies.

.. autoclass:: diff_diff.PeriodEffect
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

SyntheticDiDResults
-------------------

Results from SyntheticDiD estimation.

.. autoclass:: diff_diff.SyntheticDiDResults
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Attributes

   .. autosummary::

      ~SyntheticDiDResults.att
      ~SyntheticDiDResults.unit_weights
      ~SyntheticDiDResults.time_weights

Results-contract foundations
----------------------------

Shared bases introduced by the 4.0 API-unification program.
``BaseResults`` is the estimator-results base;
``Diagnostic`` marks diagnostic result containers (which carry no
inference row); ``EventStudyResults`` is the unified per-event-time
representation.

.. autoclass:: diff_diff.BaseResults
   :no-index:
   :members:
   :show-inheritance:

.. autoclass:: diff_diff.Diagnostic
   :no-index:
   :members:
   :show-inheritance:

.. autoclass:: diff_diff.EventStudyResults
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
