Results Classes
===============

Dataclass containers for estimation results from various estimators.

.. module:: diff_diff.results

DiDResults
----------

Results from basic DifferenceInDifferences estimation.

.. autoclass:: diff_diff.DiDResults
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Attributes

   .. autosummary::

      ~DiDResults.att
      ~DiDResults.se
      ~DiDResults.t_stat
      ~DiDResults.p_value
      ~DiDResults.ci
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
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Attributes

   .. autosummary::

      ~MultiPeriodDiDResults.period_effects
      ~MultiPeriodDiDResults.att
      ~MultiPeriodDiDResults.pre_periods
      ~MultiPeriodDiDResults.post_periods

PeriodEffect
------------

Container for a single period's treatment effect in event studies.

.. autoclass:: diff_diff.PeriodEffect
   :members:
   :undoc-members:
   :show-inheritance:

SyntheticDiDResults
-------------------

Results from SyntheticDiD estimation.

.. autoclass:: diff_diff.SyntheticDiDResults
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Attributes

   .. autosummary::

      ~SyntheticDiDResults.att
      ~SyntheticDiDResults.unit_weights
      ~SyntheticDiDResults.time_weights
