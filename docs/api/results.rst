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
representation. On calendar-time surfaces (``TwoWayFixedEffects`` event-study
mode) ``EventStudyResults.period_effects`` is a deprecated compatibility view
of the removed ``MultiPeriodDiDResults.period_effects`` (FutureWarning, ledger
M-016, removed in 5.0): a dict keyed by calendar period with the reference
period omitted; ``to_dataframe()`` is the successor.

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

.. autoclass:: diff_diff.AggregationResult
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
