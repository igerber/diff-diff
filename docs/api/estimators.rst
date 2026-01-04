Estimators
==========

Core estimator classes for Difference-in-Differences analysis.

.. module:: diff_diff.estimators

DifferenceInDifferences
-----------------------

Basic 2x2 DiD estimator.

.. autoclass:: diff_diff.DifferenceInDifferences
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   .. rubric:: Methods

   .. autosummary::

      ~DifferenceInDifferences.fit
      ~DifferenceInDifferences.get_params
      ~DifferenceInDifferences.set_params

TwoWayFixedEffects
------------------

Panel DiD with unit and time fixed effects.

.. autoclass:: diff_diff.TwoWayFixedEffects
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

MultiPeriodDiD
--------------

Event study estimator with period-specific treatment effects.

.. autoclass:: diff_diff.MultiPeriodDiD
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

SyntheticDiD
------------

Synthetic control combined with DiD (Arkhangelsky et al. 2021).

.. autoclass:: diff_diff.SyntheticDiD
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
