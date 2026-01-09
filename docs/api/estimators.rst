Estimators
==========

Core estimator classes for Difference-in-Differences analysis.

The main estimators module (``diff_diff.estimators``) contains the base classes
``DifferenceInDifferences`` and ``MultiPeriodDiD``. Additional estimators are
organized in separate modules for maintainability:

- ``diff_diff.twfe`` - ``TwoWayFixedEffects`` estimator
- ``diff_diff.synthetic_did`` - ``SyntheticDiD`` estimator

All estimators are re-exported from ``diff_diff.estimators`` and ``diff_diff``
for backward compatibility, so you can import any of them using:

.. code-block:: python

    from diff_diff import DifferenceInDifferences, TwoWayFixedEffects, MultiPeriodDiD, SyntheticDiD

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

MultiPeriodDiD
--------------

Event study estimator with period-specific treatment effects.

.. autoclass:: diff_diff.MultiPeriodDiD
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

TwoWayFixedEffects
------------------

Panel DiD with unit and time fixed effects.

.. module:: diff_diff.twfe

.. autoclass:: diff_diff.TwoWayFixedEffects
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

SyntheticDiD
------------

Synthetic control combined with DiD (Arkhangelsky et al. 2021).

.. module:: diff_diff.synthetic_did

.. autoclass:: diff_diff.SyntheticDiD
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

TripleDifference
----------------

Triple Difference (DDD) estimator for settings where treatment requires two criteria
(Ortiz-Villavicencio & Sant'Anna, 2025).

.. module:: diff_diff.triple_diff

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
~~~~~~~~~~~~~~~~~~~~~~~

Results container for Triple Difference estimation.

.. autoclass:: diff_diff.triple_diff.TripleDifferenceResults
   :members:
   :undoc-members:

Convenience Function
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: diff_diff.triple_difference
