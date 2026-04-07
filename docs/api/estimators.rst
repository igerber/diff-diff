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

Most estimators have short aliases (``TROP`` already uses its short canonical name):

.. code-block:: python

    from diff_diff import DiD, TWFE, EventStudy, SDiD, CS, CDiD, SA, BJS, Gardner, DDD, Stacked, Bacon

.. module:: diff_diff.estimators

DifferenceInDifferences (alias: ``DiD``)
----------------------------------------

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

MultiPeriodDiD (alias: ``EventStudy``)
--------------------------------------

Event study estimator with period-specific treatment effects.

.. autoclass:: diff_diff.MultiPeriodDiD
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

TwoWayFixedEffects (alias: ``TWFE``)
-------------------------------------

Panel DiD with unit and time fixed effects.

.. module:: diff_diff.twfe

.. autoclass:: diff_diff.TwoWayFixedEffects
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

SyntheticDiD (alias: ``SDiD``)
------------------------------

Synthetic control combined with DiD (Arkhangelsky et al. 2021).

.. module:: diff_diff.synthetic_did

.. autoclass:: diff_diff.SyntheticDiD
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

