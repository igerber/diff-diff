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

    from diff_diff import DiD, TWFE, EventStudy, SDiD, CS, SA, BJS, DDD, SCM, Bacon

``CDiD``, ``Gardner`` and ``Stacked`` are deprecated since 3.9 (they emit a
``FutureWarning`` and are removed in 4.0) — use ``ContinuousDiD``,
``TwoStageDiD`` and ``StackedDiD``.

.. module:: diff_diff.estimators

DifferenceInDifferences (alias: ``DiD``)
----------------------------------------

Basic 2x2 DiD estimator.

``DifferenceInDifferences.predict()`` is present for sklearn-like
discoverability, but out-of-sample prediction is not currently supported. Use
``results_.fitted_values`` for fitted training-data predictions until a broader
post-estimation result-object contract is designed.

.. autoclass:: diff_diff.DifferenceInDifferences
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   .. rubric:: Methods

   .. autosummary::

      ~DifferenceInDifferences.fit
      ~DifferenceInDifferences.predict
      ~DifferenceInDifferences.get_params
      ~DifferenceInDifferences.set_params

MultiPeriodDiD (alias: ``EventStudy``)
--------------------------------------

Event study estimator with period-specific treatment effects.

.. autoclass:: diff_diff.MultiPeriodDiD
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

TwoWayFixedEffects (alias: ``TWFE``)
-------------------------------------

Panel DiD with unit and time fixed effects.

.. module:: diff_diff.twfe

.. autoclass:: diff_diff.TwoWayFixedEffects
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

SyntheticDiD (alias: ``SDiD``)
------------------------------

Synthetic control combined with DiD (Arkhangelsky et al. 2021).

.. module:: diff_diff.synthetic_did

.. autoclass:: diff_diff.SyntheticDiD
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
