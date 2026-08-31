TWFE Weight Diagnostics (Callaway ``twfeweights``)
===================================================

What a two-way fixed effects regression *implicitly* weights.

Run on staggered-adoption data, a TWFE regression does not estimate a simple
average of the underlying group-time effects ATT(g, t). It estimates a
weighted average, and some of those weights can be **negative** -- so the
coefficient need not lie in the convex hull of the effects it summarizes.
This module reports those weights, next to the weights the target estimands
ATT\ :sup:`O` and ATT\ :sup:`simple` would use, and decomposes the regression
back into its building blocks.

**When to use these diagnostics:**

- You have a staggered design and want to see, cell by cell, what your TWFE
  specification is actually averaging
- You want to quantify how much of a TWFE estimate comes from *pre-treatment*
  cells -- i.e. from parallel-trends violations rather than from treatment
- Your TWFE and :class:`~diff_diff.CallawaySantAnna` estimates disagree and
  you want to see which cells drive the gap
- You adjusted for covariates and want to check whether the regression's
  implicit weights actually *balance* them

**How this differs from the neighbouring surfaces:**

- :func:`diff_diff.twowayfeweights` implements de Chaisemartin &
  D'Haultfoeuille (2020) Theorem 1 and weights **(unit, time) cells**. The
  functions here weight **ATT(g, t) parameters**.
- :class:`diff_diff.BaconDecomposition` decomposes TWFE into **2x2 DiD
  comparisons**. :func:`diff_diff.decompose_twfe_weights` decomposes it into
  **group-time effects**, plus a pre-trend-violation term.

**Reference:** Baker, A., Callaway, B., Cunningham, S., Goodman-Bacon, A., &
Sant'Anna, P. H. C. (2025). Difference-in-Differences Designs: A
Practitioner's Guide. arXiv:2503.13323. Callaway, B., & Sant'Anna, P. H. C.
(2021) for the ATT\ :sup:`O` / ATT\ :sup:`simple` weights.

Ported from the ``twfeweights`` R package (v0.9.0) by Brantly Callaway, MIT
License, Copyright (c) 2023 Brantly Callaway.

.. module:: diff_diff.twfe_weights

attgt_weights
-------------

Weights an estimand places on each group-time effect.

.. autofunction:: diff_diff.attgt_weights

decompose_twfe_weights
----------------------

Decomposition of a TWFE estimate into weighted group-time effects.

.. autofunction:: diff_diff.decompose_twfe_weights

plot_twfe_weights
-----------------

.. autofunction:: diff_diff.plot_twfe_weights

Result Objects
--------------

.. autoclass:: diff_diff.ATTGTWeightsResult
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: diff_diff.TWFEDecompositionResult
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Example Usage
-------------

Inspecting what a TWFE regression weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import diff_diff

   panel = diff_diff.load_mpdta()

   cs = diff_diff.CallawaySantAnna(
       control_group="never_treated",
       base_period="universal",   # required for aggregation="twfe"
   ).fit(
       panel, outcome="lemp", unit="countyreal", time="year",
       first_treat="first.treat",
   )

   weights = diff_diff.attgt_weights(cs, aggregation="twfe")
   print(weights.summary())
   print(weights.n_negative, "cells carry negative weight")

Comparing against the estimand you meant to report
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``aggregation="overall"`` and ``"simple"`` give the Callaway & Sant'Anna
target-parameter weights, which are non-negative and sum to one. The gap
between ``implied_att`` values is the cost of the TWFE specification:

.. code-block:: python

   for aggregation in ("twfe", "overall", "simple"):
       w = diff_diff.attgt_weights(cs, aggregation=aggregation)
       print(f"{aggregation:8s} {w.implied_att: .4f}  "
             f"({w.n_negative} negative weights)")

Separating treatment effects from pre-trend violations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   decomposition = diff_diff.decompose_twfe_weights(
       panel,
       outcome="lemp", unit="countyreal", time="year",
       first_treat="first.treat",
       covariates=["lpop"],
       balance_covariates=["lpop"],
   )

   print(decomposition.summary())
   print("from pre-treatment cells:", decomposition.pretrend_bias)

   # Do the implicit weights balance the covariates?
   print(decomposition.covariate_balance())

   diff_diff.plot_twfe_weights(decomposition)

Validation
----------

Validated against R ``twfeweights`` 0.9.0 output on three fixtures (``mpdta``
plus two simulated panels). Goldens live at
``benchmarks/data/twfeweights_golden.json`` and are regenerated with
``Rscript benchmarks/R/generate_twfeweights_golden.R``; R is never needed to
run the test suite. Tolerances and their rationale are in
``docs/methodology/REGISTRY.md`` under "TWFE Weight Diagnostics".
