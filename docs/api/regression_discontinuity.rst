Regression Discontinuity (Sharp)
================================

Sharp regression discontinuity estimation with robust bias-corrected
inference, parity-targeting R ``rdrobust`` 4.0.0.

Treatment is assigned by a known threshold of an observed running variable
(``running >= cutoff``; units exactly at the cutoff are treated, matching
rdrobust). The effect is the jump in the conditional expectation of the
outcome at the cutoff, estimated by kernel-weighted local polynomials on
each side with data-driven MSE/CER-optimal bandwidths (all 10 rdrobust
selectors), and reported with robust bias-corrected inference per Calonico,
Cattaneo & Titiunik (2014).

.. note::

   **Canonical inference binding.** The result's ``att``, ``se``,
   ``t_stat``, ``p_value``, and ``conf_int`` are one internally coherent
   row - the ROBUST bias-corrected row (``att`` is the bias-corrected
   estimate; ``conf_int`` is centered on it; ``t_stat == att/se``). R's
   rdrobust prints the *conventional* estimate as its headline coefficient
   while taking inference from the robust row; that estimate is available
   as ``att_conventional`` with its own full inference row, and
   ``summary()`` prints the familiar three-row rdrobust table.

.. note::

   **Scope of this release.** Sharp designs only, with the nearest-neighbor
   variance estimator (rdrobust's default). Fuzzy designs, covariate
   adjustment, cluster-robust variance, weights, kink estimands, and the
   rdplot/density-test diagnostics are documented follow-ups - see the
   methodology registry for the full deviations and seams list.

RegressionDiscontinuity
-----------------------

.. autoclass:: diff_diff.RegressionDiscontinuity
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

RegressionDiscontinuityResults
------------------------------

.. autoclass:: diff_diff.RegressionDiscontinuityResults
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
