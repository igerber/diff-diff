Regression Discontinuity
========================

Regression discontinuity estimation - sharp and fuzzy - with robust
bias-corrected inference, parity-targeting R ``rdrobust`` 4.0.0.

**Sharp** (default): treatment is assigned by a known threshold of an
observed running variable (``running >= cutoff``; units exactly at the
cutoff are treated, matching rdrobust). The effect is the jump in the
conditional expectation of the outcome at the cutoff. **Fuzzy** (pass the
observed take-up column via ``fit(..., treatment_col=...)``): crossing
the cutoff shifts take-up instead of determining it, and the estimand is
the local Wald ratio - for binary take-up under monotonicity, the LATE
for compliers at the cutoff; for non-binary take-up, the ratio of jumps
(the ``estimand`` field says which) - with the first stage exposed as a
full ``first_stage*`` block.
Both designs use kernel-weighted local polynomials on each side with
data-driven MSE/CER-optimal bandwidths (all 10 rdrobust selectors) and
robust bias-corrected inference per Calonico, Cattaneo & Titiunik (2014).

.. note::

   **Canonical inference binding.** The result's ``att``, ``se``,
   ``t_stat``, ``p_value``, and ``conf_int`` are one internally coherent
   row - the ROBUST bias-corrected row (``att`` is the bias-corrected
   estimate - the linearized bias-corrected ratio on fuzzy fits;
   ``conf_int`` is centered on it; ``t_stat == att/se``). The
   ``estimand`` field names what ``att`` measures for the fit at hand.
   R's rdrobust prints the *conventional* estimate as its headline
   coefficient while taking inference from the robust row; that estimate
   is available as ``att_conventional`` with its own full inference row,
   and ``summary()`` prints the familiar three-row rdrobust table (plus
   a first-stage block on fuzzy fits, as R does).

.. note::

   **Scope of this release.** Sharp and fuzzy designs with the
   nearest-neighbor variance estimator (rdrobust's default). Covariate
   adjustment, cluster-robust variance, weights, kink estimands,
   weak-IV-robust fuzzy inference, and the rdplot/density-test
   diagnostics are documented follow-ups - see the methodology registry
   for the full deviations and seams list.

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
