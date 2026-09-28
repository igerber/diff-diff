Regression Discontinuity
========================

Regression discontinuity estimation - sharp and fuzzy, with optional
covariate adjustment - and robust bias-corrected inference,
parity-targeting R ``rdrobust`` 4.0.0.

**Sharp** (default): treatment is assigned by a known threshold of an
observed running variable (``running >= cutoff``; units exactly at the
cutoff are treated, matching rdrobust). The effect is the jump in the
conditional expectation of the outcome at the cutoff. **Fuzzy** (pass the
observed take-up column via ``fit(..., takeup=...)``): crossing
the cutoff shifts take-up instead of determining it, and the estimand is
the local Wald ratio - for binary take-up under monotonicity, the LATE
for compliers at the cutoff; for non-binary take-up, the ratio of jumps
(the ``estimand`` field says which) - with the first stage exposed as a
full ``first_stage*`` block.
Both designs use kernel-weighted local polynomials on each side with
data-driven MSE/CER-optimal bandwidths (all 10 rdrobust selectors) and
robust bias-corrected inference per Calonico, Cattaneo & Titiunik (2014).
**Covariate adjustment** (``fit(..., covariates=[...])``; R's ``covs=``,
per Calonico, Cattaneo, Farrell & Titiunik 2019): additive
common-coefficient adjustment that leaves the estimand UNCHANGED -
unlike the DiD estimators' conditional-parallel-trends role, RD
covariates buy precision only, and require covariate balance at the
cutoff (testable by fitting each covariate as the outcome). Bandwidths
are covariate-aware; collinear covariates are dropped with a warning
naming them (``covs_drop=True``, R's default).

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

   **Scope of this release.** Sharp, fuzzy, and covariate-adjusted
   designs with the nearest-neighbor variance estimator (rdrobust's
   default), plus the ``RDPlot`` and ``RDDensityTest`` diagnostics
   below. Cluster-robust variance, weights, kink estimands,
   weak-IV-robust fuzzy inference, and a packaged covariate-balance
   helper are documented follow-ups - see the methodology registry for
   the full deviations and seams list.

**RD plots** (``RDPlot``; Calonico, Cattaneo & Titiunik 2015, parity with
R's ``rdplot()``): the standard exploratory companion - per-side global
polynomial fits plus binned local means with a data-driven number of bins
(all 8 ``binselect`` selectors: evenly/quantile spaced, IMSE-optimal or
mimicking-variance, spacings or polynomial-regression variance estimators),
implied-scale/WIMSE-weight reporting, optional covariate adjustment, and an
optional matplotlib rendering via ``RDPlotResult.plot()`` (matplotlib is
not a dependency; the numbers are the parity surface).

RegressionDiscontinuity
-----------------------

.. autoclass:: diff_diff.RegressionDiscontinuity
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

RegressionDiscontinuityResults
------------------------------

.. autoclass:: diff_diff.RegressionDiscontinuityResults
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

RDPlot
------

.. autoclass:: diff_diff.RDPlot
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

RDPlotResult
------------

.. autoclass:: diff_diff.RDPlotResult
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

**Manipulation testing** (``RDDensityTest``; Cattaneo, Jansson & Ma 2020,
parity with R's ``rddensity()`` 3.0): the density-discontinuity
(McCrary-style) manipulation test - the running variable's density is
estimated on each side of the cutoff with the CJM boundary-adaptive local
polynomial density estimator and the difference is t-tested with robust
bias-corrected inference. Unrestricted and restricted models, jackknife
(default) and plug-in variances, the full ``each``/``diff``/``sum``/``comb``
data-driven bandwidth menu, and R-parity mass-point adjustment via the
RD-family ``masspoints=`` surface.

RDDensityTest
-------------

.. autoclass:: diff_diff.RDDensityTest
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

RDDensityTestResult
-------------------

.. autoclass:: diff_diff.RDDensityTestResult
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
