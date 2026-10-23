DiD OVB Sensitivity
===================

``DIDOVBSensitivity`` implements the canonical two-period DiD omitted-variable
bias decomposition from Wang, Sant'Anna, Chernozhukov, and Cinelli. It is a
clean-room Python implementation for `diff-diff`; it is not a translation of
the GPL-3 ``dml.sensemakr`` source.

The estimator accepts a balanced two-period panel, cross-fits a control trend
regression and treatment propensity, and reports the short ATT together with
the paper's estimable scale components. The result object then evaluates
restricted bias bounds and the RV/XRV sensitivity statistics.

Basic usage
-----------

.. code-block:: python

   from diff_diff import DIDOVBSensitivity

   result = DIDOVBSensitivity(n_folds=5, seed=42).fit(
       data,
       outcome="lemp",
       treatment="treated",
       time="year",
       unit="county",
       covariates=["lpop", "white", "pov"],
   )

   print(result.summary())
   bounds = result.bounds(trend_r2=1.0, selection_r2=0.5)
   robustness = result.robustness_value(null_value=0.0)
   contour = result.contour(null_value=0.0)

``trend_r2`` is the residual trend variation explained by an omitted
confounder. ``selection_r2`` is the residual share of treatment-odds variation;
the corresponding selection factor is
``sqrt(selection_r2 / (1 - selection_r2))``. This parameterization follows the
paper's :math:`C_{0D}^2` decomposition and is intentionally different from a
generic regression ``R^2`` argument.

R parity
--------

The independent parity oracle is generated with:

.. code-block:: bash

   Rscript benchmarks/R/generate_did_ovb_parity.R \
       benchmarks/data/did_ovb_r_results.json \
       benchmarks/data/real/mpdta.csv

The committed Python parity test compares the short ATT, scale components,
bounds, RV, and XRV. The fixture uses the 2006--2007 minimum-wage slice of the
public ``mpdta`` panel with the same fixed folds and learner specification in
both languages.

Attribution and licensing
-------------------------

The method is based on Wang et al., *Omitted Variable Bias in
Difference-in-Differences Designs*. The related R package ``dml.sensemakr`` is
GPL-3; its source is not included here. Users should cite the paper and the R
package when comparing results.

.. automodule:: diff_diff.did_ovb
   :members:
   :undoc-members:
