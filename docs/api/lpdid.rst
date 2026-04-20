LPDiD
=====

`LPDiD` implements the absorbing-treatment main path of Stata's `lpdid`
command in the `diff-diff` estimator interface.

Current support includes:

- event-study and pooled pre/post effects
- clean controls and never-treated controls
- reweighting (`rw`) and common-composition samples (`nocomp`)
- pre-mean differencing (`pmd="max"` or `pmd=k`)
- covariates, outcome lags, first-difference lags, and additional absorbed fixed effects

The implementation is validated against official Stata `lpdid` examples for the
absorbing-treatment path, with additional parity coverage for key regression-
adjustment configurations.

Estimator
---------

.. autoclass:: diff_diff.LPDiD
   :members:
   :undoc-members:
   :show-inheritance:

Results
-------

.. autoclass:: diff_diff.LPDiDResults
   :members:
   :undoc-members:
   :show-inheritance:
