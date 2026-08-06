Bad-Control DiD
===============

The initial ``badcontrols`` compatibility layer implements the linear,
two-period imputation estimator from Caetano, Callaway, Payne, and Sant'Anna
(2026).  It first imputes the untreated evolution of a treatment-affected
covariate among controls, then estimates the outcome trend using that imputed
counterfactual covariate.

The doubly robust machine-learning estimator is intentionally not substituted
silently: requesting ``est_method="dr_ml"`` raises ``NotImplementedError``
until its cross-fitting and inference contract is ported and validated.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   diff_diff.didbc
   diff_diff.imputation_bad_control
   diff_diff.extract_att
   diff_diff.BadControlsResult
