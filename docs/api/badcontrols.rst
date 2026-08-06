Bad-Control DiD
===============

The ``badcontrols`` compatibility layer implements the linear, two-period
imputation estimator and the parametric doubly robust score from Caetano,
Callaway, Payne, and Sant'Anna (2026).  The imputation path first imputes the
untreated evolution of a treatment-affected covariate among controls, then
estimates the outcome trend using that imputed counterfactual covariate.

The parametric path is selected with ``est_method="dr_ml",
nuisance_method="parametric"``.  The random-forest cross-fitted path is
intentionally not substituted silently: requesting ``nuisance_method="ml"``
raises ``NotImplementedError`` until its inference contract is ported and
validated.

The implemented linear path is checked against the installed R
``badcontrols`` package on a shared fixture, including its two-step influence
function standard error.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   diff_diff.didbc
   diff_diff.imputation_bad_control
   diff_diff.extract_att
   diff_diff.BadControlsResult
