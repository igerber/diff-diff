Bad-Control DiD
===============

The ``badcontrols`` compatibility layer implements the linear, two-period
imputation estimator and the parametric doubly robust score from Caetano,
Callaway, Payne, and Sant'Anna (2026).  The imputation path first imputes the
untreated evolution of a treatment-affected covariate among controls, then
estimates the outcome trend using that imputed counterfactual covariate.

The parametric path is selected with ``est_method="dr_ml",
nuisance_method="parametric"``. The random-forest cross-fitted path is
selected with ``nuisance_method="ml"`` and requires the optional
``diff-diff[ml]`` dependency. It uses a seeded, stratified fold assignment.

The implemented linear path is checked against the installed R
``badcontrols`` package on a shared fixture, including its two-step influence
function standard error. Binary bad controls use the logistic first stage and
the Bernoulli-information influence-function correction, also checked against
R.
The same linear imputation is also available for staggered adoption through
the per-``(g,t)`` ``didbc`` loop; joint multi-period inference is still a
separate implementation step.
The parametric and random-forest DR paths also loop over staggered cells; their
joint multi-period inference is not yet combined across cells.
The DR entry point validates ``overlap_threshold`` and ``min_group_size`` and
falls back to imputation when the propensity model is not sufficiently supported.
Set ``bstrap=True`` for a seeded, cohort-stratified empirical bootstrap with
percentile confidence intervals.
The R-style cell wrapper ``dr_ml_attgt`` accepts simple additive formula
strings such as ``xformula="~ x1 + x2"`` and ``bad_control_formula="~ X"``;
simple interactions such as ``xformula="~ x1 * x2"`` are expanded to main
effects and a product column. Arbitrary function transforms are not supported.
Imputation and parametric/ML DR paths accept ``d_covariates`` and
``bad_control_d_covariates`` for post-minus-pre changes.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

    diff_diff.didbc
    diff_diff.dr_ml_attgt
    diff_diff.dr_ml_bad_control
   diff_diff.imputation_bad_control
   diff_diff.staggered_imputation_bad_control
   diff_diff.extract_att
   diff_diff.BadControlsResult
   diff_diff.simulate_bad_controls
