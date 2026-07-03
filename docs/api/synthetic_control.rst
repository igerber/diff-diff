Synthetic Control Method (SCM)
==============================

Classic synthetic control estimator for a single treated unit (Abadie, Diamond &
Hainmueller 2010; originating in Abadie & Gardeazabal 2003).

The treated unit's counterfactual is a convex combination of "donor" (never-treated)
units. Donor weights ``W*(V)`` solve a simplex-constrained, predictor-importance-weighted
least-squares fit of the treated unit's pre-period predictors; the diagonal
predictor-importance matrix ``V`` is chosen data-driven (minimizing pre-period outcome
MSPE, ``v_method="nested"``; out-of-sample cross-validation, ``v_method="cv"``; or
closed-form inverse-variance, ``v_method="inverse_variance"``) or supplied by the user
(``v_method="custom"``). The
treatment-effect path is the gap :math:`\hat{\alpha}_{1t} = Y_{1t} - \sum_j w_j Y_{jt}`
over the post periods.

**When to use SCM:**

- Exactly **one treated unit** with a long, well-fit pre-treatment period.
- A curated **donor pool** of comparable never-treated units.
- Aggregate / few-unit comparative case studies (states, regions, countries).

**Inference:** classic SCM has **no analytical standard error**. ``se``, ``t_stat``,
``p_value`` and ``conf_int`` are always NaN; ``att`` (the mean post-period gap) is the
reported estimate. Significance comes from **in-space placebo permutation inference** via
:meth:`~diff_diff.SyntheticControlResults.in_space_placebo` (post/pre RMSPE-ratio statistic,
``placebo_p_value = rank/(n_placebos+1)``). This permutation p-value is a separate field
from the (NaN) ``p_value``; ``is_significant`` stays bound to ``p_value``.

**Robustness diagnostics (ADH 2015 §4, opt-in):**
:meth:`~diff_diff.SyntheticControlResults.leave_one_out` drops each reportably-weighted (weight > 1e-6)
donor and re-fits (per-drop ATT / ``delta_att`` table — a large ``delta_att`` flags
single-donor dependence);
:meth:`~diff_diff.SyntheticControlResults.in_time_placebo` reassigns the intervention to an
earlier pre-date and checks for a spurious gap before the true treatment date (the
backdating placebo; ``placebo_att`` should be ~0);
:meth:`~diff_diff.SyntheticControlResults.regression_weights` computes the implied
regression-counterfactual donor weights ``W^reg`` (intercept-augmented) and flags those
outside ``[0, 1]`` — the extrapolation an OLS counterfactual would incur but the
simplex-constrained synthetic control cannot (pure linear algebra, no refit); and
:meth:`~diff_diff.SyntheticControlResults.sparse_synthetic_control` exhaustively searches
size-``l`` donor subsets holding ``V`` fixed at the baseline, showing how fit / ATT degrade
as the synthetic is forced sparse. All leave the analytical inference fields NaN.

**Confidence sets by test inversion (Firpo & Possebom 2018 §4, opt-in):**
:meth:`~diff_diff.SyntheticControlResults.test_sharp_null` tests a sharp null
``H_0: alpha_1t = f(t)`` (a scalar constant effect, or a post-period effect path) by
re-ranking the stored in-space placebo gaps — no refits, and ``test_sharp_null(0)`` is
identically ``placebo_p_value`` — and
:meth:`~diff_diff.SyntheticControlResults.confidence_set` (``family="constant"`` or
``"linear"``) inverts that test into a confidence set for the effect path: a
constant-effect interval (Eqs. 15–16) or a linear-slope set (Eqs. 17–18), with the
paper's strict ``p > gamma`` membership (Eq. 14), computed by exact piecewise-constant
breakpoint inversion (or a fixed grid when ``bounds=`` is supplied). The set is
summarized on ``effect_confidence_set`` and returned by
:meth:`~diff_diff.SyntheticControlResults.get_confidence_set_df`; the analytical
``conf_int`` stays NaN (this is a separate permutation set at level ``1 - gamma``,
possibly a set / unbounded / non-contiguous).

**Conformal inference (Chernozhukov-Wüthrich-Zhu 2021, opt-in).** Unlike the Firpo
path (which re-ranks the cross-unit placebo gaps), the conformal layer fits its **own**
time-permutation-invariant constrained-LS proxy (Eqs. 3–4, no ``V``-matrix) under the
null on **all** periods and permutes residuals **over time** for the single treated
unit. :meth:`~diff_diff.SyntheticControlResults.conformal_test` gives a joint sharp-null
p-value for a hypothesized effect trajectory (Eqs. 1–2; statistic order ``q in {1, 2,
inf}``); :meth:`~diff_diff.SyntheticControlResults.conformal_confidence_intervals` gives
pointwise per-period confidence intervals by test inversion (Algorithm 1 — each period
``t`` uses ``Z = (pre-periods, t)``, the other post-periods dropped); and
:meth:`~diff_diff.SyntheticControlResults.conformal_average_effect` gives a confidence
interval for the average post-period effect by collapsing into non-overlapping
``T*``-blocks (Appendix A.1). ``scheme="moving_block"`` (default; valid under serial
dependence) or ``"iid"`` (finer p-values). The most recent run is summarized on
``conformal_inference`` and the inversion grid is on
:meth:`~diff_diff.SyntheticControlResults.get_conformal_grid_df`; the analytical
``conf_int`` stays NaN.

**Distinct from** :class:`~diff_diff.SyntheticDiD` (Arkhangelsky et al. 2021), which adds
time weights and ridge regularization; classic SCM uses **donor weights only** plus the
outer ``V`` search.

**Reference:** Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic Control
Methods for Comparative Case Studies: Estimating the Effect of California's Tobacco
Control Program. *Journal of the American Statistical Association*, 105(490), 493–505.
`doi:10.1198/jasa.2009.ap08746 <https://doi.org/10.1198/jasa.2009.ap08746>`_

SyntheticControl
----------------

Main estimator class for classic synthetic control estimation.

.. autoclass:: diff_diff.SyntheticControl
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   .. rubric:: Methods

   .. autosummary::

      ~SyntheticControl.fit
      ~SyntheticControl.get_params
      ~SyntheticControl.set_params

SyntheticControlResults
-----------------------

Results container for synthetic control estimation.

.. autoclass:: diff_diff.SyntheticControlResults
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::

      ~SyntheticControlResults.in_space_placebo
      ~SyntheticControlResults.get_placebo_df
      ~SyntheticControlResults.leave_one_out
      ~SyntheticControlResults.get_leave_one_out_df
      ~SyntheticControlResults.get_leave_one_out_gaps
      ~SyntheticControlResults.in_time_placebo
      ~SyntheticControlResults.get_in_time_placebo_df
      ~SyntheticControlResults.get_in_time_placebo_gaps
      ~SyntheticControlResults.regression_weights
      ~SyntheticControlResults.get_regression_weights_df
      ~SyntheticControlResults.sparse_synthetic_control
      ~SyntheticControlResults.get_sparse_synthetic_control_df
      ~SyntheticControlResults.get_sparse_synthetic_control_gaps
      ~SyntheticControlResults.test_sharp_null
      ~SyntheticControlResults.confidence_set
      ~SyntheticControlResults.get_confidence_set_df
      ~SyntheticControlResults.conformal_test
      ~SyntheticControlResults.conformal_confidence_intervals
      ~SyntheticControlResults.conformal_average_effect
      ~SyntheticControlResults.get_conformal_grid_df
      ~SyntheticControlResults.get_conformal_ci_df
      ~SyntheticControlResults.summary
      ~SyntheticControlResults.print_summary
      ~SyntheticControlResults.to_dict
      ~SyntheticControlResults.to_dataframe
      ~SyntheticControlResults.get_gap_df
      ~SyntheticControlResults.get_weights_df

Convenience Function
--------------------

.. autofunction:: diff_diff.synthetic_control

Predictors and V selection
--------------------------

Predictor rows of ``X1`` (treated) / ``X0`` (donors) are built, in this canonical row
order (the ordering matches R ``Synth::dataprep``), from:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Argument
     - Meaning
   * - ``predictors`` + ``predictor_window`` + ``predictors_op``
     - Columns averaged over a pre-period window (default: all pre periods).
   * - ``special_predictors``
     - ``(var, periods, op)`` triples, each averaged over its own periods/operator.
   * - ``pre_period_outcomes``
     - Individual pre-period outcomes as predictor rows (``"all"`` or a list). When
       no predictor arguments are given, defaults to all pre-period outcomes.

``v_method="nested"`` selects the diagonal predictor-importance matrix ``V`` by minimizing
the pre-period **outcome** MSPE of ``W*(V)`` over a multistart Nelder-Mead search with a
derivative-free Powell polish. ``v_method="cv"`` selects ``V`` by **out-of-sample
cross-validation** (Abadie-Diamond-Hainmueller 2015; Abadie 2021): the pre-period is split
at ``v_cv_t0`` (default ``len(pre)//2``, i.e. ``t0 = T0/2``) into a training and a validation
window; ``V`` is chosen to minimize the validation-window outcome MSPE of the training-fit
weights, then the final weights are re-estimated on the validation-window predictors. Each
predictor is **re-aggregated** over each window (a separate ``dataprep`` per window, as
ADH 2015's CV does), so it must **span both windows** — the default per-period outcome lags
(single-period) are rejected; pass spanning covariate / multi-period ``special_predictors``
(see ``docs/methodology/REGISTRY.md`` §SyntheticControl).
``v_method="inverse_variance"`` uses the closed-form ``v_h = 1/Var(X_h)`` (variance over
donors+treated; no search), applied to the **raw** predictors — it intentionally bypasses
``standardize`` (inverse-variance weighting *is* the unit-variance rescaling). ``v_method="custom"`` takes a user-supplied ``custom_v``
(one entry per predictor row, trace-normalized) and skips the outer search. ``v_cv_t0``
must be ``None`` unless ``v_method="cv"``.

.. note::

   The predictor standardization (per-row SD over donors+treated, ddof=1) and the
   optimizer are pinned from the R ``Synth`` source — they are not specified in
   Abadie-Diamond-Hainmueller (2010). The outer objective uses all pre periods rather than
   R's ``time.optimize.ssr`` window, so the nested ``V`` differs from R by an
   efficiency-only choice. Predictor/outcome aggregation also **fails closed** on any
   non-finite cell, whereas R ``dataprep`` uses ``na.rm=TRUE`` — restrict
   ``predictor_window`` / ``special_predictors`` periods to where a variable is observed.
   Predictor rows support only **equal-weight** linear combinations (``mean``, ``sum``,
   per-period lags); ADH (2010) §2.3's general weighted form ``Σ_s k_s Y_is`` with
   arbitrary ``k_s`` (and non-linear ops such as ``median``) is not accepted in this
   release. See ``docs/methodology/REGISTRY.md`` §SyntheticControl for all deviation labels.

Example Usage
-------------

Basic usage with covariate and special predictors::

    from diff_diff import SyntheticControl

    scm = SyntheticControl(v_method="nested", seed=0)
    results = scm.fit(
        data,
        outcome="gdpcap",
        treatment="treated",   # absorbing 0/1 indicator
        unit="region",
        time="year",
        predictors=["invest", "school.high"],
        # Set predictor_window explicitly when a covariate is only observed on a
        # subset of the pre periods — the default averages over ALL pre periods and
        # fails closed if any selected cell is non-finite.
        predictor_window=[1964, 1965, 1966, 1967, 1968, 1969],
        special_predictors=[("gdpcap", [1960, 1965, 1969], "mean")],
    )
    results.print_summary()

    # Effect path and donor weights
    gap_df = results.get_gap_df()        # period, gap, phase
    weights_df = results.get_weights_df()  # unit, weight (descending)

Quick estimation with the convenience function::

    from diff_diff import synthetic_control

    results = synthetic_control(
        data, outcome="gdpcap", treatment="treated",
        unit="region", time="year",
    )
    print(f"ATT: {results.att:.3f}, pre-RMSPE: {results.pre_rmspe:.3f}")

In-space placebo permutation inference (opt-in; refits one synthetic control per donor)::

    placebo_df = results.in_space_placebo()       # reassigns treatment to each donor
    print(f"placebo p-value: {results.placebo_p_value:.3f} "
          f"(n_placebos={results.n_placebos})")    # p = rank/(n_placebos+1)
    print(placebo_df)   # per-unit RMSPE-ratio table used for the permutation rank

Supplying a fixed predictor-importance matrix (skips the outer V search)::

    import numpy as np

    scm = SyntheticControl(v_method="custom", custom_v=np.ones(n_predictors))
    results = scm.fit(data, outcome="gdpcap", treatment="treated",
                      unit="region", time="year", predictors=["invest"])

Comparison with Synthetic DiD
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Feature
     - SyntheticControl
     - SyntheticDiD
   * - Unit (donor) weights
     - Simplex, predictor-importance weighted
     - Simplex, ridge-regularized
   * - Time weights
     - None (level matching)
     - Simplex (double difference)
   * - Predictor-importance ``V``
     - Nested / cv / inverse-variance / custom diagonal ``V``
     - No analog
   * - Inference
     - Placebo permutation (no analytical SE)
     - Bootstrap / jackknife / placebo

Use **SCM** for a single treated unit with a long pre-period and a curated donor pool;
use **SDID** when you have several treated units and parallel trends is plausible.
