"""
Result container for the classic Synthetic Control Method (SCM) estimator.

This module contains the ``SyntheticControlResults`` dataclass, extracted from
``synthetic_control.py`` to mirror the TROP estimator/results split.

The classic synthetic control of Abadie, Diamond & Hainmueller (2010) produces a
gap path and donor/predictor weights but **no analytical standard error**.
Accordingly ``se``/``t_stat``/``p_value``/``conf_int`` are always NaN on this
object; the point estimate ``att`` (average post-period gap) is the reported
quantity. Significance comes from in-space placebo permutation inference via
:meth:`SyntheticControlResults.in_space_placebo` (a separate ``placebo_p_value``
field, not the NaN ``p_value``).
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.results import _format_survey_block, _get_significance_stars

__all__ = ["SyntheticControlResults"]


@dataclass
class _SyntheticControlFitSnapshot:
    """Panel state retained for post-hoc in-space placebo refits.

    Holds everything ``SyntheticControlResults.in_space_placebo()`` needs to
    refit ANY donor as the pseudo-treated unit without re-reading the original
    DataFrame. Built in ``SyntheticControl.fit()`` and excluded from pickling by
    ``SyntheticControlResults.__getstate__`` (it retains the full treated+donor
    outcome/predictor panel — a privacy/size hazard if serialized).

    ``specs`` is annotated ``List[Any]`` rather than ``List[_PredictorSpec]`` to
    avoid an import cycle (``_PredictorSpec`` lives in ``synthetic_control.py``,
    which imports this module). ``donor_ids`` is an ORDERED list so the placebo
    iteration order — and therefore the rank / p-value — is deterministic.
    """

    pivots: Dict[str, pd.DataFrame]
    specs: List[Any]
    outcome: str
    all_periods: List[Any]
    pre_periods: List[Any]
    post_periods: List[Any]
    donor_ids: List[Any]
    # The treated unit's reportably-weighted donor support (donor ids with weight above
    # the 1e-6 interpretability floor), FROZEN at fit time and ordered by donor_ids.
    # leave_one_out() iterates this immutable list — NOT the mutable, presentation-level
    # results.donor_weights dict — so post-fit mutation cannot change which donors are
    # dropped, and the robustness result depends only on the fit.
    weighted_donor_ids: List[Any]
    treated_id: Any
    standardize: str
    v_method: str
    custom_v: Optional[Any]
    n_starts: int
    seed: Optional[int]
    optimizer_options: Optional[Dict[str, Any]]
    inner_max_iter: int
    inner_min_decrease: float
    # Training/validation split index for v_method="cv" (positional into pre_periods);
    # None → len(pre_periods)//2 default. Carried so in-space/LOO/in-time placebo refits
    # reproduce the same CV split as the treated fit.
    v_cv_t0: Optional[int]


def _validate_conformal_bounds(
    bounds: Optional[Tuple[float, float]], n_grid: int
) -> Optional[np.ndarray]:
    """Validate an optional ``(lo, hi)`` conformal-CI grid; return the ``linspace`` grid (or None for auto)."""
    if bounds is None:
        return None
    if (
        not isinstance(bounds, (tuple, list, np.ndarray))
        or len(bounds) != 2
        or not all(isinstance(b, (int, float, np.integer, np.floating)) for b in bounds)
        or not all(np.isfinite(float(b)) for b in bounds)
    ):
        raise ValueError(f"bounds must be a finite (lo, hi) pair, got {bounds!r}")
    if float(bounds[1]) <= float(bounds[0]):
        raise ValueError(f"bounds must satisfy hi > lo, got {bounds!r}")
    return np.linspace(float(bounds[0]), float(bounds[1]), int(n_grid))


def _warn_conformal_ci_status(res: Dict[str, Any], method_name: str) -> None:
    """Emit the standard conformal-CI status warning (empty / grid-limited / non-contiguous)."""
    status = res["status"]
    if status == "empty":
        warnings.warn(
            f"{method_name}: confidence interval is empty (every value on the grid is "
            "rejected at this alpha); endpoints are NaN.",
            UserWarning,
            stacklevel=3,
        )
    elif status == "grid_limited":
        warnings.warn(
            f"{method_name}: the accepted set touches a grid edge, so the interval may "
            "extend beyond the scanned grid (grid-limited). Pass explicit bounds= / a wider "
            "grid to widen it.",
            UserWarning,
            stacklevel=3,
        )
    elif not res["contiguous"]:
        warnings.warn(
            f"{method_name}: the accepted set is non-contiguous; [lower, upper] is the hull. "
            "Inspect get_conformal_grid_df().",
            UserWarning,
            stacklevel=3,
        )


@dataclass
class SyntheticControlResults:
    """
    Results from a classic Synthetic Control Method (SCM) estimation.

    Implements Abadie, Diamond & Hainmueller (2010), "Synthetic Control Methods
    for Comparative Case Studies." A single treated unit's counterfactual is the
    convex combination ``Σ_j w_j · Y_jt`` of donor units chosen to match the
    treated unit's pre-period outcomes and predictors; the treatment effect path
    is the gap ``α̂_1t = Y_1t − Σ_j w_j · Y_jt`` over the post periods.

    Attributes
    ----------
    att : float
        Average post-period gap (the reported point estimate). The per-period
        gaps are in ``gap_path``.
    se : float
        Always NaN — classic SCM has no analytical standard error (inference is
        permutation/placebo based; see Abadie-Diamond-Hainmueller 2010 §2.4).
    t_stat, p_value : float
        Always NaN (no analytical SE).
    conf_int : tuple[float, float]
        Always (NaN, NaN) (no analytical SE).
    n_obs : int
        Number of observations (treated + donor rows over all periods) used.
    n_donors : int
        Number of donor units in the (post-filter) donor pool.
    n_pre_periods : int
        Number of pre-treatment periods.
    n_post_periods : int
        Number of post-treatment periods.
    donor_weights : dict
        Mapping ``{donor_unit_id: weight}`` on the unit simplex. Weights below
        the interpretability floor (1e-6) are dropped.
    v_weights : dict
        Mapping ``{predictor_label: v}`` — the diagonal predictor-importance
        matrix V, trace-normalized to sum to 1. On the degenerate **single-donor**
        path (one donor forces ``w=[1]``) V is unidentified — every V yields the same
        synthetic — so ``v_weights`` is **uniform** for every ``v_method`` (including
        ``cv`` / ``inverse_variance``), with a ``UserWarning`` emitted at fit time.
    predictor_balance : pandas.DataFrame
        Predictor-balance table: for each predictor, the treated value, the
        synthetic value (donor-weighted), and the donor-pool mean. Under
        ``v_method="cv"`` the reported ``donor_weights`` come from the ADH-2015 step-4
        refit on the **validation-window** re-aggregated predictors, so the ``treated`` /
        ``synthetic`` / ``donor_mean`` values are reported on that same validation-window
        basis (each spec re-aggregated over ``pre[v_cv_t0:]``) — the row's ``predictor``
        label remains the full spec identity, so it stays aligned with ``v_weights``. For
        every other ``v_method`` the values are the full-pre-period predictor aggregates.
    gap_path : dict
        Mapping ``{period: gap}`` for ALL periods (pre periods carry the fit
        residual used for ``pre_rmspe``; post periods carry the effect path).
    pre_rmspe : float
        Root mean squared prediction error over the pre-treatment periods (the
        primary fit diagnostic).
    mspe_v : float, optional
        The outer-objective value of the selected ``V``: the **pre-period** outcome
        MSPE of ``W*(V*)`` under ``v_method="nested"``, or the held-out
        **validation-window** outcome MSPE under ``v_method="cv"`` (the CV selection
        criterion). None when there is no outer search — the ``v_method="custom"``
        and ``"inverse_variance"`` paths and the degenerate single-donor path. Not
        comparable across ``v_method`` values (different objective windows).
    treated_unit : Any
        The treated unit's identifier.
    pre_periods, post_periods : list
        Calendar-sorted pre / post period values.
    v_method : str
        ``"nested"`` (data-driven V), ``"custom"`` (user-supplied V), ``"cv"``
        (out-of-sample cross-validation V), or ``"inverse_variance"`` (closed-form
        ``1/Var(X)`` V).
    v_cv_t0 : int, optional
        The training/validation split index actually used under ``v_method="cv"``
        (the resolved value — equals ``n_pre_periods // 2`` when the constructor's
        ``v_cv_t0`` was None). None for every other ``v_method``. Survives pickling.
    standardize : str
        ``"std"`` (per-row SD scaling) or ``"none"``.
    alpha : float
        Significance level recorded for downstream (placebo) inference.
    rmspe_ratio : float
        The treated unit's post/pre RMSPE ratio = ``sqrt(MSPE_post / MSPE_pre)`` —
        the in-space placebo test statistic (ADH 2010 §2.4), computed at fit time.
    placebo_p_value : float
        In-space placebo permutation p-value (``rank / (n_placebos + 1)``), NaN
        until :meth:`in_space_placebo` is run. SEPARATE from the (always-NaN)
        analytical ``p_value``; ``is_significant`` stays bound to ``p_value``.
    n_placebos, n_failed, n_infeasible : int
        Donor placebos that entered the permutation reference set / were excluded
        for solver non-convergence / were excluded as structurally infeasible (under
        ``v_method="cv"``, a re-aggregated window with no cross-donor variation once
        that donor is pseudo-treated). All 0 until :meth:`in_space_placebo` is run.
        ``n_infeasible`` mirrors the split :meth:`in_time_placebo` already reports; the
        permutation ``placebo_p_value`` uses only the ``n_placebos`` that entered the
        rank, so it is unaffected by how the excluded remainder is attributed.
    survey_metadata : Any, optional
        Reserved; always None in this release.

    Significance for classic SCM comes from :meth:`in_space_placebo` (opt-in
    in-space placebo permutation inference); :meth:`get_placebo_df` returns the
    per-unit RMSPE-ratio table used for the rank.
    """

    att: float
    se: float
    t_stat: float
    p_value: float
    conf_int: Tuple[float, float]
    n_obs: int
    n_donors: int
    n_pre_periods: int
    n_post_periods: int
    donor_weights: Dict[Any, float]
    v_weights: Dict[str, float]
    predictor_balance: pd.DataFrame
    gap_path: Dict[Any, float]
    pre_rmspe: float
    treated_unit: Any
    pre_periods: List[Any]
    post_periods: List[Any]
    v_method: str
    standardize: str
    alpha: float = 0.05
    mspe_v: Optional[float] = None
    v_cv_t0: Optional[int] = None
    survey_metadata: Optional[Any] = field(default=None)
    # In-space placebo permutation inference (Abadie-Diamond-Hainmueller 2010
    # Section 2.4), populated by ``in_space_placebo()``. ``rmspe_ratio`` (the
    # treated unit's post/pre RMSPE ratio) is computed at fit time; the rest stay
    # at their no-inference defaults until a placebo run. NOTE: the permutation
    # ``placebo_p_value`` is deliberately SEPARATE from ``p_value`` (which stays
    # NaN) — it is not an analytical p-value, has no SE / t-stat, and does not
    # flow through ``safe_inference``. ``is_significant`` likewise stays bound to
    # the (NaN) ``p_value``, NOT ``placebo_p_value``.
    placebo_p_value: float = np.nan
    rmspe_ratio: float = np.nan
    n_placebos: int = 0
    n_failed: int = 0
    # Donor placebos excluded as STRUCTURALLY infeasible (distinct from n_failed's solver
    # non-convergence): under v_method="cv", pseudo-treating a donor can leave a
    # re-aggregated CV window with no cross-donor variation, so the weights are
    # unidentified. 0 until in_space_placebo() runs. Mirrors the split in_time_placebo
    # reports via _in_time_n_infeasible. Excluded from the permutation rank just like
    # n_failed, so placebo_p_value is unaffected by the attribution.
    n_infeasible: int = 0
    # Confidence set for the treatment-effect path by test inversion (Firpo & Possebom
    # 2018, "Synthetic Control Method: Inference, Sensitivity Analysis and Confidence
    # Sets," J. Causal Inference 6(2), §4), populated by ``confidence_set()``. A small
    # summary dict ``{family, parameter, gamma, lower, upper, contiguous, boundary,
    # point_estimate, n_grid, n_placebos, status}``; None until ``confidence_set()`` runs.
    # DELIBERATELY SEPARATE from the always-NaN analytical ``conf_int`` (the Wald interval
    # classic SCM does not have): this is a PERMUTATION set at level ``1-gamma`` (with
    # ``gamma`` granular in ``1/(J+1)``), and may be a set / unbounded / non-contiguous —
    # mirrors how ``placebo_p_value`` is kept distinct from the (NaN) ``p_value``.
    effect_confidence_set: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        # Internal state set per instance by ``fit()`` / ``in_space_placebo()``.
        # Declared here (not as dataclass fields) so ``dataclasses.fields()`` /
        # ``dataclasses.asdict()`` cannot reach the retained panel state.
        # ``_fit_snapshot`` (full panel) and ``_placebo_gaps`` (per-unit gap paths)
        # are panel-derived and nulled on pickle by ``__getstate__``; ``_placebo_df``
        # holds the small per-unit aggregate table returned by ``get_placebo_df()``.
        self._fit_snapshot: Optional[_SyntheticControlFitSnapshot] = None
        self._placebo_gaps: Optional[Dict[Any, Dict[Any, float]]] = None
        self._placebo_df: Optional[pd.DataFrame] = None
        # Whether the treated unit's own inner Frank-Wolfe weight solve converged.
        # in_space_placebo() fails closed when this is False: a truncated treated
        # fit makes the ranked statistic (rmspe_ratio) not a valid SCM optimum.
        self._fit_converged: bool = True
        # Explicit reason an in-space placebo run was infeasible/absent, set by
        # in_space_placebo(). summary() / _scm_native render THIS instead of
        # reconstructing the cause from counts — n_placebos/n_failed alone cannot
        # tell a non-converged treated fit ("treated_fit_nonconverged", n_failed=0)
        # apart from too few donors ("too_few_donors", also n_failed=0). Values:
        # None (not run), "ran", "treated_fit_nonconverged", "too_few_donors",
        # "all_placebos_failed" (every excluded donor was a solver non-convergence),
        # "all_placebos_infeasible" (every excluded donor was structurally infeasible),
        # "all_placebos_unusable" (a MIX of failed + infeasible with none usable) —
        # mirrors the in_time_placebo split. A small string, so it survives pickling.
        self._placebo_status: Optional[str] = None
        # Per-unit floored pre-period denominators (treated + each converged placebo),
        # captured by in_space_placebo() so the sharp-null test inversion
        # (test_sharp_null / confidence_set, Firpo & Possebom 2018) re-ranks against the
        # SAME denominators the placebo run used (the test_sharp_null(0) == placebo_p_value
        # anchor). Each value uses that unit's OWN pre-outcome scale; the pre window is
        # f-free so the denominator is grid-invariant. Small dict → survives pickling.
        self._placebo_pre_denoms: Optional[Dict[Any, float]] = None

        # --- ADH 2015 §4 robustness diagnostics (opt-in, populated by ---
        # --- leave_one_out() / in_time_placebo()). Same panel-vs-scalar split as ---
        # --- the in-space placebo: the small per-row tables (_loo_df / _in_time_df), ---
        # --- scalar summaries and status strings survive pickling; the per-refit ---
        # --- gap-path dicts (_loo_gaps / _in_time_gaps) are panel-derived and nulled ---
        # --- by __getstate__. analytical se/t/p/ci stay NaN throughout.
        self._loo_df: Optional[pd.DataFrame] = None
        self._loo_gaps: Optional[Dict[Any, Dict[Any, float]]] = None
        # Reason a leave-one-out run was infeasible/absent. Values: None (not run),
        # "ran", "treated_fit_nonconverged", "too_few_donors", "all_refits_failed"
        # (all excluded drops were solver non-convergences), "all_refits_infeasible"
        # (all excluded drops were structurally infeasible), "all_refits_unusable" (a
        # MIX with none usable) — mirrors the in_time_placebo split.
        self._loo_status: Optional[str] = None
        # (min, max) ATT across the successful leave-one-out refits (the absolute
        # spread of counterfactual ATTs); None until run.
        self._loo_att_range: Optional[Tuple[float, float]] = None
        # The headline single-donor-dependence number: max |att_loo - baseline_att|
        # over the successful drops. Baseline-RELATIVE, so a uniform shift of every
        # drop away from the baseline is NOT masked the way a narrow raw att_range
        # would be. None until run.
        self._loo_max_abs_delta_att: Optional[float] = None
        self._loo_n_failed: int = 0
        # Leave-one-out drops excluded as STRUCTURALLY infeasible (cv donor-pool
        # indistinguishability), distinct from _loo_n_failed's solver non-convergence.
        # Mirrors _in_time_n_infeasible. 0 until leave_one_out() runs.
        self._loo_n_infeasible: int = 0
        self._in_time_df: Optional[pd.DataFrame] = None
        self._in_time_gaps: Optional[Dict[Any, Dict[Any, float]]] = None
        # Reason an in-time placebo run was infeasible/absent. Values: None (not run),
        # "ran", "treated_fit_nonconverged", "too_few_pre_periods",
        # "all_dates_infeasible", "all_dates_failed", "all_dates_unusable" (a mix of
        # failed + infeasible dates with none usable).
        self._in_time_status: Optional[str] = None
        self._in_time_n_failed: int = 0
        # Number of placebo dates that were dimensionally infeasible (too few pre-fake
        # periods, all predictors dropped, or a zero-mass surviving custom_v). Surfaced
        # alongside _in_time_n_failed so a mixed no-success run reports an accurate mix.
        self._in_time_n_infeasible: int = 0
        # Firpo & Possebom (2018) §4 test-inversion confidence set (opt-in, populated by
        # confidence_set()). The grid table {param, p_value, in_set} is small / NOT
        # panel-derived, so it survives pickling by default (NOT nulled by __getstate__);
        # the public ``effect_confidence_set`` summary dataclass field likewise survives.
        self._confidence_set_df: Optional[pd.DataFrame] = None

        # --- Chernozhukov-Wüthrich-Zhu (2021) conformal inference (opt-in, populated by ---
        # --- conformal_test() / conformal_confidence_intervals() / conformal_average_effect()). ---
        # The public ``conformal_inference`` summary dict + the small ``_conformal_ci_df``
        # (pointwise CI table) and ``_conformal_grid_df`` (inversion grid) are NOT
        # panel-derived, so they survive pickling (NOT nulled by __getstate__). The
        # conformal layer reads the donor outcome panel from ``_fit_snapshot`` (already
        # nulled on pickle), so an unpickled result fails closed in those methods. The
        # analytical ``se``/``t_stat``/``p_value``/``conf_int`` stay NaN — the conformal
        # p-value / CI is a separate permutation object (mirrors ``effect_confidence_set``).
        self.conformal_inference: Optional[Dict[str, Any]] = None
        self._conformal_ci_df: Optional[pd.DataFrame] = None
        self._conformal_grid_df: Optional[pd.DataFrame] = None

    def __getstate__(self) -> Dict[str, Any]:
        """Exclude panel-derived internal state from pickling.

        ``_fit_snapshot`` retains the full treated+donor panel and ``_placebo_gaps``
        the per-unit gap paths — both panel-derived, a privacy/size hazard if the
        pickle is sent elsewhere. The scalar placebo fields (``placebo_p_value``,
        ``rmspe_ratio``, ``n_placebos``, ``n_failed``, ``n_infeasible``) and the small
        ``_placebo_df`` aggregate table survive. An unpickled result keeps all public
        fields; a diagnostic call that needs the snapshot (``in_space_placebo``) then
        raises a ValueError directing the user to re-fit. Mirrors ``SyntheticDiDResults``.
        """
        state = self.__dict__.copy()
        state["_fit_snapshot"] = None
        state["_placebo_gaps"] = None
        # ADH-2015 diagnostic gap paths are panel-derived (same hazard as
        # _placebo_gaps); the small _loo_df / _in_time_df tables + scalar summaries
        # survive so a round-tripped result still reports the diagnostic, but the
        # overlay gap accessors raise (re-fit to recompute).
        state["_loo_gaps"] = None
        state["_in_time_gaps"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore pickled state, backfilling scalar diagnostic fields added later.

        Unpickling bypasses ``__init__`` / ``__post_init__``, so a pickle written by an
        OLDER version (before ``n_infeasible`` / ``_loo_n_infeasible`` existed) would
        otherwise leave those attributes unset and make ``summary()`` / ``to_dict()`` /
        ``DiagnosticReport`` raise ``AttributeError``. Default any missing counter to 0
        (the "no infeasible refits recorded" state) so a legacy result reports cleanly.
        """
        self.__dict__.update(state)
        for _attr, _default in (("n_infeasible", 0), ("_loo_n_infeasible", 0)):
            if not hasattr(self, _attr):
                setattr(self, _attr, _default)

    def __repr__(self) -> str:
        """Concise string representation."""
        return (
            f"SyntheticControlResults(ATT={self.att:.4f}, "
            f"pre_RMSPE={self.pre_rmspe:.4f}, "
            f"n_donors={self.n_donors}, "
            f"v_method={self.v_method!r})"
        )

    @property
    def coef_var(self) -> float:
        """Coefficient of variation: SE / abs(ATT). NaN here (SE is always NaN)."""
        if not (np.isfinite(self.se) and self.se >= 0):
            return np.nan
        if not np.isfinite(self.att) or self.att == 0:
            return np.nan
        return self.se / abs(self.att)

    @property
    def is_significant(self) -> bool:
        """Always False — classic SCM produces no analytical p-value."""
        return bool(np.isfinite(self.p_value) and self.p_value < self.alpha)

    @property
    def significance_stars(self) -> str:
        """Significance stars based on p-value (empty here — p_value is NaN)."""
        return _get_significance_stars(self.p_value)

    def summary(self, alpha: Optional[float] = None) -> str:
        """
        Generate a formatted summary of the estimation results.

        Parameters
        ----------
        alpha : float, optional
            Significance level; defaults to the alpha used during estimation.

        Returns
        -------
        str
            Formatted summary table.
        """
        alpha = alpha or self.alpha

        n_top = min(5, len(self.donor_weights))
        top_donors = sorted(self.donor_weights.items(), key=lambda kv: kv[1], reverse=True)[:n_top]

        lines = [
            "=" * 75,
            "Synthetic Control Method (SCM) Estimation Results".center(75),
            "Abadie, Diamond & Hainmueller (2010)".center(75),
            "=" * 75,
            "",
            f"{'Observations:':<28} {self.n_obs:>10}",
            f"{'Donor units:':<28} {self.n_donors:>10}",
            f"{'Pre-treatment periods:':<28} {self.n_pre_periods:>10}",
            f"{'Post-treatment periods:':<28} {self.n_post_periods:>10}",
            f"{'Treated unit:':<28} {str(self.treated_unit):>10}",
            "",
            "-" * 75,
            "Fit Diagnostics".center(75),
            "-" * 75,
            f"{'Pre-treatment RMSPE:':<28} {self.pre_rmspe:>10.4f}",
            f"{'V selection:':<28} {self.v_method:>10}",
            f"{'Standardization:':<28} {self.standardize:>10}",
        ]
        if self.mspe_v is not None and np.isfinite(self.mspe_v):
            # Under cv, mspe_v is the held-out VALIDATION-window MSPE (the CV selection
            # criterion), not the pre-period objective minimized on the nested path.
            _mspe_label = "Validation MSPE:" if self.v_method == "cv" else "Outer-objective MSPE:"
            lines.append(f"{_mspe_label:<28} {self.mspe_v:>10.6f}")
        if self.v_method == "cv" and self.v_cv_t0 is not None:
            lines.append(f"{'CV train/val split (t0):':<28} {self.v_cv_t0:>10d}")

        if self.survey_metadata is not None:
            lines.extend(_format_survey_block(self.survey_metadata, 75))

        lines.extend(
            [
                "",
                "-" * 75,
                f"{'Top donor weights (w_j)':<40}",
                "-" * 75,
            ]
        )
        for unit_id, w in top_donors:
            lines.append(f"{'  ' + str(unit_id):<40} {w:>10.4f}")

        lines.extend(
            [
                "",
                "-" * 75,
                f"{'Parameter':<15} {'Estimate':>12} {'Std. Err.':>12} "
                f"{'t-stat':>10} {'P>|t|':>10}",
                "-" * 75,
                f"{'ATT (avg gap)':<15} {self.att:>12.4f} {'n/a':>12} " f"{'n/a':>10} {'n/a':>10}",
                "-" * 75,
                "",
            ]
        )
        # Test-inversion confidence set (Firpo & Possebom 2018, §4), if computed. Like the
        # placebo p-value this is permutation-based; the analytical conf_int stays n/a.
        ecs = self.effect_confidence_set
        if ecs is not None:
            fam = ecs["family"]
            param = ecs["parameter"]
            conf_pct = 100.0 * (1.0 - ecs["gamma"])
            lines.append(
                f"Confidence set by test inversion (Firpo-Possebom 2018; {fam} effect "
                f"f(t), parameter {param}):"
            )
            if ecs["status"] == "ran":
                note = "" if ecs["contiguous"] else "  (non-contiguous; [lower, upper] hull)"
                lines.append(
                    f"  {conf_pct:.1f}% set:".ljust(34)
                    + f"[{ecs['lower']:.4f}, {ecs['upper']:.4f}]{note}"
                )
            elif ecs["status"] == "unbounded":
                tail = (
                    " and NON-CONTIGUOUS (hull shown; see get_confidence_set_df())"
                    if not ecs["contiguous"]
                    else ""
                )
                lines.append(
                    "  Unbounded (gamma below the 1/(J+1) granularity, or the treated "
                    f"unit is not the best pre-fit){tail}."
                )
            else:  # "empty"
                lines.append(
                    "  Empty: every effect in this family is rejected at "
                    f"gamma={ecs['gamma']:.3g}."
                )
            lines.extend(
                [
                    "(Permutation-based; the analytical conf_int above stays n/a.)",
                    "-" * 75,
                    "",
                ]
            )
        # Three states: (1) placebo never run -> point to in_space_placebo();
        # (2) run with a valid reference set -> show the permutation p-value;
        # (3) run but infeasible (no placebo entered the rank, e.g. J<2 or all
        # donors failed) -> say so explicitly rather than implying it was not run.
        # ``_placebo_df is not None`` is the "attempted" signal (survives pickling).
        placebo_attempted = self._placebo_df is not None
        if placebo_attempted and np.isfinite(self.placebo_p_value):
            # The classic analytical fields above stay n/a (no SE); this is the
            # permutation p-value of the post/pre RMSPE ratio, p = rank/(n_placebos+1).
            # Excluded donors split into solver failures + structural cv infeasibilities;
            # show the breakdown when any donor was infeasible so the two are not conflated.
            n_excluded = self.n_failed + self.n_infeasible
            if n_excluded and self.n_infeasible:
                excluded_suffix = (
                    f"  ({n_excluded} excluded: {self.n_failed} failed, "
                    f"{self.n_infeasible} infeasible)"
                )
            elif n_excluded:
                excluded_suffix = f"  ({n_excluded} excluded)"
            else:
                excluded_suffix = ""
            lines.extend(
                [
                    "In-space placebo permutation inference "
                    "(Abadie-Diamond-Hainmueller 2010, Section 2.4):",
                    f"{'  RMSPE ratio (post/pre):':<34} {self.rmspe_ratio:>10.4f}",
                    f"{'  Permutation p-value:':<34} {self.placebo_p_value:>10.4f}",
                    f"{'  Placebos in reference set:':<34} {self.n_placebos:>10d}"
                    + excluded_suffix,
                    "",
                    "(Analytical SE is still undefined for classic SCM; the "
                    "p-value above is permutation-based.)",
                    "=" * 75,
                ]
            )
        elif placebo_attempted:
            # Render the SPECIFIC reason recorded by in_space_placebo(); the count
            # fields (n_placebos=0, n_failed=0) cannot tell a non-converged treated
            # fit apart from too-few-donors, so do not reconstruct it from counts.
            status = getattr(self, "_placebo_status", None)
            if status == "treated_fit_nonconverged":
                reason = [
                    "In-space placebo was skipped: the treated unit's own SCM fit "
                    "did not converge at fit time (inner Frank-Wolfe weight solve",
                    "and/or outer V search), so its RMSPE ratio is not a valid "
                    "optimum to rank against placebos. placebo_p_value is undefined",
                    "— re-fit with a larger inner_max_iter / looser "
                    "inner_min_decrease and/or a larger optimizer_options['maxiter']",
                    "/ more n_starts.",
                ]
            elif status == "too_few_donors":
                reason = [
                    "In-space placebo inference requires at least 2 donors (each "
                    "placebo is fit against the other donors); too few were",
                    "available. placebo_p_value is undefined. Inspect " "get_placebo_df().",
                ]
            elif status == "all_placebos_infeasible":
                reason = [
                    "In-space placebo permutation inference was attempted but every "
                    "donor refit was structurally infeasible",
                    f"({self.n_infeasible} of {self.n_donors}; under v_method='cv' the "
                    "pseudo-treated donor pool is indistinguishable in a",
                    "re-aggregated CV window). placebo_p_value is undefined — adjust the "
                    "predictors / v_cv_t0 / donor pool. Inspect get_placebo_df().",
                ]
            elif status == "all_placebos_unusable":
                reason = [
                    "In-space placebo permutation inference was attempted but no donor "
                    "refit was usable",
                    f"({self.n_failed} failed to converge, {self.n_infeasible} "
                    "structurally infeasible under v_method='cv').",
                    "placebo_p_value is undefined. Inspect get_placebo_df().",
                ]
            else:  # "all_placebos_failed" (or a legacy unpickle without the status)
                reason = [
                    "In-space placebo permutation inference was attempted but "
                    "produced no valid reference set",
                    f"(0 placebos entered the rank; {self.n_failed} failed to "
                    "converge). placebo_p_value is undefined — all donor refits",
                    "failed. Inspect get_placebo_df().",
                ]
            lines.extend([*reason, "=" * 75])
        else:
            lines.extend(
                [
                    "Inference: classic SCM has no analytical standard error.",
                    "Run in_space_placebo() for in-space permutation inference",
                    "(Abadie-Diamond-Hainmueller 2010, Section 2.4).",
                    "=" * 75,
                ]
            )

        return "\n".join(lines)

    def print_summary(self, alpha: Optional[float] = None) -> None:
        """Print the summary to stdout."""
        print(self.summary(alpha))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert scalar results to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary of the scalar estimation results (weights/balance/gaps
            are available via the ``get_*_df`` accessors).
        """
        result = {
            "att": self.att,
            "se": self.se,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "conf_int_lower": self.conf_int[0],
            "conf_int_upper": self.conf_int[1],
            "n_obs": self.n_obs,
            "n_donors": self.n_donors,
            "n_pre_periods": self.n_pre_periods,
            "n_post_periods": self.n_post_periods,
            "pre_rmspe": self.pre_rmspe,
            "mspe_v": self.mspe_v,
            "treated_unit": self.treated_unit,
            "v_method": self.v_method,
            "v_cv_t0": self.v_cv_t0,
            "standardize": self.standardize,
            # In-space placebo permutation inference. rmspe_ratio is set at fit;
            # placebo_p_value / n_placebos / n_failed / n_infeasible stay at their
            # no-inference defaults (NaN / 0) until in_space_placebo() runs.
            "rmspe_ratio": self.rmspe_ratio,
            "placebo_p_value": self.placebo_p_value,
            "n_placebos": self.n_placebos,
            "n_failed": self.n_failed,
            "n_infeasible": self.n_infeasible,
        }
        # Test-inversion confidence set (Firpo & Possebom 2018), flattened to scalars so
        # to_dataframe() stays a single row of scalars; all None until confidence_set()
        # runs. The analytical conf_int_lower/upper above stay NaN (no Wald interval).
        ecs = self.effect_confidence_set
        result["effect_ci_family"] = ecs["family"] if ecs else None
        result["effect_ci_parameter"] = ecs["parameter"] if ecs else None
        result["effect_ci_gamma"] = ecs["gamma"] if ecs else None
        result["effect_ci_lower"] = ecs["lower"] if ecs else None
        result["effect_ci_upper"] = ecs["upper"] if ecs else None
        result["effect_ci_contiguous"] = ecs["contiguous"] if ecs else None
        result["effect_ci_status"] = ecs["status"] if ecs else None
        if self.survey_metadata is not None:
            sm = self.survey_metadata
            result["weight_type"] = sm.weight_type
            result["effective_n"] = sm.effective_n
            result["design_effect"] = sm.design_effect
        return result

    def to_dataframe(self) -> pd.DataFrame:
        """Convert scalar results to a single-row pandas DataFrame."""
        return pd.DataFrame([self.to_dict()])

    def get_gap_df(self) -> pd.DataFrame:
        """
        Get the gap (effect) path as a DataFrame, in calendar order.

        Rebuilt period-keyed from ``gap_path`` using the canonical
        ``pre_periods + post_periods`` order so the row order is independent of
        any dict-insertion order. Columns: ``period``, ``gap``, ``phase``.

        Returns
        -------
        pandas.DataFrame
        """
        rows = []
        for period in list(self.pre_periods) + list(self.post_periods):
            if period in self.gap_path:
                phase = "post" if period in self.post_periods else "pre"
                rows.append({"period": period, "gap": self.gap_path[period], "phase": phase})
        return pd.DataFrame(rows, columns=["period", "gap", "phase"])

    def get_weights_df(self) -> pd.DataFrame:
        """
        Get donor weights as a DataFrame, sorted by weight descending.

        Returns
        -------
        pandas.DataFrame
            Columns: ``unit``, ``weight``.
        """
        items = sorted(self.donor_weights.items(), key=lambda kv: kv[1], reverse=True)
        return pd.DataFrame(
            [{"unit": unit, "weight": w} for unit, w in items],
            columns=["unit", "weight"],
        )

    _PLACEBO_COLS = ["unit", "pre_mspe", "post_mspe", "rmspe_ratio", "is_treated", "status"]

    def get_placebo_df(self) -> pd.DataFrame:
        """
        Get the in-space placebo distribution as a DataFrame (one row per unit).

        This is a per-unit SUMMARY table (one row per unit), enough to reproduce
        the permutation rank and a ratio-distribution plot — NOT the per-period
        placebo gap paths needed for the classic "spaghetti" plot (those are
        retained internally on ``_placebo_gaps`` for the successful placebos).
        Columns: ``unit``, ``pre_mspe``, ``post_mspe``, ``rmspe_ratio``,
        ``is_treated``, ``status`` (``"treated"`` / ``"placebo"`` / ``"failed"``).
        The treated unit is always present as a single ``is_treated=True,
        status="treated"`` row (its ratio is the original J-donor fit). After a
        placebo run **that produced a reference set** (``>= 2`` donors AND a
        converged treated fit), the table has ``n_donors + 1`` rows — every donor
        appears, including those whose refit did not converge (``status="failed"``
        with NaN metrics, excluded from the rank). In the degenerate / fail-closed
        cases (fewer than 2 donors, or a treated fit that did not converge) the
        placebo loop does not run, so only the treated row is returned.

        Populated by :meth:`in_space_placebo`; the summary table is retained on
        pickling, so it is still returned after a round-trip. Before any placebo
        run — including on an unpickled result that never ran one — only the
        treated row is returned.

        Returns
        -------
        pandas.DataFrame
        """
        if self._placebo_df is not None:
            return self._placebo_df.copy()
        from diff_diff.synthetic_control import _mspe

        pre = _mspe(self.gap_path, self.pre_periods)
        post = _mspe(self.gap_path, self.post_periods)
        return pd.DataFrame(
            [
                {
                    "unit": self.treated_unit,
                    "pre_mspe": pre,
                    "post_mspe": post,
                    "rmspe_ratio": self.rmspe_ratio,
                    "is_treated": True,
                    "status": "treated",
                }
            ],
            columns=self._PLACEBO_COLS,
        )

    def in_space_placebo(
        self,
        n_starts: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        In-space placebo permutation inference (Abadie-Diamond-Hainmueller 2010,
        Section 2.4).

        Reassigns the treatment to each donor in turn, re-estimates a synthetic
        control for that pseudo-treated donor against the OTHER donors, and ranks
        the real treated unit's post/pre RMSPE ratio among all units. Populates
        ``placebo_p_value``, ``n_placebos``, ``n_failed`` and ``n_infeasible`` on
        this object (``rmspe_ratio`` — the treated unit's own ratio — is set at fit
        time) and returns the placebo distribution via :meth:`get_placebo_df`.

        The real treated unit is **excluded from every placebo's donor pool**: its
        post-period outcome is treatment-contaminated, so allowing a placebo to
        load weight on it would bias the placebo gap. The ranking set is therefore
        the ``J+1`` units ``{treated} ∪ {J placebos}``, with each placebo fit
        against the other ``J-1`` donors (this matches the standard
        ``SCtools::generate.placebos`` construction). The post/pre RMSPE ratio
        normalizes by pre-treatment fit, which obviates the pre-fit-cutoff
        filtering of ADH Figures 5-7 (journal p. 502), so no pre-fit filter is
        offered — every converged placebo enters the rank.

        The permutation ``placebo_p_value`` is intentionally distinct from
        ``p_value`` (which stays NaN — classic SCM has no analytical SE) and from
        ``is_significant`` (which also stays bound to the NaN ``p_value``).

        A placebo is **excluded** from the reference set for one of two reasons,
        counted separately. A **solver non-convergence** (counted in ``n_failed``,
        ``status="failed"``) is EITHER an inner Frank-Wolfe weight solve that did not
        converge (a truncated ``W`` is unusable) OR an outer ``V`` search that did not
        converge (an under-optimized ``V`` fits the pre-period worse, shrinking its
        RMSPE ratio and biasing the permutation p-value anti-conservatively). A
        **structural cv infeasibility** (counted in ``n_infeasible``,
        ``status="infeasible"``; ``v_method="cv"`` only) is a pseudo-treated donor
        pool that is indistinguishable in a re-aggregated CV window, so the weights are
        unidentified — remedied by adjusting the predictors / ``v_cv_t0`` / donor pool,
        NOT the optimizer budget. Both are excluded from the rank identically, so
        ``placebo_p_value`` is unaffected by the attribution. Each placebo refit
        **inherits the original fit's
        ``optimizer_options`` / ``n_starts``**, so valid inference requires settings
        adequate for the outer ``V`` search to converge: production defaults do;
        with cheap settings, raise ``n_starts`` here or re-fit with a larger
        ``optimizer_options['maxiter']`` (otherwise placebos are dropped as failed).
        The treated unit's own fit is held to the same standard — if its inner OR
        outer search did not converge, the whole run fails closed (see below).

        Parameters
        ----------
        n_starts : int, optional
            Override the multistart count for each placebo's outer V search (nested/cv).
            Default None inherits the original fit's ``n_starts``. The placebo
            loop is the cost driver (one outer V search per donor); lower it for a
            faster, coarser scan.

        Returns
        -------
        pandas.DataFrame
            The placebo distribution (see :meth:`get_placebo_df`).

        Raises
        ------
        ValueError
            If the fit snapshot is unavailable (e.g. this result was unpickled).
        """
        if self._fit_snapshot is None:
            raise ValueError(
                "in_space_placebo() requires the fit snapshot on the results "
                "object. This result appears to have been loaded from "
                "serialization (which excludes the snapshot) or produced by an "
                "older estimator version. Re-fit to enable in-space placebo "
                "inference."
            )
        from diff_diff.synthetic_control import _floored_pre_mspe, _mspe, _placebo_fit_unit

        snap = self._fit_snapshot
        # A rebuilt placebo reference set invalidates any previously computed confidence set
        # (test_sharp_null / confidence_set re-rank against THIS reference set), so drop the
        # cached confidence-set outputs up front — a stale set must never be reported after an
        # explicit in_space_placebo() re-run (e.g. with a different n_starts). The snapshot
        # check above has already passed, so the reference IS about to be rebuilt on every exit.
        self.effect_confidence_set = None
        self._confidence_set_df = None
        donors = list(snap.donor_ids)
        n_donors = len(donors)
        if n_starts is None:
            n_starts_eff = snap.n_starts
        else:
            # Mirror the estimator constructor's validation (synthetic_control.py)
            # so a bad override fails fast instead of silently coercing (e.g. via
            # int(0)/int(-1)) into a degenerate or invalid permutation procedure.
            if not isinstance(n_starts, (int, np.integer)) or n_starts < 1:
                raise ValueError(f"n_starts override must be a positive integer, got {n_starts!r}")
            n_starts_eff = int(n_starts)

        treated_pre = _mspe(self.gap_path, snap.pre_periods)
        treated_post = _mspe(self.gap_path, snap.post_periods)
        treated_ratio = self.rmspe_ratio

        rows: List[Dict[str, Any]] = [
            {
                "unit": snap.treated_id,
                "pre_mspe": treated_pre,
                "post_mspe": treated_post,
                "rmspe_ratio": treated_ratio,
                "is_treated": True,
                "status": "treated",
            }
        ]

        # Fail closed when the treated unit's OWN fit did not converge at fit time
        # (inner Frank-Wolfe weight solve OR outer V search): ranking a statistic
        # from a truncated / under-optimized treated fit would not be a valid ADH
        # 2010 §2.4 permutation (placebos already fail-closed on non-convergence, so
        # the treated unit must too). ``_fit_converged`` folds both failure modes, so
        # the remediation names the knobs for each.
        if not self._fit_converged:
            warnings.warn(
                "In-space placebo skipped: the treated unit's own SCM fit did not "
                "converge at fit time (inner Frank-Wolfe weight solve and/or outer V "
                "search), so its RMSPE ratio is not a valid optimum to rank against "
                "placebos. placebo_p_value is NaN — re-fit with a larger "
                "inner_max_iter / looser inner_min_decrease (inner) and/or a larger "
                "optimizer_options['maxiter'] / more n_starts (outer V search).",
                UserWarning,
                stacklevel=2,
            )
            self.placebo_p_value = np.nan
            self.n_placebos = 0
            self.n_failed = 0
            self.n_infeasible = 0
            self._placebo_gaps = {}
            self._placebo_pre_denoms = {}
            self._placebo_status = "treated_fit_nonconverged"
            self._placebo_df = pd.DataFrame(rows, columns=self._PLACEBO_COLS)
            return self._placebo_df.copy()

        if n_donors < 2:
            warnings.warn(
                "In-space placebo inference requires at least 2 donors (each "
                f"placebo is fit against the other donors); only {n_donors} "
                "available. placebo_p_value is NaN.",
                UserWarning,
                stacklevel=2,
            )
            self.placebo_p_value = np.nan
            self.n_placebos = 0
            self.n_failed = 0
            self.n_infeasible = 0
            self._placebo_gaps = {}
            self._placebo_pre_denoms = {}
            self._placebo_status = "too_few_donors"
            self._placebo_df = pd.DataFrame(rows, columns=self._PLACEBO_COLS)
            return self._placebo_df.copy()

        if n_donors == 2:
            warnings.warn(
                "In-space placebo with 2 donors: each placebo is fit against a "
                "single donor (degenerate weight w=[1]) with no V search, so the "
                "permutation p-value is coarse (only 2 placebos enter the "
                "reference set; the smallest attainable p-value is 1/3).",
                UserWarning,
                stacklevel=2,
            )

        placebo_gaps: Dict[Any, Dict[Any, float]] = {}
        ranked_ratios: List[float] = []
        n_failed = 0
        n_infeasible = 0

        for j in donors:
            pool = [d for d in donors if d != j]
            fitted, fit_status = _placebo_fit_unit(snap, j, pool, n_starts_eff)
            if fitted is None:
                # Excluded from BOTH the numerator and the denominator (never rank a
                # non-optimal fit). "failed" (a truncated inner W / outer V search) and
                # "infeasible" (a structural cv donor-indistinguishability for this
                # pseudo-treated pool) are dropped alike but COUNTED separately, mirroring
                # the split in_time_placebo reports. Still record the donor with NaN
                # metrics so get_placebo_df() returns the full treated + every-donor set.
                if fit_status == "infeasible":
                    n_infeasible += 1
                else:
                    n_failed += 1
                rows.append(
                    {
                        "unit": j,
                        "pre_mspe": np.nan,
                        "post_mspe": np.nan,
                        "rmspe_ratio": np.nan,
                        "is_treated": False,
                        "status": fit_status,
                    }
                )
                continue
            gap_path_j, ratio_j = fitted
            placebo_gaps[j] = gap_path_j
            pre_j = _mspe(gap_path_j, snap.pre_periods)
            post_j = _mspe(gap_path_j, snap.post_periods)
            ranked_ratios.append(ratio_j)
            rows.append(
                {
                    "unit": j,
                    "pre_mspe": pre_j,
                    "post_mspe": post_j,
                    "rmspe_ratio": ratio_j,
                    "is_treated": False,
                    "status": "placebo",
                }
            )

        n_placebos = len(ranked_ratios)
        if n_placebos == 0:
            warnings.warn(
                "No in-space placebo entered the reference set (all donors failed to "
                "converge, were structurally infeasible, or were filtered out of "
                f"{n_donors}); placebo_p_value is NaN.",
                UserWarning,
                stacklevel=2,
            )
            p_value = np.nan
        else:
            # Upper-tail rank on the (unsigned) RMSPE ratio, treated unit included
            # as the "+1". Ties counted via ``>=`` so the p-value is conservative.
            # (The ratio squares the gaps -> direction-agnostic, NOT a signed test.)
            rank = 1 + sum(1 for r in ranked_ratios if r >= treated_ratio)
            p_value = rank / (n_placebos + 1)

        # Two distinct exclusion causes, warned separately (mirrors in_time_placebo) so a
        # structural cv exclusion is not mis-attributed to a solver budget the user could
        # raise. Both remain out of the permutation rank; placebo_p_value uses n_placebos.
        if n_infeasible > 0:
            warnings.warn(
                f"{n_infeasible} of {n_donors} in-space placebos were STRUCTURALLY "
                "infeasible under v_method='cv' (the pseudo-treated donor pool is "
                "indistinguishable in a re-aggregated CV window, so the weights are "
                "unidentified) and were excluded with status='infeasible'; remedy by "
                "adjusting the predictors, v_cv_t0, or the donor pool (NOT inner_max_iter "
                f"/ n_starts). placebo_p_value uses the remaining {n_placebos}.",
                UserWarning,
                stacklevel=2,
            )
        if n_failed > 0:
            warnings.warn(
                f"{n_failed} of {n_donors} in-space placebos failed to reach a valid "
                "optimum (a non-converged inner weight solve or outer V search) and were "
                "excluded with status='failed'; raise n_starts or loosen the optimizer "
                f"tolerances. placebo_p_value uses the remaining {n_placebos}.",
                UserWarning,
                stacklevel=2,
            )

        # Persist each unit's floored pre-period denominator (treated + every converged
        # placebo) so the sharp-null test inversion (test_sharp_null / confidence_set,
        # Firpo & Possebom 2018) re-ranks against the SAME denominators this run used —
        # the test_sharp_null(0) == placebo_p_value anchor. The pre window is f-free so the
        # denominator is grid-invariant; each unit's floor uses its OWN pre-outcome scale.
        outcome_pivot = snap.pivots[snap.outcome]
        pre_denoms: Dict[Any, float] = {}
        for unit, gp in [(snap.treated_id, self.gap_path), *placebo_gaps.items()]:
            pre_gaps_u = np.array([gp[p] for p in snap.pre_periods], dtype=float)
            z1_u = outcome_pivot.loc[snap.pre_periods, unit].to_numpy(dtype=float)
            scale_u = float(np.max(np.abs(z1_u))) if z1_u.size else 0.0
            pre_denoms[unit] = _floored_pre_mspe(pre_gaps_u, scale_u)
        self._placebo_pre_denoms = pre_denoms

        self.placebo_p_value = float(p_value)
        self.n_placebos = int(n_placebos)
        self.n_failed = int(n_failed)
        self.n_infeasible = int(n_infeasible)
        self._placebo_gaps = placebo_gaps
        # Classify a no-reference-set run by cause (mirrors in_time_placebo): a pure
        # solver failure ("all_placebos_failed", actionable via n_starts / tolerances) and
        # pure structural infeasibility ("all_placebos_infeasible", actionable via
        # predictors / v_cv_t0 / donor pool) are distinct; a MIX gets "all_placebos_unusable"
        # (both counters surfaced). By this point too-few-donors / non-converged-treated-fit
        # have already returned, so >=1 donor was attempted.
        if n_placebos > 0:
            self._placebo_status = "ran"
        elif n_failed > 0 and n_infeasible > 0:
            self._placebo_status = "all_placebos_unusable"
        elif n_infeasible > 0:
            self._placebo_status = "all_placebos_infeasible"
        else:
            self._placebo_status = "all_placebos_failed"
        self._placebo_df = pd.DataFrame(rows, columns=self._PLACEBO_COLS)
        return self._placebo_df.copy()

    _LOO_COLS = [
        "dropped_unit",
        "att",
        "pre_rmspe",
        "post_rmspe",
        "rmspe_ratio",
        "delta_att",
        "status",
    ]

    def leave_one_out(self, n_starts: Optional[int] = None) -> pd.DataFrame:
        """
        Leave-one-out donor robustness (Abadie-Diamond-Hainmueller 2015, Section 4).

        Drops each **reportably-weighted** donor, one at a time, and re-fits the
        treated unit's synthetic control against the remaining donor pool. The
        per-drop ATTs reveal whether the estimated effect is driven by any single
        donor (ADH 2015 overlay the leave-one-out counterfactual trajectories for
        this purpose; :meth:`get_leave_one_out_gaps` returns those paths). This is a
        thin re-run of the validated SCM solver — it has **no analytical standard
        error**; ``se``/``t_stat``/``p_value``/``conf_int`` and ``is_significant``
        are unaffected (still bound to the NaN analytical ``p_value``).

        The drop set is exactly the donors in ``donor_weights`` — those above the
        ``1e-6`` interpretability floor (``synthetic_control._MIN_REPORT_WEIGHT``).
        A donor with negligible weight ``0 < w ≤ 1e-6`` is excluded (its removal
        moves the ATT by ~the weight, so its ``delta_att`` would be ~0 — an
        uninformative row), keeping the LOO table aligned with the reported support;
        a zero-weight donor's removal leaves the synthetic unchanged. (This `1e-6`
        approximation of "positive weight" is documented in REGISTRY §SyntheticControl.)
        A donor that carries ALL the weight is still dropped (the others absorb its
        mass on re-fit); its large ``delta_att`` is exactly the single-donor-dependence
        signal this diagnostic exists to surface, NOT a failure.

        Parameters
        ----------
        n_starts : int, optional
            Override the multistart count for each leave-one-out refit's outer V
            search (nested/cv). Default None inherits the original fit's ``n_starts``.

        Returns
        -------
        pandas.DataFrame
            One ``status="baseline"`` row (the full fit, ``delta_att=0``) followed by
            one row per dropped donor: ``status="loo"``, or — with NaN metrics — an
            excluded drop that is ``"failed"`` (its refit did not converge) or
            ``"infeasible"`` (under ``v_method="cv"`` the reduced donor pool is
            indistinguishable in a re-aggregated CV window). Rows are sorted by
            ``|delta_att|`` descending, with the excluded (``"failed"`` /
            ``"infeasible"``) rows last. Columns: ``dropped_unit``, ``att``,
            ``pre_rmspe``, ``post_rmspe``, ``rmspe_ratio``, ``delta_att``
            (``att_loo - full_att``), ``status``.

        Raises
        ------
        ValueError
            If the fit snapshot is unavailable (e.g. this result was unpickled).
        """
        if self._fit_snapshot is None:
            raise ValueError(
                "leave_one_out() requires the fit snapshot on the results object. "
                "This result appears to have been loaded from serialization (which "
                "excludes the snapshot) or produced by an older estimator version. "
                "Re-fit to enable leave-one-out donor robustness."
            )
        from diff_diff.synthetic_control import _mspe, _placebo_fit_unit

        snap = self._fit_snapshot
        if n_starts is None:
            n_starts_eff = snap.n_starts
        else:
            # Mirror the estimator constructor's validation so a bad override fails
            # fast instead of silently coercing into a degenerate refit (cf.
            # in_space_placebo()).
            if not isinstance(n_starts, (int, np.integer)) or n_starts < 1:
                raise ValueError(f"n_starts override must be a positive integer, got {n_starts!r}")
            n_starts_eff = int(n_starts)

        # Baseline row: read DIRECTLY from the full fit (do NOT re-fit), so the
        # reference ATT — and therefore delta_att=0.0 — is exact.
        baseline_row = {
            "dropped_unit": None,
            "att": float(self.att),
            "pre_rmspe": float(self.pre_rmspe),
            "post_rmspe": float(np.sqrt(_mspe(self.gap_path, snap.post_periods))),
            "rmspe_ratio": float(self.rmspe_ratio),
            "delta_att": 0.0,
            "status": "baseline",
        }

        # Fail closed when the treated unit's own fit did not converge: a truncated /
        # under-optimized baseline ATT makes every leave-one-out delta meaningless.
        if not self._fit_converged:
            warnings.warn(
                "Leave-one-out skipped: the treated unit's own SCM fit did not "
                "converge at fit time (inner Frank-Wolfe weight solve and/or outer V "
                "search), so the baseline ATT is not a valid optimum to compare "
                "leave-one-out refits against. Re-fit with a larger inner_max_iter / "
                "looser inner_min_decrease (inner) and/or a larger "
                "optimizer_options['maxiter'] / more n_starts (outer V search).",
                UserWarning,
                stacklevel=2,
            )
            self._loo_status = "treated_fit_nonconverged"
            self._loo_att_range = None
            self._loo_n_failed = 0
            self._loo_n_infeasible = 0
            self._loo_gaps = {}
            self._loo_df = pd.DataFrame([baseline_row], columns=self._LOO_COLS)
            return self._loo_df.copy()

        # Dropping any donor requires at least one donor left in the pool.
        if len(snap.donor_ids) < 2:
            warnings.warn(
                "Leave-one-out donor robustness requires at least 2 donors (dropping "
                f"one must leave a non-empty pool); only {len(snap.donor_ids)} "
                "available. Returning the baseline fit only.",
                UserWarning,
                stacklevel=2,
            )
            self._loo_status = "too_few_donors"
            self._loo_att_range = None
            self._loo_n_failed = 0
            self._loo_n_infeasible = 0
            self._loo_gaps = {}
            self._loo_df = pd.DataFrame([baseline_row], columns=self._LOO_COLS)
            return self._loo_df.copy()

        # Drop the FROZEN reportably-weighted support captured at fit time (donor ids
        # with weight above the 1e-6 floor, in donor_ids order). Reading the snapshot —
        # NOT the mutable presentation-level self.donor_weights — makes the result
        # depend only on the fit and immune to post-fit mutation of donor_weights.
        pos_donors = list(snap.weighted_donor_ids)
        loo_gaps: Dict[Any, Dict[Any, float]] = {}
        loo_rows: List[Dict[str, Any]] = []
        atts: List[float] = []
        n_failed = 0
        n_infeasible = 0

        for d in pos_donors:
            pool = [x for x in snap.donor_ids if x != d]
            fitted, fit_status = _placebo_fit_unit(snap, snap.treated_id, pool, n_starts_eff)
            if fitted is None:
                # "infeasible" (structural cv donor-indistinguishability of the reduced
                # pool) vs "failed" (solver non-convergence): counted separately, both
                # excluded from the ATT range. Mirrors the in_time_placebo split.
                if fit_status == "infeasible":
                    n_infeasible += 1
                else:
                    n_failed += 1
                loo_rows.append(
                    {
                        "dropped_unit": d,
                        "att": np.nan,
                        "pre_rmspe": np.nan,
                        "post_rmspe": np.nan,
                        "rmspe_ratio": np.nan,
                        "delta_att": np.nan,
                        "status": fit_status,
                    }
                )
                continue
            gap_path_d, ratio_d = fitted
            loo_gaps[d] = gap_path_d
            att_d = float(np.mean([gap_path_d[p] for p in snap.post_periods]))
            atts.append(att_d)
            loo_rows.append(
                {
                    "dropped_unit": d,
                    "att": att_d,
                    "pre_rmspe": float(np.sqrt(_mspe(gap_path_d, snap.pre_periods))),
                    "post_rmspe": float(np.sqrt(_mspe(gap_path_d, snap.post_periods))),
                    "rmspe_ratio": ratio_d,
                    "delta_att": att_d - float(self.att),
                    "status": "loo",
                }
            )

        # Sort successful drops by |delta_att| desc (most influential donor first);
        # excluded drops (failed OR infeasible) sort last.
        finite_rows = sorted(
            (r for r in loo_rows if r["status"] == "loo"),
            key=lambda r: abs(r["delta_att"]),
            reverse=True,
        )
        excluded_rows = [r for r in loo_rows if r["status"] != "loo"]
        ordered = [baseline_row] + finite_rows + excluded_rows

        # Two exclusion causes, warned separately (mirrors in_time_placebo) so a structural
        # cv exclusion is not mis-attributed to a solver budget. Both drop out of the ATT range.
        if n_infeasible > 0:
            warnings.warn(
                f"{n_infeasible} of {len(pos_donors)} leave-one-out refits were STRUCTURALLY "
                "infeasible under v_method='cv' (the reduced donor pool is indistinguishable "
                "in a re-aggregated CV window) and were excluded with status='infeasible'; "
                "remedy by adjusting the predictors, v_cv_t0, or the donor pool (NOT "
                "inner_max_iter / n_starts); the ATT range uses the remaining refits.",
                UserWarning,
                stacklevel=2,
            )
        if n_failed > 0:
            warnings.warn(
                f"{n_failed} of {len(pos_donors)} leave-one-out refits were excluded with "
                "NaN metrics (status='failed'; the refit did not reach a valid optimum — a "
                "non-converged inner weight solve or outer V search); the ATT range uses "
                "the remaining refits.",
                UserWarning,
                stacklevel=2,
            )

        self._loo_gaps = loo_gaps
        self._loo_n_failed = int(n_failed)
        self._loo_n_infeasible = int(n_infeasible)
        self._loo_att_range = (min(atts), max(atts)) if atts else None
        # Baseline-relative headline: the largest swing of any single donor-drop from
        # the full-fit ATT (max |delta_att|). Robust to a uniform shift that a raw
        # att_range would understate.
        self._loo_max_abs_delta_att = max(abs(a - float(self.att)) for a in atts) if atts else None
        # Distinguish a real run from "no valid leave-one-out estimate produced" (so DR/BR
        # do not report an empty diagnostic as completed) AND classify the no-success cause
        # by solver-failure vs structural-infeasibility vs a mix (mirrors in_time_placebo).
        # (pos_donors empty — a converged fit always has >=1 positive weight — falls through
        # to "ran": baseline-only, benign.)
        if atts or not pos_donors:
            self._loo_status = "ran"
        elif n_failed > 0 and n_infeasible > 0:
            self._loo_status = "all_refits_unusable"
        elif n_infeasible > 0:
            self._loo_status = "all_refits_infeasible"
        else:
            self._loo_status = "all_refits_failed"
        self._loo_df = pd.DataFrame(ordered, columns=self._LOO_COLS)
        return self._loo_df.copy()

    def get_leave_one_out_df(self) -> pd.DataFrame:
        """
        Get the leave-one-out donor-robustness table (see :meth:`leave_one_out`).

        Survives pickling. Raises if :meth:`leave_one_out` has not been run.

        Returns
        -------
        pandas.DataFrame
        """
        if self._loo_df is None:
            raise ValueError("No leave-one-out results yet; call leave_one_out() first.")
        return self._loo_df.copy()

    def get_leave_one_out_gaps(self) -> pd.DataFrame:
        """
        Long-form leave-one-out gap paths, for the overlay ("spaghetti") plot.

        One row per (dropped donor, period) for every converged leave-one-out refit.
        Columns: ``dropped_unit``, ``period``, ``gap``, ``phase`` (``"pre"``/
        ``"post"``) — mirroring :meth:`get_gap_df`. These per-period paths are
        panel-derived and are NOT retained after pickling.

        Returns
        -------
        pandas.DataFrame

        Raises
        ------
        ValueError
            If :meth:`leave_one_out` has not been run, or if the gap paths were
            dropped on pickling (re-fit and re-run to recompute them).
        """
        if self._loo_df is None:
            raise ValueError("No leave-one-out results yet; call leave_one_out() first.")
        if self._loo_gaps is None:
            raise ValueError(
                "Leave-one-out gap paths are not retained after pickling "
                "(panel-derived); re-run leave_one_out() on a freshly fitted result "
                "to recompute them."
            )
        rows: List[Dict[str, Any]] = []
        for unit, gap_path in self._loo_gaps.items():
            for period in list(self.pre_periods) + list(self.post_periods):
                if period in gap_path:
                    phase = "post" if period in self.post_periods else "pre"
                    rows.append(
                        {
                            "dropped_unit": unit,
                            "period": period,
                            "gap": gap_path[period],
                            "phase": phase,
                        }
                    )
        return pd.DataFrame(rows, columns=["dropped_unit", "period", "gap", "phase"])

    _IN_TIME_COLS = [
        "placebo_period",
        "placebo_att",
        "pre_fit_rmspe",
        "rmspe_ratio",
        "n_pre_fake",
        "n_post_fake",
        "n_dropped_specs",
        "status",
    ]

    def in_time_placebo(
        self,
        placebo_periods: Optional[Any] = None,
        n_starts: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        In-time (backdating) placebo (Abadie-Diamond-Hainmueller 2015, Section 4).

        Reassigns the intervention to an earlier pre-treatment date ``t_f`` and re-fits
        the synthetic control using ONLY pre-``t_f`` information, then measures the
        "effect" over the held-out window ``[t_f, T0)``. A credible synthetic control
        should show **no spurious gap** there (ADH 2015 Figure 4, German reunification
        backdated to 1975). This is a thin re-run of the validated SCM solver — it has
        **no analytical standard error**; ``se``/``t_stat``/``p_value``/``conf_int`` and
        ``is_significant`` are unaffected.

        **Windowing convention (TRUNCATE).** The placebo fit uses only periods strictly
        before ``t_f``: pre-period-outcome predictors become the pre-``t_f`` outcomes,
        and covariate / special predictor windows are intersected with the pre-``t_f``
        window. A predictor window lying ENTIRELY in the held-out region ``[t_f, T0)``
        is dropped (surfaced in ``n_dropped_specs`` + an aggregated warning). For
        outcome-predictor fits this equals the literal "lag the predictors" re-run of a
        manual ``Synth::synth`` (R has no in-time-placebo function); see
        ``docs/methodology/REGISTRY.md`` for the recognized deviation note.

        Parameters
        ----------
        placebo_periods : period value or list of period values, optional
            The pseudo-intervention date(s), each a member of ``pre_periods``. Default
            None sweeps every feasible interior pre-date (at least 2 pre-fake periods to
            fit + at least 1 post-fake period to measure the gap). A date that is a true
            post-treatment period, or not a pre-period at all, raises ``ValueError``; a
            valid pre-date that is dimensionally infeasible (too few pre-fake periods, or
            all predictors dropped) yields a ``status="infeasible"`` row (no raise).
        n_starts : int, optional
            Override the multistart count for each placebo refit's outer V search (nested/cv).
            Default None inherits the original fit's ``n_starts``.

        Returns
        -------
        pandas.DataFrame
            One row per placebo date. Columns: ``placebo_period``, ``placebo_att`` (mean
            gap over the held-out window — should be ~0 if no real pre-period effect),
            ``pre_fit_rmspe``, ``rmspe_ratio`` (post-fake/pre-fake), ``n_pre_fake``,
            ``n_post_fake``, ``n_dropped_specs``, ``status`` (``"ran"`` / ``"infeasible"``
            / ``"failed"``).

        Raises
        ------
        ValueError
            If the fit snapshot is unavailable (e.g. this result was unpickled), or an
            explicit ``placebo_periods`` entry is a post-treatment period / not a
            pre-period.
        """
        if self._fit_snapshot is None:
            raise ValueError(
                "in_time_placebo() requires the fit snapshot on the results object. "
                "This result appears to have been loaded from serialization (which "
                "excludes the snapshot) or produced by an older estimator version. "
                "Re-fit to enable the in-time placebo."
            )
        from diff_diff.synthetic_control import (
            _mspe,
            _placebo_fit_unit,
            _truncate_snapshot_in_time,
        )

        snap = self._fit_snapshot
        if n_starts is None:
            n_starts_eff = snap.n_starts
        else:
            if not isinstance(n_starts, (int, np.integer)) or n_starts < 1:
                raise ValueError(f"n_starts override must be a positive integer, got {n_starts!r}")
            n_starts_eff = int(n_starts)

        pre = list(snap.pre_periods)
        empty = pd.DataFrame([], columns=self._IN_TIME_COLS)

        # Fail closed when the treated unit's own fit did not converge: a truncated /
        # under-optimized baseline makes the placebo comparison meaningless.
        if not self._fit_converged:
            warnings.warn(
                "In-time placebo skipped: the treated unit's own SCM fit did not "
                "converge at fit time (inner Frank-Wolfe weight solve and/or outer V "
                "search). Re-fit with a larger inner_max_iter / looser "
                "inner_min_decrease (inner) and/or a larger optimizer_options['maxiter'] "
                "/ more n_starts (outer V search).",
                UserWarning,
                stacklevel=2,
            )
            self._in_time_status = "treated_fit_nonconverged"
            self._in_time_n_failed = 0
            self._in_time_gaps = {}
            self._in_time_df = empty
            return empty.copy()

        # A feasible date needs >=2 pre-fake + >=1 post-fake period -> >=3 pre periods.
        # The >=2 pre-fake rule is a deliberate Note-documented restriction (an auto-
        # swept single-pre-fake placebo is a non-credible pre-fit; see REGISTRY).
        if len(pre) < 3:
            warnings.warn(
                "In-time placebo requires at least 3 pre-treatment periods (a feasible "
                "placebo date needs >=2 pre-fake periods to fit and >=1 post-fake period "
                f"to measure the gap); only {len(pre)} available.",
                UserWarning,
                stacklevel=2,
            )
            self._in_time_status = "too_few_pre_periods"
            self._in_time_n_failed = 0
            self._in_time_gaps = {}
            self._in_time_df = empty
            return empty.copy()

        if placebo_periods is None:
            # Sweep every feasible pre-date (positional: idx>=2 gives >=2 pre-fake +
            # >=1 post-fake; idx<2 would leave fewer than 2 pre-fake periods).
            dates: List[Any] = [pre[i] for i in range(2, len(pre))]
        else:
            if isinstance(placebo_periods, (list, tuple, set, np.ndarray, pd.Index, pd.Series)):
                dates = list(placebo_periods)
            else:
                dates = [placebo_periods]
            # An explicit but EMPTY container is a malformed request (NOT "every date
            # was infeasible") — fail fast, consistent with the post-date / non-pre
            # date raises below. Pass None to sweep all feasible pre-dates.
            if not dates:
                raise ValueError(
                    "placebo_periods is empty; pass None to sweep all feasible "
                    "pre-dates, or a non-empty list of pre-period date(s)."
                )
            pre_set = set(pre)
            post_set = set(snap.post_periods)
            for d in dates:
                if d in post_set:
                    raise ValueError(
                        f"placebo_period {d!r} is a true post-treatment period; an "
                        "in-time placebo date must lie in the pre-treatment window."
                    )
                if d not in pre_set:
                    raise ValueError(
                        f"placebo_period {d!r} is not a pre-treatment period "
                        f"(pre_periods = {pre})."
                    )
            # De-duplicate + canonicalize to pre-period order (mirrors _resolve_periods):
            # duplicate / unordered explicit dates must not trigger duplicate refits or
            # inflate n_dates.
            _requested = set(dates)
            dates = [p for p in pre if p in _requested]

        in_time_gaps: Dict[Any, Dict[Any, float]] = {}
        rows: List[Dict[str, Any]] = []
        dropped_all: set = set()
        n_failed = 0
        n_infeasible = 0
        n_ran = 0

        for t_f in dates:
            idx = pre.index(t_f)
            n_pre_fake = idx
            n_post_fake = len(pre) - idx
            snap_mod, dropped = _truncate_snapshot_in_time(snap, t_f)
            dropped_all.update(dropped)
            if snap_mod is None:
                n_infeasible += 1
                rows.append(
                    {
                        "placebo_period": t_f,
                        "placebo_att": np.nan,
                        "pre_fit_rmspe": np.nan,
                        "rmspe_ratio": np.nan,
                        "n_pre_fake": n_pre_fake,
                        "n_post_fake": n_post_fake,
                        "n_dropped_specs": len(dropped),
                        "status": "infeasible",
                    }
                )
                continue
            fitted, fit_status = _placebo_fit_unit(
                snap_mod, snap.treated_id, snap.donor_ids, n_starts_eff
            )
            if fitted is None:
                # _truncate_snapshot_in_time already applied the cv structural checks to the
                # truncated snapshot, so a None here is normally a solver non-convergence
                # ("failed"); defensively honor an "infeasible" status if the solve still
                # reports one (counts it alongside the truncation-level n_infeasible).
                if fit_status == "infeasible":
                    n_infeasible += 1
                else:
                    n_failed += 1
                rows.append(
                    {
                        "placebo_period": t_f,
                        "placebo_att": np.nan,
                        "pre_fit_rmspe": np.nan,
                        "rmspe_ratio": np.nan,
                        "n_pre_fake": n_pre_fake,
                        "n_post_fake": n_post_fake,
                        "n_dropped_specs": len(dropped),
                        "status": fit_status,
                    }
                )
                continue
            gap_path, ratio = fitted
            in_time_gaps[t_f] = gap_path
            placebo_att = float(np.mean([gap_path[p] for p in snap_mod.post_periods]))
            rows.append(
                {
                    "placebo_period": t_f,
                    "placebo_att": placebo_att,
                    "pre_fit_rmspe": float(np.sqrt(_mspe(gap_path, snap_mod.pre_periods))),
                    "rmspe_ratio": ratio,
                    "n_pre_fake": n_pre_fake,
                    "n_post_fake": n_post_fake,
                    "n_dropped_specs": len(dropped),
                    "status": "ran",
                }
            )
            n_ran += 1

        if dropped_all:
            warnings.warn(
                "In-time placebo (TRUNCATE convention): predictor(s) "
                f"{sorted(map(str, dropped_all))} fell entirely in the held-out "
                "post-fake window for some placebo date(s) and were dropped from those "
                "refits (see the n_dropped_specs column).",
                UserWarning,
                stacklevel=2,
            )
        if n_infeasible > 0:
            warnings.warn(
                f"{n_infeasible} in-time placebo date(s) were structurally infeasible "
                "(too few pre-fake periods, all predictors dropped, or — under "
                "v_method='cv' — a kept predictor no longer spans both windows, or a "
                "re-aggregated window loses cross-donor variation, after truncation) and "
                "are reported with status='infeasible' (NaN metrics).",
                UserWarning,
                stacklevel=2,
            )
        if n_failed > 0:
            warnings.warn(
                f"{n_failed} in-time placebo refit(s) failed to converge and are "
                "reported with status='failed' (NaN metrics).",
                UserWarning,
                stacklevel=2,
            )

        self._in_time_gaps = in_time_gaps
        self._in_time_n_failed = int(n_failed)
        self._in_time_n_infeasible = int(n_infeasible)
        # When no date ran, classify the cause precisely so the downstream reason text
        # is never false: a pure convergence failure ("all_dates_failed", actionable —
        # raise n_starts / loosen tolerances) and pure dimensional infeasibility
        # ("all_dates_infeasible", structural) are distinct; a MIX of both gets its own
        # "all_dates_unusable" code (both counters are surfaced) rather than being
        # mislabeled as exclusively one or the other.
        if n_ran > 0:
            self._in_time_status = "ran"
        elif n_failed > 0 and n_infeasible > 0:
            self._in_time_status = "all_dates_unusable"
        elif n_failed > 0:
            self._in_time_status = "all_dates_failed"
        else:
            self._in_time_status = "all_dates_infeasible"
        self._in_time_df = pd.DataFrame(rows, columns=self._IN_TIME_COLS)
        return self._in_time_df.copy()

    def get_in_time_placebo_df(self) -> pd.DataFrame:
        """
        Get the in-time placebo table (see :meth:`in_time_placebo`).

        Survives pickling. Raises if :meth:`in_time_placebo` has not been run.

        Returns
        -------
        pandas.DataFrame
        """
        if self._in_time_df is None:
            raise ValueError("No in-time placebo results yet; call in_time_placebo() first.")
        return self._in_time_df.copy()

    def get_in_time_placebo_gaps(self) -> pd.DataFrame:
        """
        Long-form in-time placebo gap paths, for the backdating overlay plot.

        One row per (placebo date, period) for every converged in-time refit. Columns:
        ``placebo_period``, ``period``, ``gap``, ``phase`` (``"pre_fake"`` for periods
        before the placebo date, ``"post_fake"`` for the held-out window from it on).
        These per-period paths are panel-derived and are NOT retained after pickling.

        Returns
        -------
        pandas.DataFrame

        Raises
        ------
        ValueError
            If :meth:`in_time_placebo` has not been run, or if the gap paths were
            dropped on pickling (re-fit and re-run to recompute them).
        """
        if self._in_time_df is None:
            raise ValueError("No in-time placebo results yet; call in_time_placebo() first.")
        if self._in_time_gaps is None:
            raise ValueError(
                "In-time placebo gap paths are not retained after pickling "
                "(panel-derived); re-run in_time_placebo() on a freshly fitted result "
                "to recompute them."
            )
        pre = list(self.pre_periods)
        rows: List[Dict[str, Any]] = []
        for t_f, gap_path in self._in_time_gaps.items():
            split = pre.index(t_f)
            for period in pre:
                if period in gap_path:
                    phase = "post_fake" if pre.index(period) >= split else "pre_fake"
                    rows.append(
                        {
                            "placebo_period": t_f,
                            "period": period,
                            "gap": gap_path[period],
                            "phase": phase,
                        }
                    )
        return pd.DataFrame(rows, columns=["placebo_period", "period", "gap", "phase"])

    # =====================================================================
    # Confidence sets by test inversion (Firpo & Possebom 2018, §4)
    # =====================================================================

    def _require_placebo_reference(self, n_starts: Optional[int]) -> None:
        """Ensure an in-space placebo reference set is available for test inversion.

        Lazily runs :meth:`in_space_placebo` when no reference set has been built yet
        (raising the same ValueError as that method if the fit snapshot is missing, e.g.
        on an unpickled result). If a reference set already exists, a non-None ``n_starts``
        is **ignored with a UserWarning** — the test inversion reuses the single stored set
        (every sharp null re-ranks the SAME gaps), so honouring ``n_starts`` would mean an
        expensive O(J) re-fit that the caller did not ask for. Raises ValueError when no
        valid reference set could be produced (fewer than 2 donors, a non-converged treated
        fit, or all donor refits failed / were structurally infeasible) — there is then no
        permutation distribution to invert.
        """
        if self._placebo_gaps is None:
            # Builds the reference set; raises ValueError if the snapshot is unavailable.
            self.in_space_placebo(n_starts=n_starts)
        elif n_starts is not None:
            warnings.warn(
                "n_starts is ignored: the in-space placebo reference set was already "
                "computed and is reused (every sharp null / grid value re-ranks the same "
                "placebo gaps). Re-run in_space_placebo(n_starts=...) explicitly to rebuild "
                "it with a different multistart count.",
                UserWarning,
                stacklevel=3,
            )
        if not self._placebo_gaps or self._placebo_status != "ran":
            reasons = {
                "treated_fit_nonconverged": (
                    "the treated unit's own SCM fit did not converge at fit time, so its "
                    "RMSPE ratio is not a valid optimum to rank against placebos"
                ),
                "too_few_donors": (
                    "fewer than 2 donors are available (each placebo is fit against the "
                    "other donors)"
                ),
                "all_placebos_failed": (
                    "every donor refit failed to converge, so no placebo entered the "
                    "reference set"
                ),
                "all_placebos_infeasible": (
                    "every donor refit was structurally infeasible (under v_method='cv' "
                    "the pseudo-treated donor pool is indistinguishable in a re-aggregated "
                    "CV window), so no placebo entered the reference set"
                ),
                "all_placebos_unusable": (
                    "no donor refit was usable — some failed to converge and some were "
                    "structurally infeasible — so no placebo entered the reference set"
                ),
            }
            default_reason = "no valid in-space placebo reference set was produced"
            status = self._placebo_status
            reason = reasons.get(status, default_reason) if status is not None else default_reason
            raise ValueError(
                "Confidence set / sharp-null test requires a valid in-space placebo "
                f"reference set, but {reason}. (See the in_space_placebo() warning above.)"
            )

    @staticmethod
    def _coerce_effect_path(effect: Any, n_post: int) -> np.ndarray:
        """Coerce ``effect`` to a length-``n_post`` post-period effect path ``f(t)``.

        A scalar broadcasts to a constant path (Eq 11 with ``f(t) = c``); a 1-D array must
        have one finite value per post period, aligned to the calendar-sorted
        ``post_periods``. Fails closed on a wrong length or any non-finite value.
        """
        arr = np.asarray(effect, dtype=float)
        if arr.ndim == 0:
            f = np.full(n_post, float(arr), dtype=float)
        elif arr.ndim == 1:
            if arr.shape[0] != n_post:
                raise ValueError(
                    f"effect path has length {arr.shape[0]} but there are {n_post} "
                    "post-treatment periods; pass a scalar (a constant-in-time effect) or "
                    f"a length-{n_post} array aligned to post_periods (calendar order)."
                )
            f = arr
        else:
            raise ValueError(
                "effect must be a scalar (constant effect) or a 1-D array (one value per "
                f"post period), got a {arr.ndim}-D array."
            )
        if not np.all(np.isfinite(f)):
            raise ValueError("effect contains non-finite (NaN/inf) values.")
        return f

    def test_sharp_null(
        self,
        effect: Any,
        *,
        gamma: float = 0.1,
        n_starts: Optional[int] = None,
    ) -> pd.Series:
        """Test a sharp null hypothesis on the treatment-effect path (Firpo & Possebom 2018, §4.1).

        Tests ``H_0^f: α_{1,t} = f(t)`` for every post period (Eq 11) by subtracting the
        hypothesized effect path ``f(t)`` from the post-period gaps of EVERY unit and
        re-ranking the treated unit's modified RMSPE ratio against the placebo distribution
        (Eqs 12–13 at ``φ = 0``, ``v = (1,…,1)`` — the equal-weights benchmark). The
        synthetic controls are NOT refit: this reuses the gap paths and per-unit
        denominators :meth:`in_space_placebo` already computed (run lazily here if needed).
        At ``effect = 0`` the p-value is identically the benchmark ``placebo_p_value``
        (Eq 5 = Eq 13 with ``f ≡ 0``).

        Parameters
        ----------
        effect : float or array-like
            The hypothesized post-period effect ``f(t)``: a scalar (a constant-in-time
            effect, Eq 11), or a length-``n_post_periods`` array aligned to ``post_periods``
            in calendar order (an arbitrary path — e.g. an intervention cost path or a
            theory-predicted shape).
        gamma : float, default 0.1
            Test level; the null is rejected when ``p^f < gamma``. The permutation p-value
            is granular in ``1/(J+1)`` (Firpo & Possebom fn 8), so not every nominal level
            is attainable.
        n_starts : int, optional
            Multistart count for the lazy :meth:`in_space_placebo` run; ignored (with a
            warning) if the reference set already exists.

        Returns
        -------
        pandas.Series
            ``p_value`` (``p^f``), ``reject`` (``p^f < gamma``), ``gamma``,
            ``rmspe_f_treated`` (the treated unit's modified RMSPE ratio), ``n_placebos``
            (reference-set size), ``n_failed``.

        Raises
        ------
        ValueError
            If ``gamma`` is not in ``(0, 1)``, ``effect`` has the wrong shape / non-finite
            values, or no valid placebo reference set is available (see
            :meth:`in_space_placebo`).
        """
        from diff_diff.synthetic_control import _sharp_null_pvalue

        if not (0.0 < float(gamma) < 1.0):
            raise ValueError(f"gamma must be in (0, 1), got {gamma!r}")
        self._require_placebo_reference(n_starts)
        post_periods = list(self.post_periods)
        f_post = self._coerce_effect_path(effect, len(post_periods))
        assert self._placebo_gaps is not None and self._placebo_pre_denoms is not None
        p, r1, n_ref = _sharp_null_pvalue(
            self.gap_path,
            self._placebo_gaps,
            post_periods,
            f_post,
            self._placebo_pre_denoms,
            self.treated_unit,
        )
        return pd.Series(
            {
                "p_value": float(p),
                "reject": bool(p < float(gamma)),
                "gamma": float(gamma),
                "rmspe_f_treated": float(r1),
                "n_placebos": int(n_ref),
                "n_failed": int(self.n_failed),
            }
        )

    def confidence_set(
        self,
        *,
        family: str = "constant",
        gamma: float = 0.1,
        bounds: Optional[Tuple[float, float]] = None,
        n_grid: int = 200,
        n_starts: Optional[int] = None,
    ) -> pd.DataFrame:
        """Confidence set for the treatment-effect path by test inversion (Firpo & Possebom 2018, §4.2).

        Inverts the sharp-null test (:meth:`test_sharp_null`) over a one-parameter effect
        family: the confidence set is every parameter value whose sharp null is **not
        rejected**, ``{ param : p^param > gamma }`` (Eq 14, **strict** inequality). Two
        families are supported:

        - ``family="constant"`` — ``f(t) = c`` (Eq 15); the set is a confidence **interval**
          for a constant-in-time effect (Eq 16). The parameter column is ``c``.
        - ``family="linear"`` — ``f(t) = c̃·(t − T0)`` with the 1-based post-period index
          ``(t − T0)`` (Eq 17); the set is a confidence **set** over the slope ``c̃``
          (Eq 18). The parameter column is ``c_tilde``.

        The inversion is a pure re-ranking of the stored placebo gaps (no synthetic-control
        refits): :meth:`in_space_placebo` is run lazily if needed, then each value only
        recomputes ``p^param``. With ``bounds=None`` the set is recovered **exactly**:
        ``p^param`` is piecewise-constant (each placebo's indicator flips only at the real
        roots of a quadratic in ``param``), so the placebo breakpoints partition the line,
        ``p`` is evaluated once per induced interval AND at each breakpoint (where a tie
        under ``≥`` can lift ``p`` above ``gamma``), and the union of accepted
        intervals/points is the set — with NO centering or monotonicity assumption (accepted
        tails and disjoint components are handled). With explicit ``bounds`` a fixed
        ``linspace(*bounds, n_grid)`` grid is scanned instead (grid-limited membership).

        **Boundary convention (paper-sourced, Eq 14):** membership is the *strict* inequality
        ``p^param > gamma``. The permutation p-value is discrete (a multiple of ``1/(J+1)``),
        so ``p = gamma`` is reachable and is **excluded** from the set.

        The result is stored on the object: the summary on
        :attr:`effect_confidence_set` (``{family, parameter, gamma, lower, upper,
        contiguous, boundary, point_estimate, n_grid, n_placebos, status}``, surviving
        pickling) and the full grid on :meth:`get_confidence_set_df`. The analytical
        ``conf_int`` / ``se`` stay NaN — this is a separate permutation object.

        Parameters
        ----------
        family : {"constant", "linear"}, default "constant"
            The one-parameter effect family to invert over.
        gamma : float, default 0.1
            Confidence level is ``1 − gamma``; ``p^param > gamma`` defines membership.
        bounds : (float, float), optional
            Fixed ``(lo, hi)`` grid for the parameter. Default None uses exact breakpoint
            inversion (a fixed grid is used only when ``bounds`` is supplied).
        n_grid : int, default 200
            Number of grid points evaluated for the returned table (>= 2).
        n_starts : int, optional
            Multistart count for the lazy :meth:`in_space_placebo` run; ignored (with a
            warning) if the reference set already exists.

        Returns
        -------
        pandas.DataFrame
            Columns ``param`` (``c`` for constant, ``c̃`` for linear), ``p_value``
            (``p^param``), ``in_set`` (``p^param > gamma``). Empty for an ``"empty"`` set;
            an ``"unbounded"`` exact set with finite breakpoints still returns an inspection
            grid over a padded breakpoint range (see :attr:`effect_confidence_set`
            ``status``).

        Raises
        ------
        ValueError
            If ``family`` is unknown, ``gamma`` not in ``(0, 1)``, ``n_grid < 2``, ``bounds``
            malformed, or no valid placebo reference set is available.
        """
        from diff_diff.synthetic_control import _invert_sharp_null

        if family not in ("constant", "linear"):
            raise ValueError(f"family must be 'constant' or 'linear', got {family!r}")
        if not (0.0 < float(gamma) < 1.0):
            raise ValueError(f"gamma must be in (0, 1), got {gamma!r}")
        if not isinstance(n_grid, (int, np.integer)) or n_grid < 2:
            raise ValueError(f"n_grid must be an integer >= 2, got {n_grid!r}")
        if bounds is not None:
            # Guard the type/length BEFORE indexing so a malformed scalar raises the
            # documented ValueError (not a bare TypeError from len()/subscription).
            if (
                not isinstance(bounds, (tuple, list, np.ndarray))
                or len(bounds) != 2
                or not all(isinstance(b, (int, float, np.integer, np.floating)) for b in bounds)
                or not all(np.isfinite(float(b)) for b in bounds)
            ):
                raise ValueError(f"bounds must be a finite (lo, hi) pair, got {bounds!r}")
            if float(bounds[1]) <= float(bounds[0]):
                raise ValueError(f"bounds must satisfy hi > lo, got {bounds!r}")
        self._require_placebo_reference(n_starts)
        assert self._placebo_gaps is not None and self._placebo_pre_denoms is not None
        res = _invert_sharp_null(
            self.gap_path,
            self._placebo_gaps,
            list(self.post_periods),
            self._placebo_pre_denoms,
            self.treated_unit,
            family,
            float(gamma),
            bounds=(None if bounds is None else (float(bounds[0]), float(bounds[1]))),
            n_grid=int(n_grid),
        )
        status = res["status"]
        if status == "unbounded":
            extra = (
                " The accepted set is ALSO non-contiguous (e.g. two accepted tails with a "
                "rejected middle, NOT the whole line), so [lower, upper] is only the hull — "
                "inspect get_confidence_set_df() for the structure."
                if not res["contiguous"]
                else ""
            )
            warnings.warn(
                "Confidence set is unbounded: either gamma is below the permutation "
                "granularity 1/(J+1) (so no effect is ever rejected — Firpo & Possebom "
                "fn 8), or the treated unit does not have the best pre-treatment fit (so "
                "the RMSPE ratio does not grow without bound on one side). Reported "
                "endpoint(s) are +/-inf." + extra,
                UserWarning,
                stacklevel=2,
            )
        elif status == "empty":
            warnings.warn(
                f"Confidence set is empty: every {family} effect in this family is "
                f"rejected at gamma={gamma:.3g} (the largest attainable p-value does not "
                "exceed gamma). Endpoints are NaN.",
                UserWarning,
                stacklevel=2,
            )
        elif not res["contiguous"]:
            warnings.warn(
                "Confidence set is non-contiguous (the discrete permutation p-value dips "
                "below gamma at an interior grid point); [lower, upper] is reported as the "
                "hull. Inspect get_confidence_set_df() for the full grid.",
                UserWarning,
                stacklevel=2,
            )
        self.effect_confidence_set = {
            "family": family,
            "parameter": "c" if family == "constant" else "c_tilde",
            "gamma": float(gamma),
            "lower": float(res["lower"]),
            "upper": float(res["upper"]),
            "contiguous": bool(res["contiguous"]),
            "boundary": "strict",
            "point_estimate": float(res["point_estimate"]),
            "n_grid": int(n_grid),
            "n_placebos": int(res["n_ref"]),
            "status": status,
        }
        self._confidence_set_df = pd.DataFrame(res["grid"], columns=["param", "p_value", "in_set"])
        return self._confidence_set_df.copy()

    def get_confidence_set_df(self) -> pd.DataFrame:
        """Get the test-inversion confidence-set grid table (see :meth:`confidence_set`).

        Columns: ``param`` (``c`` constant / ``c̃`` linear), ``p_value`` (``p^param``),
        ``in_set`` (``p^param > gamma``). Survives pickling. Raises if
        :meth:`confidence_set` has not been run.

        Returns
        -------
        pandas.DataFrame
        """
        if self._confidence_set_df is None:
            raise ValueError("No confidence set yet; call confidence_set() first.")
        return self._confidence_set_df.copy()

    # =====================================================================
    # Conformal inference (Chernozhukov-Wüthrich-Zhu 2021) — opt-in.
    # A self-contained inference layer that fits its OWN permutation-invariant
    # constrained-LS proxy (CWZ §2.3 eqs 3-4) under the null on all periods and
    # permutes residuals OVER TIME for the single treated unit. Independent of the
    # cross-unit in-space placebo (that is the Firpo path). The analytical
    # se/t/p/ci stay NaN throughout. See diff_diff/conformal.py and the
    # ## SyntheticControl section of docs/methodology/REGISTRY.md.
    # =====================================================================

    @staticmethod
    def _coerce_q(q: Any) -> Any:
        """Validate the ``S_q`` norm order — must be ``1``, ``2``, or ``inf``."""
        from diff_diff.conformal import _INF

        if isinstance(q, str):
            if q.strip().lower() in ("inf", "infinity"):
                return _INF
            raise ValueError(f"q must be 1, 2, or inf, got {q!r}")
        if q == _INF:
            return _INF
        try:
            qf = float(q)
        except (TypeError, ValueError):
            raise ValueError(f"q must be 1, 2, or inf, got {q!r}")
        if qf == _INF:
            return _INF
        if qf == 1.0:
            return 1
        if qf == 2.0:
            return 2
        raise ValueError(f"q must be 1, 2, or inf, got {q!r}")

    def _conformal_panel(self) -> Tuple[np.ndarray, np.ndarray, int, int, float, List[Any]]:
        """Extract the calendar-ordered (treated, donor) outcome panel for the CWZ layer.

        Returns ``(y1, Y0, n_pre, n_post, pre_scale, donors)`` where ``y1`` ``(T,)`` /
        ``Y0`` ``(T, J)`` are in strict calendar order (sorted pre-period prefix +
        sorted post-period suffix — :attr:`pre_periods` / :attr:`post_periods` are
        built that way in ``fit()``), so the moving-block cyclic shift respects time
        adjacency. ``pre_scale`` is the θ0-invariant pre-window outcome norm used to
        scale the proxy's convergence tolerance. Fails closed if the snapshot is
        unavailable (unpickled result), the panel has non-finite cells, or there are
        no donors / too few periods; warns on a single donor (degenerate proxy).
        """
        if self._fit_snapshot is None:
            raise ValueError(
                "conformal inference requires the fit snapshot on the results object. "
                "This result appears to have been loaded from serialization (which "
                "excludes the snapshot) or produced by an older estimator version. "
                "Re-fit to enable conformal inference."
            )
        snap = self._fit_snapshot
        donors = list(snap.donor_ids)
        if len(donors) < 1:
            raise ValueError("conformal inference requires at least one donor unit.")
        pre = list(snap.pre_periods)
        post = list(snap.post_periods)
        n_pre, n_post = len(pre), len(post)
        if n_pre + n_post < 2:
            raise ValueError("conformal inference requires at least 2 time periods.")
        if n_post < 1:
            raise ValueError("conformal inference requires at least one post period.")
        cal = pre + post  # calendar order
        Y = snap.pivots[snap.outcome]
        y1 = Y.loc[cal, snap.treated_id].to_numpy(dtype=float)
        Y0 = Y.loc[cal, donors].to_numpy(dtype=float)
        if not (np.all(np.isfinite(y1)) and np.all(np.isfinite(Y0))):
            raise ValueError(
                "conformal inference: the outcome panel has non-finite (NaN/inf) cells."
            )
        if len(donors) == 1:
            warnings.warn(
                "conformal inference with a single donor: the synthetic control is forced "
                "to that donor (w=[1]), so the proxy is degenerate and the inference is not "
                "meaningful. Provide >= 2 donors.",
                UserWarning,
                stacklevel=3,
            )
        pre_scale = max(float(np.linalg.norm(y1[:n_pre])), 1e-12)
        return y1, Y0, n_pre, n_post, pre_scale, donors

    def conformal_test(
        self,
        effect: Any,
        *,
        q: Any = 1,
        scheme: str = "moving_block",
        n_iid: int = 10000,
        seed: Optional[int] = None,
    ) -> pd.Series:
        """Joint sharp-null conformal test ``H0: θ = effect`` (Chernozhukov-Wüthrich-Zhu 2021, §2.2).

        Imputes the counterfactual treated outcomes under the null (subtracts the
        hypothesized post-period effect path), fits the canonical CWZ constrained-LS
        synthetic-control proxy on **all** periods under that null (eqs 3-4 — simplex
        weights on raw outcomes, NO V-matrix; distinct from the headline ADH
        V-matrix weights, as the exactness theory requires a time-permutation-invariant
        proxy), and computes the permutation p-value (eq 2) of the statistic
        ``S_q(û) = ((1/√T*)·Σ_{t>T0}|û_t|^q)^{1/q}`` by reshuffling residuals over
        time. The proxy is fit ONCE (footnote 7); only residuals are permuted.

        This is a SEPARATE permutation object from the analytical inference: ``se`` /
        ``t_stat`` / ``p_value`` / ``conf_int`` / ``is_significant`` stay NaN.

        Parameters
        ----------
        effect : float or array-like
            The hypothesized post-period effect trajectory ``θ0``: a scalar (a
            constant-in-time effect) or a length-``n_post_periods`` array aligned to
            ``post_periods`` in calendar order.
        q : {1, 2, inf}, default 1
            The ``S_q`` norm order. ``1`` (robust to heavy tails — the paper's
            application default), ``2`` (permanent effects), ``inf`` (= ``max|û_t|``,
            large temporary effects).
        scheme : {"moving_block", "iid"}, default "moving_block"
            The permutation set. ``"moving_block"`` (``Π_→``, ``T`` cyclic shifts) is
            valid under serially-dependent / stationary weakly-dependent errors
            (Assumption 2.2) — the robust default; ``"iid"`` (``Π_all``, sampled) is
            valid under i.i.d. errors (Assumption 2.1) and gives finer p-values.
        n_iid : int, default 10000
            Number of random permutations drawn for ``scheme="iid"`` (ignored for
            moving-block, which is the exact ``T``-element set). Exact ``T!``
            enumeration is used when ``T! <= n_iid``.
        seed : int, optional
            RNG seed for ``scheme="iid"`` sampling. Default uses the fit's seed.
            Moving-block is deterministic.

        Returns
        -------
        pandas.Series
            ``p_value``, ``S_observed``, ``q``, ``scheme``, ``n_perms`` (``|Π|``),
            ``n_post``, ``proxy_converged``.

        Raises
        ------
        ValueError
            If ``q`` / ``scheme`` / ``n_iid`` are invalid, ``effect`` has the wrong
            shape / non-finite values, or the fit snapshot is unavailable.
        """
        from diff_diff.conformal import _INF, _make_perms, _single_null_pvalue

        q = self._coerce_q(q)
        if scheme not in ("moving_block", "iid"):
            raise ValueError(f"scheme must be 'moving_block' or 'iid', got {scheme!r}")
        if not isinstance(n_iid, (int, np.integer)) or n_iid < 1:
            raise ValueError(f"n_iid must be a positive integer, got {n_iid!r}")
        y1, Y0, n_pre, n_post, pre_scale, _ = self._conformal_panel()
        f_post = self._coerce_effect_path(effect, n_post)
        if n_post >= n_pre:
            warnings.warn(
                "CWZ conformal validity is driven by a large pre-period (T0) relative to "
                f"a short post-period (T*); here T0={n_pre} <= T*={n_post}, so the "
                "finite-sample size guarantee is weak. Interpret with caution.",
                UserWarning,
                stacklevel=2,
            )
        snap = self._fit_snapshot
        assert snap is not None  # _conformal_panel already guarded
        n_t = n_pre + n_post
        post_mask = np.zeros(n_t, dtype=bool)
        post_mask[n_pre:] = True
        rng = np.random.default_rng(snap.seed if seed is None else seed)
        perms = _make_perms(n_t, scheme, int(n_iid), rng)
        res = _single_null_pvalue(
            y1,
            Y0,
            post_mask,
            f_post,
            perms,
            q,
            max_iter=snap.inner_max_iter,
            min_decrease=snap.inner_min_decrease * pre_scale,
        )
        if not res["converged"]:
            warnings.warn(
                "conformal proxy did not fully converge (Frank-Wolfe simplex solve hit "
                "the iteration cap); the p-value uses a near-optimal proxy. Re-fit with a "
                "larger inner_max_iter or a looser inner_min_decrease for a tighter solve.",
                UserWarning,
                stacklevel=2,
            )
        q_out: Any = float("inf") if q == _INF else int(q)
        self.conformal_inference = {
            "kind": "joint",
            "scheme": scheme,
            "q": q_out,
            "alpha": None,
            "n_perms": int(res["n_perms"]),
            "n_post": int(n_post),
            "joint_p_value": float(res["p_value"]),
            "proxy_converged": bool(res["converged"]),
            "status": "ran",
        }
        return pd.Series(
            {
                "p_value": float(res["p_value"]),
                "S_observed": float(res["s_observed"]),
                "q": q_out,
                "scheme": scheme,
                "n_perms": int(res["n_perms"]),
                "n_post": int(n_post),
                "proxy_converged": bool(res["converged"]),
            }
        )

    def conformal_average_effect(
        self,
        *,
        alpha: float = 0.1,
        scheme: str = "moving_block",
        n_iid: int = 10000,
        bounds: Optional[Tuple[float, float]] = None,
        n_grid: int = 200,
        seed: Optional[int] = None,
    ) -> pd.Series:
        """Confidence interval for the AVERAGE post-period effect (Chernozhukov-Wüthrich-Zhu 2021, Appendix A.1).

        Tests ``H0: T*^{-1}·Σ_{t>T0} θ_t = θ̄0`` by **collapsing** the panel into
        non-overlapping ``T*``-blocks (each a per-unit block average), fitting the CWZ
        proxy on the collapsed series, and permuting the **block** residuals — the
        ``T/T*``-block analog of :meth:`conformal_test` (a single post-block). The CI
        is every ``θ̄0`` not rejected at level ``alpha`` (test inversion). The earliest
        ``T0 mod T*`` pre-periods are dropped so the pre-block count is integral (the
        paper assumes ``T/T*`` integer).

        Because the effective sample is only ``T/T*`` blocks, the moving-block
        permutation set has just ``T/T*`` elements (p-value granularity ``T*/T``);
        pass ``scheme="iid"`` for a finer set (``(T/T*)!`` block permutations) when the
        block count is small. Analytical ``se`` / ``conf_int`` stay NaN.

        Parameters
        ----------
        alpha : float, default 0.1
            The confidence level is ``1 − alpha``; membership is ``p^θ̄0 > alpha``.
        scheme : {"moving_block", "iid"}, default "moving_block"
            Permutation set over the collapsed blocks.
        n_iid : int, default 10000
            Random block-permutation draws for ``scheme="iid"`` (exact ``(T/T*)!``
            enumeration when it fits).
        bounds : (float, float), optional
            Fixed ``(lo, hi)`` grid for ``θ̄0``. Default None auto-centres the grid on
            the average-effect point estimate (membership outside the grid is not
            certified — flagged via ``status="grid_limited"``).
        n_grid : int, default 200
            Number of grid points (>= 2).
        seed : int, optional
            RNG seed for ``scheme="iid"``. Default uses the fit's seed.

        Returns
        -------
        pandas.Series
            ``lower``, ``upper``, ``point_estimate`` (the average-effect estimate),
            ``status`` (``"ran"``/``"grid_limited"``/``"empty"``/``"unbounded"``), ``contiguous``,
            ``n_perms``, ``n_blocks``, ``n_dropped_pre``, ``n_grid_nonconverged``.

        Raises
        ------
        ValueError
            If ``alpha`` / ``scheme`` / ``n_iid`` / ``n_grid`` / ``bounds`` are invalid,
            ``T0 < T*`` (no full pre-block), or the fit snapshot is unavailable.
        """
        from diff_diff.conformal import _block_collapse, _invert_single_post, _make_perms

        if scheme not in ("moving_block", "iid"):
            raise ValueError(f"scheme must be 'moving_block' or 'iid', got {scheme!r}")
        if not isinstance(n_iid, (int, np.integer)) or n_iid < 1:
            raise ValueError(f"n_iid must be a positive integer, got {n_iid!r}")
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
        if not isinstance(n_grid, (int, np.integer)) or n_grid < 2:
            raise ValueError(f"n_grid must be an integer >= 2, got {n_grid!r}")
        grid = _validate_conformal_bounds(bounds, int(n_grid))
        y1, Y0, n_pre, n_post, _, _ = self._conformal_panel()
        if n_pre < n_post:
            raise ValueError(
                f"conformal_average_effect requires T0 >= T* to form at least one full "
                f"pre-block, got T0={n_pre} < T*={n_post}."
            )
        if n_post >= n_pre:
            warnings.warn(
                "CWZ conformal validity is driven by a large pre-period (T0) relative to "
                f"a short post-period (T*); here T0={n_pre} <= T*={n_post}, so the "
                "finite-sample size guarantee is weak. Interpret with caution.",
                UserWarning,
                stacklevel=2,
            )
        y1b, Y0b, n_dropped = _block_collapse(y1, Y0, n_pre, n_post)
        if n_dropped:
            warnings.warn(
                f"conformal_average_effect: T0={n_pre} is not a multiple of T*={n_post}; "
                f"dropping the earliest {n_dropped} pre-period(s) to form integral T*-blocks "
                "(CWZ Appendix A.1).",
                UserWarning,
                stacklevel=2,
            )
        n_blocks = int(y1b.shape[0])
        snap = self._fit_snapshot
        assert snap is not None
        rng = np.random.default_rng(snap.seed if seed is None else seed)
        perms = _make_perms(n_blocks, scheme, int(n_iid), rng)
        n_perms = int(perms.shape[0])
        if float(alpha) < 1.0 / n_perms:
            warnings.warn(
                f"alpha={alpha:.3g} is below the permutation granularity 1/|Pi|=1/{n_perms} "
                f"(the average effect collapses to {n_blocks} blocks), so no value is ever "
                "rejected and the interval is the whole grid (unbounded). Use scheme='iid' "
                "for a finer block-permutation set or a larger alpha.",
                UserWarning,
                stacklevel=2,
            )
        block_scale = max(float(np.linalg.norm(y1b[: n_blocks - 1])), 1e-12)
        res = _invert_single_post(
            y1b,
            Y0b,
            n_blocks - 1,
            float(alpha),
            perms,
            max_iter=snap.inner_max_iter,
            min_decrease=snap.inner_min_decrease * block_scale,
            grid=grid,
            n_grid=int(n_grid),
        )
        _warn_conformal_ci_status(res, "conformal_average_effect")
        self.conformal_inference = {
            "kind": "average",
            "scheme": scheme,
            "alpha": float(alpha),
            "n_perms": n_perms,
            "n_post": int(n_post),
            "n_blocks": n_blocks,
            "n_dropped_pre": int(n_dropped),
            "lower": float(res["lower"]),
            "upper": float(res["upper"]),
            "point_estimate": float(res["point_estimate"]),
            "contiguous": bool(res["contiguous"]),
            "status": res["status"],
        }
        self._conformal_grid_df = pd.DataFrame(
            res["grid"], columns=["param", "p_value", "in_set", "converged"]
        )
        return pd.Series(
            {
                "lower": float(res["lower"]),
                "upper": float(res["upper"]),
                "point_estimate": float(res["point_estimate"]),
                "status": res["status"],
                "contiguous": bool(res["contiguous"]),
                "n_perms": n_perms,
                "n_blocks": n_blocks,
                "n_dropped_pre": int(n_dropped),
                "n_grid_nonconverged": int(res["n_nonconverged"]),
            }
        )

    def get_conformal_grid_df(self) -> pd.DataFrame:
        """Get the conformal test-inversion grid table (see :meth:`conformal_average_effect` / :meth:`conformal_confidence_intervals`).

        Columns: ``param`` (the grid value), ``p_value`` (``p^param``), ``in_set``
        (``= not (converged and p_value <= alpha)`` — a non-converged grid point is
        indeterminate and stays in the set, so ``in_set`` can be ``True`` even when the
        displayed ``p_value`` is not ``> alpha``), and ``converged`` (the proxy
        Frank-Wolfe convergence flag for that grid point). For pointwise CIs the table is
        the concatenation across post periods (with a ``period`` column). A
        granularity-``unbounded`` interval (``alpha < 1/|Π|``) short-circuits and returns
        an EMPTY grid. Survives pickling. Raises if no conformal inversion has been run.

        Returns
        -------
        pandas.DataFrame
        """
        if self._conformal_grid_df is None:
            raise ValueError(
                "No conformal inversion grid yet; call conformal_average_effect() or "
                "conformal_confidence_intervals() first."
            )
        return self._conformal_grid_df.copy()

    def conformal_confidence_intervals(
        self,
        *,
        alpha: float = 0.1,
        scheme: str = "moving_block",
        n_iid: int = 10000,
        bounds: Optional[Tuple[float, float]] = None,
        n_grid: int = 100,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Pointwise per-period conformal confidence intervals (Chernozhukov-Wüthrich-Zhu 2021, Algorithm 1).

        For each post period ``t``, inverts a conformal test of ``H0: θ_t = c`` over a
        grid of ``c``. Per the paper (§2.2), each per-period test uses the data
        ``Z = (Z_1, …, Z_{T0}, Z_t)`` — the ``T0`` pre-periods PLUS only period ``t``,
        with the **other post-periods dropped** — so it is a clean single-post-period
        (``T*=1``) conformal test on the ``(T0+1)``-length sub-series: impute
        ``Y_{1t} − c``, refit the CWZ proxy on the sub-series, permute the residuals,
        and keep ``c`` iff ``p^c > alpha``. (Because ``T*=1`` here, the ``S_q`` order
        ``q`` is inert — ``S_q = |û_t|`` for every ``q`` — so it is not a parameter.)
        The analytical ``conf_int`` stays ``(NaN, NaN)`` — this is a separate
        permutation object.

        Parameters
        ----------
        alpha : float, default 0.1
            The confidence level is ``1 − alpha``; membership is ``p^c > alpha``.
        scheme : {"moving_block", "iid"}, default "moving_block"
            Permutation set over the ``(T0+1)``-length sub-series.
        n_iid : int, default 10000
            Random permutation draws for ``scheme="iid"``.
        bounds : (float, float), optional
            A single fixed ``(lo, hi)`` grid applied to EVERY period. Default None
            auto-centres a per-period grid on that period's point estimate (membership
            outside the grid is not certified — flagged ``status="grid_limited"``).
        n_grid : int, default 100
            Grid points per period (>= 2).
        seed : int, optional
            RNG seed for ``scheme="iid"``. Default uses the fit's seed.

        Returns
        -------
        pandas.DataFrame
            One row per post period: ``period``, ``lower``, ``upper``,
            ``point_estimate``, ``status`` (``"ran"``/``"grid_limited"``/``"empty"``/``"unbounded"``),
            ``contiguous``, ``n_grid_in_set``, ``n_grid_nonconverged``. The full
            per-period inversion grid is on :meth:`get_conformal_grid_df`.

        Raises
        ------
        ValueError
            If ``alpha`` / ``scheme`` / ``n_iid`` / ``n_grid`` / ``bounds`` are invalid
            or the fit snapshot is unavailable.
        """
        from diff_diff.conformal import _invert_single_post, _make_perms

        if scheme not in ("moving_block", "iid"):
            raise ValueError(f"scheme must be 'moving_block' or 'iid', got {scheme!r}")
        if not isinstance(n_iid, (int, np.integer)) or n_iid < 1:
            raise ValueError(f"n_iid must be a positive integer, got {n_iid!r}")
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
        if not isinstance(n_grid, (int, np.integer)) or n_grid < 2:
            raise ValueError(f"n_grid must be an integer >= 2, got {n_grid!r}")
        grid_template = _validate_conformal_bounds(bounds, int(n_grid))
        y1, Y0, n_pre, n_post, pre_scale, _ = self._conformal_panel()
        if n_pre <= 1:
            warnings.warn(
                "CWZ conformal validity is driven by a large pre-period (T0); each pointwise "
                f"CI fits its proxy on a (T0+1)-length sub-series and here T0={n_pre} <= 1, so "
                "the finite-sample size guarantee is weak. Interpret with caution.",
                UserWarning,
                stacklevel=2,
            )
        snap = self._fit_snapshot
        assert snap is not None
        post_periods = list(snap.post_periods)
        m = n_pre + 1  # sub-series length (T0 pre + the single tested period)
        rng = np.random.default_rng(snap.seed if seed is None else seed)
        perms = _make_perms(m, scheme, int(n_iid), rng)
        n_perms = int(perms.shape[0])
        if float(alpha) < 1.0 / n_perms:
            warnings.warn(
                f"alpha={alpha:.3g} is below the permutation granularity 1/|Pi|=1/{n_perms}, "
                "so no value is ever rejected and every per-period interval is the whole grid "
                "(unbounded). Use scheme='iid' for a finer set or a larger alpha.",
                UserWarning,
                stacklevel=2,
            )
        md = snap.inner_min_decrease * pre_scale  # pre window is theta0-invariant
        ci_rows: List[Dict[str, Any]] = []
        grid_rows: List[Dict[str, Any]] = []
        statuses: List[str] = []
        any_noncontig = False
        for k, period in enumerate(post_periods):
            sub_idx = list(range(n_pre)) + [n_pre + k]
            res = _invert_single_post(
                y1[sub_idx],
                Y0[sub_idx],
                m - 1,
                float(alpha),
                perms,
                max_iter=snap.inner_max_iter,
                min_decrease=md,
                grid=grid_template,
                n_grid=int(n_grid),
            )
            ci_rows.append(
                {
                    "period": period,
                    "lower": float(res["lower"]),
                    "upper": float(res["upper"]),
                    "point_estimate": float(res["point_estimate"]),
                    "status": res["status"],
                    "contiguous": bool(res["contiguous"]),
                    "n_grid_in_set": int(res["n_in_set"]),
                    "n_grid_nonconverged": int(res["n_nonconverged"]),
                }
            )
            statuses.append(res["status"])
            any_noncontig = any_noncontig or not res["contiguous"]
            for theta, p, in_set, conv in res["grid"]:
                grid_rows.append(
                    {
                        "period": period,
                        "param": theta,
                        "p_value": p,
                        "in_set": in_set,
                        "converged": conv,
                    }
                )
        n_empty = statuses.count("empty")
        n_gl = statuses.count("grid_limited")
        n_unbounded = statuses.count("unbounded")
        if n_empty or n_gl or n_unbounded or any_noncontig:
            warnings.warn(
                "conformal_confidence_intervals: "
                f"{n_empty} period(s) empty, {n_gl} grid-limited (CI may extend beyond the "
                f"scanned grid — pass bounds= / widen n_grid), {n_unbounded} unbounded"
                + (", some non-contiguous (hull reported)" if any_noncontig else "")
                + ". Inspect get_conformal_grid_df().",
                UserWarning,
                stacklevel=2,
            )
        self._conformal_ci_df = pd.DataFrame(
            ci_rows,
            columns=[
                "period",
                "lower",
                "upper",
                "point_estimate",
                "status",
                "contiguous",
                "n_grid_in_set",
                "n_grid_nonconverged",
            ],
        )
        self._conformal_grid_df = pd.DataFrame(
            grid_rows, columns=["period", "param", "p_value", "in_set", "converged"]
        )
        self.conformal_inference = {
            "kind": "pointwise",
            "scheme": scheme,
            "alpha": float(alpha),
            "n_perms": n_perms,
            "n_post": int(n_post),
            "n_grid_limited": int(n_gl),
            "n_empty": int(n_empty),
            "n_unbounded": int(n_unbounded),
            "status": "ran",
        }
        return self._conformal_ci_df.copy()

    def get_conformal_ci_df(self) -> pd.DataFrame:
        """Get the pointwise per-period conformal CI table (see :meth:`conformal_confidence_intervals`).

        One row per post period: ``period``, ``lower``, ``upper``, ``point_estimate``,
        ``status``, ``contiguous``, ``n_grid_in_set``, ``n_grid_nonconverged``. Survives
        pickling. Raises if :meth:`conformal_confidence_intervals` has not been run.

        Returns
        -------
        pandas.DataFrame
        """
        if self._conformal_ci_df is None:
            raise ValueError(
                "No pointwise conformal CIs yet; call conformal_confidence_intervals() first."
            )
        return self._conformal_ci_df.copy()
