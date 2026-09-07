"""Results containers for :class:`~diff_diff.DurationDiD` (Deaner & Ku 2026).

Two containers live here:

- :class:`DurationDiDResults` — the estimator result (``BaseResults``): the
  headline absorption ATT (uniform average over the post-treatment dates),
  the per-date ATT path with pointwise and simultaneous centered-bootstrap
  bands, the fitted hazard relationship, the survival curves, and the
  bootstrap diagnostics.
- :class:`DurationDiDPretestResults` — the Appendix B Algorithm 2
  fixed-anchor pre-treatment specification test (``Diagnostic``).

Every inference family (headline, post-period path, pretest) is either fully
available or fully withheld (joint NaN); there is no partially populated
state. Statuses name the reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.aggregation import AggregationMixin
from diff_diff.results import _get_significance_stars
from diff_diff.results_base import (
    BaseResults,
    Diagnostic,
    EventStudyResults,
    _coverage_pct,
    _json_safe_label,
    _require_fit_alpha,
)

_METHOD_TITLES = {
    "cd": "common dynamics (CD): additive untreated-hazard gap",
    "ph": "proportional hazards (PH): multiplicative untreated-hazard ratio",
}


def _to_list(arr: Any) -> Any:
    """JSON-safe list form of an array-like (NaN kept as float NaN)."""
    if arr is None:
        return None
    return np.asarray(arr).tolist()


def invalid_curve_message(periods: Any, curve_status: List[str], last_pre_period: Any) -> str:
    """Diagnostic sentence for an invalid imputed counterfactual curve.

    Always names the flagged dates and statuses, then the remedy that can
    actually help: a fitted PRE-date violation cannot be repaired by a shorter
    horizon (the fit itself must change); a post-date violation admits an
    explicit refit on dates at or before the last date strictly before the
    first invalid post date, when at least one post date survives.
    """
    labels = [p.item() if hasattr(p, "item") else p for p in np.asarray(periods).tolist()]
    grid_f = np.asarray(periods, dtype=float)
    tstar_idx = int(np.nonzero(grid_f == float(last_pre_period))[0][0])
    flagged = [(labels[t], s) for t, s in enumerate(curve_status) if s != "ok"]
    msg = (
        "Invalid imputed counterfactual curve at "
        + ", ".join(f"{p!r} ({s})" for p, s in flagged)
        + ". No valid causal inference is reported for the post-period family; "
        "raw extrapolations are retained where finite."
    )
    pre_bad = any(s != "ok" for s in curve_status[1 : tstar_idx + 1])
    post_bad = [t for t in range(tstar_idx + 1, len(curve_status)) if curve_status[t] != "ok"]
    if pre_bad:
        msg += (
            " The fitted pre-treatment curve itself leaves the domain, so a shorter "
            "horizon cannot repair it: change the fit (method, pre_periods / "
            "pre_period_weights) or the design."
        )
    if post_bad:
        first_bad = post_bad[0]
        if first_bad > tstar_idx + 1:
            msg += (
                " A shorter horizon must be an explicit refit choice: subset the data "
                f"to dates at or before {labels[first_bad - 1]!r} and refit."
            )
        else:
            msg += (
                " The first post-treatment date is already invalid, so no shorter "
                "horizon with a valid counterfactual exists."
            )
    return msg


def _invalid_draw_line(n_invalid: int, n_valid: int) -> str:
    """Summary line for complete draws whose imputed curve left the domain."""
    if n_valid <= 0:
        return "  Out-of-domain imputed curves among complete draws: n/a (no complete draws)"
    share = 100.0 * n_invalid / n_valid
    return (
        f"  Out-of-domain imputed curves among complete draws: {n_invalid}/{n_valid} "
        f"({share:.1f}%) — retained as statistics, not failures; a large share signals "
        "weak extrapolation support"
    )


def _scalar(x: Any) -> Any:
    """Native Python scalar for numpy scalars; passthrough otherwise."""
    if isinstance(x, np.generic):
        return x.item()
    return x


@dataclass
class DurationDiDPretestResults(Diagnostic):
    """Appendix B Algorithm 2 pre-treatment specification test.

    For every interior pre-treatment date ``t`` (strictly between the
    baseline and ``anchor_period``, the last pre-treatment date) the
    contrast compares the hazard relationship at ``t`` with its value at the
    fixed anchor: the average-hazard GAP under ``method="cd"`` and the
    cumulative-increment RATIO under ``method="ph"``. The whole-individual
    bootstrap gives one SD per contrast, a simultaneous critical value from
    the maximum absolute centered pivot, symmetric bands
    ``contrast +/- crit_value * se`` and the max-|t| statistic. The test
    rejects when any band excludes zero, which is exactly ``p_value <=
    alpha`` under the inverse-empirical-CDF quantile.

    Attributes
    ----------
    method : str
        ``"cd"`` or ``"ph"``.
    periods : np.ndarray
        The tested interior pre-treatment dates (``J``).
    anchor_period : Any
        The last pre-treatment date (the fixed anchor).
    contrast : np.ndarray
        Point contrasts, one per tested date (NaN when unavailable).
    se : np.ndarray
        Bootstrap SDs of the contrasts (NaN when unavailable).
    band_lower, band_upper : np.ndarray
        Simultaneous bands at level ``1 - alpha``.
    crit_value : float
        Simultaneous critical value (NaN when unavailable).
    statistic : float
        ``max |contrast / se|`` over the tested dates.
    p_value : float
        Empirical tail fraction of the bootstrap maxima at or above
        ``statistic`` (NaN when unavailable).
    reject : bool or None
        ``True`` when any band excludes zero; ``None`` when unavailable.
    alpha : float
        Significance level.
    n_bootstrap, n_bootstrap_valid : int
        Requested draws and complete draws for this family.
    status : str
        ``"ok"``, ``"disabled"`` (``n_bootstrap=0``),
        ``"unavailable_insufficient_pre_periods"`` (no interior pre-date),
        ``"unavailable_ph_support"`` (a zero control increment at a tested
        date or the anchor under PH), ``"unavailable_nonfinite_moments"``,
        ``"unavailable_failed_draws"``, or ``"unavailable_zero_se"``.
    """

    method: str
    periods: np.ndarray
    anchor_period: Any
    contrast: np.ndarray
    se: np.ndarray
    band_lower: np.ndarray
    band_upper: np.ndarray
    crit_value: float
    statistic: float
    p_value: float
    reject: Optional[bool]
    alpha: float
    n_bootstrap: int
    n_bootstrap_valid: int
    status: str

    def to_dataframe(self) -> pd.DataFrame:
        """One row per tested pre-treatment date."""
        return pd.DataFrame(
            {
                "period": np.asarray(self.periods).tolist(),
                "contrast": np.asarray(self.contrast, dtype=float),
                "se": np.asarray(self.se, dtype=float),
                "band_lower": np.asarray(self.band_lower, dtype=float),
                "band_upper": np.asarray(self.band_upper, dtype=float),
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable dictionary."""
        return {
            "method": self.method,
            "periods": [_json_safe_label(p) for p in np.asarray(self.periods).tolist()],
            "anchor_period": _json_safe_label(self.anchor_period),
            "contrast": _to_list(self.contrast),
            "se": _to_list(self.se),
            "band_lower": _to_list(self.band_lower),
            "band_upper": _to_list(self.band_upper),
            "crit_value": float(self.crit_value),
            "statistic": float(self.statistic),
            "p_value": float(self.p_value),
            "reject": self.reject,
            "alpha": float(self.alpha),
            "n_bootstrap": int(self.n_bootstrap),
            "n_bootstrap_valid": int(self.n_bootstrap_valid),
            "status": self.status,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        kind = "average-hazard gap" if self.method == "cd" else "cumulative-increment ratio"
        lines = [
            "DurationDiD pre-treatment specification test (Algorithm 2, fixed anchor)",
            f"  Contrast: {kind} at each interior pre-date minus its value at "
            f"the anchor {self.anchor_period!r}",
            f"  Status: {self.status}",
            f"  Tested dates: {np.asarray(self.periods).tolist()}",
        ]
        if self.status == "ok":
            lines.append(
                f"  max|t| = {self.statistic:.4f}, simultaneous {_coverage_pct(self.alpha)}% "
                f"critical value = {self.crit_value:.4f}, p-value = {self.p_value:.4f}, "
                f"reject = {self.reject} ({self.n_bootstrap_valid}/{self.n_bootstrap} "
                "complete draws)"
            )
            lines.append("  Failure to reject does not establish post-treatment identification.")
        else:
            lines.append("  Inference unavailable for this diagnostic (see status).")
        if len(np.asarray(self.periods)) > 0:
            lines.append("")
            lines.append(self.to_dataframe().to_string(index=False))
        return "\n".join(lines)


@dataclass
class DurationDiDResults(BaseResults, AggregationMixin):
    """Results of :class:`~diff_diff.DurationDiD`.

    The headline ``att`` is the uniform average over the post-treatment
    dates of the absorption ATT ``E[Y_t - Y_t(0) | treated]`` (positive =
    more cumulative absorption/exit than the counterfactual). Per-date
    effects and both band families live in the ``*_by_period`` / ``cband_*``
    fields. Inference is whole-individual pooled bootstrap only; every family
    is either available (``inference_status == "ok"``) or fully withheld.

    Attributes
    ----------
    att, se, t_stat, p_value, conf_int
        Headline inference (centered-bootstrap SD, p-value and symmetric
        interval). ``p_value <= alpha`` is exactly the band rule.
    method : str
        ``"cd"`` or ``"ph"``.
    n_units : int
        Individuals (the bootstrap resampling unit).
    n_obs : int
        Panel rows used (``n_units * n_periods``).
    n_treated, n_control : int
        Individuals per group.
    periods : np.ndarray
        The common time grid (all dates, sorted).
    last_pre_period : Any
        The last untreated date (``tstar``).
    post_periods : np.ndarray
        Dates strictly after ``last_pre_period`` (the declared post family).
    pre_periods, pre_period_weights : np.ndarray
        The realized fitting dates and their normalized weights.
    excluded_pre_periods : dict
        Candidate fitting dates that were excluded, with the reason.
    coefficient : float
        Fitted CD gap ``c`` (per unit of time) or PH ratio ``c``.
    survival_treated, survival_control, counterfactual_survival : np.ndarray
        Group survival curves and the imputed treated counterfactual over
        the whole grid (pre-dates are the fitted values); NaN where no
        counterfactual exists or the value is non-finite.
    att_by_period, se_by_period, t_stat_by_period, p_value_by_period
        Per post-date effect and pointwise inference.
    conf_int_by_period : np.ndarray
        ``(P, 2)`` pointwise intervals.
    pointwise_crit_values, cband_lower, cband_upper, cband_crit_value
        Pointwise critical values and the simultaneous band.
    joint_p_value : float
        Simultaneous p-value for the null that every post-date effect is zero.
    vcov : np.ndarray or None
        ``(P, P)`` bootstrap covariance of the post-date effects; ``None``
        whenever the post family is unavailable.
    curve_status, period_status : list of str
        Validity of the imputed counterfactual curve at every date /
        every post-date (``"ok"``, ``"control_survival_zero"``,
        ``"counterfactual_nonfinite"``, ``"counterfactual_survival_above_one"``,
        ``"counterfactual_nonmonotone"``).
    inference_status : str
        ``"ok"``, ``"disabled"``, ``"unavailable_invalid_periods"``,
        ``"unavailable_failed_draws"`` or ``"unavailable_zero_se"``.
    bootstrap_effects : np.ndarray or None
        ``(n_bootstrap, P)`` raw post-effect draws (NaN rows for failed
        draws). Statistics of the completed draws are diagnostics, never
        inference.
    bootstrap_failure_reasons : dict
        ``{"post": {reason: count}, "pretest": {reason: count}}``.
    n_draws_invalid_counterfactual : int
        Complete post-family draws whose imputed counterfactual curve left the
        domain (survival above one or a decreasing cumulative hazard). Such
        draws are well-defined statistics and are NOT failures, but a large
        share signals weak support for the CD extrapolation; reported in
        ``summary()`` as a count and a share of the complete draws.
    pretest : DurationDiDPretestResults
        The Algorithm 2 specification test (always populated).
    """

    att: float
    se: float
    t_stat: float
    p_value: float
    conf_int: Tuple[float, float]
    method: str
    alpha: float
    n_bootstrap: int
    n_bootstrap_valid: int
    n_bootstrap_valid_pretest: int
    seed: Optional[int]
    n_units: int
    n_obs: int
    n_treated: int
    n_control: int
    n_periods: int
    periods: np.ndarray
    last_pre_period: Any
    post_periods: np.ndarray
    pre_periods: np.ndarray
    pre_period_weights: np.ndarray
    excluded_pre_periods: Dict[Any, str]
    coefficient: float
    ph_ratio_boundary: bool
    n_treated_survivors_at_last_pre: int
    n_control_survivors_at_horizon: int
    survival_treated: np.ndarray
    survival_control: np.ndarray
    counterfactual_survival: np.ndarray
    att_by_period: np.ndarray
    se_by_period: np.ndarray
    t_stat_by_period: np.ndarray
    p_value_by_period: np.ndarray
    conf_int_by_period: np.ndarray
    pointwise_crit_values: np.ndarray
    cband_lower: np.ndarray
    cband_upper: np.ndarray
    cband_crit_value: float
    joint_p_value: float
    vcov: Optional[np.ndarray]
    curve_status: List[str]
    period_status: List[str]
    inference_status: str
    bootstrap_effects: Optional[np.ndarray]
    bootstrap_failure_reasons: Dict[str, Dict[str, int]]
    n_draws_invalid_counterfactual: int
    pretest: DurationDiDPretestResults

    _AGGREGATE_SUPPORTED: ClassVar[Tuple[str, ...]] = ("event_study",)
    _AGGREGATE_BALANCE_E_TYPES: ClassVar[Tuple[str, ...]] = ()

    # ------------------------------------------------------------------ #
    # Convenience                                                          #
    # ------------------------------------------------------------------ #
    @property
    def is_significant(self) -> bool:
        """Band rule: the headline centered-bootstrap interval excludes zero.

        Under the inverse-empirical-CDF quantile this is exactly
        ``p_value <= alpha``; ``False`` when inference is unavailable.
        """
        return bool(np.isfinite(self.p_value) and self.p_value <= self.alpha)

    @property
    def significance_stars(self) -> str:
        """Significance stars for the headline p-value ("" when NaN)."""
        return _get_significance_stars(self.p_value)

    # ------------------------------------------------------------------ #
    # Serialization                                                        #
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable dictionary (arrays as lists, scalars native)."""
        return {
            "att": float(self.att),
            "se": float(self.se),
            "t_stat": float(self.t_stat),
            "p_value": float(self.p_value),
            "conf_int_lower": float(self.conf_int[0]),
            "conf_int_upper": float(self.conf_int[1]),
            "method": self.method,
            "alpha": float(self.alpha),
            "inference_method": "bootstrap" if self.n_bootstrap > 0 else "none",
            "inference_status": self.inference_status,
            "n_bootstrap": int(self.n_bootstrap),
            "n_bootstrap_valid": int(self.n_bootstrap_valid),
            "n_bootstrap_valid_pretest": int(self.n_bootstrap_valid_pretest),
            "seed": _scalar(self.seed),
            "n_units": int(self.n_units),
            "n_obs": int(self.n_obs),
            "n_treated": int(self.n_treated),
            "n_control": int(self.n_control),
            "n_periods": int(self.n_periods),
            "periods": [_json_safe_label(p) for p in np.asarray(self.periods).tolist()],
            "last_pre_period": _json_safe_label(self.last_pre_period),
            "post_periods": [_json_safe_label(p) for p in np.asarray(self.post_periods).tolist()],
            "pre_periods": [_json_safe_label(p) for p in np.asarray(self.pre_periods).tolist()],
            "pre_period_weights": _to_list(self.pre_period_weights),
            "excluded_pre_periods": {
                str(_json_safe_label(k)): v for k, v in self.excluded_pre_periods.items()
            },
            "coefficient": float(self.coefficient),
            "ph_ratio_boundary": bool(self.ph_ratio_boundary),
            "n_treated_survivors_at_last_pre": int(self.n_treated_survivors_at_last_pre),
            "n_control_survivors_at_horizon": int(self.n_control_survivors_at_horizon),
            "survival_treated": _to_list(self.survival_treated),
            "survival_control": _to_list(self.survival_control),
            "counterfactual_survival": _to_list(self.counterfactual_survival),
            "att_by_period": _to_list(self.att_by_period),
            "se_by_period": _to_list(self.se_by_period),
            "t_stat_by_period": _to_list(self.t_stat_by_period),
            "p_value_by_period": _to_list(self.p_value_by_period),
            "conf_int_by_period": _to_list(self.conf_int_by_period),
            "pointwise_crit_values": _to_list(self.pointwise_crit_values),
            "cband_lower": _to_list(self.cband_lower),
            "cband_upper": _to_list(self.cband_upper),
            "cband_crit_value": float(self.cband_crit_value),
            "joint_p_value": float(self.joint_p_value),
            "vcov": _to_list(self.vcov),
            "curve_status": list(self.curve_status),
            "period_status": list(self.period_status),
            "bootstrap_failure_reasons": {
                k: {kk: int(vv) for kk, vv in v.items()}
                for k, v in self.bootstrap_failure_reasons.items()
            },
            "n_draws_invalid_counterfactual": int(self.n_draws_invalid_counterfactual),
            "pretest": self.pretest.to_dict(),
        }

    def to_dataframe(self, level: str = "periods") -> pd.DataFrame:
        """Tabular view.

        Parameters
        ----------
        level : {"periods", "att"}
            ``"periods"``: one row per post-treatment date with the effect,
            pointwise inference, simultaneous band and validity status.
            ``"att"``: the single headline row.
        """
        if level == "att":
            return pd.DataFrame(
                [
                    {
                        "att": self.att,
                        "se": self.se,
                        "t_stat": self.t_stat,
                        "p_value": self.p_value,
                        "conf_int_lower": self.conf_int[0],
                        "conf_int_upper": self.conf_int[1],
                        "inference_status": self.inference_status,
                    }
                ]
            )
        if level == "periods":
            ci = np.asarray(self.conf_int_by_period, dtype=float).reshape(-1, 2)
            return pd.DataFrame(
                {
                    "period": np.asarray(self.post_periods).tolist(),
                    "att": np.asarray(self.att_by_period, dtype=float),
                    "se": np.asarray(self.se_by_period, dtype=float),
                    "t_stat": np.asarray(self.t_stat_by_period, dtype=float),
                    "p_value": np.asarray(self.p_value_by_period, dtype=float),
                    "conf_int_lower": ci[:, 0],
                    "conf_int_upper": ci[:, 1],
                    "cband_lower": np.asarray(self.cband_lower, dtype=float),
                    "cband_upper": np.asarray(self.cband_upper, dtype=float),
                    "status": list(self.period_status),
                }
            )
        raise ValueError(f"level must be 'periods' or 'att', got {level!r}")

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #
    def summary(self, alpha: Optional[float] = None) -> str:
        """Formatted summary. ``alpha`` must equal the fit alpha."""
        fit_alpha = _require_fit_alpha(alpha, self.alpha)
        pct = _coverage_pct(fit_alpha)
        lines = [
            "=" * 78,
            "Duration Difference-in-Differences (Deaner & Ku 2026)".center(78),
            "=" * 78,
            f"Hazard restriction: {_METHOD_TITLES.get(self.method, self.method)}",
            f"Individuals: {self.n_units} ({self.n_treated} treated, {self.n_control} control); "
            f"Observations: {self.n_obs}; Periods: {self.n_periods}",
            f"Last pre-treatment period: {self.last_pre_period!r}; "
            f"post-treatment periods: {len(self.post_periods)}",
            f"Fitting periods: {np.asarray(self.pre_periods).tolist()} "
            f"(weights {np.round(np.asarray(self.pre_period_weights, dtype=float), 4).tolist()})",
            f"Fitted coefficient: {self.coefficient:.6f}",
            "",
            f"Inference: whole-individual bootstrap ({self.n_bootstrap_valid}/{self.n_bootstrap} "
            f"complete draws); status = {self.inference_status}",
            _invalid_draw_line(self.n_draws_invalid_counterfactual, self.n_bootstrap_valid),
            "-" * 78,
            "Headline (uniform average of post-period absorption ATTs)",
            f"  ATT = {self.att:.6f}   SE = {self.se:.6f}   t = {self.t_stat:.4f}   "
            f"p = {self.p_value:.4f} {self.significance_stars}",
            f"  {pct}% CI: [{self.conf_int[0]:.6f}, {self.conf_int[1]:.6f}]",
            "",
            f"Post-period effects ({pct}% pointwise CI and simultaneous band; "
            f"simultaneous crit = {self.cband_crit_value:.4f}, joint p = {self.joint_p_value:.4f})",
        ]
        frame = self.to_dataframe(level="periods")
        lines.append(frame.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
        flagged = [
            (p, s)
            for p, s in zip(np.asarray(self.periods).tolist(), self.curve_status)
            if s != "ok"
        ]
        if flagged:
            lines.append("")
            lines.append(
                invalid_curve_message(self.periods, self.curve_status, self.last_pre_period)
            )
        if self.excluded_pre_periods:
            lines.append("")
            lines.append(f"Excluded fitting periods: {self.excluded_pre_periods}")
        lines.append("")
        lines.append(self.pretest.summary())
        lines.append("=" * 78)
        return "\n".join(lines)

    def print_summary(self, alpha: Optional[float] = None) -> None:
        """Print :meth:`summary`."""
        print(self.summary(alpha=alpha))

    # ------------------------------------------------------------------ #
    # Post-fit aggregation (AggregationMixin hook)                         #
    # ------------------------------------------------------------------ #
    def _aggregate_compute(
        self,
        level: str,
        *,
        weights: Optional[str],
        balance_e: Optional[int],
    ) -> Any:
        if level != "event_study":  # pragma: no cover - mixin validates first
            raise ValueError(f"Unsupported aggregation method: {level!r}")
        n_post = len(self.post_periods)
        event_time = np.arange(-1, n_post)
        nan = np.nan
        att = np.concatenate([[0.0], np.asarray(self.att_by_period, dtype=float)])
        se = np.concatenate([[nan], np.asarray(self.se_by_period, dtype=float)])
        t_stat = np.concatenate([[nan], np.asarray(self.t_stat_by_period, dtype=float)])
        p_value = np.concatenate([[nan], np.asarray(self.p_value_by_period, dtype=float)])
        ci = np.asarray(self.conf_int_by_period, dtype=float).reshape(-1, 2)
        ci_lo = np.concatenate([[nan], ci[:, 0]])
        ci_hi = np.concatenate([[nan], ci[:, 1]])
        is_reference = np.zeros(n_post + 1, dtype=bool)
        is_reference[0] = True
        has_band = bool(np.isfinite(self.cband_crit_value))
        has_vcov = self.vcov is not None
        return EventStudyResults(
            event_time=event_time,
            att=att,
            se=se,
            t_stat=t_stat,
            p_value=p_value,
            conf_int_lower=ci_lo,
            conf_int_upper=ci_hi,
            is_reference=is_reference,
            n=np.full(n_post + 1, float(self.n_units)),
            n_kind="units",
            time_scale="relative",
            event_time_convention="e0_first_treated",
            vcov=np.asarray(self.vcov, dtype=float) if has_vcov else None,
            vcov_index=np.arange(n_post) if has_vcov else None,
            cband_lower=(
                np.concatenate([[nan], np.asarray(self.cband_lower, dtype=float)])
                if has_band
                else None
            ),
            cband_upper=(
                np.concatenate([[nan], np.asarray(self.cband_upper, dtype=float)])
                if has_band
                else None
            ),
            cband_crit_value=float(self.cband_crit_value) if has_band else None,
            alpha=self.alpha,
            source="DurationDiDResults",
            df=None,
        )
