"""Owned result containers for Deaner--Ku Duration DiD."""

from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.aggregation import AggregationMixin, AggregationResult
from diff_diff.results_base import (
    BaseResults,
    Diagnostic,
    EventStudyResults,
    _coverage_pct,
    _json_safe_label,
    _require_fit_alpha,
)

_SUMMARY_ALPHA_MESSAGE = (
    "This result stores centered-bootstrap bands computed at alpha={fit_alpha}; "
    "summary() never recomputes or relabels stored inference (requested alpha={alpha}). "
    "Re-fit with the desired alpha to obtain the corresponding bootstrap critical values."
)

_FAILURE_COLUMNS = ["draw", "family", "period", "reason"]
_CONTRAST_COLUMNS = [
    "period",
    "elapsed_time",
    "contrast",
    "se",
    "cband_lower",
    "cband_upper",
    "status",
    "reason",
]
_EFFECT_COLUMNS = [
    "period",
    "event_time",
    "att",
    "se",
    "t_stat",
    "p_value",
    "conf_int_lower",
    "conf_int_upper",
    "cband_lower",
    "cband_upper",
    "pointwise_crit_value",
    "inference_status",
    "reason",
]
_SURVIVAL_COLUMNS = [
    "period",
    "elapsed_time",
    "treated_survival",
    "control_survival",
    "treated_survivors",
    "control_survivors",
    "raw_counterfactual_cumulative_hazard",
    "raw_counterfactual_survival",
    "raw_att",
    "counterfactual_status",
    "reason",
]


def _json_value(value: Any) -> Any:
    """Serialize nested statistical metadata without non-standard JSON numbers."""
    if isinstance(value, DurationDiDPretestResults):
        return value.to_dict()
    if isinstance(value, pd.DataFrame):
        return _json_value(value.to_dict(orient="records"))
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return _json_safe_label(value)


def _own_fields(result: Any) -> None:
    for item in fields(result):
        setattr(result, item.name, deepcopy(getattr(result, item.name)))


@dataclass
class DurationDiDPretestResults(Diagnostic):
    """Stored fixed-anchor hazard diagnostic (Deaner--Ku, Appendix B Algorithm 2).

    ``contrasts`` covers interior pre-treatment dates, excluding baseline and
    the last untreated anchor. Its bands are simultaneous at ``1-alpha``.
    ``reject=None`` denotes an unavailable test, never a non-rejection.
    ``bootstrap_contrasts`` retains all requested draw slots; failed slots are
    NaN. No test is recomputed by the accessor, summary, or serialization.
    """

    method: str
    alpha: float
    anchor_period: Any
    contrasts: pd.DataFrame
    statistic: float
    p_value: float
    critical_value: float
    reject: Optional[bool]
    status: str
    reasons: List[str]
    n_bootstrap: int
    n_bootstrap_attempted: int
    n_bootstrap_valid: int
    bootstrap_contrasts: np.ndarray
    bootstrap_failures: pd.DataFrame

    def __post_init__(self) -> None:
        _own_fields(self)

    def to_dict(self) -> Dict[str, Any]:
        """Return complete strict-JSON-safe statistics and availability metadata."""
        return {item.name: _json_value(getattr(self, item.name)) for item in fields(self)}

    def to_dataframe(self) -> pd.DataFrame:
        """Return an owned copy of the interior-date contrast table."""
        return self.contrasts.copy(deep=True)

    def summary(self, alpha: Optional[float] = None) -> str:
        """Describe the stored simultaneous hazard pretest at its fitted level."""
        _require_fit_alpha(alpha, self.alpha, message=_SUMMARY_ALPHA_MESSAGE)
        decision = (
            "unavailable"
            if self.reject is None
            else ("rejected" if self.reject else "not rejected")
        )
        return (
            f"DurationDiD hazard pretest ({self.method})\n"
            f"{_coverage_pct(self.alpha)}% simultaneous bands; {decision}\n"
            f"Statistic: {self.statistic:.6g}; p-value: {self.p_value:.6g}\n"
            f"Bootstrap draws: {self.n_bootstrap_valid}/{self.n_bootstrap_attempted} valid "
            f"({self.n_bootstrap} requested)"
            + "".join("\n" + reason for reason in self.reasons)
            + "\nNon-rejection does not establish identification or adequate power."
        )


@dataclass
class DurationDiDResults(BaseResults, AggregationMixin):
    """Duration DiD cumulative absorption effects and pooled-bootstrap inference.

    The headline is the uniform average of post-period absorption ATTs over
    the whole treated population, including people absorbed at baseline.
    Positive effects increase cumulative absorption. ``effects`` holds post
    estimates; ``survival_curve`` preserves raw extrapolations even when the
    counterfactual is invalid. ``coefficient`` is a normalized-interval hazard
    gap for common dynamics and a dimensionless ratio for proportional hazards.

    Inference is stored at ``alpha``; failed bootstrap families do not use a
    filtered distribution. ``pretrend_results`` is a separate hazard diagnostic,
    not a set of pre-treatment outcome effects. See the API page for all field
    and table schemas.
    """

    att: float
    se: float
    t_stat: float
    p_value: float
    conf_int: Tuple[float, float]
    method: str
    alpha: float
    seed: Optional[int]
    n_bootstrap: int
    n_obs: int
    n_units: int
    n_treated: int
    n_control: int
    periods: List[Any]
    pre_periods: List[Any]
    post_periods: List[Any]
    fit_periods: List[Any]
    requested_fit_periods: Optional[List[Any]]
    time_weights: Dict[Any, float]
    excluded_fit_periods: Dict[Any, str]
    time_origin: Any
    time_step: Any
    coefficient: float
    effects: pd.DataFrame
    survival_curve: pd.DataFrame
    bootstrap_effects: np.ndarray
    n_bootstrap_valid: int
    bootstrap_failures: pd.DataFrame
    vcov: Optional[np.ndarray]
    cband_crit_value: float
    estimation_status: str
    inference_status: Dict[str, str]
    inference_reasons: Dict[str, List[str]]
    support_warnings: List[str]
    pretrend_results: DurationDiDPretestResults

    _AGGREGATE_SUPPORTED: ClassVar[Tuple[str, ...]] = ("simple", "event_study")
    _AGGREGATE_BALANCE_E_TYPES: ClassVar[Tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        _own_fields(self)

    @property
    def inference_method(self) -> str:
        """The estimator's sole inference method."""
        return "bootstrap"

    @property
    def raw_att(self) -> float:
        """Uniform mean of raw post-period effects, including invalid extrapolations."""
        return float(
            np.mean(
                self.survival_curve.loc[
                    self.survival_curve["counterfactual_status"] != "not_estimated", "raw_att"
                ].to_numpy()
            )
        )

    def pretrend_test(self) -> DurationDiDPretestResults:
        """Return a defensive copy of the stored hazard Diagnostic; never refit."""
        return deepcopy(self.pretrend_results)

    def to_dict(self) -> Dict[str, Any]:
        """Return complete metadata, raw paths and inference as strict JSON values."""
        result = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name == "conf_int":
                result["conf_int_lower"] = _json_value(value[0])
                result["conf_int_upper"] = _json_value(value[1])
            elif item.name in {"time_weights", "excluded_fit_periods"}:
                result[item.name] = [
                    {"period": _json_value(k), "value": _json_value(v)} for k, v in value.items()
                ]
            else:
                result[item.name] = _json_value(value)
        result.update(inference_method=self.inference_method, raw_att=_json_value(self.raw_att))
        return result

    def to_dataframe(self, level: str = "event_study") -> pd.DataFrame:
        """Return event-study (default), simple, survival, or diagnostics tables."""
        if level in self._AGGREGATE_SUPPORTED:
            return self.aggregate(level).to_dataframe()
        if level == "survival":
            return self.survival_curve.copy(deep=True)
        if level == "diagnostics":
            return self.pretrend_results.to_dataframe()
        raise ValueError("level must be 'event_study', 'simple', 'survival', or 'diagnostics'")

    def summary(self, alpha: Optional[float] = None) -> str:
        """Summarize stored absorption ATT and availability, printing shared reasons once."""
        _require_fit_alpha(alpha, self.alpha, message=_SUMMARY_ALPHA_MESSAGE)
        units = (
            "hazard gap per normalized observation interval"
            if self.method == "common_dynamics"
            else "dimensionless hazard ratio"
        )
        reasons = dict.fromkeys(
            reason for values in self.inference_reasons.values() for reason in values
        )
        return (
            f"DurationDiD ({self.method})\n"
            f"Coefficient ({units}): {self.coefficient:.6g}\n"
            f"Mean cumulative absorption ATT: {self.att:.6g}; SE: {self.se:.6g}\n"
            f"{_coverage_pct(self.alpha)}% CI: {self.conf_int}; p-value: {self.p_value:.6g}\n"
            f"Estimation: {self.estimation_status}; inference: {self.inference_status}\n"
            f"Pooled individual bootstrap: {self.n_bootstrap_valid}/{self.n_bootstrap} valid effect draws"
            + "".join("\n" + reason for reason in reasons)
            + "".join("\n" + warning for warning in self.support_warnings)
            + "\n"
            + self.pretrend_results.summary()
        )

    def _aggregate_compute(
        self, level: str, *, weights: Optional[str], balance_e: Optional[int]
    ) -> Any:
        if level == "simple":
            return AggregationResult(
                level="simple",
                label=np.array(["overall"]),
                target=np.array(["att"]),
                att=np.array([self.att]),
                se=np.array([self.se]),
                t_stat=np.array([self.t_stat]),
                p_value=np.array([self.p_value]),
                conf_int_lower=np.array([self.conf_int[0]]),
                conf_int_upper=np.array([self.conf_int[1]]),
                n=np.array([self.n_treated]),
                df=np.array([np.nan]),
                alpha=self.alpha,
                n_kind="units",
                weight=np.array([1.0]),
                estimator="DurationDiD",
            )
        e = self.effects

        def column(name: str, reference: float = np.nan) -> np.ndarray:
            return np.r_[reference, e[name].to_numpy(dtype=float)]

        return EventStudyResults(
            event_time=np.arange(-1, len(e)),
            att=column("att", 0.0),
            se=column("se"),
            t_stat=column("t_stat"),
            p_value=column("p_value"),
            conf_int_lower=column("conf_int_lower"),
            conf_int_upper=column("conf_int_upper"),
            is_reference=np.r_[True, np.zeros(len(e), dtype=bool)],
            n=np.r_[np.nan, np.full(len(e), self.n_treated)],
            n_kind="units",
            reference_period=-1,
            time_scale="relative",
            event_time_convention="e0_first_treated",
            vcov=None if self.vcov is None else self.vcov.copy(),
            vcov_index=None if self.vcov is None else np.arange(len(e)),
            cband_lower=column("cband_lower"),
            cband_upper=column("cband_upper"),
            cband_crit_value=self.cband_crit_value,
            alpha=self.alpha,
            source="DurationDiDResults",
        )
