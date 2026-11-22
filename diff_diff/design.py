"""
Design diagnosis on the user's own data.

Before analysing a difference-in-differences study, :func:`diagnose_design`
takes an *untreated* panel, repeatedly assigns placebo treatments that follow a
declared design (how many units, which adoption periods), optionally adds a
known effect to the placebo-treated observations, fits each candidate
estimator, and reports per estimator how often it rejects when nothing
happened (size), how often it rejects an injected effect (power), and the
bias, RMSE and coverage of its estimates.

The run is outcome-blind: no real treatment effect is ever estimated. When a
real treatment column is supplied, every period from the earliest real
adoption onward is dropped before anything else happens.

References
----------
Bertrand, M., Duflo, E., & Mullainathan, S. (2004). "How Much Should We Trust
    Differences-in-Differences Estimates?" Quarterly Journal of Economics,
    119(1), 249-275.

Blair, G., Cooper, J., Coppock, A., & Humphreys, M. (2019). "Declaring and
    Diagnosing Research Designs." American Political Science Review, 113(3),
    838-859.

Coglianese, J., & Fang, J. A. (2026). "An Evaluation of Difference-in-Differences
    Methods Using Placebo Event Studies." Finance and Economics Discussion Series
    2026-045, Board of Governors of the Federal Reserve System.

Mizushima, Y., & Powell, D. (2025). "Inference with Modern
    Difference-in-Differences Methods." SSRN working paper 5221387.
"""

import dataclasses
import hashlib
import inspect
import json
import warnings
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from diff_diff.power import (
    SimulationMDEResults,
    SimulationPowerResults,
    simulate_mde,
    simulate_power,
)
from diff_diff.results_base import Diagnostic

__all__ = ["diagnose_design", "DesignDiagnosisResults"]

# Estimators that need one common post indicator (common timing only).
_COMMON_TIMING_ESTIMATORS = ("DifferenceInDifferences", "TwoWayFixedEffects")
# Estimators driven by a first-treatment-period column (common or staggered).
_STAGGERED_ESTIMATORS = (
    "CallawaySantAnna",
    "SunAbraham",
    "ImputationDiD",
    "TwoStageDiD",
    "StackedDiD",
    "EfficientDiD",
)
_SUPPORTED_ESTIMATORS = _COMMON_TIMING_ESTIMATORS + _STAGGERED_ESTIMATORS

_FIRST_TREAT_COLUMN = "_placebo_first_treat"
_TREATED_COLUMN = "_placebo_treated"
_POST_COLUMN = "_placebo_post"
_PLACEBO_COLUMNS = (_FIRST_TREAT_COLUMN, _TREATED_COLUMN, _POST_COLUMN)

# fit() keywords a caller may not pass: each would bypass or contradict the
# placebo columns this module supplies.
_REFUSED_FIT_KEYS = (
    "data",
    "outcome",
    "unit",
    "time",
    "first_treat",
    "treatment",
    "post",
    "formula",
    "post_periods",
    "survey_design",
)

# The exception classes simulate_power treats as data-dependent replicate
# failures (diff_diff/power.py). Anything else is a programming error and
# propagates.
_DATA_DEPENDENT_ERRORS = (
    ValueError,
    np.linalg.LinAlgError,
    KeyError,
    RuntimeError,
    ZeroDivisionError,
)

_PREFLIGHT_ATTEMPTS = 5
_MAX_LOST_SHARE = 0.10
_MIN_PRE_PERIODS = 2
_MAX_ERROR_MESSAGES = 5


# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------


@dataclass
class DesignDiagnosisResults(Diagnostic):
    """
    Results of :func:`diagnose_design`.

    Attributes
    ----------
    results : dict
        Label -> list of :class:`~diff_diff.SimulationPowerResults`, one per
        entry of ``effect_sizes``. The first entry is the no-effect run, so
        its ``rejection_rate`` is the estimator's size on this design.
    mde_results : dict
        Label -> :class:`~diff_diff.SimulationMDEResults` for every estimator
        whose minimum-detectable-effect search ran.
    estimators : dict
        Label -> an unfitted copy of the estimator as configured.
    size_controlled : dict
        Label -> whether the estimator passed the size gate at effect 0.
    n_nan_inference : dict
        Label -> list aligned with ``effect_sizes``: replicates excluded
        because the fit returned non-finite inference.
    mde : dict
        Label -> minimum detectable effect, NaN when not reported.
    effect_sizes : list of float
        The effects actually run: 0 first, then the requested non-zero values.
    alpha : float
        Significance level of the tests.
    target_power : float or None
        Target power of the minimum-detectable-effect search.
    outcome, unit, time : str
        Column names used.
    first_treat : str or None
        Real first-treatment column used to truncate the panel. ``None`` means
        the caller declared the panel untreated; that was not verified.
    treatment_fraction : float or None
        Declared share of treated units, when given.
    n_treated : int
        Number of placebo-treated units per replicate.
    cohort_periods : list
        Declared adoption periods (values of the ``time`` column).
    resample_units : bool
        Whether units were resampled with replacement in each replicate.
    estimator_kwargs : dict
        Extra ``fit()`` keywords applied to every estimator.
    seed : int
        Seed of the placebo draws (resolved to a concrete value).
    n_units, n_periods, n_obs : int
        Shape of the panel actually used.
    n_dropped : int
        Rows removed by truncating at the earliest real adoption.
    data_sha256 : str
        Fingerprint of the panel actually used.
    data_columns : list of str
        Columns the fingerprint covers.
    """

    results: Dict[str, List[SimulationPowerResults]]
    mde_results: Dict[str, SimulationMDEResults]
    estimators: Dict[str, Any]
    size_controlled: Dict[str, bool]
    n_nan_inference: Dict[str, List[int]]
    mde: Dict[str, float]
    effect_sizes: List[float]
    alpha: float
    target_power: Optional[float]
    outcome: str
    unit: str
    time: str
    first_treat: Optional[str]
    treatment_fraction: Optional[float]
    n_treated: int
    cohort_periods: List[Any]
    resample_units: bool
    estimator_kwargs: Dict[str, Any]
    seed: int
    n_units: int
    n_periods: int
    n_obs: int
    n_dropped: int
    data_sha256: str
    data_columns: List[str]

    def __repr__(self) -> str:
        """Concise string representation."""
        n_ok = sum(1 for ok in self.size_controlled.values() if ok)
        return (
            f"DesignDiagnosisResults(estimators={len(self.estimators)}, "
            f"size_controlled={n_ok}, effect_sizes={self.effect_sizes})"
        )

    def to_dataframe(self) -> pd.DataFrame:
        """
        One row per estimator and effect size, in input order.

        Returns
        -------
        pd.DataFrame
            Column ``label`` first, then the columns of
            ``SimulationPowerResults.to_dict()``, then ``size_controlled``,
            ``n_nan_inference`` and ``mde``.
        """
        rows: List[Dict[str, Any]] = []
        for label, cells in self.results.items():
            for i, cell in enumerate(cells):
                row: Dict[str, Any] = {"label": label}
                row.update(cell.to_dict())
                row["size_controlled"] = self.size_controlled[label]
                row["n_nan_inference"] = self.n_nan_inference[label][i]
                row["mde"] = self.mde[label]
                rows.append(row)
        return pd.DataFrame(rows)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert results to a dictionary.

        Returns
        -------
        Dict[str, Any]
            The declared design, the estimators (class name and parameters),
            and the per-estimator, per-effect rows.
        """
        return {
            "estimators": {
                label: {
                    "estimator_name": type(est).__name__,
                    "params": dict(est.get_params()),
                }
                for label, est in self.estimators.items()
            },
            "effect_sizes": list(self.effect_sizes),
            "alpha": self.alpha,
            "target_power": self.target_power,
            "outcome": self.outcome,
            "unit": self.unit,
            "time": self.time,
            "first_treat": self.first_treat,
            "treatment_fraction": self.treatment_fraction,
            "n_treated": self.n_treated,
            "cohort_periods": list(self.cohort_periods),
            "resample_units": self.resample_units,
            "estimator_kwargs": dict(self.estimator_kwargs),
            "seed": self.seed,
            "n_units": self.n_units,
            "n_periods": self.n_periods,
            "n_obs": self.n_obs,
            "n_dropped": self.n_dropped,
            "data_sha256": self.data_sha256,
            "data_columns": list(self.data_columns),
            "size_controlled": dict(self.size_controlled),
            "mde": dict(self.mde),
            "results": self.to_dataframe().to_dict("records"),
        }

    def summary(self) -> str:
        """
        Generate a formatted summary of the design diagnosis.

        Returns
        -------
        str
            Formatted summary. Estimators appear in input order; nothing is
            ranked.
        """
        width = 78
        timing = "common" if len(self.cohort_periods) == 1 else "staggered"
        untreated = (
            f"truncated at earliest real adoption ({self.n_dropped} rows dropped)"
            if self.first_treat is not None
            else "declared by caller (no first_treat column given; not verified)"
        )
        lines = [
            "=" * width,
            "Design Diagnosis (placebo assignments on the supplied panel)".center(width),
            "=" * width,
            "",
            f"{'Units x periods:':<32} {self.n_units} x {self.n_periods}",
            f"{'Untreated status:':<32} {untreated}",
            f"{'Placebo-treated units:':<32} {self.n_treated}",
            f"{'Adoption periods (' + timing + '):':<32} {list(self.cohort_periods)}",
            f"{'Units resampled:':<32} {self.resample_units}",
            f"{'Significance level (alpha):':<32} {self.alpha:.3f}",
            f"{'Seed:':<32} {self.seed}",
            "",
            "-" * width,
            "No-effect runs (size)".center(width),
            "-" * width,
            f"{'Estimator':<26}{'Size':>8}{'Scored':>8}{'Lost':>7}{'Coverage':>10}"
            f"{'Size OK':>9}{'MDE':>10}",
        ]
        for label, cells in self.results.items():
            null = cells[0]
            mde = self.mde[label]
            mde_text = f"{mde:.4g}" if np.isfinite(mde) else "-"
            lines.append(
                f"{label[:25]:<26}{_fmt_rate(null.rejection_rate):>8}"
                f"{null.n_simulations:>8}{null.n_simulation_failures:>7}"
                f"{_fmt_rate(null.coverage):>10}"
                f"{('yes' if self.size_controlled[label] else 'NO'):>9}{mde_text:>10}"
            )
        if len(self.effect_sizes) > 1:
            lines.extend(["", "-" * width, "Injected effects".center(width), "-" * width])
            lines.append(
                f"{'Estimator':<26}{'Effect':>10}{'Power':>8}{'Bias':>11}{'RMSE':>11}"
                f"{'Coverage':>10}"
            )
            for label, cells in self.results.items():
                for cell in cells[1:]:
                    lines.append(
                        f"{label[:25]:<26}{cell.true_effect:>10.4g}"
                        f"{_fmt_rate(cell.power):>8}{_fmt_num(cell.bias):>11}"
                        f"{_fmt_num(cell.rmse):>11}{_fmt_rate(cell.coverage):>10}"
                    )
        lines.extend(
            [
                "",
                "Estimators are listed in input order and are not ranked. 'Size OK' is",
                "a check on the no-effect runs only; power for an estimator that fails",
                "it is not meaningful.",
                "=" * width,
            ]
        )
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print the summary to stdout."""
        print(self.summary())


def _fmt_rate(value: float) -> str:
    return f"{value:.1%}" if np.isfinite(value) else "-"


def _fmt_num(value: float) -> str:
    return f"{value:.4g}" if np.isfinite(value) else "-"


# ---------------------------------------------------------------------------
# Warning capture
# ---------------------------------------------------------------------------


class _WarningCapture:
    """Record every warning raised in a block, regardless of caller filters."""

    def __init__(self) -> None:
        self.records: List[warnings.WarningMessage] = []
        self._ctx: Optional[warnings.catch_warnings] = None

    def __enter__(self) -> "_WarningCapture":
        self._ctx = warnings.catch_warnings(record=True)
        captured = self._ctx.__enter__()
        warnings.simplefilter("always")
        assert captured is not None
        self.records = captured
        return self

    def __exit__(self, *exc_info: Any) -> None:
        assert self._ctx is not None
        self._ctx.__exit__(*exc_info)


def _reemit(label: str, records: Sequence[warnings.WarningMessage]) -> None:
    """Re-emit captured warnings once per (category, source), labelled.

    Estimator warnings embed per-replicate values, so they are aggregated by
    where they were raised, not by message text. Re-emission goes through the
    caller's own warning filters.
    """
    grouped: Dict[Tuple[Any, str, int], Tuple[str, int]] = {}
    for record in records:
        key = (record.category, record.filename, record.lineno)
        first, count = grouped.get(key, (str(record.message), 0))
        grouped[key] = (first, count + 1)
    for (category, _filename, _lineno), (message, count) in grouped.items():
        suffix = f" (raised {count} times; first message shown)" if count > 1 else ""
        warnings.warn(f"[{label}] {message}{suffix}", category, stacklevel=3)


# ---------------------------------------------------------------------------
# Fit adapter
# ---------------------------------------------------------------------------


def _has_scalar_inference(result: Any) -> bool:
    """Whether a fit result exposes scalar att / se / p_value and a 2-interval."""
    for name in ("att", "se", "p_value", "conf_int"):
        if not hasattr(result, name):
            return False
    for name in ("att", "se", "p_value"):
        if np.ndim(getattr(result, name)) != 0:
            return False
    conf_int = result.conf_int
    try:
        return len(conf_int) == 2 and all(np.ndim(bound) == 0 for bound in conf_int)
    except TypeError:
        return False


def _has_finite_inference(result: Any) -> bool:
    """Whether att, se, p_value and both interval bounds are finite."""
    values = [result.att, result.se, result.p_value, result.conf_int[0], result.conf_int[1]]
    try:
        return bool(np.all(np.isfinite(np.asarray(values, dtype=float))))
    except (TypeError, ValueError):
        return False


class _PlaceboFit:
    """Wrap one estimator copy for one cell and count the replicates it loses.

    ``simulate_power`` counts a NaN p-value as a non-rejection; raising here
    instead excludes such a replicate from every rate. Only the exception
    classes ``simulate_power`` already treats as data-dependent are counted;
    anything else propagates.
    """

    def __init__(self, estimator: Any) -> None:
        self._estimator = estimator
        self.n_fits = 0
        self.n_nan = 0
        self.n_raised = 0
        self.errors: Dict[str, int] = {}

    @property
    def n_lost(self) -> int:
        return self.n_nan + self.n_raised

    def fit(self, data: pd.DataFrame, **kwargs: Any) -> Any:
        self.n_fits += 1
        try:
            result = self._estimator.fit(data, **kwargs)
        except _DATA_DEPENDENT_ERRORS as exc:
            self.n_raised += 1
            message = f"{type(exc).__name__}: {exc}"
            if message in self.errors or len(self.errors) < _MAX_ERROR_MESSAGES:
                self.errors[message] = self.errors.get(message, 0) + 1
            raise
        if not _has_scalar_inference(result) or not _has_finite_inference(result):
            self.n_nan += 1
            raise ValueError("non-finite inference")
        return result


def _clone(estimator: Any) -> Any:
    """An unfitted copy with the same constructor parameters."""
    return type(estimator)(**estimator.get_params())


# ---------------------------------------------------------------------------
# Size gate
# ---------------------------------------------------------------------------


def _size_controlled(rejection_rate: float, n_scored: int, n_lost: int, alpha: float) -> bool:
    """Whether a no-effect run is consistent with the nominal level.

    False when nothing was scored, or when more than 10% of the replicates
    were lost (an estimator that fails on hard draws would otherwise look
    better than it is). Otherwise true exactly when the rejection rate is at
    most ``alpha`` plus 1.96 Monte Carlo standard errors at ``alpha``.
    """
    if n_scored <= 0:
        return False
    if n_lost / (n_scored + n_lost) > _MAX_LOST_SHARE:
        return False
    bound = alpha + 1.96 * np.sqrt(alpha * (1.0 - alpha) / n_scored)
    return bool(rejection_rate <= bound)


# ---------------------------------------------------------------------------
# Data fingerprint
# ---------------------------------------------------------------------------


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NaT:
        return True
    if value is getattr(pd, "NA", None):
        return True
    return isinstance(value, (float, np.floating)) and bool(np.isnan(value))


def _encode_cell(value: Any) -> List[str]:
    """Type-tagged text for one cell, so distinct values never encode alike."""
    if _is_missing(value):
        return ["n", ""]
    if isinstance(value, (bool, np.bool_)):
        return ["b", "True" if bool(value) else "False"]
    if isinstance(value, (int, np.integer)):
        return ["i", str(int(value))]
    if isinstance(value, (float, np.floating)):
        return ["f", format(float(value), ".17g")]
    if isinstance(value, np.datetime64):
        return ["t", pd.Timestamp(value).isoformat()]
    if isinstance(value, (datetime, date)):
        return ["t", value.isoformat()]
    if isinstance(value, str):
        return ["s", value]
    return ["o:" + type(value).__name__, repr(value)]


def _data_fingerprint(panel: pd.DataFrame, unit: str, time: str) -> Tuple[str, List[str]]:
    """SHA-256 of a canonical text of the panel, and the columns it covers.

    Rows are sorted by (unit, time) and columns by name, so the digest does
    not depend on row or column order. The text is built by hand (one JSON
    line of type-tagged cells per row) so that it does not depend on the
    pandas version.
    """
    by_name = {str(c): c for c in panel.columns}
    if len(by_name) != len(panel.columns):
        raise ValueError(
            "data has column labels that are equal as strings (for example 1 and "
            "'1'); the data fingerprint needs distinct column names. Rename them."
        )
    columns = sorted(by_name)
    ordered = panel.sort_values([unit, time], kind="mergesort")
    digest = hashlib.sha256()
    digest.update(json.dumps(columns, ensure_ascii=True, separators=(",", ":")).encode("utf-8"))
    column_values = [ordered[by_name[name]].tolist() for name in columns]
    for row in zip(*column_values):
        line = json.dumps(
            [_encode_cell(value) for value in row], ensure_ascii=True, separators=(",", ":")
        )
        digest.update(b"\n")
        digest.update(line.encode("utf-8"))
    return digest.hexdigest(), columns


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _resolve_estimators(
    estimators: Union[Sequence[Any], Mapping[str, Any]],
) -> Dict[str, Any]:
    """Label -> estimator, in input order."""
    if isinstance(estimators, Mapping):
        resolved = {str(label): est for label, est in estimators.items()}
        if len(resolved) != len(estimators):
            raise ValueError("estimators has labels that are equal as strings.")
    else:
        if isinstance(estimators, (str, bytes)) or not isinstance(estimators, Sequence):
            raise TypeError(
                "estimators must be a list of estimator instances or a mapping "
                "from label to estimator instance."
            )
        resolved = {}
        for est in estimators:
            label = type(est).__name__
            if label in resolved:
                raise ValueError(
                    f"estimators contains more than one {label}. Pass a mapping "
                    f"from label to estimator to compare variants of one class, "
                    f"e.g. {{'did': {label}(), 'did_clustered': {label}(cluster=...)}}."
                )
            resolved[label] = est
    if not resolved:
        raise ValueError("estimators is empty.")
    return resolved


def _validate_estimators(
    resolved: Dict[str, Any],
    alpha: float,
    staggered: bool,
    unit: str,
    resample_units: bool,
) -> None:
    for label, est in resolved.items():
        name = type(est).__name__
        if name not in _SUPPORTED_ESTIMATORS:
            raise ValueError(
                f"Estimator '{label}' is a {name}, which diagnose_design does not "
                f"support. Supported: {', '.join(_SUPPORTED_ESTIMATORS)}."
            )
        params = est.get_params()
        est_alpha = params.get("alpha", alpha)
        if not np.isclose(est_alpha, alpha):
            raise ValueError(
                f"Estimator '{label}' has alpha={est_alpha} but diagnose_design was "
                f"called with alpha={alpha}. Coverage is computed from each fit's "
                f"own confidence interval, so the two must agree."
            )
        anticipation = params.get("anticipation", 0)
        if anticipation != 0:
            raise ValueError(
                f"Estimator '{label}' has anticipation={anticipation}. Its aggregate "
                f"then averages periods before adoption, where no effect is "
                f"injected, so bias and coverage would be scored against the wrong "
                f"target. diagnose_design supports anticipation=0 only."
            )
        cluster = params.get("cluster")
        # StackedDiD's ``cluster`` is a mode ("unit" / "unit_subexp"), not a column.
        if resample_units and name != "StackedDiD" and isinstance(cluster, str) and cluster != unit:
            raise ValueError(
                f"Estimator '{label}' clusters on column '{cluster}', but with "
                f"resample_units=True a unit drawn twice gets a fresh id in "
                f"'{unit}' only, so its copies would share one '{cluster}' cluster. "
                f"Cluster on '{unit}', or pass resample_units=False to keep the "
                f"panel, and its clusters, as observed."
            )
        if staggered and name in _COMMON_TIMING_ESTIMATORS:
            raise ValueError(
                f"Estimator '{label}' ({name}) needs one common adoption period, but "
                f"cohort_periods declares a staggered design. Pass a single adoption "
                f"period, or drop this estimator."
            )


def _validate_estimator_kwargs(resolved: Dict[str, Any], estimator_kwargs: Dict[str, Any]) -> None:
    refused = [key for key in estimator_kwargs if key in _REFUSED_FIT_KEYS]
    if refused:
        raise ValueError(
            f"estimator_kwargs may not contain {sorted(refused)}: diagnose_design "
            f"supplies the outcome and the placebo treatment columns itself."
        )
    for label, est in resolved.items():
        parameters = inspect.signature(est.fit).parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            continue
        unknown = [key for key in estimator_kwargs if key not in parameters]
        if unknown:
            raise ValueError(
                f"estimator_kwargs key(s) {sorted(unknown)} are not accepted by "
                f"{type(est).__name__}.fit (estimator '{label}'). estimator_kwargs "
                f"is applied to every estimator."
            )


def _prepare_panel(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    first_treat: Optional[str],
) -> Tuple[pd.DataFrame, int]:
    """Validate the panel and drop every row from the earliest real adoption on."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    needed = [outcome, unit, time] + ([first_treat] if first_treat is not None else [])
    missing = [col for col in needed if col not in data.columns]
    if missing:
        raise ValueError(f"Column(s) {missing} not found in data.")
    present = [col for col in _PLACEBO_COLUMNS if col in data.columns]
    if present:
        raise ValueError(
            f"data already has column(s) {present}; diagnose_design reserves the "
            f"'_placebo_' prefix. Rename them."
        )
    if not pd.api.types.is_numeric_dtype(data[time]) or pd.api.types.is_bool_dtype(data[time]):
        raise ValueError(
            f"'{time}' must be numeric: the staggered estimators need numeric "
            f"periods. Convert dates to an integer period index first."
        )
    if data[time].isna().any() or data[unit].isna().any():
        raise ValueError(f"'{unit}' and '{time}' must not contain missing values.")

    n_dropped = 0
    panel = data
    if first_treat is not None:
        if not pd.api.types.is_numeric_dtype(data[first_treat]):
            raise ValueError(f"'{first_treat}' must be numeric (0, inf or NaN = never treated).")
        if (data.groupby(unit)[first_treat].nunique(dropna=False) > 1).any():
            raise ValueError(f"'{first_treat}' must be constant within each unit.")
        values = data[first_treat].to_numpy(dtype=float)
        if np.any(values[np.isfinite(values)] < 0):
            raise ValueError(f"'{first_treat}' has negative values.")
        adopted = np.isfinite(values) & (values > 0)
        if adopted.any():
            first_period = data[time].min()
            always = sorted(data.loc[adopted & (values <= first_period), unit].unique().tolist())
            if always:
                shown = ", ".join(repr(u) for u in always[:5])
                more = f" and {len(always) - 5} more" if len(always) > 5 else ""
                raise ValueError(
                    f"{len(always)} unit(s) are treated from the first period on "
                    f"({shown}{more}), so no untreated period exists for them. "
                    f"Drop always-treated units before calling diagnose_design."
                )
            earliest = float(values[adopted].min())
            keep = data[time].to_numpy(dtype=float) < earliest
            n_dropped = int((~keep).sum())
            panel = data.loc[keep]

    if len(panel) == 0:
        raise ValueError("No untreated rows remain after truncation.")
    if not pd.api.types.is_numeric_dtype(panel[outcome]) or pd.api.types.is_bool_dtype(
        panel[outcome]
    ):
        raise ValueError(f"'{outcome}' must be numeric.")
    if panel[outcome].isna().any():
        raise ValueError(f"'{outcome}' has missing values; diagnose_design needs a complete panel.")
    if panel.duplicated([unit, time]).any():
        raise ValueError(f"data has more than one row per ({unit}, {time}).")
    n_units = panel[unit].nunique()
    n_periods = panel[time].nunique()
    if len(panel) != n_units * n_periods:
        raise ValueError(
            "diagnose_design needs a balanced panel (every unit observed in "
            f"every period); found {len(panel)} rows for {n_units} units x "
            f"{n_periods} periods."
        )
    panel = panel.sort_values([unit, time], kind="mergesort").reset_index(drop=True)
    return panel, n_dropped


def _validate_cohort_periods(cohort_periods: Any, periods: np.ndarray) -> List[Any]:
    if isinstance(cohort_periods, (str, bytes)) or not isinstance(
        cohort_periods, (list, tuple, np.ndarray)
    ):
        raise TypeError("cohort_periods must be a list of adoption periods (values of time).")
    declared = list(cohort_periods)
    if not declared:
        raise ValueError("cohort_periods is empty; declare at least one adoption period.")
    if len(set(declared)) != len(declared):
        raise ValueError("cohort_periods contains duplicates.")
    for period in declared:
        if period not in periods:
            raise ValueError(
                f"Adoption period {period!r} is not a period of the untreated panel "
                f"(periods run from {periods[0]!r} to {periods[-1]!r})."
            )
        if not period > 0:
            raise ValueError(
                f"Adoption period {period!r} is not positive; 0 marks never-treated "
                f"units. Shift the period values."
            )
        n_before = int(np.sum(periods < period))
        if n_before < _MIN_PRE_PERIODS:
            raise ValueError(
                f"Too few periods: adoption period {period!r} has {n_before} earlier "
                f"period(s) in the untreated panel; at least {_MIN_PRE_PERIODS} are "
                f"needed."
            )
    return declared


def _resolve_n_treated(
    treatment_fraction: Optional[float], n_treated: Optional[int], n_units: int
) -> int:
    if (treatment_fraction is None) == (n_treated is None):
        raise ValueError("Pass exactly one of treatment_fraction and n_treated.")
    if treatment_fraction is not None:
        if not 0.0 < treatment_fraction < 1.0:
            raise ValueError(f"treatment_fraction must be in (0, 1), got {treatment_fraction}.")
        count = int(n_units * treatment_fraction)
    else:
        if isinstance(n_treated, bool) or not isinstance(n_treated, (int, np.integer)):
            raise ValueError(f"n_treated must be an integer, got {n_treated!r}.")
        count = int(n_treated)
    if not 1 <= count <= n_units - 1:
        raise ValueError(
            f"The design gives {count} treated unit(s) out of {n_units}; it must "
            f"leave at least one treated and one untreated unit."
        )
    return count


def _resolve_effect_sizes(effect_sizes: Optional[List[float]]) -> List[float]:
    """The effects actually run: 0 first, then the non-zero values, deduplicated."""
    resolved: List[float] = [0.0]
    for effect in effect_sizes or []:
        value = float(effect)
        if not np.isfinite(value):
            raise ValueError(f"effect_sizes must be finite, got {effect!r}.")
        if value not in resolved:
            resolved.append(value)
    return resolved


def _uses_unseeded_bootstrap(estimator: Any) -> bool:
    params = estimator.get_params()
    if "seed" not in params or params["seed"] is not None:
        return False
    if type(estimator).__name__ in _COMMON_TIMING_ESTIMATORS:
        return params.get("inference") == "wild_bootstrap"
    return bool(params.get("n_bootstrap", 0))


# ---------------------------------------------------------------------------
# Placebo generator
# ---------------------------------------------------------------------------


def _make_placebo_generator(
    panel: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    n_treated: int,
    cohort_periods: List[Any],
    resample_units: bool,
) -> Callable[..., pd.DataFrame]:
    """A ``data_generator`` for ``simulate_power`` that draws placebo designs.

    ``panel`` is balanced and sorted by (unit, time). The returned callable
    uses only ``treatment_effect`` and ``seed``; the remaining keywords
    ``simulate_power`` passes describe a synthetic design and are ignored.
    """
    n_periods = panel[time].nunique()
    n_units = len(panel) // n_periods
    period_values = panel[time].to_numpy()[:n_periods]
    time_tiled = np.tile(period_values, n_units)
    offsets = np.arange(n_periods)
    outcome_position = panel.columns.get_loc(outcome)
    cohorts = np.asarray(cohort_periods)
    common = len(cohort_periods) == 1
    first_treat_dtype = np.result_type(cohorts.dtype, np.int64)

    def data_generator(
        treatment_effect: float = 0.0, seed: Optional[int] = None, **ignored: Any
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        if resample_units:
            drawn = rng.integers(0, n_units, size=n_units)
            rows = (drawn[:, None] * n_periods + offsets[None, :]).ravel()
            frame = panel.iloc[rows].reset_index(drop=True)
            # Fresh ids: a unit drawn twice becomes two distinct units.
            frame[unit] = np.repeat(np.arange(n_units), n_periods)
        else:
            frame = panel.copy()
        treated_units = rng.choice(n_units, size=n_treated, replace=False)
        first_treat_by_unit = np.zeros(n_units, dtype=first_treat_dtype)
        if common:
            first_treat_by_unit[treated_units] = cohorts[0]
        else:
            first_treat_by_unit[treated_units] = rng.choice(cohorts, size=n_treated)
        first_treat_rows = np.repeat(first_treat_by_unit, n_periods)
        ever_treated = first_treat_rows > 0
        active = ever_treated & (time_tiled >= first_treat_rows)
        if treatment_effect != 0.0:
            values = frame.iloc[:, outcome_position].to_numpy(dtype=float, copy=True)
            values[active] += treatment_effect
            frame[outcome] = values
        frame[_FIRST_TREAT_COLUMN] = first_treat_rows
        frame[_TREATED_COLUMN] = ever_treated.astype(int)
        if common:
            frame[_POST_COLUMN] = (time_tiled >= cohorts[0]).astype(int)
        return frame

    return data_generator


def _fit_kwargs(
    estimator: Any, outcome: str, unit: str, time: str, estimator_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """The complete ``fit()`` keywords for one estimator on a placebo frame."""
    if type(estimator).__name__ in _COMMON_TIMING_ESTIMATORS:
        columns: Dict[str, Any] = {
            "outcome": outcome,
            "treatment": _TREATED_COLUMN,
            "post": _POST_COLUMN,
            "unit": unit,
        }
    else:
        columns = {
            "outcome": outcome,
            "unit": unit,
            "time": time,
            "first_treat": _FIRST_TREAT_COLUMN,
        }
    columns.update(estimator_kwargs)
    return columns


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def _replicate_seeds(seed: int, count: int) -> List[int]:
    """The first ``count`` per-replicate seeds ``simulate_power`` will draw."""
    rng = np.random.default_rng(seed)
    return [int(rng.integers(0, 2**31)) for _ in range(count)]


def _preflight(
    label: str,
    estimator: Any,
    data_generator: Callable[..., pd.DataFrame],
    fit_kwargs: Dict[str, Any],
    seed: int,
) -> None:
    """Refuse an estimator that cannot fit any of the first placebo replicates."""
    first_error: Optional[str] = None
    name = type(estimator).__name__
    for replicate_seed in _replicate_seeds(seed, _PREFLIGHT_ATTEMPTS):
        frame = data_generator(treatment_effect=0.0, seed=replicate_seed)
        try:
            result = _clone(estimator).fit(frame, **fit_kwargs)
        except _DATA_DEPENDENT_ERRORS as exc:
            if first_error is None:
                first_error = f"{type(exc).__name__}: {exc}"
            continue
        if not _has_scalar_inference(result):
            raise ValueError(
                f"Estimator '{label}' ({name}) is misconfigured for diagnose_design: "
                f"its fit returned a {type(result).__name__} without scalar att, se, "
                f"p_value and a two-element conf_int."
            )
        if _has_finite_inference(result):
            return
        if first_error is None:
            first_error = "non-finite inference"
    raise ValueError(
        f"Estimator '{label}' ({name}) could not be fitted on any of the first "
        f"{_PREFLIGHT_ATTEMPTS} placebo replicates of this design. First problem: "
        f"{first_error}. Fix its configuration or drop it from estimators."
    )


def _failed_cell(effect: float, alpha: float, name: str, n_simulations: int) -> Any:
    nan = float("nan")
    return SimulationPowerResults(
        power=nan,
        power_se=nan,
        power_ci=(nan, nan),
        rejection_rate=nan,
        mean_estimate=nan,
        std_estimate=nan,
        mean_se=nan,
        coverage=nan,
        n_simulations=0,
        effect_sizes=[effect],
        powers=[nan],
        true_effect=effect,
        alpha=alpha,
        estimator_name=name,
        n_simulation_failures=n_simulations,
    )


def _run_cell(
    estimator: Any,
    effect: float,
    data_generator: Callable[..., pd.DataFrame],
    fit_kwargs: Dict[str, Any],
    n_units: int,
    n_periods: int,
    n_simulations: int,
    alpha: float,
    seed: int,
) -> Tuple[Any, _PlaceboFit]:
    """One estimator at one effect size, through ``simulate_power``."""
    name = type(estimator).__name__
    adapter = _PlaceboFit(_clone(estimator))
    try:
        cell = simulate_power(
            adapter,
            n_units=n_units,
            n_periods=n_periods,
            treatment_effect=effect,
            n_simulations=n_simulations,
            alpha=alpha,
            seed=seed,
            data_generator=data_generator,
            estimator_kwargs=fit_kwargs,
            progress=False,
        )
    except RuntimeError as exc:
        if "All simulations failed" not in str(exc):
            raise
        return _failed_cell(effect, alpha, name, n_simulations), adapter
    return dataclasses.replace(cell, estimator_name=name), adapter


def _lost_message(adapter: _PlaceboFit) -> str:
    parts = [f"{adapter.n_lost} of {adapter.n_fits} fits lost"]
    if adapter.n_nan:
        parts.append(f"{adapter.n_nan} with non-finite inference")
    for message, count in adapter.errors.items():
        parts.append(f"{count} x {message}")
    return "; ".join(parts)


def _run_mde(
    estimator: Any,
    null_cell: Any,
    data_generator: Callable[..., pd.DataFrame],
    fit_kwargs: Dict[str, Any],
    n_units: int,
    n_periods: int,
    n_simulations: int,
    alpha: float,
    power: float,
    seed: int,
) -> Tuple[Optional[Any], float]:
    """Minimum detectable effect by ``simulate_mde``; NaN when not reportable.

    Runs inside the caller's warning capture, which adds the estimator label
    on re-emission, so the warnings raised here carry none.
    """
    nan = float("nan")
    scale = float(null_cell.std_estimate)
    if not np.isfinite(scale) or scale <= 0.0:
        warnings.warn(
            f"MDE not reported: the spread of the no-effect estimates is "
            f"{scale}, so the search cannot be scaled.",
            UserWarning,
            stacklevel=3,
        )
        return None, nan
    name = type(estimator).__name__
    adapter = _PlaceboFit(_clone(estimator))
    try:
        found = simulate_mde(
            adapter,
            n_units=n_units,
            n_periods=n_periods,
            sigma=scale,
            n_simulations=n_simulations,
            power=power,
            alpha=alpha,
            seed=seed,
            data_generator=data_generator,
            estimator_kwargs=fit_kwargs,
            progress=False,
        )
    except RuntimeError as exc:
        if "All simulations failed" not in str(exc):
            raise
        warnings.warn(
            f"MDE not reported: a search step lost every replicate " f"({_lost_message(adapter)}).",
            UserWarning,
            stacklevel=3,
        )
        return None, nan
    found = dataclasses.replace(found, estimator_name=name)
    if adapter.n_fits and adapter.n_lost / adapter.n_fits > _MAX_LOST_SHARE:
        warnings.warn(
            f"MDE not reported: more than {_MAX_LOST_SHARE:.0%} of the "
            f"fits in the search were lost ({_lost_message(adapter)}).",
            UserWarning,
            stacklevel=3,
        )
        return found, nan
    if not found.power_at_mde >= power or not found.mde > 0.0:
        # simulate_mde has already warned (not bracketed, or power at effect 0
        # already at the target); the caller re-emits that warning labelled.
        return found, nan
    return found, float(found.mde)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def diagnose_design(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    estimators: Union[Sequence[Any], Mapping[str, Any]],
    cohort_periods: List[Any],
    treatment_fraction: Optional[float] = None,
    n_treated: Optional[int] = None,
    effect_sizes: Optional[List[float]] = None,
    n_simulations: int = 500,
    alpha: float = 0.05,
    power: Optional[float] = None,
    first_treat: Optional[str] = None,
    resample_units: bool = True,
    estimator_kwargs: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    progress: bool = True,
) -> DesignDiagnosisResults:
    """
    Diagnose a planned DiD design on your own untreated panel.

    Repeatedly assigns placebo treatments that follow the declared design,
    optionally adds a known effect to the placebo-treated observations, fits
    every estimator on the same placebo draws, and reports each estimator's
    rejection rate under no effect (size), its power, bias, RMSE and coverage
    at each injected effect, and optionally a minimum detectable effect.

    No real treatment effect is estimated at any point.

    Parameters
    ----------
    data : pd.DataFrame
        A balanced panel with no missing outcomes. It must be untreated: pass
        only pre-treatment data, or pass ``first_treat`` so that every period
        from the earliest real adoption onward is dropped.
    outcome : str
        Name of the outcome column.
    unit : str
        Name of the unit identifier column.
    time : str
        Name of the calendar period column (numeric).
    estimators : list or dict
        Estimator instances to compare. A list is labelled by class name; pass
        a dict ``{label: estimator}`` to compare variants of one class.
        Supported: DifferenceInDifferences, TwoWayFixedEffects,
        CallawaySantAnna, SunAbraham, ImputationDiD, TwoStageDiD, StackedDiD,
        EfficientDiD.
    cohort_periods : list
        The planned adoption periods, as values of ``time``. One value
        declares a common adoption date; two or more declare a staggered
        design in which each placebo-treated unit draws its period uniformly
        from the list. Each needs at least two earlier periods.
    treatment_fraction : float, optional
        Share of units treated; the count is ``int(n_units * treatment_fraction)``.
        Pass exactly one of ``treatment_fraction`` and ``n_treated``.
    n_treated : int, optional
        Number of units treated.
    effect_sizes : list of float, optional
        Effects to inject, on the outcome's scale, added to every treated
        observation from adoption onward. Effect 0 is always run first.
    n_simulations : int, default=500
        Placebo replicates per estimator and effect.
    alpha : float, default=0.05
        Significance level. Must equal each estimator's own ``alpha``.
    power : float, optional
        Target power for a minimum-detectable-effect search, run only for
        estimators that pass the size check. ``None`` skips the search.
    first_treat : str, optional
        Column with each unit's real first treatment period (0, inf or NaN
        for never-treated). When given, the panel is truncated just before the
        earliest real adoption.
    resample_units : bool, default=True
        Draw units with replacement in every replicate before assigning the
        placebo. With ``False`` the panel is used as observed and the expected
        rejection rate of a correctly sized test is not known exactly. With
        ``True``, an estimator may cluster only on the ``unit`` column: copies
        of a resampled unit are distinct units there and nowhere else.
    estimator_kwargs : dict, optional
        Extra ``fit()`` keywords applied to every estimator (for example
        ``covariates``). Column-role keywords are refused.
    seed : int, optional
        Seed of the placebo draws. ``None`` draws one and records it.
    progress : bool, default=True
        Print one line per estimator and effect.

    Returns
    -------
    DesignDiagnosisResults
        One :class:`~diff_diff.SimulationPowerResults` per estimator and
        effect, the size check, and the minimum detectable effects.

    Notes
    -----
    Estimators are returned in input order and are never ranked. An estimator
    that cannot fit any of the first five placebo replicates is refused with
    its error before the run starts. After that, a replicate whose fit raises
    a data-dependent error or returns non-finite inference is excluded from
    every rate and counted; an estimator that loses more than 10% of its
    no-effect replicates fails the size check.

    The methodology, its sources and the choices that have no paper source
    are documented in ``docs/methodology/REGISTRY.md`` ``## DesignDiagnosis``.

    Examples
    --------
    >>> from diff_diff import CallawaySantAnna, ImputationDiD, diagnose_design
    >>> result = diagnose_design(
    ...     panel, outcome="y", unit="id", time="year",
    ...     estimators=[CallawaySantAnna(), ImputationDiD()],
    ...     cohort_periods=[2019], treatment_fraction=0.2,
    ...     effect_sizes=[0.05], n_simulations=200, seed=1,
    ... )  # doctest: +SKIP
    >>> result.to_dataframe()  # doctest: +SKIP
    """
    if isinstance(n_simulations, bool) or not isinstance(n_simulations, (int, np.integer)):
        raise ValueError(f"n_simulations must be a positive integer, got {n_simulations!r}.")
    if n_simulations < 1:
        raise ValueError(f"n_simulations must be a positive integer, got {n_simulations}.")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if power is not None and not alpha < power < 1.0:
        raise ValueError(f"power must satisfy alpha < power < 1, got {power}.")

    resolved = _resolve_estimators(estimators)
    fit_extras = dict(estimator_kwargs or {})
    effects = _resolve_effect_sizes(effect_sizes)

    panel, n_dropped = _prepare_panel(data, outcome, unit, time, first_treat)
    periods = np.sort(panel[time].unique())
    n_periods = len(periods)
    n_units = panel[unit].nunique()
    if n_periods < _MIN_PRE_PERIODS + 1:
        raise ValueError(
            f"Too few periods: the untreated panel has {n_periods} period(s); at "
            f"least {_MIN_PRE_PERIODS + 1} are needed."
        )
    declared = _validate_cohort_periods(cohort_periods, periods)
    count = _resolve_n_treated(treatment_fraction, n_treated, n_units)
    _validate_estimators(
        resolved,
        alpha,
        staggered=len(declared) > 1,
        unit=unit,
        resample_units=resample_units,
    )
    _validate_estimator_kwargs(resolved, fit_extras)

    if seed is None:
        resolved_seed = int(np.random.SeedSequence().generate_state(1)[0])
    else:
        resolved_seed = int(seed)

    for label, est in resolved.items():
        if _uses_unseeded_bootstrap(est):
            warnings.warn(
                f"[{label}] uses bootstrap inference with seed=None, so its results "
                f"are not reproducible; the seed of diagnose_design fixes the "
                f"placebo draws only. Set the estimator's own seed.",
                UserWarning,
                stacklevel=2,
            )

    data_sha256, data_columns = _data_fingerprint(panel, unit, time)
    data_generator = _make_placebo_generator(
        panel, outcome, unit, time, count, declared, resample_units
    )
    fit_kwargs = {
        label: _fit_kwargs(est, outcome, unit, time, fit_extras) for label, est in resolved.items()
    }

    for label, est in resolved.items():
        with _WarningCapture() as capture:
            _preflight(label, est, data_generator, fit_kwargs[label], resolved_seed)
        _reemit(label, capture.records)

    results: Dict[str, List[SimulationPowerResults]] = {}
    n_nan_inference: Dict[str, List[int]] = {}
    size_controlled: Dict[str, bool] = {}
    mde_results: Dict[str, SimulationMDEResults] = {}
    mde: Dict[str, float] = {}

    for label, est in resolved.items():
        cells: List[SimulationPowerResults] = []
        nan_counts: List[int] = []
        null_adapter: Optional[_PlaceboFit] = None
        for effect in effects:
            with _WarningCapture() as capture:
                cell, adapter = _run_cell(
                    est,
                    effect,
                    data_generator,
                    fit_kwargs[label],
                    n_units,
                    n_periods,
                    int(n_simulations),
                    alpha,
                    resolved_seed,
                )
            _reemit(label, capture.records)
            cells.append(cell)
            nan_counts.append(adapter.n_nan)
            if null_adapter is None:
                null_adapter = adapter
            if progress:
                print(
                    f"  diagnose_design: {label}, effect={effect:g}: "
                    f"rejection rate {_fmt_rate(cell.power)} "
                    f"({cell.n_simulations} scored, {adapter.n_lost} lost)"
                )
        assert null_adapter is not None
        null_cell = cells[0]
        results[label] = cells
        n_nan_inference[label] = nan_counts
        size_controlled[label] = _size_controlled(
            float(null_cell.rejection_rate),
            int(null_cell.n_simulations),
            null_adapter.n_lost,
            alpha,
        )
        mde[label] = float("nan")
        if power is not None and size_controlled[label]:
            with _WarningCapture() as capture:
                found, value = _run_mde(
                    est,
                    null_cell,
                    data_generator,
                    fit_kwargs[label],
                    n_units,
                    n_periods,
                    int(n_simulations),
                    alpha,
                    power,
                    resolved_seed,
                )
            _reemit(label, capture.records)
            if found is not None:
                mde_results[label] = found
            mde[label] = value
            if progress:
                print(f"  diagnose_design: {label}, MDE search: {_fmt_num(value)}")

    return DesignDiagnosisResults(
        results=results,
        mde_results=mde_results,
        estimators={label: _clone(est) for label, est in resolved.items()},
        size_controlled=size_controlled,
        n_nan_inference=n_nan_inference,
        mde=mde,
        effect_sizes=effects,
        alpha=alpha,
        target_power=power,
        outcome=outcome,
        unit=unit,
        time=time,
        first_treat=first_treat,
        treatment_fraction=treatment_fraction,
        n_treated=count,
        cohort_periods=declared,
        resample_units=resample_units,
        estimator_kwargs=fit_extras,
        seed=resolved_seed,
        n_units=n_units,
        n_periods=n_periods,
        n_obs=len(panel),
        n_dropped=n_dropped,
        data_sha256=data_sha256,
        data_columns=data_columns,
    )
