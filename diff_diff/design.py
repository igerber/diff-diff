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

When several estimators are compared, the last periods are held out of the
comparison. :meth:`DesignDiagnosisResults.lock` then reruns the one chosen
estimator with those periods included and records the choice as a
:class:`DesignLock`; :meth:`DesignLock.reconcile` checks the analysis that was
finally run against it.

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

import copy
import dataclasses
import hashlib
import inspect
import json
import math
import warnings
from dataclasses import dataclass, field
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

__all__ = [
    "diagnose_design",
    "DesignDiagnosisResults",
    "DesignLock",
    "DesignReconciliationResults",
    "DESIGN_LOCK_SCHEMA_VERSION",
]

DESIGN_LOCK_SCHEMA_VERSION = "1.0"

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
        Shape of the panel actually used (without held-out periods).
    n_dropped : int
        Rows removed by truncating at the earliest real adoption. Held-out
        rows are not counted here.
    data_sha256 : str
        Fingerprint of the panel actually used.
    data_columns : list of str
        Columns the fingerprint covers.
    n_holdout : int
        Trailing periods of the untreated panel that were held out of the
        comparison and not read. :meth:`lock` uses them.
    time_periods : list
        Period values of the panel actually used.
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
    n_holdout: int
    time_periods: List[Any]
    # Each estimator's parameters as diagnosed (lock() refuses an instance
    # that was changed afterwards) and the labels a held-out run was made for.
    _params: Dict[str, Dict[str, Any]] = field(default_factory=dict, repr=False, compare=False)
    _holdout_labels: List[str] = field(default_factory=list, repr=False, compare=False)

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
            rows.extend(
                _records(
                    label,
                    cells,
                    self.size_controlled[label],
                    self.n_nan_inference[label],
                    self.mde[label],
                )
            )
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
            "n_holdout": self.n_holdout,
            "time_periods": list(self.time_periods),
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
        held_out = f"{self.n_holdout} (not read; used by lock())" if self.n_holdout else "0 (none)"
        lines = [
            "=" * width,
            "Design Diagnosis (placebo assignments on the supplied panel)".center(width),
            "=" * width,
            "",
            f"{'Units x periods:':<32} {self.n_units} x {self.n_periods}",
            f"{'Untreated status:':<32} {untreated}",
            f"{'Placebo-treated units:':<32} {self.n_treated}",
            f"{'Adoption periods (' + timing + '):':<32} {list(self.cohort_periods)}",
            f"{'Held-out periods:':<32} {held_out}",
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

    def lock(
        self,
        label: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        progress: bool = True,
    ) -> "DesignLock":
        """
        Record one estimator as the pre-specified choice.

        When periods were held out of the comparison (``n_holdout > 0``),
        the chosen estimator alone is rerun on the panel with those periods
        included, with fresh placebo draws, and the lock reports that run.
        Its numbers were not used to compare estimators, so they are not
        inflated by the choice.

        Parameters
        ----------
        label : str, optional
            Key of the chosen entry of ``estimators``. May be omitted only
            when one estimator was diagnosed.
        data : pd.DataFrame, optional
            The panel that was passed to :func:`diagnose_design`. Required
            when ``n_holdout > 0`` (the results do not keep the panel). Its
            fingerprint must match ``data_sha256`` on the periods that were
            compared.
        progress : bool, default=True
            Print one line per effect of the held-out run.

        Returns
        -------
        DesignLock
            The pre-specified design. Write ``lock.to_json()`` to a file to
            keep it; :meth:`DesignLock.reconcile` checks an analysis against
            it.

        Notes
        -----
        The held-out run uses a seed derived from the recorded one, so it
        cannot be re-rolled. Running it for a second label emits a warning:
        the two runs would amount to a second comparison.
        """
        return _lock(self, label, data, progress)


def _records(
    label: str,
    cells: Sequence[Any],
    size_controlled: bool,
    n_nan_inference: Sequence[int],
    mde: float,
) -> List[Dict[str, Any]]:
    """One row per effect size for one estimator (the ``to_dataframe`` rows)."""
    rows: List[Dict[str, Any]] = []
    for i, cell in enumerate(cells):
        row: Dict[str, Any] = {"label": label}
        row.update(cell.to_dict())
        row["size_controlled"] = size_controlled
        row["n_nan_inference"] = n_nan_inference[i]
        row["mde"] = mde
        rows.append(row)
    return rows


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


def _reemit(label: str, records: Sequence[warnings.WarningMessage], stacklevel: int = 3) -> None:
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
        warnings.warn(f"[{label}] {message}{suffix}", category, stacklevel=stacklevel)


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
    """Tagged text for one cell, so distinct values never encode alike.

    Numbers are encoded by value, not by storage type: ``1`` and ``1.0`` are
    the same cell. pandas stores an integer column as float as soon as any
    row is missing, so a type tag would make the digest of some rows depend
    on rows outside them.
    """
    if _is_missing(value):
        return ["n", ""]
    if isinstance(value, (bool, np.bool_)):
        return ["b", "True" if bool(value) else "False"]
    if isinstance(value, (int, np.integer)):
        return ["i", str(int(value))]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if number.is_integer():
            return ["i", str(int(number))]
        return ["f", format(number, ".17g")]
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
# JSON values of a design lock
# ---------------------------------------------------------------------------


def _json_value(value: Any, path: str, nonfinite: str = "raise") -> Any:
    """``value`` as plain JSON data, or ``ValueError`` naming ``path``.

    Strict on purpose: a lock must rebuild the estimator exactly, so anything
    that is not a JSON scalar, list or string-keyed mapping is refused instead
    of being passed through. ``nonfinite="none"`` maps NaN and infinities to
    ``None`` (result records); ``"raise"`` refuses them (configuration).
    """
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isfinite(number):
            return number
        if nonfinite == "none":
            return None
        raise ValueError(f"{path} is a non-finite number ({number!r})")
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_value(item, f"{path}[{i}]", nonfinite) for i, item in enumerate(value)]
    if isinstance(value, Mapping):
        converted: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} has a key that is not a string ({key!r})")
            converted[key] = _json_value(item, f"{path}.{key}", nonfinite)
        return converted
    raise ValueError(f"{path} holds a {type(value).__name__}, which cannot be written as JSON")


def _params_snapshot(estimator: Any) -> Dict[str, Any]:
    """Each constructor parameter as JSON data, or a marker when it has none."""
    snapshot: Dict[str, Any] = {}
    for name, value in estimator.get_params().items():
        try:
            snapshot[name] = _json_value(value, name)
        except ValueError:
            snapshot[name] = ["<not lockable>", repr(value)]
    return snapshot


def _unlockable(estimator: Any, estimator_kwargs: Dict[str, Any]) -> List[str]:
    """Why ``lock()`` would refuse this configuration (empty when it would not)."""
    problems: List[str] = []
    for group, values in (
        ("params", estimator.get_params()),
        ("estimator_kwargs", estimator_kwargs),
    ):
        for name, value in values.items():
            try:
                _json_value(value, f"{group}.{name}")
            except ValueError as exc:
                problems.append(str(exc))
    return problems


def _lock_digest(payload: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON of a lock's fields (without ``sha256``)."""
    text = json.dumps(
        payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
        if params.get("vcov_type") == "conley":
            raise ValueError(
                f"Estimator '{label}' uses vcov_type='conley', which diagnose_design "
                f"does not support. {name} takes its Conley time dimension from the "
                f"0/1 post indicator, so on a multi-period panel every pre-period "
                f"would count as one period and the spatial and serial terms would "
                f"be wrong; resampled copies of a unit would also share one location."
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


def _untreated_rows(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    first_treat: Optional[str],
) -> Tuple[pd.DataFrame, int]:
    """Drop every row from the earliest real adoption on.

    Reads only the ``unit``, ``time`` and ``first_treat`` columns, so the
    caller can remove held-out periods before any outcome is looked at.
    """
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
        # na_value: a nullable column (Int64 / Float64) marks never-treated
        # units with pd.NA, which older pandas will not convert on its own.
        values = data[first_treat].to_numpy(dtype=float, na_value=np.nan)
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
            keep = _period_values(data[time]).astype(float) < earliest
            n_dropped = int((~keep).sum())
            panel = data.loc[keep]

    if len(panel) == 0:
        raise ValueError("No untreated rows remain after truncation.")
    return panel, n_dropped


def _check_panel(panel: pd.DataFrame, outcome: str, unit: str, time: str) -> pd.DataFrame:
    """Validate outcomes and balance; return the panel sorted by (unit, time)."""
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
    return panel.sort_values([unit, time], kind="mergesort").reset_index(drop=True)


def _resolve_n_holdout(
    n_holdout: Optional[int], n_estimators: int, periods: np.ndarray, declared: List[Any]
) -> int:
    """How many trailing periods the comparison leaves unread."""
    n_post = int(np.sum(periods >= max(declared)))
    if n_holdout is None:
        if n_estimators == 1:
            return 0
        count = n_post // 2
        if count == 0:
            raise ValueError(
                f"Several estimators are compared, so periods are held out of the "
                f"comparison by default, but the latest adoption period "
                f"{max(declared)!r} is the final period of the untreated panel and "
                f"nothing can be held out. Declare an earlier adoption period, or "
                f"pass n_holdout=0 to compare on the whole panel."
            )
        return count
    if isinstance(n_holdout, bool) or not isinstance(n_holdout, (int, np.integer)):
        raise ValueError(f"n_holdout must be a non-negative integer, got {n_holdout!r}.")
    count = int(n_holdout)
    if not 0 <= count <= n_post - 1:
        raise ValueError(
            f"n_holdout={count} is out of range: the latest adoption period "
            f"{max(declared)!r} has {n_post} period(s) at or after it, and at least "
            f"one must stay in the comparison (0 <= n_holdout <= {n_post - 1})."
        )
    return count


def _select_columns(frame: pd.DataFrame, data_columns: Sequence[str]) -> pd.DataFrame:
    """The columns a fingerprint covers, found by their names as strings."""
    by_name = {str(c): c for c in frame.columns}
    missing = [name for name in data_columns if name not in by_name]
    if missing:
        raise ValueError(f"Column(s) {missing} of the diagnosed panel are not in data.")
    return frame[[by_name[name] for name in data_columns]]


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


def _period_values(column: pd.Series) -> np.ndarray:
    """The calendar column as a plain numpy array (handles nullable dtypes)."""
    if pd.api.types.is_integer_dtype(column):
        return column.to_numpy(dtype=np.int64)
    return column.to_numpy(dtype=float)


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
    period_values = _period_values(panel[time])[:n_periods]
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


def _run_estimator(
    caller: str,
    label: str,
    estimator: Any,
    effects: Sequence[float],
    data_generator: Callable[..., pd.DataFrame],
    fit_kwargs: Dict[str, Any],
    n_units: int,
    n_periods: int,
    n_simulations: int,
    alpha: float,
    power: Optional[float],
    seed: int,
    progress: bool,
    stacklevel: int = 4,
) -> Tuple[List[Any], List[int], bool, Optional[Any], float]:
    """Every effect size, the size check and the MDE search for one estimator.

    Returns the cells, the non-finite-inference counts, whether the size
    check passed, the MDE search result (``None`` when it did not run) and
    the reported MDE (NaN when not reported). Warnings are re-emitted with
    the label.
    """
    cells: List[Any] = []
    nan_counts: List[int] = []
    null_adapter: Optional[_PlaceboFit] = None
    for effect in effects:
        with _WarningCapture() as capture:
            cell, adapter = _run_cell(
                estimator,
                effect,
                data_generator,
                fit_kwargs,
                n_units,
                n_periods,
                n_simulations,
                alpha,
                seed,
            )
        _reemit(label, capture.records, stacklevel)
        cells.append(cell)
        nan_counts.append(adapter.n_nan)
        if null_adapter is None:
            null_adapter = adapter
        if progress:
            print(
                f"  {caller}: {label}, effect={effect:g}: "
                f"rejection rate {_fmt_rate(cell.power)} "
                f"({cell.n_simulations} scored, {adapter.n_lost} lost)"
            )
    assert null_adapter is not None
    null_cell = cells[0]
    size_ok = _size_controlled(
        float(null_cell.rejection_rate),
        int(null_cell.n_simulations),
        null_adapter.n_lost,
        alpha,
    )
    found: Optional[Any] = None
    value = float("nan")
    if power is not None and size_ok:
        with _WarningCapture() as capture:
            found, value = _run_mde(
                estimator,
                null_cell,
                data_generator,
                fit_kwargs,
                n_units,
                n_periods,
                n_simulations,
                alpha,
                power,
                seed,
            )
        _reemit(label, capture.records, stacklevel)
        if progress:
            print(f"  {caller}: {label}, MDE search: {_fmt_num(value)}")
    return cells, nan_counts, size_ok, found, value


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
    n_holdout: Optional[int] = None,
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
    n_holdout : int, optional
        Trailing periods of the untreated panel to hold out of the comparison.
        They are not read here; :meth:`DesignDiagnosisResults.lock` reruns the
        chosen estimator with them included, so its reported power is not
        taken from the runs that were used to choose it. ``None`` holds out
        nothing for a single estimator and, for several, half of the periods
        at or after the latest adoption period (rounded down); if that is
        zero the call raises. Pass ``0`` to compare on the whole panel.

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

    untreated, n_dropped = _untreated_rows(data, outcome, unit, time, first_treat)
    all_periods = np.unique(_period_values(untreated[time]))
    if len(all_periods) < _MIN_PRE_PERIODS + 1:
        raise ValueError(
            f"Too few periods: the untreated panel has {len(all_periods)} period(s); at "
            f"least {_MIN_PRE_PERIODS + 1} are needed."
        )
    declared = _validate_cohort_periods(cohort_periods, all_periods)
    held_out = _resolve_n_holdout(n_holdout, len(resolved), all_periods, declared)
    if held_out:
        # Held-out rows are removed by period value alone, before any outcome
        # is validated, hashed or fitted.
        periods = all_periods[:-held_out]
        keep = np.isin(_period_values(untreated[time]), periods)
        untreated = untreated.loc[keep]
    else:
        periods = all_periods
    panel = _check_panel(untreated, outcome, unit, time)
    n_periods = len(periods)
    n_units = panel[unit].nunique()
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
        problems = _unlockable(est, fit_extras)
        if problems:
            warnings.warn(
                f"[{label}] cannot be recorded in a design lock: {'; '.join(problems)}. "
                f"The diagnosis runs, but lock() will refuse this estimator.",
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
        cells, nan_counts, size_ok, found, value = _run_estimator(
            "diagnose_design",
            label,
            est,
            effects,
            data_generator,
            fit_kwargs[label],
            n_units,
            n_periods,
            int(n_simulations),
            alpha,
            power,
            resolved_seed,
            progress,
        )
        results[label] = cells
        n_nan_inference[label] = nan_counts
        size_controlled[label] = size_ok
        if found is not None:
            mde_results[label] = found
        mde[label] = value

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
        n_holdout=held_out,
        time_periods=periods.tolist(),
        _params={label: _params_snapshot(est) for label, est in resolved.items()},
    )


# ---------------------------------------------------------------------------
# Design lock
# ---------------------------------------------------------------------------

_ABSENT = "<absent>"
_MATCH = "match"
_DEVIATES = "deviates"
_NOT_CHECKED = "not checked"
_PRE_SPECIFIED = "pre-specified"
# Lock fields whose NaN entries (a cell that could not be scored, an MDE that
# was not reported) are stored as None.
_RECORD_FIELDS = ("comparison_results", "holdout_results", "mde")


def _lock(
    diagnosis: DesignDiagnosisResults,
    label: Optional[str],
    data: Optional[pd.DataFrame],
    progress: bool,
) -> "DesignLock":
    """Implementation of :meth:`DesignDiagnosisResults.lock`."""
    labels = list(diagnosis.estimators)
    if label is None:
        if len(labels) != 1:
            raise ValueError(
                f"Several estimators were diagnosed; pass the label of the one to "
                f"lock. Labels: {labels}."
            )
        label = labels[0]
    if label not in diagnosis.estimators:
        raise ValueError(f"Unknown label {label!r}. Labels: {labels}.")
    chosen = diagnosis.estimators[label]

    snapshot = diagnosis._params.get(label)
    if snapshot is not None and _params_snapshot(chosen) != snapshot:
        raise ValueError(
            f"The estimator stored under '{label}' was changed after the diagnosis "
            f"(its parameters no longer match the ones that were diagnosed). Run "
            f"diagnose_design again with the configuration you want to lock."
        )
    problems = _unlockable(chosen, diagnosis.estimator_kwargs)
    if problems:
        raise ValueError(
            f"Estimator '{label}' cannot be locked: {'; '.join(problems)}. A lock "
            f"must be able to rebuild the estimator exactly."
        )

    outcome, unit, time = diagnosis.outcome, diagnosis.unit, diagnosis.time
    if diagnosis.n_holdout > 0 and data is None:
        raise ValueError(
            f"{diagnosis.n_holdout} period(s) were held out of the comparison, so "
            f"lock() needs the panel to rerun '{label}' with them: pass data= (the "
            f"frame given to diagnose_design)."
        )

    n_units, n_periods, n_obs = diagnosis.n_units, diagnosis.n_periods, diagnosis.n_obs
    time_periods = list(diagnosis.time_periods)
    data_sha256 = diagnosis.data_sha256
    size_ok = diagnosis.size_controlled[label]
    mde = diagnosis.mde[label]
    holdout_records: List[Dict[str, Any]] = []

    if data is not None:
        untreated, _ = _untreated_rows(data, outcome, unit, time, diagnosis.first_treat)
        untreated = _select_columns(untreated, diagnosis.data_columns)
        found_periods = np.unique(_period_values(untreated[time])).tolist()
        n_compared = len(time_periods)
        if (
            found_periods[:n_compared] != time_periods
            or len(found_periods) - n_compared != diagnosis.n_holdout
        ):
            missing = [period for period in time_periods if period not in found_periods]
            other = [period for period in found_periods if period not in time_periods]
            raise ValueError(
                f"data does not have the periods of the diagnosed panel: expected the "
                f"{n_compared} compared period(s) followed by {diagnosis.n_holdout} "
                f"held-out one(s). Compared periods missing: {missing}; other periods "
                f"found: {other}."
            )
        compared = untreated.loc[np.isin(_period_values(untreated[time]), time_periods)]
        digest, _ = _data_fingerprint(compared, unit, time)
        if digest != diagnosis.data_sha256:
            raise ValueError(
                "data is not the panel that was diagnosed: its fingerprint on the "
                "compared periods differs from data_sha256."
            )

        if diagnosis.n_holdout > 0:
            if diagnosis._holdout_labels and label not in diagnosis._holdout_labels:
                warnings.warn(
                    f"The held-out periods were already used for "
                    f"{diagnosis._holdout_labels}. A held-out run for '{label}' as "
                    f"well makes the two runs a second comparison: do not use them "
                    f"to choose between estimators.",
                    UserWarning,
                    stacklevel=3,
                )
            full = _check_panel(untreated, outcome, unit, time)
            n_units = full[unit].nunique()
            n_periods = len(found_periods)
            generator = _make_placebo_generator(
                full,
                outcome,
                unit,
                time,
                diagnosis.n_treated,
                list(diagnosis.cohort_periods),
                diagnosis.resample_units,
            )
            fit_kwargs = _fit_kwargs(chosen, outcome, unit, time, dict(diagnosis.estimator_kwargs))
            # Derived, not an argument: the held-out run cannot be re-rolled.
            lock_seed = int(np.random.SeedSequence([int(diagnosis.seed), 1]).generate_state(1)[0])
            null = diagnosis.results[label][0]
            n_simulations = int(null.n_simulations) + int(null.n_simulation_failures)
            with _WarningCapture() as capture:
                _preflight(label, chosen, generator, fit_kwargs, lock_seed)
            _reemit(label, capture.records, stacklevel=4)
            cells, nan_counts, size_ok, _found, mde = _run_estimator(
                "lock",
                label,
                chosen,
                diagnosis.effect_sizes,
                generator,
                fit_kwargs,
                n_units,
                n_periods,
                n_simulations,
                diagnosis.alpha,
                diagnosis.target_power,
                lock_seed,
                progress,
                stacklevel=5,
            )
            holdout_records = _records(label, cells, size_ok, nan_counts, mde)
            n_obs = len(full)
            time_periods = found_periods
            data_sha256, _ = _data_fingerprint(full, unit, time)
            if label not in diagnosis._holdout_labels:
                diagnosis._holdout_labels.append(label)

    if not size_ok:
        warnings.warn(
            f"'{label}' did not pass the size check on the no-effect runs the lock "
            f"reports, so its power is not meaningful. The lock records "
            f"size_controlled=False.",
            UserWarning,
            stacklevel=3,
        )

    from diff_diff import __version__

    payload: Dict[str, Any] = {
        "schema_version": DESIGN_LOCK_SCHEMA_VERSION,
        "diff_diff_version": __version__,
        "label": label,
        "estimator_name": type(chosen).__name__,
        "params": chosen.get_params(),
        "estimator_kwargs": diagnosis.estimator_kwargs,
        "outcome": outcome,
        "unit": unit,
        "time": time,
        "first_treat": diagnosis.first_treat,
        "alpha": diagnosis.alpha,
        "target_power": diagnosis.target_power,
        "effect_sizes": diagnosis.effect_sizes,
        "n_treated": diagnosis.n_treated,
        "treatment_fraction": diagnosis.treatment_fraction,
        "cohort_periods": diagnosis.cohort_periods,
        "resample_units": diagnosis.resample_units,
        "seed": diagnosis.seed,
        "n_holdout": diagnosis.n_holdout,
        "n_units": n_units,
        "n_periods": n_periods,
        "n_obs": n_obs,
        "time_periods": time_periods,
        "data_sha256": data_sha256,
        "data_columns": diagnosis.data_columns,
        "comparison_results": diagnosis.to_dataframe().to_dict("records"),
        "holdout_results": holdout_records,
        "size_controlled": size_ok,
        "mde": mde,
    }
    converted = {
        key: _json_value(value, key, "none" if key in _RECORD_FIELDS else "raise")
        for key, value in payload.items()
    }
    return DesignLock(**converted, sha256=_lock_digest(converted))


@dataclass(frozen=True)
class DesignLock(Diagnostic):
    """
    A pre-specified design: the estimator chosen before the analysis.

    Made by :meth:`DesignDiagnosisResults.lock`. It records the chosen
    estimator and its configuration, the declared design, a fingerprint of
    the panel, every estimator that was compared, and the chosen estimator's
    size, power, bias, RMSE, coverage and minimum detectable effect. All
    values are plain JSON data, so ``DesignLock.from_json(lock.to_json())``
    equals ``lock``.

    Attributes
    ----------
    schema_version : str
        ``DESIGN_LOCK_SCHEMA_VERSION`` at the time the lock was made.
    diff_diff_version : str
        Library version that made the lock.
    label : str
        The chosen estimator's label in the comparison.
    estimator_name : str
        Its class name.
    params : dict
        Its constructor parameters.
    estimator_kwargs : dict
        Extra ``fit()`` keywords that were applied.
    outcome, unit, time : str
        Column names of the diagnosed panel.
    first_treat : str or None
        Real first-treatment column used to truncate the diagnosed panel.
    alpha : float
        Significance level.
    target_power : float or None
        Target power of the minimum-detectable-effect search.
    effect_sizes : list of float
        Injected effects, 0 first.
    n_treated : int
        Placebo-treated units per replicate.
    treatment_fraction : float or None
        Declared share of treated units, when given.
    cohort_periods : list
        Declared adoption periods.
    resample_units : bool
        Whether units were resampled in each replicate.
    seed : int
        Seed of the comparison. The held-out run uses a seed derived from it.
    n_holdout : int
        Periods held out of the comparison.
    n_units, n_periods, n_obs : int
        Shape of the panel the reported numbers come from: the panel with the
        held-out periods when a held-out run was made, else the compared one.
    time_periods : list
        Its period values.
    data_sha256 : str
        Its fingerprint.
    data_columns : list of str
        Columns the fingerprint covers.
    comparison_results : list of dict
        One record per estimator and effect from the comparison, for every
        estimator that was compared.
    holdout_results : list of dict
        The chosen estimator's records from the held-out run; empty when no
        periods were held out.
    size_controlled : bool
        Whether the chosen estimator passed the size check in the run the
        lock reports.
    mde : float or None
        Its minimum detectable effect; ``None`` when not reported.
    sha256 : str
        SHA-256 of the canonical JSON of every other field. It detects
        accidental edits; it is not a signature.
    """

    schema_version: str
    diff_diff_version: str
    label: str
    estimator_name: str
    params: Dict[str, Any]
    estimator_kwargs: Dict[str, Any]
    outcome: str
    unit: str
    time: str
    first_treat: Optional[str]
    alpha: float
    target_power: Optional[float]
    effect_sizes: List[float]
    n_treated: int
    treatment_fraction: Optional[float]
    cohort_periods: List[Any]
    resample_units: bool
    seed: int
    n_holdout: int
    n_units: int
    n_periods: int
    n_obs: int
    time_periods: List[Any]
    data_sha256: str
    data_columns: List[str]
    comparison_results: List[Dict[str, Any]]
    holdout_results: List[Dict[str, Any]]
    size_controlled: bool
    mde: Optional[float]
    sha256: str

    def __repr__(self) -> str:
        """Concise string representation."""
        return (
            f"DesignLock(label={self.label!r}, estimator_name={self.estimator_name!r}, "
            f"n_holdout={self.n_holdout}, sha256={self.sha256[:12]!r}...)"
        )

    def __hash__(self) -> int:
        # The generated hash would fail on the dict and list fields.
        return hash(self.sha256)

    def _payload(self) -> Dict[str, Any]:
        return {
            f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.name != "sha256"
        }

    def _verify(self) -> None:
        """Raise if the fields no longer match ``sha256``."""
        try:
            digest = _lock_digest(self._payload())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"The design lock holds a value that is not JSON data: {exc}.")
        if digest != self.sha256:
            raise ValueError(
                "The design lock does not match its sha256: it was edited after it "
                "was made. Reload it from the original file."
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the lock to a dictionary of plain JSON data.

        Returns
        -------
        Dict[str, Any]
            Every field, as a deep copy.
        """
        result = copy.deepcopy(self._payload())
        result["sha256"] = self.sha256
        return result

    def to_json(self, *, indent: int = 2) -> str:
        """Return ``to_dict()`` serialized as JSON."""
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DesignLock":
        """
        Rebuild a lock from ``to_dict()`` output.

        Raises
        ------
        ValueError
            If a key is unknown or missing, the schema version has a
            different major number, or the content does not match ``sha256``.
        """
        if not isinstance(data, Mapping):
            raise TypeError("A design lock is read from a dictionary.")
        names = [f.name for f in dataclasses.fields(cls)]
        unknown = sorted(str(key) for key in set(data) - set(names))
        missing = sorted(set(names) - set(data))
        if unknown or missing:
            raise ValueError(
                f"Not a design lock: unknown key(s) {unknown}, missing key(s) {missing}."
            )
        version = str(data["schema_version"])
        if version.split(".")[0] != DESIGN_LOCK_SCHEMA_VERSION.split(".")[0]:
            raise ValueError(
                f"Design lock schema version {version!r} is not supported by this "
                f"library version (which reads {DESIGN_LOCK_SCHEMA_VERSION!r})."
            )
        lock = cls(**copy.deepcopy(dict(data)))
        lock._verify()
        return lock

    @classmethod
    def from_json(cls, text: str) -> "DesignLock":
        """Rebuild a lock from ``to_json()`` output."""
        return cls.from_dict(json.loads(text))

    def to_dataframe(self) -> pd.DataFrame:
        """
        The chosen estimator's reported rows, one per effect size.

        Returns
        -------
        pd.DataFrame
            ``holdout_results`` when a held-out run was made, otherwise the
            chosen estimator's rows of ``comparison_results``.
        """
        if self.holdout_results:
            return pd.DataFrame(copy.deepcopy(self.holdout_results))
        rows = [row for row in self.comparison_results if row.get("label") == self.label]
        return pd.DataFrame(copy.deepcopy(rows))

    def summary(self) -> str:
        """
        Generate a formatted summary of the lock.

        Returns
        -------
        str
            Formatted summary.
        """
        width = 78
        timing = "common" if len(self.cohort_periods) == 1 else "staggered"
        source = (
            f"held-out run ({self.n_holdout} period(s) the comparison did not read)"
            if self.holdout_results
            else "comparison run (not confirmed on held-out periods)"
        )
        compared = list(dict.fromkeys(str(row.get("label")) for row in self.comparison_results))
        mde_text = f"{self.mde:.4g}" if self.mde is not None else "-"
        lines = [
            "=" * width,
            "Design Lock (pre-specified design)".center(width),
            "=" * width,
            "",
            f"{'Chosen:':<32} {self.label} ({self.estimator_name})",
            f"{'Compared:':<32} {', '.join(compared)}",
            f"{'Units x periods:':<32} {self.n_units} x {self.n_periods}",
            f"{'Placebo-treated units:':<32} {self.n_treated}",
            f"{'Adoption periods (' + timing + '):':<32} {list(self.cohort_periods)}",
            f"{'Significance level (alpha):':<32} {self.alpha:.3f}",
            f"{'Reported numbers from:':<32} {source}",
            f"{'Size OK:':<32} {'yes' if self.size_controlled else 'NO'}",
            f"{'MDE:':<32} {mde_text}",
            "",
            "-" * width,
            f"{'Effect':>10}{'Rejects':>10}{'Bias':>12}{'RMSE':>12}{'Coverage':>10}"
            f"{'Scored':>8}{'Lost':>7}",
            "-" * width,
        ]
        for row in self.to_dataframe().to_dict("records"):
            lines.append(
                f"{row['true_effect']:>10.4g}{_fmt_cell(row['rejection_rate'], rate=True):>10}"
                f"{_fmt_cell(row['bias']):>12}{_fmt_cell(row['rmse']):>12}"
                f"{_fmt_cell(row['coverage'], rate=True):>10}"
                f"{row['n_simulations']:>8}{row['n_simulation_failures']:>7}"
            )
        lines.extend(
            [
                "",
                f"diff-diff {self.diff_diff_version}, lock schema {self.schema_version}",
                f"sha256 {self.sha256}",
                "=" * width,
            ]
        )
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print the summary to stdout."""
        print(self.summary())

    def to_estimator(self) -> Any:
        """
        Build the locked estimator.

        Returns
        -------
        estimator
            A fresh, unfitted instance of ``estimator_name`` configured with
            ``params``. Fitting it, with ``estimator_kwargs``, is the
            pre-specified analysis.
        """
        self._verify()
        if self.estimator_name not in _SUPPORTED_ESTIMATORS:
            raise ValueError(
                f"The lock names {self.estimator_name!r}, which is not one of "
                f"{', '.join(_SUPPORTED_ESTIMATORS)}."
            )
        import diff_diff

        return getattr(diff_diff, self.estimator_name)(**copy.deepcopy(self.params))

    def reconcile(
        self,
        estimator: Any,
        *,
        outcome: str,
        unit: str,
        time: str,
        first_treat: Optional[str] = None,
        treatment: Optional[str] = None,
        post: Optional[str] = None,
        estimator_kwargs: Optional[Dict[str, Any]] = None,
        data: Optional[pd.DataFrame] = None,
        results: Optional[Any] = None,
    ) -> "DesignReconciliationResults":
        """
        Check the analysis that was run against this lock.

        Parameters
        ----------
        estimator : estimator
            The estimator used for the analysis.
        outcome, unit, time : str
            Outcome, unit and calendar-period column names of the analysis.
        first_treat : str, optional
            Column with each unit's real first treatment period (0, inf or
            NaN for never-treated), for the estimators that take one.
        treatment : str, optional
            0/1 group column, for ``DifferenceInDifferences`` and
            ``TwoWayFixedEffects``. Pass ``first_treat`` or ``treatment``,
            not both.
        post : str, optional
            0/1 post-period column that goes with ``treatment``.
        estimator_kwargs : dict, optional
            Extra ``fit()`` keywords the analysis used.
        data : pd.DataFrame, optional
            The analysis data. When given, the data and the realized design
            are checked too.
        results : results object, optional
            The fitted results. It must be ``estimator.results_``.

        Returns
        -------
        DesignReconciliationResults
            One check per item and an overall ``status``: ``"pre-specified"``
            when every check that was run matches, otherwise ``"deviates"``.

        Notes
        -----
        This compares what the analyst declares with the lock. Fitted results
        do not record the constructor parameters or column names that
        produced them, so a parameter changed with ``set_params()`` after
        fitting cannot be detected. ``"pre-specified"`` covers the listed
        checks only.

        The data fingerprint covers every column the diagnosed panel had, on
        the locked periods. A column whose values changed there deviates;
        that includes an adoption column that was all never-treated when the
        design was diagnosed and holds the real dates now.

        Adoption dates are not compared: placebo dates lie inside the
        diagnosed panel and real ones after it. The number of cohorts, the
        number of treated units, the length of each cohort's post window and
        whether the panel is balanced are.
        """
        self._verify()
        if not hasattr(estimator, "get_params"):
            raise TypeError("reconcile needs the estimator instance used for the analysis.")
        if first_treat is not None and treatment is not None:
            raise ValueError(
                "Pass first_treat (a first-treatment-period column) or treatment (a "
                "0/1 group column), not both."
            )
        if results is not None and getattr(estimator, "results_", None) is not results:
            raise ValueError(
                "results is not this estimator's results_: it was not produced by "
                "the estimator being reconciled."
            )
        if data is not None and not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")

        checks = [_compare("estimator_name", self.estimator_name, type(estimator).__name__)]
        checks.extend(_compare_mapping("params", self.params, estimator.get_params()))
        checks.extend(
            _compare_mapping(
                "estimator_kwargs", self.estimator_kwargs, dict(estimator_kwargs or {})
            )
        )
        checks.append(_compare("outcome", self.outcome, outcome))
        checks.append(_compare("unit", self.unit, unit))
        checks.append(_compare("time", self.time, time))
        if results is None:
            checks.append(_check_row("alpha", self.alpha, None, _NOT_CHECKED))
        else:
            checks.append(_compare("alpha", self.alpha, getattr(results, "alpha", _ABSENT)))
        checks.extend(_design_checks(self, data, unit, time, first_treat, treatment, post))

        deviates = any(check["status"] == _DEVIATES for check in checks)
        return DesignReconciliationResults(
            status=_DEVIATES if deviates else _PRE_SPECIFIED,
            checks=checks,
            label=self.label,
            lock_sha256=self.sha256,
        )


def _fmt_cell(value: Optional[float], rate: bool = False) -> str:
    if value is None:
        return "-"
    return _fmt_rate(value) if rate else _fmt_num(value)


def _check_row(item: str, locked: Any, realized: Any, status: str) -> Dict[str, Any]:
    return {"item": item, "locked": locked, "realized": realized, "status": status}


def _same(locked: Any, realized: Any) -> bool:
    """Equality of two JSON values; a boolean never equals a number."""
    if isinstance(locked, bool) != isinstance(realized, bool):
        return False
    if isinstance(locked, list) and isinstance(realized, list):
        return len(locked) == len(realized) and all(_same(a, b) for a, b in zip(locked, realized))
    if isinstance(locked, dict) and isinstance(realized, dict):
        return set(locked) == set(realized) and all(
            _same(locked[key], realized[key]) for key in locked
        )
    return bool(locked == realized)


def _compare(item: str, locked: Any, realized: Any) -> Dict[str, Any]:
    """One check. A realized value that is not JSON data cannot match a lock."""
    try:
        value = _json_value(realized, item)
    except ValueError:
        return _check_row(item, locked, repr(realized), _DEVIATES)
    return _check_row(item, locked, value, _MATCH if _same(locked, value) else _DEVIATES)


def _compare_mapping(
    group: str, locked: Mapping[str, Any], realized: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    names = list(locked) + [name for name in realized if name not in locked]
    rows = []
    for name in names:
        item = f"{group}.{name}"
        if name not in realized:
            rows.append(_check_row(item, locked[name], _ABSENT, _DEVIATES))
        elif name not in locked:
            rows.append(_check_row(item, _ABSENT, repr(realized[name]), _DEVIATES))
        else:
            rows.append(_compare(item, locked[name], realized[name]))
    return rows


def _design_checks(
    lock: DesignLock,
    data: Optional[pd.DataFrame],
    unit: str,
    time: str,
    first_treat: Optional[str],
    treatment: Optional[str],
    post: Optional[str],
) -> List[Dict[str, Any]]:
    """The checks that need the analysis data."""
    cohorts = sorted(lock.cohort_periods)
    locked: Dict[str, Any] = {
        "data_sha256": lock.data_sha256,
        "n_units": lock.n_units,
        "is_balanced": True,
        "n_treated": lock.n_treated,
        "n_cohorts": len(cohorts),
        "n_post_by_cohort": [
            sum(1 for period in lock.time_periods if period >= cohort) for cohort in cohorts
        ],
    }
    realized: Dict[str, Any] = {}
    problems: Dict[str, str] = {}
    timing = ("n_treated", "n_cohorts", "n_post_by_cohort")

    if data is not None:
        # The fingerprint is taken with the lock's own column names: it asks
        # whether the diagnosed panel is still in the data.
        try:
            selected = _select_columns(data, lock.data_columns)
            by_name = {str(column): column for column in selected.columns}
            lock_unit, lock_time = by_name[str(lock.unit)], by_name[str(lock.time)]
            rows = selected.loc[selected[lock_time].isin(lock.time_periods)]
            realized["data_sha256"], _ = _data_fingerprint(rows, lock_unit, lock_time)
        except (KeyError, ValueError) as exc:
            problems["data_sha256"] = f"unavailable: {exc}"

        absent = [column for column in (unit, time) if column not in data.columns]
        if absent:
            requested = timing if (first_treat is not None or treatment is not None) else ()
            for item in ("n_units", "is_balanced") + requested:
                problems[item] = f"column(s) {absent} not in data"
        else:
            n_units = int(data[unit].nunique())
            periods = pd.unique(data[time])
            realized["n_units"] = n_units
            realized["is_balanced"] = bool(
                not data.duplicated([unit, time]).any() and len(data) == n_units * len(periods)
            )
            if first_treat is not None:
                problem, values = _realized_first_treat(data, unit, first_treat)
                if problem is not None:
                    for item in timing:
                        problems[item] = problem
                else:
                    realized_cohorts = sorted(set(values))
                    realized["n_treated"] = len(values)
                    realized["n_cohorts"] = len(realized_cohorts)
                    try:
                        realized["n_post_by_cohort"] = [
                            int(np.sum(periods >= cohort)) for cohort in realized_cohorts
                        ]
                    except TypeError:
                        problems["n_post_by_cohort"] = (
                            f"'{time}' cannot be compared with '{first_treat}'"
                        )
            elif treatment is not None:
                if treatment not in data.columns:
                    problems["n_treated"] = problems["n_cohorts"] = (
                        f"column {treatment!r} not in data"
                    )
                elif (data.groupby(unit)[treatment].nunique(dropna=False) > 1).any():
                    problems["n_treated"] = problems["n_cohorts"] = "varies within unit"
                else:
                    n_treated = int((data.groupby(unit)[treatment].first() == 1).sum())
                    realized["n_treated"] = n_treated
                    realized["n_cohorts"] = 1 if n_treated else 0
                if post is not None:
                    if post not in data.columns:
                        problems["n_post_by_cohort"] = f"column {post!r} not in data"
                    elif (data.groupby(time)[post].nunique(dropna=False) > 1).any():
                        problems["n_post_by_cohort"] = "varies within period"
                    else:
                        realized["n_post_by_cohort"] = [
                            int((data.groupby(time)[post].first() == 1).sum())
                        ]

    rows_out = []
    for item, value in locked.items():
        if item in problems:
            rows_out.append(_check_row(item, value, problems[item], _DEVIATES))
        elif item in realized:
            rows_out.append(_compare(item, value, realized[item]))
        else:
            rows_out.append(_check_row(item, value, None, _NOT_CHECKED))
    return rows_out


def _realized_first_treat(
    data: pd.DataFrame, unit: str, first_treat: str
) -> Tuple[Optional[str], List[float]]:
    """Each treated unit's adoption period, or why it cannot be read."""
    if first_treat not in data.columns:
        return f"column {first_treat!r} not in data", []
    if not pd.api.types.is_numeric_dtype(data[first_treat]):
        return f"'{first_treat}' is not numeric", []
    if (data.groupby(unit)[first_treat].nunique(dropna=False) > 1).any():
        return "varies within unit", []
    values = data[first_treat].to_numpy(dtype=float, na_value=np.nan)
    first_rows = ~data[unit].duplicated().to_numpy()
    per_unit = values[first_rows]
    adopted = per_unit[np.isfinite(per_unit) & (per_unit > 0)]
    return None, [float(value) for value in adopted]


@dataclass
class DesignReconciliationResults(Diagnostic):
    """
    Results of :meth:`DesignLock.reconcile`.

    Attributes
    ----------
    status : str
        ``"pre-specified"`` when every check that was run matches the lock,
        otherwise ``"deviates"``. The size of a deviation is not graded.
    checks : list of dict
        One entry per reconciled item, with keys ``item``, ``locked``,
        ``realized`` and ``status`` (``"match"``, ``"deviates"`` or
        ``"not checked"``).
    label : str
        Label of the locked estimator.
    lock_sha256 : str
        ``sha256`` of the lock that was reconciled against.
    """

    status: str
    checks: List[Dict[str, Any]]
    label: str
    lock_sha256: str

    def __repr__(self) -> str:
        """Concise string representation."""
        n_deviating = sum(1 for check in self.checks if check["status"] == _DEVIATES)
        return (
            f"DesignReconciliationResults(status={self.status!r}, "
            f"checks={len(self.checks)}, deviating={n_deviating})"
        )

    def to_dataframe(self) -> pd.DataFrame:
        """
        One row per reconciled item.

        Returns
        -------
        pd.DataFrame
            Columns ``item``, ``locked``, ``realized``, ``status``.
        """
        return pd.DataFrame(
            copy.deepcopy(self.checks), columns=["item", "locked", "realized", "status"]
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert results to a dictionary.

        Returns
        -------
        Dict[str, Any]
            The overall status, the lock reconciled against and every check.
        """
        return {
            "status": self.status,
            "label": self.label,
            "lock_sha256": self.lock_sha256,
            "checks": copy.deepcopy(self.checks),
        }

    def summary(self) -> str:
        """
        Generate a formatted summary of the reconciliation.

        Returns
        -------
        str
            Formatted summary: the overall status and every item that
            deviates or was not checked.
        """
        width = 78
        counts = {status: 0 for status in (_MATCH, _DEVIATES, _NOT_CHECKED)}
        for check in self.checks:
            counts[check["status"]] = counts.get(check["status"], 0) + 1
        lines = [
            "=" * width,
            "Design Reconciliation (analysis vs. design lock)".center(width),
            "=" * width,
            "",
            f"{'Locked estimator:':<24} {self.label}",
            f"{'Status:':<24} {self.status}",
            f"{'Checks:':<24} {counts[_MATCH]} match, {counts[_DEVIATES]} deviate, "
            f"{counts[_NOT_CHECKED]} not checked",
            f"{'Lock sha256:':<24} {self.lock_sha256}",
        ]
        shown = [check for check in self.checks if check["status"] != _MATCH]
        if shown:
            lines.extend(
                [
                    "",
                    "-" * width,
                    f"{'Item':<30}{'Locked':<17}{'Realized':<17}{'Status':<14}",
                    "-" * width,
                ]
            )
            for check in shown:
                realized = "" if check["realized"] is None else str(check["realized"])
                lines.append(
                    f"{str(check['item'])[:29]:<30}{str(check['locked'])[:16]:<17}"
                    f"{realized[:16]:<17}{check['status']:<14}"
                )
        lines.extend(
            [
                "",
                "'pre-specified' means every check that was run matches the lock. It is",
                "a statement about the listed items only, and deviations are not graded.",
                "=" * width,
            ]
        )
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print the summary to stdout."""
        print(self.summary())
