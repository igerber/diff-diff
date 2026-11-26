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

Placebo treatment is assigned at random, so the diagnosis tests inference
(size, power) on the panel's own error structure. It does not test parallel
trends.

:meth:`DesignDiagnosisResults.lock` records one estimator and its
configuration as a :class:`DesignLock`. When several estimators were compared
it first reruns the chosen one alone with new placebo draws and reports that
run. :meth:`DesignLock.reconcile` checks the analysis that was finally run
against the lock.

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
from scipy import stats

from diff_diff.power import (
    SimulationMDEResults,
    SimulationPowerResults,
    simulate_mde,
    simulate_power,
)
from diff_diff.results_base import Diagnostic, EventStudyResults
from diff_diff.utils import safe_inference

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

# Candidates whose placebo event study is recorded under event_study=True: the
# classes whose aggregate("event_study") container PreTrendsPower admits.
_EVENT_STUDY_ESTIMATORS = ("CallawaySantAnna", "StackedDiD")
# Attribute of a placebo event study that holds, per placebo draw, each
# pre-period coefficient over its own reported standard error (absolute
# value). PreTrendsPower counts the pre-trend test's rejections from it for
# whichever pre-periods it is asked to plan (see placebo_pretest_rate).
_PRETEST_RECORD = "_placebo_pretest"
# Attribute of a placebo event study that holds the event-study container of
# every placebo draw behind it, and how many no-effect replicates never became
# one. HonestDiD.power_by_M runs on those containers (see _placebo_draw_record).
_PLACEBO_DRAWS = "_placebo_draws"
# The placebo covariance is not reported when its smallest eigenvalue is at or
# below this multiple of its largest (the near-singularity threshold
# results_base._validate_vcov_subblock applies when allow_singular=False).
_RANK_TOLERANCE = 1e-10
# The estimand label of a placebo event-study surface (EventStudyResults.estimand).
PLACEBO_ESTIMAND = "placebo"

# Printed by both summaries: what a placebo diagnosis does and does not test.
_SCOPE_LINES = (
    "Placebo treatment is assigned at random, so these runs test inference",
    "(size, power) on this panel's error structure, not parallel trends.",
)


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
    time_periods : list
        Period values of the panel actually used.
    event_study : bool
        Whether placebo event studies were requested.
    event_study_results : dict
        Label -> :class:`~diff_diff.EventStudyResults`, the placebo event
        study of each ``CallawaySantAnna`` or ``StackedDiD`` estimator for
        which one could be recorded; empty when ``event_study`` is False.
        Its ``att`` column is the mean of the no-effect replicates'
        event-study coefficients, ``vcov`` their sample covariance across the
        replicates, and ``se`` the square root of its diagonal: the spread of
        one fit's coefficient, which is what a single fit's standard error
        should equal. ``t_stat``, ``p_value`` and the interval compare that
        mean with that spread; they are not inference on an effect, and the
        error bars of a plot of the surface show the spread. The surface
        carries ``estimand="placebo"``. Pass this object (or one entry) to
        :class:`~diff_diff.PreTrendsPower` for planned pre-trend power.
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
    time_periods: List[Any]
    event_study: bool = False
    # Excluded from ==: the surfaces hold arrays, on which dataclass equality
    # raises.
    event_study_results: Dict[str, EventStudyResults] = field(default_factory=dict, compare=False)
    # The configuration as diagnosed (lock() refuses one that was changed
    # afterwards) and the labels lock() has rerun.
    _classes: Dict[str, type] = field(default_factory=dict, repr=False, compare=False)
    _params: Dict[str, Dict[str, Any]] = field(default_factory=dict, repr=False, compare=False)
    _fit_keywords: Optional[Dict[str, Any]] = field(default=None, repr=False, compare=False)
    _rerun_labels: List[str] = field(default_factory=list, repr=False, compare=False)
    # Per label, how its placebo event study was reduced: draws used, draws
    # excluded, the reported-SE ratio per pre-period, and how often a
    # pre-trend test over all pre-periods rejected in the no-effect draws
    # against the rate the power formula assumes.
    _event_study_checks: Dict[str, Dict[str, Any]] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore a pickled object, filling fields added after it was pickled."""
        self.__dict__.update(state)
        # Fields with a default_factory have no class-level fallback.
        self.__dict__.setdefault("event_study", False)
        self.__dict__.setdefault("event_study_results", {})
        self.__dict__.setdefault("_event_study_checks", {})

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
            "event_study": self.event_study,
            "n_units": self.n_units,
            "n_periods": self.n_periods,
            "n_obs": self.n_obs,
            "n_dropped": self.n_dropped,
            "data_sha256": self.data_sha256,
            "data_columns": list(self.data_columns),
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
        if self.event_study:
            lines.extend(_event_study_lines(self, width))
        lines.extend(
            [
                "",
                "Estimators are listed in input order and are not ranked. 'Size OK' is",
                "a check on the no-effect runs only; power for an estimator that fails",
                "it is not meaningful.",
                *_SCOPE_LINES,
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
        Record one estimator and its configuration as a :class:`DesignLock`.

        When several estimators were diagnosed, the chosen one alone is rerun
        on the same panel with new placebo draws, and the lock reports that
        run. The rerun's draws were not used to choose, which removes the
        part of the overstatement of the chosen estimator's power that comes
        from the draws. The part that comes from the panel itself remains
        (about one percentage point in the library's simulations; see the
        methodology registry). When one estimator was diagnosed nothing was
        chosen, so there is no rerun and the lock reports the diagnosis.

        Parameters
        ----------
        label : str, optional
            Key of the chosen entry of ``estimators``. May be omitted only
            when one estimator was diagnosed.
        data : pd.DataFrame, optional
            The panel that was passed to :func:`diagnose_design`. Required
            when several estimators were diagnosed (the results do not keep
            the panel). Its periods and fingerprint must match the diagnosed
            panel (``time_periods``, ``data_sha256``).
        progress : bool, default=True
            Print one line per effect of the rerun.

        Returns
        -------
        DesignLock
            The recorded design. Write ``lock.to_json()`` to a file to keep
            it; :meth:`DesignLock.reconcile` checks an analysis against it.
            The lock does not record when it was made.

        Notes
        -----
        The rerun uses a seed derived from the recorded one, so it cannot be
        re-rolled. A rerun for a second label emits a warning: the two runs
        would amount to a second comparison.

        The lock's ``size_controlled`` is the rerun's verdict. If it differs
        from the comparison's verdict a warning states both.

        A search made of separate :func:`diagnose_design` calls with one
        estimator each is not visible here: each of those locks reports its
        own diagnosis, with no rerun.
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

    def __init__(self, estimator: Any, capture: Optional[List[Any]] = None) -> None:
        self._estimator = estimator
        # None, or the list that receives one event-study record per scored
        # replicate (see _event_study_record).
        self._capture = capture
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
        if self._capture is not None:
            self._capture.append(_event_study_record(result))
        return result


def _clone(estimator: Any) -> Any:
    """An unfitted copy with the same constructor parameters."""
    return type(estimator)(**estimator.get_params())


# ---------------------------------------------------------------------------
# Placebo event study
# ---------------------------------------------------------------------------


def _event_study_skip_reason(estimator: Any) -> Optional[str]:
    """Why no placebo event study is recorded for this estimator, else None."""
    name = type(estimator).__name__
    if name not in _EVENT_STUDY_ESTIMATORS:
        return (
            f"{name} event studies are not an input PreTrendsPower accepts from "
            f"diagnose_design (recorded for: {', '.join(_EVENT_STUDY_ESTIMATORS)})"
        )
    if estimator.get_params().get("vcov_type") == "jackknife":
        return (
            "the pre-trend check of a cluster-jackknife fit compares each "
            "coefficient with a t(H-1) reference under a Bonferroni correction, "
            "which is not the test PreTrendsPower models (reports skip pre-trends "
            "power for such fits too)"
        )
    return None


def _plain(value: Any) -> Any:
    """A numpy scalar as the Python scalar it wraps; anything else unchanged."""
    return value.item() if isinstance(value, np.generic) else value


def _event_study_record(result: Any) -> Optional[Tuple[Any, np.ndarray, np.ndarray, Any]]:
    """One replicate's event study: layout, coefficient and SE vectors, container.

    The layout is the rows that are a reference row or carry a finite estimate
    and a finite positive standard error, together with the provenance the
    consumers read. Two replicates are averaged only if their layouts are
    equal, so a surface's provenance is that of every draw behind it. ``None``
    when the aggregation fails on this draw.
    """
    try:
        surface = result.aggregate("event_study")
    except _DATA_DEPENDENT_ERRORS:
        return None
    event_time = [_plain(v) for v in surface.event_time.tolist()]
    is_reference = np.asarray(surface.is_reference, dtype=bool)
    att = np.asarray(surface.att, dtype=float)
    se = np.asarray(surface.se, dtype=float)
    estimated = ~is_reference & np.isfinite(att) & np.isfinite(se) & (se > 0)
    keep = is_reference | estimated
    rows = tuple((event_time[i], bool(is_reference[i])) for i in range(len(event_time)) if keep[i])
    reference_event_times = surface.reference_event_times
    provenance = (
        surface.source,
        surface.base_period,
        (
            None
            if reference_event_times is None
            else tuple(_plain(v) for v in reference_event_times)
        ),
        surface.event_time_convention,
    )
    return (rows, provenance), att[estimated], se[estimated], surface


def _placebo_event_study(
    estimator: Any,
    records: Sequence[Optional[Tuple[Any, ...]]],
    alpha: float,
    n_lost: int = 0,
) -> Tuple[Optional[EventStudyResults], Optional[Dict[str, Any]], Optional[str]]:
    """Reduce one estimator's no-effect event studies to one planned surface.

    Returns ``(surface, checks, None)`` when a surface is reported and
    ``(None, None, reason)`` when it is not.

    The coefficients are the means over the draws and the covariance is the
    sample covariance of the coefficient vectors across the draws. The other
    inference columns follow from those two; they describe how one draw varies
    around the mean of the draws, not an estimated effect.

    A record is ``(layout, coefficients, standard errors)`` followed, when the
    replicate's event-study container was kept, by that container. ``n_lost``
    is the number of no-effect replicates that produced no record at all (the
    fit raised or its inference was not finite); it is carried on the surface
    with the draws so that a consumer can count every lost replicate.
    """
    n_captured = len(records)
    usable = [record for record in records if record is not None]
    if not usable:
        return None, None, "no no-effect replicate produced an event study"
    counts: Dict[Any, int] = {}
    for record in usable:
        counts[record[0]] = counts.get(record[0], 0) + 1
    # Most frequent layout; ties to the one with more rows, then to a fixed
    # order, so the choice never depends on iteration order.
    layout = min(counts, key=lambda key: (-counts[key], -len(key[0]), repr(key)))
    rows, provenance = layout
    used = [record for record in usable if record[0] == layout]
    n_used = len(used)
    n_excluded = n_captured - n_used
    if n_excluded / n_captured > _MAX_LOST_SHARE:
        return (
            None,
            None,
            f"{n_excluded} of {n_captured} no-effect replicates had a different "
            f"event-study layout or no event study (more than {_MAX_LOST_SHARE:.0%})",
        )
    labels = [label for label, is_ref in rows if not is_ref]
    k = len(labels)
    pre = [position for position, label in enumerate(labels) if label < 0]
    if not pre:
        name = type(estimator).__name__
        hint = (
            " (StackedDiD estimates no pre-period at kappa_pre=1; set kappa_pre >= 2)"
            if name == "StackedDiD"
            else ""
        )
        return None, None, f"the event study has no estimated pre-treatment period{hint}"
    if n_used < 2 * k + 1:
        return (
            None,
            None,
            f"{n_used} replicates are too few for the covariance of {k} "
            f"coefficients (the library requires at least {2 * k + 1}); raise n_simulations",
        )
    draws = np.vstack([record[1] for record in used])
    vcov = np.atleast_2d(np.cov(draws, rowvar=False, ddof=1))
    singular = "the placebo covariance of the event-study coefficients is "
    if not np.all(np.isfinite(vcov)):
        return None, None, singular + "not finite"
    eigenvalues = np.linalg.eigvalsh((vcov + vcov.T) / 2.0)
    if eigenvalues[-1] <= 0.0 or eigenvalues[0] <= _RANK_TOLERANCE * eigenvalues[-1]:
        return (
            None,
            None,
            singular + "rank-deficient (the replicates contain too few distinct "
            "placebo assignments; this happens on a small panel with "
            "resample_units=False)",
        )
    means = draws.mean(axis=0)
    spread = np.sqrt(np.diag(vcov))
    n_rows = len(rows)
    att = np.zeros(n_rows)
    columns = {
        name: np.full(n_rows, np.nan)
        for name in ("se", "t_stat", "p_value", "conf_int_lower", "conf_int_upper")
    }
    position = 0
    for i, (_label, is_ref) in enumerate(rows):
        if is_ref:
            continue
        t_stat, p_value, conf_int = safe_inference(
            float(means[position]), float(spread[position]), alpha=alpha
        )
        att[i] = means[position]
        columns["se"][i] = spread[position]
        columns["t_stat"][i] = t_stat
        columns["p_value"][i] = p_value
        columns["conf_int_lower"][i] = conf_int[0]
        columns["conf_int_upper"][i] = conf_int[1]
        position += 1
    source, base_period, reference_event_times, convention = provenance
    surface = EventStudyResults(
        event_time=np.asarray([label for label, _ in rows]),
        att=att,
        se=columns["se"],
        t_stat=columns["t_stat"],
        p_value=columns["p_value"],
        conf_int_lower=columns["conf_int_lower"],
        conf_int_upper=columns["conf_int_upper"],
        is_reference=np.asarray([is_ref for _, is_ref in rows], dtype=bool),
        n=np.full(n_rows, np.nan),
        n_kind=None,
        time_scale="relative",
        event_time_convention=convention,
        vcov=vcov,
        vcov_index=np.asarray(labels),
        alpha=alpha,
        source=source,
        base_period=base_period,
        anticipation=0,
        reference_event_times=reference_event_times,
        estimand=PLACEBO_ESTIMAND,
    )
    reported = np.vstack([record[2] for record in used])
    mean_reported = reported.mean(axis=0)
    pre_labels = [labels[i] for i in pre]
    # Kept on the surface so that the check below can be repeated for any
    # subset of the pre-periods (PreTrendsPower does, for the set it plans).
    setattr(
        surface,
        _PRETEST_RECORD,
        {"labels": pre_labels, "abs_t": np.abs(draws[:, pre] / reported[:, pre])},
    )
    # The draws' own containers, for consumers that run a fitted-model analysis
    # on every draw. Kept only when every averaged draw has one.
    containers = [record[3] if len(record) > 3 else None for record in used]
    if all(container is not None for container in containers):
        setattr(
            surface,
            _PLACEBO_DRAWS,
            {"draws": containers, "n_lost": int(n_lost) + n_excluded},
        )
    counted, _ = placebo_pretest_rate(surface, pre_labels, alpha)
    expected = _formula_pretest_rate(vcov[np.ix_(pre, pre)], alpha)
    checks = {
        "n_draws": n_used,
        "n_excluded": n_excluded,
        "n_pre": len(pre),
        "n_post": k - len(pre),
        "se_ratio": {labels[i]: float(mean_reported[i] / spread[i]) for i in pre},
        "pretest_rejection_rate": counted,
        "pretest_expected_rate": expected,
        "pretest_agrees": _pretest_size_ok(counted, expected, n_used),
    }
    return surface, checks, None


def _placebo_draw_record(surface: Any) -> Tuple[List[Any], int]:
    """The event-study containers of the placebo draws behind ``surface``.

    Returns the containers of the draws the placebo event study averaged, in
    order, and the number of no-effect replicates that are not among them (the
    fit raised, its inference was not finite, its event study could not be
    aggregated, or its layout differed from the retained one). Each container
    is what ``aggregate("event_study")`` returned for that placebo fit, so a
    fitted-model analysis can be run on every draw.

    Raises ``ValueError`` when the surface does not carry them (it was rebuilt
    or copied field by field, or was pickled by a version that did not keep
    them).
    """
    record = getattr(surface, _PLACEBO_DRAWS, None)
    if record is None:
        raise ValueError(
            "This placebo event study does not carry its placebo draws. Pass "
            "the object diagnose_design(..., event_study=True) returned (its "
            "event_study_results entry), not a rebuilt copy; a diagnosis "
            "pickled by an earlier version has to be run again."
        )
    return list(record["draws"]), int(record["n_lost"])


def _resolve_event_study(results: Any) -> Tuple[Any, Optional["DesignDiagnosisResults"], Any]:
    """Unwrap a ``DesignDiagnosisResults`` to the placebo event study it holds.

    Returns ``(surface, diagnosis, label)``. Any other input is returned
    unchanged as ``(results, None, None)``. Raises ``ValueError`` when the
    diagnosis was run without ``event_study=True``, recorded no placebo event
    study, or recorded several (one has to be named).
    """
    if not isinstance(results, DesignDiagnosisResults):
        return results, None, None
    if not results.event_study:
        raise ValueError(
            "This design diagnosis holds no placebo event study. Run "
            "diagnose_design(..., event_study=True) to record one."
        )
    surfaces = results.event_study_results
    if not surfaces:
        raise ValueError(
            "This design diagnosis recorded no placebo event study for any of "
            f"its estimators ({', '.join(results.estimators)}). One is recorded "
            "for CallawaySantAnna and StackedDiD estimators; see the warnings "
            "diagnose_design issued."
        )
    if len(surfaces) > 1:
        raise ValueError(
            "This design diagnosis holds placebo event studies for several "
            f"estimators ({', '.join(surfaces)}). Pass one of them: "
            "diagnosis.event_study_results[label]."
        )
    ((label, surface),) = surfaces.items()
    return surface, results, label


def placebo_pretest_rate(surface: Any, tested: Sequence[Any], alpha: float) -> Tuple[float, int]:
    """How often the pre-trend test rejected in the placebo draws, for ``tested``.

    The test is the one PreTrendsPower models by default: reject when any of
    the tested pre-period coefficients is individually significant at
    ``alpha`` (normal critical value), with each draw's own reported standard
    errors, as a fitted model would run it. No violation is present in any
    draw, so this is the test's false-alarm rate on this panel.

    Returns the rate and the number of draws. Raises ``ValueError`` when the
    surface does not carry its per-draw record (it was rebuilt or copied
    field by field) or ``tested`` names a period that is not in it.
    """
    record = getattr(surface, _PRETEST_RECORD, None)
    if record is None:
        raise ValueError(
            "This placebo event study carries no per-draw record, so the "
            "pre-trend test's rejection rate in the placebo draws cannot be "
            "counted. Pass the object diagnose_design(..., event_study=True) "
            "returned (its event_study_results entry), not a rebuilt copy."
        )
    position = {label: i for i, label in enumerate(record["labels"])}
    missing = [label for label in tested if label not in position]
    if missing:
        raise ValueError(
            f"Pre-periods {missing} are not in this placebo event study's "
            f"per-draw record (recorded: {record['labels']})."
        )
    columns = [position[label] for label in tested]
    critical = float(stats.norm.ppf(1.0 - alpha / 2.0))
    abs_t = record["abs_t"]
    rejected = np.any(abs_t[:, columns] > critical, axis=1)
    return float(rejected.mean()), int(abs_t.shape[0])


def _formula_pretest_rate(vcov: np.ndarray, alpha: float) -> float:
    """Rejection rate of the any-coefficient pre-test with no violation.

    The rate Roth's (2022) formula implies: the coefficients are normal with
    covariance ``vcov`` and each is compared with the normal critical value
    times its own standard deviation. This is PreTrendsPower's NIS power at a
    violation of zero.
    """
    from diff_diff.pretrends import _compute_nis_acceptance_prob

    critical = float(stats.norm.ppf(1.0 - alpha / 2.0))
    accept = _compute_nis_acceptance_prob(0.0, np.zeros(vcov.shape[0]), vcov, critical)
    return float(1.0 - accept)


def _pretest_size_ok(counted: float, expected: float, n_draws: int) -> bool:
    """Whether the counted no-violation rejection rate agrees with the expected one.

    True when the two differ by at most 1.96 Monte Carlo standard errors of a
    rate estimated from ``n_draws`` replicates at the expected rate (the form
    of ``_size_controlled``, applied in both directions: a pre-test that
    rejects less often than the formula assumes makes the planned power too
    high, one that rejects more often makes it meaningless).
    """
    if n_draws <= 0 or not np.isfinite(counted) or not np.isfinite(expected):
        return False
    tolerance = 1.96 * math.sqrt(expected * (1.0 - expected) / n_draws)
    return bool(abs(counted - expected) <= tolerance)


def _event_study_lines(diagnosis: "DesignDiagnosisResults", width: int) -> List[str]:
    """The placebo event-study block of ``DesignDiagnosisResults.summary()``."""
    lines = ["", "-" * width, "Placebo event studies (no-effect runs)".center(width), "-" * width]
    if not diagnosis._event_study_checks:
        lines.append("None recorded.")
        return lines
    lines.append(f"{'':<22}{'':>6}{'':>6}{'':>5}{'':>14}{'All pre-periods test':>25}")
    lines.append(
        f"{'Estimator':<22}{'Draws':>6}{'Excl.':>6}{'Pre':>5}{'SE ratio':>14}"
        f"{'counted':>9}{'assumed':>9}{'Agree':>7}"
    )
    for label, checks in diagnosis._event_study_checks.items():
        ratios = list(checks["se_ratio"].values())
        lines.append(
            f"{label[:21]:<22}{checks['n_draws']:>6}{checks['n_excluded']:>6}"
            f"{checks['n_pre']:>5}{f'{min(ratios):.2f} to {max(ratios):.2f}':>14}"
            f"{_fmt_rate(checks['pretest_rejection_rate']):>9}"
            f"{_fmt_rate(checks['pretest_expected_rate']):>9}"
            f"{('yes' if checks['pretest_agrees'] else 'NO'):>7}"
        )
    lines.extend(
        [
            "",
            "'All pre-periods test': how often a pre-trend test over every pre-period",
            "(any coefficient individually significant) rejected in the no-effect",
            "draws, and how often the power formula assumes it does. PreTrendsPower",
            "repeats this comparison for the pre-periods it is asked to plan and",
            "reports planned power only where the two agree;",
            "PreTrendsPower().power_by_pre_periods(...) shows it by number tested.",
            "'SE ratio': mean reported standard error of a pre-period coefficient",
            "over its spread across the draws.",
        ]
    )
    return lines


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


def _snapshot(values: Mapping[str, Any]) -> Dict[str, Any]:
    """Each value as JSON data, or a marker when it has no JSON form.

    Taken at diagnosis time and compared again in ``lock()``: the stored
    estimators and ``fit()`` keywords are live objects, and a lock must record
    the configuration that produced the numbers it reports.
    """
    snapshot: Dict[str, Any] = {}
    for name, value in values.items():
        try:
            snapshot[str(name)] = _json_value(value, str(name))
        except ValueError:
            snapshot[str(name)] = ["<not lockable>", repr(value)]
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

    Reads only the ``unit``, ``time`` and ``first_treat`` columns.
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


def _select_columns(frame: pd.DataFrame, data_columns: Sequence[str]) -> pd.DataFrame:
    """The columns a fingerprint covers, found by their names as strings."""
    by_name = {str(c): c for c in frame.columns}
    if len(by_name) != len(frame.columns):
        # Otherwise a diagnosed column could be swapped for a same-named one.
        raise ValueError(
            "data has column labels that are equal as strings (for example 1 and "
            "'1'), so the diagnosed columns cannot be identified. Rename them."
        )
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
    estimator: Any,
    outcome: str,
    unit: str,
    time: str,
    estimator_kwargs: Dict[str, Any],
    treatment: str = _TREATED_COLUMN,
    post: str = _POST_COLUMN,
    first_treat: str = _FIRST_TREAT_COLUMN,
) -> Dict[str, Any]:
    """
    The complete ``fit()`` keywords for one estimator.

    The treatment columns default to the placebo frame's; ``DesignLock.fit``
    passes the analysis frame's.
    """
    if type(estimator).__name__ in _COMMON_TIMING_ESTIMATORS:
        columns: Dict[str, Any] = {
            "outcome": outcome,
            "treatment": treatment,
            "post": post,
            "unit": unit,
        }
    else:
        columns = {
            "outcome": outcome,
            "unit": unit,
            "time": time,
            "first_treat": first_treat,
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
    capture: Optional[List[Any]] = None,
) -> Tuple[Any, _PlaceboFit]:
    """One estimator at one effect size, through ``simulate_power``."""
    name = type(estimator).__name__
    adapter = _PlaceboFit(_clone(estimator), capture)
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
    event_study: bool = False,
) -> Tuple[List[Any], List[int], bool, Optional[Any], float, Optional[List[Any]]]:
    """Every effect size, the size check and the MDE search for one estimator.

    Returns the cells, the non-finite-inference counts, whether the size
    check passed, the MDE search result (``None`` when it did not run), the
    reported MDE (NaN when not reported), and the event-study records of the
    no-effect cell (``None`` unless ``event_study`` is set and the estimator
    records a placebo event study). Warnings are re-emitted with the label.

    Only the no-effect cell captures event studies: the other effect cells and
    the MDE search build their adapters without a capture list.
    """
    cells: List[Any] = []
    nan_counts: List[int] = []
    null_adapter: Optional[_PlaceboFit] = None
    captured: Optional[List[Any]] = None
    if event_study and _event_study_skip_reason(estimator) is None:
        captured = []
    for position, effect in enumerate(effects):
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
                capture=captured if position == 0 else None,
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
    return cells, nan_counts, size_ok, found, value, captured


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
    event_study: bool = False,
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
    event_study : bool, default=False
        Also record a placebo event study for each ``CallawaySantAnna`` and
        ``StackedDiD`` estimator (not for one with ``vcov_type="jackknife"``):
        the mean and the covariance, across the no-effect replicates, of the
        event-study coefficients of each fit. The result is stored in
        ``event_study_results`` and is what
        :class:`~diff_diff.PreTrendsPower` reads to plan pre-trend power
        before any outcome exists. The record also keeps, per replicate,
        each pre-period coefficient over its own reported standard error, so
        that the pre-trend test's rejection rate with no violation can be
        counted for any set of pre-periods; ``PreTrendsPower`` reports
        planned power for a set only where that count agrees with the rate
        its formula assumes. ``summary()`` shows the comparison for the test
        over all pre-periods, and a warning is issued when it disagrees.
        The event-study container of every no-effect replicate is kept with
        the placebo event study as well (about 3 KB each);
        :meth:`~diff_diff.HonestDiD.power_by_M` runs a sensitivity analysis
        on each of them to plan its power.
        Unrelated to a ``fit()`` keyword of the same name passed through
        ``estimator_kwargs``.

    Returns
    -------
    DesignDiagnosisResults
        One :class:`~diff_diff.SimulationPowerResults` per estimator and
        effect, the size check, the minimum detectable effects, and the
        placebo event studies when requested.

    Notes
    -----
    Estimators are returned in input order and are never ranked. An estimator
    that cannot fit any of the first five placebo replicates is refused with
    its error before the run starts. After that, a replicate whose fit raises
    a data-dependent error or returns non-finite inference is excluded from
    every rate and counted; an estimator that loses more than 10% of its
    no-effect replicates fails the size check.

    A placebo event study is computed on the same draws as the comparison of
    estimators. If you compared several estimators and picked one, its placebo
    event study comes from draws that informed the pick. For pre-trend
    planning that the choice did not see, rerun ``diagnose_design`` with the
    chosen estimator alone and a new ``seed``, and use that run for pre-trend
    planning only: the lock is still taken from the comparison run, because a
    lock made from a single-estimator run records no comparison and no rerun.

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
    if not isinstance(event_study, (bool, np.bool_)):
        raise ValueError(f"event_study must be True or False, got {event_study!r}.")
    event_study = bool(event_study)

    resolved = _resolve_estimators(estimators)
    # A private copy: the caller's dict and the lists in it stay theirs.
    fit_extras = copy.deepcopy(dict(estimator_kwargs or {}))
    effects = _resolve_effect_sizes(effect_sizes)

    untreated, n_dropped = _untreated_rows(data, outcome, unit, time, first_treat)
    all_periods = np.unique(_period_values(untreated[time]))
    if len(all_periods) < _MIN_PRE_PERIODS + 1:
        raise ValueError(
            f"Too few periods: the untreated panel has {len(all_periods)} period(s); at "
            f"least {_MIN_PRE_PERIODS + 1} are needed."
        )
    declared = _validate_cohort_periods(cohort_periods, all_periods)
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
        if event_study:
            reason = _event_study_skip_reason(est)
            if reason is not None:
                warnings.warn(
                    f"[{label}] no placebo event study is recorded: {reason}.",
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
    event_study_results: Dict[str, EventStudyResults] = {}
    event_study_checks: Dict[str, Dict[str, Any]] = {}

    for label, est in resolved.items():
        cells, nan_counts, size_ok, found, value, captured = _run_estimator(
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
            event_study=event_study,
        )
        results[label] = cells
        n_nan_inference[label] = nan_counts
        size_controlled[label] = size_ok
        if found is not None:
            mde_results[label] = found
        mde[label] = value
        if captured is not None:
            surface, checks, reason = _placebo_event_study(
                est, captured, alpha, n_lost=int(cells[0].n_simulation_failures)
            )
            if surface is None or checks is None:
                warnings.warn(
                    f"[{label}] no placebo event study is recorded: {reason}.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            event_study_results[label] = surface
            event_study_checks[label] = checks
            if not checks["pretest_agrees"]:
                warnings.warn(
                    f"[{label}] a pre-trend test over all {checks['n_pre']} "
                    f"pre-periods (any coefficient individually significant at "
                    f"{alpha:g}) rejected {checks['pretest_rejection_rate']:.1%} of "
                    f"the {checks['n_draws']} no-effect replicates, against "
                    f"{checks['pretest_expected_rate']:.1%} assumed by the "
                    f"pre-trend power formula. PreTrendsPower repeats this "
                    f"comparison for the pre-periods it plans and reports planned "
                    f"power only where the two agree; a test of fewer pre-periods "
                    f"may still agree (see PreTrendsPower.power_by_pre_periods).",
                    UserWarning,
                    stacklevel=2,
                )
            if not size_ok:
                warnings.warn(
                    f"[{label}] did not pass the size check, so pre-trend power "
                    f"planned from its placebo event study may be overstated.",
                    UserWarning,
                    stacklevel=2,
                )

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
        time_periods=periods.tolist(),
        event_study=event_study,
        event_study_results=event_study_results,
        _event_study_checks=event_study_checks,
        _classes={label: type(est) for label, est in resolved.items()},
        _params={label: _snapshot(est.get_params()) for label, est in resolved.items()},
        _fit_keywords=_snapshot(fit_extras),
    )


# ---------------------------------------------------------------------------
# Design lock
# ---------------------------------------------------------------------------

_ABSENT = "<absent>"
_MATCH = "match"
_DEVIATES = "deviates"
_NOT_CHECKED = "not checked"
# Lock fields whose NaN entries (a cell that could not be scored, an MDE that
# was not reported) are stored as None.
_RECORD_FIELDS = ("comparison_results", "rerun_results", "mde")


def _size_facts(run: str, null: Any, verdict: bool) -> str:
    """One run's no-effect facts, for the warning about differing verdicts."""
    return (
        f"{run}: no-effect rejection rate {_fmt_rate(null.rejection_rate)}, "
        f"{int(null.n_simulations)} scored, {int(null.n_simulation_failures)} lost, "
        f"{'passed' if verdict else 'failed'}"
    )


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

    diagnosed_class = diagnosis._classes.get(label)
    if diagnosed_class is not None and type(chosen) is not diagnosed_class:
        raise ValueError(
            f"The estimator stored under '{label}' was changed after the diagnosis "
            f"(it is a {type(chosen).__name__}; a {diagnosed_class.__name__} was "
            f"diagnosed). Run diagnose_design again with the estimator you want to "
            f"lock."
        )
    snapshot = diagnosis._params.get(label)
    if snapshot is not None and _snapshot(chosen.get_params()) != snapshot:
        raise ValueError(
            f"The estimator stored under '{label}' was changed after the diagnosis "
            f"(its parameters no longer match the ones that were diagnosed). Run "
            f"diagnose_design again with the configuration you want to lock."
        )
    if (
        diagnosis._fit_keywords is not None
        and _snapshot(diagnosis.estimator_kwargs) != diagnosis._fit_keywords
    ):
        raise ValueError(
            "estimator_kwargs was changed after the diagnosis (it no longer matches "
            "the fit() keywords that were diagnosed). Run diagnose_design again with "
            "the configuration you want to lock."
        )
    problems = _unlockable(chosen, diagnosis.estimator_kwargs)
    if problems:
        raise ValueError(
            f"Estimator '{label}' cannot be locked: {'; '.join(problems)}. A lock "
            f"must be able to rebuild the estimator exactly."
        )

    outcome, unit, time = diagnosis.outcome, diagnosis.unit, diagnosis.time
    # Several estimators were compared, so one was chosen: rerun it alone on
    # draws the choice did not see. With one estimator nothing was chosen.
    rerun = len(labels) > 1
    if rerun and data is None:
        raise ValueError(
            f"{len(labels)} estimators were diagnosed, so lock() reruns '{label}' "
            f"alone with new placebo draws and needs the panel: pass data= (the "
            f"frame given to diagnose_design)."
        )

    time_periods = list(diagnosis.time_periods)
    compared_ok = diagnosis.size_controlled[label]
    size_ok = compared_ok
    mde = diagnosis.mde[label]
    rerun_records: List[Dict[str, Any]] = []
    rerun_null: Any = None

    if data is not None:
        untreated, _ = _untreated_rows(data, outcome, unit, time, diagnosis.first_treat)
        untreated = _select_columns(untreated, diagnosis.data_columns)
        found_periods = np.unique(_period_values(untreated[time])).tolist()
        if found_periods != time_periods:
            missing = [period for period in time_periods if period not in found_periods]
            other = [period for period in found_periods if period not in time_periods]
            raise ValueError(
                f"data does not have the periods of the diagnosed panel. Diagnosed "
                f"periods missing: {missing}; other periods found: {other}."
            )
        digest, _ = _data_fingerprint(untreated, unit, time)
        if digest != diagnosis.data_sha256:
            raise ValueError(
                "data is not the panel that was diagnosed: its fingerprint differs "
                "from data_sha256."
            )

        if rerun:
            if diagnosis._rerun_labels and label not in diagnosis._rerun_labels:
                warnings.warn(
                    f"A rerun was already made for {diagnosis._rerun_labels}. A rerun "
                    f"for '{label}' as well makes the two runs a second comparison: "
                    f"do not use them to choose between estimators.",
                    UserWarning,
                    stacklevel=3,
                )
            panel = _check_panel(untreated, outcome, unit, time)
            generator = _make_placebo_generator(
                panel,
                outcome,
                unit,
                time,
                diagnosis.n_treated,
                list(diagnosis.cohort_periods),
                diagnosis.resample_units,
            )
            fit_kwargs = _fit_kwargs(chosen, outcome, unit, time, dict(diagnosis.estimator_kwargs))
            # Derived, not an argument: the rerun cannot be re-rolled.
            lock_seed = int(np.random.SeedSequence([int(diagnosis.seed), 1]).generate_state(1)[0])
            null = diagnosis.results[label][0]
            n_simulations = int(null.n_simulations) + int(null.n_simulation_failures)
            with _WarningCapture() as capture:
                _preflight(label, chosen, generator, fit_kwargs, lock_seed)
            _reemit(label, capture.records, stacklevel=4)
            cells, nan_counts, size_ok, _found, mde, _captured = _run_estimator(
                "lock",
                label,
                chosen,
                diagnosis.effect_sizes,
                generator,
                fit_kwargs,
                diagnosis.n_units,
                diagnosis.n_periods,
                n_simulations,
                diagnosis.alpha,
                diagnosis.target_power,
                lock_seed,
                progress,
                stacklevel=5,
            )
            rerun_records = _records(label, cells, size_ok, nan_counts, mde)
            rerun_null = cells[0]
            if label not in diagnosis._rerun_labels:
                diagnosis._rerun_labels.append(label)

    if rerun_null is not None and size_ok != compared_ok:
        # Both runs' facts and no conclusion about the cause: the size check
        # can fail on lost replicates as well as on the rejection rate.
        meaning = "" if size_ok else " Its power is not meaningful."
        warnings.warn(
            f"The size check gives different verdicts for '{label}'. "
            f"{_size_facts('Comparison', diagnosis.results[label][0], compared_ok)}; "
            f"{_size_facts('rerun', rerun_null, size_ok)}. The lock records the "
            f"rerun's verdict, size_controlled={size_ok}.{meaning}",
            UserWarning,
            stacklevel=3,
        )
    elif not size_ok:
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
        "n_units": diagnosis.n_units,
        "n_periods": diagnosis.n_periods,
        "n_obs": diagnosis.n_obs,
        "time_periods": time_periods,
        "data_sha256": diagnosis.data_sha256,
        "data_columns": diagnosis.data_columns,
        "comparison_results": diagnosis.to_dataframe().to_dict("records"),
        "rerun_results": rerun_records,
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
    A recorded design: one estimator, its configuration and its diagnosis.

    Made by :meth:`DesignDiagnosisResults.lock`. It records the chosen
    estimator and its configuration, the declared design, a fingerprint of
    the panel, every estimator that was compared, and the chosen estimator's
    size, power, bias, RMSE, coverage and minimum detectable effect. All
    values are plain JSON data, so ``DesignLock.from_json(lock.to_json())``
    equals ``lock``.

    The lock does not record when it was made. Whether a design was fixed
    before the outcomes were seen depends on when the lock file was saved or
    registered, which the library cannot check.

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
        Seed of the comparison. The rerun uses a seed derived from it.
    n_units, n_periods, n_obs : int
        Shape of the diagnosed panel.
    time_periods : list
        Its period values.
    data_sha256 : str
        Its fingerprint.
    data_columns : list of str
        Columns the fingerprint covers.
    comparison_results : list of dict
        One record per estimator and effect from the comparison, for every
        estimator that was compared.
    rerun_results : list of dict
        The chosen estimator's records from the rerun (the chosen estimator
        alone, new placebo draws, same panel). Empty when one estimator was
        diagnosed: nothing was chosen, so there was no rerun.
    size_controlled : bool
        Whether the chosen estimator passed the size check in the run the
        lock reports (the rerun when there was one). The comparison's
        verdict is in ``comparison_results``.
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
    n_units: int
    n_periods: int
    n_obs: int
    time_periods: List[Any]
    data_sha256: str
    data_columns: List[str]
    comparison_results: List[Dict[str, Any]]
    rerun_results: List[Dict[str, Any]]
    size_controlled: bool
    mde: Optional[float]
    sha256: str

    def __repr__(self) -> str:
        """Concise string representation."""
        return (
            f"DesignLock(label={self.label!r}, estimator_name={self.estimator_name!r}, "
            f"rerun={bool(self.rerun_results)}, sha256={self.sha256[:12]!r}...)"
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
            ``rerun_results`` when a rerun was made, otherwise the chosen
            estimator's rows of ``comparison_results``.
        """
        if self.rerun_results:
            return pd.DataFrame(copy.deepcopy(self.rerun_results))
        return pd.DataFrame(copy.deepcopy(self._compared_rows()))

    def _compared_rows(self) -> List[Dict[str, Any]]:
        """The chosen estimator's rows of the comparison."""
        return [row for row in self.comparison_results if row.get("label") == self.label]

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
            "rerun of the chosen estimator (new placebo draws, same panel)"
            if self.rerun_results
            else "comparison run (one estimator, nothing was chosen)"
        )
        size_text = "yes" if self.size_controlled else "NO"
        compared_rows = self._compared_rows()
        if self.rerun_results and compared_rows:
            compared_ok = bool(compared_rows[0].get("size_controlled"))
            if compared_ok != self.size_controlled:
                size_text += f" (rerun); comparison: {'yes' if compared_ok else 'NO'}"
        compared = list(dict.fromkeys(str(row.get("label")) for row in self.comparison_results))
        mde_text = f"{self.mde:.4g}" if self.mde is not None else "-"
        lines = [
            "=" * width,
            "Design Lock".center(width),
            "=" * width,
            "",
            f"{'Chosen:':<32} {self.label} ({self.estimator_name})",
            f"{'Compared:':<32} {', '.join(compared)}",
            f"{'Units x periods:':<32} {self.n_units} x {self.n_periods}",
            f"{'Placebo-treated units:':<32} {self.n_treated}",
            f"{'Adoption periods (' + timing + '):':<32} {list(self.cohort_periods)}",
            f"{'Significance level (alpha):':<32} {self.alpha:.3f}",
            f"{'Reported numbers from:':<32} {source}",
            f"{'Size OK:':<32} {size_text}",
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
                *_SCOPE_LINES,
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
            ``params``. Fitting it, with ``estimator_kwargs``, is the locked
            analysis.
        """
        self._verify()
        if self.estimator_name not in _SUPPORTED_ESTIMATORS:
            raise ValueError(
                f"The lock names {self.estimator_name!r}, which is not one of "
                f"{', '.join(_SUPPORTED_ESTIMATORS)}."
            )
        import diff_diff

        return getattr(diff_diff, self.estimator_name)(**copy.deepcopy(self.params))

    def fit(
        self,
        data: pd.DataFrame,
        *,
        first_treat: Optional[str] = None,
        treatment: Optional[str] = None,
        post: Optional[str] = None,
    ) -> "DesignReconciliationResults":
        """
        Fit the locked analysis and reconcile it against this lock.

        Builds the locked estimator (:meth:`to_estimator`), fits it on
        ``data`` with the locked outcome, unit and time columns and the
        locked ``estimator_kwargs``, and runs the checks of
        :meth:`reconcile` on that same frame.

        Parameters
        ----------
        data : pd.DataFrame
            The analysis data.
        first_treat : str, optional
            Column with each unit's real first treatment period (0, inf or
            NaN for never-treated). Required for the estimators that take
            one, refused for the others.
        treatment : str, optional
            0/1 group column. Required, with ``post``, for
            ``DifferenceInDifferences`` and ``TwoWayFixedEffects``; refused
            for the others.
        post : str, optional
            0/1 post-period column that goes with ``treatment``.

        Returns
        -------
        DesignReconciliationResults
            The checks, with the fitted results in ``results`` and
            ``declared=False``.

        Notes
        -----
        The estimator class, its parameters, the outcome, unit and time
        columns and the ``fit()`` keywords are the locked ones by
        construction, and the data checks run on the frame that was fitted.
        The realized design can still deviate (unit count, number of treated
        units, number of cohorts, post-window lengths, the fingerprint of the
        diagnosed periods). Nothing here says when the lock was made.

        The column arguments name columns of the analysis data and are not
        taken from the lock: the lock's own ``first_treat`` is the column
        that truncated the diagnosed panel.

        An exception raised by the estimator's ``fit()`` propagates, and no
        reconciliation is returned.
        """
        estimator = self.to_estimator()
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        columns: Dict[str, str]
        if self.estimator_name in _COMMON_TIMING_ESTIMATORS:
            if treatment is None or post is None or first_treat is not None:
                raise ValueError(
                    f"{self.estimator_name} is fitted with a 0/1 group column and a "
                    f"0/1 post-period column: pass treatment= and post=, not "
                    f"first_treat=."
                )
            columns = {"treatment": treatment, "post": post}
        else:
            if first_treat is None or treatment is not None or post is not None:
                raise ValueError(
                    f"{self.estimator_name} is fitted with a first-treatment-period "
                    f"column: pass first_treat=, not treatment= or post=."
                )
            columns = {"first_treat": first_treat}
        keywords = _fit_kwargs(
            estimator,
            self.outcome,
            self.unit,
            self.time,
            copy.deepcopy(self.estimator_kwargs),
            **columns,
        )
        results = estimator.fit(data, **keywords)
        return self._reconcile(
            estimator,
            outcome=self.outcome,
            unit=self.unit,
            time=self.time,
            first_treat=first_treat,
            treatment=treatment,
            post=post,
            estimator_kwargs=self.estimator_kwargs,
            data=data,
            results=results,
            declared=False,
        )

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
            One check per item and an overall ``status``: ``"match"`` when
            every check that was run matches, otherwise ``"deviates"``.

        Notes
        -----
        This compares what the analyst declares with the lock. Fitted results
        do not record the constructor parameters or column names that
        produced them, so a parameter changed with ``set_params()`` after
        fitting cannot be detected. :meth:`fit` runs the locked analysis
        itself and does not rely on a declaration. ``"match"`` covers the
        listed checks that were run, and says nothing about when the design
        was fixed.

        The data fingerprint covers every column the diagnosed panel had, on
        the locked periods. A column whose values changed there deviates;
        that includes an adoption column that was all never-treated when the
        design was diagnosed and holds the real dates now.

        Adoption dates are not compared: placebo dates lie inside the
        diagnosed panel and real ones after it. The number of cohorts, the
        number of treated units, the length of each cohort's post window and
        whether the panel is balanced are.
        """
        return self._reconcile(
            estimator,
            outcome=outcome,
            unit=unit,
            time=time,
            first_treat=first_treat,
            treatment=treatment,
            post=post,
            estimator_kwargs=estimator_kwargs,
            data=data,
            results=results,
            declared=True,
        )

    def _reconcile(
        self,
        estimator: Any,
        *,
        outcome: str,
        unit: str,
        time: str,
        first_treat: Optional[str],
        treatment: Optional[str],
        post: Optional[str],
        estimator_kwargs: Optional[Dict[str, Any]],
        data: Optional[pd.DataFrame],
        results: Optional[Any],
        declared: bool,
    ) -> "DesignReconciliationResults":
        """The checks behind :meth:`reconcile` and :meth:`fit`."""
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
            status=_DEVIATES if deviates else _MATCH,
            checks=checks,
            label=self.label,
            lock_sha256=self.sha256,
            results=results,
            declared=declared,
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

        requested = timing if (first_treat is not None or treatment is not None) else ()
        try:
            _realized_design(
                data, unit, time, first_treat, treatment, post, requested, realized, problems
            )
        except (KeyError, TypeError, ValueError) as exc:
            # reconcile() does not raise because of what the analysis frame
            # looks like: a check that could not be read is a deviation.
            for item in ("n_units", "is_balanced") + requested:
                if item not in realized:
                    problems.setdefault(item, f"could not be read: {exc}")

    rows_out = []
    for item, value in locked.items():
        if item in problems:
            rows_out.append(_check_row(item, value, problems[item], _DEVIATES))
        elif item in realized:
            rows_out.append(_compare(item, value, realized[item]))
        else:
            rows_out.append(_check_row(item, value, None, _NOT_CHECKED))
    return rows_out


def _column_problem(data: pd.DataFrame, column: Any) -> Optional[str]:
    """Why ``column`` cannot be read from ``data``, or None when it can."""
    count = sum(1 for label in data.columns if label is column or label == column)
    if count == 0:
        return f"column {column!r} not in data"
    if count > 1:
        return f"column label {column!r} appears {count} times in data"
    return None


def _realized_design(
    data: pd.DataFrame,
    unit: str,
    time: str,
    first_treat: Optional[str],
    treatment: Optional[str],
    post: Optional[str],
    requested: Tuple[str, ...],
    realized: Dict[str, Any],
    problems: Dict[str, str],
) -> None:
    """Read the realized design from the analysis frame into ``realized``.

    Every column that is read must be present exactly once and, for the unit
    and period identifiers, complete: a row whose unit or period is missing
    cannot be placed, so the unit count, the balance and the timing are then
    reported as deviations and never as matches.
    """
    panel_items = ("n_units", "is_balanced") + requested
    unreadable = [text for text in (_column_problem(data, c) for c in (unit, time)) if text]
    if unreadable:
        for item in panel_items:
            problems[item] = "; ".join(unreadable)
        return
    incomplete = [column for column in (unit, time) if data[column].isna().any()]
    if incomplete:
        for item in panel_items:
            problems[item] = f"missing values in column(s) {incomplete}"
        return

    n_units = int(data[unit].nunique())
    periods = pd.unique(data[time])
    realized["n_units"] = n_units
    realized["is_balanced"] = bool(
        not data.duplicated([unit, time]).any() and len(data) == n_units * len(periods)
    )
    if first_treat is not None:
        problem, values = _realized_first_treat(data, unit, first_treat)
        if problem is not None:
            for item in requested:
                problems[item] = problem
            return
        realized_cohorts = sorted(set(values))
        realized["n_treated"] = len(values)
        realized["n_cohorts"] = len(realized_cohorts)
        try:
            realized["n_post_by_cohort"] = [
                int(np.sum(periods >= cohort)) for cohort in realized_cohorts
            ]
        except TypeError:
            problems["n_post_by_cohort"] = f"'{time}' cannot be compared with '{first_treat}'"
    elif treatment is not None:
        problem = _column_problem(data, treatment)
        if problem is not None:
            problems["n_treated"] = problems["n_cohorts"] = problem
        elif (data.groupby(unit)[treatment].nunique(dropna=False) > 1).any():
            problems["n_treated"] = problems["n_cohorts"] = "varies within unit"
        else:
            n_treated = int((data.groupby(unit)[treatment].first() == 1).sum())
            realized["n_treated"] = n_treated
            realized["n_cohorts"] = 1 if n_treated else 0
        if post is not None:
            problem = _column_problem(data, post)
            if problem is not None:
                problems["n_post_by_cohort"] = problem
            elif (data.groupby(time)[post].nunique(dropna=False) > 1).any():
                problems["n_post_by_cohort"] = "varies within period"
            else:
                realized["n_post_by_cohort"] = [int((data.groupby(time)[post].first() == 1).sum())]


def _realized_first_treat(
    data: pd.DataFrame, unit: str, first_treat: str
) -> Tuple[Optional[str], List[float]]:
    """Each treated unit's adoption period, or why it cannot be read."""
    problem = _column_problem(data, first_treat)
    if problem is not None:
        return problem, []
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
    Results of :meth:`DesignLock.reconcile` and :meth:`DesignLock.fit`.

    Attributes
    ----------
    status : str
        ``"match"`` when every check that was run matches the lock,
        otherwise ``"deviates"``. The size of a deviation is not graded, and
        checks whose inputs were not passed are ``"not checked"``.
    checks : list of dict
        One entry per reconciled item, with keys ``item``, ``locked``,
        ``realized`` and ``status`` (``"match"``, ``"deviates"`` or
        ``"not checked"``).
    label : str
        Label of the locked estimator.
    lock_sha256 : str
        ``sha256`` of the lock that was reconciled against.
    results : results object or None
        The fitted results the checks were made for: the fit made by
        :meth:`DesignLock.fit`, or the ``results`` passed to
        :meth:`DesignLock.reconcile` (``None`` when none was passed). Not
        part of ``to_dict()``.
    declared : bool
        ``True`` when the estimator, the column names and the ``fit()``
        keywords were declared to :meth:`DesignLock.reconcile`; ``False``
        when :meth:`DesignLock.fit` ran the analysis itself.
    """

    status: str
    checks: List[Dict[str, Any]]
    label: str
    lock_sha256: str
    results: Optional[Any] = field(default=None, repr=False, compare=False)
    declared: bool = True

    def __repr__(self) -> str:
        """Concise string representation."""
        n_deviating = sum(1 for check in self.checks if check["status"] == _DEVIATES)
        n_skipped = sum(1 for check in self.checks if check["status"] == _NOT_CHECKED)
        return (
            f"DesignReconciliationResults(status={self.status!r}, "
            f"checks={len(self.checks)}, deviating={n_deviating}, "
            f"not_checked={n_skipped}, declared={self.declared})"
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
            The overall status, the lock reconciled against, whether the
            configuration was declared, and every check. ``results`` is not
            included.
        """
        return {
            "status": self.status,
            "declared": self.declared,
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
        # The status is never shown without the checks that were not run.
        skipped = f" - {counts[_NOT_CHECKED]} not checked" if counts[_NOT_CHECKED] else ""
        lines = [
            "=" * width,
            "Design Reconciliation (analysis vs. design lock)".center(width),
            "=" * width,
            "",
            f"{'Locked estimator:':<24} {self.label}",
            f"{'Status:':<24} {self.status}{skipped}",
            f"{'Configuration:':<24} "
            f"{'declared to reconcile()' if self.declared else 'fitted from the lock'}",
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
        lines.extend(["", "'match' means every check that was run agrees with the lock. It covers"])
        if self.declared:
            lines.extend(
                [
                    "what was declared to reconcile() and the listed items only; it does not",
                    "say when the design was fixed, and deviations are not graded.",
                ]
            )
        else:
            lines.extend(
                [
                    "the listed items only; it does not say when the design was fixed, and",
                    "deviations are not graded.",
                ]
            )
        lines.append("=" * width)
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print the summary to stdout."""
        print(self.summary())
