"""Duration DiD under common dynamics or proportional untreated hazards.

Deaner and Ku (2026), Sections 2--3 and Appendix B. The committed methodology
review resolves the printed PH ambiguity in favor of the mean of ratios and
pins the fixed-anchor diagnostic. No author-software parity is claimed.
"""

from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff._base import BaseEstimator
from diff_diff.duration_did_results import (
    _CONTRAST_COLUMNS,
    _EFFECT_COLUMNS,
    _FAILURE_COLUMNS,
    _SURVIVAL_COLUMNS,
    DurationDiDPretestResults,
    DurationDiDResults,
)
from diff_diff.utils import safe_inference, validate_n_bootstrap


class _DomainError(ValueError):
    def __init__(self, reasons: Dict[Optional[int], str]):
        self.reasons = reasons
        super().__init__(str(reasons))


def _survival(y: np.ndarray, group: np.ndarray) -> np.ndarray:
    if not (np.any(group == 0) and np.any(group == 1)):
        raise _DomainError({None: "resample lost a treatment group"})
    return np.array([1.0 - y[group == k].mean(axis=0) for k in (0, 1)])


def _moments(s: np.ndarray, method: str) -> Tuple[np.ndarray, np.ndarray]:
    # Evaluate only defined logs. In particular, factual treated post-period
    # extinction is not a log-domain failure for the effect estimator.
    r = np.full_like(s, np.nan)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.log(s, out=r, where=s > 0)
        r = -r
        d = r - r[:, :1]
        moment = np.full(s.shape[1], np.nan)
        if method == "common_dynamics":
            moment[1:] = (d[1, 1:] - d[0, 1:]) / np.arange(1, s.shape[1])
        else:
            np.divide(d[1], d[0], out=moment, where=d[0] > 0)
    return moment, d


def _curve(
    s: np.ndarray, method: str, support: np.ndarray, weights: np.ndarray, start: int
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
    if np.any(s[:, 0] <= 0):
        raise _DomainError({0: "both groups need positive baseline survival"})
    moment, d = _moments(s, method)
    bad = {
        int(i): "unsupported calibration moment (required survival or PH control increment)"
        for i in support
        if not np.isfinite(moment[i])
    }
    bad.update(
        {
            i: "positive control survival required for extrapolation"
            for i in range(start, s.shape[1])
            if s[0, i] <= 0
        }
    )
    if bad:
        raise _DomainError(bad)
    coefficient = float(np.dot(weights, moment[support]))
    if not np.isfinite(coefficient) or (method == "proportional_hazards" and coefficient <= 0):
        raise _DomainError({None: "fitted coefficient is unidentified or PH ratio is not positive"})
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        r_cf = -np.log(s[1, 0]) + (
            d[0, start:] + np.arange(start, s.shape[1]) * coefficient
            if method == "common_dynamics"
            else coefficient * d[0, start:]
        )
        s_cf = np.exp(-r_cf)
        att = s_cf - s[1, start:]
    previous = np.r_[s[1, start - 1], s_cf[:-1]]
    invalid = {}
    for j, value in enumerate(s_cf):
        reasons = []
        if not np.isfinite(value) or value < -1e-12 or value > 1 + 1e-12:
            reasons.append("counterfactual survival outside finite probability bounds")
        if value > previous[j] + 1e-12:
            reasons.append("counterfactual survival increases from preceding survival")
        if reasons:
            invalid[start + j] = "; ".join(reasons)
    return coefficient, r_cf, s_cf, att, invalid


def _contrasts(
    s: np.ndarray, method: str, start: int
) -> Tuple[np.ndarray, Dict[Optional[int], str]]:
    if start < 3:
        return np.empty(0), {None: "at least three pre-periods required for a hazard pretest"}
    moments, _ = _moments(s, method)
    with np.errstate(invalid="ignore"):
        values = moments[1 : start - 1] - moments[start - 1]
    bad = {
        i: "unsupported hazard diagnostic moment or anchor"
        for i in range(1, start)
        if not np.isfinite(moments[i])
    }
    return values, bad


def _quantile(values: np.ndarray, alpha: float) -> Any:
    """Inverse empirical CDF, without interpolation or a finite-B adjustment."""
    return np.sort(values, axis=0)[int(np.ceil((1 - alpha) * len(values))) - 1]


def _inference(theta: np.ndarray, draws: np.ndarray, alpha: float) -> Dict[str, Any]:
    """Algorithm 1: SD, centered absolute tails and studentized maxima."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        se = np.std(draws, axis=0, ddof=1)
        centered = np.abs(draws - theta)
    result: Dict[str, Any] = {
        key: np.full(len(theta), np.nan)
        for key in (
            "t_stat",
            "p_value",
            "conf_int_lower",
            "conf_int_upper",
            "pointwise_crit_value",
            "cband_lower",
            "cband_upper",
        )
    }
    result["se"] = se
    result["cband_crit_value"] = np.nan
    result["statistic"] = np.nan
    result["simultaneous_p_value"] = np.nan
    good = np.isfinite(se) & (se > 0) & np.isfinite(theta)
    for j in range(len(theta)):
        stat, p, interval = safe_inference(theta[j], se[j], alpha=alpha)
        if good[j]:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                crit = float(_quantile(centered[:, j] / se[j], alpha))
                width = crit * se[j]
            if not (np.isfinite(stat) and np.isfinite(crit) and np.isfinite(width)):
                good[j] = False
                continue
            p = float(np.mean(centered[:, j] >= abs(theta[j])))
            interval = (theta[j] - width, theta[j] + width)
            result["pointwise_crit_value"][j] = crit
            result["t_stat"][j] = stat
            result["p_value"][j] = p
            result["conf_int_lower"][j], result["conf_int_upper"][j] = interval
    if len(theta) and np.all(good):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            maxima = np.max(centered / se, axis=1)
            crit = float(_quantile(maxima, alpha))
            statistic = float(np.max(np.abs(result["t_stat"])))
            lower, upper = theta - crit * se, theta + crit * se
        if np.all(np.isfinite(lower)) and np.all(np.isfinite(upper)):
            result.update(
                cband_lower=lower,
                cband_upper=upper,
                cband_crit_value=crit,
                statistic=statistic,
                simultaneous_p_value=float(np.mean(maxima >= statistic)),
            )
    return result


def _selectors(values: Any, periods: List[Any], name: str) -> List[Any]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of observed dates")
    try:
        selected = list(values)
        if len(selected) != len(set(selected)) or not set(selected).issubset(periods):
            raise ValueError(f"{name} contains duplicate or unknown dates")
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of observed dates") from exc
    return [t for t in periods if t in selected]


def _panel(
    data: pd.DataFrame, outcome: str, treatment: str, unit: str, time: str
) -> Tuple[np.ndarray, np.ndarray, List[Any], Any]:
    cols = [outcome, treatment, unit, time]
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("data must be a nonempty DataFrame")
    if len(set(cols)) != 4 or any(col not in data.columns for col in cols):
        raise ValueError("outcome, treatment, unit and time must name distinct existing columns")
    if data[cols].isna().any().any():
        raise ValueError("panel columns must not contain missing values")
    if data.duplicated([unit, time]).any():
        raise ValueError("duplicate (unit, time) observations are not supported")
    for col in (outcome, treatment):
        if not data[col].isin([0, 1]).all():
            raise ValueError(f"{col} must be finite binary 0/1")
    clock = pd.Index(data[time].unique())
    try:
        if isinstance(clock, (pd.DatetimeIndex, pd.TimedeltaIndex)):
            clock = clock.sort_values()
            differences = [clock[i] - clock[i - 1] for i in range(1, len(clock))]
            if (
                not differences
                or differences[0] <= pd.Timedelta(0)
                or any(x != differences[0] for x in differences)
            ):
                raise ValueError("datetime/timedelta dates must have equal actual spacing")
            step = differences[0]
        else:
            if any(
                isinstance(t, (bool, np.bool_))
                or not isinstance(t, (int, float, np.integer, np.floating))
                for t in clock
            ):
                raise ValueError("time labels must be numeric, Timestamp, or Timedelta")
            clock = clock.sort_values()
            if not np.isfinite(np.asarray(clock, dtype=float)).all() or len(clock) < 2:
                raise ValueError("time requires at least two finite dates")
            differences = np.array([float(clock[i] - clock[i - 1]) for i in range(1, len(clock))])
            step = float(differences[0])
            if (
                step <= 0
                or not np.isfinite(step)
                or not np.allclose(differences, step, rtol=1e-9, atol=abs(step) * 1e-12)
            ):
                raise ValueError("numeric dates must have equal spacing")
    except (TypeError, OverflowError) as exc:
        raise ValueError("unsupported or irregular observation clock") from exc
    periods = list(clock)
    codes, units = pd.factorize(data[unit], sort=False)
    if len(data) != len(units) * len(periods):
        raise ValueError("a balanced complete individual panel is required; dropout is unsupported")
    times = clock.get_indexer(data[time])
    y = np.empty((len(units), len(periods)), dtype=float)
    memberships = np.empty_like(y)
    y[codes, times] = data[outcome].to_numpy(dtype=float)
    memberships[codes, times] = data[treatment].to_numpy(dtype=float)
    if np.any(memberships != memberships[:, :1]):
        raise ValueError("treatment must be constant group membership within each individual")
    if np.any(np.diff(y, axis=1) < 0):
        raise ValueError("outcome must be absorbing: no transition from 1 back to 0")
    group = memberships[:, 0]
    if len(np.unique(group)) != 2:
        raise ValueError("both treated and control individuals are required")
    return y, group, periods, step


class DurationDiD(BaseEstimator):
    """Two-group, common-timing Duration DiD (Deaner and Ku, 2026).

    Parameters
    ----------
    method : {"common_dynamics", "proportional_hazards"}, default "common_dynamics"
        Additive or proportional restriction on untreated hazards.
    n_bootstrap : int, default 1000
        Number of pooled individual history draws, at least two. No retries.
    alpha : float, default 0.05
        Significance level used for all stored intervals and hazard diagnostics.
    seed : int or None, default None
        Nonnegative seed for numpy's Generator.

    Notes
    -----
    Supports complete balanced individual panels only. No covariates, survey
    weights, censoring, staggered adoption, or higher-level clustering.
    Bootstrap failures suppress inference for the affected family, not valid
    original point estimates. See the methodology Registry for domain policy.
    """

    def __init__(
        self,
        method: str = "common_dynamics",
        n_bootstrap: int = 1000,
        alpha: float = 0.05,
        seed: Optional[int] = None,
    ):
        self.method = method
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.seed = seed
        self._validate_params()
        self.results_: Optional[DurationDiDResults] = None
        self.is_fitted_ = False

    def _validate_params(self) -> None:
        """Apply the constructor contract again before fitting mutable parameters."""
        if self.method not in ("common_dynamics", "proportional_hazards"):
            raise ValueError("method must be 'common_dynamics' or 'proportional_hazards'")
        validate_n_bootstrap(self.n_bootstrap)
        if self.n_bootstrap < 2:
            raise ValueError("n_bootstrap must be >= 2 for DurationDiD")
        if (
            isinstance(self.alpha, (bool, np.bool_))
            or not isinstance(self.alpha, (int, float, np.integer, np.floating))
            or not np.isfinite(self.alpha)
            or not 0 < self.alpha < 1
        ):
            raise ValueError("alpha must be finite and between 0 and 1")
        if self.seed is not None and (
            isinstance(self.seed, (bool, np.bool_))
            or not isinstance(self.seed, (int, np.integer))
            or self.seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer or None")

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        unit: str,
        time: str,
        *,
        post_periods: Any,
        fit_periods: Any = None,
        time_weights: Any = None,
    ) -> DurationDiDResults:
        """Estimate cumulative absorption ATT and fixed-anchor hazard diagnostics.

        Parameters
        ----------
        data : pandas.DataFrame
            Complete long individual panel; rows may be in any order.
        outcome : str
            Absorbing binary outcome column (1 means already absorbed).
        treatment : str
            Fixed binary group indicator, not a time-varying exposure column.
        unit, time : str
            Individual identifier and equally spaced observation clock columns.
        post_periods : sequence
            Required nonempty observed suffix beginning at first treatment.
        fit_periods : sequence or None
            Calibration dates after baseline and before treatment. None selects
            all original-sample eligible dates. Diagnostic dates remain fixed.
        time_weights : mapping or None
            Nonnegative calibration weights keyed by the selected dates. Zero
            weights remove moments before evaluation. Not observation weights.

        Returns
        -------
        DurationDiDResults
            Stored effects, raw survival paths, bootstrap and diagnostic metadata.

        Raises
        ------
        ValueError
            If constructor parameters or the panel/calibration inputs are invalid.
            Parameters are revalidated on every call, including direct attribute
            updates. Rejected parameters leave any previous fitted result intact.
        """
        self._validate_params()
        y, group, periods, step = _panel(data, outcome, treatment, unit, time)
        post = _selectors(post_periods, periods, "post_periods")
        if not post:
            raise ValueError("post_periods must be a nonempty observed suffix")
        start = periods.index(post[0])
        if post != periods[start:] or start < 2:
            raise ValueError("post_periods must be a suffix leaving at least two pre-periods")
        s = _survival(y, group)
        if np.any(s[:, 0] <= 0):
            raise ValueError(
                f"positive baseline survival required in both groups at {periods[0]!r}"
            )
        moments, _ = _moments(s, self.method)
        excluded = {}
        requested = None if fit_periods is None else _selectors(fit_periods, periods, "fit_periods")
        if requested is not None and (
            not requested or any(t not in periods[1:start] for t in requested)
        ):
            raise ValueError(
                "fit_periods must select dates strictly after baseline and before treatment"
            )
        selected = []
        for t in periods[1:start] if requested is None else requested:
            if requested is None and not np.isfinite(moments[periods.index(t)]):
                excluded[t] = "unsupported original-sample calibration moment"
            else:
                selected.append(t)
        if time_weights is None:
            w = np.ones(len(selected))
        else:
            if not isinstance(time_weights, Mapping) or set(time_weights) != set(selected):
                raise ValueError(
                    "time_weights keys must match selected fit_periods or default eligible dates"
                )
            try:
                w = np.array([time_weights[t] for t in selected], dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError("time_weights must contain finite nonnegative weights") from exc
        if not len(w) or not np.isfinite(w).all() or np.any(w < 0) or not np.any(w > 0):
            raise ValueError("calibration requires finite nonnegative weights with positive total")
        # Scale before summing so finite large weights cannot overflow the total.
        originally_positive = w > 0
        w = w / np.max(w)
        if np.any(originally_positive & (w == 0)):
            raise ValueError("positive calibration weight underflows during normalization")
        for t, weight in zip(selected, w):
            if weight == 0:
                excluded[t] = "zero calibration weight"
        positive = w > 0
        selected = [t for t, keep in zip(selected, positive) if keep]
        w = w[positive] / w[positive].sum()
        if np.any(w == 0):
            raise ValueError("positive calibration weight underflows during normalization")
        support = np.array([periods.index(t) for t in selected], dtype=int)
        try:
            coefficient, r_cf, s_cf, raw, invalid = _curve(s, self.method, support, w, start)
        except _DomainError as exc:
            raise ValueError(
                "; ".join(
                    f"{periods[i] if i is not None else 'fit'}: {reason}"
                    for i, reason in exc.reasons.items()
                )
            ) from exc
        contrasts, diagnostic_errors = _contrasts(s, self.method, start)
        b = self.n_bootstrap
        draws = np.full((b, len(post)), np.nan)
        diagnostic_draws = np.full((b, len(contrasts)), np.nan)
        failures = []

        def record(draw: int, family: str, errors: Dict[Any, str]) -> None:
            failures.extend(
                {
                    "draw": draw,
                    "family": family,
                    "period": None if i is None else periods[i],
                    "reason": reason,
                }
                for i, reason in errors.items()
            )

        rng = np.random.default_rng(self.seed)
        for draw in range(b):
            indices = rng.integers(0, len(y), size=len(y))
            try:
                sampled = _survival(y[indices], group[indices])
            except _DomainError as exc:
                record(draw, "effects", exc.reasons)
                if not diagnostic_errors:
                    record(draw, "diagnostics", exc.reasons)
                continue
            try:
                _, _, _, effect, errors = _curve(sampled, self.method, support, w, start)
                if errors:
                    record(draw, "effects", errors)
                else:
                    draws[draw] = effect
            except _DomainError as exc:
                record(draw, "effects", exc.reasons)
            if not diagnostic_errors:
                delta, errors = _contrasts(sampled, self.method, start)
                if errors:
                    record(draw, "diagnostics", errors)
                else:
                    diagnostic_draws[draw] = delta
        failure_table = pd.DataFrame(failures, columns=_FAILURE_COLUMNS)
        failure_table["period"] = pd.Series([item["period"] for item in failures], dtype=object)
        valid = int(np.isfinite(draws).all(axis=1).sum())
        effects = pd.DataFrame(np.nan, index=range(len(post)), columns=_EFFECT_COLUMNS)
        effects["period"] = post
        effects["event_time"] = np.arange(len(post))
        effects["att"] = np.nan if invalid else raw
        reason = (
            "invalid original counterfactual"
            if invalid
            else (
                "one or more effect bootstrap draws failed"
                if valid != b
                else "zero or undefined bootstrap standard error"
            )
        )
        effects["inference_status"] = "unavailable"
        effects["reason"] = reason
        att = np.nan if invalid else float(np.mean(raw))
        se = np.nan
        stat, pvalue, ci = safe_inference(att, se, alpha=self.alpha)
        vcov = None
        crit = np.nan
        if not invalid and valid == b:
            inference = _inference(raw, draws, self.alpha)
            for key in effects.columns:
                if key in inference:
                    effects[key] = inference[key]
            available = np.isfinite(effects["p_value"])
            effects.loc[available, "inference_status"] = "available"
            effects.loc[available, "reason"] = None
            crit = inference["cband_crit_value"]
            vcov = np.atleast_2d(np.cov(draws, rowvar=False, ddof=1))
            headline = _inference(np.array([att]), draws.mean(axis=1)[:, None], self.alpha)
            se, stat, pvalue = (float(headline[key][0]) for key in ("se", "t_stat", "p_value"))
            ci = (float(headline["conf_int_lower"][0]), float(headline["conf_int_upper"][0]))
        n_available = int((effects["inference_status"] == "available").sum())
        statuses = {
            "pointwise": (
                "available"
                if n_available == len(post)
                else ("partial" if n_available else "unavailable")
            ),
            "simultaneous": "available" if np.isfinite(crit) else "unavailable",
            "simple": "available" if np.isfinite(pvalue) else "unavailable",
        }
        reasons = {key: [] if value == "available" else [reason] for key, value in statuses.items()}
        diagnostic = self._diagnostic(
            contrasts, diagnostic_errors, diagnostic_draws, failure_table, periods, start
        )
        counts = np.array([np.sum(1 - y[group == k], axis=0) for k in (0, 1)], dtype=int)
        support_warnings = []
        # A reporting heuristic, separate from the strict log-domain guard.
        for k, label in ((0, "control"), (1, "treated")):
            for i, count in enumerate(counts[k]):
                if count < 5:
                    support_warnings.append(
                        f"{label} survivors at {periods[i]!r}: {count} (<5 support heuristic)"
                    )
        if self.method == "proportional_hazards":
            for i in support:
                exits = counts[0, 0] - counts[0, i]
                if exits < 5:
                    support_warnings.append(
                        f"control exits since baseline at {periods[i]!r}: {exits} (<5 PH support heuristic)"
                    )
        curve = pd.DataFrame(
            {
                "period": periods,
                "elapsed_time": np.arange(len(periods), dtype=float),
                "treated_survival": s[1],
                "control_survival": s[0],
                "treated_survivors": counts[1],
                "control_survivors": counts[0],
                "raw_counterfactual_cumulative_hazard": np.r_[np.full(start, np.nan), r_cf],
                "raw_counterfactual_survival": np.r_[np.full(start, np.nan), s_cf],
                "raw_att": np.r_[np.full(start, np.nan), raw],
                "counterfactual_status": ["not_estimated"] * start
                + ["invalid" if i in invalid else "valid" for i in range(start, len(periods))],
                "reason": [invalid.get(i) for i in range(len(periods))],
            },
            columns=_SURVIVAL_COLUMNS,
        )
        result = DurationDiDResults(
            att=att,
            se=se,
            t_stat=stat,
            p_value=pvalue,
            conf_int=ci,
            method=self.method,
            alpha=self.alpha,
            seed=self.seed,
            n_bootstrap=b,
            n_obs=len(data),
            n_units=len(y),
            n_treated=int(np.sum(group == 1)),
            n_control=int(np.sum(group == 0)),
            periods=periods,
            pre_periods=periods[:start],
            post_periods=post,
            fit_periods=selected,
            requested_fit_periods=requested,
            time_weights=dict(zip(selected, map(float, w))),
            excluded_fit_periods=excluded,
            time_origin=periods[0],
            time_step=step,
            coefficient=coefficient,
            effects=effects,
            survival_curve=curve,
            bootstrap_effects=draws,
            n_bootstrap_valid=valid,
            bootstrap_failures=failure_table,
            vcov=vcov,
            cband_crit_value=crit,
            estimation_status="invalid_counterfactual" if invalid else "ok",
            inference_status=statuses,
            inference_reasons=reasons,
            support_warnings=support_warnings,
            pretrend_results=diagnostic,
        )
        self.results_ = result
        self.is_fitted_ = True
        return result

    def _diagnostic(
        self,
        values: np.ndarray,
        errors: Dict[Any, str],
        draws: np.ndarray,
        failures: pd.DataFrame,
        periods: List[Any],
        start: int,
    ) -> DurationDiDPretestResults:
        reasons = [
            f"{periods[i] if i is not None else 'pretest'}: {reason}"
            for i, reason in errors.items()
        ]
        attempted = 0 if errors else self.n_bootstrap
        valid = 0 if errors else int(np.isfinite(draws).all(axis=1).sum())
        table = pd.DataFrame(np.nan, index=range(len(values)), columns=_CONTRAST_COLUMNS)
        table["period"] = periods[1 : start - 1]
        table["elapsed_time"] = np.arange(1, start - 1, dtype=float)
        table["contrast"] = values
        statistic = pvalue = critical = np.nan
        reject = None
        if not errors and valid == self.n_bootstrap:
            inference = _inference(values, draws, self.alpha)
            table["se"] = inference["se"]
            critical = inference["cband_crit_value"]
            if np.isfinite(critical):
                table["cband_lower"] = inference["cband_lower"]
                table["cband_upper"] = inference["cband_upper"]
                statistic, pvalue = inference["statistic"], inference["simultaneous_p_value"]
                reject = bool(np.any(table["cband_lower"] > 0) or np.any(table["cband_upper"] < 0))
            else:
                reasons.append("zero or undefined diagnostic bootstrap standard error")
        elif not errors:
            reasons.append("one or more diagnostic bootstrap draws failed")
        status = "available" if reject is not None else "unavailable"
        table["status"] = status
        table["reason"] = None if not reasons else "; ".join(reasons)
        return DurationDiDPretestResults(
            method=self.method,
            alpha=self.alpha,
            anchor_period=periods[start - 1],
            contrasts=table,
            statistic=statistic,
            p_value=pvalue,
            critical_value=critical,
            reject=reject,
            status=status,
            reasons=reasons,
            n_bootstrap=self.n_bootstrap,
            n_bootstrap_attempted=attempted,
            n_bootstrap_valid=valid,
            bootstrap_contrasts=draws,
            bootstrap_failures=failures.loc[failures["family"] == "diagnostics"].reset_index(
                drop=True
            ),
        )
