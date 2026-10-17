"""Clean-room omitted-variable-bias sensitivity analysis for canonical DiD.

This module implements the estimable components and sensitivity bounds described
by Wang, Sant'Anna, Chernozhukov, and Cinelli, "Omitted Variable Bias in
Difference-in-Differences Designs".  It is an original Python implementation;
the GPL-3 ``dml.sensemakr`` source is not imported or translated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from diff_diff._crossfit import FoldAssignment, assign_folds, cross_fit_predict
from diff_diff._learners import make_learner

__all__ = [
    "DIDOVBSensitivity",
    "DIDOVBSensitivityResults",
    "DIDOVBBounds",
    "DIDOVBRobustness",
]


def _validate_probability(value: float, name: str, *, allow_one: bool = True) -> float:
    value = float(value)
    upper = 1.0 if allow_one else np.nextafter(1.0, 0.0)
    if not np.isfinite(value) or value < 0.0 or value > upper:
        right = "1" if allow_one else "1 (exclusive)"
        raise ValueError(f"{name} must be in [0, {right}], got {value!r}")
    return value


def _validate_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0.0 < alpha < 0.5:
        raise ValueError(f"alpha must be in (0, 0.5), got {alpha!r}")
    return alpha


@dataclass(frozen=True)
class DIDOVBBounds:
    """Bias bounds and delta-method confidence intervals."""

    lower: float
    upper: float
    radius: float
    lower_se: float
    upper_se: float
    lower_ci: float
    upper_ci: float
    rho_max: float
    trend_r2: float
    selection_r2: float
    alpha: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "radius": self.radius,
            "lower_se": self.lower_se,
            "upper_se": self.upper_se,
            "lower_ci": self.lower_ci,
            "upper_ci": self.upper_ci,
            "rho_max": self.rho_max,
            "trend_r2": self.trend_r2,
            "selection_r2": self.selection_r2,
            "alpha": self.alpha,
        }


@dataclass(frozen=True)
class DIDOVBRobustness:
    """RV/XRV search results and the intervals at the reported strengths."""

    null_value: float
    alpha: float
    rv: float
    xrv: float
    rv_bounds: DIDOVBBounds
    xrv_bounds: DIDOVBBounds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "null_value": self.null_value,
            "alpha": self.alpha,
            "rv": self.rv,
            "xrv": self.xrv,
            "rv_bounds": self.rv_bounds.to_dict(),
            "xrv_bounds": self.xrv_bounds.to_dict(),
        }


@dataclass
class DIDOVBSensitivityResults:
    """Fitted canonical DiD OVB sensitivity results."""

    short_att: float
    short_se: float
    sigma2_control: float
    nu2_selection: float
    scale: float
    n_obs: int
    n_treated: int
    n_control: int
    n_folds: int
    seed: Optional[int]
    propensity_learner: Any
    outcome_learner: Any
    _theta_if: np.ndarray
    _scale_if: np.ndarray

    def _radius(self, trend_r2: float, selection_r2: float, rho_max: float) -> float:
        _validate_probability(trend_r2, "trend_r2")
        _validate_probability(selection_r2, "selection_r2", allow_one=False)
        _validate_probability(rho_max, "rho_max")
        selection_factor = np.sqrt(selection_r2 / (1.0 - selection_r2))
        return float(rho_max * np.sqrt(trend_r2) * selection_factor * self.scale)

    def bounds(
        self,
        *,
        trend_r2: float = 1.0,
        selection_r2: float = 0.5,
        rho_max: float = 1.0,
        alpha: float = 0.05,
    ) -> DIDOVBBounds:
        """Return OVB bounds under paper-scale trend/selection restrictions.

        ``trend_r2`` is the share of residual control-trend variation explained
        by the omitted confounder. ``selection_r2`` is the residual share of
        treatment-odds variation; its implied selection factor is
        ``sqrt(selection_r2 / (1-selection_r2))``.
        """
        alpha = _validate_alpha(alpha)
        radius = self._radius(trend_r2, selection_r2, rho_max)
        selection_factor = np.sqrt(selection_r2 / (1.0 - selection_r2))
        multiplier = float(rho_max * np.sqrt(trend_r2) * selection_factor)
        lower_if = self._theta_if - multiplier * self._scale_if
        upper_if = self._theta_if + multiplier * self._scale_if
        lower_se = float(np.sqrt(np.mean(lower_if**2) / self.n_obs))
        upper_se = float(np.sqrt(np.mean(upper_if**2) / self.n_obs))
        critical = float(norm.ppf(1.0 - alpha / 2.0))
        return DIDOVBBounds(
            lower=self.short_att - radius,
            upper=self.short_att + radius,
            radius=radius,
            lower_se=lower_se,
            upper_se=upper_se,
            lower_ci=self.short_att - radius - critical * lower_se,
            upper_ci=self.short_att + radius + critical * upper_se,
            rho_max=float(rho_max),
            trend_r2=float(trend_r2),
            selection_r2=float(selection_r2),
            alpha=alpha,
        )

    def robustness_value(
        self, *, null_value: float = 0.0, alpha: float = 0.05, tolerance: float = 1e-10
    ) -> DIDOVBRobustness:
        """Find the paper's common-strength RV and selection-only XRV."""
        alpha = _validate_alpha(alpha)
        null_value = float(null_value)
        base = self.bounds(trend_r2=0.0, selection_r2=0.0, alpha=alpha)
        if base.lower_ci <= null_value <= base.upper_ci:
            zero = self.bounds(trend_r2=0.0, selection_r2=0.0, alpha=alpha)
            return DIDOVBRobustness(null_value, alpha, 0.0, 0.0, zero, zero)

        def search(kind: str) -> tuple[float, DIDOVBBounds]:
            def interval_contains(strength: float) -> bool:
                if kind == "rv":
                    b = self.bounds(trend_r2=strength, selection_r2=strength, alpha=alpha)
                else:
                    b = self.bounds(trend_r2=1.0, selection_r2=strength, alpha=alpha)
                return b.lower_ci <= null_value <= b.upper_ci

            high = np.nextafter(1.0, 0.0)
            if not interval_contains(high):
                if kind == "rv":
                    b = self.bounds(trend_r2=high, selection_r2=high, alpha=alpha)
                else:
                    b = self.bounds(trend_r2=1.0, selection_r2=high, alpha=alpha)
                return float("nan"), b
            low = 0.0
            for _ in range(80):
                mid = (low + high) / 2.0
                if interval_contains(mid):
                    high = mid
                else:
                    low = mid
                if high - low <= tolerance:
                    break
            if kind == "rv":
                b = self.bounds(trend_r2=high, selection_r2=high, alpha=alpha)
            else:
                b = self.bounds(trend_r2=1.0, selection_r2=high, alpha=alpha)
            return float(high), b

        rv, rv_bounds = search("rv")
        xrv, xrv_bounds = search("xrv")
        return DIDOVBRobustness(null_value, alpha, rv, xrv, rv_bounds, xrv_bounds)

    def contour(
        self, *, null_value: float = 0.0, alpha: float = 0.05, n_grid: int = 101
    ) -> pd.DataFrame:
        """Return confidence-bound endpoints over a common-strength grid."""
        alpha = _validate_alpha(alpha)
        if isinstance(n_grid, bool) or int(n_grid) != n_grid or n_grid < 2:
            raise ValueError(f"n_grid must be an integer >= 2, got {n_grid!r}")
        rows = []
        for strength in np.linspace(0.0, np.nextafter(1.0, 0.0), int(n_grid)):
            rv = self.bounds(trend_r2=float(strength), selection_r2=float(strength), alpha=alpha)
            xrv = self.bounds(trend_r2=1.0, selection_r2=float(strength), alpha=alpha)
            rows.append(
                {
                    "strength": float(strength),
                    "rv_lower": rv.lower_ci,
                    "rv_upper": rv.upper_ci,
                    "xrv_lower": xrv.lower_ci,
                    "xrv_upper": xrv.upper_ci,
                }
            )
        return pd.DataFrame(rows)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "short_att": self.short_att,
            "short_se": self.short_se,
            "sigma2_control": self.sigma2_control,
            "nu2_selection": self.nu2_selection,
            "scale": self.scale,
            "n_obs": self.n_obs,
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "n_folds": self.n_folds,
            "seed": self.seed,
            "propensity_learner": repr(self.propensity_learner),
            "outcome_learner": repr(self.outcome_learner),
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_dict()])

    def summary(self) -> str:
        return "\n".join(
            [
                "OVB Sensitivity Analysis for Difference-in-Differences",
                f"Short ATT:          {self.short_att:.6g}",
                f"Short ATT SE:       {self.short_se:.6g}",
                f"Control trend var:  {self.sigma2_control:.6g}",
                f"Selection scale:    {self.nu2_selection:.6g}",
                f"OVB scale S0:       {self.scale:.6g}",
                f"Observations:       {self.n_obs}",
            ]
        )


class DIDOVBSensitivity:
    """Cross-fitted canonical two-period DiD OVB sensitivity estimator."""

    def __init__(
        self,
        *,
        n_folds: int = 5,
        seed: Optional[int] = None,
        propensity_learner: Any = "logit",
        outcome_learner: Any = "linear",
        pscore_trim: float = 0.01,
    ) -> None:
        if isinstance(n_folds, bool) or int(n_folds) != n_folds or n_folds < 2:
            raise ValueError(f"n_folds must be an integer >= 2, got {n_folds!r}")
        if not 0.0 < float(pscore_trim) < 0.5:
            raise ValueError("pscore_trim must be in (0, 0.5)")
        self.n_folds = int(n_folds)
        self.seed = seed
        self.propensity_learner = propensity_learner
        self.outcome_learner = outcome_learner
        self.pscore_trim = float(pscore_trim)

    def fit(
        self,
        data: pd.DataFrame,
        *,
        outcome: str,
        treatment: str,
        time: str,
        unit: str,
        covariates: Sequence[str],
        outcome_change: Optional[str] = None,
        fold_ids: Optional[Sequence[int]] = None,
    ) -> DIDOVBSensitivityResults:
        """Fit on a balanced two-period panel."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        covariates = list(covariates)
        required = [outcome, treatment, time, unit, *covariates]
        missing = [name for name in required if name not in data.columns]
        if missing:
            raise KeyError(f"missing columns: {missing}")
        periods = list(pd.unique(data[time]))
        if len(periods) != 2:
            raise ValueError("DIDOVBSensitivity requires exactly two periods")
        try:
            periods = sorted(periods)
        except TypeError as exc:
            raise ValueError("time values must be sortable") from exc

        grouped = data.groupby(unit, sort=False, dropna=False)
        if not bool((grouped.size() == 2).all()):
            raise ValueError("each unit must have exactly one observation in each period")
        pre = data.loc[data[time] == periods[0]].set_index(unit)
        post = data.loc[data[time] == periods[1]].set_index(unit)
        if not pre.index.is_unique or not post.index.is_unique:
            raise ValueError("unit must identify one row per period")
        common = pre.index.intersection(post.index)
        if len(common) != len(pre) or len(common) != len(post):
            raise ValueError("the two periods must contain the same units")
        pre = pre.loc[common]
        post = post.loc[common]

        d_pre = pd.to_numeric(pre[treatment], errors="coerce").to_numpy(dtype=float)
        d_post = pd.to_numeric(post[treatment], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(d_pre)) or not np.all(np.isfinite(d_post)):
            raise ValueError("treatment must be finite and binary")
        if not np.array_equal(d_pre, d_post) or not np.all(np.isin(d_pre, [0.0, 1.0])):
            raise ValueError("treatment must be time-invariant and binary")
        y_pre = pd.to_numeric(pre[outcome], errors="coerce").to_numpy(dtype=float)
        y_post = pd.to_numeric(post[outcome], errors="coerce").to_numpy(dtype=float)
        x = pre[covariates].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if outcome_change is not None:
            if outcome_change not in data.columns:
                raise KeyError(f"missing columns: [{outcome_change!r}]")
            dy_pre = pd.to_numeric(pre[outcome_change], errors="coerce").to_numpy(dtype=float)
            dy_post = pd.to_numeric(post[outcome_change], errors="coerce").to_numpy(dtype=float)
            delta_y = dy_post - dy_pre
        else:
            delta_y = y_post - y_pre
        valid = np.isfinite(delta_y) & np.all(np.isfinite(x), axis=1)
        if not np.all(valid):
            d_pre, delta_y, x = d_pre[valid], delta_y[valid], x[valid]
        if len(delta_y) < 4:
            raise ValueError("at least four complete units are required")
        n_treated = int(np.sum(d_pre == 1.0))
        n_control = int(np.sum(d_pre == 0.0))
        if n_treated < self.n_folds or n_control < self.n_folds:
            raise ValueError("n_folds cannot exceed the number of treated or control units")

        if fold_ids is None:
            rng = np.random.default_rng(self.seed)
            folds = assign_folds(len(delta_y), self.n_folds, rng=rng, stratify=d_pre)
        else:
            provided_folds = np.asarray(fold_ids)
            if provided_folds.shape != (len(delta_y),) or not np.issubdtype(
                provided_folds.dtype, np.integer
            ):
                raise ValueError("fold_ids must be an integer vector with one entry per unit")
            provided_folds = provided_folds.astype(np.int64, copy=False)
            if np.any(provided_folds < 0) or np.any(provided_folds >= self.n_folds):
                raise ValueError("fold_ids must lie in [0, n_folds)")
            if len(np.unique(provided_folds)) != self.n_folds:
                raise ValueError("fold_ids must contain every fold")
            folds = FoldAssignment(
                n_folds=self.n_folds,
                n_units=len(delta_y),
                fold_ids=provided_folds,
                bitgen_state={},
                bitgen_name="provided",
                stratify_labels=d_pre.copy(),
            )
        ps_result = cross_fit_predict(
            make_learner(self.propensity_learner, kind="classifier"),
            x,
            d_pre,
            folds,
            predict_method="predict_proba",
            context_label="DIDOVBSensitivity propensity learner",
        )
        outcome_result = cross_fit_predict(
            make_learner(self.outcome_learner, kind="regressor"),
            x,
            delta_y,
            folds,
            predict_method="predict",
            fit_mask=d_pre == 0.0,
            context_label="DIDOVBSensitivity outcome learner",
        )
        p = float(np.mean(d_pre))
        ps = np.clip(ps_result.oof_predictions, self.pscore_trim, 1.0 - self.pscore_trim)
        omega = (ps / (1.0 - ps)) / (p / (1.0 - p))
        residual = delta_y - outcome_result.oof_predictions
        score = (d_pre / p - (1.0 - d_pre) / (1.0 - p) * omega) * residual
        short_att = float(np.mean(score))
        sigma2 = float(np.mean(residual[d_pre == 0.0] ** 2))
        nu2 = float(np.mean(omega[d_pre == 0.0] ** 2))
        if not np.isfinite(short_att) or not np.isfinite(sigma2) or not np.isfinite(nu2):
            raise ValueError("OVB components are non-finite")
        scale = float(np.sqrt(max(0.0, sigma2 * nu2)))
        theta_if = score - short_att
        sigma_if = (1.0 - d_pre) / (1.0 - p) * (residual**2 - sigma2)
        nu_if = (1.0 - d_pre) / (1.0 - p) * (omega**2 - nu2)
        scale_if = np.zeros_like(theta_if)
        if scale > 0.0:
            scale_if = (nu2 * sigma_if + sigma2 * nu_if) / (2.0 * scale)
        short_se = float(np.sqrt(np.mean(theta_if**2) / len(theta_if)))
        return DIDOVBSensitivityResults(
            short_att=short_att,
            short_se=short_se,
            sigma2_control=sigma2,
            nu2_selection=nu2,
            scale=scale,
            n_obs=len(delta_y),
            n_treated=n_treated,
            n_control=n_control,
            n_folds=self.n_folds,
            seed=self.seed,
            propensity_learner=self.propensity_learner,
            outcome_learner=self.outcome_learner,
            _theta_if=theta_if,
            _scale_if=scale_if,
        )
