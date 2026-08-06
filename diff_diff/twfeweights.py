"""ATT(g,t) weights for two-way fixed-effects decompositions.

This module ports the no-covariate part of the R package
``twfeweights``.  The implementation deliberately accepts a small, explicit
Python representation rather than depending on the internal layout of an R
``att_gt`` object: ``attgt`` may be a DataFrame with ``group``, ``time`` and
``attgt`` columns, or a fitted diff-diff result together with its input data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.special import expit


@dataclass
class MPWeightsResult:
    """Container returned by the ported weight functions.

    ``weights_df`` mirrors the data frame returned by R and is the stable
    machine-readable surface.  ``att`` is provided as a convenience and is
    not used when constructing the weights.
    """

    weights_df: pd.DataFrame

    @property
    def att(self) -> float:
        values = self.weights_df["weight"].to_numpy(float)
        effects = self.weights_df["attgt"].to_numpy(float)
        return float(np.nansum(values * effects))

    def to_dataframe(self) -> pd.DataFrame:
        return self.weights_df.copy()

    def __getitem__(self, key: Any) -> Any:
        return self.weights_df[key]


@dataclass
class TwoPeriodCovariatesResult:
    """Container for two-period regression-weight diagnostics."""

    est: float
    weights: np.ndarray
    dy: np.ndarray
    treatment: np.ndarray
    cov_balance_df: Optional[pd.DataFrame] = None
    ess: Optional[float] = None


def _coerce_inputs(
    attgt: pd.DataFrame,
    data: pd.DataFrame,
    group: str,
    time: str,
    treatment_group: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Any]]:
    required = {group, time, "attgt"}
    missing = required.difference(attgt.columns)
    if missing:
        raise ValueError(f"attgt is missing columns: {sorted(missing)}")
    if treatment_group not in data.columns:
        raise ValueError(f"data is missing treatment-group column {treatment_group!r}")

    effects = attgt[[group, time, "attgt"]].copy()
    effects = effects.rename(columns={group: "group", time: "time"})
    effects["group"] = effects["group"].replace({np.inf: 0, -np.inf: 0})
    effects["time"] = effects["time"]
    periods = sorted(pd.unique(effects["time"]))
    if not periods:
        raise ValueError("attgt must contain at least one time period")
    return effects, data, periods


def _result_frame(
    effects: pd.DataFrame,
    weights: np.ndarray,
    keep_untreated: bool,
) -> MPWeightsResult:
    out = effects.copy()
    out["weight"] = weights
    out["post"] = ((out["time"] >= out["group"]) & (out["group"] != 0)).astype(bool)
    if not keep_untreated:
        out = out.loc[out["group"] != 0].reset_index(drop=True)
    out = out.rename(columns={"time": "time.period"})
    return MPWeightsResult(out[["group", "time.period", "weight", "attgt", "post"]])


def _extract_attgt(
    attgt: Any,
    data: Optional[pd.DataFrame],
    group: str,
    time: str,
    treatment_group: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize a DataFrame or a result object into the port's inputs."""
    if isinstance(attgt, pd.DataFrame):
        if data is None:
            raise ValueError("data is required when attgt is a DataFrame")
        return _coerce_inputs(attgt, data, group, time, treatment_group)[:2]

    if data is None:
        data = getattr(attgt, "data", None)
    if data is None:
        raise ValueError("data must be supplied for a fitted result object")
    effects = getattr(attgt, "group_time_effects", None)
    periods = getattr(attgt, "time_periods", None)
    if effects is None or periods is None:
        raise TypeError("attgt must be a DataFrame or a fitted result with group_time_effects")
    rows = []
    for (g, t), value in effects.items():
        effect = (
            value.get("effect", value.get("attgt", np.nan)) if isinstance(value, dict) else value
        )
        rows.append({"group": g, "time": t, "attgt": effect})
    return _coerce_inputs(pd.DataFrame(rows), data, "group", "time", treatment_group)[:2]


def _weights_frame(
    attgt: Any,
    data: Optional[pd.DataFrame],
    group: str,
    time: str,
    treatment_group: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Any]]:
    effects, panel = _extract_attgt(attgt, data, group, time, treatment_group)
    periods = sorted(pd.unique(effects["time"]))
    return effects, panel, periods


def twfe_weights(
    attgt: Any,
    data: Optional[pd.DataFrame] = None,
    *,
    group: str = "group",
    time: str = "time",
    treatment_group: str = "first_treat",
    keep_untreated: bool = False,
) -> MPWeightsResult:
    """Compute TWFE weights on ATT(g,t), porting ``twfe_weights`` from R."""
    effects, panel, periods = _weights_frame(attgt, data, group, time, treatment_group)
    gcol = panel[treatment_group].to_numpy()
    is_treated = (~pd.isna(gcol)) & (gcol != 0) & (~np.isinf(gcol))
    treated_share = {t: np.mean((gcol <= t) & is_treated) for t in periods}
    mean_treated_share = float(np.mean(list(treated_share.values())))
    groups = [0] + sorted(pd.unique(gcol[is_treated]).tolist())
    group_share = {g: (np.mean(~is_treated) if g == 0 else np.mean(gcol == g)) for g in groups}
    max_time = max(periods)

    def numerator(g: Any, t: Any) -> float:
        if g == 0:
            h = -treated_share[t] + mean_treated_share
        else:
            h = (
                float(t >= g)
                - (max_time - g + 1) / len(periods)
                - treated_share[t]
                + mean_treated_share
            )
        return h * group_share[g]

    raw = np.array([numerator(g, t) for g, t in zip(effects.group, effects.time)], dtype=float)
    treated_cells = (effects["group"].to_numpy() != 0) & (
        effects["time"].to_numpy() >= effects["group"].to_numpy()
    )
    denominator = raw[treated_cells].sum()
    if denominator == 0:
        raise ValueError("TWFE weights cannot be normalized for this treatment design")
    return _result_frame(effects, raw / denominator, keep_untreated)


def _cohort_shares(
    panel: pd.DataFrame, treatment_group: str, weights: Optional[Any]
) -> dict[Any, float]:
    gcol = panel[treatment_group].to_numpy()
    treated = (~pd.isna(gcol)) & (gcol != 0) & (~np.isinf(gcol))
    if not treated.any():
        raise ValueError("data contains no treated units")
    w = np.ones(len(panel), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if len(w) != len(panel):
        raise ValueError("weights must have one value per row in data")
    total = float(w[treated].sum())
    if total <= 0 or not np.isfinite(total):
        raise ValueError("treated-unit weights must have a positive finite sum")
    return {
        g: float(np.sum(w[treated] * (gcol[treated] == g)) / total)
        for g in pd.unique(gcol[treated])
    }


def attO_weights(
    attgt: Any,
    data: Optional[pd.DataFrame] = None,
    *,
    group: str = "group",
    time: str = "time",
    treatment_group: str = "first_treat",
    weights: Optional[Any] = None,
    keep_untreated: bool = False,
) -> MPWeightsResult:
    """Compute overall ATT weights (``ATT^O``) from Callaway-Sant'Anna."""
    effects, panel, periods = _weights_frame(attgt, data, group, time, treatment_group)
    shares = _cohort_shares(panel, treatment_group, weights)
    max_time = max(periods)
    values = np.array(
        [
            float(t >= g) * shares.get(g, 0.0) / (max_time - g + 1)
            for g, t in zip(effects.group, effects.time)
        ]
    )
    return _result_frame(effects, values, keep_untreated)


def att_simple_weights(
    attgt: Any,
    data: Optional[pd.DataFrame] = None,
    *,
    group: str = "group",
    time: str = "time",
    treatment_group: str = "first_treat",
    weights: Optional[Any] = None,
    keep_untreated: bool = False,
) -> MPWeightsResult:
    """Compute simple ATT weights from Callaway-Sant'Anna."""
    effects, panel, _ = _weights_frame(attgt, data, group, time, treatment_group)
    shares = _cohort_shares(panel, treatment_group, weights)
    values = np.array(
        [float(t >= g) * shares.get(g, 0.0) for g, t in zip(effects.group, effects.time)]
    )
    total = values.sum()
    if total == 0:
        raise ValueError("simple ATT weights have zero post-treatment mass")
    return _result_frame(effects, values / total, keep_untreated)


def ggtwfeweights(result: MPWeightsResult) -> Any:
    """Plot weights when matplotlib is installed."""
    import matplotlib.pyplot as plt

    frame = result.weights_df
    fig, ax = plt.subplots()
    for post, values in frame.groupby("post"):
        ax.scatter(values["weight"], values["attgt"], label=str(post))
    ax.axhline(0, color="black", linewidth=1.5)
    ax.axvline(0, color="black", linewidth=1.5)
    ax.set_xlabel("weight")
    ax.set_ylabel("ATT(g,t)")
    ax.legend(title="post")
    return ax


def effective_sample_size(est_weights: Any, sampling_weights: Optional[Any] = None) -> float:
    """Compute the effective sample size of normalized estimation weights."""
    weights = np.asarray(est_weights, dtype=float)
    sampling = (
        np.ones(len(weights)) if sampling_weights is None else np.asarray(sampling_weights, float)
    )
    sampling = sampling / np.mean(sampling)
    weights = weights / np.average(weights, weights=sampling)
    return float(weights.sum() ** 2 / np.sum(weights**2))


def pooled_sd(x: Any, treatment: Any, sampling_weights: Optional[Any] = None) -> float:
    """Compute the treated/control pooled weighted standard deviation."""
    values = np.asarray(x, dtype=float)
    d = np.asarray(treatment).astype(bool)
    w = np.ones(len(values)) if sampling_weights is None else np.asarray(sampling_weights, float)
    w = w / np.mean(w)

    def variance(z: np.ndarray, z_w: np.ndarray) -> float:
        mean = np.average(z, weights=z_w)
        return float(np.average((z - mean) ** 2, weights=z_w))

    n1, n0 = w[d].sum(), w[~d].sum()
    return float(
        np.sqrt(
            ((n1 - 1) * variance(values[d], w[d]) + (n0 - 1) * variance(values[~d], w[~d]))
            / (n1 + n0 - 2)
        )
    )


def log_ratio_sd(
    x: Any,
    treatment: Any,
    est_weights: Optional[Any] = None,
    sampling_weights: Optional[Any] = None,
) -> float:
    """Compare treated/control weighted standard deviations on a log scale."""
    values = np.asarray(x, dtype=float)
    d = np.asarray(treatment).astype(bool)
    sampling = (
        np.ones(len(values)) if sampling_weights is None else np.asarray(sampling_weights, float)
    )
    sampling = sampling / np.mean(sampling)
    estimation = (
        np.ones(len(values)) if est_weights is None else np.asarray(est_weights, float).copy()
    )
    estimation[d] /= np.average(estimation[d], weights=sampling[d])
    estimation[~d] /= np.average(estimation[~d], weights=sampling[~d])

    def spread(mask: np.ndarray) -> float:
        weighted = values[mask] * estimation[mask]
        center = np.average(weighted, weights=sampling[mask])
        return float(
            np.sqrt(
                (sampling[mask].sum() - 1)
                * np.average((weighted - center) ** 2, weights=sampling[mask])
            )
        )

    return float(np.log(spread(d)) - np.log(spread(~d)))


def frac_treated_extreme(
    x: Any,
    treatment: Any,
    est_weights: Optional[Any] = None,
    sampling_weights: Optional[Any] = None,
    alpha: float = 0.05,
) -> float:
    """Fraction of treated weighted mass outside untreated quantiles."""
    values = np.asarray(x, dtype=float)
    d = np.asarray(treatment).astype(bool)
    if len(np.unique(values)) < 3:
        return float("nan")
    sampling = (
        np.ones(len(values)) if sampling_weights is None else np.asarray(sampling_weights, float)
    )
    estimation = (
        np.ones(len(values)) if est_weights is None else np.asarray(est_weights, float).copy()
    )
    estimation[d] /= np.average(estimation[d], weights=sampling[d])
    estimation[~d] /= np.average(estimation[~d], weights=sampling[~d])
    low, high = np.quantile(values[~d] * estimation[~d], [alpha / 2, 1 - alpha / 2])
    treated_mass = sampling[d] * estimation[d]
    return float(
        treated_mass[(values[d] * estimation[d] < low) | (values[d] * estimation[d] > high)].sum()
        / treated_mass.sum()
    )


def two_period_reg_weights(
    data: pd.DataFrame,
    *,
    yname: str,
    tname: str,
    idname: str,
    gname: str,
    covariates: Sequence[str] = (),
    time_invariant_covariates: Sequence[str] = (),
    weightsname: Optional[str] = None,
) -> TwoPeriodCovariatesResult:
    """Compute two-period TWFE implicit regression weights."""
    required = {yname, tname, idname, gname, *covariates, *time_invariant_covariates}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"data is missing columns: {missing}")
    periods = sorted(pd.unique(data[tname]))
    if len(periods) != 2:
        raise ValueError("two_period_reg_weights only supports two periods")
    counts = data.groupby(idname)[tname].nunique()
    if (counts != 2).any():
        raise ValueError("two_period_reg_weights requires a balanced panel")
    ordered = data.sort_values([idname, tname])
    pre = ordered[ordered[tname] == periods[0]].set_index(idname)
    post = ordered[ordered[tname] == periods[1]].set_index(idname)
    ids = pre.index
    dy = (post.loc[ids, yname] - pre.loc[ids, yname]).to_numpy(float)
    treatment = (post.loc[ids, gname].to_numpy() != 0).astype(float)
    features = [np.ones(len(ids))]
    for column in covariates:
        features.append((post.loc[ids, column] - pre.loc[ids, column]).to_numpy(float))
    for column in time_invariant_covariates:
        features.append(pre.loc[ids, column].to_numpy(float))
    x = np.column_stack(features)
    sampling = (
        np.ones(len(ids)) if weightsname is None else pre.loc[ids, weightsname].to_numpy(float)
    )
    if np.any(~np.isfinite(sampling)) or np.any(sampling <= 0):
        raise ValueError("sampling weights must be positive and finite")
    coef = np.linalg.lstsq(
        x * np.sqrt(sampling)[:, None], treatment * np.sqrt(sampling), rcond=None
    )[0]
    residual = treatment - x @ coef
    denominator = np.average(residual**2, weights=sampling)
    if denominator <= 0:
        raise ValueError("treatment is collinear with the supplied covariates")
    implicit = residual / denominator
    estimate = float(np.average(implicit * dy, weights=sampling))
    ess = effective_sample_size(implicit[treatment == 0], sampling[treatment == 0])
    return TwoPeriodCovariatesResult(estimate, implicit, dy, treatment, ess=ess)


def two_period_aipw_weights(
    data: pd.DataFrame,
    *,
    yname: str,
    tname: str,
    idname: str,
    gname: str,
    covariates: Sequence[str] = (),
) -> TwoPeriodCovariatesResult:
    """Compute a two-period AIPW ATT and its implicit group weights."""
    reg = two_period_reg_weights(
        data,
        yname=yname,
        tname=tname,
        idname=idname,
        gname=gname,
        covariates=(),
    )
    if not covariates:
        return reg
    ordered = data.sort_values([idname, tname])
    periods = sorted(pd.unique(ordered[tname]))
    pre = ordered[ordered[tname] == periods[0]].set_index(idname)
    ids = pre.index
    d = reg.treatment
    dy = reg.dy
    x = np.column_stack([np.ones(len(ids)), *[pre.loc[ids, c].to_numpy(float) for c in covariates]])
    m_coef = np.linalg.lstsq(x[d == 0], dy[d == 0], rcond=None)[0]
    m_hat = x @ m_coef
    p_coef = np.zeros(x.shape[1], dtype=float)
    for _ in range(100):
        p = np.clip(expit(x @ p_coef), 1e-8, 1 - 1e-8)
        v = np.clip(p * (1 - p), 1e-8, None)
        z = x @ p_coef + (d - p) / v
        updated = np.linalg.lstsq(x * np.sqrt(v)[:, None], z * np.sqrt(v), rcond=None)[0]
        if np.max(np.abs(updated - p_coef)) < 1e-10:
            p_coef = updated
            break
        p_coef = updated
    propensity = np.clip(expit(x @ p_coef), 1e-8, 1 - 1e-8)
    pi = float(d.mean())
    residual = dy - m_hat
    control_weight = propensity / (1 - propensity)
    score = d / pi * residual - (1 - d) / pi * control_weight * residual
    weights = np.ones(len(d), dtype=float)
    weights[d == 0] = control_weight[d == 0] / np.mean(control_weight[d == 0])
    return TwoPeriodCovariatesResult(
        float(score.mean()), weights, dy, d, ess=effective_sample_size(weights[d == 0])
    )
