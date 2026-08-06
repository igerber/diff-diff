"""Small, composable panel-treatment-effects primitives.

The API mirrors the infrastructure exposed by R ``ptetools``.  Estimators in
this module are intentionally separate from the high-level estimator classes:
``setup_pte`` describes a panel, ``two_by_two_subset`` creates one group-time
comparison, and ``did_attgt`` estimates the resulting two-period ATT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.special import expit


@dataclass
class PTEParams:
    data: pd.DataFrame
    yname: str
    gname: str
    tname: str
    idname: Optional[str]
    panel: bool
    groups: list[Any]
    time_periods: list[Any]
    anticipation: int = 0
    base_period: str = "varying"
    weightsname: Optional[str] = None

    @property
    def glist(self) -> list[Any]:
        return self.groups

    @property
    def tlist(self) -> list[Any]:
        return self.time_periods


@dataclass
class GTDataFrame:
    data: pd.DataFrame

    def __post_init__(self) -> None:
        self.data = self.data.copy()

    def __getitem__(self, key: Any) -> Any:
        return self.data[key]

    def __len__(self) -> int:
        return len(self.data)


@dataclass
class TwoByTwoSubset:
    gt_data: GTDataFrame
    n1: int
    disidx: np.ndarray


@dataclass
class ATTGTResult:
    attgt: float
    inf_func: Optional[np.ndarray] = None
    extra_gt_returns: Any = None


@dataclass
class PTEAggregateResult:
    estimate: float
    weights: pd.DataFrame
    type: str = "group"


@dataclass
class PTEResults:
    """Results from the generic group-time ATT loop."""

    att_gt: pd.DataFrame
    overall_att: float
    overall_se: float
    influence_functions: Optional[np.ndarray] = None

    def to_dataframe(self) -> pd.DataFrame:
        return self.att_gt.copy()

    def aggregate(self, type: str = "group") -> PTEAggregateResult:
        return pte_aggte(self.att_gt, type=type)


def gt_data_frame(data: pd.DataFrame) -> GTDataFrame:
    """Mark a two-period comparison table as ptetools-compatible."""
    required = {"G", "id", "period", "name", "Y", "D"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"gt_data is missing required columns: {missing}")
    return GTDataFrame(data)


def setup_pte(
    data: pd.DataFrame,
    yname: str,
    gname: str,
    tname: str,
    idname: Optional[str] = None,
    *,
    panel: bool = True,
    anticipation: int = 0,
    base_period: str = "varying",
    weightsname: Optional[str] = None,
) -> PTEParams:
    """Validate a panel and return the metadata used by ``ptetools`` steps."""
    if base_period not in {"varying", "universal"}:
        raise ValueError("base_period must be 'varying' or 'universal'")
    if not isinstance(anticipation, (int, np.integer)) or anticipation < 0:
        raise ValueError("anticipation must be a non-negative integer")
    required = {yname, gname, tname}
    if panel:
        if idname is None:
            raise ValueError("idname is required for panel data")
        required.add(idname)
    if weightsname is not None:
        required.add(weightsname)
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"data is missing required columns: {missing}")
    out = data.copy()
    if out[[yname, gname, tname]].isna().any().any():
        raise ValueError("outcome, group, and time columns cannot contain missing values")
    periods = sorted(pd.unique(out[tname]).tolist())
    if not periods:
        raise ValueError("data must contain at least one time period")
    groups = sorted(pd.unique(out[gname]).tolist())
    if 0 not in groups:
        raise ValueError("never-treated units must be coded as group 0")
    treated_groups = [g for g in groups if g != 0]
    if any(g <= min(periods) for g in treated_groups):
        raise ValueError("treated groups must have at least one pre-treatment period")
    return PTEParams(
        data=out,
        yname=yname,
        gname=gname,
        tname=tname,
        idname=idname,
        panel=panel,
        groups=treated_groups,
        time_periods=periods[1:],
        anticipation=int(anticipation),
        base_period=base_period,
        weightsname=weightsname,
    )


def two_by_two_subset(
    data: pd.DataFrame,
    g: Any,
    tp: Any,
    *,
    gname: str = "G",
    tname: str = "period",
    idname: str = "id",
    yname: str = "Y",
    control_group: str = "notyettreated",
    anticipation: int = 0,
    base_period: str = "varying",
    covariates: Sequence[str] = (),
) -> TwoByTwoSubset:
    """Construct the two-period ``(g,t)`` subset used by ATT(g,t) estimators."""
    if control_group not in {"notyettreated", "nevertreated"}:
        raise ValueError("control_group must be 'notyettreated' or 'nevertreated'")
    if base_period not in {"varying", "universal"}:
        raise ValueError("base_period must be 'varying' or 'universal'")
    pre = g - anticipation - 1 if base_period == "universal" else tp - 1
    if pre not in set(pd.unique(data[tname])):
        raise ValueError(f"base period {pre!r} is not present in data")
    cohort = data[gname]
    if control_group == "nevertreated":
        keep = cohort.isin([0, g])
    else:
        keep = cohort.isin([0, g]) | (cohort > tp)
    keep &= data[tname].isin([pre, tp])
    columns = [gname, idname, tname, yname] + list(covariates)
    missing_covariates = sorted(set(covariates).difference(data.columns))
    if missing_covariates:
        raise ValueError(f"data is missing covariates: {missing_covariates}")
    out = data.loc[keep, columns].copy()
    out = out.rename(columns={gname: "G", idname: "id", tname: "period", yname: "Y"})
    out["name"] = np.where(out["period"].eq(tp), "post", "pre")
    out["D"] = (out["G"] == g).astype(int)
    out = out.sort_values(["id", "period"]).reset_index(drop=True)
    if out.empty or out["D"].sum() == 0 or (out["D"] == 0).sum() == 0:
        raise ValueError("two_by_two_subset has no treated or comparison observations")
    ids = pd.unique(data[idname])
    disidx = np.isin(ids, pd.unique(out["id"]))
    return TwoByTwoSubset(gt_data_frame(out), int(out.loc[out.D == 1, "id"].nunique()), disidx)


def attgt_if(
    attgt: float, inf_func: Optional[Sequence[float]] = None, extra_gt_returns: Any = None
) -> ATTGTResult:
    """Create the influence-function result container used by ``pte``."""
    return ATTGTResult(
        attgt=float(attgt),
        inf_func=None if inf_func is None else np.asarray(inf_func, float),
        extra_gt_returns=extra_gt_returns,
    )


def did_attgt(
    gt_data: GTDataFrame | pd.DataFrame, *, covariates: Sequence[str] = ()
) -> ATTGTResult:
    """Estimate a two-period ATT, optionally with pre-period AIPW covariates."""
    frame = gt_data.data if isinstance(gt_data, GTDataFrame) else gt_data
    required = {"id", "D", "name", "Y"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"gt_data is missing required columns: {missing}")
    wide = frame.pivot_table(index="id", columns="name", values="Y", aggfunc="first")
    treat = frame.groupby("id", sort=False)["D"].first().reindex(wide.index).to_numpy(float)
    if not {"pre", "post"}.issubset(wide.columns):
        raise ValueError("gt_data must contain both pre and post observations")
    delta = (wide["post"] - wide["pre"]).to_numpy(float)
    treated = treat == 1
    control = treat == 0
    if treated.sum() == 0 or control.sum() == 0:
        raise ValueError("both treated and comparison units are required")
    if covariates:
        pre = frame.loc[frame["name"].eq("pre")].set_index("id")
        x = pre.loc[wide.index, list(covariates)].to_numpy(float)
        x = np.column_stack([np.ones(len(x)), x])
        control_x = x[control]
        control_delta = delta[control]
        m_coef = np.linalg.lstsq(control_x, control_delta, rcond=None)[0]
        m_hat = x @ m_coef
        p_coef = np.zeros(x.shape[1], dtype=float)
        for _ in range(100):
            probability = np.clip(expit(x @ p_coef), 1e-8, 1 - 1e-8)
            variance = np.clip(probability * (1 - probability), 1e-8, None)
            working = x @ p_coef + (treat - probability) / variance
            updated = np.linalg.lstsq(
                x * np.sqrt(variance)[:, None], working * np.sqrt(variance), rcond=None
            )[0]
            if np.max(np.abs(updated - p_coef)) < 1e-10:
                p_coef = updated
                break
            p_coef = updated
        propensity = np.clip(expit(x @ p_coef), 1e-8, 1 - 1e-8)
        pi = float(treated.mean())
        residual = delta - m_hat
        score = treated / pi * residual - control / pi * propensity / (1 - propensity) * residual
        att = float(score.mean())
    else:
        score = None
        att = float(delta[treated].mean() - delta[control].mean())
    inf = np.zeros(len(delta), dtype=float)
    if score is not None:
        inf = score - att - att / treated.mean() * (treat - treated.mean())
    else:
        inf[treated] = (delta[treated] - delta[treated].mean()) / treated.mean()
        inf[control] = -(delta[control] - delta[control].mean()) / (1.0 - treated.mean())
    return attgt_if(att, inf)


def overall_weights(
    attgt: pd.DataFrame, *, group: str = "group", time: str = "time"
) -> pd.DataFrame:
    """Return Callaway--Sant'Anna overall weights for post-treatment cells."""
    required = {group, time, "attgt"}
    missing = sorted(required.difference(attgt.columns))
    if missing:
        raise ValueError(f"attgt is missing columns: {missing}")
    frame = attgt.rename(columns={group: "group", time: "time"}).copy()
    treated = frame["group"] != 0
    periods = sorted(pd.unique(frame["time"]))
    max_time = max(periods)
    cohort_counts = frame.loc[treated, "group"].value_counts().sort_index()
    cohort_share = cohort_counts / cohort_counts.sum()
    frame["overall_weight"] = [
        float(cohort_share.get(g, 0.0) / (max_time - g + 1)) if g != 0 and t >= g else 0.0
        for g, t in zip(frame["group"], frame["time"])
    ]
    return frame[["group", "time", "overall_weight"]]


def pte(
    data: pd.DataFrame,
    *,
    yname: str,
    gname: str,
    tname: str,
    idname: str,
    control_group: str = "notyettreated",
    anticipation: int = 0,
    base_period: str = "varying",
    covariates: Sequence[str] = (),
) -> PTEResults:
    """Run the generic unadjusted panel ATT(g,t) loop."""
    params = setup_pte(
        data,
        yname,
        gname,
        tname,
        idname,
        anticipation=anticipation,
        base_period=base_period,
    )
    rows = []
    influence = []
    n_units = data[idname].nunique()
    for g in params.groups:
        for tp in params.time_periods:
            if base_period == "universal" and tp == g - anticipation - 1:
                rows.append({"group": g, "time": tp, "attgt": 0.0, "se": np.nan})
                influence.append(np.full(n_units, np.nan))
                continue
            subset = two_by_two_subset(
                data,
                g,
                tp,
                gname=gname,
                tname=tname,
                idname=idname,
                yname=yname,
                control_group=control_group,
                anticipation=anticipation,
                base_period=base_period,
                covariates=covariates,
            )
            result = did_attgt(subset.gt_data, covariates=covariates)
            if result.inf_func is None:
                raise RuntimeError("did_attgt did not return an influence function")
            se = float(np.sqrt(np.nanmean(result.inf_func**2) / len(result.inf_func)))
            rows.append({"group": g, "time": tp, "attgt": result.attgt, "se": se})
            full_if = np.full(n_units, np.nan)
            full_if[subset.disidx] = result.inf_func
            influence.append(full_if)
    att_gt = pd.DataFrame(rows)
    weights = overall_weights(att_gt)
    valid = np.isfinite(att_gt["attgt"]) & (weights["overall_weight"] > 0)
    overall_att = float(np.sum(att_gt.loc[valid, "attgt"] * weights.loc[valid, "overall_weight"]))
    full_influence = np.asarray(influence, dtype=float).T if influence else None
    return PTEResults(att_gt, overall_att, float("nan"), full_influence)


def pte_aggte(attgt: pd.DataFrame, *, type: str = "group") -> PTEAggregateResult:
    """Aggregate an ATT(g,t) table using group or dynamic weights."""
    if type not in {"group", "dynamic"}:
        raise ValueError("type must be 'group' or 'dynamic'")
    frame = attgt.copy()
    if type == "group":
        weights = overall_weights(frame)
    else:
        required = {"group", "time", "attgt"}
        if not required.issubset(frame.columns):
            raise ValueError("dynamic aggregation requires group, time, and attgt columns")
        frame["event_time"] = frame["time"] - frame["group"]
        frame = frame.loc[frame["event_time"] >= 0].copy()
        frame["overall_weight"] = frame.groupby("event_time")["group"].transform("count").rdiv(1.0)
        frame["overall_weight"] /= frame["overall_weight"].sum()
        weights = frame[["group", "time", "overall_weight"]]
    effects = frame["attgt"].to_numpy(float)
    w = weights["overall_weight"].to_numpy(float)
    if len(effects) != len(w):
        effects = frame.loc[weights.index, "attgt"].to_numpy(float)
    return PTEAggregateResult(float(np.nansum(effects * w)), weights.reset_index(drop=True), type)
