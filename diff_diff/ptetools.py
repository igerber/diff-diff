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
from scipy.interpolate import BSpline
from scipy.special import expit
from scipy.stats import norm


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


def group_time_att(
    att_gt: pd.DataFrame, *, influence_functions: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """Validate and construct a group-time ATT table."""
    required = {"group", "time", "attgt"}
    missing = sorted(required.difference(att_gt.columns))
    if missing:
        raise ValueError(f"att_gt is missing columns: {missing}")
    if influence_functions is not None and len(influence_functions) != len(att_gt):
        raise ValueError("influence_functions must have one row per ATT(g,t) cell")
    return att_gt.copy()


def process_att_gt(att_gt: pd.DataFrame, **_: Any) -> pd.DataFrame:
    """R ``process_att_gt``-style normalization of group-time output."""
    return group_time_att(att_gt)


@dataclass
class PTEAggregateResult:
    estimate: float
    weights: pd.DataFrame
    type: str = "group"
    standard_error: float = float("nan")
    conf_int: tuple[float, float] = (float("nan"), float("nan"))

    def to_dataframe(self) -> pd.DataFrame:
        out = self.weights.copy()
        out["estimate"] = self.estimate
        out["se"] = self.standard_error
        return out

    def to_dict(self) -> dict[str, object]:
        return {
            "estimate": self.estimate,
            "se": self.standard_error,
            "conf_int": self.conf_int,
            "type": self.type,
            "weights": self.weights.to_dict(orient="records"),
        }


@dataclass
class DoseResult:
    """Container for dose-response ATT/ACRT curves.

    Mirrors R ``ptetools::dose_obj``. ``att_d`` / ``acrt_d`` are either the
    single-column DataFrames accepted by ``pte_dose_results`` (``dose``+
    ``att``) or the rich per-dose tables produced by ``process_dose_gt``
    (``dose``/``att``/``se``/``crit``).
    """

    dose: Any
    overall_att: Optional[float] = None
    overall_att_se: Optional[float] = None
    att_d: Optional[pd.DataFrame] = None
    acrt_d: Optional[pd.DataFrame] = None
    overall_acrt: Optional[float] = None
    overall_acrt_se: Optional[float] = None
    overall_att_inffunc: Optional[np.ndarray] = None
    overall_acrt_inffunc: Optional[np.ndarray] = None
    att_d_se: Optional[np.ndarray] = None
    att_d_crit: Optional[float] = None
    att_d_inffunc: Optional[np.ndarray] = None
    acrt_d_se: Optional[np.ndarray] = None
    acrt_d_crit: Optional[float] = None
    acrt_d_inffunc: Optional[np.ndarray] = None
    simultaneous: bool = False
    alp: float = 0.05
    biters: int = 100

    def to_dict(self) -> dict[str, object]:
        def frame(x: Optional[pd.DataFrame]) -> Optional[list[dict[str, object]]]:
            return None if x is None else x.to_dict(orient="records")

        def array(x: Optional[np.ndarray]) -> Optional[list[float]]:
            return None if x is None else np.asarray(x, float).tolist()

        return {
            "dose": np.asarray(self.dose).tolist(),
            "overall_att": self.overall_att,
            "overall_att_se": self.overall_att_se,
            "overall_acrt": self.overall_acrt,
            "overall_acrt_se": self.overall_acrt_se,
            "att_d": frame(self.att_d),
            "att_d_se": array(self.att_d_se),
            "att_d_crit": self.att_d_crit,
            "acrt_d": frame(self.acrt_d),
            "acrt_d_se": array(self.acrt_d_se),
            "acrt_d_crit": self.acrt_d_crit,
            "simultaneous": self.simultaneous,
        }

    def summary(self) -> pd.DataFrame:
        if self.att_d is not None:
            return self.att_d.copy()
        return pd.DataFrame({"dose": np.asarray(self.dose)})


def dose_rich_table(
    dose: Any,
    est: np.ndarray,
    se: Optional[np.ndarray],
    crit: Optional[float],
) -> pd.DataFrame:
    """Build the rich ``dose``/``att``/``se``/``crit`` table stored on ``att_d``."""
    out = pd.DataFrame({"dose": np.asarray(dose)})
    out["att"] = np.asarray(est)
    if se is not None:
        out["se"] = np.squeeze(np.asarray(se))
    if crit is not None:
        out["crit"] = float(crit)
    return out


def dose_obj(
    dose: Any,
    *,
    overall_att: Optional[float] = None,
    overall_att_se: Optional[float] = None,
    att_d: Optional[pd.DataFrame] = None,
    acrt_d: Optional[pd.DataFrame] = None,
    **_: Any,
) -> DoseResult:
    """Construct a dose-response result container."""
    return DoseResult(dose, overall_att, overall_att_se, att_d, acrt_d)


def pte_dose_results(
    dose: Any,
    att_d: pd.DataFrame,
    *,
    overall_att: Optional[float] = None,
    overall_att_se: Optional[float] = None,
) -> DoseResult:
    """Construct a dose result from an ATT-by-dose table."""
    return dose_obj(dose, overall_att=overall_att, overall_att_se=overall_att_se, att_d=att_d)


def aggte_obj(
    estimate: float,
    weights: pd.DataFrame,
    *,
    type: str = "group",
    standard_error: float = float("nan"),
    conf_int: tuple[float, float] = (float("nan"), float("nan")),
) -> PTEAggregateResult:
    """Construct an aggregate treatment-effect result container."""
    return PTEAggregateResult(estimate, weights, type, standard_error, conf_int)


@dataclass
class PTEResults:
    """Results from the generic group-time ATT loop."""

    att_gt: pd.DataFrame
    overall_att: float
    overall_se: float
    influence_functions: Optional[np.ndarray] = None
    cohort_weights: Optional[dict[Any, float]] = None
    bootstrap_distribution: Optional[np.ndarray] = None
    overall_conf_int: tuple[float, float] = (float("nan"), float("nan"))

    def to_dataframe(self) -> pd.DataFrame:
        return self.att_gt.copy()

    def aggregate(self, type: str = "group") -> PTEAggregateResult:
        return pte_aggte(self.att_gt, type=type, cohort_weights=self.cohort_weights)

    def to_dict(self) -> dict[str, object]:
        return {
            "overall_att": self.overall_att,
            "overall_se": self.overall_se,
            "overall_conf_int": self.overall_conf_int,
            "att_gt": self.att_gt.to_dict(orient="records"),
            "cohort_weights": self.cohort_weights,
        }

    def summary(self) -> str:
        return (
            f"PTEResults(ATT={self.overall_att:.6f}, "
            f"SE={self.overall_se:.6f}, CI={self.overall_conf_int}, cells={len(self.att_gt)})"
        )


def crit_val_checks(crit_val: float, alpha: float = 0.05) -> tuple[float, bool]:
    """Validate a simultaneous critical value and return ``(value, cband)``."""
    pointwise = float(norm.ppf(1 - alpha / 2))
    if not np.isfinite(crit_val) or crit_val < pointwise:
        return pointwise, False
    return float(crit_val), True


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


def setup_pte_basic(
    data: pd.DataFrame,
    yname: str,
    gname: str,
    tname: str,
    idname: Optional[str] = None,
    *,
    panel: bool = True,
) -> PTEParams:
    """Basic R ``setup_pte_basic``-compatible panel description."""
    return setup_pte(data, yname, gname, tname, idname, panel=panel)


def pte_params(
    data: pd.DataFrame,
    yname: str,
    gname: str,
    tname: str,
    idname: Optional[str] = None,
    *,
    panel: bool = True,
    anticipation: int = 0,
    base_period: str = "varying",
) -> PTEParams:
    """R ``pte_params``-style constructor backed by ``setup_pte``."""
    return setup_pte(
        data,
        yname,
        gname,
        tname,
        idname,
        panel=panel,
        anticipation=anticipation,
        base_period=base_period,
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


def two_by_two_rcs_subset(
    data: pd.DataFrame,
    g: Any,
    tp: Any,
    *,
    gname: str = "G",
    tname: str = "period",
    yname: str = "Y",
    control_group: str = "notyettreated",
    anticipation: int = 0,
    base_period: str = "varying",
    covariates: Sequence[str] = (),
) -> TwoByTwoSubset:
    """Construct a two-period repeated-cross-section comparison."""
    if control_group not in {"notyettreated", "nevertreated"}:
        raise ValueError("control_group must be 'notyettreated' or 'nevertreated'")
    pre = g - anticipation - 1 if base_period == "universal" else tp - 1
    cohort = data[gname]
    if control_group == "nevertreated":
        keep = cohort.isin([0, g])
    else:
        keep = cohort.isin([0, g]) | (cohort > tp)
    keep &= data[tname].isin([pre, tp])
    columns = [gname, tname, yname] + list(covariates)
    missing = sorted(set(columns).difference(data.columns))
    if missing:
        raise ValueError(f"data is missing columns: {missing}")
    out = data.loc[keep, columns].copy().reset_index(drop=True)
    out = out.rename(columns={gname: "G", tname: "period", yname: "Y"})
    out["id"] = np.arange(len(out))
    out["name"] = np.where(out["period"].eq(tp), "post", "pre")
    out["D"] = (out["G"] == g).astype(int)
    if out.empty or out["D"].sum() == 0 or (out["D"] == 0).sum() == 0:
        raise ValueError("two_by_two_rcs_subset has no treated or comparison observations")
    return TwoByTwoSubset(gt_data_frame(out), int(out["D"].sum()), np.ones(len(out), dtype=bool))


def keep_all_untreated_subset(data: pd.DataFrame, g: Any, tp: Any) -> TwoByTwoSubset:
    """Keep all untreated history plus cohort ``g`` through period ``tp``."""
    treated_now = (data["G"] <= data["period"]) & data["G"].ne(0)
    keep = (~treated_now) | data["G"].eq(g)
    keep &= ~(data["G"].eq(g) & data["period"].gt(tp))
    out = data.loc[keep].copy()
    out["name"] = np.where(out["period"].eq(tp), "post", "pre")
    out["D"] = ((out["G"] == g) & (out["period"] >= tp)).astype(int)
    ids = pd.unique(data["id"])
    return TwoByTwoSubset(
        gt_data_frame(out), int(out["id"].nunique()), np.isin(ids, pd.unique(out["id"]))
    )


def keep_all_pretreatment_subset(data: pd.DataFrame, g: Any, tp: Any) -> TwoByTwoSubset:
    """Keep all pre-treatment history through ``tp`` for eligible cohorts."""
    out = data.loc[data["period"] <= tp].copy()
    out = out.loc[out["G"].eq(g) | out["G"].gt(tp) | out["G"].eq(0)].copy()
    out["name"] = np.where(out["period"].eq(tp), "post", "pre")
    out["D"] = ((out["G"] == g) & (out["period"] >= tp)).astype(int)
    ids = pd.unique(data["id"])
    return TwoByTwoSubset(
        gt_data_frame(out), int(out["id"].nunique()), np.isin(ids, pd.unique(out["id"]))
    )


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


def pte_attgt(
    gt_data: GTDataFrame | pd.DataFrame, *, covariates: Sequence[str] = ()
) -> ATTGTResult:
    """Alias for the panel ATT(g,t) step used by ``pte_default``."""
    return did_attgt(gt_data, covariates=covariates)


def did_rcs_attgt(
    gt_data: GTDataFrame | pd.DataFrame, *, covariates: Sequence[str] = ()
) -> ATTGTResult:
    """Estimate an RCS ATT(g,t) from period-specific group means."""
    frame = gt_data.data if isinstance(gt_data, GTDataFrame) else gt_data
    if covariates:
        raise NotImplementedError("RCS covariate adjustment is not implemented")
    treated = frame["D"].eq(1)
    post = frame["name"].eq("post")
    control = ~treated
    if not treated.any() or not control.any():
        raise ValueError("both treated and comparison observations are required")
    delta_treated = frame.loc[treated & post, "Y"].mean() - frame.loc[treated & ~post, "Y"].mean()
    delta_control = frame.loc[control & post, "Y"].mean() - frame.loc[control & ~post, "Y"].mean()
    att = float(delta_treated - delta_control)
    inf = np.zeros(len(frame), dtype=float)
    n_treated = treated.sum() / 2
    n_control = control.sum() / 2
    inf[treated & post] = (
        frame.loc[treated & post, "Y"] - frame.loc[treated & post, "Y"].mean()
    ) / n_treated
    inf[treated & ~post] = (
        -(frame.loc[treated & ~post, "Y"] - frame.loc[treated & ~post, "Y"].mean()) / n_treated
    )
    inf[control & post] = (
        -(frame.loc[control & post, "Y"] - frame.loc[control & post, "Y"].mean()) / n_control
    )
    inf[control & ~post] = (
        frame.loc[control & ~post, "Y"] - frame.loc[control & ~post, "Y"].mean()
    ) / n_control
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
    idname: Optional[str] = None,
    panel: bool = True,
    control_group: str = "notyettreated",
    anticipation: int = 0,
    base_period: str = "varying",
    covariates: Sequence[str] = (),
    bstrap: bool = False,
    biters: int = 100,
    seed: Optional[int] = None,
) -> PTEResults:
    """Run the generic unadjusted panel ATT(g,t) loop."""
    if not panel:
        data = data.copy()
        idname = idname or "_pte_rowid"
        data[idname] = np.arange(len(data))
    if idname is None:
        raise ValueError("idname is required when panel=True")
    params = setup_pte(
        data,
        yname,
        gname,
        tname,
        idname,
        panel=panel,
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
            if panel:
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
            else:
                subset = two_by_two_rcs_subset(
                    data,
                    g,
                    tp,
                    gname=gname,
                    tname=tname,
                    yname=yname,
                    control_group=control_group,
                    anticipation=anticipation,
                    base_period=base_period,
                    covariates=covariates,
                )
                result = did_rcs_attgt(subset.gt_data, covariates=covariates)
            if result.inf_func is None:
                raise RuntimeError("did_attgt did not return an influence function")
            finite_if = result.inf_func[np.isfinite(result.inf_func)]
            se = (
                float(np.sqrt(np.mean(finite_if**2) / len(result.inf_func)))
                if finite_if.size
                else float("nan")
            )
            rows.append({"group": g, "time": tp, "attgt": result.attgt, "se": se})
            full_if = np.full(n_units, np.nan)
            full_if[subset.disidx] = result.inf_func
            influence.append(full_if)
    att_gt = pd.DataFrame(rows)
    unit_groups = data.groupby(idname, sort=False)[gname].first()
    treated_groups = unit_groups[unit_groups != 0]
    cohort_weights = (treated_groups.value_counts() / len(treated_groups)).to_dict()
    weights = pte_aggte(att_gt, type="group", cohort_weights=cohort_weights).weights
    valid = np.isfinite(att_gt["attgt"]) & (weights["overall_weight"] > 0)
    overall_att = float(np.sum(att_gt.loc[valid, "attgt"] * weights.loc[valid, "overall_weight"]))
    full_influence = np.asarray(influence, dtype=float).T if influence else None
    overall_se = float("nan")
    if bstrap:
        if not isinstance(biters, (int, np.integer)) or biters < 2:
            raise ValueError("biters must be an integer greater than or equal to 2")
        rng = np.random.default_rng(seed)
        bootstrap_att = []
        for _ in range(int(biters)):
            sampled_units = []
            bootstrap_groups = [gname] if panel else [gname, tname]
            for _, group_data in data.groupby(bootstrap_groups, sort=False):
                units = pd.unique(group_data[idname])
                sampled_units.extend(rng.choice(units, size=len(units), replace=True))
            pieces = []
            for draw, unit in enumerate(sampled_units):
                piece = data.loc[data[idname].eq(unit)].copy()
                piece[idname] = draw
                pieces.append(piece)
            sampled = pd.concat(pieces, ignore_index=True)
            bootstrap_att.append(
                pte(
                    sampled,
                    yname=yname,
                    gname=gname,
                    tname=tname,
                    idname=idname,
                    control_group=control_group,
                    anticipation=anticipation,
                    base_period=base_period,
                    covariates=covariates,
                    bstrap=False,
                ).overall_att
            )
        overall_se = float(np.std(bootstrap_att, ddof=1))
    distribution = np.asarray(bootstrap_att, dtype=float) if bstrap else None
    conf_int = (
        tuple(np.quantile(distribution, [0.025, 0.975]))
        if distribution is not None
        else (float("nan"), float("nan"))
    )
    return PTEResults(
        att_gt, overall_att, overall_se, full_influence, cohort_weights, distribution, conf_int
    )


def pte_default(
    data: pd.DataFrame,
    *,
    yname: str,
    gname: str,
    tname: str,
    idname: str,
    covariates: Sequence[str] = (),
    control_group: str = "notyettreated",
    anticipation: int = 0,
    base_period: str = "varying",
    bstrap: bool = False,
    biters: int = 100,
    seed: Optional[int] = None,
) -> PTEResults:
    """R ``pte_default``-style wrapper around the generic panel estimator."""
    return pte(
        data,
        yname=yname,
        gname=gname,
        tname=tname,
        idname=idname,
        covariates=covariates,
        control_group=control_group,
        anticipation=anticipation,
        base_period=base_period,
        bstrap=bstrap,
        biters=biters,
        seed=seed,
    )


def pte_results(
    att_gt: pd.DataFrame,
    overall_att: float,
    overall_se: float = float("nan"),
) -> PTEResults:
    """Construct a ``PTEResults`` object from aggregate inputs."""
    return PTEResults(group_time_att(att_gt), overall_att, overall_se)


def pte_emp_boot(
    data: pd.DataFrame,
    **kwargs: Any,
) -> PTEResults:
    """R ``pte_emp_boot``-style wrapper for empirical bootstrap results."""
    return panel_empirical_bootstrap(data, **kwargs)


def panel_empirical_bootstrap(data: pd.DataFrame, **kwargs: Any) -> PTEResults:
    """Run the panel empirical bootstrap through the generic ``pte`` loop."""
    kwargs = dict(kwargs)
    kwargs["bstrap"] = True
    return pte(data, **kwargs)


def mboot2(
    influence_functions: np.ndarray,
    *,
    biters: int = 100,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate multiplier-bootstrap draws from an influence-function matrix."""
    if influence_functions.ndim != 2:
        raise ValueError("influence_functions must be a two-dimensional array")
    if biters < 2:
        raise ValueError("biters must be at least 2")
    rng = np.random.default_rng(seed)
    multipliers = rng.normal(size=(int(biters), influence_functions.shape[0]))
    return multipliers @ influence_functions / influence_functions.shape[0]


def _type1_quantile(values: np.ndarray, q: float) -> float:
    """R ``quantile(..., type=1)`` — inverse of the empirical distribution function."""
    values = np.sort(np.asarray(values, dtype=float))
    n = values.size
    if n == 0:
        return float("nan")
    if n == 1:
        return float(values[0])
    j = n * q
    if j < 1:
        return float(values[0])
    if np.isclose(j, round(j)):
        return float(values[max(int(j) - 1, 0)])
    return float(values[int(np.floor(j))])


def mboot_se_and_crit(
    draws: np.ndarray,
    *,
    alp: float = 0.05,
    cband: bool = True,
) -> tuple[np.ndarray, float, bool]:
    """Convert ``mboot2`` draws into R-style bootstrap SEs and a sup-t critical value.

    ``draws`` is the ``(biters, n_cols)`` matrix returned by ``mboot2`` (i.e.
    the ``colMeans(ub * inffunc)`` terms R multiplies by ``sqrt(n)`` before
    computing its IQR-based standard errors — the ``sqrt(n)`` factors cancel).
    Returns ``(se, crit_val, cband_ok)`` following ``process_att_gt::mboot2``.
    """
    draws = np.asarray(draws, dtype=float)
    if draws.ndim != 2:
        raise ValueError("draws must be a two-dimensional array")
    iqr_scale = norm.ppf(0.75) - norm.ppf(0.25)
    se = np.array(
        [(_type1_quantile(col, 0.75) - _type1_quantile(col, 0.25)) / iqr_scale for col in draws.T]
    )
    finite_se = np.all(np.isfinite(se)) and np.all(se > 0)
    if finite_se:
        sup_t = np.max(np.abs(draws / se), axis=1)
        crit_val = _type1_quantile(sup_t, 1 - alp)
    else:
        crit_val = float("nan")
    return se, float(crit_val), bool(finite_se and crit_val >= norm.ppf(1 - alp / 2))


def _weighted_combine_list(entries: Sequence[Any], weights: np.ndarray) -> np.ndarray:
    """``BMisc::weighted_combine_list`` — normalize weights, then sum ``w_i * entry_i``."""
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if total == 0:
        raise ValueError("weights sum to zero")
    weights = weights / total
    first = np.asarray(entries[0], dtype=float) * weights[0]
    for entry, weight in zip(entries[1:], weights[1:]):
        first = first + np.asarray(entry, dtype=float) * weight
    return first


def bspline_basis(
    x: Any,
    *,
    degree: int = 3,
    knots: Optional[Sequence[float]] = None,
    derivative: int = 0,
    intercept: bool = False,
) -> np.ndarray:
    """B-spline design matrix matching ``splines2::bSpline`` / ``splines2::dbs``.

    Boundary knots are the range of ``x`` (clamped, multiplicity ``degree+1``);
    ``intercept=False`` (matching splines2's default) drops the first basis
    function, so the returned matrix has ``degree + len(knots)`` columns, the
    convention R's ``process_dose_gt`` relies on before ``cbind``-ing a
    constant column.
    """
    x = np.asarray(x, dtype=float)
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if derivative not in {0, 1}:
        raise ValueError("derivative must be 0 or 1")
    knots = np.asarray([], dtype=float) if knots is None else np.asarray(knots, dtype=float)
    if np.any((knots <= x.min()) | (knots >= x.max())):
        raise ValueError("interior knots must lie strictly inside the range of x")
    if np.any(np.diff(knots) <= 0):
        raise ValueError("knots must be strictly increasing")
    t = np.concatenate(
        [
            np.repeat(x.min(), degree + 1),
            knots,
            np.repeat(x.max(), degree + 1),
        ]
    )
    n_coeff = len(t) - degree - 1
    if derivative == 0:
        design = BSpline.design_matrix(x, t, degree).toarray()
    else:
        td = t[1:-1]
        kd = degree - 1
        transform = np.zeros((n_coeff - 1, n_coeff))
        for j in range(n_coeff - 1):
            denom = t[j + degree + 1] - t[j + 1]
            transform[j, j] = -degree / denom
            transform[j, j + 1] = degree / denom
        design = np.column_stack([BSpline(td, transform[:, j], kd)(x) for j in range(n_coeff)])
    if not intercept:
        design = design[:, 1:]
    return np.asarray(design, dtype=float)


def _cell_results(gt_results: Any) -> list[dict[str, Any]]:
    """Read the per-cell ``extra_gt_returns`` entries off a ``gt_results`` dict."""
    raw = gt_results["extra_gt_returns"]
    out = []
    for entry in raw:
        inner = entry["extra_gt_returns"]
        required = {"att.d", "acrt.d", "att.overall", "acrt.overall", "bread", "Xe"}
        missing = sorted(required.difference(inner))
        if missing:
            raise ValueError(f"dose cell results are missing: {missing}")
        out.append(inner)
    return out


def process_dose_gt(
    gt_results: dict[str, Any],
    ptep: dict[str, Any],
    *,
    seed: Optional[int] = None,
) -> DoseResult:
    """Combine per-cell dose results into ATT(d) / ACRT(d) curves and overall effects.

    Mirrors R ``ptetools::process_dose_gt``. ``gt_results`` carries the
    group-time loop output — ``inffunc`` (the ``n x n_cells`` influence-function
    matrix, zero-padded off-support rows per the R ``compute.pte`` convention),
    ``attgt_list`` (``group``/``time.period``/``att``) and ``extra_gt_returns``
    whose nested ``extra_gt_returns`` give ``att.d``, ``acrt.d``, ``att.overall``,
    ``acrt.overall``, ``bet``, ``bread`` and ``Xe`` for each cell. ``ptep`` is a
    dict of parameters: ``data``/``yname``/``gname``/``tname``/``idname`` (panel
    fields), ``anticipation``, ``base_period``, ``control_group``, ``dvals``,
    ``degree``, ``knots``, ``biters``, ``alp``, ``cband`` and ``bstrap``.

    Dose standard errors always come from the multiplier bootstrap, matching R.
    """
    if not isinstance(gt_results, dict):
        raise TypeError("gt_results must be a dict")
    ptep = dict(ptep)
    for key in ("data", "yname", "gname", "tname"):
        if key not in ptep:
            raise ValueError(f"ptep is missing required field: {key}")

    def opt(key: str, default: Any) -> Any:
        return ptep.get(key, default)

    attgt_list = gt_results["attgt_list"]
    att_gt = pd.DataFrame(
        {
            "group": [cell["group"] for cell in attgt_list],
            "time": [cell["time.period"] for cell in attgt_list],
            "attgt": [cell["att"] for cell in attgt_list],
        }
    )
    o_weights = overall_weights(att_gt)
    o_weight = o_weights["overall_weight"].to_numpy(float)

    cells = _cell_results(gt_results)
    groups = [entry["group"] for entry in gt_results["extra_gt_returns"]]
    times = [entry["time.period"] for entry in gt_results["extra_gt_returns"]]
    if not (
        np.array_equal(groups, o_weights["group"]) and np.array_equal(times, o_weights["time"])
    ):
        raise ValueError(
            "in processing dose results, mismatch between order of groups and time periods"
        )

    att_d_gt = [cell["att.d"] for cell in cells]
    acrt_d_gt = [cell["acrt.d"] for cell in cells]
    att_overall_gt = np.asarray([cell["att.overall"] for cell in cells], dtype=float)
    acrt_overall_gt = np.asarray([cell["acrt.overall"] for cell in cells], dtype=float)
    bread_gt = [cell["bread"] for cell in cells]
    Xe_gt = [np.asarray(cell["Xe"], dtype=float) for cell in cells]

    acrt_gt_inffunc = np.asarray(gt_results["inffunc"], dtype=float)
    if acrt_gt_inffunc.ndim != 2:
        raise ValueError("gt_results['inffunc'] must be a two-dimensional array")
    n_units = acrt_gt_inffunc.shape[0]
    if acrt_gt_inffunc.shape[1] != att_overall_gt.size:
        raise ValueError("gt_results['inffunc'] must have one column per group-time cell")

    biters = int(opt("biters", 100))
    alp = float(opt("alp", 0.05))
    cband = bool(opt("cband", True))
    if biters < 2:
        raise ValueError("biters must be an integer greater than or equal to 2")

    # ------------------------------------------------------------------
    # overall ATT: recomputed through the generic pte loop (R's self-call
    # to pte_default), then sanity-checked against the cell contributions.
    # ------------------------------------------------------------------
    att_res = pte(
        ptep["data"],
        yname=ptep["yname"],
        gname=ptep["gname"],
        tname=ptep["tname"],
        idname=ptep.get("idname"),
        panel=bool(opt("panel", True)),
        control_group=opt("control_group", "notyettreated"),
        anticipation=int(opt("anticipation", 0)),
        base_period=opt("base_period", "varying"),
        covariates=(),
        bstrap=False,
    )
    overall_att = float(att_res.overall_att)
    att_inffunc = np.nan_to_num(np.asarray(att_res.influence_functions, dtype=float), nan=0.0)
    if att_inffunc.shape[1] != att_overall_gt.size:
        raise ValueError("influence function matrix does not align with group-time cells")
    overall_att_inffunc = att_inffunc @ (o_weight / o_weight.sum())
    overall_att_se = float(
        mboot_se_and_crit(
            mboot2(overall_att_inffunc[:, None], biters=biters, seed=seed),
            alp=alp,
            cband=False,
        )[0][0]
    )
    if not np.isclose(overall_att, float(np.average(att_overall_gt, weights=o_weight))):
        raise ValueError("failed sanity check: something off with calculating overall att")

    # ------------------------------------------------------------------
    # overall ACRT
    # ------------------------------------------------------------------
    overall_acrt = float(np.average(acrt_overall_gt, weights=o_weight))
    overall_acrt_inffunc = acrt_gt_inffunc @ (o_weight / o_weight.sum())
    overall_acrt_se = float(
        mboot_se_and_crit(
            mboot2(overall_acrt_inffunc[:, None], biters=biters, seed=seed),
            alp=alp,
            cband=False,
        )[0][0]
    )

    # point estimates of ATT(d) and ACRT(d)
    att_d = _weighted_combine_list(att_d_gt, o_weight)
    acrt_d = _weighted_combine_list(acrt_d_gt, o_weight)

    dvals = np.asarray(opt("dvals", None), dtype=float)
    if dvals is None or dvals.size == 0:
        raise ValueError("ptep['dvals'] must be a non-empty vector of dose values")
    degree = int(opt("degree", 3))
    knots = opt("knots", None)
    if knots is None:
        knots = np.array([], dtype=float)
    bs_grid = np.column_stack(
        [np.ones(dvals.size), bspline_basis(dvals, degree=degree, knots=knots)]
    )
    bs_deriv = np.column_stack(
        [np.zeros(dvals.size), bspline_basis(dvals, degree=degree, knots=knots, derivative=1)]
    )

    # per-cell influence functions for ATT(d)
    n1_vec = np.array([x.shape[0] for x in Xe_gt])
    keep_mat = acrt_gt_inffunc != 0
    if not np.array_equal(keep_mat.sum(axis=0), n1_vec):
        raise ValueError("something off with overall influence function")
    keep_mat2 = (att_inffunc != 0) & (~keep_mat)
    comparison_inffunc = np.where(keep_mat2, att_inffunc, 0.0)
    att_d_gt_inffunc = []
    for i, x in enumerate(Xe_gt):
        out = np.zeros((n_units, dvals.size))
        this_inffunc = x @ bread_gt[i] @ bs_grid.T
        out[keep_mat[:, i], :] = (n_units / n1_vec[i]) * this_inffunc
        out[keep_mat2[:, i], :] = -comparison_inffunc[keep_mat2[:, i], i][:, None]
        att_d_gt_inffunc.append(out)
    att_d_inffunc = _weighted_combine_list(att_d_gt_inffunc, o_weight)

    att_d_se, att_d_crit_val, att_cband_ok = mboot_se_and_crit(
        mboot2(att_d_inffunc, biters=biters, seed=seed), alp=alp, cband=cband
    )
    if cband and att_cband_ok:
        att_d_crit_val = float(crit_val_checks(att_d_crit_val, alp)[0])
    elif not cband:
        att_d_crit_val = float(norm.ppf(1 - alp / 2))

    # per-cell influence functions for ACRT(d): same but derivative basis,
    # no comparison-group contribution
    acrt_d_gt_inffunc = []
    for i, x in enumerate(Xe_gt):
        out = np.zeros((n_units, dvals.size))
        this_inffunc = x @ bread_gt[i] @ bs_deriv.T
        out[keep_mat[:, i], :] = (n_units / n1_vec[i]) * this_inffunc
        acrt_d_gt_inffunc.append(out)
    acrt_d_inffunc = _weighted_combine_list(acrt_d_gt_inffunc, o_weight)

    acrt_d_se, acrt_d_crit_val, acrt_cband_ok = mboot_se_and_crit(
        mboot2(acrt_d_inffunc, biters=biters, seed=seed), alp=alp, cband=cband
    )
    if cband and acrt_cband_ok:
        acrt_d_crit_val = float(crit_val_checks(acrt_d_crit_val, alp)[0])
    elif not cband:
        acrt_d_crit_val = float(norm.ppf(1 - alp / 2))

    simultaneous = bool(cband and att_cband_ok and acrt_cband_ok)
    return DoseResult(
        dose=dvals,
        overall_att=overall_att,
        overall_att_se=overall_att_se,
        overall_acrt=overall_acrt,
        overall_acrt_se=overall_acrt_se,
        overall_att_inffunc=overall_att_inffunc,
        overall_acrt_inffunc=overall_acrt_inffunc,
        att_d=dose_rich_table(dvals, att_d, att_d_se, att_d_crit_val),
        att_d_se=att_d_se,
        att_d_crit=att_d_crit_val,
        att_d_inffunc=att_d_inffunc,
        acrt_d=dose_rich_table(dvals, acrt_d, acrt_d_se, acrt_d_crit_val),
        acrt_d_se=acrt_d_se,
        acrt_d_crit=acrt_d_crit_val,
        acrt_d_inffunc=acrt_d_inffunc,
        simultaneous=simultaneous,
        alp=alp,
        biters=biters,
    )


def pte_aggte(
    attgt: pd.DataFrame,
    *,
    type: str = "group",
    cohort_weights: Optional[dict[Any, float]] = None,
) -> PTEAggregateResult:
    """Aggregate an ATT(g,t) table using group or dynamic weights."""
    if type not in {"group", "dynamic"}:
        raise ValueError("type must be 'group' or 'dynamic'")
    frame = attgt.copy()
    if type == "group":
        if cohort_weights is None:
            weights = overall_weights(frame)
        else:
            frame = frame.rename(columns={"group": "group", "time": "time"})
            post = (frame["group"] != 0) & (frame["time"] >= frame["group"])
            post_counts = frame.loc[post].groupby("group")["time"].transform("count")
            frame["overall_weight"] = 0.0
            frame.loc[post, "overall_weight"] = [
                cohort_weights.get(g, 0.0) / count
                for g, count in zip(frame.loc[post, "group"], post_counts)
            ]
            weights = frame[["group", "time", "overall_weight"]]
    else:
        required = {"group", "time", "attgt"}
        if not required.issubset(frame.columns):
            raise ValueError("dynamic aggregation requires group, time, and attgt columns")
        frame["event_time"] = frame["time"] - frame["group"]
        frame = frame.loc[frame["event_time"] >= 0].copy()
        if cohort_weights is None:
            counts = frame["group"].value_counts().astype(float)
            cohort_weights = (counts / counts.sum()).to_dict()
        frame["cohort_weight"] = frame["group"].map(cohort_weights).fillna(0.0)
        frame["overall_weight"] = frame.groupby("event_time")["cohort_weight"].transform(
            lambda values: values / values.sum() if values.sum() > 0 else values
        )
        weights = frame[["group", "time", "overall_weight"]]
    effects = frame["attgt"].to_numpy(float)
    w = weights["overall_weight"].to_numpy(float)
    if len(effects) != len(w):
        effects = frame.loc[weights.index, "attgt"].to_numpy(float)
    return PTEAggregateResult(float(np.nansum(effects * w)), weights.reset_index(drop=True), type)


def attgt_pte_aggregations(
    attgt: pd.DataFrame, *, type: str = "group", **kwargs: Any
) -> PTEAggregateResult:
    """Dispatch the standard ATT(g,t) aggregation path."""
    return pte_aggte(attgt, type=type, **kwargs)
