"""Small, composable panel-treatment-effects primitives.

The API mirrors the infrastructure exposed by R ``ptetools``.  Estimators in
this module are intentionally separate from the high-level estimator classes:
``setup_pte`` describes a panel, ``two_by_two_subset`` creates one group-time
comparison, and ``did_attgt`` estimates the resulting two-period ATT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

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
    by_event_time: Optional[pd.DataFrame] = None
    bootstrap_distribution: Optional[np.ndarray] = None

    def to_dataframe(self) -> pd.DataFrame:
        if self.by_event_time is not None:
            return self.by_event_time.copy()
        out = self.weights.copy()
        out["estimate"] = self.estimate
        out["se"] = self.standard_error
        out["conf_int_lower"] = self.conf_int[0]
        out["conf_int_upper"] = self.conf_int[1]
        return out

    def to_dict(self) -> dict[str, object]:
        return {
            "estimate": self.estimate,
            "se": self.standard_error,
            "conf_int": self.conf_int,
            "type": self.type,
            "weights": self.weights.to_dict(orient="records"),
            "by_event_time": (
                None if self.by_event_time is None else self.by_event_time.to_dict(orient="records")
            ),
            "bootstrap_distribution": (
                None
                if self.bootstrap_distribution is None
                else self.bootstrap_distribution.tolist()
            ),
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


def ggpte_cont(
    result: DoseResult,
    *,
    type: str = "att",
    show: bool = False,
    **kwargs: Any,
) -> Any:
    """Plot a dose result using the project's matplotlib dose-response API.

    This is the Python equivalent of the deprecated R ``ggpte_cont`` wrapper;
    the returned object is a matplotlib ``Axes`` (or a Plotly figure when
    ``backend='plotly'`` is passed).
    """
    if not isinstance(result, DoseResult):
        raise TypeError("result must be a DoseResult")
    target = {"att": "att_d", "acrt": "acrt_d"}.get(type)
    if target is None:
        raise ValueError("type must be 'att' or 'acrt'")
    table = getattr(result, target)
    if table is None:
        raise ValueError(f"DoseResult does not contain the {type.upper()} curve")
    estimate_name = "att" if type == "att" else "acrt"
    data = table.rename(columns={estimate_name: "effect"}).copy()
    if "effect" not in data.columns:
        raise ValueError(f"DoseResult {target} must contain '{estimate_name}'")
    if "crit" in data.columns and "se" in data.columns:
        data["conf_int_lower"] = data["effect"] - data["crit"] * data["se"]
        data["conf_int_upper"] = data["effect"] + data["crit"] * data["se"]
    from diff_diff.visualization import plot_dose_response

    return plot_dose_response(
        data=data,
        target=type,
        show=show,
        **kwargs,
    )


def ggpte(result: PTEResults, *, show: bool = False, **kwargs: Any) -> Any:
    """Plot the dynamic ATT surface of a ``PTEResults`` object."""
    if not isinstance(result, PTEResults):
        raise TypeError("result must be a PTEResults")
    frame = result.att_gt.copy()
    frame["event_time"] = frame["time"] - frame["group"]
    frame = frame.loc[frame["group"] != 0].copy()
    if frame.empty:
        raise ValueError("PTEResults has no treated event-study cells")
    cohort_weights = result.cohort_weights
    if cohort_weights is None:
        counts = frame["group"].value_counts().astype(float)
        cohort_weights = (counts / counts.sum()).to_dict()
    frame["cohort_weight"] = frame["group"].map(cohort_weights).fillna(0.0)
    frame["overall_weight"] = frame.groupby("event_time")["cohort_weight"].transform(
        lambda values: values / values.sum() if values.sum() > 0 else values
    )
    frame = frame.reset_index(drop=True)
    estimates = frame.groupby("event_time").apply(
        lambda values: float(np.sum(values["attgt"] * values["overall_weight"])),
        include_groups=False,
    )
    se: dict[Any, float] = {}
    if result.influence_functions is not None:
        influence = np.asarray(result.influence_functions, dtype=float)
        cell_weights = frame["overall_weight"].to_numpy(float)
        for event_time, positions in frame.groupby("event_time").groups.items():
            positions_array = np.asarray(list(positions), dtype=int)
            weighted_if = influence[:, positions_array] @ cell_weights[positions_array]
            se[event_time] = float(np.sqrt(np.nansum(weighted_if**2)))
    from diff_diff.visualization import plot_event_study

    return plot_event_study(
        effects=estimates.to_dict(),
        se=se or None,
        periods=list(estimates.index),
        pre_periods=[event_time for event_time in estimates.index if event_time < 0],
        post_periods=[event_time for event_time in estimates.index if event_time >= 0],
        title="Treatment Effects Over Event Time",
        show=show,
        **kwargs,
    )


def plot_qtt(
    result: PTEQTTResult,
    *,
    type: str = "overall",
    cband: bool = True,
    plot_probs: Sequence[float] = (0.5,),
    plot_ci: Optional[bool] = None,
    show: bool = False,
    ax: Any = None,
) -> Any:
    """Plot an overall or dynamic QTT curve."""
    if not isinstance(result, PTEQTTResult):
        raise TypeError("result must be a PTEQTTResult")
    if type not in {"overall", "dynamic"}:
        raise ValueError("type must be 'overall' or 'dynamic'")
    from diff_diff.visualization._common import _require_matplotlib

    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    frame = result.overall if type == "overall" else result.dynamic
    lower_name = "lower_ub" if cband else "lower_pw"
    upper_name = "upper_ub" if cband else "upper_pw"

    if type == "overall":
        ax.axhline(0.0, color="gray", linewidth=1)
        ax.plot(frame["probs"], frame["qtt"], marker="o", label="QTT")
        if lower_name in frame and upper_name in frame:
            ax.plot(frame["probs"], frame[lower_name], linestyle="--", color="gray")
            ax.plot(frame["probs"], frame[upper_name], linestyle="--", color="gray")
        ax.set_xlabel("Quantile")
        ax.set_ylabel("QTT")
        ax.set_xlim(0.0, 1.0)
        ax.set_title("Quantile Treatment Effects")
    else:
        available = set(frame["probs"].unique())
        selected = list(plot_probs)
        missing = sorted(set(selected).difference(available))
        if missing:
            raise ValueError(f"plot_probs value(s) not found: {missing}")
        if plot_ci is None:
            plot_ci = len(selected) == 1
        ax.axhline(0.0, color="gray", linewidth=1)
        ax.axvline(-0.5, color="gray", linestyle="--", linewidth=1)
        for prob in selected:
            curve = frame.loc[frame["probs"].eq(prob)].sort_values("e")
            line = ax.plot(curve["e"], curve["qtt"], marker="o", label=f"q={prob:g}")[0]
            if plot_ci and lower_name in curve and upper_name in curve:
                ax.errorbar(
                    curve["e"],
                    curve["qtt"],
                    yerr=[curve["qtt"] - curve[lower_name], curve[upper_name] - curve["qtt"]],
                    fmt="none",
                    ecolor=line.get_color(),
                    capsize=3,
                )
        ax.set_xlabel("Event Time")
        ax.set_ylabel("QTT")
        ax.set_title("Dynamic Quantile Treatment Effects")
        if len(selected) > 1:
            ax.legend(title="Quantile")
    if show:
        plt.show()
    return ax


def autoplot_pte_results(result: PTEResults, **kwargs: Any) -> Any:
    """Python-named counterpart of R ``autoplot.pte_results``."""
    return ggpte(result, **kwargs)


def plot_pte_results(result: PTEResults, **kwargs: Any) -> Any:
    """Python-named counterpart of R ``plot.pte_results``."""
    return ggpte(result, show=True, **kwargs)


def autoplot_pte_emp_boot(result: PTEResults, **kwargs: Any) -> Any:
    """Python-named counterpart of R ``autoplot.pte_emp_boot``."""
    return ggpte(result, **kwargs)


def plot_pte_emp_boot(result: PTEResults, **kwargs: Any) -> Any:
    """Python-named counterpart of R ``plot.pte_emp_boot``."""
    return ggpte(result, show=True, **kwargs)


def autoplot_pte_qtt(result: PTEQTTResult, **kwargs: Any) -> Any:
    """Python-named counterpart of R ``autoplot.pte_qtt``."""
    return plot_qtt(result, **kwargs)


def plot_pte_qtt(result: PTEQTTResult, **kwargs: Any) -> Any:
    """Python-named counterpart of R ``plot.pte_qtt``."""
    return plot_qtt(result, show=True, **kwargs)


def autoplot_dose_obj(result: DoseResult, **kwargs: Any) -> Any:
    """Python-named counterpart of R ``autoplot.dose_obj``."""
    return ggpte_cont(result, **kwargs)


def plot_dose_obj(result: DoseResult, **kwargs: Any) -> Any:
    """Python-named counterpart of R ``plot.dose_obj``."""
    return ggpte_cont(result, show=True, **kwargs)


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

    def to_dataframe(self, level: str = "att_gt") -> pd.DataFrame:
        """Return ATT(g,t) rows or a post-fit aggregate table."""
        if level == "att_gt":
            return self.att_gt.copy()
        if level in {"group", "dynamic"}:
            return self.aggregate(level).to_dataframe()
        raise ValueError("level must be 'att_gt', 'group', or 'dynamic'")

    def aggregate(
        self,
        type: str = "group",
        *,
        bstrap: bool = False,
        biters: int = 1000,
        seed: Optional[int] = None,
        alpha: float = 0.05,
    ) -> PTEAggregateResult:
        """Aggregate post-fit effects, optionally with multiplier bootstrap."""
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        if bstrap and (not isinstance(biters, (int, np.integer)) or biters < 2):
            raise ValueError("biters must be an integer greater than or equal to 2")
        aggregate = pte_aggte(self.att_gt, type=type, cohort_weights=self.cohort_weights)
        if self.influence_functions is None or aggregate.weights.empty:
            return aggregate

        inference_weights = aggregate.weights.copy()
        source_indices = []
        for group, time in zip(inference_weights["group"], inference_weights["time"]):
            matches = self.att_gt.index[
                self.att_gt["group"].eq(group) & self.att_gt["time"].eq(time)
            ]
            if len(matches) != 1:
                return aggregate
            source_indices.append(int(matches[0]))
        weights = inference_weights["overall_weight"].to_numpy(float)
        influence = np.asarray(self.influence_functions, dtype=float)
        if influence.ndim != 2 or max(source_indices) >= influence.shape[1]:
            return aggregate
        weighted_if = influence[:, source_indices] @ weights
        standard_error = float(np.sqrt(np.nansum(weighted_if**2)))
        critical = float(norm.ppf(0.975))
        conf_int = (
            float(aggregate.estimate - critical * standard_error),
            float(aggregate.estimate + critical * standard_error),
        )
        by_event_time = None
        bootstrap_distribution = None
        if type == "dynamic":
            event_frame = inference_weights.copy()
            event_frame["event_time"] = event_frame["time"] - event_frame["group"]
            event_rows = []
            event_ifs = []
            for event_time, event_group in event_frame.groupby("event_time", sort=True):
                positions = event_group.index.to_numpy(int)
                event_weights = event_group["overall_weight"].to_numpy(float)
                source_positions = np.asarray(source_indices)[positions]
                event_estimate = float(
                    np.sum(self.att_gt.iloc[source_positions]["attgt"] * event_weights)
                )
                event_if = influence[:, source_positions] @ event_weights
                event_se = float(np.sqrt(np.nansum(event_if**2)))
                event_ifs.append(event_if)
                event_rows.append(
                    {
                        "event_time": event_time,
                        "estimate": event_estimate,
                        "se": event_se,
                        "conf_int_lower": event_estimate - critical * event_se,
                        "conf_int_upper": event_estimate + critical * event_se,
                    }
                )
            by_event_time = pd.DataFrame(event_rows)
            if bstrap:
                rng = np.random.default_rng(seed)
                event_if_matrix = np.column_stack(event_ifs)
                draws = (
                    np.asarray(rng.standard_normal((int(biters), influence.shape[0])))
                    @ event_if_matrix
                )
                bootstrap_distribution = draws + by_event_time["estimate"].to_numpy(float)
                bootstrap_se = np.std(draws, axis=0, ddof=1)
                lower_pw = np.quantile(bootstrap_distribution, alpha / 2, axis=0)
                upper_pw = np.quantile(bootstrap_distribution, 1 - alpha / 2, axis=0)
                studentized = draws / np.where(bootstrap_se > 0, bootstrap_se, np.nan)
                abs_studentized = np.abs(studentized)
                row_max = np.max(
                    np.where(np.isfinite(abs_studentized), abs_studentized, -np.inf), axis=1
                )
                finite_row_max = row_max[np.isfinite(row_max)]
                critical = (
                    float(np.quantile(finite_row_max, 1 - alpha))
                    if finite_row_max.size
                    else float(norm.ppf(1 - alpha / 2))
                )
                by_event_time["se"] = bootstrap_se
                by_event_time["lower_pw"] = lower_pw
                by_event_time["upper_pw"] = upper_pw
                by_event_time["lower_ub"] = by_event_time["estimate"] - critical * bootstrap_se
                by_event_time["upper_ub"] = by_event_time["estimate"] + critical * bootstrap_se
        return PTEAggregateResult(
            estimate=aggregate.estimate,
            weights=aggregate.weights,
            type=aggregate.type,
            standard_error=standard_error,
            conf_int=conf_int,
            by_event_time=by_event_time,
            bootstrap_distribution=bootstrap_distribution,
        )

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


def attgt_noif(attgt: float, extra_gt_returns: Any = None) -> ATTGTResult:
    """Create the no-influence-function result used by R ``attgt_noif``."""
    return ATTGTResult(attgt=float(attgt), extra_gt_returns=extra_gt_returns)


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


def covid_attgt(
    gt_data: GTDataFrame | pd.DataFrame,
    *,
    covariates: Sequence[str] = (),
    d_covariates: Sequence[str] = (),
    d_outcome: bool = False,
) -> ATTGTResult:
    """Estimate the R ``ptetools::covid_attgt`` ATT(g,t).

    This is the Callaway--Li levels estimator: when ``d_outcome=False`` the
    outcome is the post-period level relative to a zero baseline; setting
    ``d_outcome=True`` uses the post-minus-pre outcome.  Pre-period covariates
    and optional covariate changes enter the DRDID panel score.  The score is
    delegated to the same DRDID-validated core used by
    :class:`CallawaySantAnna`.
    """
    frame = gt_data.data if isinstance(gt_data, GTDataFrame) else gt_data
    required = {"id", "D", "name", "Y"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"gt_data is missing required columns: {missing}")
    if not {"pre", "post"}.issubset(frame["name"].unique()):
        raise ValueError("gt_data must contain both pre and post observations")

    wide = frame.pivot_table(index="id", columns="name", values="Y", aggfunc="first")
    if wide[["pre", "post"]].isna().any().any():
        raise ValueError("each id must have one pre and one post outcome")
    treatment = frame.groupby("id", sort=False)["D"].first().reindex(wide.index).to_numpy(float)
    treated = treatment == 1
    control = treatment == 0
    if not treated.any() or not control.any():
        raise ValueError("both treated and comparison units are required")

    pre = frame.loc[frame["name"].eq("pre")].set_index("id").reindex(wide.index)
    post = frame.loc[frame["name"].eq("post")].set_index("id").reindex(wide.index)
    columns = list(covariates) + list(d_covariates)
    missing_covariates = sorted(set(columns).difference(frame.columns))
    if missing_covariates:
        raise ValueError(f"covariates are missing from gt_data: {missing_covariates}")
    if covariates:
        X = pre[list(covariates)].to_numpy(float)
    else:
        X = np.empty((len(wide), 0), dtype=float)
    if d_covariates:
        dX = post[list(d_covariates)].to_numpy(float) - pre[list(d_covariates)].to_numpy(float)
        X = np.column_stack([X, dX])

    outcome = (
        (wide["post"] - wide["pre"]).to_numpy(float) if d_outcome else wide["post"].to_numpy(float)
    )
    from diff_diff.staggered import CallawaySantAnna

    estimator = CallawaySantAnna(estimation_method="dr")
    # The score is used as a standalone cell primitive, outside fit(), where
    # CallawaySantAnna normally initializes this diagnostic accumulator.
    estimator._safe_inv_tracker = []
    att, _, inf_func = estimator._doubly_robust(
        outcome[treated], outcome[control], X[treated], X[control]
    )
    ordered_inf = np.empty(len(wide), dtype=float)
    ordered_inf[treated] = inf_func[: treated.sum()]
    ordered_inf[control] = inf_func[treated.sum() :]
    return attgt_if(att, ordered_inf)


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
    treated = frame["D"].eq(1)
    post = frame["name"].eq("post")
    control = ~treated
    if not treated.any() or not control.any():
        raise ValueError("both treated and comparison observations are required")
    if covariates:
        missing = sorted(set(covariates).difference(frame.columns))
        if missing:
            raise ValueError(f"covariates are missing from gt_data: {missing}")
        from diff_diff.staggered import CallawaySantAnna

        estimator = CallawaySantAnna(estimation_method="dr", panel=False)
        estimator._safe_inv_tracker = []
        y_gt = frame.loc[treated & post, "Y"].to_numpy(float)
        y_gs = frame.loc[treated & ~post, "Y"].to_numpy(float)
        y_ct = frame.loc[control & post, "Y"].to_numpy(float)
        y_cs = frame.loc[control & ~post, "Y"].to_numpy(float)
        X_gt = frame.loc[treated & post, list(covariates)].to_numpy(float)
        X_gs = frame.loc[treated & ~post, list(covariates)].to_numpy(float)
        X_ct = frame.loc[control & post, list(covariates)].to_numpy(float)
        X_cs = frame.loc[control & ~post, list(covariates)].to_numpy(float)
        att, _, inf_concat, _ = estimator._doubly_robust_rc(
            y_gt, y_gs, y_ct, y_cs, X_gt, X_gs, X_ct, X_cs
        )
        lengths = [len(y_gt), len(y_gs), len(y_ct), len(y_cs)]
        pieces = np.split(np.asarray(inf_concat, dtype=float), np.cumsum(lengths)[:-1])
        inf = np.zeros(len(frame), dtype=float)
        for mask, piece in zip(
            (treated & post, treated & ~post, control & post, control & ~post), pieces
        ):
            inf[mask.to_numpy()] = piece
        return attgt_if(att, inf)
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
    setup_pte_fun: Optional[Callable[..., Any]] = None,
    subset_fun: Optional[Callable[..., Any]] = None,
    attgt_fun: Optional[Callable[..., Any]] = None,
    aggte_fun: Optional[Callable[..., Any]] = None,
) -> PTEResults:
    """Run the generic group-time loop with optional custom callbacks.

    Custom callbacks receive ordinary Python objects: ``setup_pte_fun`` gets
    the panel metadata arguments and must return ``PTEParams``; ``subset_fun``
    gets ``(data, g, tp)`` and returns ``TwoByTwoSubset`` or a compatible
    ``GTDataFrame``; ``attgt_fun`` gets the selected ``GTDataFrame`` and must
    return ``ATTGTResult`` or a mapping with ``attgt`` and optional
    ``inf_func``; ``aggte_fun`` gets ``(att_gt, cohort_weights)`` and may
    return ``PTEAggregateResult``. Defaults reproduce the built-in R-style
    unadjusted path.
    """
    if not panel:
        data = data.copy()
        idname = idname or "_pte_rowid"
        data[idname] = np.arange(len(data))
    if idname is None:
        raise ValueError("idname is required when panel=True")
    if setup_pte_fun is None:
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
    else:
        params = setup_pte_fun(
            data,
            yname=yname,
            gname=gname,
            tname=tname,
            idname=idname,
            panel=panel,
            anticipation=anticipation,
            base_period=base_period,
        )
        if not isinstance(params, PTEParams):
            raise TypeError("setup_pte_fun must return PTEParams")
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
                subset = (
                    subset_fun(data, g, tp)
                    if subset_fun is not None
                    else two_by_two_subset(
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
                )
                if isinstance(subset, GTDataFrame):
                    subset = TwoByTwoSubset(subset, len(subset), np.ones(len(subset), dtype=bool))
                if not isinstance(subset, TwoByTwoSubset):
                    raise TypeError("subset_fun must return TwoByTwoSubset or GTDataFrame")
                result = (
                    attgt_fun(subset.gt_data)
                    if attgt_fun is not None
                    else did_attgt(subset.gt_data, covariates=covariates)
                )
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
                result = (
                    attgt_fun(subset.gt_data)
                    if attgt_fun is not None
                    else did_rcs_attgt(subset.gt_data, covariates=covariates)
                )
            if isinstance(result, dict):
                result = ATTGTResult(
                    float(result["attgt"]),
                    result.get("inf_func"),
                    result.get("extra_gt_returns"),
                )
            if not isinstance(result, ATTGTResult):
                raise TypeError("attgt_fun must return ATTGTResult or a mapping")
            if result.inf_func is None:
                raise RuntimeError("did_attgt did not return an influence function")
            finite_if = result.inf_func[np.isfinite(result.inf_func)]
            se = (
                float(np.sqrt(np.mean(finite_if**2) / len(result.inf_func)))
                if finite_if.size
                else float("nan")
            )
            rows.append({"group": g, "time": tp, "attgt": result.attgt, "se": se})
            # Mirror R's compute.pte influence surface: zero-pad off-support
            # units and scale the cell influence function by (n / n1) to adjust
            # for the relative size of the overall sample vs the cell.
            full_if = np.zeros(n_units)
            n1 = int(subset.n1) if subset.n1 else result.inf_func.size
            full_if[subset.disidx] = (n_units / n1) * result.inf_func
            influence.append(full_if)
    att_gt = pd.DataFrame(rows)
    unit_groups = data.groupby(idname, sort=False)[gname].first()
    treated_groups = unit_groups[unit_groups != 0]
    cohort_weights = (treated_groups.value_counts() / len(treated_groups)).to_dict()
    aggregate = (
        aggte_fun(att_gt, cohort_weights)
        if aggte_fun is not None
        else pte_aggte(att_gt, type="group", cohort_weights=cohort_weights)
    )
    if not isinstance(aggregate, PTEAggregateResult):
        raise TypeError("aggte_fun must return PTEAggregateResult")
    weights = aggregate.weights
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
                    setup_pte_fun=setup_pte_fun,
                    subset_fun=subset_fun,
                    attgt_fun=attgt_fun,
                    aggte_fun=aggte_fun,
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
    if j == np.floor(j):
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


# =============================================================================
# QTT (quantile treatment effects) machinery
#   Mirrors the ``gt_type = "qtt"`` / ``"qott"`` branch of R ``ptetools``:
#   per-cell ``(g,t)`` distributions F0/F1 (and Fte for QoTT) are mixed with
#   the R ``attgt_pte_aggregations`` weights and inverted at each quantile level
#   in ``probs``.  Standard errors / simultaneous bands come from a unit-level
#   empirical bootstrap (``qtt_empirical_bootstrap``).
# =============================================================================


def _pget(ptep: Any, key: str, default: Any = None) -> Any:
    """Read a parameter off a ``PTEParams`` or plain dict."""
    if isinstance(ptep, dict):
        return ptep.get(key, default)
    if hasattr(ptep, key):
        return getattr(ptep, key)
    raise ValueError(f"ptep is missing required field: {key}")


def _ptep_field(ptep: Any, key: str, default: Any = None) -> Any:
    """Read a parameter, tolerating missing fields on dataclass ``ptep``s.

    Unlike ``_pget``, a missing attribute on an object just yields ``default``
    (mirrors R's ``$`` extraction returning ``NULL`` for absent list elements).
    """
    if isinstance(ptep, dict):
        return ptep.get(key, default)
    return getattr(ptep, key, default)


class _ECDF:
    """A step-function empirical distribution (``make_dist``'s approxfun).

    Evaluated at a query ``q`` it returns the piecewise-constant CDF with
    ``yleft=0``, ``yright=1`` and ``ties="ordered"``, mirroring R's
    ``approxfun(x, Fx, method="constant")``.
    """

    def __init__(self, x: Any, y: Any) -> None:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        idx = np.argsort(x)
        self.x, self.y = x[idx].copy(), y[idx].copy()
        self.nobs = len(self.x)

    def __call__(self, q: Any) -> np.ndarray:
        q = np.atleast_1d(np.asarray(q, dtype=float))
        pos = np.searchsorted(self.x, q, side="right") - 1
        out = np.zeros_like(q)
        valid = pos >= 0
        clipped = np.clip(pos, 0, len(self.x) - 1)
        out[valid] = self.y[clipped[valid]]
        out[pos >= len(self.x)] = 1.0
        return out


def combine_ecdfs(
    y_seq: Any, ecdflist: Sequence[Any], weights: Optional[Sequence[float]] = None
) -> _ECDF:
    """Mix per-\\.((g,t))`` CDFs into one CDF — ``BMisc::combine_ecdfs``."""
    y_seq = np.asarray(y_seq, dtype=float)
    y_seq = np.sort(y_seq)
    if len(ecdflist) == 0:
        return _ECDF(y_seq, np.zeros_like(y_seq))
    w = (
        np.full(len(ecdflist), 1.0 / len(ecdflist))
        if weights is None
        else np.asarray(weights, dtype=float)
    )
    w = w / w.sum() if w.sum() != 0 else w
    values = np.column_stack([np.asarray(ecdf(y_seq), dtype=float) for ecdf in ecdflist])
    return _ECDF(y_seq, values @ w)


def ecdf_quantiles(ecdf: _ECDF, probs: Any) -> np.ndarray:
    """``quantile(ecdf, probs, type = 1)`` mirroring ``quantile.ecdf``.

    R reconstructs an approximate equally-weighted pseudo-sample from the stored
    breakpoints and CDF heights, then applies the type-1 inverse-CDF quantile;
    this reproduces that exactly (verified against the installed ``ptetools``).
    """
    probs = np.atleast_1d(np.asarray(probs, dtype=float))
    rounded = np.round(ecdf.nobs * ecdf.y).astype(int)
    counts = np.diff(np.concatenate([[0], rounded]))
    recon = np.repeat(ecdf.x, counts)
    if recon.size == 0:
        return np.full(probs.shape, np.nan)
    return np.array([_type1_quantile(recon, p) for p in probs])


def block_boot_sample(
    data: pd.DataFrame, idname: str, rng: Optional[np.random.Generator] = None
) -> pd.DataFrame:
    """Block-resample a panel by unit, re-indexing ids (``blockBootSample``)."""
    unique_ids = pd.unique(data[idname])
    if unique_ids.size == 0:
        raise ValueError("cannot bootstrap an empty panel")
    rng = rng if rng is not None else np.random.default_rng()
    sampled = rng.choice(unique_ids, size=unique_ids.size, replace=True)
    pieces = []
    for new_id, old_id in enumerate(sampled):
        block = data.loc[data[idname].eq(old_id)].copy()
        block[idname] = new_id
        pieces.append(block)
    return pd.concat(pieces, ignore_index=True)


def _attgt_pte_weights(attgt_list: Sequence[dict[str, Any]], ptep: Any) -> dict[str, Any]:
    """Port of the R ``attgt_pte_aggregations`` weight computation.

    Given the per-cell ``(group, time.period, att)`` entries, builds the
    group/dynamic/overall weights used to mix ``(g,t)`` CDFs.  Row order
    follows R's ``merge`` (sorted by ``group`` then ``time.period``).
    """
    groups = list(_pget(ptep, "groups"))
    periods = list(_pget(ptep, "time_periods"))
    data = _pget(ptep, "data")
    gname = _pget(ptep, "gname")
    tname = _pget(ptep, "tname")
    frame = pd.DataFrame(
        [
            {"group": c["group"], "time.period": c["time.period"], "att": float(c["att"])}
            for c in attgt_list
        ]
    )
    frame = frame.dropna(subset=["att"]).reset_index(drop=True)
    frame["e"] = frame["time.period"] - frame["group"]
    first_period = periods[0]
    n_group: dict[Any, float] = {}
    for group in groups:
        sub = data.loc[(data[gname] == group) & (data[tname] == first_period)]
        n_group[group] = float(len(sub))
    frame["n.group"] = frame["group"].map(n_group).fillna(0.0)
    frame = frame.sort_values(["group", "time.period"]).reset_index(drop=True)

    eseq = sorted(pd.unique(frame["e"]))
    dyn_rows = []
    dyn_weights = []
    for this_e in eseq:
        res_e = frame.loc[frame["e"].eq(this_e)]
        w = res_e["n.group"].to_numpy(float)
        w = w / w.sum() if w.sum() else w
        mask = frame["e"].to_numpy() == this_e
        wvec = np.zeros(len(frame))
        wvec[mask] = w
        dyn_weights.append({"e": this_e, "weights": wvec})
        dyn_rows.append({"e": this_e, "att.e": float((res_e["att"].to_numpy() * w).sum())})

    group_rows = []
    group_weights = []
    for group in groups:
        mask = (frame["group"] == group) & (frame["time.period"] >= frame["group"])
        res_g = frame.loc[mask]
        if len(res_g) == 0:
            continue
        wvec = np.zeros(len(frame))
        wvec[mask.to_numpy()] = 1.0 / len(res_g)
        group_weights.append({"g": group, "weights": wvec})
        group_rows.append(
            {
                "group": group,
                "att.g": float(res_g["att"].mean()),
                "n.group": float(frame.loc[frame["group"] == group, "n.group"].iloc[0]),
                "group_post_length": len(res_g),
            }
        )
    grp = pd.DataFrame(group_rows)
    if len(grp) == 0:
        over_weights = np.zeros(len(frame))
        att_overall = float("nan")
    else:
        grp = grp.dropna(subset=["n.group"])
        total = grp["n.group"].sum()
        att_overall = float((grp["att.g"] * grp["n.group"]).sum() / total)
        if (grp["group_post_length"] == 0).any() or total == 0:
            grp = grp.assign(g_overall_w=0.0)
        else:
            grp = grp.assign(g_overall_w=(grp["n.group"] / total) / grp["group_post_length"])
        over_map = dict(zip(grp["group"], grp["g_overall_w"]))
        gr_over = frame["group"].map(over_map).fillna(0.0).to_numpy()
        over_weights = np.where((frame["e"] >= 0).to_numpy(), gr_over, 0.0)

    return {
        "attgt_results": frame[["group", "time.period", "att"]],
        "dyn_results": (
            pd.DataFrame(dyn_rows, columns=["e", "att.e"])
            if dyn_rows
            else pd.DataFrame(columns=["e", "att.e"])
        ),
        "dyn_weights": dyn_weights,
        "group_results": (
            grp[["group", "att.g"]] if len(grp) else pd.DataFrame(columns=["group", "att.g"])
        ),
        "group_weights": group_weights,
        "overall_results": att_overall,
        "overall_weights": over_weights,
    }


def _aligned_cell_returns(
    extra_gt_returns: Sequence[dict[str, Any]],
    order: Sequence[tuple[Any, Any]],
) -> list[dict[str, Any]]:
    """Reorder per-cell extra returns to match ``_attgt_pte_aggregations`` rows."""
    lookup = {(e["group"], e["time.period"]): e.get("extra_gt_returns") for e in extra_gt_returns}
    aligned = []
    for group, time_period in order:
        cell = lookup.get((group, time_period))
        if cell is None:
            raise ValueError("extra_gt_returns is missing a group-time cell present in attgt_list")
        aligned.append(cell)
    return aligned


def pte_qtt(
    overall: pd.DataFrame,
    dynamic: pd.DataFrame,
    group: pd.DataFrame,
    *,
    F0_overall: Any = None,
    F1_overall: Any = None,
    ptep: Any = None,
) -> "PTEQTTResult":
    """Construct a ``pte_qtt`` result container (R ``ptetools::pte_qtt``)."""
    return PTEQTTResult(overall, dynamic, group, F0_overall, F1_overall, ptep)


@dataclass
class PTEQTTResult:
    """Full quantile treatment-effect curve (R ``pte_qtt`` object).

    ``overall``/``dynamic``/``group`` are DataFrames with ``probs`` + ``qtt``
    (plus ``se``/confidence-band columns after ``qtt_empirical_bootstrap``);
    ``F0_overall``/``F1_overall`` are the mixed ``_ECDF`` CDFs.
    """

    overall: pd.DataFrame
    dynamic: pd.DataFrame
    group: pd.DataFrame
    F0_overall: Any = None
    F1_overall: Any = None
    ptep: Any = None

    def to_dict(self) -> dict[str, object]:
        return {
            "overall": self.overall.to_dict(orient="records"),
            "dynamic": self.dynamic.to_dict(orient="records"),
            "group": self.group.to_dict(orient="records"),
        }

    def summary(self) -> str:
        probs = self.overall["probs"]
        qtt = self.overall["qtt"]
        return f"PTEQTTResult(overall QTT: median {np.nanmedian(qtt):.4f} over {len(probs)} quantile levels)"


def qtt_pte_aggregations(
    attgt_list: Sequence[dict[str, Any]],
    ptep: Any,
    extra_gt_returns: Sequence[dict[str, Any]],
    probs: Optional[Sequence[float]] = None,
) -> dict[str, Any]:
    """Mix ``(g,t)`` F0/F1 CDFs into overall/dynamic/group QTT curves.

    Mirrors R ``ptetools::qtt_pte_aggregations``: the per-cell CDFs are aligned
    to the aggregated weight rows (a deliberate fix of R's latent ordering
    assumption when the compute loop is time-major but the weights are
    group-major), mixed with the ``_attgt_pte_weights`` weights over a common
    ``y.seq`` grid, and inverted at each ``probs`` level (``quantile.ecdf``).
    """
    if probs is None:
        probs = np.arange(0.05, 0.951, 0.05)
    probs = np.asarray(probs, dtype=float)
    agg = _attgt_pte_weights(attgt_list, ptep)
    order = list(zip(agg["attgt_results"]["group"], agg["attgt_results"]["time.period"]))
    cells = _aligned_cell_returns(extra_gt_returns, order)
    F0_gt = [cell["F0"] for cell in cells]
    F1_gt = [cell["F1"] for cell in cells]

    data = _pget(ptep, "data")
    yname = _pget(ptep, "yname")
    y_seq = np.quantile(data[yname], np.linspace(0.0, 1.0, 1000))
    overall_w = np.asarray(agg["overall_weights"], dtype=float)
    F0_overall = combine_ecdfs(y_seq, F0_gt, overall_w)
    F1_overall = combine_ecdfs(y_seq, F1_gt, overall_w)
    overall_results = pd.DataFrame(
        {
            "probs": probs,
            "qtt": ecdf_quantiles(F1_overall, probs) - ecdf_quantiles(F0_overall, probs),
        }
    )

    dyn_rows = []
    for dw in agg["dyn_weights"]:
        w = np.asarray(dw["weights"], dtype=float)
        F0_e = combine_ecdfs(y_seq, F0_gt, w)
        F1_e = combine_ecdfs(y_seq, F1_gt, w)
        dyn_rows.append(
            pd.DataFrame(
                {
                    "e": dw["e"],
                    "probs": probs,
                    "qtt": ecdf_quantiles(F1_e, probs) - ecdf_quantiles(F0_e, probs),
                }
            )
        )
    dyn_results = (
        pd.concat(dyn_rows, ignore_index=True)
        if dyn_rows
        else pd.DataFrame(columns=["e", "probs", "qtt"])
    )

    group_rows = []
    for gw in agg["group_weights"]:
        w = np.asarray(gw["weights"], dtype=float)
        F0_g = combine_ecdfs(y_seq, F0_gt, w)
        F1_g = combine_ecdfs(y_seq, F1_gt, w)
        group_rows.append(
            pd.DataFrame(
                {
                    "group": gw["g"],
                    "probs": probs,
                    "qtt": ecdf_quantiles(F1_g, probs) - ecdf_quantiles(F0_g, probs),
                }
            )
        )
    group_results = (
        pd.concat(group_rows, ignore_index=True)
        if group_rows
        else pd.DataFrame(columns=["group", "probs", "qtt"])
    )

    return {
        "overall_results": overall_results,
        "dyn_results": dyn_results,
        "group_results": group_results,
        "F0_overall": F0_overall,
        "F1_overall": F1_overall,
    }


def qott_pte_aggregations(
    attgt_list: Sequence[dict[str, Any]],
    ptep: Any,
    extra_gt_returns: Sequence[dict[str, Any]],
    ret_quantile: Optional[Sequence[float]] = None,
) -> dict[str, Any]:
    """Aggregate ``(g,t)`` treatment-effect distributions into QoTT curves."""
    if ret_quantile is None:
        ret_quantile = _ptep_field(ptep, "ret_quantile", None)
    if ret_quantile is None:
        ret_quantile = np.arange(0.05, 0.951, 0.05)
    ret_quantile = np.asarray(ret_quantile, dtype=float)
    agg = _attgt_pte_weights(attgt_list, ptep)
    cells = _aligned_cell_returns(
        extra_gt_returns,
        list(zip(agg["attgt_results"]["group"], agg["attgt_results"]["time.period"])),
    )
    Fte_gt = [cell["Fte"] for cell in cells]
    data = _pget(ptep, "data")
    yname = _pget(ptep, "yname")
    y_seq = np.linspace(-np.max(data[yname]), np.max(data[yname]), 1000)
    overall_w = np.asarray(agg["overall_weights"], dtype=float)
    Fte_overall = combine_ecdfs(y_seq, Fte_gt, overall_w)
    overall = ecdf_quantiles(Fte_overall, ret_quantile)

    dyn_rows = []
    for dw in agg["dyn_weights"]:
        Fte_e = combine_ecdfs(y_seq, Fte_gt, np.asarray(dw["weights"], dtype=float))
        dyn_rows.append(
            pd.DataFrame(
                {"e": dw["e"], "probs": ret_quantile, "qott": ecdf_quantiles(Fte_e, ret_quantile)}
            )
        )
    dyn_results = (
        pd.concat(dyn_rows, ignore_index=True)
        if dyn_rows
        else pd.DataFrame(columns=["e", "probs", "qott"])
    )

    group_rows = []
    for gw in agg["group_weights"]:
        Fte_g = combine_ecdfs(y_seq, Fte_gt, np.asarray(gw["weights"], dtype=float))
        group_rows.append(
            pd.DataFrame(
                {
                    "group": gw["g"],
                    "probs": ret_quantile,
                    "qott": ecdf_quantiles(Fte_g, ret_quantile),
                }
            )
        )
    group_results = (
        pd.concat(group_rows, ignore_index=True)
        if group_rows
        else pd.DataFrame(columns=["group", "probs", "qott"])
    )

    return {
        "overall_results": overall,
        "dyn_results": dyn_results,
        "group_results": group_results,
        "Fte_overall": Fte_overall,
    }


def compute_pte(
    ptep: Any,
    subset_fun: Any,
    attgt_fun: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the ``(g,t)`` estimation loop — R ``ptetools::compute.pte``.

    ``subset_fun(data, g, tp, ...)`` yields a ``TwoByTwoSubset`` and
    ``attgt_fun(gt_data=gt_data, ...)`` yields an ``ATTGTResult`` whose
    ``extra_gt_returns`` carries the per-cell ``F0``/``F1`` (and ``Fte`` for
    QoTT) ``_ECDF`` objects.  Returns ``attgt.list`` (ordered time-major),
    ``inffunc`` and ``extra_gt_returns`` exactly like R.
    """
    data = _pget(ptep, "data")
    gname = _pget(ptep, "gname")
    tname = _pget(ptep, "tname")
    idname = _pget(ptep, "idname")
    panel = _pget(ptep, "panel")
    base_period = _pget(ptep, "base_period", "varying")
    anticipation = _pget(ptep, "anticipation", 0)
    time_periods = list(_pget(ptep, "time_periods") or _pget(ptep, "tlist", []))
    groups = list(_pget(ptep, "groups") or _pget(ptep, "glist", []))
    n = data[idname].nunique() if panel else len(data)

    subset_kwargs = dict(kwargs)
    subset_kwargs.setdefault("gname", gname)
    subset_kwargs.setdefault("tname", tname)
    subset_kwargs.setdefault("yname", _pget(ptep, "yname"))
    if idname is not None:
        subset_kwargs.setdefault("idname", idname)
    subset_kwargs.setdefault("anticipation", anticipation)
    subset_kwargs.setdefault("base_period", base_period)

    attgt_list: list[dict[str, Any]] = []
    extra_gt_returns: list[dict[str, Any]] = []
    inffunc = np.full((n, len(groups) * len(time_periods)), np.nan)
    counter = 0
    for tp in time_periods:
        for g in groups:
            if base_period == "universal" and tp == (g - 1 - anticipation):
                attgt_list.append({"att": 0, "group": g, "time.period": tp})
                extra_gt_returns.append({"extra_gt_returns": None, "group": g, "time.period": tp})
                counter += 1
                continue
            gt_subset = subset_fun(data, g, tp, **subset_kwargs)
            gt_data = gt_subset.gt_data
            n1 = gt_subset.n1
            disidx = gt_subset.disidx
            attgt = attgt_fun(gt_data=gt_data, **kwargs)
            attgt_list.append({"att": attgt.attgt, "group": g, "time.period": tp})
            extra_gt_returns.append(
                {"extra_gt_returns": attgt.extra_gt_returns, "group": g, "time.period": tp}
            )
            if attgt.inf_func is not None and n1:
                scaled = (n / n1) * np.asarray(attgt.inf_func, dtype=float)
                this_if = np.zeros(n)
                this_if[disidx] = scaled
                inffunc[:, counter] = this_if
            counter += 1
    return {
        "attgt.list": attgt_list,
        "inffunc": inffunc,
        "extra_gt_returns": extra_gt_returns,
    }


def _qtt_crit_val(boot_mat: np.ndarray, qtt_est: np.ndarray, alp: float) -> float:
    """Sup-t critical value over the QTT curve — R ``qtt_crit_val``.

    Standardises each quantile column by a robust ``(IQR / (z.75 - z.25))``
    scale (falling back to the sample SD clamped to ``1e-9``), takes the
    per-bootstrap maximum absolute standardised deviation, and returns the
    ``(1 - alp)`` type-1 empirical quantile of that sup statistic.
    """
    boot_mat = np.asarray(boot_mat, dtype=float)
    qtt_est = np.asarray(qtt_est, dtype=float)
    iqr_scale = np.array(
        [
            _type1_quantile(boot_mat[:, j], 0.75) - _type1_quantile(boot_mat[:, j], 0.25)
            for j in range(boot_mat.shape[1])
        ]
    )
    sigmahalf = iqr_scale / (norm.ppf(0.75) - norm.ppf(0.25))
    if np.any(sigmahalf == 0):
        sigmahalf = np.maximum(np.std(boot_mat, axis=0, ddof=1), 1e-9)
    cb = np.max(np.abs((boot_mat - qtt_est) / sigmahalf), axis=1)
    return _type1_quantile(cb, 1 - alp)


def qtt_empirical_bootstrap(
    attgt_list: Sequence[dict[str, Any]],
    ptep: Any,
    setup_pte_fun: Any,
    subset_fun: Any,
    attgt_fun: Any,
    extra_gt_returns: Sequence[dict[str, Any]],
    aggte_fun: Any = None,
    *,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> PTEQTTResult:
    """Unit-level empirical bootstrap for QTT — R ``qtt_empirical_bootstrap``.

    Repeatedly block-resamples units (or resamples rows for repeated cross
    sections), re-estimates the per-cell F0/F1 CDFs, re-aggregates the QTT
    curve, and derives bootstrap pointwise SEs (``qtt +/- z*se``) plus uniform
    bands (``qtt +/- crit*se``) using the sup-t critical value ``_qtt_crit_val``.
    """
    if aggte_fun is None:
        aggte_fun = qtt_pte_aggregations
    probs = _ptep_field(ptep, "probs", None)
    probs = np.asarray(probs, dtype=float) if probs is not None else np.arange(0.05, 0.951, 0.05)
    data = _pget(ptep, "data")
    yname = _pget(ptep, "yname")
    gname = _pget(ptep, "gname")
    tname = _pget(ptep, "tname")
    idname = _pget(ptep, "idname")
    panel = _pget(ptep, "panel")
    alp = _ptep_field(ptep, "alp", 0.05)
    biters = int(_ptep_field(ptep, "biters", 99))
    boot_type = _ptep_field(ptep, "boot_type", "empirical")
    gt_type = _ptep_field(ptep, "gt_type", "qtt")

    aggte = aggte_fun(attgt_list, ptep, extra_gt_returns)
    z = norm.ppf(1 - alp / 2)
    rng = np.random.default_rng(seed)
    boot_res: list[Any] = []
    for _ in range(int(biters)):
        if panel:
            bdata = block_boot_sample(data.copy(), idname, rng=rng)
        else:
            idx = rng.integers(0, len(data), size=len(data))
            bdata = data.iloc[idx].reset_index(drop=True).copy()
            bdata[".rowid"] = np.arange(len(bdata))
            bdata["id"] = bdata[".rowid"]
        bptep = setup_pte_fun(
            yname=yname,
            gname=gname,
            tname=tname,
            idname=idname,
            data=bdata,
            panel=panel,
            alp=alp,
            boot_type=boot_type,
            gt_type=gt_type,
            probs=probs,
            biters=biters,
            cl=kwargs.get("cl", 1),
            **kwargs,
        )
        bres_gt = compute_pte(bptep, subset_fun, attgt_fun, **kwargs)
        boot_res.append(aggte_fun(bres_gt["attgt.list"], bptep, bres_gt["extra_gt_returns"]))

    overall_boot = np.asarray([br["overall_results"]["qtt"] for br in boot_res], dtype=float)
    overall_se = np.std(overall_boot, axis=0, ddof=1)
    overall_cval = _qtt_crit_val(overall_boot, aggte["overall_results"]["qtt"].to_numpy(), alp)
    overall_results = aggte["overall_results"].copy()
    overall_results["se"] = overall_se
    overall_results["lower_pw"] = overall_results["qtt"] - z * overall_se
    overall_results["upper_pw"] = overall_results["qtt"] + z * overall_se
    overall_results["lower_ub"] = overall_results["qtt"] - overall_cval * overall_se
    overall_results["upper_ub"] = overall_results["qtt"] + overall_cval * overall_se

    dyn_se_rows: list[pd.DataFrame] = []
    for this_e in dict.fromkeys(aggte["dyn_results"]["e"]):
        boot_rows = []
        for br in boot_res:
            grp = br["dyn_results"]
            vals = grp.loc[grp["e"].eq(this_e), "qtt"].to_numpy() if not grp.empty else np.array([])
            boot_rows.append(vals if vals.size == probs.size else None)
        complete = [r for r in boot_rows if r is not None]
        if len(complete) < 2:
            continue
        boot_mat = np.asarray(complete, dtype=float)
        qtt_est = aggte["dyn_results"].loc[aggte["dyn_results"]["e"].eq(this_e), "qtt"].to_numpy()
        this_cval = _qtt_crit_val(boot_mat, qtt_est, alp)
        dyn_se_rows.append(
            pd.DataFrame(
                {
                    "e": this_e,
                    "probs": probs,
                    "se": np.std(boot_mat, axis=0, ddof=1),
                    "cval": this_cval,
                }
            )
        )
    if dyn_se_rows:
        dyn_se_df = pd.concat(dyn_se_rows, ignore_index=True)
        dyn_results = pd.merge(aggte["dyn_results"], dyn_se_df, on=["e", "probs"], how="inner")
    else:
        dyn_results = aggte["dyn_results"].copy()
    if not dyn_results.empty:
        dyn_results = dyn_results.copy()
        dyn_results["lower_pw"] = dyn_results["qtt"] - z * dyn_results["se"]
        dyn_results["upper_pw"] = dyn_results["qtt"] + z * dyn_results["se"]
        dyn_results["lower_ub"] = dyn_results["qtt"] - dyn_results["cval"] * dyn_results["se"]
        dyn_results["upper_ub"] = dyn_results["qtt"] + dyn_results["cval"] * dyn_results["se"]
        dyn_results = dyn_results.drop(columns=["cval"])

    group_se_rows: list[pd.DataFrame] = []
    for g in dict.fromkeys(aggte["group_results"]["group"]):
        boot_rows = []
        for br in boot_res:
            grp = br["group_results"]
            vals = grp.loc[grp["group"].eq(g), "qtt"].to_numpy() if not grp.empty else np.array([])
            boot_rows.append(vals if vals.size == probs.size else None)
        complete = [r for r in boot_rows if r is not None]
        if len(complete) < 2:
            continue
        boot_mat = np.asarray(complete, dtype=float)
        qtt_est = (
            aggte["group_results"].loc[aggte["group_results"]["group"].eq(g), "qtt"].to_numpy()
        )
        this_cval = _qtt_crit_val(boot_mat, qtt_est, alp)
        group_se_rows.append(
            pd.DataFrame(
                {
                    "group": g,
                    "probs": probs,
                    "se": np.std(boot_mat, axis=0, ddof=1),
                    "cval": this_cval,
                }
            )
        )
    if group_se_rows:
        group_se_df = pd.concat(group_se_rows, ignore_index=True)
        group_results = pd.merge(
            aggte["group_results"], group_se_df, on=["group", "probs"], how="inner"
        )
    else:
        group_results = aggte["group_results"].copy()
    if not group_results.empty:
        group_results = group_results.copy()
        group_results["lower_pw"] = group_results["qtt"] - z * group_results["se"]
        group_results["upper_pw"] = group_results["qtt"] + z * group_results["se"]
        group_results["lower_ub"] = (
            group_results["qtt"] - group_results["cval"] * group_results["se"]
        )
        group_results["upper_ub"] = (
            group_results["qtt"] + group_results["cval"] * group_results["se"]
        )
        group_results = group_results.drop(columns=["cval"])

    return pte_qtt(
        overall_results,
        dyn_results,
        group_results,
        F0_overall=aggte.get("F0_overall"),
        F1_overall=aggte.get("F1_overall"),
        ptep=ptep,
    )
