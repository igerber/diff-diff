"""Two-period bad-control estimators.

This is the first Python implementation layer for the R ``badcontrols``
package.  It ports the linear imputation estimator from Caetano et al. (2026)
and keeps the data contract explicit: a balanced two-period panel with a
single treatment-affected covariate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class BadControlsResult:
    """Result of a two-period bad-control imputation estimate."""

    att: float
    se: float
    att_gt: pd.DataFrame
    influence_function: np.ndarray
    method: str = "imputation"

    @property
    def overall_att(self) -> float:
        return self.att

    @property
    def overall_se(self) -> float:
        return self.se

    def to_dict(self) -> dict:
        return {
            "att": self.att,
            "se": self.se,
            "method": self.method,
            "att_gt": self.att_gt.to_dict(orient="records"),
        }


def _design(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    values = [np.ones(len(frame), dtype=float)]
    for column in columns:
        if column not in frame.columns:
            raise ValueError(f"missing covariate column {column!r}")
        values.append(pd.to_numeric(frame[column], errors="raise").to_numpy(float))
    return np.column_stack(values)


def _fit_predict(
    train: pd.DataFrame, target: str, columns: Sequence[str], new: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    x_train = _design(train, columns)
    y_train = pd.to_numeric(train[target], errors="raise").to_numpy(float)
    coef, _, _, _ = np.linalg.lstsq(x_train, y_train, rcond=None)
    return _design(new, columns) @ coef, coef


def _wide_panel(
    data: pd.DataFrame,
    yname: str,
    gname: str,
    tname: str,
    idname: str,
    pre_period: object,
    post_period: object,
    extra_columns: Sequence[str] = (),
) -> pd.DataFrame:
    required = {yname, gname, tname, idname, *extra_columns}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"data is missing required columns: {missing}")
    panel = data.loc[data[tname].isin([pre_period, post_period])].copy()
    counts = panel.groupby(idname)[tname].nunique()
    if (counts != 2).any():
        raise ValueError("data must contain exactly one pre and one post observation per unit")
    wide = panel.pivot(index=idname, columns=tname)
    wide.columns = [f"{name}_{period}" for name, period in wide.columns]
    group = panel.groupby(idname)[gname].first()
    wide[gname] = group
    wide = wide.reset_index()
    wide["D"] = (wide[gname] != 0).astype(int)
    for column in extra_columns:
        wide[column] = wide[f"{column}_{pre_period}"]
    return wide


def imputation_bad_control(
    data: pd.DataFrame,
    *,
    yname: str,
    gname: str,
    tname: str,
    idname: str,
    bad_control: Optional[str] = None,
    covariates: Sequence[str] = (),
    bad_control_covariates: Sequence[str] = (),
    bad_control_d_covariates: Sequence[str] = (),
    identification_strategy: str = "unconfoundedness",
) -> BadControlsResult:
    """Estimate ATT by imputing the untreated bad-control evolution.

    ``identification_strategy='unconfoundedness'`` fits the bad-control level
    on its lag and auxiliary covariates among controls.  ``'did'`` instead
    fits the bad-control change on auxiliary covariates, matching the
    parallel-trends-for-X branch in the R implementation.
    """
    if identification_strategy not in {"unconfoundedness", "did"}:
        raise ValueError("identification_strategy must be 'unconfoundedness' or 'did'")
    if bad_control is None and identification_strategy == "did":
        raise ValueError("identification_strategy='did' requires bad_control")
    periods = sorted(pd.unique(data[tname]).tolist())
    if len(periods) != 2:
        raise ValueError("imputation_bad_control currently requires exactly two periods")
    extra_columns = list(dict.fromkeys([bad_control] if bad_control else []))
    extra_columns += (
        list(covariates) + list(bad_control_covariates) + list(bad_control_d_covariates)
    )
    wide = _wide_panel(
        data,
        yname,
        gname,
        tname,
        idname,
        periods[0],
        periods[1],
        extra_columns,
    )
    y_pre, y_post = f"{yname}_{periods[0]}", f"{yname}_{periods[1]}"
    wide["delta_y"] = wide[y_post] - wide[y_pre]
    treated = wide["D"].eq(1)
    control = ~treated
    if not treated.any() or not control.any():
        raise ValueError("both treated and never-treated units are required")

    if bad_control is None:
        wide["bc_pre"] = 0.0
        wide["bc_post_imp"] = 0.0
        step1_columns: list[str] = []
    else:
        bc_pre, bc_post = f"{bad_control}_{periods[0]}", f"{bad_control}_{periods[1]}"
        wide["bc_pre"] = wide[bc_pre]
        wide["bc_post"] = wide[bc_post]
        auxiliary = list(bad_control_covariates) + list(bad_control_d_covariates) + list(covariates)
        step1_columns = (
            ["bc_pre"] + auxiliary if identification_strategy == "unconfoundedness" else auxiliary
        )
        if identification_strategy == "did":
            wide["delta_bc"] = wide["bc_post"] - wide["bc_pre"]
            predicted, _ = _fit_predict(wide.loc[control], "delta_bc", step1_columns, wide)
            wide["bc_post_imp"] = wide["bc_pre"] + predicted
        else:
            predicted, _ = _fit_predict(wide.loc[control], "bc_post", step1_columns, wide)
            wide["bc_post_imp"] = wide["bc_post"]
            wide.loc[treated, "bc_post_imp"] = predicted[treated.to_numpy()]

    outcome_columns = ["bc_post_imp", "bc_pre"] if bad_control is not None else []
    outcome_columns += list(covariates)
    predicted_y, _ = _fit_predict(wide.loc[control], "delta_y", outcome_columns, wide)
    residual_treated = (
        wide.loc[treated, "delta_y"].to_numpy(float) - predicted_y[treated.to_numpy()]
    )
    att = float(residual_treated.mean())
    n = len(wide)
    influence = np.zeros(n, dtype=float)
    pi = float(treated.mean())
    influence[treated.to_numpy()] = (residual_treated - att) / pi
    control_residual = (
        wide.loc[control, "delta_y"].to_numpy(float) - predicted_y[control.to_numpy()]
    )
    influence[control.to_numpy()] = -control_residual / (1.0 - pi)
    se = float(np.sqrt(np.mean((influence - influence.mean()) ** 2) / n))
    att_gt = pd.DataFrame(
        {"group": [wide.loc[treated, gname].iloc[0]], "time": [periods[1]], "attgt": [att]}
    )
    return BadControlsResult(att, se, att_gt, influence)


def didbc(
    data: pd.DataFrame,
    *,
    yname: str,
    gname: str,
    tname: str,
    idname: str,
    bad_control: Optional[str] = None,
    covariates: Sequence[str] = (),
    bad_control_covariates: Sequence[str] = (),
    identification_strategy: str = "unconfoundedness",
    est_method: str = "imputation",
    **_: object,
) -> BadControlsResult:
    """Python spelling of R ``didbc`` for its linear imputation path."""
    if est_method != "imputation":
        raise NotImplementedError("est_method='dr_ml' is not implemented in this release")
    return imputation_bad_control(
        data,
        yname=yname,
        gname=gname,
        tname=tname,
        idname=idname,
        bad_control=bad_control,
        covariates=covariates,
        bad_control_covariates=bad_control_covariates,
        identification_strategy=identification_strategy,
    )


def extract_att(result: BadControlsResult) -> dict[str, float]:
    """Return the overall ATT and standard error from a bad-control result."""
    if not isinstance(result, BadControlsResult):
        raise TypeError("result must be a BadControlsResult")
    return {"att": result.att, "se": result.se}
