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
from scipy.special import expit


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


def _logit_predict(
    train: pd.DataFrame,
    treatment: str,
    columns: Sequence[str],
    new: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a logistic working model by IRLS and return fitted probabilities."""
    x = _design(train, columns)
    y = train[treatment].to_numpy(float)
    beta = np.zeros(x.shape[1], dtype=float)
    for _ in range(100):
        probability = np.clip(expit(x @ beta), 1e-8, 1 - 1e-8)
        variance = np.clip(probability * (1 - probability), 1e-8, None)
        working = x @ beta + (y - probability) / variance
        updated = np.linalg.lstsq(
            x * np.sqrt(variance)[:, None], working * np.sqrt(variance), rcond=None
        )[0]
        if np.max(np.abs(updated - beta)) < 1e-10:
            beta = updated
            break
        beta = updated
    return np.clip(expit(_design(new, columns) @ beta), 1e-8, 1 - 1e-8), beta


def dr_parametric_bad_control(
    data: pd.DataFrame,
    *,
    yname: str,
    gname: str,
    tname: str,
    idname: str,
    bad_control: Optional[str] = None,
    covariates: Sequence[str] = (),
    bad_control_covariates: Sequence[str] = (),
) -> BadControlsResult:
    """Estimate the two-period parametric doubly robust bad-control score.

    This follows Equation (11) of Caetano et al. (2026).  It uses the
    parametric nuisance models without cross-fitting; the cross-fitted ML
    route remains a separate implementation step.
    """
    periods = sorted(pd.unique(data[tname]).tolist())
    if len(periods) != 2:
        raise ValueError("dr_parametric_bad_control currently requires exactly two periods")
    extra = list(
        dict.fromkeys(
            ([bad_control] if bad_control else []) + list(covariates) + list(bad_control_covariates)
        )
    )
    wide = _wide_panel(data, yname, gname, tname, idname, periods[0], periods[1], extra)
    wide["delta_y"] = wide[f"{yname}_{periods[1]}"] - wide[f"{yname}_{periods[0]}"]
    treated = wide["D"].eq(1)
    control = ~treated
    if not treated.any() or not control.any():
        raise ValueError("both treated and never-treated units are required")
    if bad_control is not None:
        wide["bc_pre"] = wide[f"{bad_control}_{periods[0]}"]
        wide["bc_post"] = wide[f"{bad_control}_{periods[1]}"]
        m_columns = ["bc_post", "bc_pre"] + list(covariates)
        p_columns = ["bc_pre"] + list(bad_control_covariates) + list(covariates)
    else:
        m_columns = list(covariates)
        p_columns = list(covariates)
    m_hat, _ = _fit_predict(wide.loc[control], "delta_y", m_columns, wide)
    p_hat, _ = _logit_predict(wide, "D", p_columns, wide)
    wide["m_hat"] = m_hat
    nu_hat, _ = _fit_predict(wide.loc[control], "m_hat", p_columns, wide)
    odds = p_hat / (1 - p_hat)
    wide["odds_hat"] = odds
    omega_hat, _ = _fit_predict(wide.loc[control], "odds_hat", m_columns, wide)
    delta_y = wide["delta_y"].to_numpy(float)
    d = treated.to_numpy(float)
    pi = float(d.mean())
    score = (
        d / pi * delta_y
        - d / pi * nu_hat
        - (1 - d) / pi * (m_hat - nu_hat) * odds
        - (1 - d) / pi * (delta_y - m_hat) * omega_hat
    )
    att = float(score.mean())
    influence = score - att - att / pi * (d - pi)
    se = float(np.sqrt(np.mean(influence**2) / len(wide)))
    att_gt = pd.DataFrame(
        {"group": [wide.loc[treated, gname].iloc[0]], "time": [periods[1]], "attgt": [att]}
    )
    return BadControlsResult(att, se, att_gt, influence, method="dr_ml-parametric")


def dr_ml_bad_control(
    data: pd.DataFrame,
    *,
    yname: str,
    gname: str,
    tname: str,
    idname: str,
    bad_control: Optional[str] = None,
    covariates: Sequence[str] = (),
    bad_control_covariates: Sequence[str] = (),
    n_folds: int = 5,
    random_state: Optional[int] = None,
) -> BadControlsResult:
    """Estimate the DR bad-control score with cross-fitted random forests."""
    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    except ImportError as exc:
        raise ImportError("install diff-diff[ml] to use nuisance_method='ml'") from exc
    if not isinstance(n_folds, (int, np.integer)) or n_folds < 2:
        raise ValueError("n_folds must be an integer greater than or equal to 2")
    periods = sorted(pd.unique(data[tname]).tolist())
    if len(periods) != 2:
        raise ValueError("dr_ml_bad_control currently requires exactly two periods")
    extra = list(
        dict.fromkeys(
            ([bad_control] if bad_control else []) + list(covariates) + list(bad_control_covariates)
        )
    )
    wide = _wide_panel(data, yname, gname, tname, idname, periods[0], periods[1], extra)
    wide["delta_y"] = wide[f"{yname}_{periods[1]}"] - wide[f"{yname}_{periods[0]}"]
    treated = wide["D"].eq(1).to_numpy()
    control = ~treated
    if treated.sum() < n_folds or control.sum() < n_folds:
        raise ValueError("each treatment arm must have at least n_folds units")
    if bad_control is not None:
        wide["bc_pre"] = wide[f"{bad_control}_{periods[0]}"]
        wide["bc_post"] = wide[f"{bad_control}_{periods[1]}"]
        m_columns = ["bc_post", "bc_pre"] + list(covariates)
        p_columns = ["bc_pre"] + list(bad_control_covariates) + list(covariates)
    else:
        m_columns = list(covariates)
        p_columns = list(covariates)
    x_m = _design(wide, m_columns)[:, 1:]
    x_p = _design(wide, p_columns)[:, 1:]
    rng = np.random.default_rng(random_state)
    fold_ids = np.empty(len(wide), dtype=int)
    for mask in (treated, control):
        indices = np.flatnonzero(mask)
        rng.shuffle(indices)
        fold_ids[indices] = np.arange(len(indices)) % n_folds
    m_hat = np.zeros(len(wide), dtype=float)
    p_hat = np.zeros(len(wide), dtype=float)
    nu_hat = np.zeros(len(wide), dtype=float)
    omega_hat = np.zeros(len(wide), dtype=float)
    for fold in range(n_folds):
        train = fold_ids != fold
        test = ~train
        train_control = train & control
        m_model = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, random_state=random_state
        )
        m_model.fit(x_m[train_control], wide.loc[train_control, "delta_y"])
        m_hat[test] = m_model.predict(x_m[test])
        p_model = RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, random_state=random_state
        )
        p_model.fit(x_p[train], wide.loc[train, "D"])
        p_hat[test] = np.clip(p_model.predict_proba(x_p[test])[:, 1], 1e-4, 1 - 1e-4)
        nu_model = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, random_state=random_state
        )
        nu_model.fit(x_p[train_control], m_hat[train_control])
        nu_hat[test] = nu_model.predict(x_p[test])
        omega_model = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, random_state=random_state
        )
        odds_train = p_hat[train_control] / (1 - p_hat[train_control])
        omega_model.fit(x_m[train_control], odds_train)
        omega_hat[test] = omega_model.predict(x_m[test])
    pi = float(treated.mean())
    delta_y = wide["delta_y"].to_numpy(float)
    d = treated.astype(float)
    odds = p_hat / (1 - p_hat)
    score = d / pi * delta_y - d / pi * nu_hat
    score -= (1 - d) / pi * (m_hat - nu_hat) * odds
    score -= (1 - d) / pi * (delta_y - m_hat) * omega_hat
    att = float(score.mean())
    influence = score - att - att / pi * (d - pi)
    se = float(np.sqrt(np.mean(influence**2) / len(wide)))
    att_gt = pd.DataFrame(
        {"group": [wide.loc[treated, gname].iloc[0]], "time": [periods[1]], "attgt": [att]}
    )
    return BadControlsResult(att, se, att_gt, influence, method="dr_ml")


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

    step1_pred: Optional[np.ndarray] = None
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
            step1_pred = predicted
            wide["bc_post_imp"] = wide["bc_post"]
            wide.loc[treated, "bc_post_imp"] = predicted[treated.to_numpy()]

    outcome_columns = ["bc_post_imp", "bc_pre"] if bad_control is not None else []
    outcome_columns += list(covariates)
    predicted_y, outcome_coef = _fit_predict(wide.loc[control], "delta_y", outcome_columns, wide)
    residual_treated = (
        wide.loc[treated, "delta_y"].to_numpy(float) - predicted_y[treated.to_numpy()]
    )
    att = float(residual_treated.mean())
    n = len(wide)
    influence = np.zeros(n, dtype=float)
    pi = float(treated.mean())
    treated_mask = treated.to_numpy()
    control_mask = control.to_numpy()
    influence[treated_mask] = (residual_treated - att) / pi
    if bad_control is not None and identification_strategy == "unconfoundedness":
        # Include uncertainty from both OLS steps, as in the R influence
        # function for the linear imputation estimator.
        r_control = _design(wide.loc[control], outcome_columns)
        r_treated = _design(wide.loc[treated], outcome_columns)
        s_control = _design(wide.loc[control], step1_columns)
        s_treated = _design(wide.loc[treated], step1_columns)
        u = wide.loc[control, "delta_y"].to_numpy(float) - predicted_y[control_mask]
        if step1_pred is None:
            raise RuntimeError("bad-control imputation did not produce a first-stage prediction")
        v = wide.loc[control, "bc_post"].to_numpy(float) - step1_pred[control_mask]
        sigma_r = r_control.T @ r_control / control.sum()
        sigma_s = s_control.T @ s_control / control.sum()
        beta1 = float(outcome_coef[1])
        kappa_r = np.linalg.solve(sigma_r, r_treated.mean(axis=0))
        kappa_s = np.linalg.solve(sigma_s, beta1 * s_treated.mean(axis=0))
        correction = (r_control @ kappa_r) * u + (s_control @ kappa_s) * v
        influence[control_mask] = -correction / (1.0 - pi)
    else:
        control_residual = wide.loc[control, "delta_y"].to_numpy(float) - predicted_y[control_mask]
        influence[control_mask] = -control_residual / (1.0 - pi)
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
    nuisance_method: str = "ml",
    n_folds: int = 5,
    random_state: Optional[int] = None,
    **_: object,
) -> BadControlsResult:
    """Python spelling of R ``didbc`` for its linear imputation path."""
    if est_method == "dr_ml":
        if nuisance_method == "ml":
            return dr_ml_bad_control(
                data,
                yname=yname,
                gname=gname,
                tname=tname,
                idname=idname,
                bad_control=bad_control,
                covariates=covariates,
                bad_control_covariates=bad_control_covariates,
                n_folds=n_folds,
                random_state=random_state,
            )
        if nuisance_method != "parametric":
            raise NotImplementedError(
                "est_method='dr_ml' with nuisance_method='ml' is not implemented in this release"
            )
        return dr_parametric_bad_control(
            data,
            yname=yname,
            gname=gname,
            tname=tname,
            idname=idname,
            bad_control=bad_control,
            covariates=covariates,
            bad_control_covariates=bad_control_covariates,
        )
    if est_method != "imputation":
        raise ValueError("est_method must be 'imputation' or 'dr_ml'")
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
