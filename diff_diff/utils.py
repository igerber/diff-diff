"""
Utility functions for difference-in-differences estimation.
"""

from typing import Union

import numpy as np
import pandas as pd
from scipy import stats


def validate_binary(arr: np.ndarray, name: str) -> None:
    """
    Validate that an array contains only binary values (0 or 1).

    Parameters
    ----------
    arr : np.ndarray
        Array to validate.
    name : str
        Name of the variable (for error messages).

    Raises
    ------
    ValueError
        If array contains non-binary values.
    """
    unique_values = np.unique(arr[~np.isnan(arr)])
    if not np.all(np.isin(unique_values, [0, 1])):
        raise ValueError(
            f"{name} must be binary (0 or 1). "
            f"Found values: {unique_values}"
        )


def compute_robust_se(
    X: np.ndarray,
    residuals: np.ndarray,
    cluster_ids: np.ndarray = None
) -> np.ndarray:
    """
    Compute heteroskedasticity-robust (HC1) or cluster-robust standard errors.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n, k).
    residuals : np.ndarray
        Residuals from regression of shape (n,).
    cluster_ids : np.ndarray, optional
        Cluster identifiers for cluster-robust SEs.

    Returns
    -------
    np.ndarray
        Variance-covariance matrix of shape (k, k).
    """
    n, k = X.shape

    # Compute (X'X)^(-1)
    XtX_inv = np.linalg.inv(X.T @ X)

    if cluster_ids is None:
        # HC1 robust standard errors
        # HC1 adjustment factor: n / (n - k)
        adjustment = n / (n - k)

        # Create diagonal matrix with squared residuals
        u_squared = residuals ** 2

        # Meat of the sandwich: X' * diag(u^2) * X
        meat = X.T @ (X * u_squared[:, np.newaxis])

        vcov = adjustment * XtX_inv @ meat @ XtX_inv
    else:
        # Cluster-robust standard errors
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        # Adjustment factor for cluster-robust SEs
        adjustment = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))

        # Compute the meat of the sandwich
        meat = np.zeros((k, k))
        for cluster in unique_clusters:
            mask = cluster_ids == cluster
            X_c = X[mask]
            u_c = residuals[mask]
            score_c = X_c.T @ u_c
            meat += np.outer(score_c, score_c)

        vcov = adjustment * XtX_inv @ meat @ XtX_inv

    return vcov


def compute_confidence_interval(
    estimate: float,
    se: float,
    alpha: float = 0.05,
    df: int = None
) -> tuple:
    """
    Compute confidence interval for an estimate.

    Parameters
    ----------
    estimate : float
        Point estimate.
    se : float
        Standard error.
    alpha : float
        Significance level (default 0.05 for 95% CI).
    df : int, optional
        Degrees of freedom. If None, uses normal distribution.

    Returns
    -------
    tuple
        (lower_bound, upper_bound) of confidence interval.
    """
    if df is not None:
        critical_value = stats.t.ppf(1 - alpha / 2, df)
    else:
        critical_value = stats.norm.ppf(1 - alpha / 2)

    lower = estimate - critical_value * se
    upper = estimate + critical_value * se

    return (lower, upper)


def compute_p_value(t_stat: float, df: int = None, two_sided: bool = True) -> float:
    """
    Compute p-value for a t-statistic.

    Parameters
    ----------
    t_stat : float
        T-statistic.
    df : int, optional
        Degrees of freedom. If None, uses normal distribution.
    two_sided : bool
        Whether to compute two-sided p-value (default True).

    Returns
    -------
    float
        P-value.
    """
    if df is not None:
        p_value = stats.t.sf(np.abs(t_stat), df)
    else:
        p_value = stats.norm.sf(np.abs(t_stat))

    if two_sided:
        p_value *= 2

    return p_value


def check_parallel_trends(
    data: pd.DataFrame,
    outcome: str,
    time: str,
    treatment_group: str,
    pre_periods: list = None
) -> dict:
    """
    Perform a simple check for parallel trends assumption.

    This computes the trend (slope) in the outcome variable for both
    treatment and control groups during pre-treatment periods.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    outcome : str
        Name of outcome variable column.
    time : str
        Name of time period column.
    treatment_group : str
        Name of treatment group indicator column.
    pre_periods : list, optional
        List of pre-treatment time periods. If None, infers from data.

    Returns
    -------
    dict
        Dictionary with trend statistics and test results.
    """
    if pre_periods is None:
        # Assume treatment happens at median time period
        all_periods = sorted(data[time].unique())
        mid_point = len(all_periods) // 2
        pre_periods = all_periods[:mid_point]

    pre_data = data[data[time].isin(pre_periods)]

    # Compute trends for each group
    treated_data = pre_data[pre_data[treatment_group] == 1]
    control_data = pre_data[pre_data[treatment_group] == 0]

    # Simple linear regression for trends
    def compute_trend(group_data):
        time_values = group_data[time].values
        outcome_values = group_data[outcome].values

        # Normalize time to start at 0
        time_norm = time_values - time_values.min()

        # Compute slope using least squares
        n = len(time_norm)
        if n < 2:
            return np.nan, np.nan

        mean_t = np.mean(time_norm)
        mean_y = np.mean(outcome_values)

        slope = np.sum((time_norm - mean_t) * (outcome_values - mean_y)) / np.sum((time_norm - mean_t) ** 2)

        # Compute standard error of slope
        y_hat = mean_y + slope * (time_norm - mean_t)
        residuals = outcome_values - y_hat
        mse = np.sum(residuals ** 2) / (n - 2)
        se_slope = np.sqrt(mse / np.sum((time_norm - mean_t) ** 2))

        return slope, se_slope

    treated_slope, treated_se = compute_trend(treated_data)
    control_slope, control_se = compute_trend(control_data)

    # Test for difference in trends
    slope_diff = treated_slope - control_slope
    se_diff = np.sqrt(treated_se ** 2 + control_se ** 2)
    t_stat = slope_diff / se_diff if se_diff > 0 else np.nan
    p_value = compute_p_value(t_stat) if not np.isnan(t_stat) else np.nan

    return {
        "treated_trend": treated_slope,
        "treated_trend_se": treated_se,
        "control_trend": control_slope,
        "control_trend_se": control_se,
        "trend_difference": slope_diff,
        "trend_difference_se": se_diff,
        "t_statistic": t_stat,
        "p_value": p_value,
        "parallel_trends_plausible": p_value > 0.05 if not np.isnan(p_value) else None,
    }
