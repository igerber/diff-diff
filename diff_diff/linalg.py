"""
Unified linear algebra backend for diff-diff.

This module provides optimized OLS and variance estimation that can be
swapped to a compiled backend (Rust/C++) for maximum performance.

The key optimizations are:
1. scipy.linalg.lstsq with 'gelsy' driver (QR-based, faster than SVD)
2. Vectorized cluster-robust SE via groupby (eliminates O(n*clusters) loop)
3. Single interface for all estimators (reduces code duplication)

Future: This module can be extended with a Rust backend for additional speedup.
"""

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.linalg import lstsq as scipy_lstsq


def solve_ols(
    X: np.ndarray,
    y: np.ndarray,
    *,
    cluster_ids: Optional[np.ndarray] = None,
    return_vcov: bool = True,
    return_fitted: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]],
    Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]],
]:
    """
    Solve OLS regression with optional clustered standard errors.

    This is the unified OLS solver for all diff-diff estimators. It uses
    scipy's optimized LAPACK routines and vectorized variance estimation.

    Parameters
    ----------
    X : ndarray of shape (n, k)
        Design matrix (should include intercept if desired).
    y : ndarray of shape (n,)
        Response vector.
    cluster_ids : ndarray of shape (n,), optional
        Cluster identifiers for cluster-robust standard errors.
        If None, HC1 (heteroskedasticity-robust) SEs are computed.
    return_vcov : bool, default True
        Whether to compute and return the variance-covariance matrix.
        Set to False for faster computation when SEs are not needed.
    return_fitted : bool, default False
        Whether to return fitted values in addition to residuals.

    Returns
    -------
    coefficients : ndarray of shape (k,)
        OLS coefficient estimates.
    residuals : ndarray of shape (n,)
        Residuals (y - X @ coefficients).
    fitted : ndarray of shape (n,), optional
        Fitted values (X @ coefficients). Only returned if return_fitted=True.
    vcov : ndarray of shape (k, k) or None
        Variance-covariance matrix (HC1 or cluster-robust).
        None if return_vcov=False.

    Notes
    -----
    This function uses scipy.linalg.lstsq with the 'gelsy' driver, which is
    QR-based and typically faster than NumPy's default SVD-based solver for
    well-conditioned matrices.

    The cluster-robust standard errors use the sandwich estimator with the
    standard small-sample adjustment: (G/(G-1)) * ((n-1)/(n-k)).

    Examples
    --------
    >>> import numpy as np
    >>> from diff_diff.linalg import solve_ols
    >>> X = np.column_stack([np.ones(100), np.random.randn(100)])
    >>> y = 2 + 3 * X[:, 1] + np.random.randn(100)
    >>> coef, resid, vcov = solve_ols(X, y)
    >>> print(f"Intercept: {coef[0]:.2f}, Slope: {coef[1]:.2f}")
    """
    # Validate inputs
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1-dimensional, got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have same number of observations: "
            f"{X.shape[0]} vs {y.shape[0]}"
        )

    n, k = X.shape
    if n < k:
        raise ValueError(
            f"Fewer observations ({n}) than parameters ({k}). "
            "Cannot solve underdetermined system."
        )

    # Solve OLS using scipy's optimized solver
    # 'gelsy' uses QR with column pivoting, faster than default 'gelsd' (SVD)
    # check_finite=False skips NaN/Inf checks for speed (caller should validate)
    # Note: gelsy doesn't reliably report rank, so we don't check for deficiency
    coefficients = scipy_lstsq(X, y, lapack_driver="gelsy", check_finite=False)[0]

    # Compute residuals and fitted values
    fitted = X @ coefficients
    residuals = y - fitted

    # Compute variance-covariance matrix if requested
    vcov = None
    if return_vcov:
        vcov = compute_robust_vcov(X, residuals, cluster_ids)

    if return_fitted:
        return coefficients, residuals, fitted, vcov
    else:
        return coefficients, residuals, vcov


def compute_robust_vcov(
    X: np.ndarray,
    residuals: np.ndarray,
    cluster_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute heteroskedasticity-robust or cluster-robust variance-covariance matrix.

    Uses the sandwich estimator: (X'X)^{-1} * meat * (X'X)^{-1}

    Parameters
    ----------
    X : ndarray of shape (n, k)
        Design matrix.
    residuals : ndarray of shape (n,)
        OLS residuals.
    cluster_ids : ndarray of shape (n,), optional
        Cluster identifiers. If None, computes HC1 robust SEs.

    Returns
    -------
    vcov : ndarray of shape (k, k)
        Variance-covariance matrix.

    Notes
    -----
    For HC1 (no clustering):
        meat = X' * diag(u^2) * X
        adjustment = n / (n - k)

    For cluster-robust:
        meat = sum_g (X_g' u_g)(X_g' u_g)'
        adjustment = (G / (G-1)) * ((n-1) / (n-k))

    The cluster-robust computation is vectorized using pandas groupby,
    which is much faster than a Python loop over clusters.
    """
    n, k = X.shape
    XtX = X.T @ X

    if cluster_ids is None:
        # HC1 (heteroskedasticity-robust) standard errors
        adjustment = n / (n - k)
        u_squared = residuals**2
        # Vectorized meat computation: X' diag(u^2) X = (X * u^2)' X
        meat = X.T @ (X * u_squared[:, np.newaxis])
    else:
        # Cluster-robust standard errors (vectorized via groupby)
        cluster_ids = np.asarray(cluster_ids)
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        if n_clusters < 2:
            raise ValueError(
                f"Need at least 2 clusters for cluster-robust SEs, got {n_clusters}"
            )

        # Small-sample adjustment
        adjustment = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))

        # Compute cluster-level scores: sum of X_i * u_i within each cluster
        # scores[i] = X[i] * residuals[i] for each observation
        scores = X * residuals[:, np.newaxis]  # (n, k)

        # Sum scores within each cluster using pandas groupby (vectorized)
        # This is much faster than looping over clusters
        cluster_scores = pd.DataFrame(scores).groupby(cluster_ids).sum().values  # (G, k)

        # Meat is the outer product sum: sum_g (score_g)(score_g)'
        # Equivalent to cluster_scores.T @ cluster_scores
        meat = cluster_scores.T @ cluster_scores  # (k, k)

    # Sandwich estimator: (X'X)^{-1} meat (X'X)^{-1}
    # Solve (X'X) temp = meat, then solve (X'X) vcov' = temp'
    # More stable than explicit inverse
    try:
        temp = np.linalg.solve(XtX, meat)
        vcov = adjustment * np.linalg.solve(XtX, temp.T).T
    except np.linalg.LinAlgError as e:
        if "Singular" in str(e):
            raise ValueError(
                "Design matrix is rank-deficient (singular X'X matrix). "
                "This indicates perfect multicollinearity. Check your fixed effects "
                "and covariates for linear dependencies."
            ) from e
        raise

    return vcov


def compute_r_squared(
    y: np.ndarray,
    residuals: np.ndarray,
    adjusted: bool = False,
    n_params: int = 0,
) -> float:
    """
    Compute R-squared or adjusted R-squared.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Response vector.
    residuals : ndarray of shape (n,)
        OLS residuals.
    adjusted : bool, default False
        If True, compute adjusted R-squared.
    n_params : int, default 0
        Number of parameters (including intercept). Required if adjusted=True.

    Returns
    -------
    r_squared : float
        R-squared or adjusted R-squared.
    """
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot == 0:
        return 0.0

    r_squared = 1 - (ss_res / ss_tot)

    if adjusted:
        n = len(y)
        if n <= n_params:
            return r_squared
        r_squared = 1 - (1 - r_squared) * (n - 1) / (n - n_params)

    return r_squared
