"""
Unified linear algebra backend for diff-diff.

This module provides optimized OLS and variance estimation with an optional
Rust backend for maximum performance.

The key optimizations are:
1. scipy.linalg.lstsq with 'gelsy' driver (QR-based, faster than SVD)
2. Vectorized cluster-robust SE via groupby (eliminates O(n*clusters) loop)
3. Single interface for all estimators (reduces code duplication)
4. Optional Rust backend for additional speedup (when available)

The Rust backend is automatically used when available, with transparent
fallback to NumPy/SciPy implementations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import lstsq as scipy_lstsq

# Import Rust backend if available (from _backend to avoid circular imports)
from diff_diff._backend import (
    HAS_RUST_BACKEND,
    _rust_compute_robust_vcov,
    _rust_solve_ols,
)


def solve_ols(
    X: np.ndarray,
    y: np.ndarray,
    *,
    cluster_ids: Optional[np.ndarray] = None,
    return_vcov: bool = True,
    return_fitted: bool = False,
    check_finite: bool = True,
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
    check_finite : bool, default True
        Whether to check that X and y contain only finite values (no NaN/Inf).
        Set to False for faster computation if you are certain your data is clean.

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

    # Check for NaN/Inf values if requested
    if check_finite:
        if not np.isfinite(X).all():
            raise ValueError(
                "X contains NaN or Inf values. "
                "Clean your data or set check_finite=False to skip this check."
            )
        if not np.isfinite(y).all():
            raise ValueError(
                "y contains NaN or Inf values. "
                "Clean your data or set check_finite=False to skip this check."
            )

    # Use Rust backend if available
    # Note: Fall back to NumPy if check_finite=False since Rust's LAPACK
    # doesn't support non-finite values
    if HAS_RUST_BACKEND and check_finite:
        # Ensure contiguous arrays for Rust
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)

        # Convert cluster_ids to int64 for Rust (if provided)
        cluster_ids_int = None
        if cluster_ids is not None:
            cluster_ids_int = pd.factorize(cluster_ids)[0].astype(np.int64)

        try:
            coefficients, residuals, vcov = _rust_solve_ols(
                X, y, cluster_ids_int, return_vcov
            )
        except ValueError as e:
            # Translate Rust LAPACK errors to consistent Python error messages
            error_msg = str(e)
            if "Matrix inversion failed" in error_msg or "Least squares failed" in error_msg:
                raise ValueError(
                    "Design matrix is rank-deficient (singular X'X matrix). "
                    "This indicates perfect multicollinearity. Check your fixed effects "
                    "and covariates for linear dependencies."
                ) from e
            raise

        if return_fitted:
            fitted = X @ coefficients
            return coefficients, residuals, fitted, vcov
        else:
            return coefficients, residuals, vcov

    # Fallback to NumPy/SciPy implementation
    return _solve_ols_numpy(
        X, y, cluster_ids=cluster_ids, return_vcov=return_vcov, return_fitted=return_fitted
    )


def _solve_ols_numpy(
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
    NumPy/SciPy fallback implementation of solve_ols.

    Uses normal equations (X'X)^{-1} X'y solved via np.linalg.solve for speed,
    with fallback to scipy.lstsq (QR) for rank-deficient matrices.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n, k).
    y : np.ndarray
        Response vector of shape (n,).
    cluster_ids : np.ndarray, optional
        Cluster identifiers for cluster-robust SEs.
    return_vcov : bool
        Whether to compute variance-covariance matrix.
    return_fitted : bool
        Whether to return fitted values.

    Returns
    -------
    coefficients : np.ndarray
        OLS coefficients of shape (k,).
    residuals : np.ndarray
        Residuals of shape (n,).
    fitted : np.ndarray, optional
        Fitted values if return_fitted=True.
    vcov : np.ndarray, optional
        Variance-covariance matrix if return_vcov=True.
    """
    # Solve OLS using normal equations: (X'X) beta = X'y
    # This is ~14x faster than QR-based lstsq for typical DiD problems
    # np.linalg.solve uses LAPACK's gesv (LU factorization with pivoting)
    XtX = X.T @ X
    Xty = X.T @ y

    try:
        coefficients = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        # Fall back to QR-based solver for rank-deficient matrices
        # This is slower but handles singular/near-singular cases
        coefficients = scipy_lstsq(X, y, lapack_driver="gelsy", check_finite=False)[0]

    # Compute residuals and fitted values
    fitted = X @ coefficients
    residuals = y - fitted

    # Compute variance-covariance matrix if requested
    vcov = None
    if return_vcov:
        vcov = _compute_robust_vcov_numpy(X, residuals, cluster_ids)

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
    # Use Rust backend if available
    if HAS_RUST_BACKEND:
        X = np.ascontiguousarray(X, dtype=np.float64)
        residuals = np.ascontiguousarray(residuals, dtype=np.float64)

        cluster_ids_int = None
        if cluster_ids is not None:
            cluster_ids_int = pd.factorize(cluster_ids)[0].astype(np.int64)

        try:
            return _rust_compute_robust_vcov(X, residuals, cluster_ids_int)
        except ValueError as e:
            # Translate Rust LAPACK errors to consistent Python error messages
            error_msg = str(e)
            if "Matrix inversion failed" in error_msg:
                raise ValueError(
                    "Design matrix is rank-deficient (singular X'X matrix). "
                    "This indicates perfect multicollinearity. Check your fixed effects "
                    "and covariates for linear dependencies."
                ) from e
            raise

    # Fallback to NumPy implementation
    return _compute_robust_vcov_numpy(X, residuals, cluster_ids)


def _compute_robust_vcov_numpy(
    X: np.ndarray,
    residuals: np.ndarray,
    cluster_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    NumPy fallback implementation of compute_robust_vcov.

    Computes HC1 (heteroskedasticity-robust) or cluster-robust variance-covariance
    matrix using the sandwich estimator.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n, k).
    residuals : np.ndarray
        OLS residuals of shape (n,).
    cluster_ids : np.ndarray, optional
        Cluster identifiers. If None, uses HC1. If provided, uses
        cluster-robust with G/(G-1) small-sample adjustment.

    Returns
    -------
    vcov : np.ndarray
        Variance-covariance matrix of shape (k, k).

    Notes
    -----
    Uses vectorized groupby aggregation for cluster-robust SEs to avoid
    the O(n * G) loop that would be required with explicit iteration.
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


# =============================================================================
# LinearRegression Helper Class
# =============================================================================


@dataclass
class InferenceResult:
    """
    Container for inference results on a single coefficient.

    This dataclass provides a unified way to access coefficient estimates
    and their associated inference statistics.

    Attributes
    ----------
    coefficient : float
        The point estimate of the coefficient.
    se : float
        Standard error of the coefficient.
    t_stat : float
        T-statistic (coefficient / se).
    p_value : float
        Two-sided p-value for the t-statistic.
    conf_int : tuple of (float, float)
        Confidence interval (lower, upper).
    df : int or None
        Degrees of freedom used for inference. None if using normal distribution.
    alpha : float
        Significance level used for confidence interval.

    Examples
    --------
    >>> result = InferenceResult(
    ...     coefficient=2.5, se=0.5, t_stat=5.0, p_value=0.001,
    ...     conf_int=(1.52, 3.48), df=100, alpha=0.05
    ... )
    >>> result.is_significant()
    True
    >>> result.significance_stars()
    '***'
    """

    coefficient: float
    se: float
    t_stat: float
    p_value: float
    conf_int: Tuple[float, float]
    df: Optional[int] = None
    alpha: float = 0.05

    def is_significant(self, alpha: Optional[float] = None) -> bool:
        """Check if the coefficient is statistically significant."""
        threshold = alpha if alpha is not None else self.alpha
        return self.p_value < threshold

    def significance_stars(self) -> str:
        """Return significance stars based on p-value."""
        if self.p_value < 0.001:
            return "***"
        elif self.p_value < 0.01:
            return "**"
        elif self.p_value < 0.05:
            return "*"
        elif self.p_value < 0.1:
            return "."
        return ""

    def to_dict(self) -> Dict[str, Union[float, Tuple[float, float], int, None]]:
        """Convert to dictionary representation."""
        return {
            "coefficient": self.coefficient,
            "se": self.se,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "conf_int": self.conf_int,
            "df": self.df,
            "alpha": self.alpha,
        }


class LinearRegression:
    """
    OLS regression helper with unified coefficient extraction and inference.

    This class wraps the low-level `solve_ols` function and provides a clean
    interface for fitting regressions and extracting coefficient-level inference.
    It eliminates code duplication across estimators by centralizing the common
    pattern of: fit OLS -> extract coefficient -> compute SE -> compute t-stat
    -> compute p-value -> compute CI.

    Parameters
    ----------
    include_intercept : bool, default True
        Whether to automatically add an intercept column to the design matrix.
    robust : bool, default True
        Whether to use heteroskedasticity-robust (HC1) standard errors.
        If False and cluster_ids is None, uses classical OLS standard errors.
    cluster_ids : array-like, optional
        Cluster identifiers for cluster-robust standard errors.
        Overrides the `robust` parameter if provided.
    alpha : float, default 0.05
        Significance level for confidence intervals.

    Attributes
    ----------
    coefficients_ : ndarray
        Fitted coefficient values (available after fit).
    vcov_ : ndarray
        Variance-covariance matrix (available after fit).
    residuals_ : ndarray
        Residuals from the fit (available after fit).
    fitted_values_ : ndarray
        Fitted values from the fit (available after fit).
    n_obs_ : int
        Number of observations (available after fit).
    n_params_ : int
        Number of parameters including intercept (available after fit).
    df_ : int
        Degrees of freedom (n - k) (available after fit).

    Examples
    --------
    Basic usage with automatic intercept:

    >>> import numpy as np
    >>> from diff_diff.linalg import LinearRegression
    >>> X = np.random.randn(100, 2)
    >>> y = 1 + 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100)
    >>> reg = LinearRegression().fit(X, y)
    >>> print(f"Intercept: {reg.coefficients_[0]:.2f}")
    >>> inference = reg.get_inference(1)  # inference for first predictor
    >>> print(f"Coef: {inference.coefficient:.2f}, SE: {inference.se:.2f}")

    Using with cluster-robust standard errors:

    >>> cluster_ids = np.repeat(np.arange(20), 5)  # 20 clusters of 5
    >>> reg = LinearRegression(cluster_ids=cluster_ids).fit(X, y)
    >>> inference = reg.get_inference(1)
    >>> print(f"Cluster-robust SE: {inference.se:.2f}")

    Extracting multiple coefficients at once:

    >>> results = reg.get_inference_batch([1, 2])
    >>> for idx, inf in results.items():
    ...     print(f"Coef {idx}: {inf.coefficient:.2f} ({inf.significance_stars()})")
    """

    def __init__(
        self,
        include_intercept: bool = True,
        robust: bool = True,
        cluster_ids: Optional[np.ndarray] = None,
        alpha: float = 0.05,
    ):
        self.include_intercept = include_intercept
        self.robust = robust
        self.cluster_ids = cluster_ids
        self.alpha = alpha

        # Fitted attributes (set by fit())
        self.coefficients_: Optional[np.ndarray] = None
        self.vcov_: Optional[np.ndarray] = None
        self.residuals_: Optional[np.ndarray] = None
        self.fitted_values_: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._X: Optional[np.ndarray] = None
        self.n_obs_: Optional[int] = None
        self.n_params_: Optional[int] = None
        self.df_: Optional[int] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        cluster_ids: Optional[np.ndarray] = None,
        df_adjustment: int = 0,
    ) -> "LinearRegression":
        """
        Fit OLS regression.

        Parameters
        ----------
        X : ndarray of shape (n, k)
            Design matrix. An intercept column will be added if include_intercept=True.
        y : ndarray of shape (n,)
            Response vector.
        cluster_ids : ndarray, optional
            Cluster identifiers for this fit. Overrides the instance-level
            cluster_ids if provided.
        df_adjustment : int, default 0
            Additional degrees of freedom adjustment (e.g., for absorbed fixed effects).
            The effective df will be n - k - df_adjustment.

        Returns
        -------
        self : LinearRegression
            Fitted estimator.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # Add intercept if requested
        if self.include_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])

        # Use provided cluster_ids or fall back to instance-level
        effective_cluster_ids = cluster_ids if cluster_ids is not None else self.cluster_ids

        # Determine if we need robust/cluster vcov
        compute_vcov = True

        if self.robust or effective_cluster_ids is not None:
            # Use solve_ols with robust/cluster SEs
            coefficients, residuals, fitted, vcov = solve_ols(
                X, y,
                cluster_ids=effective_cluster_ids,
                return_fitted=True,
                return_vcov=compute_vcov,
            )
        else:
            # Classical OLS - compute vcov separately
            coefficients, residuals, fitted, _ = solve_ols(
                X, y,
                return_fitted=True,
                return_vcov=False,
            )
            # Compute classical OLS variance-covariance matrix
            n, k = X.shape
            mse = np.sum(residuals**2) / (n - k)
            try:
                vcov = np.linalg.solve(X.T @ X, mse * np.eye(k))
            except np.linalg.LinAlgError:
                # Fall back to pseudo-inverse for singular matrices
                vcov = np.linalg.pinv(X.T @ X) * mse

        # Store fitted attributes
        self.coefficients_ = coefficients
        self.vcov_ = vcov
        self.residuals_ = residuals
        self.fitted_values_ = fitted
        self._y = y
        self._X = X
        self.n_obs_ = X.shape[0]
        self.n_params_ = X.shape[1]
        self.df_ = self.n_obs_ - self.n_params_ - df_adjustment

        return self

    def _check_fitted(self) -> None:
        """Raise error if model has not been fitted."""
        if self.coefficients_ is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

    def get_coefficient(self, index: int) -> float:
        """
        Get the coefficient value at a specific index.

        Parameters
        ----------
        index : int
            Index of the coefficient in the coefficient array.

        Returns
        -------
        float
            Coefficient value.
        """
        self._check_fitted()
        return float(self.coefficients_[index])

    def get_se(self, index: int) -> float:
        """
        Get the standard error for a coefficient.

        Parameters
        ----------
        index : int
            Index of the coefficient.

        Returns
        -------
        float
            Standard error.
        """
        self._check_fitted()
        return float(np.sqrt(self.vcov_[index, index]))

    def get_inference(
        self,
        index: int,
        alpha: Optional[float] = None,
        df: Optional[int] = None,
    ) -> InferenceResult:
        """
        Get full inference results for a coefficient.

        This is the primary method for extracting coefficient-level inference,
        returning all statistics in a single call.

        Parameters
        ----------
        index : int
            Index of the coefficient in the coefficient array.
        alpha : float, optional
            Significance level for CI. Defaults to instance-level alpha.
        df : int, optional
            Degrees of freedom. Defaults to fitted df (n - k - df_adjustment).
            Set to None explicitly to use normal distribution instead of t.

        Returns
        -------
        InferenceResult
            Dataclass containing coefficient, se, t_stat, p_value, conf_int.

        Examples
        --------
        >>> reg = LinearRegression().fit(X, y)
        >>> result = reg.get_inference(1)
        >>> print(f"Effect: {result.coefficient:.3f} (SE: {result.se:.3f})")
        >>> print(f"95% CI: [{result.conf_int[0]:.3f}, {result.conf_int[1]:.3f}]")
        >>> if result.is_significant():
        ...     print("Statistically significant!")
        """
        self._check_fitted()

        coef = float(self.coefficients_[index])
        se = float(np.sqrt(self.vcov_[index, index]))
        t_stat = coef / se if se > 0 else 0.0

        # Use instance alpha if not provided
        effective_alpha = alpha if alpha is not None else self.alpha

        # Use fitted df if not explicitly provided
        # Note: df=None means use normal distribution
        effective_df = df if df is not None else self.df_

        # Compute p-value
        p_value = _compute_p_value(t_stat, df=effective_df)

        # Compute confidence interval
        conf_int = _compute_confidence_interval(coef, se, effective_alpha, df=effective_df)

        return InferenceResult(
            coefficient=coef,
            se=se,
            t_stat=t_stat,
            p_value=p_value,
            conf_int=conf_int,
            df=effective_df,
            alpha=effective_alpha,
        )

    def get_inference_batch(
        self,
        indices: List[int],
        alpha: Optional[float] = None,
        df: Optional[int] = None,
    ) -> Dict[int, InferenceResult]:
        """
        Get inference results for multiple coefficients.

        Parameters
        ----------
        indices : list of int
            Indices of coefficients to extract.
        alpha : float, optional
            Significance level for CIs. Defaults to instance-level alpha.
        df : int, optional
            Degrees of freedom. Defaults to fitted df.

        Returns
        -------
        dict
            Dictionary mapping index -> InferenceResult.

        Examples
        --------
        >>> reg = LinearRegression().fit(X, y)
        >>> results = reg.get_inference_batch([1, 2, 3])
        >>> for idx, inf in results.items():
        ...     print(f"Coef {idx}: {inf.coefficient:.3f} {inf.significance_stars()}")
        """
        self._check_fitted()
        return {idx: self.get_inference(idx, alpha=alpha, df=df) for idx in indices}

    def get_all_inference(
        self,
        alpha: Optional[float] = None,
        df: Optional[int] = None,
    ) -> List[InferenceResult]:
        """
        Get inference results for all coefficients.

        Parameters
        ----------
        alpha : float, optional
            Significance level for CIs. Defaults to instance-level alpha.
        df : int, optional
            Degrees of freedom. Defaults to fitted df.

        Returns
        -------
        list of InferenceResult
            Inference results for each coefficient in order.
        """
        self._check_fitted()
        return [
            self.get_inference(i, alpha=alpha, df=df)
            for i in range(len(self.coefficients_))
        ]

    def r_squared(self, adjusted: bool = False) -> float:
        """
        Compute R-squared or adjusted R-squared.

        Parameters
        ----------
        adjusted : bool, default False
            If True, return adjusted R-squared.

        Returns
        -------
        float
            R-squared value.
        """
        self._check_fitted()
        return compute_r_squared(
            self._y, self.residuals_, adjusted=adjusted, n_params=self.n_params_
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the fitted model.

        Parameters
        ----------
        X : ndarray of shape (n, k)
            Design matrix for prediction. Should have same number of columns
            as the original X (excluding intercept if include_intercept=True).

        Returns
        -------
        ndarray
            Predicted values.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)

        if self.include_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])

        return X @ self.coefficients_


# =============================================================================
# Internal helpers for inference (used by LinearRegression)
# =============================================================================


def _compute_p_value(
    t_stat: float,
    df: Optional[int] = None,
    two_sided: bool = True,
) -> float:
    """
    Compute p-value for a t-statistic.

    Parameters
    ----------
    t_stat : float
        T-statistic.
    df : int, optional
        Degrees of freedom. If None, uses normal distribution.
    two_sided : bool, default True
        Whether to compute two-sided p-value.

    Returns
    -------
    float
        P-value.
    """
    if df is not None and df > 0:
        p_value = stats.t.sf(np.abs(t_stat), df)
    else:
        p_value = stats.norm.sf(np.abs(t_stat))

    if two_sided:
        p_value *= 2

    return float(p_value)


def _compute_confidence_interval(
    estimate: float,
    se: float,
    alpha: float = 0.05,
    df: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Compute confidence interval for an estimate.

    Parameters
    ----------
    estimate : float
        Point estimate.
    se : float
        Standard error.
    alpha : float, default 0.05
        Significance level (0.05 for 95% CI).
    df : int, optional
        Degrees of freedom. If None, uses normal distribution.

    Returns
    -------
    tuple of (float, float)
        (lower_bound, upper_bound) of confidence interval.
    """
    if df is not None and df > 0:
        critical_value = stats.t.ppf(1 - alpha / 2, df)
    else:
        critical_value = stats.norm.ppf(1 - alpha / 2)

    lower = estimate - critical_value * se
    upper = estimate + critical_value * se

    return (lower, upper)
