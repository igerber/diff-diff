"""
Unified linear algebra backend for diff-diff.

This module provides optimized OLS and variance estimation with an optional
Rust backend for maximum performance.

The key optimizations are:
1. scipy.linalg.lstsq with 'gelsd' driver (SVD-based, handles rank-deficient matrices)
2. Vectorized cluster-robust SE via groupby (eliminates O(n*clusters) loop)
3. Single interface for all estimators (reduces code duplication)
4. Optional Rust backend for additional speedup (when available)
5. R-style rank deficiency handling: detect, warn, and set NA for dropped columns

The Rust backend is automatically used when available, with transparent
fallback to NumPy/SciPy implementations.

Rank Deficiency Handling
------------------------
When a design matrix is rank-deficient (has linearly dependent columns), the OLS
solution is not unique. This module follows R's `lm()` approach:

1. Detect rank deficiency using pivoted QR decomposition
2. Identify which columns are linearly dependent
3. Drop redundant columns from the solve
4. Set NA (NaN) for coefficients of dropped columns
5. Warn with clear message listing dropped columns
6. Compute valid SEs for remaining (identified) coefficients

This is controlled by the `rank_deficient_action` parameter:
- "warn" (default): Emit warning, set NA for dropped coefficients
- "error": Raise ValueError with dropped column information
- "silent": No warning, but still set NA for dropped coefficients
"""

import warnings
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union, overload

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import lstsq as scipy_lstsq
from scipy.linalg import qr

# Import Rust backend if available (from _backend to avoid circular imports)
from diff_diff._backend import (
    HAS_RUST_BACKEND,
    _rust_compute_robust_vcov,
    _rust_solve_ols,
)

# =============================================================================
# Utility Functions
# =============================================================================


def _factorize_cluster_ids(cluster_ids: np.ndarray) -> np.ndarray:
    """
    Convert cluster IDs to contiguous integer codes for Rust backend.

    Handles string, categorical, or non-contiguous integer cluster IDs by
    mapping them to contiguous integers starting from 0.

    Parameters
    ----------
    cluster_ids : np.ndarray
        Cluster identifiers (can be strings, integers, or categorical).

    Returns
    -------
    np.ndarray
        Integer cluster codes (dtype int64) suitable for Rust backend.
    """
    # Use pandas factorize for efficient conversion of any dtype
    codes, _ = pd.factorize(cluster_ids)
    return codes.astype(np.int64)


# =============================================================================
# Rank Deficiency Detection and Handling
# =============================================================================


def _detect_rank_deficiency(
    X: np.ndarray,
    rcond: Optional[float] = None,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Detect rank deficiency using pivoted QR decomposition.

    This follows R's lm() approach of using pivoted QR to detect which columns
    are linearly dependent. The pivoting ensures we drop the "least important"
    columns (those with smallest contribution to the column space).

    Parameters
    ----------
    X : ndarray of shape (n, k)
        Design matrix.
    rcond : float, optional
        Relative condition number threshold for determining rank.
        Diagonal elements of R smaller than rcond * max(|R_ii|) are treated
        as zero. If None, uses 1e-07 to match R's qr() default tolerance.

    Returns
    -------
    rank : int
        Numerical rank of the matrix.
    dropped_cols : ndarray of int
        Indices of columns that are linearly dependent (should be dropped).
        Empty if matrix is full rank.
    pivot : ndarray of int
        Column permutation from QR decomposition.
    """
    n, k = X.shape

    # Compute pivoted QR decomposition: X @ P = Q @ R
    # P is a permutation matrix, represented as pivot indices
    Q, R, pivot = qr(X, mode="economic", pivoting=True)

    # Determine rank tolerance
    # R's qr() uses tol = 1e-07 by default, which is sqrt(eps) ≈ 1.49e-08
    # We use 1e-07 to match R's lm() behavior for consistency
    if rcond is None:
        rcond = 1e-07

    # The diagonal of R contains information about linear independence
    # After pivoting, |R[i,i]| is decreasing
    r_diag = np.abs(np.diag(R))

    # Find numerical rank: count singular values above threshold
    # The threshold is relative to the largest diagonal element
    if r_diag[0] == 0:
        rank = 0
    else:
        tol = rcond * r_diag[0]
        rank = int(np.sum(r_diag > tol))

    # Columns after rank position (in pivot order) are linearly dependent
    # We need to map back to original column indices
    if rank < k:
        dropped_cols = np.sort(pivot[rank:])
    else:
        dropped_cols = np.array([], dtype=int)

    return rank, dropped_cols, pivot


def _format_dropped_columns(
    dropped_cols: np.ndarray,
    column_names: Optional[List[str]] = None,
) -> str:
    """
    Format dropped column information for error/warning messages.

    Parameters
    ----------
    dropped_cols : ndarray of int
        Indices of dropped columns.
    column_names : list of str, optional
        Names for the columns. If None, uses indices.

    Returns
    -------
    str
        Formatted string describing dropped columns.
    """
    if len(dropped_cols) == 0:
        return ""

    if column_names is not None:
        names = [column_names[i] if i < len(column_names) else f"column {i}" for i in dropped_cols]
        if len(names) == 1:
            return f"'{names[0]}'"
        elif len(names) <= 5:
            return ", ".join(f"'{n}'" for n in names)
        else:
            shown = ", ".join(f"'{n}'" for n in names[:5])
            return f"{shown}, ... and {len(names) - 5} more"
    else:
        if len(dropped_cols) == 1:
            return f"column {dropped_cols[0]}"
        elif len(dropped_cols) <= 5:
            return ", ".join(f"column {i}" for i in dropped_cols)
        else:
            shown = ", ".join(f"column {i}" for i in dropped_cols[:5])
            return f"{shown}, ... and {len(dropped_cols) - 5} more"


def _expand_coefficients_with_nan(
    coef_reduced: np.ndarray,
    k_full: int,
    kept_cols: np.ndarray,
) -> np.ndarray:
    """
    Expand reduced coefficients to full size, filling dropped columns with NaN.

    Parameters
    ----------
    coef_reduced : ndarray of shape (rank,)
        Coefficients for kept columns only.
    k_full : int
        Total number of columns in original design matrix.
    kept_cols : ndarray of int
        Indices of columns that were kept.

    Returns
    -------
    ndarray of shape (k_full,)
        Full coefficient vector with NaN for dropped columns.
    """
    coef_full = np.full(k_full, np.nan)
    coef_full[kept_cols] = coef_reduced
    return coef_full


def _expand_vcov_with_nan(
    vcov_reduced: np.ndarray,
    k_full: int,
    kept_cols: np.ndarray,
) -> np.ndarray:
    """
    Expand reduced vcov matrix to full size, filling dropped entries with NaN.

    Parameters
    ----------
    vcov_reduced : ndarray of shape (rank, rank)
        Variance-covariance matrix for kept columns only.
    k_full : int
        Total number of columns in original design matrix.
    kept_cols : ndarray of int
        Indices of columns that were kept.

    Returns
    -------
    ndarray of shape (k_full, k_full)
        Full vcov matrix with NaN for dropped rows/columns.
    """
    vcov_full = np.full((k_full, k_full), np.nan)
    # Use advanced indexing to fill in the kept entries
    ix = np.ix_(kept_cols, kept_cols)
    vcov_full[ix] = vcov_reduced
    return vcov_full


def _solve_ols_rust(
    X: np.ndarray,
    y: np.ndarray,
    *,
    cluster_ids: Optional[np.ndarray] = None,
    return_vcov: bool = True,
    return_fitted: bool = False,
) -> Optional[
    Union[
        Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]],
        Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]],
    ]
]:
    """
    Rust backend implementation of solve_ols for full-rank matrices.

    This is only called when:
    1. The Rust backend is available
    2. The design matrix is full rank (no rank deficiency handling needed)

    For rank-deficient matrices, the Python backend is used instead to
    properly handle R-style NA coefficients for dropped columns.

    Why the backends differ (by design):
    - Rust uses SVD-based solve (minimum-norm solution for rank-deficient)
    - Python uses pivoted QR to identify and drop linearly dependent columns
    - ndarray-linalg doesn't support QR with pivoting, so Rust can't identify
      which specific columns to drop
    - For full-rank matrices, both approaches give identical results
    - For rank-deficient matrices, only Python can provide R-style NA handling

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n, k), must be full rank.
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
    None
        If Rust backend detects numerical instability and caller should
        fall back to Python backend.
    """
    # Convert cluster_ids to int64 for Rust (handles string/categorical IDs)
    if cluster_ids is not None:
        cluster_ids = _factorize_cluster_ids(cluster_ids)

    # Call Rust backend with fallback on numerical instability
    try:
        coefficients, residuals, vcov = _rust_solve_ols(
            X, y, cluster_ids=cluster_ids, return_vcov=return_vcov
        )
    except ValueError as e:
        error_msg = str(e).lower()
        if "numerically unstable" in error_msg or "singular" in error_msg:
            warnings.warn(
                f"Rust backend detected numerical instability: {e}. "
                "Falling back to Python backend.",
                UserWarning,
                stacklevel=3,
            )
            return None  # Signal caller to use Python fallback
        raise

    # Convert to numpy arrays
    coefficients = np.asarray(coefficients)
    residuals = np.asarray(residuals)
    if vcov is not None:
        vcov = np.asarray(vcov)

    # Return with optional fitted values
    if return_fitted:
        fitted = np.dot(X, coefficients)
        return coefficients, residuals, fitted, vcov
    else:
        return coefficients, residuals, vcov


@overload
def solve_ols(
    X: np.ndarray,
    y: np.ndarray,
    *,
    cluster_ids: Optional[np.ndarray] = ...,
    return_vcov: bool = ...,
    return_fitted: Literal[False] = ...,
    check_finite: bool = ...,
    rank_deficient_action: str = ...,
    column_names: Optional[List[str]] = ...,
    skip_rank_check: bool = ...,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]: ...


@overload
def solve_ols(
    X: np.ndarray,
    y: np.ndarray,
    *,
    cluster_ids: Optional[np.ndarray] = ...,
    return_vcov: bool = ...,
    return_fitted: Literal[True],
    check_finite: bool = ...,
    rank_deficient_action: str = ...,
    column_names: Optional[List[str]] = ...,
    skip_rank_check: bool = ...,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]: ...


@overload
def solve_ols(
    X: np.ndarray,
    y: np.ndarray,
    *,
    cluster_ids: Optional[np.ndarray] = ...,
    return_vcov: bool = ...,
    return_fitted: bool,
    check_finite: bool = ...,
    rank_deficient_action: str = ...,
    column_names: Optional[List[str]] = ...,
    skip_rank_check: bool = ...,
) -> Union[
    Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]],
    Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]],
]: ...


_VALID_WEIGHT_TYPES = {"pweight", "fweight", "aweight"}


def _validate_weights(weights, weight_type, n):
    """Validate weights array and weight_type for solve_ols/LinearRegression."""
    if weight_type not in _VALID_WEIGHT_TYPES:
        raise ValueError(
            f"weight_type must be one of {_VALID_WEIGHT_TYPES}, " f"got '{weight_type}'"
        )
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape[0] != n:
            raise ValueError(f"weights length ({weights.shape[0]}) must match " f"X rows ({n})")
        if np.any(np.isnan(weights)):
            raise ValueError("Weights contain NaN values")
        if np.any(np.isinf(weights)):
            raise ValueError("Weights contain Inf values")
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative")
        if np.sum(weights) <= 0:
            raise ValueError(
                "Weights sum to zero — no observations have positive weight. "
                "Cannot fit a model on an empty effective sample."
            )
        if weight_type == "fweight":
            fractional = weights - np.round(weights)
            if np.any(np.abs(fractional) > 1e-10):
                raise ValueError(
                    "Frequency weights (fweight) must be non-negative integers. "
                    "Fractional values detected. Use pweight for non-integer weights."
                )
    return weights


def solve_ols(
    X: np.ndarray,
    y: np.ndarray,
    *,
    cluster_ids: Optional[np.ndarray] = None,
    return_vcov: bool = True,
    return_fitted: bool = False,
    check_finite: bool = True,
    rank_deficient_action: str = "warn",
    column_names: Optional[List[str]] = None,
    skip_rank_check: bool = False,
    weights: Optional[np.ndarray] = None,
    weight_type: str = "pweight",
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
    rank_deficient_action : str, default "warn"
        How to handle rank-deficient design matrices:
        - "warn": Emit warning and set NaN for dropped coefficients (R-style)
        - "error": Raise ValueError with dropped column information
        - "silent": No warning, but still set NaN for dropped coefficients
    column_names : list of str, optional
        Names for the columns (used in warning/error messages).
        If None, columns are referred to by their indices.
    skip_rank_check : bool, default False
        If True, skip the pivoted QR rank check and use Rust backend directly
        (when available). This saves O(nk²) computation but will not detect
        rank-deficient matrices. Use only when you know the design matrix is
        full rank. If the matrix is actually rank-deficient, results may be
        incorrect (minimum-norm solution instead of R-style NA handling).
    weights : ndarray of shape (n,), optional
        Observation weights for Weighted Least Squares. When provided,
        minimizes sum(w_i * (y_i - X_i @ beta)^2). Weights should be
        pre-normalized (e.g., mean=1 for pweights).
    weight_type : str, default "pweight"
        Type of weights: "pweight" (inverse selection probability),
        "fweight" (frequency), or "aweight" (inverse variance).
        Affects variance estimation but not coefficient computation.

    Returns
    -------
    coefficients : ndarray of shape (k,)
        OLS coefficient estimates. For rank-deficient matrices, coefficients
        of linearly dependent columns are set to NaN.
    residuals : ndarray of shape (n,)
        Residuals (y - fitted). For rank-deficient matrices, uses only
        identified coefficients to compute fitted values.
    fitted : ndarray of shape (n,), optional
        Fitted values. For full-rank matrices, this is X @ coefficients.
        For rank-deficient matrices, uses only identified coefficients
        (X_reduced @ coefficients_reduced). Only returned if return_fitted=True.
    vcov : ndarray of shape (k, k) or None
        Variance-covariance matrix (HC1 or cluster-robust).
        For rank-deficient matrices, rows/columns for dropped coefficients
        are filled with NaN. None if return_vcov=False.

    Notes
    -----
    This function detects rank-deficient matrices using pivoted QR decomposition
    and handles them following R's lm() approach:

    1. Detect linearly dependent columns via pivoted QR
    2. Drop redundant columns and solve the reduced system
    3. Set NaN for coefficients of dropped columns
    4. Compute valid SEs for identified coefficients only
    5. Expand vcov matrix with NaN for dropped rows/columns

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

    For rank-deficient matrices with collinear columns:

    >>> X = np.random.randn(100, 3)
    >>> X[:, 2] = X[:, 0] + X[:, 1]  # Perfect collinearity
    >>> y = np.random.randn(100)
    >>> coef, resid, vcov = solve_ols(X, y)  # Emits warning
    >>> print(np.isnan(coef[2]))  # Dropped column has NaN coefficient
    True
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
            f"X and y must have same number of observations: " f"{X.shape[0]} vs {y.shape[0]}"
        )

    n, k = X.shape
    if n < k:
        raise ValueError(
            f"Fewer observations ({n}) than parameters ({k}). "
            "Cannot solve underdetermined system."
        )

    # Validate rank_deficient_action
    valid_actions = {"warn", "error", "silent"}
    if rank_deficient_action not in valid_actions:
        raise ValueError(
            f"rank_deficient_action must be one of {valid_actions}, "
            f"got '{rank_deficient_action}'"
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

    # WLS transformation: apply sqrt(w) scaling to X and y
    # This happens BEFORE routing to Rust or NumPy backends — they receive
    # pre-transformed X_w, y_w and solve standard OLS.
    # Residuals are back-transformed to original scale afterward.
    _original_X = None
    _original_y = None
    if weights is not None:
        weights = _validate_weights(weights, weight_type, n)
        _original_X = X
        _original_y = y
        sqrt_w = np.sqrt(weights)
        X = X * sqrt_w[:, np.newaxis]
        y = y * sqrt_w

    # When weights are present, compute vcov separately on original-scale data
    # to avoid double-weighting. The backend only computes point estimates.
    _weighted_vcov_external = weights is not None
    _backend_return_vcov = return_vcov and not _weighted_vcov_external

    # Fast path: skip rank check and use Rust directly when requested
    # This saves O(nk²) QR overhead but won't detect rank-deficient matrices
    result = None  # Will hold the tuple from backend functions

    if skip_rank_check:
        if HAS_RUST_BACKEND and _rust_solve_ols is not None and weights is None:
            result = _solve_ols_rust(
                X,
                y,
                cluster_ids=cluster_ids,
                return_vcov=_backend_return_vcov,
                return_fitted=return_fitted,
            )
            # result is None on numerical instability → fall through
        if result is None:
            result = _solve_ols_numpy(
                X,
                y,
                cluster_ids=cluster_ids,
                return_vcov=_backend_return_vcov,
                return_fitted=return_fitted,
                rank_deficient_action=rank_deficient_action,
                column_names=column_names,
                _skip_rank_check=True,
            )
    else:
        # Check for rank deficiency using fast pivoted QR decomposition.
        # Rank detection operates on (possibly weighted) X since collinearity
        # depends on the weighted column space.
        rank, dropped_cols, pivot = _detect_rank_deficiency(X)
        is_rank_deficient = len(dropped_cols) > 0

        # Routing strategy:
        # - Full-rank + Rust available + no weights → fast Rust backend
        # - Weighted or rank-deficient → Python backend
        # - Rust numerical instability → Python fallback (via None return)
        if (
            HAS_RUST_BACKEND
            and _rust_solve_ols is not None
            and not is_rank_deficient
            and weights is None
        ):
            result = _solve_ols_rust(
                X,
                y,
                cluster_ids=cluster_ids,
                return_vcov=_backend_return_vcov,
                return_fitted=return_fitted,
            )

            if result is not None:
                vcov_check = result[-1]
                if _backend_return_vcov and vcov_check is not None and np.any(np.isnan(vcov_check)):
                    warnings.warn(
                        "Rust backend detected ill-conditioned matrix (NaN in variance-covariance). "
                        "Re-running with Python backend for proper rank detection.",
                        UserWarning,
                        stacklevel=2,
                    )
                    result = None  # Force Python fallback below

        if result is None:
            result = _solve_ols_numpy(
                X,
                y,
                cluster_ids=cluster_ids,
                return_vcov=_backend_return_vcov,
                return_fitted=return_fitted,
                rank_deficient_action=rank_deficient_action,
                column_names=column_names,
                _precomputed_rank_info=(rank, dropped_cols, pivot),
            )

    # Back-transform residuals and compute weighted vcov on original-scale data.
    # The WLS transform (sqrt(w) scaling) is for point estimates only. Vcov must
    # be computed on original X and residuals with weights applied exactly once.
    if _original_X is not None and _original_y is not None:
        if return_fitted:
            coefficients, _resid_w, _fitted_w, vcov_out = result
        else:
            coefficients, _resid_w, vcov_out = result

        # Handle rank-deficient case: use only identified columns for fitted values
        # to avoid NaN propagation from dropped coefficients
        nan_mask = np.isnan(coefficients)
        if np.any(nan_mask):
            kept_cols = np.where(~nan_mask)[0]
            fitted_orig = np.dot(_original_X[:, kept_cols], coefficients[kept_cols])
        else:
            fitted_orig = np.dot(_original_X, coefficients)
        residuals_orig = _original_y - fitted_orig

        if return_vcov:
            if np.any(nan_mask):
                kept_cols = np.where(~nan_mask)[0]
                if len(kept_cols) > 0:
                    vcov_reduced = _compute_robust_vcov_numpy(
                        _original_X[:, kept_cols],
                        residuals_orig,
                        cluster_ids,
                        weights=weights,
                        weight_type=weight_type,
                    )
                    vcov_out = _expand_vcov_with_nan(vcov_reduced, _original_X.shape[1], kept_cols)
                else:
                    vcov_out = np.full((_original_X.shape[1], _original_X.shape[1]), np.nan)
            else:
                vcov_out = _compute_robust_vcov_numpy(
                    _original_X,
                    residuals_orig,
                    cluster_ids,
                    weights=weights,
                    weight_type=weight_type,
                )

        if return_fitted:
            result = (coefficients, residuals_orig, fitted_orig, vcov_out)
        else:
            result = (coefficients, residuals_orig, vcov_out)

    return result


@overload
def _solve_ols_numpy(
    X: np.ndarray,
    y: np.ndarray,
    *,
    cluster_ids: Optional[np.ndarray] = ...,
    return_vcov: bool = ...,
    return_fitted: Literal[False] = ...,
    rank_deficient_action: str = ...,
    column_names: Optional[List[str]] = ...,
    _precomputed_rank_info: Optional[Tuple[int, np.ndarray, np.ndarray]] = ...,
    _skip_rank_check: bool = ...,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]: ...


@overload
def _solve_ols_numpy(
    X: np.ndarray,
    y: np.ndarray,
    *,
    cluster_ids: Optional[np.ndarray] = ...,
    return_vcov: bool = ...,
    return_fitted: Literal[True],
    rank_deficient_action: str = ...,
    column_names: Optional[List[str]] = ...,
    _precomputed_rank_info: Optional[Tuple[int, np.ndarray, np.ndarray]] = ...,
    _skip_rank_check: bool = ...,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]: ...


@overload
def _solve_ols_numpy(
    X: np.ndarray,
    y: np.ndarray,
    *,
    cluster_ids: Optional[np.ndarray] = ...,
    return_vcov: bool = ...,
    return_fitted: bool,
    rank_deficient_action: str = ...,
    column_names: Optional[List[str]] = ...,
    _precomputed_rank_info: Optional[Tuple[int, np.ndarray, np.ndarray]] = ...,
    _skip_rank_check: bool = ...,
) -> Union[
    Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]],
    Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]],
]: ...


def _solve_ols_numpy(
    X: np.ndarray,
    y: np.ndarray,
    *,
    cluster_ids: Optional[np.ndarray] = None,
    return_vcov: bool = True,
    return_fitted: bool = False,
    rank_deficient_action: str = "warn",
    column_names: Optional[List[str]] = None,
    _precomputed_rank_info: Optional[Tuple[int, np.ndarray, np.ndarray]] = None,
    _skip_rank_check: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]],
    Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]],
]:
    """
    NumPy/SciPy implementation of solve_ols with R-style rank deficiency handling.

    Detects rank-deficient matrices using pivoted QR decomposition and handles
    them following R's lm() approach: drop redundant columns, set NA (NaN) for
    their coefficients, and compute valid SEs for identified coefficients only.

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
    rank_deficient_action : str
        How to handle rank deficiency: "warn", "error", or "silent".
    column_names : list of str, optional
        Names for the columns (used in warning/error messages).
    _precomputed_rank_info : tuple, optional
        Pre-computed (rank, dropped_cols, pivot) from _detect_rank_deficiency.
        Used internally to avoid redundant computation when called from solve_ols.
    _skip_rank_check : bool, default False
        If True, skip rank detection entirely and assume full rank.
        Used when caller has already determined matrix is full rank.

    Returns
    -------
    coefficients : np.ndarray
        OLS coefficients of shape (k,). NaN for dropped columns.
    residuals : np.ndarray
        Residuals of shape (n,).
    fitted : np.ndarray, optional
        Fitted values if return_fitted=True.
    vcov : np.ndarray, optional
        Variance-covariance matrix if return_vcov=True. NaN for dropped rows/cols.
    """
    n, k = X.shape

    # Determine rank deficiency status
    if _skip_rank_check:
        # Caller guarantees full rank - skip expensive QR decomposition
        is_rank_deficient = False
        dropped_cols = np.array([], dtype=int)
    elif _precomputed_rank_info is not None:
        # Use pre-computed rank info
        rank, dropped_cols, pivot = _precomputed_rank_info
        is_rank_deficient = len(dropped_cols) > 0
    else:
        # Compute rank via pivoted QR
        rank, dropped_cols, pivot = _detect_rank_deficiency(X)
        is_rank_deficient = len(dropped_cols) > 0

    if is_rank_deficient:
        # Format dropped column information for messages
        dropped_str = _format_dropped_columns(dropped_cols, column_names)

        if rank_deficient_action == "error":
            raise ValueError(
                f"Design matrix is rank-deficient. {k - rank} of {k} columns are "
                f"linearly dependent and cannot be uniquely estimated: {dropped_str}. "
                "This indicates multicollinearity in your model specification."
            )
        elif rank_deficient_action == "warn":
            warnings.warn(
                f"Rank-deficient design matrix: dropping {k - rank} of {k} columns "
                f"({dropped_str}). Coefficients for these columns are set to NA. "
                "This may indicate multicollinearity in your model specification.",
                UserWarning,
                stacklevel=3,  # Point to user code that called solve_ols
            )
        # else: "silent" - no warning

        # Extract kept columns for the reduced solve
        kept_cols = np.array([i for i in range(k) if i not in dropped_cols])
        X_reduced = X[:, kept_cols]

        # Solve the reduced system (now full-rank)
        # Use cond=1e-07 for consistency with Rust backend and QR rank tolerance
        coefficients_reduced = scipy_lstsq(
            X_reduced, y, lapack_driver="gelsd", check_finite=False, cond=1e-07
        )[0]

        # Expand coefficients to full size with NaN for dropped columns
        coefficients = _expand_coefficients_with_nan(coefficients_reduced, k, kept_cols)

        # Compute residuals using only the identified coefficients
        # Note: Dropped coefficients are NaN, so we use the reduced form
        fitted = np.dot(X_reduced, coefficients_reduced)
        residuals = y - fitted

        # Compute variance-covariance matrix for reduced system, then expand
        vcov = None
        if return_vcov:
            vcov_reduced = _compute_robust_vcov_numpy(
                X_reduced,
                residuals,
                cluster_ids,
            )
            vcov = _expand_vcov_with_nan(vcov_reduced, k, kept_cols)
    else:
        # Full-rank case: proceed normally
        # Use cond=1e-07 for consistency with Rust backend and QR rank tolerance
        coefficients = scipy_lstsq(X, y, lapack_driver="gelsd", check_finite=False, cond=1e-07)[0]

        # Compute residuals and fitted values
        fitted = np.dot(X, coefficients)
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
    weights: Optional[np.ndarray] = None,
    weight_type: str = "pweight",
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
    weights : ndarray of shape (n,), optional
        Observation weights. If provided, computes weighted sandwich estimator.
    weight_type : str, default "pweight"
        Weight type: "pweight", "fweight", or "aweight".

    Returns
    -------
    vcov : ndarray of shape (k, k)
        Variance-covariance matrix.

    Notes
    -----
    For HC1 (no clustering):
        pweight: meat = Σ s_i s_i' where s_i = w_i x_i u_i (w² in meat)
        fweight: meat = X' diag(w u²) X (matches frequency-expanded HC1)
        aweight/unweighted: meat = X' diag(u²) X
        adjustment = n / (n - k)  (fweight uses n_eff = sum(w))

    For cluster-robust:
        meat = sum_g (X_g' u_g)(X_g' u_g)'
        adjustment = (G / (G-1)) * ((n-1) / (n-k))

    The cluster-robust computation is vectorized using pandas groupby,
    which is much faster than a Python loop over clusters.
    """
    # Validate weights before dispatching to backend
    if weights is not None:
        weights = _validate_weights(weights, weight_type, X.shape[0])

    # Use Rust backend if available AND no weights (Rust doesn't support weights yet)
    if HAS_RUST_BACKEND and weights is None:
        X = np.ascontiguousarray(X, dtype=np.float64)
        residuals = np.ascontiguousarray(residuals, dtype=np.float64)

        cluster_ids_int = None
        if cluster_ids is not None:
            cluster_ids_int = pd.factorize(cluster_ids)[0].astype(np.int64)

        try:
            return _rust_compute_robust_vcov(X, residuals, cluster_ids_int)
        except ValueError as e:
            # Translate Rust errors to consistent Python error messages or fallback
            error_msg = str(e)
            if "Matrix inversion failed" in error_msg:
                raise ValueError(
                    "Design matrix is rank-deficient (singular X'X matrix). "
                    "This indicates perfect multicollinearity. Check your fixed effects "
                    "and covariates for linear dependencies."
                ) from e
            if "numerically unstable" in error_msg.lower():
                # Fall back to NumPy on numerical instability (with warning)
                warnings.warn(
                    f"Rust backend detected numerical instability: {e}. "
                    "Falling back to Python backend for variance computation.",
                    UserWarning,
                    stacklevel=2,
                )
                return _compute_robust_vcov_numpy(
                    X,
                    residuals,
                    cluster_ids,
                    weights=weights,
                    weight_type=weight_type,
                )
            raise

    # Fallback to NumPy implementation
    return _compute_robust_vcov_numpy(
        X,
        residuals,
        cluster_ids,
        weights=weights,
        weight_type=weight_type,
    )


def _compute_robust_vcov_numpy(
    X: np.ndarray,
    residuals: np.ndarray,
    cluster_ids: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    weight_type: str = "pweight",
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
    weights : np.ndarray, optional
        Observation weights. If provided, computes weighted sandwich estimator.
    weight_type : str, default "pweight"
        Weight type: "pweight", "fweight", or "aweight".

    Returns
    -------
    vcov : np.ndarray
        Variance-covariance matrix of shape (k, k).

    Notes
    -----
    Uses vectorized groupby aggregation for cluster-robust SEs to avoid
    the O(n * G) loop that would be required with explicit iteration.

    Weight type affects the meat computation:
    - pweight: scores = w_i * X_i * u_i (HC1 meat = Σ s_i s_i' = X'diag(w²u²)X)
    - fweight: scores = w_i * X_i * u_i (weighted scores), df = sum(w) - k
    - aweight: scores = X_i * u_i (no weight in meat; after WLS, errors ~homoskedastic)
    """
    n, k = X.shape

    # Bread: (X'WX) or (X'X) depending on whether weights present
    if weights is not None:
        XtWX = X.T @ (X * weights[:, np.newaxis])
        bread_matrix = XtWX
    else:
        bread_matrix = X.T @ X

    # Effective n for df computation
    # fweights: sum(w) (frequency expansion)
    # pweight/aweight with zeros: positive-weight count (zero-weight rows
    # contribute nothing to the sandwich and should not inflate df)
    n_eff = n
    if weights is not None:
        if weight_type == "fweight":
            n_eff = int(round(np.sum(weights)))
        elif np.any(weights == 0):
            n_eff = int(np.count_nonzero(weights > 0))

    # Compute weighted scores for cluster-robust meat (outer product of sums).
    # pweight/fweight multiply by w; aweight and unweighted use raw residuals.
    _use_weighted_scores = weights is not None and weight_type not in ("aweight",)
    if _use_weighted_scores:
        scores = X * (weights * residuals)[:, np.newaxis]
    else:
        scores = X * residuals[:, np.newaxis]
        # Zero out scores for zero-weight aweight rows (subpopulation invariance)
        if weights is not None and np.any(weights == 0):
            scores[weights == 0] = 0.0

    if cluster_ids is None:
        # HC1 (heteroskedasticity-robust) standard errors
        adjustment = n_eff / (n_eff - k)
        if weights is not None and weight_type == "fweight":
            # fweight: frequency-expanded HC1, meat = Σ w_i x_i x_i' u_i²
            meat = np.dot(X.T, X * (weights * residuals**2)[:, np.newaxis])
        else:
            # pweight: WLS score outer product, meat = Σ w_i² x_i x_i' u_i²
            # aweight/unweighted: meat = Σ x_i x_i' u_i² (scores have no w)
            meat = scores.T @ scores
    else:
        # Cluster-robust standard errors (vectorized via groupby)
        cluster_ids = np.asarray(cluster_ids)
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        # Exclude clusters with zero total weight (subpopulation-zeroed)
        if weights is not None and weight_type != "fweight" and np.any(weights == 0):
            cluster_weights = pd.Series(weights).groupby(cluster_ids).sum()
            n_clusters = int((cluster_weights > 0).sum())

        if n_clusters < 2:
            raise ValueError(f"Need at least 2 clusters for cluster-robust SEs, got {n_clusters}")

        # Small-sample adjustment
        adjustment = (n_clusters / (n_clusters - 1)) * ((n_eff - 1) / (n_eff - k))

        # Sum scores within each cluster using pandas groupby (vectorized)
        cluster_scores = pd.DataFrame(scores).groupby(cluster_ids).sum().values

        # Meat is the outer product sum: sum_g (score_g)(score_g)'
        meat = cluster_scores.T @ cluster_scores

    # Sandwich estimator: bread^{-1} meat bread^{-1}
    # Solve bread * temp = meat, then solve bread * vcov' = temp'
    try:
        temp = np.linalg.solve(bread_matrix, meat)
        vcov = adjustment * np.linalg.solve(bread_matrix, temp.T).T
    except np.linalg.LinAlgError as e:
        if "Singular" in str(e):
            raise ValueError(
                "Design matrix is rank-deficient (singular X'X matrix). "
                "This indicates perfect multicollinearity. Check your fixed effects "
                "and covariates for linear dependencies."
            ) from e
        raise

    return vcov


# Empirical threshold: coefficients above this magnitude suggest near-separation
# in the logistic model (predicted probabilities collapse to 0/1).
_LOGIT_SEPARATION_COEF_THRESHOLD = 10
_LOGIT_SEPARATION_PROB_THRESHOLD = 1e-5


def solve_logit(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 25,
    tol: float = 1e-8,
    check_separation: bool = True,
    rank_deficient_action: str = "warn",
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit logistic regression via IRLS (Fisher scoring).

    Matches R's ``glm(family=binomial)`` algorithm: iteratively reweighted
    least squares with working weights ``mu*(1-mu)`` and working response
    ``eta + (y-mu)/(mu*(1-mu))``.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features). Intercept added automatically.
    y : np.ndarray
        Binary outcome (0/1).
    max_iter : int, default 25
        Maximum IRLS iterations (R's ``glm`` default).
    tol : float, default 1e-8
        Convergence tolerance on coefficient change (R's ``glm`` default).
    check_separation : bool, default True
        Whether to check for near-separation and emit warnings.
    rank_deficient_action : str, default "warn"
        How to handle rank-deficient design matrices:
        - "warn": Emit warning and drop columns (default)
        - "error": Raise ValueError
        - "silent": Drop columns silently
    weights : np.ndarray, optional
        Survey/observation weights of shape (n_samples,). When provided,
        the IRLS working weights become ``weights * mu * (1 - mu)``
        instead of ``mu * (1 - mu)``. This produces the survey-weighted
        maximum likelihood estimator, matching R's ``svyglm(family=binomial)``.
        When None (default), behavior is identical to unweighted logistic
        regression.

    Returns
    -------
    beta : np.ndarray
        Fitted coefficients (including intercept as element 0).
    probs : np.ndarray
        Predicted probabilities.
    """
    n, p = X.shape
    X_with_intercept = np.column_stack([np.ones(n), X])
    k = p + 1  # number of parameters including intercept

    # Validate weights
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (n,):
            raise ValueError(f"weights must have shape ({n},), got {weights.shape}")
        if np.any(np.isnan(weights)):
            raise ValueError("weights contain NaN values")
        if np.any(~np.isfinite(weights)):
            raise ValueError("weights contain Inf values")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")
        if np.sum(weights) <= 0:
            raise ValueError(
                "weights sum to zero — no observations have positive weight"
            )

    # Validate rank_deficient_action
    valid_actions = {"warn", "error", "silent"}
    if rank_deficient_action not in valid_actions:
        raise ValueError(
            f"rank_deficient_action must be one of {valid_actions}, "
            f"got '{rank_deficient_action}'"
        )

    # Track original column count for coefficient expansion at the end
    k_original = X_with_intercept.shape[1]
    eff_dropped_original: list = []  # indices in original column space

    # Validate effective weighted sample when weights have zeros
    if weights is not None and np.any(weights == 0):
        pos_mask = weights > 0
        n_pos = int(np.sum(pos_mask))
        y_pos = y[pos_mask]
        # Need both outcome classes in the positive-weight subset
        unique_y = np.unique(y_pos)
        if len(unique_y) < 2:
            raise ValueError(
                f"Positive-weight observations have only {len(unique_y)} "
                f"outcome class(es). Logistic regression requires both 0 and 1 "
                f"in the effective (positive-weight) sample."
            )
        # Check rank deficiency on positive-weight rows FIRST — full design
        # may be full rank due to zero-weight padding. Drop columns before
        # checking sample-size identification.
        X_eff = X_with_intercept[pos_mask]
        eff_rank_info = _detect_rank_deficiency(X_eff)
        if len(eff_rank_info[1]) > 0:
            n_dropped_eff = len(eff_rank_info[1])
            if rank_deficient_action == "error":
                raise ValueError(
                    f"Effective (positive-weight) sample is rank-deficient: "
                    f"{n_dropped_eff} linearly dependent column(s). "
                    f"Cannot identify logistic model on this subpopulation."
                )
            elif rank_deficient_action == "warn":
                warnings.warn(
                    f"Effective (positive-weight) sample is rank-deficient: "
                    f"dropping {n_dropped_eff} column(s). Propensity estimates "
                    f"may be unreliable on this subpopulation.",
                    UserWarning,
                    stacklevel=2,
                )
            # Drop columns and track original indices for final expansion
            eff_dropped_original = list(eff_rank_info[1])
            X_with_intercept = np.delete(X_with_intercept, eff_rank_info[1], axis=1)
            k = X_with_intercept.shape[1]
        # Check sample-size identification AFTER column dropping
        if n_pos <= k:
            raise ValueError(
                f"Only {n_pos} positive-weight observation(s) for "
                f"{k} parameters (after rank reduction). "
                f"Cannot identify logistic model."
            )

    # Check rank deficiency once before iterating (on possibly-shrunk matrix)
    rank_info = _detect_rank_deficiency(X_with_intercept)
    rank, dropped_cols, _ = rank_info
    if len(dropped_cols) > 0:
        col_desc = _format_dropped_columns(dropped_cols)
        if rank_deficient_action == "error":
            raise ValueError(
                f"Rank-deficient design matrix in logistic regression: "
                f"dropping {col_desc}. Propensity score estimates may be unreliable."
            )
        elif rank_deficient_action == "warn":
            warnings.warn(
                f"Rank-deficient design matrix in logistic regression: "
                f"dropping {col_desc}. Propensity score estimates may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
        kept_cols = np.array([i for i in range(k) if i not in dropped_cols])
        X_solve = X_with_intercept[:, kept_cols]
    else:
        kept_cols = np.arange(k)
        X_solve = X_with_intercept

    # IRLS (Fisher scoring)
    beta_solve = np.zeros(X_solve.shape[1])
    converged = False

    for iteration in range(max_iter):
        eta = X_solve @ beta_solve
        # Clip to prevent overflow in exp
        eta = np.clip(eta, -500, 500)
        mu = 1.0 / (1.0 + np.exp(-eta))
        # Clip mu to prevent zero working weights
        mu = np.clip(mu, 1e-10, 1 - 1e-10)

        # Working weights and working response
        w_irls = mu * (1.0 - mu)
        z = eta + (y - mu) / w_irls

        if weights is not None:
            w_total = weights * w_irls
        else:
            w_total = w_irls

        # Weighted least squares: solve (X'WX) beta = X'Wz
        sqrt_w = np.sqrt(w_total)
        Xw = X_solve * sqrt_w[:, None]
        zw = z * sqrt_w
        beta_new, _, _, _ = np.linalg.lstsq(Xw, zw, rcond=None)

        # Check convergence
        if np.max(np.abs(beta_new - beta_solve)) < tol:
            beta_solve = beta_new
            converged = True
            break
        beta_solve = beta_new

    # Final predicted probabilities
    eta_final = X_solve @ beta_solve
    eta_final = np.clip(eta_final, -500, 500)
    probs = 1.0 / (1.0 + np.exp(-eta_final))

    # Warnings
    if not converged:
        warnings.warn(
            f"Logistic regression did not converge in {max_iter} iterations. "
            f"Propensity score estimates may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    if check_separation:
        if np.max(np.abs(beta_solve)) > _LOGIT_SEPARATION_COEF_THRESHOLD:
            warnings.warn(
                "Large coefficients detected in propensity score model "
                f"(max|beta| > {_LOGIT_SEPARATION_COEF_THRESHOLD}), "
                "suggesting potential separation.",
                UserWarning,
                stacklevel=2,
            )
        n_extreme = int(
            np.sum(
                (probs < _LOGIT_SEPARATION_PROB_THRESHOLD)
                | (probs > 1 - _LOGIT_SEPARATION_PROB_THRESHOLD)
            )
        )
        if n_extreme > 0:
            warnings.warn(
                f"Near-separation detected in propensity score model: "
                f"{n_extreme} of {n} observations have predicted probabilities "
                f"within {_LOGIT_SEPARATION_PROB_THRESHOLD} of 0 or 1. ATT estimates may be sensitive to "
                f"model specification.",
                UserWarning,
                stacklevel=2,
            )

    # Expand beta back to original column count, accounting for columns
    # dropped in both the effective-sample check and the full-sample check
    if len(dropped_cols) > 0 or len(eff_dropped_original) > 0:
        # First expand from X_solve columns back to post-eff-drop columns
        # Use NaN for dropped coefficients (R convention: not estimable)
        beta_post_eff = np.full(k, np.nan)
        beta_post_eff[kept_cols] = beta_solve

        # Then expand from post-eff-drop columns back to original columns
        if len(eff_dropped_original) > 0:
            beta_full = np.full(k_original, np.nan)
            kept_original = [i for i in range(k_original) if i not in eff_dropped_original]
            beta_full[kept_original] = beta_post_eff
        else:
            beta_full = beta_post_eff
    else:
        beta_full = beta_solve

    return beta_full, probs


def _check_propensity_diagnostics(
    pscore: np.ndarray,
    trim_bound: float = 0.01,
) -> None:
    """
    Warn if propensity scores are extreme.

    Parameters
    ----------
    pscore : np.ndarray
        Predicted probabilities.
    trim_bound : float, default 0.01
        Trimming threshold.
    """
    n_extreme = int(np.sum((pscore < trim_bound) | (pscore > 1 - trim_bound)))
    if n_extreme > 0:
        n_total = len(pscore)
        pct = 100.0 * n_extreme / n_total
        warnings.warn(
            f"Propensity scores for {n_extreme} of {n_total} observations "
            f"({pct:.1f}%) were outside [{trim_bound}, {1 - trim_bound}] "
            f"and will be trimmed. This may indicate near-separation in "
            f"the propensity score model.",
            UserWarning,
            stacklevel=2,
        )


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
        """Check if the coefficient is statistically significant.

        Returns False for NaN p-values (unidentified coefficients).
        """
        if np.isnan(self.p_value):
            return False
        threshold = alpha if alpha is not None else self.alpha
        return self.p_value < threshold

    def significance_stars(self) -> str:
        """Return significance stars based on p-value.

        Returns empty string for NaN p-values (unidentified coefficients).
        """
        if np.isnan(self.p_value):
            return ""
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
    rank_deficient_action : str, default "warn"
        Action when design matrix is rank-deficient (linearly dependent columns):
        - "warn": Issue warning and drop linearly dependent columns (default)
        - "error": Raise ValueError
        - "silent": Drop columns silently without warning
    weights : array-like, optional
        Observation weights. When survey_design is provided, weights are
        automatically derived from it (explicit weights are overridden).
    weight_type : str, default "pweight"
        Weight type: "pweight", "fweight", or "aweight".
    survey_design : ResolvedSurveyDesign, optional
        Resolved survey design for Taylor Series Linearization variance
        estimation. When provided, weights and weight_type are canonicalized
        from this object.

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
    n_params_effective_ : int
        Effective number of parameters after dropping linearly dependent columns.
        Equals n_params_ for full-rank matrices (available after fit).
    df_ : int
        Degrees of freedom (n - n_params_effective) (available after fit).

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
        rank_deficient_action: str = "warn",
        weights: Optional[np.ndarray] = None,
        weight_type: str = "pweight",
        survey_design: object = None,
    ):
        self.include_intercept = include_intercept
        self.robust = robust
        self.cluster_ids = cluster_ids
        self.alpha = alpha
        self.rank_deficient_action = rank_deficient_action
        self.weights = weights
        self.weight_type = weight_type
        self.survey_design = survey_design  # ResolvedSurveyDesign or None

        # Fitted attributes (set by fit())
        self.coefficients_: Optional[np.ndarray] = None
        self.vcov_: Optional[np.ndarray] = None
        self.residuals_: Optional[np.ndarray] = None
        self.fitted_values_: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._X: Optional[np.ndarray] = None
        self.n_obs_: Optional[int] = None
        self.n_params_: Optional[int] = None
        self.n_params_effective_: Optional[int] = None
        self.df_: Optional[int] = None
        self.survey_df_: Optional[int] = None

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

        # Reset replicate df from any previous fit
        self._replicate_df = None

        # Add intercept if requested
        if self.include_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])

        # Use provided cluster_ids or fall back to instance-level
        effective_cluster_ids = cluster_ids if cluster_ids is not None else self.cluster_ids

        # Determine if survey vcov should be used
        _use_survey_vcov = False
        if self.survey_design is not None:
            from diff_diff.survey import ResolvedSurveyDesign

            if isinstance(self.survey_design, ResolvedSurveyDesign):
                _use_survey_vcov = self.survey_design.needs_survey_vcov
                # Canonicalize weights from survey_design to ensure consistency
                # between coefficient estimation and survey vcov computation
                if self.weights is not None and self.weights is not self.survey_design.weights:
                    warnings.warn(
                        "Explicit weights= differ from survey_design.weights. "
                        "Using survey_design weights for both coefficient "
                        "estimation and variance computation to ensure "
                        "consistency.",
                        UserWarning,
                        stacklevel=2,
                    )
                self.weights = self.survey_design.weights
                self.weight_type = self.survey_design.weight_type

        if self.weights is not None:
            self.weights = _validate_weights(self.weights, self.weight_type, X.shape[0])

        # Inject cluster as PSU for survey variance when no PSU specified.
        # Use a local variable to avoid mutating self.survey_design, which
        # would cause stale PSU on repeated fit() calls with different clusters.
        _effective_survey_design = self.survey_design
        if (
            effective_cluster_ids is not None
            and _effective_survey_design is not None
            and _use_survey_vcov
        ):
            from diff_diff.survey import ResolvedSurveyDesign as _RSD
            from diff_diff.survey import _inject_cluster_as_psu

            if isinstance(_effective_survey_design, _RSD) and _effective_survey_design.psu is None:
                _effective_survey_design = _inject_cluster_as_psu(
                    _effective_survey_design, effective_cluster_ids
                )

        if self.robust or effective_cluster_ids is not None:
            # Use solve_ols with robust/cluster SEs
            # When survey vcov will be used, skip standard vcov computation
            coefficients, residuals, fitted, vcov = solve_ols(
                X,
                y,
                cluster_ids=effective_cluster_ids,
                return_fitted=True,
                return_vcov=not _use_survey_vcov,
                rank_deficient_action=self.rank_deficient_action,
                weights=self.weights,
                weight_type=self.weight_type,
            )
        else:
            # Classical OLS - compute vcov separately
            coefficients, residuals, fitted, _ = solve_ols(
                X,
                y,
                return_fitted=True,
                return_vcov=False,
                rank_deficient_action=self.rank_deficient_action,
                weights=self.weights,
                weight_type=self.weight_type,
            )
            # Compute classical OLS variance-covariance matrix
            # Handle rank-deficient case: use effective rank for df
            n, k = X.shape
            nan_mask = np.isnan(coefficients)
            k_effective = k - np.sum(nan_mask)  # Number of identified coefficients

            # Effective n for df: fweights use sum(w), pweight/aweight with
            # zeros use positive-weight count (zero-weight rows don't contribute)
            n_eff_df = n
            if self.weights is not None:
                if self.weight_type == "fweight":
                    n_eff_df = int(round(np.sum(self.weights)))
                elif np.any(self.weights == 0):
                    n_eff_df = int(np.count_nonzero(self.weights > 0))

            if k_effective == 0:
                # All coefficients dropped - no valid inference
                vcov = np.full((k, k), np.nan)
            elif np.any(nan_mask):
                # Rank-deficient: compute vcov for identified coefficients only
                kept_cols = np.where(~nan_mask)[0]
                X_reduced = X[:, kept_cols]
                if self.weights is not None:
                    # Weighted classical vcov: use weighted RSS and X'WX
                    w = self.weights
                    mse = np.sum(w * residuals**2) / (n_eff_df - k_effective)
                    XtWX_reduced = X_reduced.T @ (X_reduced * w[:, np.newaxis])
                    try:
                        vcov_reduced = np.linalg.solve(XtWX_reduced, mse * np.eye(k_effective))
                    except np.linalg.LinAlgError:
                        vcov_reduced = np.linalg.pinv(XtWX_reduced) * mse
                else:
                    mse = np.sum(residuals**2) / (n_eff_df - k_effective)
                    try:
                        vcov_reduced = np.linalg.solve(
                            X_reduced.T @ X_reduced, mse * np.eye(k_effective)
                        )
                    except np.linalg.LinAlgError:
                        vcov_reduced = np.linalg.pinv(X_reduced.T @ X_reduced) * mse
                # Expand to full size with NaN for dropped columns
                vcov = _expand_vcov_with_nan(vcov_reduced, k, kept_cols)
            else:
                # Full rank: standard computation
                if self.weights is not None:
                    # Weighted classical vcov: use weighted RSS and X'WX
                    w = self.weights
                    mse = np.sum(w * residuals**2) / (n_eff_df - k)
                    XtWX = X.T @ (X * w[:, np.newaxis])
                    try:
                        vcov = np.linalg.solve(XtWX, mse * np.eye(k))
                    except np.linalg.LinAlgError:
                        vcov = np.linalg.pinv(XtWX) * mse
                else:
                    mse = np.sum(residuals**2) / (n_eff_df - k)
                    try:
                        vcov = np.linalg.solve(X.T @ X, mse * np.eye(k))
                    except np.linalg.LinAlgError:
                        vcov = np.linalg.pinv(X.T @ X) * mse

        # Compute survey vcov if applicable
        if _use_survey_vcov:
            from diff_diff.survey import ResolvedSurveyDesign as _RSD

            _uses_rep = (
                isinstance(_effective_survey_design, _RSD)
                and _effective_survey_design.uses_replicate_variance
            )

            if _uses_rep:
                from diff_diff.survey import compute_replicate_vcov

                nan_mask = np.isnan(coefficients)
                if np.any(nan_mask):
                    kept_cols = np.where(~nan_mask)[0]
                    if len(kept_cols) > 0:
                        vcov_reduced, _n_valid_rep = compute_replicate_vcov(
                            X[:, kept_cols], y, coefficients[kept_cols],
                            _effective_survey_design,
                            weight_type=self.weight_type,
                        )
                        vcov = _expand_vcov_with_nan(vcov_reduced, X.shape[1], kept_cols)
                    else:
                        vcov = np.full((X.shape[1], X.shape[1]), np.nan)
                        _n_valid_rep = 0
                else:
                    vcov, _n_valid_rep = compute_replicate_vcov(
                        X, y, coefficients, _effective_survey_design,
                        weight_type=self.weight_type,
                    )
                # Store effective replicate df only when replicates were dropped
                if _n_valid_rep < _effective_survey_design.n_replicates:
                    self._replicate_df = _n_valid_rep - 1 if _n_valid_rep > 1 else None
                else:
                    self._replicate_df = None  # use rank-based df from design
            else:
                from diff_diff.survey import compute_survey_vcov

                nan_mask = np.isnan(coefficients)
                if np.any(nan_mask):
                    kept_cols = np.where(~nan_mask)[0]
                    if len(kept_cols) > 0:
                        vcov_reduced = compute_survey_vcov(
                            X[:, kept_cols], residuals, _effective_survey_design
                        )
                        vcov = _expand_vcov_with_nan(vcov_reduced, X.shape[1], kept_cols)
                    else:
                        vcov = np.full((X.shape[1], X.shape[1]), np.nan)
                else:
                    vcov = compute_survey_vcov(X, residuals, _effective_survey_design)

        # Store fitted attributes
        self.coefficients_ = coefficients
        self.vcov_ = vcov
        self.residuals_ = residuals
        self.fitted_values_ = fitted
        self._y = y
        self._X = X
        self.n_obs_ = X.shape[0]
        self.n_params_ = X.shape[1]

        # Compute effective number of parameters (excluding dropped columns)
        # This is needed for correct degrees of freedom in inference
        nan_mask = np.isnan(coefficients)
        self.n_params_effective_ = int(self.n_params_ - np.sum(nan_mask))
        # Effective n for df: fweights use sum(w), pweight/aweight with
        # zeros use positive-weight count (matches compute_robust_vcov)
        n_eff_df = self.n_obs_
        if self.weights is not None:
            if self.weight_type == "fweight":
                n_eff_df = int(round(np.sum(self.weights)))
            elif np.any(self.weights == 0):
                n_eff_df = int(np.count_nonzero(self.weights > 0))
        self.df_ = n_eff_df - self.n_params_effective_ - df_adjustment

        # Survey degrees of freedom: n_PSU - n_strata (overrides standard df)
        self.survey_df_ = None
        if _effective_survey_design is not None:
            from diff_diff.survey import ResolvedSurveyDesign

            if isinstance(_effective_survey_design, ResolvedSurveyDesign):
                self.survey_df_ = _effective_survey_design.df_survey
                # Override with effective replicate df if available
                if hasattr(self, '_replicate_df') and self._replicate_df is not None:
                    self.survey_df_ = self._replicate_df

        return self

    def compute_deff(self, coefficient_names=None):
        """Compute per-coefficient design effect diagnostics.

        Compares the survey vcov to an SRS (HC1) baseline.  Must be called
        after ``fit()`` with a survey design.

        Returns
        -------
        DEFFDiagnostics
        """
        self._check_fitted()
        if not (hasattr(self, 'survey_design') and self.survey_design is not None):
            raise ValueError(
                "compute_deff() requires a survey design. "
                "Fit with survey_design= first."
            )
        from diff_diff.survey import compute_deff_diagnostics

        # Handle rank-deficient fits: compute DEFF only on kept columns,
        # then expand back with NaN for dropped columns
        nan_mask = np.isnan(self.coefficients_)
        if np.any(nan_mask):
            kept = np.where(~nan_mask)[0]
            if len(kept) == 0:
                k = len(self.coefficients_)
                nan_arr = np.full(k, np.nan)
                from diff_diff.survey import DEFFDiagnostics
                return DEFFDiagnostics(
                    deff=nan_arr, effective_n=nan_arr.copy(),
                    srs_se=nan_arr.copy(), survey_se=nan_arr.copy(),
                    coefficient_names=coefficient_names,
                )
            # Compute on kept columns only
            X_kept = self._X[:, kept]
            vcov_kept = self.vcov_[np.ix_(kept, kept)]
            deff_kept = compute_deff_diagnostics(
                X_kept, self.residuals_, vcov_kept,
                self.weights, weight_type=self.weight_type,
            )
            # Expand back to full size with NaN for dropped
            k = len(self.coefficients_)
            full_deff = np.full(k, np.nan)
            full_eff_n = np.full(k, np.nan)
            full_srs_se = np.full(k, np.nan)
            full_survey_se = np.full(k, np.nan)
            full_deff[kept] = deff_kept.deff
            full_eff_n[kept] = deff_kept.effective_n
            full_srs_se[kept] = deff_kept.srs_se
            full_survey_se[kept] = deff_kept.survey_se
            from diff_diff.survey import DEFFDiagnostics
            return DEFFDiagnostics(
                deff=full_deff, effective_n=full_eff_n,
                srs_se=full_srs_se, survey_se=full_survey_se,
                coefficient_names=coefficient_names,
            )

        return compute_deff_diagnostics(
            self._X, self.residuals_, self.vcov_,
            self.weights, weight_type=self.weight_type,
            coefficient_names=coefficient_names,
        )

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
        assert self.coefficients_ is not None
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
        assert self.vcov_ is not None
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
        assert self.coefficients_ is not None
        assert self.vcov_ is not None

        coef = float(self.coefficients_[index])
        se = float(np.sqrt(self.vcov_[index, index]))

        # Use instance alpha if not provided
        effective_alpha = alpha if alpha is not None else self.alpha

        # Use survey df if available, otherwise fitted df
        # Note: df=None means use normal distribution
        if df is not None:
            effective_df = df
        elif self.survey_df_ is not None:
            effective_df = self.survey_df_
        elif (hasattr(self, 'survey_design') and self.survey_design is not None
              and hasattr(self.survey_design, 'uses_replicate_variance')
              and self.survey_design.uses_replicate_variance):
            # Replicate design with undefined df (rank <= 1) — NaN inference
            warnings.warn(
                "Replicate design has undefined survey d.f. (rank <= 1). "
                "Inference fields will be NaN.",
                UserWarning, stacklevel=2,
            )
            effective_df = 0  # Forces NaN from t-distribution
        else:
            effective_df = self.df_

        # Warn if df is non-positive and fall back to normal distribution
        # (skip for replicate designs — df=0 is intentional for NaN inference)
        _is_replicate = (hasattr(self, 'survey_design') and self.survey_design is not None
                         and hasattr(self.survey_design, 'uses_replicate_variance')
                         and self.survey_design.uses_replicate_variance)
        if effective_df is not None and effective_df <= 0 and not _is_replicate:
            import warnings

            warnings.warn(
                f"Degrees of freedom is non-positive (df={effective_df}). "
                "Using normal distribution instead of t-distribution for inference.",
                UserWarning,
            )
            effective_df = None

        # Use project-standard NaN-safe inference (returns all-NaN when SE <= 0)
        from diff_diff.utils import safe_inference

        t_stat, p_value, conf_int = safe_inference(coef, se, alpha=effective_alpha, df=effective_df)

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
        return [self.get_inference(i, alpha=alpha, df=df) for i in range(len(self.coefficients_))]

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

        Notes
        -----
        For rank-deficient fits, adjusted R² uses the effective number of
        parameters (excluding dropped columns) for consistency with the
        corrected degrees of freedom.
        """
        self._check_fitted()
        assert self._y is not None
        assert self.residuals_ is not None
        # Use effective params for adjusted R² to match df correction
        n_params = self.n_params_effective_ if adjusted else self.n_params_
        return compute_r_squared(self._y, self.residuals_, adjusted=adjusted, n_params=n_params)

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

        Notes
        -----
        For rank-deficient fits where some coefficients are NaN, predictions
        use only the identified (non-NaN) coefficients. This is equivalent to
        treating dropped columns as having zero coefficients.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)

        if self.include_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])

        # Handle rank-deficient case: use only identified coefficients
        # Replace NaN with 0 so they don't contribute to prediction
        assert self.coefficients_ is not None
        coef = self.coefficients_.copy()
        coef[np.isnan(coef)] = 0.0

        return np.dot(X, coef)


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
