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
from typing import Dict, List, Optional, Tuple, Union

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
    Q, R, pivot = qr(X, mode='economic', pivoting=True)

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
        names = [column_names[i] if i < len(column_names) else f"column {i}"
                 for i in dropped_cols]
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
) -> Union[
    Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]],
    Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]],
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
    """
    # Convert cluster_ids to int64 for Rust (handles string/categorical IDs)
    if cluster_ids is not None:
        cluster_ids = _factorize_cluster_ids(cluster_ids)

    # Call Rust backend
    coefficients, residuals, vcov = _rust_solve_ols(
        X, y, cluster_ids=cluster_ids, return_vcov=return_vcov
    )

    # Convert to numpy arrays
    coefficients = np.asarray(coefficients)
    residuals = np.asarray(residuals)
    if vcov is not None:
        vcov = np.asarray(vcov)

    # Return with optional fitted values
    if return_fitted:
        fitted = X @ coefficients
        return coefficients, residuals, fitted, vcov
    else:
        return coefficients, residuals, vcov


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
            f"X and y must have same number of observations: "
            f"{X.shape[0]} vs {y.shape[0]}"
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

    # Fast path: skip rank check and use Rust directly when requested
    # This saves O(nk²) QR overhead but won't detect rank-deficient matrices
    if skip_rank_check:
        if HAS_RUST_BACKEND and _rust_solve_ols is not None:
            return _solve_ols_rust(
                X, y,
                cluster_ids=cluster_ids,
                return_vcov=return_vcov,
                return_fitted=return_fitted,
            )
        # Fall through to Python without rank check (user guarantees full rank)
        return _solve_ols_numpy(
            X, y,
            cluster_ids=cluster_ids,
            return_vcov=return_vcov,
            return_fitted=return_fitted,
            rank_deficient_action=rank_deficient_action,
            column_names=column_names,
            _skip_rank_check=True,
        )

    # Check for rank deficiency using fast pivoted QR decomposition.
    # This adds O(nk²) overhead but is necessary for:
    # 1. Detecting which columns to drop (R-style NA handling)
    # 2. Routing rank-deficient cases to Python (Rust doesn't support pivoted QR)
    #
    # Trade-off: ~2x compute cost for full-rank matrices in exchange for proper
    # rank deficiency handling. For maximum performance on known full-rank data,
    # set skip_rank_check=True.
    rank, dropped_cols, pivot = _detect_rank_deficiency(X)
    is_rank_deficient = len(dropped_cols) > 0

    # Routing strategy:
    # - Full-rank + Rust available → fast Rust backend (SVD-based solve)
    # - Rank-deficient → Python backend (proper NA handling, valid SEs)
    # - No Rust → Python backend (works for all cases)
    if HAS_RUST_BACKEND and _rust_solve_ols is not None and not is_rank_deficient:
        result = _solve_ols_rust(
            X, y,
            cluster_ids=cluster_ids,
            return_vcov=return_vcov,
            return_fitted=return_fitted,
        )

        # Check for NaN vcov: Rust SVD may detect rank-deficiency that QR missed
        # for ill-conditioned matrices (QR and SVD have different numerical properties).
        # When this happens, fall back to Python's R-style handling.
        vcov = result[-1]  # vcov is always the last element
        if return_vcov and vcov is not None and np.any(np.isnan(vcov)):
            warnings.warn(
                "Rust backend detected ill-conditioned matrix (NaN in variance-covariance). "
                "Re-running with Python backend for proper rank detection.",
                UserWarning,
                stacklevel=2,
            )
            # Force fresh rank detection - don't pass cached info since QR
            # and SVD disagreed about rank. Python's QR will re-detect and
            # apply R-style NaN handling for dropped columns.
            return _solve_ols_numpy(
                X, y,
                cluster_ids=cluster_ids,
                return_vcov=return_vcov,
                return_fitted=return_fitted,
                rank_deficient_action=rank_deficient_action,
                column_names=column_names,
                _precomputed_rank_info=None,  # Force re-detection
            )
        else:
            return result

    # Use NumPy implementation for rank-deficient cases (R-style NA handling)
    # or when Rust backend is not available
    return _solve_ols_numpy(
        X, y,
        cluster_ids=cluster_ids,
        return_vcov=return_vcov,
        return_fitted=return_fitted,
        rank_deficient_action=rank_deficient_action,
        column_names=column_names,
        # Pass pre-computed rank info to avoid redundant computation
        _precomputed_rank_info=(rank, dropped_cols, pivot),
    )


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
        fitted = X_reduced @ coefficients_reduced
        residuals = y - fitted

        # Compute variance-covariance matrix for reduced system, then expand
        vcov = None
        if return_vcov:
            vcov_reduced = _compute_robust_vcov_numpy(X_reduced, residuals, cluster_ids)
            vcov = _expand_vcov_with_nan(vcov_reduced, k, kept_cols)
    else:
        # Full-rank case: proceed normally
        # Use cond=1e-07 for consistency with Rust backend and QR rank tolerance
        coefficients = scipy_lstsq(X, y, lapack_driver="gelsd", check_finite=False, cond=1e-07)[0]

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
    ):
        self.include_intercept = include_intercept
        self.robust = robust
        self.cluster_ids = cluster_ids
        self.alpha = alpha
        self.rank_deficient_action = rank_deficient_action

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
                rank_deficient_action=self.rank_deficient_action,
            )
        else:
            # Classical OLS - compute vcov separately
            coefficients, residuals, fitted, _ = solve_ols(
                X, y,
                return_fitted=True,
                return_vcov=False,
                rank_deficient_action=self.rank_deficient_action,
            )
            # Compute classical OLS variance-covariance matrix
            # Handle rank-deficient case: use effective rank for df
            n, k = X.shape
            nan_mask = np.isnan(coefficients)
            k_effective = k - np.sum(nan_mask)  # Number of identified coefficients

            if k_effective == 0:
                # All coefficients dropped - no valid inference
                vcov = np.full((k, k), np.nan)
            elif np.any(nan_mask):
                # Rank-deficient: compute vcov for identified coefficients only
                kept_cols = np.where(~nan_mask)[0]
                X_reduced = X[:, kept_cols]
                mse = np.sum(residuals**2) / (n - k_effective)
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
                mse = np.sum(residuals**2) / (n - k)
                try:
                    vcov = np.linalg.solve(X.T @ X, mse * np.eye(k))
                except np.linalg.LinAlgError:
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

        # Compute effective number of parameters (excluding dropped columns)
        # This is needed for correct degrees of freedom in inference
        nan_mask = np.isnan(coefficients)
        self.n_params_effective_ = int(self.n_params_ - np.sum(nan_mask))
        self.df_ = self.n_obs_ - self.n_params_effective_ - df_adjustment

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

        # Handle zero or negative SE (indicates perfect fit or numerical issues)
        if se <= 0:
            import warnings
            warnings.warn(
                f"Standard error is zero or negative (se={se}) for coefficient at index {index}. "
                "This may indicate perfect multicollinearity or numerical issues.",
                UserWarning,
            )
            # Use inf for t-stat when SE is zero (perfect fit scenario)
            if coef > 0:
                t_stat = np.inf
            elif coef < 0:
                t_stat = -np.inf
            else:
                t_stat = 0.0
        else:
            t_stat = coef / se

        # Use instance alpha if not provided
        effective_alpha = alpha if alpha is not None else self.alpha

        # Use fitted df if not explicitly provided
        # Note: df=None means use normal distribution
        effective_df = df if df is not None else self.df_

        # Warn if df is non-positive and fall back to normal distribution
        if effective_df is not None and effective_df <= 0:
            import warnings
            warnings.warn(
                f"Degrees of freedom is non-positive (df={effective_df}). "
                "Using normal distribution instead of t-distribution for inference.",
                UserWarning,
            )
            effective_df = None

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

        Notes
        -----
        For rank-deficient fits, adjusted R² uses the effective number of
        parameters (excluding dropped columns) for consistency with the
        corrected degrees of freedom.
        """
        self._check_fitted()
        # Use effective params for adjusted R² to match df correction
        n_params = self.n_params_effective_ if adjusted else self.n_params_
        return compute_r_squared(
            self._y, self.residuals_, adjusted=adjusted, n_params=n_params
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
        coef = self.coefficients_.copy()
        coef[np.isnan(coef)] = 0.0

        return X @ coef


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
