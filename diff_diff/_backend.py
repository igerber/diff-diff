"""
Backend detection and configuration for diff-diff.

This module handles:
1. Detection of optional Rust backend
2. Environment variable configuration (DIFF_DIFF_BACKEND)
3. Exports HAS_RUST_BACKEND and Rust function references

Other modules should import from here to avoid circular imports with __init__.py.
"""

import os

# Check for backend override via environment variable
# DIFF_DIFF_BACKEND can be: 'auto' (default), 'python', or 'rust'
_backend_env = os.environ.get("DIFF_DIFF_BACKEND", "auto").lower()

# Try to import Rust backend for accelerated operations
try:
    from diff_diff._rust_backend import (
        generate_bootstrap_weights_batch as _rust_bootstrap_weights,
        project_simplex as _rust_project_simplex,
        solve_ols as _rust_solve_ols,
        compute_robust_vcov as _rust_compute_robust_vcov,
        # TROP estimator acceleration (local method)
        compute_unit_distance_matrix as _rust_unit_distance_matrix,
        loocv_grid_search as _rust_loocv_grid_search,
        bootstrap_trop_variance as _rust_bootstrap_trop_variance,
        # TROP estimator acceleration (global method)
        loocv_grid_search_global as _rust_loocv_grid_search_global,
        bootstrap_trop_variance_global as _rust_bootstrap_trop_variance_global,
        # SDID weights (Frank-Wolfe matching R's synthdid)
        compute_sdid_unit_weights as _rust_sdid_unit_weights,
        compute_time_weights as _rust_compute_time_weights,
        compute_noise_level as _rust_compute_noise_level,
        sc_weight_fw as _rust_sc_weight_fw,
        sc_weight_fw_with_convergence as _rust_sc_weight_fw_with_convergence,
        sc_weight_fw_weighted as _rust_sc_weight_fw_weighted,
        sc_weight_fw_weighted_with_convergence as _rust_sc_weight_fw_weighted_with_convergence,
        # Diagnostics
        rust_backend_info as _rust_backend_info,
    )

    _rust_available = True
except ImportError:
    _rust_available = False
    _rust_bootstrap_weights = None
    _rust_project_simplex = None
    _rust_solve_ols = None
    _rust_compute_robust_vcov = None
    # TROP estimator acceleration (local method)
    _rust_unit_distance_matrix = None
    _rust_loocv_grid_search = None
    _rust_bootstrap_trop_variance = None
    # TROP estimator acceleration (global method)
    _rust_loocv_grid_search_global = None
    _rust_bootstrap_trop_variance_global = None
    # SDID weights (Frank-Wolfe matching R's synthdid)
    _rust_sdid_unit_weights = None
    _rust_compute_time_weights = None
    _rust_compute_noise_level = None
    _rust_sc_weight_fw = None
    _rust_sc_weight_fw_with_convergence = None
    _rust_sc_weight_fw_weighted = None
    _rust_sc_weight_fw_weighted_with_convergence = None
    _rust_backend_info = None

# FE-absorption MAP demeaning kernel: imported independently so a stale or
# mixed-version extension missing only this newer symbol degrades to the
# numpy demeaning engine WITHOUT disabling the older Rust accelerations.
try:
    from diff_diff._rust_backend import demean_map as _rust_demean_map
except ImportError:
    _rust_demean_map = None

# Batched ridge-regularized SPD solve (EfficientDiD per-unit weights):
# imported independently for the same mixed-version reason as demean_map.
try:
    from diff_diff._rust_backend import (
        batched_ridge_chol_solve_ones as _rust_batched_ridge_chol_solve,
    )
except ImportError:
    _rust_batched_ridge_chol_solve = None

# HC2 (leverage-corrected) robust vcov: imported independently for the same
# mixed-version reason as demean_map (a stale extension missing only this
# newer symbol degrades HC2 to the NumPy path without disabling the older
# Rust accelerations).
try:
    from diff_diff._rust_backend import (
        compute_robust_vcov_hc2 as _rust_compute_robust_vcov_hc2,
    )
except ImportError:
    _rust_compute_robust_vcov_hc2 = None

# Opt-in normal-equations Cholesky OLS fast path: imported independently
# for the same mixed-version reason as demean_map. A stale extension
# missing only this symbol keeps every older Rust acceleration: Rust-eligible
# fits fall back to the legacy SVD solve_ols kernel (the knob simply has no
# Rust acceleration there), while numpy-lane fits (weighted, non-hc1,
# forced-python) still use the numpy Cholesky twin.
try:
    from diff_diff._rust_backend import solve_ols_chol as _rust_solve_ols_chol
except ImportError:
    _rust_solve_ols_chol = None

# Determine final backend based on environment variable and availability
if _backend_env == "python":
    # Force pure Python mode - disable Rust even if available
    HAS_RUST_BACKEND = False
    _rust_bootstrap_weights = None
    _rust_project_simplex = None
    _rust_solve_ols = None
    _rust_compute_robust_vcov = None
    # FE-absorption MAP demeaning kernel
    _rust_demean_map = None
    # Batched ridge-regularized SPD solve
    _rust_batched_ridge_chol_solve = None
    # HC2 robust vcov
    _rust_compute_robust_vcov_hc2 = None
    # Opt-in normal-equations Cholesky OLS fast path
    _rust_solve_ols_chol = None
    # TROP estimator acceleration (local method)
    _rust_unit_distance_matrix = None
    _rust_loocv_grid_search = None
    _rust_bootstrap_trop_variance = None
    # TROP estimator acceleration (global method)
    _rust_loocv_grid_search_global = None
    _rust_bootstrap_trop_variance_global = None
    # SDID weights (Frank-Wolfe matching R's synthdid)
    _rust_sdid_unit_weights = None
    _rust_compute_time_weights = None
    _rust_compute_noise_level = None
    _rust_sc_weight_fw = None
    _rust_sc_weight_fw_with_convergence = None
    _rust_sc_weight_fw_weighted = None
    _rust_sc_weight_fw_weighted_with_convergence = None
    _rust_backend_info = None
elif _backend_env == "rust":
    # Force Rust mode - fail if not available
    if not _rust_available:
        raise ImportError(
            "DIFF_DIFF_BACKEND=rust but Rust backend is not available. "
            "Install with: pip install diff-diff[rust]"
        )
    HAS_RUST_BACKEND = True
else:
    # Auto mode - use Rust if available
    HAS_RUST_BACKEND = _rust_available


def rust_backend_info():
    """Return compile-time BLAS feature information for the Rust backend.

    Returns a dict with keys:
    - 'blas': True if any BLAS backend is linked
    - 'accelerate': True if Apple Accelerate is linked (macOS)
    - 'openblas': True if OpenBLAS is linked (Linux)

    If the Rust backend is not available, all values are False.
    """
    if _rust_backend_info is not None:
        return _rust_backend_info()
    return {"blas": False, "accelerate": False, "openblas": False}


__all__ = [
    "HAS_RUST_BACKEND",
    "rust_backend_info",
    "_rust_bootstrap_weights",
    "_rust_project_simplex",
    "_rust_solve_ols",
    "_rust_compute_robust_vcov",
    # Opt-in normal-equations Cholesky OLS fast path
    "_rust_solve_ols_chol",
    # FE-absorption MAP demeaning kernel
    "_rust_demean_map",
    # Batched ridge-regularized SPD solve (EfficientDiD per-unit weights)
    "_rust_batched_ridge_chol_solve",
    # TROP estimator acceleration (local method)
    "_rust_unit_distance_matrix",
    "_rust_loocv_grid_search",
    "_rust_bootstrap_trop_variance",
    # TROP estimator acceleration (global method)
    "_rust_loocv_grid_search_global",
    "_rust_bootstrap_trop_variance_global",
    # SDID weights (Frank-Wolfe matching R's synthdid)
    "_rust_sdid_unit_weights",
    "_rust_compute_time_weights",
    "_rust_compute_noise_level",
    "_rust_sc_weight_fw",
    "_rust_sc_weight_fw_with_convergence",
    "_rust_sc_weight_fw_weighted",
    "_rust_sc_weight_fw_weighted_with_convergence",
]
