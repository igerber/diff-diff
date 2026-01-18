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
_backend_env = os.environ.get('DIFF_DIFF_BACKEND', 'auto').lower()

# Try to import Rust backend for accelerated operations
try:
    from diff_diff._rust_backend import (
        generate_bootstrap_weights_batch as _rust_bootstrap_weights,
        compute_synthetic_weights as _rust_synthetic_weights,
        project_simplex as _rust_project_simplex,
        solve_ols as _rust_solve_ols,
        compute_robust_vcov as _rust_compute_robust_vcov,
        # TROP estimator acceleration
        compute_unit_distance_matrix as _rust_unit_distance_matrix,
        loocv_grid_search as _rust_loocv_grid_search,
        bootstrap_trop_variance as _rust_bootstrap_trop_variance,
    )
    _rust_available = True
except ImportError:
    _rust_available = False
    _rust_bootstrap_weights = None
    _rust_synthetic_weights = None
    _rust_project_simplex = None
    _rust_solve_ols = None
    _rust_compute_robust_vcov = None
    # TROP estimator acceleration
    _rust_unit_distance_matrix = None
    _rust_loocv_grid_search = None
    _rust_bootstrap_trop_variance = None

# Determine final backend based on environment variable and availability
if _backend_env == 'python':
    # Force pure Python mode - disable Rust even if available
    HAS_RUST_BACKEND = False
    _rust_bootstrap_weights = None
    _rust_synthetic_weights = None
    _rust_project_simplex = None
    _rust_solve_ols = None
    _rust_compute_robust_vcov = None
    # TROP estimator acceleration
    _rust_unit_distance_matrix = None
    _rust_loocv_grid_search = None
    _rust_bootstrap_trop_variance = None
elif _backend_env == 'rust':
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

__all__ = [
    'HAS_RUST_BACKEND',
    '_rust_bootstrap_weights',
    '_rust_synthetic_weights',
    '_rust_project_simplex',
    '_rust_solve_ols',
    '_rust_compute_robust_vcov',
    # TROP estimator acceleration
    '_rust_unit_distance_matrix',
    '_rust_loocv_grid_search',
    '_rust_bootstrap_trop_variance',
]
