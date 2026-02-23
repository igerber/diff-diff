"""
Pytest configuration and shared fixtures for diff-diff tests.

This module provides shared fixtures including lazy R availability checking
to avoid import-time subprocess latency.
"""

import math
import os
import subprocess

# Force non-interactive matplotlib backend before any test imports it.
# Prevents plt.show() from blocking the test suite on a GUI window.
os.environ.setdefault("MPLBACKEND", "Agg")

import pytest


# =============================================================================
# R Availability Fixtures (Lazy Loading)
# =============================================================================

_r_available_cache = None


def _check_r_available() -> bool:
    """
    Check if R and required packages (did, jsonlite) are available (cached).

    This is called lazily when the r_available fixture is first used,
    not at module import time, to avoid subprocess latency during test collection.

    Returns
    -------
    bool
        True if R and required packages are available, False otherwise.
    """
    global _r_available_cache
    if _r_available_cache is None:
        # Allow environment override (matches DIFF_DIFF_BACKEND pattern)
        r_env = os.environ.get("DIFF_DIFF_R", "auto").lower()
        if r_env == "skip":
            _r_available_cache = False
        else:
            try:
                result = subprocess.run(
                    ["Rscript", "-e", "library(did); library(jsonlite); cat('OK')"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                _r_available_cache = result.returncode == 0 and "OK" in result.stdout
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                _r_available_cache = False
    return _r_available_cache


@pytest.fixture(scope="session")
def r_available():
    """
    Lazy check for R availability.

    This fixture is session-scoped and cached, so R availability is only
    checked once per test session, and only when a test actually needs it.

    Returns
    -------
    bool
        True if R and required packages (did, jsonlite) are available.
    """
    return _check_r_available()


@pytest.fixture
def require_r(r_available):
    """
    Skip test if R is not available.

    Use this fixture in tests that require R:

    ```python
    def test_comparison_with_r(require_r):
        # This test will be skipped if R is not available
        ...
    ```
    """
    if not r_available:
        pytest.skip("R or did package not available")


# =============================================================================
# CI Performance: Backend-Aware Parameter Scaling
# =============================================================================

from diff_diff._backend import HAS_RUST_BACKEND

_PURE_PYTHON_MODE = (
    os.environ.get("DIFF_DIFF_BACKEND", "auto").lower() == "python"
    or not HAS_RUST_BACKEND
)


class CIParams:
    """Scale test parameters in pure Python mode for CI performance.

    When Rust backend is available, all values pass through unchanged.
    In pure Python mode, bootstrap iterations and LOOCV grids are scaled
    down to reduce CI time while preserving code path coverage.
    """

    @staticmethod
    def bootstrap(n: int, *, min_n: int = 11) -> int:
        """Scale bootstrap iterations. Guaranteed monotonic: bootstrap(n+1) >= bootstrap(n).

        Use a larger min_n for tests comparing analytical vs bootstrap SEs,
        which need more iterations for stable convergence.
        In pure Python mode, min_n is capped at 49 to keep CI fast.
        """
        if not _PURE_PYTHON_MODE or n <= 10:
            return n
        effective_min = min(min_n, 49)
        return min(n, max(effective_min, int(math.sqrt(n) * 1.6)))

    @staticmethod
    def grid(values: list) -> list:
        """Scale TROP lambda grids. Keeps first, middle, last for grids > 3 elements."""
        if not _PURE_PYTHON_MODE or len(values) <= 3:
            return values
        return [values[0], values[len(values) // 2], values[-1]]


@pytest.fixture(scope="session")
def ci_params():
    """Backend-aware parameter scaling for CI performance."""
    return CIParams()


# =============================================================================
# NaN Inference Assertion Helper
# =============================================================================

import numpy as np


def assert_nan_inference(inference_dict):
    """Assert ALL inference fields are NaN-consistent when SE is non-finite or <= 0.

    Use this helper to validate that when SE is undefined, ALL downstream
    inference fields are NaN (not zero, not some valid-looking number).

    Parameters
    ----------
    inference_dict : dict
        Dictionary with keys: ``se``, ``t_stat``, ``p_value``, ``conf_int``
        where ``conf_int`` is a 2-tuple ``(ci_lower, ci_upper)``.

    Raises
    ------
    AssertionError
        If any inference field is not NaN when it should be.
    """
    se = inference_dict["se"]
    assert not (np.isfinite(se) and se > 0), (
        f"assert_nan_inference called but SE={se} is finite and positive. "
        "This helper is for validating NaN propagation when SE is invalid."
    )
    assert np.isnan(inference_dict["t_stat"]), (
        f"t_stat should be NaN when SE={se}, got {inference_dict['t_stat']}"
    )
    assert np.isnan(inference_dict["p_value"]), (
        f"p_value should be NaN when SE={se}, got {inference_dict['p_value']}"
    )
    ci = inference_dict["conf_int"]
    assert np.isnan(ci[0]), (
        f"ci_lower should be NaN when SE={se}, got {ci[0]}"
    )
    assert np.isnan(ci[1]), (
        f"ci_upper should be NaN when SE={se}, got {ci[1]}"
    )
