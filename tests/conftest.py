"""
Pytest configuration and shared fixtures for diff-diff tests.

This module provides shared fixtures including lazy R availability checking
to avoid import-time subprocess latency.
"""

import os
import subprocess

import pytest


# =============================================================================
# R Availability Fixtures (Lazy Loading)
# =============================================================================

_r_available_cache = None


def _check_r_available() -> bool:
    """
    Check if R and the did package are available (cached).

    This is called lazily when the r_available fixture is first used,
    not at module import time, to avoid subprocess latency during test collection.

    Returns
    -------
    bool
        True if R and did package are available, False otherwise.
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
                    ["Rscript", "-e", "library(did); cat('OK')"],
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
        True if R and did package are available.
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
