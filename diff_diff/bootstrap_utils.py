"""
Shared bootstrap utilities for multiplier bootstrap inference.

Provides weight generation, percentile CI, and p-value helpers used by
both CallawaySantAnna and ContinuousDiD estimators.
"""

import warnings
from typing import Optional, Tuple

import numpy as np

from diff_diff._backend import HAS_RUST_BACKEND, _rust_bootstrap_weights

__all__ = [
    "generate_bootstrap_weights",
    "generate_bootstrap_weights_batch",
    "generate_bootstrap_weights_batch_numpy",
    "compute_percentile_ci",
    "compute_bootstrap_pvalue",
    "compute_effect_bootstrap_stats",
]


def generate_bootstrap_weights(
    n_units: int,
    weight_type: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate bootstrap weights for multiplier bootstrap.

    Parameters
    ----------
    n_units : int
        Number of units (clusters) to generate weights for.
    weight_type : str
        Type of weights: "rademacher", "mammen", or "webb".
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Array of bootstrap weights with shape (n_units,).
    """
    if weight_type == "rademacher":
        return rng.choice([-1.0, 1.0], size=n_units)
    elif weight_type == "mammen":
        sqrt5 = np.sqrt(5)
        val1 = -(sqrt5 - 1) / 2
        val2 = (sqrt5 + 1) / 2
        p1 = (sqrt5 + 1) / (2 * sqrt5)
        return rng.choice([val1, val2], size=n_units, p=[p1, 1 - p1])
    elif weight_type == "webb":
        values = np.array([
            -np.sqrt(3 / 2), -np.sqrt(2 / 2), -np.sqrt(1 / 2),
            np.sqrt(1 / 2), np.sqrt(2 / 2), np.sqrt(3 / 2)
        ])
        return rng.choice(values, size=n_units)
    else:
        raise ValueError(
            f"weight_type must be 'rademacher', 'mammen', or 'webb', "
            f"got '{weight_type}'"
        )


def generate_bootstrap_weights_batch(
    n_bootstrap: int,
    n_units: int,
    weight_type: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate all bootstrap weights at once (vectorized).

    Uses Rust backend if available for parallel generation.

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap iterations.
    n_units : int
        Number of units (clusters) to generate weights for.
    weight_type : str
        Type of weights: "rademacher", "mammen", or "webb".
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Array of bootstrap weights with shape (n_bootstrap, n_units).
    """
    if HAS_RUST_BACKEND and _rust_bootstrap_weights is not None:
        seed = rng.integers(0, 2**63 - 1)
        return _rust_bootstrap_weights(n_bootstrap, n_units, weight_type, seed)
    return generate_bootstrap_weights_batch_numpy(n_bootstrap, n_units, weight_type, rng)


def generate_bootstrap_weights_batch_numpy(
    n_bootstrap: int,
    n_units: int,
    weight_type: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    NumPy fallback implementation of :func:`generate_bootstrap_weights_batch`.

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap iterations.
    n_units : int
        Number of units (clusters) to generate weights for.
    weight_type : str
        Type of weights: "rademacher", "mammen", or "webb".
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Array of bootstrap weights with shape (n_bootstrap, n_units).
    """
    if weight_type == "rademacher":
        return rng.choice([-1.0, 1.0], size=(n_bootstrap, n_units))
    elif weight_type == "mammen":
        sqrt5 = np.sqrt(5)
        val1 = -(sqrt5 - 1) / 2
        val2 = (sqrt5 + 1) / 2
        p1 = (sqrt5 + 1) / (2 * sqrt5)
        return rng.choice([val1, val2], size=(n_bootstrap, n_units), p=[p1, 1 - p1])
    elif weight_type == "webb":
        values = np.array([
            -np.sqrt(3 / 2), -np.sqrt(2 / 2), -np.sqrt(1 / 2),
            np.sqrt(1 / 2), np.sqrt(2 / 2), np.sqrt(3 / 2)
        ])
        return rng.choice(values, size=(n_bootstrap, n_units))
    else:
        raise ValueError(
            f"weight_type must be 'rademacher', 'mammen', or 'webb', "
            f"got '{weight_type}'"
        )


def compute_percentile_ci(
    boot_dist: np.ndarray,
    alpha: float,
) -> Tuple[float, float]:
    """
    Compute percentile confidence interval from bootstrap distribution.

    Parameters
    ----------
    boot_dist : np.ndarray
        Bootstrap distribution (1-D array).
    alpha : float
        Significance level (e.g., 0.05 for 95% CI).

    Returns
    -------
    tuple of float
        ``(lower, upper)`` confidence interval bounds.
    """
    lower = float(np.percentile(boot_dist, alpha / 2 * 100))
    upper = float(np.percentile(boot_dist, (1 - alpha / 2) * 100))
    return (lower, upper)


def compute_bootstrap_pvalue(
    original_effect: float,
    boot_dist: np.ndarray,
    n_valid: Optional[int] = None,
) -> float:
    """
    Compute two-sided bootstrap p-value using the percentile method.

    Parameters
    ----------
    original_effect : float
        Original point estimate.
    boot_dist : np.ndarray
        Bootstrap distribution of the effect.
    n_valid : int, optional
        Number of valid bootstrap samples for p-value floor.
        If None, uses ``len(boot_dist)``.

    Returns
    -------
    float
        Two-sided bootstrap p-value.
    """
    if original_effect >= 0:
        p_one_sided = np.mean(boot_dist <= 0)
    else:
        p_one_sided = np.mean(boot_dist >= 0)

    p_value = min(2 * p_one_sided, 1.0)
    n_for_floor = n_valid if n_valid is not None else len(boot_dist)
    p_value = max(p_value, 1 / (n_for_floor + 1))
    return float(p_value)


def compute_effect_bootstrap_stats(
    original_effect: float,
    boot_dist: np.ndarray,
    alpha: float = 0.05,
    context: str = "bootstrap distribution",
) -> Tuple[float, Tuple[float, float], float]:
    """
    Compute bootstrap statistics for a single effect.

    Filters non-finite samples, returning NaN for all statistics if
    fewer than 50% of samples are valid.

    Parameters
    ----------
    original_effect : float
        Original point estimate.
    boot_dist : np.ndarray
        Bootstrap distribution of the effect.
    alpha : float, default=0.05
        Significance level.
    context : str, optional
        Description for warning messages.

    Returns
    -------
    se : float
        Bootstrap standard error.
    ci : tuple of float
        Percentile confidence interval.
    p_value : float
        Bootstrap p-value.
    """
    if not np.isfinite(original_effect):
        return np.nan, (np.nan, np.nan), np.nan

    finite_mask = np.isfinite(boot_dist)
    n_valid = np.sum(finite_mask)
    n_total = len(boot_dist)

    if n_valid < n_total:
        n_nonfinite = n_total - n_valid
        warnings.warn(
            f"Dropping {n_nonfinite}/{n_total} non-finite bootstrap samples "
            f"in {context}. Bootstrap estimates based on remaining valid samples.",
            RuntimeWarning,
            stacklevel=3,
        )

    if n_valid < n_total * 0.5:
        warnings.warn(
            f"Too few valid bootstrap samples ({n_valid}/{n_total}) in {context}. "
            "Returning NaN for SE/CI/p-value to signal invalid inference.",
            RuntimeWarning,
            stacklevel=3,
        )
        return np.nan, (np.nan, np.nan), np.nan

    valid_dist = boot_dist[finite_mask]
    se = float(np.std(valid_dist, ddof=1))

    # Guard: if SE is not finite or zero, all inference fields must be NaN.
    if not np.isfinite(se) or se <= 0:
        warnings.warn(
            f"Bootstrap SE is non-finite or zero (n_valid={n_valid}) in {context}. "
            "Returning NaN for SE/CI/p-value.",
            RuntimeWarning,
            stacklevel=3,
        )
        return np.nan, (np.nan, np.nan), np.nan

    ci = compute_percentile_ci(valid_dist, alpha)
    p_value = compute_bootstrap_pvalue(
        original_effect, valid_dist, n_valid=len(valid_dist)
    )
    return se, ci, p_value
