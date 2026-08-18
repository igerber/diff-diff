"""Randomization inference for LWDiD estimator.

Implements Fisher's randomization inference under the sharp null
hypothesis H0: τ_i = 0 for all i (no individual treatment effect).

References
----------
Fisher, R. A. (1935). The Design of Experiments.
Lee, S. J. & Wooldridge, J. M. (2025). Section 5. SSRN 4516518.
"""

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from diff_diff.lwdid_exceptions import RandomizationWarning

# Backward compat alias
RandomizationError = ValueError


@dataclass
class RandomizationResult:
    """Result container for randomization inference.

    Attributes
    ----------
    pvalue : float
        Two-sided p-value from the randomization distribution.
    att_observed : float
        Observed ATT estimate from the original data.
    att_distribution : np.ndarray
        Array of ATT estimates from randomization replications (includes NaN
        for failed replications).
    n_reps : int
        Total number of replications requested.
    n_valid : int
        Number of valid (non-degenerate) replications used for p-value.
    n_failed : int
        Number of failed or degenerate replications.
    failure_rate : float
        Proportion of replications that failed (n_failed / n_reps).
    method : str
        Resampling method used: 'permutation' or 'bootstrap'.
    seed : int or None
        Random seed used for reproducibility.
    """

    pvalue: float
    att_observed: float
    att_distribution: np.ndarray
    n_reps: int
    n_valid: int
    n_failed: int
    failure_rate: float
    method: str
    seed: Optional[int]


def _validate_inputs(
    y: np.ndarray,
    treatment: np.ndarray,
    controls: Optional[np.ndarray],
    n_reps: int,
    method: str,
) -> None:
    """Validate inputs for randomization inference.

    Raises
    ------
    RandomizationError
        If any validation check fails.
    """
    if n_reps is None or n_reps <= 0:
        raise ValueError("n_reps must be a positive integer")

    if method not in ("permutation", "bootstrap"):
        raise ValueError(f"method must be 'permutation' or 'bootstrap', got '{method}'")

    if y.ndim != 1:
        raise ValueError(f"y must be a 1-d array, got shape {y.shape}")

    if treatment.ndim != 1:
        raise ValueError(f"treatment must be a 1-d array, got shape {treatment.shape}")

    if len(y) == 0:
        raise ValueError("y must not be empty.")

    if len(y) != len(treatment):
        raise ValueError(
            f"y and treatment must have the same length, " f"got {len(y)} and {len(treatment)}"
        )

    n = len(y)
    if n < 3:
        raise ValueError(f"Sample size too small for randomization inference: N={n}")

    if not np.all((treatment == 0) | (treatment == 1)):
        raise ValueError(
            "treatment must be binary (0 or 1). "
            f"Got values in [{treatment.min()}, {treatment.max()}]."
        )

    n1 = int(treatment.sum())
    if n1 == 0 or n1 == n:
        raise ValueError(
            "Treatment variable is constant (all treated or all control). "
            "Randomization inference requires variation in treatment."
        )

    if controls is not None:
        if controls.ndim == 1:
            controls = controls.reshape(-1, 1)
        if controls.shape[0] != n:
            raise ValueError(f"controls must have {n} rows, got {controls.shape[0]}")
        if not np.all(np.isfinite(controls)):
            raise ValueError(
                "controls contains non-finite values (NaN or Inf). "
                "Please remove or impute missing values before calling "
                "randomization_inference()."
            )


def _compute_observed_att(
    y: np.ndarray,
    treatment: np.ndarray,
    controls: Optional[np.ndarray],
) -> float:
    """Compute the observed ATT from the data.

    When controls are present, uses OLS via lstsq.
    Otherwise computes the simple mean difference.
    """
    if controls is None:
        mask1 = treatment == 1
        return float(y[mask1].mean() - y[~mask1].mean())

    n = len(y)
    if controls.ndim == 1:
        controls = controls.reshape(-1, 1)
    X = np.column_stack([np.ones(n), treatment, controls])
    coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return float(coefs[1])


def _fast_path(
    y: np.ndarray,
    treatment: np.ndarray,
    n_reps: int,
    method: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Fast path: no controls, direct mean-difference computation.

    Returns
    -------
    att_dist : ndarray of shape (n_reps,)
        Randomization distribution of ATT. Failed reps contain NaN.
    """
    n = len(y)
    att_dist = np.empty(n_reps)

    for b in range(n_reps):
        if method == "permutation":
            d_b = rng.permutation(treatment)
        else:
            d_b = rng.choice(treatment, size=n, replace=True)

        n1_b = d_b.sum()
        if n1_b == 0 or n1_b == n:
            att_dist[b] = np.nan
            continue

        mask1 = d_b == 1
        att_dist[b] = y[mask1].mean() - y[~mask1].mean()

    return att_dist


def _slow_path(
    y: np.ndarray,
    treatment: np.ndarray,
    controls: np.ndarray,
    n_reps: int,
    method: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Slow path: with controls, OLS via pre-allocated design matrix.

    The design matrix is pre-allocated and only the treatment column
    (column 1) is updated per replication. This avoids repeated memory
    allocation and keeps the cost to O(N*K) per iteration.

    Returns
    -------
    att_dist : ndarray of shape (n_reps,)
        Randomization distribution of ATT. Failed reps contain NaN.
    """
    n = len(y)
    if controls.ndim == 1:
        controls = controls.reshape(-1, 1)

    # Pre-allocate design matrix: [intercept, treatment, controls]
    X = np.column_stack([np.ones(n), treatment, controls])
    att_dist = np.empty(n_reps)

    for b in range(n_reps):
        if method == "permutation":
            d_b = rng.permutation(treatment)
        else:
            d_b = rng.choice(treatment, size=n, replace=True)

        n1_b = d_b.sum()
        if n1_b == 0 or n1_b == n:
            att_dist[b] = np.nan
            continue

        # Update only the treatment column
        X[:, 1] = d_b

        try:
            coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            att_dist[b] = coefs[1]
        except np.linalg.LinAlgError:
            att_dist[b] = np.nan

    return att_dist


def _compute_pvalue(att_dist: np.ndarray, att_obs: float) -> tuple:
    """Compute two-sided p-value from randomization distribution.

    Uses the formula: p = (sum(|ATT*| >= |ATT_obs|) + 1) / (n_valid + 1)
    following Phipson & Smyth (2010).  The non-strict inequality counts
    replications at least as extreme as the observed statistic, so a
    fully tied distribution (e.g. constant outcome) yields p = 1.0,
    while the +1 in numerator and denominator accounts for the observed
    statistic itself and guarantees p > 0.

    Returns
    -------
    pvalue : float
    n_valid : int
    n_failed : int
    """
    valid_mask = np.isfinite(att_dist)
    n_valid = int(valid_mask.sum())
    n_failed = len(att_dist) - n_valid

    if n_valid == 0:
        return 1.0, 0, n_failed

    valid_atts = att_dist[valid_mask]
    pvalue = float((np.sum(np.abs(valid_atts) >= np.abs(att_obs)) + 1) / (n_valid + 1))
    return pvalue, n_valid, n_failed


def randomization_inference(
    y: np.ndarray,
    treatment: np.ndarray,
    controls: Optional[np.ndarray] = None,
    n_reps: int = 1000,
    method: str = "permutation",
    seed: Optional[int] = None,
) -> RandomizationResult:
    """Fisher randomization inference for testing zero treatment effect.

    Tests the sharp null hypothesis H0: τ_i = 0 for all i by permuting
    (or bootstrapping) treatment labels and computing a Monte Carlo p-value
    as the proportion of resampled test statistics at least as extreme as
    the observed statistic.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Transformed outcome variable.
    treatment : ndarray of shape (n,)
        Binary treatment indicator (0/1).
    controls : ndarray of shape (n, K) or None, optional
        Control variables to include in the regression model. When None,
        ATT is computed as a simple mean difference (fast path). When
        provided, ATT is estimated via OLS with controls (slow path).
    n_reps : int, default 1000
        Number of randomization replications for computing the p-value.
    method : {'permutation', 'bootstrap'}, default 'permutation'
        Resampling method:

        - 'permutation': Classical Fisher randomization inference. Permutes
          treatment labels without replacement, preserving the original
          number of treated and control units.
        - 'bootstrap': Resamples treatment labels with replacement. May
          produce degenerate draws which are excluded from p-value.

    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    RandomizationResult
        Dataclass containing p-value, observed ATT, randomization
        distribution, and diagnostic information.

    Raises
    ------
    RandomizationError
        If inputs are invalid, sample size is too small, treatment is
        constant, or insufficient valid replications are produced.

    Notes
    -----
    The p-value is computed as:

        p = (sum(|ATT*| >= |ATT_obs|) + 1) / (n_valid + 1)

    following Phipson & Smyth (2010).  The non-strict inequality counts
    replications at least as extreme as the observed statistic (standard
    randomization-test convention, so a degenerate all-tie distribution
    yields p = 1.0), while the +1 ensures the p-value is strictly
    positive and provides valid finite-sample inference.

    When controls are absent, ATT is computed directly as the difference
    in means between treated and control groups. With controls, a
    pre-allocated design matrix is used with ``np.linalg.lstsq`` for
    efficiency.

    Examples
    --------
    >>> import numpy as np
    >>> from diff_diff.lwdid_randomization import randomization_inference
    >>> rng = np.random.default_rng(42)
    >>> y = rng.normal(0, 1, 100)
    >>> y[:30] += 2.0
    >>> treatment = np.zeros(100); treatment[:30] = 1.0
    >>> r = randomization_inference(y, treatment, n_reps=999, seed=0)
    >>> r.pvalue < 0.05
    True
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    y = np.asarray(y, dtype=np.float64)
    treatment = np.asarray(treatment, dtype=np.float64)

    if controls is not None:
        controls = np.asarray(controls, dtype=np.float64)
        if controls.ndim == 1:
            controls = controls.reshape(-1, 1)

    # Handle NaN: drop observations with non-finite y
    if y.ndim == 1 and len(y) > 0:
        finite_mask = np.isfinite(y)
        if not finite_mask.all():
            y = y[finite_mask]
            treatment = treatment[finite_mask]
            if controls is not None:
                controls = controls[finite_mask]

    _validate_inputs(y, treatment, controls, n_reps, method)

    # ------------------------------------------------------------------
    # Compute observed ATT
    # ------------------------------------------------------------------
    att_obs = _compute_observed_att(y, treatment, controls)

    # ------------------------------------------------------------------
    # Generate randomization distribution
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)

    if controls is None:
        att_dist = _fast_path(y, treatment, n_reps, method, rng)
    else:
        att_dist = _slow_path(y, treatment, controls, n_reps, method, rng)

    # ------------------------------------------------------------------
    # Compute p-value and diagnostics
    # ------------------------------------------------------------------
    pvalue, n_valid, n_failed = _compute_pvalue(att_dist, att_obs)
    failure_rate = n_failed / n_reps

    # Warn if failure rate is high (bootstrap only; permutation preserves
    # treatment proportions and should never produce degenerate draws)
    if method == "bootstrap" and failure_rate > 0.10:
        warnings.warn(
            f"Randomization inference: {n_failed}/{n_reps} replications "
            f"produced degenerate treatment assignments "
            f"({failure_rate:.1%} failure rate). "
            f"Consider using method='permutation' or increasing sample size.",
            RandomizationWarning,
            stacklevel=2,
        )

    # Error if too few valid replications
    if n_valid < max(10, int(0.1 * n_reps)):
        raise ValueError(
            f"Insufficient valid replications for reliable inference: "
            f"{n_valid}/{n_reps} valid (failure rate {failure_rate:.1%}). "
            f"Use method='permutation' to avoid degenerate draws."
        )

    return RandomizationResult(
        pvalue=pvalue,
        att_observed=att_obs,
        att_distribution=att_dist,
        n_reps=n_reps,
        n_valid=n_valid,
        n_failed=n_failed,
        failure_rate=failure_rate,
        method=method,
        seed=seed,
    )
