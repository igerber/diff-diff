"""Randomization inference for LWDiD estimator.

Implements Fisher's randomization inference under the sharp null
hypothesis H0: τ_i = 0 for all i (no individual treatment effect).

References
----------
Fisher, R. A. (1935). The Design of Experiments.
Lee, S. J. & Wooldridge, J. M. (2026). "Simple Approaches to Inference
  with Difference-in-Differences Estimators with Small Cross-Sectional
  Sample Sizes." SSRN 5325686 (randomization inference for small-N
  designs; implemented per the authors'-package inclusive convention).
"""

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from diff_diff.linalg import solve_ols


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
        Resampling method used: always 'permutation'.
    seed : int or None
        Random seed used for reproducibility.
    n_dropped : int
        Observations dropped for non-finite y before estimation (warned).
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
    #: Observations dropped for non-finite y before estimation (warned).
    n_dropped: int = 0


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
    ValueError
        If any validation check fails.
    """
    if (
        n_reps is None
        or isinstance(n_reps, bool)
        or not isinstance(n_reps, (int, np.integer))
        or n_reps < 10
    ):
        # Round-24 review: the valid-replication floor below is
        # max(10, ...), so n_reps < 10 can NEVER satisfy it - reject up
        # front instead of failing after the permutation loop.
        raise ValueError(
            f"n_reps must be an integer >= 10 (the reliable-inference "
            f"floor requires at least 10 valid replications), got {n_reps!r}"
        )

    if method == "bootstrap":
        # Review finding: resampling treatment labels WITH replacement
        # changes the treated count and is not the complete-randomization
        # assignment mechanism of Fisher randomization inference - it was
        # presented under the Fisher umbrella without a specified
        # assignment design. The mode is removed (LWDiD is unreleased).
        raise ValueError(
            "method='bootstrap' has been removed: resampling treatment "
            "labels with replacement is not Fisher randomization inference "
            "(it changes the treated count and has no specified assignment "
            "mechanism). Use method='permutation'."
        )
    if method != "permutation":
        raise ValueError(f"method must be 'permutation', got '{method}'")

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


def _build_design(
    y: np.ndarray,
    treatment: np.ndarray,
    controls: np.ndarray,
    design: str,
) -> np.ndarray:
    """Build the regression design for a given treatment assignment.

    ``'linear'`` is the generic ``[1, D, X]`` covariate-adjusted contrast.
    ``'ra_interacted'`` is the LWDiD RA design ``[1, D, X, D(X - Xbar_1)]``
    (LW eq. E.1) with the treated covariate mean RECOMPUTED for the given
    assignment - required so each permutation tests the same estimator the
    fit reported (round-5 review).
    """
    n = len(y)
    if design == "ra_interacted":
        xbar1 = controls[treatment == 1].mean(axis=0)
        return np.column_stack(
            [np.ones(n), treatment, controls, treatment[:, None] * (controls - xbar1)]
        )
    return np.column_stack([np.ones(n), treatment, controls])


def _compute_observed_att(
    y: np.ndarray,
    treatment: np.ndarray,
    controls: Optional[np.ndarray],
    design: str = "linear",
) -> float:
    """Compute the observed ATT from the data.

    When controls are present, uses OLS via the shared solve_ols.
    Otherwise computes the simple mean difference.
    """
    if controls is None:
        mask1 = treatment == 1
        return float(y[mask1].mean() - y[~mask1].mean())

    if controls.ndim == 1:
        controls = controls.reshape(-1, 1)
    X = _build_design(y, treatment, controls, design)
    # Rank-aware shared solver (round-4 review: lstsq returned a finite
    # minimum-norm treatment coefficient when a control duplicated the
    # treatment column, so RI tested an unidentified statistic).
    coefs, _, _ = solve_ols(X, y)
    if not np.isfinite(coefs[1]):
        raise ValueError(
            "The treatment coefficient is not identified: the design is "
            "rank-deficient and the shared solver dropped the treatment "
            "column (e.g. a control collinear with treatment). Remove the "
            "collinear control(s) before running randomization inference."
        )
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
    design: str = "linear",
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

        # The design is rebuilt per assignment: under 'ra_interacted' the
        # treated covariate mean (and the interaction columns) depend on
        # the drawn assignment (round-5 review - the pre-fix code updated
        # only the treatment column of a fixed [1, D, X] matrix).
        X = _build_design(y, d_b, controls, design)

        try:
            with warnings.catch_warnings():
                # Rank warnings per draw would flood; a dropped treatment
                # coefficient is recorded as a failed replication (NaN)
                # and surfaced through the failed-rep accounting.
                warnings.simplefilter("ignore")
                coefs, _, _ = solve_ols(X, y)
            att_dist[b] = coefs[1] if np.isfinite(coefs[1]) else np.nan
        except (np.linalg.LinAlgError, ValueError):
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
    design: str = "linear",
) -> RandomizationResult:
    """Fisher randomization inference for testing zero treatment effect.

    Tests the sharp null hypothesis H0: τ_i = 0 for all i by permuting
    treatment labels and computing a Monte Carlo p-value
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
    method : {'permutation'}, default 'permutation'
        Resampling method:

        - 'permutation': Classical Fisher randomization inference. Permutes
          treatment labels without replacement, preserving the original
          number of treated and control units.

    seed : int or None, optional
        Random seed for reproducibility.
    design : {'linear', 'ra_interacted'}, default 'linear'
        Regression design used for the covariate-adjusted statistic.
        ``'linear'`` fits ``[1, D, X]``. ``'ra_interacted'`` fits the LWDiD
        RA design ``[1, D, X, D(X - Xbar_1)]`` and RECOMPUTES the treated
        covariate mean for every permuted assignment, so the permuted
        statistic is the same estimator as the observed one (used by
        ``LWDiDResults.randomization_test`` to match the fitted ATT).

    Returns
    -------
    RandomizationResult
        Dataclass containing p-value, observed ATT, randomization
        distribution, and diagnostic information.

    Raises
    ------
    ValueError
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
    pre-allocated design matrix is refit through the shared rank-aware
    ``solve_ols`` solver (draws whose treatment coefficient is dropped
    count as failed replications).

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

    # Shape/length coherence FIRST (round-18 review: applying the finite
    # mask before these checks turned a mismatched treatment/controls
    # length into a raw boolean-index IndexError instead of the
    # documented ValueError).
    if y.ndim != 1:
        raise ValueError(f"y must be a 1-d array, got shape {y.shape}")
    if treatment.ndim != 1:
        raise ValueError(f"treatment must be a 1-d array, got shape {treatment.shape}")
    if len(y) != len(treatment):
        raise ValueError(
            f"y and treatment must have the same length, got {len(y)} and {len(treatment)}"
        )
    if controls is not None and controls.shape[0] != len(y):
        raise ValueError(f"controls must have {len(y)} rows, got {controls.shape[0]}")

    # Drop observations with non-finite y WITH a warning (campaign
    # finding: silent drops) and record the count on the result.
    n_dropped = 0
    if len(y) > 0:
        finite_mask = np.isfinite(y)
        if not finite_mask.all():
            n_dropped = int((~finite_mask).sum())
            warnings.warn(
                f"randomization_inference: dropped {n_dropped} observation(s) "
                f"with non-finite y before estimation.",
                UserWarning,
                stacklevel=2,
            )
            y = y[finite_mask]
            treatment = treatment[finite_mask]
            if controls is not None:
                controls = controls[finite_mask]

    _validate_inputs(y, treatment, controls, n_reps, method)

    # ------------------------------------------------------------------
    # Compute observed ATT
    # ------------------------------------------------------------------
    if design not in ("linear", "ra_interacted"):
        raise ValueError(f"design must be 'linear' or 'ra_interacted', got {design!r}")
    if design == "ra_interacted" and controls is None:
        raise ValueError(
            "design='ra_interacted' requires controls (the design is [1, D, X, D(X - Xbar_1)])."
        )

    att_obs = _compute_observed_att(y, treatment, controls, design)

    # ------------------------------------------------------------------
    # Generate randomization distribution
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)

    if controls is None:
        att_dist = _fast_path(y, treatment, n_reps, method, rng)
    else:
        att_dist = _slow_path(y, treatment, controls, n_reps, method, rng, design)

    # ------------------------------------------------------------------
    # Compute p-value and diagnostics
    # ------------------------------------------------------------------
    pvalue, n_valid, n_failed = _compute_pvalue(att_dist, att_obs)
    failure_rate = n_failed / n_reps

    # Error if too few valid replications
    if n_valid < max(10, int(0.1 * n_reps)):
        raise ValueError(
            f"Insufficient valid replications for reliable inference: "
            f"{n_valid}/{n_reps} valid (failure rate {failure_rate:.1%}). "
            f"Increase n_reps, or check for near-constant treatment/"
            f"outcome configurations that make draws degenerate."
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
        n_dropped=n_dropped,
    )
