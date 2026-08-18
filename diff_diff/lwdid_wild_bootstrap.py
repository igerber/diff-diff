"""Wild cluster bootstrap for inference with few clusters.

This module implements the wild cluster bootstrap method (Cameron, Gelbach &
Miller 2008) for reliable inference when the number of clusters is small.
The method is particularly useful in difference-in-differences settings where
standard cluster-robust standard errors may perform poorly.

The wild cluster bootstrap is recommended when:

- Number of clusters G < 30
- Cluster sizes are unbalanced
- Few treated clusters

Key features:

- Full enumeration mode for exact p-values when G <= 12
- Multiple weight distributions: Rademacher, Mammen, Webb (6-point)
- Batch matrix computation with memory chunking for large datasets
- Precomputed projection matrices to avoid per-iteration overhead

References
----------
Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). Bootstrap-based
improvements for inference with clustered errors. *Review of Economics
and Statistics*, 90(3), 414-427.

Webb, M. D. (2014). Reworking wild bootstrap based inference for clustered
errors. *Queen's Economics Department Working Paper*, No. 1315.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import product
from typing import Optional

import numpy as np

from .lwdid_exceptions import NumericalWarning

# Backward compat alias
BootstrapConvergenceError = ValueError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FULL_ENUM_THRESHOLD = 12  # Use full enumeration when G <= this
_MEMORY_THRESHOLD = 50_000_000  # n_reps * n_obs elements before chunking
_VALID_WEIGHT_TYPES = ("rademacher", "mammen", "webb")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class WildClusterBootstrapResult:
    """Result of wild cluster bootstrap inference.

    Attributes
    ----------
    att : float
        Point estimate of the average treatment effect on the treated.
    se_bootstrap : float
        Bootstrap standard error (std of bootstrap ATT estimates).
    ci_lower : float
        Lower bound of the bootstrap confidence interval.
    ci_upper : float
        Upper bound of the bootstrap confidence interval.
    pvalue : float
        Bootstrap p-value (two-sided), computed as the fraction of
        bootstrap |t*| >= |t_original|.
    weight_type : str
        Weight distribution used ('rademacher', 'mammen', or 'webb').
    n_reps : int
        Number of bootstrap replications actually performed.
    n_clusters : int
        Number of clusters in the data.
    t_stats : np.ndarray
        Array of bootstrap t-statistics (length = n_reps).
    """

    att: float
    se_bootstrap: float
    ci_lower: float
    ci_upper: float
    pvalue: float
    weight_type: str
    n_reps: int
    n_clusters: int
    t_stats: np.ndarray = field(repr=False)

    def summary(self) -> str:
        """Return a human-readable summary string."""
        sig = (
            "***"
            if self.pvalue < 0.01
            else "**" if self.pvalue < 0.05 else "*" if self.pvalue < 0.1 else ""
        )
        return (
            f"Wild Cluster Bootstrap Results\n"
            f"{'=' * 50}\n"
            f"ATT: {self.att:.4f} {sig}\n"
            f"Bootstrap SE: {self.se_bootstrap:.4f}\n"
            f"95% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]\n"
            f"P-value: {self.pvalue:.4f}\n"
            f"N clusters: {self.n_clusters}\n"
            f"N bootstrap reps: {self.n_reps}\n"
            f"Weight type: {self.weight_type}\n"
            f"{'=' * 50}"
        )


# ---------------------------------------------------------------------------
# Weight generation functions
# ---------------------------------------------------------------------------


def _rademacher_weights(n_clusters: int, n_reps: int, rng: np.random.Generator) -> np.ndarray:
    """Generate Rademacher bootstrap weights.

    Each weight is +1 or -1 with equal probability 0.5.
    E[w] = 0, E[w^2] = 1.

    Parameters
    ----------
    n_clusters : int
        Number of clusters (G).
    n_reps : int
        Number of bootstrap replications (B).
    rng : numpy.random.Generator
        Random number generator instance.

    Returns
    -------
    np.ndarray
        Shape (n_reps, n_clusters) array of weights in {-1, +1}.
    """
    return rng.choice(np.array([-1, 1], dtype=np.float64), size=(n_reps, n_clusters))


def _mammen_weights(n_clusters: int, n_reps: int, rng: np.random.Generator) -> np.ndarray:
    """Generate Mammen two-point bootstrap weights.

    Two-point distribution matching the first three moments:
        P(w = -(sqrt(5)-1)/2) = (sqrt(5)+1) / (2*sqrt(5))
        P(w =  (sqrt(5)+1)/2) = (sqrt(5)-1) / (2*sqrt(5))

    E[w] = 0, E[w^2] = 1, E[w^3] = 1.

    Parameters
    ----------
    n_clusters : int
        Number of clusters (G).
    n_reps : int
        Number of bootstrap replications (B).
    rng : numpy.random.Generator
        Random number generator instance.

    Returns
    -------
    np.ndarray
        Shape (n_reps, n_clusters) array of Mammen weights.
    """
    sqrt5 = np.sqrt(5.0)
    p = (sqrt5 + 1.0) / (2.0 * sqrt5)
    w1 = -(sqrt5 - 1.0) / 2.0  # approx -0.618
    w2 = (sqrt5 + 1.0) / 2.0  # approx  1.618

    u = rng.random((n_reps, n_clusters))
    return np.where(u < p, w1, w2)


def _webb_weights(n_clusters: int, n_reps: int, rng: np.random.Generator) -> np.ndarray:
    """Generate Webb six-point bootstrap weights.

    Six-point distribution (Webb 2014), designed for very few clusters:
        values: +-sqrt(1/2), +-sqrt(2/2), +-sqrt(3/2)
        each with probability 1/6.

    E[w] = 0, E[w^2] = 1.

    Parameters
    ----------
    n_clusters : int
        Number of clusters (G).
    n_reps : int
        Number of bootstrap replications (B).
    rng : numpy.random.Generator
        Random number generator instance.

    Returns
    -------
    np.ndarray
        Shape (n_reps, n_clusters) array of Webb weights.
    """
    values = np.array(
        [
            -np.sqrt(3.0 / 2.0),
            -np.sqrt(2.0 / 2.0),
            -np.sqrt(1.0 / 2.0),
            np.sqrt(1.0 / 2.0),
            np.sqrt(2.0 / 2.0),
            np.sqrt(3.0 / 2.0),
        ]
    )
    return rng.choice(values, size=(n_reps, n_clusters))


def _generate_all_rademacher(n_clusters: int) -> np.ndarray:
    """Generate all 2^G Rademacher weight combinations for full enumeration.

    Parameters
    ----------
    n_clusters : int
        Number of clusters G (must be <= 12 for tractability).

    Returns
    -------
    np.ndarray
        Shape (2^G, G) array of all {-1, +1} combinations.
    """
    return np.array(list(product([-1.0, 1.0], repeat=n_clusters)), dtype=np.float64)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_design_matrix(treatment: np.ndarray, controls: Optional[np.ndarray]) -> np.ndarray:
    """Build the OLS design matrix [intercept, treatment, controls].

    Parameters
    ----------
    treatment : np.ndarray
        Treatment indicator, shape (N,).
    controls : np.ndarray or None
        Control variables, shape (N, p) or None.

    Returns
    -------
    np.ndarray
        Design matrix X of shape (N, k) where k = 2 + p.
    """
    n = len(treatment)
    parts = [np.ones((n, 1), dtype=np.float64), treatment.reshape(-1, 1).astype(np.float64)]
    if controls is not None:
        ctrl = np.asarray(controls, dtype=np.float64)
        if ctrl.ndim == 1:
            ctrl = ctrl.reshape(-1, 1)
        parts.append(ctrl)
    return np.hstack(parts)


def _precompute(
    y: np.ndarray,
    X: np.ndarray,
    cluster_ids: np.ndarray,
) -> dict:
    """Precompute matrices needed for the bootstrap loop.

    Computes once:
      - (X'X)^{-1}, projection P = (X'X)^{-1} X'
      - beta_hat, residuals
      - Cluster membership indices and masks

    Parameters
    ----------
    y : np.ndarray, shape (N,)
        Outcome vector.
    X : np.ndarray, shape (N, k)
        Design matrix (intercept + treatment + controls).
    cluster_ids : np.ndarray, shape (N,)
        Cluster identifiers.

    Returns
    -------
    dict
        Dictionary with precomputed quantities.
    """
    N, k = X.shape

    # Normal equations
    XtX = X.T @ X

    # Condition number check
    cond = np.linalg.cond(XtX)
    if cond > 1e12:
        warnings.warn(
            f"Design matrix X'X has large condition number ({cond:.2e}). "
            f"Bootstrap t-statistics may lose numerical precision.",
            NumericalWarning,
            stacklevel=3,
        )

    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        warnings.warn(
            "X'X is singular; falling back to pseudo-inverse.",
            NumericalWarning,
            stacklevel=3,
        )
        XtX_inv = np.linalg.pinv(XtX)

    P = XtX_inv @ X.T  # shape (k, N)
    beta_hat = P @ y
    residuals = y - X @ beta_hat

    # Cluster structure
    unique_clusters = np.unique(cluster_ids)
    G = len(unique_clusters)
    cluster_map = {c: i for i, c in enumerate(unique_clusters)}
    obs_cluster_idx = np.array([cluster_map[c] for c in cluster_ids], dtype=np.intp)

    # Precompute per-cluster masks
    cluster_masks: list[np.ndarray] = []
    for g in range(G):
        cluster_masks.append(np.where(obs_cluster_idx == g)[0])

    # Precompute "meat" components for cluster-robust SE
    # For each cluster g: X_g' e_g (shape k), needed for CR variance
    # Also store X_g for later use
    cluster_X: list[np.ndarray] = []
    for g in range(G):
        cluster_X.append(X[cluster_masks[g]])

    return {
        "y": y,
        "X": X,
        "P": P,
        "XtX_inv": XtX_inv,
        "beta_hat": beta_hat,
        "residuals": residuals,
        "obs_cluster_idx": obs_cluster_idx,
        "cluster_masks": cluster_masks,
        "cluster_X": cluster_X,
        "G": G,
        "N": N,
        "k": k,
    }


def _cluster_robust_se(
    X: np.ndarray,
    residuals: np.ndarray,
    XtX_inv: np.ndarray,
    cluster_masks: list[np.ndarray],
    cluster_X: list[np.ndarray],
    G: int,
    N: int,
    k: int,
    coef_idx: int = 1,
) -> float:
    """Compute cluster-robust standard error for a single coefficient.

    Uses the sandwich estimator:
        V = (X'X)^{-1} B (X'X)^{-1}
    where B = sum_g (X_g' e_g)(X_g' e_g)' with finite-sample correction.

    Parameters
    ----------
    coef_idx : int
        Index of the coefficient for which to compute SE (default=1 for treatment).

    Returns
    -------
    float
        Cluster-robust standard error for the coefficient.
    """
    # Finite-sample correction: G/(G-1) * (N-1)/(N-k)
    correction = (G / (G - 1.0)) * ((N - 1.0) / (N - k))

    # Build the "meat" of the sandwich
    B = np.zeros((k, k), dtype=np.float64)
    for g in range(G):
        idx = cluster_masks[g]
        Xg = cluster_X[g]
        eg = residuals[idx]
        score_g = Xg.T @ eg  # shape (k,)
        B += np.outer(score_g, score_g)

    B *= correction

    # Sandwich variance
    V = XtX_inv @ B @ XtX_inv
    se = np.sqrt(V[coef_idx, coef_idx])
    return se


def _fast_ols_and_t(
    y_star: np.ndarray,
    precomp: dict,
    coef_idx: int = 1,
) -> tuple[float, float]:
    """Compute OLS coefficient and cluster-robust t-stat for bootstrap y*.

    Parameters
    ----------
    y_star : np.ndarray, shape (N,)
        Bootstrap outcome vector.
    precomp : dict
        Precomputed matrices from _precompute().
    coef_idx : int
        Coefficient index (1 = treatment).

    Returns
    -------
    tuple[float, float]
        (coefficient, t-statistic)
    """
    P = precomp["P"]
    X = precomp["X"]
    XtX_inv = precomp["XtX_inv"]
    cluster_masks = precomp["cluster_masks"]
    cluster_X = precomp["cluster_X"]
    G = precomp["G"]
    N = precomp["N"]
    k = precomp["k"]

    beta_star = P @ y_star
    resid_star = y_star - X @ beta_star

    se = _cluster_robust_se(X, resid_star, XtX_inv, cluster_masks, cluster_X, G, N, k, coef_idx)

    coef = beta_star[coef_idx]
    if se > 0.0 and np.isfinite(se):
        t_stat = coef / se
    else:
        t_stat = np.nan
    return coef, t_stat


def _run_bootstrap_loop(
    weights_all: np.ndarray,
    precomp: dict,
    fitted_base: np.ndarray,
    resid_base: np.ndarray,
    n_reps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the bootstrap loop (possibly chunked for memory).

    For each replicate b:
      1. Map cluster weights to observation-level: w_i = w_{g(i)}
      2. Construct y* = fitted_base + w_i * resid_base
      3. Fit OLS, compute cluster-robust t-stat

    Parameters
    ----------
    weights_all : np.ndarray, shape (n_reps, G)
        Bootstrap weights for all reps.
    precomp : dict
        Precomputed matrices.
    fitted_base : np.ndarray, shape (N,)
        Fitted values under the null/restricted model.
    resid_base : np.ndarray, shape (N,)
        Residuals from the null/restricted model.
    n_reps : int
        Number of replications.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (att_bootstrap, t_stats_bootstrap) each of shape (n_reps,).
    """
    N = precomp["N"]
    obs_cluster_idx = precomp["obs_cluster_idx"]

    att_bootstrap = np.full(n_reps, np.nan, dtype=np.float64)
    t_stats_bootstrap = np.full(n_reps, np.nan, dtype=np.float64)

    # Determine chunking
    total_elements = n_reps * N
    if total_elements > _MEMORY_THRESHOLD:
        # Process in chunks to limit memory usage
        chunk_size = max(1, _MEMORY_THRESHOLD // N)
    else:
        chunk_size = n_reps

    for start in range(0, n_reps, chunk_size):
        end = min(start + chunk_size, n_reps)
        batch_weights = weights_all[start:end]  # shape (batch, G)
        batch_size = end - start

        # Map cluster weights to observations: shape (batch, N)
        obs_weights = batch_weights[:, obs_cluster_idx]

        for i in range(batch_size):
            b = start + i
            y_star = fitted_base + obs_weights[i] * resid_base
            try:
                coef, t_stat = _fast_ols_and_t(y_star, precomp)
                att_bootstrap[b] = coef
                t_stats_bootstrap[b] = t_stat
            except (np.linalg.LinAlgError, ValueError):
                # Leave as NaN
                pass

    return att_bootstrap, t_stats_bootstrap


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def wild_cluster_bootstrap(
    y: np.ndarray,
    treatment: np.ndarray,
    cluster_ids: np.ndarray,
    controls: Optional[np.ndarray] = None,
    n_reps: int = 999,
    weight_type: str = "rademacher",
    ci_level: float = 0.95,
    seed: Optional[int] = None,
    impose_null: bool = True,
    full_enumeration: Optional[bool] = None,
) -> WildClusterBootstrapResult:
    """Perform wild cluster bootstrap inference (Cameron, Gelbach & Miller 2008).

    Provides reliable inference when the number of clusters is small (< 30).
    Constructs a bootstrap distribution of t-statistics by resampling
    cluster-level weights and re-estimating the model.

    Algorithm
    ---------
    1. Estimate original model: y = X beta + e, get residuals e.
    2. (If impose_null) Fit restricted model without treatment: y = alpha + e_r.
    3. For each bootstrap rep b = 1, ..., B:
       a. Generate cluster-level weights w_g from chosen distribution.
       b. Construct bootstrap residuals: e*_i = w_{g(i)} * e_i.
       c. Construct bootstrap outcome: y* = X_restricted @ beta_r + e*.
       d. Fit unrestricted OLS on y*, compute cluster-robust t-stat.
    4. p-value = fraction of |t*_b| >= |t_original|.
    5. CI from quantile of |t*| distribution.

    Parameters
    ----------
    y : np.ndarray, shape (N,)
        Outcome variable.
    treatment : np.ndarray, shape (N,)
        Binary treatment indicator (0/1).
    cluster_ids : np.ndarray, shape (N,)
        Cluster membership for each observation.
    controls : np.ndarray or None, shape (N, p)
        Optional matrix of control variables.
    n_reps : int, default 999
        Number of bootstrap replications. Ignored if full_enumeration is used.
    weight_type : str, default 'rademacher'
        Bootstrap weight distribution: 'rademacher', 'mammen', or 'webb'.
    ci_level : float, default 0.95
        Confidence interval level (e.g. 0.95 for 95% CI).
    seed : int or None, default None
        Random seed for reproducibility.
    impose_null : bool, default True
        Whether to impose H0: treatment_effect = 0 when constructing
        bootstrap outcomes. Recommended for hypothesis testing.
    full_enumeration : bool or None, default None
        Whether to enumerate all 2^G Rademacher weight combinations.
        If None, automatically enabled when G <= 12 and weight_type='rademacher'.

    Returns
    -------
    WildClusterBootstrapResult
        Dataclass containing ATT, bootstrap SE, CI, p-value, and t-stats.

    Raises
    ------
    ValueError
        If inputs have incompatible shapes or invalid weight_type.
    BootstrapConvergenceError
        If all bootstrap replications produce degenerate results.

    Notes
    -----
    - For G <= 12 clusters with Rademacher weights, full enumeration produces
      exact (deterministic) p-values with no Monte Carlo error.
    - Memory chunking is applied automatically when n_reps * N > 50M elements.
    - The treatment coefficient is always at index 1 in the design matrix
      [intercept, treatment, controls...].

    Examples
    --------
    >>> import numpy as np
    >>> from diff_diff.lwdid_wild_bootstrap import wild_cluster_bootstrap
    >>> rng = np.random.default_rng(42)
    >>> n = 200
    >>> y = rng.normal(0, 1, n)
    >>> y[:50] += 1.5
    >>> treatment = np.zeros(n); treatment[:50] = 1.0
    >>> cluster_ids = np.repeat(np.arange(20), 10)
    >>> result = wild_cluster_bootstrap(y, treatment, cluster_ids, seed=123)
    >>> print(f"ATT={result.att:.3f}, p={result.pvalue:.3f}")
    """
    # ----- Input validation -----
    y = np.asarray(y, dtype=np.float64).ravel()
    treatment = np.asarray(treatment, dtype=np.float64).ravel()
    cluster_ids = np.asarray(cluster_ids).ravel()

    N = len(y)
    if N == 0:
        raise ValueError("y must not be empty.")
    if len(treatment) != N:
        raise ValueError(f"Length mismatch: y has {N} obs but treatment has {len(treatment)}.")
    if len(cluster_ids) != N:
        raise ValueError(f"Length mismatch: y has {N} obs but cluster_ids has {len(cluster_ids)}.")
    if not np.all((treatment == 0) | (treatment == 1)):
        raise ValueError(
            "treatment must be binary (0 or 1). "
            f"Got values in [{treatment.min()}, {treatment.max()}]."
        )
    if treatment.sum() == 0:
        raise ValueError("No treated observations (treatment is all zeros).")
    if treatment.sum() == N:
        raise ValueError("No control observations (treatment is all ones).")

    n_clusters = len(np.unique(cluster_ids))
    if n_clusters < 2:
        raise ValueError(f"Need at least 2 clusters for wild cluster bootstrap, got {n_clusters}.")

    if controls is not None:
        controls = np.asarray(controls, dtype=np.float64)
        if controls.ndim == 1:
            controls = controls.reshape(-1, 1)
        if controls.shape[0] != N:
            raise ValueError(f"Controls have {controls.shape[0]} rows but y has {N} obs.")
        if not np.all(np.isfinite(controls)):
            raise ValueError(
                "controls contains non-finite values (NaN or Inf). "
                "Please remove or impute missing values before calling "
                "wild_cluster_bootstrap()."
            )

    # Validate cluster_ids: must not contain NaN (for numeric arrays)
    if np.issubdtype(cluster_ids.dtype, np.floating) and not np.all(np.isfinite(cluster_ids)):
        raise ValueError(
            "cluster_ids contains non-finite values (NaN or Inf). "
            "Cluster identifiers must be valid for all observations."
        )

    if weight_type not in _VALID_WEIGHT_TYPES:
        raise ValueError(
            f"Unknown weight_type '{weight_type}'. " f"Must be one of: {_VALID_WEIGHT_TYPES}"
        )
    if not (0.0 < ci_level < 1.0):
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}.")
    if n_reps < 1:
        raise ValueError(f"n_reps must be >= 1, got {n_reps}.")

    # Handle NaN: drop observations with non-finite y
    finite_mask = np.isfinite(y)
    if not finite_mask.all():
        y = y[finite_mask]
        treatment = treatment[finite_mask]
        cluster_ids = cluster_ids[finite_mask]
        if controls is not None:
            controls = controls[finite_mask]
        N = len(y)
        if N == 0:
            raise ValueError("All observations have non-finite y values.")
        # Revalidate treatment after NaN removal
        n_treated = int(treatment.sum())
        n_control = N - n_treated
        if n_treated == 0:
            raise ValueError("After dropping non-finite y, no treated observations remain.")
        if n_control == 0:
            raise ValueError("After dropping non-finite y, no control observations remain.")
        n_clusters = len(np.unique(cluster_ids))
        if n_clusters < 2:
            raise ValueError(f"After dropping non-finite y, only {n_clusters} cluster(s) remain.")

    # ----- Setup -----
    rng = np.random.default_rng(seed)
    alpha = 1.0 - ci_level

    # Build design matrix
    X = _build_design_matrix(treatment, controls)

    # Precompute
    precomp = _precompute(y, X, cluster_ids)
    G = precomp["G"]
    k = precomp["k"]

    # ----- Original model statistics -----
    beta_hat = precomp["beta_hat"]
    att_original = beta_hat[1]  # treatment coefficient

    se_original = _cluster_robust_se(
        X,
        precomp["residuals"],
        precomp["XtX_inv"],
        precomp["cluster_masks"],
        precomp["cluster_X"],
        G,
        N,
        k,
        coef_idx=1,
    )

    # Handle degenerate case
    if se_original <= 0.0 or not np.isfinite(se_original):
        return WildClusterBootstrapResult(
            att=att_original,
            se_bootstrap=np.nan,
            ci_lower=np.nan,
            ci_upper=np.nan,
            pvalue=np.nan,
            weight_type=weight_type,
            n_reps=0,
            n_clusters=G,
            t_stats=np.array([], dtype=np.float64),
        )

    t_stat_original = att_original / se_original

    # ----- Determine full enumeration -----
    if full_enumeration is None:
        full_enumeration = G <= _FULL_ENUM_THRESHOLD and weight_type == "rademacher"

    # ----- Construct base for y* -----
    if impose_null:
        # Restricted model: y = intercept only (no treatment)
        X_restricted = np.ones((N, 1), dtype=np.float64)
        beta_r = np.linalg.lstsq(X_restricted, y, rcond=None)[0]
        fitted_base = (X_restricted @ beta_r).ravel()
        resid_base = y - fitted_base
    else:
        # Unrestricted model residuals
        fitted_base = (X @ beta_hat).ravel()
        resid_base = precomp["residuals"]

    # ----- Generate weights -----
    if full_enumeration and weight_type == "rademacher":
        weights_all = _generate_all_rademacher(G)
        actual_n_reps = weights_all.shape[0]
    else:
        actual_n_reps = n_reps
        if weight_type == "rademacher":
            weights_all = _rademacher_weights(G, actual_n_reps, rng)
        elif weight_type == "mammen":
            weights_all = _mammen_weights(G, actual_n_reps, rng)
        else:
            weights_all = _webb_weights(G, actual_n_reps, rng)

    # ----- Run bootstrap -----
    att_bootstrap, t_stats_bootstrap = _run_bootstrap_loop(
        weights_all, precomp, fitted_base, resid_base, actual_n_reps
    )

    # ----- Collect valid results -----
    valid_mask = np.isfinite(t_stats_bootstrap)
    t_stats_valid = t_stats_bootstrap[valid_mask]
    att_valid = att_bootstrap[valid_mask]

    if len(t_stats_valid) == 0:
        raise ValueError(
            "All bootstrap replications produced degenerate results (NaN t-stats). "
            "This may indicate a singular design matrix or insufficient variation."
        )

    # ----- Compute p-value -----
    # Two-sided: p = P(|t*| >= |t_orig|)
    pvalue = float(np.mean(np.abs(t_stats_valid) >= np.abs(t_stat_original)))

    # ----- Bootstrap SE -----
    se_bootstrap = float(np.std(att_valid, ddof=0))

    # ----- Confidence interval -----
    if impose_null:
        # Symmetric CI based on (1-alpha) quantile of |t*|
        t_abs_crit = np.percentile(np.abs(t_stats_valid), 100.0 * (1.0 - alpha))
        ci_lower = att_original - t_abs_crit * se_original
        ci_upper = att_original + t_abs_crit * se_original
    else:
        # Percentile CI from bootstrap ATT distribution
        ci_lower = float(np.percentile(att_valid, 100.0 * alpha / 2.0))
        ci_upper = float(np.percentile(att_valid, 100.0 * (1.0 - alpha / 2.0)))

    return WildClusterBootstrapResult(
        att=float(att_original),
        se_bootstrap=se_bootstrap,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        pvalue=pvalue,
        weight_type=weight_type,
        n_reps=actual_n_reps,
        n_clusters=G,
        t_stats=t_stats_bootstrap,
    )
