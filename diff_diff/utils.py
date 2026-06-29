"""
Utility functions for difference-in-differences estimation.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Import Rust backend if available (from _backend to avoid circular imports)
from diff_diff._backend import (
    HAS_RUST_BACKEND,
    _rust_project_simplex,
    _rust_sdid_unit_weights,
    _rust_compute_time_weights,
    _rust_compute_noise_level,
    _rust_sc_weight_fw,
    _rust_sc_weight_fw_with_convergence,
    _rust_sc_weight_fw_weighted,
    _rust_sc_weight_fw_weighted_with_convergence,
)
from diff_diff.linalg import compute_robust_vcov as _compute_robust_vcov_linalg
from diff_diff.linalg import solve_ols as _solve_ols_linalg

# Numerical constants for optimization algorithms
_OPTIMIZATION_MAX_ITER = 1000  # Maximum iterations for weight optimization
_OPTIMIZATION_TOL = 1e-8  # Convergence tolerance for optimization
_NUMERICAL_EPS = 1e-10  # Small constant to prevent division by zero

# Cache for critical values to avoid repeated scipy calls
_critical_value_cache: Dict[Tuple[float, Optional[int]], float] = {}


def _get_critical_value(alpha: float, df: Optional[int] = None) -> float:
    """Return cached critical value for (alpha, df) pair."""
    key = (alpha, df)
    if key not in _critical_value_cache:
        if df is not None:
            _critical_value_cache[key] = float(stats.t.ppf(1 - alpha / 2, df))
        else:
            _critical_value_cache[key] = float(stats.norm.ppf(1 - alpha / 2))
    return _critical_value_cache[key]


def validate_binary(arr: np.ndarray, name: str) -> None:
    """
    Validate that an array contains only binary values (0 or 1).

    Parameters
    ----------
    arr : np.ndarray
        Array to validate.
    name : str
        Name of the variable (for error messages).

    Raises
    ------
    ValueError
        If array contains non-binary values.
    """
    unique_values = np.unique(arr[~np.isnan(arr)])
    if not np.all(np.isin(unique_values, [0, 1])):
        raise ValueError(f"{name} must be binary (0 or 1). " f"Found values: {unique_values}")


def validate_covariate_names(
    covariates: Optional[List[str]],
    reserved_names: Iterable[str],
    *,
    estimator: str = "estimator",
) -> None:
    """
    Validate that covariate column names do not collide with reserved
    structural term names (and are not duplicated within ``covariates``).

    Fitted coefficients are stored in a ``name -> value`` dict built by zipping
    a variable-name list -- structural term names PLUS the user covariate column
    names appended verbatim -- with the coefficient vector. A covariate whose
    name equals a reserved structural name (the intercept ``const``, the
    treatment/time indicators, the interaction term, period dummies,
    fixed-effect dummies, or an internal working column) would silently
    overwrite the structural coefficient (Python dict last-write-wins),
    corrupting the result with no error. Duplicate names within ``covariates``
    collapse to a single dict entry the same way.

    The comparison is case-sensitive: column names and dict keys are
    case-sensitive, so e.g. ``Const`` does not actually collide with ``const``
    and is allowed.

    Parameters
    ----------
    covariates : list of str or None
        User-supplied covariate column names. ``None`` or empty is a no-op.
    reserved_names : iterable of str
        Reserved structural term names this estimator builds (estimator-specific).
    estimator : str
        Estimator name, used in the error message.

    Raises
    ------
    ValueError
        If a covariate name collides with a reserved structural name, or if
        ``covariates`` contains duplicate names.
    """
    if not covariates:
        return
    reserved = set(reserved_names)
    collisions = sorted({c for c in covariates if c in reserved})
    if collisions:
        raise ValueError(
            f"{estimator}: covariate name(s) {collisions} collide with reserved "
            f"structural term name(s). These names are used internally for the "
            f"intercept, the treatment/time indicators, the interaction term, "
            f"period dummies, fixed-effect dummies, or internal working columns, "
            f"and a colliding covariate would silently overwrite the structural "
            f"coefficient. Rename the covariate column(s). Reserved names for "
            f"this fit: {sorted(reserved)}."
        )
    seen: set = set()
    duplicates = []
    for c in covariates:
        if c in seen:
            duplicates.append(c)
        seen.add(c)
    if duplicates:
        raise ValueError(
            f"{estimator}: duplicate covariate name(s) {sorted(set(duplicates))} "
            f"in `covariates`. Each covariate maps to one coefficient; duplicates "
            f"collapse to a single entry. Remove the duplicate(s)."
        )


def validate_design_term_names(
    var_names: Iterable[str],
    *,
    estimator: str = "estimator",
) -> None:
    """
    Raise if the assembled design term-name list contains duplicates.

    Backstop for :func:`validate_covariate_names`: even after the user
    covariates are cleared, a fixed-effect dummy name (``{fe}_{value}``) can
    still collide with a structural term — most notably a ``MultiPeriodDiD``
    ``period_{p}`` event-study key when a non-time fixed effect produces matching
    dummy names — or with another dummy. Such a duplicate would silently
    overwrite a coefficient when ``var_names`` is zipped into the result's
    ``coefficients`` dict (Python dict last-write-wins). This checks the FINAL
    name list (structural terms + covariates + fixed-effect dummies) right
    before the dict is built, catching collisions that depend on the data and so
    cannot be known up front.

    Parameters
    ----------
    var_names : iterable of str
        The fully assembled design-matrix column-name list.
    estimator : str
        Estimator name, used in the error message.

    Raises
    ------
    ValueError
        If any name appears more than once.
    """
    seen: set = set()
    duplicates = []
    for name in var_names:
        if name in seen:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise ValueError(
            f"{estimator}: the fitted design has duplicate term name(s) "
            f"{sorted(set(duplicates))} — a covariate or fixed-effect dummy name "
            f"collides with a structural term (intercept, treatment/time "
            f"indicators, the interaction, or period dummies) or with another "
            f"column. This would silently overwrite a coefficient in the result. "
            f"Rename the offending fixed-effect category or covariate column."
        )


def fe_dummy_names(col: pd.Series, prefix: str) -> List[str]:
    """
    Reserved fixed-effect dummy column names for the collision guard, matching
    ``pd.get_dummies(col, prefix=prefix, drop_first=True).columns`` WITHOUT
    materializing the dense ``(n x G)`` dummy matrix.

    The within-transform ``TwoWayFixedEffects`` path is specifically designed to
    avoid expanding high-cardinality fixed-effect dummies (that is its scaling
    contract), so the collision guard must reserve those names without building
    the dummy block. ``pd.get_dummies`` orders categories via
    ``pd.Categorical(col).categories`` — sorted unique values for a plain column,
    the declared category order for a ``Categorical`` — then ``drop_first=True``
    drops the first. This derivation reproduces that exactly (including
    ``Categorical`` columns with a non-default category order) at ``O(G)`` memory.

    Parameters
    ----------
    col : pandas.Series
        The fixed-effect / unit / time column.
    prefix : str
        Dummy-name prefix (the project uses ``fe`` for ``fixed_effects`` and
        ``_fe_{unit}`` / ``_fe_{time}`` for TWFE unit/time dummies).

    Returns
    -------
    list of str
        The kept (post ``drop_first``) dummy column names.
    """
    if isinstance(col.dtype, pd.CategoricalDtype):
        cats = list(col.cat.categories)
    else:
        cats = list(pd.Categorical(col).categories)
    return [f"{prefix}_{c}" for c in cats[1:]]


def warn_if_not_converged(
    converged: bool,
    method_name: str,
    max_iter: int,
    tol: Optional[float] = None,
    stacklevel: int = 3,
) -> None:
    """Emit a UserWarning when an iterative solver exhausts max_iter without converging.

    Shared helper for axis-B silent-failure fixes (iterative loops that otherwise
    return the current iterate without signaling non-convergence).
    """
    if converged:
        return
    tol_suffix = f" (tol={tol})" if tol is not None else ""
    warnings.warn(
        f"{method_name} did not converge in {max_iter} iterations{tol_suffix}. "
        "Results may be inaccurate.",
        UserWarning,
        stacklevel=stacklevel,
    )


def compute_robust_se(
    X: np.ndarray, residuals: np.ndarray, cluster_ids: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute heteroskedasticity-robust (HC1) or cluster-robust standard errors.

    This function is a thin wrapper around the optimized implementation in
    diff_diff.linalg for backwards compatibility.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n, k).
    residuals : np.ndarray
        Residuals from regression of shape (n,).
    cluster_ids : np.ndarray, optional
        Cluster identifiers for cluster-robust SEs.

    Returns
    -------
    np.ndarray
        Variance-covariance matrix of shape (k, k).
    """
    return _compute_robust_vcov_linalg(X, residuals, cluster_ids)


def compute_confidence_interval(
    estimate: float, se: float, alpha: float = 0.05, df: Optional[int] = None
) -> Tuple[float, float]:
    """
    Compute confidence interval for an estimate.

    Parameters
    ----------
    estimate : float
        Point estimate.
    se : float
        Standard error.
    alpha : float
        Significance level (default 0.05 for 95% CI).
    df : int, optional
        Degrees of freedom. If None, uses normal distribution.

    Returns
    -------
    tuple
        (lower_bound, upper_bound) of confidence interval.
    """
    critical_value = _get_critical_value(alpha, df)
    lower = estimate - critical_value * se
    upper = estimate + critical_value * se

    return (lower, upper)


def compute_p_value(t_stat: float, df: Optional[int] = None, two_sided: bool = True) -> float:
    """
    Compute p-value for a t-statistic.

    Parameters
    ----------
    t_stat : float
        T-statistic.
    df : int, optional
        Degrees of freedom. If None, uses normal distribution.
    two_sided : bool
        Whether to compute two-sided p-value (default True).

    Returns
    -------
    float
        P-value.
    """
    if df is not None:
        p_value = stats.t.sf(np.abs(t_stat), df)
    else:
        p_value = stats.norm.sf(np.abs(t_stat))

    if two_sided:
        p_value *= 2

    return float(p_value)


def safe_inference(effect, se, alpha=0.05, df=None):
    """Compute t_stat, p_value, conf_int with NaN-safe gating.

    When SE is non-finite, zero, or negative, ALL inference fields
    are set to NaN to prevent misleading statistical output.

    Accepts scalar inputs only (not numpy arrays). All existing inference
    call sites operate on scalars within loops.

    Parameters
    ----------
    effect : float
        Point estimate (treatment effect or coefficient).
    se : float
        Standard error of the estimate.
    alpha : float, optional
        Significance level for confidence interval (default 0.05).
    df : int, optional
        Degrees of freedom. If None, uses normal distribution.

    Returns
    -------
    tuple
        (t_stat, p_value, (ci_lower, ci_upper)). All NaN when SE is
        non-finite, zero, or negative.
    """
    if not (np.isfinite(se) and se > 0):
        return np.nan, np.nan, (np.nan, np.nan)
    if df is not None and df <= 0:
        # Undefined degrees of freedom (e.g., rank-deficient replicate design)
        return np.nan, np.nan, (np.nan, np.nan)
    t_stat = effect / se
    p_value = compute_p_value(t_stat, df=df)
    conf_int = compute_confidence_interval(effect, se, alpha, df=df)
    return t_stat, p_value, conf_int


def safe_inference_batch(effects, ses, alpha=0.05, df=None):
    """Vectorized batch inference for arrays of effects and SEs.

    Parameters
    ----------
    effects : np.ndarray
        Array of point estimates.
    ses : np.ndarray
        Array of standard errors.
    alpha : float, optional
        Significance level (default 0.05).
    df : int, optional
        Degrees of freedom. If None, uses normal distribution.

    Returns
    -------
    t_stats : np.ndarray
    p_values : np.ndarray
    ci_lowers : np.ndarray
    ci_uppers : np.ndarray
    """
    effects = np.asarray(effects, dtype=float)
    ses = np.asarray(ses, dtype=float)
    n = len(effects)

    t_stats = np.full(n, np.nan)
    p_values = np.full(n, np.nan)
    ci_lowers = np.full(n, np.nan)
    ci_uppers = np.full(n, np.nan)

    # Undefined df (e.g., rank-deficient replicate design) → all NaN
    if df is not None and df <= 0:
        return t_stats, p_values, ci_lowers, ci_uppers

    valid = np.isfinite(ses) & (ses > 0)
    if not np.any(valid):
        return t_stats, p_values, ci_lowers, ci_uppers

    t_stats[valid] = effects[valid] / ses[valid]

    if df is not None:
        p_values[valid] = 2.0 * stats.t.sf(np.abs(t_stats[valid]), df)
    else:
        p_values[valid] = 2.0 * stats.norm.sf(np.abs(t_stats[valid]))

    crit = _get_critical_value(alpha, df)
    ci_lowers[valid] = effects[valid] - crit * ses[valid]
    ci_uppers[valid] = effects[valid] + crit * ses[valid]

    return t_stats, p_values, ci_lowers, ci_uppers


# =============================================================================
# Wild Cluster Bootstrap
# =============================================================================


@dataclass
class WildBootstrapResults:
    """
    Results from wild cluster bootstrap inference.

    Attributes
    ----------
    se : float
        Analytical cluster-robust (CR1) standard error of the coefficient. The
        wild bootstrap studentizes the test with this SE; it is not a rescaled
        bootstrap dispersion.
    p_value : float
        Wild cluster bootstrap p-value (two-tailed or equal-tailed).
    t_stat_original : float
        Studentized statistic of the original estimate, ``(coef - null) / se``.
    ci_lower : float
        Lower bound of the confidence interval (by test inversion).
    ci_upper : float
        Upper bound of the confidence interval (by test inversion).
    n_clusters : int
        Number of clusters in the data.
    n_bootstrap : int
        Number of bootstrap replications.
    weight_type : str
        Type of bootstrap weights used ("rademacher", "webb", or "mammen").
    alpha : float
        Significance level used for confidence interval.
    p_val_type : str
        Test shape used ("two-tailed" or "equal-tailed").
    bootstrap_distribution : np.ndarray, optional
        Bootstrap distribution of the studentized statistic ``t*`` evaluated at
        the null (if requested).

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008).
    Bootstrap-Based Improvements for Inference with Clustered Errors.
    The Review of Economics and Statistics, 90(3), 414-427.
    """

    se: float
    p_value: float
    t_stat_original: float
    ci_lower: float
    ci_upper: float
    n_clusters: int
    n_bootstrap: int
    weight_type: str
    alpha: float = 0.05
    p_val_type: str = "two-tailed"
    bootstrap_distribution: Optional[np.ndarray] = field(default=None, repr=False)

    def summary(self) -> str:
        """Generate formatted summary of bootstrap results."""
        lines = [
            "Wild Cluster Bootstrap Results",
            "=" * 40,
            f"Cluster-robust SE:   {self.se:.6f}",
            f"Bootstrap p-value:   {self.p_value:.4f}",
            f"Studentized t-stat:  {self.t_stat_original:.4f}",
            f"CI ({int((1-self.alpha)*100)}%):           [{self.ci_lower:.6f}, {self.ci_upper:.6f}]",
            f"Number of clusters:  {self.n_clusters}",
            f"Bootstrap reps:      {self.n_bootstrap}",
            f"Weight type:         {self.weight_type}",
            f"Test type:           {self.p_val_type}",
        ]
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print formatted summary to stdout."""
        print(self.summary())


def _generate_rademacher_weights(n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate Rademacher weights: +1 or -1 with probability 0.5.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Array of Rademacher weights.
    """
    return np.asarray(rng.choice([-1.0, 1.0], size=n_clusters))


def _generate_webb_weights(n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate Webb's 6-point distribution weights.

    Values: {-sqrt(3/2), -sqrt(2/2), -sqrt(1/2), sqrt(1/2), sqrt(2/2), sqrt(3/2)}
    with equal probabilities (1/6 each), giving E[w]=0 and Var(w)=1.0.

    This distribution is recommended for very few clusters (G < 10) as it
    provides better finite-sample properties than Rademacher weights.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Array of Webb weights.

    References
    ----------
    Webb, M. D. (2014). Reworking wild bootstrap based inference for
    clustered errors. Queen's Economics Department Working Paper No. 1315.

    Note: Uses equal probabilities (1/6 each) matching R's `did` package,
    which gives unit variance for consistency with other weight distributions.
    """
    values = np.array(
        [
            -np.sqrt(3 / 2),
            -np.sqrt(2 / 2),
            -np.sqrt(1 / 2),
            np.sqrt(1 / 2),
            np.sqrt(2 / 2),
            np.sqrt(3 / 2),
        ]
    )
    # Equal probabilities (1/6 each) matching R's did package, giving Var(w) = 1.0
    return np.asarray(rng.choice(values, size=n_clusters))


def _generate_mammen_weights(n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate Mammen's two-point distribution weights.

    Values: {-(sqrt(5)-1)/2, (sqrt(5)+1)/2}
    with probabilities {(sqrt(5)+1)/(2*sqrt(5)), (sqrt(5)-1)/(2*sqrt(5))}.

    This distribution satisfies E[v]=0, E[v^2]=1, E[v^3]=1, which provides
    asymptotic refinement for skewed error distributions.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Array of Mammen weights.

    References
    ----------
    Mammen, E. (1993). Bootstrap and Wild Bootstrap for High Dimensional
    Linear Models. The Annals of Statistics, 21(1), 255-285.
    """
    sqrt5 = np.sqrt(5)
    # Values from Mammen (1993)
    val1 = -(sqrt5 - 1) / 2  # approximately -0.618
    val2 = (sqrt5 + 1) / 2  # approximately 1.618 (golden ratio)

    # Probability of val1
    p1 = (sqrt5 + 1) / (2 * sqrt5)  # approximately 0.724

    return np.asarray(rng.choice([val1, val2], size=n_clusters, p=[p1, 1 - p1]))


def _wild_weight_matrix(
    n_clusters: int,
    n_bootstrap: int,
    weight_type: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build the ``(B, n_clusters)`` matrix of cluster-level bootstrap weights.

    For Rademacher weights with few clusters all ``2**n_clusters`` sign-vectors
    are enumerated (deterministic) once ``n_bootstrap`` reaches the number of
    possible draws — i.e. when ``2**n_clusters <= n_bootstrap`` (and
    ``n_clusters <= 20``, a guard against pathological memory use). This matches
    the full-enumeration switch of ``fwildclusterboot::boottest`` (verified: for
    ``G=10`` boottest samples at ``B=1023`` and enumerates at ``B=1024``); the
    reported ``n_bootstrap`` is then ``2**n_clusters``. (Only ``2**(n_clusters-1)``
    of those draws have distinct ``|t*|`` — each draw and its all-signs-flipped
    mirror share ``|t*|`` — but the full set is materialized.) Enumeration removes
    RNG dependence in the few-cluster regime where the wild bootstrap matters
    most. Otherwise ``n_bootstrap`` weight vectors are sampled. Webb/Mammen always
    sample: the sign-flip enumeration symmetry is Rademacher-specific (Mammen is
    asymmetric, Webb is a 6-point law).
    """
    if weight_type == "rademacher" and n_clusters <= 20 and 2**n_clusters <= n_bootstrap:
        n_enum = 2**n_clusters
        bits = (np.arange(n_enum)[:, None] >> np.arange(n_clusters)) & 1
        return np.where(bits == 1, 1.0, -1.0)

    generators = {
        "rademacher": _generate_rademacher_weights,
        "webb": _generate_webb_weights,
        "mammen": _generate_mammen_weights,
    }
    generate = generators[weight_type]
    weights = np.empty((n_bootstrap, n_clusters))
    for b in range(n_bootstrap):
        weights[b] = generate(n_clusters, rng)
    return weights


def wild_bootstrap_se(
    X: np.ndarray,
    y: np.ndarray,
    residuals: np.ndarray,
    cluster_ids: np.ndarray,
    coefficient_index: int,
    n_bootstrap: int = 999,
    weight_type: str = "rademacher",
    null_hypothesis: float = 0.0,
    alpha: float = 0.05,
    seed: Optional[int] = None,
    return_distribution: bool = False,
    p_val_type: str = "two-tailed",
) -> WildBootstrapResults:
    """
    Compute wild cluster bootstrap standard errors and p-values.

    Implements the Wild Cluster Restricted (WCR) bootstrap of Cameron, Gelbach,
    and Miller (2008), matching the defaults of R's ``fwildclusterboot::boottest``
    (Roodman, MacKinnon, Nielsen & Webb 2019): the null ``H0: coefficient =
    null_hypothesis`` is genuinely imposed by re-estimating the model with the
    coefficient's column dropped, the bootstrap DGP resamples the *restricted*
    residuals, and the confidence interval is obtained by **inverting the
    bootstrap test** (the set of null values not rejected at level ``alpha``) so
    that the p-value and CI are mutually consistent (``0 in CI`` iff
    ``p >= alpha``). For Rademacher weights with few clusters all
    ``2**n_clusters`` sign-vectors are enumerated (deterministic) when
    ``2**n_clusters <= n_bootstrap`` (the ``boottest`` full-enumeration trigger —
    it switches to enumeration once ``n_bootstrap`` reaches the number of
    possible draws) and ``n_clusters <= 20`` (a memory guard); the reported
    ``n_bootstrap`` is then ``2**n_clusters``. Otherwise signs are sampled.

    The reported ``se`` is the analytical cluster-robust (CR1) standard error of
    the original estimate — the studentized bootstrap drives the p-value and CI,
    not a re-scaled bootstrap dispersion.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n, k).
    y : np.ndarray
        Outcome vector of shape (n,).
    residuals : np.ndarray
        Retained for backward compatibility and IGNORED by the WCR
        implementation, which recomputes the original fit and the restricted
        (null-imposed) residualization internally from ``X`` and ``y``.
    cluster_ids : np.ndarray
        Cluster identifiers of shape (n,).
    coefficient_index : int
        Index of the coefficient for which to compute bootstrap inference.
        For DiD, this is typically 3 (the treatment*post interaction term).
    n_bootstrap : int, default=999
        Number of bootstrap replications. Odd numbers are recommended for
        exact p-value computation.
    weight_type : str, default="rademacher"
        Type of bootstrap weights:
        - "rademacher": +1 or -1 with equal probability (standard choice)
        - "webb": 6-point distribution (recommended for <10 clusters)
        - "mammen": Two-point distribution with skewness correction
    null_hypothesis : float, default=0.0
        Value of the null hypothesis for p-value computation.
    alpha : float, default=0.05
        Significance level for confidence interval.
    seed : int, optional
        Random seed for reproducibility. If None (default), results
        will vary between runs.
    return_distribution : bool, default=False
        If True, include the bootstrap distribution of the studentized statistic
        ``t*`` (evaluated at the null) in the results.
    p_val_type : str, default="two-tailed"
        Shape of the test (mirrors ``boottest``'s ``p_val_type``):

        - "two-tailed": test on ``|t*|``; two-tailed CI by inversion (the
          interval need not be symmetric about the estimate).
        - "equal-tailed": each tail tested at ``alpha/2``; equal-tailed CI.

    Returns
    -------
    WildBootstrapResults
        Dataclass containing bootstrap SE, p-value, confidence interval,
        and other inference results.

    Raises
    ------
    ValueError
        If weight_type is not recognized or if there are fewer than 2 clusters.

    Warns
    -----
    UserWarning
        If the number of clusters is less than 5, as bootstrap inference
        may be unreliable.

    Examples
    --------
    >>> from diff_diff.utils import wild_bootstrap_se
    >>> results = wild_bootstrap_se(
    ...     X, y, residuals, cluster_ids,
    ...     coefficient_index=3,  # ATT coefficient
    ...     n_bootstrap=999,
    ...     weight_type="rademacher",
    ...     seed=42
    ... )
    >>> print(f"Bootstrap SE: {results.se:.4f}")
    >>> print(f"Bootstrap p-value: {results.p_value:.4f}")

    References
    ----------
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008).
    Bootstrap-Based Improvements for Inference with Clustered Errors.
    The Review of Economics and Statistics, 90(3), 414-427.

    MacKinnon, J. G., & Webb, M. D. (2018). The wild bootstrap for
    few (treated) clusters. The Econometrics Journal, 21(2), 114-135.
    """
    # Validate inputs
    valid_weight_types = ["rademacher", "webb", "mammen"]
    if weight_type not in valid_weight_types:
        raise ValueError(f"weight_type must be one of {valid_weight_types}, got '{weight_type}'")
    valid_p_val_types = ["two-tailed", "equal-tailed"]
    if p_val_type not in valid_p_val_types:
        raise ValueError(f"p_val_type must be one of {valid_p_val_types}, got '{p_val_type}'")

    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    if n_clusters < 2:
        raise ValueError(f"Wild cluster bootstrap requires at least 2 clusters, got {n_clusters}")

    if n_clusters < 5:
        warnings.warn(
            f"Only {n_clusters} clusters detected. Wild cluster bootstrap inference may be "
            "unreliable with fewer than 5 clusters. With Rademacher weights all "
            f"{2 ** n_clusters} sign-vectors are enumerated exactly when "
            f"n_bootstrap >= 2**n_clusters = {2 ** n_clusters}; Webb weights "
            "(weight_type='webb') improve finite-sample behaviour but are sampled, not "
            "enumerated.",
            UserWarning,
        )

    rng = np.random.default_rng(seed)
    n = X.shape[0]

    def _degenerate() -> WildBootstrapResults:
        # All-or-nothing NaN contract (feedback_bootstrap_nan_on_invalid_contract):
        # when the original fit or the bootstrap is degenerate, NaN the entire
        # (se, t_stat, p_value, ci) inference family together rather than mixing
        # analytical and bootstrap quantities on the same coefficient.
        return WildBootstrapResults(
            se=float("nan"),
            p_value=float("nan"),
            t_stat_original=float("nan"),
            ci_lower=float("nan"),
            ci_upper=float("nan"),
            n_clusters=n_clusters,
            n_bootstrap=n_bootstrap,
            weight_type=weight_type,
            alpha=alpha,
            p_val_type=p_val_type,
            bootstrap_distribution=None,
        )

    # Step 1: original fit. Establishes the analytical cluster-robust (CR1) SE
    # that studentizes the test, and the set of identified (kept) columns so the
    # bootstrap stays rank-robust (e.g. an always-treated unit dummy collinear
    # with treated*post on the full-dummy TWFE path: solve_ols drops the nuisance
    # column and reports it as NaN, while the identified ATT is retained).
    # First fit WITHOUT the cluster-robust vcov: this identifies the kept
    # (full-rank) columns and lets us reject a saturated design *before*
    # requesting the cluster sandwich. The shared CR1 small-sample adjustment
    # (n_eff-1)/(n_eff-k) divides by zero on a saturated design (n == rank), so
    # routing the degenerate case here keeps the all-or-nothing NaN contract.
    beta_hat, _, _ = _solve_ols_linalg(X, y, return_vcov=False)
    original_coef = float(beta_hat[coefficient_index])
    if not np.isfinite(original_coef):
        return _degenerate()
    kept = np.isfinite(beta_hat)
    if not bool(kept[coefficient_index]):
        return _degenerate()
    X_eff = X[:, kept]
    j_eff = int(np.sum(kept[:coefficient_index]))  # position of the coef among kept columns
    k_eff = X_eff.shape[1]
    if n <= k_eff:  # no residual degrees of freedom -> CR1 undefined
        return _degenerate()

    # Now the cluster-robust (CR1) vcov is well-defined; it studentizes the test.
    _, _, vcov_original = _solve_ols_linalg(X, y, cluster_ids=cluster_ids, return_vcov=True)
    if vcov_original is None:
        return _degenerate()
    se_a = float(np.sqrt(vcov_original[coefficient_index, coefficient_index]))
    if not np.isfinite(se_a) or se_a <= 0:
        return _degenerate()

    # Projections on the (full-rank) effective design.
    XtX_inv = np.linalg.inv(X_eff.T @ X_eff)
    a_vec = X_eff @ XtX_inv[:, j_eff]  # influence of each obs on beta_j: beta*_j = a_vec . y*
    proj = XtX_inv @ X_eff.T  # (k_eff, n) OLS projection onto coefficients

    # Restricted residualization imposing H0: regress y and x_j on X_eff \ {col j}.
    # The restricted residuals u(r) = M_{-j} y - r * M_{-j} x_j are linear in the
    # candidate null r, so the whole test can be re-evaluated at any r cheaply.
    xj = X_eff[:, j_eff]
    X_reduced = np.delete(X_eff, j_eff, axis=1)
    if X_reduced.shape[1] == 0:
        # Single-regressor design: the reduced model has no regressors, so the
        # restricted fit is identically 0 and the residuals are the variables
        # themselves (solve_ols cannot fit a zero-column design).
        fit_y_red = np.zeros(n)
        fit_xj_red = np.zeros(n)
    else:
        _, _, fit_y_red, _ = _solve_ols_linalg(X_reduced, y, return_vcov=False, return_fitted=True)
        _, _, fit_xj_red, _ = _solve_ols_linalg(
            X_reduced, xj, return_vcov=False, return_fitted=True
        )
    m_y = y - fit_y_red
    m_xj = xj - fit_xj_red

    # CR1 small-sample correction. NOTE: this constant cancels in |t*| vs |t0|
    # (it scales se* and se_a identically), so it affects only the reported SE,
    # not the p-value or CI. Kept for fidelity with the analytical CR1 SE.
    corr = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k_eff))

    # Cluster membership: indicator matrix C (G, n) for fast per-cluster score sums.
    cluster_pos = {c: i for i, c in enumerate(unique_clusters)}
    cl_idx = np.array([cluster_pos[c] for c in cluster_ids])
    cluster_indicator = np.zeros((n_clusters, n))
    cluster_indicator[cl_idx, np.arange(n)] = 1.0

    # Fixed bootstrap weights, held constant across the whole test inversion so
    # that p(r) is a stable (monotone, step) function amenable to root-finding.
    weights = _wild_weight_matrix(n_clusters, n_bootstrap, weight_type, rng)
    n_boot_eff = int(weights.shape[0])
    weights_obs = weights[:, cl_idx]  # (B, n)

    def _t_star(r: float) -> np.ndarray:
        """Studentized bootstrap statistics t*(r) under H0: beta_j = r."""
        u_r = m_y - r * m_xj  # restricted residuals at r (n,)
        # WCR DGP: y* = fitted_restricted + u_r * w = (y - u_r) + u_r * w_obs.
        y_star = y[None, :] - u_r[None, :] * (1.0 - weights_obs)  # (B, n)
        beta_j_star = y_star @ a_vec  # (B,)
        coef_full = proj @ y_star.T  # (k_eff, B)
        resid_star = y_star.T - X_eff @ coef_full  # (n, B) bootstrap residuals
        scores = cluster_indicator @ (a_vec[:, None] * resid_star)  # (G, B) per-cluster scores
        se_star = np.sqrt(corr * np.sum(scores**2, axis=0))  # (B,)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (beta_j_star - r) / se_star
        t[~(se_star > 0)] = np.nan
        return t

    t_star = _t_star(null_hypothesis)
    finite = np.isfinite(t_star)
    n_valid = int(finite.sum())
    if n_valid < 2:
        return _degenerate()
    t_star_valid = t_star[finite]
    t0 = (original_coef - null_hypothesis) / se_a

    # Strict-inequality tail counts, matching fwildclusterboot/boottest: a
    # bootstrap statistic is counted only if it *exceeds* the observed one. In
    # the fully-enumerated few-cluster case the all-(+1) / all-(-1) sign-vectors
    # reproduce t* = +/- t0 exactly (the observed draw and its mirror); strict
    # ">" excludes those boundary ties, as boottest does. The small relative
    # guard (~1e-9) makes the exclusion robust to floating-point noise from the
    # fast-form path so a true tie never sneaks in as a strict exceedance.
    def _frac_gt(vals: np.ndarray, thresh: float) -> float:
        return float(np.mean(vals > thresh + 1e-9 * max(1.0, abs(thresh))))

    def _frac_lt(vals: np.ndarray, thresh: float) -> float:
        return float(np.mean(vals < thresh - 1e-9 * max(1.0, abs(thresh))))

    # p-value at the test null (two-tailed on |t*|, or equal-tailed).
    if p_val_type == "two-tailed":
        raw_p = _frac_gt(np.abs(t_star_valid), abs(t0))
    else:
        p_low = _frac_lt(t_star_valid, t0)
        p_up = _frac_gt(t_star_valid, t0)
        raw_p = 2.0 * min(p_low, p_up)
    # Floor the reported p-value to avoid an exact zero (a documented departure
    # from boottest, which can report p == 0) — but NEVER let the floor reach
    # the significance level. With very few valid draws 1/(n_valid+1) can exceed
    # alpha, and flooring there would flip a bootstrap-significant result (0
    # outside the inverted CI) to "non-significant", re-creating the very
    # p-vs-CI contradiction this estimator fixes. When the floor would cross
    # alpha we report the raw p-value (which is < alpha in exactly those cases),
    # so the significance verdict always agrees with the inverted CI.
    floor = 1.0 / (n_valid + 1)
    p_value = max(raw_p, floor) if floor < alpha else raw_p
    p_value = float(min(1.0, p_value))

    # ---- Confidence interval by test inversion ------------------------------
    # The CI is the set of nulls r not rejected at level alpha. The relevant
    # rejection frequency is monotonically decreasing as r moves away from the
    # point estimate, so each endpoint is found by outward bracketing + plain
    # bisection — robust to the step-function nature of a finite bootstrap
    # (unlike brentq, which assumes a continuous sign change).
    def _reject_two_tailed(r: float) -> float:
        t = _t_star(r)
        t = t[np.isfinite(t)]
        if t.size < 2:
            return 0.0
        t0_r = (original_coef - r) / se_a
        return _frac_gt(np.abs(t), abs(t0_r))

    def _tail_freq(r: float, upper: bool) -> float:
        t = _t_star(r)
        t = t[np.isfinite(t)]
        if t.size < 2:
            return 0.0
        t0_r = (original_coef - r) / se_a
        return _frac_gt(t, t0_r) if upper else _frac_lt(t, t0_r)

    def _bisect(f: Any, level: float, direction: int) -> float:
        # f(center) >= level; search outward (direction -1 lower, +1 upper) for
        # the crossing f(r) = level (f decreasing in |r - center|), then bisect.
        center = original_coef
        scale = se_a if se_a > 0 else 1.0
        step = scale
        hi = center + direction * step
        bracketed = False
        for _ in range(64):
            if f(hi) < level:
                bracketed = True
                break
            step *= 2.0
            hi = center + direction * step
        if not bracketed:
            # The test never rejects arbitrarily far out: the inverted CI is
            # genuinely unbounded on this side. Represent it with a signed
            # infinity (NOT NaN) so the (se, t, p, CI) inference family stays
            # internally consistent — 0 still lies inside an unbounded interval
            # exactly when the test fails to reject it, preserving
            # 0 ∈ CI ⟺ p ≥ alpha.
            return float(direction) * np.inf
        lo = center  # f(lo) >= level, f(hi) < level
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if f(mid) >= level:
                lo = mid
            else:
                hi = mid
            if abs(hi - lo) <= 1e-10 * max(1.0, abs(center)):
                break
        return 0.5 * (lo + hi)

    if p_val_type == "two-tailed":
        ci_lower = _bisect(_reject_two_tailed, alpha, -1)
        ci_upper = _bisect(_reject_two_tailed, alpha, +1)
    else:
        # equal-tailed: lower endpoint where the upper-tail frequency hits
        # alpha/2; upper endpoint where the lower-tail frequency hits alpha/2.
        ci_lower = _bisect(lambda r: _tail_freq(r, True), alpha / 2.0, -1)
        ci_upper = _bisect(lambda r: _tail_freq(r, False), alpha / 2.0, +1)

    return WildBootstrapResults(
        se=se_a,
        p_value=p_value,
        t_stat_original=t0,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        n_clusters=n_clusters,
        n_bootstrap=n_boot_eff,
        weight_type=weight_type,
        alpha=alpha,
        p_val_type=p_val_type,
        bootstrap_distribution=t_star_valid if return_distribution else None,
    )


def check_parallel_trends(
    data: pd.DataFrame,
    outcome: str,
    time: str,
    treatment_group: str,
    pre_periods: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    Perform a simple check for parallel trends assumption.

    This computes the trend (slope) in the outcome variable for both
    treatment and control groups during pre-treatment periods.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    outcome : str
        Name of outcome variable column.
    time : str
        Name of time period column.
    treatment_group : str
        Name of treatment group indicator column.
    pre_periods : list, optional
        List of pre-treatment time periods. If None, infers from data.

    Returns
    -------
    dict
        Dictionary with trend statistics and test results.
    """
    if pre_periods is None:
        # Assume treatment happens at median time period
        all_periods = sorted(data[time].unique())
        mid_point = len(all_periods) // 2
        pre_periods = all_periods[:mid_point]

    pre_data = data[data[time].isin(pre_periods)]

    # Compute trends for each group
    treated_data = pre_data[pre_data[treatment_group] == 1]
    control_data = pre_data[pre_data[treatment_group] == 0]

    # Simple linear regression for trends
    def compute_trend(group_data: pd.DataFrame) -> Tuple[float, float]:
        time_values = group_data[time].values
        outcome_values = group_data[outcome].values

        # Normalize time to start at 0
        time_norm = time_values - time_values.min()

        # Compute slope using least squares
        n = len(time_norm)
        if n < 2:
            return np.nan, np.nan

        mean_t = np.mean(time_norm)
        mean_y = np.mean(outcome_values)

        # Check for zero variance in time (all same time period)
        time_var = np.sum((time_norm - mean_t) ** 2)
        if time_var == 0:
            return np.nan, np.nan

        slope = np.sum((time_norm - mean_t) * (outcome_values - mean_y)) / time_var

        # Compute standard error of slope
        y_hat = mean_y + slope * (time_norm - mean_t)
        residuals = outcome_values - y_hat
        mse = np.sum(residuals**2) / (n - 2)
        se_slope = np.sqrt(mse / time_var)

        return slope, se_slope

    treated_slope, treated_se = compute_trend(treated_data)
    control_slope, control_se = compute_trend(control_data)

    # Test for difference in trends
    slope_diff = treated_slope - control_slope
    se_diff = np.sqrt(treated_se**2 + control_se**2)
    t_stat, p_value, _ = safe_inference(slope_diff, se_diff)

    return {
        "treated_trend": treated_slope,
        "treated_trend_se": treated_se,
        "control_trend": control_slope,
        "control_trend_se": control_se,
        "trend_difference": slope_diff,
        "trend_difference_se": se_diff,
        "t_statistic": t_stat,
        "p_value": p_value,
        "parallel_trends_plausible": p_value > 0.05 if not np.isnan(p_value) else None,
    }


def check_parallel_trends_robust(
    data: pd.DataFrame,
    outcome: str,
    time: str,
    treatment_group: str,
    unit: Optional[str] = None,
    pre_periods: Optional[List[Any]] = None,
    n_permutations: int = 1000,
    seed: Optional[int] = None,
    wasserstein_threshold: float = 0.2,
) -> Dict[str, Any]:
    """
    Perform robust parallel trends testing using distributional comparisons.

    Uses the Wasserstein (Earth Mover's) distance to compare the full
    distribution of outcome changes between treated and control groups,
    with permutation-based inference.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with repeated observations over time.
    outcome : str
        Name of outcome variable column.
    time : str
        Name of time period column.
    treatment_group : str
        Name of treatment group indicator column (0/1).
    unit : str, optional
        Name of unit identifier column. If provided, computes unit-level
        changes. Otherwise uses observation-level data.
    pre_periods : list, optional
        List of pre-treatment time periods. If None, uses first half of periods.
    n_permutations : int, default=1000
        Number of permutations for computing p-value.
    seed : int, optional
        Random seed for reproducibility.
    wasserstein_threshold : float, default=0.2
        Threshold for normalized Wasserstein distance. Values below this
        threshold (combined with p > 0.05) suggest parallel trends are plausible.

    Returns
    -------
    dict
        Dictionary containing:
        - wasserstein_distance: Wasserstein distance between group distributions
        - wasserstein_p_value: Permutation-based p-value
        - ks_statistic: Kolmogorov-Smirnov test statistic
        - ks_p_value: KS test p-value
        - mean_difference: Difference in mean changes
        - variance_ratio: Ratio of variances in changes
        - treated_changes: Array of outcome changes for treated
        - control_changes: Array of outcome changes for control
        - parallel_trends_plausible: Boolean assessment

    Examples
    --------
    >>> results = check_parallel_trends_robust(
    ...     data, outcome='sales', time='year',
    ...     treatment_group='treated', unit='firm_id'
    ... )
    >>> print(f"Wasserstein distance: {results['wasserstein_distance']:.4f}")
    >>> print(f"P-value: {results['wasserstein_p_value']:.4f}")

    Notes
    -----
    The Wasserstein distance (Earth Mover's Distance) measures the minimum
    "cost" of transforming one distribution into another. Unlike simple
    mean comparisons, it captures differences in the entire distribution
    shape, making it more robust to non-normal data and heterogeneous effects.

    A small Wasserstein distance and high p-value suggest the distributions
    of pre-treatment changes are similar, supporting the parallel trends
    assumption.
    """
    # Use local RNG to avoid affecting global random state
    rng = np.random.default_rng(seed)

    # Identify pre-treatment periods
    if pre_periods is None:
        all_periods = sorted(data[time].unique())
        mid_point = len(all_periods) // 2
        pre_periods = all_periods[:mid_point]

    pre_data = data[data[time].isin(pre_periods)].copy()

    # Compute outcome changes
    treated_changes, control_changes = _compute_outcome_changes(
        pre_data,
        outcome,
        time,
        treatment_group,
        unit,
        caller_label="check_parallel_trends_robust",
    )

    if len(treated_changes) < 2 or len(control_changes) < 2:
        return {
            "wasserstein_distance": np.nan,
            "wasserstein_p_value": np.nan,
            "ks_statistic": np.nan,
            "ks_p_value": np.nan,
            "mean_difference": np.nan,
            "variance_ratio": np.nan,
            "treated_changes": treated_changes,
            "control_changes": control_changes,
            "parallel_trends_plausible": None,
            "error": "Insufficient data for comparison",
        }

    # Compute Wasserstein distance
    wasserstein_dist = stats.wasserstein_distance(treated_changes, control_changes)

    # Permutation test for Wasserstein distance
    all_changes = np.concatenate([treated_changes, control_changes])
    n_treated = len(treated_changes)
    n_total = len(all_changes)

    permuted_distances = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm_idx = rng.permutation(n_total)
        perm_treated = all_changes[perm_idx[:n_treated]]
        perm_control = all_changes[perm_idx[n_treated:]]
        permuted_distances[i] = stats.wasserstein_distance(perm_treated, perm_control)

    # P-value: proportion of permuted distances >= observed
    wasserstein_p = np.mean(permuted_distances >= wasserstein_dist)

    # Kolmogorov-Smirnov test
    ks_stat, ks_p = stats.ks_2samp(treated_changes, control_changes)

    # Additional summary statistics
    mean_diff = np.mean(treated_changes) - np.mean(control_changes)
    var_treated = np.var(treated_changes, ddof=1)
    var_control = np.var(control_changes, ddof=1)
    var_ratio = var_treated / var_control if var_control > 0 else np.nan

    # Normalized Wasserstein (relative to pooled std)
    pooled_std = np.std(all_changes, ddof=1)
    wasserstein_normalized = wasserstein_dist / pooled_std if pooled_std > 0 else np.nan

    # Assessment: parallel trends plausible if p-value > 0.05
    # and normalized Wasserstein is small (below threshold)
    plausible = bool(
        wasserstein_p > 0.05
        and (
            wasserstein_normalized < wasserstein_threshold
            if not np.isnan(wasserstein_normalized)
            else True
        )
    )

    return {
        "wasserstein_distance": wasserstein_dist,
        "wasserstein_normalized": wasserstein_normalized,
        "wasserstein_p_value": wasserstein_p,
        "ks_statistic": ks_stat,
        "ks_p_value": ks_p,
        "mean_difference": mean_diff,
        "variance_ratio": var_ratio,
        "n_treated": len(treated_changes),
        "n_control": len(control_changes),
        "treated_changes": treated_changes,
        "control_changes": control_changes,
        "parallel_trends_plausible": plausible,
    }


def _compute_outcome_changes(
    data: pd.DataFrame,
    outcome: str,
    time: str,
    treatment_group: str,
    unit: Optional[str] = None,
    caller_label: str = "parallel-trend diagnostic",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute period-to-period outcome changes for treated and control groups.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    outcome : str
        Outcome variable column.
    time : str
        Time period column.
    treatment_group : str
        Treatment group indicator column.
    unit : str, optional
        Unit identifier column.

    Returns
    -------
    tuple
        (treated_changes, control_changes) as numpy arrays.
    """
    if unit is not None:
        # Unit-level changes: compute change for each unit across periods
        data_sorted = data.sort_values([unit, time])
        data_sorted["_outcome_change"] = data_sorted.groupby(unit)[outcome].diff()

        # Remove NaN from first period of each unit. The first period per unit
        # has no prior observation to diff against, so n_units drops are
        # expected. Anything beyond that is a silent side-effect of gaps or
        # NaN outcomes — surface the excess via warning (axis-E drop counter).
        n_units_observed = int(data_sorted[unit].nunique())
        n_dropped = int(data_sorted["_outcome_change"].isna().sum())
        n_unexpected_drops = max(0, n_dropped - n_units_observed)
        if n_unexpected_drops > 0:
            warnings.warn(
                f"{caller_label}: dropped {n_dropped} row(s) with NaN "
                f"first-differences; {n_units_observed} are the expected "
                f"first-period-per-unit drops, and {n_unexpected_drops} are "
                f"additional NaN first-differences (e.g. NaN outcomes or "
                f"unit-period gaps upstream). Parallel-trend statistics are "
                f"computed on the remaining rows.",
                UserWarning,
                stacklevel=3,
            )
        changes_data = data_sorted.dropna(subset=["_outcome_change"])

        treated_changes = changes_data[changes_data[treatment_group] == 1]["_outcome_change"].values

        control_changes = changes_data[changes_data[treatment_group] == 0]["_outcome_change"].values
    else:
        # Aggregate changes: compute mean change per period per group
        treated_data = data[data[treatment_group] == 1]
        control_data = data[data[treatment_group] == 0]

        # Compute period means
        treated_means = treated_data.groupby(time)[outcome].mean()
        control_means = control_data.groupby(time)[outcome].mean()

        # Compute changes between consecutive periods
        treated_changes = np.diff(treated_means.values)
        control_changes = np.diff(control_means.values)

    return treated_changes.astype(float), control_changes.astype(float)


def equivalence_test_trends(
    data: pd.DataFrame,
    outcome: str,
    time: str,
    treatment_group: str,
    unit: Optional[str] = None,
    pre_periods: Optional[List[Any]] = None,
    equivalence_margin: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Perform equivalence testing (TOST) for parallel trends.

    Tests whether the difference in trends is practically equivalent to zero
    using Two One-Sided Tests (TOST) procedure.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    outcome : str
        Name of outcome variable column.
    time : str
        Name of time period column.
    treatment_group : str
        Name of treatment group indicator column.
    unit : str, optional
        Name of unit identifier column.
    pre_periods : list, optional
        List of pre-treatment time periods.
    equivalence_margin : float, optional
        The margin for equivalence (delta). If None, uses 0.5 * pooled SD
        of outcome changes as a default.

    Returns
    -------
    dict
        Dictionary containing:
        - mean_difference: Difference in mean changes
        - equivalence_margin: The margin used
        - lower_p_value: P-value for lower bound test
        - upper_p_value: P-value for upper bound test
        - tost_p_value: Maximum of the two p-values
        - equivalent: Boolean indicating equivalence at alpha=0.05
    """
    # Get pre-treatment periods
    if pre_periods is None:
        all_periods = sorted(data[time].unique())
        mid_point = len(all_periods) // 2
        pre_periods = all_periods[:mid_point]

    pre_data = data[data[time].isin(pre_periods)].copy()

    # Compute outcome changes
    treated_changes, control_changes = _compute_outcome_changes(
        pre_data,
        outcome,
        time,
        treatment_group,
        unit,
        caller_label="equivalence_test_trends",
    )

    # Need at least 2 observations per group to compute variance
    # and at least 3 total for meaningful df calculation
    if len(treated_changes) < 2 or len(control_changes) < 2:
        return {
            "mean_difference": np.nan,
            "se_difference": np.nan,
            "equivalence_margin": np.nan,
            "lower_t_stat": np.nan,
            "upper_t_stat": np.nan,
            "lower_p_value": np.nan,
            "upper_p_value": np.nan,
            "tost_p_value": np.nan,
            "degrees_of_freedom": np.nan,
            "equivalent": None,
            "error": "Insufficient data (need at least 2 observations per group)",
        }

    # Compute statistics
    var_t = np.var(treated_changes, ddof=1)
    var_c = np.var(control_changes, ddof=1)
    n_t = len(treated_changes)
    n_c = len(control_changes)

    mean_diff = np.mean(treated_changes) - np.mean(control_changes)

    # Handle zero variance case
    if var_t == 0 and var_c == 0:
        return {
            "mean_difference": mean_diff,
            "se_difference": 0.0,
            "equivalence_margin": np.nan,
            "lower_t_stat": np.nan,
            "upper_t_stat": np.nan,
            "lower_p_value": np.nan,
            "upper_p_value": np.nan,
            "tost_p_value": np.nan,
            "degrees_of_freedom": np.nan,
            "equivalent": None,
            "error": "Zero variance in both groups - cannot perform t-test",
        }

    se_diff = np.sqrt(var_t / n_t + var_c / n_c)

    # Handle zero SE case (cannot divide by zero in t-stat calculation)
    if se_diff == 0:
        return {
            "mean_difference": mean_diff,
            "se_difference": 0.0,
            "equivalence_margin": np.nan,
            "lower_t_stat": np.nan,
            "upper_t_stat": np.nan,
            "lower_p_value": np.nan,
            "upper_p_value": np.nan,
            "tost_p_value": np.nan,
            "degrees_of_freedom": np.nan,
            "equivalent": None,
            "error": "Zero standard error - cannot perform t-test",
        }

    # Set equivalence margin if not provided
    if equivalence_margin is None:
        pooled_changes = np.concatenate([treated_changes, control_changes])
        equivalence_margin = 0.5 * np.std(pooled_changes, ddof=1)

    # Degrees of freedom (Welch-Satterthwaite approximation)
    # Guard against division by zero when one group has zero variance
    numerator = (var_t / n_t + var_c / n_c) ** 2
    denom_t = (var_t / n_t) ** 2 / (n_t - 1) if var_t > 0 else 0
    denom_c = (var_c / n_c) ** 2 / (n_c - 1) if var_c > 0 else 0
    denominator = denom_t + denom_c

    if denominator == 0:
        # Fall back to minimum of n_t-1 and n_c-1 when one variance is zero
        df = min(n_t - 1, n_c - 1)
    else:
        df = numerator / denominator

    # TOST: Two one-sided tests
    # Test 1: H0: diff <= -margin vs H1: diff > -margin
    t_lower = (mean_diff - (-equivalence_margin)) / se_diff
    p_lower = stats.t.sf(t_lower, df)

    # Test 2: H0: diff >= margin vs H1: diff < margin
    t_upper = (mean_diff - equivalence_margin) / se_diff
    p_upper = stats.t.cdf(t_upper, df)

    # TOST p-value is the maximum of the two
    tost_p = max(p_lower, p_upper)

    return {
        "mean_difference": mean_diff,
        "se_difference": se_diff,
        "equivalence_margin": equivalence_margin,
        "lower_t_stat": t_lower,
        "upper_t_stat": t_upper,
        "lower_p_value": p_lower,
        "upper_p_value": p_upper,
        "tost_p_value": tost_p,
        "degrees_of_freedom": df,
        "equivalent": bool(tost_p < 0.05),
    }


# compute_synthetic_weights and _compute_synthetic_weights_numpy removed in the
# silent-failures audit post-cleanup (finding #22). The one caller
# (`diff_diff.prep.rank_control_units`) inlines a single-pass, uncentered
# Frank-Wolfe via the shared `_sc_weight_fw` dispatcher — a ranking heuristic,
# NOT the canonical SDID/R `synthdid::sc.weight.fw` two-pass procedure
# (intercept=True, 100-iter -> sparsify -> 10000-iter). Canonical SDID unit
# weights go through `compute_sdid_unit_weights` (see `_sc_weight_fw_numpy`
# below and REGISTRY.md SDID section).


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """
    Project vector onto probability simplex (sum to 1, non-negative).

    Uses the algorithm from Duchi et al. (2008).

    Parameters
    ----------
    v : np.ndarray
        Vector to project.

    Returns
    -------
    np.ndarray
        Projected vector on the simplex.
    """
    n = len(v)
    if n == 0:
        return v

    # Sort in descending order
    u = np.sort(v)[::-1]

    # Find the threshold
    cssv = np.cumsum(u)
    rho = np.where(u > (cssv - 1) / np.arange(1, n + 1))[0]

    if len(rho) == 0:
        # All elements are negative or zero
        rho_val = 0
    else:
        rho_val = rho[-1]

    theta = (cssv[rho_val] - 1) / (rho_val + 1)

    return np.asarray(np.maximum(v - theta, 0))


# =============================================================================
# SDID Weight Optimization (Frank-Wolfe, matching R's synthdid)
# =============================================================================


def _sum_normalize(v: np.ndarray) -> np.ndarray:
    """Normalize vector to sum to 1. Fallback to uniform if sum is zero.

    Matches R's synthdid ``sum_normalize()`` helper.
    """
    s = np.sum(v)
    if s > 0:
        return v / s
    return np.ones(len(v)) / len(v)


def _compute_noise_level(Y_pre_control: np.ndarray) -> float:
    """Compute noise level from first-differences of control outcomes.

    Matches R's ``sd(apply(Y[1:N0, 1:T0], 1, diff))`` which computes
    first-differences across time for each control unit, then takes the
    pooled standard deviation.

    Parameters
    ----------
    Y_pre_control : np.ndarray
        Control unit pre-treatment outcomes, shape (n_pre, n_control).

    Returns
    -------
    float
        Noise level (standard deviation of first-differences).
    """
    if HAS_RUST_BACKEND:
        return float(_rust_compute_noise_level(np.ascontiguousarray(Y_pre_control)))
    return _compute_noise_level_numpy(Y_pre_control)


def _compute_noise_level_numpy(Y_pre_control: np.ndarray) -> float:
    """Pure NumPy implementation of noise level computation."""
    if Y_pre_control.shape[0] < 2:
        return 0.0
    # R: apply(Y[1:N0, 1:T0], 1, diff) computes diff per row (unit).
    # Our matrix is (T, N) so diff along axis=0 gives (T-1, N).
    first_diffs = np.diff(Y_pre_control, axis=0)  # (T_pre-1, N_co)
    if first_diffs.size <= 1:
        return 0.0
    return float(np.std(first_diffs, ddof=1))


def _compute_regularization(
    Y_pre_control: np.ndarray,
    n_treated: int,
    n_post: int,
) -> tuple:
    """Compute auto-regularization parameters matching R's synthdid.

    Parameters
    ----------
    Y_pre_control : np.ndarray
        Control unit pre-treatment outcomes, shape (n_pre, n_control).
    n_treated : int
        Number of treated units.
    n_post : int
        Number of post-treatment periods.

    Returns
    -------
    tuple
        (zeta_omega, zeta_lambda) regularization parameters.
    """
    sigma = _compute_noise_level(Y_pre_control)
    eta_omega = (n_treated * n_post) ** 0.25
    eta_lambda = 1e-6
    return eta_omega * sigma, eta_lambda * sigma


def _fw_step(
    A: np.ndarray,
    x: np.ndarray,
    b: np.ndarray,
    eta: float,
) -> np.ndarray:
    """Single Frank-Wolfe step on the simplex.

    Matches R's ``fw.step()`` in synthdid's ``sc.weight.fw()``.

    Parameters
    ----------
    A : np.ndarray
        Matrix of shape (N, T0).
    x : np.ndarray
        Current weight vector of shape (T0,).
    b : np.ndarray
        Target vector of shape (N,).
    eta : float
        Regularization strength (N * zeta^2).

    Returns
    -------
    np.ndarray
        Updated weight vector on the simplex.
    """
    Ax = A @ x
    half_grad = A.T @ (Ax - b) + eta * x
    i = int(np.argmin(half_grad))
    d_x = -x.copy()
    d_x[i] += 1.0
    if np.allclose(d_x, 0.0):
        return x.copy()
    d_err = A[:, i] - Ax
    denom = d_err @ d_err + eta * (d_x @ d_x)
    if denom <= 0:
        return x.copy()
    step = -(half_grad @ d_x) / denom
    step = float(np.clip(step, 0.0, 1.0))
    return x + step * d_x


def _sc_weight_fw(
    Y: np.ndarray,
    zeta: float,
    intercept: bool = True,
    init_weights: Optional[np.ndarray] = None,
    min_decrease: float = 1e-5,
    max_iter: int = 10000,
    return_convergence: bool = False,
    reg_weights: Optional[np.ndarray] = None,
):
    """Compute synthetic control weights via Frank-Wolfe optimization.

    Matches R's ``sc.weight.fw()`` from the synthdid package. Solves::

        min_{lambda on simplex}  zeta^2 * ||lambda||^2
            + (1/N) * ||A_centered @ lambda - b_centered||^2

    With ``reg_weights`` set, solves the weighted-regularization variant
    used by SDID survey-bootstrap (PR #352)::

        min_{lambda on simplex}  zeta^2 * sum_j reg_weights[j] * lambda[j]^2
            + (1/N) * ||A_centered @ lambda - b_centered||^2

    Parameters
    ----------
    Y : np.ndarray
        Matrix of shape (N, T0+1). Last column is the target (post-period
        mean or treated pre-period mean depending on context).
    zeta : float
        Regularization strength.
    intercept : bool, default True
        If True, column-center Y before optimization.
    init_weights : np.ndarray, optional
        Initial weights. If None, starts with uniform weights.
    min_decrease : float, default 1e-5
        Convergence criterion: stop when objective decreases by less than
        ``min_decrease**2``. R uses ``1e-5 * noise_level``; the caller
        should pass the data-dependent value for best results.
    max_iter : int, default 10000
        Maximum number of iterations. Matches R's default.
    return_convergence : bool, default False
        If True, returns a tuple ``(weights, converged)`` where
        ``converged`` is ``True`` iff the min-decrease criterion fired
        rather than ``max_iter`` being reached. Dispatches to the Rust
        ``sc_weight_fw_with_convergence`` entry point when available, and
        to ``_sc_weight_fw_numpy(return_convergence=True)`` otherwise. Used
        by SDID bootstrap to surface per-draw FW non-convergence
        explicitly instead of relying on ``warnings.catch_warnings`` (the
        default Rust FW entry point is silent on non-convergence).
    reg_weights : np.ndarray, optional
        Per-coordinate regularization weights of shape ``(T0,)``. When
        set, switches to the weighted-regularization Rust kernel
        (``sc_weight_fw_weighted`` / ``_with_convergence``) which solves
        the SDID survey-bootstrap objective with ``ζ²·Σ rw·ω²`` in place
        of the uniform ``ζ²·||ω||²``. The caller is responsible for any
        column-scaling of ``Y`` to match the loss form. Default ``None``
        delegates to the unweighted kernel — preserves the legacy ABI for
        all existing callers.

    Returns
    -------
    np.ndarray or Tuple[np.ndarray, bool]
        Weights of shape (T0,) on the simplex; with
        ``return_convergence=True``, additionally the convergence flag.
    """
    Y_c = np.ascontiguousarray(Y, dtype=np.float64)
    init_c = (
        np.ascontiguousarray(init_weights, dtype=np.float64) if init_weights is not None else None
    )
    rw_c = np.ascontiguousarray(reg_weights, dtype=np.float64) if reg_weights is not None else None

    if rw_c is not None:
        # Validate reg_weights shape at the dispatcher so Rust and NumPy
        # backends share a single failure surface. The Rust
        # ``sc_weight_fw_weighted_internal`` silently falls back to the
        # unweighted kernel on a length mismatch, while the NumPy
        # implementation raises — dispatching without a shared upstream
        # check would let callers get the wrong objective on the Rust
        # path with no error (PR #355 R5 P2).
        expected_t0 = Y_c.shape[1] - 1
        if rw_c.shape != (expected_t0,):
            raise ValueError(
                f"reg_weights shape {rw_c.shape} does not match expected "
                f"({expected_t0},) — must equal Y.shape[1] - 1"
            )

    if HAS_RUST_BACKEND:
        if reg_weights is not None:
            if return_convergence:
                weights, converged = _rust_sc_weight_fw_weighted_with_convergence(
                    Y_c,
                    zeta,
                    intercept,
                    init_c,
                    min_decrease,
                    max_iter,
                    rw_c,
                )
                return np.asarray(weights), converged
            return np.asarray(
                _rust_sc_weight_fw_weighted(
                    Y_c,
                    zeta,
                    intercept,
                    init_c,
                    min_decrease,
                    max_iter,
                    rw_c,
                )
            )
        if return_convergence:
            weights, converged = _rust_sc_weight_fw_with_convergence(
                Y_c,
                zeta,
                intercept,
                init_c,
                min_decrease,
                max_iter,
            )
            return np.asarray(weights), converged
        return np.asarray(
            _rust_sc_weight_fw(
                Y_c,
                zeta,
                intercept,
                init_c,
                min_decrease,
                max_iter,
            )
        )
    return _sc_weight_fw_numpy(
        Y,
        zeta,
        intercept,
        init_weights,
        min_decrease,
        max_iter,
        return_convergence=return_convergence,
        reg_weights=reg_weights,
    )


def _sc_weight_fw_numpy(
    Y: np.ndarray,
    zeta: float,
    intercept: bool = True,
    init_weights: Optional[np.ndarray] = None,
    min_decrease: float = 1e-5,
    max_iter: int = 10000,
    return_convergence: bool = False,
    reg_weights: Optional[np.ndarray] = None,
):
    """Pure NumPy implementation of Frank-Wolfe SC weight solver.

    When ``return_convergence=True``, returns a tuple ``(weights, converged)``
    and suppresses the default ``warn_if_not_converged`` side effect — the
    caller is responsible for deciding how to surface non-convergence.

    With ``reg_weights`` set, solves the weighted-regularization variant
    (matches the Rust ``sc_weight_fw_weighted`` kernel; PR #352). The loss
    term is unchanged; only the regularization becomes
    ``ζ²·Σ_j reg_weights[j]·lam[j]²`` and the FW step uses the diag(rw)-
    weighted simplex direction norm.
    """
    T0 = Y.shape[1] - 1
    N = Y.shape[0]

    if T0 <= 0:
        lam_trivial = np.ones(max(T0, 1))
        if return_convergence:
            return lam_trivial, True
        return lam_trivial

    # Column-center if using intercept (matches R's intercept=TRUE default)
    if intercept:
        Y = Y - Y.mean(axis=0)

    A = Y[:, :T0]
    b = Y[:, T0]
    eta = N * zeta**2

    if init_weights is not None:
        lam = init_weights.copy()
    else:
        lam = np.ones(T0) / T0

    if reg_weights is not None:
        rw = np.asarray(reg_weights, dtype=np.float64)
        if rw.shape != (T0,):
            raise ValueError(
                f"reg_weights shape {rw.shape} does not match expected "
                f"({T0},) — must equal A.shape[1]"
            )
    else:
        rw = None

    vals = np.full(max_iter, np.nan)
    converged = False
    for t in range(max_iter):
        if rw is None:
            lam = _fw_step(A, lam, b, eta)
            err = Y @ np.append(lam, -1.0)
            vals[t] = zeta**2 * np.sum(lam**2) + np.sum(err**2) / N
        else:
            # Weighted FW step with diag(rw) regularization. Mirrors the
            # Rust sc_weight_fw_*_weighted derivation in rust/src/weights.rs.
            ax_minus_b = A @ lam - b
            half_grad = A.T @ ax_minus_b + eta * rw * lam
            i = int(np.argmin(half_grad))
            d = -lam.copy()
            d[i] += 1.0
            d_x_w_norm_sq = float(np.sum(rw * d * d))
            if d_x_w_norm_sq < 1e-24:
                err = ax_minus_b
                vals[t] = zeta**2 * float(np.sum(rw * lam * lam)) + float(np.sum(err**2)) / N
                if t >= 1 and vals[t - 1] - vals[t] < min_decrease**2:
                    converged = True
                    break
                continue
            d_err_sq = float(np.sum((A @ d) ** 2))
            denom = d_err_sq + eta * d_x_w_norm_sq
            if denom <= 0.0:
                err = ax_minus_b
                vals[t] = zeta**2 * float(np.sum(rw * lam * lam)) + float(np.sum(err**2)) / N
                if t >= 1 and vals[t - 1] - vals[t] < min_decrease**2:
                    converged = True
                    break
                continue
            hg_dot_dx = float(half_grad @ d)
            step = float(np.clip(-hg_dot_dx / denom, 0.0, 1.0))
            lam = lam + step * d
            err = A @ lam - b
            vals[t] = zeta**2 * float(np.sum(rw * lam * lam)) + float(np.sum(err**2)) / N
        if t >= 1 and vals[t - 1] - vals[t] < min_decrease**2:
            converged = True
            break
    if return_convergence:
        return lam, converged
    warn_if_not_converged(converged, "Frank-Wolfe SC weight solver", max_iter, min_decrease)

    return lam


def _sparsify(v: np.ndarray) -> np.ndarray:
    """Sparsify weight vector by zeroing out small entries.

    Matches R's synthdid ``sparsify_function``:
    ``v[v <= max(v)/4] = 0; v = v / sum(v)``

    Parameters
    ----------
    v : np.ndarray
        Weight vector.

    Returns
    -------
    np.ndarray
        Sparsified weight vector summing to 1.
    """
    v = v.copy()
    max_v = np.max(v)
    if max_v <= 0:
        return np.ones(len(v)) / len(v)
    v[v <= max_v / 4] = 0.0
    return _sum_normalize(v)


def compute_time_weights(
    Y_pre_control: np.ndarray,
    Y_post_control: np.ndarray,
    zeta_lambda: float,
    intercept: bool = True,
    min_decrease: float = 1e-5,
    max_iter_pre_sparsify: int = 100,
    max_iter: int = 10000,
    init_weights: Optional[np.ndarray] = None,
    return_convergence: bool = False,
):
    """Compute SDID time weights via Frank-Wolfe optimization.

    Matches R's ``synthdid::sc.weight.fw(Yc[1:N0, ], zeta=zeta.lambda,
    intercept=TRUE)`` where ``Yc`` is the collapsed-form matrix. Uses
    two-pass optimization with sparsification (same as unit weights),
    matching R's default ``sparsify=sparsify_function``.

    Parameters
    ----------
    Y_pre_control : np.ndarray
        Control outcomes in pre-treatment periods, shape (n_pre, n_control).
    Y_post_control : np.ndarray
        Control outcomes in post-treatment periods, shape (n_post, n_control).
    zeta_lambda : float
        Regularization parameter for time weights.
    intercept : bool, default True
        If True, column-center the optimization matrix.
    min_decrease : float, default 1e-5
        Convergence criterion for Frank-Wolfe. R uses ``1e-5 * noise_level``.
    max_iter_pre_sparsify : int, default 100
        Iterations for first pass (before sparsification).
    max_iter : int, default 10000
        Maximum iterations for second pass (after sparsification).
        Matches R's default.
    init_weights : np.ndarray, optional
        Warm-start weights for the first Frank-Wolfe pass, shape ``(n_pre,)``.
        If None (default), the solver starts from uniform, matching the
        top-level ``synthdid_estimate(update.lambda=TRUE)`` path. When
        provided, the Rust fast-path is skipped in favor of the Python
        two-pass dispatcher so the first-pass init can be threaded
        through; this matches R's ``synthdid::bootstrap_sample`` shape
        (which passes ``weights$lambda`` as FW init per draw). Used by
        ``SyntheticDiD._bootstrap_se`` on the refit loop.
    return_convergence : bool, default False
        If True, returns a tuple ``(weights, converged)`` where ``converged``
        is the AND of the first-pass and second-pass convergence flags from
        the underlying ``_sc_weight_fw`` calls (True iff the min-decrease
        criterion fired on BOTH passes; False if either hit ``max_iter``).
        Setting this flag also forces the Python two-pass dispatcher even
        when ``init_weights`` is None, because the Rust top-level fast-path
        is silent on non-convergence. Used by SDID bootstrap to surface
        per-draw FW non-convergence explicitly; standalone callers can
        leave this at the default to preserve the legacy ABI.

    Returns
    -------
    np.ndarray or Tuple[np.ndarray, bool]
        Time weights of shape (n_pre,) on the simplex. With
        ``return_convergence=True``, additionally the two-pass convergence
        flag (as described above).
    """
    if Y_post_control.shape[0] == 0:
        raise ValueError(
            "Y_post_control has no rows. At least one post-treatment period "
            "is required for time weight computation."
        )

    # When the caller asks for convergence tracking, skip the Rust top-level
    # fast-path even if init_weights is None — that entry point bypasses the
    # Python two-pass dispatcher and is silent on FW non-convergence.
    if HAS_RUST_BACKEND and init_weights is None and not return_convergence:
        return np.asarray(
            _rust_compute_time_weights(
                np.ascontiguousarray(Y_pre_control, dtype=np.float64),
                np.ascontiguousarray(Y_post_control, dtype=np.float64),
                zeta_lambda,
                intercept,
                min_decrease,
                max_iter_pre_sparsify,
                max_iter,
            )
        )

    n_pre = Y_pre_control.shape[0]

    if n_pre <= 1:
        lam_trivial = np.ones(n_pre)
        if return_convergence:
            return lam_trivial, True
        return lam_trivial

    # Build collapsed form: (N_co, T_pre + 1), last col = per-control post mean
    post_means = np.mean(Y_post_control, axis=0)  # (N_co,)
    Y_time = np.column_stack([Y_pre_control.T, post_means])  # (N_co, T_pre+1)

    # First pass: limited iterations (matching R's max.iter.pre.sparsify).
    # init_weights is either None (uniform start) or the caller-supplied
    # warm-start; the inner _sc_weight_fw still dispatches to Rust for the
    # 100-iter run, so we only pay a Python-level dispatch overhead.
    if return_convergence:
        lam, conv1 = _sc_weight_fw(
            Y_time,
            zeta=zeta_lambda,
            intercept=intercept,
            init_weights=init_weights,
            min_decrease=min_decrease,
            max_iter=max_iter_pre_sparsify,
            return_convergence=True,
        )
    else:
        lam = _sc_weight_fw(
            Y_time,
            zeta=zeta_lambda,
            intercept=intercept,
            init_weights=init_weights,
            min_decrease=min_decrease,
            max_iter=max_iter_pre_sparsify,
        )

    # Sparsify: zero out small weights, renormalize (R's sparsify_function)
    lam = _sparsify(lam)

    # Second pass: from sparsified initialization (matching R's max.iter)
    if return_convergence:
        lam, conv2 = _sc_weight_fw(
            Y_time,
            zeta=zeta_lambda,
            intercept=intercept,
            init_weights=lam,
            min_decrease=min_decrease,
            max_iter=max_iter,
            return_convergence=True,
        )
        return lam, bool(conv1 and conv2)

    lam = _sc_weight_fw(
        Y_time,
        zeta=zeta_lambda,
        intercept=intercept,
        init_weights=lam,
        min_decrease=min_decrease,
        max_iter=max_iter,
    )

    return lam


def compute_sdid_unit_weights(
    Y_pre_control: np.ndarray,
    Y_pre_treated_mean: np.ndarray,
    zeta_omega: float,
    intercept: bool = True,
    min_decrease: float = 1e-5,
    max_iter_pre_sparsify: int = 100,
    max_iter: int = 10000,
    init_weights: Optional[np.ndarray] = None,
    return_convergence: bool = False,
):
    """Compute SDID unit weights via Frank-Wolfe with two-pass sparsification.

    Matches R's ``synthdid::sc.weight.fw(t(Yc[, 1:T0]), zeta=zeta.omega,
    intercept=TRUE)`` followed by the sparsify/re-optimize pass.

    Parameters
    ----------
    Y_pre_control : np.ndarray
        Control outcomes in pre-treatment periods, shape (n_pre, n_control).
    Y_pre_treated_mean : np.ndarray
        Mean treated outcomes in pre-treatment periods, shape (n_pre,).
    zeta_omega : float
        Regularization parameter for unit weights.
    intercept : bool, default True
        If True, column-center the optimization matrix.
    min_decrease : float, default 1e-5
        Convergence criterion for Frank-Wolfe. R uses ``1e-5 * noise_level``.
    max_iter_pre_sparsify : int, default 100
        Iterations for first pass (before sparsification).
    max_iter : int, default 10000
        Iterations for second pass (after sparsification). Matches R's default.
    init_weights : np.ndarray, optional
        Warm-start weights for the first Frank-Wolfe pass, shape
        ``(n_control,)``. If None (default), the solver starts from
        uniform — matching the top-level ``synthdid_estimate(update.omega=TRUE)``
        path. When provided, the Rust fast-path is skipped in favor of the
        Python two-pass dispatcher so the first-pass init can be threaded
        through; this matches R's ``synthdid::bootstrap_sample`` shape
        (which passes ``sum_normalize(weights$omega[...])`` as FW init per
        draw). Used by ``SyntheticDiD._bootstrap_se`` on the refit loop.
    return_convergence : bool, default False
        If True, returns a tuple ``(weights, converged)`` where ``converged``
        is the AND of the first-pass and second-pass convergence flags from
        the underlying ``_sc_weight_fw`` calls (True iff the min-decrease
        criterion fired on BOTH passes; False if either hit ``max_iter``).
        Setting this flag also forces the Python two-pass dispatcher even
        when ``init_weights`` is None, because the Rust top-level fast-path
        is silent on non-convergence. Used by SDID bootstrap to surface
        per-draw FW non-convergence explicitly; standalone callers can
        leave this at the default to preserve the legacy ABI.

    Returns
    -------
    np.ndarray or Tuple[np.ndarray, bool]
        Unit weights of shape (n_control,) on the simplex. With
        ``return_convergence=True``, additionally the two-pass convergence
        flag (as described above).
    """
    n_control = Y_pre_control.shape[1]

    if n_control == 0:
        empty = np.asarray([])
        if return_convergence:
            return empty, True
        return empty
    if n_control == 1:
        singleton = np.asarray([1.0])
        if return_convergence:
            return singleton, True
        return singleton

    # When the caller asks for convergence tracking, skip the Rust top-level
    # fast-path even if init_weights is None — that entry point bypasses the
    # Python two-pass dispatcher and is silent on FW non-convergence.
    if HAS_RUST_BACKEND and init_weights is None and not return_convergence:
        return np.asarray(
            _rust_sdid_unit_weights(
                np.ascontiguousarray(Y_pre_control, dtype=np.float64),
                np.ascontiguousarray(Y_pre_treated_mean, dtype=np.float64),
                zeta_omega,
                intercept,
                min_decrease,
                max_iter_pre_sparsify,
                max_iter,
            )
        )

    # Build collapsed form: (T_pre, N_co + 1), last col = treated pre means
    Y_unit = np.column_stack([Y_pre_control, Y_pre_treated_mean.reshape(-1, 1)])

    # First pass: limited iterations. init_weights is either None (uniform
    # start) or the caller-supplied warm-start; the inner _sc_weight_fw
    # still dispatches to Rust for the 100-iter run, so we only pay a
    # Python-level dispatch overhead.
    if return_convergence:
        omega, conv1 = _sc_weight_fw(
            Y_unit,
            zeta=zeta_omega,
            intercept=intercept,
            init_weights=init_weights,
            max_iter=max_iter_pre_sparsify,
            min_decrease=min_decrease,
            return_convergence=True,
        )
    else:
        omega = _sc_weight_fw(
            Y_unit,
            zeta=zeta_omega,
            intercept=intercept,
            init_weights=init_weights,
            max_iter=max_iter_pre_sparsify,
            min_decrease=min_decrease,
        )

    # Sparsify: zero out weights <= max/4, renormalize
    omega = _sparsify(omega)

    # Second pass: from sparsified initialization
    if return_convergence:
        omega, conv2 = _sc_weight_fw(
            Y_unit,
            zeta=zeta_omega,
            intercept=intercept,
            init_weights=omega,
            max_iter=max_iter,
            min_decrease=min_decrease,
            return_convergence=True,
        )
        return omega, bool(conv1 and conv2)

    omega = _sc_weight_fw(
        Y_unit,
        zeta=zeta_omega,
        intercept=intercept,
        init_weights=omega,
        max_iter=max_iter,
        min_decrease=min_decrease,
    )

    return omega


# =============================================================================
# Survey-weighted SDID FW helpers (PR #352 — internal, called from
# SyntheticDiD._bootstrap_se on per-draw survey-weighted refits)
# =============================================================================


def compute_sdid_unit_weights_survey(
    Y_pre_control: np.ndarray,
    Y_pre_treated_mean: np.ndarray,
    rw_control: np.ndarray,
    zeta_omega: float,
    intercept: bool = True,
    min_decrease: float = 1e-5,
    max_iter_pre_sparsify: int = 100,
    max_iter: int = 10000,
    init_weights: Optional[np.ndarray] = None,
    return_convergence: bool = False,
):
    """Survey-weighted SDID unit weights via two-pass weighted Frank-Wolfe.

    Solves the weighted-FW objective (PR #352 §2.2)::

        min_{ω on simplex}
            Σ_t (Σ_i rw_control[i]·ω[i]·Y_pre_control[t,i] - Y_pre_treated_mean[t])²
            + ζ²·Σ_i rw_control[i]·ω[i]²

    Implementation: pre-scales each control column of Y_unit by
    ``rw_control`` (so the loss term picks up the per-control linear
    combination) and passes ``rw_control`` as ``reg_weights`` to
    ``_sc_weight_fw`` (so the regularization picks up the per-ω scaling).
    Two-pass sparsify-refit structure mirrors ``compute_sdid_unit_weights``.

    The returned ω is on the standard simplex. The caller (typically
    ``SyntheticDiD._bootstrap_se``) is responsible for composing
    ``ω_eff = rw_control·ω / Σ(rw_control·ω)`` for the downstream SDID
    estimator, which expects a normalized weight vector.

    Parameters
    ----------
    Y_pre_control : np.ndarray
        Control outcomes in pre-treatment periods, shape (n_pre, n_control).
    Y_pre_treated_mean : np.ndarray
        Mean treated outcomes in pre-treatment periods, shape (n_pre,).
    rw_control : np.ndarray
        Per-control survey weights, shape (n_control,). Must be non-negative.
        For pweight-only bootstrap this is the constant survey weight per
        control unit; for Rao-Wu bootstrap this is the per-draw rescaled
        weight (``generate_rao_wu_weights`` output sliced to control units).
    zeta_omega : float
        Regularization parameter (already normalized by Y_scale).
    intercept : bool, default True
        Column-center the optimization matrix.
    min_decrease : float, default 1e-5
        Convergence criterion.
    max_iter_pre_sparsify : int, default 100
        First-pass iteration cap before sparsification.
    max_iter : int, default 10000
        Second-pass iteration cap.
    init_weights : np.ndarray, optional
        Warm-start weights for the first pass; shape (n_control,).
    return_convergence : bool, default False
        If True, returns ``(ω, converged)`` where converged is the AND of
        both passes' convergence flags.

    Returns
    -------
    np.ndarray or Tuple[np.ndarray, bool]
        ω on the simplex (NOT ω_eff).
    """
    n_control = Y_pre_control.shape[1]

    if rw_control.shape != (n_control,):
        raise ValueError(
            f"rw_control shape {rw_control.shape} does not match expected " f"({n_control},)"
        )

    if n_control == 0:
        empty = np.asarray([])
        return (empty, True) if return_convergence else empty
    if n_control == 1:
        singleton = np.asarray([1.0])
        return (singleton, True) if return_convergence else singleton

    # Build the column-scaled Y matrix: each control column j is multiplied by
    # rw_control[j], so A·ω in the loss equals Σ_j rw_j·ω_j·Y_j,pre.
    rw = np.ascontiguousarray(rw_control, dtype=np.float64)
    Y_scaled = np.column_stack(
        [
            Y_pre_control * rw[np.newaxis, :],
            Y_pre_treated_mean.reshape(-1, 1),
        ]
    )

    if return_convergence:
        omega, conv1 = _sc_weight_fw(
            Y_scaled,
            zeta=zeta_omega,
            intercept=intercept,
            init_weights=init_weights,
            max_iter=max_iter_pre_sparsify,
            min_decrease=min_decrease,
            return_convergence=True,
            reg_weights=rw,
        )
    else:
        omega = _sc_weight_fw(
            Y_scaled,
            zeta=zeta_omega,
            intercept=intercept,
            init_weights=init_weights,
            max_iter=max_iter_pre_sparsify,
            min_decrease=min_decrease,
            reg_weights=rw,
        )

    omega = _sparsify(omega)

    if return_convergence:
        omega, conv2 = _sc_weight_fw(
            Y_scaled,
            zeta=zeta_omega,
            intercept=intercept,
            init_weights=omega,
            max_iter=max_iter,
            min_decrease=min_decrease,
            return_convergence=True,
            reg_weights=rw,
        )
        return omega, bool(conv1 and conv2)

    return _sc_weight_fw(
        Y_scaled,
        zeta=zeta_omega,
        intercept=intercept,
        init_weights=omega,
        max_iter=max_iter,
        min_decrease=min_decrease,
        reg_weights=rw,
    )


def compute_time_weights_survey(
    Y_pre_control: np.ndarray,
    Y_post_control: np.ndarray,
    rw_control: np.ndarray,
    zeta_lambda: float,
    intercept: bool = True,
    min_decrease: float = 1e-5,
    max_iter_pre_sparsify: int = 100,
    max_iter: int = 10000,
    init_weights: Optional[np.ndarray] = None,
    return_convergence: bool = False,
):
    """Survey-weighted SDID time weights via two-pass row-weighted FW.

    Solves the WLS-style time-weight objective (PR #352 §2.2)::

        min_{λ on simplex}
            Σ_u rw_control[u]·(Σ_t λ[t]·Y_u,pre-centered[t] - Y_u,post_mean-centered)²
            + ζ²·||λ||²

    Regularization stays uniform on λ (rw is per-control, λ is per-period —
    no alignment for per-λ reg weighting). The loss term uses WLS-style
    row weights; when ``intercept=True``, the column-centering step is
    *also* survey-weighted (weighted mean across controls, weights
    ``rw_control``) so the centered loss minimizes
    ``Σ_u rw_u·(A_u·λ - b_u)²`` on the rw-centered matrix — equivalent
    to the stated weighted objective. The Rust kernel then sees the
    weighted-centered + sqrt(rw)-row-scaled matrix with
    ``intercept=False`` (no additional unweighted centering).

    The returned λ is on the standard simplex.

    Parameters
    ----------
    Y_pre_control : np.ndarray
        Shape (n_pre, n_control).
    Y_post_control : np.ndarray
        Shape (n_post, n_control).
    rw_control : np.ndarray
        Shape (n_control,), non-negative.
    zeta_lambda : float
        Regularization parameter (already normalized by Y_scale).
    Other parameters mirror ``compute_time_weights``.

    Returns
    -------
    np.ndarray or Tuple[np.ndarray, bool]
        λ on the simplex.
    """
    n_pre = Y_pre_control.shape[0]
    n_control = Y_pre_control.shape[1]

    if rw_control.shape != (n_control,):
        raise ValueError(
            f"rw_control shape {rw_control.shape} does not match expected " f"({n_control},)"
        )

    if Y_post_control.shape[0] == 0:
        raise ValueError(
            "Y_post_control has no rows. At least one post-treatment period "
            "is required for time weight computation."
        )

    if n_pre <= 1:
        lam_trivial = np.ones(n_pre)
        return (lam_trivial, True) if return_convergence else lam_trivial

    # Build collapsed form like compute_time_weights: (N_co, T_pre+1)
    post_means = np.mean(Y_post_control, axis=0)
    Y_time = np.column_stack([Y_pre_control.T, post_means])  # (N_co, T_pre+1)

    # Column-center the (N_co, T_pre+1) matrix using the SURVEY-WEIGHTED
    # mean across control units when ``intercept=True``. Plain
    # ``intercept=True`` inside the FW kernel would use an unweighted
    # column mean which does not correspond to the stated weighted-loss
    # objective once ``rw_control`` varies. Perform the weighted
    # centering here and pass ``intercept=False`` below so the kernel
    # does not re-center on the row-scaled matrix.
    rw_sum = float(np.sum(rw_control))
    if intercept and rw_sum > 0:
        col_weighted_means = (Y_time * rw_control[:, np.newaxis]).sum(axis=0) / rw_sum
        Y_time = Y_time - col_weighted_means[np.newaxis, :]

    # Row-scale by sqrt(rw): after weighted centering (if any), each
    # control unit's contribution to the loss is weighted by
    # ``rw_control[u]`` via the sqrt(rw) row scaling, which reproduces
    # ``||diag(sqrt(rw))·(A·λ - b)||²`` = ``Σ_u rw_u·(A_u·λ - b_u)²``.
    # Reg on λ stays uniform (no reg_weights).
    sqrt_rw = np.sqrt(np.maximum(rw_control, 0.0))
    Y_weighted = Y_time * sqrt_rw[:, np.newaxis]

    if return_convergence:
        lam, conv1 = _sc_weight_fw(
            Y_weighted,
            zeta=zeta_lambda,
            intercept=False,  # weighted centering already applied above
            init_weights=init_weights,
            min_decrease=min_decrease,
            max_iter=max_iter_pre_sparsify,
            return_convergence=True,
        )
    else:
        lam = _sc_weight_fw(
            Y_weighted,
            zeta=zeta_lambda,
            intercept=False,  # weighted centering already applied above
            init_weights=init_weights,
            min_decrease=min_decrease,
            max_iter=max_iter_pre_sparsify,
        )

    lam = _sparsify(lam)

    if return_convergence:
        lam, conv2 = _sc_weight_fw(
            Y_weighted,
            zeta=zeta_lambda,
            intercept=False,  # weighted centering already applied above
            init_weights=lam,
            min_decrease=min_decrease,
            max_iter=max_iter,
            return_convergence=True,
        )
        return lam, bool(conv1 and conv2)

    return _sc_weight_fw(
        Y_weighted,
        zeta=zeta_lambda,
        intercept=False,  # weighted centering already applied above
        init_weights=lam,
        min_decrease=min_decrease,
        max_iter=max_iter,
    )


def compute_sdid_estimator(
    Y_pre_control: np.ndarray,
    Y_post_control: np.ndarray,
    Y_pre_treated: np.ndarray,
    Y_post_treated: np.ndarray,
    unit_weights: np.ndarray,
    time_weights: np.ndarray,
) -> float:
    """
    Compute the Synthetic DiD estimator.

    Parameters
    ----------
    Y_pre_control : np.ndarray
        Control outcomes in pre-treatment periods, shape (n_pre, n_control).
    Y_post_control : np.ndarray
        Control outcomes in post-treatment periods, shape (n_post, n_control).
    Y_pre_treated : np.ndarray
        Treated unit outcomes in pre-treatment periods, shape (n_pre,).
    Y_post_treated : np.ndarray
        Treated unit outcomes in post-treatment periods, shape (n_post,).
    unit_weights : np.ndarray
        Weights for control units, shape (n_control,).
    time_weights : np.ndarray
        Weights for pre-treatment periods, shape (n_pre,).

    Returns
    -------
    float
        The synthetic DiD treatment effect estimate.

    Notes
    -----
    The SDID estimator is:

        τ̂ = (Ȳ_treated,post - Σ_t λ_t * Y_treated,t)
            - Σ_j ω_j * (Ȳ_j,post - Σ_t λ_t * Y_j,t)

    Where:
    - ω_j are unit weights
    - λ_t are time weights
    - Ȳ denotes average over post periods
    """
    # Weighted pre-treatment averages
    weighted_pre_control = time_weights @ Y_pre_control  # shape: (n_control,)
    weighted_pre_treated = time_weights @ Y_pre_treated  # scalar

    # Post-treatment averages
    mean_post_control = np.mean(Y_post_control, axis=0)  # shape: (n_control,)
    mean_post_treated = np.mean(Y_post_treated)  # scalar

    # DiD for treated: post - weighted pre
    did_treated = mean_post_treated - weighted_pre_treated

    # Weighted DiD for controls: sum over j of omega_j * (post_j - weighted_pre_j)
    did_control = unit_weights @ (mean_post_control - weighted_pre_control)

    # SDID estimator
    tau = did_treated - did_control

    return float(tau)


def demean_by_group(
    data: pd.DataFrame,
    variables: List[str],
    group_var: str,
    inplace: bool = False,
    suffix: str = "",
    weights: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, int]:
    """
    Demean variables by a grouping variable (one-way within transformation).

    For each variable, computes: x_ig - mean(x_g) where g is the group.
    When weights are provided, uses weighted group means:
    mean_g = sum(w_i * x_i) / sum(w_i) for i in group g.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the variables to demean.
    variables : list of str
        Column names to demean.
    group_var : str
        Column name for the grouping variable.
    inplace : bool, default False
        If True, modifies the original columns. If False, leaves original
        columns unchanged (demeaning is still applied to return value).
    suffix : str, default ""
        Suffix to add to demeaned column names (only used when inplace=False
        and you want to keep both original and demeaned columns).
    weights : np.ndarray, optional
        Observation weights for weighted group means.

    Returns
    -------
    data : pd.DataFrame
        DataFrame with demeaned variables.
    n_effects : int
        Number of absorbed fixed effects (nunique - 1).

    Examples
    --------
    >>> df, n_fe = demean_by_group(df, ['y', 'x1', 'x2'], 'unit')
    >>> # df['y'], df['x1'], df['x2'] are now demeaned by unit
    """
    if not inplace:
        data = data.copy()

    # Count fixed effects (categories - 1 for identification)
    n_effects = data[group_var].nunique() - 1

    if weights is not None:
        # Weighted demeaning: weighted_mean_g = sum(w*x) / sum(w) per group
        groups = data[group_var].values
        w = np.asarray(weights, dtype=np.float64)
        # Cache weight sums per group (invariant across variables)
        w_sum = pd.Series(w).groupby(groups).transform("sum").values
        for var in variables:
            col_name = var if not suffix else f"{var}{suffix}"
            x = data[var].values.astype(np.float64)
            wx = pd.Series(w * x).groupby(groups).transform("sum").values
            # Guard zero-total-weight groups (survey subpopulation / zero-weight
            # domain padding) so those rows pass through unchanged and stay inert
            # in WLS. For positive-weight groups this is bit-identical to wx/w_sum.
            weighted_means = np.divide(
                wx, w_sum, out=np.zeros_like(wx, dtype=np.float64), where=w_sum > 0
            )
            data[col_name] = x - weighted_means
    else:
        # Cache the groupby object for efficiency
        grouper = data.groupby(group_var, sort=False)
        for var in variables:
            col_name = var if not suffix else f"{var}{suffix}"
            group_means = grouper[var].transform("mean")
            data[col_name] = data[var] - group_means

    return data, n_effects


def demean_by_groups(
    data: pd.DataFrame,
    variables: List[str],
    group_vars: List[str],
    inplace: bool = False,
    suffix: str = "",
    weights: Optional[np.ndarray] = None,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> Tuple[pd.DataFrame, int]:
    """N-way within transformation via the method of alternating projections (MAP).

    Removes ``len(group_vars)`` absorbed fixed-effect dimensions by repeatedly
    demeaning each variable by each group in ``group_vars`` order until the
    iterate stops changing (``max|x - x_old| < tol``) or ``max_iter`` is reached.
    This converges to the exact (W)LS Frisch-Waugh-Lovell residualization onto the
    combined column space of all ``group_vars`` dummies, for both balanced and
    unbalanced panels and for both unweighted and survey-weighted designs (it is
    the algorithm used by R ``fixest`` / ``reghdfe`` / ``lfe``).

    Single-pass sequential demeaning (one sweep) is only the one-iteration
    approximation of this projection; it is exact only when the FE subspaces are
    orthogonal (balanced fully-crossed panels). For ``len(group_vars) == 1`` the
    projection is exact in one pass, so this delegates to :func:`demean_by_group`
    (byte-identical to the prior one-way behavior).

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the variables to demean.
    variables : list of str
        Column names to demean.
    group_vars : list of str
        Grouping (fixed-effect) columns to absorb. Must be non-empty.
    inplace : bool, default False
        If True, writes the demeaned values onto ``data`` (the caller must own it)
        and returns it. If False, attaches them as a consolidated block via
        ``pd.concat`` and returns a new frame (the input is not mutated).
    suffix : str, default ""
        Demeaned column naming. Empty overwrites the source columns; a non-empty
        suffix writes to ``f"{var}{suffix}"`` columns (originals preserved).
    weights : np.ndarray, optional
        Observation weights. When provided, uses weighted group means
        ``sum(w*x)/sum(w)`` per group and converges to the WLS-FWL residual.
    max_iter : int, default 100
        Maximum number of alternating-projection iterations per variable. Emits a
        single ``UserWarning`` per call listing any variable that fails to converge.
    tol : float, default 1e-10
        Convergence tolerance on the max absolute change across the iterate.

    Returns
    -------
    data : pd.DataFrame
        DataFrame with demeaned variables.
    n_effects : int
        Number of absorbed fixed effects, ``sum_d (nunique_d - 1)`` over
        ``group_vars`` (the standard DOF-accounting convention).

    Examples
    --------
    >>> df, n_fe = demean_by_groups(df, ['y', 'x'], ['unit', 'period'])
    """
    if not group_vars:
        raise ValueError("demean_by_groups requires at least one grouping variable.")

    # One dimension: the within projection is exact in a single pass. Delegate so
    # the one-way callers stay byte-identical to demean_by_group.
    if len(group_vars) == 1:
        return demean_by_group(
            data,
            variables,
            group_vars[0],
            inplace=inplace,
            suffix=suffix,
            weights=weights,
        )

    # N >= 2: method of alternating projections. Per-variable independent
    # convergence (outer loop = variable, inner = iterations) — this mirrors
    # within_transform's weighted loop exactly so that the two-way weighted case
    # (group_vars == [unit, time]) is bit-identical.
    n_effects = sum(int(data[g].nunique()) - 1 for g in group_vars)
    target_cols = [var if not suffix else f"{var}{suffix}" for var in variables]
    group_arrays = [data[g].values for g in group_vars]
    demeaned_values: List[np.ndarray] = []
    non_converged_vars: List[str] = []

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        # Cache per-group weight sums once (invariant across variables/iterations).
        w_sums = [pd.Series(w).groupby(g).transform("sum").values for g in group_arrays]

        def _weighted_group_demean(x, groups, w, w_sum):
            wx_sum = pd.Series(w * x).groupby(groups).transform("sum").values
            # Guard zero-total-weight groups (survey subpopulation / zero-weight
            # domain padding): leave such rows unchanged (mean 0) so they remain
            # inert in the downstream WLS instead of poisoning the design with
            # NaN/Inf. For positive-weight groups this is bit-identical to wx/w_sum.
            means = np.divide(
                wx_sum, w_sum, out=np.zeros_like(wx_sum, dtype=np.float64), where=w_sum > 0
            )
            return x - means

        for var in variables:
            x = data[var].values.astype(np.float64)
            converged = False
            for _iter in range(max_iter):
                x_old = x.copy()
                for groups, w_sum in zip(group_arrays, w_sums):
                    x = _weighted_group_demean(x, groups, w, w_sum)
                if np.max(np.abs(x - x_old)) < tol:
                    converged = True
                    break
            if not converged:
                non_converged_vars.append(var)
            demeaned_values.append(x)
    else:
        for var in variables:
            x = data[var].values.astype(np.float64)
            converged = False
            for _iter in range(max_iter):
                x_old = x.copy()
                for groups in group_arrays:
                    x = x - pd.Series(x).groupby(groups).transform("mean").values
                if np.max(np.abs(x - x_old)) < tol:
                    converged = True
                    break
            if not converged:
                non_converged_vars.append(var)
            demeaned_values.append(x)

    if non_converged_vars:
        warn_if_not_converged(
            False,
            f"demean_by_groups alternating projection (variables: {non_converged_vars})",
            max_iter,
            tol,
        )

    if inplace:
        for col, vals in zip(target_cols, demeaned_values):
            data[col] = vals
        return data, n_effects

    # Non-inplace: attach the demeaned columns as a single consolidated block via
    # pd.concat (no defensive copy — the demean above is read-only). Mirrors the
    # within_transform attach contract: a colliding target name (suffix="" or a
    # re-demean) is dropped first so concat replaces rather than duplicates.
    new_block = pd.DataFrame(dict(zip(target_cols, demeaned_values)), index=data.index)
    collisions = [c for c in target_cols if c in data.columns]
    if collisions:
        data = data.drop(columns=collisions)
    return pd.concat([data, new_block], axis=1), n_effects


def within_transform(
    data: pd.DataFrame,
    variables: List[str],
    unit: str,
    time: str,
    inplace: bool = False,
    suffix: str = "_demeaned",
    weights: Optional[np.ndarray] = None,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> pd.DataFrame:
    """
    Apply two-way within transformation to remove unit and time fixed effects.

    Computed via the method of alternating projections (a thin two-way wrapper
    over :func:`demean_by_groups` with ``group_vars=[unit, time]``): each variable
    is demeaned by unit, then by time, iterated until convergence. This is the
    exact (weighted) Frisch-Waugh-Lovell residualization for both balanced and
    unbalanced panels. The closed-form additive expression ``y_it - y_i. - y_.t +
    y_..`` is the balanced fully-crossed special case (it equals the converged
    iterate only when the unit and time FE subspaces are orthogonal). When weights
    are provided, weighted group means are used at each step.

    This is the standard fixed effects transformation for panel data that
    removes both unit-specific and time-specific effects.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data containing the variables to transform.
    variables : list of str
        Column names to transform.
    unit : str
        Column name for unit identifier.
    time : str
        Column name for time period identifier.
    inplace : bool, default False
        Controls how the demeaned columns are attached. If False (default), they
        are concatenated onto the input as a single block and a new frame is
        returned; the input frame is not mutated (no defensive deep copy is taken
        — the demean is read-only and ``concat`` does not mutate its inputs). If
        True, they are written onto the passed frame in place (the caller must own
        it) and that frame is returned. Independent of ``suffix``.
    suffix : str, default "_demeaned"
        Column naming, independent of ``inplace``. A non-empty suffix writes the
        demeaned values to new ``f"{var}{suffix}"`` columns (originals preserved);
        ``suffix=""`` overwrites the source columns. Assigning to an existing
        column name overwrites it rather than appending a duplicate label.
    weights : np.ndarray, optional
        Observation weights for weighted group means.
    max_iter : int, default 100
        Maximum number of alternating-projection iterations. Applies to BOTH the
        weighted and unweighted paths (both now iterate via the method of
        alternating projections, exact for unbalanced panels). Emits a
        ``UserWarning`` per call when any variable fails to converge within this
        budget. Balanced panels converge in ~2 iterations.
    tol : float, default 1e-8
        Convergence tolerance on the max absolute change across the iterate.

    Returns
    -------
    pd.DataFrame
        DataFrame with within-transformed variables.

    Notes
    -----
    The within transformation removes variation that is constant within units
    (unit fixed effects) and constant within time periods (time fixed effects).
    The resulting estimates are equivalent to including unit and time dummies
    but is computationally more efficient for large panels.

    Examples
    --------
    >>> df = within_transform(df, ['y', 'x'], 'unit_id', 'year')
    >>> # df now has 'y_demeaned' and 'x_demeaned' columns
    """
    # Two-way (unit + time) within transformation is the N-way method of
    # alternating projections specialized to two FE dimensions. Delegate to the
    # shared engine so there is one MAP implementation. The weighted path is
    # bit-identical to the previous inline loop (same per-variable convergence,
    # unit-then-time order, and forwarded tol=1e-8); the unweighted path now also
    # iterates (MAP) instead of the closed-form additive demean, which was exact
    # only for balanced panels. ``tol`` is forwarded explicitly (within_transform's
    # default is 1e-8, not demean_by_groups' 1e-10) to preserve weighted goldens.
    return demean_by_groups(
        data,
        variables,
        [unit, time],
        inplace=inplace,
        suffix=suffix,
        weights=weights,
        max_iter=max_iter,
        tol=tol,
    )[0]
