"""
Utility functions for difference-in-differences estimation.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from diff_diff.linalg import compute_robust_vcov as _compute_robust_vcov_linalg
from diff_diff.linalg import solve_ols as _solve_ols_linalg

# Import Rust backend if available (from _backend to avoid circular imports)
from diff_diff._backend import (
    HAS_RUST_BACKEND,
    _rust_project_simplex,
    _rust_synthetic_weights,
    _rust_sdid_unit_weights,
    _rust_compute_time_weights,
    _rust_compute_noise_level,
    _rust_sc_weight_fw,
)

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
        Bootstrap standard error of the coefficient.
    p_value : float
        Bootstrap p-value (two-sided).
    t_stat_original : float
        Original t-statistic from the data.
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    n_clusters : int
        Number of clusters in the data.
    n_bootstrap : int
        Number of bootstrap replications.
    weight_type : str
        Type of bootstrap weights used ("rademacher", "webb", or "mammen").
    alpha : float
        Significance level used for confidence interval.
    bootstrap_distribution : np.ndarray, optional
        Full bootstrap distribution of coefficients (if requested).

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
    bootstrap_distribution: Optional[np.ndarray] = field(default=None, repr=False)

    def summary(self) -> str:
        """Generate formatted summary of bootstrap results."""
        lines = [
            "Wild Cluster Bootstrap Results",
            "=" * 40,
            f"Bootstrap SE:        {self.se:.6f}",
            f"Bootstrap p-value:   {self.p_value:.4f}",
            f"Original t-stat:     {self.t_stat_original:.4f}",
            f"CI ({int((1-self.alpha)*100)}%):           [{self.ci_lower:.6f}, {self.ci_upper:.6f}]",
            f"Number of clusters:  {self.n_clusters}",
            f"Bootstrap reps:      {self.n_bootstrap}",
            f"Weight type:         {self.weight_type}",
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
) -> WildBootstrapResults:
    """
    Compute wild cluster bootstrap standard errors and p-values.

    Implements the Wild Cluster Residual (WCR) bootstrap procedure from
    Cameron, Gelbach, and Miller (2008). Uses the restricted residuals
    approach (imposing H0: coefficient = null_hypothesis) for more accurate
    p-value computation.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n, k).
    y : np.ndarray
        Outcome vector of shape (n,).
    residuals : np.ndarray
        OLS residuals from unrestricted regression, shape (n,).
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
        If True, include full bootstrap distribution in results.

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

    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    if n_clusters < 2:
        raise ValueError(f"Wild cluster bootstrap requires at least 2 clusters, got {n_clusters}")

    if n_clusters < 5:
        warnings.warn(
            f"Only {n_clusters} clusters detected. Wild bootstrap inference may be "
            "unreliable with fewer than 5 clusters. Consider using Webb weights "
            "(weight_type='webb') for improved finite-sample properties.",
            UserWarning,
        )

    # Initialize RNG
    rng = np.random.default_rng(seed)

    # Select weight generator
    weight_generators = {
        "rademacher": _generate_rademacher_weights,
        "webb": _generate_webb_weights,
        "mammen": _generate_mammen_weights,
    }
    generate_weights = weight_generators[weight_type]

    n = X.shape[0]

    # Step 1: Compute original coefficient and cluster-robust SE
    beta_hat, _, vcov_original = _solve_ols_linalg(X, y, cluster_ids=cluster_ids, return_vcov=True)
    original_coef = beta_hat[coefficient_index]
    assert vcov_original is not None
    se_original = np.sqrt(vcov_original[coefficient_index, coefficient_index])
    t_stat_original = (original_coef - null_hypothesis) / se_original

    # Step 2: Impose null hypothesis (restricted estimation)
    # Create restricted y: y_restricted = y - X[:, coef_index] * null_hypothesis
    # This imposes the null that the coefficient equals null_hypothesis
    y_restricted = y - X[:, coefficient_index] * null_hypothesis

    # Fit restricted model (but we need to drop the column for the restricted coef)
    # Actually, for WCR bootstrap we keep all columns but impose the null via residuals
    # Re-estimate with the restricted dependent variable
    beta_restricted, residuals_restricted, _ = _solve_ols_linalg(X, y_restricted, return_vcov=False)

    # Create cluster-to-observation mapping for efficiency
    cluster_map = {c: np.where(cluster_ids == c)[0] for c in unique_clusters}
    cluster_indices = [cluster_map[c] for c in unique_clusters]

    # Step 3: Bootstrap loop
    bootstrap_t_stats = np.zeros(n_bootstrap)
    bootstrap_coefs = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        # Generate cluster-level weights
        cluster_weights = generate_weights(n_clusters, rng)

        # Map cluster weights to observations
        obs_weights = np.zeros(n)
        for g, indices in enumerate(cluster_indices):
            obs_weights[indices] = cluster_weights[g]

        # Construct bootstrap sample: y* = X @ beta_restricted + e_restricted * weights
        y_star = np.dot(X, beta_restricted) + residuals_restricted * obs_weights

        # Estimate bootstrap coefficients with cluster-robust SE
        beta_star, residuals_star, vcov_star = _solve_ols_linalg(
            X, y_star, cluster_ids=cluster_ids, return_vcov=True
        )
        bootstrap_coefs[b] = beta_star[coefficient_index]
        assert vcov_star is not None
        se_star = np.sqrt(vcov_star[coefficient_index, coefficient_index])

        # Compute bootstrap t-statistic (under null hypothesis)
        if se_star > 0:
            bootstrap_t_stats[b] = (beta_star[coefficient_index] - null_hypothesis) / se_star
        else:
            bootstrap_t_stats[b] = 0.0

    # Step 4: Compute bootstrap p-value
    # P-value is proportion of |t*| >= |t_original|
    p_value = np.mean(np.abs(bootstrap_t_stats) >= np.abs(t_stat_original))

    # Ensure p-value is at least 1/(n_bootstrap+1) to avoid exact zero
    p_value = float(max(float(p_value), 1 / (n_bootstrap + 1)))

    # Step 5: Compute bootstrap SE and confidence interval
    # SE from standard deviation of bootstrap coefficient distribution
    se_bootstrap = float(np.std(bootstrap_coefs, ddof=1))

    # Percentile confidence interval from bootstrap distribution
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    ci_lower = float(np.percentile(bootstrap_coefs, lower_percentile))
    ci_upper = float(np.percentile(bootstrap_coefs, upper_percentile))

    return WildBootstrapResults(
        se=se_bootstrap,
        p_value=p_value,
        t_stat_original=t_stat_original,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_clusters=n_clusters,
        n_bootstrap=n_bootstrap,
        weight_type=weight_type,
        alpha=alpha,
        bootstrap_distribution=bootstrap_coefs if return_distribution else None,
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
        pre_data, outcome, time, treatment_group, unit
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
    data: pd.DataFrame, outcome: str, time: str, treatment_group: str, unit: Optional[str] = None
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

        # Remove NaN from first period of each unit
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
        pre_data, outcome, time, treatment_group, unit
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


def compute_synthetic_weights(
    Y_control: np.ndarray, Y_treated: np.ndarray, lambda_reg: float = 0.0, min_weight: float = 1e-6
) -> np.ndarray:
    """
    Compute synthetic control unit weights using constrained optimization.

    Finds weights ω that minimize the squared difference between the
    weighted average of control unit outcomes and the treated unit outcomes
    during pre-treatment periods.

    Parameters
    ----------
    Y_control : np.ndarray
        Control unit outcomes matrix of shape (n_pre_periods, n_control_units).
        Each column is a control unit, each row is a pre-treatment period.
    Y_treated : np.ndarray
        Treated unit mean outcomes of shape (n_pre_periods,).
        Average across treated units for each pre-treatment period.
    lambda_reg : float, default=0.0
        L2 regularization parameter. Larger values shrink weights toward
        uniform (1/n_control). Helps prevent overfitting when n_pre < n_control.
    min_weight : float, default=1e-6
        Minimum weight threshold. Weights below this are set to zero.

    Returns
    -------
    np.ndarray
        Unit weights of shape (n_control_units,) that sum to 1.

    Notes
    -----
    Solves the quadratic program:

        min_ω ||Y_treated - Y_control @ ω||² + λ||ω - 1/n||²
        s.t. ω >= 0, sum(ω) = 1

    Uses a simplified coordinate descent approach with projection onto simplex.
    """
    n_pre, n_control = Y_control.shape

    if n_control == 0:
        return np.asarray([])

    if n_control == 1:
        return np.asarray([1.0])

    # Use Rust backend if available
    if HAS_RUST_BACKEND:
        Y_control = np.ascontiguousarray(Y_control, dtype=np.float64)
        Y_treated = np.ascontiguousarray(Y_treated, dtype=np.float64)
        weights = _rust_synthetic_weights(
            Y_control, Y_treated, lambda_reg, _OPTIMIZATION_MAX_ITER, _OPTIMIZATION_TOL
        )
    else:
        # Fallback to NumPy implementation
        weights = _compute_synthetic_weights_numpy(Y_control, Y_treated, lambda_reg)

    # Set small weights to zero for interpretability
    weights[weights < min_weight] = 0
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
    else:
        # Fallback to uniform if all weights are zeroed
        weights = np.ones(n_control) / n_control

    return np.asarray(weights)


def _compute_synthetic_weights_numpy(
    Y_control: np.ndarray,
    Y_treated: np.ndarray,
    lambda_reg: float = 0.0,
) -> np.ndarray:
    """NumPy fallback implementation of compute_synthetic_weights."""
    n_pre, n_control = Y_control.shape

    # Initialize with uniform weights
    weights = np.ones(n_control) / n_control

    # Precompute matrices for optimization
    # Objective: ||Y_treated - Y_control @ w||^2 + lambda * ||w - w_uniform||^2
    # = w' @ (Y_control' @ Y_control + lambda * I) @ w - 2 * (Y_control' @ Y_treated + lambda * w_uniform)' @ w + const
    YtY = Y_control.T @ Y_control
    YtT = Y_control.T @ Y_treated
    w_uniform = np.ones(n_control) / n_control

    # Add regularization
    H = YtY + lambda_reg * np.eye(n_control)
    f = YtT + lambda_reg * w_uniform

    # Solve with projected gradient descent
    # Project onto probability simplex
    step_size = 1.0 / (np.linalg.norm(H, 2) + _NUMERICAL_EPS)

    for _ in range(_OPTIMIZATION_MAX_ITER):
        weights_old = weights.copy()

        # Gradient step: minimize ||Y - Y_control @ w||^2
        grad = H @ weights - f
        weights = weights - step_size * grad

        # Project onto simplex (sum to 1, non-negative)
        weights = _project_simplex(weights)

        # Check convergence
        if np.linalg.norm(weights - weights_old) < _OPTIMIZATION_TOL:
            break

    return weights


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
) -> np.ndarray:
    """Compute synthetic control weights via Frank-Wolfe optimization.

    Matches R's ``sc.weight.fw()`` from the synthdid package. Solves::

        min_{lambda on simplex}  zeta^2 * ||lambda||^2
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

    Returns
    -------
    np.ndarray
        Weights of shape (T0,) on the simplex.
    """
    if HAS_RUST_BACKEND:
        return np.asarray(
            _rust_sc_weight_fw(
                np.ascontiguousarray(Y, dtype=np.float64),
                zeta,
                intercept,
                (
                    np.ascontiguousarray(init_weights, dtype=np.float64)
                    if init_weights is not None
                    else None
                ),
                min_decrease,
                max_iter,
            )
        )
    return _sc_weight_fw_numpy(Y, zeta, intercept, init_weights, min_decrease, max_iter)


def _sc_weight_fw_numpy(
    Y: np.ndarray,
    zeta: float,
    intercept: bool = True,
    init_weights: Optional[np.ndarray] = None,
    min_decrease: float = 1e-5,
    max_iter: int = 10000,
) -> np.ndarray:
    """Pure NumPy implementation of Frank-Wolfe SC weight solver."""
    T0 = Y.shape[1] - 1
    N = Y.shape[0]

    if T0 <= 0:
        return np.ones(max(T0, 1))

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

    vals = np.full(max_iter, np.nan)
    for t in range(max_iter):
        lam = _fw_step(A, lam, b, eta)
        err = Y @ np.append(lam, -1.0)
        vals[t] = zeta**2 * np.sum(lam**2) + np.sum(err**2) / N
        if t >= 1 and vals[t - 1] - vals[t] < min_decrease**2:
            break

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
) -> np.ndarray:
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

    Returns
    -------
    np.ndarray
        Time weights of shape (n_pre,) on the simplex.
    """
    if Y_post_control.shape[0] == 0:
        raise ValueError(
            "Y_post_control has no rows. At least one post-treatment period "
            "is required for time weight computation."
        )

    if HAS_RUST_BACKEND:
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
        return np.ones(n_pre)

    # Build collapsed form: (N_co, T_pre + 1), last col = per-control post mean
    post_means = np.mean(Y_post_control, axis=0)  # (N_co,)
    Y_time = np.column_stack([Y_pre_control.T, post_means])  # (N_co, T_pre+1)

    # First pass: limited iterations (matching R's max.iter.pre.sparsify)
    lam = _sc_weight_fw(
        Y_time,
        zeta=zeta_lambda,
        intercept=intercept,
        min_decrease=min_decrease,
        max_iter=max_iter_pre_sparsify,
    )

    # Sparsify: zero out small weights, renormalize (R's sparsify_function)
    lam = _sparsify(lam)

    # Second pass: from sparsified initialization (matching R's max.iter)
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
) -> np.ndarray:
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

    Returns
    -------
    np.ndarray
        Unit weights of shape (n_control,) on the simplex.
    """
    n_control = Y_pre_control.shape[1]

    if n_control == 0:
        return np.asarray([])
    if n_control == 1:
        return np.asarray([1.0])

    if HAS_RUST_BACKEND:
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

    # First pass: limited iterations
    omega = _sc_weight_fw(
        Y_unit,
        zeta=zeta_omega,
        intercept=intercept,
        max_iter=max_iter_pre_sparsify,
        min_decrease=min_decrease,
    )

    # Sparsify: zero out weights <= max/4, renormalize
    omega = _sparsify(omega)

    # Second pass: from sparsified initialization
    omega = _sc_weight_fw(
        Y_unit,
        zeta=zeta_omega,
        intercept=intercept,
        init_weights=omega,
        max_iter=max_iter,
        min_decrease=min_decrease,
    )

    return omega


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
        w_sum = pd.Series(w).groupby(groups).transform("sum")
        for var in variables:
            col_name = var if not suffix else f"{var}{suffix}"
            x = data[var].values.astype(np.float64)
            wx = pd.Series(w * x).groupby(groups).transform("sum")
            weighted_means = wx / w_sum
            data[col_name] = x - weighted_means.values
    else:
        # Cache the groupby object for efficiency
        grouper = data.groupby(group_var, sort=False)
        for var in variables:
            col_name = var if not suffix else f"{var}{suffix}"
            group_means = grouper[var].transform("mean")
            data[col_name] = data[var] - group_means

    return data, n_effects


def within_transform(
    data: pd.DataFrame,
    variables: List[str],
    unit: str,
    time: str,
    inplace: bool = False,
    suffix: str = "_demeaned",
    weights: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Apply two-way within transformation to remove unit and time fixed effects.

    Computes: y_it - y_i. - y_.t + y_.. for each variable.
    When weights are provided, uses weighted group means at each step.

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
        If True, modifies the original columns. If False, creates new columns
        with the specified suffix.
    suffix : str, default "_demeaned"
        Suffix for new column names when inplace=False.
    weights : np.ndarray, optional
        Observation weights for weighted group means.

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
    if not inplace:
        data = data.copy()

    if weights is not None:
        # Weighted within-transformation via iterative alternating projections
        w = np.asarray(weights, dtype=np.float64)
        unit_groups = data[unit].values
        time_groups = data[time].values

        # Cache weight sums per group (invariant across variables)
        unit_w_sum = pd.Series(w).groupby(unit_groups).transform("sum").values
        time_w_sum = pd.Series(w).groupby(time_groups).transform("sum").values

        def _weighted_group_demean(x, groups, w, w_sum):
            wx_sum = pd.Series(w * x).groupby(groups).transform("sum").values
            return x - wx_sum / w_sum

        if inplace:
            for var in variables:
                x = data[var].values.astype(np.float64)
                for _iter in range(100):  # max iterations
                    x_old = x.copy()
                    x = _weighted_group_demean(x, unit_groups, w, unit_w_sum)
                    x = _weighted_group_demean(x, time_groups, w, time_w_sum)
                    if np.max(np.abs(x - x_old)) < 1e-8:
                        break
                data[var] = x
        else:
            demeaned_data = {}
            for var in variables:
                x = data[var].values.astype(np.float64)
                for _iter in range(100):
                    x_old = x.copy()
                    x = _weighted_group_demean(x, unit_groups, w, unit_w_sum)
                    x = _weighted_group_demean(x, time_groups, w, time_w_sum)
                    if np.max(np.abs(x - x_old)) < 1e-8:
                        break
                demeaned_data[f"{var}{suffix}"] = x
            demeaned_df = pd.DataFrame(demeaned_data, index=data.index)
            data = pd.concat([data, demeaned_df], axis=1)
    else:
        # Cache groupby objects for efficiency
        unit_grouper = data.groupby(unit, sort=False)
        time_grouper = data.groupby(time, sort=False)

        if inplace:
            for var in variables:
                unit_means = unit_grouper[var].transform("mean")
                time_means = time_grouper[var].transform("mean")
                grand_mean = data[var].mean()
                data[var] = data[var] - unit_means - time_means + grand_mean
        else:
            demeaned_data = {}
            for var in variables:
                unit_means = unit_grouper[var].transform("mean")
                time_means = time_grouper[var].transform("mean")
                grand_mean = data[var].mean()
                demeaned_data[f"{var}{suffix}"] = (
                    data[var] - unit_means - time_means + grand_mean
                ).values
            demeaned_df = pd.DataFrame(demeaned_data, index=data.index)
            data = pd.concat([data, demeaned_df], axis=1)

    return data
