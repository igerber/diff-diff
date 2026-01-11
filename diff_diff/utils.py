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

# Numerical constants for optimization algorithms
_OPTIMIZATION_MAX_ITER = 1000  # Maximum iterations for weight optimization
_OPTIMIZATION_TOL = 1e-8  # Convergence tolerance for optimization
_NUMERICAL_EPS = 1e-10  # Small constant to prevent division by zero


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
        raise ValueError(
            f"{name} must be binary (0 or 1). "
            f"Found values: {unique_values}"
        )


def compute_robust_se(
    X: np.ndarray,
    residuals: np.ndarray,
    cluster_ids: Optional[np.ndarray] = None
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
    estimate: float,
    se: float,
    alpha: float = 0.05,
    df: Optional[int] = None
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
    if df is not None:
        critical_value = stats.t.ppf(1 - alpha / 2, df)
    else:
        critical_value = stats.norm.ppf(1 - alpha / 2)

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
    with probabilities proportional to {1, 2, 3, 3, 2, 1}.

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
    """
    values = np.array([
        -np.sqrt(3 / 2), -np.sqrt(2 / 2), -np.sqrt(1 / 2),
        np.sqrt(1 / 2), np.sqrt(2 / 2), np.sqrt(3 / 2)
    ])
    probs = np.array([1, 2, 3, 3, 2, 1]) / 12
    return np.asarray(rng.choice(values, size=n_clusters, p=probs))


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
    val2 = (sqrt5 + 1) / 2   # approximately 1.618 (golden ratio)

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
    return_distribution: bool = False
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
        raise ValueError(
            f"weight_type must be one of {valid_weight_types}, got '{weight_type}'"
        )

    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    if n_clusters < 2:
        raise ValueError(
            f"Wild cluster bootstrap requires at least 2 clusters, got {n_clusters}"
        )

    if n_clusters < 5:
        warnings.warn(
            f"Only {n_clusters} clusters detected. Wild bootstrap inference may be "
            "unreliable with fewer than 5 clusters. Consider using Webb weights "
            "(weight_type='webb') for improved finite-sample properties.",
            UserWarning
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
    beta_hat, _, vcov_original = _solve_ols_linalg(
        X, y, cluster_ids=cluster_ids, return_vcov=True
    )
    original_coef = beta_hat[coefficient_index]
    se_original = np.sqrt(vcov_original[coefficient_index, coefficient_index])
    t_stat_original = (original_coef - null_hypothesis) / se_original

    # Step 2: Impose null hypothesis (restricted estimation)
    # Create restricted y: y_restricted = y - X[:, coef_index] * null_hypothesis
    # This imposes the null that the coefficient equals null_hypothesis
    y_restricted = y - X[:, coefficient_index] * null_hypothesis

    # Fit restricted model (but we need to drop the column for the restricted coef)
    # Actually, for WCR bootstrap we keep all columns but impose the null via residuals
    # Re-estimate with the restricted dependent variable
    beta_restricted, residuals_restricted, _ = _solve_ols_linalg(
        X, y_restricted, return_vcov=False
    )

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
        y_star = X @ beta_restricted + residuals_restricted * obs_weights

        # Estimate bootstrap coefficients with cluster-robust SE
        beta_star, residuals_star, vcov_star = _solve_ols_linalg(
            X, y_star, cluster_ids=cluster_ids, return_vcov=True
        )
        bootstrap_coefs[b] = beta_star[coefficient_index]
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
        bootstrap_distribution=bootstrap_coefs if return_distribution else None
    )


def check_parallel_trends(
    data: pd.DataFrame,
    outcome: str,
    time: str,
    treatment_group: str,
    pre_periods: Optional[List[Any]] = None
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
        mse = np.sum(residuals ** 2) / (n - 2)
        se_slope = np.sqrt(mse / time_var)

        return slope, se_slope

    treated_slope, treated_se = compute_trend(treated_data)
    control_slope, control_se = compute_trend(control_data)

    # Test for difference in trends
    slope_diff = treated_slope - control_slope
    se_diff = np.sqrt(treated_se ** 2 + control_se ** 2)
    t_stat = slope_diff / se_diff if se_diff > 0 else np.nan
    p_value = compute_p_value(t_stat) if not np.isnan(t_stat) else np.nan

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
    wasserstein_threshold: float = 0.2
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
        wasserstein_p > 0.05 and
        (wasserstein_normalized < wasserstein_threshold if not np.isnan(wasserstein_normalized) else True)
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
    unit: Optional[str] = None
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

        treated_changes = changes_data[
            changes_data[treatment_group] == 1
        ]["_outcome_change"].values

        control_changes = changes_data[
            changes_data[treatment_group] == 0
        ]["_outcome_change"].values
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
    equivalence_margin: Optional[float] = None
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
    numerator = (var_t/n_t + var_c/n_c)**2
    denom_t = (var_t/n_t)**2/(n_t-1) if var_t > 0 else 0
    denom_c = (var_c/n_c)**2/(n_c-1) if var_c > 0 else 0
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
    Y_control: np.ndarray,
    Y_treated: np.ndarray,
    lambda_reg: float = 0.0,
    min_weight: float = 1e-6
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

    # Set small weights to zero for interpretability
    weights[weights < min_weight] = 0
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
    else:
        # Fallback to uniform if all weights are zeroed
        weights = np.ones(n_control) / n_control

    return np.asarray(weights)


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


def compute_time_weights(
    Y_control: np.ndarray,
    Y_treated: np.ndarray,
    zeta: float = 1.0
) -> np.ndarray:
    """
    Compute time weights for synthetic DiD.

    Time weights emphasize pre-treatment periods where the outcome
    is more informative for constructing the synthetic control.
    Based on the SDID approach from Arkhangelsky et al. (2021).

    Parameters
    ----------
    Y_control : np.ndarray
        Control unit outcomes of shape (n_pre_periods, n_control_units).
    Y_treated : np.ndarray
        Treated unit mean outcomes of shape (n_pre_periods,).
    zeta : float, default=1.0
        Regularization parameter for time weights. Higher values
        give more uniform weights.

    Returns
    -------
    np.ndarray
        Time weights of shape (n_pre_periods,) that sum to 1.

    Notes
    -----
    The time weights help interpolate between DiD (uniform weights)
    and synthetic control (weights concentrated on similar periods).
    """
    n_pre = len(Y_treated)

    if n_pre <= 1:
        return np.asarray(np.ones(n_pre))

    # Compute mean control outcomes per period
    control_means = np.mean(Y_control, axis=1)

    # Compute differences from treated
    diffs = np.abs(Y_treated - control_means)

    # Inverse weighting: periods with smaller differences get higher weight
    # Add regularization to prevent extreme weights
    inv_diffs = 1.0 / (diffs + zeta * np.std(diffs) + _NUMERICAL_EPS)

    # Normalize to sum to 1
    weights = inv_diffs / np.sum(inv_diffs)

    return np.asarray(weights)


def compute_sdid_estimator(
    Y_pre_control: np.ndarray,
    Y_post_control: np.ndarray,
    Y_pre_treated: np.ndarray,
    Y_post_treated: np.ndarray,
    unit_weights: np.ndarray,
    time_weights: np.ndarray
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


def compute_placebo_effects(
    Y_pre_control: np.ndarray,
    Y_post_control: np.ndarray,
    Y_pre_treated: np.ndarray,
    unit_weights: np.ndarray,
    time_weights: np.ndarray,
    control_unit_ids: List[Any],
    n_placebo: Optional[int] = None
) -> np.ndarray:
    """
    Compute placebo treatment effects by treating each control as treated.

    Used for inference in synthetic DiD when bootstrap is not appropriate.

    Parameters
    ----------
    Y_pre_control : np.ndarray
        Control outcomes in pre-treatment periods, shape (n_pre, n_control).
    Y_post_control : np.ndarray
        Control outcomes in post-treatment periods, shape (n_post, n_control).
    Y_pre_treated : np.ndarray
        Treated outcomes in pre-treatment periods, shape (n_pre,).
    unit_weights : np.ndarray
        Unit weights, shape (n_control,).
    time_weights : np.ndarray
        Time weights, shape (n_pre,).
    control_unit_ids : list
        List of control unit identifiers.
    n_placebo : int, optional
        Number of placebo tests. If None, uses all control units.

    Returns
    -------
    np.ndarray
        Array of placebo treatment effects.

    Notes
    -----
    For each control unit j, we pretend it was treated and compute
    the SDID estimate using the remaining controls. The distribution
    of these placebo effects provides a reference for inference.
    """
    n_pre, n_control = Y_pre_control.shape

    if n_placebo is None:
        n_placebo = n_control

    placebo_effects = []

    for j in range(min(n_placebo, n_control)):
        # Treat unit j as the "treated" unit
        Y_pre_placebo_treated = Y_pre_control[:, j]
        Y_post_placebo_treated = Y_post_control[:, j]

        # Use remaining units as controls
        remaining_idx = [i for i in range(n_control) if i != j]

        if len(remaining_idx) == 0:
            continue

        Y_pre_remaining = Y_pre_control[:, remaining_idx]
        Y_post_remaining = Y_post_control[:, remaining_idx]

        # Recompute weights for remaining controls
        remaining_weights = compute_synthetic_weights(
            Y_pre_remaining,
            Y_pre_placebo_treated
        )

        # Compute placebo effect
        placebo_tau = compute_sdid_estimator(
            Y_pre_remaining,
            Y_post_remaining,
            Y_pre_placebo_treated,
            Y_post_placebo_treated,
            remaining_weights,
            time_weights
        )

        placebo_effects.append(placebo_tau)

    return np.asarray(placebo_effects)
