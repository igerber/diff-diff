"""
Honest DiD sensitivity analysis (Rambachan & Roth 2023).

Provides robust inference for difference-in-differences designs when
parallel trends may be violated. Instead of assuming parallel trends
holds exactly, this module allows for bounded violations and computes
partially identified treatment effect bounds.

References
----------
Rambachan, A., & Roth, J. (2023). A More Credible Approach to Parallel Trends.
The Review of Economic Studies, 90(5), 2555-2591.
https://doi.org/10.1093/restud/rdad018

See Also
--------
https://github.com/asheshrambachan/HonestDiD - R package implementation
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import optimize, stats

from diff_diff.results import (
    MultiPeriodDiDResults,
)

# =============================================================================
# Delta Restriction Classes
# =============================================================================


@dataclass
class DeltaSD:
    """
    Smoothness restriction on trend violations (Delta^{SD}).

    Restricts the second differences of the trend violations:
        |delta_{t+1} - 2*delta_t + delta_{t-1}| <= M

    When M=0, this enforces that violations follow a linear trend
    (linear extrapolation of pre-trends). Larger M allows more
    curvature in the violation path.

    Parameters
    ----------
    M : float
        Maximum allowed second difference. M=0 means linear trends only.

    Examples
    --------
    >>> delta = DeltaSD(M=0.5)
    >>> delta.M
    0.5
    """

    M: float = 0.0

    def __post_init__(self):
        if self.M < 0:
            raise ValueError(f"M must be non-negative, got M={self.M}")

    def __repr__(self) -> str:
        return f"DeltaSD(M={self.M})"


@dataclass
class DeltaRM:
    """
    Relative magnitudes restriction on trend violations (Delta^{RM}).

    Post-treatment violations are bounded by Mbar times the maximum
    absolute pre-treatment violation:
        |delta_post| <= Mbar * max(|delta_pre|)

    When Mbar=0, this enforces exact parallel trends post-treatment.
    Mbar=1 means post-period violations can be as large as the worst
    observed pre-period violation.

    Parameters
    ----------
    Mbar : float
        Scaling factor for maximum pre-period violation.

    Examples
    --------
    >>> delta = DeltaRM(Mbar=1.0)
    >>> delta.Mbar
    1.0
    """

    Mbar: float = 1.0

    def __post_init__(self):
        if self.Mbar < 0:
            raise ValueError(f"Mbar must be non-negative, got Mbar={self.Mbar}")

    def __repr__(self) -> str:
        return f"DeltaRM(Mbar={self.Mbar})"


@dataclass
class DeltaSDRM:
    """
    Combined smoothness and relative magnitudes restriction.

    Imposes both:
    1. Smoothness: |delta_{t+1} - 2*delta_t + delta_{t-1}| <= M
    2. Relative magnitudes: |delta_post| <= Mbar * max(|delta_pre|)

    This is more restrictive than either constraint alone.

    Parameters
    ----------
    M : float
        Maximum allowed second difference (smoothness).
    Mbar : float
        Scaling factor for maximum pre-period violation (relative magnitudes).

    Examples
    --------
    >>> delta = DeltaSDRM(M=0.5, Mbar=1.0)
    """

    M: float = 0.0
    Mbar: float = 1.0

    def __post_init__(self):
        if self.M < 0:
            raise ValueError(f"M must be non-negative, got M={self.M}")
        if self.Mbar < 0:
            raise ValueError(f"Mbar must be non-negative, got Mbar={self.Mbar}")

    def __repr__(self) -> str:
        return f"DeltaSDRM(M={self.M}, Mbar={self.Mbar})"


DeltaType = Union[DeltaSD, DeltaRM, DeltaSDRM]


# =============================================================================
# Results Classes
# =============================================================================


@dataclass
class HonestDiDResults:
    """
    Results from Honest DiD sensitivity analysis.

    Contains bounds on the treatment effect under the specified
    restrictions on violations of parallel trends.

    Attributes
    ----------
    lb : float
        Lower bound of identified set.
    ub : float
        Upper bound of identified set.
    ci_lb : float
        Lower bound of robust confidence interval.
    ci_ub : float
        Upper bound of robust confidence interval.
    M : float
        The restriction parameter value used.
    method : str
        The type of restriction ("smoothness", "relative_magnitude", or "combined").
    original_estimate : float
        The original point estimate (under parallel trends).
    original_se : float
        The original standard error.
    alpha : float
        Significance level for confidence interval.
    ci_method : str
        Method used for CI construction ("FLCI" or "C-LF").
    original_results : Any
        The original estimation results object.
    """

    lb: float
    ub: float
    ci_lb: float
    ci_ub: float
    M: float
    method: str
    original_estimate: float
    original_se: float
    alpha: float = 0.05
    ci_method: str = "FLCI"
    original_results: Optional[Any] = field(default=None, repr=False)
    # Event study bounds (optional)
    event_study_bounds: Optional[Dict[Any, Dict[str, float]]] = field(
        default=None, repr=False
    )

    def __repr__(self) -> str:
        sig = "" if self.ci_lb <= 0 <= self.ci_ub else "*"
        return (
            f"HonestDiDResults(bounds=[{self.lb:.4f}, {self.ub:.4f}], "
            f"CI=[{self.ci_lb:.4f}, {self.ci_ub:.4f}]{sig}, "
            f"M={self.M})"
        )

    @property
    def is_significant(self) -> bool:
        """Check if CI excludes zero (effect is robust to violations)."""
        return not (self.ci_lb <= 0 <= self.ci_ub)

    @property
    def significance_stars(self) -> str:
        """
        Return significance indicator if robust CI excludes zero.

        Note: Unlike point estimation, partial identification does not yield
        a single p-value. This returns "*" if the robust CI excludes zero
        at the specified alpha level, indicating the effect is robust to
        the assumed violations of parallel trends.
        """
        return "*" if self.is_significant else ""

    @property
    def identified_set_width(self) -> float:
        """Width of the identified set."""
        return self.ub - self.lb

    @property
    def ci_width(self) -> float:
        """Width of the confidence interval."""
        return self.ci_ub - self.ci_lb

    def summary(self) -> str:
        """
        Generate formatted summary of sensitivity analysis results.

        Returns
        -------
        str
            Formatted summary.
        """
        conf_level = int((1 - self.alpha) * 100)

        method_names = {
            "smoothness": "Smoothness (Delta^SD)",
            "relative_magnitude": "Relative Magnitudes (Delta^RM)",
            "combined": "Combined (Delta^SDRM)",
        }
        method_display = method_names.get(self.method, self.method)

        lines = [
            "=" * 70,
            "Honest DiD Sensitivity Analysis Results".center(70),
            "(Rambachan & Roth 2023)".center(70),
            "=" * 70,
            "",
            f"{'Method:':<30} {method_display}",
            f"{'Restriction parameter (M):':<30} {self.M:.4f}",
            f"{'CI method:':<30} {self.ci_method}",
            "",
            "-" * 70,
            "Original Estimate (under parallel trends)".center(70),
            "-" * 70,
            f"{'Point estimate:':<30} {self.original_estimate:.4f}",
            f"{'Standard error:':<30} {self.original_se:.4f}",
            "",
            "-" * 70,
            "Robust Results (allowing for violations)".center(70),
            "-" * 70,
            f"{'Identified set:':<30} [{self.lb:.4f}, {self.ub:.4f}]",
            f"{f'{conf_level}% Robust CI:':<30} [{self.ci_lb:.4f}, {self.ci_ub:.4f}]",
            "",
            f"{'Effect robust to violations:':<30} {'Yes' if self.is_significant else 'No'}",
            "",
        ]

        # Interpretation
        lines.extend([
            "-" * 70,
            "Interpretation".center(70),
            "-" * 70,
        ])

        if self.method == "relative_magnitude":
            lines.append(
                f"Post-treatment violations bounded at {self.M:.1f}x max pre-period violation."
            )
        elif self.method == "smoothness":
            if self.M == 0:
                lines.append("Violations follow linear extrapolation of pre-trends.")
            else:
                lines.append(
                    f"Violation curvature (second diff) bounded by {self.M:.4f} per period."
                )
        else:
            lines.append(
                f"Combined smoothness (M={self.M:.2f}) and relative magnitude bounds."
            )

        if self.is_significant:
            if self.ci_lb > 0:
                lines.append(f"Effect remains POSITIVE even with violations up to M={self.M}.")
            else:
                lines.append(f"Effect remains NEGATIVE even with violations up to M={self.M}.")
        else:
            lines.append(
                f"Cannot rule out zero effect when allowing violations up to M={self.M}."
            )

        lines.extend(["", "=" * 70])

        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print summary to stdout."""
        print(self.summary())

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            "lb": self.lb,
            "ub": self.ub,
            "ci_lb": self.ci_lb,
            "ci_ub": self.ci_ub,
            "M": self.M,
            "method": self.method,
            "original_estimate": self.original_estimate,
            "original_se": self.original_se,
            "alpha": self.alpha,
            "ci_method": self.ci_method,
            "is_significant": self.is_significant,
            "identified_set_width": self.identified_set_width,
            "ci_width": self.ci_width,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        return pd.DataFrame([self.to_dict()])


@dataclass
class SensitivityResults:
    """
    Results from sensitivity analysis over a grid of M values.

    Contains bounds and confidence intervals for each M value,
    plus the breakdown value.

    Attributes
    ----------
    M_values : np.ndarray
        Grid of M parameter values.
    bounds : List[Tuple[float, float]]
        List of (lb, ub) identified set bounds for each M.
    robust_cis : List[Tuple[float, float]]
        List of (ci_lb, ci_ub) robust CIs for each M.
    breakdown_M : float
        Smallest M where robust CI includes zero.
    method : str
        Type of restriction used.
    original_estimate : float
        Original point estimate.
    original_se : float
        Original standard error.
    alpha : float
        Significance level.
    """

    M_values: np.ndarray
    bounds: List[Tuple[float, float]]
    robust_cis: List[Tuple[float, float]]
    breakdown_M: Optional[float]
    method: str
    original_estimate: float
    original_se: float
    alpha: float = 0.05

    def __repr__(self) -> str:
        breakdown_str = f"{self.breakdown_M:.4f}" if self.breakdown_M else "None"
        return (
            f"SensitivityResults(n_M={len(self.M_values)}, "
            f"breakdown_M={breakdown_str})"
        )

    @property
    def has_breakdown(self) -> bool:
        """Check if there is a finite breakdown value."""
        return self.breakdown_M is not None

    def summary(self) -> str:
        """Generate formatted summary."""
        lines = [
            "=" * 70,
            "Honest DiD Sensitivity Analysis".center(70),
            "=" * 70,
            "",
            f"{'Method:':<30} {self.method}",
            f"{'Original estimate:':<30} {self.original_estimate:.4f}",
            f"{'Original SE:':<30} {self.original_se:.4f}",
            f"{'M values tested:':<30} {len(self.M_values)}",
            "",
        ]

        if self.breakdown_M is not None:
            lines.append(f"{'Breakdown value:':<30} {self.breakdown_M:.4f}")
            lines.append("")
            lines.append(
                f"Result is robust to violations up to M = {self.breakdown_M:.4f}"
            )
        else:
            lines.append(f"{'Breakdown value:':<30} None (always significant)")

        lines.extend([
            "",
            "-" * 70,
            f"{'M':<10} {'Lower Bound':>12} {'Upper Bound':>12} {'CI Lower':>12} {'CI Upper':>12}",
            "-" * 70,
        ])

        for i, M in enumerate(self.M_values):
            lb, ub = self.bounds[i]
            ci_lb, ci_ub = self.robust_cis[i]
            lines.append(f"{M:<10.4f} {lb:>12.4f} {ub:>12.4f} {ci_lb:>12.4f} {ci_ub:>12.4f}")

        lines.extend(["", "=" * 70])

        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print summary to stdout."""
        print(self.summary())

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with one row per M value."""
        rows = []
        for i, M in enumerate(self.M_values):
            lb, ub = self.bounds[i]
            ci_lb, ci_ub = self.robust_cis[i]
            rows.append({
                "M": M,
                "lb": lb,
                "ub": ub,
                "ci_lb": ci_lb,
                "ci_ub": ci_ub,
                "is_significant": not (ci_lb <= 0 <= ci_ub),
            })
        return pd.DataFrame(rows)

    def plot(self, ax=None, show_bounds: bool = True, show_ci: bool = True,
             breakdown_line: bool = True, **kwargs):
        """
        Plot sensitivity analysis results.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        show_bounds : bool
            Whether to show identified set bounds.
        show_ci : bool
            Whether to show confidence intervals.
        breakdown_line : bool
            Whether to show vertical line at breakdown value.
        **kwargs
            Additional arguments passed to plotting functions.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the plot.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting")

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        M = self.M_values
        bounds_arr = np.array(self.bounds)
        ci_arr = np.array(self.robust_cis)

        # Plot original estimate
        ax.axhline(y=self.original_estimate, color='black', linestyle='-',
                   linewidth=1.5, label='Original estimate', alpha=0.7)

        # Plot zero line
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

        if show_bounds:
            ax.fill_between(M, bounds_arr[:, 0], bounds_arr[:, 1],
                            alpha=0.3, color='blue', label='Identified set')

        if show_ci:
            ax.plot(M, ci_arr[:, 0], 'b-', linewidth=1.5, label='Robust CI')
            ax.plot(M, ci_arr[:, 1], 'b-', linewidth=1.5)

        if breakdown_line and self.breakdown_M is not None:
            ax.axvline(x=self.breakdown_M, color='red', linestyle=':',
                       linewidth=2, label=f'Breakdown (M={self.breakdown_M:.2f})')

        ax.set_xlabel('M (restriction parameter)')
        ax.set_ylabel('Treatment Effect')
        ax.set_title('Sensitivity Analysis: Treatment Effect Bounds')
        ax.legend(loc='best')

        return ax


# =============================================================================
# Helper Functions
# =============================================================================


def _extract_event_study_params(
    results: Union[MultiPeriodDiDResults, Any]
) -> Tuple[np.ndarray, np.ndarray, int, int, List[Any], List[Any]]:
    """
    Extract event study parameters from results objects.

    Parameters
    ----------
    results : MultiPeriodDiDResults or CallawaySantAnnaResults
        Estimation results with event study structure.

    Returns
    -------
    beta_hat : np.ndarray
        Vector of event study coefficients (pre + post periods).
    sigma : np.ndarray
        Variance-covariance matrix of coefficients.
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.
    pre_periods : list
        Pre-period identifiers.
    post_periods : list
        Post-period identifiers.
    """
    if isinstance(results, MultiPeriodDiDResults):
        # Extract from MultiPeriodDiD
        pre_periods = results.pre_periods
        post_periods = results.post_periods

        # Get coefficients - need to extract from period_effects
        # Note: MultiPeriodDiD stores effects for post-periods only in period_effects
        # Pre-period effects would be in the coefficients dict if estimated
        effects = []
        ses = []

        # For now, we'll work with post-period effects
        # In a full event study, we'd also have pre-period coefficients
        for period in post_periods:
            pe = results.period_effects[period]
            effects.append(pe.effect)
            ses.append(pe.se)

        beta_hat = np.array(effects)
        num_post_periods = len(post_periods)
        num_pre_periods = len(pre_periods) if pre_periods else 0

        # Get vcov if available
        if results.vcov is not None:
            sigma = results.vcov
        else:
            # Construct diagonal vcov from SEs
            sigma = np.diag(np.array(ses) ** 2)

        return beta_hat, sigma, num_pre_periods, num_post_periods, pre_periods, post_periods

    else:
        # Try CallawaySantAnnaResults
        try:
            from diff_diff.staggered import CallawaySantAnnaResults
            if isinstance(results, CallawaySantAnnaResults):
                if results.event_study_effects is None:
                    raise ValueError(
                        "CallawaySantAnnaResults must have event_study_effects for HonestDiD. "
                        "Re-run CallawaySantAnna.fit() with aggregate='event_study' to compute "
                        "event study effects."
                    )

                # Extract event study effects by relative time
                event_effects = results.event_study_effects
                rel_times = sorted(event_effects.keys())

                # Split into pre and post
                pre_times = [t for t in rel_times if t < 0]
                post_times = [t for t in rel_times if t >= 0]

                effects = []
                ses = []
                for t in rel_times:
                    effects.append(event_effects[t]['effect'])
                    ses.append(event_effects[t]['se'])

                beta_hat = np.array(effects)
                sigma = np.diag(np.array(ses) ** 2)

                return (
                    beta_hat, sigma,
                    len(pre_times), len(post_times),
                    pre_times, post_times
                )
        except ImportError:
            pass

        raise TypeError(
            f"Unsupported results type: {type(results)}. "
            "Expected MultiPeriodDiDResults or CallawaySantAnnaResults."
        )


def _construct_A_sd(num_periods: int) -> np.ndarray:
    """
    Construct constraint matrix for smoothness (second differences).

    For T periods, creates matrix A such that:
    A @ delta gives the second differences.

    Parameters
    ----------
    num_periods : int
        Number of time periods.

    Returns
    -------
    A : np.ndarray
        Constraint matrix of shape (num_periods - 2, num_periods).
    """
    if num_periods < 3:
        return np.zeros((0, num_periods))

    n_constraints = num_periods - 2
    A = np.zeros((n_constraints, num_periods))

    for i in range(n_constraints):
        # Second difference: delta_{t+1} - 2*delta_t + delta_{t-1}
        A[i, i] = 1      # delta_{t-1}
        A[i, i + 1] = -2  # delta_t
        A[i, i + 2] = 1   # delta_{t+1}

    return A


def _construct_constraints_sd(
    num_pre_periods: int,
    num_post_periods: int,
    M: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct smoothness constraint matrices.

    Returns A, b such that delta in DeltaSD iff |A @ delta| <= b.

    Parameters
    ----------
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.
    M : float
        Smoothness parameter.

    Returns
    -------
    A_ineq : np.ndarray
        Inequality constraint matrix.
    b_ineq : np.ndarray
        Inequality constraint vector.
    """
    total_periods = num_pre_periods + num_post_periods
    A_base = _construct_A_sd(total_periods)

    if A_base.shape[0] == 0:
        return np.zeros((0, total_periods)), np.zeros(0)

    # |A @ delta| <= M becomes:
    # A @ delta <= M  and  -A @ delta <= M
    A_ineq = np.vstack([A_base, -A_base])
    b_ineq = np.full(2 * A_base.shape[0], M)

    return A_ineq, b_ineq


def _construct_constraints_rm(
    num_pre_periods: int,
    num_post_periods: int,
    Mbar: float,
    max_pre_violation: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct relative magnitudes constraint matrices.

    Parameters
    ----------
    num_pre_periods : int
        Number of pre-treatment periods.
    num_post_periods : int
        Number of post-treatment periods.
    Mbar : float
        Relative magnitude scaling factor.
    max_pre_violation : float
        Maximum absolute pre-period violation (estimated from data).

    Returns
    -------
    A_ineq : np.ndarray
        Inequality constraint matrix.
    b_ineq : np.ndarray
        Inequality constraint vector.
    """
    total_periods = num_pre_periods + num_post_periods

    # Bound post-period violations: |delta_post| <= Mbar * max_pre_violation
    bound = Mbar * max_pre_violation

    # Create constraints for each post-period
    # delta_post[i] <= bound  and  -delta_post[i] <= bound
    n_constraints = 2 * num_post_periods
    A_ineq = np.zeros((n_constraints, total_periods))
    b_ineq = np.full(n_constraints, bound)

    for i in range(num_post_periods):
        post_idx = num_pre_periods + i
        A_ineq[2 * i, post_idx] = 1      # delta <= bound
        A_ineq[2 * i + 1, post_idx] = -1  # -delta <= bound

    return A_ineq, b_ineq


def _solve_bounds_lp(
    beta_post: np.ndarray,
    l_vec: np.ndarray,
    A_ineq: np.ndarray,
    b_ineq: np.ndarray,
    num_pre_periods: int,
    lp_method: str = 'highs'
) -> Tuple[float, float]:
    """
    Solve for identified set bounds using linear programming.

    The parameter of interest is theta = l' @ (beta_post - delta_post).
    We find min and max over delta in the constraint set.

    Note: The optimization is over delta for ALL periods (pre + post), but
    only the post-period components contribute to the objective function.
    This correctly handles smoothness constraints that link pre and post periods.

    Parameters
    ----------
    beta_post : np.ndarray
        Post-period coefficient estimates.
    l_vec : np.ndarray
        Weighting vector for aggregation.
    A_ineq : np.ndarray
        Inequality constraint matrix (for all periods).
    b_ineq : np.ndarray
        Inequality constraint vector.
    num_pre_periods : int
        Number of pre-periods (for indexing).
    lp_method : str
        LP solver method for scipy.optimize.linprog. Default 'highs' requires
        scipy >= 1.6.0. Alternatives: 'interior-point', 'revised simplex'.

    Returns
    -------
    lb : float
        Lower bound.
    ub : float
        Upper bound.
    """
    num_post = len(beta_post)
    total_periods = A_ineq.shape[1] if A_ineq.shape[0] > 0 else num_pre_periods + num_post

    # theta = l' @ beta_post - l' @ delta_post
    # We optimize over delta (all periods including pre for smoothness constraints)

    # Extract post-period part of constraints
    # For delta in R^total_periods, we want min/max of -l' @ delta_post
    # where delta_post = delta[num_pre_periods:]

    c = np.zeros(total_periods)
    c[num_pre_periods:num_pre_periods + num_post] = -l_vec  # min -l'@delta = max l'@delta

    # For upper bound: max l'@(beta - delta) = l'@beta + max(-l'@delta)
    # For lower bound: min l'@(beta - delta) = l'@beta + min(-l'@delta)

    if A_ineq.shape[0] == 0:
        # No constraints - unbounded
        return -np.inf, np.inf

    # Solve for lower bound of -l'@delta (which gives upper bound of theta)
    try:
        result_min = optimize.linprog(
            c, A_ub=A_ineq, b_ub=b_ineq,
            bounds=(None, None),
            method=lp_method
        )
        if result_min.success:
            min_val = result_min.fun
        else:
            min_val = -np.inf
    except Exception:
        min_val = -np.inf

    # Solve for upper bound of -l'@delta (which gives lower bound of theta)
    try:
        result_max = optimize.linprog(
            -c, A_ub=A_ineq, b_ub=b_ineq,
            bounds=(None, None),
            method=lp_method
        )
        if result_max.success:
            max_val = -result_max.fun
        else:
            max_val = np.inf
    except Exception:
        max_val = np.inf

    theta_base = np.dot(l_vec, beta_post)
    lb = theta_base + min_val  # = l'@beta + min(-l'@delta) = min(l'@(beta-delta))
    ub = theta_base + max_val  # = l'@beta + max(-l'@delta) = max(l'@(beta-delta))

    return lb, ub


def _compute_flci(
    lb: float,
    ub: float,
    se: float,
    alpha: float = 0.05
) -> Tuple[float, float]:
    """
    Compute Fixed Length Confidence Interval (FLCI).

    The FLCI extends the identified set by a critical value times
    the standard error on each side.

    Parameters
    ----------
    lb : float
        Lower bound of identified set.
    ub : float
        Upper bound of identified set.
    se : float
        Standard error of the estimator.
    alpha : float
        Significance level.

    Returns
    -------
    ci_lb : float
        Lower bound of confidence interval.
    ci_ub : float
        Upper bound of confidence interval.

    Raises
    ------
    ValueError
        If se <= 0 or alpha is not in (0, 1).
    """
    if se <= 0:
        raise ValueError(f"Standard error must be positive, got se={se}")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be between 0 and 1, got alpha={alpha}")

    z = stats.norm.ppf(1 - alpha / 2)
    ci_lb = lb - z * se
    ci_ub = ub + z * se
    return ci_lb, ci_ub


def _compute_clf_ci(
    beta_post: np.ndarray,
    sigma_post: np.ndarray,
    l_vec: np.ndarray,
    Mbar: float,
    max_pre_violation: float,
    alpha: float = 0.05,
    n_draws: int = 1000
) -> Tuple[float, float, float, float]:
    """
    Compute Conditional Least Favorable (C-LF) confidence interval.

    For relative magnitudes, accounts for estimation of max_pre_violation.

    Parameters
    ----------
    beta_post : np.ndarray
        Post-period coefficient estimates.
    sigma_post : np.ndarray
        Variance-covariance matrix for post-period coefficients.
    l_vec : np.ndarray
        Weighting vector.
    Mbar : float
        Relative magnitude parameter.
    max_pre_violation : float
        Estimated max pre-period violation.
    alpha : float
        Significance level.
    n_draws : int
        Number of Monte Carlo draws for conditional CI.

    Returns
    -------
    lb : float
        Lower bound of identified set.
    ub : float
        Upper bound of identified set.
    ci_lb : float
        Lower bound of confidence interval.
    ci_ub : float
        Upper bound of confidence interval.
    """
    # For simplicity, use FLCI approach with adjustment for estimation uncertainty
    # A full implementation would condition on the estimated max_pre_violation

    theta = np.dot(l_vec, beta_post)
    se = np.sqrt(l_vec @ sigma_post @ l_vec)

    bound = Mbar * max_pre_violation

    # Simple bounds: theta +/- bound
    lb = theta - bound
    ub = theta + bound

    # CI with estimation uncertainty
    z = stats.norm.ppf(1 - alpha / 2)
    ci_lb = lb - z * se
    ci_ub = ub + z * se

    return lb, ub, ci_lb, ci_ub


# =============================================================================
# Main Class
# =============================================================================


class HonestDiD:
    """
    Honest DiD sensitivity analysis (Rambachan & Roth 2023).

    Computes robust inference for difference-in-differences allowing
    for bounded violations of parallel trends.

    Parameters
    ----------
    method : {"smoothness", "relative_magnitude", "combined"}
        Type of restriction on trend violations:
        - "smoothness": Bounds on second differences (Delta^SD)
        - "relative_magnitude": Post violations <= M * max pre violation (Delta^RM)
        - "combined": Both restrictions (Delta^SDRM)
    M : float, optional
        Restriction parameter. Interpretation depends on method:
        - smoothness: Max second difference
        - relative_magnitude: Scaling factor for max pre-period violation
        Default is 1.0 for relative_magnitude, 0.0 for smoothness.
    alpha : float
        Significance level for confidence intervals.
    l_vec : array-like or None
        Weighting vector for scalar parameter (length = num_post_periods).
        If None, uses uniform weights (average effect).

    Examples
    --------
    >>> from diff_diff import MultiPeriodDiD
    >>> from diff_diff.honest_did import HonestDiD
    >>>
    >>> # Fit event study
    >>> mp_did = MultiPeriodDiD()
    >>> results = mp_did.fit(data, outcome='y', treatment='treated',
    ...                      time='period', post_periods=[4,5,6,7])
    >>>
    >>> # Sensitivity analysis with relative magnitudes
    >>> honest = HonestDiD(method='relative_magnitude', M=1.0)
    >>> bounds = honest.fit(results)
    >>> print(bounds.summary())
    >>>
    >>> # Sensitivity curve over M values
    >>> sensitivity = honest.sensitivity_analysis(results, M_grid=[0, 0.5, 1, 1.5, 2])
    >>> sensitivity.plot()
    """

    def __init__(
        self,
        method: Literal["smoothness", "relative_magnitude", "combined"] = "relative_magnitude",
        M: Optional[float] = None,
        alpha: float = 0.05,
        l_vec: Optional[np.ndarray] = None,
    ):
        self.method = method
        self.alpha = alpha
        self.l_vec = l_vec

        # Set default M based on method
        if M is None:
            self.M = 1.0 if method == "relative_magnitude" else 0.0
        else:
            self.M = M

        self._validate_params()

    def _validate_params(self):
        """Validate initialization parameters."""
        if self.method not in ["smoothness", "relative_magnitude", "combined"]:
            raise ValueError(
                f"method must be 'smoothness', 'relative_magnitude', or 'combined', "
                f"got method='{self.method}'"
            )
        if self.M < 0:
            raise ValueError(f"M must be non-negative, got M={self.M}")
        if not 0 < self.alpha < 1:
            raise ValueError(f"alpha must be between 0 and 1, got alpha={self.alpha}")

    def get_params(self) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        return {
            "method": self.method,
            "M": self.M,
            "alpha": self.alpha,
            "l_vec": self.l_vec,
        }

    def set_params(self, **params) -> "HonestDiD":
        """Set parameters for this estimator."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        self._validate_params()
        return self

    def fit(
        self,
        results: Union[MultiPeriodDiDResults, Any],
        M: Optional[float] = None,
    ) -> HonestDiDResults:
        """
        Compute bounds and robust confidence intervals.

        Parameters
        ----------
        results : MultiPeriodDiDResults or CallawaySantAnnaResults
            Results from event study estimation.
        M : float, optional
            Override the M parameter for this fit.

        Returns
        -------
        HonestDiDResults
            Results containing bounds and robust confidence intervals.
        """
        M = M if M is not None else self.M

        # Extract event study parameters
        (beta_hat, sigma, num_pre, num_post,
         pre_periods, post_periods) = _extract_event_study_params(results)

        # beta_hat from MultiPeriodDiDResults already contains only post-periods
        # Check if we have the right number of coefficients
        if len(beta_hat) == num_post:
            # Already just post-period effects
            beta_post = beta_hat
        elif len(beta_hat) == num_pre + num_post:
            # Full event study, extract post-periods
            beta_post = beta_hat[num_pre:]
        else:
            # Assume it's post-period effects
            beta_post = beta_hat
            num_post = len(beta_hat)

        # Handle sigma extraction for post periods
        if sigma.shape[0] == num_post and sigma.shape[0] == len(beta_post):
            sigma_post = sigma
        elif sigma.shape[0] == num_pre + num_post:
            sigma_post = sigma[num_pre:, num_pre:]
        else:
            # Construct diagonal from available dimensions
            sigma_post = sigma[:len(beta_post), :len(beta_post)]

        # Update num_post to match actual data
        num_post = len(beta_post)

        # Set up weighting vector
        if self.l_vec is None:
            l_vec = np.ones(num_post) / num_post  # Uniform weights
        else:
            l_vec = np.asarray(self.l_vec)
            if len(l_vec) != num_post:
                raise ValueError(
                    f"l_vec must have length {num_post}, got {len(l_vec)}"
                )

        # Compute original estimate and SE
        original_estimate = np.dot(l_vec, beta_post)
        original_se = np.sqrt(l_vec @ sigma_post @ l_vec)

        # Compute bounds based on method
        if self.method == "smoothness":
            lb, ub, ci_lb, ci_ub = self._compute_smoothness_bounds(
                beta_post, sigma_post, l_vec, num_pre, num_post, M
            )
            ci_method = "FLCI"

        elif self.method == "relative_magnitude":
            lb, ub, ci_lb, ci_ub = self._compute_rm_bounds(
                beta_post, sigma_post, l_vec, num_pre, num_post, M,
                pre_periods, results
            )
            ci_method = "C-LF"

        else:  # combined
            lb, ub, ci_lb, ci_ub = self._compute_combined_bounds(
                beta_post, sigma_post, l_vec, num_pre, num_post, M,
                pre_periods, results
            )
            ci_method = "FLCI"

        return HonestDiDResults(
            lb=lb,
            ub=ub,
            ci_lb=ci_lb,
            ci_ub=ci_ub,
            M=M,
            method=self.method,
            original_estimate=original_estimate,
            original_se=original_se,
            alpha=self.alpha,
            ci_method=ci_method,
            original_results=results,
        )

    def _compute_smoothness_bounds(
        self,
        beta_post: np.ndarray,
        sigma_post: np.ndarray,
        l_vec: np.ndarray,
        num_pre: int,
        num_post: int,
        M: float
    ) -> Tuple[float, float, float, float]:
        """Compute bounds under smoothness restriction."""
        # Construct constraints
        A_ineq, b_ineq = _construct_constraints_sd(num_pre, num_post, M)

        # Solve for bounds
        lb, ub = _solve_bounds_lp(beta_post, l_vec, A_ineq, b_ineq, num_pre)

        # Compute FLCI
        se = np.sqrt(l_vec @ sigma_post @ l_vec)
        ci_lb, ci_ub = _compute_flci(lb, ub, se, self.alpha)

        return lb, ub, ci_lb, ci_ub

    def _compute_rm_bounds(
        self,
        beta_post: np.ndarray,
        sigma_post: np.ndarray,
        l_vec: np.ndarray,
        num_pre: int,
        num_post: int,
        Mbar: float,
        pre_periods: List,
        results: Any
    ) -> Tuple[float, float, float, float]:
        """Compute bounds under relative magnitudes restriction."""
        # Estimate max pre-period violation from pre-trends
        # For relative magnitudes, we use the pre-period coefficients
        max_pre_violation = self._estimate_max_pre_violation(results, pre_periods)

        if max_pre_violation == 0:
            # No pre-period violations detected - use point estimate
            theta = np.dot(l_vec, beta_post)
            se = np.sqrt(l_vec @ sigma_post @ l_vec)
            z = stats.norm.ppf(1 - self.alpha / 2)
            return theta, theta, theta - z * se, theta + z * se

        # Compute bounds
        lb, ub, ci_lb, ci_ub = _compute_clf_ci(
            beta_post, sigma_post, l_vec, Mbar, max_pre_violation, self.alpha
        )

        return lb, ub, ci_lb, ci_ub

    def _compute_combined_bounds(
        self,
        beta_post: np.ndarray,
        sigma_post: np.ndarray,
        l_vec: np.ndarray,
        num_pre: int,
        num_post: int,
        M: float,
        pre_periods: List,
        results: Any
    ) -> Tuple[float, float, float, float]:
        """Compute bounds under combined smoothness + RM restriction."""
        # Get smoothness bounds
        lb_sd, ub_sd, _, _ = self._compute_smoothness_bounds(
            beta_post, sigma_post, l_vec, num_pre, num_post, M
        )

        # Get RM bounds (use M as Mbar for combined)
        lb_rm, ub_rm, _, _ = self._compute_rm_bounds(
            beta_post, sigma_post, l_vec, num_pre, num_post, M, pre_periods, results
        )

        # Combined bounds are intersection
        lb = max(lb_sd, lb_rm)
        ub = min(ub_sd, ub_rm)

        # If bounds cross, use the original estimate
        if lb > ub:
            theta = np.dot(l_vec, beta_post)
            lb = ub = theta

        # Compute FLCI on combined bounds
        se = np.sqrt(l_vec @ sigma_post @ l_vec)
        ci_lb, ci_ub = _compute_flci(lb, ub, se, self.alpha)

        return lb, ub, ci_lb, ci_ub

    def _estimate_max_pre_violation(
        self,
        results: Any,
        pre_periods: List
    ) -> float:
        """
        Estimate the maximum pre-period violation.

        Uses pre-period coefficients if available, otherwise returns
        a default based on the overall SE.
        """
        if isinstance(results, MultiPeriodDiDResults):
            # Check if we have pre-period effects
            # In a standard event study, pre-period coefficients should be ~0
            # Their magnitude indicates the pre-trend violation
            if hasattr(results, 'coefficients') and results.coefficients:
                # Look for pre-period coefficients
                pre_effects = []
                for period in pre_periods:
                    key = f"treated:period_{period}"
                    if key in results.coefficients:
                        pre_effects.append(abs(results.coefficients[key]))

                if pre_effects:
                    return max(pre_effects)

            # Fallback: use avg_se as a scale
            return results.avg_se

        # For CallawaySantAnna, use pre-period event study effects
        try:
            from diff_diff.staggered import CallawaySantAnnaResults
            if isinstance(results, CallawaySantAnnaResults):
                if results.event_study_effects:
                    pre_effects = [
                        abs(results.event_study_effects[t]['effect'])
                        for t in results.event_study_effects
                        if t < 0
                    ]
                    if pre_effects:
                        return max(pre_effects)
                return results.overall_se
        except ImportError:
            pass

        # Default fallback
        return 0.1

    def sensitivity_analysis(
        self,
        results: Union[MultiPeriodDiDResults, Any],
        M_grid: Optional[List[float]] = None,
    ) -> SensitivityResults:
        """
        Perform sensitivity analysis over a grid of M values.

        Parameters
        ----------
        results : MultiPeriodDiDResults or CallawaySantAnnaResults
            Results from event study estimation.
        M_grid : list of float, optional
            Grid of M values to evaluate. If None, uses default grid
            based on method.

        Returns
        -------
        SensitivityResults
            Results containing bounds and CIs for each M value.
        """
        if M_grid is None:
            if self.method == "relative_magnitude":
                M_grid = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
            else:
                M_grid = [0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]

        M_values = np.array(M_grid)
        bounds_list = []
        ci_list = []

        for M in M_values:
            result = self.fit(results, M=M)
            bounds_list.append((result.lb, result.ub))
            ci_list.append((result.ci_lb, result.ci_ub))

        # Find breakdown value
        breakdown_M = self._find_breakdown(results, M_values, ci_list)

        # Get original estimate info
        first_result = self.fit(results, M=0)

        return SensitivityResults(
            M_values=M_values,
            bounds=bounds_list,
            robust_cis=ci_list,
            breakdown_M=breakdown_M,
            method=self.method,
            original_estimate=first_result.original_estimate,
            original_se=first_result.original_se,
            alpha=self.alpha,
        )

    def _find_breakdown(
        self,
        results: Any,
        M_values: np.ndarray,
        ci_list: List[Tuple[float, float]]
    ) -> Optional[float]:
        """
        Find the breakdown value where CI first includes zero.

        Uses binary search for precision.
        """
        # Check if any CI includes zero
        includes_zero = [ci_lb <= 0 <= ci_ub for ci_lb, ci_ub in ci_list]

        if not any(includes_zero):
            # Always significant - no breakdown
            return None

        if all(includes_zero):
            # Never significant - breakdown at 0
            return 0.0

        # Find first transition point
        for i, (inc, M) in enumerate(zip(includes_zero, M_values)):
            if inc and (i == 0 or not includes_zero[i - 1]):
                # Binary search between M_values[i-1] and M_values[i]
                if i == 0:
                    return 0.0

                lo, hi = M_values[i - 1], M_values[i]

                for _ in range(20):  # 20 iterations for precision
                    mid = (lo + hi) / 2
                    result = self.fit(results, M=mid)
                    if result.ci_lb <= 0 <= result.ci_ub:
                        hi = mid
                    else:
                        lo = mid

                return (lo + hi) / 2

        return None

    def breakdown_value(
        self,
        results: Union[MultiPeriodDiDResults, Any],
        tol: float = 0.01
    ) -> Optional[float]:
        """
        Find the breakdown value directly using binary search.

        The breakdown value is the smallest M where the robust
        confidence interval includes zero.

        Parameters
        ----------
        results : MultiPeriodDiDResults or CallawaySantAnnaResults
            Results from event study estimation.
        tol : float
            Tolerance for binary search.

        Returns
        -------
        float or None
            Breakdown value, or None if effect is always significant.
        """
        # Check at M=0
        result_0 = self.fit(results, M=0)
        if result_0.ci_lb <= 0 <= result_0.ci_ub:
            return 0.0

        # Check if significant even for large M
        result_large = self.fit(results, M=10)
        if not (result_large.ci_lb <= 0 <= result_large.ci_ub):
            return None  # Always significant

        # Binary search
        lo, hi = 0.0, 10.0

        while hi - lo > tol:
            mid = (lo + hi) / 2
            result = self.fit(results, M=mid)
            if result.ci_lb <= 0 <= result.ci_ub:
                hi = mid
            else:
                lo = mid

        return (lo + hi) / 2


# =============================================================================
# Convenience Functions
# =============================================================================


def compute_honest_did(
    results: Union[MultiPeriodDiDResults, Any],
    method: str = "relative_magnitude",
    M: float = 1.0,
    alpha: float = 0.05,
) -> HonestDiDResults:
    """
    Convenience function for computing Honest DiD bounds.

    Parameters
    ----------
    results : MultiPeriodDiDResults or CallawaySantAnnaResults
        Results from event study estimation.
    method : str
        Type of restriction ("smoothness", "relative_magnitude", "combined").
    M : float
        Restriction parameter.
    alpha : float
        Significance level.

    Returns
    -------
    HonestDiDResults
        Bounds and robust confidence intervals.

    Examples
    --------
    >>> bounds = compute_honest_did(event_study_results, method='relative_magnitude', M=1.0)
    >>> print(f"Robust CI: [{bounds.ci_lb:.3f}, {bounds.ci_ub:.3f}]")
    """
    honest = HonestDiD(method=method, M=M, alpha=alpha)
    return honest.fit(results)


def sensitivity_plot(
    results: Union[MultiPeriodDiDResults, Any],
    method: str = "relative_magnitude",
    M_grid: Optional[List[float]] = None,
    alpha: float = 0.05,
    ax=None,
    **kwargs
):
    """
    Create a sensitivity analysis plot.

    Parameters
    ----------
    results : MultiPeriodDiDResults or CallawaySantAnnaResults
        Results from event study estimation.
    method : str
        Type of restriction.
    M_grid : list of float, optional
        Grid of M values.
    alpha : float
        Significance level.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    **kwargs
        Additional arguments passed to plot method.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.
    """
    honest = HonestDiD(method=method, alpha=alpha)
    sensitivity = honest.sensitivity_analysis(results, M_grid=M_grid)
    return sensitivity.plot(ax=ax, **kwargs)
