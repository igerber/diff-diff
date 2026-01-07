"""
Pre-trends power analysis for difference-in-differences designs.

This module implements the power analysis framework from Roth (2022) for assessing
the informativeness of pre-trends tests. It answers the question: "If my pre-trends
test passed, what violations would I have been able to detect?"

Key concepts:
- **Minimum Detectable Violation (MDV)**: The smallest pre-trends violation that
  would be detected with given power (e.g., 80%).
- **Power of Pre-Trends Test**: Probability of rejecting parallel trends given
  a specific violation pattern.
- **Relationship to HonestDiD**: If MDV is large relative to your estimated effect,
  a passing pre-trends test provides limited reassurance.

References
----------
Roth, J. (2022). Pretest with Caution: Event-Study Estimates after Testing for
    Parallel Trends. American Economic Review: Insights, 4(3), 305-322.
    https://doi.org/10.1257/aeri.20210236

See Also
--------
https://github.com/jonathandroth/pretrends - R package implementation
diff_diff.honest_did - Sensitivity analysis for parallel trends violations
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats, optimize

from diff_diff.results import MultiPeriodDiDResults


# =============================================================================
# Results Classes
# =============================================================================


@dataclass
class PreTrendsPowerResults:
    """
    Results from pre-trends power analysis.

    Attributes
    ----------
    power : float
        Power to detect the specified violation pattern at given alpha.
    mdv : float
        Minimum detectable violation (smallest M detectable at target power).
    violation_magnitude : float
        The magnitude of violation tested (M parameter).
    violation_type : str
        Type of violation pattern ('linear', 'constant', 'last_period', 'custom').
    alpha : float
        Significance level for the pre-trends test.
    target_power : float
        Target power level used for MDV calculation.
    n_pre_periods : int
        Number of pre-treatment periods in the event study.
    test_statistic : float
        Expected test statistic under the specified violation.
    critical_value : float
        Critical value for the pre-trends test.
    noncentrality : float
        Non-centrality parameter under the alternative hypothesis.
    pre_period_effects : np.ndarray
        Estimated pre-period effects from the event study.
    pre_period_ses : np.ndarray
        Standard errors of pre-period effects.
    vcov : np.ndarray
        Variance-covariance matrix of pre-period effects.
    """

    power: float
    mdv: float
    violation_magnitude: float
    violation_type: str
    alpha: float
    target_power: float
    n_pre_periods: int
    test_statistic: float
    critical_value: float
    noncentrality: float
    pre_period_effects: np.ndarray = field(repr=False)
    pre_period_ses: np.ndarray = field(repr=False)
    vcov: np.ndarray = field(repr=False)
    original_results: Optional[Any] = field(default=None, repr=False)

    def __repr__(self) -> str:
        return (
            f"PreTrendsPowerResults(power={self.power:.3f}, "
            f"mdv={self.mdv:.4f}, M={self.violation_magnitude:.4f})"
        )

    @property
    def is_informative(self) -> bool:
        """
        Check if the pre-trends test is informative.

        A pre-trends test is considered informative if the MDV is reasonably
        small relative to typical effect sizes. This is a heuristic check;
        see the summary for interpretation guidance.
        """
        # Heuristic: MDV < 2x the max observed pre-period SE
        max_se = np.max(self.pre_period_ses) if len(self.pre_period_ses) > 0 else 1.0
        return bool(self.mdv < 2 * max_se)

    @property
    def power_adequate(self) -> bool:
        """Check if power meets the target threshold."""
        return bool(self.power >= self.target_power)

    def summary(self) -> str:
        """
        Generate formatted summary of pre-trends power analysis.

        Returns
        -------
        str
            Formatted summary.
        """
        lines = [
            "=" * 70,
            "Pre-Trends Power Analysis Results".center(70),
            "(Roth 2022)".center(70),
            "=" * 70,
            "",
            f"{'Number of pre-periods:':<35} {self.n_pre_periods}",
            f"{'Significance level (alpha):':<35} {self.alpha:.3f}",
            f"{'Target power:':<35} {self.target_power:.1%}",
            f"{'Violation type:':<35} {self.violation_type}",
            "",
            "-" * 70,
            "Power Analysis".center(70),
            "-" * 70,
            f"{'Violation magnitude (M):':<35} {self.violation_magnitude:.4f}",
            f"{'Power to detect this violation:':<35} {self.power:.1%}",
            f"{'Minimum detectable violation:':<35} {self.mdv:.4f}",
            "",
            f"{'Test statistic (expected):':<35} {self.test_statistic:.4f}",
            f"{'Critical value:':<35} {self.critical_value:.4f}",
            f"{'Non-centrality parameter:':<35} {self.noncentrality:.4f}",
            "",
            "-" * 70,
            "Interpretation".center(70),
            "-" * 70,
        ]

        if self.power_adequate:
            lines.append(
                f"✓ Power ({self.power:.0%}) meets target ({self.target_power:.0%})."
            )
            lines.append(
                f"  The pre-trends test would detect violations of magnitude {self.violation_magnitude:.3f}."
            )
        else:
            lines.append(
                f"✗ Power ({self.power:.0%}) below target ({self.target_power:.0%})."
            )
            lines.append(
                f"  Would need violations of {self.mdv:.3f} to achieve {self.target_power:.0%} power."
            )

        lines.append("")
        lines.append(
            f"Minimum detectable violation (MDV): {self.mdv:.4f}"
        )
        lines.append(
            "  → Passing pre-trends test does NOT rule out violations up to this size."
        )

        lines.extend(["", "=" * 70])

        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print summary to stdout."""
        print(self.summary())

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            "power": self.power,
            "mdv": self.mdv,
            "violation_magnitude": self.violation_magnitude,
            "violation_type": self.violation_type,
            "alpha": self.alpha,
            "target_power": self.target_power,
            "n_pre_periods": self.n_pre_periods,
            "test_statistic": self.test_statistic,
            "critical_value": self.critical_value,
            "noncentrality": self.noncentrality,
            "is_informative": self.is_informative,
            "power_adequate": self.power_adequate,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        return pd.DataFrame([self.to_dict()])


@dataclass
class PreTrendsPowerCurve:
    """
    Power curve across violation magnitudes.

    Attributes
    ----------
    M_values : np.ndarray
        Grid of violation magnitudes tested.
    powers : np.ndarray
        Power at each violation magnitude.
    mdv : float
        Minimum detectable violation.
    alpha : float
        Significance level.
    target_power : float
        Target power level.
    violation_type : str
        Type of violation pattern.
    """

    M_values: np.ndarray
    powers: np.ndarray
    mdv: float
    alpha: float
    target_power: float
    violation_type: str

    def __repr__(self) -> str:
        return (
            f"PreTrendsPowerCurve(n_points={len(self.M_values)}, "
            f"mdv={self.mdv:.4f})"
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with M and power columns."""
        return pd.DataFrame({
            "M": self.M_values,
            "power": self.powers,
        })

    def plot(self, ax=None, show_mdv: bool = True, show_target: bool = True,
             color: str = "#2563eb", mdv_color: str = "#dc2626",
             target_color: str = "#22c55e", **kwargs):
        """
        Plot the power curve.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        show_mdv : bool, default=True
            Whether to show vertical line at MDV.
        show_target : bool, default=True
            Whether to show horizontal line at target power.
        color : str
            Color for power curve line.
        mdv_color : str
            Color for MDV vertical line.
        target_color : str
            Color for target power horizontal line.
        **kwargs
            Additional arguments passed to plt.plot().

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

        # Plot power curve
        ax.plot(self.M_values, self.powers, color=color, linewidth=2,
                label="Power", **kwargs)

        # Target power line
        if show_target:
            ax.axhline(y=self.target_power, color=target_color, linestyle="--",
                       linewidth=1.5, alpha=0.7,
                       label=f"Target power ({self.target_power:.0%})")

        # MDV line
        if show_mdv and self.mdv is not None and np.isfinite(self.mdv):
            ax.axvline(x=self.mdv, color=mdv_color, linestyle=":",
                       linewidth=1.5, alpha=0.7,
                       label=f"MDV = {self.mdv:.3f}")

        ax.set_xlabel("Violation Magnitude (M)")
        ax.set_ylabel("Power")
        ax.set_title("Pre-Trends Test Power Curve")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        return ax


# =============================================================================
# Main Class
# =============================================================================


class PreTrendsPower:
    """
    Pre-trends power analysis (Roth 2022).

    Computes the power of pre-trends tests to detect violations of parallel
    trends, and the minimum detectable violation (MDV).

    Parameters
    ----------
    alpha : float, default=0.05
        Significance level for the pre-trends test.
    power : float, default=0.80
        Target power level for MDV calculation.
    violation_type : str, default='linear'
        Type of violation pattern to consider:
        - 'linear': Violations follow a linear trend (most common)
        - 'constant': Same violation in all pre-periods
        - 'last_period': Violation only in the last pre-period
        - 'custom': User-specified violation pattern (via violation_weights)
    violation_weights : array-like, optional
        Custom weights for violation pattern. Length must equal number of
        pre-periods. Only used when violation_type='custom'.

    Examples
    --------
    Basic usage with MultiPeriodDiD results:

    >>> from diff_diff import MultiPeriodDiD
    >>> from diff_diff.pretrends import PreTrendsPower
    >>>
    >>> # Fit event study
    >>> mp_did = MultiPeriodDiD()
    >>> results = mp_did.fit(data, outcome='y', treatment='treated',
    ...                      time='period', post_periods=[4, 5, 6, 7])
    >>>
    >>> # Analyze pre-trends power
    >>> pt = PreTrendsPower(alpha=0.05, power=0.80)
    >>> power_results = pt.fit(results)
    >>> print(power_results.summary())
    >>>
    >>> # Get power curve
    >>> curve = pt.power_curve(results)
    >>> curve.plot()

    Notes
    -----
    The pre-trends test is typically a joint test that all pre-period
    coefficients are zero. This test has limited power to detect small
    violations, especially when:

    1. There are few pre-periods
    2. Standard errors are large
    3. The violation pattern is smooth (e.g., linear trend)

    Passing a pre-trends test does NOT mean parallel trends holds. It means
    violations smaller than the MDV cannot be ruled out. For robust inference,
    combine with HonestDiD sensitivity analysis.

    References
    ----------
    Roth, J. (2022). Pretest with Caution: Event-Study Estimates after Testing
        for Parallel Trends. American Economic Review: Insights, 4(3), 305-322.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        power: float = 0.80,
        violation_type: Literal["linear", "constant", "last_period", "custom"] = "linear",
        violation_weights: Optional[np.ndarray] = None,
    ):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
        if not 0 < power < 1:
            raise ValueError(f"power must be between 0 and 1, got {power}")
        if violation_type not in ["linear", "constant", "last_period", "custom"]:
            raise ValueError(
                f"violation_type must be 'linear', 'constant', 'last_period', or 'custom', "
                f"got '{violation_type}'"
            )
        if violation_type == "custom" and violation_weights is None:
            raise ValueError(
                "violation_weights must be provided when violation_type='custom'"
            )

        self.alpha = alpha
        self.target_power = power
        self.violation_type = violation_type
        self.violation_weights = (
            np.asarray(violation_weights) if violation_weights is not None else None
        )

    def get_params(self) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        return {
            "alpha": self.alpha,
            "power": self.target_power,
            "violation_type": self.violation_type,
            "violation_weights": self.violation_weights,
        }

    def set_params(self, **params) -> "PreTrendsPower":
        """Set parameters for this estimator."""
        for key, value in params.items():
            if key == "power":
                self.target_power = value
            elif hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self

    def _get_violation_weights(self, n_pre: int) -> np.ndarray:
        """
        Get violation weights based on violation type.

        Parameters
        ----------
        n_pre : int
            Number of pre-treatment periods.

        Returns
        -------
        np.ndarray
            Violation weights, normalized to have L2 norm of 1.
        """
        if self.violation_type == "custom":
            if len(self.violation_weights) != n_pre:
                raise ValueError(
                    f"violation_weights has length {len(self.violation_weights)}, "
                    f"but there are {n_pre} pre-periods"
                )
            weights = self.violation_weights.copy()
        elif self.violation_type == "linear":
            # Linear trend: weights = [-n+1, -n+2, ..., -1, 0] for periods ending at -1
            # Normalized so that violation at period -1 = 0 and grows linearly backward
            weights = np.arange(-n_pre + 1, 1, dtype=float)
            # Shift so that weights are positive and represent deviation from PT
            weights = -weights  # Now [n-1, n-2, ..., 1, 0]
        elif self.violation_type == "constant":
            # Same violation in all periods
            weights = np.ones(n_pre)
        elif self.violation_type == "last_period":
            # Violation only in last pre-period (period -1)
            weights = np.zeros(n_pre)
            weights[-1] = 1.0
        else:
            raise ValueError(f"Unknown violation_type: {self.violation_type}")

        # Normalize to unit norm (if not all zeros)
        norm = np.linalg.norm(weights)
        if norm > 0:
            weights = weights / norm

        return weights

    def _extract_pre_period_params(
        self,
        results: Union[MultiPeriodDiDResults, Any],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Extract pre-period parameters from results.

        Returns
        -------
        effects : np.ndarray
            Pre-period effect estimates.
        ses : np.ndarray
            Pre-period standard errors.
        vcov : np.ndarray
            Variance-covariance matrix for pre-period effects.
        n_pre : int
            Number of pre-periods.
        """
        if isinstance(results, MultiPeriodDiDResults):
            # Get pre-period information
            all_pre_periods = results.pre_periods

            if len(all_pre_periods) == 0:
                raise ValueError(
                    "No pre-treatment periods found in results. "
                    "Pre-trends power analysis requires pre-period coefficients."
                )

            # Only include periods with actual estimated coefficients
            # (excludes the reference period which is omitted from estimation)
            if hasattr(results, 'coefficients') and results.coefficients:
                # Find which pre-periods have estimated coefficients
                estimated_pre_periods = [
                    p for p in all_pre_periods
                    if f"treated:period_{p}" in results.coefficients
                ]

                if len(estimated_pre_periods) == 0:
                    raise ValueError(
                        "No estimated pre-period coefficients found. "
                        "The pre-trends test requires at least one estimated "
                        "pre-period coefficient (excluding the reference period)."
                    )

                n_pre = len(estimated_pre_periods)

                # Extract effects for estimated periods only
                effects = np.array([
                    results.coefficients[f"treated:period_{p}"]
                    for p in estimated_pre_periods
                ])

                # Extract SEs - try period_effects first, fall back to avg_se
                ses = []
                for p in estimated_pre_periods:
                    if p in results.period_effects:
                        ses.append(results.period_effects[p].se)
                    else:
                        ses.append(results.avg_se)
                ses = np.array(ses)

                # Extract vcov for estimated pre-periods
                # Build mapping from period to vcov index
                if results.vcov is not None:
                    # Get ordered list of all coefficient keys
                    coef_keys = list(results.coefficients.keys())
                    pre_indices = [
                        coef_keys.index(f"treated:period_{p}")
                        for p in estimated_pre_periods
                        if f"treated:period_{p}" in coef_keys
                    ]
                    if len(pre_indices) == n_pre and results.vcov.shape[0] > max(pre_indices):
                        vcov = results.vcov[np.ix_(pre_indices, pre_indices)]
                    else:
                        # Fall back to diagonal
                        vcov = np.diag(ses ** 2)
                else:
                    vcov = np.diag(ses ** 2)
            else:
                # No coefficients available - try period_effects for pre-periods
                # Exclude reference period (the one with effect=0 and se=0 or missing)
                estimated_pre_periods = [
                    p for p in all_pre_periods
                    if p in results.period_effects
                    and results.period_effects[p].se > 0
                ]

                if len(estimated_pre_periods) == 0:
                    raise ValueError(
                        "No estimated pre-period effects found. "
                        "The pre-trends test requires at least one estimated "
                        "pre-period effect (excluding the reference period)."
                    )

                n_pre = len(estimated_pre_periods)
                effects = np.array([
                    results.period_effects[p].effect
                    for p in estimated_pre_periods
                ])
                ses = np.array([
                    results.period_effects[p].se
                    for p in estimated_pre_periods
                ])
                vcov = np.diag(ses ** 2)

            return effects, ses, vcov, n_pre

        # Try CallawaySantAnnaResults
        try:
            from diff_diff.staggered import CallawaySantAnnaResults
            if isinstance(results, CallawaySantAnnaResults):
                if results.event_study_effects is None:
                    raise ValueError(
                        "CallawaySantAnnaResults must have event_study_effects. "
                        "Re-run with aggregate='event_study'."
                    )

                # Get pre-period effects (negative relative times)
                pre_effects = {
                    t: data for t, data in results.event_study_effects.items()
                    if t < 0
                }

                if not pre_effects:
                    raise ValueError("No pre-treatment periods found in event study.")

                pre_periods = sorted(pre_effects.keys())
                n_pre = len(pre_periods)

                effects = np.array([pre_effects[t]['effect'] for t in pre_periods])
                ses = np.array([pre_effects[t]['se'] for t in pre_periods])
                vcov = np.diag(ses ** 2)

                return effects, ses, vcov, n_pre
        except ImportError:
            pass

        # Try SunAbrahamResults
        try:
            from diff_diff.sun_abraham import SunAbrahamResults
            if isinstance(results, SunAbrahamResults):
                # Get pre-period effects (negative relative times)
                pre_effects = {
                    t: data for t, data in results.event_study_effects.items()
                    if t < 0
                }

                if not pre_effects:
                    raise ValueError("No pre-treatment periods found in event study.")

                pre_periods = sorted(pre_effects.keys())
                n_pre = len(pre_periods)

                effects = np.array([pre_effects[t]['effect'] for t in pre_periods])
                ses = np.array([pre_effects[t]['se'] for t in pre_periods])
                vcov = np.diag(ses ** 2)

                return effects, ses, vcov, n_pre
        except ImportError:
            pass

        raise TypeError(
            f"Unsupported results type: {type(results)}. "
            "Expected MultiPeriodDiDResults, CallawaySantAnnaResults, or SunAbrahamResults."
        )

    def _compute_power(
        self,
        M: float,
        weights: np.ndarray,
        vcov: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """
        Compute power to detect violation of magnitude M.

        The pre-trends test is a Wald test: H0: delta = 0 vs H1: delta != 0
        Under H1 with violation delta = M * weights, the test statistic follows
        a non-central chi-squared distribution.

        Parameters
        ----------
        M : float
            Violation magnitude.
        weights : np.ndarray
            Normalized violation pattern.
        vcov : np.ndarray
            Variance-covariance matrix.

        Returns
        -------
        power : float
            Power to detect this violation.
        noncentrality : float
            Non-centrality parameter.
        test_stat : float
            Expected test statistic under H1.
        critical_value : float
            Critical value for the test.
        """
        n_pre = len(weights)

        # Violation vector: delta = M * weights
        delta = M * weights

        # Non-centrality parameter for chi-squared test
        # lambda = delta' * V^{-1} * delta
        try:
            vcov_inv = np.linalg.inv(vcov)
            noncentrality = delta @ vcov_inv @ delta
        except np.linalg.LinAlgError:
            # Singular matrix - use pseudo-inverse
            vcov_inv = np.linalg.pinv(vcov)
            noncentrality = delta @ vcov_inv @ delta

        # Critical value from chi-squared distribution
        critical_value = stats.chi2.ppf(1 - self.alpha, df=n_pre)

        # Power = P(chi2_nc > critical_value) where chi2_nc is non-central chi2
        if noncentrality > 0:
            power = 1 - stats.ncx2.cdf(critical_value, df=n_pre, nc=noncentrality)
        else:
            power = self.alpha  # Size under null

        # Expected test statistic under H1
        test_stat = n_pre + noncentrality  # Mean of non-central chi2

        return power, noncentrality, test_stat, critical_value

    def _compute_mdv(
        self,
        weights: np.ndarray,
        vcov: np.ndarray,
    ) -> float:
        """
        Compute minimum detectable violation.

        Find the smallest M such that power >= target_power.

        Parameters
        ----------
        weights : np.ndarray
            Normalized violation pattern.
        vcov : np.ndarray
            Variance-covariance matrix.

        Returns
        -------
        mdv : float
            Minimum detectable violation.
        """
        n_pre = len(weights)

        # Critical value
        critical_value = stats.chi2.ppf(1 - self.alpha, df=n_pre)

        # Find non-centrality parameter for target power
        # We need: P(ncx2 > critical_value) = target_power
        # Use inverse: find lambda such that ncx2.cdf(cv, df, lambda) = 1 - target_power

        def power_minus_target(nc):
            if nc <= 0:
                return self.alpha - self.target_power
            return stats.ncx2.sf(critical_value, df=n_pre, nc=nc) - self.target_power

        # Binary search for non-centrality parameter
        # Start with bounds
        nc_low, nc_high = 0, 1

        # Expand upper bound until power exceeds target
        while power_minus_target(nc_high) < 0 and nc_high < 1000:
            nc_high *= 2

        if nc_high >= 1000:
            # Target power not achievable - return inf
            return np.inf

        # Binary search
        try:
            result = optimize.brentq(power_minus_target, nc_low, nc_high)
            target_nc = result
        except ValueError:
            # Fallback: use approximate formula
            # For chi2, power ≈ Phi(sqrt(2*nc) - sqrt(2*cv))
            # Solving: sqrt(2*nc) = z_power + sqrt(2*cv)
            z_power = stats.norm.ppf(self.target_power)
            target_nc = 0.5 * (z_power + np.sqrt(2 * critical_value)) ** 2

        # Convert non-centrality to M
        # nc = delta' * V^{-1} * delta = M^2 * w' * V^{-1} * w
        try:
            vcov_inv = np.linalg.inv(vcov)
            w_Vinv_w = weights @ vcov_inv @ weights
        except np.linalg.LinAlgError:
            vcov_inv = np.linalg.pinv(vcov)
            w_Vinv_w = weights @ vcov_inv @ weights

        if w_Vinv_w > 0:
            mdv = np.sqrt(target_nc / w_Vinv_w)
        else:
            mdv = np.inf

        return mdv

    def fit(
        self,
        results: Union[MultiPeriodDiDResults, Any],
        M: Optional[float] = None,
    ) -> PreTrendsPowerResults:
        """
        Compute pre-trends power analysis.

        Parameters
        ----------
        results : MultiPeriodDiDResults, CallawaySantAnnaResults, or SunAbrahamResults
            Results from an event study estimation.
        M : float, optional
            Specific violation magnitude to evaluate. If None, evaluates at
            a default magnitude based on the data.

        Returns
        -------
        PreTrendsPowerResults
            Power analysis results including power and MDV.
        """
        # Extract pre-period parameters
        effects, ses, vcov, n_pre = self._extract_pre_period_params(results)

        # Get violation weights
        weights = self._get_violation_weights(n_pre)

        # Compute MDV
        mdv = self._compute_mdv(weights, vcov)

        # Default M: use MDV if not specified
        if M is None:
            M = mdv if np.isfinite(mdv) else np.max(ses)

        # Compute power at specified M
        power, noncentrality, test_stat, critical_value = self._compute_power(
            M, weights, vcov
        )

        return PreTrendsPowerResults(
            power=power,
            mdv=mdv,
            violation_magnitude=M,
            violation_type=self.violation_type,
            alpha=self.alpha,
            target_power=self.target_power,
            n_pre_periods=n_pre,
            test_statistic=test_stat,
            critical_value=critical_value,
            noncentrality=noncentrality,
            pre_period_effects=effects,
            pre_period_ses=ses,
            vcov=vcov,
            original_results=results,
        )

    def power_at(
        self,
        results: Union[MultiPeriodDiDResults, Any],
        M: float,
    ) -> float:
        """
        Compute power to detect a specific violation magnitude.

        Parameters
        ----------
        results : results object
            Event study results.
        M : float
            Violation magnitude.

        Returns
        -------
        float
            Power to detect violation of magnitude M.
        """
        result = self.fit(results, M=M)
        return result.power

    def power_curve(
        self,
        results: Union[MultiPeriodDiDResults, Any],
        M_grid: Optional[List[float]] = None,
        n_points: int = 50,
    ) -> PreTrendsPowerCurve:
        """
        Compute power across a range of violation magnitudes.

        Parameters
        ----------
        results : results object
            Event study results.
        M_grid : list of float, optional
            Specific violation magnitudes to evaluate. If None, creates
            automatic grid from 0 to 2.5 * MDV.
        n_points : int, default=50
            Number of points in automatic grid.

        Returns
        -------
        PreTrendsPowerCurve
            Power curve data with plot method.
        """
        # Extract parameters
        effects, ses, vcov, n_pre = self._extract_pre_period_params(results)
        weights = self._get_violation_weights(n_pre)

        # Compute MDV
        mdv = self._compute_mdv(weights, vcov)

        # Create M grid if not provided
        if M_grid is None:
            max_M = min(2.5 * mdv if np.isfinite(mdv) else 10 * np.max(ses), 100)
            M_grid = np.linspace(0, max_M, n_points)
        else:
            M_grid = np.asarray(M_grid)

        # Compute power at each M
        powers = np.array([
            self._compute_power(M, weights, vcov)[0]
            for M in M_grid
        ])

        return PreTrendsPowerCurve(
            M_values=M_grid,
            powers=powers,
            mdv=mdv,
            alpha=self.alpha,
            target_power=self.target_power,
            violation_type=self.violation_type,
        )

    def sensitivity_to_honest_did(
        self,
        results: Union[MultiPeriodDiDResults, Any],
    ) -> Dict[str, Any]:
        """
        Compare pre-trends power analysis with HonestDiD sensitivity.

        This method helps interpret how informative a passing pre-trends
        test is in the context of HonestDiD's relative magnitudes restriction.

        Parameters
        ----------
        results : results object
            Event study results.

        Returns
        -------
        dict
            Dictionary with:
            - mdv: Minimum detectable violation from pre-trends test
            - honest_M_at_mdv: Corresponding M value for HonestDiD
            - interpretation: Text explaining the relationship
        """
        pt_results = self.fit(results)
        mdv = pt_results.mdv

        # The MDV represents the size of violation the test could detect
        # In HonestDiD's relative magnitudes framework, M=1 means
        # post-treatment violations can be as large as the max pre-period violation
        # The MDV gives us a sense of how large that max violation could be

        max_pre_se = np.max(pt_results.pre_period_ses)

        interpretation = []
        interpretation.append(
            f"Minimum Detectable Violation (MDV): {mdv:.4f}"
        )
        interpretation.append(
            f"Max pre-period SE: {max_pre_se:.4f}"
        )

        if np.isfinite(mdv):
            # Ratio of MDV to max SE - gives sense of how many SEs the MDV is
            mdv_in_ses = mdv / max_pre_se if max_pre_se > 0 else np.inf
            interpretation.append(
                f"MDV / max(SE): {mdv_in_ses:.2f}"
            )

            if mdv_in_ses < 1:
                interpretation.append(
                    "→ Pre-trends test is fairly sensitive to violations."
                )
            elif mdv_in_ses < 2:
                interpretation.append(
                    "→ Pre-trends test has moderate sensitivity."
                )
            else:
                interpretation.append(
                    "→ Pre-trends test has low power to detect violations."
                )
                interpretation.append(
                    "  Consider using HonestDiD with larger M values for robustness."
                )
        else:
            interpretation.append(
                "→ Pre-trends test cannot achieve target power for any violation size."
            )
            interpretation.append(
                "  Use HonestDiD sensitivity analysis for inference."
            )

        return {
            "mdv": mdv,
            "max_pre_se": max_pre_se,
            "mdv_in_ses": mdv / max_pre_se if max_pre_se > 0 and np.isfinite(mdv) else np.inf,
            "interpretation": "\n".join(interpretation),
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def compute_pretrends_power(
    results: Union[MultiPeriodDiDResults, Any],
    M: Optional[float] = None,
    alpha: float = 0.05,
    target_power: float = 0.80,
    violation_type: str = "linear",
) -> PreTrendsPowerResults:
    """
    Convenience function for pre-trends power analysis.

    Parameters
    ----------
    results : results object
        Event study results.
    M : float, optional
        Violation magnitude to evaluate.
    alpha : float, default=0.05
        Significance level.
    target_power : float, default=0.80
        Target power for MDV calculation.
    violation_type : str, default='linear'
        Type of violation pattern.

    Returns
    -------
    PreTrendsPowerResults
        Power analysis results.

    Examples
    --------
    >>> from diff_diff import MultiPeriodDiD
    >>> from diff_diff.pretrends import compute_pretrends_power
    >>>
    >>> results = MultiPeriodDiD().fit(data, ...)
    >>> power_results = compute_pretrends_power(results)
    >>> print(f"MDV: {power_results.mdv:.3f}")
    >>> print(f"Power: {power_results.power:.1%}")
    """
    pt = PreTrendsPower(
        alpha=alpha,
        power=target_power,
        violation_type=violation_type,
    )
    return pt.fit(results, M=M)


def compute_mdv(
    results: Union[MultiPeriodDiDResults, Any],
    alpha: float = 0.05,
    target_power: float = 0.80,
    violation_type: str = "linear",
) -> float:
    """
    Compute minimum detectable violation.

    Parameters
    ----------
    results : results object
        Event study results.
    alpha : float, default=0.05
        Significance level.
    target_power : float, default=0.80
        Target power.
    violation_type : str, default='linear'
        Type of violation pattern.

    Returns
    -------
    float
        Minimum detectable violation.
    """
    pt = PreTrendsPower(
        alpha=alpha,
        power=target_power,
        violation_type=violation_type,
    )
    result = pt.fit(results)
    return result.mdv
