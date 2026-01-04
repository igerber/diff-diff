"""
Power analysis tools for difference-in-differences study design.

This module provides power calculations and simulation-based power analysis
for DiD study design, helping practitioners answer questions like:
- "How many units do I need to detect an effect of size X?"
- "What is the minimum detectable effect given my sample size?"
- "What power do I have to detect a given effect?"

References
----------
Bloom, H. S. (1995). "Minimum Detectable Effects: A Simple Way to Report the
    Statistical Power of Experimental Designs." Evaluation Review, 19(5), 547-556.

Burlig, F., Preonas, L., & Woerman, M. (2020). "Panel Data and Experimental Design."
    Journal of Development Economics, 144, 102458.

Djimeu, E. W., & Houndolo, D.-G. (2016). "Power Calculation for Causal Inference
    in Social Science: Sample Size and Minimum Detectable Effect Determination."
    Journal of Development Effectiveness, 8(4), 508-527.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Maximum sample size returned when effect is too small to detect
# (e.g., zero effect or extremely small relative to noise)
MAX_SAMPLE_SIZE = 2**31 - 1


@dataclass
class PowerResults:
    """
    Results from analytical power analysis.

    Attributes
    ----------
    power : float
        Statistical power (probability of rejecting H0 when effect exists).
    mde : float
        Minimum detectable effect size.
    required_n : int
        Required total sample size (treated + control).
    effect_size : float
        Effect size used in calculation.
    alpha : float
        Significance level.
    alternative : str
        Alternative hypothesis ('two-sided', 'greater', 'less').
    n_treated : int
        Number of treated units.
    n_control : int
        Number of control units.
    n_pre : int
        Number of pre-treatment periods.
    n_post : int
        Number of post-treatment periods.
    sigma : float
        Residual standard deviation.
    rho : float
        Intra-cluster correlation (for panel data).
    design : str
        Study design type ('basic_did', 'panel', 'staggered').
    """

    power: float
    mde: float
    required_n: int
    effect_size: float
    alpha: float
    alternative: str
    n_treated: int
    n_control: int
    n_pre: int
    n_post: int
    sigma: float
    rho: float = 0.0
    design: str = "basic_did"

    def __repr__(self) -> str:
        """Concise string representation."""
        return (
            f"PowerResults(power={self.power:.3f}, mde={self.mde:.4f}, "
            f"required_n={self.required_n})"
        )

    def summary(self) -> str:
        """
        Generate a formatted summary of power analysis results.

        Returns
        -------
        str
            Formatted summary table.
        """
        lines = [
            "=" * 60,
            "Power Analysis for Difference-in-Differences".center(60),
            "=" * 60,
            "",
            f"{'Design:':<30} {self.design}",
            f"{'Significance level (alpha):':<30} {self.alpha:.3f}",
            f"{'Alternative hypothesis:':<30} {self.alternative}",
            "",
            "-" * 60,
            "Sample Size".center(60),
            "-" * 60,
            f"{'Treated units:':<30} {self.n_treated:>10}",
            f"{'Control units:':<30} {self.n_control:>10}",
            f"{'Total units:':<30} {self.n_treated + self.n_control:>10}",
            f"{'Pre-treatment periods:':<30} {self.n_pre:>10}",
            f"{'Post-treatment periods:':<30} {self.n_post:>10}",
            "",
            "-" * 60,
            "Variance Parameters".center(60),
            "-" * 60,
            f"{'Residual SD (sigma):':<30} {self.sigma:>10.4f}",
            f"{'Intra-cluster correlation:':<30} {self.rho:>10.4f}",
            "",
            "-" * 60,
            "Power Analysis Results".center(60),
            "-" * 60,
            f"{'Effect size:':<30} {self.effect_size:>10.4f}",
            f"{'Power:':<30} {self.power:>10.1%}",
            f"{'Minimum detectable effect:':<30} {self.mde:>10.4f}",
            f"{'Required sample size:':<30} {self.required_n:>10}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print the summary to stdout."""
        print(self.summary())

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert results to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing all power analysis results.
        """
        return {
            "power": self.power,
            "mde": self.mde,
            "required_n": self.required_n,
            "effect_size": self.effect_size,
            "alpha": self.alpha,
            "alternative": self.alternative,
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "n_pre": self.n_pre,
            "n_post": self.n_post,
            "sigma": self.sigma,
            "rho": self.rho,
            "design": self.design,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with power analysis results.
        """
        return pd.DataFrame([self.to_dict()])


@dataclass
class SimulationPowerResults:
    """
    Results from simulation-based power analysis.

    Attributes
    ----------
    power : float
        Estimated power (proportion of simulations rejecting H0).
    power_se : float
        Standard error of power estimate.
    power_ci : Tuple[float, float]
        Confidence interval for power estimate.
    rejection_rate : float
        Proportion of simulations with p-value < alpha.
    mean_estimate : float
        Mean treatment effect estimate across simulations.
    std_estimate : float
        Standard deviation of estimates across simulations.
    mean_se : float
        Mean standard error across simulations.
    coverage : float
        Proportion of CIs containing true effect.
    n_simulations : int
        Number of simulations performed.
    effect_sizes : List[float]
        Effect sizes tested (if multiple).
    powers : List[float]
        Power at each effect size (if multiple).
    true_effect : float
        True treatment effect used in simulation.
    alpha : float
        Significance level.
    estimator_name : str
        Name of the estimator used.
    """

    power: float
    power_se: float
    power_ci: Tuple[float, float]
    rejection_rate: float
    mean_estimate: float
    std_estimate: float
    mean_se: float
    coverage: float
    n_simulations: int
    effect_sizes: List[float]
    powers: List[float]
    true_effect: float
    alpha: float
    estimator_name: str
    bias: float = field(init=False)
    rmse: float = field(init=False)
    simulation_results: Optional[List[Dict[str, Any]]] = field(default=None, repr=False)

    def __post_init__(self):
        """Compute derived statistics."""
        self.bias = self.mean_estimate - self.true_effect
        self.rmse = np.sqrt(self.bias**2 + self.std_estimate**2)

    def __repr__(self) -> str:
        """Concise string representation."""
        return (
            f"SimulationPowerResults(power={self.power:.3f} "
            f"[{self.power_ci[0]:.3f}, {self.power_ci[1]:.3f}], "
            f"n_simulations={self.n_simulations})"
        )

    def summary(self) -> str:
        """
        Generate a formatted summary of simulation power results.

        Returns
        -------
        str
            Formatted summary table.
        """
        lines = [
            "=" * 65,
            "Simulation-Based Power Analysis Results".center(65),
            "=" * 65,
            "",
            f"{'Estimator:':<35} {self.estimator_name}",
            f"{'Number of simulations:':<35} {self.n_simulations}",
            f"{'True treatment effect:':<35} {self.true_effect:.4f}",
            f"{'Significance level (alpha):':<35} {self.alpha:.3f}",
            "",
            "-" * 65,
            "Power Estimates".center(65),
            "-" * 65,
            f"{'Power (rejection rate):':<35} {self.power:.1%}",
            f"{'Standard error:':<35} {self.power_se:.4f}",
            f"{'95% CI:':<35} [{self.power_ci[0]:.3f}, {self.power_ci[1]:.3f}]",
            "",
            "-" * 65,
            "Estimation Performance".center(65),
            "-" * 65,
            f"{'Mean estimate:':<35} {self.mean_estimate:.4f}",
            f"{'Bias:':<35} {self.bias:.4f}",
            f"{'Std. deviation of estimates:':<35} {self.std_estimate:.4f}",
            f"{'RMSE:':<35} {self.rmse:.4f}",
            f"{'Mean standard error:':<35} {self.mean_se:.4f}",
            f"{'Coverage (CI contains true):':<35} {self.coverage:.1%}",
            "=" * 65,
        ]
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print the summary to stdout."""
        print(self.summary())

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert results to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing simulation power results.
        """
        return {
            "power": self.power,
            "power_se": self.power_se,
            "power_ci_lower": self.power_ci[0],
            "power_ci_upper": self.power_ci[1],
            "rejection_rate": self.rejection_rate,
            "mean_estimate": self.mean_estimate,
            "std_estimate": self.std_estimate,
            "bias": self.bias,
            "rmse": self.rmse,
            "mean_se": self.mean_se,
            "coverage": self.coverage,
            "n_simulations": self.n_simulations,
            "true_effect": self.true_effect,
            "alpha": self.alpha,
            "estimator_name": self.estimator_name,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with simulation power results.
        """
        return pd.DataFrame([self.to_dict()])

    def power_curve_df(self) -> pd.DataFrame:
        """
        Get power curve data as a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with effect_size and power columns.
        """
        return pd.DataFrame({
            "effect_size": self.effect_sizes,
            "power": self.powers
        })


class PowerAnalysis:
    """
    Power analysis for difference-in-differences designs.

    Provides analytical power calculations for basic 2x2 DiD and panel DiD
    designs. For complex designs like staggered adoption, use simulate_power()
    instead.

    Parameters
    ----------
    alpha : float, default=0.05
        Significance level for hypothesis testing.
    power : float, default=0.80
        Target statistical power.
    alternative : str, default='two-sided'
        Alternative hypothesis: 'two-sided', 'greater', or 'less'.

    Examples
    --------
    Calculate minimum detectable effect:

    >>> from diff_diff import PowerAnalysis
    >>> pa = PowerAnalysis(alpha=0.05, power=0.80)
    >>> results = pa.mde(n_treated=50, n_control=50, sigma=1.0)
    >>> print(f"MDE: {results.mde:.3f}")

    Calculate required sample size:

    >>> results = pa.sample_size(effect_size=0.5, sigma=1.0)
    >>> print(f"Required N: {results.required_n}")

    Calculate power for given sample and effect:

    >>> results = pa.power(effect_size=0.5, n_treated=50, n_control=50, sigma=1.0)
    >>> print(f"Power: {results.power:.1%}")

    Notes
    -----
    The power calculations are based on the variance of the DiD estimator:

    For basic 2x2 DiD:
        Var(ATT) = sigma^2 * (1/n_treated_post + 1/n_treated_pre
                            + 1/n_control_post + 1/n_control_pre)

    For panel DiD with T periods:
        Var(ATT) = sigma^2 * (1/(N_treated * T) + 1/(N_control * T))
                 * (1 + (T-1)*rho) / (1 + (T-1)*rho)

    Where rho is the intra-cluster correlation coefficient.

    References
    ----------
    Bloom, H. S. (1995). "Minimum Detectable Effects."
    Burlig, F., Preonas, L., & Woerman, M. (2020). "Panel Data and Experimental Design."
    """

    def __init__(
        self,
        alpha: float = 0.05,
        power: float = 0.80,
        alternative: str = "two-sided",
    ):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        if not 0 < power < 1:
            raise ValueError("power must be between 0 and 1")
        if alternative not in ("two-sided", "greater", "less"):
            raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

        self.alpha = alpha
        self.target_power = power
        self.alternative = alternative

    def _get_critical_values(self) -> Tuple[float, float]:
        """Get z critical values for alpha and power."""
        if self.alternative == "two-sided":
            z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        else:
            z_alpha = stats.norm.ppf(1 - self.alpha)
        z_beta = stats.norm.ppf(self.target_power)
        return z_alpha, z_beta

    def _compute_variance(
        self,
        n_treated: int,
        n_control: int,
        n_pre: int,
        n_post: int,
        sigma: float,
        rho: float = 0.0,
        design: str = "basic_did",
    ) -> float:
        """
        Compute variance of the DiD estimator.

        Parameters
        ----------
        n_treated : int
            Number of treated units.
        n_control : int
            Number of control units.
        n_pre : int
            Number of pre-treatment periods.
        n_post : int
            Number of post-treatment periods.
        sigma : float
            Residual standard deviation.
        rho : float
            Intra-cluster correlation (for panel data).
        design : str
            Study design type.

        Returns
        -------
        float
            Variance of the DiD estimator.
        """
        if design == "basic_did":
            # For basic 2x2 DiD, each cell has n_treated/2 or n_control/2 obs
            # assuming balanced design
            n_t_pre = n_treated  # treated units in pre-period
            n_t_post = n_treated  # treated units in post-period
            n_c_pre = n_control
            n_c_post = n_control

            variance = sigma**2 * (
                1 / n_t_post + 1 / n_t_pre + 1 / n_c_post + 1 / n_c_pre
            )
        elif design == "panel":
            # Panel DiD with multiple periods
            # Account for serial correlation via ICC
            T = n_pre + n_post

            # Design effect for clustering
            design_effect = 1 + (T - 1) * rho

            # Base variance (as if independent)
            base_var = sigma**2 * (1 / n_treated + 1 / n_control)

            # Adjust for clustering (Moulton factor)
            variance = base_var * design_effect / T
        else:
            raise ValueError(f"Unknown design: {design}")

        return variance

    def power(
        self,
        effect_size: float,
        n_treated: int,
        n_control: int,
        sigma: float,
        n_pre: int = 1,
        n_post: int = 1,
        rho: float = 0.0,
    ) -> PowerResults:
        """
        Calculate statistical power for given effect size and sample.

        Parameters
        ----------
        effect_size : float
            Expected treatment effect size.
        n_treated : int
            Number of treated units.
        n_control : int
            Number of control units.
        sigma : float
            Residual standard deviation.
        n_pre : int, default=1
            Number of pre-treatment periods.
        n_post : int, default=1
            Number of post-treatment periods.
        rho : float, default=0.0
            Intra-cluster correlation for panel data.

        Returns
        -------
        PowerResults
            Power analysis results.

        Examples
        --------
        >>> pa = PowerAnalysis()
        >>> results = pa.power(effect_size=2.0, n_treated=50, n_control=50, sigma=5.0)
        >>> print(f"Power: {results.power:.1%}")
        """
        T = n_pre + n_post
        design = "panel" if T > 2 else "basic_did"

        variance = self._compute_variance(
            n_treated, n_control, n_pre, n_post, sigma, rho, design
        )
        se = np.sqrt(variance)

        # Calculate power
        if self.alternative == "two-sided":
            z_alpha = stats.norm.ppf(1 - self.alpha / 2)
            # Power = P(reject | effect) = P(|Z| > z_alpha | effect)
            power_val = (
                1 - stats.norm.cdf(z_alpha - effect_size / se)
                + stats.norm.cdf(-z_alpha - effect_size / se)
            )
        elif self.alternative == "greater":
            z_alpha = stats.norm.ppf(1 - self.alpha)
            power_val = 1 - stats.norm.cdf(z_alpha - effect_size / se)
        else:  # less
            z_alpha = stats.norm.ppf(1 - self.alpha)
            power_val = stats.norm.cdf(-z_alpha - effect_size / se)

        # Also compute MDE and required N for reference
        mde = self._compute_mde_from_se(se)
        required_n = self._compute_required_n(
            effect_size, sigma, n_pre, n_post, rho, design,
            n_treated / (n_treated + n_control)
        )

        return PowerResults(
            power=power_val,
            mde=mde,
            required_n=required_n,
            effect_size=effect_size,
            alpha=self.alpha,
            alternative=self.alternative,
            n_treated=n_treated,
            n_control=n_control,
            n_pre=n_pre,
            n_post=n_post,
            sigma=sigma,
            rho=rho,
            design=design,
        )

    def _compute_mde_from_se(self, se: float) -> float:
        """Compute MDE given standard error."""
        z_alpha, z_beta = self._get_critical_values()
        return (z_alpha + z_beta) * se

    def mde(
        self,
        n_treated: int,
        n_control: int,
        sigma: float,
        n_pre: int = 1,
        n_post: int = 1,
        rho: float = 0.0,
    ) -> PowerResults:
        """
        Calculate minimum detectable effect given sample size.

        The MDE is the smallest effect size that can be detected with the
        specified power and significance level.

        Parameters
        ----------
        n_treated : int
            Number of treated units.
        n_control : int
            Number of control units.
        sigma : float
            Residual standard deviation.
        n_pre : int, default=1
            Number of pre-treatment periods.
        n_post : int, default=1
            Number of post-treatment periods.
        rho : float, default=0.0
            Intra-cluster correlation for panel data.

        Returns
        -------
        PowerResults
            Power analysis results including MDE.

        Examples
        --------
        >>> pa = PowerAnalysis(power=0.80)
        >>> results = pa.mde(n_treated=100, n_control=100, sigma=10.0)
        >>> print(f"MDE: {results.mde:.2f}")
        """
        T = n_pre + n_post
        design = "panel" if T > 2 else "basic_did"

        variance = self._compute_variance(
            n_treated, n_control, n_pre, n_post, sigma, rho, design
        )
        se = np.sqrt(variance)

        mde = self._compute_mde_from_se(se)

        return PowerResults(
            power=self.target_power,
            mde=mde,
            required_n=n_treated + n_control,
            effect_size=mde,
            alpha=self.alpha,
            alternative=self.alternative,
            n_treated=n_treated,
            n_control=n_control,
            n_pre=n_pre,
            n_post=n_post,
            sigma=sigma,
            rho=rho,
            design=design,
        )

    def _compute_required_n(
        self,
        effect_size: float,
        sigma: float,
        n_pre: int,
        n_post: int,
        rho: float,
        design: str,
        treat_frac: float = 0.5,
    ) -> int:
        """Compute required sample size for given effect."""
        # Handle edge case of zero effect size
        if effect_size == 0:
            return MAX_SAMPLE_SIZE  # Can't detect zero effect

        z_alpha, z_beta = self._get_critical_values()

        T = n_pre + n_post

        if design == "basic_did":
            # Var = sigma^2 * (1/n_t + 1/n_t + 1/n_c + 1/n_c) = sigma^2 * (2/n_t + 2/n_c)
            # For balanced: Var = sigma^2 * 4/n where n = n_t = n_c
            # SE = sqrt(Var), effect_size = (z_alpha + z_beta) * SE
            # n = 4 * sigma^2 * (z_alpha + z_beta)^2 / effect_size^2

            # For general allocation with treat_frac:
            # Var = sigma^2 * 2 * (1/(N*p) + 1/(N*(1-p)))
            #     = 2 * sigma^2 / N * (1/p + 1/(1-p))
            #     = 2 * sigma^2 / N * (1/(p*(1-p)))

            n_total = (
                2 * sigma**2 * (z_alpha + z_beta)**2
                / (effect_size**2 * treat_frac * (1 - treat_frac))
            )
        else:  # panel
            design_effect = 1 + (T - 1) * rho

            # Var = sigma^2 * (1/n_t + 1/n_c) * design_effect / T
            # For balanced: Var = 2 * sigma^2 / N * design_effect / T

            n_total = (
                2 * sigma**2 * (z_alpha + z_beta)**2 * design_effect
                / (effect_size**2 * treat_frac * (1 - treat_frac) * T)
            )

        # Handle infinity case (extremely small effect)
        if np.isinf(n_total):
            return MAX_SAMPLE_SIZE

        return max(4, int(np.ceil(n_total)))  # At least 4 units

    def sample_size(
        self,
        effect_size: float,
        sigma: float,
        n_pre: int = 1,
        n_post: int = 1,
        rho: float = 0.0,
        treat_frac: float = 0.5,
    ) -> PowerResults:
        """
        Calculate required sample size to detect given effect.

        Parameters
        ----------
        effect_size : float
            Treatment effect to detect.
        sigma : float
            Residual standard deviation.
        n_pre : int, default=1
            Number of pre-treatment periods.
        n_post : int, default=1
            Number of post-treatment periods.
        rho : float, default=0.0
            Intra-cluster correlation for panel data.
        treat_frac : float, default=0.5
            Fraction of units assigned to treatment.

        Returns
        -------
        PowerResults
            Power analysis results including required sample size.

        Examples
        --------
        >>> pa = PowerAnalysis(power=0.80)
        >>> results = pa.sample_size(effect_size=5.0, sigma=10.0)
        >>> print(f"Required N: {results.required_n}")
        """
        T = n_pre + n_post
        design = "panel" if T > 2 else "basic_did"

        n_total = self._compute_required_n(
            effect_size, sigma, n_pre, n_post, rho, design, treat_frac
        )

        n_treated = max(2, int(np.ceil(n_total * treat_frac)))
        n_control = max(2, n_total - n_treated)
        n_total = n_treated + n_control

        # Compute actual power achieved
        variance = self._compute_variance(
            n_treated, n_control, n_pre, n_post, sigma, rho, design
        )
        se = np.sqrt(variance)
        mde = self._compute_mde_from_se(se)

        return PowerResults(
            power=self.target_power,
            mde=mde,
            required_n=n_total,
            effect_size=effect_size,
            alpha=self.alpha,
            alternative=self.alternative,
            n_treated=n_treated,
            n_control=n_control,
            n_pre=n_pre,
            n_post=n_post,
            sigma=sigma,
            rho=rho,
            design=design,
        )

    def power_curve(
        self,
        n_treated: int,
        n_control: int,
        sigma: float,
        effect_sizes: Optional[List[float]] = None,
        n_pre: int = 1,
        n_post: int = 1,
        rho: float = 0.0,
    ) -> pd.DataFrame:
        """
        Compute power for a range of effect sizes.

        Parameters
        ----------
        n_treated : int
            Number of treated units.
        n_control : int
            Number of control units.
        sigma : float
            Residual standard deviation.
        effect_sizes : list of float, optional
            Effect sizes to evaluate. If None, uses a range from 0 to 3*MDE.
        n_pre : int, default=1
            Number of pre-treatment periods.
        n_post : int, default=1
            Number of post-treatment periods.
        rho : float, default=0.0
            Intra-cluster correlation.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns 'effect_size' and 'power'.

        Examples
        --------
        >>> pa = PowerAnalysis()
        >>> curve = pa.power_curve(n_treated=50, n_control=50, sigma=5.0)
        >>> print(curve)
        """
        # First get MDE to determine default range
        mde_result = self.mde(n_treated, n_control, sigma, n_pre, n_post, rho)

        if effect_sizes is None:
            # Generate range from 0 to 2*MDE
            effect_sizes = np.linspace(0, 2.5 * mde_result.mde, 50).tolist()

        powers = []
        for es in effect_sizes:
            result = self.power(
                effect_size=es,
                n_treated=n_treated,
                n_control=n_control,
                sigma=sigma,
                n_pre=n_pre,
                n_post=n_post,
                rho=rho,
            )
            powers.append(result.power)

        return pd.DataFrame({"effect_size": effect_sizes, "power": powers})

    def sample_size_curve(
        self,
        effect_size: float,
        sigma: float,
        sample_sizes: Optional[List[int]] = None,
        n_pre: int = 1,
        n_post: int = 1,
        rho: float = 0.0,
        treat_frac: float = 0.5,
    ) -> pd.DataFrame:
        """
        Compute power for a range of sample sizes.

        Parameters
        ----------
        effect_size : float
            Treatment effect size.
        sigma : float
            Residual standard deviation.
        sample_sizes : list of int, optional
            Total sample sizes to evaluate. If None, uses sensible range.
        n_pre : int, default=1
            Number of pre-treatment periods.
        n_post : int, default=1
            Number of post-treatment periods.
        rho : float, default=0.0
            Intra-cluster correlation.
        treat_frac : float, default=0.5
            Fraction assigned to treatment.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns 'sample_size' and 'power'.
        """
        # Get required N to determine default range
        required = self.sample_size(
            effect_size, sigma, n_pre, n_post, rho, treat_frac
        )

        if sample_sizes is None:
            min_n = max(10, required.required_n // 4)
            max_n = required.required_n * 2
            sample_sizes = list(range(min_n, max_n + 1, max(1, (max_n - min_n) // 50)))

        powers = []
        for n in sample_sizes:
            n_treated = max(2, int(n * treat_frac))
            n_control = max(2, n - n_treated)
            result = self.power(
                effect_size=effect_size,
                n_treated=n_treated,
                n_control=n_control,
                sigma=sigma,
                n_pre=n_pre,
                n_post=n_post,
                rho=rho,
            )
            powers.append(result.power)

        return pd.DataFrame({"sample_size": sample_sizes, "power": powers})


def simulate_power(
    estimator: Any,
    n_units: int = 100,
    n_periods: int = 4,
    treatment_effect: float = 5.0,
    treatment_fraction: float = 0.5,
    treatment_period: int = 2,
    sigma: float = 1.0,
    n_simulations: int = 500,
    alpha: float = 0.05,
    effect_sizes: Optional[List[float]] = None,
    seed: Optional[int] = None,
    data_generator: Optional[Callable] = None,
    data_generator_kwargs: Optional[Dict[str, Any]] = None,
    estimator_kwargs: Optional[Dict[str, Any]] = None,
    progress: bool = True,
) -> SimulationPowerResults:
    """
    Estimate power using Monte Carlo simulation.

    This function simulates datasets with known treatment effects and estimates
    power as the fraction of simulations where the null hypothesis is rejected.
    This is the recommended approach for complex designs like staggered adoption.

    Parameters
    ----------
    estimator : estimator object
        DiD estimator to use (e.g., DifferenceInDifferences, CallawaySantAnna).
    n_units : int, default=100
        Number of units per simulation.
    n_periods : int, default=4
        Number of time periods.
    treatment_effect : float, default=5.0
        True treatment effect to simulate.
    treatment_fraction : float, default=0.5
        Fraction of units that are treated.
    treatment_period : int, default=2
        First post-treatment period (0-indexed).
    sigma : float, default=1.0
        Residual standard deviation (noise level).
    n_simulations : int, default=500
        Number of Monte Carlo simulations.
    alpha : float, default=0.05
        Significance level for hypothesis tests.
    effect_sizes : list of float, optional
        Multiple effect sizes to evaluate for power curve.
        If None, uses only treatment_effect.
    seed : int, optional
        Random seed for reproducibility.
    data_generator : callable, optional
        Custom data generation function. Should accept same signature as
        generate_did_data(). If None, uses generate_did_data().
    data_generator_kwargs : dict, optional
        Additional keyword arguments for data generator.
    estimator_kwargs : dict, optional
        Additional keyword arguments for estimator.fit().
    progress : bool, default=True
        Whether to print progress updates.

    Returns
    -------
    SimulationPowerResults
        Simulation-based power analysis results.

    Examples
    --------
    Basic power simulation:

    >>> from diff_diff import DifferenceInDifferences, simulate_power
    >>> did = DifferenceInDifferences()
    >>> results = simulate_power(
    ...     estimator=did,
    ...     n_units=100,
    ...     treatment_effect=5.0,
    ...     sigma=5.0,
    ...     n_simulations=500,
    ...     seed=42
    ... )
    >>> print(f"Power: {results.power:.1%}")

    Power curve over multiple effect sizes:

    >>> results = simulate_power(
    ...     estimator=did,
    ...     effect_sizes=[1.0, 2.0, 3.0, 5.0, 7.0],
    ...     n_simulations=200,
    ...     seed=42
    ... )
    >>> print(results.power_curve_df())

    With Callaway-Sant'Anna for staggered designs:

    >>> from diff_diff import CallawaySantAnna
    >>> cs = CallawaySantAnna()
    >>> # Custom data generator for staggered adoption
    >>> def staggered_data(n_units, n_periods, treatment_effect, **kwargs):
    ...     # Your staggered data generation logic
    ...     ...
    >>> results = simulate_power(cs, data_generator=staggered_data, ...)

    Notes
    -----
    The simulation approach:
    1. Generate data with known treatment effect
    2. Fit the estimator and record the p-value
    3. Repeat n_simulations times
    4. Power = fraction of simulations where p-value < alpha

    For staggered designs, you'll need to provide a custom data_generator
    that creates appropriate staggered treatment timing.

    References
    ----------
    Burlig, F., Preonas, L., & Woerman, M. (2020). "Panel Data and Experimental Design."
    """
    from diff_diff.prep import generate_did_data

    rng = np.random.default_rng(seed)

    # Use default data generator if none provided
    if data_generator is None:
        data_generator = generate_did_data

    data_gen_kwargs = data_generator_kwargs or {}
    est_kwargs = estimator_kwargs or {}

    # Determine effect sizes to test
    if effect_sizes is None:
        effect_sizes = [treatment_effect]

    all_powers = []

    # For the primary effect (last in list), collect detailed results
    # Use index-based comparison to avoid float precision issues
    if len(effect_sizes) == 1:
        primary_idx = 0
    else:
        # Find index of treatment_effect in effect_sizes
        primary_idx = -1
        for i, es in enumerate(effect_sizes):
            if np.isclose(es, treatment_effect):
                primary_idx = i
                break
        if primary_idx == -1:
            primary_idx = len(effect_sizes) - 1  # Default to last

    primary_effect = effect_sizes[primary_idx]

    for effect_idx, effect in enumerate(effect_sizes):
        is_primary = (effect_idx == primary_idx)

        estimates = []
        ses = []
        p_values = []
        rejections = []
        ci_contains_true = []
        n_failures = 0

        for sim in range(n_simulations):
            if progress and sim % 100 == 0 and sim > 0:
                pct = (sim + effect_idx * n_simulations) / (len(effect_sizes) * n_simulations)
                print(f"  Simulation progress: {pct:.0%}")

            # Generate data
            sim_seed = rng.integers(0, 2**31)
            data = data_generator(
                n_units=n_units,
                n_periods=n_periods,
                treatment_effect=effect,
                treatment_fraction=treatment_fraction,
                treatment_period=treatment_period,
                noise_sd=sigma,
                seed=sim_seed,
                **data_gen_kwargs
            )

            try:
                # Fit estimator
                # Try to determine the right arguments based on estimator type
                estimator_name = type(estimator).__name__

                if estimator_name == "DifferenceInDifferences":
                    result = estimator.fit(
                        data,
                        outcome="outcome",
                        treatment="treated",
                        time="post",
                        **est_kwargs
                    )
                elif estimator_name == "TwoWayFixedEffects":
                    result = estimator.fit(
                        data,
                        outcome="outcome",
                        treatment="treated",
                        time="period",
                        unit="unit",
                        **est_kwargs
                    )
                elif estimator_name == "MultiPeriodDiD":
                    post_periods = list(range(treatment_period, n_periods))
                    result = estimator.fit(
                        data,
                        outcome="outcome",
                        treatment="treated",
                        time="period",
                        post_periods=post_periods,
                        **est_kwargs
                    )
                elif estimator_name == "CallawaySantAnna":
                    # Need to create first_treat column for staggered
                    # For standard generate_did_data, convert to first_treat format
                    data = data.copy()
                    data["first_treat"] = np.where(
                        data["treated"] == 1, treatment_period, 0
                    )
                    result = estimator.fit(
                        data,
                        outcome="outcome",
                        unit="unit",
                        time="period",
                        first_treat="first_treat",
                        **est_kwargs
                    )
                else:
                    # Generic fallback - try common signature
                    result = estimator.fit(
                        data,
                        outcome="outcome",
                        treatment="treated",
                        time="post",
                        **est_kwargs
                    )

                # Extract results
                att = result.att if hasattr(result, 'att') else result.avg_att
                se = result.se if hasattr(result, 'se') else result.avg_se
                p_val = result.p_value if hasattr(result, 'p_value') else result.avg_p_value
                ci = result.conf_int if hasattr(result, 'conf_int') else result.avg_conf_int

                estimates.append(att)
                ses.append(se)
                p_values.append(p_val)
                rejections.append(p_val < alpha)
                ci_contains_true.append(ci[0] <= effect <= ci[1])

            except Exception as e:
                # Track failed simulations
                n_failures += 1
                if progress:
                    print(f"  Warning: Simulation {sim} failed: {e}")
                continue

        # Warn if too many simulations failed
        failure_rate = n_failures / n_simulations
        if failure_rate > 0.1:
            warnings.warn(
                f"{n_failures}/{n_simulations} simulations ({failure_rate:.1%}) failed "
                f"for effect_size={effect}. Check estimator and data generator.",
                UserWarning
            )

        if len(estimates) == 0:
            raise RuntimeError("All simulations failed. Check estimator and data generator.")

        # Compute power and SE
        power_val = np.mean(rejections)
        power_se = np.sqrt(power_val * (1 - power_val) / len(rejections))

        all_powers.append(power_val)

        # Store detailed results for primary effect
        if is_primary:
            primary_estimates = estimates
            primary_ses = ses
            primary_p_values = p_values
            primary_rejections = rejections
            primary_ci_contains = ci_contains_true

    # Compute confidence interval for power (primary effect)
    power_val = all_powers[primary_idx]
    n_valid = len(primary_rejections)
    power_se = np.sqrt(power_val * (1 - power_val) / n_valid)
    z = stats.norm.ppf(0.975)
    power_ci = (
        max(0.0, power_val - z * power_se),
        min(1.0, power_val + z * power_se)
    )

    # Compute summary statistics
    mean_estimate = np.mean(primary_estimates)
    std_estimate = np.std(primary_estimates, ddof=1)
    mean_se = np.mean(primary_ses)
    coverage = np.mean(primary_ci_contains)

    return SimulationPowerResults(
        power=power_val,
        power_se=power_se,
        power_ci=power_ci,
        rejection_rate=power_val,
        mean_estimate=mean_estimate,
        std_estimate=std_estimate,
        mean_se=mean_se,
        coverage=coverage,
        n_simulations=n_valid,
        effect_sizes=effect_sizes,
        powers=all_powers,
        true_effect=primary_effect,
        alpha=alpha,
        estimator_name=type(estimator).__name__,
        simulation_results=[
            {"estimate": e, "se": s, "p_value": p, "rejected": r}
            for e, s, p, r in zip(primary_estimates, primary_ses,
                                   primary_p_values, primary_rejections)
        ],
    )


def compute_mde(
    n_treated: int,
    n_control: int,
    sigma: float,
    power: float = 0.80,
    alpha: float = 0.05,
    n_pre: int = 1,
    n_post: int = 1,
    rho: float = 0.0,
) -> float:
    """
    Convenience function to compute minimum detectable effect.

    Parameters
    ----------
    n_treated : int
        Number of treated units.
    n_control : int
        Number of control units.
    sigma : float
        Residual standard deviation.
    power : float, default=0.80
        Target statistical power.
    alpha : float, default=0.05
        Significance level.
    n_pre : int, default=1
        Number of pre-treatment periods.
    n_post : int, default=1
        Number of post-treatment periods.
    rho : float, default=0.0
        Intra-cluster correlation.

    Returns
    -------
    float
        Minimum detectable effect size.

    Examples
    --------
    >>> mde = compute_mde(n_treated=50, n_control=50, sigma=10.0)
    >>> print(f"MDE: {mde:.2f}")
    """
    pa = PowerAnalysis(alpha=alpha, power=power)
    result = pa.mde(n_treated, n_control, sigma, n_pre, n_post, rho)
    return result.mde


def compute_power(
    effect_size: float,
    n_treated: int,
    n_control: int,
    sigma: float,
    alpha: float = 0.05,
    n_pre: int = 1,
    n_post: int = 1,
    rho: float = 0.0,
) -> float:
    """
    Convenience function to compute power for given effect and sample.

    Parameters
    ----------
    effect_size : float
        Expected treatment effect.
    n_treated : int
        Number of treated units.
    n_control : int
        Number of control units.
    sigma : float
        Residual standard deviation.
    alpha : float, default=0.05
        Significance level.
    n_pre : int, default=1
        Number of pre-treatment periods.
    n_post : int, default=1
        Number of post-treatment periods.
    rho : float, default=0.0
        Intra-cluster correlation.

    Returns
    -------
    float
        Statistical power.

    Examples
    --------
    >>> power = compute_power(effect_size=5.0, n_treated=50, n_control=50, sigma=10.0)
    >>> print(f"Power: {power:.1%}")
    """
    pa = PowerAnalysis(alpha=alpha)
    result = pa.power(effect_size, n_treated, n_control, sigma, n_pre, n_post, rho)
    return result.power


def compute_sample_size(
    effect_size: float,
    sigma: float,
    power: float = 0.80,
    alpha: float = 0.05,
    n_pre: int = 1,
    n_post: int = 1,
    rho: float = 0.0,
    treat_frac: float = 0.5,
) -> int:
    """
    Convenience function to compute required sample size.

    Parameters
    ----------
    effect_size : float
        Treatment effect to detect.
    sigma : float
        Residual standard deviation.
    power : float, default=0.80
        Target statistical power.
    alpha : float, default=0.05
        Significance level.
    n_pre : int, default=1
        Number of pre-treatment periods.
    n_post : int, default=1
        Number of post-treatment periods.
    rho : float, default=0.0
        Intra-cluster correlation.
    treat_frac : float, default=0.5
        Fraction assigned to treatment.

    Returns
    -------
    int
        Required total sample size.

    Examples
    --------
    >>> n = compute_sample_size(effect_size=5.0, sigma=10.0)
    >>> print(f"Required N: {n}")
    """
    pa = PowerAnalysis(alpha=alpha, power=power)
    result = pa.sample_size(effect_size, sigma, n_pre, n_post, rho, treat_frac)
    return result.required_n
