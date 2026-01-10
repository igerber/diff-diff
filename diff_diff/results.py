"""
Results classes for difference-in-differences estimation.

Provides statsmodels-style output with a more Pythonic interface.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class DiDResults:
    """
    Results from a Difference-in-Differences estimation.

    Provides easy access to coefficients, standard errors, confidence intervals,
    and summary statistics in a Pythonic way.

    Attributes
    ----------
    att : float
        Average Treatment effect on the Treated (ATT).
    se : float
        Standard error of the ATT estimate.
    t_stat : float
        T-statistic for the ATT estimate.
    p_value : float
        P-value for the null hypothesis that ATT = 0.
    conf_int : tuple[float, float]
        Confidence interval for the ATT.
    n_obs : int
        Number of observations used in estimation.
    n_treated : int
        Number of treated units.
    n_control : int
        Number of control units.
    """

    att: float
    se: float
    t_stat: float
    p_value: float
    conf_int: Tuple[float, float]
    n_obs: int
    n_treated: int
    n_control: int
    alpha: float = 0.05
    coefficients: Optional[Dict[str, float]] = field(default=None)
    vcov: Optional[np.ndarray] = field(default=None)
    residuals: Optional[np.ndarray] = field(default=None)
    fitted_values: Optional[np.ndarray] = field(default=None)
    r_squared: Optional[float] = field(default=None)
    # Bootstrap inference fields
    inference_method: str = field(default="analytical")
    n_bootstrap: Optional[int] = field(default=None)
    n_clusters: Optional[int] = field(default=None)
    bootstrap_distribution: Optional[np.ndarray] = field(default=None, repr=False)

    def __repr__(self) -> str:
        """Concise string representation."""
        return (
            f"DiDResults(ATT={self.att:.4f}{self.significance_stars}, "
            f"SE={self.se:.4f}, "
            f"p={self.p_value:.4f})"
        )

    def summary(self, alpha: Optional[float] = None) -> str:
        """
        Generate a formatted summary of the estimation results.

        Parameters
        ----------
        alpha : float, optional
            Significance level for confidence intervals. Defaults to the
            alpha used during estimation.

        Returns
        -------
        str
            Formatted summary table.
        """
        alpha = alpha or self.alpha
        conf_level = int((1 - alpha) * 100)

        lines = [
            "=" * 70,
            "Difference-in-Differences Estimation Results".center(70),
            "=" * 70,
            "",
            f"{'Observations:':<25} {self.n_obs:>10}",
            f"{'Treated units:':<25} {self.n_treated:>10}",
            f"{'Control units:':<25} {self.n_control:>10}",
        ]

        if self.r_squared is not None:
            lines.append(f"{'R-squared:':<25} {self.r_squared:>10.4f}")

        # Add inference method info
        if self.inference_method != "analytical":
            lines.append(f"{'Inference method:':<25} {self.inference_method:>10}")
            if self.n_bootstrap is not None:
                lines.append(f"{'Bootstrap replications:':<25} {self.n_bootstrap:>10}")
            if self.n_clusters is not None:
                lines.append(f"{'Number of clusters:':<25} {self.n_clusters:>10}")

        lines.extend([
            "",
            "-" * 70,
            f"{'Parameter':<15} {'Estimate':>12} {'Std. Err.':>12} {'t-stat':>10} {'P>|t|':>10} {'':>5}",
            "-" * 70,
            f"{'ATT':<15} {self.att:>12.4f} {self.se:>12.4f} {self.t_stat:>10.3f} {self.p_value:>10.4f} {self.significance_stars:>5}",
            "-" * 70,
            "",
            f"{conf_level}% Confidence Interval: [{self.conf_int[0]:.4f}, {self.conf_int[1]:.4f}]",
        ])

        # Add significance codes
        lines.extend([
            "",
            "Signif. codes: '***' 0.001, '**' 0.01, '*' 0.05, '.' 0.1",
            "=" * 70,
        ])

        return "\n".join(lines)

    def print_summary(self, alpha: Optional[float] = None) -> None:
        """Print the summary to stdout."""
        print(self.summary(alpha))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert results to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing all estimation results.
        """
        result = {
            "att": self.att,
            "se": self.se,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "conf_int_lower": self.conf_int[0],
            "conf_int_upper": self.conf_int[1],
            "n_obs": self.n_obs,
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "r_squared": self.r_squared,
            "inference_method": self.inference_method,
        }
        if self.n_bootstrap is not None:
            result["n_bootstrap"] = self.n_bootstrap
        if self.n_clusters is not None:
            result["n_clusters"] = self.n_clusters
        return result

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with estimation results.
        """
        return pd.DataFrame([self.to_dict()])

    @property
    def is_significant(self) -> bool:
        """Check if the ATT is statistically significant at the alpha level."""
        return bool(self.p_value < self.alpha)

    @property
    def significance_stars(self) -> str:
        """Return significance stars based on p-value."""
        return _get_significance_stars(self.p_value)


def _get_significance_stars(p_value: float) -> str:
    """Return significance stars based on p-value."""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    elif p_value < 0.1:
        return "."
    return ""


@dataclass
class PeriodEffect:
    """
    Treatment effect for a single time period.

    Attributes
    ----------
    period : any
        The time period identifier.
    effect : float
        The treatment effect estimate for this period.
    se : float
        Standard error of the effect estimate.
    t_stat : float
        T-statistic for the effect estimate.
    p_value : float
        P-value for the null hypothesis that effect = 0.
    conf_int : tuple[float, float]
        Confidence interval for the effect.
    """

    period: Any
    effect: float
    se: float
    t_stat: float
    p_value: float
    conf_int: Tuple[float, float]

    def __repr__(self) -> str:
        """Concise string representation."""
        sig = _get_significance_stars(self.p_value)
        return (
            f"PeriodEffect(period={self.period}, effect={self.effect:.4f}{sig}, "
            f"SE={self.se:.4f}, p={self.p_value:.4f})"
        )

    @property
    def is_significant(self) -> bool:
        """Check if the effect is statistically significant at 0.05 level."""
        return bool(self.p_value < 0.05)

    @property
    def significance_stars(self) -> str:
        """Return significance stars based on p-value."""
        return _get_significance_stars(self.p_value)


@dataclass
class MultiPeriodDiDResults:
    """
    Results from a Multi-Period Difference-in-Differences estimation.

    Provides access to period-specific treatment effects as well as
    an aggregate average treatment effect.

    Attributes
    ----------
    period_effects : dict[any, PeriodEffect]
        Dictionary mapping period identifiers to their PeriodEffect objects.
    avg_att : float
        Average Treatment effect on the Treated across all post-periods.
    avg_se : float
        Standard error of the average ATT.
    avg_t_stat : float
        T-statistic for the average ATT.
    avg_p_value : float
        P-value for the null hypothesis that average ATT = 0.
    avg_conf_int : tuple[float, float]
        Confidence interval for the average ATT.
    n_obs : int
        Number of observations used in estimation.
    n_treated : int
        Number of treated observations.
    n_control : int
        Number of control observations.
    pre_periods : list
        List of pre-treatment period identifiers.
    post_periods : list
        List of post-treatment period identifiers.
    """

    period_effects: Dict[Any, PeriodEffect]
    avg_att: float
    avg_se: float
    avg_t_stat: float
    avg_p_value: float
    avg_conf_int: Tuple[float, float]
    n_obs: int
    n_treated: int
    n_control: int
    pre_periods: List[Any]
    post_periods: List[Any]
    alpha: float = 0.05
    coefficients: Optional[Dict[str, float]] = field(default=None)
    vcov: Optional[np.ndarray] = field(default=None)
    residuals: Optional[np.ndarray] = field(default=None)
    fitted_values: Optional[np.ndarray] = field(default=None)
    r_squared: Optional[float] = field(default=None)

    def __repr__(self) -> str:
        """Concise string representation."""
        sig = _get_significance_stars(self.avg_p_value)
        return (
            f"MultiPeriodDiDResults(avg_ATT={self.avg_att:.4f}{sig}, "
            f"SE={self.avg_se:.4f}, "
            f"n_post_periods={len(self.post_periods)})"
        )

    def summary(self, alpha: Optional[float] = None) -> str:
        """
        Generate a formatted summary of the estimation results.

        Parameters
        ----------
        alpha : float, optional
            Significance level for confidence intervals. Defaults to the
            alpha used during estimation.

        Returns
        -------
        str
            Formatted summary table.
        """
        alpha = alpha or self.alpha
        conf_level = int((1 - alpha) * 100)

        lines = [
            "=" * 80,
            "Multi-Period Difference-in-Differences Estimation Results".center(80),
            "=" * 80,
            "",
            f"{'Observations:':<25} {self.n_obs:>10}",
            f"{'Treated observations:':<25} {self.n_treated:>10}",
            f"{'Control observations:':<25} {self.n_control:>10}",
            f"{'Pre-treatment periods:':<25} {len(self.pre_periods):>10}",
            f"{'Post-treatment periods:':<25} {len(self.post_periods):>10}",
        ]

        if self.r_squared is not None:
            lines.append(f"{'R-squared:':<25} {self.r_squared:>10.4f}")

        # Period-specific effects
        lines.extend([
            "",
            "-" * 80,
            "Period-Specific Treatment Effects".center(80),
            "-" * 80,
            f"{'Period':<15} {'Estimate':>12} {'Std. Err.':>12} {'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}",
            "-" * 80,
        ])

        for period in self.post_periods:
            pe = self.period_effects[period]
            stars = pe.significance_stars
            lines.append(
                f"{str(period):<15} {pe.effect:>12.4f} {pe.se:>12.4f} "
                f"{pe.t_stat:>10.3f} {pe.p_value:>10.4f} {stars:>6}"
            )

        # Average effect
        lines.extend([
            "-" * 80,
            "",
            "-" * 80,
            "Average Treatment Effect (across post-periods)".center(80),
            "-" * 80,
            f"{'Parameter':<15} {'Estimate':>12} {'Std. Err.':>12} {'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}",
            "-" * 80,
            f"{'Avg ATT':<15} {self.avg_att:>12.4f} {self.avg_se:>12.4f} "
            f"{self.avg_t_stat:>10.3f} {self.avg_p_value:>10.4f} {self.significance_stars:>6}",
            "-" * 80,
            "",
            f"{conf_level}% Confidence Interval: [{self.avg_conf_int[0]:.4f}, {self.avg_conf_int[1]:.4f}]",
        ])

        # Add significance codes
        lines.extend([
            "",
            "Signif. codes: '***' 0.001, '**' 0.01, '*' 0.05, '.' 0.1",
            "=" * 80,
        ])

        return "\n".join(lines)

    def print_summary(self, alpha: Optional[float] = None) -> None:
        """Print the summary to stdout."""
        print(self.summary(alpha))

    def get_effect(self, period) -> PeriodEffect:
        """
        Get the treatment effect for a specific period.

        Parameters
        ----------
        period : any
            The period identifier.

        Returns
        -------
        PeriodEffect
            The treatment effect for the specified period.

        Raises
        ------
        KeyError
            If the period is not found in post-treatment periods.
        """
        if period not in self.period_effects:
            raise KeyError(
                f"Period '{period}' not found. "
                f"Available post-periods: {list(self.period_effects.keys())}"
            )
        return self.period_effects[period]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert results to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing all estimation results.
        """
        result: Dict[str, Any] = {
            "avg_att": self.avg_att,
            "avg_se": self.avg_se,
            "avg_t_stat": self.avg_t_stat,
            "avg_p_value": self.avg_p_value,
            "avg_conf_int_lower": self.avg_conf_int[0],
            "avg_conf_int_upper": self.avg_conf_int[1],
            "n_obs": self.n_obs,
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "n_pre_periods": len(self.pre_periods),
            "n_post_periods": len(self.post_periods),
            "r_squared": self.r_squared,
        }

        # Add period-specific effects
        for period, pe in self.period_effects.items():
            result[f"effect_period_{period}"] = pe.effect
            result[f"se_period_{period}"] = pe.se
            result[f"pval_period_{period}"] = pe.p_value

        return result

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert period-specific effects to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per post-treatment period.
        """
        rows = []
        for period, pe in self.period_effects.items():
            rows.append({
                "period": period,
                "effect": pe.effect,
                "se": pe.se,
                "t_stat": pe.t_stat,
                "p_value": pe.p_value,
                "conf_int_lower": pe.conf_int[0],
                "conf_int_upper": pe.conf_int[1],
                "is_significant": pe.is_significant,
            })
        return pd.DataFrame(rows)

    @property
    def is_significant(self) -> bool:
        """Check if the average ATT is statistically significant at the alpha level."""
        return bool(self.avg_p_value < self.alpha)

    @property
    def significance_stars(self) -> str:
        """Return significance stars for the average ATT based on p-value."""
        return _get_significance_stars(self.avg_p_value)


@dataclass
class SyntheticDiDResults:
    """
    Results from a Synthetic Difference-in-Differences estimation.

    Combines DiD with synthetic control by re-weighting control units to match
    pre-treatment trends of treated units.

    Attributes
    ----------
    att : float
        Average Treatment effect on the Treated (ATT).
    se : float
        Standard error of the ATT estimate (bootstrap or placebo-based).
    t_stat : float
        T-statistic for the ATT estimate.
    p_value : float
        P-value for the null hypothesis that ATT = 0.
    conf_int : tuple[float, float]
        Confidence interval for the ATT.
    n_obs : int
        Number of observations used in estimation.
    n_treated : int
        Number of treated units.
    n_control : int
        Number of control units.
    unit_weights : dict
        Dictionary mapping control unit IDs to their synthetic weights.
    time_weights : dict
        Dictionary mapping pre-treatment periods to their time weights.
    pre_periods : list
        List of pre-treatment period identifiers.
    post_periods : list
        List of post-treatment period identifiers.
    variance_method : str
        Method used for variance estimation: "bootstrap" or "placebo".
    """

    att: float
    se: float
    t_stat: float
    p_value: float
    conf_int: Tuple[float, float]
    n_obs: int
    n_treated: int
    n_control: int
    unit_weights: Dict[Any, float]
    time_weights: Dict[Any, float]
    pre_periods: List[Any]
    post_periods: List[Any]
    alpha: float = 0.05
    variance_method: str = field(default="bootstrap")
    lambda_reg: Optional[float] = field(default=None)
    pre_treatment_fit: Optional[float] = field(default=None)
    placebo_effects: Optional[np.ndarray] = field(default=None)
    n_bootstrap: Optional[int] = field(default=None)

    def __repr__(self) -> str:
        """Concise string representation."""
        sig = _get_significance_stars(self.p_value)
        return (
            f"SyntheticDiDResults(ATT={self.att:.4f}{sig}, "
            f"SE={self.se:.4f}, "
            f"p={self.p_value:.4f})"
        )

    def summary(self, alpha: Optional[float] = None) -> str:
        """
        Generate a formatted summary of the estimation results.

        Parameters
        ----------
        alpha : float, optional
            Significance level for confidence intervals. Defaults to the
            alpha used during estimation.

        Returns
        -------
        str
            Formatted summary table.
        """
        alpha = alpha or self.alpha
        conf_level = int((1 - alpha) * 100)

        lines = [
            "=" * 75,
            "Synthetic Difference-in-Differences Estimation Results".center(75),
            "=" * 75,
            "",
            f"{'Observations:':<25} {self.n_obs:>10}",
            f"{'Treated units:':<25} {self.n_treated:>10}",
            f"{'Control units:':<25} {self.n_control:>10}",
            f"{'Pre-treatment periods:':<25} {len(self.pre_periods):>10}",
            f"{'Post-treatment periods:':<25} {len(self.post_periods):>10}",
        ]

        if self.lambda_reg is not None:
            lines.append(f"{'Regularization (lambda):':<25} {self.lambda_reg:>10.4f}")

        if self.pre_treatment_fit is not None:
            lines.append(f"{'Pre-treatment fit (RMSE):':<25} {self.pre_treatment_fit:>10.4f}")

        # Variance method info
        lines.append(f"{'Variance method:':<25} {self.variance_method:>10}")
        if self.variance_method == "bootstrap" and self.n_bootstrap is not None:
            lines.append(f"{'Bootstrap replications:':<25} {self.n_bootstrap:>10}")

        lines.extend([
            "",
            "-" * 75,
            f"{'Parameter':<15} {'Estimate':>12} {'Std. Err.':>12} {'t-stat':>10} {'P>|t|':>10} {'':>5}",
            "-" * 75,
            f"{'ATT':<15} {self.att:>12.4f} {self.se:>12.4f} {self.t_stat:>10.3f} {self.p_value:>10.4f} {self.significance_stars:>5}",
            "-" * 75,
            "",
            f"{conf_level}% Confidence Interval: [{self.conf_int[0]:.4f}, {self.conf_int[1]:.4f}]",
        ])

        # Show top unit weights
        if self.unit_weights:
            sorted_weights = sorted(
                self.unit_weights.items(), key=lambda x: x[1], reverse=True
            )
            top_n = min(5, len(sorted_weights))
            lines.extend([
                "",
                "-" * 75,
                "Top Unit Weights (Synthetic Control)".center(75),
                "-" * 75,
            ])
            for unit, weight in sorted_weights[:top_n]:
                if weight > 0.001:  # Only show meaningful weights
                    lines.append(f"  Unit {unit}: {weight:.4f}")

            # Show how many units have non-trivial weight
            n_nonzero = sum(1 for w in self.unit_weights.values() if w > 0.001)
            lines.append(f"  ({n_nonzero} units with weight > 0.001)")

        # Add significance codes
        lines.extend([
            "",
            "Signif. codes: '***' 0.001, '**' 0.01, '*' 0.05, '.' 0.1",
            "=" * 75,
        ])

        return "\n".join(lines)

    def print_summary(self, alpha: Optional[float] = None) -> None:
        """Print the summary to stdout."""
        print(self.summary(alpha))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert results to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing all estimation results.
        """
        result = {
            "att": self.att,
            "se": self.se,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "conf_int_lower": self.conf_int[0],
            "conf_int_upper": self.conf_int[1],
            "n_obs": self.n_obs,
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "n_pre_periods": len(self.pre_periods),
            "n_post_periods": len(self.post_periods),
            "variance_method": self.variance_method,
            "lambda_reg": self.lambda_reg,
            "pre_treatment_fit": self.pre_treatment_fit,
        }
        if self.n_bootstrap is not None:
            result["n_bootstrap"] = self.n_bootstrap
        return result

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with estimation results.
        """
        return pd.DataFrame([self.to_dict()])

    def get_unit_weights_df(self) -> pd.DataFrame:
        """
        Get unit weights as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with unit IDs and their weights.
        """
        return pd.DataFrame([
            {"unit": unit, "weight": weight}
            for unit, weight in self.unit_weights.items()
        ]).sort_values("weight", ascending=False)

    def get_time_weights_df(self) -> pd.DataFrame:
        """
        Get time weights as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with time periods and their weights.
        """
        return pd.DataFrame([
            {"period": period, "weight": weight}
            for period, weight in self.time_weights.items()
        ])

    @property
    def is_significant(self) -> bool:
        """Check if the ATT is statistically significant at the alpha level."""
        return bool(self.p_value < self.alpha)

    @property
    def significance_stars(self) -> str:
        """Return significance stars based on p-value."""
        return _get_significance_stars(self.p_value)
