"""
Results classes for difference-in-differences estimation.

Provides statsmodels-style output with a more Pythonic interface.
"""

from dataclasses import dataclass, field
from typing import Optional

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
    conf_int: tuple
    n_obs: int
    n_treated: int
    n_control: int
    alpha: float = 0.05
    coefficients: Optional[dict] = field(default=None)
    vcov: Optional[np.ndarray] = field(default=None)
    residuals: Optional[np.ndarray] = field(default=None)
    fitted_values: Optional[np.ndarray] = field(default=None)
    r_squared: Optional[float] = field(default=None)

    def __repr__(self) -> str:
        """Concise string representation."""
        sig = "***" if self.p_value < 0.001 else "**" if self.p_value < 0.01 else "*" if self.p_value < 0.05 else ""
        return (
            f"DiDResults(ATT={self.att:.4f}{sig}, "
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

        lines.extend([
            "",
            "-" * 70,
            f"{'Parameter':<15} {'Estimate':>12} {'Std. Err.':>12} {'t-stat':>10} {'P>|t|':>10}",
            "-" * 70,
            f"{'ATT':<15} {self.att:>12.4f} {self.se:>12.4f} {self.t_stat:>10.3f} {self.p_value:>10.4f}",
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

    def to_dict(self) -> dict:
        """
        Convert results to a dictionary.

        Returns
        -------
        dict
            Dictionary containing all estimation results.
        """
        return {
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
        }

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
        if self.p_value < 0.001:
            return "***"
        elif self.p_value < 0.01:
            return "**"
        elif self.p_value < 0.05:
            return "*"
        elif self.p_value < 0.1:
            return "."
        return ""
