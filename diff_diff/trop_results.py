"""
Result containers for the Triply Robust Panel (TROP) estimator.

This module contains the TROPResults dataclass, _PrecomputedStructures TypedDict,
and _LAMBDA_INF sentinel value. Extracted from trop.py for module size management.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

from diff_diff.results import _format_survey_block, _get_significance_stars

__all__ = [
    "_LAMBDA_INF",
    "_PrecomputedStructures",
    "TROPResults",
]


# Sentinel value for "disabled" λ_nn in LOOCV parameter search.
# Per paper's footnote 2: λ_nn=∞ disables the factor model (L=0).
# For λ_time and λ_unit, 0.0 means disabled (uniform weights) per Eq. 3:
#   exp(-0 × dist) = 1 for all distances.
_LAMBDA_INF: float = float("inf")


class _PrecomputedStructures(TypedDict):
    """Type definition for pre-computed structures used across LOOCV iterations.

    These structures are computed once in `_precompute_structures()` and reused
    to avoid redundant computation during LOOCV and final estimation.
    """

    unit_dist_matrix: np.ndarray
    """Pairwise unit distance matrix (n_units x n_units)."""
    time_dist_matrix: np.ndarray
    """Time distance matrix where [t, s] = |t - s| (n_periods x n_periods)."""
    control_mask: np.ndarray
    """Boolean mask for control observations (D == 0)."""
    treated_mask: np.ndarray
    """Boolean mask for treated observations (D == 1)."""
    treated_observations: List[Tuple[int, int]]
    """List of (t, i) tuples for treated observations."""
    control_obs: List[Tuple[int, int]]
    """List of (t, i) tuples for valid control observations."""
    control_unit_idx: np.ndarray
    """Array of never-treated unit indices (for backward compatibility)."""
    D: np.ndarray
    """Treatment indicator matrix (n_periods x n_units) for dynamic control sets."""
    Y: np.ndarray
    """Outcome matrix (n_periods x n_units)."""
    n_units: int
    """Number of units."""
    n_periods: int
    """Number of time periods."""


@dataclass
class TROPResults:
    """
    Results from a Triply Robust Panel (TROP) estimation.

    TROP combines nuclear norm regularized factor estimation with
    exponential distance-based unit weights and time decay weights.

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
    n_treated_obs : int
        Number of treated unit-time observations.
    unit_effects : dict
        Estimated unit fixed effects (alpha_i).
    time_effects : dict
        Estimated time fixed effects (beta_t).
    treatment_effects : dict
        Individual treatment effects for each treated (unit, time) pair.
    lambda_time : float
        Selected time weight decay parameter from grid. 0.0 = uniform time
        weights (disabled) per Eq. 3.
    lambda_unit : float
        Selected unit weight decay parameter from grid. 0.0 = uniform unit
        weights (disabled) per Eq. 3.
    lambda_nn : float
        Selected nuclear norm regularization parameter from grid. inf = factor
        model disabled (L=0); converted to 1e10 internally for computation.
    factor_matrix : np.ndarray
        Estimated low-rank factor matrix L (n_periods x n_units).
    effective_rank : float
        Effective rank of the factor matrix (sum of singular values / max).
    loocv_score : float
        Leave-one-out cross-validation score for selected parameters.
    alpha : float
        Significance level for confidence interval.
    n_pre_periods : int
        Number of pre-treatment periods.
    n_post_periods : int
        Number of post-treatment periods (periods with D=1 observations).
    n_bootstrap : int, optional
        Number of bootstrap replications (if bootstrap variance).
    bootstrap_distribution : np.ndarray, optional
        Bootstrap distribution of estimates.
    """

    att: float
    se: float
    t_stat: float
    p_value: float
    conf_int: Tuple[float, float]
    n_obs: int
    n_treated: int
    n_control: int
    n_treated_obs: int
    unit_effects: Dict[Any, float]
    time_effects: Dict[Any, float]
    treatment_effects: Dict[Tuple[Any, Any], float]
    lambda_time: float
    lambda_unit: float
    lambda_nn: float
    factor_matrix: np.ndarray
    effective_rank: float
    loocv_score: float
    alpha: float = 0.05
    n_pre_periods: int = 0
    n_post_periods: int = 0
    n_bootstrap: Optional[int] = field(default=None)
    bootstrap_distribution: Optional[np.ndarray] = field(default=None, repr=False)
    # Survey design metadata (SurveyMetadata instance from diff_diff.survey)
    survey_metadata: Optional[Any] = field(default=None)

    def __repr__(self) -> str:
        """Concise string representation."""
        sig = _get_significance_stars(self.p_value)
        return (
            f"TROPResults(ATT={self.att:.4f}{sig}, "
            f"SE={self.se:.4f}, "
            f"eff_rank={self.effective_rank:.1f}, "
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
            "Triply Robust Panel (TROP) Estimation Results".center(75),
            "Athey, Imbens, Qu & Viviano (2025)".center(75),
            "=" * 75,
            "",
            f"{'Observations:':<25} {self.n_obs:>10}",
            f"{'Treated units:':<25} {self.n_treated:>10}",
            f"{'Control units:':<25} {self.n_control:>10}",
            f"{'Treated observations:':<25} {self.n_treated_obs:>10}",
            f"{'Pre-treatment periods:':<25} {self.n_pre_periods:>10}",
            f"{'Post-treatment periods:':<25} {self.n_post_periods:>10}",
            "",
            "-" * 75,
            "Tuning Parameters (selected via LOOCV)".center(75),
            "-" * 75,
            f"{'Lambda (time decay):':<25} {self.lambda_time:>10.4f}",
            f"{'Lambda (unit distance):':<25} {self.lambda_unit:>10.4f}",
            f"{'Lambda (nuclear norm):':<25} {self.lambda_nn:>10.4f}",
            f"{'Effective rank:':<25} {self.effective_rank:>10.2f}",
            f"{'LOOCV score:':<25} {self.loocv_score:>10.6f}",
        ]

        # Variance info
        if self.n_bootstrap is not None:
            lines.append(f"{'Bootstrap replications:':<25} {self.n_bootstrap:>10}")

        # Add survey design info
        if self.survey_metadata is not None:
            sm = self.survey_metadata
            lines.extend(_format_survey_block(sm, 75))

        lines.extend(
            [
                "",
                "-" * 75,
                f"{'Parameter':<15} {'Estimate':>12} {'Std. Err.':>12} "
                f"{'t-stat':>10} {'P>|t|':>10} {'':>5}",
                "-" * 75,
                f"{'ATT':<15} {self.att:>12.4f} {self.se:>12.4f} "
                f"{self.t_stat:>10.3f} {self.p_value:>10.4f} {self.significance_stars:>5}",
                "-" * 75,
                "",
                f"{conf_level}% Confidence Interval: [{self.conf_int[0]:.4f}, {self.conf_int[1]:.4f}]",
            ]
        )

        # Add significance codes
        lines.extend(
            [
                "",
                "Signif. codes: '***' 0.001, '**' 0.01, '*' 0.05, '.' 0.1",
                "=" * 75,
            ]
        )

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
            "n_treated_obs": self.n_treated_obs,
            "n_pre_periods": self.n_pre_periods,
            "n_post_periods": self.n_post_periods,
            "lambda_time": self.lambda_time,
            "lambda_unit": self.lambda_unit,
            "lambda_nn": self.lambda_nn,
            "effective_rank": self.effective_rank,
            "loocv_score": self.loocv_score,
        }
        if self.survey_metadata is not None:
            sm = self.survey_metadata
            result["weight_type"] = sm.weight_type
            result["effective_n"] = sm.effective_n
            result["design_effect"] = sm.design_effect
            result["sum_weights"] = sm.sum_weights
            result["n_strata"] = sm.n_strata
            result["n_psu"] = sm.n_psu
            result["df_survey"] = sm.df_survey
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

    def get_treatment_effects_df(self) -> pd.DataFrame:
        """
        Get individual treatment effects as a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with unit, time, and treatment effect columns.
        """
        return pd.DataFrame(
            [
                {"unit": unit, "time": time, "effect": effect}
                for (unit, time), effect in self.treatment_effects.items()
            ]
        )

    def get_unit_effects_df(self) -> pd.DataFrame:
        """
        Get unit fixed effects as a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with unit and effect columns.
        """
        return pd.DataFrame(
            [{"unit": unit, "effect": effect} for unit, effect in self.unit_effects.items()]
        )

    def get_time_effects_df(self) -> pd.DataFrame:
        """
        Get time fixed effects as a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with time and effect columns.
        """
        return pd.DataFrame(
            [{"time": time, "effect": effect} for time, effect in self.time_effects.items()]
        )

    @property
    def is_significant(self) -> bool:
        """Check if the ATT is statistically significant at the alpha level."""
        return bool(self.p_value < self.alpha)

    @property
    def significance_stars(self) -> str:
        """Return significance stars based on p-value."""
        return _get_significance_stars(self.p_value)
