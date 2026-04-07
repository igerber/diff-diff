"""
Result container classes for Staggered Triple Difference estimator.

This module provides dataclass containers for storing and presenting
group-time DDD effects and their aggregations.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.results import _format_survey_block, _get_significance_stars

if TYPE_CHECKING:
    from diff_diff.staggered_bootstrap import CSBootstrapResults


@dataclass
class StaggeredTripleDiffResults:
    """
    Results from Staggered Triple Difference (DDD) estimation.

    Implements the Ortiz-Villavicencio & Sant'Anna (2025) estimator for
    staggered adoption settings with an eligibility dimension.

    Attributes
    ----------
    group_time_effects : dict
        Dictionary mapping (group, time) tuples to effect dictionaries.
    overall_att : float
        Overall average treatment effect (weighted average of ATT(g,t)).
    overall_se : float
        Standard error of overall ATT.
    overall_t_stat : float
        T-statistic for overall ATT.
    overall_p_value : float
        P-value for overall ATT.
    overall_conf_int : tuple
        Confidence interval for overall ATT.
    groups : list
        List of enabling cohorts (first treatment periods).
    time_periods : list
        List of all time periods.
    n_obs : int
        Total number of observations.
    n_treated_units : int
        Number of treated units (S < inf AND Q = 1).
    n_control_units : int
        Number of units not in treated group.
    n_never_enabled : int
        Number of never-enabled units (S = inf or 0).
    n_eligible : int
        Number of eligible units (Q = 1).
    n_ineligible : int
        Number of ineligible units (Q = 0).
    """

    group_time_effects: Dict[Tuple[Any, Any], Dict[str, Any]]
    overall_att: float
    overall_se: float
    overall_t_stat: float
    overall_p_value: float
    overall_conf_int: Tuple[float, float]
    groups: List[Any]
    time_periods: List[Any]
    n_obs: int
    n_treated_units: int
    n_control_units: int
    n_never_enabled: int
    n_eligible: int
    n_ineligible: int
    alpha: float = 0.05
    control_group: str = "notyettreated"
    base_period: str = "varying"
    estimation_method: str = "dr"
    event_study_effects: Optional[Dict[int, Dict[str, Any]]] = field(default=None)
    group_effects: Optional[Dict[Any, Dict[str, Any]]] = field(default=None)
    influence_functions: Optional["np.ndarray"] = field(default=None, repr=False)
    bootstrap_results: Optional["CSBootstrapResults"] = field(default=None, repr=False)
    cband_crit_value: Optional[float] = None
    pscore_trim: float = 0.01
    survey_metadata: Optional[Any] = field(default=None, repr=False)
    comparison_group_counts: Optional[Dict[Tuple, int]] = field(default=None, repr=False)
    gmm_weights: Optional[Dict[Tuple, Dict]] = field(default=None, repr=False)

    def __repr__(self) -> str:
        """Concise string representation."""
        sig = _get_significance_stars(self.overall_p_value)
        return (
            f"StaggeredTripleDiffResults(ATT={self.overall_att:.4f}{sig}, "
            f"SE={self.overall_se:.4f}, "
            f"n_groups={len(self.groups)}, "
            f"n_periods={len(self.time_periods)})"
        )

    def summary(self, alpha: Optional[float] = None) -> str:
        """
        Generate formatted summary of estimation results.

        Parameters
        ----------
        alpha : float, optional
            Significance level. Defaults to alpha used in estimation.

        Returns
        -------
        str
            Formatted summary.
        """
        alpha = alpha or self.alpha
        conf_level = int((1 - alpha) * 100)

        lines = [
            "=" * 85,
            "Staggered Triple Difference (DDD) Results".center(85),
            "=" * 85,
            "",
            f"{'Total observations:':<30} {self.n_obs:>10}",
            f"{'Treated units (S<inf, Q=1):':<30} {self.n_treated_units:>10}",
            f"{'Control units:':<30} {self.n_control_units:>10}",
            f"{'Never-enabled units:':<30} {self.n_never_enabled:>10}",
            f"{'Eligible units (Q=1):':<30} {self.n_eligible:>10}",
            f"{'Ineligible units (Q=0):':<30} {self.n_ineligible:>10}",
            f"{'Enabling cohorts:':<30} {len(self.groups):>10}",
            f"{'Time periods:':<30} {len(self.time_periods):>10}",
            f"{'Estimation method:':<30} {self.estimation_method:>10}",
            f"{'Control group:':<30} {self.control_group:>10}",
            f"{'Base period:':<30} {self.base_period:>10}",
            "",
        ]

        if self.survey_metadata is not None:
            sm = self.survey_metadata
            lines.extend(_format_survey_block(sm, 85))

        # Overall ATT
        lines.extend(
            [
                "-" * 85,
                "Overall Average Treatment Effect on the Treated".center(85),
                "-" * 85,
                f"{'Parameter':<15} {'Estimate':>12} {'Std. Err.':>12} "
                f"{'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}",
                "-" * 85,
                f"{'ATT':<15} {self.overall_att:>12.4f} {self.overall_se:>12.4f} "
                f"{self.overall_t_stat:>10.3f} {self.overall_p_value:>10.4f} "
                f"{_get_significance_stars(self.overall_p_value):>6}",
                "-" * 85,
                "",
                f"{conf_level}% Confidence Interval: "
                f"[{self.overall_conf_int[0]:.4f}, {self.overall_conf_int[1]:.4f}]",
                "",
            ]
        )

        # Event study effects
        if self.event_study_effects:
            ci_label = (
                "Simult. CI"
                if self.cband_crit_value is not None
                else "Pointwise CI"
            )
            lines.extend(
                [
                    "-" * 85,
                    "Event Study (Dynamic) Effects".center(85),
                    "-" * 85,
                    f"{'Rel. Period':<15} {'Estimate':>12} {'Std. Err.':>12} "
                    f"{'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}",
                    "-" * 85,
                ]
            )

            for rel_t in sorted(self.event_study_effects.keys()):
                eff = self.event_study_effects[rel_t]
                sig = _get_significance_stars(eff["p_value"])
                lines.append(
                    f"{rel_t:<15} {eff['effect']:>12.4f} {eff['se']:>12.4f} "
                    f"{eff['t_stat']:>10.3f} {eff['p_value']:>10.4f} {sig:>6}"
                )

            lines.extend(["-" * 85])
            if self.cband_crit_value is not None:
                lines.append(
                    f"{ci_label}: critical value = {self.cband_crit_value:.4f} "
                    f"(sup-t bootstrap, {conf_level}% family-wise)"
                )
            lines.append("")

        # Group effects
        if self.group_effects:
            lines.extend(
                [
                    "-" * 85,
                    "Effects by Enabling Cohort".center(85),
                    "-" * 85,
                    f"{'Cohort':<15} {'Estimate':>12} {'Std. Err.':>12} "
                    f"{'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}",
                    "-" * 85,
                ]
            )

            for group in sorted(self.group_effects.keys()):
                eff = self.group_effects[group]
                sig = _get_significance_stars(eff["p_value"])
                lines.append(
                    f"{group:<15} {eff['effect']:>12.4f} {eff['se']:>12.4f} "
                    f"{eff['t_stat']:>10.3f} {eff['p_value']:>10.4f} {sig:>6}"
                )

            lines.extend(["-" * 85, ""])

        lines.extend(
            [
                "Signif. codes: '***' 0.001, '**' 0.01, '*' 0.05, '.' 0.1",
                "=" * 85,
            ]
        )

        return "\n".join(lines)

    def print_summary(self, alpha: Optional[float] = None) -> None:
        """Print summary to stdout."""
        print(self.summary(alpha))

    def to_dataframe(self, level: str = "group_time") -> pd.DataFrame:
        """
        Convert results to DataFrame.

        Parameters
        ----------
        level : str, default="group_time"
            Level of aggregation: "group_time", "event_study", or "group".

        Returns
        -------
        pd.DataFrame
            Results as DataFrame.
        """
        if level == "group_time":
            rows = []
            for (g, t), data in self.group_time_effects.items():
                rows.append(
                    {
                        "group": g,
                        "time": t,
                        "effect": data["effect"],
                        "se": data["se"],
                        "t_stat": data["t_stat"],
                        "p_value": data["p_value"],
                        "conf_int_lower": data["conf_int"][0],
                        "conf_int_upper": data["conf_int"][1],
                    }
                )
            return pd.DataFrame(rows)

        elif level == "event_study":
            if self.event_study_effects is None:
                raise ValueError(
                    "Event study effects not computed. Use aggregate='event_study'."
                )
            rows = []
            for rel_t, data in sorted(self.event_study_effects.items()):
                cband_ci = data.get("cband_conf_int", (np.nan, np.nan))
                rows.append(
                    {
                        "relative_period": rel_t,
                        "effect": data["effect"],
                        "se": data["se"],
                        "t_stat": data["t_stat"],
                        "p_value": data["p_value"],
                        "conf_int_lower": data["conf_int"][0],
                        "conf_int_upper": data["conf_int"][1],
                        "cband_lower": cband_ci[0],
                        "cband_upper": cband_ci[1],
                    }
                )
            return pd.DataFrame(rows)

        elif level == "group":
            if self.group_effects is None:
                raise ValueError(
                    "Group effects not computed. Use aggregate='group'."
                )
            rows = []
            for group, data in sorted(self.group_effects.items()):
                rows.append(
                    {
                        "group": group,
                        "effect": data["effect"],
                        "se": data["se"],
                        "t_stat": data["t_stat"],
                        "p_value": data["p_value"],
                        "conf_int_lower": data["conf_int"][0],
                        "conf_int_upper": data["conf_int"][1],
                    }
                )
            return pd.DataFrame(rows)

        else:
            raise ValueError(
                f"Unknown level: {level}. "
                "Use 'group_time', 'event_study', or 'group'."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        d = {
            "overall_att": self.overall_att,
            "overall_se": self.overall_se,
            "overall_t_stat": self.overall_t_stat,
            "overall_p_value": self.overall_p_value,
            "overall_conf_int": self.overall_conf_int,
            "n_obs": self.n_obs,
            "n_treated_units": self.n_treated_units,
            "n_control_units": self.n_control_units,
            "n_never_enabled": self.n_never_enabled,
            "n_eligible": self.n_eligible,
            "n_ineligible": self.n_ineligible,
            "n_groups": len(self.groups),
            "n_periods": len(self.time_periods),
            "groups": self.groups,
            "time_periods": self.time_periods,
            "estimation_method": self.estimation_method,
            "control_group": self.control_group,
            "base_period": self.base_period,
            "alpha": self.alpha,
            "pscore_trim": self.pscore_trim,
        }
        if self.event_study_effects is not None:
            d["event_study_effects"] = self.event_study_effects
        if self.group_effects is not None:
            d["group_effects"] = self.group_effects
        if self.comparison_group_counts is not None:
            d["comparison_group_counts"] = self.comparison_group_counts
        return d

    @property
    def is_significant(self) -> bool:
        """Check if overall ATT is significant."""
        return bool(self.overall_p_value < self.alpha)

    @property
    def significance_stars(self) -> str:
        """Significance stars for overall ATT."""
        return _get_significance_stars(self.overall_p_value)
