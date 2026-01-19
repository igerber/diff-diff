"""
Result container classes for Callaway-Sant'Anna estimator.

This module provides dataclass containers for storing and presenting
group-time average treatment effects and their aggregations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from diff_diff.results import _get_significance_stars

if TYPE_CHECKING:
    from diff_diff.staggered_bootstrap import CSBootstrapResults


@dataclass
class GroupTimeEffect:
    """
    Treatment effect for a specific group-time combination.

    Attributes
    ----------
    group : any
        The treatment cohort (first treatment period).
    time : any
        The time period.
    effect : float
        The ATT(g,t) estimate.
    se : float
        Standard error.
    n_treated : int
        Number of treated observations.
    n_control : int
        Number of control observations.
    """
    group: Any
    time: Any
    effect: float
    se: float
    t_stat: float
    p_value: float
    conf_int: Tuple[float, float]
    n_treated: int
    n_control: int

    @property
    def is_significant(self) -> bool:
        """Check if effect is significant at 0.05 level."""
        return bool(self.p_value < 0.05)

    @property
    def significance_stars(self) -> str:
        """Return significance stars based on p-value."""
        return _get_significance_stars(self.p_value)


@dataclass
class CallawaySantAnnaResults:
    """
    Results from Callaway-Sant'Anna (2021) staggered DiD estimation.

    This class stores group-time average treatment effects ATT(g,t) and
    provides methods for aggregation into summary measures.

    Attributes
    ----------
    group_time_effects : dict
        Dictionary mapping (group, time) tuples to effect dictionaries.
    overall_att : float
        Overall average treatment effect (weighted average of ATT(g,t)).
    overall_se : float
        Standard error of overall ATT.
    overall_p_value : float
        P-value for overall ATT.
    overall_conf_int : tuple
        Confidence interval for overall ATT.
    groups : list
        List of treatment cohorts (first treatment periods).
    time_periods : list
        List of all time periods.
    n_obs : int
        Total number of observations.
    n_treated_units : int
        Number of ever-treated units.
    n_control_units : int
        Number of never-treated units.
    event_study_effects : dict, optional
        Effects aggregated by relative time (event study).
    group_effects : dict, optional
        Effects aggregated by treatment cohort.
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
    alpha: float = 0.05
    control_group: str = "never_treated"
    event_study_effects: Optional[Dict[int, Dict[str, Any]]] = field(default=None)
    group_effects: Optional[Dict[Any, Dict[str, Any]]] = field(default=None)
    influence_functions: Optional["np.ndarray"] = field(default=None, repr=False)
    bootstrap_results: Optional["CSBootstrapResults"] = field(default=None, repr=False)

    def __repr__(self) -> str:
        """Concise string representation."""
        sig = _get_significance_stars(self.overall_p_value)
        return (
            f"CallawaySantAnnaResults(ATT={self.overall_att:.4f}{sig}, "
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
            "Callaway-Sant'Anna Staggered Difference-in-Differences Results".center(85),
            "=" * 85,
            "",
            f"{'Total observations:':<30} {self.n_obs:>10}",
            f"{'Treated units:':<30} {self.n_treated_units:>10}",
            f"{'Control units:':<30} {self.n_control_units:>10}",
            f"{'Treatment cohorts:':<30} {len(self.groups):>10}",
            f"{'Time periods:':<30} {len(self.time_periods):>10}",
            f"{'Control group:':<30} {self.control_group:>10}",
            "",
        ]

        # Overall ATT
        lines.extend([
            "-" * 85,
            "Overall Average Treatment Effect on the Treated".center(85),
            "-" * 85,
            f"{'Parameter':<15} {'Estimate':>12} {'Std. Err.':>12} {'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}",
            "-" * 85,
            f"{'ATT':<15} {self.overall_att:>12.4f} {self.overall_se:>12.4f} "
            f"{self.overall_t_stat:>10.3f} {self.overall_p_value:>10.4f} "
            f"{_get_significance_stars(self.overall_p_value):>6}",
            "-" * 85,
            "",
            f"{conf_level}% Confidence Interval: [{self.overall_conf_int[0]:.4f}, {self.overall_conf_int[1]:.4f}]",
            "",
        ])

        # Event study effects if available
        if self.event_study_effects:
            lines.extend([
                "-" * 85,
                "Event Study (Dynamic) Effects".center(85),
                "-" * 85,
                f"{'Rel. Period':<15} {'Estimate':>12} {'Std. Err.':>12} {'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}",
                "-" * 85,
            ])

            for rel_t in sorted(self.event_study_effects.keys()):
                eff = self.event_study_effects[rel_t]
                sig = _get_significance_stars(eff['p_value'])
                lines.append(
                    f"{rel_t:<15} {eff['effect']:>12.4f} {eff['se']:>12.4f} "
                    f"{eff['t_stat']:>10.3f} {eff['p_value']:>10.4f} {sig:>6}"
                )

            lines.extend(["-" * 85, ""])

        # Group effects if available
        if self.group_effects:
            lines.extend([
                "-" * 85,
                "Effects by Treatment Cohort".center(85),
                "-" * 85,
                f"{'Cohort':<15} {'Estimate':>12} {'Std. Err.':>12} {'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}",
                "-" * 85,
            ])

            for group in sorted(self.group_effects.keys()):
                eff = self.group_effects[group]
                sig = _get_significance_stars(eff['p_value'])
                lines.append(
                    f"{group:<15} {eff['effect']:>12.4f} {eff['se']:>12.4f} "
                    f"{eff['t_stat']:>10.3f} {eff['p_value']:>10.4f} {sig:>6}"
                )

            lines.extend(["-" * 85, ""])

        lines.extend([
            "Signif. codes: '***' 0.001, '**' 0.01, '*' 0.05, '.' 0.1",
            "=" * 85,
        ])

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
                rows.append({
                    'group': g,
                    'time': t,
                    'effect': data['effect'],
                    'se': data['se'],
                    't_stat': data['t_stat'],
                    'p_value': data['p_value'],
                    'conf_int_lower': data['conf_int'][0],
                    'conf_int_upper': data['conf_int'][1],
                })
            return pd.DataFrame(rows)

        elif level == "event_study":
            if self.event_study_effects is None:
                raise ValueError("Event study effects not computed. Use aggregate='event_study'.")
            rows = []
            for rel_t, data in sorted(self.event_study_effects.items()):
                rows.append({
                    'relative_period': rel_t,
                    'effect': data['effect'],
                    'se': data['se'],
                    't_stat': data['t_stat'],
                    'p_value': data['p_value'],
                    'conf_int_lower': data['conf_int'][0],
                    'conf_int_upper': data['conf_int'][1],
                })
            return pd.DataFrame(rows)

        elif level == "group":
            if self.group_effects is None:
                raise ValueError("Group effects not computed. Use aggregate='group'.")
            rows = []
            for group, data in sorted(self.group_effects.items()):
                rows.append({
                    'group': group,
                    'effect': data['effect'],
                    'se': data['se'],
                    't_stat': data['t_stat'],
                    'p_value': data['p_value'],
                    'conf_int_lower': data['conf_int'][0],
                    'conf_int_upper': data['conf_int'][1],
                })
            return pd.DataFrame(rows)

        else:
            raise ValueError(f"Unknown level: {level}. Use 'group_time', 'event_study', or 'group'.")

    @property
    def is_significant(self) -> bool:
        """Check if overall ATT is significant."""
        return bool(self.overall_p_value < self.alpha)

    @property
    def significance_stars(self) -> str:
        """Significance stars for overall ATT."""
        return _get_significance_stars(self.overall_p_value)
