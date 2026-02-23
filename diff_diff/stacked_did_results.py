"""
Result containers for the Stacked DiD estimator.

This module contains StackedDiDResults dataclass for Wing, Freedman &
Hollingsworth (2024) stacked difference-in-differences estimation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.results import _get_significance_stars

__all__ = [
    "StackedDiDResults",
]


@dataclass
class StackedDiDResults:
    """
    Results from Stacked DiD estimation (Wing, Freedman & Hollingsworth 2024).

    Attributes
    ----------
    overall_att : float
        Overall average treatment effect on the treated (average of
        post-treatment event-study coefficients).
    overall_se : float
        Standard error of overall ATT (delta method on VCV).
    overall_t_stat : float
        T-statistic for overall ATT.
    overall_p_value : float
        P-value for overall ATT.
    overall_conf_int : tuple
        Confidence interval for overall ATT.
    event_study_effects : dict, optional
        Dictionary mapping event time h to effect dict with keys:
        'effect', 'se', 't_stat', 'p_value', 'conf_int', 'n_obs'.
    group_effects : dict, optional
        Dictionary mapping cohort g to effect dict.
    stacked_data : pd.DataFrame
        Full stacked dataset with _sub_exp, _event_time, _D_sa,
        _Q_weight columns. Accessible for custom analysis.
    groups : list
        Adoption events in the trimmed set (Omega_kappa).
    trimmed_groups : list
        Adoption events excluded by IC1/IC2.
    time_periods : list
        All time periods in the original data.
    n_obs : int
        Number of observations in the original data.
    n_stacked_obs : int
        Number of observations in the stacked dataset.
    n_sub_experiments : int
        Number of sub-experiments in the stack.
    n_treated_units : int
        Distinct treated units across trimmed set.
    n_control_units : int
        Distinct control units across trimmed set.
    kappa_pre : int
        Pre-treatment event-time window size.
    kappa_post : int
        Post-treatment event-time window size.
    weighting : str
        Weighting scheme used.
    clean_control : str
        Clean control definition used.
    alpha : float
        Significance level used.
    """

    overall_att: float
    overall_se: float
    overall_t_stat: float
    overall_p_value: float
    overall_conf_int: Tuple[float, float]
    event_study_effects: Optional[Dict[int, Dict[str, Any]]]
    group_effects: Optional[Dict[Any, Dict[str, Any]]]
    stacked_data: pd.DataFrame = field(repr=False)
    groups: List[Any] = field(default_factory=list)
    trimmed_groups: List[Any] = field(default_factory=list)
    time_periods: List[Any] = field(default_factory=list)
    n_obs: int = 0
    n_stacked_obs: int = 0
    n_sub_experiments: int = 0
    n_treated_units: int = 0
    n_control_units: int = 0
    kappa_pre: int = 1
    kappa_post: int = 1
    weighting: str = "aggregate"
    clean_control: str = "not_yet_treated"
    alpha: float = 0.05

    def __repr__(self) -> str:
        """Concise string representation."""
        sig = _get_significance_stars(self.overall_p_value)
        return (
            f"StackedDiDResults(ATT={self.overall_att:.4f}{sig}, "
            f"SE={self.overall_se:.4f}, "
            f"n_sub_exp={self.n_sub_experiments}, "
            f"n_stacked_obs={self.n_stacked_obs})"
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
            "Stacked DiD Estimator Results (Wing, Freedman & Hollingsworth 2024)".center(85),
            "=" * 85,
            "",
            f"{'Original observations:':<30} {self.n_obs:>10}",
            f"{'Stacked observations:':<30} {self.n_stacked_obs:>10}",
            f"{'Sub-experiments:':<30} {self.n_sub_experiments:>10}",
            f"{'Treated units:':<30} {self.n_treated_units:>10}",
            f"{'Control units:':<30} {self.n_control_units:>10}",
            f"{'Treatment cohorts:':<30} {len(self.groups):>10}",
            f"{'Trimmed cohorts:':<30} {len(self.trimmed_groups):>10}",
            f"{'Event window:':<30} {'[' + str(-self.kappa_pre) + ', ' + str(self.kappa_post) + ']':>10}",
            f"{'Weighting:':<30} {self.weighting:>10}",
            f"{'Clean control:':<30} {self.clean_control:>10}",
            "",
        ]

        # Overall ATT
        lines.extend(
            [
                "-" * 85,
                "Overall Average Treatment Effect on the Treated".center(85),
                "-" * 85,
                f"{'Parameter':<15} {'Estimate':>12} {'Std. Err.':>12} "
                f"{'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}",
                "-" * 85,
            ]
        )

        t_str = (
            f"{self.overall_t_stat:>10.3f}" if np.isfinite(self.overall_t_stat) else f"{'NaN':>10}"
        )
        p_str = (
            f"{self.overall_p_value:>10.4f}"
            if np.isfinite(self.overall_p_value)
            else f"{'NaN':>10}"
        )
        sig = _get_significance_stars(self.overall_p_value)

        lines.extend(
            [
                f"{'ATT':<15} {self.overall_att:>12.4f} {self.overall_se:>12.4f} "
                f"{t_str} {p_str} {sig:>6}",
                "-" * 85,
                "",
                f"{conf_level}% Confidence Interval: "
                f"[{self.overall_conf_int[0]:.4f}, {self.overall_conf_int[1]:.4f}]",
                "",
            ]
        )

        # Event study effects
        if self.event_study_effects:
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

            for h in sorted(self.event_study_effects.keys()):
                eff = self.event_study_effects[h]
                if eff.get("n_obs", 1) == 0:
                    # Reference period marker
                    lines.append(
                        f"[ref: {h}]" f"{'0.0000':>17} {'---':>12} {'---':>10} {'---':>10} {'':>6}"
                    )
                elif np.isnan(eff["effect"]):
                    lines.append(f"{h:<15} {'NaN':>12} {'NaN':>12} {'NaN':>10} {'NaN':>10} {'':>6}")
                else:
                    e_sig = _get_significance_stars(eff["p_value"])
                    e_t = (
                        f"{eff['t_stat']:>10.3f}" if np.isfinite(eff["t_stat"]) else f"{'NaN':>10}"
                    )
                    e_p = (
                        f"{eff['p_value']:>10.4f}"
                        if np.isfinite(eff["p_value"])
                        else f"{'NaN':>10}"
                    )
                    lines.append(
                        f"{h:<15} {eff['effect']:>12.4f} {eff['se']:>12.4f} "
                        f"{e_t} {e_p} {e_sig:>6}"
                    )

            lines.extend(["-" * 85, ""])

        # Group effects
        if self.group_effects:
            lines.extend(
                [
                    "-" * 85,
                    "Group (Cohort) Effects".center(85),
                    "-" * 85,
                    f"{'Cohort':<15} {'Estimate':>12} {'Std. Err.':>12} "
                    f"{'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}",
                    "-" * 85,
                ]
            )

            for g in sorted(self.group_effects.keys()):
                eff = self.group_effects[g]
                if np.isnan(eff["effect"]):
                    lines.append(f"{g:<15} {'NaN':>12} {'NaN':>12} {'NaN':>10} {'NaN':>10} {'':>6}")
                else:
                    g_sig = _get_significance_stars(eff["p_value"])
                    g_t = (
                        f"{eff['t_stat']:>10.3f}" if np.isfinite(eff["t_stat"]) else f"{'NaN':>10}"
                    )
                    g_p = (
                        f"{eff['p_value']:>10.4f}"
                        if np.isfinite(eff["p_value"])
                        else f"{'NaN':>10}"
                    )
                    lines.append(
                        f"{g:<15} {eff['effect']:>12.4f} {eff['se']:>12.4f} "
                        f"{g_t} {g_p} {g_sig:>6}"
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

    def to_dataframe(self, level: str = "event_study") -> pd.DataFrame:
        """
        Convert results to DataFrame.

        Parameters
        ----------
        level : str, default="event_study"
            Level of aggregation:
            - "event_study": Event study effects by relative time
            - "group": Group (cohort) effects

        Returns
        -------
        pd.DataFrame
            Results as DataFrame.
        """
        if level == "event_study":
            if self.event_study_effects is None:
                raise ValueError(
                    "Event study effects not computed. " "Use aggregate='event_study'."
                )
            rows = []
            for h, data in sorted(self.event_study_effects.items()):
                rows.append(
                    {
                        "relative_period": h,
                        "effect": data["effect"],
                        "se": data["se"],
                        "t_stat": data["t_stat"],
                        "p_value": data["p_value"],
                        "conf_int_lower": data["conf_int"][0],
                        "conf_int_upper": data["conf_int"][1],
                        "n_obs": data.get("n_obs", np.nan),
                    }
                )
            return pd.DataFrame(rows)

        elif level == "group":
            raise ValueError(
                "Group aggregation is not supported by StackedDiD. "
                "The pooled stacked regression cannot produce cohort-specific "
                "effects. Use CallawaySantAnna or ImputationDiD for "
                "cohort-level estimates."
            )

        else:
            raise ValueError(f"Unknown level: {level}. Use 'event_study' or 'group'.")

    @property
    def is_significant(self) -> bool:
        """Check if overall ATT is significant."""
        return bool(self.overall_p_value < self.alpha)

    @property
    def significance_stars(self) -> str:
        """Significance stars for overall ATT."""
        return _get_significance_stars(self.overall_p_value)
