"""
Result container classes for Continuous Difference-in-Differences estimator.

Provides dataclass containers for dose-response curves, group-time effects,
and aggregated estimation results.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.results import _get_significance_stars

__all__ = ["ContinuousDiDResults", "DoseResponseCurve"]


@dataclass
class DoseResponseCurve:
    """
    Dose-response curve from continuous DiD estimation.

    Attributes
    ----------
    dose_grid : np.ndarray
        Evaluation points, shape ``(n_grid,)``.
    effects : np.ndarray
        ATT(d) or ACRT(d) values, shape ``(n_grid,)``.
    se : np.ndarray
        Standard errors, shape ``(n_grid,)``.
    conf_int_lower : np.ndarray
        Lower CI bounds, shape ``(n_grid,)``.
    conf_int_upper : np.ndarray
        Upper CI bounds, shape ``(n_grid,)``.
    target : str
        ``"att"`` or ``"acrt"``.
    """

    dose_grid: np.ndarray
    effects: np.ndarray
    se: np.ndarray
    conf_int_lower: np.ndarray
    conf_int_upper: np.ndarray
    target: str
    p_value: Optional[np.ndarray] = None
    n_bootstrap: int = 0

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with dose, effect, se, CI, t_stat, p_value."""
        n = len(self.effects)
        if self.n_bootstrap > 0 and self.p_value is not None:
            # Bootstrap inference: use stored p-values, t-stat is undefined
            t_stat = np.full(n, np.nan)
            p_value = self.p_value
        else:
            # Analytic inference: compute t-stat and p-value from normal approx
            from diff_diff.utils import safe_inference

            t_stat = np.full(n, np.nan)
            p_value = np.full(n, np.nan)
            for i in range(n):
                t_i, p_i, _ = safe_inference(self.effects[i], self.se[i])
                t_stat[i] = t_i
                p_value[i] = p_i
        return pd.DataFrame(
            {
                "dose": self.dose_grid,
                "effect": self.effects,
                "se": self.se,
                "conf_int_lower": self.conf_int_lower,
                "conf_int_upper": self.conf_int_upper,
                "t_stat": t_stat,
                "p_value": p_value,
            }
        )


@dataclass
class ContinuousDiDResults:
    """
    Results from Continuous Difference-in-Differences estimation.

    Implements Callaway, Goodman-Bacon & Sant'Anna (2024).

    Attributes
    ----------
    dose_response_att : DoseResponseCurve
        ATT(d) dose-response curve.
    dose_response_acrt : DoseResponseCurve
        ACRT(d) dose-response curve.
    overall_att : float
        Binarized overall ATT (ATT^{loc} under PT, equals ATT^{glob} under SPT).
    overall_acrt : float
        Plug-in overall ACRT^{glob}.
    group_time_effects : dict
        Per (g,t) cell results.
    base_period : str
        Base period strategy (``"varying"`` or ``"universal"``).
    anticipation : int
        Number of anticipation periods.
    n_bootstrap : int
        Number of bootstrap iterations used.
    bootstrap_weights : str
        Bootstrap weight type (``"rademacher"``, ``"mammen"``, or ``"webb"``).
    seed : int or None
        Random seed used for bootstrap.
    rank_deficient_action : str
        How rank deficiency is handled (``"warn"``, ``"error"``, ``"silent"``).
    """

    dose_response_att: DoseResponseCurve
    dose_response_acrt: DoseResponseCurve
    overall_att: float
    overall_att_se: float
    overall_att_t_stat: float
    overall_att_p_value: float
    overall_att_conf_int: Tuple[float, float]
    overall_acrt: float
    overall_acrt_se: float
    overall_acrt_t_stat: float
    overall_acrt_p_value: float
    overall_acrt_conf_int: Tuple[float, float]
    group_time_effects: Dict[Tuple[Any, Any], Dict[str, Any]]
    dose_grid: np.ndarray
    groups: List[Any]
    time_periods: List[Any]
    n_obs: int
    n_treated_units: int
    n_control_units: int
    alpha: float = 0.05
    control_group: str = "never_treated"
    degree: int = 3
    num_knots: int = 0
    base_period: str = "varying"
    anticipation: int = 0
    n_bootstrap: int = 0
    bootstrap_weights: str = "rademacher"
    seed: Optional[int] = None
    rank_deficient_action: str = "warn"
    event_study_effects: Optional[Dict[int, Dict[str, Any]]] = field(default=None)

    def __repr__(self) -> str:
        sig_att = _get_significance_stars(self.overall_att_p_value)
        sig_acrt = _get_significance_stars(self.overall_acrt_p_value)
        return (
            f"ContinuousDiDResults("
            f"ATT_glob={self.overall_att:.4f}{sig_att}, "
            f"ACRT_glob={self.overall_acrt:.4f}{sig_acrt}, "
            f"n_groups={len(self.groups)}, "
            f"n_periods={len(self.time_periods)})"
        )

    def summary(self, alpha: Optional[float] = None) -> str:
        """Generate formatted summary."""
        alpha = alpha or self.alpha
        conf_level = int((1 - alpha) * 100)
        w = 85

        lines = [
            "=" * w,
            "Continuous Difference-in-Differences Results".center(w),
            "(Callaway, Goodman-Bacon & Sant'Anna 2024)".center(w),
            "=" * w,
            "",
            f"{'Total observations:':<30} {self.n_obs:>10}",
            f"{'Treated units:':<30} {self.n_treated_units:>10}",
            f"{'Control units:':<30} {self.n_control_units:>10}",
            f"{'Treatment cohorts:':<30} {len(self.groups):>10}",
            f"{'Time periods:':<30} {len(self.time_periods):>10}",
            f"{'Control group:':<30} {self.control_group:>10}",
            f"{'B-spline degree:':<30} {self.degree:>10}",
            f"{'Interior knots:':<30} {self.num_knots:>10}",
            f"{'Base period:':<30} {self.base_period:>10}",
            f"{'Anticipation:':<30} {self.anticipation:>10}",
            "",
        ]

        # Overall summary parameters
        lines.extend([
            "-" * w,
            "Overall Summary Parameters".center(w),
            "-" * w,
            f"{'Parameter':<15} {'Estimate':>12} {'Std. Err.':>12} "
            f"{'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}",
            "-" * w,
        ])
        for label, est, se, t, p in [
            (
                "ATT_glob",
                self.overall_att,
                self.overall_att_se,
                self.overall_att_t_stat,
                self.overall_att_p_value,
            ),
            (
                "ACRT_glob",
                self.overall_acrt,
                self.overall_acrt_se,
                self.overall_acrt_t_stat,
                self.overall_acrt_p_value,
            ),
        ]:
            t_str = f"{t:>10.3f}" if np.isfinite(t) else f"{'NaN':>10}"
            p_str = f"{p:>10.4f}" if np.isfinite(p) else f"{'NaN':>10}"
            sig = _get_significance_stars(p)
            lines.append(
                f"{label:<15} {est:>12.4f} {se:>12.4f} {t_str} {p_str} {sig:>6}"
            )
        lines.extend([
            "-" * w,
            "",
            f"{conf_level}% CI for ATT_glob: "
            f"[{self.overall_att_conf_int[0]:.4f}, {self.overall_att_conf_int[1]:.4f}]",
            f"{conf_level}% CI for ACRT_glob: "
            f"[{self.overall_acrt_conf_int[0]:.4f}, {self.overall_acrt_conf_int[1]:.4f}]",
            "",
        ])

        # Dose-response curve summary (first/mid/last points)
        if len(self.dose_grid) > 0:
            lines.extend([
                "-" * w,
                "Dose-Response Curve (selected points)".center(w),
                "-" * w,
                f"{'Dose':>10} {'ATT(d)':>12} {'SE':>10} "
                f"{'ACRT(d)':>12} {'SE':>10}",
                "-" * w,
            ])
            n_grid = len(self.dose_grid)
            indices = sorted(set([0, n_grid // 4, n_grid // 2, 3 * n_grid // 4, n_grid - 1]))
            for idx in indices:
                if idx < n_grid:
                    lines.append(
                        f"{self.dose_grid[idx]:>10.3f} "
                        f"{self.dose_response_att.effects[idx]:>12.4f} "
                        f"{self.dose_response_att.se[idx]:>10.4f} "
                        f"{self.dose_response_acrt.effects[idx]:>12.4f} "
                        f"{self.dose_response_acrt.se[idx]:>10.4f}"
                    )
            lines.extend(["-" * w, ""])

        # Event study effects if available
        if self.event_study_effects:
            lines.extend([
                "-" * w,
                "Event Study (Dynamic) Effects (Binarized ATT)".center(w),
                "-" * w,
                f"{'Rel. Period':<15} {'Estimate':>12} {'Std. Err.':>12} "
                f"{'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}",
                "-" * w,
            ])
            for rel_t in sorted(self.event_study_effects.keys()):
                eff = self.event_study_effects[rel_t]
                sig = _get_significance_stars(eff["p_value"])
                t_str = (
                    f"{eff['t_stat']:>10.3f}"
                    if np.isfinite(eff["t_stat"])
                    else f"{'NaN':>10}"
                )
                p_str = (
                    f"{eff['p_value']:>10.4f}"
                    if np.isfinite(eff["p_value"])
                    else f"{'NaN':>10}"
                )
                lines.append(
                    f"{rel_t:<15} {eff['effect']:>12.4f} {eff['se']:>12.4f} "
                    f"{t_str} {p_str} {sig:>6}"
                )
            lines.extend(["-" * w, ""])

        lines.extend([
            "Signif. codes: '***' 0.001, '**' 0.01, '*' 0.05, '.' 0.1",
            "=" * w,
        ])
        return "\n".join(lines)

    def print_summary(self, alpha: Optional[float] = None) -> None:
        """Print summary to stdout."""
        print(self.summary(alpha))

    def to_dataframe(self, level: str = "dose_response") -> pd.DataFrame:
        """
        Convert results to DataFrame.

        Parameters
        ----------
        level : str, default="dose_response"
            ``"dose_response"``, ``"group_time"``, or ``"event_study"``.
        """
        if level == "dose_response":
            att_df = self.dose_response_att.to_dataframe()
            acrt_df = self.dose_response_acrt.to_dataframe()
            return pd.DataFrame(
                {
                    "dose": att_df["dose"],
                    "att": att_df["effect"],
                    "att_se": att_df["se"],
                    "att_ci_lower": att_df["conf_int_lower"],
                    "att_ci_upper": att_df["conf_int_upper"],
                    "acrt": acrt_df["effect"],
                    "acrt_se": acrt_df["se"],
                    "acrt_ci_lower": acrt_df["conf_int_lower"],
                    "acrt_ci_upper": acrt_df["conf_int_upper"],
                }
            )
        elif level == "group_time":
            rows = []
            for (g, t), data in sorted(self.group_time_effects.items()):
                rows.append(
                    {
                        "group": g,
                        "time": t,
                        "att_glob": data.get("att_glob", np.nan),
                        "acrt_glob": data.get("acrt_glob", np.nan),
                        "n_treated": data.get("n_treated", 0),
                        "n_control": data.get("n_control", 0),
                    }
                )
            return pd.DataFrame(rows)
        elif level == "event_study":
            if self.event_study_effects is None:
                raise ValueError(
                    "Event study effects not computed. Use aggregate='eventstudy'."
                )
            rows = []
            for rel_t, data in sorted(self.event_study_effects.items()):
                rows.append(
                    {
                        "relative_period": rel_t,
                        "att_glob": data["effect"],
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
                f"Unknown level: {level}. Use 'dose_response', 'group_time', or 'event_study'."
            )

    @property
    def is_significant(self) -> bool:
        """Check if overall ATT is significant."""
        return bool(self.overall_att_p_value < self.alpha)

    @property
    def significance_stars(self) -> str:
        """Significance stars for overall ATT."""
        return _get_significance_stars(self.overall_att_p_value)
