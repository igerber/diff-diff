"""
Result container for the Efficient DiD estimator.

Follows the CallawaySantAnnaResults pattern: dataclass with summary(),
to_dataframe(), and significance properties.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.results import _format_survey_block, _get_significance_stars

if TYPE_CHECKING:
    from diff_diff.efficient_did_bootstrap import EDiDBootstrapResults


@dataclass
class HausmanPretestResult:
    """Result of Hausman pretest for PT-All vs PT-Post (Theorem A.1).

    Under H0 (PT-All holds), both estimators are consistent but PT-All
    is efficient.  Rejection suggests PT-All is too strong; use PT-Post.
    """

    statistic: float
    """Hausman H statistic."""
    p_value: float
    """Chi-squared p-value."""
    df: int
    """Degrees of freedom (effective rank of V)."""
    reject: bool
    """True if p_value < alpha."""
    alpha: float
    """Significance level used."""
    att_all: float
    """Overall ATT under PT-All."""
    att_post: float
    """Overall ATT under PT-Post."""
    recommendation: str
    """``"pt_all"`` if fail to reject, ``"pt_post"`` if reject, ``"inconclusive"`` if test unavailable."""
    gt_details: Optional[pd.DataFrame] = None
    """Per-event-study-horizon details: relative_period, es_all, es_post, delta."""

    def __repr__(self) -> str:
        return (
            f"HausmanPretestResult(H={self.statistic:.3f}, p={self.p_value:.4f}, "
            f"df={self.df}, recommend={self.recommendation})"
        )


@dataclass
class EfficientDiDResults:
    """
    Results from Efficient DiD (Chen, Sant'Anna & Xie 2025) estimation.

    Stores group-time ATT(g,t) estimates with efficient weights, plus
    optional aggregations (overall ATT, event study, group effects).

    Attributes
    ----------
    group_time_effects : dict
        ``{(g, t): {'effect', 'se', 't_stat', 'p_value', 'conf_int',
        'n_treated', 'n_control'}}``
    overall_att : float
        Overall ATT (cohort-size weighted average of post-treatment
        group-time effects, matching CallawaySantAnna convention).
    overall_se : float
        Standard error of overall ATT.
    overall_t_stat : float
        t-statistic for overall ATT.
    overall_p_value : float
        p-value for overall ATT.
    overall_conf_int : tuple
        Confidence interval for overall ATT.
    groups : list
        Treatment cohort identifiers.
    time_periods : list
        All time periods.
    n_obs : int
        Total observations (units x periods).
    n_treated_units : int
        Number of ever-treated units.
    n_control_units : int
        Number of never-treated units.
    alpha : float
        Significance level.
    pt_assumption : str
        ``"all"`` or ``"post"``.
    anticipation : int
        Number of anticipation periods used.
    n_bootstrap : int
        Number of bootstrap iterations (0 = analytical only).
    bootstrap_weights : str
        Bootstrap weight distribution (``"rademacher"``, ``"mammen"``, ``"webb"``).
    seed : int or None
        Random seed used for bootstrap.
    event_study_effects : dict, optional
        ``{relative_time: effect_dict}``
    group_effects : dict, optional
        ``{group: effect_dict}``
    efficient_weights : dict, optional
        ``{(g, t): ndarray}`` — diagnostic: weight vector per target.
    omega_condition_numbers : dict, optional
        ``{(g, t): float}`` — diagnostic: Omega* condition numbers.
    influence_functions : dict, optional
        ``{(g, t): ndarray(n_units,)}`` — per-unit EIF values for each
        group-time cell.  Only populated when ``store_eif=True`` in
        :meth:`~EfficientDiD.fit` (used internally by ``hausman_pretest``).
    bootstrap_results : EDiDBootstrapResults, optional
        Bootstrap inference results.
    estimation_path : str
        ``"nocov"`` or ``"dr"`` — which estimation path was used.
    sieve_k_max : int or None
        Maximum polynomial degree for sieve ratio estimation.
    sieve_criterion : str
        Information criterion used (``"aic"`` or ``"bic"``).
    ratio_clip : float
        Clipping bound for sieve propensity ratios.
    kernel_bandwidth : float or None
        Bandwidth used for kernel-smoothed conditional Omega*.
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
    pt_assumption: str = "all"
    anticipation: int = 0
    n_bootstrap: int = 0
    bootstrap_weights: str = "rademacher"
    seed: Optional[int] = None
    event_study_effects: Optional[Dict[int, Dict[str, Any]]] = field(default=None)
    group_effects: Optional[Dict[Any, Dict[str, Any]]] = field(default=None)
    efficient_weights: Optional[Dict[Tuple[Any, Any], "np.ndarray"]] = field(
        default=None, repr=False
    )
    omega_condition_numbers: Optional[Dict[Tuple[Any, Any], float]] = field(
        default=None, repr=False
    )
    control_group: str = "never_treated"
    influence_functions: Optional[Dict[Tuple[Any, Any], "np.ndarray"]] = field(
        default=None, repr=False
    )
    bootstrap_results: Optional["EDiDBootstrapResults"] = field(default=None, repr=False)
    estimation_path: str = "nocov"
    sieve_k_max: Optional[int] = None
    sieve_criterion: str = "bic"
    ratio_clip: float = 20.0
    kernel_bandwidth: Optional[float] = None
    # Survey design metadata (SurveyMetadata instance from diff_diff.survey)
    survey_metadata: Optional[Any] = field(default=None)

    def __repr__(self) -> str:
        sig = _get_significance_stars(self.overall_p_value)
        path = "DR" if self.estimation_path == "dr" else "nocov"
        return (
            f"EfficientDiDResults(ATT={self.overall_att:.4f}{sig}, "
            f"SE={self.overall_se:.4f}, "
            f"pt={self.pt_assumption}, path={path}, "
            f"n_groups={len(self.groups)}, "
            f"n_periods={len(self.time_periods)})"
        )

    def summary(self, alpha: Optional[float] = None) -> str:
        """Generate formatted summary of estimation results."""
        alpha = alpha or self.alpha
        conf_level = int((1 - alpha) * 100)

        lines = [
            "=" * 85,
            "Efficient DiD (Chen-Sant'Anna-Xie 2025) Results".center(85),
            "=" * 85,
            "",
            f"{'Total observations:':<30} {self.n_obs:>10}",
            f"{'Treated units:':<30} {self.n_treated_units:>10}",
            f"{'Control units:':<30} {self.n_control_units:>10}",
            f"{'Treatment cohorts:':<30} {len(self.groups):>10}",
            f"{'Time periods:':<30} {len(self.time_periods):>10}",
            f"{'PT assumption:':<30} {self.pt_assumption:>10}",
            f"{'Estimation path:':<30} {'doubly robust' if self.estimation_path == 'dr' else 'no covariates':>10}",
        ]
        if self.control_group != "never_treated":
            lines.append(f"{'Control group:':<30} {self.control_group:>10}")
        if self.anticipation > 0:
            lines.append(f"{'Anticipation periods:':<30} {self.anticipation:>10}")
        if self.n_bootstrap > 0:
            lines.append(f"{'Bootstrap:':<30} {self.n_bootstrap:>10} ({self.bootstrap_weights})")
        lines.append("")

        # Add survey design info
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
            lines.extend(["-" * 85, ""])

        # Group effects
        if self.group_effects:
            lines.extend(
                [
                    "-" * 85,
                    "Effects by Treatment Cohort".center(85),
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
        """Convert results to DataFrame.

        Parameters
        ----------
        level : str
            ``"group_time"``, ``"event_study"``, or ``"group"``.
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
                raise ValueError("Event study effects not computed. Use aggregate='event_study'.")
            rows = []
            for rel_t, data in sorted(self.event_study_effects.items()):
                rows.append(
                    {
                        "relative_period": rel_t,
                        "effect": data["effect"],
                        "se": data["se"],
                        "t_stat": data["t_stat"],
                        "p_value": data["p_value"],
                        "conf_int_lower": data["conf_int"][0],
                        "conf_int_upper": data["conf_int"][1],
                    }
                )
            return pd.DataFrame(rows)

        elif level == "group":
            if self.group_effects is None:
                raise ValueError("Group effects not computed. Use aggregate='group'.")
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
                f"Unknown level: {level}. " "Use 'group_time', 'event_study', or 'group'."
            )

    @property
    def is_significant(self) -> bool:
        """Check if overall ATT is significant."""
        return bool(self.overall_p_value < self.alpha)

    @property
    def significance_stars(self) -> str:
        """Significance stars for overall ATT."""
        return _get_significance_stars(self.overall_p_value)
