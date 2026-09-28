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
from diff_diff.results_base import BaseResults, _coverage_pct, _require_fit_alpha

if TYPE_CHECKING:
    from diff_diff.staggered_bootstrap import CSBootstrapResults


@dataclass
class StaggeredTripleDiffResults(BaseResults):
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
    overall_att_es : float, optional
        Paper Eq. (4.14) "overall" ATT: the unweighted mean of the post-treatment
        event-study effects ES(e). Populated only when ``aggregate`` is
        ``"event_study"`` or ``"all"``; ``None`` otherwise. Distinct from
        ``overall_att`` (the Callaway-Sant'Anna simple post-treatment (g,t) average,
        which is the default headline ATT).
    overall_se_es, overall_t_stat_es, overall_p_value_es, overall_conf_int_es : optional
        Standard error, t-statistic, p-value, and confidence interval for
        ``overall_att_es``; ``None`` when ``overall_att_es`` is ``None``.
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
    # Anticipation periods (``k``) used at fit time. Persisted so
    # downstream diagnostics in ``BusinessReport`` / ``DiagnosticReport``
    # can render the anticipation-aware assumption block and
    # horizon-classification cutoffs accurately on real fits.
    anticipation: int = 0
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
    epv_diagnostics: Optional[Dict[Tuple[Any, Any], Dict[str, Any]]] = field(
        default=None, repr=False
    )
    epv_threshold: float = 10
    pscore_fallback: str = "error"
    # Paper Eq. (4.14) "overall" ATT: the unweighted mean of the post-treatment
    # event-study effects ES(e). Populated only when aggregate in {"event_study",
    # "all"}; None otherwise. Distinct from the default ``overall_att`` (the
    # Callaway-Sant'Anna simple post-treatment (g,t) average). See REGISTRY
    # ## StaggeredTripleDifference "Aggregation".
    overall_att_es: Optional[float] = None
    overall_se_es: Optional[float] = None
    overall_t_stat_es: Optional[float] = None
    overall_p_value_es: Optional[float] = None
    overall_conf_int_es: Optional[Tuple[float, float]] = None

    # --- Inference-field aliases (balance/external-adapter compatibility) ---
    @property
    def att(self) -> float:
        return self.overall_att

    @property
    def se(self) -> float:
        return self.overall_se

    @property
    def conf_int(self) -> Tuple[float, float]:
        return self.overall_conf_int

    @property
    def p_value(self) -> float:
        return self.overall_p_value

    @property
    def t_stat(self) -> float:
        return self.overall_t_stat

    def __repr__(self) -> str:
        """Concise string representation."""
        sig = _get_significance_stars(self.overall_p_value)
        return (
            f"StaggeredTripleDiffResults(ATT={self.overall_att:.4f}{sig}, "
            f"SE={self.overall_se:.4f}, "
            f"n_groups={len(self.groups)}, "
            f"n_periods={len(self.time_periods)})"
        )

    @property
    def coef_var(self) -> float:
        """Coefficient of variation: SE / abs(overall ATT). NaN when ATT is 0 or SE non-finite."""
        if not (np.isfinite(self.overall_se) and self.overall_se >= 0):
            return np.nan
        if not np.isfinite(self.overall_att) or self.overall_att == 0:
            return np.nan
        return self.overall_se / abs(self.overall_att)

    def summary(self, alpha: Optional[float] = None) -> str:
        """
        Generate formatted summary of estimation results.

        Parameters
        ----------
        alpha : float, optional
            Accepted for signature uniformity. The stored intervals were
            computed at fit time; a value different from the stored
            ``alpha`` raises ValueError rather than silently recomputing
            or relabeling (bootstrap percentile intervals cannot be
            reconstructed from the reported SE). Re-fit at the desired
            alpha instead.

        Returns
        -------
        str
            Formatted summary.
        """
        alpha = _require_fit_alpha(alpha, self.alpha)
        conf_level = _coverage_pct(alpha)

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
            ]
        )

        # Paper Eq. (4.14) overall (event-study average), when computed
        # (aggregate in {"event_study", "all"}). The headline ATT above remains the
        # Callaway-Sant'Anna simple post-treatment (g,t) average.
        if (
            self.overall_att_es is not None
            and self.overall_se_es is not None
            and self.overall_conf_int_es is not None
        ):
            p_es = self.overall_p_value_es if self.overall_p_value_es is not None else float("nan")
            t_es = self.overall_t_stat_es if self.overall_t_stat_es is not None else float("nan")
            sig_es = _get_significance_stars(p_es)
            lines.extend(
                [
                    "",
                    "Overall ATT (event-study average, paper Eq. 4.14):",
                    f"{'ATT':<15} {self.overall_att_es:>12.4f} {self.overall_se_es:>12.4f} "
                    f"{t_es:>10.3f} {p_es:>10.4f} {sig_es:>6}",
                    f"{conf_level}% Confidence Interval: "
                    f"[{self.overall_conf_int_es[0]:.4f}, {self.overall_conf_int_es[1]:.4f}]",
                ]
            )

        cv = self.coef_var
        if np.isfinite(cv):
            lines.append(f"{'CV (SE/abs(ATT)):':<25} {cv:>10.4f}")

        lines.append("")

        # EPV diagnostics block (if any cohort has low EPV)
        if self.epv_diagnostics:
            low_epv = {k: v for k, v in self.epv_diagnostics.items() if v.get("is_low")}
            if low_epv:
                n_affected = len(low_epv)
                n_total = len(self.epv_diagnostics)
                min_entry = min(low_epv.values(), key=lambda v: v["epv"])
                min_g = min(low_epv.keys(), key=lambda k: low_epv[k]["epv"])
                lines.extend(
                    [
                        "-" * 85,
                        "Propensity Score Diagnostics".center(85),
                        "-" * 85,
                        f"WARNING: Low Events Per Variable (EPV) in "
                        f"{n_affected} of {n_total} cohort-time cell(s).",
                        f"Minimum EPV: {min_entry['epv']:.1f} "
                        f"(cohort g={min_g[0]}). "
                        f"Threshold: {self.epv_threshold:.0f}.",
                        "Consider: estimation_method='reg' or fewer covariates.",
                        "Call results.epv_summary() for per-cohort details.",
                        "-" * 85,
                        "",
                    ]
                )

        # Event study effects
        if self.event_study_effects:
            ci_label = "Simult. CI" if self.cband_crit_value is not None else "Pointwise CI"
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

    def epv_summary(self, show_all: bool = False) -> pd.DataFrame:
        """
        Return per-cohort EPV diagnostics as a DataFrame.

        Parameters
        ----------
        show_all : bool, default False
            If False, only show cells with low EPV. If True, show all cells.

        Returns
        -------
        pd.DataFrame
            Columns: group, time, epv, n_events, n_params, is_low.
        """
        if not self.epv_diagnostics:
            return pd.DataFrame(columns=["group", "time", "epv", "n_events", "n_params", "is_low"])
        rows = []
        for (g, t), diag in sorted(self.epv_diagnostics.items()):
            if show_all or diag.get("is_low", False):
                rows.append(
                    {
                        "group": g,
                        "time": t,
                        "epv": diag.get("epv"),
                        "n_events": diag.get("n_events"),
                        "n_params": diag.get("k"),
                        "is_low": diag.get("is_low", False),
                    }
                )
        cols = ["group", "time", "epv", "n_events", "n_params", "is_low"]
        return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

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
                row = {
                    "group": g,
                    "time": t,
                    "effect": data["effect"],
                    "se": data["se"],
                    "t_stat": data["t_stat"],
                    "p_value": data["p_value"],
                    "conf_int_lower": data["conf_int"][0],
                    "conf_int_upper": data["conf_int"][1],
                }
                if self.epv_diagnostics and (g, t) in self.epv_diagnostics:
                    row["epv"] = self.epv_diagnostics[(g, t)].get("epv")
                rows.append(row)
            return pd.DataFrame(rows)

        elif level == "event_study":
            if self.event_study_effects is None:
                raise ValueError("Event study effects not computed. Use aggregate='event_study'.")
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
        if self.overall_att_es is not None:
            d["overall_att_es"] = self.overall_att_es
            d["overall_se_es"] = self.overall_se_es
            d["overall_t_stat_es"] = self.overall_t_stat_es
            d["overall_p_value_es"] = self.overall_p_value_es
            d["overall_conf_int_es"] = self.overall_conf_int_es
        return d

    @property
    def is_significant(self) -> bool:
        """Check if overall ATT is significant."""
        return bool(self.overall_p_value < self.alpha)

    @property
    def significance_stars(self) -> str:
        """Significance stars for overall ATT."""
        return _get_significance_stars(self.overall_p_value)
