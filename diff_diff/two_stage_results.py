"""
Result containers for the Two-Stage DiD estimator.

This module contains TwoStageBootstrapResults and TwoStageDiDResults
dataclasses. Extracted from two_stage.py for module size management.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.results import _format_survey_block, _get_significance_stars
from diff_diff.results_base import BaseResults

__all__ = [
    "TwoStageBootstrapResults",
    "TwoStageDiDResults",
]


@dataclass
class TwoStageBootstrapResults:
    """
    Results from TwoStageDiD bootstrap inference.

    Bootstrap uses multiplier bootstrap on the GMM influence function,
    consistent with other library estimators. The R `did2s` package defaults
    to analytical corrected clustered SEs (``bootstrap = FALSE``); its optional
    block bootstrap (``bootstrap = TRUE``) and this multiplier bootstrap are
    asymptotically equivalent.

    Attributes
    ----------
    n_bootstrap : int
        Number of bootstrap iterations.
    weight_type : str
        Type of bootstrap weights: "rademacher", "mammen", or "webb".
    alpha : float
        Significance level used for confidence intervals.
    overall_att_se : float
        Bootstrap standard error for overall ATT.
    overall_att_ci : tuple
        Bootstrap confidence interval for overall ATT.
    overall_att_p_value : float
        Bootstrap p-value for overall ATT.
    event_study_ses : dict, optional
        Bootstrap SEs for event study effects.
    event_study_cis : dict, optional
        Bootstrap CIs for event study effects.
    event_study_p_values : dict, optional
        Bootstrap p-values for event study effects.
    group_ses : dict, optional
        Bootstrap SEs for group effects.
    group_cis : dict, optional
        Bootstrap CIs for group effects.
    group_p_values : dict, optional
        Bootstrap p-values for group effects.
    bootstrap_distribution : np.ndarray, optional
        Full bootstrap distribution of overall ATT.
    """

    n_bootstrap: int
    weight_type: str
    alpha: float
    overall_att_se: float
    overall_att_ci: Tuple[float, float]
    overall_att_p_value: float
    event_study_ses: Optional[Dict[int, float]] = None
    event_study_cis: Optional[Dict[int, Tuple[float, float]]] = None
    event_study_p_values: Optional[Dict[int, float]] = None
    group_ses: Optional[Dict[Any, float]] = None
    group_cis: Optional[Dict[Any, Tuple[float, float]]] = None
    group_p_values: Optional[Dict[Any, float]] = None
    bootstrap_distribution: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class TwoStageDiDResults(BaseResults):
    """
    Results from Gardner (2022) two-stage DiD estimation.

    Attributes
    ----------
    treatment_effects : pd.DataFrame
        Per-observation treatment effects with columns: unit, time,
        tau_hat, weight. tau_hat is the residualized outcome y_tilde
        for treated observations; weight is 1/n_treated.
    overall_att : float
        Overall average treatment effect on the treated.
    overall_se : float
        Standard error of overall ATT (GMM sandwich).
    overall_t_stat : float
        T-statistic for overall ATT.
    overall_p_value : float
        P-value for overall ATT.
    overall_conf_int : tuple
        Confidence interval for overall ATT.
    event_study_effects : dict, optional
        Dictionary mapping relative time h to effect dict with keys:
        'effect', 'se', 't_stat', 'p_value', 'conf_int', 'n_obs'.
    group_effects : dict, optional
        Dictionary mapping cohort g to effect dict.
    groups : list
        List of treatment cohorts.
    time_periods : list
        List of all time periods.
    n_obs : int
        Total number of observations.
    n_treated_obs : int
        Number of treated observations.
    n_untreated_obs : int
        Number of untreated observations.
    n_treated_units : int
        Number of ever-treated units.
    n_control_units : int
        Number of units contributing to untreated observations.
    alpha : float
        Significance level used.
    bootstrap_results : TwoStageBootstrapResults, optional
        Bootstrap inference results.
    """

    treatment_effects: pd.DataFrame
    overall_att: float
    overall_se: float
    overall_t_stat: float
    overall_p_value: float
    overall_conf_int: Tuple[float, float]
    event_study_effects: Optional[Dict[int, Dict[str, Any]]]
    group_effects: Optional[Dict[Any, Dict[str, Any]]]
    groups: List[Any]
    time_periods: List[Any]
    n_obs: int
    n_treated_obs: int
    n_untreated_obs: int
    n_treated_units: int
    n_control_units: int
    alpha: float = 0.05
    anticipation: int = 0
    bootstrap_results: Optional[TwoStageBootstrapResults] = field(default=None, repr=False)
    # Survey design metadata (SurveyMetadata instance from diff_diff.survey)
    survey_metadata: Optional[Any] = field(default=None, repr=False)
    # --- Variance-estimator metadata (Phase 1b vcov_type threading) ---
    # vcov_type is permanently narrow to {"hc1"} per the Gardner (2022) GMM
    # cluster-sandwich. cluster_name/n_clusters carry the cluster label (the
    # Gardner sandwich always clusters — default at the unit column, see
    # two_stage.py:1547) so summary() renders the unit-cluster CR1 label rather
    # than generic HC1. Both are None under any survey design (the survey block
    # already reports the design's PSU/strata metadata).
    vcov_type: str = "hc1"
    cluster_name: Optional[str] = None
    n_clusters: Optional[int] = None
    # --- Unified event-study surface support (spec section 5, row M-092) ---
    # event_study_vcov: the full Gardner-GMM covariance over the ESTIMATED
    # horizon coefficients, ordered by event_study_vcov_index. The reference
    # period and Proposition-5 horizons are never Stage-2 regression columns,
    # so they appear in event_study_effects but not here; all-filtered
    # horizons (n_obs == 0) ARE columns, carrying the rank guard's NaN
    # rows/columns (consistent with their NaN marginal SEs). Both are None
    # under bootstrap (percentile inference, no covariance) and under
    # replicate-weight survey designs (the replicate VCV has a mixed
    # [overall, ES, groups] layout; not persisted - CS precedent).
    # event_study_df: the scalar df every estimated ES row's safe_inference
    # received (the survey df; None on non-survey fits -> normal theory, and
    # None under bootstrap where the stored inference never used a df).
    event_study_vcov: Optional[np.ndarray] = field(default=None, repr=False)
    event_study_vcov_index: Optional[List[int]] = field(default=None, repr=False)
    event_study_df: Optional[float] = field(default=None, repr=False)

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
            f"TwoStageDiDResults(ATT={self.overall_att:.4f}{sig}, "
            f"SE={self.overall_se:.4f}, "
            f"n_groups={len(self.groups)}, "
            f"n_treated_obs={self.n_treated_obs})"
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
            "Two-Stage DiD Estimator Results (Gardner 2022)".center(85),
            "=" * 85,
            "",
            f"{'Total observations:':<30} {self.n_obs:>10}",
            f"{'Treated observations:':<30} {self.n_treated_obs:>10}",
            f"{'Untreated observations:':<30} {self.n_untreated_obs:>10}",
            f"{'Treated units:':<30} {self.n_treated_units:>10}",
            f"{'Control units:':<30} {self.n_control_units:>10}",
            f"{'Treatment cohorts:':<30} {len(self.groups):>10}",
            f"{'Time periods:':<30} {len(self.time_periods):>10}",
            "",
        ]

        # Survey design info
        if self.survey_metadata is not None:
            sm = self.survey_metadata
            lines.extend(_format_survey_block(sm, 85))

        # Variance-estimator label (Phase 1b vcov_type), with two suppression
        # gates mirroring DiDResults.summary() (results.py:213-226) and
        # ImputationDiDResults.summary():
        #   1. Bootstrap fits: fit() overwrites the reported SE/CI/p-value with
        #      bootstrap_results, so the analytical variance-family label would
        #      mislabel the inference source — surface "Inference method:
        #      bootstrap" + the replication count instead.
        #   2. Survey fits: _format_survey_block above already reports the design
        #      (weight type, strata/PSU counts, replicate method), so a parallel
        #      variance line would be redundant/misleading.
        # Default cluster=None still clusters at the unit column (Gardner GMM
        # sandwich, two_stage.py:1547), so cluster_name carries the unit column
        # and _format_vcov_label renders the unit-cluster CR1 label, not HC1.
        if self.bootstrap_results is not None:
            lines.append(f"{'Inference method:':<30} {'bootstrap':>15}")
            lines.append(
                f"{'Bootstrap replications:':<30} {self.bootstrap_results.n_bootstrap:>15}"
            )
            lines.append("")
        elif self.survey_metadata is None:
            from diff_diff.results import _format_vcov_label

            vcov_label = _format_vcov_label(
                self.vcov_type,
                cluster_name=self.cluster_name,
                n_clusters=self.n_clusters,
                n_obs=self.n_obs,
            )
            if vcov_label:
                lines.append(f"{'Variance estimator:':<30} {vcov_label:>15}")
                lines.append("")

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
            ]
        )

        cv = self.coef_var
        if np.isfinite(cv):
            lines.append(f"{'CV (SE/abs(ATT)):':<25} {cv:>10.4f}")

        lines.append("")

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
            - "observation": Per-observation treatment effects

        Returns
        -------
        pd.DataFrame
            Results as DataFrame.
        """
        if level == "observation":
            return self.treatment_effects.copy()

        elif level == "event_study":
            if self.event_study_effects is None:
                raise ValueError(
                    "Event study effects not computed. "
                    "Use aggregate='event_study' or aggregate='all'."
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
            if self.group_effects is None:
                raise ValueError(
                    "Group effects not computed. " "Use aggregate='group' or aggregate='all'."
                )
            rows = []
            for g, data in sorted(self.group_effects.items()):
                rows.append(
                    {
                        "group": g,
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

        else:
            raise ValueError(
                f"Unknown level: {level}. Use 'event_study', 'group', or 'observation'."
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert headline results to a dictionary.

        Provides flat aliases (``att``/``se``/``t_stat``/``p_value``/
        ``conf_int_lower``/``conf_int_upper``) plus variance-estimator metadata
        (``vcov_type``, optional ``cluster_name``/``n_clusters``, optional
        ``n_bootstrap``, ``inference_method``). Per-cohort / per-horizon detail is
        exposed via :meth:`to_dataframe`. ``inference_method`` reports
        ``"cluster"`` for the default fit because the Gardner GMM sandwich
        clusters at the unit column (``n_clusters`` populated) — consistent with
        the CR1-at-unit summary label.

        Returns
        -------
        Dict[str, Any]
            Headline overall ATT plus inference metadata.
        """
        result: Dict[str, Any] = {
            "att": self.overall_att,
            "se": self.overall_se,
            "t_stat": self.overall_t_stat,
            "p_value": self.overall_p_value,
            "conf_int_lower": self.overall_conf_int[0],
            "conf_int_upper": self.overall_conf_int[1],
            "n_obs": self.n_obs,
            "n_treated_obs": self.n_treated_obs,
            "n_untreated_obs": self.n_untreated_obs,
            "n_treated_units": self.n_treated_units,
            "n_control_units": self.n_control_units,
            "alpha": self.alpha,
            "anticipation": self.anticipation,
            "vcov_type": self.vcov_type,
        }
        if self.cluster_name is not None:
            result["cluster_name"] = self.cluster_name
        if self.n_clusters is not None:
            result["n_clusters"] = self.n_clusters
        if self.bootstrap_results is not None:
            result["n_bootstrap"] = self.bootstrap_results.n_bootstrap
            result["inference_method"] = "bootstrap"
        elif self.survey_metadata is not None:
            result["inference_method"] = "survey"
        elif self.n_clusters is not None:
            result["inference_method"] = "cluster"
        else:
            result["inference_method"] = "analytical"
        return result

    @property
    def is_significant(self) -> bool:
        """Check if overall ATT is significant."""
        return bool(self.overall_p_value < self.alpha)

    @property
    def significance_stars(self) -> str:
        """Significance stars for overall ATT."""
        return _get_significance_stars(self.overall_p_value)
