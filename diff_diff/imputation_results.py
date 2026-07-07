"""
Result containers for the Imputation DiD estimator.

This module contains ImputationBootstrapResults and ImputationDiDResults
dataclasses. Extracted from imputation.py for module size management.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.results import _format_survey_block, _get_significance_stars

__all__ = [
    "ImputationBootstrapResults",
    "ImputationDiDResults",
]


@dataclass
class ImputationBootstrapResults:
    """
    Results from ImputationDiD bootstrap inference.

    Bootstrap is a library extension beyond Borusyak et al. (2024), which
    proposes only analytical inference via the conservative variance estimator.
    Provided for consistency with CallawaySantAnna and SunAbraham.

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
class ImputationDiDResults:
    """
    Results from Borusyak-Jaravel-Spiess (2024) imputation DiD estimation.

    Attributes
    ----------
    treatment_effects : pd.DataFrame
        Unit-level treatment effects with columns: unit, time, tau_hat, weight.
    overall_att : float
        Overall average treatment effect on the treated.
    overall_se : float
        Standard error of overall ATT.
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
        Number of treated observations (:math:`|\\Omega_1|`).
    n_untreated_obs : int
        Number of untreated observations (:math:`|\\Omega_0|`).
    n_treated_units : int
        Number of ever-treated units.
    n_control_units : int
        Number of units contributing to Omega_0.
    alpha : float
        Significance level used.
    pretrend_results : dict, optional
        Populated by pretrend_test().
    bootstrap_results : ImputationBootstrapResults, optional
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
    pretrend_results: Optional[Dict[str, Any]] = field(default=None, repr=False)
    bootstrap_results: Optional[ImputationBootstrapResults] = field(default=None, repr=False)
    # Internal: stores data needed for pretrend_test()
    _estimator_ref: Optional[Any] = field(default=None, repr=False)
    # Survey design metadata (SurveyMetadata instance from diff_diff.survey)
    survey_metadata: Optional[Any] = field(default=None, repr=False)
    # Variance-estimator metadata (Phase 1b interstitial #3).
    # vcov_type is permanently narrow to {"hc1"} per the IF-based variance
    # contract (see REGISTRY.md). cluster_name + n_clusters are populated
    # only under bare cluster=; suppressed under survey designs (the survey
    # block in summary() already renders the design's PSU/strata metadata).
    vcov_type: str = field(default="hc1")
    cluster_name: Optional[str] = field(default=None)
    n_clusters: Optional[int] = field(default=None)
    # BJS 2024 Supp. App. A.9 leave-one-out finite-sample variance refinement
    # (opt-in). Recorded here so reported SEs are self-describing.
    leave_one_out: bool = field(default=False)

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
            f"ImputationDiDResults(ATT={self.overall_att:.4f}{sig}, "
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
            "Imputation DiD Estimator Results (Borusyak et al. 2024)".center(85),
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

        # Inference / variance metadata. Two suppression rules — match the
        # canonical DiDResults pattern at diff_diff/results.py:213-226:
        #   1. Survey designs: the survey block above already names the
        #      design + n_psu + df; the analytical SE is TSL on the combined
        #      IF (or replicate reweighting), not the raw HC1/CR1 sandwich.
        #   2. Bootstrap fits: fit() overwrites the reported SE/CI/p-value
        #      with bootstrap_results, so the analytical variance-family
        #      label would misstate the actual inference source. Surface an
        #      "Inference method: bootstrap" + replication count instead.
        if self.bootstrap_results is not None:
            lines.append(f"{'Inference method:':<30} {'bootstrap':>15}")
            lines.append(
                f"{'Bootstrap replications:':<30} {self.bootstrap_results.n_bootstrap:>15}"
            )
        elif self.survey_metadata is None:
            # Analytical, non-survey path: render the variance-family label.
            # For cluster=None ImputationDiD still clusters at unit by default
            # (Theorem 3 equation 7 conservative variance on per-unit IF
            # sums), so cluster_name is populated with the unit column name
            # and _format_vcov_label renders the unit-cluster CR1 label.
            from diff_diff.results import _format_vcov_label

            vcov_label = _format_vcov_label(
                self.vcov_type,
                cluster_name=self.cluster_name,
                n_clusters=self.n_clusters,
                n_obs=self.n_obs,
            )
            if vcov_label:
                lines.append(f"{'Variance estimator:':<30} {vcov_label:>15}")
        if self.n_clusters is not None and self.bootstrap_results is None:
            lines.append(f"{'Number of clusters:':<30} {self.n_clusters:>15}")
        if self.leave_one_out:
            lines.append(f"{'Leave-one-out variance:':<30} {'A.9 (BJS 2024)':>15}")

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

        # Pre-trend test
        if self.pretrend_results is not None:
            pt = self.pretrend_results
            lines.extend(
                [
                    "-" * 85,
                    "Pre-Trend Test (Equation 9)".center(85),
                    "-" * 85,
                    f"{'F-statistic:':<30} {pt['f_stat']:>10.3f}",
                    f"{'P-value:':<30} {pt['p_value']:>10.4f}",
                    f"{'Degrees of freedom:':<30} {pt['df']:>10}",
                    f"{'Number of leads:':<30} {pt['n_leads']:>10}",
                    "-" * 85,
                    "",
                ]
            )

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

    def to_dataframe(self, level: str = "observation") -> pd.DataFrame:
        """
        Convert results to DataFrame.

        Parameters
        ----------
        level : str, default="observation"
            Level of aggregation:
            - "observation": Unit-level treatment effects
            - "event_study": Event study effects by relative time
            - "group": Group (cohort) effects

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
                f"Unknown level: {level}. Use 'observation', 'event_study', or 'group'."
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert results to a dictionary.

        Provides flat headline aliases (``att``/``se``/``t_stat``/``p_value``/
        ``conf_int_lower``/``conf_int_upper``) plus variance-estimator
        metadata (``vcov_type``, optional ``cluster_name``/``n_clusters``,
        optional ``n_bootstrap``, ``inference_method``).

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the headline overall ATT and inference
            metadata. Per-cohort / per-horizon detail is exposed via
            :meth:`to_dataframe`.
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
            "leave_one_out": self.leave_one_out,
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

    def pretrend_test(self, n_leads: Optional[int] = None) -> Dict[str, Any]:
        """
        Run a pre-trend test (Equation 9 of Borusyak et al. 2024).

        Adds pre-treatment lead indicators to the Step 1 OLS and tests
        their joint significance via a Wald F-test (cluster-robust, or
        design-based survey VCV when survey_design was provided at fit).

        Parameters
        ----------
        n_leads : int, optional
            Number of pre-treatment leads to include. If None, uses all
            available pre-treatment periods minus one (for the reference period).

        Returns
        -------
        dict
            Dictionary with keys: 'f_stat', 'p_value', 'df', 'n_leads',
            'lead_coefficients'.
        """
        if self._estimator_ref is None:
            raise RuntimeError(
                "Pre-trend test requires internal estimator reference. "
                "Re-fit the model to use this method."
            )
        result = self._estimator_ref._pretrend_test(n_leads=n_leads)
        self.pretrend_results = result
        return result

    @property
    def is_significant(self) -> bool:
        """Check if overall ATT is significant."""
        return bool(self.overall_p_value < self.alpha)

    @property
    def significance_stars(self) -> str:
        """Significance stars for overall ATT."""
        return _get_significance_stars(self.overall_p_value)
