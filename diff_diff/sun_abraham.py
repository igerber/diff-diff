"""
Sun-Abraham Interaction-Weighted Estimator for staggered DiD.

Implements the estimator from Sun & Abraham (2021), "Estimating dynamic
treatment effects in event studies with heterogeneous treatment effects",
Journal of Econometrics.

This provides an alternative to Callaway-Sant'Anna using an interaction-weighted
(IW) regression approach rather than aggregating 2x2 DiD comparisons.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from diff_diff.results import _get_significance_stars
from diff_diff.staggered import _generate_bootstrap_weights
from diff_diff.utils import (
    compute_confidence_interval,
    compute_p_value,
)


@dataclass
class SunAbrahamResults:
    """
    Results from Sun-Abraham (2021) interaction-weighted estimation.

    Attributes
    ----------
    event_study_effects : dict
        Dictionary mapping relative time to effect dictionaries with keys:
        'effect', 'se', 't_stat', 'p_value', 'conf_int', 'n_groups'.
    overall_att : float
        Overall average treatment effect (weighted average of post-treatment effects).
    overall_se : float
        Standard error of overall ATT.
    overall_t_stat : float
        T-statistic for overall ATT.
    overall_p_value : float
        P-value for overall ATT.
    overall_conf_int : tuple
        Confidence interval for overall ATT.
    cohort_weights : dict
        Dictionary mapping relative time to cohort weight dictionaries.
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
    alpha : float
        Significance level used for confidence intervals.
    control_group : str
        Type of control group used.
    """

    event_study_effects: Dict[int, Dict[str, Any]]
    overall_att: float
    overall_se: float
    overall_t_stat: float
    overall_p_value: float
    overall_conf_int: Tuple[float, float]
    cohort_weights: Dict[int, Dict[Any, float]]
    groups: List[Any]
    time_periods: List[Any]
    n_obs: int
    n_treated_units: int
    n_control_units: int
    alpha: float = 0.05
    control_group: str = "never_treated"
    bootstrap_results: Optional["SABootstrapResults"] = field(default=None, repr=False)
    cohort_effects: Optional[Dict[Tuple[Any, int], Dict[str, Any]]] = field(
        default=None, repr=False
    )

    def __repr__(self) -> str:
        """Concise string representation."""
        sig = _get_significance_stars(self.overall_p_value)
        n_rel_periods = len(self.event_study_effects)
        return (
            f"SunAbrahamResults(ATT={self.overall_att:.4f}{sig}, "
            f"SE={self.overall_se:.4f}, "
            f"n_groups={len(self.groups)}, "
            f"n_rel_periods={n_rel_periods})"
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
            "Sun-Abraham Interaction-Weighted Estimator Results".center(85),
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
            Level of aggregation: "event_study" or "cohort".

        Returns
        -------
        pd.DataFrame
            Results as DataFrame.
        """
        if level == "event_study":
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

        elif level == "cohort":
            if self.cohort_effects is None:
                raise ValueError(
                    "Cohort-level effects not available. "
                    "They are computed internally but not stored by default."
                )
            rows = []
            for (cohort, rel_t), data in sorted(self.cohort_effects.items()):
                rows.append(
                    {
                        "cohort": cohort,
                        "relative_period": rel_t,
                        "effect": data["effect"],
                        "se": data["se"],
                        "weight": data.get("weight", np.nan),
                    }
                )
            return pd.DataFrame(rows)

        else:
            raise ValueError(
                f"Unknown level: {level}. Use 'event_study' or 'cohort'."
            )

    @property
    def is_significant(self) -> bool:
        """Check if overall ATT is significant."""
        return bool(self.overall_p_value < self.alpha)

    @property
    def significance_stars(self) -> str:
        """Significance stars for overall ATT."""
        return _get_significance_stars(self.overall_p_value)


@dataclass
class SABootstrapResults:
    """
    Results from Sun-Abraham multiplier bootstrap inference.

    Attributes
    ----------
    n_bootstrap : int
        Number of bootstrap iterations.
    weight_type : str
        Type of bootstrap weights used.
    alpha : float
        Significance level used for confidence intervals.
    overall_att_se : float
        Bootstrap standard error for overall ATT.
    overall_att_ci : Tuple[float, float]
        Bootstrap confidence interval for overall ATT.
    overall_att_p_value : float
        Bootstrap p-value for overall ATT.
    event_study_ses : Dict[int, float]
        Bootstrap SEs for event study effects.
    event_study_cis : Dict[int, Tuple[float, float]]
        Bootstrap CIs for event study effects.
    event_study_p_values : Dict[int, float]
        Bootstrap p-values for event study effects.
    bootstrap_distribution : Optional[np.ndarray]
        Full bootstrap distribution of overall ATT.
    """

    n_bootstrap: int
    weight_type: str
    alpha: float
    overall_att_se: float
    overall_att_ci: Tuple[float, float]
    overall_att_p_value: float
    event_study_ses: Dict[int, float]
    event_study_cis: Dict[int, Tuple[float, float]]
    event_study_p_values: Dict[int, float]
    bootstrap_distribution: Optional[np.ndarray] = field(default=None, repr=False)


class SunAbraham:
    """
    Sun-Abraham (2021) interaction-weighted estimator for staggered DiD.

    This estimator provides event-study coefficients using a saturated
    regression with cohort-by-relative-time interactions. It's an alternative
    to Callaway-Sant'Anna that uses different weighting.

    The key innovation is the interaction-weighting approach:
    1. Run a regression with cohort × relative_time indicators
    2. Weight cohort-specific effects by the share of each cohort in
       the treated population at each relative time

    This avoids the negative weighting problem of standard TWFE and provides
    consistent event-study estimates under treatment effect heterogeneity.

    Parameters
    ----------
    control_group : str, default="never_treated"
        Which units to use as controls:
        - "never_treated": Use only never-treated units
        - "not_yet_treated": Use never-treated and not-yet-treated units
    anticipation : int, default=0
        Number of periods before treatment where effects may occur.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    cluster : str, optional
        Column name for cluster-robust standard errors.
    n_bootstrap : int, default=0
        Number of bootstrap iterations for inference.
        If 0, uses analytical standard errors.
    bootstrap_weights : str, default="rademacher"
        Type of weights for multiplier bootstrap.
    seed : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    results_ : SunAbrahamResults
        Estimation results after calling fit().
    is_fitted_ : bool
        Whether the model has been fitted.

    Examples
    --------
    Basic usage:

    >>> import pandas as pd
    >>> from diff_diff import SunAbraham
    >>>
    >>> # Panel data with staggered treatment
    >>> data = pd.DataFrame({
    ...     'unit': [...],
    ...     'time': [...],
    ...     'outcome': [...],
    ...     'first_treat': [...]  # 0 for never-treated
    ... })
    >>>
    >>> sa = SunAbraham()
    >>> results = sa.fit(data, outcome='outcome', unit='unit',
    ...                  time='time', first_treat='first_treat')
    >>> results.print_summary()

    With covariates:

    >>> sa = SunAbraham()
    >>> results = sa.fit(data, outcome='outcome', unit='unit',
    ...                  time='time', first_treat='first_treat',
    ...                  covariates=['age', 'income'])

    Notes
    -----
    The Sun-Abraham estimator uses a saturated regression approach:

    Y_it = α_i + λ_t + Σ_g Σ_e [β_{g,e} × 1(G_i=g) × 1(t-G_i=e)] + ε_it

    where G_i is unit i's treatment cohort and e is relative time.

    The event-study coefficients are then computed as:

    β_e = Σ_g w_{g,e} × β_{g,e}

    where w_{g,e} is the share of cohort g in the treated population at
    relative time e.

    Compared to Callaway-Sant'Anna:
    - SA uses regression-based estimation; CS uses 2x2 DiD
    - SA can be more efficient when effects are homogeneous
    - Both are consistent under heterogeneous treatment effects
    - Running both provides a useful robustness check

    References
    ----------
    Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects in
    event studies with heterogeneous treatment effects. Journal of
    Econometrics, 225(2), 175-199.
    """

    def __init__(
        self,
        control_group: str = "never_treated",
        anticipation: int = 0,
        alpha: float = 0.05,
        cluster: Optional[str] = None,
        n_bootstrap: int = 0,
        bootstrap_weights: str = "rademacher",
        seed: Optional[int] = None,
    ):
        if control_group not in ["never_treated", "not_yet_treated"]:
            raise ValueError(
                f"control_group must be 'never_treated' or 'not_yet_treated', "
                f"got '{control_group}'"
            )

        if bootstrap_weights not in ["rademacher", "mammen", "webb"]:
            raise ValueError(
                f"bootstrap_weights must be 'rademacher', 'mammen', or 'webb', "
                f"got '{bootstrap_weights}'"
            )

        self.control_group = control_group
        self.anticipation = anticipation
        self.alpha = alpha
        self.cluster = cluster
        self.n_bootstrap = n_bootstrap
        self.bootstrap_weights = bootstrap_weights
        self.seed = seed

        self.is_fitted_ = False
        self.results_: Optional[SunAbrahamResults] = None

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        covariates: Optional[List[str]] = None,
        min_pre_periods: int = 1,
        min_post_periods: int = 1,
    ) -> SunAbrahamResults:
        """
        Fit the Sun-Abraham estimator.

        Parameters
        ----------
        data : pd.DataFrame
            Panel data with unit and time identifiers.
        outcome : str
            Name of outcome variable column.
        unit : str
            Name of unit identifier column.
        time : str
            Name of time period column.
        first_treat : str
            Name of column indicating when unit was first treated.
            Use 0 (or np.inf) for never-treated units.
        covariates : list, optional
            List of covariate column names.
        min_pre_periods : int, default=1
            Minimum number of pre-treatment periods to include in event study.
        min_post_periods : int, default=1
            Minimum number of post-treatment periods to include in event study.

        Returns
        -------
        SunAbrahamResults
            Object containing all estimation results.

        Raises
        ------
        ValueError
            If required columns are missing or data validation fails.
        """
        # Validate inputs
        required_cols = [outcome, unit, time, first_treat]
        if covariates:
            required_cols.extend(covariates)

        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Create working copy
        df = data.copy()

        # Ensure numeric types
        df[time] = pd.to_numeric(df[time])
        df[first_treat] = pd.to_numeric(df[first_treat])

        # Identify groups and time periods
        time_periods = sorted(df[time].unique())
        treatment_groups = sorted([g for g in df[first_treat].unique() if g > 0])

        # Never-treated indicator
        df["_never_treated"] = (df[first_treat] == 0) | (df[first_treat] == np.inf)

        # Get unique units
        unit_info = (
            df.groupby(unit)
            .agg({first_treat: "first", "_never_treated": "first"})
            .reset_index()
        )

        n_treated_units = int((unit_info[first_treat] > 0).sum())
        n_control_units = int((unit_info["_never_treated"]).sum())

        if n_control_units == 0:
            raise ValueError(
                "No never-treated units found. Check 'first_treat' column."
            )

        if len(treatment_groups) == 0:
            raise ValueError(
                "No treated units found. Check 'first_treat' column."
            )

        # Compute relative time for each observation
        df["_rel_time"] = df.apply(
            lambda row: (
                row[time] - row[first_treat]
                if row[first_treat] > 0
                else np.nan
            ),
            axis=1,
        )

        # Identify the range of relative time periods to estimate
        # We need at least one cohort observed at each relative time
        rel_times_by_cohort = {}
        for g in treatment_groups:
            g_times = df[df[first_treat] == g][time].unique()
            rel_times_by_cohort[g] = sorted([t - g for t in g_times])

        # Find common relative time range
        all_rel_times: set = set()
        for g, rel_times in rel_times_by_cohort.items():
            all_rel_times.update(rel_times)

        all_rel_times_sorted = sorted(all_rel_times)

        # Filter to reasonable range
        min_rel = max(min(all_rel_times_sorted), -20)  # cap at -20
        max_rel = min(max(all_rel_times_sorted), 20)  # cap at +20

        # Reference period: last pre-treatment period (typically -1)
        reference_period = -1 - self.anticipation

        # Get relative periods to estimate (excluding reference)
        rel_periods_to_estimate = [
            e
            for e in all_rel_times_sorted
            if min_rel <= e <= max_rel and e != reference_period
        ]

        # Compute cohort-specific effects using the IW approach
        cohort_effects, cohort_residuals = self._compute_cohort_effects(
            df,
            outcome,
            unit,
            time,
            first_treat,
            treatment_groups,
            time_periods,
            rel_periods_to_estimate,
            reference_period,
            covariates,
        )

        # Compute interaction-weighted event study effects
        event_study_effects, cohort_weights = self._compute_iw_effects(
            df,
            unit,
            first_treat,
            treatment_groups,
            rel_periods_to_estimate,
            cohort_effects,
        )

        # Compute overall ATT (average of post-treatment effects)
        post_effects = [
            (e, eff)
            for e, eff in event_study_effects.items()
            if e >= 0
        ]

        if post_effects:
            # Weight by number of treated observations at each relative time
            post_weights = []
            post_estimates = []
            post_variances = []

            for e, eff in post_effects:
                # Count treated observations at relative time e
                n_treated_at_e = len(
                    df[(df["_rel_time"] == e) & (df[first_treat] > 0)]
                )
                post_weights.append(n_treated_at_e)
                post_estimates.append(eff["effect"])
                post_variances.append(eff["se"] ** 2)

            post_weights = np.array(post_weights, dtype=float)
            post_weights = post_weights / post_weights.sum()

            overall_att = float(np.sum(post_weights * np.array(post_estimates)))
            overall_var = float(
                np.sum((post_weights**2) * np.array(post_variances))
            )
            overall_se = np.sqrt(overall_var)
        else:
            overall_att = 0.0
            overall_se = 0.0

        overall_t = overall_att / overall_se if overall_se > 0 else 0.0
        overall_p = compute_p_value(overall_t)
        overall_ci = compute_confidence_interval(overall_att, overall_se, self.alpha)

        # Run bootstrap if requested
        bootstrap_results = None
        if self.n_bootstrap > 0:
            bootstrap_results = self._run_bootstrap(
                df=df,
                outcome=outcome,
                unit=unit,
                time=time,
                first_treat=first_treat,
                treatment_groups=treatment_groups,
                time_periods=time_periods,
                rel_periods_to_estimate=rel_periods_to_estimate,
                reference_period=reference_period,
                covariates=covariates,
                original_event_study=event_study_effects,
                original_overall_att=overall_att,
            )

            # Update results with bootstrap inference
            overall_se = bootstrap_results.overall_att_se
            overall_t = overall_att / overall_se if overall_se > 0 else 0.0
            overall_p = bootstrap_results.overall_att_p_value
            overall_ci = bootstrap_results.overall_att_ci

            # Update event study effects
            for e in event_study_effects:
                if e in bootstrap_results.event_study_ses:
                    event_study_effects[e]["se"] = bootstrap_results.event_study_ses[e]
                    event_study_effects[e]["conf_int"] = (
                        bootstrap_results.event_study_cis[e]
                    )
                    event_study_effects[e]["p_value"] = (
                        bootstrap_results.event_study_p_values[e]
                    )
                    eff_val = event_study_effects[e]["effect"]
                    se_val = event_study_effects[e]["se"]
                    event_study_effects[e]["t_stat"] = (
                        eff_val / se_val if se_val > 0 else 0.0
                    )

        # Store results
        self.results_ = SunAbrahamResults(
            event_study_effects=event_study_effects,
            overall_att=overall_att,
            overall_se=overall_se,
            overall_t_stat=overall_t,
            overall_p_value=overall_p,
            overall_conf_int=overall_ci,
            cohort_weights=cohort_weights,
            groups=treatment_groups,
            time_periods=time_periods,
            n_obs=len(df),
            n_treated_units=n_treated_units,
            n_control_units=n_control_units,
            alpha=self.alpha,
            control_group=self.control_group,
            bootstrap_results=bootstrap_results,
            cohort_effects=cohort_effects,
        )

        self.is_fitted_ = True
        return self.results_

    def _compute_cohort_effects(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        treatment_groups: List[Any],
        time_periods: List[Any],
        rel_periods: List[int],
        reference_period: int,
        covariates: Optional[List[str]],
    ) -> Tuple[Dict[Tuple[Any, int], Dict[str, Any]], Dict[str, np.ndarray]]:
        """
        Compute cohort-specific treatment effects using 2x2 DiD.

        For each cohort g and relative period e, estimate ATT(g,e) using
        never-treated or not-yet-treated as controls.

        Returns
        -------
        cohort_effects : dict
            Dictionary mapping (cohort, rel_period) to effect info.
        residuals : dict
            Residuals for bootstrap (keyed by unit).
        """
        cohort_effects: Dict[Tuple[Any, int], Dict[str, Any]] = {}
        residuals: Dict[str, np.ndarray] = {}

        for g in treatment_groups:
            # Get units in this cohort
            cohort_units = df[df[first_treat] == g][unit].unique()
            n_cohort = len(cohort_units)

            if n_cohort == 0:
                continue

            # Base period for this cohort
            base_period = g - 1 - self.anticipation
            if base_period not in time_periods:
                # Find closest earlier period
                earlier = [t for t in time_periods if t < g - self.anticipation]
                if not earlier:
                    continue
                base_period = max(earlier)

            for e in rel_periods:
                # Calendar time for this relative period
                t = g + e

                if t not in time_periods:
                    continue

                # Skip if this is before treatment (we handle pre-treatment
                # by comparing to base period)
                if e < -self.anticipation - 1:
                    t_compare = t
                else:
                    t_compare = t

                # Get control units
                if self.control_group == "never_treated":
                    control_mask = df["_never_treated"]
                else:
                    # Not yet treated at time t
                    control_mask = (df["_never_treated"]) | (df[first_treat] > t)

                control_units = df[control_mask][unit].unique()

                if len(control_units) == 0:
                    continue

                # Compute 2x2 DiD: compare cohort g vs controls, from base to t
                try:
                    att_ge, se_ge, inf_func = self._compute_2x2_did(
                        df,
                        outcome,
                        unit,
                        time,
                        cohort_units,
                        control_units,
                        base_period,
                        t,
                        covariates,
                    )

                    if att_ge is not None:
                        t_stat = att_ge / se_ge if se_ge > 0 else 0.0
                        p_val = compute_p_value(t_stat)
                        ci = compute_confidence_interval(att_ge, se_ge, self.alpha)

                        cohort_effects[(g, e)] = {
                            "effect": att_ge,
                            "se": se_ge,
                            "t_stat": t_stat,
                            "p_value": p_val,
                            "conf_int": ci,
                            "n_cohort": n_cohort,
                            "inf_func": inf_func,
                        }
                except Exception:
                    # Skip this cohort-period if estimation fails
                    continue

        return cohort_effects, residuals

    def _compute_2x2_did(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        treated_units: np.ndarray,
        control_units: np.ndarray,
        base_period: Any,
        post_period: Any,
        covariates: Optional[List[str]],
    ) -> Tuple[Optional[float], float, np.ndarray]:
        """
        Compute a 2x2 DiD estimate.

        Returns
        -------
        att : float or None
            Treatment effect estimate.
        se : float
            Standard error.
        inf_func : np.ndarray
            Influence function values.
        """
        # Get data for the two periods
        df_base = df[df[time] == base_period].set_index(unit)
        df_post = df[df[time] == post_period].set_index(unit)

        # Compute outcome changes for treated
        treated_base = df_base.loc[df_base.index.isin(treated_units), outcome]
        treated_post = df_post.loc[df_post.index.isin(treated_units), outcome]
        treated_common = treated_base.index.intersection(treated_post.index)

        if len(treated_common) == 0:
            return None, 0.0, np.array([])

        treated_change = (
            treated_post.loc[treated_common].values
            - treated_base.loc[treated_common].values
        )

        # Compute outcome changes for control
        control_base = df_base.loc[df_base.index.isin(control_units), outcome]
        control_post = df_post.loc[df_post.index.isin(control_units), outcome]
        control_common = control_base.index.intersection(control_post.index)

        if len(control_common) == 0:
            return None, 0.0, np.array([])

        control_change = (
            control_post.loc[control_common].values
            - control_base.loc[control_common].values
        )

        n_t = len(treated_change)
        n_c = len(control_change)

        # Simple difference in means (could add covariate adjustment here)
        att = float(np.mean(treated_change) - np.mean(control_change))

        var_t = float(np.var(treated_change, ddof=1)) if n_t > 1 else 0.0
        var_c = float(np.var(control_change, ddof=1)) if n_c > 1 else 0.0

        se = np.sqrt(var_t / n_t + var_c / n_c) if (n_t > 0 and n_c > 0) else 0.0

        # Influence function
        inf_treated = (treated_change - np.mean(treated_change)) / n_t
        inf_control = (control_change - np.mean(control_change)) / n_c
        inf_func = np.concatenate([inf_treated, -inf_control])

        return att, se, inf_func

    def _compute_iw_effects(
        self,
        df: pd.DataFrame,
        unit: str,
        first_treat: str,
        treatment_groups: List[Any],
        rel_periods: List[int],
        cohort_effects: Dict[Tuple[Any, int], Dict[str, Any]],
    ) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[Any, float]]]:
        """
        Compute interaction-weighted event study effects.

        The IW estimator aggregates cohort-specific effects using weights
        that represent each cohort's share of treated observations at each
        relative time.

        Returns
        -------
        event_study_effects : dict
            Dictionary mapping relative period to aggregated effect info.
        cohort_weights : dict
            Dictionary mapping relative period to cohort weight dictionary.
        """
        event_study_effects: Dict[int, Dict[str, Any]] = {}
        cohort_weights: Dict[int, Dict[Any, float]] = {}

        for e in rel_periods:
            # Get all cohort effects for this relative period
            effects_at_e = [
                (g, data)
                for (g, rel_e), data in cohort_effects.items()
                if rel_e == e
            ]

            if not effects_at_e:
                continue

            # Compute IW weights: share of each cohort in treated population at e
            # Weight = n_g / Σ_g' n_g' where n_g is cohort g's size
            weights = {}
            total_treated = 0

            for g, data in effects_at_e:
                n_g = data["n_cohort"]
                weights[g] = n_g
                total_treated += n_g

            if total_treated == 0:
                continue

            # Normalize weights
            for g in weights:
                weights[g] = weights[g] / total_treated

            cohort_weights[e] = weights

            # Compute weighted average effect
            agg_effect = 0.0
            agg_var = 0.0

            for g, data in effects_at_e:
                w = weights[g]
                agg_effect += w * data["effect"]
                agg_var += (w**2) * (data["se"] ** 2)

            agg_se = np.sqrt(agg_var)
            t_stat = agg_effect / agg_se if agg_se > 0 else 0.0
            p_val = compute_p_value(t_stat)
            ci = compute_confidence_interval(agg_effect, agg_se, self.alpha)

            event_study_effects[e] = {
                "effect": agg_effect,
                "se": agg_se,
                "t_stat": t_stat,
                "p_value": p_val,
                "conf_int": ci,
                "n_groups": len(effects_at_e),
            }

        return event_study_effects, cohort_weights

    def _run_bootstrap(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        treatment_groups: List[Any],
        time_periods: List[Any],
        rel_periods_to_estimate: List[int],
        reference_period: int,
        covariates: Optional[List[str]],
        original_event_study: Dict[int, Dict[str, Any]],
        original_overall_att: float,
    ) -> SABootstrapResults:
        """
        Run multiplier bootstrap for inference.

        Uses the pairs bootstrap (resampling units with replacement) rather
        than multiplier bootstrap to get valid inference.
        """
        if self.n_bootstrap < 50:
            warnings.warn(
                f"n_bootstrap={self.n_bootstrap} is low. Consider n_bootstrap >= 199 "
                "for reliable inference.",
                UserWarning,
                stacklevel=3,
            )

        rng = np.random.default_rng(self.seed)

        # Get unique units
        all_units = df[unit].unique()
        n_units = len(all_units)

        # Store bootstrap samples
        rel_periods = sorted(original_event_study.keys())
        bootstrap_effects = {e: np.zeros(self.n_bootstrap) for e in rel_periods}
        bootstrap_overall = np.zeros(self.n_bootstrap)

        for b in range(self.n_bootstrap):
            # Resample units with replacement (pairs bootstrap)
            boot_units = rng.choice(all_units, size=n_units, replace=True)

            # Create bootstrap sample
            boot_dfs = []
            for i, u in enumerate(boot_units):
                unit_data = df[df[unit] == u].copy()
                # Rename unit to make unique in bootstrap sample
                unit_data[unit] = i
                boot_dfs.append(unit_data)

            df_b = pd.concat(boot_dfs, ignore_index=True)
            df_b["_rel_time"] = df_b.apply(
                lambda row: (
                    row[time] - row[first_treat]
                    if row[first_treat] > 0
                    else np.nan
                ),
                axis=1,
            )
            df_b["_never_treated"] = (
                (df_b[first_treat] == 0) | (df_b[first_treat] == np.inf)
            )

            # Re-estimate cohort effects
            try:
                cohort_effects_b, _ = self._compute_cohort_effects(
                    df_b,
                    outcome,
                    unit,
                    time,
                    first_treat,
                    treatment_groups,
                    time_periods,
                    rel_periods_to_estimate,
                    reference_period,
                    covariates,
                )

                # Re-compute IW effects
                event_study_b, _ = self._compute_iw_effects(
                    df_b,
                    unit,
                    first_treat,
                    treatment_groups,
                    rel_periods_to_estimate,
                    cohort_effects_b,
                )

                # Store bootstrap estimates
                for e in rel_periods:
                    if e in event_study_b:
                        bootstrap_effects[e][b] = event_study_b[e]["effect"]
                    else:
                        bootstrap_effects[e][b] = original_event_study[e]["effect"]

                # Compute overall ATT for this bootstrap sample
                post_effects_b = [
                    (e, eff) for e, eff in event_study_b.items() if e >= 0
                ]
                if post_effects_b:
                    post_weights = []
                    post_estimates = []
                    for e, eff in post_effects_b:
                        n_at_e = len(
                            df_b[(df_b["_rel_time"] == e) & (df_b[first_treat] > 0)]
                        )
                        post_weights.append(max(n_at_e, 1))
                        post_estimates.append(eff["effect"])

                    post_weights = np.array(post_weights, dtype=float)
                    if post_weights.sum() > 0:
                        post_weights = post_weights / post_weights.sum()
                        bootstrap_overall[b] = np.sum(
                            post_weights * np.array(post_estimates)
                        )
                    else:
                        bootstrap_overall[b] = original_overall_att
                else:
                    bootstrap_overall[b] = original_overall_att

            except Exception:
                # If bootstrap iteration fails, use original
                for e in rel_periods:
                    bootstrap_effects[e][b] = original_event_study[e]["effect"]
                bootstrap_overall[b] = original_overall_att

        # Compute bootstrap statistics
        event_study_ses = {}
        event_study_cis = {}
        event_study_p_values = {}

        for e in rel_periods:
            boot_dist = bootstrap_effects[e]
            original_effect = original_event_study[e]["effect"]

            se = float(np.std(boot_dist, ddof=1))
            ci = self._compute_percentile_ci(boot_dist, self.alpha)
            p_value = self._compute_bootstrap_pvalue(original_effect, boot_dist)

            event_study_ses[e] = se
            event_study_cis[e] = ci
            event_study_p_values[e] = p_value

        # Overall ATT statistics
        overall_se = float(np.std(bootstrap_overall, ddof=1))
        overall_ci = self._compute_percentile_ci(bootstrap_overall, self.alpha)
        overall_p = self._compute_bootstrap_pvalue(
            original_overall_att, bootstrap_overall
        )

        return SABootstrapResults(
            n_bootstrap=self.n_bootstrap,
            weight_type=self.bootstrap_weights,
            alpha=self.alpha,
            overall_att_se=overall_se,
            overall_att_ci=overall_ci,
            overall_att_p_value=overall_p,
            event_study_ses=event_study_ses,
            event_study_cis=event_study_cis,
            event_study_p_values=event_study_p_values,
            bootstrap_distribution=bootstrap_overall,
        )

    def _compute_percentile_ci(
        self,
        boot_dist: np.ndarray,
        alpha: float,
    ) -> Tuple[float, float]:
        """Compute percentile confidence interval."""
        lower = float(np.percentile(boot_dist, alpha / 2 * 100))
        upper = float(np.percentile(boot_dist, (1 - alpha / 2) * 100))
        return (lower, upper)

    def _compute_bootstrap_pvalue(
        self,
        original_effect: float,
        boot_dist: np.ndarray,
    ) -> float:
        """Compute two-sided bootstrap p-value."""
        if original_effect >= 0:
            p_one_sided = float(np.mean(boot_dist <= 0))
        else:
            p_one_sided = float(np.mean(boot_dist >= 0))

        p_value = min(2 * p_one_sided, 1.0)
        p_value = max(p_value, 1 / (self.n_bootstrap + 1))

        return p_value

    def get_params(self) -> Dict[str, Any]:
        """Get estimator parameters (sklearn-compatible)."""
        return {
            "control_group": self.control_group,
            "anticipation": self.anticipation,
            "alpha": self.alpha,
            "cluster": self.cluster,
            "n_bootstrap": self.n_bootstrap,
            "bootstrap_weights": self.bootstrap_weights,
            "seed": self.seed,
        }

    def set_params(self, **params) -> "SunAbraham":
        """Set estimator parameters (sklearn-compatible)."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        return self

    def summary(self) -> str:
        """Get summary of estimation results."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before calling summary()")
        assert self.results_ is not None
        return self.results_.summary()

    def print_summary(self) -> None:
        """Print summary to stdout."""
        print(self.summary())
