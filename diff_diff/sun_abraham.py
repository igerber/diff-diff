"""
Sun-Abraham Interaction-Weighted Estimator for staggered DiD.

Implements the estimator from Sun & Abraham (2021), "Estimating dynamic
treatment effects in event studies with heterogeneous treatment effects",
Journal of Econometrics.

This provides an alternative to Callaway-Sant'Anna using a saturated
regression with cohort × relative-time interactions.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.linalg import LinearRegression, compute_robust_vcov
from diff_diff.results import _get_significance_stars
from diff_diff.utils import (
    compute_confidence_interval,
    compute_p_value,
    within_transform as _within_transform_util,
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
    Results from Sun-Abraham bootstrap inference.

    Attributes
    ----------
    n_bootstrap : int
        Number of bootstrap iterations.
    weight_type : str
        Type of bootstrap used (always "pairs" for pairs bootstrap).
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
    TWFE regression with cohort × relative-time interactions, following
    the methodology in Sun & Abraham (2021).

    The estimation procedure follows three steps:
    1. Run a saturated TWFE regression with cohort × relative-time dummies
    2. Compute cohort shares (weights) at each relative time
    3. Aggregate cohort-specific effects using interaction weights

    This avoids the negative weighting problem of standard TWFE and provides
    consistent event-study estimates under treatment effect heterogeneity.

    Parameters
    ----------
    control_group : str, default="never_treated"
        Which units to use as controls:
        - "never_treated": Use only never-treated units (recommended)
        - "not_yet_treated": Use never-treated and not-yet-treated units
    anticipation : int, default=0
        Number of periods before treatment where effects may occur.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    cluster : str, optional
        Column name for cluster-robust standard errors.
        If None, clusters at the unit level by default.
    n_bootstrap : int, default=0
        Number of bootstrap iterations for inference.
        If 0, uses analytical cluster-robust standard errors.
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

    Y_it = α_i + λ_t + Σ_g Σ_e [δ_{g,e} × 1(G_i=g) × D_{it}^e] + X'γ + ε_it

    where:
    - α_i = unit fixed effects
    - λ_t = time fixed effects
    - G_i = unit i's treatment cohort (first treatment period)
    - D_{it}^e = indicator for being e periods from treatment
    - δ_{g,e} = cohort-specific effect (CATT) at relative time e

    The event-study coefficients are then computed as:

    β_e = Σ_g w_{g,e} × δ_{g,e}

    where w_{g,e} is the share of cohort g in the treated population at
    relative time e (interaction weights).

    Compared to Callaway-Sant'Anna:
    - SA uses saturated regression; CS uses 2x2 DiD comparisons
    - SA can be more efficient when model is correctly specified
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
        seed: Optional[int] = None,
    ):
        if control_group not in ["never_treated", "not_yet_treated"]:
            raise ValueError(
                f"control_group must be 'never_treated' or 'not_yet_treated', "
                f"got '{control_group}'"
            )

        self.control_group = control_group
        self.anticipation = anticipation
        self.alpha = alpha
        self.cluster = cluster
        self.n_bootstrap = n_bootstrap
        self.seed = seed

        self.is_fitted_ = False
        self.results_: Optional[SunAbrahamResults] = None
        self._reference_period = -1  # Will be set during fit

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
        Fit the Sun-Abraham estimator using saturated regression.

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
            List of covariate column names to include in regression.
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

        # Compute relative time for each observation (vectorized)
        df["_rel_time"] = np.where(
            df[first_treat] > 0,
            df[time] - df[first_treat],
            np.nan
        )

        # Identify the range of relative time periods to estimate
        rel_times_by_cohort = {}
        for g in treatment_groups:
            g_times = df[df[first_treat] == g][time].unique()
            rel_times_by_cohort[g] = sorted([t - g for t in g_times])

        # Find all relative time values
        all_rel_times: set = set()
        for g, rel_times in rel_times_by_cohort.items():
            all_rel_times.update(rel_times)

        all_rel_times_sorted = sorted(all_rel_times)

        # Filter to reasonable range
        min_rel = max(min(all_rel_times_sorted), -20)  # cap at -20
        max_rel = min(max(all_rel_times_sorted), 20)   # cap at +20

        # Reference period: last pre-treatment period (typically -1)
        self._reference_period = -1 - self.anticipation

        # Get relative periods to estimate (excluding reference)
        rel_periods_to_estimate = [
            e
            for e in all_rel_times_sorted
            if min_rel <= e <= max_rel and e != self._reference_period
        ]

        # Determine cluster variable
        cluster_var = self.cluster if self.cluster is not None else unit

        # Filter data based on control_group setting
        if self.control_group == "never_treated":
            # Only keep never-treated as controls
            df_reg = df[df["_never_treated"] | (df[first_treat] > 0)].copy()
        else:
            # Keep all units (not_yet_treated will be handled by the regression)
            df_reg = df.copy()

        # Fit saturated regression
        (
            cohort_effects,
            cohort_ses,
            vcov_cohort,
            coef_index_map,
        ) = self._fit_saturated_regression(
            df_reg,
            outcome,
            unit,
            time,
            first_treat,
            treatment_groups,
            rel_periods_to_estimate,
            covariates,
            cluster_var,
        )

        # Compute interaction-weighted event study effects
        event_study_effects, cohort_weights = self._compute_iw_effects(
            df,
            unit,
            first_treat,
            treatment_groups,
            rel_periods_to_estimate,
            cohort_effects,
            cohort_ses,
            vcov_cohort,
            coef_index_map,
        )

        # Compute overall ATT (average of post-treatment effects)
        overall_att, overall_se = self._compute_overall_att(
            df,
            first_treat,
            event_study_effects,
            cohort_effects,
            cohort_weights,
            vcov_cohort,
            coef_index_map,
        )

        overall_t = overall_att / overall_se if overall_se > 0 else 0.0
        overall_p = compute_p_value(overall_t)
        overall_ci = compute_confidence_interval(overall_att, overall_se, self.alpha)

        # Run bootstrap if requested
        bootstrap_results = None
        if self.n_bootstrap > 0:
            bootstrap_results = self._run_bootstrap(
                df=df_reg,
                outcome=outcome,
                unit=unit,
                time=time,
                first_treat=first_treat,
                treatment_groups=treatment_groups,
                rel_periods_to_estimate=rel_periods_to_estimate,
                covariates=covariates,
                cluster_var=cluster_var,
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

        # Convert cohort effects to storage format
        cohort_effects_storage: Dict[Tuple[Any, int], Dict[str, Any]] = {}
        for (g, e), effect in cohort_effects.items():
            weight = cohort_weights.get(e, {}).get(g, 0.0)
            se = cohort_ses.get((g, e), 0.0)
            cohort_effects_storage[(g, e)] = {
                "effect": effect,
                "se": se,
                "weight": weight,
            }

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
            cohort_effects=cohort_effects_storage,
        )

        self.is_fitted_ = True
        return self.results_

    def _fit_saturated_regression(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        treatment_groups: List[Any],
        rel_periods: List[int],
        covariates: Optional[List[str]],
        cluster_var: str,
    ) -> Tuple[
        Dict[Tuple[Any, int], float],
        Dict[Tuple[Any, int], float],
        np.ndarray,
        Dict[Tuple[Any, int], int],
    ]:
        """
        Fit saturated TWFE regression with cohort × relative-time interactions.

        Y_it = α_i + λ_t + Σ_g Σ_e [δ_{g,e} × D_{g,e,it}] + X'γ + ε

        Uses within-transformation for unit fixed effects and time dummies.

        Returns
        -------
        cohort_effects : dict
            Mapping (cohort, rel_period) -> effect estimate δ_{g,e}
        cohort_ses : dict
            Mapping (cohort, rel_period) -> standard error
        vcov : np.ndarray
            Variance-covariance matrix for cohort effects
        coef_index_map : dict
            Mapping (cohort, rel_period) -> index in coefficient vector
        """
        df = df.copy()

        # Create cohort × relative-time interaction dummies
        # Exclude reference period
        # Build all columns at once to avoid fragmentation
        interaction_data = {}
        coef_index_map: Dict[Tuple[Any, int], int] = {}
        idx = 0

        for g in treatment_groups:
            for e in rel_periods:
                col_name = f"_D_{g}_{e}"
                # Indicator: unit is in cohort g AND at relative time e
                indicator = (
                    (df[first_treat] == g) &
                    (df["_rel_time"] == e)
                ).astype(float)

                # Only include if there are observations
                if indicator.sum() > 0:
                    interaction_data[col_name] = indicator.values
                    coef_index_map[(g, e)] = idx
                    idx += 1

        # Add all interaction columns at once
        interaction_cols = list(interaction_data.keys())
        if interaction_data:
            interaction_df = pd.DataFrame(interaction_data, index=df.index)
            df = pd.concat([df, interaction_df], axis=1)

        if len(interaction_cols) == 0:
            raise ValueError(
                "No valid cohort × relative-time interactions found. "
                "Check your data structure."
            )

        # Apply within-transformation for unit and time fixed effects
        variables_to_demean = [outcome] + interaction_cols
        if covariates:
            variables_to_demean.extend(covariates)

        df_demeaned = self._within_transform(df, variables_to_demean, unit, time)

        # Build design matrix
        X_cols = [f"{col}_dm" for col in interaction_cols]
        if covariates:
            X_cols.extend([f"{cov}_dm" for cov in covariates])

        X = df_demeaned[X_cols].values
        y = df_demeaned[f"{outcome}_dm"].values

        # Fit OLS using LinearRegression helper (more stable than manual X'X inverse)
        cluster_ids = df_demeaned[cluster_var].values
        reg = LinearRegression(
            include_intercept=False,  # Already demeaned, no intercept needed
            robust=True,
            cluster_ids=cluster_ids,
        ).fit(X, y)

        coefficients = reg.coefficients_
        vcov = reg.vcov_

        # Extract cohort effects and standard errors using get_inference
        cohort_effects: Dict[Tuple[Any, int], float] = {}
        cohort_ses: Dict[Tuple[Any, int], float] = {}

        n_interactions = len(interaction_cols)
        for (g, e), coef_idx in coef_index_map.items():
            inference = reg.get_inference(coef_idx)
            cohort_effects[(g, e)] = inference.coefficient
            cohort_ses[(g, e)] = inference.se

        # Extract just the vcov for cohort effects (excluding covariates)
        vcov_cohort = vcov[:n_interactions, :n_interactions]

        return cohort_effects, cohort_ses, vcov_cohort, coef_index_map

    def _within_transform(
        self,
        df: pd.DataFrame,
        variables: List[str],
        unit: str,
        time: str,
    ) -> pd.DataFrame:
        """
        Apply two-way within transformation to remove unit and time fixed effects.

        y_it - y_i. - y_.t + y_..
        """
        return _within_transform_util(df, variables, unit, time, suffix="_dm")

    def _compute_iw_effects(
        self,
        df: pd.DataFrame,
        unit: str,
        first_treat: str,
        treatment_groups: List[Any],
        rel_periods: List[int],
        cohort_effects: Dict[Tuple[Any, int], float],
        cohort_ses: Dict[Tuple[Any, int], float],
        vcov_cohort: np.ndarray,
        coef_index_map: Dict[Tuple[Any, int], int],
    ) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[Any, float]]]:
        """
        Compute interaction-weighted event study effects.

        β_e = Σ_g w_{g,e} × δ_{g,e}

        where w_{g,e} is the share of cohort g among treated units at relative time e.

        Returns
        -------
        event_study_effects : dict
            Dictionary mapping relative period to aggregated effect info.
        cohort_weights : dict
            Dictionary mapping relative period to cohort weight dictionary.
        """
        event_study_effects: Dict[int, Dict[str, Any]] = {}
        cohort_weights: Dict[int, Dict[Any, float]] = {}

        # Get cohort sizes
        unit_cohorts = df.groupby(unit)[first_treat].first()
        cohort_sizes = unit_cohorts[unit_cohorts > 0].value_counts().to_dict()

        for e in rel_periods:
            # Get cohorts that have observations at this relative time
            cohorts_at_e = [
                g for g in treatment_groups
                if (g, e) in cohort_effects
            ]

            if not cohorts_at_e:
                continue

            # Compute IW weights: share of each cohort among those observed at e
            weights = {}
            total_size = 0
            for g in cohorts_at_e:
                n_g = cohort_sizes.get(g, 0)
                weights[g] = n_g
                total_size += n_g

            if total_size == 0:
                continue

            # Normalize weights
            for g in weights:
                weights[g] = weights[g] / total_size

            cohort_weights[e] = weights

            # Compute weighted average effect
            agg_effect = 0.0
            for g in cohorts_at_e:
                w = weights[g]
                agg_effect += w * cohort_effects[(g, e)]

            # Compute SE using delta method with vcov
            # Var(β_e) = w' Σ w where w is weight vector and Σ is vcov submatrix
            indices = [coef_index_map[(g, e)] for g in cohorts_at_e]
            weight_vec = np.array([weights[g] for g in cohorts_at_e])
            vcov_subset = vcov_cohort[np.ix_(indices, indices)]
            agg_var = float(weight_vec @ vcov_subset @ weight_vec)
            agg_se = np.sqrt(max(agg_var, 0))

            t_stat = agg_effect / agg_se if agg_se > 0 else 0.0
            p_val = compute_p_value(t_stat)
            ci = compute_confidence_interval(agg_effect, agg_se, self.alpha)

            event_study_effects[e] = {
                "effect": agg_effect,
                "se": agg_se,
                "t_stat": t_stat,
                "p_value": p_val,
                "conf_int": ci,
                "n_groups": len(cohorts_at_e),
            }

        return event_study_effects, cohort_weights

    def _compute_overall_att(
        self,
        df: pd.DataFrame,
        first_treat: str,
        event_study_effects: Dict[int, Dict[str, Any]],
        cohort_effects: Dict[Tuple[Any, int], float],
        cohort_weights: Dict[int, Dict[Any, float]],
        vcov_cohort: np.ndarray,
        coef_index_map: Dict[Tuple[Any, int], int],
    ) -> Tuple[float, float]:
        """
        Compute overall ATT as weighted average of post-treatment effects.

        Returns (att, se) tuple.
        """
        post_effects = [
            (e, eff)
            for e, eff in event_study_effects.items()
            if e >= 0
        ]

        if not post_effects:
            return 0.0, 0.0

        # Weight by number of treated observations at each relative time
        post_weights = []
        post_estimates = []

        for e, eff in post_effects:
            n_at_e = len(df[(df["_rel_time"] == e) & (df[first_treat] > 0)])
            post_weights.append(max(n_at_e, 1))
            post_estimates.append(eff["effect"])

        post_weights = np.array(post_weights, dtype=float)
        post_weights = post_weights / post_weights.sum()

        overall_att = float(np.sum(post_weights * np.array(post_estimates)))

        # Compute SE using delta method
        # Need to trace back through the full weighting scheme
        # ATT = Σ_e w_e × β_e = Σ_e w_e × Σ_g w_{g,e} × δ_{g,e}
        # Collect all (g, e) pairs and their overall weights
        overall_weights_by_coef: Dict[Tuple[Any, int], float] = {}

        for i, (e, _) in enumerate(post_effects):
            period_weight = post_weights[i]
            if e in cohort_weights:
                for g, cw in cohort_weights[e].items():
                    key = (g, e)
                    if key in coef_index_map:
                        if key not in overall_weights_by_coef:
                            overall_weights_by_coef[key] = 0.0
                        overall_weights_by_coef[key] += period_weight * cw

        if not overall_weights_by_coef:
            # Fallback to simple variance calculation
            overall_var = float(
                np.sum((post_weights ** 2) * np.array([eff["se"] ** 2 for _, eff in post_effects]))
            )
            return overall_att, np.sqrt(overall_var)

        # Build full weight vector and compute variance
        indices = [coef_index_map[key] for key in overall_weights_by_coef.keys()]
        weight_vec = np.array(list(overall_weights_by_coef.values()))
        vcov_subset = vcov_cohort[np.ix_(indices, indices)]
        overall_var = float(weight_vec @ vcov_subset @ weight_vec)
        overall_se = np.sqrt(max(overall_var, 0))

        return overall_att, overall_se

    def _run_bootstrap(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        treatment_groups: List[Any],
        rel_periods_to_estimate: List[int],
        covariates: Optional[List[str]],
        cluster_var: str,
        original_event_study: Dict[int, Dict[str, Any]],
        original_overall_att: float,
    ) -> SABootstrapResults:
        """
        Run pairs bootstrap for inference.

        Resamples units with replacement and re-estimates the full model.
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

            # Create bootstrap sample efficiently
            # Build index array for all selected units
            boot_indices = np.concatenate([
                df.index[df[unit] == u].values for u in boot_units
            ])
            df_b = df.iloc[boot_indices].copy()

            # Reassign unique unit IDs for bootstrap sample
            # Each resampled unit gets a unique ID
            new_unit_ids = []
            current_id = 0
            for u in boot_units:
                unit_rows = df[df[unit] == u]
                for _ in range(len(unit_rows)):
                    new_unit_ids.append(current_id)
                current_id += 1
            df_b[unit] = new_unit_ids[:len(df_b)]

            # Recompute relative time (vectorized)
            df_b["_rel_time"] = np.where(
                df_b[first_treat] > 0,
                df_b[time] - df_b[first_treat],
                np.nan
            )
            df_b["_never_treated"] = (
                (df_b[first_treat] == 0) | (df_b[first_treat] == np.inf)
            )

            try:
                # Re-estimate saturated regression
                (
                    cohort_effects_b,
                    cohort_ses_b,
                    vcov_b,
                    coef_map_b,
                ) = self._fit_saturated_regression(
                    df_b,
                    outcome,
                    unit,
                    time,
                    first_treat,
                    treatment_groups,
                    rel_periods_to_estimate,
                    covariates,
                    cluster_var,
                )

                # Compute IW effects for this bootstrap sample
                event_study_b, cohort_weights_b = self._compute_iw_effects(
                    df_b,
                    unit,
                    first_treat,
                    treatment_groups,
                    rel_periods_to_estimate,
                    cohort_effects_b,
                    cohort_ses_b,
                    vcov_b,
                    coef_map_b,
                )

                # Store bootstrap estimates
                for e in rel_periods:
                    if e in event_study_b:
                        bootstrap_effects[e][b] = event_study_b[e]["effect"]
                    else:
                        bootstrap_effects[e][b] = original_event_study[e]["effect"]

                # Compute overall ATT for this bootstrap sample
                overall_b, _ = self._compute_overall_att(
                    df_b,
                    first_treat,
                    event_study_b,
                    cohort_effects_b,
                    cohort_weights_b,
                    vcov_b,
                    coef_map_b,
                )
                bootstrap_overall[b] = overall_b

            except (ValueError, np.linalg.LinAlgError) as exc:
                # If bootstrap iteration fails, use original
                warnings.warn(
                    f"Bootstrap iteration {b} failed: {exc}. Using original estimate.",
                    UserWarning,
                    stacklevel=2,
                )
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
            weight_type="pairs",
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
