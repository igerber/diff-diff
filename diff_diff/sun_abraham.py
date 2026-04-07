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

from diff_diff.bootstrap_utils import compute_effect_bootstrap_stats
from diff_diff.linalg import LinearRegression
from diff_diff.results import _format_survey_block, _get_significance_stars
from diff_diff.utils import (
    safe_inference,
)
from diff_diff.utils import (
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
    # Survey design metadata (SurveyMetadata instance from diff_diff.survey)
    survey_metadata: Optional[Any] = field(default=None)

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
            raise ValueError(f"Unknown level: {level}. Use 'event_study' or 'cohort'.")

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
    rank_deficient_action : str, default="warn"
        Action when design matrix is rank-deficient (linearly dependent columns):
        - "warn": Issue warning and drop linearly dependent columns (default)
        - "error": Raise ValueError
        - "silent": Drop columns silently without warning

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
        rank_deficient_action: str = "warn",
    ):
        if control_group not in ["never_treated", "not_yet_treated"]:
            raise ValueError(
                f"control_group must be 'never_treated' or 'not_yet_treated', "
                f"got '{control_group}'"
            )

        if rank_deficient_action not in ["warn", "error", "silent"]:
            raise ValueError(
                f"rank_deficient_action must be 'warn', 'error', or 'silent', "
                f"got '{rank_deficient_action}'"
            )

        self.control_group = control_group
        self.anticipation = anticipation
        self.alpha = alpha
        self.cluster = cluster
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.rank_deficient_action = rank_deficient_action

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
        survey_design: object = None,
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
        survey_design : SurveyDesign, optional
            Survey design specification for design-based inference.
            Supports weighted estimation and Taylor series linearization
            variance with strata, PSU, and FPC.

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

        # Resolve survey design if provided
        from diff_diff.survey import (
            _resolve_effective_cluster,
            _resolve_survey_for_fit,
            _validate_unit_constant_survey,
        )

        resolved_survey, survey_weights, survey_weight_type, survey_metadata = (
            _resolve_survey_for_fit(survey_design, data, "analytical")
        )

        # Validate survey columns are constant within units (required for
        # unit-level collapse in Rao-Wu bootstrap)
        if resolved_survey is not None:
            _validate_unit_constant_survey(data, unit, survey_design)

        # Reject replicate-weight designs — SunAbraham's weighted
        # within-transformation bakes survey weights into X and y, so
        # replicate refits on the already-transformed design are incorrect.
        # Full estimator-level replicate refits are not yet implemented.
        if resolved_survey is not None and resolved_survey.uses_replicate_variance:
            raise NotImplementedError(
                "SunAbraham does not yet support replicate-weight survey designs. "
                "The weighted within-transformation must be recomputed for each "
                "replicate, which requires estimator-level replicate refits. "
                "Use a TSL-based survey design (strata/psu/fpc) instead."
            )

        # Bootstrap + survey supported via Rao-Wu rescaled bootstrap.
        # Determine Rao-Wu eligibility from the *original* survey_design
        # (before cluster-as-PSU injection which adds PSU to weights-only designs).
        _use_rao_wu = False
        if survey_design is not None and resolved_survey is not None:
            _has_explicit_strata = getattr(survey_design, "strata", None) is not None
            _has_explicit_psu = getattr(survey_design, "psu", None) is not None
            _has_explicit_fpc = getattr(survey_design, "fpc", None) is not None
            if _has_explicit_strata or _has_explicit_psu or _has_explicit_fpc:
                _use_rao_wu = True

        # Create working copy
        df = data.copy()

        # Ensure numeric types
        df[time] = pd.to_numeric(df[time])
        df[first_treat] = pd.to_numeric(df[first_treat])

        # Never-treated indicator (must precede treatment_groups to exclude np.inf)
        df["_never_treated"] = (df[first_treat] == 0) | (df[first_treat] == np.inf)
        # Normalize np.inf → 0 so all downstream `> 0` checks exclude never-treated
        df.loc[df[first_treat] == np.inf, first_treat] = 0

        # Identify groups and time periods
        time_periods = sorted(df[time].unique())
        treatment_groups = sorted([g for g in df[first_treat].unique() if g > 0])

        # Get unique units
        unit_info = (
            df.groupby(unit).agg({first_treat: "first", "_never_treated": "first"}).reset_index()
        )

        n_treated_units = int((unit_info[first_treat] > 0).sum())
        n_control_units = int((unit_info["_never_treated"]).sum())

        if n_control_units == 0:
            raise ValueError("No never-treated units found. Check 'first_treat' column.")

        if len(treatment_groups) == 0:
            raise ValueError("No treated units found. Check 'first_treat' column.")

        # Compute relative time for each observation (vectorized)
        df["_rel_time"] = np.where(df[first_treat] > 0, df[time] - df[first_treat], np.nan)

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

        # Use full range of relative times (no artificial truncation, matches R's fixest::sunab())
        min_rel = min(all_rel_times_sorted)
        max_rel = max(all_rel_times_sorted)

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

        # Resolve effective cluster and inject cluster-as-PSU
        cluster_ids_raw = df_reg[cluster_var].values if cluster_var in df_reg.columns else None
        effective_cluster_ids = _resolve_effective_cluster(
            resolved_survey, cluster_ids_raw, cluster_var if self.cluster is not None else None
        )
        if resolved_survey is not None and effective_cluster_ids is not None:
            from diff_diff.survey import _inject_cluster_as_psu, compute_survey_metadata

            resolved_survey = _inject_cluster_as_psu(resolved_survey, effective_cluster_ids)
            if resolved_survey.psu is not None and survey_metadata is not None:
                raw_w = (
                    data[survey_design.weights].values.astype(np.float64)
                    if survey_design.weights
                    else np.ones(len(data), dtype=np.float64)
                )
                survey_metadata = compute_survey_metadata(resolved_survey, raw_w)

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
            survey_weights=survey_weights,
            survey_weight_type=survey_weight_type,
            resolved_survey=resolved_survey,
        )

        # Resolve survey weight column name for cohort aggregation
        survey_weight_col = (
            survey_design.weights
            if survey_design is not None
            and hasattr(survey_design, "weights")
            and survey_design.weights
            else None
        )

        # Survey degrees of freedom for t-distribution inference
        _sa_survey_df = (
            max(survey_metadata.df_survey, 1)
            if survey_metadata is not None and survey_metadata.df_survey is not None
            else None
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
            survey_weight_col=survey_weight_col,
            survey_df=_sa_survey_df,
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
            survey_weight_col=survey_weight_col,
        )

        overall_t, overall_p, overall_ci = safe_inference(
            overall_att, overall_se, alpha=self.alpha, df=_sa_survey_df
        )

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
                resolved_survey=resolved_survey,
                survey_weights=survey_weights,
                survey_weight_type=survey_weight_type,
                survey_weight_col=survey_weight_col,
                use_rao_wu=_use_rao_wu,
            )

            # Update results with bootstrap inference
            overall_se = bootstrap_results.overall_att_se
            overall_t = safe_inference(overall_att, overall_se, alpha=self.alpha)[0]
            overall_p = bootstrap_results.overall_att_p_value
            overall_ci = bootstrap_results.overall_att_ci

            # Update event study effects
            for e in event_study_effects:
                if e in bootstrap_results.event_study_ses:
                    event_study_effects[e]["se"] = bootstrap_results.event_study_ses[e]
                    event_study_effects[e]["conf_int"] = bootstrap_results.event_study_cis[e]
                    event_study_effects[e]["p_value"] = bootstrap_results.event_study_p_values[e]
                    eff_val = event_study_effects[e]["effect"]
                    se_val = event_study_effects[e]["se"]
                    event_study_effects[e]["t_stat"] = safe_inference(
                        eff_val, se_val, alpha=self.alpha
                    )[0]

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
            survey_metadata=survey_metadata,
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
        survey_weights: Optional[np.ndarray] = None,
        survey_weight_type: str = "pweight",
        resolved_survey: object = None,
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
                indicator = ((df[first_treat] == g) & (df["_rel_time"] == e)).astype(float)

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
                "No valid cohort × relative-time interactions found. " "Check your data structure."
            )

        # Apply within-transformation for unit and time fixed effects
        variables_to_demean = [outcome] + interaction_cols
        if covariates:
            variables_to_demean.extend(covariates)

        df_demeaned = _within_transform_util(
            df, variables_to_demean, unit, time, suffix="_dm", weights=survey_weights
        )

        # Build design matrix
        X_cols = [f"{col}_dm" for col in interaction_cols]
        if covariates:
            X_cols.extend([f"{cov}_dm" for cov in covariates])

        X = df_demeaned[X_cols].values
        y = df_demeaned[f"{outcome}_dm"].values

        # Fit OLS using LinearRegression helper (more stable than manual X'X inverse)
        cluster_ids = df_demeaned[cluster_var].values

        # Degrees of freedom adjustment for absorbed unit and time fixed effects
        n_units_fe = df[unit].nunique()
        n_times_fe = df[time].nunique()
        df_adj = n_units_fe + n_times_fe - 1

        reg = LinearRegression(
            include_intercept=False,  # Already demeaned, no intercept needed
            robust=True,
            cluster_ids=cluster_ids,
            rank_deficient_action=self.rank_deficient_action,
            weights=survey_weights,
            weight_type=survey_weight_type,
            survey_design=resolved_survey,
        ).fit(X, y, df_adjustment=df_adj)

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
        assert vcov is not None
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
        survey_weight_col: Optional[str] = None,
        survey_df: Optional[int] = None,
    ) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[Any, float]]]:
        """
        Compute interaction-weighted event study effects.

        β_e = Σ_g w_{g,e} × δ_{g,e}

        where w_{g,e} = n_{g,e} / Σ_g n_{g,e} is the share of observations from cohort g
        at event-time e among all treated observations at that event-time.

        When survey weights are provided, n_{g,e} is the survey-weighted mass
        (sum of weights) rather than raw observation counts, so the estimand
        reflects the survey-weighted cohort composition.

        Returns
        -------
        event_study_effects : dict
            Dictionary mapping relative period to aggregated effect info.
        cohort_weights : dict
            Dictionary mapping relative period to cohort weight dictionary.
        """
        event_study_effects: Dict[int, Dict[str, Any]] = {}
        cohort_weights: Dict[int, Dict[Any, float]] = {}

        # Pre-compute per-event-time observation mass: n_{g,e}
        # With survey weights, use weighted sum; otherwise raw counts.
        treated_mask = df[first_treat] > 0
        if survey_weight_col is not None and survey_weight_col in df.columns:
            event_time_counts = (
                df[treated_mask].groupby([first_treat, "_rel_time"])[survey_weight_col].sum()
            )
        else:
            event_time_counts = df[treated_mask].groupby([first_treat, "_rel_time"]).size()

        for e in rel_periods:
            # Get cohorts that have observations at this relative time
            cohorts_at_e = [g for g in treatment_groups if (g, e) in cohort_effects]

            if not cohorts_at_e:
                continue

            # Compute IW weights: n_{g,e} / Σ_g n_{g,e}
            weights = {}
            total_size = 0
            for g in cohorts_at_e:
                n_g_e = event_time_counts.get((g, e), 0)
                weights[g] = n_g_e
                total_size += n_g_e

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

            t_stat, p_val, ci = safe_inference(agg_effect, agg_se, alpha=self.alpha, df=survey_df)

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
        survey_weight_col: Optional[str] = None,
    ) -> Tuple[float, float]:
        """
        Compute overall ATT as weighted average of post-treatment effects.

        When survey weights are provided, the per-period weights use
        survey-weighted mass rather than raw observation counts.

        Returns (att, se) tuple.
        """
        post_effects = [(e, eff) for e, eff in event_study_effects.items() if e >= 0]

        if not post_effects:
            return np.nan, np.nan

        # Weight by (survey-weighted) mass of treated observations at each relative time
        post_weights = []
        post_estimates = []

        for e, eff in post_effects:
            mask = (df["_rel_time"] == e) & (df[first_treat] > 0)
            if survey_weight_col is not None and survey_weight_col in df.columns:
                # No floor for survey weights — valid masses can be < 1
                n_at_e = df.loc[mask, survey_weight_col].sum()
                post_weights.append(n_at_e if n_at_e > 0 else 0.0)
            else:
                n_at_e = len(df[mask])
                post_weights.append(max(n_at_e, 1))
            post_estimates.append(eff["effect"])

        post_weights_arr = np.array(post_weights, dtype=float)
        post_weights_arr = post_weights_arr / post_weights_arr.sum()

        overall_att = float(np.sum(post_weights_arr * np.array(post_estimates)))

        # Compute SE using delta method
        # Need to trace back through the full weighting scheme
        # ATT = Σ_e w_e × β_e = Σ_e w_e × Σ_g w_{g,e} × δ_{g,e}
        # Collect all (g, e) pairs and their overall weights
        overall_weights_by_coef: Dict[Tuple[Any, int], float] = {}

        for i, (e, _) in enumerate(post_effects):
            period_weight = post_weights_arr[i]
            if e in cohort_weights:
                for g, cw in cohort_weights[e].items():
                    key = (g, e)
                    if key in coef_index_map:
                        if key not in overall_weights_by_coef:
                            overall_weights_by_coef[key] = 0.0
                        overall_weights_by_coef[key] += period_weight * cw

        if not overall_weights_by_coef:
            # Fallback to simplified variance that ignores covariances between periods
            warnings.warn(
                "Could not construct full weight vector for overall ATT SE. "
                "Using simplified variance that ignores covariances between periods.",
                UserWarning,
                stacklevel=2,
            )
            overall_var = float(
                np.sum(
                    (post_weights_arr**2) * np.array([eff["se"] ** 2 for _, eff in post_effects])
                )
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
        resolved_survey: object = None,
        survey_weights: Optional[np.ndarray] = None,
        survey_weight_type: str = "pweight",
        survey_weight_col: Optional[str] = None,
        use_rao_wu: bool = False,
    ) -> SABootstrapResults:
        """
        Run bootstrap for inference.

        When use_rao_wu is True (survey design with explicit strata/PSU/FPC),
        uses Rao-Wu rescaled bootstrap (weight perturbation). Otherwise, uses
        pairs bootstrap (resampling units with replacement).
        """
        if self.n_bootstrap < 50:
            warnings.warn(
                f"n_bootstrap={self.n_bootstrap} is low. Consider n_bootstrap >= 199 "
                "for reliable inference.",
                UserWarning,
                stacklevel=3,
            )

        rng = np.random.default_rng(self.seed)

        if use_rao_wu:
            return self._run_rao_wu_bootstrap(
                df=df,
                outcome=outcome,
                unit=unit,
                time=time,
                first_treat=first_treat,
                treatment_groups=treatment_groups,
                rel_periods_to_estimate=rel_periods_to_estimate,
                covariates=covariates,
                cluster_var=cluster_var,
                original_event_study=original_event_study,
                original_overall_att=original_overall_att,
                resolved_survey=resolved_survey,
                survey_weight_type=survey_weight_type,
                survey_weight_col=survey_weight_col,
                rng=rng,
            )

        # --- Pairs bootstrap (non-survey or weights-only survey) ---

        # Get unique units
        all_units = df[unit].unique()
        n_units = len(all_units)

        # Pre-compute unit -> row indices mapping (avoids repeated boolean scans)
        unit_row_indices = {u: df.index[df[unit] == u].values for u in all_units}
        unit_row_counts = {u: len(idx) for u, idx in unit_row_indices.items()}

        # Store bootstrap samples
        rel_periods = sorted(original_event_study.keys())
        bootstrap_effects = {e: np.zeros(self.n_bootstrap) for e in rel_periods}
        bootstrap_overall = np.zeros(self.n_bootstrap)

        for b in range(self.n_bootstrap):
            # Resample units with replacement (pairs bootstrap)
            boot_units = rng.choice(all_units, size=n_units, replace=True)

            # Create bootstrap sample using pre-computed index mapping
            boot_indices = np.concatenate([unit_row_indices[u] for u in boot_units])
            df_b = df.iloc[boot_indices].copy()

            # Reassign unique unit IDs (vectorized)
            rows_per_unit = np.array([unit_row_counts[u] for u in boot_units])
            df_b[unit] = np.repeat(np.arange(n_units), rows_per_unit)

            # Recompute relative time (vectorized)
            df_b["_rel_time"] = np.where(
                df_b[first_treat] > 0, df_b[time] - df_b[first_treat], np.nan
            )
            # np.inf was normalized to 0 in fit(), so the np.inf check is defensive only
            df_b["_never_treated"] = (df_b[first_treat] == 0) | (df_b[first_treat] == np.inf)

            try:
                # Extract survey weights from resampled data if present
                boot_survey_weights = None
                if survey_weight_col is not None and survey_weight_col in df_b.columns:
                    boot_survey_weights = df_b[survey_weight_col].values

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
                    survey_weights=boot_survey_weights,
                    survey_weight_type=survey_weight_type,
                    resolved_survey=None,  # Use explicit weights, not stale design
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
                    survey_weight_col=survey_weight_col,
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
                    survey_weight_col=survey_weight_col,
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
            se, ci, p_value = compute_effect_bootstrap_stats(
                original_effect,
                boot_dist,
                alpha=self.alpha,
                context=f"event study e={e}",
            )
            event_study_ses[e] = se
            event_study_cis[e] = ci
            event_study_p_values[e] = p_value

        # Overall ATT statistics
        overall_se, overall_ci, overall_p = compute_effect_bootstrap_stats(
            original_overall_att,
            bootstrap_overall,
            alpha=self.alpha,
            context="overall ATT",
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

    def _run_rao_wu_bootstrap(
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
        resolved_survey: object,
        survey_weight_type: str,
        survey_weight_col: Optional[str],
        rng: np.random.Generator,
    ) -> SABootstrapResults:
        """
        Run Rao-Wu rescaled bootstrap for survey-aware inference.

        Instead of physically resampling units, each iteration generates
        rescaled observation weights via Rao-Wu (1988) weight perturbation.
        The rescaled weights feed into the existing WLS regression path.
        """
        from diff_diff.bootstrap_utils import generate_rao_wu_weights
        from diff_diff.survey import ResolvedSurveyDesign

        # Column name for rescaled weights in the bootstrap DataFrame
        _rw_col = "__rw_boot_weight"

        # Collapse survey design to unit level so Rao-Wu respects panel
        # structure: each unit gets one set of weights regardless of how
        # many time periods it has.  Without this, when there is no
        # explicit PSU, generate_rao_wu_weights treats each observation as
        # its own PSU and different obs of the same unit can get different
        # weights, breaking panel semantics.
        all_units = df[unit].unique()

        weights_unit = (
            pd.Series(resolved_survey.weights, index=df.index)
            .groupby(df[unit])
            .first()
            .reindex(all_units)
            .values
            .astype(np.float64)
        )

        strata_unit = None
        if resolved_survey.strata is not None:
            strata_unit = (
                pd.Series(resolved_survey.strata, index=df.index)
                .groupby(df[unit])
                .first()
                .reindex(all_units)
                .values
            )

        psu_unit = None
        if resolved_survey.psu is not None:
            psu_unit = (
                pd.Series(resolved_survey.psu, index=df.index)
                .groupby(df[unit])
                .first()
                .reindex(all_units)
                .values
            )

        fpc_unit = None
        if resolved_survey.fpc is not None:
            fpc_unit = (
                pd.Series(resolved_survey.fpc, index=df.index)
                .groupby(df[unit])
                .first()
                .reindex(all_units)
                .values
            )

        unit_resolved = ResolvedSurveyDesign(
            weights=weights_unit,
            weight_type=resolved_survey.weight_type,
            strata=strata_unit,
            psu=psu_unit,
            fpc=fpc_unit,
            n_strata=resolved_survey.n_strata,
            n_psu=resolved_survey.n_psu,
            lonely_psu=resolved_survey.lonely_psu,
        )

        # Build unit -> row indices mapping for expanding unit-level weights
        unit_to_rows = {u: df.index[df[unit] == u].values for u in all_units}
        unit_order = {u: i for i, u in enumerate(all_units)}

        # Store bootstrap samples
        rel_periods = sorted(original_event_study.keys())
        bootstrap_effects = {e: np.full(self.n_bootstrap, np.nan) for e in rel_periods}
        bootstrap_overall = np.full(self.n_bootstrap, np.nan)

        for b in range(self.n_bootstrap):
            try:
                # Generate Rao-Wu rescaled weights at unit level
                unit_boot_weights = generate_rao_wu_weights(unit_resolved, rng)

                # Expand unit-level weights to observation level
                boot_weights = np.empty(len(df), dtype=np.float64)
                for u, idx in unit_to_rows.items():
                    boot_weights[idx] = unit_boot_weights[unit_order[u]]

                # Drop observations with zero weight (PSUs not drawn in this
                # iteration) to avoid NaN/Inf in within-transformation.
                positive_mask = boot_weights > 0
                if positive_mask.sum() < 2:
                    # Too few observations with positive weight
                    raise ValueError("Rao-Wu iteration produced < 2 positive weights")

                df_b = df[positive_mask].reset_index(drop=True)
                boot_weights_b = boot_weights[positive_mask]
                df_b[_rw_col] = boot_weights_b

                # Verify we still have both treated and control observations
                has_treated = (df_b[first_treat] > 0).any()
                has_control = ((df_b[first_treat] == 0) | (df_b[first_treat] == np.inf)).any()
                if not has_treated or not has_control:
                    raise ValueError("Rao-Wu iteration dropped all treated or control units")

                # Re-estimate saturated regression with rescaled weights.
                # Pass resolved_survey=None since inference comes from the
                # bootstrap distribution, not from within-iteration vcov.
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
                    survey_weights=boot_weights_b,
                    survey_weight_type=survey_weight_type,
                    resolved_survey=None,
                )

                # Compute IW effects using rescaled weights for cohort shares
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
                    survey_weight_col=_rw_col,
                )

                # Store bootstrap estimates
                for e in rel_periods:
                    if e in event_study_b:
                        bootstrap_effects[e][b] = event_study_b[e]["effect"]
                    else:
                        bootstrap_effects[e][b] = original_event_study[e]["effect"]

                # Compute overall ATT using rescaled weights
                overall_b, _ = self._compute_overall_att(
                    df_b,
                    first_treat,
                    event_study_b,
                    cohort_effects_b,
                    cohort_weights_b,
                    vcov_b,
                    coef_map_b,
                    survey_weight_col=_rw_col,
                )
                bootstrap_overall[b] = overall_b

            except (ValueError, np.linalg.LinAlgError) as exc:
                # Failed draws stored as NaN (not original estimate) to avoid
                # shrinking bootstrap dispersion.  compute_effect_bootstrap_stats
                # handles NaN draws via nanstd.
                warnings.warn(
                    f"Bootstrap iteration {b} failed: {exc}. Storing NaN.",
                    UserWarning,
                    stacklevel=2,
                )
                for e in rel_periods:
                    bootstrap_effects[e][b] = np.nan
                bootstrap_overall[b] = np.nan

        # Compute bootstrap statistics
        event_study_ses = {}
        event_study_cis = {}
        event_study_p_values = {}

        for e in rel_periods:
            boot_dist = bootstrap_effects[e]
            original_effect = original_event_study[e]["effect"]
            se, ci, p_value = compute_effect_bootstrap_stats(
                original_effect,
                boot_dist,
                alpha=self.alpha,
                context=f"event study e={e}",
            )
            event_study_ses[e] = se
            event_study_cis[e] = ci
            event_study_p_values[e] = p_value

        # Overall ATT statistics
        overall_se, overall_ci, overall_p = compute_effect_bootstrap_stats(
            original_overall_att,
            bootstrap_overall,
            alpha=self.alpha,
            context="overall ATT",
        )

        return SABootstrapResults(
            n_bootstrap=self.n_bootstrap,
            weight_type="rao_wu",
            alpha=self.alpha,
            overall_att_se=overall_se,
            overall_att_ci=overall_ci,
            overall_att_p_value=overall_p,
            event_study_ses=event_study_ses,
            event_study_cis=event_study_cis,
            event_study_p_values=event_study_p_values,
            bootstrap_distribution=bootstrap_overall,
        )

    def get_params(self) -> Dict[str, Any]:
        """Get estimator parameters (sklearn-compatible)."""
        return {
            "control_group": self.control_group,
            "anticipation": self.anticipation,
            "alpha": self.alpha,
            "cluster": self.cluster,
            "n_bootstrap": self.n_bootstrap,
            "seed": self.seed,
            "rank_deficient_action": self.rank_deficient_action,
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
