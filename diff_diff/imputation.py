"""
Borusyak-Jaravel-Spiess (2024) Imputation DiD Estimator.

Implements the efficient imputation estimator for staggered
Difference-in-Differences from Borusyak, Jaravel & Spiess (2024),
"Revisiting Event-Study Designs: Robust and Efficient Estimation",
Review of Economic Studies.

The estimator:
1. Runs OLS on untreated observations to estimate unit + time fixed effects
2. Imputes counterfactual Y(0) for treated observations
3. Aggregates imputed treatment effects with researcher-chosen weights

Inference uses the conservative clustered variance estimator (Theorem 3).
"""

import warnings
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.sparse.linalg import spsolve

from diff_diff.imputation_bootstrap import ImputationDiDBootstrapMixin, _compute_target_weights
from diff_diff.imputation_results import (  # noqa: F401 (re-export)
    ImputationBootstrapResults,
    ImputationDiDResults,
)
from diff_diff.linalg import solve_ols
from diff_diff.utils import safe_inference

# =============================================================================
# Main Estimator
# =============================================================================


class ImputationDiD(ImputationDiDBootstrapMixin):
    """
    Borusyak-Jaravel-Spiess (2024) imputation DiD estimator.

    This is the efficient estimator for staggered Difference-in-Differences
    under parallel trends. It produces shorter confidence intervals than
    Callaway-Sant'Anna (~50% shorter) and Sun-Abraham (2-3.5x shorter)
    under homogeneous treatment effects.

    The estimation procedure:
    1. Run OLS on untreated observations to estimate unit + time fixed effects
    2. Impute counterfactual Y(0) for treated observations
    3. Aggregate imputed treatment effects with researcher-chosen weights

    Inference uses the conservative clustered variance estimator from Theorem 3
    of the paper.

    Parameters
    ----------
    anticipation : int, default=0
        Number of periods before treatment where effects may occur.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    cluster : str, optional
        Column name for cluster-robust standard errors.
        If None, clusters at the unit level by default.
    n_bootstrap : int, default=0
        Number of bootstrap iterations. If 0, uses analytical inference
        (conservative variance from Theorem 3).
    bootstrap_weights : str, default="rademacher"
        Type of bootstrap weights: "rademacher", "mammen", or "webb".
    seed : int, optional
        Random seed for reproducibility.
    rank_deficient_action : str, default="warn"
        Action when design matrix is rank-deficient:
        - "warn": Issue warning and drop linearly dependent columns
        - "error": Raise ValueError
        - "silent": Drop columns silently
    horizon_max : int, optional
        Maximum event-study horizon. If set, event study effects are only
        computed for |h| <= horizon_max.
    aux_partition : str, default="cohort_horizon"
        Controls the auxiliary model partition for Theorem 3 variance:
        - "cohort_horizon": Groups by cohort x relative time (tightest SEs)
        - "cohort": Groups by cohort only (more conservative)
        - "horizon": Groups by relative time only (more conservative)

    Attributes
    ----------
    results_ : ImputationDiDResults
        Estimation results after calling fit().
    is_fitted_ : bool
        Whether the model has been fitted.

    Examples
    --------
    Basic usage:

    >>> from diff_diff import ImputationDiD, generate_staggered_data
    >>> data = generate_staggered_data(n_units=200, seed=42)
    >>> est = ImputationDiD()
    >>> results = est.fit(data, outcome='outcome', unit='unit',
    ...                   time='time', first_treat='first_treat')
    >>> results.print_summary()

    With event study:

    >>> est = ImputationDiD()
    >>> results = est.fit(data, outcome='outcome', unit='unit',
    ...                   time='time', first_treat='first_treat',
    ...                   aggregate='event_study')
    >>> from diff_diff import plot_event_study
    >>> plot_event_study(results)

    Notes
    -----
    The imputation estimator uses ALL untreated observations (never-treated +
    not-yet-treated periods of eventually-treated units) to estimate the
    counterfactual model. There is no ``control_group`` parameter because this
    is fundamental to the method's efficiency.

    References
    ----------
    Borusyak, K., Jaravel, X., & Spiess, J. (2024). Revisiting Event-Study
    Designs: Robust and Efficient Estimation. Review of Economic Studies,
    91(6), 3253-3285.
    """

    def __init__(
        self,
        anticipation: int = 0,
        alpha: float = 0.05,
        cluster: Optional[str] = None,
        n_bootstrap: int = 0,
        bootstrap_weights: str = "rademacher",
        seed: Optional[int] = None,
        rank_deficient_action: str = "warn",
        horizon_max: Optional[int] = None,
        aux_partition: str = "cohort_horizon",
    ):
        if rank_deficient_action not in ("warn", "error", "silent"):
            raise ValueError(
                f"rank_deficient_action must be 'warn', 'error', or 'silent', "
                f"got '{rank_deficient_action}'"
            )
        if bootstrap_weights not in ("rademacher", "mammen", "webb"):
            raise ValueError(
                f"bootstrap_weights must be 'rademacher', 'mammen', or 'webb', "
                f"got '{bootstrap_weights}'"
            )
        if aux_partition not in ("cohort_horizon", "cohort", "horizon"):
            raise ValueError(
                f"aux_partition must be 'cohort_horizon', 'cohort', or 'horizon', "
                f"got '{aux_partition}'"
            )

        self.anticipation = anticipation
        self.alpha = alpha
        self.cluster = cluster
        self.n_bootstrap = n_bootstrap
        self.bootstrap_weights = bootstrap_weights
        self.seed = seed
        self.rank_deficient_action = rank_deficient_action
        self.horizon_max = horizon_max
        self.aux_partition = aux_partition

        self.is_fitted_ = False
        self.results_: Optional[ImputationDiDResults] = None

        # Internal state preserved for pretrend_test()
        self._fit_data: Optional[Dict[str, Any]] = None

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        covariates: Optional[List[str]] = None,
        aggregate: Optional[str] = None,
        balance_e: Optional[int] = None,
        survey_design: object = None,
    ) -> ImputationDiDResults:
        """
        Fit the imputation DiD estimator.

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
        covariates : list of str, optional
            List of covariate column names.
        aggregate : str, optional
            Aggregation mode: None/"simple" (overall ATT only),
            "event_study", "group", or "all".
        balance_e : int, optional
            When computing event study, restrict to cohorts observed at all
            relative times in [-balance_e, max_h].
        survey_design : SurveyDesign, optional
            Survey design specification for design-based inference. Supports
            pweight only (aweight/fweight raise ValueError). FPC raises
            NotImplementedError. PSU is used as cluster variable for Theorem 3
            variance. Strata enters survey df for t-distribution inference.
            Both analytical (n_bootstrap=0) and bootstrap inference are supported.

        Returns
        -------
        ImputationDiDResults
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

        # Resolve survey design if provided
        from diff_diff.survey import (
            _inject_cluster_as_psu,
            _resolve_effective_cluster,
            _resolve_survey_for_fit,
            _validate_unit_constant_survey,
        )

        resolved_survey, survey_weights, _, survey_metadata = _resolve_survey_for_fit(
            survey_design, data, "analytical"
        )

        # Validate within-unit constancy for panel survey designs
        if resolved_survey is not None:
            if resolved_survey.uses_replicate_variance:
                raise NotImplementedError(
                    "ImputationDiD does not yet support replicate-weight survey "
                    "designs. Use a TSL-based survey design (strata/psu/fpc)."
                )
            _validate_unit_constant_survey(data, unit, survey_design)
            if resolved_survey.weight_type != "pweight":
                raise ValueError(
                    f"ImputationDiD survey support requires weight_type='pweight', "
                    f"got '{resolved_survey.weight_type}'. The survey variance math "
                    f"assumes probability weights (pweight)."
                )
            if resolved_survey.fpc is not None:
                raise NotImplementedError(
                    "ImputationDiD does not yet support FPC (finite population "
                    "correction) in SurveyDesign. Weights, strata (for survey df), "
                    "and PSU (for cluster-robust variance) are supported."
                )

        # Bootstrap + survey supported via PSU-level multiplier bootstrap.

        # Ensure numeric types
        df[time] = pd.to_numeric(df[time])
        df[first_treat] = pd.to_numeric(df[first_treat])

        # Validate absorbing treatment: first_treat must be constant within each unit
        ft_nunique = df.groupby(unit)[first_treat].nunique()
        non_constant = ft_nunique[ft_nunique > 1]
        if len(non_constant) > 0:
            example_unit = non_constant.index[0]
            example_vals = sorted(df.loc[df[unit] == example_unit, first_treat].unique())
            warnings.warn(
                f"{len(non_constant)} unit(s) have non-constant '{first_treat}' "
                f"values (e.g., unit '{example_unit}' has values {example_vals}). "
                f"ImputationDiD assumes treatment is an absorbing state "
                f"(once treated, always treated) with a single treatment onset "
                f"time per unit. Non-constant first_treat violates this assumption "
                f"and may produce unreliable estimates.",
                UserWarning,
                stacklevel=2,
            )

            # Coerce to per-unit value so downstream code
            # (_never_treated, _treated, _rel_time) uses a single
            # consistent first_treat per unit.
            df[first_treat] = df.groupby(unit)[first_treat].transform("first")

        # Identify treatment status
        df["_never_treated"] = (df[first_treat] == 0) | (df[first_treat] == np.inf)

        # Check for always-treated units (treated in all observed periods)
        min_time = df[time].min()
        always_treated_mask = (~df["_never_treated"]) & (df[first_treat] <= min_time)
        n_always_treated = df.loc[always_treated_mask, unit].nunique()
        if n_always_treated > 0:
            warnings.warn(
                f"{n_always_treated} unit(s) are treated in all observed periods "
                f"(first_treat <= {min_time}). These units have no untreated "
                "observations and cannot contribute to the counterfactual model. "
                "Their treatment effects will be imputed but may be unreliable.",
                UserWarning,
                stacklevel=2,
            )

        # Create treatment indicator D_it
        # D_it = 1 if t >= first_treat and first_treat > 0
        # With anticipation: D_it = 1 if t >= first_treat - anticipation
        effective_treat = df[first_treat] - self.anticipation
        df["_treated"] = (~df["_never_treated"]) & (df[time] >= effective_treat)

        # Identify Omega_0 (untreated) and Omega_1 (treated)
        omega_0_mask = ~df["_treated"]
        omega_1_mask = df["_treated"]

        n_omega_0 = int(omega_0_mask.sum())
        n_omega_1 = int(omega_1_mask.sum())

        if n_omega_0 == 0:
            raise ValueError(
                "No untreated observations found. Cannot estimate counterfactual model."
            )
        if n_omega_1 == 0:
            raise ValueError("No treated observations found. Nothing to estimate.")

        # Identify groups and time periods
        time_periods = sorted(df[time].unique())
        treatment_groups = sorted([g for g in df[first_treat].unique() if g > 0 and g != np.inf])

        if len(treatment_groups) == 0:
            raise ValueError("No treated units found. Check 'first_treat' column.")

        # Unit info
        unit_info = (
            df.groupby(unit).agg({first_treat: "first", "_never_treated": "first"}).reset_index()
        )
        n_treated_units = int((~unit_info["_never_treated"]).sum())
        # Control units = units with at least one untreated observation
        units_in_omega_0 = df.loc[omega_0_mask, unit].unique()
        n_control_units = len(units_in_omega_0)

        # Cluster variable
        cluster_var = self.cluster if self.cluster is not None else unit
        if self.cluster is not None and self.cluster not in df.columns:
            raise ValueError(
                f"Cluster column '{self.cluster}' not found in data. "
                f"Available columns: {list(df.columns)}"
            )

        # Resolve effective cluster and inject cluster-as-PSU for survey variance
        if resolved_survey is not None:
            cluster_ids_raw = df[cluster_var].values if cluster_var in df.columns else None
            effective_cluster_ids = _resolve_effective_cluster(
                resolved_survey,
                cluster_ids_raw,
                cluster_var if self.cluster is not None else None,
            )
            resolved_survey = _inject_cluster_as_psu(resolved_survey, effective_cluster_ids)
            # When survey PSU is present, use it as the effective cluster for
            # Theorem 3 variance (PSU overrides unit-level clustering)
            if resolved_survey.psu is not None:
                # Create a temporary column with PSU IDs for cluster_var
                df["_survey_cluster"] = resolved_survey.psu
                cluster_var = "_survey_cluster"
            # Recompute metadata after PSU injection
            if resolved_survey.psu is not None and survey_metadata is not None:
                from diff_diff.survey import compute_survey_metadata

                raw_w = (
                    data[survey_design.weights].values.astype(np.float64)
                    if survey_design.weights
                    else np.ones(len(data), dtype=np.float64)
                )
                survey_metadata = compute_survey_metadata(resolved_survey, raw_w)

        # Compute relative time
        df["_rel_time"] = np.where(
            ~df["_never_treated"],
            df[time] - df[first_treat],
            np.nan,
        )

        # ---- Step 1: OLS on untreated observations ----
        unit_fe, time_fe, grand_mean, delta_hat, kept_cov_mask = self._fit_untreated_model(
            df, outcome, unit, time, covariates, omega_0_mask, weights=survey_weights
        )

        # ---- Rank condition checks ----
        # Check: every treated unit should have >= 1 untreated period (for unit FE)
        treated_unit_ids = df.loc[omega_1_mask, unit].unique()
        units_with_fe = set(unit_fe.keys())
        units_missing_fe = set(treated_unit_ids) - units_with_fe

        # Check: every post-treatment period should have >= 1 untreated unit (for time FE)
        post_period_ids = df.loc[omega_1_mask, time].unique()
        periods_with_fe = set(time_fe.keys())
        periods_missing_fe = set(post_period_ids) - periods_with_fe

        if units_missing_fe or periods_missing_fe:
            parts = []
            if units_missing_fe:
                sorted_missing = sorted(units_missing_fe)
                parts.append(
                    f"{len(units_missing_fe)} treated unit(s) have no untreated "
                    f"periods (units: {sorted_missing[:5]}"
                    f"{'...' if len(units_missing_fe) > 5 else ''})"
                )
            if periods_missing_fe:
                sorted_missing = sorted(periods_missing_fe)
                parts.append(
                    f"{len(periods_missing_fe)} post-treatment period(s) have no "
                    f"untreated units (periods: {sorted_missing[:5]}"
                    f"{'...' if len(periods_missing_fe) > 5 else ''})"
                )
            msg = (
                "Rank condition violated: "
                + "; ".join(parts)
                + ". Affected treatment effects will be NaN."
            )
            if self.rank_deficient_action == "error":
                raise ValueError(msg)
            elif self.rank_deficient_action == "warn":
                warnings.warn(msg, UserWarning, stacklevel=2)
            # "silent": continue without warning

        # ---- Step 2: Impute treatment effects ----
        tau_hat, y_hat_0 = self._impute_treatment_effects(
            df,
            outcome,
            unit,
            time,
            covariates,
            omega_1_mask,
            unit_fe,
            time_fe,
            grand_mean,
            delta_hat,
        )

        # Store tau_hat in dataframe
        df["_tau_hat"] = np.nan
        df.loc[omega_1_mask, "_tau_hat"] = tau_hat

        # ---- Step 3: Aggregate ----
        # Always compute overall ATT (simple aggregation)
        finite_mask = np.isfinite(tau_hat)
        valid_tau = tau_hat[finite_mask]

        if len(valid_tau) == 0:
            overall_att = np.nan
        elif survey_weights is not None:
            # Survey-weighted ATT: use treated obs' survey weights
            treated_survey_w = survey_weights[omega_1_mask.values]
            w_finite = treated_survey_w[finite_mask]
            overall_att = float(np.average(valid_tau, weights=w_finite))
        else:
            overall_att = float(np.mean(valid_tau))

        # ---- Conservative variance (Theorem 3) ----
        # Build weights matching the ATT: proportional to survey weights for
        # finite tau_hat, uniform when no survey
        overall_weights = np.zeros(n_omega_1)
        n_valid = int(finite_mask.sum())
        if n_valid > 0:
            if survey_weights is not None:
                treated_sw = survey_weights[omega_1_mask.values]
                sw_finite = treated_sw[finite_mask]
                overall_weights[finite_mask] = sw_finite / sw_finite.sum()
            else:
                overall_weights[finite_mask] = 1.0 / n_valid

        if n_valid == 0:
            overall_se = np.nan
        else:
            overall_se = self._compute_conservative_variance(
                df=df,
                outcome=outcome,
                unit=unit,
                time=time,
                first_treat=first_treat,
                covariates=covariates,
                omega_0_mask=omega_0_mask,
                omega_1_mask=omega_1_mask,
                unit_fe=unit_fe,
                time_fe=time_fe,
                grand_mean=grand_mean,
                delta_hat=delta_hat,
                weights=overall_weights,
                cluster_var=cluster_var,
                kept_cov_mask=kept_cov_mask,
                survey_weights=survey_weights,
            )

        # Survey degrees of freedom for t-distribution inference
        _survey_df = resolved_survey.df_survey if resolved_survey is not None else None

        overall_t, overall_p, overall_ci = safe_inference(
            overall_att, overall_se, alpha=self.alpha, df=_survey_df
        )

        # Event study and group aggregation
        event_study_effects = None
        group_effects = None

        if aggregate in ("event_study", "all"):
            event_study_effects = self._aggregate_event_study(
                df=df,
                outcome=outcome,
                unit=unit,
                time=time,
                first_treat=first_treat,
                covariates=covariates,
                omega_0_mask=omega_0_mask,
                omega_1_mask=omega_1_mask,
                unit_fe=unit_fe,
                time_fe=time_fe,
                grand_mean=grand_mean,
                delta_hat=delta_hat,
                cluster_var=cluster_var,
                treatment_groups=treatment_groups,
                balance_e=balance_e,
                kept_cov_mask=kept_cov_mask,
                survey_weights=survey_weights,
                survey_df=_survey_df,
            )

        if aggregate in ("group", "all"):
            group_effects = self._aggregate_group(
                df=df,
                outcome=outcome,
                unit=unit,
                time=time,
                first_treat=first_treat,
                covariates=covariates,
                omega_0_mask=omega_0_mask,
                omega_1_mask=omega_1_mask,
                unit_fe=unit_fe,
                time_fe=time_fe,
                grand_mean=grand_mean,
                delta_hat=delta_hat,
                cluster_var=cluster_var,
                treatment_groups=treatment_groups,
                kept_cov_mask=kept_cov_mask,
                survey_weights=survey_weights,
                survey_df=_survey_df,
            )

        # Build treatment effects dataframe
        treated_df = df.loc[omega_1_mask, [unit, time, "_tau_hat", "_rel_time"]].copy()
        treated_df = treated_df.rename(columns={"_tau_hat": "tau_hat", "_rel_time": "rel_time"})
        # Weights consistent with actual ATT: zero for NaN tau_hat
        tau_finite = treated_df["tau_hat"].notna()
        n_valid_te = int(tau_finite.sum())
        if n_valid_te > 0:
            if survey_weights is not None:
                # Survey-weighted: use normalized survey weights for treated obs
                treated_sw = survey_weights[omega_1_mask.values]
                sw_finite = np.where(tau_finite, treated_sw, 0.0)
                sw_sum = sw_finite.sum()
                treated_df["weight"] = sw_finite / sw_sum if sw_sum > 0 else 0.0
            else:
                treated_df["weight"] = np.where(tau_finite, 1.0 / n_valid_te, 0.0)
        else:
            treated_df["weight"] = 0.0

        # Store fit data for pretrend_test
        self._fit_data = {
            "df": df,
            "outcome": outcome,
            "unit": unit,
            "time": time,
            "first_treat": first_treat,
            "covariates": covariates,
            "omega_0_mask": omega_0_mask,
            "omega_1_mask": omega_1_mask,
            "cluster_var": cluster_var,
            "unit_fe": unit_fe,
            "time_fe": time_fe,
            "grand_mean": grand_mean,
            "delta_hat": delta_hat,
            "kept_cov_mask": kept_cov_mask,
            "survey_design": survey_design,
        }

        # Pre-compute cluster psi sums for bootstrap
        psi_data = None
        if self.n_bootstrap > 0 and n_valid > 0:
            try:
                # Extract survey weights for untreated obs (same as analytical path)
                _sw_0 = survey_weights[omega_0_mask.values] if survey_weights is not None else None
                # Extract survey weights for treated obs (event-study/group bootstrap paths)
                _sw_1 = survey_weights[omega_1_mask.values] if survey_weights is not None else None
                psi_data = self._precompute_bootstrap_psi(
                    df=df,
                    outcome=outcome,
                    unit=unit,
                    time=time,
                    first_treat=first_treat,
                    covariates=covariates,
                    omega_0_mask=omega_0_mask,
                    omega_1_mask=omega_1_mask,
                    unit_fe=unit_fe,
                    time_fe=time_fe,
                    grand_mean=grand_mean,
                    delta_hat=delta_hat,
                    cluster_var=cluster_var,
                    kept_cov_mask=kept_cov_mask,
                    overall_weights=overall_weights,
                    event_study_effects=event_study_effects,
                    group_effects=group_effects,
                    treatment_groups=treatment_groups,
                    tau_hat=tau_hat,
                    balance_e=balance_e,
                    survey_weights_0=_sw_0,
                    survey_weights_1=_sw_1,
                )
            except Exception as e:
                warnings.warn(
                    f"Bootstrap pre-computation failed: {e}. " "Skipping bootstrap inference.",
                    UserWarning,
                    stacklevel=2,
                )
                psi_data = None

        # Bootstrap
        bootstrap_results = None
        if self.n_bootstrap > 0 and psi_data is not None:
            bootstrap_results = self._run_bootstrap(
                original_att=overall_att,
                original_event_study=event_study_effects,
                original_group=group_effects,
                psi_data=psi_data,
                resolved_survey=resolved_survey,
            )

            # Update inference with bootstrap results
            overall_se = bootstrap_results.overall_att_se
            overall_t = (
                overall_att / overall_se if np.isfinite(overall_se) and overall_se > 0 else np.nan
            )
            overall_p = bootstrap_results.overall_att_p_value
            overall_ci = bootstrap_results.overall_att_ci

            # Update event study
            if event_study_effects and bootstrap_results.event_study_ses:
                for h in event_study_effects:
                    if (
                        h in bootstrap_results.event_study_ses
                        and event_study_effects[h].get("n_obs", 1) > 0
                    ):
                        event_study_effects[h]["se"] = bootstrap_results.event_study_ses[h]
                        assert bootstrap_results.event_study_cis is not None
                        event_study_effects[h]["conf_int"] = bootstrap_results.event_study_cis[h]
                        assert bootstrap_results.event_study_p_values is not None
                        event_study_effects[h]["p_value"] = bootstrap_results.event_study_p_values[
                            h
                        ]
                        eff_val = event_study_effects[h]["effect"]
                        se_val = event_study_effects[h]["se"]
                        event_study_effects[h]["t_stat"] = safe_inference(
                            eff_val, se_val, alpha=self.alpha
                        )[0]

            # Update group effects
            if group_effects and bootstrap_results.group_ses:
                for g in group_effects:
                    if g in bootstrap_results.group_ses:
                        group_effects[g]["se"] = bootstrap_results.group_ses[g]
                        assert bootstrap_results.group_cis is not None
                        group_effects[g]["conf_int"] = bootstrap_results.group_cis[g]
                        assert bootstrap_results.group_p_values is not None
                        group_effects[g]["p_value"] = bootstrap_results.group_p_values[g]
                        eff_val = group_effects[g]["effect"]
                        se_val = group_effects[g]["se"]
                        group_effects[g]["t_stat"] = safe_inference(
                            eff_val, se_val, alpha=self.alpha
                        )[0]

        # Construct results
        self.results_ = ImputationDiDResults(
            treatment_effects=treated_df,
            overall_att=overall_att,
            overall_se=overall_se,
            overall_t_stat=overall_t,
            overall_p_value=overall_p,
            overall_conf_int=overall_ci,
            event_study_effects=event_study_effects,
            group_effects=group_effects,
            groups=treatment_groups,
            time_periods=time_periods,
            n_obs=len(df),
            n_treated_obs=n_omega_1,
            n_untreated_obs=n_omega_0,
            n_treated_units=n_treated_units,
            n_control_units=n_control_units,
            alpha=self.alpha,
            bootstrap_results=bootstrap_results,
            _estimator_ref=self,
            survey_metadata=survey_metadata,
        )

        self.is_fitted_ = True
        return self.results_

    # =========================================================================
    # Step 1: OLS on untreated observations
    # =========================================================================

    def _iterative_fe(
        self,
        y: np.ndarray,
        unit_vals: np.ndarray,
        time_vals: np.ndarray,
        idx: pd.Index,
        max_iter: int = 100,
        tol: float = 1e-10,
        weights: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[Any, float], Dict[Any, float]]:
        """
        Estimate unit and time FE via iterative alternating projection (Gauss-Seidel).

        Converges to the exact OLS solution for both balanced and unbalanced panels.
        For balanced panels, converges in 1-2 iterations (identical to one-pass).
        For unbalanced panels, typically 5-20 iterations.

        Parameters
        ----------
        weights : np.ndarray, optional
            Survey weights. When provided, uses weighted group means
            (sum(w*x)/sum(w)) instead of unweighted means.

        Returns
        -------
        unit_fe : dict
            Mapping from unit -> unit fixed effect.
        time_fe : dict
            Mapping from time -> time fixed effect.
        """
        n = len(y)
        alpha = np.zeros(n)  # unit FE broadcast to obs level
        beta = np.zeros(n)  # time FE broadcast to obs level

        # Precompute per-group weight sums (invariant across iterations)
        if weights is not None:
            w_series = pd.Series(weights, index=idx)
            wsum_t = w_series.groupby(time_vals).transform("sum").values
            wsum_u = w_series.groupby(unit_vals).transform("sum").values

        with np.errstate(invalid="ignore", divide="ignore"):
            for iteration in range(max_iter):
                resid_after_alpha = y - alpha
                if weights is not None:
                    wr_t = pd.Series(resid_after_alpha * weights, index=idx)
                    beta_new = wr_t.groupby(time_vals).transform("sum").values / wsum_t
                else:
                    beta_new = (
                        pd.Series(resid_after_alpha, index=idx)
                        .groupby(time_vals)
                        .transform("mean")
                        .values
                    )

                resid_after_beta = y - beta_new
                if weights is not None:
                    wr_u = pd.Series(resid_after_beta * weights, index=idx)
                    alpha_new = wr_u.groupby(unit_vals).transform("sum").values / wsum_u
                else:
                    alpha_new = (
                        pd.Series(resid_after_beta, index=idx)
                        .groupby(unit_vals)
                        .transform("mean")
                        .values
                    )

                # Check convergence on FE changes
                max_change = max(
                    np.max(np.abs(alpha_new - alpha)),
                    np.max(np.abs(beta_new - beta)),
                )
                alpha = alpha_new
                beta = beta_new
                if max_change < tol:
                    break

        unit_fe = pd.Series(alpha, index=idx).groupby(unit_vals).first().to_dict()
        time_fe = pd.Series(beta, index=idx).groupby(time_vals).first().to_dict()
        return unit_fe, time_fe

    @staticmethod
    def _iterative_demean(
        vals: np.ndarray,
        unit_vals: np.ndarray,
        time_vals: np.ndarray,
        idx: pd.Index,
        max_iter: int = 100,
        tol: float = 1e-10,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Demean a vector by iterative alternating projection (unit + time FE removal).

        Converges to the exact within-transformation for both balanced and
        unbalanced panels. For balanced panels, converges in 1-2 iterations.

        Parameters
        ----------
        weights : np.ndarray, optional
            Survey weights. When provided, uses weighted group means
            (sum(w*x)/sum(w)) instead of unweighted means.
        """
        result = vals.copy()

        # Precompute per-group weight sums (invariant across iterations)
        if weights is not None:
            w_series = pd.Series(weights, index=idx)
            wsum_t = w_series.groupby(time_vals).transform("sum").values
            wsum_u = w_series.groupby(unit_vals).transform("sum").values

        with np.errstate(invalid="ignore", divide="ignore"):
            for _ in range(max_iter):
                if weights is not None:
                    wr_t = pd.Series(result * weights, index=idx)
                    time_means = wr_t.groupby(time_vals).transform("sum").values / wsum_t
                else:
                    time_means = (
                        pd.Series(result, index=idx).groupby(time_vals).transform("mean").values
                    )
                result_after_time = result - time_means
                if weights is not None:
                    wr_u = pd.Series(result_after_time * weights, index=idx)
                    unit_means = wr_u.groupby(unit_vals).transform("sum").values / wsum_u
                else:
                    unit_means = (
                        pd.Series(result_after_time, index=idx)
                        .groupby(unit_vals)
                        .transform("mean")
                        .values
                    )
                result_new = result_after_time - unit_means
                if np.max(np.abs(result_new - result)) < tol:
                    result = result_new
                    break
                result = result_new
        return result

    @staticmethod
    def _compute_balanced_cohort_mask(
        df_treated: pd.DataFrame,
        first_treat: str,
        all_horizons: List[int],
        balance_e: int,
        cohort_rel_times: Dict[Any, Set[int]],
    ) -> np.ndarray:
        """Compute boolean mask selecting treated obs from balanced cohorts.

        A cohort is 'balanced' if it has observations at every relative time
        in [-balance_e, max(all_horizons)].

        Parameters
        ----------
        df_treated : pd.DataFrame
            Post-treatment observations (Omega_1).
        first_treat : str
            Column name for cohort identifier.
        all_horizons : list of int
            Post-treatment horizons in the event study.
        balance_e : int
            Number of pre-treatment periods to require.
        cohort_rel_times : dict
            Maps each cohort value to the set of all observed relative times
            (including pre-treatment) from the full panel. Built by
            _build_cohort_rel_times().
        """
        if not all_horizons:
            return np.ones(len(df_treated), dtype=bool)

        max_h = max(all_horizons)
        required_range = set(range(-balance_e, max_h + 1))

        balanced_cohorts = set()
        for g, horizons in cohort_rel_times.items():
            if required_range.issubset(horizons):
                balanced_cohorts.add(g)

        return df_treated[first_treat].isin(balanced_cohorts).values

    @staticmethod
    def _build_cohort_rel_times(
        df: pd.DataFrame,
        first_treat: str,
    ) -> Dict[Any, Set[int]]:
        """Build mapping of cohort -> set of observed relative times from full panel.

        Precondition: df must have '_never_treated' and '_rel_time' columns
        (set by fit() before any aggregation calls).
        """
        treated_mask = ~df["_never_treated"]
        treated_df = df.loc[treated_mask]
        result: Dict[Any, Set[int]] = {}
        ft_vals = treated_df[first_treat].values
        rt_vals = treated_df["_rel_time"].values
        for i in range(len(treated_df)):
            h = rt_vals[i]
            if np.isfinite(h):
                result.setdefault(ft_vals[i], set()).add(int(h))
        return result

    def _fit_untreated_model(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        covariates: Optional[List[str]],
        omega_0_mask: pd.Series,
        weights: Optional[np.ndarray] = None,
    ) -> Tuple[
        Dict[Any, float], Dict[Any, float], float, Optional[np.ndarray], Optional[np.ndarray]
    ]:
        """
        Step 1: Estimate unit + time FE on untreated observations.

        Uses iterative alternating projection (Gauss-Seidel) to compute exact
        OLS fixed effects for both balanced and unbalanced panels. For balanced
        panels, converges in 1-2 iterations (identical to one-pass demeaning).

        Parameters
        ----------
        weights : np.ndarray, optional
            Full-panel survey weights (same length as df). The untreated subset
            is extracted internally via omega_0_mask. When None, unweighted.

        Returns
        -------
        unit_fe : dict
            Unit fixed effects {unit_id: alpha_i}.
        time_fe : dict
            Time fixed effects {time_period: beta_t}.
        grand_mean : float
            Grand mean (0.0 — absorbed into iterative FE).
        delta_hat : np.ndarray or None
            Covariate coefficients (if covariates provided).
        kept_cov_mask : np.ndarray or None
            Boolean mask of shape (n_covariates,) indicating which covariates
            have finite coefficients. None if no covariates.
        """
        df_0 = df.loc[omega_0_mask]
        w_0 = weights[omega_0_mask.values] if weights is not None else None

        if covariates is None or len(covariates) == 0:
            # No covariates: estimate FE via iterative alternating projection
            # (exact OLS for both balanced and unbalanced panels)
            y = df_0[outcome].values.copy()
            unit_fe, time_fe = self._iterative_fe(
                y, df_0[unit].values, df_0[time].values, df_0.index, weights=w_0
            )
            # grand_mean = 0: iterative FE absorb the intercept
            return unit_fe, time_fe, 0.0, None, None

        else:
            # With covariates: iteratively demean Y and X, OLS for delta,
            # then recover FE from covariate-adjusted outcome
            y = df_0[outcome].values.copy()
            X_raw = df_0[covariates].values.copy()
            units = df_0[unit].values
            times = df_0[time].values
            n_cov = len(covariates)

            # Step A: Iteratively demean Y and all X columns to remove unit+time FE
            y_dm = self._iterative_demean(y, units, times, df_0.index, weights=w_0)
            X_dm = np.column_stack(
                [
                    self._iterative_demean(X_raw[:, j], units, times, df_0.index, weights=w_0)
                    for j in range(n_cov)
                ]
            )

            # Step B: OLS for covariate coefficients on demeaned data
            result = solve_ols(
                X_dm,
                y_dm,
                return_vcov=False,
                rank_deficient_action=self.rank_deficient_action,
                column_names=covariates,
                weights=w_0,
            )
            delta_hat = result[0]

            # Mask of covariates with finite coefficients (before cleaning)
            # Used to exclude rank-deficient covariates from variance design matrices
            kept_cov_mask = np.isfinite(delta_hat)

            # Replace NaN coefficients with 0 for adjustment
            # (rank-deficient covariates are dropped)
            delta_hat_clean = np.where(np.isfinite(delta_hat), delta_hat, 0.0)

            # Step C: Recover FE from covariate-adjusted outcome using iterative FE
            y_adj = y - np.dot(X_raw, delta_hat_clean)
            unit_fe, time_fe = self._iterative_fe(y_adj, units, times, df_0.index, weights=w_0)

            # grand_mean = 0: iterative FE absorb the intercept
            return unit_fe, time_fe, 0.0, delta_hat_clean, kept_cov_mask

    # =========================================================================
    # Step 2: Impute counterfactuals
    # =========================================================================

    def _impute_treatment_effects(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        covariates: Optional[List[str]],
        omega_1_mask: pd.Series,
        unit_fe: Dict[Any, float],
        time_fe: Dict[Any, float],
        grand_mean: float,
        delta_hat: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step 2: Impute Y(0) for treated observations and compute tau_hat.

        Returns
        -------
        tau_hat : np.ndarray
            Imputed treatment effects for each treated observation.
        y_hat_0 : np.ndarray
            Imputed counterfactual Y(0).
        """
        df_1 = df.loc[omega_1_mask]
        n_1 = len(df_1)

        # Look up unit and time FE
        alpha_i = df_1[unit].map(unit_fe).values
        beta_t = df_1[time].map(time_fe).values

        # Handle missing FE (set to NaN)
        alpha_i = np.where(pd.isna(alpha_i), np.nan, alpha_i).astype(float)
        beta_t = np.where(pd.isna(beta_t), np.nan, beta_t).astype(float)

        y_hat_0 = grand_mean + alpha_i + beta_t

        if delta_hat is not None and covariates:
            X_1 = df_1[covariates].values
            y_hat_0 = y_hat_0 + np.dot(X_1, delta_hat)

        tau_hat = df_1[outcome].values - y_hat_0

        return tau_hat, y_hat_0

    # =========================================================================
    # Conservative Variance (Theorem 3)
    # =========================================================================

    def _compute_cluster_psi_sums(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        covariates: Optional[List[str]],
        omega_0_mask: pd.Series,
        omega_1_mask: pd.Series,
        unit_fe: Dict[Any, float],
        time_fe: Dict[Any, float],
        grand_mean: float,
        delta_hat: Optional[np.ndarray],
        weights: np.ndarray,
        cluster_var: str,
        kept_cov_mask: Optional[np.ndarray] = None,
        survey_weights_0: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute cluster-level influence function sums (Theorem 3).

        psi_i = sum_t v_it * epsilon_tilde_it, summed within each cluster.

        Returns
        -------
        cluster_psi_sums : np.ndarray
            Array of cluster-level psi sums.
        cluster_ids_unique : np.ndarray
            Unique cluster identifiers (matching order of psi sums).
        """
        df_0 = df.loc[omega_0_mask]
        df_1 = df.loc[omega_1_mask]
        n_0 = len(df_0)
        n_1 = len(df_1)

        # ---- Compute v_it for treated observations ----
        v_treated = weights.copy()

        # ---- Compute v_it for untreated observations ----
        if covariates is None or len(covariates) == 0:
            # FE-only case: closed-form
            treated_units = df_1[unit].values
            treated_times = df_1[time].values

            w_by_unit: Dict[Any, float] = {}
            for i_idx in range(n_1):
                u = treated_units[i_idx]
                w_by_unit[u] = w_by_unit.get(u, 0.0) + weights[i_idx]

            w_by_time: Dict[Any, float] = {}
            for i_idx in range(n_1):
                t = treated_times[i_idx]
                w_by_time[t] = w_by_time.get(t, 0.0) + weights[i_idx]

            w_total = float(np.sum(weights))

            # Use survey-weighted sums for untreated denominators when present
            if survey_weights_0 is not None:
                sw0_series = pd.Series(survey_weights_0, index=df_0.index)
                n0_by_unit = sw0_series.groupby(df_0[unit]).sum().to_dict()
                n0_by_time = sw0_series.groupby(df_0[time]).sum().to_dict()
                n0_denom = float(np.sum(survey_weights_0))
            else:
                n0_by_unit = df_0.groupby(unit).size().to_dict()
                n0_by_time = df_0.groupby(time).size().to_dict()
                n0_denom = n_0

            untreated_units = df_0[unit].values
            untreated_times = df_0[time].values
            v_untreated = np.zeros(n_0)

            for j in range(n_0):
                u = untreated_units[j]
                t = untreated_times[j]
                w_i = w_by_unit.get(u, 0.0)
                w_t = w_by_time.get(t, 0.0)
                n0_i = n0_by_unit.get(u, 1)
                n0_t = n0_by_time.get(t, 1)
                base_v = -(w_i / n0_i + w_t / n0_t - w_total / n0_denom)
                # WLS projection requires per-obs survey weight factor
                if survey_weights_0 is not None:
                    base_v *= survey_weights_0[j]
                v_untreated[j] = base_v
        else:
            v_untreated = self._compute_v_untreated_with_covariates(
                df_0,
                df_1,
                unit,
                time,
                covariates,
                weights,
                delta_hat,
                kept_cov_mask=kept_cov_mask,
                survey_weights_0=survey_weights_0,
            )

        # ---- Compute auxiliary model residuals (Equation 8) ----
        epsilon_treated = self._compute_auxiliary_residuals_treated(
            df_1,
            outcome,
            unit,
            time,
            first_treat,
            covariates,
            unit_fe,
            time_fe,
            grand_mean,
            delta_hat,
            v_treated,
        )
        epsilon_untreated = self._compute_residuals_untreated(
            df_0, outcome, unit, time, covariates, unit_fe, time_fe, grand_mean, delta_hat
        )

        # ---- psi_it = v_it * epsilon_tilde_it ----
        v_all = np.empty(len(df))
        v_all[omega_1_mask.values] = v_treated
        v_all[omega_0_mask.values] = v_untreated

        eps_all = np.empty(len(df))
        eps_all[omega_1_mask.values] = epsilon_treated
        eps_all[omega_0_mask.values] = epsilon_untreated

        ve_product = v_all * eps_all
        # NaN eps from missing FE (rank condition violation). Zero their variance
        # contribution — matches R's did_imputation which drops unimputable obs.
        np.nan_to_num(ve_product, copy=False, nan=0.0)

        # Sum within clusters
        cluster_ids = df[cluster_var].values
        ve_series = pd.Series(ve_product, index=df.index)
        cluster_sums = ve_series.groupby(cluster_ids).sum()

        return cluster_sums.values, cluster_sums.index.values

    def _compute_conservative_variance(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        covariates: Optional[List[str]],
        omega_0_mask: pd.Series,
        omega_1_mask: pd.Series,
        unit_fe: Dict[Any, float],
        time_fe: Dict[Any, float],
        grand_mean: float,
        delta_hat: Optional[np.ndarray],
        weights: np.ndarray,
        cluster_var: str,
        kept_cov_mask: Optional[np.ndarray] = None,
        survey_weights: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute conservative clustered variance (Theorem 3, Equation 7).

        Parameters
        ----------
        weights : np.ndarray
            Aggregation weights w_it for treated observations.
            Shape: (n_treated,), must sum to 1.
        survey_weights : np.ndarray, optional
            Full-panel survey weights. When provided, untreated denominators
            in v_it use survey-weighted sums instead of raw counts.

        Returns
        -------
        float
            Standard error.
        """
        sw_0 = survey_weights[omega_0_mask.values] if survey_weights is not None else None
        cluster_psi_sums, _ = self._compute_cluster_psi_sums(
            df=df,
            outcome=outcome,
            unit=unit,
            time=time,
            first_treat=first_treat,
            covariates=covariates,
            omega_0_mask=omega_0_mask,
            omega_1_mask=omega_1_mask,
            unit_fe=unit_fe,
            time_fe=time_fe,
            grand_mean=grand_mean,
            delta_hat=delta_hat,
            weights=weights,
            cluster_var=cluster_var,
            kept_cov_mask=kept_cov_mask,
            survey_weights_0=sw_0,
        )
        sigma_sq = float((cluster_psi_sums**2).sum())
        return np.sqrt(max(sigma_sq, 0.0))

    def _compute_v_untreated_with_covariates(
        self,
        df_0: pd.DataFrame,
        df_1: pd.DataFrame,
        unit: str,
        time: str,
        covariates: List[str],
        weights: np.ndarray,
        delta_hat: Optional[np.ndarray],
        kept_cov_mask: Optional[np.ndarray] = None,
        survey_weights_0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute v_it for untreated observations with covariates.

        Uses the projection: v_untreated = -A_0 (A_0'A_0)^{-1} A_1' w_treated
        When survey_weights_0 is provided, uses weighted normal equations:
        v_untreated = -A_0 (A_0' W A_0)^{-1} A_1' w_treated

        Uses scipy.sparse for FE dummy columns to reduce memory from O(N*(U+T))
        to O(N) for the FE portion.
        """
        # Exclude rank-deficient covariates from design matrices
        if kept_cov_mask is not None and not np.all(kept_cov_mask):
            covariates = [c for c, k in zip(covariates, kept_cov_mask) if k]

        units_0 = df_0[unit].values
        times_0 = df_0[time].values
        units_1 = df_1[unit].values
        times_1 = df_1[time].values

        all_units = np.unique(np.concatenate([units_0, units_1]))
        all_times = np.unique(np.concatenate([times_0, times_1]))
        unit_to_idx = {u: i for i, u in enumerate(all_units)}
        time_to_idx = {t: i for i, t in enumerate(all_times)}
        n_units = len(all_units)
        n_times = len(all_times)
        n_cov = len(covariates)
        n_fe_cols = (n_units - 1) + (n_times - 1)

        def _build_A_sparse(df_sub, unit_vals, time_vals):
            n = len(df_sub)

            # Unit dummies (drop first) — vectorized
            u_indices = np.array([unit_to_idx[u] for u in unit_vals])
            u_mask = u_indices > 0  # skip first unit (dropped)
            u_rows = np.arange(n)[u_mask]
            u_cols = u_indices[u_mask] - 1

            # Time dummies (drop first) — vectorized
            t_indices = np.array([time_to_idx[t] for t in time_vals])
            t_mask = t_indices > 0
            t_rows = np.arange(n)[t_mask]
            t_cols = (n_units - 1) + t_indices[t_mask] - 1

            rows = np.concatenate([u_rows, t_rows])
            cols = np.concatenate([u_cols, t_cols])
            data = np.ones(len(rows))

            A_fe = sparse.csr_matrix((data, (rows, cols)), shape=(n, n_fe_cols))

            # Covariates (dense, typically few columns)
            if n_cov > 0:
                A_cov = sparse.csr_matrix(df_sub[covariates].values)
                A = sparse.hstack([A_fe, A_cov], format="csr")
            else:
                A = A_fe

            return A

        A_0 = _build_A_sparse(df_0, units_0, times_0)
        A_1 = _build_A_sparse(df_1, units_1, times_1)

        # Compute A_1' w (sparse.T @ dense -> dense)
        A1_w = A_1.T @ weights  # shape (p,)

        # Solve (A_0' [W] A_0) z = A_1' w using sparse direct solver
        # When survey weights present, use weighted normal equations A_0' W A_0
        if survey_weights_0 is not None:
            A0tA0_sparse = A_0.T @ A_0.multiply(survey_weights_0[:, None])
        else:
            A0tA0_sparse = A_0.T @ A_0  # stays sparse
        try:
            z = spsolve(A0tA0_sparse.tocsc(), A1_w)
        except Exception:
            # Fallback to dense lstsq if sparse solver fails (e.g., singular matrix)
            A0tA0_dense = A0tA0_sparse.toarray()
            z, _, _, _ = np.linalg.lstsq(A0tA0_dense, A1_w, rcond=None)

        # v_untreated = -[W_0] A_0 z (WLS projection requires per-obs weight)
        v_untreated = -(A_0 @ z)
        if survey_weights_0 is not None:
            v_untreated = v_untreated * survey_weights_0
        return v_untreated

    def _compute_auxiliary_residuals_treated(
        self,
        df_1: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        covariates: Optional[List[str]],
        unit_fe: Dict[Any, float],
        time_fe: Dict[Any, float],
        grand_mean: float,
        delta_hat: Optional[np.ndarray],
        v_treated: np.ndarray,
    ) -> np.ndarray:
        """
        Compute v_it-weighted auxiliary residuals for treated obs (Equation 8).

        Computes v_it-weighted tau_tilde_g per Equation 8 of Borusyak et al. (2024):
        tau_tilde_g = sum(v_it * tau_hat_it) / sum(v_it) within group g.

        epsilon_tilde_it = Y_it - alpha_i - beta_t [- X'delta] - tau_tilde_g
        """
        n_1 = len(df_1)

        # Compute base residuals (Y - Y_hat(0) = tau_hat)
        # NaN for missing FE (consistent with _impute_treatment_effects)
        alpha_i = df_1[unit].map(unit_fe).values.astype(float)  # NaN for missing
        beta_t = df_1[time].map(time_fe).values.astype(float)  # NaN for missing
        y_hat_0 = grand_mean + alpha_i + beta_t

        if delta_hat is not None and covariates:
            y_hat_0 = y_hat_0 + np.dot(df_1[covariates].values, delta_hat)

        tau_hat = df_1[outcome].values - y_hat_0

        # Partition Omega_1 and compute tau_tilde for each group
        if self.aux_partition == "cohort_horizon":
            group_keys = list(zip(df_1[first_treat].values, df_1["_rel_time"].values))
        elif self.aux_partition == "cohort":
            group_keys = list(df_1[first_treat].values)
        elif self.aux_partition == "horizon":
            group_keys = list(df_1["_rel_time"].values)
        else:
            group_keys = list(range(n_1))  # each obs is its own group

        # Compute v_it-weighted average tau within each partition group (Equation 8)
        # tau_tilde_g = sum(v_it * tau_hat_it) / sum(v_it) within group g
        group_series = pd.Series(group_keys, index=df_1.index)
        tau_series = pd.Series(tau_hat, index=df_1.index)
        v_series = pd.Series(v_treated, index=df_1.index)

        weighted_tau_sum = (v_series * tau_series).groupby(group_series).sum()
        weight_sum = v_series.groupby(group_series).sum()

        # Guard: zero-weight groups -> their tau_tilde doesn't affect variance
        # (v_it ~ 0 means these obs contribute nothing to the estimand)
        # Use simple mean as fallback. This is common for event-study SE computation
        # where weights target a specific horizon, making other partition groups zero.
        zero_weight_groups = weight_sum.abs() < 1e-15
        if zero_weight_groups.any():
            simple_means = tau_series.groupby(group_series).mean()
            tau_tilde_map = weighted_tau_sum / weight_sum
            tau_tilde_map = tau_tilde_map.where(~zero_weight_groups, simple_means)
        else:
            tau_tilde_map = weighted_tau_sum / weight_sum

        tau_tilde = group_series.map(tau_tilde_map).values

        # Auxiliary residuals
        epsilon_treated = tau_hat - tau_tilde

        return epsilon_treated

    def _compute_residuals_untreated(
        self,
        df_0: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        covariates: Optional[List[str]],
        unit_fe: Dict[Any, float],
        time_fe: Dict[Any, float],
        grand_mean: float,
        delta_hat: Optional[np.ndarray],
    ) -> np.ndarray:
        """Compute Step 1 residuals for untreated observations."""
        alpha_i = df_0[unit].map(unit_fe).fillna(0.0).values
        beta_t = df_0[time].map(time_fe).fillna(0.0).values
        y_hat = grand_mean + alpha_i + beta_t

        if delta_hat is not None and covariates:
            y_hat = y_hat + np.dot(df_0[covariates].values, delta_hat)

        return df_0[outcome].values - y_hat

    # =========================================================================
    # Aggregation
    # =========================================================================

    def _aggregate_event_study(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        covariates: Optional[List[str]],
        omega_0_mask: pd.Series,
        omega_1_mask: pd.Series,
        unit_fe: Dict[Any, float],
        time_fe: Dict[Any, float],
        grand_mean: float,
        delta_hat: Optional[np.ndarray],
        cluster_var: str,
        treatment_groups: List[Any],
        balance_e: Optional[int] = None,
        kept_cov_mask: Optional[np.ndarray] = None,
        survey_weights: Optional[np.ndarray] = None,
        survey_df: Optional[int] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """Aggregate treatment effects by event-study horizon."""
        df_1 = df.loc[omega_1_mask]
        tau_hat = df["_tau_hat"].loc[omega_1_mask].values
        rel_times = df_1["_rel_time"].values

        # Get all horizons
        all_horizons = sorted(set(int(h) for h in rel_times if np.isfinite(h)))

        # Apply horizon_max filter
        if self.horizon_max is not None:
            all_horizons = [h for h in all_horizons if abs(h) <= self.horizon_max]

        # Apply balance_e filter
        if balance_e is not None:
            cohort_rel_times = self._build_cohort_rel_times(df, first_treat)
            balanced_mask = pd.Series(
                self._compute_balanced_cohort_mask(
                    df_1, first_treat, all_horizons, balance_e, cohort_rel_times
                ),
                index=df_1.index,
            )
        else:
            balanced_mask = pd.Series(True, index=df_1.index)

        # Check Proposition 5: no never-treated units
        has_never_treated = df["_never_treated"].any()
        h_bar = np.inf
        if not has_never_treated and len(treatment_groups) > 1:
            h_bar = max(treatment_groups) - min(treatment_groups)

        # Reference period
        ref_period = -1 - self.anticipation

        event_study_effects: Dict[int, Dict[str, Any]] = {}

        # Add reference period marker
        event_study_effects[ref_period] = {
            "effect": 0.0,
            "se": 0.0,
            "t_stat": np.nan,
            "p_value": np.nan,
            "conf_int": (0.0, 0.0),
            "n_obs": 0,
        }

        # Collect horizons with Proposition 5 violations
        prop5_horizons = []

        for h in all_horizons:
            if h == ref_period:
                continue

            # Select treated obs at this horizon from balanced cohorts
            h_mask = (rel_times == h) & balanced_mask.values
            n_h = int(h_mask.sum())

            if n_h == 0:
                continue

            # Proposition 5 check
            if not has_never_treated and h >= h_bar:
                prop5_horizons.append(h)
                event_study_effects[h] = {
                    "effect": np.nan,
                    "se": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "conf_int": (np.nan, np.nan),
                    "n_obs": n_h,
                }
                continue

            tau_h = tau_hat[h_mask]
            finite_h = np.isfinite(tau_h)
            valid_tau = tau_h[finite_h]

            if len(valid_tau) == 0:
                event_study_effects[h] = {
                    "effect": np.nan,
                    "se": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "conf_int": (np.nan, np.nan),
                    "n_obs": n_h,
                }
                continue

            # Survey-weighted or simple mean for per-horizon effect
            if survey_weights is not None:
                treated_sw = survey_weights[omega_1_mask.values]
                sw_h = treated_sw[h_mask]
                sw_valid = sw_h[finite_h]
                effect = float(np.average(valid_tau, weights=sw_valid))
            else:
                effect = float(np.mean(valid_tau))

            # Compute SE via conservative variance with horizon-specific weights
            # When survey, aggregation weights are proportional to survey weights
            if survey_weights is not None:
                treated_sw = survey_weights[omega_1_mask.values]
                n_1 = len(tau_hat)
                weights_h = np.zeros(n_1)
                sw_h = treated_sw[h_mask]
                finite_in_h = np.isfinite(tau_h)
                sw_finite = sw_h[finite_in_h]
                # Set weights proportional to survey weights, summing to 1
                if sw_finite.sum() > 0:
                    h_indices = np.where(h_mask)[0]
                    finite_indices = h_indices[finite_in_h]
                    weights_h[finite_indices] = sw_finite / sw_finite.sum()
                n_valid = int(finite_in_h.sum())
            else:
                weights_h, n_valid = _compute_target_weights(tau_hat, h_mask)

            se = self._compute_conservative_variance(
                df=df,
                outcome=outcome,
                unit=unit,
                time=time,
                first_treat=first_treat,
                covariates=covariates,
                omega_0_mask=omega_0_mask,
                omega_1_mask=omega_1_mask,
                unit_fe=unit_fe,
                time_fe=time_fe,
                grand_mean=grand_mean,
                delta_hat=delta_hat,
                weights=weights_h,
                cluster_var=cluster_var,
                kept_cov_mask=kept_cov_mask,
                survey_weights=survey_weights,
            )

            t_stat, p_value, conf_int = safe_inference(effect, se, alpha=self.alpha, df=survey_df)

            event_study_effects[h] = {
                "effect": effect,
                "se": se,
                "t_stat": t_stat,
                "p_value": p_value,
                "conf_int": conf_int,
                "n_obs": n_h,
            }

        # Proposition 5 warning
        if prop5_horizons:
            warnings.warn(
                f"Horizons {prop5_horizons} are not identified without "
                f"never-treated units (Proposition 5). Set to NaN.",
                UserWarning,
                stacklevel=3,
            )

        # Check for empty result set after filtering
        real_effects = [
            h for h, v in event_study_effects.items() if h != ref_period and v.get("n_obs", 0) > 0
        ]
        if len(real_effects) == 0:
            filter_info = []
            if balance_e is not None:
                filter_info.append(f"balance_e={balance_e}")
            if self.horizon_max is not None:
                filter_info.append(f"horizon_max={self.horizon_max}")
            filter_str = " and ".join(filter_info) if filter_info else "filters"
            warnings.warn(
                f"Event study aggregation produced no horizons with observations "
                f"after applying {filter_str}. The result contains only the "
                f"reference period marker. Consider relaxing filter parameters.",
                UserWarning,
                stacklevel=3,
            )

        return event_study_effects

    def _aggregate_group(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        covariates: Optional[List[str]],
        omega_0_mask: pd.Series,
        omega_1_mask: pd.Series,
        unit_fe: Dict[Any, float],
        time_fe: Dict[Any, float],
        grand_mean: float,
        delta_hat: Optional[np.ndarray],
        cluster_var: str,
        treatment_groups: List[Any],
        kept_cov_mask: Optional[np.ndarray] = None,
        survey_weights: Optional[np.ndarray] = None,
        survey_df: Optional[int] = None,
    ) -> Dict[Any, Dict[str, Any]]:
        """Aggregate treatment effects by cohort."""
        df_1 = df.loc[omega_1_mask]
        tau_hat = df["_tau_hat"].loc[omega_1_mask].values
        cohorts = df_1[first_treat].values

        group_effects: Dict[Any, Dict[str, Any]] = {}

        for g in treatment_groups:
            g_mask = cohorts == g
            n_g = int(g_mask.sum())

            if n_g == 0:
                continue

            tau_g = tau_hat[g_mask]
            finite_g = np.isfinite(tau_g)
            valid_tau = tau_g[finite_g]

            if len(valid_tau) == 0:
                group_effects[g] = {
                    "effect": np.nan,
                    "se": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "conf_int": (np.nan, np.nan),
                    "n_obs": n_g,
                }
                continue

            # Survey-weighted or simple mean for per-group effect
            if survey_weights is not None:
                treated_sw = survey_weights[omega_1_mask.values]
                sw_g = treated_sw[g_mask]
                sw_valid = sw_g[finite_g]
                effect = float(np.average(valid_tau, weights=sw_valid))
            else:
                effect = float(np.mean(valid_tau))

            # Compute SE with group-specific weights
            # When survey, aggregation weights proportional to survey weights
            if survey_weights is not None:
                treated_sw = survey_weights[omega_1_mask.values]
                n_1 = len(tau_hat)
                weights_g = np.zeros(n_1)
                sw_g = treated_sw[g_mask]
                sw_finite = sw_g[finite_g]
                if sw_finite.sum() > 0:
                    g_indices = np.where(g_mask)[0]
                    finite_indices = g_indices[finite_g]
                    weights_g[finite_indices] = sw_finite / sw_finite.sum()
            else:
                weights_g, _ = _compute_target_weights(tau_hat, g_mask)

            se = self._compute_conservative_variance(
                df=df,
                outcome=outcome,
                unit=unit,
                time=time,
                first_treat=first_treat,
                covariates=covariates,
                omega_0_mask=omega_0_mask,
                omega_1_mask=omega_1_mask,
                unit_fe=unit_fe,
                time_fe=time_fe,
                grand_mean=grand_mean,
                delta_hat=delta_hat,
                weights=weights_g,
                cluster_var=cluster_var,
                kept_cov_mask=kept_cov_mask,
                survey_weights=survey_weights,
            )

            t_stat, p_value, conf_int = safe_inference(effect, se, alpha=self.alpha, df=survey_df)

            group_effects[g] = {
                "effect": effect,
                "se": se,
                "t_stat": t_stat,
                "p_value": p_value,
                "conf_int": conf_int,
                "n_obs": n_g,
            }

        return group_effects

    # =========================================================================
    # Pre-trend test (Equation 9)
    # =========================================================================

    def _pretrend_test(self, n_leads: Optional[int] = None) -> Dict[str, Any]:
        """
        Run pre-trend test (Equation 9).

        Adds pre-treatment lead indicators to the Step 1 OLS on Omega_0
        and tests their joint significance via cluster-robust Wald F-test.
        """
        if self._fit_data is None:
            raise RuntimeError("Must call fit() before pretrend_test().")

        if self._fit_data.get("survey_design") is not None:
            raise NotImplementedError(
                "pretrend_test() is not yet survey-aware. The pre-trend F-test "
                "uses unweighted demeaning and cluster-count degrees of freedom, "
                "which do not account for survey weights. Survey-weighted "
                "pretrend_test() is planned for future work."
            )

        fd = self._fit_data
        df = fd["df"]
        outcome = fd["outcome"]
        unit = fd["unit"]
        time = fd["time"]
        first_treat = fd["first_treat"]
        covariates = fd["covariates"]
        omega_0_mask = fd["omega_0_mask"]
        cluster_var = fd["cluster_var"]

        df_0 = df.loc[omega_0_mask].copy()

        # Compute relative time for untreated obs
        # For not-yet-treated units in their pre-treatment periods
        rel_time_0 = np.where(
            ~df_0["_never_treated"],
            df_0[time] - df_0[first_treat],
            np.nan,
        )

        # Get available pre-treatment relative times (negative values)
        pre_rel_times = sorted(
            set(int(h) for h in rel_time_0 if np.isfinite(h) and h < -self.anticipation)
        )

        if len(pre_rel_times) == 0:
            return {
                "f_stat": np.nan,
                "p_value": np.nan,
                "df": 0,
                "n_leads": 0,
                "lead_coefficients": {},
            }

        # Exclude the reference period (last pre-treatment period)
        ref = -1 - self.anticipation
        pre_rel_times = [h for h in pre_rel_times if h != ref]

        if n_leads is not None:
            # Take the n_leads periods closest to treatment
            pre_rel_times = sorted(pre_rel_times, reverse=True)[:n_leads]
            pre_rel_times = sorted(pre_rel_times)

        if len(pre_rel_times) == 0:
            return {
                "f_stat": np.nan,
                "p_value": np.nan,
                "df": 0,
                "n_leads": 0,
                "lead_coefficients": {},
            }

        # Build lead indicators
        lead_cols = []
        for h in pre_rel_times:
            col_name = f"_lead_{h}"
            df_0[col_name] = ((rel_time_0 == h)).astype(float)
            lead_cols.append(col_name)

        # Within-transform via iterative demeaning (exact for unbalanced panels)
        y_dm = self._iterative_demean(
            df_0[outcome].values, df_0[unit].values, df_0[time].values, df_0.index
        )

        all_x_cols = lead_cols[:]
        if covariates:
            all_x_cols.extend(covariates)

        X_dm = np.column_stack(
            [
                self._iterative_demean(
                    df_0[col].values, df_0[unit].values, df_0[time].values, df_0.index
                )
                for col in all_x_cols
            ]
        )

        # OLS with cluster-robust SEs
        cluster_ids = df_0[cluster_var].values
        result = solve_ols(
            X_dm,
            y_dm,
            cluster_ids=cluster_ids,
            return_vcov=True,
            rank_deficient_action=self.rank_deficient_action,
            column_names=all_x_cols,
        )
        coefficients = result[0]
        vcov = result[2]
        assert vcov is not None

        # Extract lead coefficients and their sub-VCV
        n_leads_actual = len(lead_cols)
        gamma = coefficients[:n_leads_actual]
        V_gamma = vcov[:n_leads_actual, :n_leads_actual]

        # Wald F-test: F = (gamma' V^{-1} gamma) / n_leads
        try:
            V_inv_gamma = np.linalg.solve(V_gamma, gamma)
            wald_stat = float(gamma @ V_inv_gamma)
            f_stat = wald_stat / n_leads_actual
        except np.linalg.LinAlgError:
            f_stat = np.nan

        # P-value from F distribution
        if np.isfinite(f_stat) and f_stat >= 0:
            n_clusters = len(np.unique(cluster_ids))
            df_denom = max(n_clusters - 1, 1)
            p_value = float(stats.f.sf(f_stat, n_leads_actual, df_denom))
        else:
            p_value = np.nan

        # Store lead coefficients
        lead_coefficients = {}
        for j, h in enumerate(pre_rel_times):
            lead_coefficients[h] = float(gamma[j])

        return {
            "f_stat": f_stat,
            "p_value": p_value,
            "df": n_leads_actual,
            "n_leads": n_leads_actual,
            "lead_coefficients": lead_coefficients,
        }

    # =========================================================================
    # sklearn-compatible interface
    # =========================================================================

    def get_params(self) -> Dict[str, Any]:
        """Get estimator parameters (sklearn-compatible)."""
        return {
            "anticipation": self.anticipation,
            "alpha": self.alpha,
            "cluster": self.cluster,
            "n_bootstrap": self.n_bootstrap,
            "bootstrap_weights": self.bootstrap_weights,
            "seed": self.seed,
            "rank_deficient_action": self.rank_deficient_action,
            "horizon_max": self.horizon_max,
            "aux_partition": self.aux_partition,
        }

    def set_params(self, **params) -> "ImputationDiD":
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


# =============================================================================
# Convenience function
# =============================================================================


def imputation_did(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    first_treat: str,
    covariates: Optional[List[str]] = None,
    aggregate: Optional[str] = None,
    balance_e: Optional[int] = None,
    survey_design: object = None,
    **kwargs,
) -> ImputationDiDResults:
    """
    Convenience function for imputation DiD estimation.

    This is a shortcut for creating an ImputationDiD estimator and calling fit().

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    outcome : str
        Outcome variable column name.
    unit : str
        Unit identifier column name.
    time : str
        Time period column name.
    first_treat : str
        Column indicating first treatment period (0 for never-treated).
    covariates : list of str, optional
        Covariate column names.
    aggregate : str, optional
        Aggregation mode: None, "simple", "event_study", "group", "all".
    balance_e : int, optional
        Balance event study to cohorts observed at all relative times.
    survey_design : SurveyDesign, optional
        Survey design specification for design-based inference. Supports
        pweight only (aweight/fweight raise ValueError). FPC raises
        NotImplementedError. PSU is used as cluster variable for Theorem 3
        variance. Strata enters survey df for t-distribution inference.
        Both analytical (n_bootstrap=0) and bootstrap inference are supported.
    **kwargs
        Additional keyword arguments passed to ImputationDiD constructor.

    Returns
    -------
    ImputationDiDResults
        Estimation results.

    Examples
    --------
    >>> from diff_diff import imputation_did, generate_staggered_data
    >>> data = generate_staggered_data(seed=42)
    >>> results = imputation_did(data, 'outcome', 'unit', 'time', 'first_treat',
    ...                          aggregate='event_study')
    >>> results.print_summary()
    """
    est = ImputationDiD(**kwargs)
    return est.fit(
        data,
        outcome=outcome,
        unit=unit,
        time=time,
        first_treat=first_treat,
        covariates=covariates,
        aggregate=aggregate,
        balance_e=balance_e,
        survey_design=survey_design,
    )
