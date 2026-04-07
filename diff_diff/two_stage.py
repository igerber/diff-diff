"""
Gardner (2022) Two-Stage Difference-in-Differences Estimator.

Implements the two-stage DiD estimator from Gardner (2022), "Two-stage
differences in differences". The method:
1. Estimates unit + time fixed effects on untreated observations only
2. Residualizes ALL outcomes using estimated FEs
3. Regresses residualized outcomes on treatment indicators (Stage 2)

Inference uses the GMM sandwich variance estimator from Butts & Gardner
(2022) that correctly accounts for first-stage estimation uncertainty.

Point estimates are identical to ImputationDiD (Borusyak et al. 2024);
the key difference is the variance estimator (GMM sandwich vs. conservative).

References
----------
Gardner, J. (2022). Two-stage differences in differences.
    arXiv:2207.05943.
Butts, K. & Gardner, J. (2022). did2s: Two-Stage
    Difference-in-Differences. R Journal, 14(1), 162-173.
"""

import warnings
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import factorized as sparse_factorized

# Maximum number of elements before falling back to per-column sparse aggregation.
# 10M float64 elements ≈ 80 MB peak allocation. Above this, per-column .getcol()
# trades throughput for bounded memory. Keep in sync with two_stage_bootstrap.py.
_SPARSE_DENSE_THRESHOLD = 10_000_000

from diff_diff.linalg import solve_ols
from diff_diff.two_stage_bootstrap import TwoStageDiDBootstrapMixin
from diff_diff.two_stage_results import (
    TwoStageBootstrapResults,  # noqa: F401
    TwoStageDiDResults,
)  # noqa: F401 (re-export)
from diff_diff.utils import safe_inference

# =============================================================================
# Main Estimator
# =============================================================================


class TwoStageDiD(TwoStageDiDBootstrapMixin):
    """
    Gardner (2022) two-stage Difference-in-Differences estimator.

    This estimator addresses TWFE bias under heterogeneous treatment
    effects by:
    1. Estimating unit + time FEs on untreated observations only
    2. Residualizing ALL outcomes using estimated FEs
    3. Regressing residualized outcomes on treatment indicators

    Point estimates are identical to ImputationDiD (Borusyak et al. 2024).
    The key difference is the variance estimator: TwoStageDiD uses a GMM
    sandwich variance that accounts for first-stage estimation uncertainty,
    while ImputationDiD uses the conservative variance from Theorem 3.

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
        Number of bootstrap iterations. If 0, uses analytical GMM
        sandwich inference.
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

    Attributes
    ----------
    results_ : TwoStageDiDResults
        Estimation results after calling fit().
    is_fitted_ : bool
        Whether the model has been fitted.

    Examples
    --------
    Basic usage:

    >>> from diff_diff import TwoStageDiD, generate_staggered_data
    >>> data = generate_staggered_data(n_units=200, seed=42)
    >>> est = TwoStageDiD()
    >>> results = est.fit(data, outcome='outcome', unit='unit',
    ...                   time='period', first_treat='first_treat')
    >>> results.print_summary()

    With event study:

    >>> est = TwoStageDiD()
    >>> results = est.fit(data, outcome='outcome', unit='unit',
    ...                   time='period', first_treat='first_treat',
    ...                   aggregate='event_study')
    >>> from diff_diff import plot_event_study
    >>> plot_event_study(results)

    Notes
    -----
    The two-stage estimator uses ALL untreated observations (never-treated +
    not-yet-treated periods of eventually-treated units) to estimate the
    counterfactual model.

    References
    ----------
    Gardner, J. (2022). Two-stage differences in differences.
        arXiv:2207.05943.
    Butts, K. & Gardner, J. (2022). did2s: Two-Stage
        Difference-in-Differences. R Journal, 14(1), 162-173.
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

        self.anticipation = anticipation
        self.alpha = alpha
        self.cluster = cluster
        self.n_bootstrap = n_bootstrap
        self.bootstrap_weights = bootstrap_weights
        self.seed = seed
        self.rank_deficient_action = rank_deficient_action
        self.horizon_max = horizon_max

        self.is_fitted_ = False
        self.results_: Optional[TwoStageDiDResults] = None

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
    ) -> TwoStageDiDResults:
        """
        Fit the two-stage DiD estimator.

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
        TwoStageDiDResults
            Object containing all estimation results.

        Raises
        ------
        ValueError
            If required columns are missing or data validation fails.
        """
        # ---- Data validation ----
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

        resolved_survey, survey_weights, survey_weight_type, survey_metadata = (
            _resolve_survey_for_fit(survey_design, data, "analytical")
        )

        # Validate within-unit constancy for panel survey designs
        if resolved_survey is not None:
            if resolved_survey.uses_replicate_variance:
                raise NotImplementedError(
                    "TwoStageDiD does not yet support replicate-weight survey "
                    "designs. Use a TSL-based survey design (strata/psu/fpc)."
                )
            _validate_unit_constant_survey(data, unit, survey_design)
            if resolved_survey.weight_type != "pweight":
                raise ValueError(
                    f"TwoStageDiD survey support requires weight_type='pweight', "
                    f"got '{resolved_survey.weight_type}'. The survey variance math "
                    f"assumes probability weights (pweight)."
                )
            if resolved_survey.fpc is not None:
                raise NotImplementedError(
                    "TwoStageDiD does not yet support FPC (finite population "
                    "correction) in SurveyDesign. Weights, strata (for survey df), "
                    "and PSU (for cluster-robust variance) are supported."
                )

        # Bootstrap + survey supported via PSU-level multiplier bootstrap.

        df[time] = pd.to_numeric(df[time])
        df[first_treat] = pd.to_numeric(df[first_treat])

        # Validate absorbing treatment
        ft_nunique = df.groupby(unit)[first_treat].nunique()
        non_constant = ft_nunique[ft_nunique > 1]
        if len(non_constant) > 0:
            example_unit = non_constant.index[0]
            example_vals = sorted(df.loc[df[unit] == example_unit, first_treat].unique())
            warnings.warn(
                f"{len(non_constant)} unit(s) have non-constant '{first_treat}' "
                f"values (e.g., unit '{example_unit}' has values {example_vals}). "
                f"TwoStageDiD assumes treatment is an absorbing state "
                f"(once treated, always treated) with a single treatment onset "
                f"time per unit. Non-constant first_treat violates this assumption "
                f"and may produce unreliable estimates.",
                UserWarning,
                stacklevel=2,
            )
            df[first_treat] = df.groupby(unit)[first_treat].transform("first")

        # Identify treatment status
        df["_never_treated"] = (df[first_treat] == 0) | (df[first_treat] == np.inf)

        # Check for always-treated units
        min_time = df[time].min()
        always_treated_mask = (~df["_never_treated"]) & (df[first_treat] <= min_time)
        always_treated_units = df.loc[always_treated_mask, unit].unique()
        n_always_treated = len(always_treated_units)
        if n_always_treated > 0:
            unit_list = ", ".join(str(u) for u in always_treated_units[:10])
            suffix = f" (and {n_always_treated - 10} more)" if n_always_treated > 10 else ""
            warnings.warn(
                f"{n_always_treated} unit(s) are treated in all observed periods "
                f"(first_treat <= {min_time}): [{unit_list}{suffix}]. "
                "These units have no untreated observations and cannot contribute "
                "to the counterfactual model. Excluding from estimation.",
                UserWarning,
                stacklevel=2,
            )
            df = df[~df[unit].isin(always_treated_units)].copy()

            # Subset survey arrays to match filtered df
            if survey_weights is not None:
                keep_mask = ~data[unit].isin(always_treated_units)
                survey_weights = survey_weights[keep_mask.values]
            if resolved_survey is not None:
                keep_mask = ~data[unit].isin(always_treated_units)
                resolved_survey = replace(
                    resolved_survey,
                    weights=resolved_survey.weights[keep_mask.values],
                    strata=(
                        resolved_survey.strata[keep_mask.values]
                        if resolved_survey.strata is not None
                        else None
                    ),
                    psu=(
                        resolved_survey.psu[keep_mask.values]
                        if resolved_survey.psu is not None
                        else None
                    ),
                    fpc=(
                        resolved_survey.fpc[keep_mask.values]
                        if resolved_survey.fpc is not None
                        else None
                    ),
                )
                # Recompute n_psu/n_strata after subsetting
                new_n_psu = (
                    len(np.unique(resolved_survey.psu)) if resolved_survey.psu is not None else 0
                )
                new_n_strata = (
                    len(np.unique(resolved_survey.strata))
                    if resolved_survey.strata is not None
                    else 0
                )
                resolved_survey = replace(resolved_survey, n_psu=new_n_psu, n_strata=new_n_strata)
                # Recompute survey_metadata since it depends on these counts
                from diff_diff.survey import compute_survey_metadata

                raw_w = (
                    df[survey_design.weights].values.astype(np.float64)
                    if survey_design.weights
                    else np.ones(len(df), dtype=np.float64)
                )
                survey_metadata = compute_survey_metadata(resolved_survey, raw_w)

        # Treatment indicator with anticipation
        effective_treat = df[first_treat] - self.anticipation
        df["_treated"] = (~df["_never_treated"]) & (df[time] >= effective_treat)

        # Partition into Omega_0 (untreated) and Omega_1 (treated)
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

        # Groups and time periods
        time_periods = sorted(df[time].unique())
        treatment_groups = sorted([g for g in df[first_treat].unique() if g > 0 and g != np.inf])

        if len(treatment_groups) == 0:
            raise ValueError("No treated units found. Check 'first_treat' column.")

        # Unit info
        unit_info = (
            df.groupby(unit).agg({first_treat: "first", "_never_treated": "first"}).reset_index()
        )
        n_treated_units = int((~unit_info["_never_treated"]).sum())
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
            # GMM variance (PSU overrides unit-level clustering)
            if resolved_survey.psu is not None:
                df["_survey_cluster"] = resolved_survey.psu
                cluster_var = "_survey_cluster"
            # Recompute metadata after PSU injection
            if resolved_survey.psu is not None and survey_metadata is not None:
                from diff_diff.survey import compute_survey_metadata

                raw_w = (
                    df[survey_design.weights].values.astype(np.float64)
                    if survey_design.weights
                    else np.ones(len(df), dtype=np.float64)
                )
                survey_metadata = compute_survey_metadata(resolved_survey, raw_w)

        # Relative time
        df["_rel_time"] = np.where(
            ~df["_never_treated"],
            df[time] - df[first_treat],
            np.nan,
        )

        # ---- Stage 1: OLS on untreated observations ----
        unit_fe, time_fe, grand_mean, delta_hat, kept_cov_mask = self._fit_untreated_model(
            df, outcome, unit, time, covariates, omega_0_mask, weights=survey_weights
        )

        # ---- Rank condition checks ----
        treated_unit_ids = df.loc[omega_1_mask, unit].unique()
        units_with_fe = set(unit_fe.keys())
        units_missing_fe = set(treated_unit_ids) - units_with_fe

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

        # ---- Residualize ALL observations ----
        y_tilde = self._residualize(
            df, outcome, unit, time, covariates, unit_fe, time_fe, grand_mean, delta_hat
        )
        df["_y_tilde"] = y_tilde

        # ---- Stage 2: OLS of y_tilde on treatment indicators ----
        # Build design matrices and compute effects + GMM variance
        ref_period = -1 - self.anticipation

        # Survey degrees of freedom for t-distribution inference
        _survey_df = resolved_survey.df_survey if resolved_survey is not None else None

        # Always compute overall ATT (static specification)
        overall_att, overall_se = self._stage2_static(
            df=df,
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
            survey_weights=survey_weights,
            survey_weight_type=survey_weight_type,
        )

        overall_t, overall_p, overall_ci = safe_inference(
            overall_att, overall_se, alpha=self.alpha, df=_survey_df
        )

        # Event study and group aggregation
        event_study_effects = None
        group_effects = None

        if aggregate in ("event_study", "all"):
            event_study_effects = self._stage2_event_study(
                df=df,
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
                ref_period=ref_period,
                balance_e=balance_e,
                kept_cov_mask=kept_cov_mask,
                survey_weights=survey_weights,
                survey_weight_type=survey_weight_type,
                survey_df=_survey_df,
            )

        if aggregate in ("group", "all"):
            group_effects = self._stage2_group(
                df=df,
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
                survey_weight_type=survey_weight_type,
                survey_df=_survey_df,
            )

        # Build treatment effects DataFrame
        treated_df = df.loc[omega_1_mask, [unit, time, "_y_tilde", "_rel_time"]].copy()
        treated_df = treated_df.rename(columns={"_y_tilde": "tau_hat", "_rel_time": "rel_time"})
        tau_finite = treated_df["tau_hat"].notna() & np.isfinite(treated_df["tau_hat"].values)
        n_valid_te = int(tau_finite.sum())
        if n_valid_te > 0:
            if survey_weights is not None:
                treated_sw = survey_weights[omega_1_mask.values]
                sw_finite = np.where(tau_finite, treated_sw, 0.0)
                sw_sum = sw_finite.sum()
                treated_df["weight"] = sw_finite / sw_sum if sw_sum > 0 else 0.0
            else:
                treated_df["weight"] = np.where(tau_finite, 1.0 / n_valid_te, 0.0)
        else:
            treated_df["weight"] = 0.0

        # ---- Bootstrap ----
        bootstrap_results = None
        if self.n_bootstrap > 0:
            try:
                bootstrap_results = self._run_bootstrap(
                    df=df,
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
                    treatment_groups=treatment_groups,
                    ref_period=ref_period,
                    balance_e=balance_e,
                    original_att=overall_att,
                    original_event_study=event_study_effects,
                    original_group=group_effects,
                    aggregate=aggregate,
                    resolved_survey=resolved_survey,
                )
            except NotImplementedError:
                raise  # Don't swallow explicit rejections (e.g. lonely_psu="adjust")
            except Exception as e:
                warnings.warn(
                    f"Bootstrap failed: {e}. Skipping bootstrap inference.",
                    UserWarning,
                    stacklevel=2,
                )

            if bootstrap_results is not None:
                # Update inference with bootstrap results
                overall_se = bootstrap_results.overall_att_se
                overall_t = (
                    overall_att / overall_se
                    if np.isfinite(overall_se) and overall_se > 0
                    else np.nan
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
                            event_study_effects[h]["conf_int"] = bootstrap_results.event_study_cis[
                                h
                            ]
                            assert bootstrap_results.event_study_p_values is not None
                            event_study_effects[h]["p_value"] = (
                                bootstrap_results.event_study_p_values[h]
                            )
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
        self.results_ = TwoStageDiDResults(
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
            survey_metadata=survey_metadata,
        )

        self.is_fitted_ = True
        return self.results_

    # =========================================================================
    # Stage 1: OLS on untreated observations
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
        Estimate unit and time FE via iterative alternating projection.

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
        alpha = np.zeros(n)
        beta = np.zeros(n)

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

        Parameters
        ----------
        weights : np.ndarray, optional
            Survey weights. When provided, uses weighted group means
            (sum(w*x)/sum(w)) instead of unweighted means.
        """
        result = vals.copy()

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
        Stage 1: Estimate unit + time FE on untreated observations.

        Parameters
        ----------
        weights : np.ndarray, optional
            Full-panel survey weights (same length as df). The untreated subset
            is extracted internally via omega_0_mask. When None, unweighted.

        Returns
        -------
        unit_fe, time_fe, grand_mean, delta_hat, kept_cov_mask
        """
        df_0 = df.loc[omega_0_mask]
        w_0 = weights[omega_0_mask.values] if weights is not None else None

        if covariates is None or len(covariates) == 0:
            y = df_0[outcome].values.copy()
            unit_fe, time_fe = self._iterative_fe(
                y, df_0[unit].values, df_0[time].values, df_0.index, weights=w_0
            )
            return unit_fe, time_fe, 0.0, None, None

        else:
            y = df_0[outcome].values.copy()
            X_raw = df_0[covariates].values.copy()
            units = df_0[unit].values
            times = df_0[time].values
            n_cov = len(covariates)

            y_dm = self._iterative_demean(y, units, times, df_0.index, weights=w_0)
            X_dm = np.column_stack(
                [
                    self._iterative_demean(X_raw[:, j], units, times, df_0.index, weights=w_0)
                    for j in range(n_cov)
                ]
            )

            result = solve_ols(
                X_dm,
                y_dm,
                return_vcov=False,
                rank_deficient_action=self.rank_deficient_action,
                column_names=covariates,
                weights=w_0,
            )
            delta_hat = result[0]
            kept_cov_mask = np.isfinite(delta_hat)
            delta_hat_clean = np.where(np.isfinite(delta_hat), delta_hat, 0.0)

            y_adj = y - np.dot(X_raw, delta_hat_clean)
            unit_fe, time_fe = self._iterative_fe(y_adj, units, times, df_0.index, weights=w_0)

            return unit_fe, time_fe, 0.0, delta_hat_clean, kept_cov_mask

    # =========================================================================
    # Residualization
    # =========================================================================

    def _residualize(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        covariates: Optional[List[str]],
        unit_fe: Dict[Any, float],
        time_fe: Dict[Any, float],
        grand_mean: float,
        delta_hat: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Compute residualized outcome y_tilde for ALL observations.

        y_tilde_i = y_i - mu_hat_i - eta_hat_t [- X_i @ delta_hat]
        """
        alpha_i = df[unit].map(unit_fe).values
        beta_t = df[time].map(time_fe).values

        # Handle missing FE (NaN for units/periods not in untreated sample)
        alpha_i = np.where(pd.isna(alpha_i), np.nan, alpha_i).astype(float)
        beta_t = np.where(pd.isna(beta_t), np.nan, beta_t).astype(float)

        y_hat = grand_mean + alpha_i + beta_t

        if delta_hat is not None and covariates:
            y_hat = y_hat + np.dot(df[covariates].values, delta_hat)

        y_tilde = df[outcome].values - y_hat
        return y_tilde

    # =========================================================================
    # Stage 2 specifications
    # =========================================================================

    def _stage2_static(
        self,
        df: pd.DataFrame,
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
        kept_cov_mask: Optional[np.ndarray],
        survey_weights: Optional[np.ndarray] = None,
        survey_weight_type: str = "pweight",
    ) -> Tuple[float, float]:
        """
        Static (simple ATT) Stage 2: OLS of y_tilde on D_it.

        Returns (att, se).
        """
        y_tilde = df["_y_tilde"].values.copy()

        # Handle NaN y_tilde (from unidentified FEs — e.g., rank condition violations)
        # Set to 0 so solve_ols doesn't reject; these obs have X_2=0 (untreated)
        # or contribute NaN treatment effects (excluded from point estimate).
        nan_mask = ~np.isfinite(y_tilde)
        if nan_mask.any():
            y_tilde[nan_mask] = 0.0

        D = omega_1_mask.values.astype(float)
        # Zero out treatment indicator for NaN y_tilde obs (don't count in ATT)
        D[nan_mask] = 0.0

        # X_2: treatment indicator (no intercept)
        X_2 = D.reshape(-1, 1)

        # Avoid degenerate case where all treated obs have NaN y_tilde
        if D.sum() == 0:
            return np.nan, np.nan

        # Stage 2 OLS for point estimate (discard naive SE)
        coef, residuals, _ = solve_ols(
            X_2,
            y_tilde,
            return_vcov=False,
            weights=survey_weights,
            weight_type=survey_weight_type,
        )
        att = float(coef[0])

        # GMM sandwich variance
        eps_2 = y_tilde - np.dot(X_2, coef)  # Stage 2 residuals

        V = self._compute_gmm_variance(
            df=df,
            unit=unit,
            time=time,
            covariates=covariates,
            omega_0_mask=omega_0_mask,
            unit_fe=unit_fe,
            time_fe=time_fe,
            delta_hat=delta_hat,
            kept_cov_mask=kept_cov_mask,
            X_2=X_2,
            eps_2=eps_2,
            cluster_ids=df[cluster_var].values,
            survey_weights=survey_weights,
        )

        se = float(np.sqrt(max(V[0, 0], 0.0)))
        return att, se

    def _stage2_event_study(
        self,
        df: pd.DataFrame,
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
        ref_period: int,
        balance_e: Optional[int],
        kept_cov_mask: Optional[np.ndarray],
        survey_weights: Optional[np.ndarray] = None,
        survey_weight_type: str = "pweight",
        survey_df: Optional[int] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """Event study Stage 2: OLS of y_tilde on relative-time dummies."""
        y_tilde = df["_y_tilde"].values.copy()
        # Handle NaN y_tilde (unidentified FEs)
        nan_mask = ~np.isfinite(y_tilde)
        if nan_mask.any():
            y_tilde[nan_mask] = 0.0
        rel_times = df["_rel_time"].values
        n = len(df)

        # Get all horizons from treated observations
        treated_rel = rel_times[omega_1_mask.values]
        all_horizons = sorted(set(int(h) for h in treated_rel if np.isfinite(h)))

        # Apply horizon_max filter
        if self.horizon_max is not None:
            all_horizons = [h for h in all_horizons if abs(h) <= self.horizon_max]

        # Apply balance_e filter
        if balance_e is not None:
            cohort_rel_times = self._build_cohort_rel_times(df, first_treat)
            balanced_cohorts = set()
            if all_horizons:
                max_h = max(all_horizons)
                required_range = set(range(-balance_e, max_h + 1))
                for g, horizons in cohort_rel_times.items():
                    if required_range.issubset(horizons):
                        balanced_cohorts.add(g)
            if not balanced_cohorts:
                warnings.warn(
                    f"No cohorts satisfy balance_e={balance_e} requirement. "
                    "Event study results will contain only the reference period. "
                    "Consider reducing balance_e.",
                    UserWarning,
                    stacklevel=2,
                )
                return {
                    ref_period: {
                        "effect": 0.0,
                        "se": 0.0,
                        "t_stat": np.nan,
                        "p_value": np.nan,
                        "conf_int": (0.0, 0.0),
                        "n_obs": 0,
                    }
                }
            balance_mask = df[first_treat].isin(balanced_cohorts).values
        else:
            balance_mask = np.ones(n, dtype=bool)

        # Check Proposition 5: no never-treated units
        has_never_treated = df["_never_treated"].any()
        h_bar = np.inf
        if not has_never_treated and len(treatment_groups) > 1:
            h_bar = max(treatment_groups) - min(treatment_groups)

        # Identify Prop 5 horizons and compute their actual treated obs counts.
        # Treated obs have NaN y_tilde at these horizons (counterfactual
        # unidentified), but actual_n counts them to distinguish from truly
        # empty horizons. rel_times is NaN for untreated/never-treated obs
        # (line ~653), so (rel_times == h) is False for them.
        prop5_horizons = []
        prop5_effects: Dict[int, Dict[str, Any]] = {}
        if h_bar < np.inf:
            for h in all_horizons:
                if h == ref_period:
                    continue
                if h >= h_bar:
                    actual_n = int(np.sum((rel_times == h) & omega_1_mask.values & balance_mask))
                    if actual_n > 0:
                        prop5_horizons.append(h)
                        prop5_effects[h] = {
                            "effect": np.nan,
                            "se": np.nan,
                            "t_stat": np.nan,
                            "p_value": np.nan,
                            "conf_int": (np.nan, np.nan),
                            "n_obs": actual_n,
                        }

        # Remove reference period AND Prop 5 horizons from estimation
        prop5_set = set(prop5_horizons)
        est_horizons = [h for h in all_horizons if h != ref_period and h not in prop5_set]

        if len(est_horizons) == 0:
            # No horizons to estimate — return just reference period
            return {
                ref_period: {
                    "effect": 0.0,
                    "se": 0.0,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "conf_int": (0.0, 0.0),
                    "n_obs": 0,
                }
            }

        # Build Stage 2 design: one column per horizon (no intercept)
        # Never-treated obs get all-zero rows (undefined relative time -> NaN)
        # With no intercept, they contribute zero to X'_2 X_2 and X'_2 y_tilde
        horizon_to_col = {h: j for j, h in enumerate(est_horizons)}
        k = len(est_horizons)
        X_2 = np.zeros((n, k))

        for i in range(n):
            if not balance_mask[i]:
                continue
            if nan_mask[i]:
                continue  # NaN y_tilde -> don't include in event study
            h = rel_times[i]
            if np.isfinite(h):
                h_int = int(h)
                if h_int in horizon_to_col:
                    X_2[i, horizon_to_col[h_int]] = 1.0

        # Stage 2 OLS
        coef, residuals, _ = solve_ols(
            X_2,
            y_tilde,
            return_vcov=False,
            weights=survey_weights,
            weight_type=survey_weight_type,
        )
        eps_2 = y_tilde - np.dot(X_2, coef)

        # GMM variance for full coefficient vector
        V = self._compute_gmm_variance(
            df=df,
            unit=unit,
            time=time,
            covariates=covariates,
            omega_0_mask=omega_0_mask,
            unit_fe=unit_fe,
            time_fe=time_fe,
            delta_hat=delta_hat,
            kept_cov_mask=kept_cov_mask,
            X_2=X_2,
            eps_2=eps_2,
            cluster_ids=df[cluster_var].values,
            survey_weights=survey_weights,
        )

        # Build results dict
        event_study_effects: Dict[int, Dict[str, Any]] = {}

        # Reference period marker
        event_study_effects[ref_period] = {
            "effect": 0.0,
            "se": 0.0,
            "t_stat": np.nan,
            "p_value": np.nan,
            "conf_int": (0.0, 0.0),
            "n_obs": 0,
        }

        for h in est_horizons:
            j = horizon_to_col[h]
            n_obs = int(np.sum(X_2[:, j]))

            if n_obs == 0:
                event_study_effects[h] = {
                    "effect": np.nan,
                    "se": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "conf_int": (np.nan, np.nan),
                    "n_obs": 0,
                }
                continue

            effect = float(coef[j])
            se = float(np.sqrt(max(V[j, j], 0.0)))

            t_stat, p_val, ci = safe_inference(effect, se, alpha=self.alpha, df=survey_df)

            event_study_effects[h] = {
                "effect": effect,
                "se": se,
                "t_stat": t_stat,
                "p_value": p_val,
                "conf_int": ci,
                "n_obs": n_obs,
            }

        # Add Proposition 5 entries (unidentified horizons with n_obs > 0)
        event_study_effects.update(prop5_effects)

        if prop5_horizons:
            warnings.warn(
                f"Horizons {prop5_horizons} are not identified without "
                f"never-treated units (Proposition 5). Set to NaN.",
                UserWarning,
                stacklevel=2,
            )

        return event_study_effects

    def _stage2_group(
        self,
        df: pd.DataFrame,
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
        kept_cov_mask: Optional[np.ndarray],
        survey_weights: Optional[np.ndarray] = None,
        survey_weight_type: str = "pweight",
        survey_df: Optional[int] = None,
    ) -> Dict[Any, Dict[str, Any]]:
        """Group (cohort) Stage 2: OLS of y_tilde on cohort dummies."""
        y_tilde = df["_y_tilde"].values.copy()
        nan_mask = ~np.isfinite(y_tilde)
        if nan_mask.any():
            y_tilde[nan_mask] = 0.0
        n = len(df)

        # Build Stage 2 design: one column per cohort (no intercept)
        group_to_col = {g: j for j, g in enumerate(treatment_groups)}
        k = len(treatment_groups)
        X_2 = np.zeros((n, k))

        ft_vals = df[first_treat].values
        treated_mask = omega_1_mask.values
        for i in range(n):
            if treated_mask[i] and not nan_mask[i]:
                g = ft_vals[i]
                if g in group_to_col:
                    X_2[i, group_to_col[g]] = 1.0

        # Stage 2 OLS
        coef, residuals, _ = solve_ols(
            X_2,
            y_tilde,
            return_vcov=False,
            weights=survey_weights,
            weight_type=survey_weight_type,
        )
        eps_2 = y_tilde - np.dot(X_2, coef)

        # GMM variance
        V = self._compute_gmm_variance(
            df=df,
            unit=unit,
            time=time,
            covariates=covariates,
            omega_0_mask=omega_0_mask,
            unit_fe=unit_fe,
            time_fe=time_fe,
            delta_hat=delta_hat,
            kept_cov_mask=kept_cov_mask,
            X_2=X_2,
            eps_2=eps_2,
            cluster_ids=df[cluster_var].values,
            survey_weights=survey_weights,
        )

        group_effects: Dict[Any, Dict[str, Any]] = {}
        for g in treatment_groups:
            j = group_to_col[g]
            n_obs = int(np.sum(X_2[:, j]))

            if n_obs == 0:
                group_effects[g] = {
                    "effect": np.nan,
                    "se": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "conf_int": (np.nan, np.nan),
                    "n_obs": 0,
                }
                continue

            effect = float(coef[j])
            se = float(np.sqrt(max(V[j, j], 0.0)))

            t_stat, p_val, ci = safe_inference(effect, se, alpha=self.alpha, df=survey_df)

            group_effects[g] = {
                "effect": effect,
                "se": se,
                "t_stat": t_stat,
                "p_value": p_val,
                "conf_int": ci,
                "n_obs": n_obs,
            }

        return group_effects

    # =========================================================================
    # GMM score computation
    # =========================================================================

    @staticmethod
    def _compute_gmm_scores(
        c_by_cluster: np.ndarray,
        gamma_hat: np.ndarray,
        s2_by_cluster: np.ndarray,
    ) -> np.ndarray:
        """
        Compute per-cluster GMM scores S_g = gamma_hat' c_g - X'_{2g} eps_{2g}.

        Handles NaN/overflow from rank-deficient FE by wrapping in errstate
        and replacing non-finite values with 0.

        Parameters
        ----------
        c_by_cluster : np.ndarray, shape (G, p)
            Per-cluster Stage 1 scores.
        gamma_hat : np.ndarray, shape (p, k)
            Cross-moment correction matrix.
        s2_by_cluster : np.ndarray, shape (G, k)
            Per-cluster Stage 2 scores.

        Returns
        -------
        np.ndarray, shape (G, k)
            Per-cluster influence scores.
        """
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            correction = np.dot(c_by_cluster, gamma_hat)
        np.nan_to_num(correction, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return correction - s2_by_cluster

    # =========================================================================
    # GMM Sandwich Variance (Butts & Gardner 2022)
    # =========================================================================

    def _compute_gmm_variance(
        self,
        df: pd.DataFrame,
        unit: str,
        time: str,
        covariates: Optional[List[str]],
        omega_0_mask: pd.Series,
        unit_fe: Dict[Any, float],
        time_fe: Dict[Any, float],
        delta_hat: Optional[np.ndarray],
        kept_cov_mask: Optional[np.ndarray],
        X_2: np.ndarray,
        eps_2: np.ndarray,
        cluster_ids: np.ndarray,
        survey_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute GMM sandwich variance (Butts & Gardner 2022).

        Matches the R `did2s` source code implementation: uses the GLOBAL
        Hessian inverse (not per-cluster) and NO finite-sample adjustments.

        The per-observation influence function is:
            IF_i = (X'_2 X_2)^{-1} [gamma_hat' x_{10i} eps_{10i} - x_{2i} eps_{2i}]

        where gamma_hat = (X'_{10} X_{10})^{-1} (X'_1 X_2) uses the GLOBAL
        cross-moment.

        The cluster-robust variance is:
            V = (X'_2 X_2)^{-1} (sum_g S_g S'_g) (X'_2 X_2)^{-1}
            S_g = gamma_hat' c_g - X'_{2g} eps_{2g}
            c_g = X'_{10g} eps_{10g}

        With survey weights W (diagonal):
            Bread: (X'_2 W X_2)^{-1}
            gamma_hat: (X'_{10} W X_{10})^{-1} (X'_1 W X_2)
            c_g = sum_{i in g} w_i * x_{10i} * eps_{10i}
            s2_g = sum_{i in g} w_i * x_{2i} * eps_{2i}

        Parameters
        ----------
        X_2 : np.ndarray, shape (n, k)
            Stage 2 design matrix (treatment indicators).
        eps_2 : np.ndarray, shape (n,)
            Stage 2 residuals.
        cluster_ids : np.ndarray, shape (n,)
            Cluster identifiers.
        survey_weights : np.ndarray, optional
            Survey weights of shape (n,). When None, unweighted (identical
            to current code).

        Returns
        -------
        np.ndarray, shape (k, k)
            Variance-covariance matrix.
        """
        n = len(df)
        k = X_2.shape[1]

        # Exclude rank-deficient covariates
        cov_list = covariates
        if covariates and kept_cov_mask is not None and not np.all(kept_cov_mask):
            cov_list = [c for c, k_ in zip(covariates, kept_cov_mask) if k_]

        # Build sparse FE design matrices X_1 (all obs) and X_10 (untreated only)
        X_1_sparse, X_10_sparse, unit_to_idx, time_to_idx = self._build_fe_design(
            df, unit, time, cov_list, omega_0_mask
        )

        p = X_1_sparse.shape[1]

        # eps_10 = Y - X_10 @ gamma_hat
        # Untreated: stage 1 residual (Y - fitted). Treated: Y (X_10 rows = 0).
        # Reconstruct Y from y_tilde: Y = y_tilde + fitted_stage1
        alpha_i = df[unit].map(unit_fe).values
        beta_t = df[time].map(time_fe).values
        alpha_i = np.where(pd.isna(alpha_i), 0.0, alpha_i).astype(float)
        beta_t = np.where(pd.isna(beta_t), 0.0, beta_t).astype(float)
        fitted_1 = alpha_i + beta_t
        if delta_hat is not None and cov_list:
            if kept_cov_mask is not None and not np.all(kept_cov_mask):
                fitted_1 = fitted_1 + np.dot(df[cov_list].values, delta_hat[kept_cov_mask])
            else:
                fitted_1 = fitted_1 + np.dot(df[cov_list].values, delta_hat)

        y_tilde = df["_y_tilde"].values
        y_vals = y_tilde + fitted_1  # reconstruct Y

        # eps_10: for untreated, stage 1 residual; for treated, Y_i (since X_10 rows = 0)
        eps_10 = np.empty(n)
        omega_0 = omega_0_mask.values
        eps_10[omega_0] = y_vals[omega_0] - fitted_1[omega_0]  # Stage 1 residual
        eps_10[~omega_0] = y_vals[~omega_0]  # x_{10i} = 0, so eps_10 = Y

        # 1. gamma_hat = (X'_{10} W X_{10})^{-1} (X'_1 W X_2)  [p x k]
        # With survey weights, both cross-products need W
        if survey_weights is not None:
            XtWX_10 = X_10_sparse.T @ X_10_sparse.multiply(survey_weights[:, None])
            Xt1_WX2 = X_1_sparse.T @ (X_2 * survey_weights[:, None])
        else:
            XtWX_10 = X_10_sparse.T @ X_10_sparse  # (p x p) sparse
            Xt1_WX2 = X_1_sparse.T @ X_2  # (p x k) dense

        try:
            solve_XtX = sparse_factorized(XtWX_10.tocsc())
            if Xt1_WX2.ndim == 1:
                gamma_hat = solve_XtX(Xt1_WX2).reshape(-1, 1)
            else:
                gamma_hat = np.column_stack(
                    [solve_XtX(Xt1_WX2[:, j]) for j in range(Xt1_WX2.shape[1])]
                )
        except RuntimeError:
            # Singular matrix — fall back to dense least-squares
            gamma_hat = np.linalg.lstsq(XtWX_10.toarray(), Xt1_WX2, rcond=None)[0]
            if gamma_hat.ndim == 1:
                gamma_hat = gamma_hat.reshape(-1, 1)

        # 2. Per-cluster Stage 1 scores: c_g = sum_{i in g} w_i * x_{10i} * eps_{10i}
        # Only untreated obs have non-zero X_10 rows
        # With survey weights: multiply eps_10 by survey_weights before sparse multiply
        if survey_weights is not None:
            weighted_eps_10 = survey_weights * eps_10
        else:
            weighted_eps_10 = eps_10
        weighted_X10 = X_10_sparse.multiply(weighted_eps_10[:, None])  # sparse element-wise

        unique_clusters, cluster_indices = np.unique(cluster_ids, return_inverse=True)
        G = len(unique_clusters)

        n_elements = weighted_X10.shape[0] * weighted_X10.shape[1]
        c_by_cluster = np.zeros((G, p))
        if n_elements > _SPARSE_DENSE_THRESHOLD:
            # Per-column path: limits peak memory for large FE matrices
            weighted_X10_csc = weighted_X10.tocsc()
            for j_col in range(p):
                col_data = weighted_X10_csc.getcol(j_col).toarray().ravel()
                np.add.at(c_by_cluster[:, j_col], cluster_indices, col_data)
        else:
            # Dense path: faster for moderate-size matrices
            weighted_X10_dense = weighted_X10.toarray()
            for j_col in range(p):
                np.add.at(c_by_cluster[:, j_col], cluster_indices, weighted_X10_dense[:, j_col])

        # 3. Per-cluster Stage 2 scores: s2_g = sum_{i in g} w_i * x_{2i} * eps_{2i}
        if survey_weights is not None:
            weighted_eps_2 = survey_weights * eps_2
        else:
            weighted_eps_2 = eps_2
        weighted_X2 = X_2 * weighted_eps_2[:, None]  # (n x k) dense
        s2_by_cluster = np.zeros((G, k))
        for j_col in range(k):
            np.add.at(s2_by_cluster[:, j_col], cluster_indices, weighted_X2[:, j_col])

        # 4. S_g = gamma_hat' c_g - X'_{2g} eps_{2g}
        S = self._compute_gmm_scores(c_by_cluster, gamma_hat, s2_by_cluster)

        # 5. Meat: sum_g S_g S'_g = S' S
        with np.errstate(invalid="ignore", over="ignore"):
            meat = S.T @ S  # (k x k)

        # 6. Bread: (X'_2 W X_2)^{-1}
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            if survey_weights is not None:
                XtWX_2 = X_2.T @ (X_2 * survey_weights[:, None])
            else:
                XtWX_2 = X_2.T @ X_2
        try:
            bread = np.linalg.solve(XtWX_2, np.eye(k))
        except np.linalg.LinAlgError:
            bread = np.linalg.lstsq(XtWX_2, np.eye(k), rcond=None)[0]

        # 7. V = bread @ meat @ bread
        V = bread @ meat @ bread
        return V

    def _build_fe_design(
        self,
        df: pd.DataFrame,
        unit: str,
        time: str,
        covariates: Optional[List[str]],
        omega_0_mask: pd.Series,
    ) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, Dict[Any, int], Dict[Any, int]]:
        """
        Build sparse FE design matrices X_1 (all obs) and X_10 (untreated rows only).

        Column layout: [unit_0, ..., unit_{U-2}, time_0, ..., time_{T-2}, cov_1, ..., cov_C]
        (Drop first unit and first time for identification.)

        X_10 is identical to X_1 except that rows for treated observations are zeroed out.

        Returns
        -------
        X_1_sparse : sparse.csr_matrix, shape (n, p)
        X_10_sparse : sparse.csr_matrix, shape (n, p)
        unit_to_idx : dict
        time_to_idx : dict
        """
        n = len(df)
        unit_vals = df[unit].values
        time_vals = df[time].values
        omega_0 = omega_0_mask.values

        all_units = np.unique(unit_vals)
        all_times = np.unique(time_vals)
        unit_to_idx = {u: i for i, u in enumerate(all_units)}
        time_to_idx = {t: i for i, t in enumerate(all_times)}
        n_units = len(all_units)
        n_times = len(all_times)
        n_cov = len(covariates) if covariates else 0
        n_fe_cols = (n_units - 1) + (n_times - 1)

        def _build_rows(mask=None):
            """Build sparse matrix for given observation mask."""
            # Unit dummies (drop first)
            u_indices = np.array([unit_to_idx[u] for u in unit_vals])
            u_mask = u_indices > 0
            if mask is not None:
                u_mask = u_mask & mask

            u_rows = np.arange(n)[u_mask]
            u_cols = u_indices[u_mask] - 1

            # Time dummies (drop first)
            t_indices = np.array([time_to_idx[t] for t in time_vals])
            t_mask = t_indices > 0
            if mask is not None:
                t_mask = t_mask & mask

            t_rows = np.arange(n)[t_mask]
            t_cols = (n_units - 1) + t_indices[t_mask] - 1

            rows = np.concatenate([u_rows, t_rows])
            cols = np.concatenate([u_cols, t_cols])
            data = np.ones(len(rows))

            A_fe = sparse.csr_matrix((data, (rows, cols)), shape=(n, n_fe_cols))

            if n_cov > 0:
                cov_data = df[covariates].values.copy()
                if mask is not None:
                    cov_data[~mask] = 0.0
                A_cov = sparse.csr_matrix(cov_data)
                A = sparse.hstack([A_fe, A_cov], format="csr")
            else:
                A = A_fe

            return A

        X_1 = _build_rows(mask=None)
        X_10 = _build_rows(mask=omega_0)

        return X_1, X_10, unit_to_idx, time_to_idx

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
        }

    def set_params(self, **params) -> "TwoStageDiD":
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


def two_stage_did(
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
) -> TwoStageDiDResults:
    """
    Convenience function for two-stage DiD estimation.

    This is a shortcut for creating a TwoStageDiD estimator and calling fit().

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
        Requires analytical inference (n_bootstrap=0).
    **kwargs
        Additional keyword arguments passed to TwoStageDiD constructor.

    Returns
    -------
    TwoStageDiDResults
        Estimation results.

    Examples
    --------
    >>> from diff_diff import two_stage_did, generate_staggered_data
    >>> data = generate_staggered_data(seed=42)
    >>> results = two_stage_did(data, 'outcome', 'unit', 'period',
    ...                         'first_treat', aggregate='event_study')
    >>> results.print_summary()
    """
    est = TwoStageDiD(**kwargs)
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
