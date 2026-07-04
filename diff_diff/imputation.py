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

The ``vcov_type`` input contract is permanently narrow to ``{"hc1"}`` per
the influence-function-based variance decomposition: the per-unit IF
aggregation (Theorem 3 equation 7) has no equivalent single design matrix
on which analytical-sandwich families (``classical``, ``hc2``, ``hc2_bm``)
or spatial-HAC composition (``conley``) can be defined. ``cluster=``
invokes per-cluster IF summation; ``survey_design=`` invokes TSL on the
combined IF. See ``docs/methodology/REGISTRY.md`` for the cross-estimator
IF-vs-sandwich taxonomy.
"""

import warnings
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.sparse.linalg import factorized as sparse_factorized

from diff_diff.imputation_bootstrap import ImputationDiDBootstrapMixin, _compute_target_weights
from diff_diff.imputation_results import (  # noqa: F401 (re-export)
    ImputationBootstrapResults,
    ImputationDiDResults,
)
from diff_diff.linalg import solve_ols
from diff_diff.utils import _iterative_fe_solve, demean_by_groups, safe_inference


class _UntreatedProjection(NamedTuple):
    """Cached, target-invariant pieces of the untreated imputation projection
    ``v_untreated = -A_0 (A_0' [W] A_0)^{-1} A_1' w`` (BJS 2024 Theorem 3).

    Within a single ``fit()`` the untreated design (``df_0``/``df_1``, covariates,
    survey weights) is identical across every estimand target (overall ATT, each
    event-study horizon, each group, and the bootstrap precompute) -- only the
    treated aggregation ``weights`` (the RHS ``A_1' w``) vary. So ``A_0``, ``A_1``
    and the factorization of ``A_0'[W]A_0`` are built once and reused across
    targets (factorize-once / solve-many), mirroring the TwoStageDiD GMM-sandwich
    ``sparse_factorized`` pattern.
    """

    A_0: sparse.csr_matrix
    A_1: sparse.csr_matrix
    # solver(rhs) -> z; None when the factorization was exactly singular (the
    # solve path then routes to a dense lstsq fallback).
    solver: Optional[Callable[[np.ndarray], np.ndarray]]
    A0tA0_csc: sparse.csc_matrix  # retained for the dense-lstsq fallback
    survey_weights_0: Optional[np.ndarray]
    singular: bool


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
    vcov_type : str, default="hc1"
        Variance estimator family. Permanently narrow to ``{"hc1"}`` per
        the IF-based variance contract (Theorem 3): analytical-sandwich
        families ``{classical, hc2, hc2_bm}`` and ``conley`` are rejected
        at ``__init__`` with methodology-rooted messages. ``cluster=``
        invokes per-cluster IF summation; ``survey_design=`` invokes TSL
        on the combined IF. See REGISTRY.md for the cross-estimator
        IF-vs-sandwich taxonomy.
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
        computed for abs(h) <= horizon_max.
    aux_partition : str, default="cohort_horizon"
        Controls the auxiliary model partition for Theorem 3 variance:
        - "cohort_horizon": Groups by cohort x relative time (tightest SEs)
        - "cohort": Groups by cohort only (more conservative)
        - "horizon": Groups by relative time only (more conservative)
    pretrends : bool, default=False
        If True, event study includes pre-treatment horizons for visual
        pre-trends assessment. Pre-period effects should be ~0 under
        parallel trends. Only affects event_study aggregation; overall
        ATT and group aggregation are unchanged.

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
        vcov_type: str = "hc1",
        n_bootstrap: int = 0,
        bootstrap_weights: str = "rademacher",
        seed: Optional[int] = None,
        rank_deficient_action: str = "warn",
        horizon_max: Optional[int] = None,
        aux_partition: str = "cohort_horizon",
        pretrends: bool = False,
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
        self._validate_vcov_type(vcov_type)

        self.anticipation = anticipation
        self.alpha = alpha
        self.cluster = cluster
        self.vcov_type = vcov_type
        self.n_bootstrap = n_bootstrap
        self.bootstrap_weights = bootstrap_weights
        self.seed = seed
        self.rank_deficient_action = rank_deficient_action
        self.horizon_max = horizon_max
        self.aux_partition = aux_partition
        self.pretrends = pretrends

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
            pweight only (aweight/fweight raise ValueError). Supports strata,
            PSU, and FPC for design-based variance via compute_survey_if_variance().
            Strata enters survey df for t-distribution inference.
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
        # Re-validate vcov_type at fit-time so sklearn-style set_params
        # mutations (e.g. set_params(vcov_type="classical")) are re-checked
        # at use rather than silently accepted by the parameter setter.
        self._validate_vcov_type(self.vcov_type)

        # Validate inputs
        required_cols = [outcome, unit, time, first_treat]
        if covariates:
            required_cols.extend(covariates)

        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # pretrends + analytical survey is supported (Phase 8e-iii).
        # Replicate-weight surveys need per-replicate lead regression refits
        # which are not yet implemented — reject that combination.
        if (
            self.pretrends
            and survey_design is not None
            and survey_design.replicate_method is not None
            and aggregate in ("event_study", "all")
        ):
            raise NotImplementedError(
                "pretrends=True is not yet compatible with replicate-weight "
                "survey designs. Analytical survey designs (strata/PSU/FPC) "
                "are supported. Use pretrends=False with replicate weights."
            )

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

        _uses_replicate_imp = (
            resolved_survey is not None and resolved_survey.uses_replicate_variance
        )
        if _uses_replicate_imp and self.n_bootstrap > 0:
            raise ValueError(
                "Cannot use n_bootstrap > 0 with replicate-weight survey designs. "
                "Replicate weights provide their own variance estimation."
            )
        # Reject replicate-weight + cluster=: replicate IF variance is
        # computed by replicate reweighting (BRR / Fay / JK1 / JKn / SDR)
        # and ignores PSU/cluster entirely (survey.py enforces that
        # replicate_weights are mutually exclusive with strata/psu/fpc).
        # Honoring bare cluster= here would silently have no effect on
        # variance while populating cluster_name/n_clusters on Results
        # dishonestly. Fail-closed mirroring CallawaySantAnna.
        if (
            self.cluster is not None
            and survey_design is not None
            and getattr(survey_design, "replicate_weights", None) is not None
        ):
            raise NotImplementedError(
                f"ImputationDiD(cluster={self.cluster!r}) is not supported "
                "with replicate-weight survey designs. Replicate-weight "
                "variance is computed by replicate reweighting (BRR / Fay / "
                "JK1 / JKn / SDR) and ignores PSU/cluster entirely — setting "
                "cluster= would silently have no effect on the variance "
                "estimate. Either omit cluster= (the replicate weights encode "
                "the design structure implicitly) or use a non-replicate "
                "survey design (with explicit strata/psu/fpc)."
            )
        # Validate within-unit constancy for panel survey designs
        if resolved_survey is not None:
            _validate_unit_constant_survey(data, unit, survey_design)
            if resolved_survey.weight_type != "pweight":
                raise ValueError(
                    f"ImputationDiD survey support requires weight_type='pweight', "
                    f"got '{resolved_survey.weight_type}'. The survey variance math "
                    f"assumes probability weights (pweight)."
                )
            # FPC is supported — threaded through compute_survey_if_variance()
            # in _compute_conservative_variance().

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

        # Per-fit cache of the target-invariant untreated-projection design +
        # factorization, shared across every estimand target (overall ATT, each
        # event-study horizon, each group) AND the bootstrap precompute. A
        # fit-time local (not self.* state) so fit() stays idempotent; see
        # _compute_cluster_psi_sums for the key derivation.
        proj_cache: Dict[Any, _UntreatedProjection] = {}

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

        # ---- Variance ----
        _n_valid_rep_imp = None
        _vcov_rep_imp = None
        overall_se = np.nan  # placeholder; overridden by replicate or conservative path

        if not _uses_replicate_imp:
            # Conservative variance (Theorem 3)
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
                    resolved_survey=(resolved_survey if not _uses_replicate_imp else None),
                    proj_cache=proj_cache,
                )

        # Survey degrees of freedom for t-distribution inference
        _survey_df = resolved_survey.df_survey if resolved_survey is not None else None
        # Replicate df: rank-deficient → NaN inference; dropped replicates → n_valid-1
        if _uses_replicate_imp and _survey_df is None:
            _survey_df = 0  # rank-deficient replicate → NaN inference

        # Compute overall inference (may be overridden by replicate below)
        overall_t, overall_p, overall_ci = safe_inference(
            overall_att, overall_se, alpha=self.alpha, df=_survey_df
        )

        # Event study and group aggregation (full-sample, for point estimates)
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
                resolved_survey=(resolved_survey if not _uses_replicate_imp else None),
                proj_cache=proj_cache,
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
                resolved_survey=(resolved_survey if not _uses_replicate_imp else None),
                proj_cache=proj_cache,
            )

        # Replicate variance: derive keys from actual outputs (after filtering)
        if _uses_replicate_imp:
            from diff_diff.survey import compute_replicate_refit_variance

            _rel_times_treated = df.loc[omega_1_mask, "_rel_time"].values
            _cohorts_treated = df.loc[omega_1_mask, first_treat].values

            # Derive keys from actual outputs (excludes filtered/Prop5/ref)
            _sorted_rel_times = sorted(
                e
                for e in (event_study_effects or {}).keys()
                if np.isfinite(event_study_effects[e]["effect"])
                and event_study_effects[e].get("n_obs", 1) > 0
            )
            _sorted_groups = sorted(
                g for g in (group_effects or {}).keys() if np.isfinite(group_effects[g]["effect"])
            )
            _n_es = len(_sorted_rel_times)

            # Pre-compute balanced cohort mask for balance_e
            _balanced_mask_treated = None
            if balance_e is not None and _sorted_rel_times:
                df_1 = df.loc[omega_1_mask]
                rel_times_all = df_1["_rel_time"].values
                all_horizons_full = sorted(set(int(h) for h in rel_times_all if np.isfinite(h)))
                if self.horizon_max is not None:
                    all_horizons_full = [h for h in all_horizons_full if abs(h) <= self.horizon_max]
                cohort_rel_times = self._build_cohort_rel_times(df, first_treat)
                _balanced_mask_treated = self._compute_balanced_cohort_mask(
                    df_1, first_treat, all_horizons_full, balance_e, cohort_rel_times
                )

            # Single vectorized refit: [overall, es_e0..., grp_g0...]
            def _refit_imp(w_r):
                ufe_r, tfe_r, gm_r, delta_r, _ = self._fit_untreated_model(
                    df,
                    outcome,
                    unit,
                    time,
                    covariates,
                    omega_0_mask,
                    weights=w_r,
                )
                tau_r, _ = self._impute_treatment_effects(
                    df,
                    outcome,
                    unit,
                    time,
                    covariates,
                    omega_1_mask,
                    ufe_r,
                    tfe_r,
                    gm_r,
                    delta_r,
                )
                fin = np.isfinite(tau_r)
                treated_w = w_r[omega_1_mask.values]
                results = []
                # [0] Overall ATT
                tw_fin = treated_w[fin]
                tw_sum = np.sum(tw_fin)
                results.append(
                    float(np.sum(tau_r[fin] * tw_fin) / tw_sum) if tw_sum > 0 else np.nan
                )
                # [1..n_es] Event-study (identified only)
                for e in _sorted_rel_times:
                    mask_e = fin & (_rel_times_treated == e)
                    if _balanced_mask_treated is not None:
                        mask_e = mask_e & _balanced_mask_treated
                    tw_e = treated_w[mask_e]
                    s = np.sum(tw_e)
                    results.append(float(np.sum(tau_r[mask_e] * tw_e) / s) if s > 0 else np.nan)
                # [n_es+1..] Group (identified only)
                for g in _sorted_groups:
                    mask_g = fin & (_cohorts_treated == g)
                    tw_g = treated_w[mask_g]
                    s = np.sum(tw_g)
                    results.append(float(np.sum(tau_r[mask_g] * tw_g) / s) if s > 0 else np.nan)
                return np.array(results)

            # Build full-sample estimate from actual effects
            _full_est = [overall_att]
            _full_est.extend([event_study_effects[e]["effect"] for e in _sorted_rel_times])
            _full_est.extend([group_effects[g]["effect"] for g in _sorted_groups])

            _vcov_rep_imp, _n_valid_rep_imp = compute_replicate_refit_variance(
                _refit_imp, np.array(_full_est), resolved_survey
            )
            overall_se = float(np.sqrt(max(_vcov_rep_imp[0, 0], 0.0)))

            # Override df if replicates were dropped
            if _n_valid_rep_imp < resolved_survey.n_replicates:
                _survey_df = _n_valid_rep_imp - 1 if _n_valid_rep_imp > 1 else 0
            if survey_metadata is not None:
                survey_metadata.df_survey = _survey_df if _survey_df and _survey_df > 0 else None

            overall_t, overall_p, overall_ci = safe_inference(
                overall_att, overall_se, alpha=self.alpha, df=_survey_df
            )

            # Override event-study SEs from vcov diagonal
            for i, e in enumerate(_sorted_rel_times):
                if event_study_effects is not None and e in event_study_effects:
                    se_e = float(np.sqrt(max(_vcov_rep_imp[1 + i, 1 + i], 0.0)))
                    eff_e = event_study_effects[e]["effect"]
                    t_e, p_e, ci_e = safe_inference(eff_e, se_e, alpha=self.alpha, df=_survey_df)
                    event_study_effects[e]["se"] = se_e
                    event_study_effects[e]["t_stat"] = t_e
                    event_study_effects[e]["p_value"] = p_e
                    event_study_effects[e]["conf_int"] = ci_e

            # Override group SEs from vcov diagonal
            for j, g in enumerate(_sorted_groups):
                if group_effects is not None and g in group_effects:
                    se_g = float(np.sqrt(max(_vcov_rep_imp[1 + _n_es + j, 1 + _n_es + j], 0.0)))
                    eff_g = group_effects[g]["effect"]
                    t_g, p_g, ci_g = safe_inference(eff_g, se_g, alpha=self.alpha, df=_survey_df)
                    group_effects[g]["se"] = se_g
                    group_effects[g]["t_stat"] = t_g
                    group_effects[g]["p_value"] = p_g
                    group_effects[g]["conf_int"] = ci_g

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
            "resolved_survey": resolved_survey,
            "survey_weights": survey_weights,
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
                    proj_cache=proj_cache,
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

        # Resolve cluster_name / n_clusters for Results metadata.
        # Suppress under ANY survey design (the survey block in summary()
        # already renders the design's PSU/strata/replicate metadata, and
        # replicate-weight variance ignores PSU/cluster entirely — keeping
        # cluster_name/n_clusters populated on a replicate fit would
        # misreport the inference source).
        # Otherwise:
        #   bare cluster= -> populate with the user-named cluster column
        #   cluster=None  -> the Theorem 3 variance still clusters at the
        #                    `unit` column by default (cluster_var = unit
        #                    at L418), so the summary label must report
        #                    unit-cluster CR1, not generic HC1.
        if resolved_survey is not None:
            _cluster_name_for_results: Optional[str] = None
            _n_clusters_for_results: Optional[int] = None
        elif self.cluster is not None:
            _cluster_name_for_results = self.cluster
            _n_clusters_for_results = int(data[self.cluster].nunique())
        else:
            _cluster_name_for_results = unit
            _n_clusters_for_results = int(data[unit].nunique())

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
            anticipation=self.anticipation,
            bootstrap_results=bootstrap_results,
            _estimator_ref=self,
            survey_metadata=survey_metadata,
            vcov_type=self.vcov_type,
            cluster_name=_cluster_name_for_results,
            n_clusters=_n_clusters_for_results,
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
        max_iter: int = 10_000,
        tol: float = 1e-10,
        weights: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[Any, float], Dict[Any, float]]:
        """
        Estimate unit and time FE via iterative alternating projection (Gauss-Seidel).

        Thin wrapper over the shared bincount solver
        (``diff_diff.utils._iterative_fe_solve``): factorize unit/time once,
        solve on integer codes, map the level arrays back to dicts.
        Converges to the exact (W)LS solution for balanced and unbalanced
        panels; balanced panels converge in 1-2 iterations.

        Parameters
        ----------
        idx : pd.Index
            Unused; retained for call-site stability.
        weights : np.ndarray, optional
            Survey weights (weighted group means ``sum(w*x)/sum(w)``). A
            unit/period whose observations ALL carry zero weight has no
            identifying contribution and gets ``NaN`` FE (its key is kept so
            the rank-condition membership check still sees the group).

        Returns
        -------
        unit_fe : dict
            Mapping from unit -> unit fixed effect.
        time_fe : dict
            Mapping from time -> time fixed effect.
        """
        unit_codes, unit_uniques = pd.factorize(unit_vals, sort=False)
        time_codes, time_uniques = pd.factorize(time_vals, sort=False)
        if (unit_codes < 0).any() or (time_codes < 0).any():
            raise ValueError(
                "ImputationDiD: unit or time column contains NaN. Drop or "
                "impute missing group keys before fitting."
            )
        unit_fe_arr, time_fe_arr = _iterative_fe_solve(
            np.asarray(y, dtype=np.float64),
            unit_codes.astype(np.intp, copy=False),
            time_codes.astype(np.intp, copy=False),
            len(unit_uniques),
            len(time_uniques),
            weights=weights,
            max_iter=max_iter,
            tol=tol,
            method_name="ImputationDiD iterative FE solver",
        )
        unit_fe = dict(zip(unit_uniques, unit_fe_arr))
        time_fe = dict(zip(time_uniques, time_fe_arr))
        return unit_fe, time_fe

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

            # Step A: within-transform Y and all X columns through the shared
            # MAP engine (factorize-once + bincount + optional Rust kernel),
            # one dispatch for every column. within_transform pins
            # [unit, time]; [time, unit] here preserves the historical
            # time-then-unit sweep order of the per-estimator loops.
            narrow = df_0[[outcome, *covariates, time, unit]].copy()
            demeaned, _ = demean_by_groups(
                narrow,
                [outcome, *covariates],
                [time, unit],
                inplace=True,
                weights=w_0,
                max_iter=10_000,
                tol=1e-10,
            )
            y_dm = demeaned[outcome].to_numpy(dtype=np.float64)
            X_dm = demeaned[covariates].to_numpy(dtype=np.float64)

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
        proj_cache: Optional[Dict[Any, _UntreatedProjection]] = None,
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

        # ---- Compute v_it for treated observations ----
        v_treated = weights.copy()

        # ---- Compute v_it for untreated observations ----
        # Exact two-way-FE imputation projection
        # v_untreated = -A_0 (A_0' [W] A_0)^{-1} A_1' w_treated  (Theorem 3 / the
        # implied weights of Supplementary Proposition A3), used for BOTH the
        # FE-only and the covariate case. The earlier FE-only closed form
        # -(w_i/n0_i + w_t/n0_t - w/N_0) is exact only for a *balanced* untreated
        # panel; Omega_0 is generically unbalanced in staggered designs (treated
        # observations are removed), which biased the analytical SE downward
        # (~27% on the parity panel). The projection matches R `didimputation`
        # exactly -- see tests/test_methodology_imputation.py::TestImputationDiDParityR.
        # Build the target-invariant projection design + factorization once per
        # fit() (cached in proj_cache), then solve only the target-specific RHS.
        # survey_weights is DELIBERATELY excluded from the key: the cache is a
        # fit-LOCAL dict, and within one fit() survey_weights is a single fixed
        # object, so the masks deterministically map to one sw_0 =
        # survey_weights[omega_0_mask]. The masks + covariates + kept_cov_mask
        # therefore FULLY identify the design (sw_0 itself is a fresh-sliced array
        # per call -- keying on its id() would miss every time and balloon the
        # cache to 1+H+G full A_0/A_1/factorization entries). id()-keys are safe:
        # the masks are fit() locals alive for the whole fit and the cache is a
        # fit-local dict, so no cross-fit leak / id reuse.
        cov_list = covariates if covariates is not None else []
        ctx: Optional[_UntreatedProjection] = None
        if proj_cache is not None:
            key = (
                id(omega_0_mask),
                id(omega_1_mask),
                tuple(cov_list),
                kept_cov_mask.tobytes() if kept_cov_mask is not None else None,
            )
            ctx = proj_cache.get(key)
        if ctx is None:
            ctx = self._build_untreated_projection(
                df_0,
                df_1,
                unit,
                time,
                cov_list,
                kept_cov_mask=kept_cov_mask,
                survey_weights_0=survey_weights_0,
            )
            if proj_cache is not None:
                proj_cache[key] = ctx
        v_untreated = self._solve_untreated_v(ctx, weights)

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

        return cluster_sums.values, cluster_sums.index.values, ve_product

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
        resolved_survey=None,
        proj_cache: Optional[Dict[Any, _UntreatedProjection]] = None,
    ) -> float:
        """
        Compute conservative clustered variance (Theorem 3, Equation 7).

        Parameters
        ----------
        weights : np.ndarray
            Aggregation weights w_it for treated observations.
            Shape: (n_treated,), must sum to 1.
        survey_weights : np.ndarray, optional
            Full-panel survey weights. When provided, they enter the untreated
            v_it WLS projection (weighted normal equations plus the left
            per-observation weight factor) and the design-based variance path.
        resolved_survey : ResolvedSurveyDesign, optional
            When provided, uses design-based variance via
            ``compute_survey_if_variance()`` (supports strata, PSU, FPC).

        Returns
        -------
        float
            Standard error.
        """
        sw_0 = survey_weights[omega_0_mask.values] if survey_weights is not None else None
        cluster_psi_sums, _, ve_product = self._compute_cluster_psi_sums(
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
            proj_cache=proj_cache,
        )

        if resolved_survey is not None:
            # Design-based variance with strata/PSU/FPC support
            from diff_diff.survey import compute_survey_if_variance

            variance = compute_survey_if_variance(ve_product, resolved_survey)
            if np.isnan(variance):
                return np.nan
            return np.sqrt(max(variance, 0.0))

        sigma_sq = float((cluster_psi_sums**2).sum())
        return np.sqrt(max(sigma_sq, 0.0))

    def _build_untreated_projection(
        self,
        df_0: pd.DataFrame,
        df_1: pd.DataFrame,
        unit: str,
        time: str,
        covariates: List[str],
        kept_cov_mask: Optional[np.ndarray] = None,
        survey_weights_0: Optional[np.ndarray] = None,
    ) -> _UntreatedProjection:
        """
        Build the target-INVARIANT pieces of the exact imputation projection
        ``v_untreated = -A_0 (A_0' [W] A_0)^{-1} A_1' w_treated`` and factorize the
        normal-equations matrix once. The result is cached per ``fit()`` (see
        ``_compute_cluster_psi_sums``) and reused across all estimand targets;
        only the target-specific RHS ``A_1' w`` is solved per target in
        ``_solve_untreated_v``.

        This is the GENERAL path -- used for both the FE-only and the covariate
        cases (an empty ``covariates`` list builds a pure two-way-FE design;
        ``n_cov == 0`` is the FE-only path). When survey_weights_0 is provided,
        uses the weighted normal equations ``A_0' W A_0`` (the per-observation
        survey weight is reapplied to the solved v in ``_solve_untreated_v``).

        Uses scipy.sparse for FE dummy columns to reduce memory from O(N*(U+T))
        to O(N) for the FE portion. An exactly singular ``A_0'[W]A_0`` makes
        ``sparse_factorized`` raise ``RuntimeError``; we emit a UserWarning (once
        per fit) and record ``singular=True`` so the solve routes to dense lstsq.
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
        # Two-way FE design = all unit dummies (their sum spans the intercept) +
        # time dummies dropping the first (identification). Dropping the first
        # unit dummy too -- with no intercept column -- would omit the baseline
        # level dimension and project onto a space one rank short of the true
        # two-way-FE span, biasing the imputation weights (and hence the SE).
        n_fe_cols = n_units + (n_times - 1)

        def _build_A_sparse(df_sub, unit_vals, time_vals):
            n = len(df_sub)

            # Unit dummies — keep ALL (together they span the intercept).
            u_indices = np.array([unit_to_idx[u] for u in unit_vals])
            u_rows = np.arange(n)
            u_cols = u_indices

            # Time dummies (drop first) — vectorized
            t_indices = np.array([time_to_idx[t] for t in time_vals])
            t_mask = t_indices > 0
            t_rows = np.arange(n)[t_mask]
            t_cols = n_units + (t_indices[t_mask] - 1)

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

        # Form (A_0' [W] A_0). When survey weights present, use the weighted
        # normal equations A_0' W A_0.
        if survey_weights_0 is not None:
            A0tA0_sparse = A_0.T @ A_0.multiply(survey_weights_0[:, None])
        else:
            A0tA0_sparse = A_0.T @ A_0  # stays sparse
        A0tA0_csc = A0tA0_sparse.tocsc()

        # Factorize once (factorize-once / solve-many). An exactly singular
        # matrix makes sparse_factorized raise RuntimeError -- the same condition
        # that previously surfaced as spsolve's MatrixRankWarning -> non-finite
        # solution. Mirror the TwoStageDiD GMM-sandwich pattern: warn once and
        # fall back to dense lstsq per target. (Bit-identical to the prior
        # per-target spsolve for a single dense RHS -- both use the SuperLU
        # simple driver with the same defaults.)
        try:
            solver: Optional[Callable[[np.ndarray], np.ndarray]] = sparse_factorized(A0tA0_csc)
            singular = False
        except RuntimeError as exc:
            # Silent-failure audit axis C: emit a UserWarning on fallback instead
            # of swallowing the error. Keep the "dense lstsq" substring (asserted
            # by tests).
            warnings.warn(
                "ImputationDiD variance: sparse factorization of (A_0' [W] A_0) "
                f"failed ({type(exc).__name__}); falling back to dense lstsq. This "
                "may indicate a rank-deficient or near-singular normal-equations "
                "matrix and variance estimates may be less reliable.",
                UserWarning,
                stacklevel=2,
            )
            solver = None
            singular = True

        return _UntreatedProjection(
            A_0=A_0,
            A_1=A_1,
            solver=solver,
            A0tA0_csc=A0tA0_csc,
            survey_weights_0=survey_weights_0,
            singular=singular,
        )

    def _solve_untreated_v(self, ctx: _UntreatedProjection, weights: np.ndarray) -> np.ndarray:
        """
        Solve the target-SPECIFIC RHS of the untreated imputation projection using
        the cached design + factorization in ``ctx``:
        ``v_untreated = -[W_0] A_0 (A_0'[W]A_0)^{-1} A_1' w_treated``.
        """
        A1_w = ctx.A_1.T @ weights  # (p,)

        if ctx.singular:
            # Factorization was singular at build time (warned once already).
            z, _, _, _ = np.linalg.lstsq(ctx.A0tA0_csc.toarray(), A1_w, rcond=None)
        else:
            assert ctx.solver is not None
            z = ctx.solver(A1_w)
            if not np.all(np.isfinite(z)):
                # Defensive, target-specific: a non-finite solve on an otherwise
                # factorizable matrix routes this RHS to dense lstsq. Warn per
                # target (silent-failure audit axis C) -- distinct from the
                # once-per-fit build-time singular warning.
                warnings.warn(
                    "ImputationDiD variance: sparse solve of (A_0' [W] A_0) z = "
                    "A_1' w returned a non-finite solution; falling back to dense "
                    "lstsq for this target. Variance estimates may be less reliable.",
                    UserWarning,
                    stacklevel=2,
                )
                z, _, _, _ = np.linalg.lstsq(ctx.A0tA0_csc.toarray(), A1_w, rcond=None)

        # v_untreated = -[W_0] A_0 z (WLS projection requires per-obs weight)
        v_untreated = -(ctx.A_0 @ z)
        if ctx.survey_weights_0 is not None:
            v_untreated = v_untreated * ctx.survey_weights_0
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
        Compute auxiliary residuals for treated obs (Theorem 3, Equation 8).

        Implements the paper's *unit-clustered* group aggregator (Borusyak,
        Jaravel & Spiess 2024, eq. 8, p. 3272), which minimizes the excess
        variance of the conservative estimator under a within-group
        constant-effect auxiliary model (Supplementary Appendix A.8):

            tau_tilde_g = sum_i (sum_{t in G_g,i} v_it)(sum_{t in G_g,i} v_it * tau_hat_it)
                          ----------------------------------------------------------------
                                          sum_i (sum_{t in G_g,i} v_it)^2

        i.e. for each unit i form the within-unit weight sum a_{i,g} and the
        within-unit weighted-effect sum b_{i,g} over the unit's observations in
        group g, then combine across units. At the default cohort x event-time
        partition (<=1 obs/unit/group) this reduces to sum(v^2 * tau_hat) /
        sum(v^2) -- the form the R `didimputation` package implements -- and
        equals the naive observation-level mean sum(v * tau_hat) / sum(v) only
        when within-group weights are uniform. Under coarser `cohort` / `horizon`
        partitions (a unit contributes several observations to a group) or
        non-uniform v_it (e.g. survey weights) the two genuinely differ.

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

        # Partition Omega_1 into groups G_g
        if self.aux_partition == "cohort_horizon":
            group_keys = list(zip(df_1[first_treat].values, df_1["_rel_time"].values))
        elif self.aux_partition == "cohort":
            group_keys = list(df_1[first_treat].values)
        elif self.aux_partition == "horizon":
            group_keys = list(df_1["_rel_time"].values)
        else:
            group_keys = list(range(n_1))  # each obs is its own group

        # Factorize group keys to integer codes (robust to tuple-valued keys).
        group_codes = pd.factorize(pd.Series(group_keys), sort=False)[0]
        gc_series = pd.Series(group_codes, index=df_1.index)
        tau_series = pd.Series(tau_hat, index=df_1.index)

        # Unit-clustered Equation 8. Only v_it != 0 observations contribute: a
        # zero-weight row adds exactly 0 to both a_{i,g} and b_{i,g}, so dropping
        # it is exact for finite tau_hat AND avoids letting an unimputable row
        # (NaN tau_hat, which always carries v_it == 0 by construction in
        # _compute_target_weights) poison its whole group via 0 * NaN = NaN. The
        # previous observation-level pandas sum relied on skipna to drop them.
        contrib = (v_treated != 0.0) & np.isfinite(tau_hat)
        if contrib.any():
            inner = pd.DataFrame(
                {
                    "g": group_codes[contrib],
                    "u": df_1[unit].values[contrib],
                    "v": v_treated[contrib],
                    "vt": v_treated[contrib] * tau_hat[contrib],
                }
            )
            # Per (group, unit): a_{i,g} = sum v_it, b_{i,g} = sum v_it * tau_hat
            per_unit = inner.groupby(["g", "u"], sort=False).agg(a=("v", "sum"), b=("vt", "sum"))
            # Per group: numerator sum_i a*b, denominator sum_i a^2
            per_group = (
                per_unit.assign(ab=per_unit["a"] * per_unit["b"], a2=per_unit["a"] ** 2)
                .groupby(level="g")
                .agg(num=("ab", "sum"), den=("a2", "sum"))
            )
            den_ok = per_group["den"].abs() >= 1e-15
            tau_tilde_map = (per_group["num"] / per_group["den"]).where(den_ok)
        else:
            tau_tilde_map = pd.Series(dtype=float)

        tau_tilde_per_obs = gc_series.map(tau_tilde_map)

        # Groups with no contributing (v_it != 0, finite tau_hat) observations --
        # e.g. off-target horizons in an event-study SE -- are a variance no-op
        # (psi_g = sum_t v_it * eps_tilde_it = 0 there regardless of tau_tilde_g),
        # so fall back to the unweighted group mean of tau_hat for a finite value.
        if tau_tilde_per_obs.isna().any():
            simple_means = tau_series.groupby(gc_series).mean()
            tau_tilde_per_obs = tau_tilde_per_obs.fillna(gc_series.map(simple_means))

        tau_tilde = tau_tilde_per_obs.values

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
        # Preserve NaN for any missing FE, symmetric with the treated path in
        # _compute_auxiliary_residuals_treated. On valid data this is inert --
        # every untreated observation's unit and period appear in the Step 1 FE
        # dicts (the dicts are estimated FROM Omega_0) -- but it stops a missing
        # FE from silently becoming a 0 residual, which would mask a rank-
        # condition logic error. Any NaN is zeroed downstream in the variance
        # product (np.nan_to_num), exactly like the treated path.
        alpha_i = df_0[unit].map(unit_fe).values.astype(float)
        beta_t = df_0[time].map(time_fe).values.astype(float)
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
        resolved_survey=None,
        proj_cache: Optional[Dict[Any, _UntreatedProjection]] = None,
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

        # Pre-period coefficients via BJS Test 1 lead regression
        if self.pretrends:
            df_0 = df.loc[omega_0_mask].copy()

            # Determine which cohorts' lead indicators to include.
            # balance_e restricts which cohorts contribute lead dummies,
            # but the full Omega_0 sample (including never-treated controls)
            # is kept for the within-transformed OLS (BJS Test 1, Equation 9).
            balanced_cohorts = None
            skip_preperiods = False
            if balance_e is not None:
                cohort_rel_times_0 = self._build_cohort_rel_times(df, first_treat)
                balanced_cohorts = set()
                if all_horizons:
                    max_h = max(all_horizons)
                    required_range = set(range(-balance_e, max_h + 1))
                    for g, horizons in cohort_rel_times_0.items():
                        if required_range.issubset(horizons):
                            balanced_cohorts.add(g)
                if not balanced_cohorts:
                    skip_preperiods = True  # No cohorts qualify — skip entirely

            if not skip_preperiods:
                rel_time_0 = np.where(
                    ~df_0["_never_treated"],
                    df_0[time] - df_0[first_treat],
                    np.nan,
                )

                # When balance_e is set, only include leads from balanced cohorts
                if balanced_cohorts is not None:
                    is_balanced = df_0[first_treat].isin(balanced_cohorts).values
                    rel_time_for_leads = np.where(is_balanced, rel_time_0, np.nan)
                else:
                    rel_time_for_leads = rel_time_0

                pre_rel_times = sorted(
                    set(
                        int(h)
                        for h in rel_time_for_leads
                        if np.isfinite(h) and h < -self.anticipation
                    )
                )
                pre_rel_times = [h for h in pre_rel_times if h != ref_period]
                if self.horizon_max is not None:
                    pre_rel_times = [h for h in pre_rel_times if abs(h) <= self.horizon_max]
                if pre_rel_times:
                    # Survey pretrends: pass full design (subpopulation approach)
                    _sw_0_pre = None
                    _rs_full_pre = None
                    _n_full_pre = None
                    _o0_idx_pre = None
                    if survey_weights is not None and resolved_survey is not None:
                        _sw_0_pre = survey_weights[omega_0_mask.values]
                        _rs_full_pre = resolved_survey
                        _n_full_pre = len(df)
                        _o0_idx_pre = np.where(omega_0_mask.values)[0]
                    _survey_df_pre = (
                        resolved_survey.df_survey if resolved_survey is not None else None
                    )
                    pre_effects, _, _ = self._compute_lead_coefficients(
                        df_0,
                        outcome,
                        unit,
                        time,
                        first_treat,
                        covariates,
                        cluster_var,
                        pre_rel_times,
                        alpha=self.alpha,
                        balanced_cohorts=balanced_cohorts,
                        survey_weights_0=_sw_0_pre,
                        resolved_survey_full=_rs_full_pre,
                        n_obs_full=_n_full_pre,
                        omega_0_indices=_o0_idx_pre,
                        survey_df=_survey_df_pre,
                    )
                    event_study_effects.update(pre_effects)

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
                resolved_survey=resolved_survey,
                proj_cache=proj_cache,
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
        resolved_survey=None,
        proj_cache: Optional[Dict[Any, _UntreatedProjection]] = None,
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
                resolved_survey=resolved_survey,
                proj_cache=proj_cache,
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
    # Pre-trend test (Equation 9) & pre-period lead coefficients
    # =========================================================================

    def _compute_lead_coefficients(
        self,
        df_0: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        covariates: Optional[List[str]],
        cluster_var: str,
        pre_rel_times: List[int],
        alpha: float = 0.05,
        balanced_cohorts: Optional[set] = None,
        survey_weights_0: Optional[np.ndarray] = None,
        resolved_survey_full=None,
        n_obs_full: Optional[int] = None,
        omega_0_indices: Optional[np.ndarray] = None,
        survey_df: Optional[int] = None,
    ) -> Tuple[Dict[int, Dict[str, Any]], np.ndarray, np.ndarray]:
        """
        Compute pre-period lead coefficients via within-transformed OLS (Test 1).

        Adds lead indicator dummies W_it(h) = 1[K_it = h] to the untreated
        model and estimates their coefficients. Uses cluster-robust SEs by
        default, or design-based survey VCV when ``resolved_survey_full``
        is provided (subpopulation approach: scores zero-padded to full
        panel length to preserve PSU/strata structure).

        The full Omega_0 sample (including never-treated controls) is always
        used for within-transformation. When balanced_cohorts is provided,
        lead indicators are restricted to observations from those cohorts only.

        Returns
        -------
        effects : dict
            Per-horizon event_study_effects entries.
        gamma : ndarray
            Lead coefficient vector.
        V_gamma : ndarray
            Sub-VCV matrix for lead coefficients.
        """
        rel_time_0 = np.where(
            ~df_0["_never_treated"],
            df_0[time] - df_0[first_treat],
            np.nan,
        )

        # Build lead indicators — restrict to balanced cohorts if specified
        if balanced_cohorts is not None:
            is_balanced = df_0[first_treat].isin(balanced_cohorts).values
        else:
            is_balanced = None

        lead_cols = []
        for h in pre_rel_times:
            col_name = f"_lead_{h}"
            indicator = (rel_time_0 == h).astype(float)
            if is_balanced is not None:
                indicator = indicator * is_balanced  # zero out non-balanced cohorts
            df_0[col_name] = indicator
            lead_cols.append(col_name)

        all_x_cols = lead_cols[:]
        if covariates:
            all_x_cols.extend(covariates)

        # Within-transform through the shared MAP engine (survey-weighted when
        # present), one dispatch for outcome + leads + covariates. Demean into
        # a narrow copy: df_0's raw lead indicators must survive for the
        # per-horizon n_obs counts below. within_transform pins [unit, time];
        # [time, unit] here preserves the historical time-then-unit sweep order.
        narrow = df_0[[outcome, *all_x_cols, time, unit]].copy()
        demeaned, _ = demean_by_groups(
            narrow,
            [outcome, *all_x_cols],
            [time, unit],
            inplace=True,
            weights=survey_weights_0,
            max_iter=10_000,
            tol=1e-10,
        )
        y_dm = demeaned[outcome].to_numpy(dtype=np.float64)
        X_dm = demeaned[all_x_cols].to_numpy(dtype=np.float64)

        # OLS for point estimates + VCV. When survey VCV will replace the
        # cluster-robust VCV, skip cluster_ids to avoid errors on domains
        # with few PSUs (the cluster-robust VCV is discarded anyway).
        cluster_ids = df_0[cluster_var].values
        _ols_weights = survey_weights_0
        _ols_weight_type = "pweight" if survey_weights_0 is not None else None
        _use_survey_vcov = resolved_survey_full is not None
        try:
            result = solve_ols(
                X_dm,
                y_dm,
                weights=_ols_weights,
                weight_type=_ols_weight_type,
                cluster_ids=None if _use_survey_vcov else cluster_ids,
                return_vcov=True,
                rank_deficient_action=self.rank_deficient_action,
                column_names=all_x_cols,
            )
        except (IndexError, np.linalg.LinAlgError):
            # All lead columns dropped (rank deficient after demeaning)
            effects: Dict[int, Dict[str, Any]] = {}
            for h in pre_rel_times:
                n_obs = int(df_0[f"_lead_{h}"].sum())
                effects[h] = {
                    "effect": np.nan,
                    "se": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "conf_int": (np.nan, np.nan),
                    "n_obs": n_obs,
                }
            for col in lead_cols:
                df_0.drop(columns=col, inplace=True)
            return (
                effects,
                np.full(len(pre_rel_times), np.nan),
                np.full((len(pre_rel_times), len(pre_rel_times)), np.nan),
            )

        coefficients = result[0]
        vcov = result[2]
        assert vcov is not None

        # Replace cluster-robust VCV with survey design-based VCV.
        # Use the FULL survey design (subpopulation approach): zero-pad
        # the Omega_0 scores back to full-panel length so PSU/strata
        # structure is preserved for variance estimation.
        if resolved_survey_full is not None:
            from diff_diff.survey import compute_survey_vcov

            # Use residuals from solve_ols (safe for rank-deficient fits).
            residuals_0 = result[1]

            # Reduce to kept (finite-coefficient) columns for VCV
            kept_mask = np.isfinite(coefficients)
            if np.all(kept_mask):
                X_for_vcov = X_dm
                res_for_vcov = residuals_0
            else:
                X_for_vcov = X_dm[:, kept_mask]
                res_for_vcov = residuals_0

            # Zero-pad to full panel length (subpopulation approach):
            # observations outside Omega_0 contribute zero to the score,
            # but preserve PSU/strata structure for design-based variance.
            n_full_obs = n_obs_full
            k_vcov = X_for_vcov.shape[1]
            X_full = np.zeros((n_full_obs, k_vcov), dtype=np.float64)
            res_full = np.zeros(n_full_obs, dtype=np.float64)
            X_full[omega_0_indices] = X_for_vcov
            res_full[omega_0_indices] = res_for_vcov

            vcov_kept = compute_survey_vcov(X_full, res_full, resolved_survey_full)

            if not np.all(kept_mask):
                # Expand back: NaN rows/cols for dropped columns
                n_coef = len(coefficients)
                vcov = np.full((n_coef, n_coef), np.nan)
                kept_idx = np.where(kept_mask)[0]
                vcov[np.ix_(kept_idx, kept_idx)] = vcov_kept
            else:
                vcov = vcov_kept

        n_leads = len(lead_cols)
        gamma = coefficients[:n_leads]
        V_gamma = vcov[:n_leads, :n_leads]

        # Use full-design survey df for t-distribution inference
        _df = survey_df

        # Build per-horizon effects
        effects = {}
        for j, h in enumerate(pre_rel_times):
            effect = float(gamma[j])
            se = float(np.sqrt(max(V_gamma[j, j], 0.0)))
            # n_obs from the lead indicator (respects balanced_cohorts restriction)
            n_obs = int(df_0[f"_lead_{h}"].sum())
            t_stat, p_value, conf_int = safe_inference(effect, se, alpha=alpha, df=_df)
            effects[h] = {
                "effect": effect,
                "se": se,
                "t_stat": t_stat,
                "p_value": p_value,
                "conf_int": conf_int,
                "n_obs": n_obs,
            }

        # Clean up temporary columns
        for col in lead_cols:
            df_0.drop(columns=col, inplace=True)

        return effects, gamma, V_gamma

    def _pretrend_test(self, n_leads: Optional[int] = None) -> Dict[str, Any]:
        """
        Run pre-trend test (Equation 9).

        Adds pre-treatment lead indicators to the Step 1 OLS on Omega_0
        and tests their joint significance via Wald F-test (cluster-robust
        or design-based survey VCV when survey_design is present).
        """
        if self._fit_data is None:
            raise RuntimeError("Must call fit() before pretrend_test().")

        fd = self._fit_data
        resolved_survey = fd.get("resolved_survey")
        if resolved_survey is not None and resolved_survey.uses_replicate_variance:
            raise NotImplementedError(
                "pretrend_test() is not yet supported for replicate-weight "
                "survey designs. Per-replicate Equation 9 lead regression "
                "refits are not implemented. Use analytical survey designs "
                "(strata/PSU/FPC) or call pretrend_test() without survey."
            )

        df = fd["df"]
        outcome = fd["outcome"]
        unit = fd["unit"]
        time = fd["time"]
        first_treat = fd["first_treat"]
        covariates = fd["covariates"]
        omega_0_mask = fd["omega_0_mask"]
        cluster_var = fd["cluster_var"]
        resolved_survey = fd.get("resolved_survey")
        survey_weights = fd.get("survey_weights")

        df_0 = df.loc[omega_0_mask].copy()

        # Compute relative time for untreated obs
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

        # Survey pretrends: pass full design (subpopulation approach)
        _sw_0_pt = None
        _rs_full_pt = None
        _n_full_pt = None
        _o0_idx_pt = None
        if survey_weights is not None and resolved_survey is not None:
            _sw_0_pt = survey_weights[omega_0_mask.values]
            _rs_full_pt = resolved_survey
            _n_full_pt = len(fd["df"])
            _o0_idx_pt = np.where(omega_0_mask.values)[0]

        # Use shared lead coefficient computation
        effects, gamma, V_gamma = self._compute_lead_coefficients(
            df_0,
            outcome,
            unit,
            time,
            first_treat,
            covariates,
            cluster_var,
            pre_rel_times,
            alpha=self.alpha,
            survey_weights_0=_sw_0_pt,
            resolved_survey_full=_rs_full_pt,
            n_obs_full=_n_full_pt,
            omega_0_indices=_o0_idx_pt,
            survey_df=(resolved_survey.df_survey if resolved_survey is not None else None),
        )

        n_leads_actual = len(pre_rel_times)

        # Wald F-test: F = (gamma' V^{-1} gamma) / n_leads
        try:
            V_inv_gamma = np.linalg.solve(V_gamma, gamma)
            wald_stat = float(gamma @ V_inv_gamma)
            f_stat = wald_stat / n_leads_actual
        except np.linalg.LinAlgError:
            f_stat = np.nan

        # P-value from F distribution (survey df when available)
        if np.isfinite(f_stat) and f_stat >= 0:
            if resolved_survey is not None and resolved_survey.df_survey is not None:
                df_denom = resolved_survey.df_survey
            else:
                cluster_ids = df_0[cluster_var].values
                n_clusters = len(np.unique(cluster_ids))
                df_denom = max(n_clusters - 1, 1)
            if df_denom <= 0:
                p_value = np.nan
            else:
                p_value = float(stats.f.sf(f_stat, n_leads_actual, df_denom))
        else:
            p_value = np.nan

        lead_coefficients = {h: effects[h]["effect"] for h in pre_rel_times}

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
            "vcov_type": self.vcov_type,
            "n_bootstrap": self.n_bootstrap,
            "bootstrap_weights": self.bootstrap_weights,
            "seed": self.seed,
            "rank_deficient_action": self.rank_deficient_action,
            "horizon_max": self.horizon_max,
            "aux_partition": self.aux_partition,
            "pretrends": self.pretrends,
        }

    def set_params(self, **params) -> "ImputationDiD":
        """Set estimator parameters (sklearn-compatible)."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        return self

    @staticmethod
    def _validate_vcov_type(vcov_type: str) -> None:
        """Validate ``vcov_type`` membership against ImputationDiD's
        permanently-narrow influence-function variance contract.

        Called from ``__init__`` AND ``fit()`` so sklearn-style
        ``set_params(vcov_type=...)`` mutations are re-checked at use
        time rather than silently accepted by the parameter setter.
        Mirrors the TripleDifference / CallawaySantAnna pattern (no
        single design matrix on which hat-matrix leverage or Bell-
        McCaffrey Satterthwaite DOF can be defined).
        """
        _accepted_vcov = {"hc1"}
        _if_incompatible_vcov = {"classical", "hc2", "hc2_bm"}
        _deferred_vcov = {"conley"}

        if vcov_type in _if_incompatible_vcov:
            raise ValueError(
                f"ImputationDiD(vcov_type={vcov_type!r}) is rejected: "
                "ImputationDiD uses influence-function-based variance per "
                "Borusyak, Jaravel, and Spiess (2024) Theorem 3. The "
                "per-unit influence function aggregation has no equivalent "
                "single design matrix on which hat matrix leverage or "
                "Bell-McCaffrey Satterthwaite DOF can be defined, so "
                "analytical-sandwich families {classical, hc2, hc2_bm} are "
                "not paper-prescribed. Use vcov_type='hc1' (the default) "
                "with cluster=<col> for per-cluster influence-function "
                "summation (Theorem 3 equation 7 conservative variance)."
            )
        if vcov_type in _deferred_vcov:
            raise ValueError(
                f"ImputationDiD(vcov_type={vcov_type!r}) is not yet "
                "supported: spatial-HAC composition with Theorem 3 "
                "per-unit IF aggregation has no reference implementation "
                "today. See TODO.md for the deferred follow-up row. Use "
                "vcov_type='hc1' (the default) with cluster=<col> for "
                "cluster-robust inference."
            )
        if vcov_type not in _accepted_vcov:
            raise ValueError(
                f"ImputationDiD(vcov_type={vcov_type!r}) is invalid. "
                f"Accepted: {sorted(_accepted_vcov)}."
            )

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
    vcov_type: str = "hc1",
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
        pweight only (aweight/fweight raise ValueError). Supports strata,
        PSU, and FPC for design-based variance. Strata enters survey df
        for t-distribution inference.
        Both analytical (n_bootstrap=0) and bootstrap inference are supported.
    vcov_type : str, default="hc1"
        Variance estimator family. ImputationDiD permanently accepts
        ``{"hc1"}`` only — analytical-sandwich families
        ``{classical, hc2, hc2_bm}`` are rejected at ``__init__`` because the
        Theorem 3 per-unit IF aggregation has no single design matrix on
        which hat-matrix leverage or Bell-McCaffrey Satterthwaite DOF can
        be defined. ``cluster=`` invokes per-cluster IF summation;
        ``survey_design=`` invokes TSL on the combined IF.
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
    est = ImputationDiD(vcov_type=vcov_type, **kwargs)
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
