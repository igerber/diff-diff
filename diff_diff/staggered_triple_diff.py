"""
Staggered Triple Difference (DDD) estimator.

Implements Ortiz-Villavicencio & Sant'Anna (2025) for staggered adoption
settings with an eligibility dimension, combining group-time DDD effects
via GMM-optimal weighting.

Core pairwise DiD computation matches R's triplediff::compute_did() exactly
(Riesz/Hajek normalization, separate M1/M3 OR corrections, hessian = (X'WX)^{-1}*n).
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.linalg import (
    _check_propensity_diagnostics,
    solve_logit,
)
from diff_diff.staggered_aggregation import (
    CallawaySantAnnaAggregationMixin,
)
from diff_diff.staggered_bootstrap import (
    CallawaySantAnnaBootstrapMixin,
)
from diff_diff.staggered_triple_diff_results import StaggeredTripleDiffResults
from diff_diff.utils import safe_inference

__all__ = [
    "StaggeredTripleDifference",
    "StaggeredTripleDiffResults",
]

# Type alias for pre-computed structures
PrecomputedData = Dict[str, Any]


class StaggeredTripleDifference(
    CallawaySantAnnaBootstrapMixin,
    CallawaySantAnnaAggregationMixin,
):
    """
    Staggered Triple Difference (DDD) estimator.

    Computes group-time average treatment effects ATT(g,t) for settings
    with staggered adoption and a binary eligibility dimension, using the
    three-DiD decomposition of Ortiz-Villavicencio & Sant'Anna (2025).

    Multiple comparison groups are combined via GMM-optimal (inverse-variance)
    weighting. Event study, group, and overall aggregations are supported.

    Parameters
    ----------
    estimation_method : str, default="dr"
        Estimation method: "dr" (doubly robust), "ipw" (inverse probability
        weighting), or "reg" (regression adjustment).
    alpha : float, default=0.05
        Significance level.
    anticipation : int, default=0
        Number of anticipation periods.
    base_period : str, default="varying"
        Base period selection: "varying" (consecutive comparisons) or
        "universal" (always vs g-1-anticipation).
    n_bootstrap : int, default=0
        Number of multiplier bootstrap repetitions. 0 disables bootstrap.
    bootstrap_weights : str, default="rademacher"
        Bootstrap weight distribution: "rademacher", "mammen", or "webb".
    seed : int or None, default=None
        Random seed for reproducibility.
    cband : bool, default=True
        Whether to compute simultaneous confidence bands.
    pscore_trim : float, default=0.01
        Propensity score trimming bound.
    cluster : str or None, default=None
        Column name for cluster-robust standard errors.
    rank_deficient_action : str, default="warn"
        Action for rank-deficient design matrices: "warn", "error", "silent".

    References
    ----------
    Ortiz-Villavicencio, M. & Sant'Anna, P.H.C. (2025). "Better Understanding
    Triple Differences Estimators." arXiv:2505.09942.
    """

    def __init__(
        self,
        estimation_method: str = "dr",
        control_group: str = "notyettreated",
        alpha: float = 0.05,
        anticipation: int = 0,
        base_period: str = "varying",
        n_bootstrap: int = 0,
        bootstrap_weights: str = "rademacher",
        seed: Optional[int] = None,
        cband: bool = True,
        pscore_trim: float = 0.01,
        cluster: Optional[str] = None,
        rank_deficient_action: str = "warn",
    ):
        if estimation_method not in ["dr", "ipw", "reg"]:
            raise ValueError(
                f"estimation_method must be 'dr', 'ipw', or 'reg', " f"got '{estimation_method}'"
            )
        if control_group not in ["nevertreated", "notyettreated"]:
            raise ValueError(
                f"control_group must be 'nevertreated' or 'notyettreated', "
                f"got '{control_group}'"
            )
        if not (0 < pscore_trim < 0.5):
            raise ValueError(f"pscore_trim must be in (0, 0.5), got {pscore_trim}")
        if bootstrap_weights not in ["rademacher", "mammen", "webb"]:
            raise ValueError(
                f"bootstrap_weights must be 'rademacher', 'mammen', or 'webb', "
                f"got '{bootstrap_weights}'"
            )
        if rank_deficient_action not in ["warn", "error", "silent"]:
            raise ValueError(
                f"rank_deficient_action must be 'warn', 'error', or 'silent', "
                f"got '{rank_deficient_action}'"
            )
        if base_period not in ["varying", "universal"]:
            raise ValueError(
                f"base_period must be 'varying' or 'universal', " f"got '{base_period}'"
            )

        self.estimation_method = estimation_method
        self.control_group = control_group
        self.alpha = alpha
        self.anticipation = anticipation
        self.base_period = base_period
        self.n_bootstrap = n_bootstrap
        self.bootstrap_weights = bootstrap_weights
        self.bootstrap_weight_type = bootstrap_weights
        self.seed = seed
        self.cband = cband
        self.pscore_trim = pscore_trim
        self.cluster = cluster
        self.rank_deficient_action = rank_deficient_action

        self.is_fitted_ = False
        self.results_: Optional[StaggeredTripleDiffResults] = None

    def get_params(self) -> Dict[str, Any]:
        """Get estimator parameters (sklearn-compatible)."""
        return {
            "estimation_method": self.estimation_method,
            "control_group": self.control_group,
            "alpha": self.alpha,
            "anticipation": self.anticipation,
            "base_period": self.base_period,
            "n_bootstrap": self.n_bootstrap,
            "bootstrap_weights": self.bootstrap_weights,
            "seed": self.seed,
            "cband": self.cband,
            "pscore_trim": self.pscore_trim,
            "cluster": self.cluster,
            "rank_deficient_action": self.rank_deficient_action,
        }

    def set_params(self, **params) -> "StaggeredTripleDifference":
        """Set estimator parameters (sklearn-compatible)."""
        valid_params = self.get_params()
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(f"Unknown parameter: {key}")
            setattr(self, key, value)
        if "bootstrap_weights" in params:
            self.bootstrap_weight_type = params["bootstrap_weights"]
        return self

    # ------------------------------------------------------------------
    # fit()
    # ------------------------------------------------------------------

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        eligibility: str,
        covariates: Optional[List[str]] = None,
        aggregate: Optional[str] = None,
        balance_e: Optional[int] = None,
        survey_design: object = None,
    ) -> StaggeredTripleDiffResults:
        """
        Fit the staggered triple difference estimator.

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
            Column with the enabling period for each unit's group.
            Use 0 or np.inf for never-enabled units.
        eligibility : str
            Binary eligibility indicator column (0/1, time-invariant).
        covariates : list of str, optional
            Covariate column names.
        aggregate : str, optional
            Aggregation method: "event_study", "group", "simple", or "all".
        balance_e : int, optional
            Event time to balance on for event study.
        survey_design : SurveyDesign, optional
            Survey design specification for complex survey data. When
            provided, uses survey weights for estimation (weighted Riesz
            representers, weighted logit, weighted OLS) and design-based
            variance for aggregated SEs (overall, event study, group) via
            Taylor Series Linearization or replicate weights. Requires
            ``weight_type='pweight'``.

        Returns
        -------
        StaggeredTripleDiffResults
        """
        from diff_diff.survey import (
            _resolve_survey_for_fit,
            _validate_unit_constant_survey,
            compute_survey_metadata,
        )

        resolved_survey, survey_weights, survey_weight_type, survey_metadata = (
            _resolve_survey_for_fit(survey_design, data, "analytical")
        )

        if resolved_survey is not None:
            _validate_unit_constant_survey(data, unit, survey_design)
            if resolved_survey.weight_type != "pweight":
                raise ValueError(
                    f"StaggeredTripleDifference survey support requires "
                    f"weight_type='pweight', got '{resolved_survey.weight_type}'. "
                    f"The survey variance math assumes probability weights."
                )
        if aggregate is not None and aggregate not in [
            "event_study",
            "group",
            "simple",
            "all",
        ]:
            raise ValueError(
                f"aggregate must be 'event_study', 'group', 'simple', or 'all', "
                f"got '{aggregate}'"
            )

        df = data.copy()
        self._validate_inputs(df, outcome, unit, time, first_treat, eligibility, covariates)

        if self.cluster is not None:
            warnings.warn(
                "cluster parameter is accepted but cluster-robust analytical SEs "
                "are not yet implemented for staggered DDD. Use n_bootstrap > 0 "
                "for unit-level clustered inference via multiplier bootstrap.",
                UserWarning,
                stacklevel=2,
            )

        if first_treat != "first_treat":
            df["first_treat"] = df[first_treat]
        df["first_treat"] = df["first_treat"].replace([np.inf, float("inf")], 0)

        precomputed = self._precompute_structures(
            df,
            outcome,
            unit,
            time,
            eligibility,
            covariates,
            resolved_survey=resolved_survey,
        )

        # Recompute survey metadata from unit-level resolved survey
        if resolved_survey is not None and survey_metadata is not None:
            resolved_survey_unit = precomputed.get("resolved_survey_unit")
            if resolved_survey_unit is not None:
                unit_w = resolved_survey_unit.weights
                survey_metadata = compute_survey_metadata(resolved_survey_unit, unit_w)

        # Survey df for t-distribution critical values
        df_survey = precomputed.get("df_survey")
        if (
            df_survey is None
            and resolved_survey is not None
            and hasattr(resolved_survey, "uses_replicate_variance")
            and resolved_survey.uses_replicate_variance
        ):
            df_survey = 0  # Forces NaN inference for undefined replicate df

        has_survey = resolved_survey is not None

        treatment_groups = precomputed["treatment_groups"]
        time_periods = precomputed["time_periods"]
        all_units = precomputed["all_units"]
        time_to_col = precomputed["time_to_col"]
        unit_cohorts = precomputed["unit_cohorts"]
        eligibility_per_unit = precomputed["eligibility_per_unit"]
        n_units = len(all_units)

        pscore_cache: Dict = {}
        # Skip Cholesky OR cache when survey weights present (X'WX != X'X)
        cho_cache: Dict = {} if not has_survey else None

        group_time_effects: Dict[Tuple, Dict[str, Any]] = {}
        influence_func_info: Dict[Tuple, Dict[str, Any]] = {}
        comparison_group_counts: Dict[Tuple, int] = {}
        gmm_weights_store: Dict[Tuple, Dict] = {}

        for g in treatment_groups:
            # In universal mode, skip the reference period (t == g-1-anticipation)
            # so it's omitted from GT estimation. The event-study mixin injects
            # a synthetic reference row with effect=0, matching CS behavior.
            if self.base_period == "universal":
                universal_base = g - 1 - self.anticipation
                valid_periods = [t for t in time_periods if t != universal_base]
            else:
                valid_periods = time_periods

            for t in valid_periods:
                base_period_val = self._get_base_period(g, t)
                if base_period_val is None:
                    continue
                if base_period_val not in time_to_col:
                    warnings.warn(
                        f"Base period {base_period_val} for (g={g}, t={t}) is "
                        "outside the observed panel. Skipping this cell.",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue
                if t not in time_to_col:
                    continue

                has_never_enabled = bool(np.any(unit_cohorts == 0))

                if self.control_group == "nevertreated":
                    # Only use never-enabled cohort as comparison
                    valid_gc = [0] if has_never_enabled else []
                else:
                    # Use all valid comparison cohorts (not-yet-treated + never)
                    # Threshold accounts for anticipation: cohorts that start
                    # treatment within the anticipation window are contaminated.
                    nyt_threshold = max(t, base_period_val) + self.anticipation
                    valid_gc = [gc for gc in treatment_groups if gc > nyt_threshold and gc != g]
                    if has_never_enabled:
                        valid_gc = [0] + valid_gc

                if not valid_gc:
                    warnings.warn(
                        f"No valid comparison groups for (g={g}, t={t}), skipping.",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue

                treated_mask = (unit_cohorts == g) & (eligibility_per_unit == 1)
                n_treated = int(np.sum(treated_mask))
                if n_treated == 0:
                    continue

                att_vec = []
                inf_raw = []  # unrescaled IFs
                gc_labels = []
                gc_cell_sizes = []  # size_gt_ctrl per surviving gc

                for gc in valid_gc:
                    result = self._compute_ddd_gt_gc(
                        precomputed,
                        g,
                        gc,
                        t,
                        base_period_val,
                        covariates,
                        pscore_cache,
                        cho_cache,
                    )
                    if result is None:
                        continue
                    att_gc, inf_gc, size_gt_ctrl = result
                    if not np.isfinite(att_gc):
                        continue

                    att_vec.append(att_gc)
                    inf_raw.append(inf_gc)
                    gc_labels.append(gc)
                    gc_cell_sizes.append(size_gt_ctrl)

                if not att_vec:
                    continue

                # Compute size_gt from SURVIVING comparison cohorts only
                # (not from all initially valid gc's)
                surviving_units = treated_mask.copy()
                for gc in gc_labels:
                    surviving_units |= (unit_cohorts == gc) | (unit_cohorts == g)
                survey_w = precomputed.get("survey_weights")
                if survey_w is not None:
                    size_gt = float(np.sum(survey_w[surviving_units]))
                else:
                    size_gt = float(np.sum(surviving_units))

                # Apply IF rescaling now that size_gt is known
                inf_matrix = []
                for inf_gc, size_gt_ctrl in zip(inf_raw, gc_cell_sizes):
                    if size_gt_ctrl > 0:
                        inf_gc = inf_gc * (size_gt / size_gt_ctrl)
                    inf_matrix.append(inf_gc)

                att_gmm, inf_gmm, gmm_w, se_gt = self._combine_gmm(
                    np.array(att_vec),
                    np.array(inf_matrix),
                    n_units,
                )

                if not np.isfinite(att_gmm):
                    continue

                # R's single-gc SE uses size_gt in denominator, not n_total.
                # For multi-gc (GMM), the size_gt factor is already in Omega
                # via the per-gc rescaling, so n_total is correct.
                if len(gc_labels) == 1:
                    se_gt = float(np.sqrt(np.sum(inf_gmm**2) / size_gt**2))

                if not np.isfinite(se_gt) or se_gt <= 0:
                    se_gt = np.nan

                t_stat, p_value, conf_int = safe_inference(
                    att_gmm, se_gt, alpha=self.alpha, df=df_survey
                )

                # Rescale IF for mixin compatibility.
                # R stores IF * (n/size_gt) in inf_func_mat, then uses
                # SE = sqrt(sum(IF^2)/n^2) = sqrt(sum(psi^2)) with psi = IF/n.
                # We need psi = IF_rescaled / n so mixin's sqrt(sum(psi^2)) works.
                # IF is already at size_gt/size_gt_ctrl scale from above.
                # Apply the final n/size_gt factor, then divide by n for mixin.
                inf_gmm_rescaled = inf_gmm * (n_units / size_gt)
                inf_gmm_scaled = inf_gmm_rescaled / n_units

                treated_idx = np.where(treated_mask)[0]
                treated_inf = inf_gmm_scaled[treated_idx]
                nonzero_mask = (inf_gmm_scaled != 0) & ~treated_mask
                control_idx = np.where(nonzero_mask)[0]
                control_inf = inf_gmm_scaled[control_idx]
                n_control = int(np.sum(nonzero_mask))

                group_time_effects[(g, t)] = {
                    "effect": att_gmm,
                    "se": se_gt,
                    "t_stat": t_stat,
                    "p_value": p_value,
                    "conf_int": conf_int,
                    "n_treated": n_treated,
                    "n_control": n_control,
                }
                influence_func_info[(g, t)] = {
                    "treated_idx": treated_idx,
                    "control_idx": control_idx,
                    "treated_units": all_units[treated_idx],
                    "control_units": all_units[control_idx],
                    "treated_inf": treated_inf,
                    "control_inf": control_inf,
                }
                comparison_group_counts[(g, t)] = len(gc_labels)
                gmm_weights_store[(g, t)] = dict(zip(gc_labels, gmm_w.tolist()))

        if not group_time_effects:
            raise ValueError(
                "No valid group-time effects could be computed. "
                "Check that the data has sufficient variation in treatment "
                "timing and eligibility."
            )

        # For aggregation: use eligible-treated-only cohort assignments so
        # WIF weights match the point estimate weights (n_treated per cohort,
        # i.e. P(S=g, Q=1)). This matches the paper's Eq 4.13 which defines
        # aggregation weights over the treated population (G_i defined only
        # for Q=1 units). Ineligible units get cohort=0 so they don't
        # contribute to pg for any treatment group.
        # Both precomputed["unit_cohorts"] AND df["first_treat"] must be
        # zeroed for ineligible units because the WIF code reads both.
        precomputed_agg = dict(precomputed)
        cohorts_for_agg = precomputed["unit_cohorts"].copy()
        cohorts_for_agg[eligibility_per_unit == 0] = 0
        precomputed_agg["unit_cohorts"] = cohorts_for_agg

        df_agg = df.copy()
        df_agg.loc[df_agg[eligibility] == 0, "first_treat"] = 0

        # Overall ATT via aggregation mixin
        overall_att, overall_se, overall_effective_df = self._aggregate_simple(
            group_time_effects, influence_func_info, df_agg, unit, precomputed_agg
        )
        # Use per-statistic effective df from replicate aggregation if available;
        # otherwise fall back to the original df from the survey design.
        if overall_effective_df is not None:
            df_survey = overall_effective_df
            if survey_metadata is not None:
                survey_metadata.df_survey = df_survey
        overall_t_stat, overall_p_value, overall_conf_int = safe_inference(
            overall_att, overall_se, alpha=self.alpha, df=df_survey
        )

        # Aggregations
        event_study_effects = None
        group_effects = None
        if aggregate in ("event_study", "all"):
            event_study_effects = self._aggregate_event_study(
                group_time_effects,
                influence_func_info,
                treatment_groups,
                time_periods,
                balance_e,
                df_agg,
                unit,
                precomputed_agg,
            )
        if aggregate in ("group", "all"):
            group_effects = self._aggregate_by_group(
                group_time_effects,
                influence_func_info,
                treatment_groups,
                precomputed_agg,
                df_agg,
                unit,
            )

        # Reject replicate-weight designs for bootstrap — replicate variance
        # is an analytical alternative, not compatible with bootstrap
        if (
            self.n_bootstrap > 0
            and resolved_survey is not None
            and hasattr(resolved_survey, "uses_replicate_variance")
            and resolved_survey.uses_replicate_variance
        ):
            raise NotImplementedError(
                "StaggeredTripleDifference bootstrap (n_bootstrap > 0) is not "
                "supported with replicate-weight survey designs. Replicate "
                "weights provide analytical variance; use n_bootstrap=0 instead."
            )

        # Bootstrap
        bootstrap_results = None
        cband_crit_value = None
        if self.n_bootstrap > 0:
            bootstrap_results = self._run_multiplier_bootstrap(
                group_time_effects,
                influence_func_info,
                aggregate,
                balance_e,
                treatment_groups,
                time_periods,
                df_agg,
                unit,
                precomputed_agg,
                self.cband,
            )
            if bootstrap_results is not None:
                overall_se = bootstrap_results.overall_att_se
                overall_t_stat, overall_p_value, overall_conf_int = safe_inference(
                    overall_att, overall_se, alpha=self.alpha, df=df_survey
                )
                overall_conf_int = bootstrap_results.overall_att_ci
                overall_p_value = bootstrap_results.overall_att_p_value
                if bootstrap_results.cband_crit_value is not None:
                    cband_crit_value = bootstrap_results.cband_crit_value

                # Update group-time effects with bootstrap SEs
                if bootstrap_results.group_time_ses:
                    for gt_key in group_time_effects:
                        if gt_key in bootstrap_results.group_time_ses:
                            group_time_effects[gt_key]["se"] = bootstrap_results.group_time_ses[
                                gt_key
                            ]
                            group_time_effects[gt_key]["conf_int"] = (
                                bootstrap_results.group_time_cis[gt_key]
                            )
                            group_time_effects[gt_key]["p_value"] = (
                                bootstrap_results.group_time_p_values[gt_key]
                            )
                            t_val, _, _ = safe_inference(
                                group_time_effects[gt_key]["effect"],
                                bootstrap_results.group_time_ses[gt_key],
                                alpha=self.alpha,
                                df=df_survey,
                            )
                            group_time_effects[gt_key]["t_stat"] = t_val

                if event_study_effects and bootstrap_results.event_study_ses:
                    for e_key in event_study_effects:
                        if e_key in bootstrap_results.event_study_ses:
                            event_study_effects[e_key]["se"] = bootstrap_results.event_study_ses[
                                e_key
                            ]
                            event_study_effects[e_key]["conf_int"] = (
                                bootstrap_results.event_study_cis[e_key]
                            )
                            event_study_effects[e_key]["p_value"] = (
                                bootstrap_results.event_study_p_values[e_key]
                            )
                            t_val, _, _ = safe_inference(
                                event_study_effects[e_key]["effect"],
                                bootstrap_results.event_study_ses[e_key],
                                alpha=self.alpha,
                                df=df_survey,
                            )
                            event_study_effects[e_key]["t_stat"] = t_val
                            if cband_crit_value is not None:
                                bs_se = bootstrap_results.event_study_ses[e_key]
                                eff = event_study_effects[e_key]["effect"]
                                event_study_effects[e_key]["cband_conf_int"] = (
                                    eff - cband_crit_value * bs_se,
                                    eff + cband_crit_value * bs_se,
                                )

                # Update group effects with bootstrap SEs
                if (
                    group_effects
                    and bootstrap_results.group_effect_ses is not None
                    and bootstrap_results.group_effect_cis is not None
                    and bootstrap_results.group_effect_p_values is not None
                ):
                    grp_keys = [g for g in group_effects if g in bootstrap_results.group_effect_ses]
                    for g_key in grp_keys:
                        group_effects[g_key]["se"] = bootstrap_results.group_effect_ses[g_key]
                        group_effects[g_key]["conf_int"] = bootstrap_results.group_effect_cis[g_key]
                        group_effects[g_key]["p_value"] = bootstrap_results.group_effect_p_values[
                            g_key
                        ]
                        t_val, _, _ = safe_inference(
                            group_effects[g_key]["effect"],
                            bootstrap_results.group_effect_ses[g_key],
                            alpha=self.alpha,
                            df=df_survey,
                        )
                        group_effects[g_key]["t_stat"] = t_val

        n_treated_units = int(np.sum((unit_cohorts > 0) & (eligibility_per_unit == 1)))
        n_control_units = n_units - n_treated_units
        n_never_enabled = int(np.sum(unit_cohorts == 0))
        n_eligible = int(np.sum(eligibility_per_unit == 1))
        n_ineligible = int(np.sum(eligibility_per_unit == 0))

        self.results_ = StaggeredTripleDiffResults(
            group_time_effects=group_time_effects,
            overall_att=overall_att,
            overall_se=overall_se,
            overall_t_stat=overall_t_stat,
            overall_p_value=overall_p_value,
            overall_conf_int=overall_conf_int,
            groups=treatment_groups,
            time_periods=time_periods,
            n_obs=len(df),
            n_treated_units=n_treated_units,
            n_control_units=n_control_units,
            n_never_enabled=n_never_enabled,
            n_eligible=n_eligible,
            n_ineligible=n_ineligible,
            alpha=self.alpha,
            control_group=self.control_group,
            base_period=self.base_period,
            estimation_method=self.estimation_method,
            event_study_effects=event_study_effects,
            group_effects=group_effects,
            bootstrap_results=bootstrap_results,
            cband_crit_value=cband_crit_value,
            pscore_trim=self.pscore_trim,
            survey_metadata=survey_metadata,
            comparison_group_counts=comparison_group_counts,
            gmm_weights=gmm_weights_store,
        )
        self.is_fitted_ = True
        return self.results_

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_inputs(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        eligibility: str,
        covariates: Optional[List[str]],
    ) -> None:
        """Validate input data."""
        required_cols = [outcome, unit, time, first_treat, eligibility]
        if covariates:
            required_cols.extend(covariates)
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        elig_vals = df[eligibility].dropna().unique()
        if not set(elig_vals).issubset({0, 1, 0.0, 1.0}):
            raise ValueError(
                f"Eligibility column '{eligibility}' must be binary (0/1). "
                f"Found values: {sorted(elig_vals)}"
            )
        elig_by_unit = df.groupby(unit)[eligibility].nunique()
        varying = elig_by_unit[elig_by_unit > 1]
        if len(varying) > 0:
            raise ValueError(
                f"Eligibility must be time-invariant within units. "
                f"Found {len(varying)} units with varying eligibility."
            )
        for col in [outcome, first_treat, eligibility]:
            if df[col].isna().any():
                raise ValueError(f"Column '{col}' contains missing values.")

        # Reject non-finite outcomes (Inf/-Inf)
        if not np.all(np.isfinite(df[outcome])):
            raise ValueError(
                f"Column '{outcome}' contains non-finite values (Inf/-Inf). "
                "All outcome values must be finite."
            )

        # Reject non-finite covariates
        if covariates:
            for cov in covariates:
                if df[cov].isna().any():
                    raise ValueError(f"Covariate '{cov}' contains missing values.")
                if not np.all(np.isfinite(df[cov])):
                    raise ValueError(f"Covariate '{cov}' contains non-finite values.")
        if df[eligibility].nunique() < 2:
            raise ValueError(
                "Need both eligible (Q=1) and ineligible (Q=0) units. "
                f"Only found Q={df[eligibility].unique()[0]}."
            )

        # Check unique (unit, time) pairs — no duplicate rows
        dup = df.duplicated(subset=[unit, time], keep=False)
        if dup.any():
            raise ValueError(
                f"Duplicate (unit, time) rows found. "
                f"{int(dup.sum())} duplicates detected. Panel must have unique rows."
            )

        # Check balanced panel — every unit observed in exactly the global period set
        global_periods = set(df[time].unique())
        n_global_periods = len(global_periods)
        unit_period_sets = df.groupby(unit)[time].apply(set)
        mismatched = unit_period_sets[unit_period_sets != global_periods]
        if len(mismatched) > 0:
            raise ValueError(
                "Unbalanced panel detected. All units must be observed in "
                f"all {n_global_periods} periods. "
                f"Found {len(mismatched)} units with different period sets."
            )

        # Check time-invariant first_treat
        ft_by_unit = df.groupby(unit)[first_treat].nunique()
        varying_ft = ft_by_unit[ft_by_unit > 1]
        if len(varying_ft) > 0:
            raise ValueError(
                f"first_treat must be time-invariant within units. "
                f"Found {len(varying_ft)} units with varying first_treat."
            )

        # Check time-invariant covariates
        if covariates:
            for cov in covariates:
                cov_nunique = df.groupby(unit)[cov].nunique()
                varying_cov = cov_nunique[cov_nunique > 1]
                if len(varying_cov) > 0:
                    raise ValueError(
                        f"Covariate '{cov}' must be time-invariant within units. "
                        f"Found {len(varying_cov)} units with varying values."
                    )

    # ------------------------------------------------------------------
    # Precomputation
    # ------------------------------------------------------------------

    def _precompute_structures(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        eligibility: str,
        covariates: Optional[List[str]],
        resolved_survey=None,
    ) -> PrecomputedData:
        """Build precomputed structures for efficient computation."""
        all_units = np.array(sorted(df[unit].unique()))
        time_periods = sorted(df[time].unique())
        n_units = len(all_units)
        n_periods = len(time_periods)

        unit_to_idx = {u: i for i, u in enumerate(all_units)}
        time_to_col = {t: j for j, t in enumerate(time_periods)}

        outcome_matrix = np.full((n_units, n_periods), np.nan)
        for _, row in df.iterrows():
            u_idx = unit_to_idx[row[unit]]
            t_idx = time_to_col[row[time]]
            outcome_matrix[u_idx, t_idx] = row[outcome]

        unit_df = df.groupby(unit).first().reindex(all_units)
        unit_cohorts = unit_df["first_treat"].values.astype(float)
        eligibility_per_unit = unit_df[eligibility].values.astype(int)

        treatment_groups = sorted([g for g in np.unique(unit_cohorts) if g > 0])

        covariate_matrix = None
        if covariates:
            cov_wide = {}
            for cov in covariates:
                cov_vals = np.full(n_units, np.nan)
                for u_id, idx in unit_to_idx.items():
                    u_data = df.loc[df[unit] == u_id, cov]
                    if len(u_data) > 0:
                        cov_vals[idx] = u_data.iloc[0]
                cov_wide[cov] = cov_vals
            covariate_matrix = np.column_stack(list(cov_wide.values()))

        # Extract per-unit survey weights and collapse design to unit level
        survey_weights_arr = None
        resolved_survey_unit = None
        if resolved_survey is not None:
            from diff_diff.survey import collapse_survey_to_unit_level

            survey_weights_arr = (
                pd.Series(resolved_survey.weights, index=df.index)
                .groupby(df[unit])
                .first()
                .reindex(all_units)
                .values.astype(np.float64)
            )
            # Normalize to sum=n for aggregation/rescaling (matches pweight
            # convention). Raw weights preserved in resolved_survey_unit for
            # replicate w_r/w_full ratios — those are inherently scale-invariant.
            sw_sum = np.sum(survey_weights_arr)
            if sw_sum > 0:
                survey_weights_arr = survey_weights_arr * (len(survey_weights_arr) / sw_sum)
            resolved_survey_unit = collapse_survey_to_unit_level(
                resolved_survey, df, unit, all_units
            )

        return {
            "all_units": all_units,
            "unit_to_idx": unit_to_idx,
            "time_periods": time_periods,
            "time_to_col": time_to_col,
            "outcome_matrix": outcome_matrix,
            "unit_cohorts": unit_cohorts,
            "eligibility_per_unit": eligibility_per_unit,
            "treatment_groups": treatment_groups,
            "covariate_matrix": covariate_matrix,
            "n_units": n_units,
            "n_periods": n_periods,
            "survey_weights": survey_weights_arr,
            "resolved_survey_unit": resolved_survey_unit,
            "df_survey": (
                resolved_survey_unit.df_survey if resolved_survey_unit is not None else None
            ),
        }

    # ------------------------------------------------------------------
    # Base period
    # ------------------------------------------------------------------

    def _get_base_period(self, g: Any, t: Any) -> Optional[Any]:
        """Determine base period for a (g, t) pair."""
        if self.base_period == "universal":
            return g - 1 - self.anticipation
        else:
            if t < g - self.anticipation:
                return t - 1
            else:
                return g - 1 - self.anticipation

    # ------------------------------------------------------------------
    # Three-DiD DDD for one (g, g_c, t) triple
    # ------------------------------------------------------------------

    def _compute_ddd_gt_gc(
        self,
        precomputed: PrecomputedData,
        g: Any,
        g_c: Any,
        t: Any,
        base_period_val: Any,
        covariates: Optional[List[str]],
        pscore_cache: Dict,
        cho_cache: Optional[Dict],
    ) -> Optional[Tuple[float, np.ndarray, float]]:
        """
        Compute DDD ATT for one (g, g_c, t) triple.

        Returns (att_ddd, inf_full_n_units, size_gt_ctrl) or None.
        """
        outcome_matrix = precomputed["outcome_matrix"]
        time_to_col = precomputed["time_to_col"]
        unit_cohorts = precomputed["unit_cohorts"]
        eligibility_per_unit = precomputed["eligibility_per_unit"]
        covariate_matrix = precomputed["covariate_matrix"]
        n_units = precomputed["n_units"]
        survey_weights = precomputed.get("survey_weights")

        t_col = time_to_col[t]
        b_col = time_to_col[base_period_val]

        # Four sub-groups within this (g, g_c) cell
        treated_mask = (unit_cohorts == g) & (eligibility_per_unit == 1)  # subgroup 4
        sub_a_mask = (unit_cohorts == g) & (eligibility_per_unit == 0)  # subgroup 3
        sub_b_mask = (unit_cohorts == g_c) & (eligibility_per_unit == 1)  # subgroup 2
        sub_c_mask = (unit_cohorts == g_c) & (eligibility_per_unit == 0)  # subgroup 1

        n_treated = int(np.sum(treated_mask))
        n_a = int(np.sum(sub_a_mask))
        n_b = int(np.sum(sub_b_mask))
        n_c = int(np.sum(sub_c_mask))

        # Check for empty subgroups (by count or by survey weight mass)
        empty = []
        if n_treated == 0:
            empty.append(f"(S={g},Q=1)")
        if n_a == 0:
            empty.append(f"(S={g},Q=0)")
        if n_b == 0:
            empty.append(f"(S={g_c},Q=1)")
        if n_c == 0:
            empty.append(f"(S={g_c},Q=0)")
        # Zero survey-weight mass after subpopulation filtering = effectively empty
        if not empty and survey_weights is not None:
            if np.sum(survey_weights[treated_mask]) <= 0:
                empty.append(f"(S={g},Q=1,mass=0)")
            if np.sum(survey_weights[sub_a_mask]) <= 0:
                empty.append(f"(S={g},Q=0,mass=0)")
            if np.sum(survey_weights[sub_b_mask]) <= 0:
                empty.append(f"(S={g_c},Q=1,mass=0)")
            if np.sum(survey_weights[sub_c_mask]) <= 0:
                empty.append(f"(S={g_c},Q=0,mass=0)")
        if empty:
            warnings.warn(
                f"Empty subgroup(s) {', '.join(empty)} for "
                f"(g={g}, g_c={g_c}, t={t}). "
                "Comparison unidentified, skipping.",
                UserWarning,
                stacklevel=3,
            )
            return None

        if min(n_treated, n_a, n_b, n_c) < 5:
            warnings.warn(
                f"Small cell size for (g={g}, g_c={g_c}, t={t}). " "Estimates may be unreliable.",
                UserWarning,
                stacklevel=3,
            )

        # Outcome changes
        delta_y_all = outcome_matrix[:, t_col] - outcome_matrix[:, b_col]
        valid = np.isfinite(delta_y_all)
        for m in [treated_mask, sub_a_mask, sub_b_mask, sub_c_mask]:
            if not np.all(valid[m]):
                return None

        # Three pairwise DiDs, each on a 2-cell subset
        # DiD_A: subgroup 4 vs 3 (treated-eligible vs treated-ineligible)
        pair_a_mask = treated_mask | sub_a_mask
        did_a = self._run_pairwise_did(
            delta_y_all,
            pair_a_mask,
            treated_mask,
            sub_a_mask,
            covariate_matrix,
            pscore_cache,
            (g, g, 0, base_period_val),
            cho_cache,
            ("a", g, g, base_period_val),
            survey_weights=survey_weights,
        )

        # DiD_B: subgroup 4 vs 2 (treated-eligible vs control-eligible)
        pair_b_mask = treated_mask | sub_b_mask
        did_b = self._run_pairwise_did(
            delta_y_all,
            pair_b_mask,
            treated_mask,
            sub_b_mask,
            covariate_matrix,
            pscore_cache,
            (g, g_c, 1, base_period_val),
            cho_cache,
            ("b", g, g_c, base_period_val),
            survey_weights=survey_weights,
        )

        # DiD_C: subgroup 4 vs 1 (treated-eligible vs control-ineligible)
        pair_c_mask = treated_mask | sub_c_mask
        did_c = self._run_pairwise_did(
            delta_y_all,
            pair_c_mask,
            treated_mask,
            sub_c_mask,
            covariate_matrix,
            pscore_cache,
            (g, g_c, 0, base_period_val),
            cho_cache,
            ("c", g, g_c, base_period_val),
            survey_weights=survey_weights,
        )

        if did_a is None or did_b is None or did_c is None:
            return None

        att_a, inf_a = did_a
        att_b, inf_b = did_b
        att_c, inf_c = did_c

        att_ddd = att_a + att_b - att_c

        # Three-DiD IF combination: w_j = n_cell / n_pair_j (R's att_dr convention)
        # With survey weights, use survey-weighted cell sizes
        if survey_weights is not None:
            sw_4 = float(np.sum(survey_weights[treated_mask]))
            sw_3 = float(np.sum(survey_weights[sub_a_mask]))
            sw_2 = float(np.sum(survey_weights[sub_b_mask]))
            sw_1 = float(np.sum(survey_weights[sub_c_mask]))
            n_cell_w = sw_4 + sw_3 + sw_2 + sw_1
            n_pair_a_w = sw_4 + sw_3
            n_pair_b_w = sw_4 + sw_2
            n_pair_c_w = sw_4 + sw_1
            w_3 = n_cell_w / n_pair_a_w if n_pair_a_w > 0 else 1.0
            w_2 = n_cell_w / n_pair_b_w if n_pair_b_w > 0 else 1.0
            w_1 = n_cell_w / n_pair_c_w if n_pair_c_w > 0 else 1.0
            size_gt_ctrl = n_cell_w
        else:
            n_cell = n_treated + n_a + n_b + n_c
            n_pair_a = n_treated + n_a
            n_pair_b = n_treated + n_b
            n_pair_c = n_treated + n_c
            w_3 = n_cell / n_pair_a if n_pair_a > 0 else 1.0
            w_2 = n_cell / n_pair_b if n_pair_b > 0 else 1.0
            w_1 = n_cell / n_pair_c if n_pair_c > 0 else 1.0
            size_gt_ctrl = float(n_cell)

        # Scatter pair-level IFs into n_units-length vector
        inf_full = np.zeros(n_units)
        pair_a_idx = np.where(pair_a_mask)[0]
        pair_b_idx = np.where(pair_b_mask)[0]
        pair_c_idx = np.where(pair_c_mask)[0]

        inf_full[pair_a_idx] += w_3 * inf_a
        inf_full[pair_b_idx] += w_2 * inf_b
        inf_full[pair_c_idx] -= w_1 * inf_c

        return att_ddd, inf_full, size_gt_ctrl

    # ------------------------------------------------------------------
    # Pairwise DiD (matches R's compute_did)
    # ------------------------------------------------------------------

    def _run_pairwise_did(
        self,
        delta_y_all: np.ndarray,
        pair_mask: np.ndarray,
        treated_mask: np.ndarray,
        control_mask: np.ndarray,
        covariate_matrix: Optional[np.ndarray],
        pscore_cache: Dict,
        pscore_key: Any,
        cho_cache: Optional[Dict],
        cho_key: Any,
        survey_weights: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[float, np.ndarray]]:
        """
        Compute a single pairwise DiD ATT and IF on a 2-cell subset.

        Matches R's triplediff::compute_did() formulation exactly:
        Riesz/Hajek normalization, PS + OR IF corrections.

        Returns (att, inf_func) where inf_func has length n_pair,
        ordered by pair_mask indices. Returns None if insufficient data.
        """
        pair_idx = np.where(pair_mask)[0]
        n_pair = len(pair_idx)
        if n_pair == 0:
            return None

        delta_y = delta_y_all[pair_idx]
        PA4 = treated_mask[pair_idx].astype(float)
        PAa = control_mask[pair_idx].astype(float)
        sw_pair = survey_weights[pair_idx] if survey_weights is not None else None

        n_t = int(np.sum(PA4))
        n_c = int(np.sum(PAa))
        if n_t == 0 or n_c == 0:
            return None

        has_covariates = covariate_matrix is not None and self.estimation_method != "none"

        # Build covariate matrix with intercept for the pair
        covX = None
        if has_covariates:
            X_pair = covariate_matrix[pair_idx]
            covX = np.column_stack([np.ones(n_pair), X_pair])

        # Compute nuisance parameters based on estimation method
        pscore = None
        hessian = None
        or_delta = np.zeros(n_pair)

        if self.estimation_method in ("ipw", "dr") and covX is not None:
            pscore, hessian = self._compute_pscore(
                PA4,
                covX,
                pscore_cache,
                pscore_key,
                survey_weights=sw_pair,
            )

        if self.estimation_method in ("reg", "dr") and covX is not None:
            # Skip Cholesky cache when survey weights present (cho_cache=None)
            or_delta = self._compute_or(
                delta_y,
                PAa,
                covX,
                cho_cache,
                cho_key,
                survey_weights=sw_pair,
            )

        # Compute ATT and IF (R's compute_did formulation)
        return self._compute_did_panel(
            delta_y,
            PA4,
            PAa,
            covX,
            pscore,
            hessian,
            or_delta,
            survey_weights=sw_pair,
        )

    # ------------------------------------------------------------------
    # Core DR/IPW/RA computation (matches R's compute_did exactly)
    # ------------------------------------------------------------------

    def _compute_did_panel(
        self,
        delta_y: np.ndarray,
        PA4: np.ndarray,
        PAa: np.ndarray,
        covX: Optional[np.ndarray],
        pscore: Optional[np.ndarray],
        hessian: Optional[np.ndarray],
        or_delta: np.ndarray,
        survey_weights: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray]:
        """
        Pairwise DiD ATT and influence function.
        Matches R's triplediff::compute_did() line-by-line.

        Parameters
        ----------
        delta_y : outcome changes for 2-cell subset (n_pair,)
        PA4 : treated indicator (n_pair,)
        PAa : control indicator (n_pair,)
        covX : covariate matrix with intercept (n_pair, p) or None
        pscore : propensity scores (n_pair,) or None
        hessian : (X'WX)^{-1} * n_pair or None
        or_delta : OR predictions (n_pair,), zeros if no covariates
        survey_weights : per-observation survey weights (n_pair,) or None

        Returns
        -------
        (att, inf_func) where inf_func has length n_pair.
        """
        n_pair = len(delta_y)
        est = self.estimation_method

        # Riesz representers (R lines 243-250)
        if est == "reg" or pscore is None:
            w_treat = PA4.copy()
            w_control = PAa.copy()
        else:
            w_treat = PA4.copy()
            pscore_safe = np.clip(pscore, self.pscore_trim, 1 - self.pscore_trim)
            w_control = pscore_safe * PAa / (1 - pscore_safe)

        # Incorporate survey weights into Riesz representers
        if survey_weights is not None:
            w_treat = w_treat * survey_weights
            w_control = w_control * survey_weights

        # DR ATT via Hajek normalization (R lines 251-256)
        resid = delta_y - or_delta
        riesz_treat = w_treat * resid
        riesz_control = w_control * resid

        mean_w_treat = np.mean(w_treat)
        mean_w_control = np.mean(w_control)

        if mean_w_treat <= 0 or mean_w_control <= 0:
            return float("nan"), np.zeros(n_pair)

        att_treat = np.mean(riesz_treat) / mean_w_treat
        att_control = np.mean(riesz_control) / mean_w_control
        dr_att = att_treat - att_control

        # Base IF (R lines 302-304)
        inf_treat_did = riesz_treat - w_treat * att_treat
        inf_control_did = riesz_control - w_control * att_control

        # PS correction (R lines 262-273) — IPW and DR only
        inf_control_pscore = 0.0
        if est != "reg" and hessian is not None and covX is not None:
            M2 = np.mean((w_control * (resid - att_control))[:, None] * covX, axis=0)
            if survey_weights is not None:
                score_ps = survey_weights[:, None] * (PA4 - pscore_safe)[:, None] * covX
            else:
                score_ps = (PA4 - pscore_safe)[:, None] * covX
            asy_lin_rep_ps = score_ps @ hessian
            inf_control_pscore = asy_lin_rep_ps @ M2

        # OR correction (R lines 278-300) — reg and DR only
        inf_treat_or = 0.0
        inf_cont_or = 0.0
        if est != "ipw" and covX is not None:
            M1 = np.mean(w_treat[:, None] * covX, axis=0)
            M3 = np.mean(w_control[:, None] * covX, axis=0)

            if survey_weights is not None:
                or_x = (PAa * survey_weights)[:, None] * covX
                or_ex = (PAa * survey_weights * resid)[:, None] * covX
            else:
                or_x = PAa[:, None] * covX
                or_ex = (PAa * resid)[:, None] * covX
            XpX = or_x.T @ covX / n_pair

            try:
                asy_linear_or = (np.linalg.solve(XpX, or_ex.T)).T
            except np.linalg.LinAlgError:
                asy_linear_or = (np.linalg.lstsq(XpX, or_ex.T, rcond=None)[0]).T

            inf_treat_or = -(asy_linear_or @ M1)
            inf_cont_or = -(asy_linear_or @ M3)

        # Final IF assembly (R lines 307-310)
        inf_control = (inf_control_did + inf_control_pscore + inf_cont_or) / mean_w_control
        inf_treat = (inf_treat_did + inf_treat_or) / mean_w_treat
        inf_func = inf_treat - inf_control

        return float(dr_att), inf_func

    # ------------------------------------------------------------------
    # Nuisance parameter computation
    # ------------------------------------------------------------------

    def _compute_pscore(
        self,
        PA4: np.ndarray,
        covX: np.ndarray,
        pscore_cache: Dict,
        pscore_key: Any,
        survey_weights: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit logistic P(PA4=1|X). Returns (pscore, hessian).

        hessian = (X'WX)^{-1} * n_pair, matching R's convention.
        When survey_weights is provided, IRLS uses survey-weighted
        working weights and the hessian accounts for survey weights.
        """
        cached = pscore_cache.get(pscore_key)
        n_pair = len(PA4)

        if cached is not None:
            beta_logistic = cached
            z = np.dot(covX, beta_logistic)
            z = np.clip(z, -500, 500)
            pscore = 1 / (1 + np.exp(-z))
        else:
            X_no_intercept = covX[:, 1:]  # solve_logit adds its own intercept
            try:
                beta_logistic, pscore = solve_logit(
                    X_no_intercept,
                    PA4,
                    rank_deficient_action=self.rank_deficient_action,
                    weights=survey_weights,
                )
                _check_propensity_diagnostics(pscore, self.pscore_trim)
                # Zero-fill NaN coefficients (from rank-deficient columns)
                # before caching, so cache reuse doesn't propagate NaN.
                beta_clean = np.where(np.isfinite(beta_logistic), beta_logistic, 0.0)
                pscore_cache[pscore_key] = beta_clean
            except (np.linalg.LinAlgError, ValueError):
                if self.rank_deficient_action == "error":
                    raise
                warnings.warn(
                    "Propensity score estimation failed. " "Falling back to unconditional.",
                    UserWarning,
                    stacklevel=5,
                )
                # Use survey-weighted treated share when weights available
                if survey_weights is not None:
                    pos = survey_weights > 0
                    if np.any(pos):
                        p_uc = np.average(PA4[pos], weights=survey_weights[pos])
                    else:
                        p_uc = np.mean(PA4)
                else:
                    p_uc = np.mean(PA4)
                pscore = np.full(n_pair, p_uc)
                pscore = np.clip(pscore, self.pscore_trim, 1 - self.pscore_trim)
                # No hessian for unconditional fallback
                return pscore, None

        pscore = np.clip(pscore, 1e-6, 1 - 1e-6)

        # Hessian: (X'WX)^{-1} * n (matching R's compute_pscore)
        W = pscore * (1 - pscore)
        if survey_weights is not None:
            W = W * survey_weights
        XWX = covX.T @ (W[:, None] * covX)
        try:
            hessian = np.linalg.inv(XWX) * n_pair
        except np.linalg.LinAlgError:
            hessian = np.linalg.lstsq(XWX, np.eye(XWX.shape[0]), rcond=None)[0] * n_pair

        return pscore, hessian

    def _compute_or(
        self,
        delta_y: np.ndarray,
        PAa: np.ndarray,
        covX: np.ndarray,
        cho_cache: Optional[Dict],
        cho_key: Any,
        survey_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fit OLS on control outcome changes. Returns or_delta for all pair units.

        Honors self.rank_deficient_action for collinear covariates.
        When survey_weights is provided, uses WLS via solve_ols(weights=...).
        Cholesky cache is disabled for the survey path (cho_cache=None).
        """
        from diff_diff.linalg import solve_ols as _solve_ols

        control_mask = PAa > 0
        n_c = int(np.sum(control_mask))
        if n_c == 0:
            return np.zeros(len(delta_y))

        X_control = covX[control_mask]
        y_control = delta_y[control_mask]
        sw_control = survey_weights[control_mask] if survey_weights is not None else None

        # Try Cholesky cache for fast path (full-rank only)
        # Skipped when cho_cache is None (survey weights present)
        beta = None
        if cho_cache is not None:
            cached_cho = cho_cache.get(cho_key)
            if cached_cho is False:
                pass  # Previously detected rank-deficient; skip Cholesky
            elif cached_cho is not None:
                from scipy import linalg as sp_linalg

                Xty = X_control.T @ y_control
                beta = sp_linalg.cho_solve(cached_cho, Xty)
                if np.any(~np.isfinite(beta)):
                    beta = None
            elif cho_key not in cho_cache:
                XtX = X_control.T @ X_control
                try:
                    from scipy import linalg as sp_linalg

                    cho_factor = sp_linalg.cho_factor(XtX)
                    cho_cache[cho_key] = cho_factor
                    Xty = X_control.T @ y_control
                    beta = sp_linalg.cho_solve(cho_factor, Xty)
                    if np.any(~np.isfinite(beta)):
                        beta = None
                except np.linalg.LinAlgError:
                    cho_cache[cho_key] = False

        if beta is None:
            # Fallback (or survey path): use solve_ols with optional weights
            beta, _, _ = _solve_ols(
                X_control,
                y_control,
                rank_deficient_action=self.rank_deficient_action,
                weights=sw_control,
            )
            beta = np.where(np.isfinite(beta), beta, 0.0)

        return covX @ beta

    # ------------------------------------------------------------------
    # GMM-optimal combination (matches R's att_gt GMM procedure)
    # ------------------------------------------------------------------

    def _combine_gmm(
        self,
        att_vec: np.ndarray,
        inf_func_matrix: np.ndarray,
        n_units: int,
    ) -> Tuple[float, np.ndarray, np.ndarray, float]:
        """
        Combine comparison-group-specific estimates via GMM-optimal weights.

        Returns (att_gmm, inf_gmm, weights, se_gmm).
        """
        k = len(att_vec)

        if k == 1:
            att_gmm = float(att_vec[0])
            inf_gmm = inf_func_matrix[0].copy()
            # R's SE: sqrt(sum(IF^2) / n^2)
            se_gmm = float(np.sqrt(np.sum(inf_gmm**2) / n_units**2))
            return att_gmm, inf_gmm, np.array([1.0]), se_gmm

        # R: OMEGA <- cov(inf_mat_local) — sample covariance, ddof=1
        Omega = np.cov(inf_func_matrix)

        ones = np.ones(k)
        try:
            Omega_inv = np.linalg.inv(Omega)
        except np.linalg.LinAlgError:
            warnings.warn(
                "Singular covariance matrix in GMM combination. " "Using pseudoinverse.",
                UserWarning,
                stacklevel=3,
            )
            Omega_inv = np.linalg.pinv(Omega)

        denom = float(ones @ Omega_inv @ ones)
        if denom <= 0 or not np.isfinite(denom):
            weights = np.full(k, 1.0 / k)
            att_gmm = float(weights @ att_vec)
            inf_gmm = weights @ inf_func_matrix
            se_gmm = float(np.sqrt(np.sum(inf_gmm**2) / n_units**2))
        else:
            weights = (Omega_inv @ ones) / denom
            att_gmm = float(weights @ att_vec)
            inf_gmm = weights @ inf_func_matrix
            # R: gmm_se <- sqrt(1 / (n * sum(inv_OMEGA)))
            se_gmm = float(np.sqrt(1.0 / (n_units * denom)))

        return att_gmm, inf_gmm, weights, se_gmm
