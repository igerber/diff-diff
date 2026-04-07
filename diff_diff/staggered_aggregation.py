"""
Aggregation methods mixin for Callaway-Sant'Anna estimator.

This module provides the mixin class containing methods for aggregating
group-time average treatment effects into summary measures.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from diff_diff.utils import safe_inference_batch

# Type alias for pre-computed structures (defined at module scope for runtime access)
PrecomputedData = Dict[str, Any]


class CallawaySantAnnaAggregationMixin:
    """
    Mixin class providing aggregation methods for CallawaySantAnna estimator.

    This class is not intended to be used standalone. It provides methods
    that are used by the main CallawaySantAnna class to aggregate group-time
    effects into summary measures.
    """

    # Type hints for attributes accessed from the main class
    alpha: float

    # Type hint for anticipation attribute accessed from main class
    anticipation: int

    # Type hint for base_period attribute accessed from main class
    base_period: str

    def _aggregate_simple(
        self,
        group_time_effects: Dict,
        influence_func_info: Dict,
        df: pd.DataFrame,
        unit: str,
        precomputed: Optional["PrecomputedData"] = None,
    ) -> Tuple[float, float]:
        """
        Compute simple weighted average of ATT(g,t).

        Weights by group size (number of treated units).

        Standard errors are computed using influence function aggregation,
        which properly accounts for covariances across (g,t) pairs due to
        shared control units. This includes the wif (weight influence function)
        adjustment from R's `did` package that accounts for uncertainty in
        estimating the group-size weights.

        Note: Only post-treatment effects (t >= g - anticipation) are included
        in the overall ATT. Pre-treatment effects are computed for parallel
        trends assessment but are not aggregated into the overall ATT.
        """
        effects = []
        weights_list = []
        gt_pairs = []
        groups_for_gt = []

        # For survey: compute fixed per-cohort weight sums from the full
        # unit-level sample (matching R's did::aggte pg = n_g / N).
        survey_cohort_weights = None
        if precomputed is not None and precomputed.get("survey_weights") is not None:
            sw = precomputed["survey_weights"]
            unit_cohorts = precomputed["unit_cohorts"]
            survey_cohort_weights = {}
            for g in np.unique(unit_cohorts):
                if g > 0:  # exclude never-treated (0)
                    survey_cohort_weights[g] = float(np.sum(sw[unit_cohorts == g]))

        for (g, t), data in group_time_effects.items():
            # Only include post-treatment effects (t >= g - anticipation)
            # Pre-treatment effects are for parallel trends, not overall ATT
            if t < g - self.anticipation:
                continue
            effects.append(data["effect"])
            # Use fixed cohort-level survey weight sum for aggregation.
            # For RCS, data["agg_weight"] holds the fixed cohort mass;
            # for panel, fallback to data["n_treated"].
            if survey_cohort_weights is not None and g in survey_cohort_weights:
                weights_list.append(survey_cohort_weights[g])
            else:
                weights_list.append(data.get("agg_weight", data["n_treated"]))
            gt_pairs.append((g, t))
            groups_for_gt.append(g)

        # Guard against empty post-treatment set
        if len(effects) == 0:
            import warnings

            warnings.warn(
                "No post-treatment effects available for overall ATT aggregation. "
                "This can occur when cohorts lack post-treatment periods in the data.",
                UserWarning,
                stacklevel=2,
            )
            return np.nan, np.nan, None

        effects = np.array(effects)
        weights = np.array(weights_list, dtype=float)
        groups_for_gt = np.array(groups_for_gt)

        # Exclude NaN effects from aggregation (R's aggte() convention)
        finite_mask = np.isfinite(effects)
        n_nan = int(np.sum(~finite_mask))
        if n_nan > 0:
            import warnings

            warnings.warn(
                f"{n_nan} group-time effect(s) are NaN and excluded from overall ATT "
                "aggregation. Inspect group_time_effects for details.",
                UserWarning,
                stacklevel=2,
            )
            effects = effects[finite_mask]
            weights = weights[finite_mask]
            gt_pairs = [gt for gt, m in zip(gt_pairs, finite_mask) if m]
            groups_for_gt = groups_for_gt[finite_mask]

        if len(effects) == 0:
            import warnings

            warnings.warn(
                "All post-treatment effects are NaN. Cannot compute overall ATT.",
                UserWarning,
                stacklevel=2,
            )
            return np.nan, np.nan, None

        # Normalize weights
        total_weight = np.sum(weights)
        weights_norm = weights / total_weight

        # Weighted average
        overall_att = np.sum(weights_norm * effects)

        # Compute SE using influence function aggregation with wif adjustment
        overall_se, effective_df = self._compute_aggregated_se_with_wif(
            gt_pairs,
            weights_norm,
            effects,
            groups_for_gt,
            influence_func_info,
            df,
            unit,
            precomputed,
        )

        return overall_att, overall_se, effective_df

    def _compute_aggregated_se(
        self,
        gt_pairs: List[Tuple[Any, Any]],
        weights: np.ndarray,
        influence_func_info: Dict,
        n_units: Optional[int] = None,
    ) -> float:
        """
        Compute standard error using influence function aggregation.

        This properly accounts for covariances across (g,t) pairs by
        aggregating unit-level influence functions:

            ψ_i(overall) = Σ_{(g,t)} w_(g,t) × ψ_i(g,t)
            Var(overall) = (1/n) Σ_i [ψ_i]²

        This matches R's `did` package analytical SE formula.

        Parameters
        ----------
        n_units : int, optional
            Size of the canonical index space (len(precomputed['all_units'])).
            When provided, influence function indices (treated_idx, control_idx)
            index directly into this space, eliminating dict lookups.
        """
        if not influence_func_info:
            return 0.0

        if n_units is None:
            # Fallback: infer size from influence function info
            max_idx = 0
            for g, t in gt_pairs:
                if (g, t) in influence_func_info:
                    info = influence_func_info[(g, t)]
                    if len(info["treated_idx"]) > 0:
                        max_idx = max(max_idx, info["treated_idx"].max())
                    if len(info["control_idx"]) > 0:
                        max_idx = max(max_idx, info["control_idx"].max())
            n_units = max_idx + 1

        if n_units == 0:
            return 0.0

        # Aggregate influence functions across (g,t) pairs
        psi_overall = np.zeros(n_units)

        for j, (g, t) in enumerate(gt_pairs):
            if (g, t) not in influence_func_info:
                continue

            info = influence_func_info[(g, t)]
            w = weights[j]

            # Vectorized influence function aggregation using index arrays
            treated_idx = info["treated_idx"]
            if len(treated_idx) > 0:
                np.add.at(psi_overall, treated_idx, w * info["treated_inf"])

            control_idx = info["control_idx"]
            if len(control_idx) > 0:
                np.add.at(psi_overall, control_idx, w * info["control_inf"])

        # Compute variance: Var(θ̄) = (1/n) Σᵢ ψᵢ²
        variance = np.sum(psi_overall**2)
        return np.sqrt(variance)

    def _compute_combined_influence_function(
        self,
        gt_pairs: List[Tuple[Any, Any]],
        weights: np.ndarray,
        effects: np.ndarray,
        groups_for_gt: np.ndarray,
        influence_func_info: Dict,
        df: pd.DataFrame,
        unit: str,
        precomputed: Optional["PrecomputedData"] = None,
        global_unit_to_idx: Optional[Dict[Any, int]] = None,
        n_global_units: Optional[int] = None,
    ) -> Tuple[np.ndarray, Optional[List]]:
        """
        Compute the combined (standard IF + WIF) influence function vector.

        If global_unit_to_idx / n_global_units are provided, the returned vector
        is zero-padded to the global unit set for bootstrap alignment.
        Otherwise, the returned vector is indexed by the local unit set
        (all units appearing in the (g,t) pairs).

        Returns
        -------
        combined_if : np.ndarray
            Per-unit combined influence function (standard IF + WIF).
        all_units : list or None
            Ordered list of units (only when using local indexing).
        """
        if not influence_func_info:
            if n_global_units is not None:
                return np.zeros(n_global_units), None
            return np.zeros(0), None

        # Detect RCS mode via explicit flag. In RCS, obs indices ARE array positions.
        _is_rcs = precomputed is not None and not precomputed.get("is_panel", True)

        # Build unit index mapping (local or global)
        if _is_rcs and n_global_units is not None:
            # RCS: direct indexing — obs indices are the array positions
            n_units = n_global_units
            all_units = None
        elif global_unit_to_idx is not None and n_global_units is not None:
            n_units = n_global_units
            all_units = None  # caller already has the unit list
        else:
            all_units_set: Set[Any] = set()
            for g, t in gt_pairs:
                if (g, t) in influence_func_info:
                    info = influence_func_info[(g, t)]
                    all_units_set.update(info["treated_units"])
                    all_units_set.update(info["control_units"])

            if not all_units_set:
                return np.zeros(0), []

            all_units = sorted(all_units_set)
            n_units = len(all_units)

        # Get unique groups and their information
        unique_groups = sorted(set(groups_for_gt))
        unique_groups_set = set(unique_groups)
        group_to_idx = {g: i for i, g in enumerate(unique_groups)}

        # Check for survey weights in precomputed data
        survey_w = precomputed.get("survey_weights") if precomputed is not None else None

        # Compute group-level probabilities matching R's formula:
        # pg[g] = n_g / n_all (fraction of ALL units in group g)
        # With survey weights: pg[g] = sum(sw_g) / sum(sw_all)
        group_sizes = {}
        if survey_w is not None:
            # Survey-weighted group sizes
            precomputed_cohorts = precomputed["unit_cohorts"]
            for g in unique_groups:
                mask_g = precomputed_cohorts == g
                group_sizes[g] = float(np.sum(survey_w[mask_g]))
            total_weight = float(np.sum(survey_w))
        elif _is_rcs:
            # RCS without survey: count observations per cohort
            precomputed_cohorts = precomputed["unit_cohorts"]
            for g in unique_groups:
                group_sizes[g] = int(np.sum(precomputed_cohorts == g))
            total_weight = float(n_units)
        else:
            for g in unique_groups:
                treated_in_g = df[df["first_treat"] == g][unit].nunique()
                group_sizes[g] = treated_in_g
            total_weight = float(n_units)

        # pg indexed by group
        pg_by_group = np.array([group_sizes[g] / total_weight for g in unique_groups])

        # pg indexed by keeper (each (g,t) pair gets its group's pg)
        pg_keepers = np.array([pg_by_group[group_to_idx[g]] for g in groups_for_gt])
        sum_pg_keepers = np.sum(pg_keepers)

        # Guard against zero weights (no keepers = no variance)
        if sum_pg_keepers == 0:
            return np.zeros(n_units), all_units

        # Standard aggregated influence (without wif)
        psi_standard = np.zeros(n_units)

        for j, (g, t) in enumerate(gt_pairs):
            if (g, t) not in influence_func_info:
                continue

            info = influence_func_info[(g, t)]
            w = weights[j]

            # Vectorized influence function aggregation using precomputed index arrays
            treated_idx = info["treated_idx"]
            if len(treated_idx) > 0:
                np.add.at(psi_standard, treated_idx, w * info["treated_inf"])

            control_idx = info["control_idx"]
            if len(control_idx) > 0:
                np.add.at(psi_standard, control_idx, w * info["control_inf"])

        # Build unit-group array: normalize iterator to (idx, uid) pairs
        unit_groups_array = np.full(n_units, -1, dtype=np.float64)

        if _is_rcs:
            # RCS: direct vectorized assignment — obs indices are positions
            precomputed_cohorts = precomputed["unit_cohorts"]
            for g in unique_groups:
                mask_g = precomputed_cohorts == g
                unit_groups_array[mask_g] = g
        elif global_unit_to_idx is not None:
            idx_uid_pairs = [(idx, uid) for uid, idx in global_unit_to_idx.items()]

            if precomputed is not None:
                precomputed_cohorts = precomputed["unit_cohorts"]
                precomputed_unit_to_idx = precomputed["unit_to_idx"]
                for idx, uid in idx_uid_pairs:
                    if uid in precomputed_unit_to_idx:
                        cohort = precomputed_cohorts[precomputed_unit_to_idx[uid]]
                        if cohort in unique_groups_set:
                            unit_groups_array[idx] = cohort
            else:
                for idx, uid in idx_uid_pairs:
                    unit_first_treat = df[df[unit] == uid]["first_treat"].iloc[0]
                    if unit_first_treat in unique_groups_set:
                        unit_groups_array[idx] = unit_first_treat
        else:
            idx_uid_pairs = list(enumerate(all_units))
            for idx, uid in idx_uid_pairs:
                unit_first_treat = df[df[unit] == uid]["first_treat"].iloc[0]
                if unit_first_treat in unique_groups_set:
                    unit_groups_array[idx] = unit_first_treat

        # Vectorized WIF computation
        groups_for_gt_array = np.array(groups_for_gt)
        indicator_matrix = (
            unit_groups_array[:, np.newaxis] == groups_for_gt_array[np.newaxis, :]
        ).astype(np.float64)

        if survey_w is not None:
            # Survey-weighted WIF matching R's did::wif() / compute.aggte.R.
            # pg_k = E[w_i * 1{G_i=g}] is the weighted group share.
            # IF_i(p_g) = (w_i * 1{G_i=g} - pg_k), NOT s_i * (1{G_i=g} - pg_k).
            # The pg subtraction is NOT weighted by s_i because pg is already
            # the population-level expected value of w_i * 1{G_i=g}.
            if _is_rcs and precomputed is not None:
                # RCS: survey weights are already per-observation, direct indexing
                unit_sw = survey_w
            elif global_unit_to_idx is not None and precomputed is not None:
                unit_sw = np.zeros(n_units)
                precomputed_unit_to_idx_local = precomputed["unit_to_idx"]
                idx_uid_pairs_sw = [(idx, uid) for uid, idx in global_unit_to_idx.items()]
                for idx, uid in idx_uid_pairs_sw:
                    if uid in precomputed_unit_to_idx_local:
                        pc_idx = precomputed_unit_to_idx_local[uid]
                        unit_sw[idx] = survey_w[pc_idx]
            else:
                unit_sw = np.ones(n_units)

            # w_i * 1{G_i == g_k} - pg_k  (matches R's did::wif)
            weighted_indicator = indicator_matrix * unit_sw[:, np.newaxis]
            indicator_diff = weighted_indicator - pg_keepers
            indicator_sum_w = np.sum(indicator_diff, axis=1)

            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                if1_matrix = indicator_diff / sum_pg_keepers
                if2_matrix = np.outer(indicator_sum_w, pg_keepers) / (sum_pg_keepers**2)
                wif_matrix = if1_matrix - if2_matrix
                wif_contrib = wif_matrix @ effects
        else:
            indicator_sum = np.sum(indicator_matrix - pg_keepers, axis=1)

            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                if1_matrix = (indicator_matrix - pg_keepers) / sum_pg_keepers
                if2_matrix = np.outer(indicator_sum, pg_keepers) / (sum_pg_keepers**2)
                wif_matrix = if1_matrix - if2_matrix
                wif_contrib = wif_matrix @ effects

        # Check for non-finite values from edge cases
        if not np.all(np.isfinite(wif_contrib)):
            import warnings

            n_nonfinite = np.sum(~np.isfinite(wif_contrib))
            warnings.warn(
                f"Non-finite values ({n_nonfinite}/{len(wif_contrib)}) in weight influence "
                "function computation. This may occur with very small samples or extreme "
                "weights. Returning NaN for SE to signal invalid inference.",
                RuntimeWarning,
                stacklevel=2,
            )
            nan_result = np.full(n_units, np.nan)
            return nan_result, all_units

        # Scale by 1/total_weight to match R's getSE formula
        # (for non-survey, total_weight == n_units; for survey, total_weight == sum(sw))
        psi_wif = wif_contrib / total_weight

        # Combine standard and wif terms
        psi_total = psi_standard + psi_wif

        return psi_total, all_units

    def _compute_aggregated_se_with_wif(
        self,
        gt_pairs: List[Tuple[Any, Any]],
        weights: np.ndarray,
        effects: np.ndarray,
        groups_for_gt: np.ndarray,
        influence_func_info: Dict,
        df: pd.DataFrame,
        unit: str,
        precomputed: Optional["PrecomputedData"] = None,
        return_psi: bool = False,
    ) -> "Union[float, Tuple[float, np.ndarray]]":
        """
        Compute SE with weight influence function (wif) adjustment.

        This matches R's `did` package approach for aggregation,
        which accounts for uncertainty in estimating group-size weights.

        When a full survey design (strata/PSU/FPC) is available in
        ``precomputed['resolved_survey']``, the design-based variance
        :func:`compute_survey_if_variance` is used instead of the simple
        ``sum(psi^2)`` formula.

        Formula (matching R's did::aggte):
            agg_inf_i = Σ_k w_k × inf_i_k + wif_i × ATT_k
            se = sqrt(mean(agg_inf^2) / n)
        """
        # Extract global unit info for correct pg = n_g / N_total scaling.
        # Without this, the local path builds the unit set from only units in
        # the selected (g,t) pairs, causing pg overestimation at extreme event
        # times where only early-adopter groups have data.
        global_unit_to_idx = None
        n_global_units = None
        if precomputed is not None:
            global_unit_to_idx = precomputed["unit_to_idx"]  # None for RCS
            n_global_units = precomputed.get(
                "canonical_size", len(precomputed.get("all_units", []))
            )
        elif df is not None and unit is not None:
            n_global_units = df[unit].nunique()

        psi_total, _ = self._compute_combined_influence_function(
            gt_pairs,
            weights,
            effects,
            groups_for_gt,
            influence_func_info,
            df,
            unit,
            precomputed,
            global_unit_to_idx=global_unit_to_idx,
            n_global_units=n_global_units,
        )

        if len(psi_total) == 0:
            return (0.0, psi_total) if return_psi else 0.0

        # Check for NaN propagation from non-finite WIF
        if not np.all(np.isfinite(psi_total)):
            return (np.nan, psi_total) if return_psi else np.nan

        # Use design-based variance when full survey design is available
        # Use unit-level resolved survey (panel IF is indexed by unit, not obs)
        resolved_survey = (
            precomputed.get("resolved_survey_unit") if precomputed is not None else None
        )
        if (
            resolved_survey is not None
            and hasattr(resolved_survey, "uses_replicate_variance")
            and resolved_survey.uses_replicate_variance
        ):
            from diff_diff.survey import compute_replicate_if_variance

            variance, n_valid_rep = compute_replicate_if_variance(psi_total, resolved_survey)
            # Compute effective df for this statistic (don't mutate shared state)
            effective_df = None
            if n_valid_rep < resolved_survey.n_replicates:
                effective_df = n_valid_rep - 1 if n_valid_rep > 1 else 0
            if np.isnan(variance):
                se = np.nan
            else:
                se = np.sqrt(max(variance, 0.0))
            if return_psi:
                return (se, psi_total, effective_df)
            return (se, effective_df)

        if resolved_survey is not None and (
            resolved_survey.strata is not None
            or resolved_survey.psu is not None
            or resolved_survey.fpc is not None
        ):
            from diff_diff.survey import compute_survey_if_variance

            variance = compute_survey_if_variance(psi_total, resolved_survey)
            if np.isnan(variance):
                se = np.nan
            else:
                se = np.sqrt(max(variance, 0.0))
            if return_psi:
                return (se, psi_total, None)
            return (se, None)

        variance = np.sum(psi_total**2)
        se = np.sqrt(variance)
        if return_psi:
            return (se, psi_total, None)
        return (se, None)

    def _aggregate_event_study(
        self,
        group_time_effects: Dict,
        influence_func_info: Dict,
        groups: List[Any],
        time_periods: List[Any],
        balance_e: Optional[int] = None,
        df: Optional[pd.DataFrame] = None,
        unit: Optional[str] = None,
        precomputed: Optional["PrecomputedData"] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Aggregate effects by relative time (event study).

        Computes average effect at each event time e = t - g.

        Standard errors include the weight influence function (WIF)
        adjustment that accounts for uncertainty in group-size weights,
        matching R's did::aggte(..., type="dynamic").
        """
        # Organize effects by relative time, keeping track of (g,t) pairs
        effects_by_e: Dict[int, List[Tuple[Tuple[Any, Any], float, float]]] = {}

        # Fixed per-cohort survey weights for aggregation
        survey_cohort_weights = None
        if precomputed is not None and precomputed.get("survey_weights") is not None:
            sw = precomputed["survey_weights"]
            unit_cohorts = precomputed["unit_cohorts"]
            survey_cohort_weights = {}
            for g in np.unique(unit_cohorts):
                if g > 0:
                    survey_cohort_weights[g] = float(np.sum(sw[unit_cohorts == g]))

        for (g, t), data in group_time_effects.items():
            e = t - g  # Relative time
            if e not in effects_by_e:
                effects_by_e[e] = []
            # For RCS, data["agg_weight"] holds the fixed cohort mass;
            # for panel, fallback to data["n_treated"].
            w = (
                survey_cohort_weights[g]
                if survey_cohort_weights is not None and g in survey_cohort_weights
                else data.get("agg_weight", data["n_treated"])
            )
            effects_by_e[e].append(
                (
                    (g, t),  # Keep track of the (g,t) pair
                    data["effect"],
                    w,
                )
            )

        # Balance the panel if requested
        if balance_e is not None:
            # Keep only groups that have effects at relative time balance_e
            groups_at_e = set()
            for (g, t), data in group_time_effects.items():
                if t - g == balance_e and np.isfinite(data["effect"]):
                    groups_at_e.add(g)

            # Filter effects to only include balanced groups
            balanced_effects: Dict[int, List[Tuple[Tuple[Any, Any], float, float]]] = {}
            for (g, t), data in group_time_effects.items():
                if g in groups_at_e:
                    e = t - g
                    if e not in balanced_effects:
                        balanced_effects[e] = []
                    w = (
                        survey_cohort_weights[g]
                        if survey_cohort_weights is not None and g in survey_cohort_weights
                        else data.get("agg_weight", data["n_treated"])
                    )
                    balanced_effects[e].append(
                        (
                            (g, t),
                            data["effect"],
                            w,
                        )
                    )
            effects_by_e = balanced_effects

        # Compute aggregated effects and SEs for all relative periods
        sorted_periods = sorted(effects_by_e.items())
        agg_effects_list = []
        agg_ses_list = []
        agg_n_groups = []
        agg_effective_dfs = []  # Per-horizon effective df (replicate designs)
        _psi_vectors = []  # Per-event-time combined IF vectors for VCV
        _psi_event_times = []  # Event times that contributed a psi column
        for e, effect_list in sorted_periods:
            gt_pairs = [x[0] for x in effect_list]
            effs = np.array([x[1] for x in effect_list])
            ns = np.array([x[2] for x in effect_list], dtype=float)

            # Exclude NaN effects from this period's aggregation
            finite_mask = np.isfinite(effs)
            if not np.all(finite_mask):
                effs = effs[finite_mask]
                ns = ns[finite_mask]
                gt_pairs = [gt for gt, m in zip(gt_pairs, finite_mask) if m]
                if len(effs) == 0:
                    agg_effects_list.append(np.nan)
                    agg_ses_list.append(np.nan)
                    agg_n_groups.append(0)
                    agg_effective_dfs.append(None)
                    continue

            weights = ns / np.sum(ns)
            agg_effect = np.sum(weights * effs)

            # Compute SE with WIF adjustment (matching R's did::aggte)
            groups_for_gt = np.array([g for (g, t) in gt_pairs])
            agg_se, psi_e, eff_df = self._compute_aggregated_se_with_wif(
                gt_pairs,
                weights,
                effs,
                groups_for_gt,
                influence_func_info,
                df,
                unit,
                precomputed,
                return_psi=True,
            )

            agg_effects_list.append(agg_effect)
            agg_ses_list.append(agg_se)
            agg_n_groups.append(len(effect_list))
            agg_effective_dfs.append(eff_df)
            _psi_vectors.append(psi_e)
            _psi_event_times.append(e)

        # Batch inference for all relative periods
        if not agg_effects_list:
            return {}
        # Use per-horizon effective df if any replicate aggregation overrode it;
        # otherwise fall back to the original df from the survey design.
        df_survey_val = precomputed.get("df_survey") if precomputed is not None else None
        # Guard: replicate design with undefined df → NaN inference
        if (
            df_survey_val is None
            and precomputed is not None
            and precomputed.get("resolved_survey_unit") is not None
            and hasattr(precomputed["resolved_survey_unit"], "uses_replicate_variance")
            and precomputed["resolved_survey_unit"].uses_replicate_variance
        ):
            df_survey_val = 0
        # If any horizon has a per-statistic effective df (dropped replicates),
        # use the minimum across horizons for conservative batch inference.
        non_none_dfs = [d for d in agg_effective_dfs if d is not None]
        if non_none_dfs:
            df_survey_val = min(non_none_dfs)
        t_stats, p_values, ci_lowers, ci_uppers = safe_inference_batch(
            np.array(agg_effects_list),
            np.array(agg_ses_list),
            alpha=self.alpha,
            df=df_survey_val,
        )

        event_study_effects = {}
        for idx, (e, _) in enumerate(sorted_periods):
            event_study_effects[e] = {
                "effect": agg_effects_list[idx],
                "se": agg_ses_list[idx],
                "t_stat": float(t_stats[idx]),
                "p_value": float(p_values[idx]),
                "conf_int": (float(ci_lowers[idx]), float(ci_uppers[idx])),
                "n_groups": agg_n_groups[idx],
            }

        # Add reference period for universal base period mode (matches R did package)
        if getattr(self, "base_period", "varying") == "universal":
            ref_period = -1 - self.anticipation
            if event_study_effects and ref_period not in event_study_effects:
                event_study_effects[ref_period] = {
                    "effect": 0.0,
                    "se": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "conf_int": (np.nan, np.nan),
                    "n_groups": 0,
                }

        # Compute full event-study VCV from per-event-time IF vectors (Phase 7d)
        # This enables HonestDiD to use the full covariance structure
        event_study_vcov = None
        valid_psi = [p for p in _psi_vectors if len(p) > 0]
        if valid_psi:
            try:
                Psi = np.column_stack(valid_psi)  # (n_units, n_event_times)
                resolved_survey = (
                    precomputed.get("resolved_survey_unit") if precomputed is not None else None
                )
                if (
                    resolved_survey is not None
                    and not (
                        hasattr(resolved_survey, "uses_replicate_variance")
                        and resolved_survey.uses_replicate_variance
                    )
                    and (
                        resolved_survey.strata is not None
                        or resolved_survey.psu is not None
                        or resolved_survey.fpc is not None
                    )
                ):
                    from diff_diff.survey import _compute_stratified_psu_meat

                    meat, _, _ = _compute_stratified_psu_meat(Psi, resolved_survey)
                    event_study_vcov = meat
                elif (
                    resolved_survey is not None
                    and hasattr(resolved_survey, "uses_replicate_variance")
                    and resolved_survey.uses_replicate_variance
                ):
                    # Replicate-weight: fall back to None (diagonal in HonestDiD)
                    # until multivariate replicate VCV is implemented
                    event_study_vcov = None
                else:
                    # No survey: simple sum-of-outer-products
                    event_study_vcov = Psi.T @ Psi
            except (ValueError, np.linalg.LinAlgError):
                pass  # Fall back to diagonal (None)

        # Store the event-time index that matches VCV columns (for subsetting
        # in HonestDiD when some event times are filtered out)
        self._event_study_vcov_index = _psi_event_times if event_study_vcov is not None else None

        # Attach VCV to self for CallawaySantAnna to pick up
        self._event_study_vcov = event_study_vcov

        return event_study_effects

    def _aggregate_by_group(
        self,
        group_time_effects: Dict,
        influence_func_info: Dict,
        groups: List[Any],
        precomputed: Optional["PrecomputedData"] = None,
        df: Optional[pd.DataFrame] = None,
        unit: Optional[str] = None,
    ) -> Dict[Any, Dict[str, Any]]:
        """
        Aggregate effects by treatment cohort.

        Computes average effect for each cohort across all post-treatment periods.

        Standard errors use influence function aggregation with WIF adjustment
        to account for covariances across time periods within a cohort.
        When a full survey design is present in precomputed, uses design-based
        variance via compute_survey_if_variance().
        """
        # Collect all group aggregation data first
        group_data_list = []
        for g in groups:
            g_effects = [
                ((g, t), data["effect"])
                for (gg, t), data in group_time_effects.items()
                if gg == g and t >= g - self.anticipation
            ]

            if not g_effects:
                continue

            gt_pairs = [x[0] for x in g_effects]
            effs = np.array([x[1] for x in g_effects])

            # Exclude NaN effects from this group's aggregation
            finite_mask = np.isfinite(effs)
            if not np.all(finite_mask):
                effs = effs[finite_mask]
                gt_pairs = [gt for gt, m in zip(gt_pairs, finite_mask) if m]
                if len(effs) == 0:
                    continue

            weights = np.ones(len(effs)) / len(effs)
            agg_effect = np.sum(weights * effs)

            # Use WIF-adjusted SE (with survey design support)
            groups_for_gt = np.array([gg for (gg, t) in gt_pairs])
            agg_se, eff_df = self._compute_aggregated_se_with_wif(
                gt_pairs, weights, effs, groups_for_gt, influence_func_info, df, unit, precomputed
            )
            group_data_list.append((g, agg_effect, agg_se, len(g_effects), eff_df))

        if not group_data_list:
            return {}

        # Batch inference
        agg_effects = np.array([x[1] for x in group_data_list])
        agg_ses = np.array([x[2] for x in group_data_list])
        df_survey_val = precomputed.get("df_survey") if precomputed is not None else None
        # Guard: replicate design with undefined df → NaN inference
        if (
            df_survey_val is None
            and precomputed is not None
            and precomputed.get("resolved_survey_unit") is not None
            and hasattr(precomputed["resolved_survey_unit"], "uses_replicate_variance")
            and precomputed["resolved_survey_unit"].uses_replicate_variance
        ):
            df_survey_val = 0
        # Use minimum per-group effective df if any dropped replicates
        non_none_dfs = [x[4] for x in group_data_list if x[4] is not None]
        if non_none_dfs:
            df_survey_val = min(non_none_dfs)
        t_stats, p_values, ci_lowers, ci_uppers = safe_inference_batch(
            agg_effects,
            agg_ses,
            alpha=self.alpha,
            df=df_survey_val,
        )

        group_effects = {}
        for idx, (g, agg_effect, agg_se, n_periods, _eff_df) in enumerate(group_data_list):
            group_effects[g] = {
                "effect": agg_effect,
                "se": agg_se,
                "t_stat": float(t_stats[idx]),
                "p_value": float(p_values[idx]),
                "conf_int": (float(ci_lowers[idx]), float(ci_uppers[idx])),
                "n_periods": n_periods,
            }

        return group_effects
