"""
Aggregation methods mixin for Callaway-Sant'Anna estimator.

This module provides the mixin class containing methods for aggregating
group-time average treatment effects into summary measures.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from diff_diff.utils import (
    compute_confidence_interval,
    compute_p_value,
)

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

        for (g, t), data in group_time_effects.items():
            # Only include post-treatment effects (t >= g - anticipation)
            # Pre-treatment effects are for parallel trends, not overall ATT
            if t < g - self.anticipation:
                continue
            effects.append(data['effect'])
            weights_list.append(data['n_treated'])
            gt_pairs.append((g, t))
            groups_for_gt.append(g)

        # Guard against empty post-treatment set
        if len(effects) == 0:
            import warnings
            warnings.warn(
                "No post-treatment effects available for overall ATT aggregation. "
                "This can occur when cohorts lack post-treatment periods in the data.",
                UserWarning,
                stacklevel=2
            )
            return np.nan, np.nan

        effects = np.array(effects)
        weights = np.array(weights_list, dtype=float)
        groups_for_gt = np.array(groups_for_gt)

        # Normalize weights
        total_weight = np.sum(weights)
        weights_norm = weights / total_weight

        # Weighted average
        overall_att = np.sum(weights_norm * effects)

        # Compute SE using influence function aggregation with wif adjustment
        overall_se = self._compute_aggregated_se_with_wif(
            gt_pairs, weights_norm, effects, groups_for_gt,
            influence_func_info, df, unit, precomputed
        )

        return overall_att, overall_se

    def _compute_aggregated_se(
        self,
        gt_pairs: List[Tuple[Any, Any]],
        weights: np.ndarray,
        influence_func_info: Dict,
    ) -> float:
        """
        Compute standard error using influence function aggregation.

        This properly accounts for covariances across (g,t) pairs by
        aggregating unit-level influence functions:

            ψ_i(overall) = Σ_{(g,t)} w_(g,t) × ψ_i(g,t)
            Var(overall) = (1/n) Σ_i [ψ_i]²

        This matches R's `did` package analytical SE formula.
        """
        if not influence_func_info:
            # Fallback if no influence functions available
            return 0.0

        # Build unit index mapping from all (g,t) pairs
        all_units = set()
        for (g, t) in gt_pairs:
            if (g, t) in influence_func_info:
                info = influence_func_info[(g, t)]
                all_units.update(info['treated_units'])
                all_units.update(info['control_units'])

        if not all_units:
            return 0.0

        all_units = sorted(all_units)
        n_units = len(all_units)
        unit_to_idx = {u: i for i, u in enumerate(all_units)}

        # Aggregate influence functions across (g,t) pairs
        psi_overall = np.zeros(n_units)

        for j, (g, t) in enumerate(gt_pairs):
            if (g, t) not in influence_func_info:
                continue

            info = influence_func_info[(g, t)]
            w = weights[j]

            # Treated unit contributions
            for i, unit_id in enumerate(info['treated_units']):
                idx = unit_to_idx[unit_id]
                psi_overall[idx] += w * info['treated_inf'][i]

            # Control unit contributions
            for i, unit_id in enumerate(info['control_units']):
                idx = unit_to_idx[unit_id]
                psi_overall[idx] += w * info['control_inf'][i]

        # Compute variance: Var(θ̄) = (1/n) Σᵢ ψᵢ²
        variance = np.sum(psi_overall ** 2)
        return np.sqrt(variance)

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
    ) -> float:
        """
        Compute SE with weight influence function (wif) adjustment.

        This matches R's `did` package approach for "simple" aggregation,
        which accounts for uncertainty in estimating group-size weights.

        The wif adjustment adds variance due to the fact that aggregation
        weights w_g = n_g / N depend on estimated group sizes.

        Formula (matching R's did::aggte):
            agg_inf_i = Σ_k w_k × inf_i_k + wif_i × ATT_k
            se = sqrt(mean(agg_inf^2) / n)

        where:
        - k indexes "keepers" (post-treatment (g,t) pairs)
        - w_k = pg[k] / sum(pg[keepers]) where pg = n_g / n_all
        - wif captures how unit i influences the weight estimation
        """
        if not influence_func_info:
            return 0.0

        # Build unit index mapping
        all_units_set: Set[Any] = set()
        for (g, t) in gt_pairs:
            if (g, t) in influence_func_info:
                info = influence_func_info[(g, t)]
                all_units_set.update(info['treated_units'])
                all_units_set.update(info['control_units'])

        if not all_units_set:
            return 0.0

        all_units = sorted(all_units_set)
        n_units = len(all_units)
        unit_to_idx = {u: i for i, u in enumerate(all_units)}

        # Get unique groups and their information
        unique_groups = sorted(set(groups_for_gt))
        unique_groups_set = set(unique_groups)
        group_to_idx = {g: i for i, g in enumerate(unique_groups)}

        # Compute group-level probabilities matching R's formula:
        # pg[g] = n_g / n_all (fraction of ALL units in group g)
        # This differs from our old formula which used n_g / total_treated
        group_sizes = {}
        for g in unique_groups:
            treated_in_g = df[df['first_treat'] == g][unit].nunique()
            group_sizes[g] = treated_in_g

        # pg indexed by group
        pg_by_group = np.array([group_sizes[g] / n_units for g in unique_groups])

        # pg indexed by keeper (each (g,t) pair gets its group's pg)
        # This matches R's: pg <- pgg[match(group, originalglist)]
        pg_keepers = np.array([pg_by_group[group_to_idx[g]] for g in groups_for_gt])
        sum_pg_keepers = np.sum(pg_keepers)

        # Guard against zero weights (no keepers = no variance)
        if sum_pg_keepers == 0:
            return 0.0

        # Standard aggregated influence (without wif)
        psi_standard = np.zeros(n_units)

        for j, (g, t) in enumerate(gt_pairs):
            if (g, t) not in influence_func_info:
                continue

            info = influence_func_info[(g, t)]
            w = weights[j]

            # Vectorized influence function aggregation for treated units
            treated_indices = np.array([unit_to_idx[uid] for uid in info['treated_units']])
            if len(treated_indices) > 0:
                np.add.at(psi_standard, treated_indices, w * info['treated_inf'])

            # Vectorized influence function aggregation for control units
            control_indices = np.array([unit_to_idx[uid] for uid in info['control_units']])
            if len(control_indices) > 0:
                np.add.at(psi_standard, control_indices, w * info['control_inf'])

        # Build unit-group array using precomputed data if available
        # This is O(n_units) instead of O(n_units × n_obs) DataFrame lookups
        if precomputed is not None:
            # Use precomputed cohort mapping
            precomputed_units = precomputed['all_units']
            precomputed_cohorts = precomputed['unit_cohorts']
            precomputed_unit_to_idx = precomputed['unit_to_idx']

            # Build unit_groups_array for the units in this SE computation
            # A value of -1 indicates never-treated or other (not in unique_groups)
            unit_groups_array = np.full(n_units, -1, dtype=np.float64)
            for i, uid in enumerate(all_units):
                if uid in precomputed_unit_to_idx:
                    cohort = precomputed_cohorts[precomputed_unit_to_idx[uid]]
                    if cohort in unique_groups_set:
                        unit_groups_array[i] = cohort
        else:
            # Fallback: build from DataFrame (slow path for backward compatibility)
            unit_groups_array = np.full(n_units, -1, dtype=np.float64)
            for i, uid in enumerate(all_units):
                unit_first_treat = df[df[unit] == uid]['first_treat'].iloc[0]
                if unit_first_treat in unique_groups_set:
                    unit_groups_array[i] = unit_first_treat

        # Vectorized WIF computation
        # R's wif formula:
        #   if1[i,k] = (indicator(G_i == group_k) - pg[k]) / sum(pg[keepers])
        #   if2[i,k] = indicator_sum[i] * pg[k] / sum(pg[keepers])^2
        #   wif[i,k] = if1[i,k] - if2[i,k]
        #   wif_contrib[i] = sum_k(wif[i,k] * att[k])

        # Build indicator matrix: (n_units, n_keepers)
        # indicator_matrix[i, k] = 1.0 if unit i belongs to group for keeper k
        groups_for_gt_array = np.array(groups_for_gt)
        indicator_matrix = (unit_groups_array[:, np.newaxis] == groups_for_gt_array[np.newaxis, :]).astype(np.float64)

        # Vectorized indicator_sum: sum over keepers
        # indicator_sum[i] = sum_k(indicator(G_i == group_k) - pg[k])
        indicator_sum = np.sum(indicator_matrix - pg_keepers, axis=1)

        # Vectorized wif matrix computation
        # Suppress RuntimeWarnings for edge cases (small samples, extreme weights)
        # in division operations and matrix multiplication
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            # if1_matrix[i,k] = (indicator[i,k] - pg[k]) / sum_pg
            if1_matrix = (indicator_matrix - pg_keepers) / sum_pg_keepers
            # if2_matrix[i,k] = indicator_sum[i] * pg[k] / sum_pg^2
            if2_matrix = np.outer(indicator_sum, pg_keepers) / (sum_pg_keepers ** 2)
            wif_matrix = if1_matrix - if2_matrix

            # Single matrix-vector multiply for all contributions
            # wif_contrib[i] = sum_k(wif[i,k] * att[k])
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
                stacklevel=2
            )
            return np.nan  # Signal invalid inference instead of biased SE

        # Scale by 1/n_units to match R's getSE formula: sqrt(mean(IF^2)/n)
        psi_wif = wif_contrib / n_units

        # Combine standard and wif terms
        psi_total = psi_standard + psi_wif

        # Compute variance and SE
        # R's formula: sqrt(mean(IF^2) / n) = sqrt(sum(IF^2) / n^2)
        variance = np.sum(psi_total ** 2)
        return np.sqrt(variance)

    def _aggregate_event_study(
        self,
        group_time_effects: Dict,
        influence_func_info: Dict,
        groups: List[Any],
        time_periods: List[Any],
        balance_e: Optional[int] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Aggregate effects by relative time (event study).

        Computes average effect at each event time e = t - g.

        Standard errors use influence function aggregation to account for
        covariances across (g,t) pairs.
        """
        # Organize effects by relative time, keeping track of (g,t) pairs
        effects_by_e: Dict[int, List[Tuple[Tuple[Any, Any], float, int]]] = {}

        for (g, t), data in group_time_effects.items():
            e = t - g  # Relative time
            if e not in effects_by_e:
                effects_by_e[e] = []
            effects_by_e[e].append((
                (g, t),  # Keep track of the (g,t) pair
                data['effect'],
                data['n_treated']
            ))

        # Balance the panel if requested
        if balance_e is not None:
            # Keep only groups that have effects at relative time balance_e
            groups_at_e = set()
            for (g, t), data in group_time_effects.items():
                if t - g == balance_e:
                    groups_at_e.add(g)

            # Filter effects to only include balanced groups
            balanced_effects: Dict[int, List[Tuple[Tuple[Any, Any], float, int]]] = {}
            for (g, t), data in group_time_effects.items():
                if g in groups_at_e:
                    e = t - g
                    if e not in balanced_effects:
                        balanced_effects[e] = []
                    balanced_effects[e].append((
                        (g, t),
                        data['effect'],
                        data['n_treated']
                    ))
            effects_by_e = balanced_effects

        # Compute aggregated effects
        event_study_effects = {}

        for e, effect_list in sorted(effects_by_e.items()):
            gt_pairs = [x[0] for x in effect_list]
            effs = np.array([x[1] for x in effect_list])
            ns = np.array([x[2] for x in effect_list], dtype=float)

            # Weight by group size
            weights = ns / np.sum(ns)

            agg_effect = np.sum(weights * effs)

            # Compute SE using influence function aggregation
            agg_se = self._compute_aggregated_se(
                gt_pairs, weights, influence_func_info
            )

            t_stat = agg_effect / agg_se if np.isfinite(agg_se) and agg_se > 0 else np.nan
            p_val = compute_p_value(t_stat)
            ci = compute_confidence_interval(agg_effect, agg_se, self.alpha)

            event_study_effects[e] = {
                'effect': agg_effect,
                'se': agg_se,
                't_stat': t_stat,
                'p_value': p_val,
                'conf_int': ci,
                'n_groups': len(effect_list),
            }

        # Add reference period for universal base period mode (matches R did package)
        # The reference period e = -1 - anticipation has effect = 0 by construction
        # Only add if there are actual computed effects (guard against empty data)
        if getattr(self, 'base_period', 'varying') == "universal":
            ref_period = -1 - self.anticipation
            # Only inject reference if we have at least one real effect
            if event_study_effects and ref_period not in event_study_effects:
                event_study_effects[ref_period] = {
                    'effect': 0.0,
                    'se': np.nan,  # Undefined - no data, normalization constraint
                    't_stat': np.nan,  # Undefined - normalization constraint
                    'p_value': np.nan,
                    'conf_int': (np.nan, np.nan),  # NaN propagation for undefined inference
                    'n_groups': 0,  # No groups contribute - fixed by construction
                }

        return event_study_effects

    def _aggregate_by_group(
        self,
        group_time_effects: Dict,
        influence_func_info: Dict,
        groups: List[Any],
    ) -> Dict[Any, Dict[str, Any]]:
        """
        Aggregate effects by treatment cohort.

        Computes average effect for each cohort across all post-treatment periods.

        Standard errors use influence function aggregation to account for
        covariances across time periods within a cohort.
        """
        group_effects = {}

        for g in groups:
            # Get all effects for this group (post-treatment only: t >= g - anticipation)
            # Keep track of (g, t) pairs for influence function aggregation
            g_effects = [
                ((g, t), data['effect'])
                for (gg, t), data in group_time_effects.items()
                if gg == g and t >= g - self.anticipation
            ]

            if not g_effects:
                continue

            gt_pairs = [x[0] for x in g_effects]
            effs = np.array([x[1] for x in g_effects])

            # Equal weight across time periods for a group
            weights = np.ones(len(effs)) / len(effs)

            agg_effect = np.sum(weights * effs)

            # Compute SE using influence function aggregation
            agg_se = self._compute_aggregated_se(
                gt_pairs, weights, influence_func_info
            )

            t_stat = agg_effect / agg_se if np.isfinite(agg_se) and agg_se > 0 else np.nan
            p_val = compute_p_value(t_stat)
            ci = compute_confidence_interval(agg_effect, agg_se, self.alpha)

            group_effects[g] = {
                'effect': agg_effect,
                'se': agg_se,
                't_stat': t_stat,
                'p_value': p_val,
                'conf_int': ci,
                'n_periods': len(g_effects),
            }

        return group_effects
