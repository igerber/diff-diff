"""
Aggregation methods mixin for Callaway-Sant'Anna estimator.

This module provides the mixin class containing methods for aggregating
group-time average treatment effects into summary measures.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union, overload

import numpy as np
import pandas as pd

from diff_diff.utils import safe_inference_batch

# Type alias for pre-computed structures (defined at module scope for runtime access)
PrecomputedData = Dict[str, Any]


@dataclass
class EventStudyAggregation:
    """Everything one event-study aggregation produces.

    ``_aggregate_event_study`` previously returned only ``effects`` and stashed
    the other four values on ``self`` as side channels. Returning them makes
    the aggregator PURE, which is what lets it run post-fit from a retained kit
    without mutating the results object it was called from (spec section 6).

    Attributes
    ----------
    effects : dict
        Per-event-time effect records, keyed by event time.
    overall : dict or None
        Eq. (4.14) overall ATT (``att`` / ``se`` / ``effective_df``) - the
        unweighted mean of post-treatment ES(e). Consumed by
        ``StaggeredTripleDifference`` as ``overall_att_es``; CallawaySantAnna
        leaves it unread. ``None`` when no post-treatment horizon qualifies.
    df_used : float or None
        The ONE df every ES row's inference actually used, recorded iff it
        governs a t-reference (finite and > 0). Provenance for
        ``CallawaySantAnnaResults.event_study_df``.
    vcov : np.ndarray or None
        Full event-study covariance, when computable.
    vcov_index : list or None
        Event times aligned 1:1 with ``vcov``'s columns.
    """

    effects: Dict[Any, Dict[str, Any]] = field(default_factory=dict)
    overall: Optional[Dict[str, Any]] = None
    df_used: Optional[float] = None
    vcov: Optional[np.ndarray] = None
    vcov_index: Optional[List[Any]] = None
    #: Distinct base EVENT TIMES of the cohorts RETAINED by this
    #: aggregation (derived from the materialized universal-base
    #: is_reference cells; None when there are none, e.g. varying base).
    #: Surface-faithful, unlike the fit-level fit-wide tuple: balance_e
    #: can drop the cohort responsible for a second base, and the
    #: container's common-reference guard must reflect the cohorts that
    #: actually entered the reported estimand.
    reference_event_times: Optional[Tuple[Any, ...]] = None


def fixed_cohort_agg_weights(
    precomputed: Optional["PrecomputedData"],
) -> Optional[Dict[Any, float]]:
    """Fixed per-cohort aggregation masses (R's ``pg = n_g / N`` numerator) for
    the treated cohorts ``g > 0``, or ``None`` when the caller should fall back
    to per-cell weights (``agg_weight`` / ``n_treated``).

    Priority: unit-level ``agg_cohort_masses`` (RC-on-panel and true RC, exposed
    by ``_precompute_structures_rc``) → per-observation survey cohort mass
    (survey designs) → ``None`` (panel non-survey). Preferring
    ``agg_cohort_masses`` over the raw ``survey_weights`` sum is what makes an
    unbalanced panel routed as RC (``allow_unbalanced_panel=True``, which
    synthesizes ``SurveyDesign(psu=unit)``) weight every aggregation — simple,
    event-study, group, AND the multiplier bootstrap — by fixed UNIT cohort
    mass rather than observation count. Single source of truth so the analytical
    and bootstrap paths cannot diverge.
    """
    if precomputed is None:
        return None
    agg_masses = precomputed.get("agg_cohort_masses")
    if agg_masses is not None:
        return {g: m for g, m in agg_masses.items() if g > 0}
    sw = precomputed.get("survey_weights")
    if sw is not None:
        unit_cohorts = precomputed["unit_cohorts"]
        return {g: float(np.sum(sw[unit_cohorts == g])) for g in np.unique(unit_cohorts) if g > 0}
    return None


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
        df: Optional[pd.DataFrame],
        unit: Optional[str],
        precomputed: Optional["PrecomputedData"] = None,
    ) -> Tuple[float, float, Optional[int]]:
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
        effects_list: List[Any] = []
        weights_list = []
        gt_pairs = []
        groups_list: List[Any] = []

        # Fixed per-cohort aggregation weights (R's did::aggte pg = n_g / N),
        # preferring the unit-level RC mass so allow_unbalanced_panel weights the
        # overall ATT by fixed UNIT cohort mass, not observation count.
        survey_cohort_weights = fixed_cohort_agg_weights(precomputed)

        for (g, t), data in group_time_effects.items():
            # Only include post-treatment effects (t >= g - anticipation)
            # Pre-treatment effects are for parallel trends, not overall ATT
            if t < g - self.anticipation:
                continue
            effects_list.append(data["effect"])
            # Use fixed cohort-level survey weight sum for aggregation.
            # For RCS, data["agg_weight"] holds the fixed cohort mass;
            # for panel, fallback to data["n_treated"].
            if survey_cohort_weights is not None and g in survey_cohort_weights:
                weights_list.append(survey_cohort_weights[g])
            else:
                weights_list.append(data.get("agg_weight", data["n_treated"]))
            gt_pairs.append((g, t))
            groups_list.append(g)

        # Guard against empty post-treatment set
        if len(effects_list) == 0:
            import warnings

            warnings.warn(
                "No post-treatment effects available for overall ATT aggregation. "
                "This can occur when cohorts lack post-treatment periods in the data.",
                UserWarning,
                stacklevel=2,
            )
            return np.nan, np.nan, None

        effects = np.array(effects_list)
        weights = np.array(weights_list, dtype=float)
        groups_for_gt = np.array(groups_list)

        # Exclude NaN effects from aggregation (R's aggte() convention).
        # No warning here — fit() emits a consolidated skip warning covering
        # all estimation paths (vectorized, covariate, general, RC).
        finite_mask = np.isfinite(effects)
        if not np.all(finite_mask):
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

    @staticmethod
    def _get_agg_cache(precomputed: "PrecomputedData") -> Dict[str, Any]:
        """
        Per-fit cohort tables for the combined-IF fast path, lazily memoized
        on the precomputed dict.

        The cache is validated by ARRAY IDENTITY, not dict residency:
        StaggeredTripleDifference aggregates through a shallow copy of
        precomputed with a replaced (eligibility-zeroed) ``unit_cohorts``,
        so a cache keyed to the dict could serve stale tables across the
        copy. ``cohorts_ref``/``sw_ref`` pin the exact arrays the tables
        were built from; any mismatch rebuilds into a FRESH dict (never
        mutated in place - the shallow copy shares the cache reference).
        """
        unit_cohorts = precomputed["unit_cohorts"]
        survey_w = precomputed.get("survey_weights")
        cache = precomputed.get("_agg_cache")
        if (
            cache is not None
            and cache["cohorts_ref"] is unit_cohorts
            and cache["sw_ref"] is survey_w
        ):
            return cache

        cohort_values, cohort_codes = np.unique(unit_cohorts, return_inverse=True)
        agg_masses = precomputed.get("agg_cohort_masses")
        if agg_masses is not None:
            # RC path: pg basis is per-UNIT cohort mass (R's pg = n_g / N over
            # units), exposed by _precompute_structures_rc. `cohort_codes` stays
            # per-observation (the WIF scatter is per-obs, divided by
            # obs_per_unit downstream). No-op for a true RC (per-unit ==
            # per-obs); the fix for an unbalanced panel routed as RC.
            cohort_masses = np.array(
                [float(agg_masses.get(float(cv), 0.0)) for cv in cohort_values],
                dtype=np.float64,
            )
            total_weight = float(precomputed.get("agg_total_weight", float(np.sum(cohort_masses))))
        elif survey_w is not None:
            # Survey-weighted cohort masses. np.bincount accumulation order
            # differs from the historical per-group mask-sums at the ~1 ULP
            # level (documented drift budget; REGISTRY CallawaySantAnna SE
            # notes).
            cohort_masses = np.bincount(
                cohort_codes, weights=survey_w, minlength=len(cohort_values)
            )
            total_weight = float(np.sum(survey_w))
        else:
            cohort_masses = np.bincount(cohort_codes, minlength=len(cohort_values)).astype(
                np.float64
            )
            total_weight = float(len(unit_cohorts))

        cache = {
            "cohorts_ref": unit_cohorts,
            "sw_ref": survey_w,
            "cohort_values": cohort_values,
            "cohort_codes": cohort_codes,
            "cohort_masses": cohort_masses,
            "total_weight": total_weight,
            # Per-obs unit multiplicity for the WIF over-count correction (all
            # 1.0 on panel / true RC → the division below is a no-op there).
            "obs_per_unit": precomputed.get("obs_per_unit"),
        }
        precomputed["_agg_cache"] = cache
        return cache

    def _combined_if_fast(
        self,
        gt_pairs: List[Tuple[Any, Any]],
        weights: np.ndarray,
        effects: np.ndarray,
        groups_for_gt: np.ndarray,
        influence_func_info: Dict,
        precomputed: "PrecomputedData",
        n_units: int,
    ) -> Optional[Tuple[np.ndarray, None]]:
        """
        O(n_units) combined-IF assembly over per-fit cohort tables.

        Replaces the general path's per-group full-DataFrame scans, per-unit
        Python loops, and dense (n_units x n_gt) WIF matrices with cohort-
        indexed lookups. The WIF uses the closed form (algebraically
        identical to the dense ``wif_matrix @ effects``, floating-point
        accumulation order differs - not bit-for-bit):

            wif_i = w_i * (E(c_i)/S - K(c_i) * d / S**2)

        where c_i is unit i's cohort, E(c) sums ``effects`` over keeper
        (g,t) pairs with g == c, K(c) counts them, S = sum of keeper pg,
        d = pg_keepers @ effects, and w_i is the survey weight (1 when
        unweighted). Units whose cohort is not among the keepers get
        exactly 0 (the old dense form realizes the same value through
        cancelling terms).

        Returns None when the cohort lookup cannot be resolved exactly
        (non-numeric cohort dtypes, or a keeper group missing from the
        cohort table) - the caller then falls back to the general path.
        """
        cache = self._get_agg_cache(precomputed)
        cohort_values = cache["cohort_values"]
        cohort_codes = cache["cohort_codes"]
        cohort_masses = cache["cohort_masses"]
        total_weight = cache["total_weight"]
        survey_w = cache["sw_ref"]

        groups_arr = np.asarray(groups_for_gt)
        if not (
            np.issubdtype(groups_arr.dtype, np.number)
            and np.issubdtype(cohort_values.dtype, np.number)
        ):
            return None

        # Unique keeper groups + exact positions in the cohort table.
        unique_groups = np.unique(groups_arr)
        pos = np.searchsorted(cohort_values, unique_groups)
        if np.any(pos >= len(cohort_values)) or np.any(
            cohort_values[np.minimum(pos, len(cohort_values) - 1)] != unique_groups
        ):
            return None  # keeper group absent from cohort table

        # pg per keeper (same values as the general path's group_sizes /
        # total_weight; survey masses differ only in accumulation order).
        pg_by_group = cohort_masses[pos] / total_weight
        kpos = np.searchsorted(unique_groups, groups_arr)
        pg_keepers = pg_by_group[kpos]
        sum_pg_keepers = np.sum(pg_keepers)

        # Guard against zero weights (no keepers = no variance). Must stay
        # BEFORE the psi_standard scatter - the general path returns zeros
        # without ever accumulating the standard IF.
        if sum_pg_keepers == 0:
            return np.zeros(n_units), None

        # Standard aggregated influence (without wif). Index arrays are
        # unique within each cell by construction at every producer
        # (np.where on disjoint masks), so fancy += is exact.
        psi_standard = np.zeros(n_units)
        for j, (g, t) in enumerate(gt_pairs):
            if (g, t) not in influence_func_info:
                continue
            info = influence_func_info[(g, t)]
            w = weights[j]
            treated_idx = info["treated_idx"]
            if len(treated_idx) > 0:
                psi_standard[treated_idx] += w * info["treated_inf"]
            control_idx = info["control_idx"]
            if len(control_idx) > 0:
                psi_standard[control_idx] += w * info["control_inf"]

        # Closed-form WIF over per-cohort tables.
        n_ug = len(unique_groups)
        E_keepers = np.bincount(kpos, weights=effects, minlength=n_ug)
        K_keepers = np.bincount(kpos, minlength=n_ug).astype(np.float64)
        E_full = np.zeros(len(cohort_values))
        K_full = np.zeros(len(cohort_values))
        E_full[pos] = E_keepers
        K_full[pos] = K_keepers

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            d = pg_keepers @ effects
            wif_contrib = E_full[cohort_codes] / sum_pg_keepers - K_full[cohort_codes] * (
                d / sum_pg_keepers**2
            )
            if survey_w is not None:
                wif_contrib = wif_contrib * survey_w

        # Check for non-finite values from edge cases (same fail-closed
        # contract as the general path: warn + all-NaN vector, before the
        # 1/total_weight scaling).
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
            return np.full(n_units, np.nan), None

        # Scale by 1/total_weight to match R's getSE formula. On a panel routed
        # as RC, additionally divide by obs_per_unit: the WIF is a per-UNIT
        # quantity but wif_contrib is per-observation, so the unit-clustered sum
        # would otherwise over-count each unit's WIF by its observation count.
        # obs_per_unit is 1.0 for panel / true RC (a no-op there).
        obs_per_unit = cache.get("obs_per_unit")
        if obs_per_unit is not None:
            psi_wif = wif_contrib / (obs_per_unit * total_weight)
        else:
            psi_wif = wif_contrib / total_weight

        return psi_standard + psi_wif, None

    def _compute_combined_influence_function(
        self,
        gt_pairs: List[Tuple[Any, Any]],
        weights: np.ndarray,
        effects: np.ndarray,
        groups_for_gt: np.ndarray,
        influence_func_info: Dict,
        df: Optional[pd.DataFrame],
        unit: Optional[str],
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

        # Fast-path dispatch: all in-package callers thread the SAME
        # precomputed structures they index psi by, so the cohort tables can
        # be looked up in O(n_units) instead of re-scanning the DataFrame per
        # group and looping units in Python. Order matters: the RCS check
        # must precede the panel identity check (for RCS both
        # global_unit_to_idx and precomputed["unit_to_idx"] are None, so the
        # identity guard alone would spuriously pass). Anything not exactly
        # matched (direct callers with foreign index maps, size mismatches,
        # non-numeric cohorts) falls through to the general path below
        # (same mathematical contract; its IF scatter uses the fancy-+=
        # form, bit-identical to np.add.at on the producers' duplicate-free
        # index arrays).
        if precomputed is not None and n_global_units is not None:
            _fast_ok = False
            if _is_rcs:
                _fast_ok = n_global_units == len(precomputed["unit_cohorts"])
            elif global_unit_to_idx is not None and global_unit_to_idx is precomputed.get(
                "unit_to_idx"
            ):
                _fast_ok = n_global_units == len(precomputed["unit_cohorts"])
            if _fast_ok:
                fast = self._combined_if_fast(
                    gt_pairs,
                    weights,
                    effects,
                    groups_for_gt,
                    influence_func_info,
                    precomputed,
                    n_global_units,
                )
                if fast is not None:
                    return fast

        # Build unit index mapping (local or global)
        if _is_rcs and n_global_units is not None:
            # RCS: direct indexing — obs indices are the array positions
            n_units = n_global_units
            all_units = None
        elif global_unit_to_idx is not None and n_global_units is not None:
            n_units = n_global_units
            all_units = None  # caller already has the unit list
        else:
            # Local-units fallback for direct callers without precomputed /
            # global unit ids. It needs per-cell unit-LABEL arrays, which
            # in-package fits stopped materializing (v3.8 per-cell
            # allocation shave — labels are always all_units[idx], and every
            # in-package caller threads global_unit_to_idx + n_global_units,
            # so this branch is unreachable from a fit). Direct callers must
            # either thread the global ids or supply label arrays.
            all_units_set: Set[Any] = set()
            for g, t in gt_pairs:
                if (g, t) in influence_func_info:
                    info = influence_func_info[(g, t)]
                    t_units = info.get("treated_units")
                    c_units = info.get("control_units")
                    if t_units is None or c_units is None:
                        raise ValueError(
                            "Combined-IF assembly without precomputed/global unit "
                            "ids requires per-cell 'treated_units'/'control_units' "
                            "label arrays in influence_func_info; in-package fits "
                            "no longer materialize them. Pass global_unit_to_idx "
                            "and n_global_units (as all in-package callers do), "
                            "or add the label arrays to your influence_func_info."
                        )
                    all_units_set.update(t_units)
                    all_units_set.update(c_units)

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
            # Survey weights come from precomputed, so it is present here.
            assert precomputed is not None
            # Survey-weighted group sizes
            precomputed_cohorts = precomputed["unit_cohorts"]
            for g in unique_groups:
                mask_g = precomputed_cohorts == g
                group_sizes[g] = float(np.sum(survey_w[mask_g]))
            total_weight = float(np.sum(survey_w))
        elif _is_rcs:
            # The RCS path always builds precomputed (obs-level bookkeeping).
            assert precomputed is not None
            # RCS without survey: count observations per cohort
            precomputed_cohorts = precomputed["unit_cohorts"]
            for g in unique_groups:
                group_sizes[g] = int(np.sum(precomputed_cohorts == g))
            total_weight = float(n_units)
        elif precomputed is not None:
            # Panel without survey. ``unit_cohorts`` is the per-unit cohort array
            # (``df.groupby(unit)[first_treat].first().values``), so counting its
            # matches is IDENTICAL to the frame lookup below - and it lets
            # post-fit aggregation run from the retained kit with no frame.
            precomputed_cohorts = precomputed["unit_cohorts"]
            for g in unique_groups:
                group_sizes[g] = int(np.sum(precomputed_cohorts == g))
            total_weight = float(n_units)
        else:
            # No precomputed bookkeeping (direct internal callers only): fall
            # back to the fit-time frame. Reaching here without one is the
            # fail-closed case - post-fit callers always carry the kit, so a
            # None frame here means neither source is available.
            if df is None or unit is None:
                raise ValueError(
                    "Cohort sizes need either precomputed bookkeeping or the fit-time "
                    "frame; neither was supplied."
                )
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

            # Vectorized IF aggregation using precomputed index arrays. Index
            # arrays are unique within each cell by construction at every
            # producer (np.where on disjoint masks), so fancy += is exact —
            # same scatter contract as _combined_if_fast.
            treated_idx = info["treated_idx"]
            if len(treated_idx) > 0:
                psi_standard[treated_idx] += w * info["treated_inf"]

            control_idx = info["control_idx"]
            if len(control_idx) > 0:
                psi_standard[control_idx] += w * info["control_inf"]

        # Build unit-group array: normalize iterator to (idx, uid) pairs
        unit_groups_array = np.full(n_units, -1, dtype=np.float64)

        if _is_rcs:
            # The RCS path always builds precomputed (obs-level bookkeeping).
            assert precomputed is not None
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
                if df is None or unit is None:
                    raise ValueError(
                        "Per-unit cohorts need either precomputed bookkeeping or the "
                        "fit-time frame; neither was supplied."
                    )
                for idx, uid in idx_uid_pairs:
                    unit_first_treat = df[df[unit] == uid]["first_treat"].iloc[0]
                    if unit_first_treat in unique_groups_set:
                        unit_groups_array[idx] = unit_first_treat
        else:
            idx_uid_pairs = list(enumerate(all_units))
            if precomputed is not None and precomputed.get("unit_to_idx") is not None:
                # Same per-unit cohort lookup as the branch above, from the kit
                # rather than the frame - keeps post-fit aggregation frame-free.
                precomputed_cohorts = precomputed["unit_cohorts"]
                precomputed_unit_to_idx = precomputed["unit_to_idx"]
                for idx, uid in idx_uid_pairs:
                    if uid in precomputed_unit_to_idx:
                        cohort = precomputed_cohorts[precomputed_unit_to_idx[uid]]
                        if cohort in unique_groups_set:
                            unit_groups_array[idx] = cohort
            else:
                if df is None or unit is None:
                    raise ValueError(
                        "Per-unit cohorts need either precomputed bookkeeping or the "
                        "fit-time frame; neither was supplied."
                    )
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

    @overload
    def _compute_aggregated_se_with_wif(
        self,
        gt_pairs: List[Tuple[Any, Any]],
        weights: np.ndarray,
        effects: np.ndarray,
        groups_for_gt: np.ndarray,
        influence_func_info: Dict,
        df: Optional[pd.DataFrame],
        unit: Optional[str],
        precomputed: Optional["PrecomputedData"] = None,
        return_psi: Literal[False] = False,
    ) -> Tuple[float, Optional[int]]: ...

    @overload
    def _compute_aggregated_se_with_wif(
        self,
        gt_pairs: List[Tuple[Any, Any]],
        weights: np.ndarray,
        effects: np.ndarray,
        groups_for_gt: np.ndarray,
        influence_func_info: Dict,
        df: Optional[pd.DataFrame],
        unit: Optional[str],
        precomputed: Optional["PrecomputedData"] = None,
        *,
        return_psi: Literal[True],
    ) -> Tuple[float, np.ndarray, Optional[int]]: ...

    def _compute_aggregated_se_with_wif(
        self,
        gt_pairs: List[Tuple[Any, Any]],
        weights: np.ndarray,
        effects: np.ndarray,
        groups_for_gt: np.ndarray,
        influence_func_info: Dict,
        df: Optional[pd.DataFrame],
        unit: Optional[str],
        precomputed: Optional["PrecomputedData"] = None,
        return_psi: bool = False,
    ) -> "Union[Tuple[float, Optional[int]], Tuple[float, np.ndarray, Optional[int]]]":
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

        Returns
        -------
        ``(se, effective_df)`` when ``return_psi=False``; ``(se, psi_total,
        effective_df)`` when ``return_psi=True``. This 2-tuple / 3-tuple arity is
        held on EVERY branch — including the empty-IF (``se=0.0``) and non-finite-IF
        (``se=NaN``) early returns — so callers that unpack two or three values fail
        soft instead of raising on degenerate influence functions. ``effective_df``
        is non-None only for replicate designs that dropped replicates.
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

        # Consistent return arity across ALL branches: return_psi=True -> 3-tuple
        # (se, psi, effective_df); return_psi=False -> 2-tuple (se, effective_df).
        # The empty / non-finite-IF branches must match so callers that unpack three
        # values (``_aggregate_event_study``) or two (``_aggregate_simple``) fail soft
        # (NaN SE) instead of raising on degenerate IF/WIF edge cases.
        if len(psi_total) == 0:
            return (0.0, psi_total, None) if return_psi else (0.0, None)

        # Check for NaN propagation from non-finite WIF
        if not np.all(np.isfinite(psi_total)):
            return (np.nan, psi_total, None) if return_psi else (np.nan, None)

        se, effective_df = self._se_from_psi(psi_total, precomputed)
        if return_psi:
            return (se, psi_total, effective_df)
        return (se, effective_df)

    def _se_from_psi(
        self,
        psi_total: np.ndarray,
        precomputed: Optional["PrecomputedData"] = None,
    ) -> "Tuple[float, Optional[int]]":
        """Standard error (and per-statistic effective df) from a combined IF vector.

        Routes a finite, non-empty influence-function vector through the same
        variance estimator the per-event-time and simple-aggregation SE paths use:
        replicate-weight variance, full survey-design variance, or the simple
        ``sqrt(sum(psi^2))``. Callers must guard emptiness/finiteness first.
        Returns ``(se, effective_df)``; ``effective_df`` is non-None only for
        replicate designs that dropped replicates.
        """
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
            return se, effective_df

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
            return se, None

        variance = np.sum(psi_total**2)
        return np.sqrt(variance), None

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
    ) -> EventStudyAggregation:
        """
        Aggregate effects by relative time (event study).

        Computes average effect at each event time e = t - g.

        Standard errors include the weight influence function (WIF)
        adjustment that accounts for uncertainty in group-size weights,
        matching R's did::aggte(..., type="dynamic").
        """
        # Organize effects by relative time, keeping track of (g,t) pairs
        effects_by_e: Dict[int, List[Tuple[Tuple[Any, Any], float, float]]] = {}

        # Fixed per-cohort aggregation weights (shared with _aggregate_simple and
        # the bootstrap): unit-level RC mass preferred so allow_unbalanced_panel
        # weights each multi-cell horizon by fixed UNIT cohort mass, not obs count.
        survey_cohort_weights = fixed_cohort_agg_weights(precomputed)

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

        # Common-reference provenance for THIS aggregation's surface: the
        # distinct base event times of the RETAINED cohorts, read off the
        # materialized universal-base is_reference cells (varying-base
        # fits have none -> None). Surface-faithful by construction:
        # balance_e can drop the cohort responsible for a second base, in
        # which case the fit-level fit-wide tuple would over-restrict the
        # balanced container.
        _retained_cohorts = {g for cells in effects_by_e.values() for (g, _t), _eff, _w in cells}
        _ref_es = {
            t - g
            for (g, t), data in group_time_effects.items()
            if data.get("is_reference") and g in _retained_cohorts
        }
        es_reference_event_times: Optional[Tuple[Any, ...]] = (
            tuple(sorted(_ref_es)) if _ref_es else None
        )

        # Universal base period: each cohort's positional base is materialized in
        # `group_time_effects` / `influence_func_info` (with a zero effect and a
        # zero influence function) by `fit()` before aggregation, so it is already
        # grouped into `effects_by_e` above and weighted into the dynamic horizon
        # exactly like R `did::aggte(type="dynamic")` (a reference cell dilutes the
        # real cells at an overlapping negative horizon). We only flag which cells
        # are references so a reference-only horizon reports NaN (not a spurious
        # se=0) and does not count toward `n_groups`.
        reference_cells: Set[Tuple[Any, Any]] = {
            (g, t) for (g, t), data in group_time_effects.items() if data.get("is_reference")
        }

        # Compute aggregated effects and SEs for all relative periods
        sorted_periods = sorted(effects_by_e.items())
        agg_effects_list = []
        agg_ses_list = []
        agg_n_groups = []
        agg_effective_dfs = []  # Per-horizon effective df (replicate designs)
        agg_periods = []  # Relative times that yielded an estimable aggregate row
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
                    # Every cell in this relative-time bucket is non-estimable
                    # (materialized NaN). Omit the bucket entirely so the
                    # event-study surface matches the prior omit behavior and R
                    # did::aggte() (a relative time with no estimable cell yields
                    # no row), and stays consistent with _aggregate_by_group,
                    # which already drops all-NaN groups.
                    continue

            # Reference-only horizon (universal base): every cell is a zero
            # reference (att=0, no influence function), so there is no estimated
            # effect. Report att=0, se=NaN — matching R `did` (base rows carry
            # `se = NA`) — instead of a spurious se=0 from the all-zero IF.
            if reference_cells and all(gt in reference_cells for gt in gt_pairs):
                agg_effects_list.append(0.0)
                agg_ses_list.append(np.nan)
                agg_n_groups.append(0)
                agg_effective_dfs.append(None)
                agg_periods.append(e)
                # No influence-function column for a reference-only horizon (it
                # carries no estimated effect); leaving it out of _psi_event_times
                # keeps the VCV index aligned with the VCV columns (valid_psi).
                continue

            weights = ns / np.sum(ns)
            agg_effect = np.sum(weights * effs)

            # Compute SE with WIF adjustment (matching R's did::aggte). Zero-IF
            # reference cells contribute nothing to the variance but their cohort
            # weight dilutes the real cells, matching R's dynamic aggregation.
            groups_for_gt = np.array([g for (g, t) in gt_pairs])
            # The wif-SE path needs EITHER the fit-time frame or the precomputed
            # bookkeeping. Post-fit re-aggregation supplies only the latter (the
            # frame is deliberately not retained), and every frame dereference
            # inside now prefers `precomputed`.
            if precomputed is None and (df is None or unit is None):
                raise ValueError(
                    "Event-study aggregation needs either the fit-time frame "
                    "(df + unit) or precomputed bookkeeping; got neither."
                )
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
            # Count only finite-contributing NON-reference cells so materialized
            # NaN cells and zero references don't inflate n_groups — matches the
            # all-NaN early-return which already reports 0.
            agg_n_groups.append(sum(1 for gt in gt_pairs if gt not in reference_cells))
            agg_effective_dfs.append(eff_df)
            agg_periods.append(e)
            _psi_vectors.append(psi_e)
            _psi_event_times.append(e)

        # The Eq. (4.14) overall starts unset. It used to be reset on ``self``
        # before any early return so a reused estimator never read a stale value
        # from a prior fit; returning it removes that hazard by construction.
        es_overall: Optional[Dict[str, Any]] = None
        es_df_used: Optional[float] = None

        # Batch inference for all relative periods
        if not agg_effects_list:
            return EventStudyAggregation()
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
        # Stash the ONE df every ES row's inference is about to use (the
        # conservative min under dropped replicates) as provenance for
        # CallawaySantAnnaResults.event_study_df. Recorded iff it will
        # govern a t-reference (finite, > 0; the df=0 replicate sentinel
        # yields NaN inference, not a t-law). Returned on the aggregation
        # object alongside the VCV rather than stashed on ``self``.
        if df_survey_val is not None and np.isfinite(df_survey_val) and df_survey_val > 0:
            es_df_used = float(df_survey_val)
        t_stats, p_values, ci_lowers, ci_uppers = safe_inference_batch(
            np.array(agg_effects_list),
            np.array(agg_ses_list),
            alpha=self.alpha,
            df=df_survey_val,
        )

        event_study_effects: Dict[Any, Dict[str, Any]] = {}
        for idx, e in enumerate(agg_periods):
            event_study_effects[e] = {
                "effect": agg_effects_list[idx],
                "se": agg_ses_list[idx],
                "t_stat": float(t_stats[idx]),
                "p_value": float(p_values[idx]),
                "conf_int": (float(ci_lowers[idx]), float(ci_uppers[idx])),
                "n_groups": agg_n_groups[idx],
            }

        # (Universal-mode zero reference rows are now materialized per cohort at
        # their positional base event time e = base - g during the aggregation
        # above — matching R `did::aggte(type="dynamic")` — rather than as a
        # single fixed e = -1-anticipation display row.)

        # Compute full event-study VCV from per-event-time IF vectors (Phase 7d)
        # This enables HonestDiD to use the full covariance structure
        event_study_vcov = None
        # Pair event times with their IF vectors and keep only non-empty psi, so
        # the stored VCV index (below) always aligns 1:1 with the VCV columns.
        _valid_pairs = [(et, p) for et, p in zip(_psi_event_times, _psi_vectors) if len(p) > 0]
        valid_psi = [p for _, p in _valid_pairs]
        valid_event_times = [et for et, _ in _valid_pairs]
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
        # in HonestDiD when some event times are filtered out). Uses the
        # non-empty-psi event times so the index aligns 1:1 with the VCV columns
        # (reference-only and empty-IF horizons never get a column).
        event_study_vcov_index = valid_event_times if event_study_vcov is not None else None

        # Eq. (4.14) overall ATT: the unweighted mean of the post-treatment
        # event-study effects ES(e). Stashed on self (mirroring _event_study_vcov)
        # so the StaggeredTripleDifference estimator can expose it as overall_att_es;
        # CallawaySantAnna leaves it unread. Post-treatment is the library predicate
        # e >= -anticipation (matching _aggregate_simple and the default overall_att),
        # NOT a hardcoded e >= 0.
        #
        # The POINT ESTIMATE averages EVERY finite post-treatment ES(e) effect (read
        # from event_study_effects by event-time key), so it is always the true Eq.
        # 4.14 average -- it must NOT be silently restricted to horizons with a finite
        # influence function. The SE is the influence function of that mean (the
        # average of the per-event-time combined IFs, via the same survey-aware
        # variance routine as the per-e effects). If any contributing horizon lacks a
        # finite, well-formed combined IF (a finite ES(e) can have a non-finite
        # WIF/IF, which the per-e path already surfaces as a NaN SE), the combined IF
        # for the mean is undefined: the SE is NaN while the point estimate is
        # retained, and the consumer (fit) warns and NaN-propagates the inference.
        post_e = [
            e
            for e in event_study_effects
            if e >= -self.anticipation and np.isfinite(event_study_effects[e]["effect"])
        ]
        if post_e:
            att_es = float(np.mean([event_study_effects[e]["effect"] for e in post_e]))
            psi_by_e = {e: psi for e, psi in zip(_psi_event_times, _psi_vectors)}
            psis = [psi_by_e.get(e) for e in post_e]
            se_es: float = np.nan
            eff_df_es: Optional[int] = None
            if all(p is not None and len(p) > 0 and np.all(np.isfinite(p)) for p in psis):
                if len({len(p) for p in psis}) == 1:
                    psi_es = np.column_stack(psis).mean(axis=1)
                    se_es, eff_df_es = self._se_from_psi(psi_es, precomputed)
            es_overall = {
                "att": att_es,
                "se": float(se_es),
                "effective_df": eff_df_es,
            }

        return EventStudyAggregation(
            effects=event_study_effects,
            overall=es_overall,
            df_used=es_df_used,
            vcov=event_study_vcov,
            vcov_index=event_study_vcov_index,
            reference_event_times=es_reference_event_times,
        )

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
            # Count only finite-contributing cells (gt_pairs is finite-filtered
            # above) so materialized NaN cells don't inflate n_periods.
            group_data_list.append((g, agg_effect, agg_se, len(gt_pairs), eff_df))

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

        # Provenance: the ONE df every group row's inference just used (the
        # conservative min under dropped replicates). Recorded per row iff it
        # will govern a t-reference - finite and > 0; the df=0 replicate
        # sentinel yields NaN inference, not a t-law. Carried on the effect
        # dicts so a post-fit ``aggregate("group")`` reports the df that
        # actually produced the stored p-value/CI rather than re-deriving it
        # from whichever df field happens to be populated on the results.
        group_df_used: Optional[float] = None
        if df_survey_val is not None and np.isfinite(df_survey_val) and df_survey_val > 0:
            group_df_used = float(df_survey_val)

        group_effects = {}
        for idx, (g, agg_effect, agg_se, n_periods, _eff_df) in enumerate(group_data_list):
            group_effects[g] = {
                "effect": agg_effect,
                "se": agg_se,
                "t_stat": float(t_stats[idx]),
                "p_value": float(p_values[idx]),
                "conf_int": (float(ci_lowers[idx]), float(ci_uppers[idx])),
                "n_periods": n_periods,
                "df_used": group_df_used,
            }

        return group_effects
