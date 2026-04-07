"""
Bootstrap inference for Callaway-Sant'Anna estimator.

This module provides the bootstrap results container and the mixin class
with bootstrap inference methods. Weight generation and statistical helpers
are in :mod:`diff_diff.bootstrap_utils`.
"""

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from diff_diff.bootstrap_utils import (
    compute_bootstrap_pvalue as _compute_bootstrap_pvalue_func,
)
from diff_diff.bootstrap_utils import (
    compute_effect_bootstrap_stats as _compute_effect_bootstrap_stats_func,
)
from diff_diff.bootstrap_utils import (
    compute_effect_bootstrap_stats_batch as _compute_effect_bootstrap_stats_batch_func,
)
from diff_diff.bootstrap_utils import (
    compute_percentile_ci as _compute_percentile_ci_func,
)
from diff_diff.bootstrap_utils import (
    generate_bootstrap_weights_batch as _generate_bootstrap_weights_batch,
)
from diff_diff.bootstrap_utils import (
    generate_survey_multiplier_weights_batch as _generate_survey_multiplier_weights_batch,
)

if TYPE_CHECKING:
    import pandas as pd

    from diff_diff.staggered_aggregation import PrecomputedData


# =============================================================================
# Bootstrap Results Container
# =============================================================================


@dataclass
class CSBootstrapResults:
    """
    Results from Callaway-Sant'Anna multiplier bootstrap inference.

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
    group_time_ses : Dict[Tuple[Any, Any], float]
        Bootstrap SEs for each ATT(g,t).
    group_time_cis : Dict[Tuple[Any, Any], Tuple[float, float]]
        Bootstrap CIs for each ATT(g,t).
    group_time_p_values : Dict[Tuple[Any, Any], float]
        Bootstrap p-values for each ATT(g,t).
    event_study_ses : Optional[Dict[int, float]]
        Bootstrap SEs for event study effects.
    event_study_cis : Optional[Dict[int, Tuple[float, float]]]
        Bootstrap CIs for event study effects.
    event_study_p_values : Optional[Dict[int, float]]
        Bootstrap p-values for event study effects.
    group_effect_ses : Optional[Dict[Any, float]]
        Bootstrap SEs for group effects.
    group_effect_cis : Optional[Dict[Any, Tuple[float, float]]]
        Bootstrap CIs for group effects.
    group_effect_p_values : Optional[Dict[Any, float]]
        Bootstrap p-values for group effects.
    bootstrap_distribution : Optional[np.ndarray]
        Full bootstrap distribution of overall ATT (if requested).
    """

    n_bootstrap: int
    weight_type: str
    alpha: float
    overall_att_se: float
    overall_att_ci: Tuple[float, float]
    overall_att_p_value: float
    group_time_ses: Dict[Tuple[Any, Any], float]
    group_time_cis: Dict[Tuple[Any, Any], Tuple[float, float]]
    group_time_p_values: Dict[Tuple[Any, Any], float]
    event_study_ses: Optional[Dict[int, float]] = None
    event_study_cis: Optional[Dict[int, Tuple[float, float]]] = None
    event_study_p_values: Optional[Dict[int, float]] = None
    group_effect_ses: Optional[Dict[Any, float]] = None
    group_effect_cis: Optional[Dict[Any, Tuple[float, float]]] = None
    group_effect_p_values: Optional[Dict[Any, float]] = None
    bootstrap_distribution: Optional[np.ndarray] = field(default=None, repr=False)
    cband_crit_value: Optional[float] = None


# =============================================================================
# Bootstrap Mixin Class
# =============================================================================


class CallawaySantAnnaBootstrapMixin:
    """
    Mixin class providing bootstrap inference methods for CallawaySantAnna.

    This class is not intended to be used standalone. It provides methods
    that are used by the main CallawaySantAnna class for multiplier bootstrap
    inference.
    """

    # Type hints for attributes accessed from the main class
    n_bootstrap: int
    bootstrap_weight_type: str
    alpha: float
    seed: Optional[int]
    anticipation: int

    if TYPE_CHECKING:

        def _compute_combined_influence_function(
            self,
            gt_pairs: List[Tuple[Any, Any]],
            weights: np.ndarray,
            effects: np.ndarray,
            groups_for_gt: np.ndarray,
            influence_func_info: Dict,
            df: "pd.DataFrame",
            unit: str,
            precomputed: Optional["PrecomputedData"] = None,
            global_unit_to_idx: Optional[Dict[Any, int]] = None,
            n_global_units: Optional[int] = None,
        ) -> Tuple[np.ndarray, Optional[List]]: ...

    def _run_multiplier_bootstrap(
        self,
        group_time_effects: Dict[Tuple[Any, Any], Dict[str, Any]],
        influence_func_info: Dict[Tuple[Any, Any], Dict[str, Any]],
        aggregate: Optional[str],
        balance_e: Optional[int],
        treatment_groups: List[Any],
        time_periods: List[Any],
        df: Any = None,
        unit: Optional[str] = None,
        precomputed: Any = None,
        cband: bool = True,
    ) -> CSBootstrapResults:
        """
        Run multiplier bootstrap for inference on all parameters.

        This implements the multiplier bootstrap procedure from Callaway & Sant'Anna (2021).
        The key idea is to perturb the influence function contributions with random
        weights at the cluster (unit) level, then recompute aggregations.

        Parameters
        ----------
        group_time_effects : dict
            Dictionary of ATT(g,t) effects with analytical SEs.
        influence_func_info : dict
            Dictionary mapping (g,t) to influence function information.
        aggregate : str, optional
            Type of aggregation requested.
        balance_e : int, optional
            Balance parameter for event study.
        treatment_groups : list
            List of treatment cohorts.
        time_periods : list
            List of time periods.

        Returns
        -------
        CSBootstrapResults
            Bootstrap inference results.
        """
        # Warn about low bootstrap iterations
        if self.n_bootstrap < 50:
            warnings.warn(
                f"n_bootstrap={self.n_bootstrap} is low. Consider n_bootstrap >= 199 "
                "for reliable inference. Percentile confidence intervals and p-values "
                "may be unreliable with few iterations.",
                UserWarning,
                stacklevel=3,
            )

        rng = np.random.default_rng(self.seed)

        # Use global unit set for correct pg = n_g / N_total scaling.
        # Without this, pg is overestimated in unbalanced panels where some
        # units don't appear in any influence function.
        if precomputed is not None:
            all_units = precomputed["all_units"]
            n_units = precomputed.get("canonical_size", len(all_units))
            unit_to_idx = precomputed["unit_to_idx"]  # None for RCS
        else:
            # Fallback: collect units from influence functions
            all_units_set = set()
            for (g, t), info in influence_func_info.items():
                all_units_set.update(info["treated_units"])
                all_units_set.update(info["control_units"])
            all_units = sorted(all_units_set)
            # Use global N from dataframe when available
            n_units = (
                df[unit].nunique() if (df is not None and unit is not None) else len(all_units)
            )
            unit_to_idx = {u: i for i, u in enumerate(all_units)}

        # Get list of (g,t) pairs that have influence function info
        # (skip zero-mass cells that recorded NaN ATT without IF)
        gt_pairs = [gt for gt in group_time_effects.keys() if gt in influence_func_info]
        n_gt = len(gt_pairs)

        # Identify post-treatment (g,t) pairs for overall ATT
        # Pre-treatment effects are for parallel trends assessment, not aggregated
        post_treatment_mask = np.array([t >= g - self.anticipation for (g, t) in gt_pairs])
        post_treatment_indices = np.where(post_treatment_mask)[0]

        # Compute aggregation weights for overall ATT (post-treatment only)
        # When survey weights are present, use fixed cohort survey masses
        # (from precomputed survey_weights × unit_cohorts), matching the
        # analytical _aggregate_simple() path in staggered_aggregation.py.
        # Do NOT use per-cell survey_weight_sum (which varies by cell on
        # unbalanced panels).
        survey_w = precomputed.get("survey_weights") if precomputed is not None else None
        if survey_w is not None:
            unit_cohorts = precomputed["unit_cohorts"]
            # Precompute fixed cohort masses (same formula as _aggregate_simple)
            _cohort_mass_cache: dict = {}
            for gt in gt_pairs:
                g = gt[0]
                if g not in _cohort_mass_cache:
                    _cohort_mass_cache[g] = float(np.sum(survey_w[unit_cohorts == g]))
            all_n_treated = np.array([_cohort_mass_cache[gt[0]] for gt in gt_pairs], dtype=float)
        else:
            # Use agg_weight if available (RCS: fixed cohort mass);
            # fall back to n_treated for panel data
            all_n_treated = np.array(
                [
                    group_time_effects[gt].get("agg_weight", group_time_effects[gt]["n_treated"])
                    for gt in gt_pairs
                ],
                dtype=float,
            )
        post_n_treated = all_n_treated[post_treatment_mask]

        # Filter out NaN ATT(g,t) cells from overall aggregation (matches analytical path)
        post_effects_raw = np.array(
            [group_time_effects[gt_pairs[i]]["effect"] for i in post_treatment_indices]
        )
        finite_post = np.isfinite(post_effects_raw)
        if not np.all(finite_post):
            post_treatment_indices = post_treatment_indices[finite_post]
            post_n_treated = post_n_treated[finite_post]

        # Flag to skip overall ATT aggregation when no post-treatment effects
        # But continue bootstrap for per-effect SEs (pre-treatment effects need bootstrap SEs too)
        skip_overall_aggregation = False
        if len(post_treatment_indices) == 0:
            warnings.warn(
                "No post-treatment effects for bootstrap aggregation. "
                "Overall ATT statistics will be NaN, but per-effect SEs will be computed.",
                UserWarning,
                stacklevel=2,
            )
            skip_overall_aggregation = True
            overall_weights_post = np.array([])
        else:
            overall_weights_post = post_n_treated / np.sum(post_n_treated)

        # Original point estimates
        original_atts = np.array([group_time_effects[gt]["effect"] for gt in gt_pairs])
        if skip_overall_aggregation:
            original_overall = np.nan
        else:
            original_overall = np.sum(overall_weights_post * original_atts[post_treatment_indices])

        # Prepare event study and group aggregation info if needed
        event_study_info = None
        group_agg_info = None

        if aggregate in ["event_study", "all"]:
            event_study_info = self._prepare_event_study_aggregation(
                gt_pairs,
                group_time_effects,
                balance_e,
                influence_func_info=influence_func_info,
                df=df,
                unit=unit,
                precomputed=precomputed,
                global_unit_to_idx=unit_to_idx,
                n_global_units=n_units,
            )

        if aggregate in ["group", "all"]:
            group_agg_info = self._prepare_group_aggregation(
                gt_pairs, group_time_effects, treatment_groups
            )

        # Pre-compute unit index arrays for each (g,t) pair (done once, not per iteration)
        gt_treated_indices = []
        gt_control_indices = []
        gt_treated_inf = []
        gt_control_inf = []

        for j, gt in enumerate(gt_pairs):
            info = influence_func_info[gt]
            gt_treated_indices.append(info["treated_idx"])
            gt_control_indices.append(info["control_idx"])
            gt_treated_inf.append(np.asarray(info["treated_inf"]))
            gt_control_inf.append(np.asarray(info["control_inf"]))

        # Generate bootstrap weights — PSU-level when survey design is present,
        # unit-level otherwise.
        resolved_survey_unit = (
            precomputed.get("resolved_survey_unit") if precomputed is not None else None
        )
        _use_survey_bootstrap = resolved_survey_unit is not None and (
            resolved_survey_unit.strata is not None
            or resolved_survey_unit.psu is not None
            or resolved_survey_unit.fpc is not None
        )

        if _use_survey_bootstrap:
            # PSU-level multiplier weights
            psu_weights, psu_ids = _generate_survey_multiplier_weights_batch(
                self.n_bootstrap, resolved_survey_unit, self.bootstrap_weight_type, rng
            )
            # Build unit → PSU column map
            if resolved_survey_unit.psu is not None:
                unit_psu = resolved_survey_unit.psu
                psu_id_to_col = {int(p): c for c, p in enumerate(psu_ids)}
                unit_to_psu_col = np.array(
                    [psu_id_to_col[int(unit_psu[i])] for i in range(n_units)]
                )
            else:
                # Each unit is its own PSU — identity mapping
                unit_to_psu_col = np.arange(n_units)

            # Expand PSU weights to unit level for per-(g,t) perturbation
            # Shape: (n_bootstrap, n_units)
            all_bootstrap_weights = psu_weights[:, unit_to_psu_col]
        else:
            # Standard unit-level weights (no survey or weights-only)
            all_bootstrap_weights = _generate_bootstrap_weights_batch(
                self.n_bootstrap, n_units, self.bootstrap_weight_type, rng
            )

        # Vectorized bootstrap ATT(g,t) computation
        # Compute all bootstrap ATTs for all (g,t) pairs using matrix operations
        bootstrap_atts_gt = np.zeros((self.n_bootstrap, n_gt))

        for j in range(n_gt):
            treated_idx = gt_treated_indices[j]
            control_idx = gt_control_indices[j]
            treated_inf = gt_treated_inf[j]
            control_inf = gt_control_inf[j]

            # Extract weights for this (g,t)'s units across all bootstrap iterations
            # Shape: (n_bootstrap, n_treated) and (n_bootstrap, n_control)
            treated_weights = all_bootstrap_weights[:, treated_idx]
            control_weights = all_bootstrap_weights[:, control_idx]

            # Vectorized perturbation: matrix-vector multiply
            # Shape: (n_bootstrap,)
            # Suppress RuntimeWarnings for edge cases (small samples, extreme weights)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                perturbations = treated_weights @ treated_inf + control_weights @ control_inf

            # Let non-finite values propagate - they will be handled at statistics computation
            bootstrap_atts_gt[:, j] = original_atts[j] + perturbations

        # Vectorized overall ATT using combined IF (includes WIF)
        # Shape: (n_bootstrap,)
        if skip_overall_aggregation:
            bootstrap_overall = np.full(self.n_bootstrap, np.nan)
        else:
            # Use combined IF (standard IF + WIF) for proper bootstrap
            post_gt_pairs = [gt_pairs[i] for i in post_treatment_indices]
            post_groups = np.array([gt_pairs[i][0] for i in post_treatment_indices])
            post_effects = original_atts[post_treatment_indices]
            overall_combined_if, _ = self._compute_combined_influence_function(
                post_gt_pairs,
                overall_weights_post,
                post_effects,
                post_groups,
                influence_func_info,
                df,
                unit,
                precomputed,
                global_unit_to_idx=unit_to_idx,
                n_global_units=n_units,
            )
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                bootstrap_overall = original_overall + all_bootstrap_weights @ overall_combined_if

        # Vectorized event study aggregation using combined IFs
        # Non-finite values handled at statistics computation stage
        rel_periods: List[int] = []
        bootstrap_event_study: Optional[Dict[int, np.ndarray]] = None
        if event_study_info is not None:
            rel_periods = sorted(event_study_info.keys())
            bootstrap_event_study = {}
            for e in rel_periods:
                agg_info = event_study_info[e]
                # Use combined IF (standard IF + WIF) for proper bootstrap
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    bootstrap_event_study[e] = (
                        agg_info["effect"] + all_bootstrap_weights @ agg_info["combined_if"]
                    )

        # Vectorized group aggregation
        # Non-finite values handled at statistics computation stage
        group_list: List[Any] = []
        bootstrap_group: Optional[Dict[Any, np.ndarray]] = None
        if group_agg_info is not None:
            group_list = sorted(group_agg_info.keys())
            bootstrap_group = {}
            for g in group_list:
                agg_info = group_agg_info[g]
                gt_indices = agg_info["gt_indices"]
                weights = agg_info["weights"]
                # Suppress RuntimeWarnings for edge cases
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    bootstrap_group[g] = bootstrap_atts_gt[:, gt_indices] @ weights

        # Batch compute bootstrap statistics for ATT(g,t)
        batch_ses, batch_ci_lo, batch_ci_hi, batch_pv = _compute_effect_bootstrap_stats_batch_func(
            original_atts,
            bootstrap_atts_gt,
            alpha=self.alpha,
        )
        gt_ses = {}
        gt_cis = {}
        gt_p_values = {}
        for j, gt in enumerate(gt_pairs):
            gt_ses[gt] = float(batch_ses[j])
            gt_cis[gt] = (float(batch_ci_lo[j]), float(batch_ci_hi[j]))
            gt_p_values[gt] = float(batch_pv[j])

        # Compute bootstrap statistics for overall ATT
        if skip_overall_aggregation:
            overall_se = np.nan
            overall_ci = (np.nan, np.nan)
            overall_p_value = np.nan
        else:
            overall_se, overall_ci, overall_p_value = _compute_effect_bootstrap_stats_func(
                original_overall,
                bootstrap_overall,
                alpha=self.alpha,
                context="overall ATT",
            )

        # Batch compute bootstrap statistics for event study effects
        event_study_ses = None
        event_study_cis = None
        event_study_p_values = None

        if bootstrap_event_study is not None and event_study_info is not None:
            es_effects = np.array([event_study_info[e]["effect"] for e in rel_periods])
            es_boot_matrix = np.column_stack([bootstrap_event_study[e] for e in rel_periods])
            es_ses, es_ci_lo, es_ci_hi, es_pv = _compute_effect_bootstrap_stats_batch_func(
                es_effects,
                es_boot_matrix,
                alpha=self.alpha,
            )
            event_study_ses = {e: float(es_ses[i]) for i, e in enumerate(rel_periods)}
            event_study_cis = {
                e: (float(es_ci_lo[i]), float(es_ci_hi[i])) for i, e in enumerate(rel_periods)
            }
            event_study_p_values = {e: float(es_pv[i]) for i, e in enumerate(rel_periods)}

        # Batch compute bootstrap statistics for group effects
        group_effect_ses = None
        group_effect_cis = None
        group_effect_p_values = None

        if bootstrap_group is not None and group_agg_info is not None:
            grp_effects = np.array([group_agg_info[g]["effect"] for g in group_list])
            grp_boot_matrix = np.column_stack([bootstrap_group[g] for g in group_list])
            grp_ses, grp_ci_lo, grp_ci_hi, grp_pv = _compute_effect_bootstrap_stats_batch_func(
                grp_effects,
                grp_boot_matrix,
                alpha=self.alpha,
            )
            group_effect_ses = {g: float(grp_ses[i]) for i, g in enumerate(group_list)}
            group_effect_cis = {
                g: (float(grp_ci_lo[i]), float(grp_ci_hi[i])) for i, g in enumerate(group_list)
            }
            group_effect_p_values = {g: float(grp_pv[i]) for i, g in enumerate(group_list)}

        # Compute simultaneous confidence band critical value (sup-t)
        cband_crit_value = None
        if (
            cband
            and bootstrap_event_study is not None
            and event_study_ses is not None
            and event_study_info is not None
        ):
            valid_es = [
                e
                for e in rel_periods
                if e in event_study_ses
                and np.isfinite(event_study_ses[e])
                and event_study_ses[e] > 0
            ]
            if valid_es:
                # Vectorized sup_t: max_e |(boot_att_e[b] - att_e) / se_e|
                boot_matrix = np.array([bootstrap_event_study[e] for e in valid_es])
                effects_vec = np.array([event_study_info[e]["effect"] for e in valid_es])
                ses_vec = np.array([event_study_ses[e] for e in valid_es])
                with np.errstate(divide="ignore", invalid="ignore"):
                    sup_t_dist = np.max(
                        np.abs((boot_matrix - effects_vec[:, None]) / ses_vec[:, None]),
                        axis=0,
                    )
                finite_mask = np.isfinite(sup_t_dist)
                n_valid = int(np.sum(finite_mask))
                n_total = len(sup_t_dist)
                if n_valid < n_total * 0.5:
                    warnings.warn(
                        f"Too few valid sup-t bootstrap samples ({n_valid}/{n_total}). "
                        "Returning None for cband critical value.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                elif n_valid > 0:
                    cband_crit_value = float(np.quantile(sup_t_dist[finite_mask], 1 - self.alpha))

        return CSBootstrapResults(
            n_bootstrap=self.n_bootstrap,
            weight_type=self.bootstrap_weight_type,
            alpha=self.alpha,
            overall_att_se=overall_se,
            overall_att_ci=overall_ci,
            overall_att_p_value=overall_p_value,
            group_time_ses=gt_ses,
            group_time_cis=gt_cis,
            group_time_p_values=gt_p_values,
            event_study_ses=event_study_ses,
            event_study_cis=event_study_cis,
            event_study_p_values=event_study_p_values,
            group_effect_ses=group_effect_ses,
            group_effect_cis=group_effect_cis,
            group_effect_p_values=group_effect_p_values,
            bootstrap_distribution=bootstrap_overall,
            cband_crit_value=cband_crit_value,
        )

    def _prepare_event_study_aggregation(
        self,
        gt_pairs: List[Tuple[Any, Any]],
        group_time_effects: Dict,
        balance_e: Optional[int],
        influence_func_info: Any = None,
        df: Any = None,
        unit: Optional[str] = None,
        precomputed: Any = None,
        global_unit_to_idx: Optional[Dict[Any, int]] = None,
        n_global_units: Optional[int] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """Prepare aggregation info for event study bootstrap."""
        # Use fixed cohort survey masses (not per-cell survey_weight_sum) when
        # survey weights are present, matching the analytical
        # _aggregate_event_study() path.
        survey_w = precomputed.get("survey_weights") if precomputed is not None else None
        _cohort_mass: Optional[dict] = None
        if survey_w is not None:
            unit_cohorts = precomputed["unit_cohorts"]
            _cohort_mass = {}

        def _agg_weight(g: Any, t: Any) -> float:
            if _cohort_mass is not None:
                if g not in _cohort_mass:
                    _cohort_mass[g] = float(np.sum(survey_w[unit_cohorts == g]))
                return _cohort_mass[g]
            # Use agg_weight if available (RCS: fixed cohort mass)
            return group_time_effects[(g, t)].get(
                "agg_weight", group_time_effects[(g, t)]["n_treated"]
            )

        # Organize by relative time
        effects_by_e: Dict[int, List[Tuple[int, float, float]]] = {}

        for j, (g, t) in enumerate(gt_pairs):
            e = t - g
            if e not in effects_by_e:
                effects_by_e[e] = []
            effects_by_e[e].append(
                (
                    j,  # index in gt_pairs
                    group_time_effects[(g, t)]["effect"],
                    _agg_weight(g, t),
                )
            )

        # Balance if requested
        if balance_e is not None:
            groups_at_e = set()
            for j, (g, t) in enumerate(gt_pairs):
                if t - g == balance_e and np.isfinite(group_time_effects[(g, t)]["effect"]):
                    groups_at_e.add(g)

            balanced_effects: Dict[int, List[Tuple[int, float, float]]] = {}
            for j, (g, t) in enumerate(gt_pairs):
                if g in groups_at_e:
                    e = t - g
                    if e not in balanced_effects:
                        balanced_effects[e] = []
                    balanced_effects[e].append(
                        (
                            j,
                            group_time_effects[(g, t)]["effect"],
                            _agg_weight(g, t),
                        )
                    )
            effects_by_e = balanced_effects

        # Compute aggregation weights
        result = {}
        for e, effect_list in effects_by_e.items():
            indices = np.array([x[0] for x in effect_list])
            effects = np.array([x[1] for x in effect_list])
            n_treated = np.array([x[2] for x in effect_list], dtype=float)

            # Exclude NaN effects (matches analytical aggregation path)
            finite_mask = np.isfinite(effects)
            if not np.all(finite_mask):
                indices = indices[finite_mask]
                effects = effects[finite_mask]
                n_treated = n_treated[finite_mask]
                if len(effects) == 0:
                    continue

            weights = n_treated / np.sum(n_treated)
            agg_effect = np.sum(weights * effects)

            entry: Dict[str, Any] = {
                "gt_indices": indices,
                "weights": weights,
                "effect": agg_effect,
            }

            # Compute combined IF for this event time if args available
            if influence_func_info is not None and df is not None and unit is not None:
                gt_pairs_for_e = [gt_pairs[i] for i in indices]
                groups_for_gt = np.array([gt_pairs[i][0] for i in indices])
                combined_if, _ = self._compute_combined_influence_function(
                    gt_pairs_for_e,
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
                entry["combined_if"] = combined_if

            result[e] = entry

        return result

    def _prepare_group_aggregation(
        self,
        gt_pairs: List[Tuple[Any, Any]],
        group_time_effects: Dict,
        treatment_groups: List[Any],
    ) -> Dict[Any, Dict[str, Any]]:
        """Prepare aggregation info for group-level bootstrap."""
        result = {}

        for g in treatment_groups:
            # Get all effects for this group (post-treatment only: t >= g - anticipation)
            group_data = []
            for j, (gg, t) in enumerate(gt_pairs):
                if gg == g and t >= g - self.anticipation:
                    group_data.append(
                        (
                            j,
                            group_time_effects[(gg, t)]["effect"],
                        )
                    )

            if not group_data:
                continue

            indices = np.array([x[0] for x in group_data])
            effects = np.array([x[1] for x in group_data])

            # Exclude NaN effects (matches analytical aggregation path)
            finite_mask = np.isfinite(effects)
            if not np.all(finite_mask):
                indices = indices[finite_mask]
                effects = effects[finite_mask]
                if len(effects) == 0:
                    continue

            # Equal weights across time periods
            weights = np.ones(len(effects)) / len(effects)
            agg_effect = np.sum(weights * effects)

            result[g] = {
                "gt_indices": indices,
                "weights": weights,
                "effect": agg_effect,
            }

        return result

    def _compute_percentile_ci(
        self,
        boot_dist: np.ndarray,
        alpha: float,
    ) -> Tuple[float, float]:
        """Compute percentile confidence interval from bootstrap distribution."""
        return _compute_percentile_ci_func(boot_dist, alpha)

    def _compute_bootstrap_pvalue(
        self,
        original_effect: float,
        boot_dist: np.ndarray,
        n_valid: Optional[int] = None,
    ) -> float:
        """
        Compute two-sided bootstrap p-value.

        Delegates to :func:`bootstrap_utils.compute_bootstrap_pvalue`.
        """
        return _compute_bootstrap_pvalue_func(original_effect, boot_dist, n_valid=n_valid)

    def _compute_effect_bootstrap_stats(
        self,
        original_effect: float,
        boot_dist: np.ndarray,
        context: str = "bootstrap distribution",
    ) -> Tuple[float, Tuple[float, float], float]:
        """
        Compute bootstrap statistics for a single effect.

        Delegates to :func:`bootstrap_utils.compute_effect_bootstrap_stats`.
        """
        return _compute_effect_bootstrap_stats_func(
            original_effect, boot_dist, alpha=self.alpha, context=context
        )
