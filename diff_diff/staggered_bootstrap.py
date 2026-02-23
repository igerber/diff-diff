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
    compute_percentile_ci as _compute_percentile_ci_func,
)
from diff_diff.bootstrap_utils import (
    generate_bootstrap_weights_batch as _generate_bootstrap_weights_batch,
)

if TYPE_CHECKING:
    pass


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

    def _run_multiplier_bootstrap(
        self,
        group_time_effects: Dict[Tuple[Any, Any], Dict[str, Any]],
        influence_func_info: Dict[Tuple[Any, Any], Dict[str, Any]],
        aggregate: Optional[str],
        balance_e: Optional[int],
        treatment_groups: List[Any],
        time_periods: List[Any],
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

        # Collect all unique units across all (g,t) combinations
        all_units = set()
        for (g, t), info in influence_func_info.items():
            all_units.update(info['treated_units'])
            all_units.update(info['control_units'])
        all_units = sorted(all_units)
        n_units = len(all_units)
        unit_to_idx = {u: i for i, u in enumerate(all_units)}

        # Get list of (g,t) pairs
        gt_pairs = list(group_time_effects.keys())
        n_gt = len(gt_pairs)

        # Identify post-treatment (g,t) pairs for overall ATT
        # Pre-treatment effects are for parallel trends assessment, not aggregated
        post_treatment_mask = np.array([
            t >= g - self.anticipation for (g, t) in gt_pairs
        ])
        post_treatment_indices = np.where(post_treatment_mask)[0]

        # Compute aggregation weights for overall ATT (post-treatment only)
        all_n_treated = np.array([
            group_time_effects[gt]['n_treated'] for gt in gt_pairs
        ], dtype=float)
        post_n_treated = all_n_treated[post_treatment_mask]

        # Flag to skip overall ATT aggregation when no post-treatment effects
        # But continue bootstrap for per-effect SEs (pre-treatment effects need bootstrap SEs too)
        skip_overall_aggregation = False
        if len(post_treatment_indices) == 0:
            warnings.warn(
                "No post-treatment effects for bootstrap aggregation. "
                "Overall ATT statistics will be NaN, but per-effect SEs will be computed.",
                UserWarning,
                stacklevel=2
            )
            skip_overall_aggregation = True
            overall_weights_post = np.array([])
        else:
            overall_weights_post = post_n_treated / np.sum(post_n_treated)

        # Original point estimates
        original_atts = np.array([group_time_effects[gt]['effect'] for gt in gt_pairs])
        if skip_overall_aggregation:
            original_overall = np.nan
        else:
            original_overall = np.sum(overall_weights_post * original_atts[post_treatment_mask])

        # Prepare event study and group aggregation info if needed
        event_study_info = None
        group_agg_info = None

        if aggregate in ["event_study", "all"]:
            event_study_info = self._prepare_event_study_aggregation(
                gt_pairs, group_time_effects, balance_e
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
            treated_idx = np.array([unit_to_idx[u] for u in info['treated_units']])
            control_idx = np.array([unit_to_idx[u] for u in info['control_units']])
            gt_treated_indices.append(treated_idx)
            gt_control_indices.append(control_idx)
            gt_treated_inf.append(np.asarray(info['treated_inf']))
            gt_control_inf.append(np.asarray(info['control_inf']))

        # Generate ALL bootstrap weights upfront: shape (n_bootstrap, n_units)
        # This is much faster than generating one at a time
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
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                perturbations = (
                    treated_weights @ treated_inf +
                    control_weights @ control_inf
                )

            # Let non-finite values propagate - they will be handled at statistics computation
            bootstrap_atts_gt[:, j] = original_atts[j] + perturbations

        # Vectorized overall ATT: matrix-vector multiply (post-treatment only)
        # Shape: (n_bootstrap,)
        if skip_overall_aggregation:
            bootstrap_overall = np.full(self.n_bootstrap, np.nan)
        else:
            # Suppress RuntimeWarnings for edge cases - non-finite values handled at statistics computation
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                bootstrap_overall = bootstrap_atts_gt[:, post_treatment_indices] @ overall_weights_post

        # Vectorized event study aggregation
        # Non-finite values handled at statistics computation stage
        rel_periods: List[int] = []
        bootstrap_event_study: Optional[Dict[int, np.ndarray]] = None
        if event_study_info is not None:
            rel_periods = sorted(event_study_info.keys())
            bootstrap_event_study = {}
            for e in rel_periods:
                agg_info = event_study_info[e]
                gt_indices = agg_info['gt_indices']
                weights = agg_info['weights']
                # Vectorized: select columns and multiply by weights
                # Suppress RuntimeWarnings for edge cases
                with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                    bootstrap_event_study[e] = bootstrap_atts_gt[:, gt_indices] @ weights

        # Vectorized group aggregation
        # Non-finite values handled at statistics computation stage
        group_list: List[Any] = []
        bootstrap_group: Optional[Dict[Any, np.ndarray]] = None
        if group_agg_info is not None:
            group_list = sorted(group_agg_info.keys())
            bootstrap_group = {}
            for g in group_list:
                agg_info = group_agg_info[g]
                gt_indices = agg_info['gt_indices']
                weights = agg_info['weights']
                # Suppress RuntimeWarnings for edge cases
                with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                    bootstrap_group[g] = bootstrap_atts_gt[:, gt_indices] @ weights

        # Compute bootstrap statistics for ATT(g,t)
        gt_ses = {}
        gt_cis = {}
        gt_p_values = {}

        for j, gt in enumerate(gt_pairs):
            se, ci, p_value = self._compute_effect_bootstrap_stats(
                original_atts[j], bootstrap_atts_gt[:, j],
                context=f"ATT(g={gt[0]}, t={gt[1]})"
            )
            gt_ses[gt] = se
            gt_cis[gt] = ci
            gt_p_values[gt] = p_value

        # Compute bootstrap statistics for overall ATT
        if skip_overall_aggregation:
            overall_se = np.nan
            overall_ci = (np.nan, np.nan)
            overall_p_value = np.nan
        else:
            overall_se, overall_ci, overall_p_value = self._compute_effect_bootstrap_stats(
                original_overall, bootstrap_overall,
                context="overall ATT"
            )

        # Compute bootstrap statistics for event study effects
        event_study_ses = None
        event_study_cis = None
        event_study_p_values = None

        if bootstrap_event_study is not None and event_study_info is not None:
            event_study_ses = {}
            event_study_cis = {}
            event_study_p_values = {}

            for e in rel_periods:
                se, ci, p_value = self._compute_effect_bootstrap_stats(
                    event_study_info[e]['effect'], bootstrap_event_study[e],
                    context=f"event study (e={e})"
                )
                event_study_ses[e] = se
                event_study_cis[e] = ci
                event_study_p_values[e] = p_value

        # Compute bootstrap statistics for group effects
        group_effect_ses = None
        group_effect_cis = None
        group_effect_p_values = None

        if bootstrap_group is not None and group_agg_info is not None:
            group_effect_ses = {}
            group_effect_cis = {}
            group_effect_p_values = {}

            for g in group_list:
                se, ci, p_value = self._compute_effect_bootstrap_stats(
                    group_agg_info[g]['effect'], bootstrap_group[g],
                    context=f"group effect (g={g})"
                )
                group_effect_ses[g] = se
                group_effect_cis[g] = ci
                group_effect_p_values[g] = p_value

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
        )

    def _prepare_event_study_aggregation(
        self,
        gt_pairs: List[Tuple[Any, Any]],
        group_time_effects: Dict,
        balance_e: Optional[int],
    ) -> Dict[int, Dict[str, Any]]:
        """Prepare aggregation info for event study bootstrap."""
        # Organize by relative time
        effects_by_e: Dict[int, List[Tuple[int, float, float]]] = {}

        for j, (g, t) in enumerate(gt_pairs):
            e = t - g
            if e not in effects_by_e:
                effects_by_e[e] = []
            effects_by_e[e].append((
                j,  # index in gt_pairs
                group_time_effects[(g, t)]['effect'],
                group_time_effects[(g, t)]['n_treated']
            ))

        # Balance if requested
        if balance_e is not None:
            groups_at_e = set()
            for j, (g, t) in enumerate(gt_pairs):
                if t - g == balance_e:
                    groups_at_e.add(g)

            balanced_effects: Dict[int, List[Tuple[int, float, float]]] = {}
            for j, (g, t) in enumerate(gt_pairs):
                if g in groups_at_e:
                    e = t - g
                    if e not in balanced_effects:
                        balanced_effects[e] = []
                    balanced_effects[e].append((
                        j,
                        group_time_effects[(g, t)]['effect'],
                        group_time_effects[(g, t)]['n_treated']
                    ))
            effects_by_e = balanced_effects

        # Compute aggregation weights
        result = {}
        for e, effect_list in effects_by_e.items():
            indices = np.array([x[0] for x in effect_list])
            effects = np.array([x[1] for x in effect_list])
            n_treated = np.array([x[2] for x in effect_list], dtype=float)

            weights = n_treated / np.sum(n_treated)
            agg_effect = np.sum(weights * effects)

            result[e] = {
                'gt_indices': indices,
                'weights': weights,
                'effect': agg_effect,
            }

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
                    group_data.append((
                        j,
                        group_time_effects[(gg, t)]['effect'],
                    ))

            if not group_data:
                continue

            indices = np.array([x[0] for x in group_data])
            effects = np.array([x[1] for x in group_data])

            # Equal weights across time periods
            weights = np.ones(len(effects)) / len(effects)
            agg_effect = np.sum(weights * effects)

            result[g] = {
                'gt_indices': indices,
                'weights': weights,
                'effect': agg_effect,
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
