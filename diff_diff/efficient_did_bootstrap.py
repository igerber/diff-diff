"""
Multiplier bootstrap inference for the Efficient DiD estimator.

Pattern follows CallawaySantAnnaBootstrapMixin (staggered_bootstrap.py).
Perturbs EIF values with random weights to obtain bootstrap distributions
of ATT(g,t) and aggregated parameters.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from diff_diff.bootstrap_utils import (
    compute_effect_bootstrap_stats as _compute_effect_bootstrap_stats_func,
)
from diff_diff.bootstrap_utils import (
    generate_bootstrap_weights_batch as _generate_bootstrap_weights_batch,
)


@dataclass
class EDiDBootstrapResults:
    """Bootstrap inference results for EfficientDiD."""

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


class EfficientDiDBootstrapMixin:
    """Mixin providing multiplier bootstrap for EfficientDiD."""

    n_bootstrap: int
    bootstrap_weights: str
    alpha: float
    seed: Optional[int]
    anticipation: int

    def _run_multiplier_bootstrap(
        self,
        group_time_effects: Dict[Tuple[Any, Any], Dict[str, Any]],
        eif_by_gt: Dict[Tuple[Any, Any], np.ndarray],
        n_units: int,
        aggregate: Optional[str],
        balance_e: Optional[int],
        treatment_groups: List[Any],
        cohort_fractions: Dict[float, float],
        cluster_indices: Optional[np.ndarray] = None,
        n_clusters: Optional[int] = None,
        resolved_survey: object = None,
    ) -> EDiDBootstrapResults:
        """Run multiplier bootstrap on stored EIF values.

        For each bootstrap draw *b*, perturb ATT(g,t) as::

            ATT_b(g,t) = ATT(g,t) + (1/n) * xi_b @ eif_gt

        where ``xi_b`` is an i.i.d. weight vector of length ``n_units``.
        When ``cluster_indices`` is provided, weights are generated at the
        cluster level and expanded to units.

        Aggregations (overall, event study, group) are recomputed from
        the perturbed ATT(g,t) values.

        Note: Bootstrap aggregation uses fixed cohort-size weights, consistent
        with the Callaway-Sant'Anna bootstrap pattern (staggered_bootstrap.py).
        The analytical path includes a WIF correction for aggregated SEs, but
        the bootstrap captures weight uncertainty through EIF perturbation.
        This matches the R ``did`` package approach.
        """
        if self.n_bootstrap < 50:
            warnings.warn(
                f"n_bootstrap={self.n_bootstrap} is low. Consider n_bootstrap >= 199 "
                "for reliable inference.",
                UserWarning,
                stacklevel=3,
            )

        rng = np.random.default_rng(self.seed)

        gt_pairs = list(group_time_effects.keys())
        n_gt = len(gt_pairs)

        # Generate bootstrap weights — PSU-level when survey design is present,
        # cluster-level if clustered, unit-level otherwise.
        _use_survey_bootstrap = resolved_survey is not None and (
            resolved_survey.strata is not None
            or resolved_survey.psu is not None
            or resolved_survey.fpc is not None
        )

        if _use_survey_bootstrap:
            from diff_diff.bootstrap_utils import (
                generate_survey_multiplier_weights_batch as _gen_survey_weights,
            )

            psu_weights, psu_ids = _gen_survey_weights(
                self.n_bootstrap, resolved_survey, self.bootstrap_weights, rng
            )
            # Build unit -> PSU column map
            if resolved_survey.psu is not None:
                psu_id_to_col = {int(p): c for c, p in enumerate(psu_ids)}
                unit_to_psu_col = np.array(
                    [psu_id_to_col[int(resolved_survey.psu[i])] for i in range(n_units)]
                )
            else:
                unit_to_psu_col = np.arange(n_units)
            all_weights = psu_weights[:, unit_to_psu_col]
        elif cluster_indices is not None and n_clusters is not None:
            cluster_weights = _generate_bootstrap_weights_batch(
                self.n_bootstrap, n_clusters, self.bootstrap_weights, rng
            )
            # Expand cluster weights to unit level
            all_weights = cluster_weights[:, cluster_indices]
        else:
            all_weights = _generate_bootstrap_weights_batch(
                self.n_bootstrap, n_units, self.bootstrap_weights, rng
            )

        # Original ATTs
        original_atts = np.array([group_time_effects[gt]["effect"] for gt in gt_pairs])

        # Perturbed ATTs: (n_bootstrap, n_gt)
        bootstrap_atts = np.zeros((self.n_bootstrap, n_gt))
        for j, gt in enumerate(gt_pairs):
            eif_gt = eif_by_gt[gt]  # shape (n_units,)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                perturbation = (all_weights @ eif_gt) / n_units
            bootstrap_atts[:, j] = original_atts[j] + perturbation

        # Post-treatment mask — also exclude NaN effects
        post_mask = np.array(
            [
                t >= g - self.anticipation and np.isfinite(original_atts[j])
                for j, (g, t) in enumerate(gt_pairs)
            ]
        )
        post_indices = np.where(post_mask)[0]

        # Overall ATT: fixed-weight re-aggregation of perturbed cell ATTs.
        # This matches CallawaySantAnna._run_multiplier_bootstrap
        # (staggered_bootstrap.py:281). The analytical path includes a WIF
        # correction; bootstrap captures sampling variability through per-cell
        # EIF perturbation without re-estimating weights — this is standard
        # in both this library's CS implementation and the R did package.
        skip_overall = len(post_indices) == 0
        if skip_overall:
            bootstrap_overall = np.full(self.n_bootstrap, np.nan)
            original_overall = np.nan
        else:
            post_groups = [gt_pairs[i][0] for i in post_indices]
            pg = np.array([cohort_fractions.get(g, 0.0) for g in post_groups])
            agg_w = pg / pg.sum() if pg.sum() > 0 else np.ones(len(pg)) / len(pg)
            original_overall = float(np.sum(agg_w * original_atts[post_mask]))
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                bootstrap_overall = bootstrap_atts[:, post_indices] @ agg_w

        # Event study: fixed-weight re-aggregation (same pattern as overall).
        # See note above re: WIF — analytical WIF is not needed in bootstrap.
        bootstrap_event_study = None
        event_study_info = None
        if aggregate in ("event_study", "all"):
            event_study_info = self._prepare_es_agg_boot(
                gt_pairs, original_atts, cohort_fractions, balance_e
            )
            bootstrap_event_study = {}
            for e, info in event_study_info.items():
                idx = info["gt_indices"]
                w = info["weights"]
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    bootstrap_event_study[e] = bootstrap_atts[:, idx] @ w

        # Group aggregation
        bootstrap_group = None
        group_agg_info = None
        if aggregate in ("group", "all"):
            group_agg_info = self._prepare_group_agg_boot(gt_pairs, original_atts, treatment_groups)
            bootstrap_group = {}
            for g, info in group_agg_info.items():
                idx = info["gt_indices"]
                w = info["weights"]
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    bootstrap_group[g] = bootstrap_atts[:, idx] @ w

        # Compute statistics
        gt_ses: Dict[Tuple[Any, Any], float] = {}
        gt_cis: Dict[Tuple[Any, Any], Tuple[float, float]] = {}
        gt_pvals: Dict[Tuple[Any, Any], float] = {}
        for j, gt in enumerate(gt_pairs):
            se, ci, pv = _compute_effect_bootstrap_stats_func(
                original_atts[j],
                bootstrap_atts[:, j],
                alpha=self.alpha,
                context=f"ATT(g={gt[0]}, t={gt[1]})",
            )
            gt_ses[gt] = se
            gt_cis[gt] = ci
            gt_pvals[gt] = pv

        if skip_overall:
            ov_se, ov_ci, ov_pv = np.nan, (np.nan, np.nan), np.nan
        else:
            ov_se, ov_ci, ov_pv = _compute_effect_bootstrap_stats_func(
                original_overall,
                bootstrap_overall,
                alpha=self.alpha,
                context="overall ATT",
            )

        es_ses = es_cis = es_pvs = None
        if bootstrap_event_study is not None and event_study_info is not None:
            es_ses, es_cis, es_pvs = {}, {}, {}
            for e in sorted(event_study_info.keys()):
                se, ci, pv = _compute_effect_bootstrap_stats_func(
                    event_study_info[e]["effect"],
                    bootstrap_event_study[e],
                    alpha=self.alpha,
                    context=f"event study (e={e})",
                    )
                es_ses[e] = se
                es_cis[e] = ci
                es_pvs[e] = pv

        g_ses = g_cis = g_pvs = None
        if bootstrap_group is not None and group_agg_info is not None:
            g_ses, g_cis, g_pvs = {}, {}, {}
            for g in sorted(group_agg_info.keys()):
                se, ci, pv = _compute_effect_bootstrap_stats_func(
                    group_agg_info[g]["effect"],
                    bootstrap_group[g],
                    alpha=self.alpha,
                    context=f"group effect (g={g})",
                    )
                g_ses[g] = se
                g_cis[g] = ci
                g_pvs[g] = pv

        return EDiDBootstrapResults(
            n_bootstrap=self.n_bootstrap,
            weight_type=self.bootstrap_weights,
            alpha=self.alpha,
            overall_att_se=ov_se,
            overall_att_ci=ov_ci,
            overall_att_p_value=ov_pv,
            group_time_ses=gt_ses,
            group_time_cis=gt_cis,
            group_time_p_values=gt_pvals,
            event_study_ses=es_ses,
            event_study_cis=es_cis,
            event_study_p_values=es_pvs,
            group_effect_ses=g_ses,
            group_effect_cis=g_cis,
            group_effect_p_values=g_pvs,
            bootstrap_distribution=bootstrap_overall,
        )

    def _prepare_es_agg_boot(
        self,
        gt_pairs: List[Tuple[Any, Any]],
        original_atts: np.ndarray,
        cohort_fractions: Dict[float, float],
        balance_e: Optional[int],
    ) -> Dict[int, Dict[str, Any]]:
        """Prepare event-study aggregation info for bootstrap."""
        effects_by_e: Dict[int, List[Tuple[int, float, float]]] = {}
        for j, (g, t) in enumerate(gt_pairs):
            if not np.isfinite(original_atts[j]):
                continue  # Skip NaN cells
            e = t - g
            if e not in effects_by_e:
                effects_by_e[e] = []
            effects_by_e[e].append((j, original_atts[j], cohort_fractions.get(g, 0.0)))

        if balance_e is not None:
            groups_at_e = {
                gt_pairs[j][0]
                for j, (g, t) in enumerate(gt_pairs)
                if t - g == balance_e and np.isfinite(original_atts[j])
            }
            balanced: Dict[int, List[Tuple[int, float, float]]] = {}
            for j, (g, t) in enumerate(gt_pairs):
                if g in groups_at_e:
                    if not np.isfinite(original_atts[j]):
                        continue  # Skip NaN cells even in balanced set
                    e = t - g
                    if e not in balanced:
                        balanced[e] = []
                    balanced[e].append((j, original_atts[j], cohort_fractions.get(g, 0.0)))
            effects_by_e = balanced

        if balance_e is not None and not effects_by_e:
            warnings.warn(
                f"balance_e={balance_e}: no cohort has a finite effect at the "
                "anchor horizon. Event study will be empty.",
                UserWarning,
                stacklevel=2,
            )

        result = {}
        for e, elist in effects_by_e.items():
            indices = np.array([x[0] for x in elist])
            effs = np.array([x[1] for x in elist])
            pgs = np.array([x[2] for x in elist])
            w = pgs / pgs.sum() if pgs.sum() > 0 else np.ones(len(pgs)) / len(pgs)
            result[e] = {
                "gt_indices": indices,
                "weights": w,
                "effect": float(np.sum(w * effs)),
            }
        return result

    def _prepare_group_agg_boot(
        self,
        gt_pairs: List[Tuple[Any, Any]],
        original_atts: np.ndarray,
        treatment_groups: List[Any],
    ) -> Dict[Any, Dict[str, Any]]:
        """Prepare group-level aggregation info for bootstrap."""
        result = {}
        for g in treatment_groups:
            group_data = [
                (j, original_atts[j])
                for j, (gg, t) in enumerate(gt_pairs)
                if gg == g and t >= g - self.anticipation and np.isfinite(original_atts[j])
            ]
            if not group_data:
                continue
            indices = np.array([x[0] for x in group_data])
            effs = np.array([x[1] for x in group_data])
            w = np.ones(len(effs)) / len(effs)
            result[g] = {
                "gt_indices": indices,
                "weights": w,
                "effect": float(np.sum(w * effs)),
            }
        return result
