"""
Shared event-study aggregation machinery for ContinuousDiD (row M-025).

Leaf module: hosts ``_ContinuousDiDAggregationMixin``, the binarized
event-study aggregation shared between fit-time ``fit(aggregate=
"eventstudy")`` (deprecated) and the post-fit
``ContinuousDiDResults.aggregate("event_study")`` recompute. The import
DAG forces the split: ``continuous_did.py`` imports
``continuous_did_results.py``, so the results module cannot reach the
estimator module - both instead import this leaf (numpy + diff_diff.utils
/ diff_diff.survey only).

The mixin's two methods are the verbatim bodies previously private to
``continuous_did.py``:

- ``_aggregate_event_study`` - the per-relative-period reweighting of
  per-(g, t) binarized ``att_glob`` values (cohort survey mass or
  ``n_treated`` weights).
- ``_compute_event_study_inference`` - fit's formerly-inline analytical
  ES-SE block: per-bin per-unit influence functions from the pruned
  per-cell payload, with plain / TSL-survey / replicate-weight variance
  and ``safe_inference`` at the fit's ``alpha`` and survey df.

The post-fit route calls both on a throwaway
``_ContinuousKitAggregator`` host (``continuous_did_results.py``) whose
inputs come exclusively from the fit-built ``AggregationKit``; fit()
calls them with its locals. Neither method writes ``self`` state.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from diff_diff.survey import compute_survey_vcov
from diff_diff.utils import safe_inference

if TYPE_CHECKING:
    from diff_diff.survey import ResolvedSurveyDesign


class _ContinuousDiDAggregationMixin:
    """Binarized event-study aggregation shared by fit-time and post-fit.

    Host attribute contract (the throwaway kit aggregator sets exactly
    this; ``ContinuousDiD`` carries it as an ``__init__`` param):

    - ``alpha`` - significance level for ``safe_inference``.
    """

    alpha: float

    def _aggregate_event_study(
        self,
        gt_results: Dict[Tuple, Dict],
        gt_bootstrap_info: Optional[Dict[Tuple, Dict]] = None,
        unit_survey_weights: Optional[np.ndarray] = None,
        unit_cohorts: Optional[np.ndarray] = None,
        anticipation: int = 0,
    ) -> Dict[int, Dict[str, Any]]:
        """Aggregate binarized ATT_glob by relative period."""
        effects_by_e: Dict[int, List[Tuple[float, float, Tuple]]] = {}

        for (g, t), r in gt_results.items():
            e = t - g
            if anticipation > 0 and e < -anticipation:
                continue
            if e not in effects_by_e:
                effects_by_e[e] = []
            # Compute weight for this (g,t) cell
            if unit_survey_weights is not None and unit_cohorts is not None:
                # Survey-weighted: sum of survey weights for treated units in group g
                g_mask = unit_cohorts == g
                cell_weight = float(np.sum(unit_survey_weights[g_mask]))
            else:
                cell_weight = float(r["n_treated"])
            effects_by_e[e].append((r["att_glob"], cell_weight, (g, t)))

        result = {}
        for e, entries in sorted(effects_by_e.items()):
            effects = np.array([x[0] for x in entries])
            weights = np.array([x[1] for x in entries])
            if np.sum(weights) > 0:
                w = weights / np.sum(weights)
                agg = float(np.sum(w * effects))
            else:
                agg = np.nan
            result[e] = {
                "effect": agg,
                "se": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "conf_int": (np.nan, np.nan),
            }
        return result

    def _compute_event_study_inference(
        self,
        event_study_effects: Dict[int, Dict[str, Any]],
        gt_summary: Dict[Tuple, Dict],
        gt_es_payload: Dict[Tuple, Dict],
        n_units: int,
        unit_cohorts: np.ndarray,
        unit_survey_weights: Optional[np.ndarray],
        unit_first_panel_row: np.ndarray,
        resolved_survey: Optional["ResolvedSurveyDesign"],
        survey_df: Optional[int],
    ) -> None:
        """Fill analytical se/t/p/CI on the binarized event-study rows.

        The verbatim body of fit's former inline "Event study analytical
        SEs" block: mutates ``event_study_effects`` rows in place. The
        fit-time caller passes its ``gt_results`` / ``gt_bootstrap_info``
        locals; the post-fit caller passes the kit's ``gt_summary`` /
        pruned ``gt_es_payload`` (key-compatible by construction). Only
        runs when the fit was analytical with post-treatment cells - the
        callers own that gating.
        """
        unit_sw = unit_survey_weights

        # Build unit-level ResolvedSurveyDesign once (reused per bin)
        unit_resolved_es = None
        if resolved_survey is not None:
            row_idx = unit_first_panel_row
            unit_resolved_es = resolved_survey.subset_to_units_by_row_idx(
                row_idx, unit_weights=unit_survey_weights
            )

        for e_val, info_e in event_study_effects.items():
            # Collect (g,t) cells for this event-time bin
            e_gts = [gt for gt in gt_summary if gt[1] - gt[0] == e_val]
            if not e_gts:
                continue
            # Weights within this bin: survey-weighted mass or n_treated
            if unit_sw is not None:
                ns = np.array(
                    [float(np.sum(unit_sw[unit_cohorts == gt[0]])) for gt in e_gts],
                    dtype=float,
                )
            else:
                ns = np.array(
                    [gt_summary[gt]["n_treated"] for gt in e_gts],
                    dtype=float,
                )
            total_n = ns.sum()
            if total_n == 0:
                continue
            ws = ns / total_n

            # Build per-unit IF for this event-time bin
            if_es = np.zeros(n_units)
            for idx_cell, gt in enumerate(e_gts):
                b_info = gt_es_payload.get(gt, {})
                if not b_info:
                    continue
                w = ws[idx_cell]
                # Covariate path: the binarized event-study effect is
                # att_glob, whose per-unit cell IF is precomputed.
                cov_if = b_info.get("cov_if")
                if cov_if is not None:
                    np.add.at(
                        if_es,
                        cov_if["cell_indices"],
                        w * cov_if["if_att_glob"],
                    )
                    continue
                treated_idx = b_info["treated_indices"]
                control_idx = b_info["control_indices"]
                n_t = b_info["n_treated"]
                n_c = b_info["n_control"]
                # Use survey-weighted masses when available
                if "w_treated" in b_info:
                    n_t = b_info["w_treated"]
                    n_c = b_info["w_control"]
                n_total_gt = n_t + n_c
                p_1 = n_t / n_total_gt
                p_0 = n_c / n_total_gt
                att_glob_gt = b_info["att_glob"]
                mu_0 = b_info["mu_0"]
                delta_y_treated = b_info["delta_y_treated"]
                ee_control = b_info["ee_control"]
                sw_treated = b_info.get("w_treated_arr")

                for k, uid in enumerate(treated_idx):
                    score_k = delta_y_treated[k] - att_glob_gt - mu_0
                    if sw_treated is not None:
                        score_k = sw_treated[k] * score_k
                    if_es[uid] += w * score_k / p_1 / n_total_gt
                for k, uid in enumerate(control_idx):
                    if_es[uid] -= w * ee_control[k] / p_0 / n_total_gt

            # Compute SE: survey-aware TSL or standard sqrt(sum(IF^2))
            if unit_resolved_es is not None:
                if unit_resolved_es.uses_replicate_variance:
                    from diff_diff.survey import compute_replicate_if_variance

                    # Score-scale: psi = w * if_es (matches TSL bread)
                    psi_es = unit_resolved_es.weights * if_es
                    variance, _nv = compute_replicate_if_variance(psi_es, unit_resolved_es)
                    es_se = float(np.sqrt(max(variance, 0.0))) if np.isfinite(variance) else np.nan
                else:
                    X_ones_es = np.ones((n_units, 1))
                    tsl_scale_es = float(unit_resolved_es.weights.sum())
                    if_es_tsl = if_es * tsl_scale_es
                    vcov_es = compute_survey_vcov(X_ones_es, if_es_tsl, unit_resolved_es)
                    es_se = float(np.sqrt(np.abs(vcov_es[0, 0])))
            else:
                es_se = float(np.sqrt(np.sum(if_es**2)))

            t_stat, p_val, ci_es = safe_inference(info_e["effect"], es_se, self.alpha, df=survey_df)
            info_e["se"] = es_se
            info_e["t_stat"] = t_stat
            info_e["p_value"] = p_val
            info_e["conf_int"] = ci_es
